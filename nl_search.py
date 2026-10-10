import os
import json
import re
import sqlite3
import datetime
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Union
from sentence_transformers import SentenceTransformer

class NaturalLanguageSearchEngine:
    def __init__(
        self, 
        conn: sqlite3.Connection, 
        templates: Union[str, List[Dict[str, Any]], Dict[str, Any]] = None, 
        template_dict: Union[List[Dict[str, Any]], Dict[str, Any]] = None,
        model_path: str = "/opt/models/Qwen/Qwen3-4B-Instruct-2507", 
        embed_model_path: str = "/opt/models/pkshatech/GLuCoSE-base-ja-v2",
        log_file: str = "failed_queries.jsonl",
        device: Optional[str] = None,
        **kwargs
    ):
        self.conn = conn
        self.log_file = log_file
        self.device = device
        
        target = templates if templates is not None else template_dict
        if target is None:
            raise ValueError("テンプレートを指定してください。")
            
        # テンプレートおよびスロット定義のロード
        self.template_dict, self.slots_definition = self._load_templates_and_metadata(target)
        
        print(f"[DEBUG] ロードされたスロット定義: {list(self.slots_definition.keys())}")
        print(f"[DEBUG] ロードされたテンプレート数: {len(self.template_dict)}")
        
        # 1. LLMの初期化
        print(f"🔄 自然言語検索エンジン用 LLMをロード中... (モデル: {model_path})")
        self.llm, self.sampling_params = self._init_vllm(model_path)
        
        # 2. テンプレート埋め込みモデルの初期化 & ベクトル事前生成
        print(f"🔄 テンプレート埋め込みモデルをロード中... (モデル: {embed_model_path})")
        self.embed_model = self._init_embed_model(embed_model_path)
        self.template_embeddings = self._build_template_embeddings()
        
        print("✅ 自然言語検索エンジンの準備完了")

    def _load_templates_and_metadata(self, templates: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        raw_data = templates
        if isinstance(templates, str):
            if not os.path.exists(templates):
                raise FileNotFoundError(f"テンプレートファイルが見つかりません: {templates}")
            with open(templates, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

        slots_def = {}
        templates_body = {}

        if isinstance(raw_data, dict):
            slots_def = raw_data.get("slots", {})
            body_source = raw_data.get("templates", raw_data)
            if isinstance(body_source, dict):
                templates_body = {k: v for k, v in body_source.items() if k not in ["slots"]}
            elif isinstance(body_source, list):
                templates_body = {t["id"]: t for t in body_source if "id" in t}
        elif isinstance(raw_data, list):
            templates_body = {t["id"]: t for t in raw_data if "id" in t}

        return templates_body, slots_def

    def _init_vllm(self, model_path: str):
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
        )
        sampling_params = SamplingParams(
            temperature=0.0,  # 決定論的な抽出を行うため 0.0 に設定
            max_tokens=128,
            seed=42,
            repetition_penalty=1.1
        )
        return llm, sampling_params

    def _init_embed_model(self, model_path: str):
        load_kwargs = {}
        if self.device:
            load_kwargs["device"] = self.device
        try:
            return SentenceTransformer(model_path, **load_kwargs)
        except Exception:
            return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", **load_kwargs)

    def _build_template_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings_dict = {}
        for t_id, t_info in self.template_dict.items():
            q_text = t_info.get("question", "")
            vec = self.embed_model.encode(q_text, convert_to_numpy=True, show_progress_bar=False)
            embeddings_dict[t_id] = vec.astype(np.float32)
        return embeddings_dict

    @staticmethod
    def _calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _log_failed_query(self, user_query: str, reason: str):
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": user_query,
            "reason": reason
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[警告] 失敗ログの書き込みに失敗しました: {e}")

    def _parse_query_with_llm(self, user_query: str) -> Dict[str, Any]:
        desc_lines = []
        default_json_dict = {}
        
        for slot, description in self.slots_definition.items():
            key_name = slot.lower()
            default_json_dict[key_name] = None
            desc_lines.append(f"- \"{key_name}\": {description}")
        
        json_format_str = json.dumps(default_json_dict, ensure_ascii=False, indent=2)

        # 💡 メタデータ定義に基づいてLLMに確実に抽出させる汎用プロンプト
        prompt = (
            "You are a precise query parsing assistant for a university course search system.\n"
            "Analyze the user's question and extract the corresponding slot values into a JSON object.\n\n"
            "Extraction Rules:\n"
            "1. Do not translate, normalize, or rephrase extracted values; copy the exact characters or numbers as they appear in the user's question.\n"
            "2. If a slot value is present in the question, extract it. If not present, set it to null.\n"
            "3. Output ONLY valid JSON matching the exact schema below, with no conversational filler or extra text.\n\n"
            "Slots Definition:\n"
            + "\n".join(desc_lines) + "\n\n"
            f"User Question: {user_query}\n\n"
            f"JSON Output Format:\n{json_format_str}"
        )
        
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            raw_output = outputs[0].outputs[0].text.strip()
            print(f"\n[DEBUG] LLMの出力:\n{raw_output}")
            
            start_idx = raw_output.find('{')
            if start_idx == -1:
                return {}
                
            depth = 0
            in_string = False
            escape = False
            end_idx = -1
            
            for i in range(start_idx, len(raw_output)):
                char = raw_output[i]
                if escape:
                    escape = False
                    continue
                if char == '\\' and in_string:
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i
                            break
            
            if end_idx != -1:
                json_str = raw_output[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                return parsed if isinstance(parsed, dict) else {}
            
            return {}
        except Exception as e:
            print(f"[警告] LLMによるクエリ解析エラー: {e}")
            return {}

    def _resolve_entity(self, slot_name: str, value: Any) -> Any:
        if slot_name == "INSTRUCTOR_NAME" and value:
            try:
                instructors_df = pd.read_sql("SELECT DISTINCT last_name, user_id FROM User", self.conn)
                kw_lower = str(value).lower()
                for _, row in instructors_df.iterrows():
                    uid = str(row['user_id'])
                    last_name = str(row['last_name']).lower()
                    if (last_name and last_name in kw_lower) or (kw_lower in last_name) or (kw_lower in uid.lower()):
                        return uid
            except Exception as db_err:
                print(f"[警告] エンティティ解決エラー: {db_err}")
        return value

    def process_query(self, user_query: str) -> Tuple[Optional[pd.DataFrame], str]:
        try:
            # 1. LLMによる条件抽出（完全にメタデータ定義駆動）
            parsed_conds = self._parse_query_with_llm(user_query)
            print(f"\n[DEBUG] LLM抽出された生データ: {parsed_conds}")
            
            # 大文字・小文字の差異を完全に吸収してスロット値にマッピング
            slot_values = {}
            for slot in self.slots_definition.keys():
                val = None
                for k, v in parsed_conds.items():
                    if k.upper() == slot.upper() and v is not None:
                        val = v
                        break
                slot_values[slot] = val
            
            print(f"[DEBUG] マッピングされたスロット値: {slot_values}")

            # 2. 質問文の動的マスク処理（匿名化）
            masked_query = user_query
            for slot in self.slots_definition.keys():
                val = slot_values.get(slot)
                if val and str(val) in masked_query:
                    masked_query = masked_query.replace(str(val), f"[{slot}]")
            
            print(f"[DEBUG] 匿名化されたクエリ: {masked_query}")
            
            # 3. マスク済みクエリのベクトル化
            query_vec = self.embed_model.encode(masked_query, convert_to_numpy=True).astype(np.float32)
            
            # 4. 全テンプレートとのコサイン類似度を計算してスコア順に並び替え
            similarities = []
            for t_id, t_info in self.template_dict.items():
                t_vec = self.template_embeddings[t_id]
                score = self._calculate_cosine_similarity(query_vec, t_vec)
                similarities.append({
                    "id": t_id,
                    "question": t_info.get("question"),
                    "sql": t_info.get("sql"),
                    "score": score
                })
            
            similarities.sort(key=lambda x: x['score'], reverse=True)
            
            # テンプレートマッチングの上位候補をデバッグ表示
            print("\n[DEBUG] テンプレート類似度マッチング上位候補:")
            for rank, item in enumerate(similarities[:3]):
                print(f"  [{rank+1}] スコア: {item['score']:.4f} | テンプレート文: {item['question']} (id: {item['id']})")
            
            if not similarities or similarities[0]['score'] < 0.3:
                reason = "No suitable template matched via embedding similarity."
                self._log_failed_query(user_query, reason)
                return None, "申し訳ありません。入力された条件から授業を特定できませんでした。「月曜 2限」や先生の名前を指定して再度お試しください。"
            
            # 5. 上位の候補（最大3つ）を順番に試行
            df = None
            filled_sql = ""
            selected_template = None
            
            for candidate in similarities[:3]:
                if candidate['score'] < 0.2:
                    break
                
                temp_sql = candidate["sql"]
                required_slots_in_tmpl = re.findall(r'\[([A-Z_]+)\]', candidate["question"] + " " + temp_sql)
                
                can_fill = True
                current_filled_sql = temp_sql
                
                for slot in required_slots_in_tmpl:
                    if slot == "USER_ID":
                        instructor_val = slot_values.get("INSTRUCTOR_NAME")
                        resolved_uid = self._resolve_entity("INSTRUCTOR_NAME", instructor_val)
                        if not resolved_uid:
                            can_fill = False
                            break
                        current_filled_sql = current_filled_sql.replace("[USER_ID]", str(resolved_uid))
                    elif slot in slot_values:
                        val = slot_values.get(slot)
                        if val is None:
                            can_fill = False
                            break
                        current_filled_sql = current_filled_sql.replace(f"[{slot}]", str(val))
                    else:
                        can_fill = False
                        break
                
                if not can_fill:
                    print(f"[DEBUG] テンプレート {candidate['id']} は必要なスロット値が不足しているためスキップします。")
                    continue
                
                print(f"\n[DEBUG] 試行中のテンプレート: {candidate['id']} (スコア: {candidate['score']:.4f})")
                print(f"[DEBUG] 生成SQL:\n{current_filled_sql}")
                
                try:
                    temp_df = pd.read_sql(current_filled_sql, self.conn)
                    print(f"[DEBUG] 実行結果件数: {len(temp_df)} 件")
                    if not temp_df.empty:
                        df = temp_df
                        filled_sql = current_filled_sql
                        selected_template = candidate
                        break
                    else:
                        df = temp_df
                        filled_sql = current_filled_sql
                        selected_template = candidate
                except Exception as sql_err:
                    print(f"[DEBUG] SQL実行エラー: {sql_err}")
                    continue
            
            if selected_template is None or df is None:
                reason = "All top matching templates failed to execute or yield results."
                self._log_failed_query(user_query, reason)
                return None, "該当する条件の授業は見つかりませんでした。"
            
            print(f"\n[DEBUG] 最終採用テンプレート: {selected_template['id']} (スコア: {selected_template['score']:.4f})")
            print(f"[DEBUG] 実行SQL:\n{filled_sql}")
            
            if df.empty:
                reason = f"Query executed successfully but returned 0 results. SQL: {filled_sql}"
                self._log_failed_query(user_query, reason)
                return None, "該当する条件の授業は見つかりませんでした。"
            
            print("\n【検索結果（詳細）】")
            print(df.to_string(index=False))

            return df, "以下の授業が見つかりました。次のアクションを選択してください："

        except Exception as e:
            error_reason = f"Exception occurred: {str(e)}"
            self._log_failed_query(user_query, error_reason)
            print(f"[エラー] クエリ処理中に例外が発生しました: {e}")
            return None, "処理中にエラーが発生しました。別の言い方で試してみてください。"