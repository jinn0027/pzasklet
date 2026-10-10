import os
import json
import re
import sqlite3
import datetime
import difflib
import numpy as np
import pandas as pd
import unicodedata
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

        if not slots_def:
            extracted_slots = set()
            for t_info in templates_body.values():
                q = t_info.get("question", "")
                sql = t_info.get("sql", "")
                found = re.findall(r'\[([A-Z_]+)\]', q + " " + sql)
                for f in found:
                    if f != "USER_ID":
                        extracted_slots.add(f)
            slots_def = {s: f"Automatic slot for {s}" for s in extracted_slots}

        return templates_body, slots_def

    def _init_vllm(self, model_path: str):
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
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
        
        target_mapping = {
            "DAY_OF_WEEK": ("Course_Schedule", "day_of_week"),
            "PERIOD": ("Course_Schedule", "period"),
            "ROOM_ID": ("Room", "room_id"),
            "FORMAT": ("Course", "format"),
            "FACULTY": ("Course", "faculty"),
        }

        for slot, description in self.slots_definition.items():
            key_name = slot.lower()
            default_json_dict[key_name] = None
            
            slot_upper = slot.upper()
            hint_str = ""
            try:
                if slot_upper == "INSTRUCTOR_NAME":
                    df = pd.read_sql("SELECT DISTINCT last_name, user_id FROM User", self.conn)
                    names = [f"{row['last_name']} ({row['user_id']})" for _, row in df.iterrows() if row['last_name']]
                    if names:
                        hint_str = f" [登録例: {', '.join(names[:15])}]"
                elif slot_upper in target_mapping:
                    t_name, c_name = target_mapping[slot_upper]
                    df = pd.read_sql(f"SELECT DISTINCT {c_name} FROM {t_name}", self.conn)
                    vals = [str(v) for v in df[c_name].dropna().tolist()]
                    if vals:
                        hint_str = f" [選択肢例: {', '.join(vals[:15])}]"
            except Exception:
                pass
                
            desc_lines.append(f"- {key_name}: {description}{hint_str}")
        
        json_format_str = json.dumps(default_json_dict, ensure_ascii=False, indent=2)

        prompt = (
            "あなたは大学の授業検索システムのクエリパーサーです。\n"
            "ユーザーの質問から検索条件を分析し、以下のJSON形式で抽出してください。\n\n"
            "【抽出ルール】\n"
            "1. 下記の【スロット定義】にあるキー名（小文字のキー）をそのままJSONのキーとして使用してください。新しいキーを追加しないでください。\n"
            "2. 質問文に含まれる値（人名や条件）をそのままの文字列で切り出してください。\n"
            "3. 質問文に該当する条件がないスロットは、すべて `null` にしてください。\n"
            "4. 「string」などのプレースホルダーや説明文は絶対に出力せず、純粋なJSONオブジェクトのみを出力してください。\n\n"
            "【スロット定義】\n"
            + "\n".join(desc_lines) + "\n\n"
            f"ユーザーの質問: {user_query}\n\n"
            "出力JSONフォーマット:\n"
            f"{json_format_str}"
        )
        
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            raw_output = outputs[0].outputs[0].text.strip()
            print(f"\n[DEBUG] LLMの出力:\n{raw_output}")
            
            json_str = None
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                start_idx = raw_output.find('{')
                if start_idx != -1:
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
            
            if json_str:
                parsed = json.loads(json_str)
                return parsed if isinstance(parsed, dict) else {}
            
            return {}
        except Exception as e:
            print(f"[警告] LLMによるクエリ解析エラー: {e}")
            return {}

    def _resolve_entity(self, slot_name: str, value: Any) -> Any:
        """表（データベース）の実際の値のリストを取得し、大文字小文字を無視して最もフィットする正しい値（ID等）に解決する"""
        if value is None:
            return None
            
        val_str = str(value).strip()
        val_lower = val_str.lower()
        slot_upper = slot_name.upper()

        target_mapping = {
            "DAY_OF_WEEK": ("Course_Schedule", "day_of_week"),
            "PERIOD": ("Course_Schedule", "period"),
            "ROOM_ID": ("Room", "room_id"),
            "FORMAT": ("Course", "format"),
            "FACULTY": ("Course", "faculty"),
        }

        try:
            # 1. 担当教員 (INSTRUCTOR_NAME / USER_ID) の解決
            if slot_upper == "INSTRUCTOR_NAME":
                df = pd.read_sql("SELECT DISTINCT user_id, last_name, first_name FROM User", self.conn)
                
                # ① 大文字小文字を無視した完全一致・部分一致（user_id, last_name, first_name）
                for _, row in df.iterrows():
                    uid = str(row['user_id'])
                    lname = str(row.get('last_name', ''))
                    fname = str(row.get('first_name', ''))
                    
                    if val_lower in uid.lower() or uid.lower() in val_lower:
                        return uid
                    if lname and (val_lower in lname.lower() or lname.lower() in val_lower):
                        return uid
                    if fname and (val_lower in fname.lower() or fname.lower() in val_lower):
                        return uid

                # ② データベース上のすべての名前候補に対するあいまい一致 (difflib)
                all_candidates = []
                candidate_to_uid = {}
                for _, row in df.iterrows():
                    uid = row['user_id']
                    for col in ['user_id', 'last_name', 'first_name']:
                        v = row.get(col)
                        if v:
                            s_v = str(v)
                            all_candidates.append(s_v)
                            candidate_to_uid[s_v.lower()] = uid
                
                matches = difflib.get_close_matches(val_lower, [c.lower() for c in all_candidates], n=1, cutoff=0.3)
                if matches:
                    matched_key = matches[0]
                    if matched_key in candidate_to_uid:
                        return candidate_to_uid[matched_key]
                
                return val_str

            # 2. 一般的なスロットの解決（テーブル定義に基づく大文字小文字非依存マッチング）
            if slot_upper in target_mapping:
                table_name, col_name = target_mapping[slot_upper]
                df = pd.read_sql(f"SELECT DISTINCT {col_name} FROM {table_name}", self.conn)
                candidates = df[col_name].dropna().astype(str).tolist()
                
                # 小文字化して完全一致
                cand_lower_map = {c.lower(): c for c in candidates}
                if val_lower in cand_lower_map:
                    return cand_lower_map[val_lower]
                    
                # 部分一致
                for cand in candidates:
                    if cand.lower() in val_lower or val_lower in cand.lower():
                        return cand
                        
                # あいまい一致 (difflib)
                matches = difflib.get_close_matches(val_lower, [c.lower() for c in candidates], n=1, cutoff=0.3)
                if matches:
                    matched_key = matches[0]
                    if matched_key in cand_lower_map:
                        return cand_lower_map[matched_key]

        except Exception as db_err:
            print(f"[警告] スロット '{slot_name}' のデータベース解決エラー: {db_err}")

        return value

    def process_query(self, user_query: str) -> Tuple[Optional[pd.DataFrame], str]:
        try:
            # 1. LLMによる条件抽出
            parsed_conds = self._parse_query_with_llm(user_query)
            print(f"\n[DEBUG] LLM抽出された生データ: {parsed_conds}")
            
            synonym_map = {
                "instructor": "INSTRUCTOR_NAME",
                "teacher": "INSTRUCTOR_NAME",
                "professor": "INSTRUCTOR_NAME",
                "name": "INSTRUCTOR_NAME",
                "day": "DAY_OF_WEEK",
                "week": "DAY_OF_WEEK",
                "period": "PERIOD",
                "time": "PERIOD",
                "room": "ROOM_ID",
                "location": "ROOM_ID",
                "place": "ROOM_ID",
                "credit": "CREDITS",
                "credits": "CREDITS",
                "format": "FORMAT",
                "eval": "EVAL_ITEM",
                "evaluation": "EVAL_ITEM",
                "faculty": "FACULTY",
                "department": "FACULTY"
            }

            slot_values = {}
            for slot in self.slots_definition.keys():
                slot_upper = slot.upper()
                val = None
                
                for k, v in parsed_conds.items():
                    if v is not None and str(v).lower() == "string":
                        continue
                        
                    k_lower = k.lower()
                    k_upper = k.upper()
                    
                    if k_upper == slot_upper:
                        val = v
                        break
                        
                    mapped_slot = synonym_map.get(k_lower)
                    if mapped_slot and mapped_slot.upper() == slot_upper:
                        val = v
                        break
                        
                    if k_lower in slot.lower() or slot.lower() in k_lower:
                        val = v
                        break
                
                # エンティティ解決（表のデータを用いて正しいIDや値に変換）
                resolved_val = self._resolve_entity(slot, val)
                slot_values[slot] = resolved_val
            
            print(f"[DEBUG] マッピング・解決されたスロット値: {slot_values}")

            # 2. クエリの匿名化（マスク処理：大文字小文字を無視して質問文から抽出値をプレースホルダーに置換）
            masked_query = user_query
            norm_masked_query = unicodedata.normalize('NFKC', str(masked_query))
            norm_masked_lower = norm_masked_query.lower()
            
            for slot in self.slots_definition.keys():
                slot_upper = slot.upper()
                raw_val = parsed_conds.get(slot.lower())
                
                if raw_val is not None:
                    norm_raw = unicodedata.normalize('NFKC', str(raw_val))
                    raw_lower = norm_raw.lower()
                    
                    idx = norm_masked_lower.find(raw_lower)
                    if idx != -1:
                        masked_query = masked_query[:idx] + f"[{slot_upper}]" + masked_query[idx + len(raw_val):]
                        norm_masked_query = unicodedata.normalize('NFKC', str(masked_query))
                        norm_masked_lower = norm_masked_query.lower()

            print(f"[DEBUG] 匿名化されたクエリ: {masked_query}")
            
            # 3. マスク済みクエリのベクトル化
            query_vec = self.embed_model.encode(masked_query, convert_to_numpy=True).astype(np.float32)
            
            # 4. 全テンプレートとのコサイン類似度計算
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
            
            print("\n[DEBUG] テンプレート類似度マッチング上位候補:")
            for rank, item in enumerate(similarities[:3]):
                print(f"  [{rank+1}] スコア: {item['score']:.4f} | テンプレート文: {item['question']} (id: {item['id']})")
            
            if not similarities or similarities[0]['score'] < 0.3:
                reason = "No suitable template matched via embedding similarity."
                self._log_failed_query(user_query, reason)
                return None, "申し訳ありません。入力された条件から授業を特定できませんでした。「月曜 2限」や先生の名前を指定して再度お試しください。"
            
            # 5. 上位候補を順に試行
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
                        resolved_uid = slot_values.get("INSTRUCTOR_NAME")
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
                    df = temp_df
                    filled_sql = current_filled_sql
                    selected_template = candidate
                    break
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