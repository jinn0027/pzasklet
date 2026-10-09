import os
import json
import sqlite3
import datetime
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Union

class NaturalLanguageSearchEngine:
    def __init__(
        self, 
        conn: sqlite3.Connection, 
        templates: Union[str, List[Dict[str, Any]], Dict[str, Any]] = None, 
        template_dict: Union[List[Dict[str, Any]], Dict[str, Any]] = None,
        model_path: str = "/opt/models/Qwen/Qwen3-4B-Instruct-2507", 
        log_file: str = "failed_queries.jsonl",
        device: Optional[str] = None,  # 将来の拡張・インターフェース保持用
        **kwargs
    ):
        self.conn = conn
        self.log_file = log_file
        self.device = device
        
        target = templates if templates is not None else template_dict
        if target is None:
            raise ValueError("テンプレートを指定してください。")
        self.template_dict = self._normalize_to_dict(target)
        
        print(f"🔄 自然言語検索エンジン用 LLMをロード中... (モデル: {model_path}, デバイス: {self.device or '自動'})")
        self.llm, self.sampling_params = self._init_vllm(model_path)
        print("✅ 自然言語検索エンジンのLLM準備完了")

    def _normalize_to_dict(self, templates: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if isinstance(templates, str):
            if not os.path.exists(templates):
                raise FileNotFoundError(f"テンプレートファイルが見つかりません: {templates}")
            with open(templates, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {t["id"]: t for t in data if "id" in t}
                return data
        elif isinstance(templates, list):
            return {t["id"]: t for t in templates if "id" in t}
        elif isinstance(templates, dict):
            return templates
        else:
            raise TypeError("テンプレートの形式が不正です。")

    def _init_vllm(self, model_path: str):
        from vllm import LLM, SamplingParams
        
        # vLLMはデバイス引数を取らないため環境側の自動検出に任せる
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
            seed=42
        )
        return llm, sampling_params

    def _log_failed_query(self, user_query: str, reason: str):
        """失敗したクエリをJSONL形式でログファイルに記録する"""
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
        """
        プロンプトに個人名を一切載せず、意味（抽出ルール）だけでLLMに条件を解析させる
        """
        prompt = (
            "あなたは大学の授業検索システムのクエリパーサーです。\n"
            "以下のユーザーの質問から、検索条件を分析し、JSON形式で抽出してください。\n\n"
            "【抽出するキー】\n"
            "- instructor: 教員の名前、名字、またはそのローマ字・よみがな（例: 'yatsuyanagi', '八柳'）。ない場合は null。\n"
            "- day_of_week: 曜日（月、火、水、木、金、土、日）。ない場合は null。\n"
            "- period: 時限の数字（1〜5）。ない場合は null。\n"
            "- target_info: 知りたい情報（'schedule'=日時, 'room'=場所, 'courses'=担当授業一覧, 'general'=指定なし）。\n\n"
            f"ユーザーの質問: {user_query}\n\n"
            "出力には必ず以下のJSONフォーマットを1回だけ含めてください。\n"
            "{\"instructor\": null, \"day_of_week\": null, \"period\": null, \"target_info\": \"general\"}"
        )
        
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            raw_output = outputs[0].outputs[0].text.strip()
            print(f"\n[DEBUG] LLMの出力:\n{raw_output}")
            
            # 💡 最初の '{' から対応する閉じ括弧 '}' までを正確に追跡して最初のJSONオブジェクトのみを抽出
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

    def process_query(self, user_query: str) -> Tuple[Optional[pd.DataFrame], str]:
        try:
            filled_sql = ""
            
            # 1. LLMに意味ベースで条件を抽出させる（個人名リストは一切プロンプトに渡さない）
            parsed_conds = self._parse_query_with_llm(user_query)
            
            instructor_keyword = parsed_conds.get("instructor")
            day_val = parsed_conds.get("day_of_week")
            period_val = parsed_conds.get("period")
            target_info = parsed_conds.get("target_info", "general")
            
            # 2. 教員名キーワードが取れた場合の汎用的なDB照合・SQL生成
            if instructor_keyword:
                try:
                    instructors_df = pd.read_sql("SELECT DISTINCT last_name, user_id FROM User", self.conn)
                    
                    matched_uid = None
                    kw_lower = str(instructor_keyword).lower()
                    
                    for _, row in instructors_df.iterrows():
                        uid = str(row['user_id'])
                        last_name = str(row['last_name']).lower()
                        
                        # 漢字の名字の部分一致、または user_id のパーツ（ローマ字等）への部分一致を汎用的にチェック
                        if (last_name and last_name in kw_lower) or (kw_lower in last_name) or (kw_lower in uid.lower()):
                            matched_uid = uid
                            break
                    
                    if matched_uid:
                        # ユーザーの質問の意図（いつ、どこ、など）に応じたテンプレート選択
                        target_tmpl_id = "instructor_to_courses"
                        if target_info == "schedule":
                            target_tmpl_id = "instructor_to_schedule"
                        elif target_info == "room":
                            target_tmpl_id = "instructor_to_room"
                            
                        if target_tmpl_id in self.template_dict:
                            filled_sql = self.template_dict[target_tmpl_id]["sql"].replace("[USER_ID]", matched_uid)
                        else:
                            if target_tmpl_id == "instructor_to_schedule":
                                filled_sql = (
                                    f"SELECT C.course_id, C.title, S.day_of_week, S.period "
                                    f"FROM Course AS C JOIN User AS U ON C.user_id__instructor = U.user_id "
                                    f"JOIN Course_Schedule AS S ON C.course_id = S.course_id "
                                    f"WHERE U.user_id = '{matched_uid}'"
                                )
                            elif target_tmpl_id == "instructor_to_room":
                                filled_sql = (
                                    f"SELECT C.course_id, C.title, R.building_name, R.room_id "
                                    f"FROM Course AS C JOIN User AS U ON C.user_id__instructor = U.user_id "
                                    f"JOIN Course_Schedule AS S ON C.course_id = S.course_id "
                                    f"JOIN Room AS R ON S.room_id = R.room_id "
                                    f"WHERE U.user_id = '{matched_uid}'"
                                )
                            else:
                                filled_sql = self.template_dict.get("instructor_to_courses", {}).get("sql", "").replace("[USER_ID]", matched_uid)
                except Exception as db_err:
                    print(f"[警告] DB検索エラー: {db_err}")

            # 3. 曜日・時限パターンによるテンプレート充填
            if not filled_sql and day_val and period_val:
                if "schedule_to_courses" in self.template_dict:
                    filled_sql = (
                        self.template_dict["schedule_to_courses"]["sql"]
                        .replace("[DAY_OF_WEEK]", str(day_val))
                        .replace("[PERIOD]", str(period_val))
                    )

            # ❌ 条件を特定できなかった場合のログ記録
            if not filled_sql:
                reason = "Could not resolve query intent or fill template slots."
                self._log_failed_query(user_query, reason)
                return None, "申し訳ありません。入力された条件から授業を特定できませんでした。「月曜 2限」や先生の名前を指定して再度お試しください。"

            print(f"\n[DEBUG] 実行SQL:\n{filled_sql}")

            df = pd.read_sql(filled_sql, self.conn)
            
            # ❌ 実行結果が空だった場合のログ記録
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
