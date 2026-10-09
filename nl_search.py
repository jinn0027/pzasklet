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
        device: Optional[str] = None,  # 💡 将来の拡張用に引数は保持しておく
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
        
        # 💡 vLLMは device 引数をサポートしていないため、環境側の自動検出に任せて標準引数のみで初期化する
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=64,
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

    def _llm_select_instructor(self, user_query: str, instructors_df: pd.DataFrame) -> Optional[str]:
        """LLMを使って、教員名リストの中からユーザーの質問（日英対応・表記ゆれ対応）に最も合致するものを選択する"""
        if instructors_df.empty:
            return None
            
        instructor_names = instructors_df['last_name'].tolist()
        
        prompt = (
            "以下のユーザーの質問（日本語または英語）に含まれている、または意図されている教員名を、候補リストの中から1つだけ選んでください。\n"
            "※注意: 候補リストは漢字（例: 八柳）ですが、ユーザーの質問には英語のローマ字（例: yatuyanagi）、ひらがな、カタカナで書かれている場合があります。\n"
            "言語や表記が異なっていても、同一人物を指していると推測できる場合は、候補リストに含まれる正確な漢字の名称を1つだけ出力してください。\n"
            "該当するものが全くない場合は「なし」とだけ出力してください。余計な説明や理由、文字は一切含めないこと。\n\n"
            f"ユーザーの質問: {user_query}\n"
            f"教員名の候補: {instructor_names}\n\n"
            "選択された教員名:"
        )
        
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            raw_output = outputs[0].outputs[0].text.strip()
            print(f"\n[DEBUG] LLMの出力: {raw_output}")
            
            selected_name = raw_output.strip("「」『』\"'。. ")
            
            if selected_name == "なし" or not selected_name:
                return None

            matched_name = None
            if selected_name in instructor_names:
                matched_name = selected_name
            else:
                for name in instructor_names:
                    if name in selected_name or selected_name in name:
                        matched_name = name
                        break
            
            if matched_name:
                matched_row = instructors_df[instructors_df['last_name'] == matched_name]
                if not matched_row.empty:
                    return matched_row.iloc[0]['user_id']
                    
        except Exception as e:
            print(f"[警告] LLMによる教員名選択中にエラーが発生しました: {e}")
            
        return None

    def process_query(self, user_query: str) -> Tuple[Optional[pd.DataFrame], str]:
        try:
            filled_sql = ""
            query_lower = user_query.lower()
            
            # 1. 先生の名前による検索の検出
            instructor_keywords = ["先生", "教授", "担当", "dr.", "prof", "class", "teacher", "course"]
            if any(kw in query_lower for kw in instructor_keywords) or any(name.lower() in query_lower for name in ["yatuyanagi", "八柳"]):
                try:
                    instructors_df = pd.read_sql("SELECT DISTINCT last_name, user_id FROM User", self.conn)
                    uid = self._llm_select_instructor(user_query, instructors_df)
                    
                    if uid:
                        is_asking_when = any(w in query_lower for w in ["いつ", "曜日", "時限", "時間", "何限", "何曜", "when", "time", "schedule"])
                        is_asking_where = any(w in query_lower for w in ["どこ", "教室", "場所", "ビル", "where", "room", "building", "class"])
                        
                        target_tmpl_id = "instructor_to_courses"
                        if is_asking_when:
                            target_tmpl_id = "instructor_to_schedule"
                        elif is_asking_where and "when" not in query_lower:
                            target_tmpl_id = "instructor_to_room"
                        
                        if target_tmpl_id in self.template_dict:
                            filled_sql = self.template_dict[target_tmpl_id]["sql"].replace("[USER_ID]", uid)
                        else:
                            if target_tmpl_id == "instructor_to_schedule":
                                filled_sql = (
                                    f"SELECT C.course_id, C.title, S.day_of_week, S.period "
                                    f"FROM Course AS C JOIN User AS U ON C.user_id__instructor = U.user_id "
                                    f"JOIN Course_Schedule AS S ON C.course_id = S.course_id "
                                    f"WHERE U.user_id = '{uid}'"
                                )
                            elif target_tmpl_id == "instructor_to_room":
                                filled_sql = (
                                    f"SELECT C.course_id, C.title, R.building_name, R.room_id "
                                    f"FROM Course AS C JOIN User AS U ON C.user_id__instructor = U.user_id "
                                    f"JOIN Course_Schedule AS S ON C.course_id = S.course_id "
                                    f"JOIN Room AS R ON S.room_id = R.room_id "
                                    f"WHERE U.user_id = '{uid}'"
                                )
                            else:
                                filled_sql = self.template_dict.get("instructor_to_courses", {}).get("sql", "").replace("[USER_ID]", uid)
                except Exception as db_err:
                    print(f"[警告] DB読み込みエラー (Userテーブル): {db_err}")

            # 2. 曜日・時限パターン
            if not filled_sql:
                days = ["月", "火", "水", "木", "金", "土", "日"]
                found_day = next((d for d in days if d in user_query), None)
                found_period = next((str(p) for p in range(1, 6) if f"{p}限" in user_query or f"{p}コマ" in user_query), None)
                
                if found_day and found_period:
                    if "schedule_to_courses" in self.template_dict:
                        filled_sql = (
                            self.template_dict["schedule_to_courses"]["sql"]
                            .replace("[DAY_OF_WEEK]", found_day)
                            .replace("[PERIOD]", found_period)
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
