import os
import json
import sqlite3
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Union

class NaturalLanguageSearchEngine:
    def __init__(
        self, 
        conn: sqlite3.Connection, 
        templates: Union[str, List[Dict[str, Any]], Dict[str, Any]] = None, 
        template_dict: Union[List[Dict[str, Any]], Dict[str, Any]] = None,
        model_path: str = "/opt/models/Qwen/Qwen3-4B-Instruct-2507", 
        **kwargs
    ):
        self.conn = conn
        
        target_templates = templates if templates is not None else template_dict
        if target_templates is None:
            raise ValueError("テンプレートを指定してください。")
        self.templates = self._resolve_templates(target_templates)
        
        print(f"🔄 自然言語検索エンジン用 LLMをロード中... (モデル: {model_path})")
        self.llm, self.sampling_params = self._init_vllm(model_path)
        print("✅ 自然言語検索エンジンのLLM準備完了")

    def _resolve_templates(self, templates: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(templates, str):
            if not os.path.exists(templates):
                raise FileNotFoundError(f"テンプレートファイルが見つかりません: {templates}")
            with open(templates, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else list(data.values())
        elif isinstance(templates, list):
            return templates
        elif isinstance(templates, dict):
            return list(templates.values())
        else:
            raise TypeError("テンプレートにはファイルパス（str）、リスト、または辞書を指定してください。")

    def _init_vllm(self, model_path: str):
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            seed=42
        )
        return llm, sampling_params

    def process_query(self, user_query: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        ユーザーの自然言語クエリを解析し、テンプレートに基づいてSQLを実行する。
        詳細な検索結果の表示やSQLのデバッグ出力はすべてこの中で完結させ、
        メニュー側は変更せずにそのまま利用できるようにする。
        """
        try:
            filled_sql = ""
            
            # 1. 先生の名前による検索の検出
            if "先生" in user_query or "教授" in user_query:
                try:
                    instructors_df = pd.read_sql("SELECT DISTINCT last_name, user_id FROM User", self.conn)
                    for _, row in instructors_df.iterrows():
                        name = row['last_name']
                        uid = row['user_id']
                        if name in user_query:
                            is_asking_when = any(w in user_query for w in ["いつ", "曜日", "時限", "時間", "何限", "何曜"])
                            is_asking_where = any(w in user_query for w in ["どこ", "教室", "場所", "ビル"])
                            
                            target_tmpl_id = "instructor_to_courses"
                            if is_asking_when:
                                target_tmpl_id = "instructor_to_schedule"
                            elif is_asking_where:
                                target_tmpl_id = "instructor_to_room"
                            
                            for tmpl in self.templates:
                                if tmpl.get("id") == target_tmpl_id:
                                    filled_sql = tmpl["sql"].replace("[USER_ID]", uid)
                                    break
                            
                            if not filled_sql:
                                for tmpl in self.templates:
                                    if tmpl.get("id") == "instructor_to_courses":
                                        filled_sql = tmpl["sql"].replace("[USER_ID]", uid)
                                        break
                            break
                except Exception as db_err:
                    print(f"[警告] DB読み込みエラー (Userテーブル): {db_err}")

            # 2. 曜日・時限パターン（例: "月曜日の2限"）
            if not filled_sql:
                days = ["月", "火", "水", "木", "金", "土", "日"]
                found_day = next((d for d in days if d in user_query), None)
                found_period = next((str(p) for p in range(1, 6) if f"{p}限" in user_query or f"{p}コマ" in user_query), None)
                
                if found_day and found_period:
                    for tmpl in self.templates:
                        if tmpl.get("id") == "schedule_to_courses":
                            filled_sql = tmpl["sql"].replace("[DAY_OF_WEEK]", found_day).replace("[PERIOD]", found_period)
                            break

            if not filled_sql:
                return None, "申し訳ありません。入力された条件から授業を特定できませんでした。「月曜 2限」や先生の名前を指定して再度お試しください。"

            # 🔍 投入したSQLのデバッグ表示
            print(f"\n[DEBUG] 実行SQL:\n{filled_sql}")

            # SQLの実行
            df = pd.read_sql(filled_sql, self.conn)
            if df.empty:
                return None, "該当する条件の授業は見つかりませんでした。"
            
            # 💡 メニュー側を汚さず、自然言語検索の処理内で詳細な検索結果（日時や場所など）を分かりやすく表示
            print("\n【検索結果（詳細）】")
            print(df.to_string(index=False))

            # メニュー側の選択処理（select_course_from_list）へ渡すためのDataFrameを返す
            return df, "以下の授業が見つました。次のアクションを選択してください："

        except Exception as e:
            print(f"[エラー] クエリ処理中に例外が発生しました: {e}")
            return None, "処理中にエラーが発生しました。別の言い方で試してみてください。"
