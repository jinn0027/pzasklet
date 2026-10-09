import sqlite3
import pandas as pd
from typing import Tuple, Optional, Dict, Any

class NaturalLanguageSearchEngine:
    """
    自然言語のクエリを解析し、テンプレートとスロットに分解して
    安全にSQLを実行するための検索エンジンクラス。
    """
    def __init__(self, conn: sqlite3.Connection, template_dict: Dict[str, Any]):
        self.conn = conn
        self.template_dict = template_dict
        # 将来的にここでLLMのクライアントや埋め込みモデルを保持することも可能です

    def _render_template(self, template_entry: dict, slots: dict) -> Tuple[str, str]:
        """テンプレートの質問文とSQLのプレースホルダーをスロット値で置換する"""
        q = template_entry["question"]
        sql = template_entry["sql"]
        for key, val in slots.items():
            placeholder = f"[{key}]"
            q = q.replace(placeholder, str(val))
            sql = sql.replace(placeholder, str(val))
        return q, sql

    def process_query(self, query_text: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        自然言語のクエリを解析し、適切なテンプレートにスロットを埋めてSQLを実行する。
        成功した場合は (DataFrame, 案内メッセージ) を返し、
        失敗した場合は (None, エラーメッセージ) を返す。
        """
        query_text_lower = query_text.lower()
        
        # 1. 日時（曜日・時限）のパターンマッチング
        days = ["月", "火", "水", "木", "金", "土", "日"]
        found_day = next((d for d in days if d in query_text), None)
        
        period = None
        for p in range(1, 6):
            if f"{p}限" in query_text or f"第{p}限" in query_text or f" {p} " in query_text:
                period = p
                break

        if found_day and period:
            if "schedule_to_courses" in self.template_dict:
                tmpl = self.template_dict["schedule_to_courses"]
                q, sql = self._render_template(tmpl, {"DAY_OF_WEEK": found_day, "PERIOD": period})
                
                try:
                    df = pd.read_sql(sql, self.conn)
                    if df.empty:
                        return None, f"ご指定の条件（{found_day}曜日 {period}限）に一致する授業は見つかりませんでした。"
                    return df, f"【検索結果】{found_day}曜日 {period}限に行われる授業が見つかりました："
                except Exception as e:
                    return None, f"データベースの検索中にエラーが発生しました: {e}"

        # 2. 先生の名前による検索のパターン
        try:
            instructors_df = pd.read_sql("SELECT DISTINCT user_id, last_name FROM User", self.conn)
            matched_instructor = None
            for _, row in instructors_df.iterrows():
                if row['last_name'] in query_text:
                    matched_instructor = row
                    break
            
            if matched_instructor is not None and "instructor_to_courses" in self.template_dict:
                tmpl = self.template_dict["instructor_to_courses"]
                q, sql = self._render_template(tmpl, {"USER_ID": matched_instructor['user_id']})
                
                df = pd.read_sql(sql, self.conn)
                if df.empty:
                    return None, f"{matched_instructor['last_name']}先生が担当している授業は見つかりませんでした。"
                return df, f"【検索結果】{matched_instructor['last_name']}先生が担当している授業です："
        except Exception:
            pass

        # 3. フォールバック
        return None, "申し訳ありません。入力された条件から授業を特定できませんでした。「月曜 3限」や先生の名前などを含めて再度お試しください。"
