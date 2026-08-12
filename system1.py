import sqlite3
import pandas as pd
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer

class CourseSimilarityNormalizer:
    """
    SentenceTransformer（GLuCoSE-base-ja-v2など）を使用して、
    授業のテキスト（テーマ・概要）の埋め込み生成・コサイン類似度検索を行うクラスです。
    """
    def __init__(self, model_path: str = "/opt/models/pkshatech/GLuCoSE-base-ja-v2"):
        self.model_path = model_path
        self.model = None
        self.courses_df = None
        self.course_embeddings = None
        
        try:
            self.model = SentenceTransformer(self.model_path)
        except Exception as e:
            try:
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e2:
                raise RuntimeError(f"致命的なエラー: 埋め込みモデルのロードに失敗しました。パス: {model_path}, 詳細: {e} / {e2}") from e2

    @staticmethod
    def _calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """NumPyを使用して2つのベクトル間のコサイン類似度を計算します。"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def build_course_embeddings(self, conn: sqlite3.Connection):
        """データベースから全授業を取得し、テーマ・概要を結合したテキストの埋め込みを生成します。"""
        self.courses_df = pd.read_sql("SELECT course_id, title, theme, class_abstract FROM Course WHERE course_id IS NOT NULL", conn)
        self.courses_df['text_for_embedding'] = self.courses_df['theme'].fillna('') + " " + self.courses_df['class_abstract'].fillna('')
        
        texts = self.courses_df['text_for_embedding'].tolist()
        if texts:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            self.course_embeddings = embeddings.astype(np.float32)
        else:
            self.course_embeddings = np.array([], dtype=np.float32)

    def search_similar_courses(self, target_course_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """指定した授業IDに対して、内容（テーマ・概要）が類似している上位k個の授業を返します。"""
        if self.courses_df is None or self.course_embeddings is None:
            return []

        target_matches = self.courses_df[self.courses_df['course_id'] == target_course_id]
        if target_matches.empty:
            return []
        
        target_idx = target_matches.index[0]
        target_vec = self.course_embeddings[target_idx]

        results = []
        for idx, row in self.courses_df.iterrows():
            vec = self.course_embeddings[idx]
            score = self._calculate_cosine_similarity(target_vec, vec)
            results.append({
                'course_id': row['course_id'],
                'title': row['title'],
                'score': score
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]


def load_templates(json_path="query_templates1.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"テンプレートファイル '{json_path}' が見つかりません。")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def init_db():
    conn = sqlite3.connect(':memory:')
    csv_files = ['Course.csv', 'Course_Schedule.csv', 'Evaluation_Method.csv', 'Room.csv', 'User.csv']
    for file in csv_files:
        if os.path.exists(file):
            table_name = os.path.splitext(file)[0]
            df = pd.read_csv(file, skiprows=[1, 2])
            df.to_sql(table_name, conn, if_exists='replace', index=False)
    return conn

def render_template(template_entry: dict, slots: dict) -> tuple[str, str]:
    """テンプレートの質問文とSQLのプレースホルダーをスロット値で置換する"""
    q = template_entry["question"]
    sql = template_entry["sql"]
    for key, val in slots.items():
        placeholder = f"[{key}]"
        q = q.replace(placeholder, str(val))
        sql = sql.replace(placeholder, str(val))
    return q, sql

def inspect_course(conn, template_dict, target_course_id, target_title, normalizer):
    """特定の授業に対する検索と、その後のアクションを処理する関数"""
    while True:
        print(f"\n選択された授業: {target_title} ({target_course_id})")

        print("\n何を知りたいですか？")
        print("1. 担当の先生")
        print("2. 日時（曜日・時限）")
        print("3. 場所（教室）")
        print("4. 内容（テーマ・概要）")
        print("-1. 終了（トップに戻る）")
        
        choice = input("番号を選択してください: ")
        if choice == '-1':
            print("\n--- トップメニューに戻ります ---")
            return

        sql = ""
        q = ""
        instructor_id = None
        day_of_week = None
        period = None
        room_id = None
        is_content_search = False

        if choice == '1':
            attr_df = pd.read_sql(f"SELECT U.user_id FROM Course C JOIN User U ON C.user_id__instructor = U.user_id WHERE C.course_id = '{target_course_id}'", conn)
            if attr_df.empty or attr_df.iloc[0, 0] is None:
                print("担当教員の情報が見つかりませんでした。")
                continue
            instructor_id = attr_df.iloc[0, 0]
            
            q = f"ID: {target_course_id} の授業の担当教員を教えてください。"
            sql = f"SELECT U.last_name, U.first_name FROM Course C JOIN User U ON C.user_id__instructor = U.user_id WHERE C.course_id = '{target_course_id}'"

        elif choice == '2':
            attr_df = pd.read_sql(f"SELECT day_of_week, period FROM Course_Schedule WHERE course_id = '{target_course_id}'", conn)
            if attr_df.empty:
                print("日時の情報が見つかりませんでした。")
                continue
            day_of_week = attr_df.iloc[0]['day_of_week']
            period = str(attr_df.iloc[0]['period'])
            q = f"ID: {target_course_id} の授業はいつ行われますか？"
            sql = f"SELECT day_of_week, period FROM Course_Schedule WHERE course_id = '{target_course_id}'"

        elif choice == '3':
            attr_df = pd.read_sql(f"SELECT room_id FROM Course_Schedule WHERE course_id = '{target_course_id}'", conn)
            if attr_df.empty:
                print("場所の情報が見つかりませんでした。")
                continue
            room_id = attr_df.loc[0, 'room_id']
            q = f"ID: {target_course_id} の授業の教室（場所）を教えてください。"
            sql = f"SELECT DISTINCT R.building_name, R.room_id FROM Course_Schedule S JOIN Room R ON S.room_id = R.room_id WHERE S.course_id = '{target_course_id}'"

        elif choice == '4':
            q = f"ID: {target_course_id} の授業内容やテーマについて教えてください。"
            sql = f"SELECT theme, class_abstract FROM Course WHERE course_id = '{target_course_id}'"
            is_content_search = True

        else:
            print("無効な選択です。")
            continue

        print(f"\n[DEBUG] 自然言語: {q}")
        print(f"[DEBUG] 実行SQL: {sql}")
        
        res = pd.read_sql(sql, conn)
        
        print("\n【検索結果】")
        if res.empty:
            print("該当する情報が見つかりませんでした。")
        else:
            print(res.to_string(index=False))

        # アクション選択ループ（統合メニュー）
        while True:
            print("\n次のアクションを選んでください:")
            print("1. 担当の先生")
            print("2. 日時（曜日・時限）")
            print("3. 場所（教室）")
            print("4. 内容（テーマ・概要）")
            print("5. 類似の授業一覧を表示する")
            print("-1. 終了（トップに戻る）")
            
            sub_choice = input("番号を選択してください: ")
            
            if sub_choice == '-1':
                print("\n--- トップメニューに戻ります ---")
                return
            
            elif sub_choice in ['1', '2', '3', '4']:
                choice = sub_choice
                if choice == '1':
                    attr_df = pd.read_sql(f"SELECT U.user_id FROM Course C JOIN User U ON C.user_id__instructor = U.user_id WHERE C.course_id = '{target_course_id}'", conn)
                    if attr_df.empty or attr_df.iloc[0, 0] is None:
                        print("担当教員の情報が見つかりませんでした。")
                        continue
                    instructor_id = attr_df.iloc[0, 0]
                    q = f"ID: {target_course_id} の授業の担当教員を教えてください。"
                    sql = f"SELECT U.last_name, U.first_name FROM Course C JOIN User U ON C.user_id__instructor = U.user_id WHERE C.course_id = '{target_course_id}'"
                    is_content_search = False
                elif choice == '2':
                    attr_df = pd.read_sql(f"SELECT day_of_week, period FROM Course_Schedule WHERE course_id = '{target_course_id}'", conn)
                    if attr_df.empty:
                        print("日時の情報が見つかりませんでした。")
                        continue
                    day_of_week = attr_df.iloc[0]['day_of_week']
                    period = str(attr_df.iloc[0]['period'])
                    q = f"ID: {target_course_id} の授業はいつ行われますか？"
                    sql = f"SELECT day_of_week, period FROM Course_Schedule WHERE course_id = '{target_course_id}'"
                    is_content_search = False
                elif choice == '3':
                    attr_df = pd.read_sql(f"SELECT room_id FROM Course_Schedule WHERE course_id = '{target_course_id}'", conn)
                    if attr_df.empty:
                        print("場所の情報が見つかりませんでした。")
                        continue
                    room_id = attr_df.loc[0, 'room_id']
                    q = f"ID: {target_course_id} の授業の教室（場所）を教えてください。"
                    sql = f"SELECT DISTINCT R.building_name, R.room_id FROM Course_Schedule S JOIN Room R ON S.room_id = R.room_id WHERE S.course_id = '{target_course_id}'"
                    is_content_search = False
                elif choice == '4':
                    q = f"ID: {target_course_id} の授業内容やテーマについて教えてください。"
                    sql = f"SELECT theme, class_abstract FROM Course WHERE course_id = '{target_course_id}'"
                    is_content_search = True

                print(f"\n[DEBUG] 自然言語: {q}")
                print(f"[DEBUG] 実行SQL: {sql}")
                
                res = pd.read_sql(sql, conn)
                print("\n【検索結果】")
                if res.empty:
                    print("該当する情報が見つかりませんでした。")
                else:
                    print(res.to_string(index=False))
                continue
            
            elif sub_choice == '5':
                if is_content_search:
                    similar_results = normalizer.search_similar_courses(target_course_id, k=10)
                    print(f"\n【内容（テーマ・概要）が類似している授業一覧 (コサイン類似度順)】")
                    if not similar_results:
                        print("類似授業が見つかりませんでした。")
                        continue
                    
                    for rank, res_item in enumerate(similar_results):
                        marker = " (※選択中の授業)" if res_item['course_id'] == target_course_id else ""
                        print(f"[{rank}] (類似度: {res_item['score']:.4f}) {res_item['course_id']}: {res_item['title']}{marker}")
                    
                    try:
                        s_selected_idx = int(input("\n気になる授業の番号を選んでください (戻る場合は -1): "))
                        if s_selected_idx == -1:
                            continue
                        selected_item = similar_results[s_selected_idx]
                        inspect_course(conn, template_dict, selected_item['course_id'], selected_item['title'], normalizer)
                        return
                    except (ValueError, KeyError, IndexError):
                        print("無効な番号です。")
                        continue
                else:
                    sim_sql = ""
                    sim_q = ""
                    if choice == '1' and instructor_id:
                        tmpl = template_dict["instructor_to_courses"]
                        sim_q, sim_sql = render_template(tmpl, {"USER_ID": instructor_id})
                    elif choice == '2' and day_of_week and period:
                        tmpl = template_dict["schedule_to_courses"]
                        sim_q, sim_sql = render_template(tmpl, {"DAY_OF_WEEK": day_of_week, "PERIOD": period})
                    elif choice == '3' and room_id:
                        tmpl = template_dict["room_to_courses"]
                        sim_q, sim_sql = render_template(tmpl, {"ROOM_ID": room_id})

                    print(f"\n[DEBUG] 自然言語: {sim_q}")
                    print(f"[DEBUG] 類似授業SQL: {sim_sql}")
                    
                    if sim_sql:
                        similar_res = pd.read_sql(sim_sql, conn)
                        print(f"\n【類似の授業一覧（もとの授業を含む）】")
                        if similar_res.empty:
                            print("該当する他の授業はありませんでした。")
                            continue
                        else:
                            for s_idx, s_row in similar_res.reset_index(drop=True).iterrows():
                                marker = " (※選択中の授業)" if s_row['course_id'] == target_course_id else ""
                                print(f"[{s_idx}] {s_row['course_id']}: {s_row['title']}{marker}")
                            
                            try:
                                s_selected_idx = int(input("\n気になる授業の番号を選んでください (戻る場合は -1): "))
                                if s_selected_idx == -1:
                                    continue
                                selected_sim_row = similar_res.reset_index(drop=True).loc[s_selected_idx]
                                inspect_course(conn, template_dict, selected_sim_row['course_id'], selected_sim_row['title'], normalizer)
                                return
                            except (ValueError, KeyError):
                                print("無効な番号です。")
                                continue
                    else:
                        print("属性ベースの類似授業を検索するための情報がありません。")
            else:
                print("無効な選択です。もう一度入力してください。")

def select_course_from_list(conn, template_dict, normalizer, courses_df):
    """指定された授業のDataFrameから授業を選択させ、詳細画面へ進む関数"""
    if courses_df.empty:
        print("該当する授業がありません。")
        return

    print("\n=== 授業一覧 ===")
    for idx, row in courses_df.reset_index(drop=True).iterrows():
        print(f"[{idx}] {row['course_id']}: {row['title']}")
    
    try:
        selected_idx = int(input("\n授業の番号を選んでください (戻る場合は -1): "))
        if selected_idx == -1:
            return
        selected_row = courses_df.reset_index(drop=True).loc[selected_idx]
        target_course_id = selected_row['course_id']
        target_title = selected_row['title']
    except (ValueError, KeyError, IndexError):
        print("無効な番号です。")
        return

    inspect_course(conn, template_dict, target_course_id, target_title, normalizer)

def main():
    try:
        query_templates = load_templates("query_templates1.json")
    except Exception as e:
        print(f"エラー: {e}")
        return

    template_dict = {t["id"]: t for t in query_templates}
    conn = init_db()
    print("--- 授業情報検索システム1 (pzasklet) ---")
    
    model_path = os.environ.get("EMBEDDING_MODEL_PATH", "/opt/models/pkshatech/GLuCoSE-base-ja-v2")
    print(f"埋め込みモデル ({model_path}) をロードし、授業テキストのベクトルを生成中...")
    try:
        normalizer = CourseSimilarityNormalizer(model_path=model_path)
        normalizer.build_course_embeddings(conn)
        print("準備完了しました。\n")
    except Exception as e:
        print(f"モデルの初期化またはベクトル生成に失敗しました: {e}")
        return
    
    while True:
        print("\n=== トップメニュー: 検索の切り口を選んでください ===")
        print("1. 授業一覧から選ぶ")
        print("2. 日時一覧から選んで授業を表示する")
        print("3. 場所一覧から選んで授業を表示する")
        print("4. 先生一覧から選んで授業を表示する")
        print("-1. システム終了")
        
        top_choice = input("番号を選択してください: ")
        
        if top_choice == '-1':
            print("\nシステムを終了します。")
            break
            
        elif top_choice == '1':
            select_course_from_list(conn, template_dict, normalizer, normalizer.courses_df)
            
        elif top_choice == '2':
            schedules_df = pd.read_sql("SELECT DISTINCT day_of_week, period FROM Course_Schedule ORDER BY day_of_week, period", conn)
            if schedules_df.empty:
                print("日時の情報が見つかりませんでした。")
                continue
            
            print("\n=== 日時一覧 ===")
            for idx, row in schedules_df.reset_index(drop=True).iterrows():
                print(f"[{idx}] 曜日: {row['day_of_week']}, 時限: {row['period']}")
            
            try:
                s_idx = int(input("\n日時の番号を選んでください (戻る場合は -1): "))
                if s_idx == -1:
                    continue
                sel_sched = schedules_df.reset_index(drop=True).loc[s_idx]
                d_val = sel_sched['day_of_week']
                p_val = sel_sched['period']
            except (ValueError, KeyError, IndexError):
                print("無効な番号です。")
                continue
            
            tmpl = template_dict["schedule_to_courses"]
            q, sql = render_template(tmpl, {"DAY_OF_WEEK": d_val, "PERIOD": p_val})
            print(f"\n[DEBUG] 自然言語: {q}")
            print(f"[DEBUG] 実行SQL: {sql}")
            
            courses_res = pd.read_sql(sql, conn)
            select_course_from_list(conn, template_dict, normalizer, courses_res)
            
        elif top_choice == '3':
            rooms_df = pd.read_sql("SELECT DISTINCT R.room_id, R.building_name FROM Course_Schedule S JOIN Room R ON S.room_id = R.room_id ORDER BY R.room_id", conn)
            if rooms_df.empty:
                print("場所の情報が見つかりませんでした。")
                continue
            
            print("\n=== 場所一覧 ===")
            for idx, row in rooms_df.reset_index(drop=True).iterrows():
                print(f"[{idx}] 教室: {row['room_id']} ({row['building_name']})")
            
            try:
                s_idx = int(input("\n場所の番号を選んでください (戻る場合は -1): "))
                if s_idx == -1:
                    continue
                sel_room = rooms_df.reset_index(drop=True).loc[s_idx]
                r_val = sel_room['room_id']
            except (ValueError, KeyError, IndexError):
                print("無効な番号です。")
                continue
            
            tmpl = template_dict["room_to_courses"]
            q, sql = render_template(tmpl, {"ROOM_ID": r_val})
            print(f"\n[DEBUG] 自然言語: {q}")
            print(f"[DEBUG] 実行SQL: {sql}")
            
            courses_res = pd.read_sql(sql, conn)
            select_course_from_list(conn, template_dict, normalizer, courses_res)
            
        elif top_choice == '4':
            instructors_df = pd.read_sql("SELECT DISTINCT U.user_id, U.last_name, U.first_name FROM Course C JOIN User U ON C.user_id__instructor = U.user_id ORDER BY U.last_name", conn)
            if instructors_df.empty:
                print("先生の情報が見つかりませんでした。")
                continue
            
            print("\n=== 先生一覧 ===")
            for idx, row in instructors_df.reset_index(drop=True).iterrows():
                print(f"[{idx}] {row['last_name']} {row['first_name']} (ID: {row['user_id']})")
            
            try:
                s_idx = int(input("\n先生の番号を選んでください (戻る場合は -1): "))
                if s_idx == -1:
                    continue
                sel_inst = instructors_df.reset_index(drop=True).loc[s_idx]
                i_id = sel_inst['user_id']
            except (ValueError, KeyError, IndexError):
                print("無効な番号です。")
                continue
            
            tmpl = template_dict["instructor_to_courses"]
            q, sql = render_template(tmpl, {"USER_ID": i_id})
            print(f"\n[DEBUG] 自然言語: {q}")
            print(f"[DEBUG] 実行SQL: {sql}")
            
            courses_res = pd.read_sql(sql, conn)
            select_course_from_list(conn, template_dict, normalizer, courses_res)
            
        else:
            print("無効な選択です。もう一度入力してください。")

if __name__ == "__main__":
    main()