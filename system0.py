import sqlite3
import pandas as pd
import os
import json

def load_templates(json_path="query_templates0.json"):
    """外部のJSONファイルからテンプレートを読み込む"""
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

def main():
    # テンプレートの読み込み
    try:
        query_templates = load_templates("query_templates0.json")
    except Exception as e:
        print(f"エラー: {e}")
        return

    conn = init_db()
    
    print("--- 授業情報検索システム0 (JSONテンプレート駆動版) ---")
    
    # 授業一覧の取得
    courses_df = pd.read_sql("SELECT course_id, title FROM Course WHERE course_id IS NOT NULL AND title IS NOT NULL", conn)
    
    for idx, row in courses_df.iterrows():
        print(f"[{idx}] {row['course_id']}: {row['title']}")
    
    # 1. ユーザが授業を選ぶ
    try:
        selected_idx = int(input("\n知りたい授業の番号を選んでください: "))
        target_course_id = courses_df.loc[selected_idx, 'course_id']
        target_title = courses_df.loc[selected_idx, 'title']
    except (ValueError, KeyError):
        print("無効な番号です。")
        return

    print(f"\n選択された授業: {target_title} ({target_course_id})")

    # 2. ユーザが知りたい項目（テンプレートの種類）を選択
    print("\n何を知りたいですか？")
    for idx, tmpl in enumerate(query_templates):
        print(f"{idx + 1}. {tmpl['menu_name']}")
    
    try:
        choice_idx = int(input("番号を選択してください: ")) - 1
        selected_template = query_templates[choice_idx]
    except (ValueError, IndexError):
        print("無効な選択です。")
        return

    # 3. テンプレートから自然言語の質問を復元
    generated_question = selected_template["question"].replace("[COURSE_ID]", target_course_id)
    
    # 4. スロットに値をバインドしてSQLを生成
    executable_sql = selected_template["sql"].replace("[COURSE_ID]", target_course_id)    
    
    # デバッグ表示
    print(f"\n[DEBUG] 対応する自然言語: {generated_question}")
    print(f"[DEBUG] 実行SQL: {executable_sql}")
    
    res = pd.read_sql(executable_sql, conn)
    
    # 5. 結果の表示
    print("\n【検索結果】")
    if res.empty:
        print("該当する情報が見つかりませんでした。")
    else:
        print(res.to_string(index=False))

if __name__ == "__main__":
    main()
