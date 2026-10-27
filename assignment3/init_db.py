import sqlite3
import json

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS news")
    cursor.execute("""
        CREATE TABLE news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            link TEXT NOT NULL UNIQUE,
            content TEXT,
            summary TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_news_data(news_data, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    seen_links = set()
    for news in news_data:
        link = news["link"]
        if link in seen_links:
            continue
        seen_links.add(link)
        cursor.execute("""
            INSERT INTO news (title, link, content, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (news["title"], link, news["content"], news.get("summary",""), news.get("created_at","")))
    conn.commit()
    count = cursor.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.close()
    return count

if __name__ == "__main__":
    with open("news_data.json", "r", encoding="utf-8") as f:
        news_data = json.load(f)
    create_database("news.db")
    count = insert_news_data(news_data, "news.db")
    print(f"db ready: {count} articles")