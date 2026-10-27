import json
import ollama

MODEL = "gemma3:4b"

with open("news_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    prompt = "請用繁體中文摘要以下新聞，100字內：\n標題：" + item["title"] + "\n內容：" + item["content"]
    try:
        resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        item["summary"] = resp["message"]["content"].strip()
    except Exception as e:
        item["summary"] = item["title"]
        print("err", e)

with open("news_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

import sqlite3
conn = sqlite3.connect("news.db")
cur = conn.cursor()
for item in data:
    cur.execute("UPDATE news SET summary=? WHERE link=?", (item["summary"], item["link"]))
conn.commit()
conn.close()
print("done")