from flask import Flask, render_template, request, jsonify
import sqlite3
import ollama
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

ENCODER = SentenceTransformer("BAAI/bge-m3")
INDEX = faiss.read_index("vector.index")
MODEL = "gemma3:4b"
RETRIEVAL_K = 5
MIN_SCORE = 0.55
history = []


def retrieve(query, k=RETRIEVAL_K, min_score=MIN_SCORE):
    emb = ENCODER.encode([query], normalize_embeddings=True)
    D, I = INDEX.search(emb, k)
    conn = sqlite3.connect("news.db")
    rows = []
    for score, did in zip(D[0], I[0]):
        if did == -1 or float(score) < min_score:
            continue
        r = conn.execute("SELECT title, link, created_at, summary FROM news WHERE id=?", (int(did),)).fetchone()
        if r:
            rows.append({"title": r[0], "link": r[1], "created_at": r[2], "summary": r[3]})
    conn.close()
    return rows


def unique_sources(sources):
    seen = set()
    rows = []
    for source in sources:
        if source["link"] in seen:
            continue
        seen.add(source["link"])
        rows.append(source)
    return rows


def source_text(sources):
    return "\n".join(f"- {s['title']} | {s['link']} | {s['created_at']}" for s in sources)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    history.clear()
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data["question"]
    use_retrieval = bool(data.get("retrieve", True))
    sources = retrieve(question) if use_retrieval else []
    context = "\n".join(f"- {s['title']}: {s['summary']}" for s in sources)
    if use_retrieval and not sources:
        return jsonify({
            "answer": "目前檢索不到與問題足夠相關的新聞，請換個問法或補充關鍵字。",
            "sources": [],
        })
    system_prompt = "你是健康諮詢小幫手，以繁體中文回答。"
    if context:
        system_prompt += "只根據提供的參考資料回答；若資料不足，請明確說明，不要自行推測。"
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history:
        msgs.append(h)
    user_content = question if not context else f"參考資料：\n{context}\n\n問題：{question}"
    msgs.append({"role": "user", "content": user_content})
    resp = ollama.chat(model=MODEL, messages=msgs)
    answer = resp["message"]["content"].strip()
    if sources:
        answer += "\n\n參考新聞：\n" + source_text(sources)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return jsonify({"answer": answer, "sources": sources})


@app.route("/summarize", methods=["POST"])
def summarize():
    qs = [h["content"] for h in history if h["role"] == "user"]
    if not qs:
        return jsonify({"summary": "目前沒有可整理的問題。", "sources": []})

    sources = unique_sources(
        source
        for question in qs
        for source in retrieve(question)
    )
    context = "\n".join(f"- {s['title']}: {s['summary']}" for s in sources)
    prompt = (
        "請根據以下對話問題與參考資料，用繁體中文整理重點並回答。"
        "只使用參考資料中的內容；若資料不足，請明確說明，不要自行推測。\n"
        f"問題：\n" + "\n".join(qs)
    )
    if context:
        prompt += f"\n\n參考資料：\n{context}"

    resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    answer = resp["message"]["content"].strip()
    if sources:
        answer += "\n\n參考新聞：\n" + source_text(sources)
    history.append({"role": "assistant", "content": answer})
    return jsonify({"summary": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True)