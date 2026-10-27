import os
import sqlite3
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model_name = 'BAAI/bge-m3'
index_path = './vector.index'

encoder = SentenceTransformer(model_name)

docs = []
doc_ids = []
conn = sqlite3.connect('./news.db')
cursor = conn.cursor()
try:
    rows = cursor.execute('SELECT id, title, summary, content FROM news')
    for news_id, title, summary, content in rows:
        text = summary if summary else f"{title} {content}"
        docs.append(f'{title} {text}')
        doc_ids.append(news_id)
finally:
    conn.close()

if not docs:
    raise RuntimeError('news.db is empty; run init_db.py first')

doc_ids = np.array(doc_ids, dtype='int64')

embeddings = encoder.encode(
    docs,
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=True
)

index = faiss.IndexFlatIP(embeddings.shape[1])
index = faiss.IndexIDMap(index)
index.add_with_ids(embeddings, doc_ids)

if os.path.exists(index_path):
    os.remove(index_path)
faiss.write_index(index, index_path)

print(embeddings.shape)
print(len(doc_ids))
print("索引已儲存到", index_path)