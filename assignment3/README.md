# 健康諮詢小幫手（RAG）

## 專案簡介

本專案以 RAG（Retrieval-Augmented Generation）建立健康諮詢小幫手。系統會爬取健康醫療網新聞，使用地端 Ollama 模型產生摘要，將摘要轉為向量並建立 FAISS 索引，最後透過 Flask 網頁提供多輪問答。

## 功能

- 爬取健康醫療網新聞並去除重複連結
- 使用地端模型 `gemma3:4b` 產生新聞摘要
- 使用 `BAAI/bge-m3` 建立 FAISS 向量索引
- Flask 網頁支援多輪對話
- 可透過 checkbox 決定是否啟用新聞檢索
- 每次回答附上檢索到的新聞標題、連結與發布時間
- 可將對話中的問題整理成重點
- 可清除目前對話歷史

## 資料來源與資料量

- 來源：健康醫療網新聞分類頁（`crawl_data.py`）
- 資料量：30 篇不重複新聞
- 摘要：由地端 Ollama `gemma3:4b` 產生
- 為避免重複資料：
  - `init_db.py` 每次執行都會重建 `news` 資料表
  - `link` 欄位設為 UNIQUE
  - `make_index.py` 每次執行都會重建 `vector.index`，不會累加舊向量

正常重建後應為：

- `news.db`：30 筆資料
- `vector.index`：30 個向量

## 系統架構

```text
健康醫療網
   ↓ crawl_data.py
news_data.json
   ↓ summarize.py
news_data.json（含摘要）
   ↓ init_db.py
news.db
   ↓ make_index.py + BAAI/bge-m3
vector.index
   ↓ app.py + Flask + Ollama gemma3:4b
網頁問答介面
```

## 使用模型

- 問答與摘要：Ollama `gemma3:4b`
- 向量編碼：`BAAI/bge-m3`（sentence-transformers）

## 資料表設計

| 欄位 | 說明 |
|------|------|
| id | 自動遞增主鍵 |
| title | 新聞標題 |
| link | 新聞連結（UNIQUE） |
| content | 爬取的文章全文 |
| summary | 由地端模型產生的摘要 |
| created_at | 文章發布時間 |

## 安裝套件

需求：Python 3.10+、Ollama。

```bash
ollama pull gemma3:4b
pip install -r requirements.txt
```

## 執行方法

```bash
python crawl_data.py      # 重新爬取新聞（可選）
python init_db.py         # 重建 news.db
python summarize.py       # 重新產生摘要（需 Ollama）
python make_index.py      # 重建 vector.index
python app.py             # 啟動 Flask
```

啟動後開啟 `http://127.0.0.1:5000`。

> 若已有 `news_data.json`，可跳過 `crawl_data.py` 與 `summarize.py`，直接執行 `init_db.py` 與 `make_index.py`。

## 網頁使用方式

1. 勾選「進行檢索」時，系統會先檢索新聞，再將檢索結果交給模型回答。
2. 取消勾選時，系統只使用對話歷史回答。
3. 每次回答下方會列出本次檢索到的新聞標題、連結與時間。
4. 點擊「整理重點」會整理目前對話中的所有問題，並附上相關新聞來源。
5. 點擊「清除對話」會清空目前的多輪對話歷史。

## 遠端存取

Flask 啟動後，可使用 Cloudflare Tunnel：

```bash
python tunnel.py
```

或手動執行：

```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:5000
```

## 成果

- 爬取 30 篇健康醫療網新聞文章
- 使用 `BAAI/bge-m3` 建立 FAISS 向量索引
- 網頁支援 checkbox 決定是否檢索
- 每次回答附上新聞標題、連結與新增時間
- 支援多輪對話、重點整理與清除對話

## 截圖與操作影片

> 待補：執行畫面截圖與操作影片連結。

## 備註

- `news.db` 與 `vector.index` 為可重建的產物；若重新執行 `init_db.py` 與 `make_index.py`，內容會還原為 30 筆不重複資料。
- 多輪對話歷史儲存於記憶體，重新啟動 Flask 後會清空。
