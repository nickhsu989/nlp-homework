import json
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.healthnews.com.tw"
CATEGORY_URLS = [
    "https://www.healthnews.com.tw/news/article/category/1",
    "https://www.healthnews.com.tw/news",
    "https://www.healthnews.com.tw",
]
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_ARTICLES = 30


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.encoding = r.apparent_encoding
    return r.text


def parse_article(url):
    html = fetch(url)
    soup = BeautifulSoup(html, "lxml")
    title = ""
    t = soup.find("h1")
    if t:
        title = t.get_text(strip=True)
    content = ""
    for sel in [".article-content", ".entry-content", "article", ".content"]:
        node = soup.select_one(sel)
        if node:
            content = node.get_text("\n", strip=True)
            break
    if not content:
        content = soup.get_text("\n", strip=True)
    date = ""
    m = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", html)
    if m:
        date = m.group(0)
    return {"title": title, "link": url, "content": content, "created_at": date}


def collect_links():
    links = set()
    for cu in CATEGORY_URLS:
        try:
            html = fetch(cu)
        except Exception as e:
            print("fetch fail", cu, e)
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/article/" in href or "/news/" in href:
                links.add(urljoin(BASE, href))
        print(cu, "-> links so far:", len(links))
    return list(links)


def main():
    links = collect_links()
    # dedupe by path
    seen = set()
    uniq = []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    uniq = uniq[:MAX_ARTICLES]
    print("total unique:", len(uniq))

    news_data = []
    for url in uniq:
        try:
            item = parse_article(url)
            if item["title"]:
                news_data.append(item)
            time.sleep(0.5)
        except Exception as e:
            print("skip", url, e)

    with open("news_data.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)
    print(f"saved {len(news_data)} articles")


if __name__ == "__main__":
    main()