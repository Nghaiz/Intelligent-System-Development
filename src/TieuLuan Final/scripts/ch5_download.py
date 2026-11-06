"""Tải dữ liệu thô Chương 5 vào data/ch5/raw: Hetionet, giá VN30 (Yahoo), Wikipedia tiếng Việt, HotpotQA.

    python scripts/ch5_download.py [hetionet|yahoo|wiki|hotpot]
Mọi tệp tải về chỉ được đọc như dữ liệu; không có bước nào chạy mã từ nguồn ngoài. Tệp đã có thì bỏ qua.
"""
import gzip
import sys
import time

import pandas as pd
import requests

from tlf.ch5_data import END, RAW, START, VN30, write_jsonl

HET = "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/"
WIKI = "https://vi.wikipedia.org/w/api.php"
UA = {"User-Agent": "tieuluan-final-ptit/1.0 (bai tap hoc phan; lien he qua sinh vien)"}


def fetch(url, path):
    if path.exists():
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print("->", path.name, path.stat().st_size, flush=True)


def hetionet():
    fetch(HET + "hetionet-v1.0-nodes.tsv", RAW / "hetionet-nodes.tsv")
    fetch(HET + "hetionet-v1.0-edges.sif.gz", RAW / "hetionet-edges.sif.gz")
    with gzip.open(RAW / "hetionet-edges.sif.gz", "rt", encoding="utf-8") as f:
        print("cột cạnh:", f.readline().strip())


def yahoo():
    import yfinance as yf
    out = RAW / "vn30_prices.csv"
    if out.exists():
        return
    frames = []
    for tk, *_ in VN30:
        d = yf.Ticker(f"{tk}.VN").history(start=START, end=END, auto_adjust=True)
        if d.empty:
            print("!! rỗng:", tk, flush=True)
            continue
        d = d.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        d["Date"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
        d.insert(0, "Ticker", tk)
        frames.append(d)
        print(tk, len(d), d.Date.min(), d.Date.max(), flush=True)
        time.sleep(0.3)
    pd.concat(frames).to_csv(out, index=False)


def wiki_get(params, tries=6):
    """API Wikipedia trả HTML 429 khi gọi dồn dập: chờ theo Retry-After rồi thử lại, hết lượt thì báo lỗi."""
    for i in range(tries):
        r = requests.get(WIKI, params=params, headers=UA, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        wait = int(r.headers.get("retry-after", 0)) or 5 * 2 ** i
        print(f"[wiki] HTTP {r.status_code}, chờ {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"Wikipedia không trả lời sau {tries} lần: {params}")


def wiki_title(query):
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"}
    hits = wiki_get(params)["query"]["search"]
    return hits[0]["title"] if hits else None


def wiki():
    out = RAW / "vn30_wiki.jsonl"
    if out.exists():
        return
    rows = []
    for tk, name, _, query in VN30:
        title = wiki_title(query)
        if title is None:
            print("!! không thấy bài:", tk, flush=True)
            continue
        params = {"action": "query", "prop": "extracts|info|revisions", "explaintext": 1, "titles": title, "rvprop": "ids",
                  "inprop": "url", "format": "json", "formatversion": 2, "redirects": 1}
        page = wiki_get(params)["query"]["pages"][0]
        rows.append({"ticker": tk, "company": name, "title": page["title"], "url": page["fullurl"],
                     "revid": page["revisions"][0]["revid"], "text": page.get("extract", "")})
        print(tk, "->", page["title"], len(rows[-1]["text"]), flush=True)
        time.sleep(2)
    write_jsonl(out, rows)


def hotpot(n=3000):
    from datasets import load_dataset
    out = RAW / "hotpot_val.jsonl"
    if out.exists():
        return
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{n}]",
                      cache_dir=str(RAW / "hf_cache"))
    rows = []
    for r in ds:
        rows.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                     "level": r["level"], "support": sorted(set(r["supporting_facts"]["title"])),
                     "paragraphs": [{"title": t, "text": " ".join(s)} for t, s in
                                    zip(r["context"]["title"], r["context"]["sentences"])]})
    write_jsonl(out, rows)
    print("hotpot", len(rows), flush=True)


if __name__ == "__main__":
    steps = {"hetionet": hetionet, "yahoo": yahoo, "wiki": wiki, "hotpot": hotpot}
    for name in sys.argv[1:] or steps:
        steps[name]()
