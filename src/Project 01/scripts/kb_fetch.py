"""Thu thập kho tri thức đái tháo đường vào data/kb/raw/<nguồn>.jsonl, mỗi nguồn một luồng, có tôn trọng robots.txt.

Chạy: python scripts/kb_fetch.py [--only wiki,epmc] [--limit 5]. Văn bản web là DỮ LIỆU, không bao giờ là chỉ dẫn.
"""
import argparse
import io
import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import DATA, save_json  # noqa: E402
from dnx.mem import require_disk  # noqa: E402

RAW = DATA / "kb" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = "DiaNexus-KB/1.0 (academic research project)"
# Chính sách Wikimedia đòi User-Agent có địa chỉ liên hệ (URL hoặc email); UA thiếu liên hệ nhận 429 liên tục. Dùng URL repo, không dùng email.
WIKI_UA = "DiaNexusKB/1.0 (https://github.com/Nghaiz/Intelligent-System-Development) python-requests"
TODAY = time.strftime("%Y-%m-%d")
DELAY = {"default": 1.0, "vi.wikipedia.org": 1.1, "wsearch.nlm.nih.gov": 0.8, "www.ebi.ac.uk": 0.34}
EPMC_TARGET = 22000
EPMC_QUERY = ('(TITLE:"diabetes" OR MESH:"Diabetes Mellitus") AND HAS_ABSTRACT:y AND SRC:MED AND LANG:eng '
              'AND PUB_YEAR:[2010 TO 2023]')
WIKI_TERMS = """đái tháo đường|tiểu đường|đái tháo đường type 2|đái tháo đường type 1|đái tháo đường thai kỳ|insulin|
glucose máu|hạ đường huyết|tăng đường huyết|HbA1c|hemoglobin A1c|nghiệm pháp dung nạp glucose|bệnh võng mạc đái tháo đường|
bệnh thận đái tháo đường|bệnh thần kinh đái tháo đường|bàn chân đái tháo đường|nhiễm toan ceton|metformin|sulfonylurea|
thuốc hạ đường huyết|chất ức chế SGLT2|chất chủ vận GLP-1|chất ức chế DPP-4|thiazolidinedione|béo phì|chỉ số khối cơ thể|
hội chứng chuyển hóa|tăng huyết áp|xơ vữa động mạch|bệnh mạch vành|đột quỵ|suy thận mạn|mỡ máu cholesterol|triglyceride|
tuyến tụy|tế bào beta|kháng insulin|chỉ số đường huyết|chế độ ăn|chất xơ|carbohydrate|đường fructose|hoạt động thể chất|
hút thuốc lá|glucagon|lọc máu|mù lòa|đục thủy tinh thể|cắt cụt chi|protein niệu""".replace("\n", "").split("|")
WIKI_KEY = re.compile(r"đái tháo đường|tiểu đường|insulin|glucose|đường huyết|đường máu|chuyển hóa|béo phì|huyết áp|"
                      r"cholesterol|tim mạch|tụy|dinh dưỡng", re.I)
MED_TERMS = ["diabetes", "diabetes type 1", "diabetes type 2", "gestational diabetes", "prediabetes", "insulin",
             "hemoglobin A1c", "blood glucose", "hypoglycemia", "hyperglycemia", "diabetic eye problems",
             "diabetic kidney problems", "diabetic nerve problems", "diabetic foot", "diabetes medicines",
             "diabetic diet", "metabolic syndrome", "obesity", "high blood pressure", "cholesterol"]
MOH_PDFS = [("Hướng dẫn chẩn đoán và điều trị đái tháo đường típ 2 (3280/QĐ-BYT, 2011)",
             "https://kcb.vn/upload/2005611/20210723/8e4a6b72bc14089e7c9028042a34193f3280_Q-BYT.pdf"),
            ("Quy trình lâm sàng chẩn đoán và điều trị đái tháo đường típ 2",
             "https://files01.duytan.edu.vn/svruploads/kdieuduong-duytan/upload/2024/3/31/Quy trình lâm sàng chẩn đoán và điều trị Đái tháo đường típ 2.pdf"),
            ("Tài liệu Sở Y tế Bắc Giang về đái tháo đường típ 2",
             "https://syt.bacgiang.gov.vn/documents/21459/10381054/120210224081543.pdf/0885ce05-fa4d-4af6-b9e1-5b2fae74fb78"),
            ("Hướng dẫn chẩn đoán và điều trị tiền đái tháo đường (3087/QĐ-BYT)",
             "https://syt.bacgiang.gov.vn/documents/21459/17049790/1687527717201_3087_QD-BYT_Huong+dan+chan+doan+va+dieu+tri+tien+dai+thao+duong.pdf/11b97040-ca12-4a3c-8f84-18cbcc511647"),
            ("Hướng dẫn chẩn đoán và điều trị đái tháo đường típ 1 (Bộ Y tế)",
             "https://syt.bacgiang.gov.vn/documents/21459/21336928/1719415239778_HD_chan_doan_dieu_tri_DTD_typs_1_f5ded58eaa.pdf/4df7bc0d-7d72-4f4c-b21f-5915358d4d11")]
VI_KEY = re.compile(r"diabet|blood sugar|insulin|a1c|glucose|hypoglyc|kidney|cholesterol|high blood pressure|nutrition", re.I)
WHO_PAGES = ["news-room/fact-sheets/detail/diabetes", "health-topics/diabetes", "news-room/fact-sheets/detail/hypertension",
             "news-room/fact-sheets/detail/obesity-and-overweight", "news-room/fact-sheets/detail/healthy-diet",
             "news-room/fact-sheets/detail/physical-activity", "news-room/fact-sheets/detail/tobacco",
             "news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
             "news-room/fact-sheets/detail/noncommunicable-diseases", "news-room/fact-sheets/detail/blindness-and-visual-impairment",
             "news-room/fact-sheets/detail/malnutrition", "news-room/fact-sheets/detail/stroke"]
WHO_PDF_PAGES = ["publications/i/item/classification-of-diabetes-mellitus", "publications/i/item/who-ucn-ncd-20.1"]

_robots, _last, _lock = {}, {}, threading.Lock()
_sess = requests.Session()
_sess.headers["User-Agent"] = UA
_status = {}


def allowed(url):
    host = urlparse(url).netloc
    with _lock:
        rp = _robots.get(host)
        if rp is None:
            rp = RobotFileParser()
            try:
                r = _sess.get(f"https://{host}/robots.txt", timeout=20)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])
            _robots[host] = rp
    return rp.can_fetch(UA, url)


def get(url, retries=3, api=False, **kw):
    """GET có giãn cách theo máy chủ; trả None nếu robots cấm hoặc máy chủ từ chối hẳn."""
    if not api and not allowed(url):
        print("[robots] bỏ", url, flush=True)
        return None
    host = urlparse(url).netloc
    for i in range(retries):
        with _lock:
            wait = _last.get(host, 0) + DELAY.get(host, DELAY["default"]) - time.time()
            _last[host] = time.time() + max(wait, 0)
        if wait > 0:
            time.sleep(wait)
        try:
            r = _sess.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                print(f"[{r.status_code}] {url}", flush=True)
                return None
            print(f"[{r.status_code}] thử lại {url}", flush=True)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 30)))
        except requests.RequestException as e:
            print(f"[lỗi] {url}: {type(e).__name__}", flush=True)
        time.sleep(3 * (i + 1))
    return None


def html_text(html):
    """Lấy phần chính của trang; tiêu đề thành dòng '== x ==', hàng bảng nối bằng ' | ' để giữ ngưỡng xét nghiệm."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "button"]):
        t.decompose()
    root = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body or soup
    out = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "dd"]):
        if el.name != "tr" and el.find_parent("tr"):
            continue
        if el.name == "p" and el.find_parent("li"):
            continue
        if el.name == "tr":
            s = " | ".join(c.get_text(" ", strip=True) for c in el.find_all(["th", "td"]))
        else:
            s = el.get_text(" ", strip=True)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out.append(f"== {s} ==" if el.name[0] == "h" and el.name != "hr" else s)
    return "\n".join(dict.fromkeys(out))


def links(html, base):
    soup = BeautifulSoup(html, "lxml")
    return [urldefrag(urljoin(base, a["href"]))[0] for a in soup.find_all("a", href=True)]


def doc(source, did, title, url, lang, text, **meta):
    return {"doc_id": f"{source}:{did}", "source": source, "title": title, "url": url, "lang": lang, "text": text,
            "fetched": TODAY, "meta": meta}


def write(source, rows):
    with open(RAW / f"{source}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{source}] ghi {len(rows)} tài liệu", flush=True)


def fetch_wiki(limit):
    api = "https://vi.wikipedia.org/w/api.php"
    titles = {}
    for q in WIKI_TERMS[:limit or None]:
        r = get(api, api=True, headers={"User-Agent": WIKI_UA}, params={"action": "query", "list": "search", "srsearch": q, "srlimit": 8, "format": "json"})
        for h in (r.json()["query"]["search"] if r else []):
            titles.setdefault(h["title"], q)
    rows = []
    for t in titles:
        r = get(api, api=True, headers={"User-Agent": WIKI_UA}, params={"action": "query", "prop": "extracts|info", "explaintext": 1, "inprop": "url",
                                       "exsectionformat": "wiki", "redirects": 1, "titles": t, "format": "json"})
        p = next(iter(r.json()["query"]["pages"].values())) if r else {}
        text = p.get("extract", "")
        if len(text) >= 600 and WIKI_KEY.search(text[:4000]):
            rows.append(doc("wikipedia_vi", p["pageid"], p["title"], p["fullurl"], "vi", text))
    return {r["doc_id"]: r for r in rows}.values()


def fetch_medline(limit):
    urls = {}
    for term in MED_TERMS[:limit or None]:
        r = get("https://wsearch.nlm.nih.gov/ws/query", params={"db": "healthTopics", "term": term, "retmax": 10})
        if r:
            for u in re.findall(r'url="(https://medlineplus\.gov/[^"]+)"', r.text):
                urls.setdefault(u, 1)
    rows, seen, extra = [], set(), []
    for u in list(urls):
        r = get(u)
        if not r:
            continue
        seen.add(u)
        rows.append(doc("medlineplus", u.rsplit("/", 1)[-1], BeautifulSoup(r.text, "lxml").title.get_text().split("|")[0].strip(),
                        u, "en", html_text(r.text)))
        extra += [x for x in links(r.text, u) if re.search(r"medlineplus\.gov/(lab-tests/|ency/article/|druginfo/meds/)", x)]
    for u in list(dict.fromkeys(extra))[:(limit or 150)]:
        r = get(u) if u not in seen else None
        if r:
            title = (BeautifulSoup(r.text, "lxml").title.get_text() or u).split(":")[0].strip()
            rows.append(doc("medlineplus", urlparse(u).path.strip("/").replace("/", "_"), title, u, "en", html_text(r.text)))
    return rows


def fetch_niddk(limit):
    prefix = "https://www.niddk.nih.gov/health-information/diabetes"
    queue = [prefix, "https://www.niddk.nih.gov/health-information/professionals/clinical-tools-patient-management/diabetes"]
    seen, rows, pdfs = set(), [], []
    while queue and len(rows) < (limit or 130):
        u = queue.pop(0)
        if u in seen:
            continue
        seen.add(u)
        r = get(u)
        if not r or "html" not in r.headers.get("content-type", ""):
            continue
        rows.append(doc("niddk", urlparse(u).path.strip("/").replace("/", "_"),
                        BeautifulSoup(r.text, "lxml").title.get_text().split("|")[0].split(" - NIDDK")[0].strip(), u, "en", html_text(r.text)))
        for x in links(r.text, u):
            if x.startswith(("https://www.niddk.nih.gov/health-information/diabetes", "https://www.niddk.nih.gov/health-information/professionals/clinical-tools-patient-management/diabetes")):
                queue.append(x.split("?")[0])
    return rows


def fetch_cdc(limit):
    """www.cdc.gov trả 403 cho công cụ thu thập; chỉ dùng API phân phối công khai tools.cdc.gov, không né chặn."""
    api = "https://tools.cdc.gov/api/v2/resources/media"
    rows, ids = [], {}
    for q in ("diabetes", "prediabetes", "insulin", "blood sugar", "A1C"):
        r = get(api, api=True, params={"q": q, "max": 40, "mediatype": "Html", "language": "English", "format": "json"})
        for m in (r.json().get("results", []) if r else []):
            ids[m["id"]] = m
    for i, m in list(ids.items())[:limit or 80]:
        r = get(f"{api}/{i}/syndicate.json", api=True)
        html = (r.json().get("results") or {}).get("content", "") if r else ""
        text = html_text(html) if html else ""
        if len(text) > 300 and re.search("diabet|insulin|blood sugar", text, re.I):
            rows.append(doc("cdc", i, m["name"], m.get("sourceUrl") or m.get("url"), "en", text))
    return rows


def pdf_text(content):
    import fitz
    with fitz.open(stream=io.BytesIO(content), filetype="pdf") as d:
        return "\n".join(p.get_text() for p in d), len(d)


def fetch_who(limit):
    rows = []
    for p in WHO_PAGES[:limit or None]:
        u = f"https://www.who.int/{p}"
        r = get(u)
        if r:
            rows.append(doc("who", p.rsplit("/", 1)[-1], BeautifulSoup(r.text, "lxml").title.get_text().strip(), u, "en", html_text(r.text)))
    for p in WHO_PDF_PAGES:
        r = get(f"https://www.who.int/{p}")
        if not r:
            continue
        for x in dict.fromkeys(y for y in links(r.text, r.url) if ".pdf" in y.lower() or "bitstream" in y):
            q = get(x, api=True)
            if q and q.content[:4] == b"%PDF":
                text, pages = pdf_text(q.content)
                rows.append(doc("who", "pdf_" + p.rsplit("/", 1)[-1], p.rsplit("/", 1)[-1].replace("-", " "), x, "en", text, pages=pages))
                break
    return rows


def fetch_pdfs(limit):
    """Bộ Y tế và tờ hướng dẫn tiếng Việt do MedlinePlus dẫn tới. PDF ảnh quét không có lớp chữ thì ghi nhận và bỏ."""
    rows, skipped = [], []
    for title, u in MOH_PDFS[:limit or None]:
        r = get(u, api=True)
        if not r or r.content[:4] != b"%PDF":
            skipped.append({"url": u, "why": "không tải được PDF"})
            continue
        text, pages = pdf_text(r.content)
        if len(text) < 500:
            skipped.append({"url": u, "why": f"PDF ảnh quét {pages} trang, không có lớp chữ"})
            continue
        rows.append(doc("bo_y_te", re.sub(r"\W+", "_", title)[:40], title, u, "vi", text, pages=pages))
    r = get("https://medlineplus.gov/languages/vietnamese.html")
    found = re.findall(r'href="([^"]+\.pdf|https://www\.fda\.gov/media/[^"]+)"[^>]*title="([^"]+Vietnamese\)[^"]*)"', r.text if r else "")
    picked = list({u: t for u, t in found if VI_KEY.search(t)}.items())[:limit or 60]
    for u, t in picked:
        q = get(u, api=True)
        if q and q.content[:4] == b"%PDF":
            text, pages = pdf_text(q.content)
            title = t.split(" - ")[0].strip()
            if len(text) >= 200:
                rows.append(doc("medlineplus_vi", re.sub(r"\W+", "_", title)[:40], title, u, "vi", text, pages=pages))
    _status["pdf_skipped"] = skipped
    return rows


def fetch_epmc(limit):
    """Europe PMC, bền với gián đoạn: mỗi trang ghi nối và lưu con trỏ."""
    path, ckpt = RAW / "europepmc.jsonl", RAW / "europepmc.cursor"
    target = limit * 100 if limit else EPMC_TARGET
    n, cursor = 0, "*"
    if ckpt.exists() and path.exists():
        cursor = ckpt.read_text(encoding="utf-8").strip()
        n = sum(1 for _ in open(path, encoding="utf-8"))
    else:
        path.unlink(missing_ok=True)
    total = None
    while n < target:
        r = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", retries=6, api=True,
                params={"query": EPMC_QUERY, "resultType": "core", "pageSize": 1000 if not limit else 100,
                        "format": "json", "cursorMark": cursor})
        if r is None:
            raise RuntimeError(f"Europe PMC dừng ở {n} bản ghi")
        d = r.json()
        total = d.get("hitCount")
        with open(path, "a", encoding="utf-8") as f:
            for a in d["resultList"]["result"]:
                mesh = (a.get("meshHeadingList") or {}).get("meshHeading") or []
                if not mesh or not a.get("abstractText") or not a.get("pmid"):
                    continue
                n += 1
                text = BeautifulSoup(a["abstractText"], "lxml").get_text(" ", strip=True)
                f.write(json.dumps(doc("europepmc", a["pmid"], a.get("title", "").strip(), f"https://europepmc.org/article/MED/{a['pmid']}",
                                       "en", text, pmid=a["pmid"], year=a.get("pubYear"), journal=(a.get("journalInfo") or {}).get("journal", {}).get("title"),
                                       mesh=[{"name": m["descriptorName"], "major": m.get("majorTopic_YN") == "Y"} for m in mesh],
                                       keywords=(a.get("keywordList") or {}).get("keyword", [])), ensure_ascii=False) + "\n")
        next_c = d.get("nextCursorMark")
        ckpt.write_text(next_c or "", encoding="utf-8")
        print(f"[europepmc] {n}/{target} (hit {total})", flush=True)
        if not next_c or next_c == cursor:
            break
        cursor = next_c
    _status["epmc_hits"] = total
    return None


SOURCES = {"wiki": ("wikipedia_vi", fetch_wiki), "medline": ("medlineplus", fetch_medline), "niddk": ("niddk", fetch_niddk),
           "cdc": ("cdc", fetch_cdc), "who": ("who", fetch_who), "pdf": ("pdf", fetch_pdfs), "epmc": ("europepmc", fetch_epmc)}


def run(key, limit, errors):
    name, fn = SOURCES[key]
    try:
        rows = fn(limit)
        if rows is None:
            return
        rows = list(rows)
        if key == "pdf":
            for s in ("bo_y_te", "medlineplus_vi"):
                write(s, [r for r in rows if r["source"] == s])
        else:
            write(name, rows)
    except Exception:
        errors[key] = traceback.format_exc()
        print(f"[{key}] LỖI\n{errors[key]}", flush=True)


def manifest():
    """Đếm lại từ tệp thô: số trong kb_sources.json là số đọc được thật, không phải số dự định."""
    out = {}
    for f in sorted(RAW.glob("*.jsonl")):
        rows = [json.loads(x) for x in open(f, encoding="utf-8")]
        hosts = sorted({urlparse(r["url"]).netloc for r in rows if r.get("url")})
        out[f.stem] = {"docs": len(rows), "chars": sum(len(r["text"]) for r in rows), "lang": sorted({r["lang"] for r in rows}),
                       "hosts": hosts, "fetched": sorted({r["fetched"] for r in rows}), "example_url": rows[0]["url"] if rows else None}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SOURCES))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    require_disk(1, "kb_fetch")
    errors = {}
    threads = [threading.Thread(target=run, args=(k, a.limit, errors)) for k in a.only.split(",")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    notes = {"wikipedia_user_agent": "Wikipedia dùng User-Agent riêng chứa URL repo theo chính sách Wikimedia, tốc độ ~1 yêu cầu/s.", "pubmed_eutils": "Không dùng: eutils trả 302 về trang misuse.ncbi của NCBI khi thăm dò ngày " + TODAY + "; dùng Europe PMC REST.",
             "cdc_www": "www.cdc.gov (kể cả robots.txt) trả 403 cho thu thập; chỉ dùng API phân phối tools.cdc.gov."}
    save_json("kb_sources", {"date": TODAY, "user_agent": UA, "epmc_query": EPMC_QUERY, "sources": manifest(), "errors": errors,
                             "status": _status, "notes": notes})
    print("XONG", "lỗi:" if errors else "", list(errors), flush=True)
