"""Chương 5: đổi dữ liệu thô thành node, cạnh, chunk cho Neo4j. Mỗi hàm build_* trả dict rows, không chạm CSDL."""
import gzip
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .ch5_chunk import recursive_chunks
from .ch5_data import RAW, VN30, read_jsonl
from .ch5_schema import DATASETS

SEEDS = ("Disease::DOID:9352", "Disease::DOID:9744")   # đái tháo đường type 2, type 1
# ký hiệu Hetionet -> (kiểu cạnh Neo4j, cụm tiếng Việt chiều xuôi, cụm chiều ngược)
METAEDGES = {
    "CtD": ("TREATS", "điều trị", "được điều trị bởi"), "CpD": ("PALLIATES", "giảm nhẹ", "được giảm nhẹ bởi"),
    "DaG": ("ASSOCIATES", "liên quan tới gen", "liên quan tới bệnh"),
    "DuG": ("UPREGULATES", "tăng biểu hiện gen", "được tăng biểu hiện trong bệnh"),
    "DdG": ("DOWNREGULATES", "giảm biểu hiện gen", "bị giảm biểu hiện trong bệnh"),
    "DlA": ("LOCALIZES", "khu trú ở", "là vị trí của bệnh"), "DpS": ("PRESENTS", "có triệu chứng", "là triệu chứng của"),
    "DrD": ("RESEMBLES", "tương tự bệnh", "tương tự bệnh"), "CrC": ("RESEMBLES", "tương tự chất", "tương tự chất"),
    "CbG": ("BINDS", "liên kết với gen", "được liên kết bởi hợp chất"),
    "CcSE": ("CAUSES", "gây tác dụng phụ", "là tác dụng phụ của"),
    "PCiC": ("INCLUDES", "bao gồm hợp chất", "thuộc nhóm dược lý"),
    "GpPW": ("PARTICIPATES", "tham gia đường truyền", "có gen thành viên"),
    "GpMF": ("PARTICIPATES", "có chức năng phân tử", "là chức năng của gen"),
    "GpCC": ("PARTICIPATES", "nằm ở thành phần tế bào", "chứa gen"),
    "GiG": ("INTERACTS", "tương tác với gen", "tương tác với gen"),
    "AeG": ("EXPRESSES", "biểu hiện gen", "biểu hiện ở mô"), "AuG": ("UPREGULATES", "tăng biểu hiện gen", "tăng ở mô"),
    "AdG": ("DOWNREGULATES", "giảm biểu hiện gen", "giảm ở mô"),
    "Gr>G": ("REGULATES", "điều hòa gen", "được điều hòa bởi gen"), "GcG": ("COVARIES", "đồng biến với gen", "đồng biến với gen"),
    "CuG": ("UPREGULATES", "tăng biểu hiện gen", "được tăng bởi hợp chất"),
    "CdG": ("DOWNREGULATES", "giảm biểu hiện gen", "bị giảm bởi hợp chất"),
}
# cạnh đi ra khỏi tập hạt: bỏ GpBP, AeG, Gr>G... vì hàng chục nghìn cạnh mà ít giá trị cho câu hỏi về thuốc và bệnh
OUTWARD = {"GpPW", "GpMF", "GpCC", "CbG", "CcSE", "DaG", "GiG", "CtD", "CpD", "DlA", "DpS", "CrC", "PCiC", "DuG", "DdG"}
DOC_KINDS = {"Disease", "Compound", "Symptom", "Pathway", "Anatomy", "PharmacologicClass"}
KIND_VN = {"Disease": "Bệnh", "Compound": "Hợp chất", "Gene": "Gen", "Symptom": "Triệu chứng", "Anatomy": "Mô, cơ quan",
           "Pathway": "Đường truyền", "SideEffect": "Tác dụng phụ", "PharmacologicClass": "Nhóm dược lý",
           "MolecularFunction": "Chức năng phân tử", "CellularComponent": "Thành phần tế bào"}
NAMES_PER_DOC = 70


def _label(kind):
    return kind.replace(" ", "")


def hetionet_edges():
    """Đọc 2,25 triệu cạnh, giữ lân cận của hai bệnh hạt theo ba luật: mọi cạnh chạm hạt, mọi cạnh nội bộ, OUTWARD."""
    with gzip.open(RAW / "hetionet-edges.sif.gz", "rt", encoding="utf-8") as f:
        next(f)
        rows = [line.rstrip("\n").split("\t") for line in f]
    hop1 = set(SEEDS)
    for s, _, t in rows:
        if s in SEEDS or t in SEEDS:
            hop1 |= {s, t}
    kept = []
    for s, m, t in rows:
        if m not in METAEDGES:
            continue
        inside = s in hop1 and t in hop1
        if inside or ((s in hop1 or t in hop1) and m in OUTWARD):
            kept.append((s, m, t))
    return kept, hop1, len(rows)


def _hetionet_docs(kept, names, hop1):
    """Mỗi cặp (nút, quan hệ) thành một chunk liệt kê hàng xóm; danh sách dài chia thành nhiều phần."""
    groups = defaultdict(list)
    for s, m, t in kept:
        groups[(s, m, 0)].append(t)
        groups[(t, m, 1)].append(s)
    rows = []
    for (node, m, rev), others in groups.items():
        kind = node.split("::")[0]
        if kind not in DOC_KINDS and node not in hop1:
            continue
        phrase = METAEDGES[m][1 + rev]
        vals = sorted(names[o] for o in others)
        for p in range(0, len(vals), NAMES_PER_DOC):
            part = vals[p:p + NAMES_PER_DOC]
            cid = f"het:{node}:{m}:{rev}:{p // NAMES_PER_DOC}"
            text = f"{KIND_VN.get(kind, kind)} {names[node]} {phrase}: {', '.join(part)}."
            rows.append({"id": cid, "label": DATASETS["hetionet"], "about": [f"H:{node}"],
                         "props": {"dataset": "hetionet", "title": f"{names[node]} ({m})", "text": text,
                                   "source": f"Hetionet v1.0, {m}"}})
    return rows


def build_hetionet():
    nodes = pd.read_csv(RAW / "hetionet-nodes.tsv", sep="\t")
    names = dict(zip(nodes["id"], nodes["name"].astype(str)))
    kinds = dict(zip(nodes["id"], nodes["kind"]))
    kept, hop1, total = hetionet_edges()
    used = {x for s, _, t in kept for x in (s, t)}
    rows = [{"id": f"H:{i}", "label": _label(kinds[i]), "props": {"name": names[i], "dataset": "hetionet", "hid": i}}
            for i in sorted(used)]
    edges = [{"source": f"H:{s}", "target": f"H:{t}", "type": METAEDGES[m][0], "props": {"metaedge": m}}
             for s, m, t in kept]
    return {"nodes": rows, "edges": edges, "chunks": _hetionet_docs(kept, names, hop1), "stats": [], "mentions": [],
            "meta": {"edges_full": total, "seed_hop1": len(hop1), "edges_kept": len(kept), "nodes_kept": len(used)}}


# ---------------------------------------------------------------- VN30
def fmt_pct(x):
    return f"{x:+.2f}".replace(".", ",") + "%"


def fmt_int(x):
    return f"{x:,.0f}".replace(",", ".")


def company_frame():
    df = pd.read_csv(RAW / "vn30_prices.csv", parse_dates=["Date"]).sort_values(["Ticker", "Date"])
    df["logret"] = df.groupby("Ticker")["Close"].transform(lambda s: np.log(s).diff())
    return df


def period_stats(g):
    """Thống kê một kỳ từ các phiên trong kỳ; lợi suất tính từ giá đóng cửa đầu kỳ tới cuối kỳ."""
    close = g["Close"]
    return {"ret_pct": round(100 * (close.iloc[-1] / close.iloc[0] - 1), 2), "high": round(float(g["High"].max()), 0),
            "low": round(float(g["Low"].min()), 0), "close": round(float(close.iloc[-1]), 0), "sessions": int(len(g)),
            "vol_pct": round(100 * float(g["logret"].std()) * np.sqrt(252), 2),
            "avg_volume": round(float(g["Volume"].mean()), 0)}


def correlations(df, min_rho=0.35):
    wide = df.pivot(index="Date", columns="Ticker", values="logret")
    rho = wide.corr(min_periods=250)
    pairs = []
    cols = list(rho.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = rho.loc[a, b]
            if pd.notna(r):
                pairs.append((a, b, round(float(r), 4), int(wide[[a, b]].dropna().shape[0])))
    return rho, pairs, [p for p in pairs if p[2] >= min_rho]


def _vn30_entities(years, months):
    rows = [{"id": "INDEX:VN30", "label": "MarketIndex", "props": {"name": "VN30", "dataset": "vn30"}}]
    for s in sorted({v[2] for v in VN30}):
        rows.append({"id": f"SECTOR:{s}", "label": "Sector", "props": {"name": s, "dataset": "vn30"}})
    for tk, name, _, _ in VN30:
        rows.append({"id": f"VN30:{tk}", "label": "Company", "props": {"name": name, "ticker": tk, "dataset": "vn30"}})
    rows += [{"id": f"YEAR:{y}", "label": "Year", "props": {"name": str(y), "year": int(y), "dataset": "vn30"}}
             for y in years]
    rows += [{"id": f"MONTH:{m}", "label": "Month", "props": {"name": m, "dataset": "vn30"}} for m in months]
    return rows


def _vn30_stats(df):
    """Các nút YearStat, MonthStat và câu văn của từng năm. Năm có dưới 100 phiên bị bỏ vì quá ít để so sánh."""
    stats, year_text = [], {}
    for (tk, y), g in df.groupby(["Ticker", df["Date"].dt.year]):
        if len(g) < 100:
            continue
        s = period_stats(g)
        stats.append({"id": f"{tk}:{y}", "label": "YearStat", "ticker": f"{tk}", "period": f"YEAR:{y}", "has": "HAS_YEAR",
                      "of": "OF_YEAR", "props": s | {"year": int(y), "ticker": tk}})
        months = []
        for m, gm in g.groupby(g["Date"].dt.month):
            sm = period_stats(gm)
            key = f"{y}-{m:02d}"
            stats.append({"id": f"{tk}:{key}", "label": "MonthStat", "ticker": tk, "period": f"MONTH:{key}", "has": "HAS_MONTH",
                          "of": "OF_MONTH", "props": sm | {"month": key, "ticker": tk}})
            months.append(f"tháng {m}: {fmt_pct(sm['ret_pct'])}")
        year_text[(tk, y)] = (s, months)
    return stats, year_text


def _vn30_stat_chunks(year_text, pairs, info):
    rows = []
    src = "Yahoo Finance, tính lại từ giá đóng cửa đã điều chỉnh"

    def add(cid, title, text, about):
        rows.append({"id": cid, "label": DATASETS["vn30"], "about": about,
                     "props": {"dataset": "vn30", "title": title, "text": text, "source": src}})
    for (tk, y), (s, months) in year_text.items():
        name, sector = info[tk]
        text = (f"Năm {y}, cổ phiếu {tk} ({name}, ngành {sector}) có lợi suất {fmt_pct(s['ret_pct'])}, độ biến động năm hóa "
                f"{fmt_pct(s['vol_pct'])[1:]}, giá cao nhất {fmt_int(s['high'])} đồng, thấp nhất {fmt_int(s['low'])} đồng, "
                f"đóng cửa cuối kỳ {fmt_int(s['close'])} đồng, {s['sessions']} phiên. Lợi suất theo tháng: "
                + ", ".join(months) + ".")
        add(f"year:{tk}:{y}", f"{tk} năm {y}", text, [f"VN30:{tk}", f"YEAR:{y}"])
    peers = defaultdict(list)
    for a, b, r, _ in pairs:
        peers[a].append((r, b))
        peers[b].append((r, a))
    for tk, ps in peers.items():
        top = ", ".join(f"{b} ({r:.2f})".replace(".", ",") for r, b in sorted(ps, reverse=True)[:8])
        name, sector = info[tk]
        add(f"corr:{tk}", f"Tương quan của {tk}", f"Các mã tương quan cao nhất với {tk} ({name}) theo lợi suất ngày: {top}.",
            [f"VN30:{tk}"])
    return rows


def _vn30_profiles(info):
    rows = []
    sectors = defaultdict(list)
    for tk, (_, s) in info.items():
        sectors[s].append(tk)
    for tk, (name, s) in info.items():
        peers = ", ".join(x for x in sorted(sectors[s]) if x != tk) or "không có mã nào khác"
        rows.append({"id": f"profile:{tk}", "label": DATASETS["vn30"], "about": [f"VN30:{tk}", f"SECTOR:{s}"],
                     "props": {"dataset": "vn30", "title": f"Hồ sơ {tk}", "source": "Danh mục tự khai báo",
                               "text": f"{tk} là mã của {name}, thuộc rổ VN30, ngành {s}. Cùng ngành: {peers}."}})
    return rows


def _wiki_chunks():
    rows, seen = [], set()
    for page in read_jsonl(RAW / "vn30_wiki.jsonl"):
        if page["title"] in seen:   # VRE không có bài riêng, tìm kiếm trả lại bài Vingroup của VIC
            continue
        seen.add(page["title"])
        for i, text in enumerate(recursive_chunks(page["text"])):
            if len(text.split()) < 15:
                continue
            rows.append({"id": f"wiki:{page['ticker']}:{i}", "label": DATASETS["vn30"], "about": [f"VN30:{page['ticker']}"],
                         "props": {"dataset": "vn30", "title": page["title"], "text": text, "url": page["url"],
                                   "source": f"Wikipedia tiếng Việt, phiên bản {page['revid']}"}})
    return rows


def build_vn30():
    df = company_frame()
    info = {tk: (name, sector) for tk, name, sector, _ in VN30}
    stats, year_text = _vn30_stats(df)
    years = sorted({y for _, y in year_text})
    months = sorted({s["period"][6:] for s in stats if s["label"] == "MonthStat"})
    rho, all_pairs, pairs = correlations(df)
    edges = [{"source": f"VN30:{tk}", "target": "INDEX:VN30", "type": "MEMBER_OF", "props": {}} for tk, *_ in VN30]
    edges += [{"source": f"VN30:{tk}", "target": f"SECTOR:{s}", "type": "IN_SECTOR", "props": {}} for tk, _, s, _ in VN30]
    edges += [{"source": f"VN30:{a}", "target": f"VN30:{b}", "type": "CORRELATED", "props": {"rho": r, "sessions": n}}
              for a, b, r, n in pairs]
    edges += [{"source": f"MONTH:{m}", "target": f"YEAR:{m[:4]}", "type": "IN_YEAR", "props": {}} for m in months
              if int(m[:4]) in years]
    chunks = _vn30_profiles(info) + _vn30_stat_chunks(year_text, all_pairs, info) + _wiki_chunks()
    return {"nodes": _vn30_entities(years, months), "edges": edges, "chunks": chunks, "stats": stats, "mentions": [],
            "meta": {"sessions": int(len(df)), "tickers": int(df["Ticker"].nunique()), "pairs": len(all_pairs),
                     "pairs_kept": len(pairs), "date_min": str(df["Date"].min().date()),
                     "date_max": str(df["Date"].max().date())}}


# ---------------------------------------------------------------- HotpotQA
PHRASE = re.compile(r"\b[A-Z][\w'’.-]+(?:\s+(?:of\s+|the\s+|de\s+)?[A-Z][\w'’.-]+)*")
PAREN = re.compile(r"\s*\([^)]*\)$")


def clean_title(t):
    return PAREN.sub("", t).strip()


def _hotpot_entities(paras):
    """Thực thể gồm tiêu đề bài (hub của tài liệu) và cụm viết hoa xuất hiện ở ít nhất 2 đoạn."""
    freq = Counter()
    found = {}
    for p in paras:
        found[p["title"]] = {m.group().rstrip(".") for m in PHRASE.finditer(p["text"]) if len(m.group()) >= 4}
        freq.update(found[p["title"]])
    titles = {clean_title(p["title"]): p["title"] for p in paras}
    keep = {w for w, c in freq.items() if c >= 2} | set(titles)
    return found, titles, keep


def build_hotpot(n_questions=1000):
    qs = read_jsonl(RAW / "hotpot_val.jsonl")[:n_questions]
    paras = list({p["title"]: p for q in qs for p in q["paragraphs"]}.values())
    found, titles, keep = _hotpot_entities(paras)
    nodes = [{"id": f"E:{w}", "label": "TextEntity", "props": {"name": w, "dataset": "hotpot"}} for w in sorted(keep)]
    chunks, mentions, edges = [], [], []
    for p in paras:
        cid = f"hot:{p['title']}"
        name = clean_title(p["title"])
        chunks.append({"id": cid, "label": DATASETS["hotpot"], "about": [f"E:{name}"],
                       "props": {"dataset": "hotpot", "title": p["title"], "text": f"{p['title']}. {p['text']}",
                                 "source": f"Wikipedia tiếng Anh, HotpotQA: {p['title']}"}})
        for w in sorted((found[p["title"]] & keep) - {name}):
            mentions.append({"chunk": cid, "entity": f"E:{w}"})
            edges.append({"source": f"E:{name}", "target": f"E:{w}", "type": "LINKS_TO", "props": {}})
    return {"nodes": nodes, "edges": edges, "chunks": chunks, "stats": [], "mentions": mentions,
            "meta": {"questions": len(qs), "paragraphs": len(paras)}}
