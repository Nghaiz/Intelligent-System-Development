"""Chương 5: sinh câu hỏi có đáp án từ dữ liệu và chấm câu trả lời.

Câu hỏi Hetionet và VN30 sinh bằng khuôn, đáp án tính bằng Cypher trên Neo4j. HotpotQA dùng nguyên câu hỏi gốc.
Mỗi câu: {qid, dataset, type, hops, question, answer, gold: [chunk id]}.
"""
import random
import re
import string
import unicodedata

from . import ch5_neo as N
from .ch5_data import DATA5, RAW, VN30, read_jsonl
from .ch5_kg import SEEDS

D_NAMES = {"Disease::DOID:9352": "type 2 diabetes mellitus", "Disease::DOID:9744": "type 1 diabetes mellitus"}


def _parts(by_prefix, node, m, rev, name=None):
    """Id các chunk của cặp (nút, quan hệ); có name thì chỉ giữ phần chứa tên đó."""
    rows = by_prefix.get(f"het:{node}:{m}:{rev}", [])
    return [r["id"] for r in rows if name is None or name in r["text"]]


def hetionet_questions(n, seed=42):
    rng = random.Random(seed)
    chunks = read_jsonl(DATA5 / "chunks_hetionet.jsonl")
    by_prefix = {}
    for c in chunks:
        by_prefix.setdefault(c["id"].rsplit(":", 1)[0], []).append(c)
    gens = [_het_count, _het_drug_gene, _het_side_effect, _het_class, _het_pathway]
    per = n // len(gens) + 1
    out = []
    for g in gens:
        rows = g(by_prefix, rng)
        rng.shuffle(rows)
        out += rows[:per]
    rng.shuffle(out)
    return out[:n]


def _het_count(by_prefix, rng):
    rows = N.cypher("""MATCH (c:Compound)-[:BINDS]->(g:Gene) WITH c, count(g) AS n WHERE n >= 2
                       RETURN c.hid AS hid, c.name AS name, n ORDER BY name""")
    return [{"type": "count_1hop", "hops": 1, "question": f"Trong đồ thị con, hợp chất {r['name']} liên kết với bao nhiêu gen?",
             "answer": str(r["n"]), "gold": _parts(by_prefix, r["hid"], "CbG", 0)} for r in rows]


def _het_drug_gene(by_prefix, rng):
    out = []
    for d in SEEDS:
        for drug in N.cypher("""MATCH (c:Compound)-[:TREATS]->(:Disease {hid: $d}) MATCH (c)-[:BINDS]->(g:Gene)
                                <-[:ASSOCIATES]-(:Disease {hid: $d}) WITH c, collect(g.name) AS gs
                                WHERE size(gs) = 1 RETURN c.hid AS hid, c.name AS name, gs[0] AS gene""", {"d": d}):
            out.append({"type": "intersect_2hop", "hops": 2, "answer": drug["gene"],
                        "question": f"Gen nào vừa liên quan tới bệnh {D_NAMES[d]} vừa được {drug['name']} liên kết?",
                        "gold": _parts(by_prefix, drug["hid"], "CbG", 0, drug["gene"])
                        + _parts(by_prefix, d, "DaG", 0, drug["gene"])})
    return out


def _het_side_effect(by_prefix, rng):
    out = []
    for d in SEEDS:
        for r in N.cypher("""MATCH (c:Compound)-[:TREATS]->(:Disease {hid: $d}) MATCH (c)-[:CAUSES]->(s:SideEffect)
                             WITH s, collect(c) AS cs WHERE size(cs) = 1
                             RETURN s.name AS se, cs[0].name AS drug, cs[0].hid AS hid ORDER BY se""", {"d": d}):
            out.append({"type": "filter_2hop", "hops": 2, "answer": r["drug"],
                        "question": f"Trong các thuốc điều trị {D_NAMES[d]}, thuốc nào gây tác dụng phụ {r['se']}?",
                        "gold": _parts(by_prefix, d, "CtD", 1, r["drug"])
                        + _parts(by_prefix, r["hid"], "CcSE", 0, r["se"])})
    return out[:]


def _het_class(by_prefix, rng):
    rows = N.cypher("""MATCH (p:PharmacologicClass)-[:INCLUDES]->(c:Compound)-[:TREATS]->(:Disease)
                       WITH c, collect(DISTINCT p) AS ps WHERE size(ps) = 1
                       RETURN c.name AS name, c.hid AS hid, ps[0].name AS cls""")
    return [{"type": "lookup_1hop", "hops": 1, "question": f"Hợp chất {r['name']} thuộc nhóm dược lý nào?",
             "answer": r["cls"], "gold": _parts(by_prefix, r["hid"], "PCiC", 1)} for r in rows]


def _het_pathway(by_prefix, rng):
    rows = N.cypher("""MATCH (d:Disease {hid: $d})-[:ASSOCIATES]->(g:Gene)-[:PARTICIPATES]->(p:Pathway)
                       WITH p, collect(g) AS gs WHERE size(gs) = 1
                       RETURN p.name AS pw, p.hid AS phid, gs[0].name AS gene, gs[0].hid AS ghid""", {"d": SEEDS[0]})
    return [{"type": "intersect_2hop", "hops": 2, "answer": r["gene"],
             "question": f"Gen nào liên quan tới {D_NAMES[SEEDS[0]]} và tham gia đường truyền {r['pw']}?",
             "gold": _parts(by_prefix, r["phid"], "GpPW", 1, r["gene"])
             + _parts(by_prefix, SEEDS[0], "DaG", 0, r["gene"])} for r in rows]


def _num(x):
    return f"{x:.2f}".replace(".", ",")


def vn30_questions(n, seed=42):
    rng = random.Random(seed)
    stats = N.cypher("""MATCH (c:Company)-[:HAS_YEAR]->(y:YearStat)-[:OF_YEAR]->(p:Year)
                        MATCH (c)-[:IN_SECTOR]->(s:Sector)
                        RETURN c.ticker AS t, p.year AS y, y.ret_pct AS ret, y.high AS high, s.name AS sector""")
    rng.shuffle(stats)
    n_each = n // 7 + 1
    out = [_vn_sector(t) for t, *_ in VN30]
    rng.shuffle(out)
    out = out[:n_each]
    out += [_vn_year(r) for r in stats[:n_each]]
    out += [_vn_high(r) for r in stats[n_each:2 * n_each]]
    out += _vn_month(stats[2 * n_each:3 * n_each], rng)
    out += _vn_top_sector(stats, rng)[:n_each]
    peers = [r["t"] for r in N.cypher("MATCH (c:Company)-[:CORRELATED]-() RETURN DISTINCT c.ticker AS t ORDER BY t")]
    out += [_vn_corr(t) for t in rng.sample(peers, min(n_each, len(peers)))]
    out += _vn_compare(stats, rng)[:n_each]
    rng.shuffle(out)
    return out[:n]


def _vn_sector(t):
    s = N.cypher("MATCH (:Company {ticker: $t})-[:IN_SECTOR]->(s) RETURN s.name AS n", {"t": t})[0]["n"]
    return {"type": "lookup_1hop", "hops": 1, "question": f"Mã {t} thuộc ngành nào?", "answer": s,
            "gold": [f"profile:{t}"]}


def _vn_year(r):
    return {"type": "stat_1hop", "hops": 1, "question": f"Năm {r['y']}, cổ phiếu {r['t']} có lợi suất bao nhiêu phần trăm?",
            "answer": _num(r["ret"]), "gold": [f"year:{r['t']}:{r['y']}"]}


def _vn_high(r):
    return {"type": "stat_1hop", "hops": 1, "question": f"Giá cao nhất của cổ phiếu {r['t']} trong năm {r['y']} là bao nhiêu đồng?",
            "answer": f"{r['high']:,.0f}".replace(",", "."), "gold": [f"year:{r['t']}:{r['y']}"]}


def _vn_month(rows, rng):
    out = []
    for r in rows:
        m = rng.randint(1, 12)
        got = N.cypher("MATCH (s:MonthStat {id: $id}) RETURN s.ret_pct AS ret", {"id": f"{r['t']}:{r['y']}-{m:02d}"})
        if got:
            out.append({"type": "stat_1hop", "hops": 1, "answer": _num(got[0]["ret"]), "gold": [f"year:{r['t']}:{r['y']}"],
                        "question": f"Tháng {m} năm {r['y']}, lợi suất của cổ phiếu {r['t']} là bao nhiêu phần trăm?"})
    return out


def _vn_top_sector(stats, rng):
    groups = {}
    for r in stats:
        groups.setdefault((r["sector"], r["y"]), []).append(r)
    out = []
    for (sector, y), rows in sorted(groups.items()):
        if len(rows) >= 3:
            best = max(rows, key=lambda r: r["ret"])
            out.append({"type": "aggregate_2hop", "hops": 2, "answer": best["t"],
                        "question": f"Trong ngành {sector}, mã nào có lợi suất cao nhất năm {y}?",
                        "gold": [f"year:{best['t']}:{y}", f"profile:{best['t']}"]})
    rng.shuffle(out)
    return out


def _vn_corr(t):
    peer = N.cypher("""MATCH (:Company {ticker: $t})-[r:CORRELATED]-(b) RETURN b.ticker AS p
                       ORDER BY r.rho DESC LIMIT 1""", {"t": t})[0]["p"]
    return {"type": "lookup_1hop", "hops": 1, "question": f"Mã nào có tương quan lợi suất ngày cao nhất với {t}?",
            "answer": peer, "gold": [f"corr:{t}"]}


def _vn_compare(stats, rng):
    by_year = {}
    for r in stats:
        by_year.setdefault(r["y"], []).append(r)
    out = []
    for y, rows in sorted(by_year.items()):
        for _ in range(4):
            a, b = rng.sample(rows, 2)
            win = a if a["ret"] > b["ret"] else b
            out.append({"type": "compare_2hop", "hops": 2, "answer": win["t"],
                        "question": f"Năm {y}, cổ phiếu {a['t']} hay {b['t']} có lợi suất cao hơn?",
                        "gold": [f"year:{a['t']}:{y}", f"year:{b['t']}:{y}"]})
    rng.shuffle(out)
    return out


def hotpot_questions(n):
    rows = read_jsonl(RAW / "hotpot_val.jsonl")[:n]
    return [{"type": r["type"], "hops": 2, "question": r["question"], "answer": r["answer"],
             "gold": [f"hot:{t}" for t in r["support"]], "level": r["level"]} for r in rows]


# ---------------------------------------------------------------- chấm điểm
NUM = re.compile(r"[-+−]?\d+(?:[.,]\d+)*")
ARTICLES = re.compile(r"\b(a|an|the)\b")
NO_DATA = re.compile(r"không đủ dữ liệu|không có thông tin|not enough", re.I)


def normalize(s):
    s = unicodedata.normalize("NFC", str(s)).lower().replace("−", "-")
    s = "".join(" " if c in string.punctuation and c not in ",." else c for c in s)
    s = ARTICLES.sub(" ", s)
    return " ".join(w.strip(",.") for w in s.split()).strip()


def to_float(tok):
    t = tok.replace("−", "-").replace("+", "")
    if "," in t:          # kiểu Việt: 1.234,56
        t = t.replace(".", "").replace(",", ".")
    elif t.count(".") > 1 or re.fullmatch(r"-?\d{1,3}\.\d{3}", t):
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def numbers(s):
    return [v for v in (to_float(t) for t in NUM.findall(str(s))) if v is not None]


def exact_match(pred, gold):
    """Đúng khi chuỗi đã chuẩn hóa trùng khớp; đáp án số chấp nhận mọi cách viết, sai số tối đa 0,5 %."""
    if normalize(pred) == normalize(gold):
        return 1.0
    g = to_float(str(gold)) if re.fullmatch(r"[-+−]?\d+(?:[.,]\d+)*", str(gold).strip()) else None
    if g is None:
        return 0.0
    got = numbers(pred)
    return float(len(got) == 1 and abs(abs(got[0]) - abs(g)) <= max(0.005 * abs(g), 0.011))


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def faithfulness(answer, context):
    """Tỷ lệ token của câu trả lời có mặt trong ngữ cảnh; câu từ chối vì thiếu dữ liệu coi như bám ngữ cảnh."""
    if NO_DATA.search(answer):
        return 1.0
    toks = normalize(answer).split()
    if not toks:
        return 0.0
    ctx = set(normalize(context).split())
    return sum(t in ctx for t in toks) / len(toks)
