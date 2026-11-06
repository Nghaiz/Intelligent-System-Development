"""Truy xuất KB theo hợp đồng DiaNexus: search, entities, subgraph, thresholds, stats.

Năm chế độ: bm25, dense (e5), graph (thực thể -> lân cận Neo4j -> chunk), hybrid (RRF của ba nguồn), hybrid_rerank.
Nếu Neo4j không tới được, hybrid chạy với hai nguồn còn lại, ghi cảnh báo và trường retrievers cho thấy điều đó;
chế độ graph, subgraph, stats thì báo lỗi thẳng.
"""
import logging
import re
import time

import numpy as np

from . import lex, store
from .text import nfc, tokenize

log = logging.getLogger("dnx.kb")
RRF_C = 60
DEPTH = 60          # số ứng viên mỗi nguồn đưa vào RRF
RERANK_N = 16       # số ứng viên chấm lại bằng cross-encoder
NEIGHBORS = 40      # số hàng xóm mỗi thực thể hạt
W_SEED, W_NEIGHBOR = 1.0, 0.25
W_ENTITY = 0.15
MODES = ("hybrid", "bm25", "dense", "graph", "hybrid_rerank")
# Quan hệ có ý nghĩa lâm sàng được ưu tiên khi mở rộng lân cận và vẽ đồ thị con
CLINICAL_RELS = ["TREATS", "PALLIATES", "PREVENTS", "CAUSES", "RISK_FACTOR_FOR", "COMPLICATION_OF", "PRESENTS", "DIAGNOSED_BY",
                 "MONITORED_BY", "AFFECTS", "CONTRAINDICATED_IN", "ASSOCIATED_WITH", "SAME_AS", "HAS_THRESHOLD"]
# Quan hệ giữa các nút tri thức; bỏ HAS_MESH, MENTIONS, PART_OF, FROM_SOURCE vì nút hub có hàng chục nghìn cạnh loại đó
KNOWLEDGE_RELS = "|".join(CLINICAL_RELS + ["BINDS", "UPREGULATES", "DOWNREGULATES", "LOCALIZES", "RESEMBLES", "CO_OCCURS", "INTERACTS",
                                           "INCLUDES", "PARTICIPATES", "EXPRESSES", "REGULATES", "COVARIES", "MAPS_TO", "USES_FEATURE"])
NODE = "Entity|Threshold|Feature|Rule"
NEIGHBOR_Q = f"""UNWIND $ids AS sid
MATCH (s:{NODE} {{id: sid}})
CALL (s) {{
  MATCH (s)-[r:{KNOWLEDGE_RELS}]-(m:{NODE})
  WITH m, min(CASE WHEN type(r) IN $clinical THEN 0 ELSE 1 END) AS p
  RETURN m.id AS nid ORDER BY p, m.degree DESC LIMIT $n
}}
RETURN DISTINCT nid"""


def _bm25(query, depth):
    m, vocab = store.bm25()
    en, vi = lex.expand(query)
    toks = list(dict.fromkeys(tokenize(" ".join([query] + en + vi))))
    cols = [vocab[t] for t in toks if t in vocab]
    if not cols:
        return []
    s = np.asarray(m[:, cols].sum(axis=1)).ravel()
    top = np.argpartition(-s, min(depth, len(s) - 1))[:depth]
    top = top[np.argsort(-s[top], kind="stable")]
    return [(int(i), float(s[i])) for i in top if s[i] > 0]


def _dense(vec, depth):
    s = store.vectors() @ vec
    top = np.argpartition(-s, depth)[:depth]
    top = top[np.argsort(-s[top], kind="stable")]
    return [(int(i), float(s[i])) for i in top]


def _link(text):
    """Quét n-gram 1..6 từ tiếng trong bộ từ điển thực thể, khớp dài nhất trước."""
    gaz = store.gazetteer()
    en, vi = lex.expand(text)
    found = {}
    for part in [text] + en + vi:
        toks = re.findall(r"[^\W_]+(?:[-.][^\W_]+)*", nfc(part).lower())
        i = 0
        while i < len(toks):
            for n in range(min(6, len(toks) - i), 0, -1):
                hit = gaz.get(" ".join(toks[i:i + n]))
                if hit:
                    found.setdefault(hit[0], hit)
                    i += n - 1
                    break
            i += 1
    return list(found.values())


def _graph(query, vec, depth):
    """Thực thể trong câu hỏi -> lân cận trên Neo4j -> chunk nhắc tới chúng; xếp bằng cosin + thưởng thực thể hiếm."""
    seeds = [h[0] for h in _link(query)]
    if not seeds:
        return []
    near = store.cypher(NEIGHBOR_Q, ids=seeds, clinical=CLINICAL_RELS, n=NEIGHBORS)
    m, eidx = store.mention_index()
    weights = {e: W_SEED for e in seeds}
    for r in near:
        weights.setdefault(r["nid"], W_NEIGHBOR)
    n_docs = m.shape[0]
    s = np.zeros(n_docs, dtype=np.float32)
    for e, w in weights.items():
        j = eidx.get(e)
        if j is None:
            continue
        col = m.indices[m.indptr[j]:m.indptr[j + 1]]
        if len(col):
            s[col] += w * float(np.log1p(n_docs / len(col)))
    idx = np.flatnonzero(s)
    if not len(idx):
        return []
    # thực thể chung (hub) khớp hàng chục nghìn chunk nên không cắt theo điểm thực thể; cosin quyết định, thực thể cộng điểm thưởng
    final = store.vectors()[idx] @ vec + W_ENTITY * s[idx] / s[idx].max()
    order = np.argsort(-final, kind="stable")[:depth]
    return [(int(idx[i]), float(final[i])) for i in order]


def _rrf(lists, k):
    fused, who = {}, {}
    for name, rows in lists.items():
        for rank, (i, _) in enumerate(rows, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_C + rank)
            who.setdefault(i, []).append(name)
    top = sorted(fused.items(), key=lambda x: -x[1])[:k]
    return [(i, s, who[i]) for i, s in top]


def _rows(triples):
    rows, _ = store.chunks()
    out, seen = [], set()
    for i, s, who in triples:
        c = rows[i]
        if c["chunk_id"] in seen:   # Europe PMC trả trùng 3 bản ghi
            continue
        seen.add(c["chunk_id"])
        out.append({"chunk_id": c["chunk_id"], "text": c["text"], "title": c["title"], "source": c["source"], "url": c["url"],
                    "lang": c["lang"], "score": round(float(s), 5), "retrievers": who})
    return out


def _hybrid(query, depth):
    vec = store.encode_query(query)
    lists = {"bm25": _bm25(query, DEPTH), "dense": _dense(vec, DEPTH)}
    try:
        lists["graph"] = _graph(query, vec, DEPTH)
    except Exception as e:  # Neo4j tắt: cảnh báo, retrievers không có "graph"
        log.warning("hybrid bỏ nguồn graph: %s", e)
    return _rrf(lists, depth)


def search(query, k=8, mode="hybrid"):
    if mode not in MODES:
        raise ValueError(f"mode không hợp lệ: {mode}; chọn trong {MODES}")
    if not query or not query.strip():
        return []
    if mode == "bm25":
        return _rows([(i, s, ["bm25"]) for i, s in _bm25(query, k)])
    if mode == "dense":
        return _rows([(i, s, ["dense"]) for i, s in _dense(store.encode_query(query), k)])
    if mode == "graph":
        return _rows([(i, s, ["graph"]) for i, s in _graph(query, store.encode_query(query), k)])
    if mode == "hybrid":
        return _rows(_hybrid(query, k))
    cands = _hybrid(query, max(k, RERANK_N))
    rows, _ = store.chunks()
    pairs = [(query, rows[i]["text"]) for i, _, _ in cands]
    try:
        scores = store.reranker().predict(pairs, batch_size=len(pairs), show_progress_bar=False)
    except RuntimeError as e:   # GPU hết bộ nhớ khi LLM đang nạp: trả thứ tự RRF, retrievers không có "rerank"
        log.warning("bỏ rerank: %s", e)
        return _rows(cands[:k])
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return _rows([(cands[j][0], float(scores[j]), cands[j][2] + ["rerank"]) for j in order])


def entities(text):
    return [{"id": h[0], "name": h[1], "label": h[2]} for h in _link(text)]


def subgraph(entity_ids, hops=1, limit=120):
    """Lân cận theo từng bước, ưu tiên quan hệ lâm sàng; chỉ trả nút Entity, cạnh giữa các nút đã chọn."""
    keep = list(dict.fromkeys(entity_ids))
    frontier = keep
    per = max(8, limit // max(1, len(keep)))
    for _ in range(int(hops)):
        if len(keep) >= limit or not frontier:
            break
        near = [r["nid"] for r in store.cypher(NEIGHBOR_Q, ids=frontier, clinical=CLINICAL_RELS, n=per)]
        frontier = [x for x in dict.fromkeys(near) if x not in set(keep)][:limit - len(keep)]
        keep += frontier
    nodes = store.cypher(f"""UNWIND $ids AS i MATCH (n:{NODE} {{id: i}})
        RETURN n.id AS id, n.name AS name, [x IN labels(n) WHERE x <> 'Entity'][0] AS label""", ids=keep)
    links = store.cypher(f"""UNWIND $ids AS i MATCH (a:{NODE} {{id: i}})-[r:{KNOWLEDGE_RELS}]->(b:{NODE}) WHERE b.id IN $ids
        RETURN DISTINCT a.id AS source, b.id AS target, type(r) AS type LIMIT $m""", ids=keep, m=limit * 8)
    return {"nodes": nodes, "links": links}


def thresholds(test=""):
    t = nfc(test or "").lower().strip()
    rows = store.thresholds_table()
    return [r for r in rows if not t or t in [a.lower() for a in r["aliases"]] or t == r["test"].lower()]


def stats():
    labels = {r["l"]: store.cypher(f"MATCH (n:`{r['l']}`) RETURN count(n) AS c")[0]["c"]
              for r in store.cypher("CALL db.labels() YIELD label RETURN label AS l")}
    rels = {r["t"]: store.cypher(f"MATCH ()-[r:`{r['t']}`]->() RETURN count(r) AS c")[0]["c"]
            for r in store.cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t")}
    rows, _ = store.chunks()
    return {"nodes": store.cypher("MATCH (n) RETURN count(n) AS c")[0]["c"], "relationships": sum(rels.values()),
            "by_label": labels, "by_type": rels, "chunks": len(rows), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
