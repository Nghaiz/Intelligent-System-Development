"""Chương 5: bốn bộ truy xuất trên kho chunk: BM25, Dense (chỉ mục vector Neo4j), GraphRAG (Cypher), Hybrid (RRF).

Mọi bộ truy xuất có search(q, vec, k) -> [(chunk_id, score)]; vec là vector e5 của câu hỏi, BM25 bỏ qua nó.
"""
import gc
import math
import os
import re

import numpy as np

os.environ.setdefault("USE_TF", "0")

from . import ch5_neo as N  # noqa: E402
from . import ch5_schema as S  # noqa: E402
from .ch5_data import DATA5, read_jsonl  # noqa: E402

TOKEN = re.compile(r"\w+", re.U)
STOP = set("""the of a an in on at to for and or is was were are be by with from that which who whom what when where
how did does do has had have as it its his her their this these those than also both same
là của và có nào gì bao nhiêu trong với các những một được cho theo năm tháng mã cổ phiếu hay vừa
ngoài mà ở thì bệnh gen hợp chất""".split())
_embedder = None


def tokenize(text):
    return TOKEN.findall(text.lower())


def embedder(device=None):
    """e5-base dùng chung; họ e5 đòi tiền tố "query: " và "passage: ". Web gọi với device="cpu"."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        from .common import hf_path
        _embedder = SentenceTransformer(hf_path("e5-base"), device=device or "cuda")
    return _embedder


def release_embedder():
    """Bỏ tham chiếu tới mô hình rồi thu gom; trả True nếu thực sự có mô hình để giải phóng."""
    global _embedder
    held = _embedder is not None
    _embedder = None
    gc.collect()
    return held


def encode(texts, kind="query", batch_size=64, device=None):
    prefixed = [f"{kind}: {t}" for t in texts]
    return embedder(device).encode(prefixed, batch_size=batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


class BM25:
    name = "BM25"

    def __init__(self, ds):
        from rank_bm25 import BM25Okapi
        self.rows = read_jsonl(DATA5 / f"chunks_{ds}.jsonl")
        self.ids = [r["id"] for r in self.rows]
        self.model = BM25Okapi([tokenize(r["text"]) for r in self.rows], k1=1.5, b=0.75)

    def search(self, q, vec=None, k=10):
        s = self.model.get_scores(tokenize(q))
        idx = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[i], float(s[i])) for i in idx if s[i] > 0]


class Dense:
    name = "Dense"

    def __init__(self, ds):
        self.index = f"vec_{ds}"

    def search(self, q, vec, k=10):
        rows = N.cypher(S.VECTOR_SEARCH, {"index": self.index, "k": k, "vec": [float(x) for x in vec]})
        return [(r["id"], r["score"]) for r in rows]


def lucene_query(q):
    """Câu hỏi -> truy vấn OR trên chỉ mục toàn văn; chỉ giữ token chữ-số nên không cần thoát ký tự đặc biệt."""
    toks = [t for t in tokenize(q) if t not in STOP and len(t) > 1]
    return " OR ".join(dict.fromkeys(toks)) or "khong_co_thuc_the"


class Graph:
    """GraphRAG cục bộ: liên kết thực thể + hạt vector, mở rộng một bước trên đồ thị, xếp hạng bằng cosin + số khớp."""
    name = "GraphRAG"

    def __init__(self, ds, n_linked=8, n_vec=5, fanout=120, boost=0.15):
        self.ds, self.index = ds, f"vec_{ds}"
        self.n_linked, self.n_vec, self.fanout, self.boost = n_linked, n_vec, fanout, boost

    def link(self, q):
        return N.cypher(S.LINK_ENTITIES, {"lucene": lucene_query(q), "ds": self.ds, "n": self.n_linked})

    def search(self, q, vec, k=10):
        v = [float(x) for x in vec]
        linked = [r["id"] for r in self.link(q)]
        top = N.cypher(S.VECTOR_SEARCH, {"index": self.index, "k": self.n_vec, "vec": v})
        extra = N.cypher(S.SEED_ENTITIES, {"chunks": [r["id"] for r in top]})
        seeds = list(dict.fromkeys(linked + [r["id"] for r in extra]))
        rows = N.cypher(S.GRAPH_RETRIEVE, {"seeds": seeds, "linked": linked, "vec": v, "fanout": self.fanout,
                                           "boost": self.boost, "k": k})
        return [(r["id"], r["score"]) for r in rows]


class Hybrid:
    """Reciprocal Rank Fusion: s(d) = tổng 1 / (c + hạng của d trong từng nguồn)."""
    name = "Hybrid"

    def __init__(self, parts, c=60, depth=50):
        self.parts, self.c, self.depth = parts, c, depth

    def search(self, q, vec, k=10):
        fused = {}
        for r in self.parts:
            for rank, (cid, _) in enumerate(r.search(q, vec, self.depth), 1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.c + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


def retrievers(ds):
    b, d, g = BM25(ds), Dense(ds), Graph(ds)
    return {"BM25": b, "Dense": d, "GraphRAG": g, "Hybrid": Hybrid([b, d, g])}


def rank_metrics(ranked, gold, ks=(1, 3, 5, 10)):
    """Recall@k, Hit@k với mọi đoạn vàng, MRR@10 và nDCG@10 (liên quan nhị phân) cho một câu hỏi."""
    gold = set(gold)
    rel = [c in gold for c in ranked[:10]]
    out = {f"recall@{k}": sum(rel[:k]) / len(gold) for k in ks}
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    out["ndcg"] = dcg / sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return out
