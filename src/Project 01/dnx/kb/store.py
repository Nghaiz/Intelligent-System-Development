"""Đường dẫn và nạp lười các chỉ mục KB; không nạp torch ở đây để máy chủ web khởi động nhanh."""
import json
import threading

import numpy as np
from scipy import sparse

from ..common import DATA, MODELS

KB = DATA / "kb"
IDX = MODELS / "kb"
CHUNKS = KB / "chunks.jsonl"
GAZ = KB / "gazetteer.json"
MENTION = IDX / "mentions.npz"
ENT_IDS = IDX / "mention_entities.json"
THRESH = KB / "thresholds.json"
EMB = IDX / "embeddings_f16.npy"
BM25_M = IDX / "bm25.npz"
BM25_V = IDX / "bm25_vocab.json"
_lock = threading.RLock()
_c = {}


def _once(key, fn):
    with _lock:
        if key not in _c:
            _c[key] = fn()
        return _c[key]


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"Chưa có chỉ mục KB {path}; chạy scripts/kb_index.py và scripts/kb_graph.py")
    return path


def chunks():
    def load():
        rows = [json.loads(x) for x in open(_need(CHUNKS), encoding="utf-8")]
        return rows, {r["chunk_id"]: i for i, r in enumerate(rows)}
    return _once("chunks", load)


def bm25():
    def load():
        return sparse.load_npz(_need(BM25_M)).tocsc(), json.loads(_need(BM25_V).read_text(encoding="utf-8"))
    return _once("bm25", load)


def vectors():
    return _once("vec", lambda: np.load(_need(EMB)).astype(np.float32))


def mention_index():
    def load():
        ents = json.loads(_need(ENT_IDS).read_text(encoding="utf-8"))
        return sparse.load_npz(_need(MENTION)).tocsc(), {e: i for i, e in enumerate(ents)}
    return _once("mentions", load)


def gazetteer():
    def load():
        return json.loads(_need(GAZ).read_text(encoding="utf-8"))
    return _once("gaz", load)


def thresholds_table():
    return _once("thr", lambda: json.loads(_need(THRESH).read_text(encoding="utf-8")))


def encoder():
    """e5-base; mặc định CPU để không tranh GPU với LLM, đặt DNX_KB_DEVICE=cuda để dùng GPU."""
    def load():
        import os
        from sentence_transformers import SentenceTransformer

        from ..common import hf_path
        return SentenceTransformer(hf_path("e5-base"), device=os.environ.get("DNX_KB_DEVICE", "cpu"))
    return _once("enc", load)


def reranker():
    """bge-reranker-v2-m3 trên CPU mất ~1 s mỗi cặp nên mặc định dùng GPU float16 (~1,2 GB VRAM); đổi bằng DNX_KB_RERANK_DEVICE."""
    def load():
        import os

        import torch
        from sentence_transformers import CrossEncoder

        from ..common import hf_path
        dev = os.environ.get("DNX_KB_RERANK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        return CrossEncoder(hf_path("bge-reranker"), device=dev, max_length=192,
                            model_kwargs={"torch_dtype": "float16"} if dev != "cpu" else {})
    return _once("rerank", load)


def encode_query(text):
    return encoder().encode([f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)[0]


NEO_URI, NEO_AUTH = "bolt://127.0.0.1:7687", ("neo4j", "dianexus2026")


def driver():
    def load():
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(NEO_URI, auth=NEO_AUTH, connection_timeout=5, notifications_min_severity="OFF")
        d.verify_connectivity()
        return d
    return _once("neo", load)


def cypher(q, **params):
    with driver().session() as s:
        return [r.data() for r in s.run(q, params)]
