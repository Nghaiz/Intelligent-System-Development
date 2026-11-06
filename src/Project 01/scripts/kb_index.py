"""Chunk + BM25 + embedding e5. Chạy: kb_index.py chunk | bm25 | embed (có cache từng lô, chạy tiếp được)."""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import hf_path, limit_threads, save_json  # noqa: E402
from dnx.gpu import heavy  # noqa: E402
from dnx.kb import store  # noqa: E402
from dnx.kb.text import chunk_doc, tokenize  # noqa: E402
from dnx.mem import require_disk, slot  # noqa: E402
from dnx.runlog import log_run  # noqa: E402

RAW = store.KB / "raw"
K1, B = 1.5, 0.75
BATCH = 256
SHARD = 5120


def do_chunk():
    rows = []
    per = Counter()
    for f in sorted(RAW.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            seen = 0
            for i, (sec, text) in enumerate(chunk_doc(d["text"], d["source"])):
                rows.append({"chunk_id": f"{d['doc_id']}#{i}", "doc_id": d["doc_id"], "text": text, "title": d["title"],
                             "section": sec, "source": d["source"], "url": d["url"], "lang": d["lang"],
                             "words": len(text.split())})
                seen += 1
            per[d["source"]] += seen
    with open(store.CHUNKS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    w = [r["words"] for r in rows]
    save_json("kb_chunks", {"chunks": len(rows), "by_source": dict(per), "by_lang": dict(Counter(r["lang"] for r in rows)),
                            "docs": len({r["doc_id"] for r in rows}), "words_mean": statistics.fmean(w),
                            "words_median": statistics.median(w), "words_max": max(w), "words_min": min(w),
                            "path": str(store.CHUNKS), "chunking": "theo câu, ≤180 từ, chồng lấn ≤30 từ"})
    print("chunks", len(rows), dict(per), flush=True)


def do_bm25():
    t0 = time.time()
    rows = [json.loads(x) for x in open(store.CHUNKS, encoding="utf-8")]
    vocab, ind, ptr, data, lens = {}, [], [0], [], []
    for r in rows:
        tf = Counter(tokenize(f"{r['title']} {r['section']} {r['text']}"))
        lens.append(sum(tf.values()))
        for t, c in tf.items():
            ind.append(vocab.setdefault(t, len(vocab)))
            data.append(c)
        ptr.append(len(ind))
    n, v = len(rows), len(vocab)
    tf = sparse.csr_matrix((np.array(data, np.float32), np.array(ind, np.int32), np.array(ptr, np.int64)), shape=(n, v))
    df = np.bincount(tf.indices, minlength=v)
    idf = np.maximum(np.log((n - df + 0.5) / (df + 0.5)), 0.01).astype(np.float32)
    dl = np.array(lens, np.float32)
    denom = K1 * (1 - B + B * dl / dl.mean())
    rep = np.repeat(denom, np.diff(tf.indptr))
    tf.data = tf.data * (K1 + 1) / (tf.data + rep) * idf[tf.indices]
    store.IDX.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(store.BM25_M, tf.tocsc())
    store.BM25_V.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    log_run("kb", "kb_chunks", "bm25", "scipy", {"chunks": n, "vocab": v, "nnz": tf.nnz}, time.time() - t0, "cpu", {"k1": K1, "b": B})
    print(f"bm25 n={n} vocab={v} nnz={tf.nnz}", flush=True)


def do_embed(cpu=False):
    """Mỗi mảnh SHARD chunk ghi một tệp để chạy tiếp; giữ khoá GPU theo từng mảnh để làn khác xen vào được."""
    from sentence_transformers import SentenceTransformer
    rows = [json.loads(x) for x in open(store.CHUNKS, encoding="utf-8")]
    n = len(rows)
    require_disk(1, "kb embed")
    tmp = store.IDX / "emb_shards"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = None
    for s in range(0, n, SHARD):
        p = tmp / f"{s:07d}.npy"
        if p.exists() and np.load(p).shape[0] == min(SHARD, n - s):
            continue
        if cpu:   # GPU đang bận lâu: chạy CPU, không giữ khoá GPU
            with slot(3, "e5 embed cpu"):
                if model is None:
                    model = SentenceTransformer(hf_path("e5-base"), device="cpu")
                    model.max_seq_length = 320
                v = model.encode([f"passage: {r['title']}. {r['text']}" for r in rows[s:s + SHARD]], batch_size=32,
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        else:
            with slot(4, "e5 embed"), heavy("e5 embed"):
                if model is None:
                    model = SentenceTransformer(hf_path("e5-base"), device="cpu", model_kwargs={"torch_dtype": "float16"})
                    model.max_seq_length = 320
                model.to("cuda")
                v = model.encode([f"passage: {r['title']}. {r['text']}" for r in rows[s:s + SHARD]], batch_size=BATCH,
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
                model.to("cpu")   # nhường VRAM ngay khi trả khoá
        tmp_file = p.with_suffix(".tmp.npy")
        np.save(tmp_file, v)
        tmp_file.replace(p)
        print(f"embed {s + len(v)}/{n} {time.time() - t0:.0f}s", flush=True)
    mats = [np.load(tmp / f"{s:07d}.npy") for s in range(0, n, SHARD)]
    m = np.concatenate(mats)
    assert m.shape == (n, 768), m.shape
    np.save(store.EMB, m)
    log_run("kb", "kb_chunks", "e5-base", "sentence-transformers", {"chunks": n}, time.time() - t0, "cpu" if cpu else "cuda", {"max_seq": 320})
    print("embed xong", m.shape, flush=True)


if __name__ == "__main__":
    limit_threads(8)
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["chunk", "bm25", "embed", "all"])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.step in ("chunk", "all"):
        do_chunk()
    if a.step in ("bm25", "all"):
        do_bm25()
    if a.step in ("embed", "all"):
        for attempt in range(30):   # cạn commit (1455) là tình trạng tạm thời của máy; mảnh đã xong nằm trong cache
            try:
                do_embed(a.cpu)
                break
            except OSError as e:
                if "1455" not in str(e):
                    raise
                print(f"[mem] cạn commit, thử lại sau 60s (lần {attempt + 1})", flush=True)
                time.sleep(60)
        else:
            raise SystemExit("embed thất bại: commit không đủ sau 30 lần thử")
