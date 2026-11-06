"""Nhúng mọi chunk và câu hỏi bằng e5-base: chunk ghi vào thuộc tính embedding của Neo4j, câu hỏi ghi ra data/ch5/."""
import sys
import time

import numpy as np

from tlf import ch5_neo as N
from tlf import ch5_schema as S
from tlf.ch5_data import DATA5, read_jsonl
from tlf.ch5_retrieval import encode
from tlf.common import METRICS, limit_threads, load_json, save_json
from tlf.gpu import heavy
from tlf.mem import require_disk, slot
from tlf.runlog import log_run


def embed_dataset(ds):
    rows = read_jsonl(DATA5 / f"chunks_{ds}.jsonl")
    t0 = time.perf_counter()
    vecs = encode([r["text"] for r in rows], "passage", batch_size=32)
    spent = time.perf_counter() - t0
    N.batched(S.SET_EMBEDDINGS, [{"id": r["id"], "vec": v.tolist()} for r, v in zip(rows, vecs)], size=500)
    qs = read_jsonl(DATA5 / f"qa_{ds}.jsonl")
    qv = encode([q["question"] for q in qs], "query")
    np.save(DATA5 / f"qvec_{ds}.npy", qv)
    print(ds, vecs.shape, qv.shape, round(spent, 1), flush=True)
    log_run("ch5", ds, "e5-base", "sentence-transformers", {"chunks": len(rows)}, spent, "cuda")
    return {"chunks": len(rows), "questions": len(qs), "seconds": round(spent, 1), "per_s": round(len(rows) / spent, 1)}


if __name__ == "__main__":
    require_disk(1, "ch5 nhúng")
    limit_threads(4)
    with slot(3, "ch5 e5-base"), heavy("ch5 e5-base"):
        out = {ds: embed_dataset(ds) for ds in (sys.argv[1:] or S.DATASETS)}
    N.cypher("CALL db.awaitIndexes(600)")
    if (METRICS / "ch5_embed.json").exists():
        out = load_json("ch5_embed") | out
    save_json("ch5_embed", out)
    print(out)
