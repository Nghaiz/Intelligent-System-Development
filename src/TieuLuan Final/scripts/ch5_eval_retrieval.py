"""Đánh giá BM25, Dense, GraphRAG, Hybrid trên 3 dataset. Không cần GPU: vector câu hỏi đã tính sẵn."""
import sys
import time
from collections import defaultdict

import numpy as np

from tlf import ch5_schema as S
from tlf.ch5_data import DATA5, read_jsonl
from tlf.ch5_retrieval import rank_metrics, retrievers
from tlf.common import save_json
from tlf.runlog import log_run

KEYS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg")


def evaluate(ds, name, r, qs, qv):
    per = []
    t0 = time.perf_counter()
    for q, v in zip(qs, qv):
        t1 = time.perf_counter()
        hits = [c for c, _ in r.search(q["question"], v, 10)]
        per.append(rank_metrics(hits, q["gold"]) | {"qid": q["qid"], "type": q["type"], "hops": q["hops"],
                                                    "ms": 1000 * (time.perf_counter() - t1), "top": hits[:3]})
    row = {"dataset": ds, "model": name, "framework": "Neo4j" if name != "BM25" else "rank_bm25", "n": len(qs),
           **{k: float(np.mean([p[k] for p in per])) for k in KEYS}, "ms_p50": float(np.median([p["ms"] for p in per])),
           "seconds": time.perf_counter() - t0}
    log_run("ch5", ds, name, row["framework"], {k: row[k] for k in KEYS}, row["seconds"], "cpu", {"n": len(qs)})
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    return row, per


def by_type(per):
    groups = defaultdict(list)
    for p in per:
        groups[p["type"]].append(p)
    return [{"type": t, "n": len(g), "recall@5": float(np.mean([p["recall@5"] for p in g])),
             "mrr": float(np.mean([p["mrr"] for p in g]))} for t, g in sorted(groups.items())]


def main(names):
    rows, types, per_q = [], [], []
    for ds in names:
        qs = read_jsonl(DATA5 / f"qa_{ds}.jsonl")
        qv = np.load(DATA5 / f"qvec_{ds}.npy")
        for name, r in retrievers(ds).items():
            row, per = evaluate(ds, name, r, qs, qv)
            rows.append(row)
            types += [{"dataset": ds, "model": name} | t for t in by_type(per)]
            per_q += [{"dataset": ds, "model": name} | p for p in per]
    if len(names) == len(S.DATASETS):
        save_json("ch5_retrieval", {"rows": rows, "by_type": {"rows": types}, "per_q": per_q})


if __name__ == "__main__":
    main(sys.argv[1:] or list(S.DATASETS))
