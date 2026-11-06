"""Đánh giá sinh: một LLM trên 3 dataset với ngữ cảnh Hybrid top-4. Ghi nối đuôi, chạy lại thì tiếp từ câu chưa làm.

    python scripts/ch5_eval_gen.py --model qwen2.5-7b --n 50
"""
import argparse
import json

import numpy as np

from tlf import ch5_qa as Q
from tlf import ch5_rag as G
from tlf import ch5_schema as S
from tlf.ch5_data import DATA5, read_jsonl
from tlf.ch5_retrieval import retrievers
from tlf.common import limit_threads
from tlf.llm import LLM, gpu_session
from tlf.mem import require_disk
from tlf.runlog import log_run

TOP_K = 4


def done_ids(path):
    return {r["qid"] for r in read_jsonl(path)} if path.exists() else set()


def run(llm, model, ds, n, path):
    qs = read_jsonl(DATA5 / f"qa_{ds}.jsonl")[:n]
    qv = np.load(DATA5 / f"qvec_{ds}.npy")
    hybrid = retrievers(ds)["Hybrid"]
    skip = done_ids(path)
    with open(path, "a", encoding="utf-8") as f:
        for q, v in zip(qs, qv):
            if q["qid"] in skip:
                continue
            ids = [c for c, _ in hybrid.search(q["question"], v, TOP_K)]
            out = G.generate(llm, q["question"], ds, ids)
            row = {"qid": q["qid"], "dataset": ds, "model": model, "type": q["type"], "question": q["question"],
                   "gold": q["answer"], "answer": out["answer"], "seconds": out["seconds"],
                   "em": Q.exact_match(out["answer"], q["answer"]), "f1": Q.f1(out["answer"], q["answer"]),
                   "faith": Q.faithfulness(out["answer"], out["context"]), "hit": bool(set(ids) & set(q["gold"]))}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(q["qid"], round(out["seconds"], 2), row["em"], repr(out["answer"][:70]), flush=True)


def summarize(path, model):
    rows = read_jsonl(path)
    for ds in S.DATASETS:
        g = [r for r in rows if r["dataset"] == ds]
        if g:
            m = {k: float(np.mean([r[k] for r in g])) for k in ("em", "f1", "faith")}
            log_run("ch5", ds, model, "LLM cục bộ", m, sum(r["seconds"] for r in g), "cuda", {"n": len(g), "top_k": TOP_K})
            print(ds, len(g), {k: round(v, 3) for k, v in m.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    require_disk(1, "ch5 đánh giá sinh")
    limit_threads(4)
    target = DATA5 / f"gen_{a.model}.jsonl"
    with gpu_session(a.model):
        llm = LLM(a.model)
        for name in S.DATASETS:
            run(llm, a.model, name, a.n, target)
    summarize(target, a.model)
