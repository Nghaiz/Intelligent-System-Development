"""Tải GSM8K và PubMedQA (parquet, không chạy mã trong repo), sinh ca tổng hợp, ghi JSONL vào data/ch6."""
import json

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from tlf.ch6_rules import generate_cases
from tlf.common import data_dir
from tlf.mem import require_disk

OUT = data_dir("ch6")


def parquet_rows(repo, prefix):
    names = [f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet")]
    rows = []
    for name in names:
        path = hf_hub_download(repo, name, repo_type="dataset", local_dir=OUT / "raw")
        rows += pq.read_table(path).to_pylist()
    print(repo, prefix, len(rows), names, flush=True)
    return rows


def dump(name, rows):
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    require_disk(1, "ch6 dữ liệu")
    gsm = parquet_rows("openai/gsm8k", "main/test")
    dump("gsm8k", [{"id": i, "question": r["question"], "context": "", "solution": r["answer"],
                    "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(gsm)])
    pub = parquet_rows("qiaojin/PubMedQA", "pqa_labeled/")
    dump("pubmed", [{"id": str(r["pubid"]), "question": r["question"],
                     "context": " ".join(r["context"]["contexts"]), "long_answer": r["long_answer"],
                     "gold": r["final_decision"]} for r in pub])
    dump("clinical", generate_cases(200))
