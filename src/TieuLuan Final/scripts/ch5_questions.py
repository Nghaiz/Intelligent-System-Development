"""Sinh bộ câu hỏi đánh giá data/ch5/qa_<miền>.jsonl; đáp án Hetionet và VN30 tính bằng Cypher trên Neo4j."""
from collections import Counter

from tlf import ch5_qa as Q
from tlf.ch5_data import DATA5, write_jsonl
from tlf.common import save_json

SIZES = {"hetionet": 210, "vn30": 210, "hotpot": 1000}


def finish(ds, rows):
    rows = [r for r in rows if r["gold"] and r["answer"]]
    for i, r in enumerate(rows):
        r.update(qid=f"{ds[:3]}{i:04d}", dataset=ds)
    write_jsonl(DATA5 / f"qa_{ds}.jsonl", rows)
    return {"n": len(rows), "by_type": dict(Counter(r["type"] for r in rows)),
            "gold_per_q": round(sum(len(r["gold"]) for r in rows) / len(rows), 2)}


if __name__ == "__main__":
    summary = {"hetionet": finish("hetionet", Q.hetionet_questions(SIZES["hetionet"])),
               "vn30": finish("vn30", Q.vn30_questions(SIZES["vn30"])),
               "hotpot": finish("hotpot", Q.hotpot_questions(SIZES["hotpot"]))}
    save_json("ch5_questions", summary)
    print(summary)
