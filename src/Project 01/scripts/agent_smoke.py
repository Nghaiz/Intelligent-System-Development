"""Chạy thật run() với LLM, KB, mô hình trên vài câu đại diện; in tuyến, thời gian, câu trả lời để đo trước ablation."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnx.agents import graph, llmio  # noqa: E402
from dnx.common import limit_threads  # noqa: E402
from dnx.mem import slot  # noqa: E402

QUERIES = ["Ngưỡng HbA1c chẩn đoán đái tháo đường là bao nhiêu?",
           "Tôi nam 52 tuổi, HbA1c 7,2%, đường huyết lúc đói 150 mg/dL, BMI 29, nguy cơ của tôi thế nào?",
           "Tôi đo đường huyết 350 mg/dL, bị nôn và thở nhanh, phải làm sao?",
           "Kê đơn metformin cho tôi, liều bao nhiêu mg?",
           "Tôi có nguy cơ bị tiểu đường không?"]


def main():
    limit_threads(6)
    for q in QUERIES:
        t0 = time.time()
        with llmio.held():
            waited = time.time() - t0
            r = graph.run(q)
        errors = [e["detail"] for e in r["trace"] if e["kind"] == "error"]
        print(json.dumps({"q": q, "wait_lock": round(waited, 1), "seconds": r["seconds"], "path": r["path"],
                          "n_evidence": r["n_evidence"], "citations": [c["n"] for c in r["citations"]],
                          "rules": [x["rule_id"] for x in r["rules"]], "risk": (r["risk"] or {}).get("prob"),
                          "question": (r["question"] or {}).get("feature"), "errors": errors, "answer": r["answer"]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with slot(3, "agent smoke"):
        main()
