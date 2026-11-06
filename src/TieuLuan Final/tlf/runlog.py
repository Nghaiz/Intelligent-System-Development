"""Nhật ký mọi lượt huấn luyện/đánh giá: outputs/runs.jsonl, nguồn của trang Experiment Tracker trên Web App."""
import json
import time

from .common import OUT

RUNS = OUT / "runs.jsonl"


def log_run(chapter, dataset, model, framework, metrics, seconds, device="cpu", params=None):
    """Ghi một dòng; ghi nối đuôi một lần nên an toàn khi nhiều tiến trình chạy song song."""
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "chapter": chapter, "dataset": dataset, "model": model,
           "framework": framework, "device": device, "seconds": round(float(seconds), 2),
           "params": params or {}, "metrics": {k: (round(float(v), 6) if v is not None else None)
                                               for k, v in metrics.items()}}
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
