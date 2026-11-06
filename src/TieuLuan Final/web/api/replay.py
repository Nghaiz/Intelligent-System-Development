"""Phát lại các vết thông điệp THẬT đã ghi ở data/ch6/runs/*.jsonl, xem Arena được mà không cần nạp LLM."""
import json

from tlf.common import DATA

RUNS_DIR = DATA / "ch6" / "runs"
MAX_ITEMS = 40
PER_FILE = 3


def _rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue  # dòng cuối có thể đang được ghi


def _collect():
    items = []
    for path in sorted(RUNS_DIR.glob("*.jsonl")):
        items += [r for r in _rows(path) if len(r.get("messages") or []) >= 2][:PER_FILE]
    return items[:MAX_ITEMS]


def listing():
    return [{"index": i, "dataset": r.get("dataset"), "arch": r.get("arch"), "correct": r.get("correct"),
             "messages": len(r["messages"]), "tokens": (r.get("tokens_in") or 0) + (r.get("tokens_out") or 0)}
            for i, r in enumerate(_collect())]


def load(index):
    items = _collect()
    return items[index] if 0 <= index < len(items) else None
