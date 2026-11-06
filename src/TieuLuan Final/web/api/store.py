"""Đọc kết quả thật của các làn: outputs/metrics/*.json và outputs/runs.jsonl.

Các làn đang ghi song song nên tệp có thể chưa có hoặc ghi dở; mọi hàm đọc bỏ qua tệp hỏng thay vì sập,
và không bao giờ điền số thay thế: thiếu thì trả rỗng để giao diện báo thiếu.
"""
import json
import re

from tlf.common import METRICS
from tlf.runlog import RUNS

CHAPTER_RE = re.compile(r"^ch(\d+)_")


def read_json(path):
    """Trả None khi tệp chưa có hoặc đang ghi dở (JSON cắt ngang)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def chapter_of(path):
    m = CHAPTER_RE.match(path.name)
    return int(m.group(1)) if m else None


def _by_suffix(suffix):
    for path in sorted(METRICS.glob(f"ch*{suffix}.json"), key=lambda p: (chapter_of(p) or 0, p.name)):
        data = read_json(path)
        if isinstance(data, dict):
            yield chapter_of(path), data


def runs():
    """Mọi lượt chạy trong runs.jsonl; dòng đang ghi dở bị bỏ qua."""
    if not RUNS.exists():
        return []
    rows = []
    with open(RUNS, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "model" in row:
                rows.append(row)
    return rows


def compare():
    """Các bảng chN_compare.json, mỗi bảng kèm số chương và chỉ số chính do làn khai."""
    return [{"chapter": ch, "primary": d.get("primary"), "primary_by_dataset": d.get("primary_by_dataset") or {},
             "rows": d.get("rows", [])}
            for ch, d in _by_suffix("_compare")]


def datasets():
    out = []
    for ch, d in _by_suffix("_datasets"):
        out += [dict(item, chapter=ch) for item in d.get("datasets", []) if isinstance(item, dict)]
    return out


def _sample_total(items):
    return sum(x["n_samples"] for x in items if isinstance(x.get("n_samples"), (int, float)))


def _chapter_rollup(all_runs, tables, items):
    """Gộp theo chương: số lượt chạy, dataset và số mô hình thật có trên đĩa."""
    chapters = {}

    def slot(n):
        return chapters.setdefault(n, {"chapter": n, "runs": 0, "datasets": 0, "models": 0})
    for r in all_runs:
        m = re.search(r"\d+", str(r.get("chapter", "")))
        if m:
            slot(int(m.group()))["runs"] += 1
    for item in items:
        slot(item["chapter"])["datasets"] += 1
    for t in tables:
        slot(t["chapter"])["models"] = len({row.get("model") for row in t["rows"]})
    return [chapters[k] for k in sorted(chapters)]


def overview():
    all_runs, tables, items = runs(), compare(), datasets()
    models = {r.get("model") for r in all_runs} | {row.get("model") for t in tables for row in t["rows"]}
    return {"runs": len(all_runs), "datasets": len(items), "samples": _sample_total(items),
            "models": len(models - {None}), "compare_rows": sum(len(t["rows"]) for t in tables),
            "train_seconds": round(sum(r.get("seconds") or 0 for r in all_runs), 1),
            "metric_files": len(list(METRICS.glob("ch*.json"))),
            "chapters": _chapter_rollup(all_runs, tables, items)}
