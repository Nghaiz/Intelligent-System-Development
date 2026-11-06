"""Đọc kết quả thật của các làn: outputs/metrics/*.json và outputs/runs.jsonl. Tệp đang ghi dở bị bỏ qua, không điền số."""
import json
import re

from dnx.common import METRICS
from dnx.runlog import RUNS

from .lazy import Unavailable

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,80}$")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def metric_names():
    return sorted(p.stem for p in METRICS.glob("*.json"))


def metric_count():
    return len(metric_names())


def metric(name):
    """Tên chỉ được khớp mẫu và phải có trong danh sách đã quét, không ghép đường dẫn từ dữ liệu người dùng."""
    if not NAME_RE.match(name):
        raise ValueError("Tên số liệu không hợp lệ.")
    if name not in metric_names():
        raise Unavailable(f"Chưa có số liệu {name}.")
    data = read_json(METRICS / f"{name}.json")
    if data is None:
        raise Unavailable(f"Số liệu {name} đang được ghi.")
    return data


def runs():
    if not RUNS.exists():
        return []
    rows = []
    with open(RUNS, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue  # dòng cuối có thể đang ghi dở
            if isinstance(row, dict) and "model" in row:
                rows.append(row)
    return rows


def run_count():
    return len(runs())


def chunk_count():
    d = read_json(METRICS / "kb_chunks.json")
    if not isinstance(d, dict):
        return None
    for k in ("n_chunks", "chunks", "total", "n"):
        if isinstance(d.get(k), int):
            return d[k]
    return None
