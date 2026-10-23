"""Đường dẫn và ghi kết quả dùng chung cho Bài tập Agent 01."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"
REPORT = ROOT.parents[1] / "Report" / "Agent 01"
GEN = REPORT / "generated"
FIGS = REPORT / "figures"
SEED = 42

for _d in (METRICS, GEN, FIGS):
    _d.mkdir(parents=True, exist_ok=True)


def save_json(name, obj):
    """Ghi outputs/metrics/<name>.json — nguồn duy nhất của mọi con số \\SL{name/...} trong báo cáo."""
    path = METRICS / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_to_py), encoding="utf-8")
    return path


def load_json(name):
    return json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8"))


def _to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Không tuần tự hóa được {type(o)}")
