"""Nơi duy nhất ghi dữ liệu ra đĩa cho báo cáo.

- outputs/figdata/*.dat  : bảng cách nhau bằng dấu cách, pgfplots đọc trực tiếp
- outputs/metrics/*.json : số liệu, build.py của báo cáo biến thành macro LaTeX
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDATA = ROOT / "outputs" / "figdata"
METRICS = ROOT / "outputs" / "metrics"
MODELS = ROOT / "models"
DATA = ROOT / "data"
for _d in (FIGDATA, METRICS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def write_dat(name, columns):
    """columns: dict tên_cột -> dãy giá trị cùng độ dài. Ghi FIGDATA/name.dat."""
    columns = {k: np.asarray(v) for k, v in columns.items()}   # Series pandas đánh chỉ số theo nhãn, không theo vị trí
    keys = list(columns)
    n = len(columns[keys[0]])
    if any(len(columns[k]) != n for k in keys):
        raise ValueError(f"{name}: các cột dài khác nhau")
    path = FIGDATA / f"{name}.dat"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(" ".join(keys) + "\n")
        for i in range(n):
            f.write(" ".join(_fmt(columns[k][i]) for k in keys) + "\n")
    return path


def write_matrix_dat(name, M):
    """Ma trận số (H, W) -> cột 'x y v' cho matrix plot* với colormap (hàng 0 ở trên cùng)."""
    M = np.asarray(M, dtype=float)
    H, W = M.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return write_dat(name, {"x": xs.ravel(), "y": (H - 1 - ys).ravel(), "v": M.ravel()})


def _to_builtin(o):
    if isinstance(o, dict):
        return {str(k): _to_builtin(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return _to_builtin(o.tolist())
    if isinstance(o, (list, tuple)):
        return [_to_builtin(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_metrics(name, data):
    """Ghi METRICS/name.json (ghi đè). Khoá lồng nhau thành macro a/b/c."""
    path = METRICS / f"{name}.json"
    path.write_text(json.dumps(_to_builtin(data), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_metrics(name):
    return json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8"))
