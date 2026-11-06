"""Chương 4: suy luận cho Web App. Dự báo N bước cho từng mã và từng chuỗi, kèm khoảng dao động 90 %.

Mô hình nạp lười từ models/ch4/, nạp xong giữ lại, chạy CPU. Đặc trưng dựng lại từ CSV trong data/ch4/ bằng đúng hàm
lúc huấn luyện rồi chuẩn hóa bằng thống kê tập train lưu trong tệp mô hình.
"""
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from .ch4_data import SPECS as SERIES, frame_of
from .ch4_torch import ARCH_NAMES, ARCHS, SeqRegressor
from .common import METRICS, model_dir

DEFAULT_ARCH = "gru"
HISTORY_POINTS = 120


def _spec(s):
    return {"id": f"ch4_{s.id}", "title": f"Dự báo {s.name}", "dataset": s.id, "output": "series",
            "inputs": [{"name": "steps", "type": "number", "label": "Số bước dự báo", "default": min(s.horizon, 10),
                        "min": 1, "max": s.horizon},
                       {"name": "model", "type": "select", "label": "Kiến trúc", "default": DEFAULT_ARCH,
                        "options": [{"value": a, "label": ARCH_NAMES[a]} for a in ARCHS]}],
            "unit": s.unit, "note": "Kết quả chỉ mang tính học thuật, không phải khuyến nghị đầu tư."}


SPECS = [_spec(s) for s in SERIES.values()]


@lru_cache(maxsize=None)
def _load(series, arch):
    """weights_only chặn pickle tùy ý khi đọc tệp mô hình."""
    import torch
    path = model_dir("ch4") / f"{series}__{arch}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có mô hình {arch} cho {series}: {path.name}")
    st = torch.load(path, map_location="cpu", weights_only=True)
    spec = SERIES[series]
    model = SeqRegressor(len(st["columns"]), spec.horizon, arch, st["hidden"], st["dropout"])
    model.load_state_dict(st["state"])
    return model.eval(), st


@lru_cache(maxsize=None)
def _frame(series):
    return frame_of(series)


def _future(index, spec, n):
    """Mốc thời gian tương lai; lịch phiên chỉ bỏ cuối tuần, chưa trừ ngày lễ."""
    return pd.date_range(index[-1], periods=n + 1, freq=spec.freq)[1:]


@lru_cache(maxsize=None)
def _skill(series, arch):
    """RMSE một bước trên tập test của mô hình và của Persistence, để giao diện nói rõ mô hình có thắng không."""
    rows = json.loads((METRICS / "ch4_compare.json").read_text(encoding="utf-8"))["rows"]
    pick = {(r["model"], r["framework"]): r["rmse"] for r in rows if r["dataset"] == series}
    return {"rmse": pick.get((ARCH_NAMES[arch], "pytorch")), "rmse_persistence": pick.get(("Persistence", "baseline"))}


def _forecast(series, arch):
    """Trả mảng (H,) giá trị dự báo cùng biên dưới, biên trên ở đơn vị gốc, và khung dữ liệu dùng để dự báo."""
    import torch
    spec, (model, st), df = SERIES[series], _load(series, arch), _frame(series)
    raw = df[st["columns"]].to_numpy(np.float64)[-spec.window:]
    x = (raw - st["mean"].numpy()) / st["std"].numpy()
    with torch.no_grad():
        out = model(torch.as_tensor(x[None], dtype=torch.float32))[0].numpy()
    sd, q = st["sd_h"].numpy(), st["q"].numpy()
    last = float(df["target"].iloc[-1])
    base = np.log(last) if spec.log_target else last
    mid, lo, hi = (base + v for v in (out * sd, out * sd + q[0], out * sd + q[1]))
    if spec.log_target:
        return np.exp(mid), np.exp(lo), np.exp(hi), df
    return mid, np.maximum(lo, 0.0) if series == "power" else lo, hi, df   # công suất không âm


def predict(id, payload):
    """payload: {steps, model}. Trả history {t, y} và forecast {t, y, lo, hi}, cùng đơn vị với chuỗi gốc."""
    series = str(id).removeprefix("ch4_")
    if series not in SERIES:
        raise KeyError(f"id không hợp lệ: {id}")
    arch = payload.get("model") or DEFAULT_ARCH
    if arch not in ARCHS:
        raise ValueError(f"Kiến trúc không hợp lệ: {arch}")
    spec = SERIES[series]
    try:
        steps = int(payload.get("steps") or min(spec.horizon, 10))
    except (TypeError, ValueError) as e:
        raise ValueError("steps phải là số nguyên") from e
    if not 1 <= steps <= spec.horizon:
        raise ValueError(f"steps phải trong khoảng 1 đến {spec.horizon}")
    t0 = time.perf_counter()
    mid, lo, hi, df = _forecast(series, arch)
    past = df["target"].iloc[-HISTORY_POINTS:]
    fmt = "%Y-%m-%d" if spec.log_target else "%Y-%m-%d %H:%M"
    return {"history": {"t": past.index.strftime(fmt).tolist(), "y": [float(v) for v in past]},
            "forecast": {"t": _future(df.index, spec, steps).strftime(fmt).tolist(), "y": mid[:steps].tolist(),
                         "lo": lo[:steps].tolist(), "hi": hi[:steps].tolist()},
            "interval": "90 % theo phân vị sai số validation", "model": ARCH_NAMES[arch], "unit": spec.unit,
            "dataset": spec.name, "skill": _skill(series, arch), "latency_ms": 1000 * (time.perf_counter() - t0)}
