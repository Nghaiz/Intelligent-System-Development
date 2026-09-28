"""Hạng mục 5 (Ch.5): Web App Flask cho hai mô hình hồi quy của bài.

Chạy:  python app/app.py   rồi mở http://localhost:8080
Hai phân hệ: dự báo giá AMZN phiên kế tiếp và dự báo churn KKBox từ chuỗi 31 ngày nghe nhạc.
Mọi mô hình, scaler, ngưỡng và hai người dùng mẫu đọc từ models/ (do notebook 06 ghi), không cần dữ liệu gốc.
"""
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a06.data_stock import latest_window, load_amzn  # noqa: E402
from a06.models_torch import CELLS, CELL_LABELS, build_torch  # noqa: E402

MODELS = ROOT / "models"
META = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
FRAMEWORKS = ("torch", "keras")
torch.set_num_threads(1)

app = Flask(__name__)


@lru_cache(maxsize=None)
def load_model(dataset, framework, cell):
    """Nạp một lần rồi giữ trong bộ nhớ. Trả về hàm f(X: (B, T, D) float32) -> đầu ra thô (B,).

    Chạy thử một lần ngay khi nạp (Keras trace đồ thị ở lần gọi đầu) để độ trễ trả về là của lúc chạy ổn định.
    """
    info = META["models"][f"{dataset}_{framework}_{cell}"]
    path = MODELS / info["file"]
    if framework == "torch":
        model = build_torch(cell)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

        def run(X):
            with torch.no_grad():
                return model(torch.as_tensor(X)).numpy()
    else:
        import keras                              # chỉ nạp TensorFlow khi thật sự cần mô hình Keras
        import tensorflow as tf
        model = keras.models.load_model(path)
        graph = tf.function(lambda X: model(X, training=False), reduce_retracing=True)   # eager chậm hơn ~100 lần

        def run(X):
            return graph(X).numpy().ravel()
    run(np.zeros((1, META["seq_len"][dataset], 8), dtype=np.float32))
    return run


def pick(dataset, body):
    """Mô hình người dùng chọn, mặc định là mô hình có val loss nhỏ nhất của framework đó."""
    framework = body.get("framework", "torch")
    if framework not in FRAMEWORKS:
        raise ValueError(f"framework phải là một trong {FRAMEWORKS}")
    cell = body.get("model") or META["best"][f"{dataset}_{framework}"]
    if cell not in CELLS:
        raise ValueError(f"model phải là một trong {CELLS}")
    return framework, cell


def timed(fn, X):
    t0 = time.perf_counter()
    out = fn(X)
    return out, 1000 * (time.perf_counter() - t0)


# ---------------------------------------------------------------- phân hệ 1: giá AMZN
STOCK_DF = load_amzn()
S_SCALER = META["scalers"]["amzn"]


@app.get("/api/stock/history")
def stock_history():
    tail = STOCK_DF.tail(100)
    return jsonify({"dates": [d.strftime("%Y-%m-%d") for d in tail.index],
                    "close": tail.Close.round(2).tolist(), "ma20": tail.MA20.round(2).tolist()})


@app.post("/api/stock/predict")
def stock_predict():
    body = request.get_json(silent=True) or {}
    try:
        framework, cell = pick("amzn", body)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    w, close, last = latest_window(STOCK_DF)
    X = ((w - np.array(S_SCALER["mean"])) / np.array(S_SCALER["std"])).astype(np.float32)[None]
    z, ms = timed(load_model("amzn", framework, cell), X)
    r_hat = float(z[0]) * S_SCALER["y_std"] + S_SCALER["y_mean"]     # return đã chuẩn hoá -> log-return
    pred = close * float(np.exp(r_hat))
    return jsonify({"framework": framework, "model": cell, "model_label": CELL_LABELS[cell],
                    "last_date": last.strftime("%Y-%m-%d"), "last_close": round(close, 2),
                    "pred_close": round(pred, 2), "change_pct": round(100 * (pred / close - 1), 3),
                    "latency_ms": round(ms, 3)})


# ---------------------------------------------------------------- phân hệ 2: churn KKBox
C_SCALER = META["scalers"]["kkbox"]


@app.post("/api/churn/predict")
def churn_predict():
    body = request.get_json(silent=True) or {}
    try:
        framework, cell = pick("kkbox", body)
        if "preset" in body:
            seq = np.array(META["presets"][body["preset"]]["sequence"], dtype=np.float64)
        else:
            seq = np.array(body["sequence"], dtype=np.float64)
        if seq.shape != (31, 8):
            raise ValueError(f"sequence phải có shape (31, 8), nhận {seq.shape}")
    except KeyError as e:
        return jsonify({"error": f"thiếu hoặc sai khoá {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    X = ((seq - np.array(C_SCALER["mean"])) / np.array(C_SCALER["std"])).astype(np.float32)[None]
    z, ms = timed(load_model("kkbox", framework, cell), X)
    prob = float(1 / (1 + np.exp(-z[0])))
    thr = META["thresholds"][f"kkbox_{framework}_{cell}"]
    risk = "cao" if prob >= thr else "trung bình" if prob >= thr / 2 else "thấp"
    return jsonify({"framework": framework, "model": cell, "model_label": CELL_LABELS[cell],
                    "prob": round(prob, 4), "threshold": round(thr, 4), "risk": risk,
                    "latency_ms": round(ms, 3), "sequence": seq.round(4).tolist()})


@app.get("/api/meta")
def meta():
    return jsonify({"models": {c: CELL_LABELS[c] for c in CELLS}, "frameworks": FRAMEWORKS,
                    "best": META["best"], "presets": {k: v["label"] for k, v in META["presets"].items()},
                    "features": META["kkbox_features"]})


@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
