"""Hạng mục 5 (Ch.5): Web App Flask cho hai mô hình hồi quy của bài.

Chạy:  python app/app.py   rồi mở http://localhost:8080
Hai phân hệ: dự báo giá AMZN phiên kế tiếp và dự báo churn KKBox từ chuỗi 31 ngày nghe nhạc.
Mọi mô hình, scaler, ngưỡng và hai người dùng mẫu đọc từ models/ (do notebook 06 ghi), chỉ số test của thẻ mô hình
đọc từ outputs/metrics/; không cần dữ liệu gốc của KKBox.
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


HISTORY_SIZES = (60, 100, 250)


@app.get("/api/stock/history")
def stock_history():
    """n phiên gần nhất (60, 100 hoặc 250): Close, MA20, RSI-14 và thông tin phiên cuối cho thẻ thống kê."""
    n = request.args.get("n", 100, type=int)
    if n not in HISTORY_SIZES:
        return jsonify({"error": f"n phải là một trong {HISTORY_SIZES}"}), 400
    tail = STOCK_DF.tail(n)
    last, prev = STOCK_DF.iloc[-1], STOCK_DF.iloc[-2]
    return jsonify({"dates": [d.strftime("%Y-%m-%d") for d in tail.index],
                    "close": tail.Close.round(2).tolist(), "ma20": tail.MA20.round(2).tolist(),
                    "rsi14": tail.RSI14.round(2).tolist(),
                    "last": {"open": round(float(last.Open), 2), "high": round(float(last.High), 2),
                             "low": round(float(last.Low), 2), "close": round(float(last.Close), 2),
                             "volume": int(last.Volume), "change_pct": round(100 * (last.Close / prev.Close - 1), 3)}})


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
    rsi = float(STOCK_DF.RSI14.iloc[-1])
    return jsonify({"framework": framework, "model": cell, "model_label": CELL_LABELS[cell],
                    "last_date": last.strftime("%Y-%m-%d"), "last_close": round(close, 2),
                    "pred_close": round(pred, 2), "change_pct": round(100 * (pred / close - 1), 3),
                    "rsi14": round(rsi, 2), "rsi_zone": rsi_zone(rsi), "note": STOCK_NOTE,
                    "latency_ms": round(ms, 3)})


# Ghi chú đi kèm mọi dự báo giá: Chương 4 cho thấy không mô hình nào thắng mốc "giá ngày mai = giá hôm nay".
STOCK_NOTE = ("Mô hình không thắng được mốc 'giá ngày mai bằng giá hôm nay' trên tập test; "
              "dự báo chỉ để minh hoạ, không dùng làm khuyến nghị đầu tư.")


def rsi_zone(rsi):
    """Ngưỡng 70/30 quy ước của Wilder."""
    return "quá mua" if rsi >= 70 else "quá bán" if rsi <= 30 else "trung tính"


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
                    "advice": ADVICE[risk], "trend": listening_trend(seq),
                    "latency_ms": round(ms, 3), "sequence": seq.round(4).tolist()})


# Khuyến nghị theo mức rủi ro cho bộ phận chăm sóc khách hàng.
ADVICE = {"cao": "Liên hệ trước ngày hết hạn gói, tặng ưu đãi gia hạn và gợi ý danh sách phát theo lịch sử nghe.",
          "trung bình": "Gửi nhắc gia hạn và gợi ý nội dung mới; theo dõi hành vi nghe trong tuần tới.",
          "thấp": "Không cần can thiệp; giữ chăm sóc thông thường."}


def listening_trend(seq):
    """Xu hướng nghe: log(1+giây) trung bình 7 ngày cuối trừ 7 ngày đầu (Chương 2: đây là tín hiệu mạnh nhất)."""
    secs = seq[:, META["kkbox_features"].index("total_secs")]
    return round(float(secs[-7:].mean() - secs[:7].mean()), 4)


# Chỉ số test của 16 mô hình và thống kê hai tập, do notebook 03-06 ghi; giao diện hiện chúng trên thẻ mô hình.
METRICS = ROOT / "outputs" / "metrics"
BENCH = json.loads((METRICS / "bench.json").read_text(encoding="utf-8"))
DATA_STATS = {name: json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8")) for name in ("amzn", "kkbox")}
CARD_KEYS = {"amzn": ("rmse", "mae", "dir_acc"), "kkbox": ("f1", "roc_auc", "pr_auc")}


def model_card(key):
    dataset = key.split("_")[0]
    b = BENCH["models"][key]
    card = {k: round(b[k], 4) for k in CARD_KEYS[dataset]}
    card.update(params=b["params"], latency_ms=round(b["latency_ms"], 3), size_kb=round(b["size_kb"], 1),
                best_epoch=b["best_epoch"], val_loss=round(b["best_val_loss"], 4))
    return card


@app.get("/api/meta")
def meta():
    amzn, kkbox = DATA_STATS["amzn"], DATA_STATS["kkbox"]
    # jsonify sắp khoá theo chữ cái, nên thứ tự hiển thị RNN -> LSTM -> GRU -> BiLSTM đi riêng trong "cells"
    return jsonify({"models": {c: CELL_LABELS[c] for c in CELLS}, "cells": list(CELLS), "frameworks": FRAMEWORKS,
                    "best": META["best"], "cards": {k: model_card(k) for k in BENCH["models"]},
                    "naive_rmse": round(BENCH["naive"]["rmse"], 4),
                    "presets": {k: {"label": v["label"], "sequence": v["sequence"]} for k, v in META["presets"].items()},
                    "features": META["kkbox_features"], "seq_len": META["seq_len"],
                    "stats": {"amzn": {"n": amzn["n_feat"], "start": amzn["start"], "end": amzn["end"],
                                       "window": amzn["window"], "n_features": amzn["n_features"]},
                              "kkbox": {"n_sample": kkbox["n_sample"], "n_test": kkbox["n_test"],
                                        "churn_rate": round(kkbox["churn_rate_sample"], 4), "T": kkbox["T"]}}})


@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
