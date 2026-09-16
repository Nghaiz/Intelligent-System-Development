"""REST API — Hệ thống 3: Phân loại nhận xét thương mại điện tử (23,486 đánh giá).

Chạy độc lập trên cổng 5003.

Luồng suy luận gồm hai chặng:

    văn bản thô  →  TF-IDF 1,000 chiều (joblib)  →  forward pass NumPy  →  xác suất

Bộ vector hoá TF-IDF phải nạp bằng joblib vì bảng từ điển và trọng số IDF là dữ
liệu học được từ tập Train. Nhưng phần mạng nơ-ron thì vẫn thuần NumPy: không có
framework học sâu nào tham gia vào việc tính ra con số cuối cùng.

Chạy:  python customer_comments/api/rest_api.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

APP_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = APP_DIR.parent
MODEL_DIR = APP_DIR / "model"
sys.path.insert(0, str(ROOT_DIR))

from knowledge_graph import knowledge  # noqa: E402

PORT = 5003
PREFIX = "/comments/v1"

# ---------------------------------------------------------------------------
# Nạp artifact suy luận
# ---------------------------------------------------------------------------
META = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
VECTORIZER = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

_Z = np.load(MODEL_DIR / "dl_scratch_weights.npz")
_N_LAYERS = sum(1 for k in _Z.files if k.startswith("W"))
WS = [_Z[f"W{i}"] for i in range(_N_LAYERS)]
BS = [_Z[f"b{i}"] for i in range(_N_LAYERS)]

THRESHOLD = float(META["dl_model"]["threshold"])
SENTIMENT_TERMS = META["sentiment_terms"]

# Vùng lân cận ngưỡng được coi là "trung tính" — mô hình chưa đủ tự tin
NEUTRAL_BAND = 0.12

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# Suy luận
# ---------------------------------------------------------------------------
def forward(X: np.ndarray) -> np.ndarray:
    """Lan truyền tiến thuần NumPy — ReLU ở tầng ẩn, Sigmoid ở tầng ra."""
    A = X
    for i, (W, b) in enumerate(zip(WS, BS)):
        Z = A @ W + b
        if i < len(WS) - 1:
            A = np.maximum(0.0, Z)
        else:
            A = 1.0 / (1.0 + np.exp(-np.clip(Z, -25, 25)))
    return A


def vectorize(texts: list[str]) -> np.ndarray:
    return np.asarray(VECTORIZER.transform(texts).todense(), dtype=np.float64)


def resolve_sentiment_case(prob: float) -> str:
    """Ba tầng: tiêu cực · trung tính (quanh ngưỡng) · tích cực."""
    if prob < THRESHOLD - NEUTRAL_BAND:
        return "cmt_negative"
    if prob < THRESHOLD + NEUTRAL_BAND:
        return "cmt_neutral"
    return "cmt_positive"


def explain(text: str, top_k: int = 6) -> list[dict]:
    """Liệt kê các từ khoá trong câu có trọng số cảm xúc mạnh nhất.

    Đây là phần diễn giải: cho người dùng thấy mô hình "nhìn" vào từ nào. Trọng
    số lấy từ hệ số Logistic Regression đã huấn luyện cùng bộ đặc trưng.
    """
    weights = {w: c for w, c in SENTIMENT_TERMS["positive"]}
    weights.update({w: c for w, c in SENTIMENT_TERMS["negative"]})
    low = text.lower()
    hits = [{"term": w, "weight": round(c, 3),
             "polarity": "tích cực" if c > 0 else "tiêu cực"}
            for w, c in weights.items() if w in low]
    hits.sort(key=lambda h: abs(h["weight"]), reverse=True)
    return hits[:top_k]


def validate(payload):
    if not isinstance(payload, dict):
        return None, "Nội dung yêu cầu phải là một đối tượng JSON."
    text = payload.get("text")
    if text is None:
        # cho phép gửi riêng title và review giống cấu trúc dữ liệu gốc
        title = str(payload.get("title", "") or "").strip()
        review = str(payload.get("review", "") or "").strip()
        text = f"{title} {review}".strip()
    text = str(text).strip()
    if not text:
        return None, "Thiếu nội dung nhận xét: cần trường 'text' (hoặc 'title' + 'review')."
    if len(text) > 5000:
        return None, "Nội dung nhận xét quá dài (giới hạn 5,000 ký tự)."
    return text, None


# ---------------------------------------------------------------------------
# Các điểm cuối
# ---------------------------------------------------------------------------
@app.get(f"{PREFIX}/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Phân loại nhận xét thương mại điện tử (Hệ thống 3)",
        "model": META["dl_model"]["name"],
        "architecture": META["dl_model"]["layers"],
        "n_params": META["dl_model"]["n_params"],
        "vectorizer": f"TF-IDF {META['vectorizer']['max_features']} chiều, "
                      f"ngram {tuple(META['vectorizer']['ngram_range'])}",
        "threshold": THRESHOLD,
        "knowledge_graph": "connected" if knowledge.is_available() else "unavailable",
    })


@app.get(f"{PREFIX}/schema")
def schema():
    return jsonify({
        "fields": {"text": "Nội dung nhận xét đầy đủ (hoặc gửi 'title' + 'review')"},
        "vectorizer": META["vectorizer"],
        "threshold": THRESHOLD,
        "max_length": 5000,
    })


@app.get(f"{PREFIX}/metrics")
def metrics():
    return jsonify({
        "dataset": META["dataset"], "dl_model": META["dl_model"],
        "scores": META["scores"], "architecture_study": META["architecture_study"],
        "top_keywords": META["top_keywords"], "champion": META["champion"],
    })


@app.get(f"{PREFIX}/terms")
def terms():
    return jsonify(SENTIMENT_TERMS)


@app.post(f"{PREFIX}/predict")
def predict():
    text, err = validate(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400

    X = vectorize([text])
    prob = float(forward(X)[0, 0])
    label = int(prob >= THRESHOLD)

    case = resolve_sentiment_case(prob)
    groups, tier_name, k_err = knowledge.fetch(case)

    n_active = int((X > 0).sum())
    return jsonify({
        "probability": round(prob, 6),
        "label": label,
        "label_text": "Khách hàng KHUYẾN NGHỊ sản phẩm" if label
                      else "Khách hàng KHÔNG khuyến nghị sản phẩm",
        "threshold": THRESHOLD,
        "sentiment_case": case,
        "sentiment_tier": tier_name,
        "matched_terms": explain(text),
        "tfidf": {"n_active_dims": n_active,
                  "total_dims": int(X.shape[1]),
                  "sparsity_pct": round((1 - n_active / X.shape[1]) * 100, 2)},
        "model": {"name": META["dl_model"]["name"],
                  "architecture": META["dl_model"]["layers"],
                  "n_params": META["dl_model"]["n_params"],
                  "inference": "pure NumPy forward pass"},
        "input": {"text": text, "n_words": len(text.split())},
        "knowledge": groups,
        "knowledge_error": k_err,
    })


# ---- phục vụ giao diện -----------------------------------------------------
@app.get("/")
def web_index():
    return send_from_directory(APP_DIR / "web", "index.html")


@app.get("/web/<path:filename>")
def web_files(filename):
    return send_from_directory(APP_DIR / "web", filename)


@app.get("/mobile")
@app.get("/mobile/")
def mobile_index():
    return send_from_directory(APP_DIR / "mobile", "index.html")


@app.get("/mobile/<path:filename>")
def mobile_files(filename):
    return send_from_directory(APP_DIR / "mobile", filename)


if __name__ == "__main__":
    print("Hệ thống 3 — Phân loại nhận xét thương mại điện tử")
    print(f"  Mô hình  : {META['dl_model']['name']} ({META['dl_model']['layers']})")
    print(f"  Tham số  : {META['dl_model']['n_params']:,} · suy luận thuần NumPy")
    print(f"  TF-IDF   : {META['vectorizer']['max_features']} chiều")
    print(f"  Ngưỡng   : {THRESHOLD}")
    print(f"  Đồ thị   : {'đã kết nối' if knowledge.is_available() else 'không khả dụng'}")
    print(f"  Web      : http://127.0.0.1:{PORT}/")
    print(f"  Mobile   : http://127.0.0.1:{PORT}/mobile")
    app.run(host="127.0.0.1", port=PORT, debug=False)
