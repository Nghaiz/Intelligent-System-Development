"""REST API — Hệ thống 1: Sàng lọc nguy cơ tiểu đường (100,000 hồ sơ).

Chạy độc lập trên cổng 5001.

Điểm đáng chú ý nhất của dịch vụ này: nó KHÔNG dùng bất kỳ framework học sâu nào
để suy luận. Trọng số của mạng nơ-ron được nạp từ tệp `.npz` thuần NumPy, và
hàm `forward()` phía dưới tự thực hiện đúng chuỗi phép nhân ma trận + ReLU +
Sigmoid đã viết trong notebook 03. Đây là bằng chứng rằng mạng tự viết tay chạy
được thật trong môi trường triển khai.

Chạy:  python diabetes_large/api/rest_api.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

APP_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = APP_DIR.parent
MODEL_DIR = APP_DIR / "model"
sys.path.insert(0, str(ROOT_DIR))

from knowledge_graph import knowledge  # noqa: E402  (phải nằm sau khi chèn ROOT_DIR vào sys.path)

PORT = 5001
PREFIX = "/diabetes/v1"

# ---------------------------------------------------------------------------
# Nạp artifact suy luận
# ---------------------------------------------------------------------------
META = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
_Z = np.load(MODEL_DIR / "dl_scratch_weights.npz")
_N_LAYERS = sum(1 for k in _Z.files if k.startswith("W"))
WS = [_Z[f"W{i}"] for i in range(_N_LAYERS)]
BS = [_Z[f"b{i}"] for i in range(_N_LAYERS)]

PRE = META["preprocess"]
CAT_COLS = PRE["cat_onehot"]
NUM_COLS = PRE["num_scale"]
BIN_COLS = PRE["bin_pass"]
CATEGORIES = PRE["categories"]
MU = np.array(PRE["mu"], dtype=np.float64)
SD = np.array(PRE["sd"], dtype=np.float64)
THRESHOLD = float(META["dl_model"]["threshold"])
FEATURE_NAMES = META["feature_names"]

# Các trường mà người dùng phải nhập
RAW_NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
RAW_BINARY = ["hypertension", "heart_disease"]
RAW_CATEGORICAL = ["gender", "smoking_history"]

GENDER_OPTIONS = ["Female", "Male"]
SMOKING_OPTIONS = ["No Info", "current", "ever", "former", "never", "not current"]

# Ngưỡng phân tầng nguy cơ — chỉ dùng để TRA CỨU tư vấn, không ảnh hưởng dự đoán
RISK_TIERS = [(0.60, "dia_high"), (0.30, "dia_moderate"), (0.00, "dia_low")]

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# Tiền xử lý — tái hiện CHÍNH XÁC ColumnTransformer của notebook 03
# ---------------------------------------------------------------------------
def build_feature_vector(v: dict) -> np.ndarray:
    """Dựng vector 14 chiều đúng theo thứ tự đã huấn luyện."""
    blocks: list[float] = []

    # 1. One-Hot (drop='first') cho hai biến phân loại
    for col in CAT_COLS:
        val = str(v[col])
        blocks.extend(1.0 if val == c else 0.0 for c in CATEGORIES[col])

    # 2. Hai đặc trưng tương tác lâm sàng — công thức lấy từ notebook 03
    glucose_hba1c = v["blood_glucose_level"] * v["HbA1c_level"] / 100.0
    age_hyper = v["age"] * v["hypertension"]
    raw_num = np.array([v["age"], v["bmi"], v["HbA1c_level"],
                        v["blood_glucose_level"], glucose_hba1c, age_hyper],
                       dtype=np.float64)

    # 3. Chuẩn hoá Z-Score bằng đúng μ, σ đã học trên tập Train
    blocks.extend((raw_num - MU) / SD)

    # 4. Hai biến nhị phân giữ nguyên
    blocks.extend(float(v[c]) for c in BIN_COLS)

    return np.array(blocks, dtype=np.float64).reshape(1, -1)


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


# ---------------------------------------------------------------------------
# Xác thực dữ liệu đầu vào
# ---------------------------------------------------------------------------
def validate(payload):
    if not isinstance(payload, dict):
        return None, "Nội dung yêu cầu phải là một đối tượng JSON."

    values = {}

    for field in RAW_NUMERIC:
        if field not in payload or payload[field] in (None, ""):
            return None, f"Thiếu trường bắt buộc: '{field}'."
        try:
            x = float(payload[field])
        except (TypeError, ValueError):
            return None, f"Trường '{field}' phải là một số."
        if not math.isfinite(x):
            return None, f"Trường '{field}' phải là một số hữu hạn."
        values[field] = x

    for field in RAW_BINARY:
        raw = payload.get(field, 0)
        try:
            b = int(float(raw))
        except (TypeError, ValueError):
            return None, f"Trường '{field}' phải là 0 hoặc 1."
        if b not in (0, 1):
            return None, f"Trường '{field}' phải là 0 hoặc 1."
        values[field] = b

    gender = str(payload.get("gender", "")).strip()
    if gender not in GENDER_OPTIONS:
        return None, f"Trường 'gender' phải thuộc: {', '.join(GENDER_OPTIONS)}."
    values["gender"] = gender

    smoking = str(payload.get("smoking_history", "")).strip()
    if smoking not in SMOKING_OPTIONS:
        return None, f"Trường 'smoking_history' phải thuộc: {', '.join(SMOKING_OPTIONS)}."
    values["smoking_history"] = smoking

    # Chặn các giá trị phi lý sinh học — cùng tiêu chí lọc của notebook 03
    if not (1 <= values["age"] <= 120):
        return None, "Tuổi phải nằm trong khoảng 1 – 120."
    if not (10 <= values["bmi"] <= 80):
        return None, "Chỉ số BMI phải nằm trong khoảng 10 – 80."
    if not (3 <= values["HbA1c_level"] <= 20):
        return None, "Chỉ số HbA1c phải nằm trong khoảng 3 – 20 (%)."
    if not (50 <= values["blood_glucose_level"] <= 500):
        return None, "Đường huyết phải nằm trong khoảng 50 – 500 (mg/dL)."

    return values, None


# ---------------------------------------------------------------------------
# Các điểm cuối (endpoints)
# ---------------------------------------------------------------------------
@app.get(f"{PREFIX}/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Sàng lọc nguy cơ tiểu đường (Hệ thống 1)",
        "model": META["dl_model"]["name"],
        "architecture": META["dl_model"]["layers"],
        "n_params": META["dl_model"]["n_params"],
        "inference": "Forward pass thuần NumPy từ dl_scratch_weights.npz",
        "threshold": THRESHOLD,
        "knowledge_graph": "connected" if knowledge.is_available() else "unavailable",
    })


@app.get(f"{PREFIX}/schema")
def schema():
    return jsonify({
        "numeric": RAW_NUMERIC, "binary": RAW_BINARY,
        "categorical": {"gender": GENDER_OPTIONS, "smoking_history": SMOKING_OPTIONS},
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "threshold": THRESHOLD,
    })


@app.get(f"{PREFIX}/metrics")
def metrics():
    return jsonify({
        "dataset": META["dataset"],
        "dl_model": META["dl_model"],
        "scores": META["scores"],
        "architecture_study": META["architecture_study"],
        "champion": META["champion"],
    })


@app.post(f"{PREFIX}/predict")
def predict():
    values, err = validate(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400

    x = build_feature_vector(values)
    prob = float(forward(x)[0, 0])
    label = int(prob >= THRESHOLD)

    case = knowledge.resolve_case(RISK_TIERS, prob)
    groups, tier_name, k_err = knowledge.fetch(case)

    return jsonify({
        "probability": round(prob, 6),
        "label": label,
        "label_text": "Có nguy cơ mắc tiểu đường" if label else "Nguy cơ thấp",
        "threshold": THRESHOLD,
        "risk_case": case,
        "risk_tier": tier_name,
        "model": {"name": META["dl_model"]["name"],
                  "architecture": META["dl_model"]["layers"],
                  "n_params": META["dl_model"]["n_params"],
                  "inference": "pure NumPy forward pass"},
        "input": values,
        "knowledge": groups,
        "knowledge_error": k_err,
    })


# ---- phục vụ giao diện Web và Mobile ---------------------------------------
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
    print("Hệ thống 1 — Sàng lọc nguy cơ tiểu đường")
    print(f"  Mô hình  : {META['dl_model']['name']} ({META['dl_model']['layers']})")
    print(f"  Tham số  : {META['dl_model']['n_params']:,} · suy luận thuần NumPy")
    print(f"  Ngưỡng   : {THRESHOLD}")
    print(f"  Đồ thị   : {'đã kết nối' if knowledge.is_available() else 'không khả dụng'}")
    print(f"  Web      : http://127.0.0.1:{PORT}/")
    print(f"  Mobile   : http://127.0.0.1:{PORT}/mobile")
    app.run(host="127.0.0.1", port=PORT, debug=False)
