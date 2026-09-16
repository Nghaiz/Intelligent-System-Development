"""REST API cho ứng dụng dự đoán hành vi khách hàng (Customer Behavior).

Chạy độc lập trên cổng 5003. Toàn bộ artifact suy luận (mô hình + bộ tiền xử
lý) được nạp bằng joblib từ thư mục `../model` khi khởi động — ứng dụng
KHÔNG huấn luyện lại bất cứ thứ gì trong quá trình phục vụ. Tùy theo cách
biểu diễn (tabular / hybrid / text) của từng mô hình, bộ tiền xử lý tương
ứng sẽ được chọn để biến đổi dữ liệu đầu vào.
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "model"
WEB_DIR = APP_DIR / "web"
MOBILE_DIR = APP_DIR / "mobile"

MODEL_FILES = {
    "logistic_regression_tabular": "logistic_regression_tabular.joblib",
    "decision_tree_tabular": "decision_tree_tabular.joblib",
    "random_forest_tabular": "random_forest_tabular.joblib",
    "logistic_regression_hybrid": "logistic_regression_hybrid.joblib",
    "linear_svm_hybrid": "linear_svm_hybrid.joblib",
    "multinomial_nb_text": "multinomial_nb_text.joblib",
}

REPRESENTATION_LABELS = {
    "tabular": "Chỉ bảng",
    "hybrid": "Bảng + văn bản",
    "text": "Chỉ văn bản",
}

metadata_path = MODEL_DIR / "metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(f"Không tìm thấy file metadata: {metadata_path}")

with metadata_path.open("r", encoding="utf-8") as fh:
    METADATA = json.load(fh)

NUMERIC_FEATURES = METADATA["numeric_features"]
CATEGORICAL_FEATURES = METADATA["categorical_features"]
TEXT_COLUMN = METADATA.get("text_column", "full_text")
TABULAR_FEATURES = METADATA.get("tabular_features", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
CATEGORICAL_OPTIONS = METADATA.get("categorical_options", {})
CLASS_LABELS = METADATA.get("class_labels", {"0": "Không khuyến nghị", "1": "Khuyến nghị"})
MODEL_REPR = METADATA.get("model_repr", {})
MODEL_LABELS = METADATA.get("model_labels", {})
BEST_MODEL = METADATA.get("best_model", "logistic_regression_hybrid")


def load_artifact(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {path}")
    return joblib.load(path)


preprocessor_hybrid = load_artifact("preprocessor.joblib")
preprocessor_tabular = load_artifact("preprocessor_tabular.joblib")
vectorizer_text = load_artifact("tfidf_text_only.joblib")
loaded_models = {model_id: load_artifact(filename) for model_id, filename in MODEL_FILES.items()}

FULL_COLUMNS = ["Age", "Positive Feedback Count", "word_count", "Division Name", "Department Name", "Class Name", "full_text"]

app = Flask(__name__)


def validate_payload(payload):
    if not isinstance(payload, dict):
        return None, None, "Nội dung yêu cầu phải là một đối tượng JSON."

    title = str(payload.get("Title") or "").strip()
    review_text = str(payload.get("Review Text") or "").strip()
    if not review_text:
        return None, None, "Trường 'Review Text' không được để trống."

    full_text = f"{title} {review_text}".strip()
    word_count = len(full_text.split())

    values = {"full_text": full_text, "word_count": word_count}

    age_raw = payload.get("Age")
    if age_raw is None or age_raw == "":
        return None, None, "Thiếu trường bắt buộc: Age."
    try:
        age = float(age_raw)
    except (TypeError, ValueError):
        return None, None, "Trường 'Age' phải là một số."
    if not math.isfinite(age) or age <= 0 or age > 120:
        return None, None, "Trường 'Age' phải trong khoảng hợp lệ (1-120)."
    values["Age"] = age

    feedback_raw = payload.get("Positive Feedback Count", 0)
    if feedback_raw in (None, ""):
        feedback_raw = 0
    try:
        feedback_count = float(feedback_raw)
    except (TypeError, ValueError):
        return None, None, "Trường 'Positive Feedback Count' phải là một số."
    if not math.isfinite(feedback_count) or feedback_count < 0:
        return None, None, "Trường 'Positive Feedback Count' không được âm."
    values["Positive Feedback Count"] = feedback_count

    for field in CATEGORICAL_FEATURES:
        raw = payload.get(field)
        options = CATEGORICAL_OPTIONS.get(field, [])
        if raw is None or raw == "":
            raw = "Unknown" if "Unknown" in options else (options[0] if options else "Unknown")
        elif options and raw not in options:
            return None, None, f"Giá trị '{raw}' không hợp lệ cho trường '{field}'."
        values[field] = raw

    model_id = payload.get("model") or BEST_MODEL
    if model_id not in loaded_models:
        return None, None, f"Mô hình không hợp lệ. Các mô hình hỗ trợ: {', '.join(loaded_models)}"

    return values, model_id, None


def compute_confidence(model, X, prediction_class):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        return round(float(probabilities[prediction_class]) * 100, 2)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        score = float(score[0]) if hasattr(score, "__len__") else float(score)
        prob = 1.0 / (1.0 + math.exp(-abs(score)))
        return round(prob * 100, 2)
    return None


@app.get("/")
@app.get("/web")
def web_client():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/mobile")
def mobile_client():
    return send_from_directory(MOBILE_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        application="customer_behavior",
        models=list(loaded_models),
        feature_columns=TABULAR_FEATURES + [TEXT_COLUMN],
        best_model=BEST_MODEL,
    )


@app.get("/metadata")
def metadata():
    return jsonify(METADATA)


@app.get("/models")
def models():
    """Danh sách mô hình khả dụng kèm điểm số — đọc từ metadata, không hardcode."""
    test_metrics = METADATA.get("test_metrics", {})
    items = [
        {
            "id": model_id,
            "label": MODEL_LABELS.get(model_id, model_id),
            "representation": REPRESENTATION_LABELS.get(MODEL_REPR.get(model_id, "hybrid"), MODEL_REPR.get(model_id, "")),
            "is_best": model_id == BEST_MODEL,
            "metrics": test_metrics.get(model_id, {}),
        }
        for model_id in loaded_models
    ]
    return jsonify(application="customer_behavior", best_model=BEST_MODEL, models=items)


@app.post("/customer-behavior/v1/predict")
def predict():
    payload = request.get_json(silent=True)
    values, model_id, error = validate_payload(payload)
    if error:
        return jsonify(error=error), 400

    representation = MODEL_REPR.get(model_id, "hybrid")
    selected_model = loaded_models[model_id]

    if representation == "text":
        X = vectorizer_text.transform([values["full_text"]])
    else:
        row = {col: values.get(col) for col in FULL_COLUMNS}
        raw_input = pd.DataFrame([row], columns=FULL_COLUMNS)
        pre = preprocessor_tabular if representation == "tabular" else preprocessor_hybrid
        X = pre.transform(raw_input)

    prediction_class = int(selected_model.predict(X)[0])
    confidence = compute_confidence(selected_model, X, prediction_class)

    prediction_label = CLASS_LABELS.get(str(prediction_class), "Không xác định")
    model_label = MODEL_LABELS.get(model_id, model_id)

    if prediction_class == 1:
        interpretation = (
            "Dựa trên nội dung đánh giá và thông tin sản phẩm, mô hình dự đoán khách hàng có khả năng "
            "khuyến nghị sản phẩm này cho người khác."
        )
    else:
        interpretation = (
            "Dựa trên nội dung đánh giá và thông tin sản phẩm, mô hình dự đoán khách hàng khó có khả năng "
            "khuyến nghị sản phẩm này."
        )

    return jsonify(
        application="customer_behavior",
        model=model_id,
        model_label=model_label,
        prediction_class=prediction_class,
        prediction=prediction_label,
        confidence=confidence,
        representation=REPRESENTATION_LABELS.get(representation, representation),
        interpretation=interpretation,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
