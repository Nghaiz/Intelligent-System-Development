"""REST API cho ứng dụng dự đoán giá nhà (House Price).

Chạy độc lập trên cổng 5002. Toàn bộ artifact suy luận (mô hình + bộ tiền xử
lý) được nạp bằng joblib từ thư mục `../model` khi khởi động — ứng dụng
KHÔNG huấn luyện lại bất cứ thứ gì trong quá trình phục vụ. Các đặc trưng
phái sinh (Total_Area, Room_Density, Frontage_Ratio, Area_per_Bedroom,
Log_Area) được tính lại đúng công thức đã dùng khi huấn luyện.
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
    "linear_regression": {"label": "Linear Regression", "filename": "linear_regression.joblib"},
    "ridge_regression": {"label": "Ridge Regression", "filename": "ridge_regression.joblib"},
    "decision_tree_regressor": {"label": "Decision Tree Regressor", "filename": "decision_tree_regressor.joblib"},
    "random_forest_regressor": {"label": "Random Forest Regressor", "filename": "random_forest_regressor.joblib"},
    "gradient_boosting_regressor": {"label": "Gradient Boosting Regressor", "filename": "gradient_boosting_regressor.joblib"},
}

BASE_NUMERIC_FIELDS = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
REQUIRED_NUMERIC_FIELDS = ["Area"]
OPTIONAL_NUMERIC_FIELDS = ["Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]

metadata_path = MODEL_DIR / "metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(f"Không tìm thấy file metadata: {metadata_path}")

with metadata_path.open("r", encoding="utf-8") as fh:
    METADATA = json.load(fh)

NUMERIC_FEATURES = METADATA["numeric_features"]
CATEGORICAL_FEATURES = METADATA["categorical_features"]
FEATURE_COLUMNS = METADATA["feature_columns"]
CATEGORICAL_OPTIONS = METADATA.get("categorical_options", {})
TARGET_UNIT = METADATA.get("target_unit", "tỷ VNĐ")
RESIDUAL_STD = METADATA.get("residual_std", 0.0)
BEST_MODEL = METADATA.get("best_model", "gradient_boosting_regressor")
VALUE_RANGES = METADATA.get("value_ranges", {})


def load_artifact(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {path}")
    return joblib.load(path)


preprocessor = load_artifact("preprocessor.joblib")
loaded_models = {model_id: load_artifact(cfg["filename"]) for model_id, cfg in MODEL_FILES.items()}

app = Flask(__name__)


def compute_derived_features(base_values):
    """Tính lại các đặc trưng phái sinh đúng công thức đã dùng lúc huấn luyện."""
    area = base_values.get("Area")
    floors = base_values.get("Floors")
    bedrooms = base_values.get("Bedrooms")
    bathrooms = base_values.get("Bathrooms")
    frontage = base_values.get("Frontage")

    floors_safe = floors if floors and floors > 0 else 1
    bedrooms_safe = bedrooms if bedrooms and bedrooms > 0 else 1
    area_safe = area if area and area > 0 else 1

    derived = {}
    if "Total_Area" in NUMERIC_FEATURES:
        derived["Total_Area"] = area * max(floors_safe, 1)
    if "Room_Density" in NUMERIC_FEATURES:
        bedrooms_for_density = bedrooms if bedrooms is not None else 0
        bathrooms_for_density = bathrooms if bathrooms is not None else 0
        derived["Room_Density"] = (bedrooms_for_density + bathrooms_for_density) / max(area_safe, 1)
    if "Frontage_Ratio" in NUMERIC_FEATURES:
        derived["Frontage_Ratio"] = (frontage / max(area_safe, 1)) if frontage is not None else np.nan
    if "Area_per_Bedroom" in NUMERIC_FEATURES:
        derived["Area_per_Bedroom"] = area / max(bedrooms_safe, 1)
    if "Log_Area" in NUMERIC_FEATURES:
        derived["Log_Area"] = np.log1p(area)
    return derived


def validate_payload(payload):
    if not isinstance(payload, dict):
        return None, None, "Nội dung yêu cầu phải là một đối tượng JSON."

    if "Area" not in payload or payload["Area"] in (None, ""):
        return None, None, "Thiếu trường bắt buộc: Area."

    base_values = {}
    try:
        area = float(payload["Area"])
    except (TypeError, ValueError):
        return None, None, "Trường 'Area' phải là một số."
    if not math.isfinite(area) or area <= 0:
        return None, None, "Diện tích (Area) phải là số dương."
    base_values["Area"] = area

    for field in OPTIONAL_NUMERIC_FIELDS:
        raw = payload.get(field)
        if raw is None or raw == "":
            base_values[field] = None
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, None, f"Trường '{field}' phải là một số."
        if not math.isfinite(value):
            return None, None, f"Trường '{field}' phải là số hữu hạn."
        if value < 0:
            return None, None, f"Trường '{field}' không được âm."
        base_values[field] = value

    categorical_values = {}
    for field in CATEGORICAL_FEATURES:
        raw = payload.get(field)
        if raw is None or raw == "":
            categorical_values[field] = "Không rõ" if "Không rõ" in CATEGORICAL_OPTIONS.get(field, []) else None
            continue
        options = CATEGORICAL_OPTIONS.get(field)
        if options and raw not in options:
            return None, None, f"Giá trị '{raw}' không hợp lệ cho trường '{field}'."
        categorical_values[field] = raw

    model_id = payload.get("model") or BEST_MODEL
    if model_id not in loaded_models:
        return None, None, f"Mô hình không hợp lệ. Các mô hình hỗ trợ: {', '.join(loaded_models)}"

    derived_values = compute_derived_features(base_values)

    row = {}
    row.update(base_values)
    row.update(derived_values)
    row.update(categorical_values)

    return row, model_id, None


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
        application="house_price",
        models=list(loaded_models),
        feature_columns=FEATURE_COLUMNS,
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
            "label": cfg["label"],
            "is_best": model_id == BEST_MODEL,
            "metrics": test_metrics.get(model_id, {}),
        }
        for model_id, cfg in MODEL_FILES.items()
    ]
    return jsonify(application="house_price", best_model=BEST_MODEL, models=items)


@app.post("/house-price/v1/predict")
def predict():
    payload = request.get_json(silent=True)
    row, model_id, error = validate_payload(payload)
    if error:
        return jsonify(error=error), 400

    raw_input = pd.DataFrame([row])
    for col in FEATURE_COLUMNS:
        if col not in raw_input.columns:
            raw_input[col] = np.nan
    raw_input = raw_input[FEATURE_COLUMNS]

    processed_input = preprocessor.transform(raw_input)
    selected_model = loaded_models[model_id]

    predicted_price = float(selected_model.predict(processed_input)[0])
    price_low = max(0.0, round(predicted_price - RESIDUAL_STD, 2))
    price_high = max(0.0, round(predicted_price + RESIDUAL_STD, 2))
    predicted_price = round(predicted_price, 2)

    interpretation = (
        f"Dựa trên các đặc điểm bất động sản đã nhập, mô hình {MODEL_FILES[model_id]['label']} ước tính "
        f"giá trị khoảng {predicted_price} {TARGET_UNIT}, dao động trong khoảng {price_low} - {price_high} {TARGET_UNIT}."
    )

    return jsonify(
        application="house_price",
        model=model_id,
        model_label=MODEL_FILES[model_id]["label"],
        predicted_price=predicted_price,
        unit=TARGET_UNIT,
        price_low=price_low,
        price_high=price_high,
        interpretation=interpretation,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
