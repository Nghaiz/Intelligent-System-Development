"""REST API cho ứng dụng dự đoán nguy cơ tiểu đường (Diabetes).

Chạy độc lập trên cổng 5001. Toàn bộ artifact suy luận (mô hình + bộ tiền xử
lý) được nạp bằng joblib từ thư mục `../model` khi khởi động — ứng dụng
KHÔNG huấn luyện lại bất cứ thứ gì trong quá trình phục vụ.
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import __main__
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv luôn có trong requirements
    load_dotenv = None

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - cho phép chạy khi chưa cài neo4j
    GraphDatabase = None


# ---------------------------------------------------------------------------
# Đường dẫn & cấu hình
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "model"
WEB_DIR = APP_DIR / "web"
MOBILE_DIR = APP_DIR / "mobile"
ROOT_DIR = APP_DIR.parent

if load_dotenv is not None:
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Đồ thị tri thức phân tầng nguy cơ theo `case`; xác suất do MÔ HÌNH sinh ra
# quyết định chọn tầng nào, còn đồ thị chỉ cung cấp nội dung tư vấn cho tầng ấy.
RISK_TIERS = [
    (0.25, "med_high"),
    (0.15, "med_moderate"),
    (0.00, "med_low"),
]

KNOWLEDGE_GROUPS = {
    "Action": "Hướng dẫn cần làm",
    "Device": "Thiết bị theo dõi",
    "Nutrition": "Dinh dưỡng hỗ trợ",
    "Service": "Dịch vụ đi kèm",
}


def resolve_risk_case(probability_positive):
    """Chọn tầng nguy cơ theo xác suất dương tính mà mô hình trả về."""
    for threshold, case in RISK_TIERS:
        if probability_positive >= threshold:
            return case
    return "med_low"

MODEL_FILES = {
    "logistic_regression": {"label": "Logistic Regression", "filename": "logistic_regression.joblib"},
    "knn": {"label": "K-Nearest Neighbors", "filename": "knn.joblib"},
    "decision_tree": {"label": "Decision Tree", "filename": "decision_tree.joblib"},
    "random_forest": {"label": "Random Forest", "filename": "random_forest.joblib"},
    "svm_rbf": {"label": "SVM (RBF)", "filename": "svm_rbf.joblib"},
}


# ---------------------------------------------------------------------------
# Nạp metadata (SSOT cho danh sách đặc trưng, nhãn lớp, chỉ số mô hình...)
# ---------------------------------------------------------------------------
metadata_path = MODEL_DIR / "metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(f"Không tìm thấy file metadata: {metadata_path}")

with metadata_path.open("r", encoding="utf-8") as fh:
    METADATA = json.load(fh)

FEATURE_COLUMNS = METADATA["feature_columns"]
INVALID_ZERO_COLUMNS = METADATA.get("invalid_zero_columns", ["Glucose", "BMI"])
CLASS_LABELS = METADATA.get("class_labels", {"0": "Âm tính", "1": "Dương tính"})
BEST_MODEL = METADATA.get("best_model", "random_forest")


# ---------------------------------------------------------------------------
# Bộ tiền xử lý được lưu từ notebook dùng FunctionTransformer trỏ tới hàm
# `convert_invalid_zero_to_nan` thuộc __main__. Phải đăng ký lại hàm này
# trước khi joblib.load(preprocessor) để unpickle thành công.
# ---------------------------------------------------------------------------
def convert_invalid_zero_to_nan(data):
    cleaned = data.copy()
    cleaned[INVALID_ZERO_COLUMNS] = cleaned[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    return cleaned


setattr(__main__, "convert_invalid_zero_to_nan", convert_invalid_zero_to_nan)


def load_artifact(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {path}")
    return joblib.load(path)


preprocessor = load_artifact("preprocessor.joblib")
loaded_models = {model_id: load_artifact(cfg["filename"]) for model_id, cfg in MODEL_FILES.items()}

neo4j_driver = None
if GraphDatabase is not None and NEO4J_PASSWORD:
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    except Exception:  # pragma: no cover - lỗi kết nối không được chặn khởi động
        neo4j_driver = None

app = Flask(__name__)
if neo4j_driver is None:
    app.logger.warning("Đồ thị tri thức Neo4j đang tắt (thiếu package/biến môi trường).")
else:
    app.logger.info("Đồ thị tri thức Neo4j đã cấu hình tới %s", NEO4J_URI)


# ---------------------------------------------------------------------------
# Neo4j — chỉ tra cứu bổ sung khi kết quả dự đoán là nguy cơ cao (class 1).
# Không bao giờ ảnh hưởng tới kết quả dự đoán hay xác suất.
# ---------------------------------------------------------------------------
def get_diabetes_knowledge(probability_positive):
    """Tra cứu tư vấn từ đồ thị tri thức Neo4j theo tầng nguy cơ.

    Neo4j chỉ BỔ SUNG nội dung tư vấn sau khi mô hình đã dự đoán xong. Nó không
    tham gia vào việc tính nhãn hay xác suất, và mọi lỗi ở đây đều được nuốt lại
    thành một thông báo — dự đoán vẫn phải trả về bình thường khi đồ thị hỏng.
    """
    if neo4j_driver is None:
        return [], None, "Chưa cấu hình kết nối Neo4j (thiếu NEO4J_PASSWORD hoặc package neo4j)."

    case = resolve_risk_case(probability_positive)
    query = """
    MATCH (tier:RiskTier {case: $case})
    OPTIONAL MATCH (tier)-[:ADVISES]->(action:Action)
    OPTIONAL MATCH (tier)-[:RECOMMENDS_PACKAGE]->(:CarePackage)-[:INCLUDES_PRODUCT]->(item)
    RETURN tier.name AS tier_name,
           collect(DISTINCT {kind: 'Action', name: action.name,
                             detail: action.`Nhóm`, price: null, url: null}) AS actions,
           collect(DISTINCT {kind: item.kind, name: item.name,
                             detail: item.`Công dụng`, price: item.`Khoảng giá`,
                             url: item.`Tra cứu giá`}) AS products
    """
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            record = session.run(query, case=case).single()
        if record is None:
            return [], case, "Đồ thị tri thức không có dữ liệu cho tầng nguy cơ này."

        grouped = defaultdict(list)
        for entry in list(record["actions"]) + list(record["products"]):
            if not entry.get("name"):
                continue
            group = KNOWLEDGE_GROUPS.get(entry.get("kind"), entry.get("kind") or "Khác")
            grouped[group].append({
                "title": entry["name"],
                "content": entry.get("detail"),
                "price": entry.get("price"),
                "url": entry.get("url"),
            })

        order = ["Hướng dẫn cần làm", "Thiết bị theo dõi", "Dinh dưỡng hỗ trợ", "Dịch vụ đi kèm"]
        groups = [{"title": g, "items": grouped[g]} for g in order if grouped.get(g)]
        groups += [{"title": g, "items": items} for g, items in grouped.items() if g not in order]
        return groups, record["tier_name"], None
    except Exception:
        app.logger.exception("Không thể truy vấn đồ thị tri thức Neo4j")
        return [], case, "Không thể tải kiến thức Neo4j lúc này."


# ---------------------------------------------------------------------------
# Xác thực dữ liệu đầu vào
# ---------------------------------------------------------------------------
def validate_payload(payload):
    if not isinstance(payload, dict):
        return None, None, "Nội dung yêu cầu phải là một đối tượng JSON."

    missing = [field for field in FEATURE_COLUMNS if field not in payload or payload[field] in (None, "")]
    if missing:
        return None, None, f"Thiếu các trường bắt buộc: {', '.join(missing)}"

    values = {}
    for field in FEATURE_COLUMNS:
        raw = payload[field]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, None, f"Trường '{field}' phải là một số."
        if not math.isfinite(value):
            return None, None, f"Trường '{field}' phải là một số hữu hạn."
        values[field] = value

    if values["Glucose"] <= 0:
        return None, None, "Chỉ số Glucose phải lớn hơn 0."
    if values["BMI"] <= 0:
        return None, None, "Chỉ số BMI phải lớn hơn 0."
    if values["Age"] <= 0 or values["Age"] > 120:
        return None, None, "Tuổi phải trong khoảng hợp lệ (1-120)."
    if values["Pregnancies"] < 0:
        return None, None, "Số lần mang thai không được âm."
    if values["DiabetesPedigreeFunction"] < 0:
        return None, None, "DiabetesPedigreeFunction không được âm."

    model_id = payload.get("model") or BEST_MODEL
    if model_id not in loaded_models:
        return None, None, f"Mô hình không hợp lệ. Các mô hình hỗ trợ: {', '.join(loaded_models)}"

    return values, model_id, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
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
        application="diabetes",
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
    return jsonify(application="diabetes", best_model=BEST_MODEL, models=items)


@app.post("/diabetes/v1/predict")
def predict():
    payload = request.get_json(silent=True)
    values, model_id, error = validate_payload(payload)
    if error:
        return jsonify(error=error), 400

    raw_input = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    processed_input = preprocessor.transform(raw_input)
    selected_model = loaded_models[model_id]

    prediction_class = int(selected_model.predict(processed_input)[0])
    confidence = None
    if hasattr(selected_model, "predict_proba"):
        probabilities = selected_model.predict_proba(processed_input)[0]
        confidence = round(float(probabilities[prediction_class]) * 100, 2)

    risk_level = "high" if prediction_class == 1 else "low"
    prediction_label = CLASS_LABELS.get(str(prediction_class), "Không xác định")

    if prediction_class == 1:
        interpretation = (
            "Mô hình phát hiện các chỉ số sức khỏe tương đồng với nhóm có nguy cơ mắc tiểu đường cao. "
            "Nên tham khảo ý kiến bác sĩ để được xét nghiệm và tư vấn chính xác."
        )
    else:
        interpretation = (
            "Các chỉ số sức khỏe hiện tại tương đồng với nhóm có nguy cơ thấp. "
            "Vẫn nên duy trì lối sống lành mạnh và kiểm tra sức khỏe định kỳ."
        )

    # Xác suất dương tính là đầu vào duy nhất mà đồ thị tri thức nhận từ mô hình.
    # Tra cứu ở MỌI tầng nguy cơ, không chỉ tầng cao — nhóm khoẻ mạnh và nhóm tiền
    # đái tháo đường cũng có nội dung tư vấn riêng trong đồ thị.
    probability_positive = 0.0
    if hasattr(selected_model, "predict_proba"):
        probability_positive = float(selected_model.predict_proba(processed_input)[0][1])
    elif prediction_class == 1:
        probability_positive = 1.0

    knowledge, risk_tier, knowledge_error = get_diabetes_knowledge(probability_positive)
    if knowledge_error:
        app.logger.warning("Không lấy được kiến thức Neo4j (%s)", knowledge_error)

    return jsonify(
        application="diabetes",
        model=model_id,
        model_label=MODEL_FILES[model_id]["label"],
        prediction_class=prediction_class,
        prediction=prediction_label,
        risk_level=risk_level,
        confidence=confidence,
        interpretation=interpretation,
        probability_positive=round(probability_positive * 100, 2),
        risk_tier=risk_tier,
        knowledge=knowledge,
        knowledge_error=knowledge_error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
