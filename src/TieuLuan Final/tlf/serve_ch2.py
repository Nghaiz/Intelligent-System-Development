"""Chương 2: suy luận cho Web App. Mô hình nạp lười một lần từ models/ch2/, chạy CPU, không cần TensorFlow."""
import math
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .ch2_data import CDC_BINARY, CDC_SCALE, HOUSE_CAT, HOUSE_NUM, HOUSE_TE, VSFC_LABELS, parse_address
from .common import MODELS, load_json, vn

ACADEMIC = "Kết quả mang tính học thuật, không thay thế chẩn đoán của bác sĩ."
HOUSE_NOTE = "Ước lượng theo dữ liệu tin rao 2024, giá bị chặn trong khoảng 1 đến 11,5 tỷ đồng."
NO_YES = [0, 1]
DIRECTIONS = ["", "Bắc", "Nam", "Đông", "Tây", "Đông - Bắc", "Đông - Nam", "Tây - Bắc", "Tây - Nam"]


def _num(name, label, default, lo, hi):
    return {"name": name, "type": "number", "label": label, "default": default, "min": lo, "max": hi}


def _sel(name, label, options, default):
    return {"name": name, "type": "select", "label": label, "default": default, "options": options}


CDC_INPUTS = [
    _num("BMI", "Chỉ số khối cơ thể", 28, 12, 98),
    _num("GenHlth", "Tự đánh giá sức khỏe (1 tốt nhất, 5 kém nhất)", 3, 1, 5),
    _num("MentHlth", "Số ngày tinh thần kém trong 30 ngày", 0, 0, 30),
    _num("PhysHlth", "Số ngày thể chất kém trong 30 ngày", 0, 0, 30),
    _num("Age", "Nhóm tuổi (1: 18-24 ... 13: từ 80)", 9, 1, 13),
    _num("Education", "Học vấn (1 thấp nhất, 6 cao nhất)", 5, 1, 6),
    _num("Income", "Thu nhập (1 thấp nhất, 8 cao nhất)", 6, 1, 8),
] + [_sel(n, lb, NO_YES, d) for n, lb, d in (
    ("HighBP", "Huyết áp cao", 1), ("HighChol", "Cholesterol cao", 1), ("CholCheck", "Đo cholesterol trong 5 năm", 1),
    ("Smoker", "Từng hút trên 100 điếu", 0), ("Stroke", "Từng đột quỵ", 0),
    ("HeartDiseaseorAttack", "Bệnh tim mạch vành", 0), ("PhysActivity", "Vận động trong 30 ngày", 1),
    ("Fruits", "Ăn trái cây mỗi ngày", 1), ("Veggies", "Ăn rau mỗi ngày", 1),
    ("HvyAlcoholConsump", "Uống rượu nhiều", 0), ("AnyHealthcare", "Có bảo hiểm y tế", 1),
    ("NoDocbcCost", "Bỏ khám vì chi phí", 0), ("DiffWalk", "Khó đi bộ, leo cầu thang", 0),
    ("Sex", "Giới tính (0 nữ, 1 nam)", 0))]

HOUSE_INPUTS = [
    {"name": "Address", "type": "text", "label": "Địa chỉ: đường, phường, quận, tỉnh",
     "default": "Đường Quang Trung, Phường 8, Gò Vấp, Hồ Chí Minh"},
    _num("Area", "Diện tích (m²)", 60, 10, 600), _num("Frontage", "Mặt tiền (m), để trống nếu không rõ", 5, 1, 80),
    _num("Access Road", "Đường vào (m), để trống nếu không rõ", 6, 1, 85), _num("Floors", "Số tầng", 3, 1, 10),
    _num("Bedrooms", "Phòng ngủ", 3, 1, 9), _num("Bathrooms", "Phòng tắm", 3, 1, 9),
    _sel("House direction", "Hướng nhà", DIRECTIONS, ""), _sel("Balcony direction", "Hướng ban công", DIRECTIONS, ""),
    _sel("Legal status", "Pháp lý", ["", "Have certificate", "Sale contract"], "Have certificate"),
    _sel("Furniture state", "Nội thất", ["", "Basic", "Full"], ""),
]

SPECS = [
    {"id": "ch2_diabetes", "title": "Sàng lọc nguy cơ tiểu đường", "dataset": "CDC Diabetes",
     "inputs": CDC_INPUTS, "output": "probs"},
    {"id": "ch2_housing", "title": "Ước lượng giá nhà Việt Nam", "dataset": "Vietnam Housing",
     "inputs": HOUSE_INPUTS, "output": "label"},
    {"id": "ch2_sentiment", "title": "Cảm xúc phản hồi sinh viên", "dataset": "UIT-VSFC", "output": "probs",
     "inputs": [{"name": "text", "type": "text", "label": "Câu phản hồi tiếng Việt",
                 "default": "thầy giảng dễ hiểu và rất nhiệt tình"}]},
]


@lru_cache(maxsize=None)
def _model(ds):
    return joblib.load(MODELS / "ch2" / f"{ds}_best.joblib")


@lru_cache(maxsize=None)
def _best(ds):
    return load_json(f"ch2_res_{ds}_classic")["best"]


def _inputs(spec_id):
    return next(s for s in SPECS if s["id"] == spec_id)["inputs"]


def _number(payload, spec):
    """Giá trị số trong miền của biểu mẫu; ô trống hoặc vượt miền báo lỗi rõ ràng, không tự sửa."""
    v = payload.get(spec["name"], spec["default"])
    v = float(v)
    if not math.isfinite(v) or not spec["min"] <= v <= spec["max"]:
        raise ValueError(f"{spec['label']} phải trong [{spec['min']}, {spec['max']}]")
    return v


def predict_diabetes(payload):
    row = {s["name"]: _number(payload, s) if s["type"] == "number" else float(payload.get(s["name"], s["default"]))
           for s in CDC_INPUTS}
    p = float(_model("cdc").predict_proba(pd.DataFrame([row])[CDC_SCALE + CDC_BINARY])[0, 1])
    best = _best("cdc")
    label = "Nguy cơ cao, nên đi xét nghiệm" if p >= best["threshold"] else "Nguy cơ thấp"
    return {"label": label, "probs": {"Không tiểu đường": 1 - p, "Tiểu đường": p}, "threshold": best["threshold"],
            "model": best["name"], "note": ACADEMIC}


def predict_housing(payload):
    """Ô số để trống thành NaN và đi qua nhánh trung vị + cờ thiếu, đúng như lúc huấn luyện."""
    row = {}
    for s in HOUSE_INPUTS:
        v = payload.get(s["name"], s["default"])
        if s["type"] == "number":
            row[s["name"]] = np.nan if v in ("", None) else _number(payload, s)
        else:
            row[s["name"]] = v if v else np.nan
    row.update(parse_address(row.pop("Address")))
    value = float(_model("housing").predict(pd.DataFrame([row])[HOUSE_NUM + HOUSE_CAT + HOUSE_TE])[0])
    return {"label": f"{vn(value, 2)} tỷ đồng", "value": value, "model": _best("housing")["name"], "note": HOUSE_NOTE}


def predict_sentiment(payload):
    """XGBoost học trên ma trận đặc nên phải đổi sang đặc:
    với ma trận thưa, số 0 bị hiểu là giá trị thiếu và dự đoán sai."""
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("Chưa nhập câu phản hồi")
    pipe = _model("vsfc")
    x = pipe["pre"].transform([text]).toarray().astype(np.float32)
    p = pipe["model"].predict_proba(x)[0]
    label = VSFC_LABELS[int(np.argmax(p))]
    return {"label": label, "probs": dict(zip(VSFC_LABELS, map(float, p))),
            "model": _best("vsfc")["name"]}


HANDLERS = {"ch2_diabetes": predict_diabetes, "ch2_housing": predict_housing, "ch2_sentiment": predict_sentiment}


def predict(id, payload):
    """Điểm vào chung của Web App; id lạ báo lỗi."""
    if id not in HANDLERS:
        raise KeyError(f"Không có dự đoán '{id}'")
    return HANDLERS[id](payload)
