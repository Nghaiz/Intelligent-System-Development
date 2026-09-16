"""REST API — Hệ thống 2: Định giá bất động sản (150,000 giao dịch).

Chạy độc lập trên cổng 5002.

Cũng như hệ thống 1, dịch vụ này suy luận hoàn toàn bằng NumPy. Khác biệt nằm ở
tầng ngõ ra: mạng hồi quy không có Sigmoid, và kết quả phải đi qua hai bước
nghịch đảo mới ra được con số Dollar mà người dùng đọc được:

    ŷ_scaled  →  ŷ_log = ŷ_scaled · σ_y + μ_y  →  giá = exp(ŷ_log) − 1

Phần khó nhất là mã hoá vị trí địa lý: API phải tra lại đúng bảng Smooth
Bayesian Target Encoding đã dựng trong notebook 04 (lưu ở `geo_encoding.json`),
và rơi về giá trị trung bình toàn quốc khi gặp khu vực chưa từng thấy.

Chạy:  python house_price_large/api/rest_api.py
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

from knowledge_graph import knowledge  # noqa: E402

PORT = 5002
PREFIX = "/house-price/v1"

# ---------------------------------------------------------------------------
# Nạp artifact suy luận
# ---------------------------------------------------------------------------
META = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
GEO = json.loads((MODEL_DIR / "geo_encoding.json").read_text(encoding="utf-8"))

_Z = np.load(MODEL_DIR / "dl_scratch_weights.npz")
_N_LAYERS = sum(1 for k in _Z.files if k.startswith("W"))
WS = [_Z[f"W{i}"] for i in range(_N_LAYERS)]
BS = [_Z[f"b{i}"] for i in range(_N_LAYERS)]

PRE = META["preprocess"]
MU = np.array(PRE["mu"], dtype=np.float64)
SD = np.array(PRE["sd"], dtype=np.float64)
Y_MU = float(PRE["y_mu"])
Y_SD = float(PRE["y_sd"])
ACRE_MEDIAN = float(PRE["acre_lot_median"])
FEATURE_NAMES = META["feature_names"]

GLOBAL_MEAN = float(GEO["global_mean"])
GLOBAL_SIZE = float(GEO["global_house_size"])
ZIP_SIZE_MAP = GEO["zip_size_map"]
TE_TABLES = GEO["tables"]

REQUIRED = ["bed", "bath", "house_size"]
OPTIONAL = ["acre_lot", "city", "state", "zip_code"]

# Ngưỡng phân khúc thị trường — chỉ dùng để tra cứu tư vấn
PRICE_TIERS = [(800_000, "house_luxury"), (300_000, "house_mid"), (0, "house_standard")]

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# Tiền xử lý — tái hiện chính xác pipeline của notebook 04
# ---------------------------------------------------------------------------
def lookup_te(level: str, key: str) -> tuple[float, bool]:
    """Tra bảng Target Encoding; trả về (giá_trị, có_tìm_thấy_không)."""
    table = TE_TABLES.get(level, {})
    if key in table:
        return float(table[key]), True
    return GLOBAL_MEAN, False       # khu vực mới → giá trung bình toàn quốc


def build_feature_vector(v: dict) -> tuple[np.ndarray, dict]:
    """Dựng vector 13 chiều đúng thứ tự đã huấn luyện."""
    size = float(v["house_size"])
    bed = float(v["bed"])
    bath = float(v["bath"])
    acre = float(v["acre_lot"]) if v.get("acre_lot") is not None else ACRE_MEDIAN
    rooms = bed + bath

    zip_code = str(v.get("zip_code") or "").strip()
    zip5 = zip_code.zfill(5) if zip_code else ""
    zip3 = zip5[:3] if zip5 else ""
    state = str(v.get("state") or "").strip()
    city = str(v.get("city") or "").strip()

    te_state, hit_state = lookup_te("state", state)
    te_zip3, hit_zip3 = lookup_te("zip3", zip3)
    te_zip5, hit_zip5 = lookup_te("zip_code", zip5)
    te_city, hit_city = lookup_te("city", city)

    zip_mean_size = float(ZIP_SIZE_MAP.get(zip5, GLOBAL_SIZE))

    raw = np.array([
        math.log(size),                       # log_size
        math.log1p(acre),                     # log_lot
        bed, bath, rooms,                     # bed, bath, total_rooms
        size / max(rooms, 1.0),               # sqft_per_room
        bath / max(bed, 1.0),                 # bath_bed_ratio
        bed * bath,                           # bed_bath_prod
        size / max(zip_mean_size, 1.0),       # relative_sqft
        te_state, te_zip3, te_zip5, te_city,  # 4 cấp mã hoá địa lý
    ], dtype=np.float64)

    geo_info = {
        "state_matched": hit_state, "zip3_matched": hit_zip3,
        "zip_code_matched": hit_zip5, "city_matched": hit_city,
        "acre_lot_used": acre, "zip_mean_house_size": zip_mean_size,
    }
    return ((raw - MU) / SD).reshape(1, -1), geo_info


def forward(X: np.ndarray) -> np.ndarray:
    """Lan truyền tiến hồi quy — ReLU ở tầng ẩn, TUYẾN TÍNH ở tầng ra."""
    A = X
    for i, (W, b) in enumerate(zip(WS, BS)):
        Z = A @ W + b
        A = np.maximum(0.0, Z) if i < len(WS) - 1 else Z
    return A


def to_usd(pred_scaled: float) -> float:
    """Nghịch đảo hai bước: chuẩn hoá → log → Dollar thực."""
    return float(np.expm1(pred_scaled * Y_SD + Y_MU))


# ---------------------------------------------------------------------------
# Xác thực dữ liệu đầu vào
# ---------------------------------------------------------------------------
def validate(payload):
    if not isinstance(payload, dict):
        return None, "Nội dung yêu cầu phải là một đối tượng JSON."

    values = {}
    for field in REQUIRED:
        if field not in payload or payload[field] in (None, ""):
            return None, f"Thiếu trường bắt buộc: '{field}'."
        try:
            x = float(payload[field])
        except (TypeError, ValueError):
            return None, f"Trường '{field}' phải là một số."
        if not math.isfinite(x):
            return None, f"Trường '{field}' phải là một số hữu hạn."
        values[field] = x

    if not (300 <= values["house_size"] <= 10_000):
        return None, "Diện tích sàn phải nằm trong khoảng 300 – 10,000 sqft."
    if not (1 <= values["bed"] <= 12):
        return None, "Số phòng ngủ phải nằm trong khoảng 1 – 12."
    if not (1 <= values["bath"] <= 12):
        return None, "Số phòng tắm phải nằm trong khoảng 1 – 12."

    acre = payload.get("acre_lot")
    if acre in (None, ""):
        values["acre_lot"] = None
    else:
        try:
            a = float(acre)
        except (TypeError, ValueError):
            return None, "Trường 'acre_lot' phải là một số."
        if a < 0 or not math.isfinite(a):
            return None, "Diện tích lô đất phải là số không âm."
        values["acre_lot"] = a

    for field in ("city", "state", "zip_code"):
        values[field] = str(payload.get(field, "") or "").strip()

    return values, None


# ---------------------------------------------------------------------------
# Các điểm cuối
# ---------------------------------------------------------------------------
@app.get(f"{PREFIX}/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Định giá bất động sản (Hệ thống 2)",
        "model": META["dl_model"]["name"],
        "architecture": META["dl_model"]["layers"],
        "n_params": META["dl_model"]["n_params"],
        "inference": "Forward pass thuần NumPy, tầng ra tuyến tính",
        "knowledge_graph": "connected" if knowledge.is_available() else "unavailable",
    })


@app.get(f"{PREFIX}/schema")
def schema():
    return jsonify({
        "required": REQUIRED, "optional": OPTIONAL,
        "feature_names": FEATURE_NAMES, "n_features": len(FEATURE_NAMES),
        "geo_levels": {k: len(v) for k, v in TE_TABLES.items()},
        "acre_lot_median": ACRE_MEDIAN,
    })


@app.get(f"{PREFIX}/metrics")
def metrics():
    return jsonify({
        "dataset": META["dataset"], "dl_model": META["dl_model"],
        "scores": META["scores"], "architecture_study": META["architecture_study"],
        "market_segments": META["market_segments"], "champion": META["champion"],
    })


@app.post(f"{PREFIX}/predict")
def predict():
    values, err = validate(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400

    x, geo_info = build_feature_vector(values)
    pred_scaled = float(forward(x)[0, 0])
    price = to_usd(pred_scaled)

    case = knowledge.resolve_case(PRICE_TIERS, price)
    groups, tier_name, k_err = knowledge.fetch(case)

    return jsonify({
        "price_usd": round(price, 2),
        "price_display": f"${price:,.0f}",
        "price_log": round(pred_scaled * Y_SD + Y_MU, 6),
        "price_per_sqft": round(price / max(values["house_size"], 1.0), 2),
        "segment_case": case,
        "segment": tier_name,
        "model": {"name": META["dl_model"]["name"],
                  "architecture": META["dl_model"]["layers"],
                  "n_params": META["dl_model"]["n_params"],
                  "inference": "pure NumPy forward pass"},
        "input": values,
        "geo_encoding": geo_info,
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
    print("Hệ thống 2 — Định giá bất động sản")
    print(f"  Mô hình  : {META['dl_model']['name']} ({META['dl_model']['layers']})")
    print(f"  Tham số  : {META['dl_model']['n_params']:,} · suy luận thuần NumPy")
    print(f"  Địa lý   : " + " · ".join(f"{k}={len(v):,}" for k, v in TE_TABLES.items()))
    print(f"  Đồ thị   : {'đã kết nối' if knowledge.is_available() else 'không khả dụng'}")
    print(f"  Web      : http://127.0.0.1:{PORT}/")
    print(f"  Mobile   : http://127.0.0.1:{PORT}/mobile")
    app.run(host="127.0.0.1", port=PORT, debug=False)
