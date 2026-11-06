"""Phục vụ dự đoán: nạp lười XGBoost + isotonic theo bài toán, SHAP qua pred_contribs, chạy trên CPU."""
import json
import math
from functools import lru_cache

from .tasks import MODEL_DIR, TASKS, feature_groups, hospital_features

SEX = {"male": 1, "female": 0}
# Isotonic trên bộ 100k cho đúng 0 và 1 vì nhãn gần như tất định theo ngưỡng; xác suất phục vụ không được khẳng định
# chắc chắn có hoặc không mắc bệnh nên kẹp lại. Nhãn và chỉ số đánh giá không đổi.
PROB_RANGE = (0.005, 0.995)
SMOKING = ("never", "former", "current", "unknown")
LABEL_VI = {
    "age": "Tuổi", "sex": "Giới tính", "bmi": "Chỉ số khối cơ thể (BMI)", "hba1c": "HbA1c", "glucose": "Đường huyết",
    "hypertension": "Tăng huyết áp", "heart_disease": "Bệnh tim", "smoking": "Hút thuốc", "high_bp": "Huyết áp cao",
    "high_chol": "Cholesterol cao", "chol_check": "Đã kiểm tra cholesterol trong 5 năm", "smoker": "Từng hút ≥ 100 điếu",
    "stroke": "Từng đột quỵ", "heart_disease_or_attack": "Bệnh mạch vành hoặc nhồi máu cơ tim",
    "phys_activity": "Vận động thể lực", "fruits": "Ăn trái cây mỗi ngày", "veggies": "Ăn rau mỗi ngày",
    "hvy_alcohol_consump": "Uống nhiều rượu bia", "any_healthcare": "Có bảo hiểm y tế",
    "no_docbc_cost": "Không đi khám vì chi phí", "gen_hlth": "Sức khỏe tự đánh giá", "ment_hlth": "Số ngày tinh thần kém",
    "phys_hlth": "Số ngày thể chất kém", "diff_walk": "Khó khăn khi đi lại", "education": "Trình độ học vấn",
    "income": "Mức thu nhập",
}


def _na(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


@lru_cache(maxsize=3)
def _bundle(task):
    """Nạp lười một lần; thiếu tệp thì báo lỗi rõ ràng, không dùng mô hình thay thế."""
    import joblib
    import xgboost as xgb
    meta_path = MODEL_DIR / f"{task}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Chưa có mô hình {task}: chạy scripts/ml_select.py")
    booster = xgb.Booster()
    booster.load_model(MODEL_DIR / f"{task}_xgb.json")
    booster.set_param({"nthread": 4})
    pre = joblib.load(MODEL_DIR / f"{task}_pre.joblib")
    return {"booster": booster, "pre": pre, "iso": joblib.load(MODEL_DIR / f"{task}_calib.joblib"),
            "meta": json.loads(meta_path.read_text(encoding="utf-8")), "groups": feature_groups(pre, task)}


def _age_code(age):
    """Tuổi thật thành mã nhóm 13 mức của BRFSS: 18–24 là 1, 80+ là 13."""
    return None if _na(age) else int(min(13, max(1, (float(age) - 25) // 5 + 2)))


def _check_cat(name, v, allowed):
    if v not in allowed:
        raise ValueError(f"{name} không hợp lệ: {v!r}; cho phép {sorted(allowed)}")
    return v


def _row_full(p):
    hyp = p.get("hypertension") if not _na(p.get("hypertension")) else p.get("high_bp")
    row = {k: p.get(k) for k in ("age", "bmi", "hba1c", "glucose", "heart_disease")}
    sex = p.get("sex")
    return {**row, "hypertension": hyp, "sex": None if _na(sex) else _check_cat("sex", sex, SEX),
            "smoking": _check_cat("smoking", p.get("smoking") or "unknown", SMOKING)}


def _row_screening(p):
    row = {k: p.get(k) for k in TASKS["screening"]["num"]}
    row["age"] = _age_code(p.get("age"))
    if not _na(p.get("sex")):
        row["sex"] = SEX[_check_cat("sex", p["sex"], SEX)]
    if _na(row["high_bp"]):
        row["high_bp"] = p.get("hypertension")
    if _na(row["heart_disease_or_attack"]):
        row["heart_disease_or_attack"] = p.get("heart_disease")
    smoke = p.get("smoking")
    if _na(row["smoker"]) and smoke not in (None, "unknown"):
        row["smoker"] = int(_check_cat("smoking", smoke, SMOKING) != "never")
    return row


def _pick_mode(profile, mode):
    if mode == "auto":
        return "full" if not _na(profile.get("hba1c")) and not _na(profile.get("glucose")) else "screening"
    if mode not in ("full", "screening"):
        raise ValueError(f"mode không hợp lệ: {mode!r}")
    if mode == "full" and (_na(profile.get("hba1c")) or _na(profile.get("glucose"))):
        raise ValueError("Chế độ full cần cả hba1c và glucose")
    return mode


def _transform(task, row):
    import pandas as pd
    t, b = TASKS[task], _bundle(task)
    cols = t["num"] + t["cat"]
    d = pd.DataFrame([{c: row.get(c) for c in cols}])
    d[t["num"]] = d[t["num"]].apply(pd.to_numeric, errors="raise").astype(float)
    for c in t["cat"]:
        d[c] = d[c].astype(object)
    return b["pre"].transform(d)


def _score(task, X):
    import xgboost as xgb
    b = _bundle(task)
    dm = xgb.DMatrix(X)
    raw = float(b["booster"].predict(dm)[0])
    prob = float(b["iso"].predict([raw])[0])
    return raw, min(PROB_RANGE[1], max(PROB_RANGE[0], prob)), dm


def predict(profile, mode="auto"):
    """Xác suất đã hiệu chỉnh và đóng góp SHAP (thang log-odds, gộp cột one-hot) giảm dần theo trị tuyệt đối."""
    task = _pick_mode(profile, mode)
    b = _bundle(task)
    row = (_row_full if task == "full" else _row_screening)(profile)
    raw, prob, dm = _score(task, _transform(task, row))
    contribs = b["booster"].predict(dm, pred_contribs=True)[0]
    by = {}
    for g, c in zip(b["groups"], contribs[:-1]):
        by[g] = by.get(g, 0.0) + float(c)
    shap = [{"feature": f, "label_vi": LABEL_VI[f], "value": row.get(f), "contribution": c}
            for f, c in sorted(by.items(), key=lambda kv: -abs(kv[1]))]
    th = b["meta"]["threshold"]
    return {"mode": task, "prob": prob, "label": int(prob >= th), "threshold": th, "model": b["meta"]["model"],
            "calibrated": True, "shap": shap, "raw_prob": raw, "base_value": float(contribs[-1]),
            "imputed": [c for c in TASKS[task]["num"] + TASKS[task]["cat"] if _na(row.get(c))]}


def what_if(profile, changes):
    """So sánh hai xác suất cùng một chế độ, chọn theo hồ sơ sau thay đổi để đổi HbA1c không làm đổi mô hình giữa chừng."""
    after_profile = {**profile, **changes}
    mode = _pick_mode(after_profile, "auto")
    before = predict(profile, mode)["prob"]
    after = predict(after_profile, mode)["prob"]
    return {"before": before, "after": after, "delta": after - before}


def readmission(record):
    """Nguy cơ tái nhập viện dưới 30 ngày; bản ghi dùng tên cột của bảng hospital, cột thiếu được điền như huấn luyện."""
    import pandas as pd
    t, b = TASKS["readmission"], _bundle("readmission")
    d = hospital_features(pd.DataFrame([record]))
    d[t["num"]] = d[t["num"]].apply(pd.to_numeric, errors="raise").astype(float)
    _, prob, _ = _score("readmission", b["pre"].transform(d[t["num"] + t["cat"]]))
    th = b["meta"]["threshold"]
    return {"prob": prob, "label": int(prob >= th), "threshold": th}


def model_card():
    """Thẻ mô hình do scripts/ml_select.py sinh từ số đo thật."""
    path = MODEL_DIR / "model_card.json"
    if not path.exists():
        raise FileNotFoundError("Chưa có model_card.json: chạy scripts/ml_select.py")
    return json.loads(path.read_text(encoding="utf-8"))
