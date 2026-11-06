"""Khai báo ba bài toán (dữ liệu, đặc trưng, nhãn) và bộ tiền xử lý dùng chung cho huấn luyện lẫn phục vụ."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..common import MODELS
from ..db import connect

MODEL_DIR = MODELS / "ml"
PRED_DIR = MODEL_DIR / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)

CDC_NUM = ["high_bp", "high_chol", "chol_check", "bmi", "smoker", "stroke", "heart_disease_or_attack", "phys_activity",
           "fruits", "veggies", "hvy_alcohol_consump", "any_healthcare", "no_docbc_cost", "gen_hlth", "ment_hlth",
           "phys_hlth", "diff_walk", "sex", "age", "education", "income"]
HOSP_NUM = ["age_mid", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses", "n_meds_active",
            "n_meds_changed"]
HOSP_MEDS = ["metformin", "insulin", "glipizide", "glyburide", "glimepiride", "pioglitazone", "rosiglitazone",
             "repaglinide"]
HOSP_CAT = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id", "specialty",
            "diag_1_group", "diag_2_group", "diag_3_group", "max_glu_serum", "a1cresult", "change", "diabetesmed",
            *HOSP_MEDS]
ALL_MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
            "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
            "tolazamide", "examide", "citoglipton", "insulin", "glyburide_metformin", "glipizide_metformin",
            "glimepiride_pioglitazone", "metformin_rosiglitazone", "metformin_pioglitazone"]
TOP_SPECIALTY = ["InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology", "Surgery-General",
                 "Nephrology", "Orthopedics", "Orthopedics-Reconstructive", "Radiologist", "Pulmonology", "Psychiatry"]

TASKS = {
    "full": {"table": "patients_lab", "target": "diabetes", "cat": ["sex", "smoking"],
             "num": ["age", "bmi", "hba1c", "glucose", "hypertension", "heart_disease"]},
    "screening": {"table": "brfss", "target": "diabetes", "num": CDC_NUM, "cat": []},
    "readmission": {"table": "hospital", "target": "readmit_30", "num": HOSP_NUM, "cat": HOSP_CAT},
}


def icd9_group(code):
    """Nhóm mã ICD-9 theo chương lớn (Strack và cộng sự, 2014)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    s = str(code)
    if s[0] in "VE":
        return "other"
    c = int(float(s))
    if c == 250:
        return "diabetes"
    for name, lo, hi, extra in (("circulatory", 390, 459, 785), ("respiratory", 460, 519, 786),
                                ("digestive", 520, 579, 787), ("genitourinary", 580, 629, 788),
                                ("musculoskeletal", 710, 739, None), ("injury", 800, 999, None),
                                ("neoplasm", 140, 239, None)):
        if lo <= c <= hi or c == extra:
            return name
    return "other"


def _age_mid(v):
    """Khoảng tuổi dạng [70-80) thành 75; số giữ nguyên."""
    if isinstance(v, str) and v.startswith("["):
        lo, hi = v[1:-1].split("-")
        return (int(lo) + int(hi)) / 2
    return v


def hospital_features(df):
    """Dòng hospital (hoặc bản ghi rút gọn từ API) thành khung đặc trưng; cột thiếu thành NaN để bộ điền xử lý."""
    d = df.reindex(columns=sorted({*df.columns, *HOSP_NUM, *HOSP_CAT, *ALL_MEDS, "age", "medical_specialty",
                                   "diag_1", "diag_2", "diag_3"})).copy()
    meds = d[ALL_MEDS].fillna("No")
    d["age_mid"] = d["age"].map(_age_mid).astype(float)
    d["n_meds_active"] = (meds != "No").sum(axis=1)
    d["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1)
    d["specialty"] = d["medical_specialty"].where(d["medical_specialty"].isin(TOP_SPECIALTY), "Other/Unknown")
    for i in (1, 2, 3):
        d[f"diag_{i}_group"] = d[f"diag_{i}"].map(icd9_group).astype(object)
    for c in ("max_glu_serum", "a1cresult"):
        d[c] = d[c].astype(object).where(d[c].notna(), "not_measured")
    for c in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        d[c] = d[c].map(lambda v: None if pd.isna(v) else str(int(v))).astype(object)
    return d


def frame(task):
    """Bảng của bài toán sau khi tính đặc trưng; có cột split và nhãn."""
    t = TASKS[task]
    with connect() as con:
        d = pd.read_sql(f"select * from {t['table']}", con)
    return hospital_features(d) if task == "readmission" else d


def make_pre(task):
    t = TASKS[task]
    steps = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), t["num"])]
    if t["cat"]:
        steps.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                           OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                                         dtype=np.float32)), t["cat"]))
    return ColumnTransformer(steps, verbose_feature_names_out=False)


def feature_groups(pre, task):
    """Cột sau biến đổi thuộc đặc trưng gốc nào; dùng để cộng SHAP của các cột one-hot về một đặc trưng."""
    t = TASKS[task]
    groups = list(t["num"])
    if t["cat"]:
        ohe = pre.named_transformers_["cat"].named_steps["onehotencoder"]
        for c, cats in zip(t["cat"], ohe.categories_):
            groups += [c] * len(cats)
    return groups


def splits(task):
    """Trả (pre đã fit trên train, khung, mặt nạ, X/y theo train/val/test)."""
    t, d = TASKS[task], frame(task)
    cols = t["num"] + t["cat"]
    masks = {s: (d["split"] == s).values for s in ("train", "val", "test")}
    y = d[t["target"]].values.astype(int)
    pre = make_pre(task).fit(d.loc[masks["train"], cols])
    data = {s: (np.asarray(pre.transform(d.loc[m, cols]), dtype=np.float32), y[m]) for s, m in masks.items()}
    return pre, d, masks, data
