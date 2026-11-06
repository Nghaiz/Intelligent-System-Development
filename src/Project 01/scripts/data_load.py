"""Nạp, làm sạch, khử trùng 5 bộ dữ liệu vào SQLite và gắn cột split cố định; ghi data_summary.json."""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402

from dnx import db  # noqa: E402
from dnx.common import DATA, ROOT, SEED, save_json  # noqa: E402
from dnx.mem import slot  # noqa: E402

SRC = ROOT.parent
RAW = {
    "lab": SRC / "Assignment 03/diabetes_large/data/diabetes_prediction_dataset.csv",
    "cdc": SRC / "TieuLuan Final/data/ch2/cdc_diabetes.csv",
    "pima": SRC / "Assignment 01/data/raw/diabetes.csv",
    "frankfurt": SRC / "Assignment 01/data/external/diabetes_hospital_frankfurt_2000.csv",
    "uci": DATA / "raw/uci/uci130.zip",
}
SMOKING = {"never": "never", "former": "former", "ever": "former", "not current": "former",
           "current": "current", "No Info": "unknown"}
CDC_NAMES = {"Diabetes_binary": "diabetes", "HighBP": "high_bp", "HighChol": "high_chol", "CholCheck": "chol_check",
             "BMI": "bmi", "Smoker": "smoker", "Stroke": "stroke", "HeartDiseaseorAttack": "heart_disease_or_attack",
             "PhysActivity": "phys_activity", "Fruits": "fruits", "Veggies": "veggies",
             "HvyAlcoholConsump": "hvy_alcohol_consump", "AnyHealthcare": "any_healthcare",
             "NoDocbcCost": "no_docbc_cost", "GenHlth": "gen_hlth", "MentHlth": "ment_hlth", "PhysHlth": "phys_hlth",
             "DiffWalk": "diff_walk", "Sex": "sex", "Age": "age", "Education": "education", "Income": "income"}
PIMA_NAMES = {"Pregnancies": "pregnancies", "Glucose": "glucose", "BloodPressure": "blood_pressure",
              "SkinThickness": "skin_thickness", "Insulin": "insulin", "BMI": "bmi",
              "DiabetesPedigreeFunction": "dpf", "Age": "age", "Outcome": "diabetes"}
ZERO_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]  # giá trị 0 là mã thiếu, không phải đo được
# Mã xuất viện tử vong/chăm sóc cuối đời: không thể tái nhập viện nên bỏ khỏi bài toán (Strack và cộng sự, 2014)
DEAD_OR_HOSPICE = [11, 13, 14, 19, 20, 21]


def split_col(y, groups=None):
    """Nhãn train/val/test 70/15/15, phân tầng; nếu có groups thì tách theo nhóm để không rò rỉ giữa các tập."""
    idx = np.arange(len(y))
    out = np.full(len(y), "train", dtype=object)
    if groups is None:
        a, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
        _, va = train_test_split(a, test_size=0.15 / 0.85, stratify=y[a], random_state=SEED)
    else:
        a, te = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(idx, y, groups))
        tr, va = next(GroupShuffleSplit(1, test_size=0.15 / 0.85, random_state=SEED).split(a, y[a], groups[a]))
        va = a[va]
    out[va], out[te] = "val", "test"
    return out


def describe(raw, clean, target, notes):
    """Một dòng thống kê của bảng sau làm sạch; dup_raw đếm trên dữ liệu thô."""
    y = clean[target].astype(int)
    cols = [c for c in clean.columns if c != "split"]
    return {"rows_raw": len(raw), "rows_clean": len(clean), "columns": cols, "n_columns": len(cols),
            "positive_rate": float(y.mean()), "positive_rate_raw": float(notes.pop("pos_raw")),
            "duplicates_raw": int(raw.duplicated().sum()),
            "missing_cells": int(clean[cols].isna().sum().sum()),
            "missing_by_col": {c: int(v) for c, v in clean[cols].isna().sum().items() if v},
            "split_counts": clean["split"].value_counts().to_dict(), **notes}


def load_lab():
    raw = pd.read_csv(RAW["lab"])
    d = raw.drop_duplicates()
    n_dup = len(raw) - len(d)
    n_other = int((d.gender == "Other").sum())
    d = d[d.gender != "Other"]  # 18 dòng thô: quá hiếm để học, hợp đồng Profile chỉ có male/female
    d = pd.DataFrame({"age": d.age, "sex": d.gender.str.lower(), "bmi": d.bmi, "hba1c": d.HbA1c_level,
                      "glucose": d.blood_glucose_level.astype(float), "hypertension": d.hypertension,
                      "heart_disease": d.heart_disease, "smoking": d.smoking_history.map(SMOKING),
                      "diabetes": d.diabetes}).reset_index(drop=True)
    if d.smoking.isna().any():
        raise ValueError("Có giá trị smoking_history lạ chưa được ánh xạ")
    d["split"] = split_col(d.diabetes.values)
    return raw, d, "diabetes", {"pos_raw": raw.diabetes.mean(), "dropped_duplicates": n_dup, "dropped_other_sex": n_other}


def load_cdc():
    raw = pd.read_csv(RAW["cdc"]).drop(columns="ID")
    d = raw.rename(columns=CDC_NAMES).drop_duplicates().reset_index(drop=True).astype(int)
    d["split"] = split_col(d.diabetes.values)
    return raw, d, "diabetes", {"pos_raw": raw.Diabetes_binary.mean(), "dropped_duplicates": len(raw) - len(d)}


def load_pima_like(key):
    raw = pd.read_csv(RAW[key])
    d = raw.rename(columns=PIMA_NAMES).drop_duplicates().reset_index(drop=True)
    zeros = {c: int((d[c] == 0).sum()) for c in ZERO_MISSING}
    d[ZERO_MISSING] = d[ZERO_MISSING].replace(0, np.nan)
    d["split"] = split_col(d.diabetes.values)
    return raw, d, "diabetes", {"pos_raw": raw.Outcome.mean(), "dropped_duplicates": len(raw) - len(d),
                                "zero_as_missing": zeros}


def load_hospital():
    with zipfile.ZipFile(RAW["uci"]) as z, z.open("diabetic_data.csv") as f:
        raw = pd.read_csv(f, na_values="?", low_memory=False)
    d = raw.copy()
    d.columns = [c.lower().replace("-", "_") for c in d.columns]
    n0 = len(d)
    d = d[~d.discharge_disposition_id.isin(DEAD_OR_HOSPICE)]
    n_dead = n0 - len(d)
    d = d[d.gender != "Unknown/Invalid"]
    n_sex = n0 - n_dead - len(d)
    miss_weight = float(d.weight.isna().mean())
    d = d.drop(columns=["weight", "payer_code"])  # thiếu 97% và 40%, không suy ra được tại giường bệnh
    d["readmit_30"] = (d.readmitted == "<30").astype(int)
    d = d.drop_duplicates(subset="encounter_id").reset_index(drop=True)
    d["split"] = split_col(d.readmit_30.values, d.patient_nbr.values)
    sets = {s: set(d.patient_nbr[d.split == s]) for s in ("train", "val", "test")}
    overlap = len(sets["train"] & sets["val"]) + len(sets["train"] & sets["test"]) + len(sets["val"] & sets["test"])
    if overlap:
        raise ValueError(f"Rò rỉ bệnh nhân giữa các tập: {overlap}")
    return raw, d, "readmit_30", {"pos_raw": (raw.readmitted == "<30").mean(), "dropped_dead_or_hospice": n_dead,
                                  "dropped_invalid_sex": n_sex, "weight_missing_rate": miss_weight,
                                  "patients": int(d.patient_nbr.nunique()), "patient_overlap_between_splits": overlap,
                                  "dropped_columns": ["weight", "payer_code"]}


def main():
    summary = {}
    with slot(2, "data_load"), db.connect() as con:
        for table, fn in {"patients_lab": load_lab, "brfss": load_cdc, "hospital": load_hospital,
                          "pima": lambda: load_pima_like("pima"),
                          "frankfurt": lambda: load_pima_like("frankfurt")}.items():
            raw, d, target, notes = fn()
            d.to_sql(table, con, if_exists="replace", index_label="rowid_src", chunksize=5000)
            con.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_split ON {table}(split)")
            summary[table] = describe(raw, d, target, notes)
            print(table, len(raw), "->", len(d), f"dương {summary[table]['positive_rate']:.4f}", flush=True)
    # Frankfurt và Pima chung lược đồ: đếm dòng trùng nhau để đối chuẩn chéo miền không bị rò rỉ
    with db.connect() as con:
        p = pd.read_sql("select * from pima", con).drop(columns=["rowid_src", "split"])
        f = pd.read_sql("select * from frankfurt", con).drop(columns=["rowid_src", "split"])
    summary["pima_frankfurt_shared_rows"] = int(len(f.merge(p.drop_duplicates(), how="inner")))
    print("Frankfurt trùng Pima:", summary["pima_frankfurt_shared_rows"], flush=True)
    save_json("data_summary", summary)


if __name__ == "__main__":
    main()
