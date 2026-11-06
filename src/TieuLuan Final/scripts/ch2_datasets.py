"""Thống kê ba dataset, kiểm tra rò rỉ và ghi outputs/metrics/ch2_datasets.json."""
import pandas as pd

from tlf.ch2_data import (CDC_TARGET, D, VSFC_LABELS, group_split_idx, load_cdc, load_housing, load_vsfc,
                          normalize_vi, split_70_15_15)
from tlf.common import save_json
from tlf.mem import slot

CDC_SHOW = ["Diabetes_binary", "BMI", "HighBP", "HighChol", "GenHlth", "PhysHlth", "Age", "Income"]
HOUSE_SHOW = ["Address", "Area", "Floors", "Bedrooms", "Legal status", "Price"]
PRICE_BINS = [1, 3, 5, 7, 9, 11.5]


def mb(*names):
    return round(sum((D / n).stat().st_size for n in names) / 2**20, 2)


def cdc_entry():
    raw = pd.read_csv(D / "cdc_diabetes.csv")
    df = load_cdc()
    X = df.drop(columns=CDC_TARGET)
    _, _, X_te, _, _, _ = split_70_15_15(X, df[CDC_TARGET].to_numpy())
    c = df[CDC_TARGET].value_counts().sort_index()
    return {
        "id": "cdc", "name": "CDC Diabetes Health Indicators", "source": "UCI Machine Learning Repository, id 891",
        "url": "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
        "task": "Phân loại nhị phân", "n_samples": len(raw), "n_features": X.shape[1],
        "size_mb": mb("cdc_diabetes.csv"), "license": "Xem trang UCI; chưa xác minh được qua API",
        "note": "Khảo sát BRFSS 2015, tự khai báo; bỏ hồ sơ trùng khít trước khi chia.",
        "sample_rows": raw[CDC_SHOW].sample(5, random_state=1).to_dict("records"),
        "distribution": {"labels": ["Không", "Có"], "values": c.tolist()},
        "stats": {"n_raw": len(raw), "n_clean": len(df), "n_dup": len(raw) - len(df),
                  "n_pos": int(c[1]), "pos_rate": float(df[CDC_TARGET].mean()),
                  "raw_pos_rate": float(raw[CDC_TARGET].mean()), "missing": int(raw.isna().sum().sum()),
                  "n_test": len(X_te), "bmi_med_pos": float(df.loc[df[CDC_TARGET] == 1, "BMI"].median()),
                  "bmi_med_neg": float(df.loc[df[CDC_TARGET] == 0, "BMI"].median())}}


def housing_entry():
    df = load_housing()
    tr, va, te = group_split_idx(df["Address"].to_numpy())
    addr = df["Address"].to_numpy()
    overlap = len(set(addr[tr]) & set(addr[te]))
    bins = pd.cut(df["Price"], PRICE_BINS, include_lowest=True).value_counts().sort_index()
    top = df["Province"].value_counts()
    return {
        "id": "housing", "name": "Vietnam Housing 2024", "source": "Kaggle, nguyentiennhan",
        "url": "https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024",
        "task": "Hồi quy", "n_samples": len(df), "n_features": 11, "size_mb": mb("vietnam_housing.csv"),
        "license": "CC0: Public Domain", "note": "Tin rao bán, giá trong khoảng 1 đến 11,5 tỷ đồng.",
        "sample_rows": df[HOUSE_SHOW].sample(5, random_state=3).to_dict("records"),
        "distribution": {"labels": [f"{b.left:g}-{b.right:g}" for b in bins.index], "values": bins.tolist()},
        "stats": {"n": len(df), "n_address": df["Address"].nunique(), "n_province": df["Province"].nunique(),
                  "n_district": df["District"].nunique(), "price_mean": float(df["Price"].mean()),
                  "price_med": float(df["Price"].median()), "price_min": float(df["Price"].min()),
                  "price_max": float(df["Price"].max()), "price_std": float(df["Price"].std()),
                  "area_med": float(df["Area"].median()), "corr_area": float(df["Area"].corr(df["Price"])),
                  "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                  "address_overlap": overlap, "top2_share": float(top.iloc[:2].sum() / len(df)),
                  "balcony_missing": float(df["Balcony direction"].isna().mean()),
                  "furniture_missing": float(df["Furniture state"].isna().mean()),
                  "frontage_missing": float(df["Frontage"].isna().mean()),
                  "near_dup": int(df.duplicated(["Address", "Area", "Floors", "Bedrooms", "Bathrooms",
                                                 "Frontage"]).sum())}}


def vsfc_entry():
    parts = {p: load_vsfc(p) for p in ("train", "validation", "test")}
    tr, te = parts["train"], parts["test"]
    full = pd.concat(parts.values())
    c = full["sentiment"].value_counts().sort_index()
    lens = full["sentence"].map(lambda s: len(normalize_vi(s).split()))
    sample = te.sample(5, random_state=7)
    return {
        "id": "vsfc", "name": "UIT-VSFC", "source": "Hugging Face, uitnlp/vietnamese_students_feedback",
        "url": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback",
        "task": "Phân loại ba lớp", "n_samples": len(full), "n_features": 1,
        "size_mb": mb("vsfc_train.parquet", "vsfc_validation.parquet", "vsfc_test.parquet"),
        "license": "Thẻ dữ liệu ghi unknown, nhóm UIT phát hành cho nghiên cứu",
        "note": "Giữ nguyên phân chia gốc của tác giả.",
        "sample_rows": [{"sentence": r.sentence, "sentiment": VSFC_LABELS[r.sentiment]}
                        for r in sample.itertuples()],
        "distribution": {"labels": VSFC_LABELS, "values": c.tolist()},
        "stats": {"n_train": len(tr), "n_val": len(parts["validation"]), "n_test": len(te),
                  "neutral_rate": float((full["sentiment"] == 1).mean()), "len_mean": float(lens.mean()),
                  "len_med": float(lens.median()), "len_max": int(lens.max()),
                  "test_in_train": int(te["sentence"].isin(tr["sentence"]).sum()),
                  "masked": int(full["sentence"].str.contains("wzjwz").sum())}}


if __name__ == "__main__":
    with slot(1.5, "ch2 datasets"):
        rows = [cdc_entry(), housing_entry(), vsfc_entry()]
    save_json("ch2_datasets", {"datasets": rows, **{d["id"]: d["stats"] for d in rows}})
