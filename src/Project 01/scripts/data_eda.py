"""Mười hình khám phá dữ liệu cho báo cáo: nhãn, tương quan, phân bố theo nhãn, tỉ lệ bệnh theo tuổi/BMI, thuốc nội viện."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import limit_threads, load_json, save_json  # noqa: E402

limit_threads(4)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dnx.db import connect  # noqa: E402
from dnx.mem import slot  # noqa: E402
from dnx.ml.serve import LABEL_VI  # noqa: E402
from dnx.ml.tasks import HOSP_MEDS, frame  # noqa: E402
from dnx.plot_style import C, save, vn  # noqa: E402

W = 6.3
NEG, POS = C["blue"], C["orange"]
NAMES = {"patients_lab": "100k xét nghiệm", "brfss": "CDC BRFSS", "hospital": "130 bệnh viện", "pima": "Pima",
         "frankfurt": "Frankfurt"}
DIAG_VI = {"circulatory": "Tuần hoàn", "respiratory": "Hô hấp", "digestive": "Tiêu hóa", "diabetes": "Đái tháo đường",
           "injury": "Chấn thương", "genitourinary": "Tiết niệu", "musculoskeletal": "Cơ xương", "neoplasm": "Ung thư",
           "other": "Khác", "missing": "Thiếu"}
AGE_CDC = ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "80+"]


def bars(ax, labels, vals, color=C["blue"], fmt=lambda v: vn(v, 1) + " %", pad=0.01):
    """Cột ngang kèm nhãn số ở đầu cột; trục giá trị đã có nhãn trực tiếp nên bỏ lưới."""
    y = np.arange(len(labels))
    ax.barh(y, vals, color=color, height=0.68)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    m = max(vals)
    for i, v in enumerate(vals):
        ax.text(v + pad * m, i, fmt(v), va="center", fontsize=12)
    ax.set_xlim(0, m * 1.2)
    ax.grid(False)


def fig_labels(s):
    keys = list(NAMES)
    fig, ax = plt.subplots(figsize=(W, 3.1))
    bars(ax, [NAMES[k] for k in keys], [s[k]["positive_rate"] * 100 for k in keys])
    ax.set_xlabel("Tỉ lệ lớp dương (%)")
    save(fig, "data_label_balance")


def fig_rows(s):
    keys = list(NAMES)
    raw, clean = [s[k]["rows_raw"] for k in keys], [s[k]["rows_clean"] for k in keys]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    y = np.arange(len(keys))
    ax.barh(y - 0.2, raw, 0.36, color=C["gray"], label="Thô")
    ax.barh(y + 0.2, clean, 0.36, color=C["blue"], label="Sau làm sạch")
    for i in range(len(keys)):
        ax.text(max(raw[i], clean[i]) * 1.15, i, f"{vn(raw[i])} → {vn(clean[i])}", va="center", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(300, 6e7)
    ax.set_yticks(y, [NAMES[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Số dòng (thang log)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, -0.42), ncol=2)
    save(fig, "data_rows")


def fig_lab_corr(lab):
    cols = ["age", "bmi", "hba1c", "glucose", "hypertension", "heart_disease", "diabetes"]
    names = ["Tuổi", "BMI", "HbA1c", "Đường huyết", "THA", "Bệnh tim", "ĐTĐ"]
    m = lab[cols].corr().values
    fig, ax = plt.subplots(figsize=(W, 4.9))
    im = ax.imshow(m, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    ax.set_xticks(range(7), names, rotation=40, ha="right")
    ax.set_yticks(range(7), names)
    ax.grid(False)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, vn(m[i, j], 2), ha="center", va="center", fontsize=12,
                    color="white" if abs(m[i, j]) > 0.38 else "#111827")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Hệ số Pearson (cắt ở ±0,5)")
    save(fig, "data_lab_corr")


def fig_brfss_corr(b):
    c = b.drop(columns=["rowid_src", "split"]).corr()["diabetes"].drop("diabetes")
    c = c.reindex(c.abs().sort_values(ascending=False).index)[:10]
    fig, ax = plt.subplots(figsize=(W, 4.2))
    y = np.arange(len(c))
    ax.barh(y, c.values, color=[C["red"] if v > 0 else C["blue"] for v in c.values], height=0.68)
    ax.set_yticks(y, [LABEL_VI.get(k, k) if k != "age" else "Nhóm tuổi" for k in c.index])
    ax.invert_yaxis()
    ax.axvline(0, color="#374151", lw=1)
    ax.set_xlabel("Tương quan với đái tháo đường (Pearson)")
    save(fig, "data_brfss_corr")


def fig_lab_dist(lab):
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.1), sharey=False)
    for ax, col, name in ((axs[0], "hba1c", "HbA1c (%)"), (axs[1], "glucose", "Đường huyết (mg/dL)")):
        vals = sorted(lab[col].unique())
        rate = lab.groupby(col).diabetes.agg(["mean", "size"]).reindex(vals)
        ax.bar(range(len(vals)), rate["mean"] * 100, color=C["blue"], width=0.74)
        ticks = range(0, len(vals), 3)  # 18 mức rời rạc, ghi nhãn cách 3 mức cho khỏi chồng chữ
        ax.set_xticks(list(ticks), [vn(vals[i], 1) if col == "hba1c" else f"{vals[i]:.0f}" for i in ticks], fontsize=12)
        ax.set_xlabel(name)
        ax.grid(False)
    axs[0].set_ylabel("Tỉ lệ ĐTĐ (%)")
    fig.tight_layout()
    save(fig, "data_lab_rate_by_test")


def binned_rate(x, y, edges):
    g = pd.cut(x, edges, right=False)
    return pd.DataFrame({"y": y.values, "g": g}).groupby("g", observed=True).y.mean() * 100


def fig_age(lab, b):
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.3))
    r = binned_rate(lab.age, lab.diabetes, list(range(0, 91, 10)))
    axs[0].plot(range(len(r)), r.values, marker="o", ms=7)
    axs[0].set_xticks(range(len(r)), [f"{int(i.left)}+" for i in r.index], rotation=60)
    axs[0].set_title("100k xét nghiệm")
    rb = b.groupby("age").diabetes.mean() * 100
    axs[1].plot(range(len(rb)), rb.values, marker="o", ms=7, color=C["teal"])
    axs[1].set_xticks(range(0, 13, 2), AGE_CDC[::2], rotation=60)
    axs[1].set_title("CDC BRFSS")
    axs[0].set_ylabel("Tỉ lệ ĐTĐ (%)")
    axs[0].set_xlabel("Nhóm tuổi")
    axs[1].set_xlabel("Nhóm tuổi")
    fig.tight_layout(w_pad=1.5)
    save(fig, "data_rate_by_age")


def fig_bmi(lab, b):
    edges = [10, 18.5, 25, 30, 35, 40, 100]
    names = ["<18,5", "18,5–25", "25–30", "30–35", "35–40", "≥40"]
    r1, r2 = binned_rate(lab.bmi, lab.diabetes, edges), binned_rate(b.bmi, b.diabetes, edges)
    fig, ax = plt.subplots(figsize=(W, 3.2))
    x = np.arange(6)
    ax.bar(x - 0.2, r1.values, 0.36, color=C["blue"], label="100k xét nghiệm")
    ax.bar(x + 0.2, r2.values, 0.36, color=C["teal"], label="CDC BRFSS")
    ax.set_xticks(x, names)
    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Tỉ lệ ĐTĐ (%)")
    ax.legend()
    ax.grid(False, axis="x")
    save(fig, "data_rate_by_bmi")


def fig_meds(h):
    use = {m: float((h[m] != "No").mean() * 100) for m in HOSP_MEDS}
    use = dict(sorted(use.items(), key=lambda kv: -kv[1]))
    fig, ax = plt.subplots(figsize=(W, 3.7))
    bars(ax, [k.capitalize() for k in use], list(use.values()), color=C["purple"])
    ax.set_xlabel("Tỉ lệ lượt nhập viện có dùng thuốc (%)")
    save(fig, "data_hospital_meds")
    return use


def fig_hosp_rate(h):
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    r = h.groupby(h.number_inpatient.clip(upper=5)).readmit_30.mean() * 100
    axs[0].bar(range(len(r)), r.values, color=C["blue"], width=0.72)
    axs[0].set_xticks(range(len(r)), [f"{i}" if i < 5 else "5+" for i in r.index])
    axs[0].set_xlabel("Số lần nội trú năm trước")
    axs[0].set_ylabel("Tái nhập viện < 30 ngày (%)")
    axs[0].grid(False, axis="x")
    g = h.groupby("diag_1_group").readmit_30.agg(["mean", "size"])
    g = g[g["size"] > 500].sort_values("mean", ascending=False)
    bars(axs[1], [DIAG_VI.get(k, k) for k in g.index], list(g["mean"] * 100))
    axs[1].set_xlabel("Theo chẩn đoán chính (%)")
    fig.tight_layout(w_pad=3.0)  # chừa khoảng cho nhãn chẩn đoán khỏi đè cột 5+ của ô trái
    save(fig, "data_hospital_readmit")


def fig_shift(lab, pima, frank):
    fig, ax = plt.subplots(figsize=(W, 3.4))
    data = [(lab, "100k xét nghiệm", C["blue"]), (pima, "Pima", C["orange"]), (frank, "Frankfurt", C["green"])]
    for d, name, col in data:
        for lab_y, ls in ((0, ":"), (1, "-")):
            v = d.glucose[d.diabetes == lab_y].dropna()
            h, e = np.histogram(v, bins=np.arange(40, 310, 15), density=True)
            ax.plot((e[:-1] + e[1:]) / 2, h, color=col, ls=ls, label=f"{name}, {'dương' if lab_y else 'âm'}")
    ax.set_xlabel("Đường huyết (mg/dL)")
    ax.set_ylabel("Mật độ")
    ax.set_yticks([])
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0))
    save(fig, "data_domain_shift")


def main():
    s = load_json("data_summary")
    with slot(2, "data_eda"), connect() as con:
        lab = pd.read_sql("select * from patients_lab", con)
        b = pd.read_sql("select * from brfss", con)
        pima = pd.read_sql("select * from pima", con)
        frank = pd.read_sql("select * from frankfurt", con)
        h = frame("readmission")
        fig_labels(s)
        fig_rows(s)
        fig_lab_corr(lab)
        fig_brfss_corr(b)
        fig_lab_dist(lab)
        fig_age(lab, b)
        fig_bmi(lab, b)
        use = fig_meds(h)
        fig_hosp_rate(h)
        fig_shift(lab, pima, frank)
    # Kiểm tra bản chất nhãn của bộ 100k: nhãn dương có nằm hoàn toàn trên ngưỡng xét nghiệm hay không
    low = lab[(lab.hba1c < 6.5) & (lab.glucose < 200)]
    high = lab[(lab.hba1c >= 6.5) | (lab.glucose >= 200)]
    save_json("data_eda", {
        "lab_hba1c_values": sorted(float(v) for v in lab.hba1c.unique()),
        "lab_glucose_values": sorted(float(v) for v in lab.glucose.unique()),
        "lab_rate_below_both_thresholds": float(low.diabetes.mean()), "lab_n_below_both": len(low),
        "lab_rate_above_any_threshold": float(high.diabetes.mean()), "lab_n_above_any": len(high),
        "lab_positive_max_hba1c_rate_by_value": {str(k): float(v) for k, v in lab.groupby("hba1c").diabetes.mean().items()},
        "hospital_med_use_pct": use})


if __name__ == "__main__":
    main()
