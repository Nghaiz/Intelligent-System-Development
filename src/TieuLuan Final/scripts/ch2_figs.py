"""Hình dữ liệu Chương 2, bản nâng cấp từ scripts/ch2_figs.py của Tiểu luận 03 cho ba dataset mới."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from tlf.ch2_data import CDC_TARGET, load_cdc, load_housing, load_vsfc, normalize_vi
from tlf.ch2_models import NAMES
from tlf.common import load_json, model_dir
from tlf.plot_style import C, H1, W, W2, plt, save, vn, vn_axis

M = model_dir("ch2")
H2 = 3.0
SHORT = {"dummy": "Dummy", "linear": "Tuyến tính", "svm": "SVM", "tree": "Cây", "rf": "RF", "xgb": "XGB",
         "mlp_numpy": "NumPy", "mlp_keras": "Keras", "mlp_torch": "PyTorch"}
COLOR = {"dummy": C["gray"], "linear": C["blue"], "svm": C["sky"], "tree": C["lime"], "rf": C["green"],
         "xgb": C["orange"], "mlp_numpy": C["purple"], "mlp_keras": C["pink"], "mlp_torch": C["red"]}


def heat(ax, M_, xt, yt, fs=12):
    """Bản đồ nhiệt có số trong ô, lấy từ ch2_figs của Tiểu luận 03."""
    ax.imshow(M_, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    ax.grid(False)
    for i in range(M_.shape[0]):
        for j in range(M_.shape[1]):
            ax.text(j, i, vn(M_[i, j], 2), ha="center", va="center", fontsize=fs,
                    color="white" if M_[i, j] > 0.6 else "black")


def compare_bar(rows, key, name, xlabel, lower=False):
    """Cột ngang có thanh sai số bootstrap; bỏ thanh khi chỉ số không phải chỉ số có khoảng tin cậy."""
    rows = sorted(rows, key=lambda r: r[key], reverse=lower)
    fig, ax = plt.subplots(figsize=(W2, H2 + 0.6))
    y = np.arange(len(rows))
    v = np.array([r[key] for r in rows])
    ax.barh(y, v, color=[COLOR[r["family"]] for r in rows])
    if key in ("f1_macro", "rmse"):
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        ax.errorbar(v, y, xerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], fmt="none", ecolor="black", capsize=2)
    ax.set_yticks(y, [SHORT[r["family"]] for r in rows])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    vn_axis(ax, "x", 2)
    save(fig, name)


def importance_bar(rows, name, labels, xlabel):
    rows = rows[:8][::-1]
    fig, ax = plt.subplots(figsize=(W2, H2 + 0.6))
    ax.barh([labels.get(r["feature"], r["feature"]) for r in rows], [r["drop"] for r in rows], color=C["orange"])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    vn_axis(ax, "x", 2)
    save(fig, name)


def cm_fig(y, pred, labels, name):
    cm = confusion_matrix(y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(W2, H2))
    heat(ax, cm, labels, labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    save(fig, name)


def cdc_figs():
    df = load_cdc()
    fig, ax = plt.subplots(figsize=(W2, H2))
    rate = df.groupby("Age")[CDC_TARGET].mean() * 100
    ax.bar(rate.index, rate.values, color=C["blue"])
    ax.set_xlabel("Nhóm tuổi (1: 18-24, 13: từ 80)")
    ax.set_ylabel("Tỷ lệ tiểu đường (%)")
    save(fig, "ch2_cdc_age")
    fig, ax = plt.subplots(figsize=(W2, H2))
    bins = np.arange(12, 62, 2)
    for lab, col, nm in [(0, C["blue"], "Không"), (1, C["red"], "Có")]:
        ax.hist(df.loc[df[CDC_TARGET] == lab, "BMI"], bins=bins, density=True, alpha=0.55, color=col, label=nm)
    ax.set_xlabel("BMI")
    ax.set_ylabel("Mật độ")
    ax.legend()
    save(fig, "ch2_cdc_bmi")
    d = np.load(M / "pred_cdc.npz")
    fig, ax = plt.subplots(figsize=(W2, H2))
    for fam in ("linear", "tree", "rf", "xgb"):
        fpr, tpr, _ = roc_curve(d["y"], d[f"{fam}_p"])
        ax.plot(fpr, tpr, color=COLOR[fam], label=SHORT[fam])
    ax.plot([0, 1], [0, 1], color=C["gray"], ls="--", lw=1)
    ax.set_xlabel("Tỷ lệ dương giả")
    ax.set_ylabel("Tỷ lệ dương thật")
    ax.legend()
    vn_axis(ax, "x", 1)
    vn_axis(ax, "y", 1)
    save(fig, "ch2_cdc_roc")
    cm_fig(d["y"], d["xgb"], ["Không", "Có"], "ch2_cdc_cm")


def housing_figs():
    df = load_housing()
    fig, ax = plt.subplots(figsize=(W2, H2))
    ax.hist(df["Price"], bins=np.arange(1, 11.75, 0.5), color=C["teal"])
    ax.set_xlabel("Giá (tỷ đồng)")
    ax.set_ylabel("Số tin")
    vn_axis(ax)
    save(fig, "ch2_housing_price")
    fig, ax = plt.subplots(figsize=(W2, H2))
    ax.hexbin(df["Area"], df["Price"], gridsize=32, xscale="log", cmap="YlGnBu", mincnt=1, bins="log")
    ax.set_xlabel("Diện tích (m², thang log)")
    ax.set_ylabel("Giá (tỷ đồng)")
    ax.grid(False)
    save(fig, "ch2_housing_area")
    d = np.load(M / "pred_housing.npz")
    y, pred = d["y"], d["xgb"]
    fig, ax = plt.subplots(figsize=(W2, H2))
    ax.hexbin(y, pred, gridsize=30, cmap="OrRd", mincnt=1, bins="log")
    ax.plot([1, 11.5], [1, 11.5], color=C["navy"], ls="--")
    ax.set_xlabel("Giá thật (tỷ đồng)")
    ax.set_ylabel("Giá dự đoán")
    ax.grid(False)
    save(fig, "ch2_housing_pred")
    edges = np.array([1, 3, 5, 7, 9, 11.5])
    idx = np.clip(np.digitize(y, edges) - 1, 0, 4)
    bias = [float(np.mean((pred - y)[idx == i])) for i in range(5)]
    fig, ax = plt.subplots(figsize=(W2, H2))
    ax.bar([f"{edges[i]:g}-{edges[i + 1]:g}".replace(".", ",") for i in range(5)], bias,
           color=[C["red"] if b > 0 else C["blue"] for b in bias])
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Khoảng giá thật (tỷ đồng)")
    ax.set_ylabel("Sai lệch TB (tỷ đồng)")
    ax.tick_params(axis="x", labelsize=10.5)
    vn_axis(ax, "y", 1)
    save(fig, "ch2_housing_bias")


def vsfc_figs():
    tr = load_vsfc("train")
    labels = ["Tiêu cực", "Trung tính", "Tích cực"]
    fig, ax = plt.subplots(figsize=(W2, H2))
    c = tr["sentiment"].value_counts().sort_index()
    bars = ax.bar(labels, c.values, color=[C["red"], C["gray"], C["green"]])
    ax.bar_label(bars, [vn(v) for v in c.values], fontsize=11)
    ax.set_ylabel("Số câu (train)")
    ax.set_ylim(0, c.max() * 1.15)
    vn_axis(ax)
    save(fig, "ch2_vsfc_label")
    fig, ax = plt.subplots(figsize=(W2, H2))
    n = tr["sentence"].map(lambda s: len(normalize_vi(s).split()))
    ax.hist(n.clip(upper=50), bins=np.arange(0, 52, 2), color=C["purple"])
    ax.set_xlabel("Số âm tiết mỗi câu (gộp từ 50)")
    ax.set_ylabel("Số câu")
    vn_axis(ax)
    save(fig, "ch2_vsfc_len")
    d = np.load(M / "pred_vsfc.npz")
    cm_fig(d["y"], d["xgb"], ["Tiêu", "Trung", "Tích"], "ch2_vsfc_cm")


def compare_figs():
    rows = load_json("ch2_compare")["rows"]
    pick = lambda t: [r for r in rows if r["dataset"] == t]  # noqa: E731
    compare_bar(pick("CDC Diabetes"), "f1_macro", "ch2_cdc_compare", "F1 macro (test)")
    compare_bar(pick("Vietnam Housing"), "rmse", "ch2_housing_compare", "RMSE, tỷ đồng (test)", lower=True)
    compare_bar(pick("UIT-VSFC"), "f1_macro", "ch2_vsfc_compare", "F1 macro (test)")
    cdc_names = {"HighBP": "Huyết áp cao", "GenHlth": "Sức khỏe chung", "BMI": "BMI", "Age": "Tuổi",
                 "HighChol": "Cholesterol cao", "Income": "Thu nhập", "CholCheck": "Đo cholesterol", "Sex": "Giới tính",
                 "HeartDiseaseorAttack": "Bệnh tim", "HvyAlcoholConsump": "Rượu nhiều", "DiffWalk": "Khó đi bộ",
                 "PhysHlth": "Ngày thể chất kém", "Education": "Học vấn"}
    importance_bar(load_json("ch2_res_cdc_classic")["importance"]["rows"], "ch2_cdc_imp", cdc_names, "Giảm AUC")
    house_names = {"Area": "Diện tích", "Street": "Mã hóa đường", "Ward": "Mã hóa phường", "District": "Mã hóa quận",
                   "Floors": "Số tầng", "Frontage": "Mặt tiền", "Access Road": "Đường vào", "Bathrooms": "Phòng tắm",
                   "Bedrooms": "Phòng ngủ", "missingindicator_Frontage": "Thiếu mặt tiền",
                   "missingindicator_Access Road": "Thiếu đường vào", "missingindicator_Floors": "Thiếu số tầng",
                   "missingindicator_Bedrooms": "Thiếu phòng ngủ", "missingindicator_Bathrooms": "Thiếu phòng tắm"}
    importance_bar(load_json("ch2_res_housing_classic")["importance"]["rows"], "ch2_housing_imp", house_names, "Giảm $R^2$")


def mlp_curves():
    """Ba bài, mỗi bài ba đường mất mát validation; các nền tảng phải hội tụ về cùng một mức."""
    fig, axes = plt.subplots(1, 3, figsize=(W, H1 - 0.8))
    for ax, ds, t in zip(axes, ("cdc", "housing", "vsfc"), ("CDC", "Housing", "VSFC")):
        for fw in ("numpy", "keras", "torch"):
            h = load_json(f"ch2_res_{ds}_mlp_{fw}")["history"]
            ax.plot([r["epoch"] for r in h], [r["val_loss"] for r in h], color=COLOR[f"mlp_{fw}"], label=SHORT[f"mlp_{fw}"])
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        vn_axis(ax, "y", 3 if ds == "cdc" else 2)
    axes[0].set_ylabel("Mất mát val")
    axes[0].legend(fontsize=10)
    fig.subplots_adjust(wspace=0.5)
    save(fig, "ch2_mlp_curves")


if __name__ == "__main__":
    assert NAMES
    for f in (cdc_figs, housing_figs, vsfc_figs, compare_figs, mlp_curves):
        f()
