"""Hình mô hình bảng: ROC/PR, reliability, loss, so sánh mô hình, SHAP, ma trận nhầm lẫn, chéo miền. Đọc số từ outputs/metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import limit_threads, load_json  # noqa: E402

limit_threads(4)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from dnx.ml.tasks import MODEL_DIR, PRED_DIR, TASKS  # noqa: E402
from dnx.plot_style import C, save, vn  # noqa: E402

W = 6.3
ORDER = ["logreg", "random_forest", "xgboost", "mlp", "resnet", "ft_transformer"]
VI = {"logreg": "Hồi quy logistic", "random_forest": "Rừng ngẫu nhiên", "xgboost": "XGBoost", "mlp": "MLP",
      "resnet": "ResNet bảng", "ft_transformer": "FT-Transformer"}
COL = dict(zip(ORDER, (C["gray"], C["teal"], C["blue"], C["orange"], C["purple"], C["pink"])))  # tránh cặp đỏ-lục khó phân biệt
T_VI = {"full": "Có xét nghiệm", "screening": "Sàng lọc", "readmission": "Tái nhập viện"}


def preds(task, m):
    return np.load(PRED_DIR / f"{task}_{m}.npz")


def calibrated(task, m):
    from dnx.ml.metrics import calibrate
    p = preds(task, m)
    return p["yt"], calibrate(p["yv"], p["pv"]).predict(p["pt"]), p["pt"]


def fig_roc_pr(task):
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.9))
    for m in ORDER:
        p = preds(task, m)
        fpr, tpr, _ = roc_curve(p["yt"], p["pt"])
        pr, rc, _ = precision_recall_curve(p["yt"], p["pt"])
        lw = 2.4 if m == "xgboost" else 1.4
        axs[0].plot(fpr, tpr, color=COL[m], lw=lw, label=VI[m])
        axs[1].plot(rc, pr, color=COL[m], lw=lw)
    axs[0].plot([0, 1], [0, 1], color="#9CA3AF", lw=1, ls="--")
    axs[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
    axs[1].set(xlabel="Recall", ylabel="Precision", title="PR")
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    save(fig, f"ml_roc_pr_{task}")


def fig_reliability():
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.0))
    for ax, task in zip(axs, TASKS):
        y, cal, raw = calibrated(task, "xgboost")
        for p, col, name in ((raw, C["gray"], "Chưa hiệu chỉnh"), (cal, C["blue"], "Isotonic")):
            f, m = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(m, f, marker="o", ms=4, color=col, lw=1.6, label=name)
        hi = max(0.3, float(np.quantile(raw, 0.995)))
        ax.plot([0, hi], [0, hi], color="#9CA3AF", lw=1, ls="--")
        ax.set_title(T_VI[task], fontsize=12)
        ax.tick_params(labelsize=12)
    axs[0].set_ylabel("Tỉ lệ thực")
    axs[1].set_xlabel("Xác suất dự đoán")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    save(fig, "ml_reliability")


def fig_compare(s):
    """So sánh trên test theo ROC-AUC; mỗi bài toán một bảng con vì thang khác nhau."""
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.5), sharey=True)
    for ax, task in zip(axs, TASKS):
        vals = [s[task]["models"][m]["test_roc_auc"] for m in ORDER]
        y = np.arange(len(ORDER))
        # Chấm thay cột: trục cắt thì độ dài cột không còn ý nghĩa, vị trí chấm vẫn đúng
        ax.hlines(y, min(vals) - 1, vals, color="#E5E7EB", lw=1.2, zorder=1)
        ax.scatter(vals, y, s=64, zorder=2, color=[C["blue"] if m == "xgboost" else "#93C5FD" for m in ORDER])
        lo = min(vals) - (max(vals) - min(vals)) * 0.6 - 0.002
        ax.set_xlim(lo, max(vals) + (max(vals) - lo) * 0.62)
        for i, v in enumerate(vals):
            ax.text(v, i, "  " + vn(v, 3), va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_title(T_VI[task], fontsize=12)
        ax.grid(False)
    axs[0].set_yticks(np.arange(len(ORDER)), [VI[m] for m in ORDER])
    axs[0].invert_yaxis()
    fig.supxlabel("ROC-AUC trên test sau hiệu chỉnh (trục cắt, không bắt đầu từ 0)", fontsize=12, y=0.0)
    save(fig, "ml_compare")


def fig_loss():
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.3))
    for ax, task in zip(axs, TASKS):
        d = load_json(f"dl_{task}")["models"]
        for m in ("mlp", "resnet", "ft_transformer"):
            h = d[m]["history"]
            ep = [r["epoch"] for r in h]
            ax.plot(ep, [r["train_loss"] for r in h], color=COL[m], ls=":", lw=1.4)
            ax.plot(ep, [r["val_loss"] for r in h], color=COL[m], lw=1.8, label=VI[m])
        ax.set_title(T_VI[task], fontsize=12)
        ax.tick_params(labelsize=12)
    axs[0].set_ylabel("BCE")
    axs[1].set_xlabel("Epoch (nét liền: val, chấm: train)", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), fontsize=12)
    save(fig, "dl_loss")


def fig_shap():
    fig, axs = plt.subplots(1, 2, figsize=(W, 4.3))
    for ax, task in zip(axs, ("full", "screening")):
        rows = load_json(f"ml_shap_{task}")["mean_abs"][:8]
        y = np.arange(len(rows))
        ax.barh(y, [r["mean_abs_shap"] for r in rows], color=C["blue"], height=0.68)
        short = [r["label_vi"].split(" (")[0].replace("Số ngày ", "Ngày ") for r in rows]
        ax.set_yticks(y, [(n[:19] + "…") if len(n) > 20 else n for n in short], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("|SHAP| (log-odds)")
        ax.set_title(T_VI[task], fontsize=13)
        ax.grid(False, axis="y")
    fig.tight_layout()
    save(fig, "ml_shap_summary")


def fig_conf(s):
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.0), sharey=True)
    for ax, task in zip(axs, TASKS):
        c = s[task]["served_test"]["confusion"]
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, vn(m[i, j]), ha="center", va="center", fontsize=12,
                        color="white" if m[i, j] / m[i].sum() > 0.6 else "#111827")
        ax.set_xticks([0, 1], ["Âm", "Dương"])
        ax.set_title(T_VI[task], fontsize=12)
        ax.grid(False)
    axs[0].set_yticks([0, 1], ["Âm", "Dương"])
    axs[0].set_ylabel("Thực tế")
    axs[1].set_xlabel("Dự đoán")
    fig.tight_layout(w_pad=0.8)
    save(fig, "ml_confusion")


def fig_cross():
    r = [x for x in load_json("ml_crossdomain")["results"] if x["model"] == "xgboost"]
    dom = ["patients_lab", "pima", "frankfurt"]
    names = ["100k", "Pima", "Frankfurt"]
    m = np.array([[next(x["roc_auc"] for x in r if x["source"] == a and x["target"] == b) for b in dom] for a in dom])
    fig, ax = plt.subplots(figsize=(W, 3.7))
    im = ax.imshow(m, vmin=0.5, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, vn(m[i, j], 3), ha="center", va="center", fontsize=13, color="white" if m[i, j] > 0.78 else "#111827")
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set(xlabel="Miền đích", ylabel="Miền nguồn")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="ROC-AUC", shrink=0.85)
    save(fig, "ml_crossdomain")


def main():
    s = load_json("ml_summary")
    if not (MODEL_DIR / "full_calib.joblib").exists():
        raise FileNotFoundError("Chạy scripts/ml_select.py trước")
    for t in TASKS:
        fig_roc_pr(t)
    fig_reliability()
    fig_compare(s)
    fig_loss()
    fig_shap()
    fig_conf(s)
    fig_cross()


if __name__ == "__main__":
    main()
