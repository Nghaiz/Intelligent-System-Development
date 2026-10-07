"""Hình kết quả thực nghiệm của Chương 2, 3, 4, đọc từ outputs/metrics (không huấn luyện lại gì).

    python scripts/make_figures.py [ch2 ch3 ch4]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import C, plt, save, vn  # noqa: E402

from tl.common import METRICS, load_json  # noqa: E402

FW = {"scratch": ("NumPy", C["gray"]), "keras": ("Keras", C["blue"]), "torch": ("PyTorch", C["red"])}
ARCH = {"custom": ("Custom CNN", C["gray"]), "vgg11": ("VGG-11", C["blue"]),
        "mobilenetv2": ("MobileNetV2", C["orange"]), "resnet18": ("ResNet-18", C["green"])}
CELL = {"rnn": ("Simple RNN", C["gray"]), "lstm": ("LSTM", C["blue"]), "gru": ("GRU", C["orange"]),
        "bilstm": ("BiLSTM", C["green"])}


def cm_plot(ax, cm, labels, title, norm=True):
    cm = np.asarray(cm, float)
    show = cm / cm.sum(1, keepdims=True) if norm else cm
    ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if norm else None)
    ax.grid(False)
    n = len(labels)
    ax.set_xticks(range(n), labels, rotation=45 if n > 3 else 0, ha="right" if n > 3 else "center")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    fs = 6 if n > 5 else 8
    for i in range(n):
        for j in range(n):
            txt = f"{vn(int(cm[i, j]))}" if n <= 3 else f"{vn(100 * show[i, j], 0)}"
            if n <= 3:
                txt += f"\n({vn(100 * show[i, j], 1)} %)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs, color="white" if show[i, j] > 0.6 else "black")


def bars(ax, groups, series, colors, ylim=None, fmt=3):
    """groups: tên chỉ số; series: {tên mô hình: [giá trị theo groups]}."""
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for k, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + (k - (n - 1) / 2) * w, vals, w, label=name, color=colors[k])
        if n * len(groups) <= 24:
            for r, v in zip(b, vals):
                ax.text(r.get_x() + r.get_width() / 2, v, vn(v, fmt), ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.set_xticks(x, groups)
    if ylim:
        ax.set_ylim(*ylim)


# ================================================================= Chương 2
def ch2():
    sk = load_json("ch2_sklearn")
    mlp = {fw: load_json(f"ch2_mlp_{fw}") for fw in FW}
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.9))
    for ax, ds, t in zip(axs, ["diabetes", "house", "reviews"], ["Tiểu đường (BCE)", "Giá nhà (MSE, nhãn chuẩn hóa)", "Cảm xúc (entropy chéo)"]):
        for fw, (lab, c) in FW.items():
            h = mlp[fw][ds]["hist"]
            ep = np.arange(1, len(h["loss"]) + 1)
            ax.plot(ep, h["loss"], color=c, ls="--", lw=1)
            ax.plot(ep, h["val_loss"], color=c, lw=1.6, label=lab)
        ax.set_title(t)
        ax.set_xlabel("Epoch")
    axs[0].set_ylabel("Mất mát")
    axs[0].legend(title="nét liền: kiểm định\nnét đứt: huấn luyện", title_fontsize=7)
    fig.tight_layout()
    save(fig, "f2_loss_mlp")

    p = np.load(METRICS / "ch2_pred_house_sklearn.npz")
    y = p["y_test"]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, (name, yp) in zip(axs, [("Random Forest", p["rf"]), ("MLP PyTorch", np.load(METRICS / "ch2_pred_house_mlp_torch.npy"))]):
        ax.scatter(y, yp, s=3, alpha=0.25, color=C["blue"])
        ax.plot([0, 12], [0, 12], color=C["red"], lw=1.2)
        r = sk["house/rf"] if name == "Random Forest" else mlp["torch"]["house"]
        ax.set_title(f"{name}: R² = {vn(r['r2'], 3)}, MAE = {vn(r['mae'], 2)} tỷ")
        ax.set_xlabel("Giá thực tế (tỷ đồng)")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
    axs[0].set_ylabel("Giá dự báo (tỷ đồng)")
    fig.tight_layout()
    save(fig, "f2_house_pred")

    cols = [C["sky"], C["purple"], C["orange"], C["gray"], C["blue"], C["red"]]
    fig, ax = plt.subplots(figsize=(9, 3))
    ser = {"Logistic Reg.": sk["diabetes/logreg"], "SVM (RBF)": sk["diabetes/svm"], "Random Forest": sk["diabetes/rf"],
           "MLP NumPy": mlp["scratch"]["diabetes"], "MLP Keras": mlp["keras"]["diabetes"], "MLP PyTorch": mlp["torch"]["diabetes"]}
    keys = ["acc", "precision", "recall", "f1", "auc"]
    bars(ax, ["Accuracy", "Precision", "Recall", "F1 (lớp dương)", "ROC-AUC"], {k: [v[m] for m in keys] for k, v in ser.items()}, cols, (0, 1.12))
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "f2_bar_diab")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [2, 1]})
    ser = {"OLS": sk["house/ols"], "Ridge": sk["house/ridge"], "SVR (RBF)": sk["house/svr"], "Random Forest": sk["house/rf"],
           "MLP NumPy": mlp["scratch"]["house"], "MLP Keras": mlp["keras"]["house"], "MLP PyTorch": mlp["torch"]["house"]}
    cols7 = [C["sky"], C["yellow"], C["purple"], C["orange"], C["gray"], C["blue"], C["red"]]
    bars(axs[0], ["MAE (tỷ)", "RMSE (tỷ)"], {k: [v["mae"], v["rmse"]] for k, v in ser.items()}, cols7, (0, 1.9), 2)
    bars(axs[1], ["R²"], {k: [v["r2"]] for k, v in ser.items()}, cols7, (0, 0.8), 3)
    axs[0].legend(ncol=4, loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "f2_bar_house")

    fig, ax = plt.subplots(figsize=(9, 3))
    ser = {"Logistic Reg.": sk["reviews/logreg"], "Linear SVM": sk["reviews/linsvm"], "Random Forest": sk["reviews/rf"],
           "MLP NumPy": mlp["scratch"]["reviews"], "MLP Keras": mlp["keras"]["reviews"], "MLP PyTorch": mlp["torch"]["reviews"]}
    bars(ax, ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"], {k: [v[m] for m in keys[:4]] for k, v in ser.items()}, cols, (0, 1.05))
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "f2_bar_rev")

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
    for ax, (t, cm) in zip(axs, [("Logistic Regression", sk["diabetes/logreg"]["confusion"]), ("Random Forest", sk["diabetes/rf"]["confusion"]),
                                 ("MLP PyTorch", mlp["torch"]["diabetes"]["confusion"])]):
        cm_plot(ax, cm, ["Không mắc", "Mắc bệnh"], t)
    fig.tight_layout()
    save(fig, "f2_cm_diab")
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    lab = ["Tiêu cực", "Trung tính", "Tích cực"]
    for ax, (t, cm) in zip(axs, [("Logistic Regression", sk["reviews/logreg"]["confusion"]), ("Linear SVM", sk["reviews/linsvm"]["confusion"]),
                                 ("MLP PyTorch", mlp["torch"]["reviews"]["confusion"])]):
        cm_plot(ax, cm, lab, t)
    fig.tight_layout()
    save(fig, "f2_cm_rev")


# ================================================================= Chương 3
def ch3():
    eda = load_json("eda")
    bench = load_json("bench")
    for ds in ("svhn", "eurosat"):
        r = load_json(f"ch3_{ds}")
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.1))
        for a, (lab, c) in ARCH.items():
            h = r[a]["hist"]
            ep = np.arange(1, len(h["loss"]) + 1)
            axs[0].plot(ep, h["loss"], color=c, lw=1.5, label=lab)
            axs[1].plot(ep, np.array(h["val_acc"]) * 100, color=c, lw=1.5, label=lab)
        if "custom_keras" in r:
            h = r["custom_keras"]["hist"]
            ep = np.arange(1, len(h["loss"]) + 1)
            axs[0].plot(ep, h["loss"], color=C["purple"], lw=1.2, ls="--", label="Custom CNN (Keras)")
            axs[1].plot(ep, np.array(h["val_acc"]) * 100, color=C["purple"], lw=1.2, ls="--", label="Custom CNN (Keras)")
        axs[0].set_ylabel("Mất mát huấn luyện")
        axs[1].set_ylabel("Độ chính xác kiểm thử (%)")
        lo = 85 if ds == "svhn" else 80
        axs[1].set_ylim(lo, 100)
        for a in axs:
            a.set_xlabel("Epoch")
        axs[1].legend(loc="lower right")
        fig.tight_layout()
        save(fig, f"f3_curves_{ds}")
        fig, ax = plt.subplots(figsize=(5.4, 4.6) if ds == "eurosat" else (4.6, 4.2))
        cm_plot(ax, r["resnet18"]["confusion"], eda[ds]["classes"], f"ResNet-18 trên {ds.upper() if ds == 'svhn' else 'EuroSAT'} (% theo hàng)")
        save(fig, f"f3_cm_{ds}")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    x = np.arange(4)
    for ds, off, hatch in (("svhn", -0.2, None), ("eurosat", 0.2, "//")):
        g = [bench[f"ch3/{ds}/{a}"]["gpu_ms"] for a in ARCH]
        c = [bench[f"ch3/{ds}/{a}"]["cpu_ms"] for a in ARCH]
        f = [bench[f"ch3/{ds}/{a}"]["fps"] for a in ARCH]
        lab = "SVHN 32×32" if ds == "svhn" else "EuroSAT 64×64"
        for ax, v in zip(axs, (g, c, f)):
            ax.bar(x + off, v, 0.4, color=[ARCH[a][1] for a in ARCH], hatch=hatch, edgecolor="white", label=lab)
    for ax, t in zip(axs, ["Độ trễ GPU, lô 1 (ms)", "Độ trễ CPU 4 luồng, lô 1 (ms)", "Thông lượng GPU, lô 256 (ảnh/s)"]):
        ax.set_xticks(x, [v[0] for v in ARCH.values()], rotation=15)
        ax.set_title(t)
    axs[2].set_yscale("log")
    axs[0].legend(["SVHN (đặc)", "EuroSAT (gạch chéo)"], loc="upper left")
    fig.tight_layout()
    save(fig, "f3_latency")


# ================================================================= Chương 4
def ch4():
    t, k = load_json("ch4_torch"), load_json("ch4_keras")
    bench = load_json("bench")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    for c, (lab, col) in CELL.items():
        h = t[f"stock/{c}"]["hist"]
        ep = np.arange(1, len(h["loss"]) + 1)
        axs[0].plot(ep, h["loss"], color=col, lw=1.4, label=lab)
        axs[0].plot(ep, h["val_loss"], color=col, lw=1.4, ls="--")
        h = t[f"churn/{c}"]["hist"]
        ep = np.arange(1, len(h["loss"]) + 1)
        axs[1].plot(ep, h["loss"], color=col, lw=1.4, label=lab)
        axs[1].plot(ep, h["val_loss"], color=col, lw=1.4, ls="--")
    axs[0].set_title("FPT: MSE trên lợi suất (%²)")
    axs[1].set_title("KKBox: entropy chéo nhị phân")
    for a in axs:
        a.set_xlabel("Epoch")
    axs[0].legend(title="nét liền: huấn luyện\nnét đứt: kiểm định", title_fontsize=7)
    fig.tight_layout()
    save(fig, "f4_loss")

    p = np.load(METRICS / "ch4_stock_pred_torch.npz")
    d = np.array(p["dates"], dtype="datetime64[D]")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.7, 1]})
    for ax, sl in zip(axs, (slice(None), slice(-60, None))):
        ax.plot(d[sl], p["y_true"][sl], color="k", lw=1.3, label="Thực tế")
        ax.plot(d[sl], p["naive"][sl], color=C["gray"], lw=1, ls=":", label="Giữ nguyên giá")
        for c in ("rnn", "lstm", "bilstm"):
            ax.plot(d[sl], p[c][sl], color=CELL[c][1], lw=1, alpha=0.9, label=CELL[c][0])
        ax.set_ylabel("Nghìn đồng")
        ax.tick_params(axis="x", rotation=20)
    axs[0].set_title("Toàn bộ giai đoạn kiểm thử")
    axs[1].set_title("Phóng to 60 phiên cuối")
    axs[0].legend(ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, "f4_pred_stock")

    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, c in zip(axs, ("lstm", "bilstm")):
        cm_plot(ax, t[f"churn/{c}"]["confusion"], ["Ở lại", "Rời bỏ"], f"{CELL[c][0]} (ngưỡng {vn(t[f'churn/{c}']['threshold'], 2)})")
    fig.tight_layout()
    save(fig, "f4_cm_churn")

    fig, axs = plt.subplots(1, 4, figsize=(12, 2.9))
    x = np.arange(4)
    rows = [("RMSE FPT (nghìn đ)", lambda r, c: r[f"stock/{c}"]["rmse"], None),
            ("F1 KKBox", lambda r, c: r[f"churn/{c}"]["f1"], None),
            ("Số tham số (FPT)", lambda r, c: r[f"stock/{c}"]["params"], None),
            ("Độ trễ CPU, lô 1 (ms)", None, None)]
    for ax, (title, get, _) in zip(axs, rows):
        if get is None:
            vt = [bench[f"ch4/stock/{c}_torch"]["cpu_ms"] for c in CELL]
            vk = [bench[f"ch4/stock/{c}_keras"]["cpu_ms"] for c in CELL]
        else:
            vt = [get(t, c) for c in CELL]
            vk = [get(k, c) for c in CELL]
        ax.bar(x - 0.2, vt, 0.4, color="#EE4C2C", label="PyTorch")
        ax.bar(x + 0.2, vk, 0.4, color=C["blue"], label="Keras")
        ax.set_xticks(x, [v[0] for v in CELL.values()], rotation=20)
        ax.set_title(title)
        if title.startswith("RMSE"):
            ax.axhline(t["stock/naive"]["rmse"], color="k", ls=":", lw=1)
            ax.set_ylim(1.5, max(vt + vk) * 1.02)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "f4_fw")


if __name__ == "__main__":
    which = sys.argv[1:] or ["ch2", "ch3", "ch4"]
    for w in which:
        globals()[w]()
