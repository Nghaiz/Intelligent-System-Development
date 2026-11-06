"""Hình võng mạc: lưới ảnh thật theo cấp, đường học, ma trận nhầm lẫn và Grad-CAM trên ảnh test."""
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import ROOT, limit_threads, load_json  # noqa: E402

limit_threads(4)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dnx.ml.retina import RDIR, SHORT_VI, preprocess  # noqa: E402
from dnx.ml.serve_retina import grade  # noqa: E402
from dnx.plot_style import C, save, vn  # noqa: E402

W = 6.3
SHOW = ROOT / "web" / "static" / "img" / "retina"
VI = {"resnet34": "ResNet-34", "efficientnet_b0": "EfficientNet-B0"}


def fig_grid():
    fig, axs = plt.subplots(2, 5, figsize=(W, 3.5))
    for k in range(5):
        for r in range(2):
            ax = axs[r, k]
            ax.imshow(preprocess(Image.open(SHOW / f"grade{k}_{r + 1}.jpg")))  # cùng cắt viền và cỡ như đầu vào mô hình
            ax.axis("off")
            if r == 0:
                ax.set_title(f"Cấp {k}\n{SHORT_VI[k]}", fontsize=12, fontweight="normal")
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    save(fig, "retina_samples")


def fig_counts(s):
    n = np.array(s["splits"]["train"]["class_counts"])
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.bar(range(5), n, color=C["blue"], width=0.72)
    for i, v in enumerate(n):
        ax.text(i, v, vn(int(v)), ha="center", va="bottom", fontsize=12)
    ax.set_xticks(range(5), SHORT_VI)
    ax.set_ylabel("Số ảnh train")
    ax.grid(False)
    ax.set_yticks([])
    save(fig, "retina_class_balance")


def fig_curves(r):
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.3))
    for name, col in zip(r["models"], (C["blue"], C["orange"])):
        h = r["models"][name]["history"]
        ep = [x["epoch"] for x in h]
        axs[0].plot(ep, [x["train_loss"] for x in h], color=col, ls=":", lw=1.6)
        axs[0].plot(ep, [x["val_loss"] for x in h], color=col, label=VI[name])
        axs[1].plot(ep, [x["val_qwk"] for x in h], color=col, marker="o", ms=4)
    axs[0].set(xlabel="Epoch (liền: val, chấm: train)", ylabel="Cross-entropy")
    axs[1].set(xlabel="Epoch", ylabel="QWK trên val")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    save(fig, "retina_loss")


def fig_conf(r):
    sel = r["selected"]
    m = np.array(r["models"][sel]["test"]["confusion"])
    n = m / m.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(W, 4.6))
    im = ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, vn(int(m[i, j])), ha="center", va="center", fontsize=12,
                    color="white" if n[i, j] > 0.55 else "#111827")
    ax.set_xticks(range(5), SHORT_VI, rotation=30, ha="right")
    ax.set_yticks(range(5), SHORT_VI)
    ax.set(xlabel=f"Dự đoán ({VI[sel]})", ylabel="Thực tế")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Tỉ lệ theo hàng")
    save(fig, "retina_confusion")


def fig_cam():
    X, y = np.load(RDIR / "test_x.npy", mmap_mode="r"), np.load(RDIR / "test_y.npy")
    rng = np.random.default_rng(7)
    picks = [int(rng.choice(np.where(y == k)[0])) for k in (0, 2, 3, 4)]
    fig, axs = plt.subplots(2, 4, figsize=(W, 3.5))
    for c, i in enumerate(picks):
        buf = io.BytesIO()
        Image.fromarray(np.array(X[i])).save(buf, format="PNG")
        g = grade(buf.getvalue())
        axs[0, c].imshow(X[i])
        axs[0, c].set_title(f"Thực: {y[i]}", fontsize=12)
        axs[1, c].imshow(Image.open(io.BytesIO(base64.b64decode(g["gradcam_png_b64"]))))
        axs[1, c].set_title(f"Dự đoán: {g['grade']}", fontsize=12)
        axs[0, c].axis("off")
        axs[1, c].axis("off")
    save(fig, "retina_gradcam")


def main():
    s, r = load_json("data_retina"), load_json("retina_results")
    fig_grid()
    fig_counts(s)
    fig_curves(r)
    fig_conf(r)
    fig_cam()


if __name__ == "__main__":
    main()
