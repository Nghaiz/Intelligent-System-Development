"""Vẽ hình dữ liệu Chương 3 vào Report/TieuLuan Final/figures/ch3_*.

    python scripts/ch3_figures.py [--part data results viz]

data    : lưới ảnh, phân bố lớp, histogram kênh màu
results : đường cong học, so sánh mô hình, ma trận nhầm lẫn, F1 theo lớp, ảnh đoán sai
viz     : Grad-CAM ba kiến trúc (PyTorch trên CPU)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.ch3_common import ARCHS, ARCH_NAME, DS_IDS, MODELS, load_proba, load_result  # noqa: E402
from tlf.ch3_data import DATASETS, labels, load, net_input, splits  # noqa: E402
from tlf.common import limit_threads  # noqa: E402
from tlf.plot_style import C, W, W2, plt, save, vn  # noqa: E402

DS_NAME = {d: DATASETS[d]["name"] for d in DS_IDS}
ACOL = {"custom": C["gray"], "vgg11": C["purple"], "resnet18": C["blue"], "mobilenetv2": C["orange"]}
RGB = ("#DC2626", "#16A34A", "#2563EB")
CAM_ARCHS = ["custom", "resnet18", "mobilenetv2"]


def have(ds):
    """Các kiến trúc đã có kết quả; làn PathMNIST có hạn GPU nên có thể thiếu VGG-11."""
    return [a for a in ARCHS if load_result(f"{ds}__{a}")]


def best_arch(ds):
    return max(have(ds), key=lambda a: load_result(f"{ds}__{a}")["f1_macro"])


def bare(ax):
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_grid(ds, rows=3):
    """Lưới ảnh thật: mỗi cột một lớp, lấy ngẫu nhiên có hạt giống từ tập train."""
    x, y = load(ds, "train")
    names, rng = labels(ds), np.random.default_rng(3)
    k = len(names)
    cell, head = W / k, 0.35 if ds == "svhn" else 0.95
    fig, ax = plt.subplots(rows, k, figsize=(W, cell * rows + head), squeeze=False)
    for c in range(k):
        idx = rng.choice(np.flatnonzero(y == c), rows, replace=False)
        for r in range(rows):
            ax[r, c].imshow(x[idx[r]], interpolation="nearest")
            bare(ax[r, c])
        size = 13 if ds == "svhn" else 11
        ax[0, c].set_title(names[c], fontsize=size, fontweight="normal", rotation=40 if ds != "svhn" else 0,
                           ha="left" if ds != "svhn" else "center", pad=4)
    fig.subplots_adjust(left=0, right=0.92 if ds != "svhn" else 1, bottom=0, wspace=0.04, hspace=0.04,
                        top=cell * rows / (cell * rows + head))
    save(fig, f"ch3_{ds}_grid", "png", 220)


def fig_dist(ds):
    """Số ảnh mỗi lớp, xếp chồng ba phần train, val, test."""
    names = labels(ds)
    left = np.zeros(len(names))
    fig, ax = plt.subplots(figsize=(W2, 3.5))
    yy = np.arange(len(names))
    for split, col in zip(splits(ds), (C["blue"], C["orange"], C["green"])):
        n = np.bincount(load(ds, split)[1], minlength=len(names))
        ax.barh(yy, n, left=left, color=col, label=split)
        left += n
    ax.set_yticks(yy, names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Số ảnh", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: vn(v / 1000) + "k" if v else "0"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, f"ch3_{ds}_dist")


def fig_hist(ds):
    """Histogram cường độ ba kênh trên tập train; chuẩn hoá theo tổng số điểm ảnh của từng kênh."""
    x, _ = load(ds, "train")
    a = np.asarray(x[:: max(1, len(x) // 5000)])
    fig, ax = plt.subplots(figsize=(W2, 3.5))
    for c, (col, nm) in enumerate(zip(RGB, "RGB")):
        h = np.bincount(a[..., c].ravel(), minlength=256) / a[..., c].size
        ax.plot(h, color=col, label=f"{nm} (TB {vn(a[..., c].mean(), 1)})", lw=1.8)
    ax.set_xlabel("Cường độ điểm ảnh", fontsize=12)
    ax.set_ylabel("Tỉ lệ", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xlim(0, 255)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: vn(v, 3)))
    ax.legend(fontsize=10)
    save(fig, f"ch3_{ds}_hist")


def fig_curves():
    """Hàng trên: val accuracy theo epoch. Hàng dưới: loss train (nét đứt) và val (nét liền)."""
    fig, ax = plt.subplots(2, 3, figsize=(W, 4.9), sharex="col")
    for j, ds in enumerate(DS_IDS):
        for a in have(ds):
            h = load_result(f"{ds}__{a}")["history"]
            e = np.arange(1, len(h["loss"]) + 1)
            ax[0, j].plot(e, h["val_acc"], color=ACOL[a], label=ARCH_NAME[a])
            ax[1, j].plot(e, h["val_loss"], color=ACOL[a])
            ax[1, j].plot(e, h["loss"], color=ACOL[a], ls="--", lw=1.3)
        ax[0, j].set_title(DS_NAME[ds], fontsize=13)
        ax[1, j].set_xlabel("Epoch", fontsize=12)
        for r in (0, 1):
            ax[r, j].tick_params(labelsize=11)
            ax[r, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: vn(v, 2)))
    ax[0, 0].set_ylabel("Val accuracy", fontsize=12)
    ax[1, 0].set_ylabel("Loss", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, "ch3_curves")


def fig_compare():
    """F1 macro theo bộ dữ liệu cho baseline softmax và các kiến trúc."""
    fig, ax = plt.subplots(figsize=(W, 3.1))
    groups = [("baseline_softmax", "Softmax trên điểm ảnh", C["sky"])] + \
        [(a, ARCH_NAME[a], ACOL[a]) for a in ARCHS]
    w = 0.16
    for i, (mid, name, col) in enumerate(groups):
        rs = [(k, r) for k, d in enumerate(DS_IDS) if (r := load_result(f"{d}__{mid}"))]
        v = [r["f1_macro"] for _, r in rs]
        bars = ax.bar([k + (i - 2) * w for k, _ in rs], v, w, color=col, label=name)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + w / 2, val + 0.012, vn(val, 3), ha="center", fontsize=9, rotation=90)
    ax.set_xticks(range(3), [DS_NAME[d] for d in DS_IDS])
    ax.set_ylim(0, 1.6)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: vn(v, 2)))
    ax.set_ylabel("F1 macro trên test")
    ax.legend(ncol=3, fontsize=10, loc="upper center")
    ax.grid(axis="x", visible=False)
    save(fig, "ch3_compare")


def fig_tradeoff():
    """Đánh đổi giữa số tham số và F1 macro; trục ngang thang log."""
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    for ds, mk in zip(DS_IDS, "os^"):
        for a in have(ds):
            r = load_result(f"{ds}__{a}")
            ax.scatter(r["params"], r["f1_macro"], s=70, color=ACOL[a], marker=mk,
                       edgecolor="white", zorder=3)
        ax.scatter([], [], color="#444", marker=mk, label=DS_NAME[ds])
    ax.set_xscale("log")
    ax.set_xlabel("Số tham số", fontsize=12)
    ax.set_ylabel("F1 macro", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: vn(v, 2)))
    ax.legend(fontsize=10)
    save(fig, "ch3_tradeoff")


def fig_time():
    """Thời gian huấn luyện mỗi epoch của từng kiến trúc; số ảnh train mỗi bộ khác nhau."""
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    for i, a in enumerate(ARCHS):
        rs = [(k, r) for k, d in enumerate(DS_IDS) if (r := load_result(f"{d}__{a}"))]
        ax.bar([k + (i - 1.5) * 0.2 for k, _ in rs], [r["seconds"] / r["epochs"] for _, r in rs], 0.2,
               color=ACOL[a], label=ARCH_NAME[a])
    ax.set_xticks(range(3), ["SVHN", "EuroSAT", "PathMNIST"], fontsize=10)
    ax.set_ylabel("Giây mỗi epoch", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "ch3_time")


def fig_cm(ds):
    """Ma trận nhầm lẫn của kiến trúc tốt nhất, chuẩn hoá theo hàng (lớp thật), ô ghi phần trăm."""
    arch = best_arch(ds)
    cm = np.array(load_result(f"{ds}__{arch}")["cm"], float)
    cm = 100 * cm / cm.sum(1, keepdims=True)
    k, names = len(cm), [str(i) for i in range(len(cm))]
    fig, ax = plt.subplots(figsize=(W2, W2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            if cm[i, j] >= 0.5:
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=9.5,
                        color="white" if cm[i, j] > 60 else "#123A75")
    ax.set_xticks(range(k), names, fontsize=11)
    ax.set_yticks(range(k), names, fontsize=11)
    ax.set_xlabel("Lớp dự đoán", fontsize=12)
    ax.set_ylabel("Lớp thật", fontsize=12)
    ax.grid(False)
    save(fig, f"ch3_cm_{ds}")


def fig_f1_class():
    """F1 từng lớp của PathMNIST: nơi ba kiến trúc lệch nhau rõ nhất do tập test khác trung tâm."""
    ds, names = "pathmnist", labels("pathmnist")
    fig, ax = plt.subplots(figsize=(W2, W2))
    yy = np.arange(len(names))
    for i, a in enumerate(have(ds)):
        ax.barh(yy + (i - 1.5) * 0.2, load_result(f"{ds}__{a}")["f1_per_class"], 0.2, color=ACOL[a],
                label=ARCH_NAME[a])
    ax.set_yticks(yy, [f"{i}. {n}" for i, n in enumerate(names)], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 trên test", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: vn(v, 1)))
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.55, 1.08))
    save(fig, "ch3_f1_class")


def fig_wrong(per_ds=6):
    """Mỗi bộ một hàng ảnh test bị kiến trúc tốt nhất đoán sai với độ tin cao nhất."""
    fig, ax = plt.subplots(3, per_ds, figsize=(W, 3.9))
    for r, ds in enumerate(DS_IDS):
        x, y = load(ds, "test")
        p = load_proba(f"{ds}__{best_arch(ds)}")
        wrong = np.flatnonzero(p.argmax(1) != y)
        pick = wrong[np.argsort(-p[wrong].max(1))[:per_ds]]
        for c, i in enumerate(pick):
            ax[r, c].imshow(x[i], interpolation="nearest")
            bare(ax[r, c])
            hat = int(p[i].argmax())
            ax[r, c].set_title(f"{y[i]}→{hat} ({vn(p[i, hat], 2)})", fontsize=11, fontweight="normal", pad=2)
        ax[r, 0].set_ylabel(DS_NAME[ds].replace(" RGB", ""), fontsize=10)
    fig.subplots_adjust(wspace=0.08, hspace=0.3)
    save(fig, "ch3_wrong", "png", 220)


def cam_row(ds, axes, pick):
    """Một hàng hình: ảnh gốc rồi Grad-CAM từng kiến trúc, cho mỗi ảnh đã chọn."""
    import torch
    from tlf.ch3_models import build
    from tlf.ch3_torch import GradCAM, checkpoint_model
    from tlf.serve_ch3 import heat_overlay
    x, _ = load(ds, "test")
    k = len(labels(ds))
    for s, i in enumerate(pick):
        axes[(1 + len(CAM_ARCHS)) * s].imshow(x[i], interpolation="nearest")
    for a_i, a in enumerate(CAM_ARCHS):
        m, mean, std = checkpoint_model(MODELS / f"{ds}__{a}.pt", lambda kw: build(a, k, **kw))
        gc = GradCAM(m)
        for s, i in enumerate(pick):
            raw = torch.from_numpy(net_input(x[i:i + 1]).copy()).permute(0, 3, 1, 2).float()
            xb = (raw / 255 - mean) / std
            cam = gc(xb)[0][0]
            if x.shape[1] == 28:
                cam = cam[2:-2, 2:-2]
            axes[(1 + len(CAM_ARCHS)) * s + 1 + a_i].imshow(heat_overlay(np.array(x[i]), cam))


def fig_gradcam(samples=2):
    """Mỗi hàng một bộ: hai ảnh test được đoán đúng, mỗi ảnh kèm Grad-CAM của ba kiến trúc."""
    step = 1 + len(CAM_ARCHS)
    fig, ax = plt.subplots(3, step * samples, figsize=(W, 2.75))
    heads = ["Ảnh", "Cơ bản", "ResNet", "Mobile"]
    for r, ds in enumerate(DS_IDS):
        y = load(ds, "test")[1]
        hit = np.flatnonzero(load_proba(f"{ds}__{best_arch(ds)}").argmax(1) == y)
        cam_row(ds, ax[r], np.random.default_rng(5).choice(hit, samples, replace=False))
    for c in range(step * samples):
        ax[0, c].set_title(heads[c % step], fontsize=10, fontweight="normal", pad=2)
    for a in ax.ravel():
        bare(a)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save(fig, "ch3_gradcam", "png", 220)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="*", default=["data", "results", "viz"])
    args = ap.parse_args()
    limit_threads(2)
    if "data" in args.part:
        for ds in DS_IDS:
            if (Path("data/ch3") / ds / "y_test.npy").exists():
                fig_grid(ds), fig_dist(ds), fig_hist(ds)
    if "results" in args.part:
        fig_curves(), fig_compare(), fig_tradeoff(), fig_time(), fig_wrong(), fig_f1_class()
        for ds in DS_IDS:
            fig_cm(ds)
    if "viz" in args.part:
        fig_gradcam()


if __name__ == "__main__":
    main()
