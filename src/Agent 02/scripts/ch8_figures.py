"""Hình Chương 8: đường học, ma trận nhầm lẫn và mô phỏng lỗi kho (từ outputs/metrics/ch8_shop.json)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ag2.common import load_json  # noqa: E402
from ag2.plot_style import C, W, W2, plt, save, vn  # noqa: E402

LABELS = ["Tìm SP", "So sánh", "Tồn kho", "Giá/KM", "Chào hỏi", "Khác"]


def learning(d):
    h = d["train"]["history"]
    ep = [x["epoch"] for x in h]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.6))
    a.plot(ep, [x["train_loss"] for x in h], color=C["blue"], label="train")
    a.plot(ep, [x["val_loss"] for x in h], color=C["orange"], label="val")
    a.set(title="Hàm mất mát (cross-entropy)", xlabel="epoch")
    b.plot(ep, [100 * x["train_acc"] for x in h], color=C["blue"], label="train")
    b.plot(ep, [100 * x["val_acc"] for x in h], color=C["orange"], label="val")
    best = d["train"]["best_epoch"]
    b.axvline(best, color=C["red"], ls="--", lw=1.4)
    b.text(best - 0.6, 76, f"epoch tốt nhất = {best}", color=C["red"], fontsize=10.5, ha="right")
    b.set(title="Độ chính xác (%)", xlabel="epoch", ylim=(30, 102))
    a.legend(), b.legend(loc="lower right")
    save(fig, "ch8_learning")


def confusion(d):
    cm = np.array(d["test"]["conf"]["rows"])
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10.5,
                    color="white" if cm[i, j] > 0.6 * cm[i].sum() else "black")
    ax.set_xticks(range(6), LABELS, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(6), LABELS, fontsize=10)
    ax.set(xlabel="dự đoán", ylabel="thực tế")
    ax.grid(False)
    save(fig, "ch8_confusion")


def faults(d):
    rows = list(d["faults"].values())
    x = np.arange(len(rows))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 1.85))
    systems = (("mas", "MAS + Coordinator", C["green"]), ("pipe", "Pipeline đơn khối", C["red"]))
    for k, (sys_key, name, col) in enumerate(systems):
        pos = x + (k - 0.5) * 0.38
        bars = ax.bar(pos, [100 * r[sys_key]["completion"] for r in rows], 0.36, color=col, label=name)
        for bar, r in zip(bars, rows):
            if bar.get_height() < 99.95:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, vn(bar.get_height(), 1),
                        ha="center", fontsize=9.5)
            if r[sys_key]["unsafe"]:
                ax.text(bar.get_x() + bar.get_width() / 2, 88.6, f"{r[sys_key]['unsafe']} sai",
                        ha="center", fontsize=10, color="white", fontweight="bold")
        bx.bar(pos, [r[sys_key]["lat_mean"] for r in rows], 0.36, color=col, label=name)
        bx.scatter(pos, [r[sys_key]["lat_p95"] for r in rows], marker="v", s=55, color="black", zorder=3)
    ticks = [f"lỗi {int(100 * r['rate'])}%" for r in rows]
    ax.set_xticks(x, ticks)
    ax.set(ylim=(88, 101.8), title="Hoàn thành đúng (%)")
    bx.set_xticks(x, ticks)
    bx.set(yscale="log", title="Trễ (ms): cột = TB, ▼ = p95")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2)
    save(fig, "ch8_faults")


if __name__ == "__main__":
    data = load_json("ch8_shop")
    learning(data)
    confusion(data)
    faults(data)
