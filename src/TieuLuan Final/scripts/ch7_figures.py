"""Chương 7: vẽ mọi hình dữ liệu từ outputs/metrics/ch7_*.json."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.common import limit_threads, load_json  # noqa: E402
from tlf.plot_style import C, SEQ, W, plt, save, vn_axis  # noqa: E402

KEYS = (("cora", "Cora"), ("photo", "Amazon Photo"), ("elliptic", "Elliptic"))
MODELS = (("lp", "LP"), ("mlp", "MLP"), ("gcn", "GCN"), ("sage", "SAGE"), ("gat", "GAT"))
COLOR = {"lp": C["gray"], "mlp": C["orange"], "gcn": C["blue"], "sage": C["green"], "gat": C["purple"]}


def fig_data():
    d = load_json("ch7_data")
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.3))
    for j, (key, name) in enumerate(KEYS):
        st = d[key]
        ax = axes[0, j]
        counts = st["class_counts"]
        ax.bar(range(len(counts)), counts, color=SEQ[:len(counts)])
        ax.set_title(name, fontsize=12)
        ax.set_xticks(range(len(counts)))
        ax.tick_params(labelsize=10)
        if key == "elliptic":
            ax.set_xticklabels(["licit", "illicit"])
        ax.grid(axis="x", visible=False)
        vn_axis(ax)
        ax = axes[1, j]
        rows = st["degree_curve"]["rows"]
        ax.loglog([r["deg"] for r in rows], [r["count"] for r in rows], "o-", color=C["teal"], ms=3.5, lw=1.4)
        ax.tick_params(labelsize=10)
        ax.set_xlabel("Bậc", fontsize=11)
    axes[0, 0].set_ylabel("Số nút", fontsize=11)
    axes[1, 0].set_ylabel("Số nút", fontsize=11)
    fig.tight_layout()
    save(fig, "ch7_data")


def fig_compare():
    node = load_json("ch7_node")
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.7), sharey=True)
    for ax, (key, name) in zip(axes, KEYS):
        e = node[key]
        v = [e[k]["f1_macro"] for k, _ in MODELS]
        s = [e[k]["f1_macro_std"] for k, _ in MODELS]
        ax.bar(range(5), v, yerr=s, capsize=2.5, color=[COLOR[k] for k, _ in MODELS])
        ax.set_xticks(range(5), [n for _, n in MODELS], rotation=45, fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
        vn_axis(ax, d=1)
    axes[0].set_ylabel("F1 macro", fontsize=11)
    fig.tight_layout()
    save(fig, "ch7_compare")


def fig_depth():
    r = load_json("ch7_reused")["cora_depth"]
    plain = [x for x in r["rows"] if not x["residual"]]
    res = [x for x in r["rows"] if x["residual"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.7))
    a.errorbar([x["layers"] for x in plain], [x["test_mean"] for x in plain], [x["test_std"] for x in plain],
               marker="o", ms=4, color=C["blue"], label="GCN thường", capsize=2)
    a.errorbar([x["layers"] for x in res], [x["test_mean"] for x in res], [x["test_std"] for x in res],
               marker="s", ms=4, color=C["orange"], label="Có nối tắt", capsize=2)
    a.set_xlabel("Số lớp", fontsize=11)
    a.set_ylabel("Độ đúng test", fontsize=11)
    a.legend(fontsize=10)
    vn_axis(a, d=1)
    p = [x for x in r["propagation"]["rows"]]
    b.plot([x["k"] + 1 for x in p], [x["mad"] for x in p], "o-", ms=4, color=C["purple"])
    b.set_xscale("log", base=2)
    b.set_xticks([x["k"] + 1 for x in p], [str(x["k"]) for x in p], fontsize=10)
    b.set_xlabel("Số lần nhân $\\hat{A}$", fontsize=11)
    b.set_ylabel("MAD của $\\hat{A}^k X$", fontsize=11)
    vn_axis(b, d=1)
    fig.tight_layout()
    save(fig, "ch7_depth")


def scatter(ax, z, y, title, sil):
    z = np.array(z)
    y = np.array(y)
    for i, c in enumerate(np.unique(y)):
        m = y == c
        ax.scatter(z[m, 0], z[m, 1], s=2.2, color=SEQ[i], rasterized=True, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"{title}\nsilhouette {sil:.2f}".replace(".", ","), fontsize=10.5)


def fig_tsne():
    re = load_json("ch7_reused")["cora_tsne"]
    t = load_json("ch7_tsne")
    groups = [("Cora", re), ("Photo", t["photo"]), ("Elliptic", t["elliptic"])]
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.2))
    for j, (name, g) in enumerate(groups):
        y = g["labels"]["series"]
        scatter(axes[0, j], g["raw"]["series"], y, f"{name}: đặc trưng thô", g["raw"]["silhouette"])
        scatter(axes[1, j], g["gcn"]["series"], y, f"{name}: ẩn GCN", g["gcn"]["silhouette"])
    fig.tight_layout()
    save(fig, "ch7_tsne", ext="pdf", dpi=220)


def fig_elliptic():
    node = load_json("ch7_node")["elliptic"]
    time = load_json("ch7_data")["elliptic_time"]["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.7))
    a.bar([r["step"] for r in time], [r["illicit"] for r in time], color=C["red"], width=0.8)
    a.axvline(29.5, color=C["gray"], ls="--", lw=1.2)
    a.axvline(34.5, color=C["gray"], ls="--", lw=1.2)
    a.set_xlabel("Bước thời gian", fontsize=11)
    a.set_ylabel("Số giao dịch illicit", fontsize=11)
    a.grid(axis="x", visible=False)
    rows = node["steps"]["rows"]
    for k, n in MODELS[1:]:
        b.plot([r["step"] for r in rows], [r[k] for r in rows], "o-", ms=3, color=COLOR[k], label=n, lw=1.5)
    b.set_xlabel("Bước thời gian (test)", fontsize=11)
    b.set_ylabel("F1 illicit", fontsize=11)
    b.legend(fontsize=9, ncol=2)
    vn_axis(b, d=1)
    fig.tight_layout()
    save(fig, "ch7_elliptic")


def fig_err():
    node = load_json("ch7_node")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for ax, (key, name) in zip(axes, KEYS[:2]):
        rows = [r for r in node[key]["homophily_bins"]["rows"] if r["n"] > 0]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r["mlp"] for r in rows], 0.4, color=COLOR["mlp"], label="MLP")
        ax.bar(x + 0.2, [r["gcn"] for r in rows], 0.4, color=COLOR["gcn"], label="GCN")
        ax.set_xticks(x, [f"{r['bin']}\n(n={r['n']})" for r in rows], fontsize=9)
        ax.set_xlabel("% láng giềng cùng nhãn", fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.grid(axis="x", visible=False)
        vn_axis(ax, d=1)
    axes[0].set_ylabel("Độ đúng test", fontsize=11)
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    save(fig, "ch7_err")


def main():
    limit_threads(2)
    for f in (fig_data, fig_compare, fig_depth, fig_tsne, fig_elliptic, fig_err):
        f()


if __name__ == "__main__":
    main()
