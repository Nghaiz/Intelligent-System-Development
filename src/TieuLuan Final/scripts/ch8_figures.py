"""Ch.8: vẽ lại các hình của bài Agent 02 ở cỡ chữ của tiểu luận, từ data/ch8/raw (không chạy lại thí nghiệm)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from tlf.common import DATA, load_json  # noqa: E402
from tlf.plot_style import C, SEQ, W, plt, save, vn  # noqa: E402

RAW = DATA / "ch8" / "raw"
WIN = 100
H = 2.75
OPT = 35.0


def raw(exp):
    return json.loads((RAW / f"{exp}.json").read_text(encoding="utf-8"))


def smooth(x, k=WIN):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def fig_learning():
    """Trái: đường học lượt tham chiếu. Phải: thưởng đánh giá của RL và luật trên mười seed."""
    z = np.load(RAW / "baseline.npz")
    seeds = load_json("ch8_e")["seeds"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, H))
    ep = np.arange(len(smooth(z["reward"][:, 0]))) + WIN
    a1.plot(ep, smooth(z["reward"][:, 0]), color=C["blue"], label="Q-learning")
    a1.plot(ep, smooth(z["reward"][:, 1]), color=C["orange"], label="Luật")
    a1.set_xlabel("episode")
    a1.set_title("Thưởng huấn luyện / bước")
    a1.set_xticks([0, 2500, 5000], ["0", "2.500", "5.000"])
    a1.set_ylim(30.2, 38)
    a1.legend(loc="upper center", ncol=2, columnspacing=0.9, handlelength=1.2, fontsize=11.5)
    x = np.arange(len(seeds))
    a2.bar(x - 0.2, [s["eval_mean"] for s in seeds], 0.4, color=C["blue"], label="Q-learning")
    a2.bar(x + 0.2, [s["rule_mean"] for s in seeds], 0.4, color=C["orange"], label="Luật")
    a2.set_xticks(x[::3], [s["seed"] for s in seeds][::3])
    a2.set_ylim(26, 36.5)
    a2.set_xlabel("seed")
    a2.set_title("Thưởng đánh giá / bước")
    fig.tight_layout(w_pad=1.4)
    save(fig, "ch8_learning")


def seed_bars(ax, groups, labels, title):
    """Cột trung bình ± SD trên 10 seed, chấm đen là từng seed, nét đỏ là mức tối ưu 35."""
    vals = [np.array([r["eval_mean"] for r in g]) for g in groups]
    ax.bar(range(len(vals)), [v.mean() for v in vals], yerr=[v.std(ddof=1) for v in vals], color=SEQ[:len(vals)],
           capsize=4, alpha=0.85)
    for i, v in enumerate(vals):
        ax.scatter(np.full(len(v), i) + np.linspace(-0.22, 0.22, len(v)), v, s=11, color="black", zorder=3)
    ax.axhline(OPT, color=C["red"], ls=":", lw=1.5)
    ax.set_xticks(range(len(vals)), labels)
    ax.set_ylim(min(v.min() for v in vals) - 1, 37)
    ax.set_yticks([25, 30, 35] if min(v.min() for v in vals) < 27 else [28, 32, 36])
    ax.set_title(title)


def fig_sweeps():
    """Thí nghiệm C và D: thưởng đánh giá theo tốc độ học và theo lịch khám phá."""
    c, d = raw("C"), raw("D")
    alpha = ("0.01", "0.10", "0.50")
    sched = ("khong", "thap", "cao")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, H))
    seed_bars(a1, [[r for r in c if r["key"] == k] for k in alpha], ["0,01", "0,10", "0,50"], "Tốc độ học α")
    groups = [[r for r in d if r["key"] == k] for k in sched]
    groups.insert(2, [r for r in c if r["key"] == "0.10"])
    seed_bars(a2, groups, ["không", "thấp", "cơ sở", "cao"], "Lịch khám phá ε")
    fig.tight_layout(w_pad=1.4)
    save(fig, "ch8_sweeps")


def fig_delta():
    """Thí nghiệm F: thưởng của RL, luật, tối ưu và tỉ lệ giảm giá theo mức giảm δ."""
    f = load_json("ch8_f")
    keys = sorted(f)
    x = [int(k) for k in keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, H))
    a1.errorbar(x, [f[k]["eval_mean"] for k in keys], yerr=[f[k]["eval_sd"] for k in keys], marker="o", ms=5,
                color=C["blue"], capsize=4, label="Q-learning")
    a1.plot(x, [f[k]["exact_rule"] for k in keys], marker="s", ms=5, color=C["orange"], label="Luật")
    a1.plot(x, [f[k]["optimal"] for k in keys], ls="--", color=C["red"], label="Tối ưu")
    a1.set_xlabel("mức giảm giá δ (%)")
    a1.set_title("Thưởng / bước")
    a1.set_ylim(28, 38)
    a1.set_xticks([5, 10, 20, 30])
    a1.legend(loc="lower left", handlelength=1.4, labelspacing=0.15, fontsize=11.5)
    a2.bar(x, [100 * f[k]["eval_disc"] for k in keys], width=3.2, color=C["teal"])
    a2.set_xlabel("mức giảm giá δ (%)")
    a2.set_title("Tỉ lệ giảm giá của RL (%)")
    fig.tight_layout(w_pad=1.4)
    save(fig, "ch8_delta")


def fig_regret():
    """Câu 10: regret của ba tác tử trong năm kịch bản."""
    q = load_json("ch8_q10")
    scen = ["tham_chieu", "giam_5", "an_mieng", "luon_giam", "ben_vung"]
    lab = ["Tham chiếu", "δ = 5%", "Ăn miếng\ntrả miếng", "Luôn\ngiảm giá", "Nhu cầu\nbền"]
    ags = [("q", "Q-learning (d, c)"), ("q_ext", "Q-learning mở rộng"), ("model", "Dựa trên mô hình")]
    x = np.arange(len(scen))
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for j, (ag, name) in enumerate(ags):
        vals = [q[s][ag]["regret"] for s in scen]
        ax.bar(x + (j - 1) * 0.27, vals, 0.27, color=SEQ[j], label=name)
        if ag == "model":   # regret bằng 0 nên cột không có chiều cao: ghi số để không bị đọc là thiếu dữ liệu
            for xi, v in zip(x, vals):
                ax.text(xi + 0.27, 0.03, vn(max(v, 0.0), 0), ha="center", va="bottom", color=SEQ[j], fontweight="bold")
    ax.set_xticks(x, lab)
    ax.set_ylabel("regret / bước")
    ax.yaxis.set_major_formatter(lambda v, _: vn(v, 1))
    ax.set_ylim(0, 1.65)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="upper center", ncol=3, fontsize=11.5, columnspacing=1.0, handlelength=1.3)
    save(fig, "ch8_regret")


def main():
    fig_learning()
    fig_sweeps()
    fig_delta()
    fig_regret()


if __name__ == "__main__":
    main()
