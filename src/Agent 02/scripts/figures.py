"""Vẽ mọi hình dữ liệu của Ch.4, Ch.6, Ch.7 từ outputs/raw và outputs/metrics (xuất PDF vector)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from ag2.common import OUT, load_json  # noqa: E402
from ag2.market import MarketConfig, expected_reward  # noqa: E402
from ag2.plot_style import C, H1, SEQ, W, plt, save  # noqa: E402

RAW = OUT / "raw"
WIN = 100
EP_AXIS = np.arange(WIN // 2, 5000, WIN)


def raw(exp):
    return json.loads((RAW / f"{exp}.json").read_text(encoding="utf-8"))


def smooth(x, k=WIN):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def band(ax, rs, key, color, label):
    """Đường trung bình ± 1 độ lệch chuẩn qua các seed (cửa sổ 100 episode)."""
    m = np.array([r[key] for r in rs])
    mu, sd = m.mean(axis=0), m.std(axis=0, ddof=1)
    ax.plot(EP_AXIS, mu, color=color, label=label)
    ax.fill_between(EP_AXIS, mu - sd, mu + sd, color=color, alpha=0.15)


def fig_ch4_delta():
    """Thưởng kỳ vọng một bước theo mức giảm giá δ cho từng trạng thái (d, a đối thủ)."""
    deltas = np.linspace(0.0, 0.4, 81)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), sharex=True)
    for ax, d in zip(axes, (0, 1)):
        for b, col in ((0, C["blue"]), (1, C["red"])):
            keep = [expected_reward(MarketConfig().with_(discount_rate=x), d, 0, b) for x in deltas]
            disc = [expected_reward(MarketConfig().with_(discount_rate=x), d, 1, b) for x in deltas]
            ax.plot(deltas, keep, color=col, ls="--", label=f"giữ giá, đối thủ a={b}")
            ax.plot(deltas, disc, color=col, label=f"giảm giá, đối thủ a={b}")
        ax.axvline(0.20, color=C["gray"], lw=1.2, ls=":")
        ax.set_title(f"Nhu cầu {'cao' if d else 'thấp'} (d = {d})")
        ax.set_xlabel("mức giảm giá δ")
    axes[0].set_ylabel("E[r] mỗi bước")
    axes[0].legend(fontsize=9, loc="lower left")
    save(fig, "ch4_reward_delta")


def fig_ch6_baseline():
    z = np.load(RAW / "baseline.npz")
    b = load_json("ch6_b")
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ep = np.arange(len(smooth(z["reward"][:, 0]))) + WIN
    ax.plot(z["reward"][:, 0], color=C["blue"], alpha=0.12, lw=0.6)
    ax.plot(ep, smooth(z["reward"][:, 0]), color=C["blue"], label="Q-learning (TB trượt 100 episode)")
    ax.plot(ep, smooth(z["reward"][:, 1]), color=C["orange"], label="Tác tử luật (TB trượt 100 episode)")
    ax.set_xlabel("episode")
    ax.set_ylabel("thưởng TB mỗi bước")
    ax2 = ax.twinx()
    ax2.plot(z["epsilon"], color=C["purple"], ls="--", lw=1.6, label="ε")
    ax2.set_ylabel("ε", color=C["purple"])
    ax2.grid(False)
    ax.legend(loc="lower right")
    save(fig, "ch6_b_curve")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75))
    a1.plot(b["exact_curve"]["ep"], b["exact_curve"]["v"], color=C["green"], marker="o", ms=2.5, lw=1.4)
    a1.axhline(35.0, color=C["red"], ls=":", lw=1.4, label="tối ưu 35")
    a1.set_title("Giá trị chính xác $V^\\pi$")
    a1.set_xlabel("episode")
    a1.legend(loc="lower left")
    a2.plot(np.arange(len(smooth(z["discount"][:, 0]))) + WIN, smooth(z["discount"][:, 0]), color=C["red"],
            label="Q-learning")
    a2.plot(np.arange(len(smooth(z["discount"][:, 1]))) + WIN, smooth(z["discount"][:, 1]), color=C["orange"],
            label="Luật")
    a2.set_title("Tỉ lệ giảm giá (HL)")
    a2.set_xlabel("episode")
    a2.legend()
    save(fig, "ch6_b_exact")

    fig, axes = plt.subplots(2, 2, figsize=(W, 3.5), sharex=True)
    names = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
    for k, ax in enumerate(axes.flat):
        ax.plot(z["q"][:, 2 * k], color=C["blue"], label="Q(s, giữ giá)")
        ax.plot(z["q"][:, 2 * k + 1], color=C["red"], label="Q(s, giảm giá)")
        ax.set_title(f"s = (d, c) = {names[k]}")
    for ax in axes[1]:
        ax.set_xlabel("episode")
    axes[0, 0].legend(fontsize=9.5)
    save(fig, "ch6_b_qtraj")


def fig_sweep(exp, keys, labels, name, extra=None):
    """Đường học (TB ± SD 10 seed) và thưởng đánh giá theo cấu hình."""
    g = {}
    for r in raw(exp) + (extra or []):
        g.setdefault(r["key"], []).append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw={"width_ratios": [1.6, 1]})
    for k, lab, col in zip(keys, labels, SEQ):
        band(a1, g[k], "curve", col, lab)
    a1.set_xlabel("episode")
    a1.set_ylabel("thưởng huấn luyện / bước")
    a1.legend(fontsize=9.5, loc="lower right")
    ev = [np.array([r["eval_mean"] for r in g[k]]) for k in keys]
    a2.bar(range(len(keys)), [e.mean() for e in ev], yerr=[e.std(ddof=1) for e in ev], color=SEQ[:len(keys)],
           capsize=4, alpha=0.85)
    for i, e in enumerate(ev):
        a2.scatter(np.full(len(e), i) + np.linspace(-0.2, 0.2, len(e)), e, s=10, color="black", zorder=3)
    a2.axhline(35.0, color=C["red"], ls=":", lw=1.4)
    a2.set_xticks(range(len(keys)), labels, rotation=20)
    a2.set_ylim(min(e.min() for e in ev) - 1, 36.5)
    a2.set_title("Thưởng đánh giá (ε = 0)")
    save(fig, name)


def fig_ch6_e():
    e = load_json("ch6_e")["seeds"]
    x = np.arange(len(e))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75))
    a1.bar(x - 0.2, [s["eval_mean"] for s in e], 0.4, color=C["blue"], label="Q-learning")
    a1.bar(x + 0.2, [s["rule_mean"] for s in e], 0.4, color=C["orange"], label="Luật")
    a1.set_xticks(x, [s["seed"] for s in e], fontsize=8.5)
    a1.set_ylim(26, 36.5)
    a1.set_xlabel("seed")
    a1.set_title("Thưởng đánh giá")
    a1.legend(fontsize=9, loc="lower left", ncol=2)
    a2.errorbar(x, [s["paired_diff"] for s in e], yerr=[1.96 * s["paired_se"] for s in e], fmt="o",
                color=C["purple"], capsize=4)
    a2.axhline(0, color=C["gray"], lw=1.2)
    a2.set_xticks(x, [s["seed"] for s in e], fontsize=8.5)
    a2.set_xlabel("seed")
    a2.set_title("RL − luật (KTC 95%)")
    save(fig, "ch6_e_seeds")


def fig_ch6_f():
    f = load_json("ch6_f")
    keys = sorted(f)
    x = [int(k) / 100 for k in keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75))
    a1.errorbar(x, [f[k]["eval_mean"] for k in keys], yerr=[f[k]["eval_sd"] for k in keys], marker="o",
                color=C["blue"], capsize=4, label="Q-learning (đánh giá)")
    a1.plot(x, [f[k]["exact_rule"] for k in keys], marker="s", color=C["orange"], label="Luật (chính xác)")
    a1.plot(x, [f[k]["optimal"] for k in keys], ls="--", color=C["red"], label="Tối ưu")
    a1.set_xlabel("mức giảm giá δ")
    a1.set_ylabel("thưởng / bước")
    a1.legend(fontsize=9.5)
    a2.bar(x, [f[k]["eval_disc"] for k in keys], width=0.03, color=C["teal"])
    a2.set_xlabel("mức giảm giá δ")
    a2.set_title("Tỉ lệ giảm giá của RL (ĐG)")
    save(fig, "ch6_f_delta")


def fig_ch6_h():
    h = raw("H")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75))
    band(a1, h, "curve_disc", C["blue"], "Q-learning 1")
    band(a1, h, "curve_opp_disc", C["red"], "Q-learning 2")
    band(a1, raw("C")[10:20], "curve_disc", C["gray"], "Q-learning vs luật")
    a1.set_xlabel("episode")
    a1.set_title("Tỉ lệ giảm giá (HL)")
    a1.legend(fontsize=9.5)
    band(a2, h, "curve", C["blue"], "Q-learning 1")
    band(a2, h, "curve_opp", C["red"], "Q-learning 2")
    a2.set_xlabel("episode")
    a2.set_title("Thưởng HL / bước")
    save(fig, "ch6_h_two")


def fig_ch7_q10():
    q = load_json("ch7_q10")
    scen = ["tham_chieu", "giam_5", "an_mieng", "luon_giam", "ben_vung"]
    lab = ["Tham chiếu", "δ = 5%", "Ăn miếng\ntrả miếng", "Luôn giảm", "Nhu cầu bền"]
    ags = [("q", "Q-learning (d,c)"), ("q_ext", "Q-learning mở rộng"), ("model", "Dựa trên mô hình")]
    x = np.arange(len(scen))
    fig, ax = plt.subplots(figsize=(W, 2.75))
    for j, (ag, name) in enumerate(ags):
        ax.bar(x + (j - 1) * 0.26, [q[s][ag]["regret"] for s in scen], 0.26, color=SEQ[j], label=name)
    ax.set_xticks(x, lab)
    ax.set_ylabel("regret = tối ưu − giá trị chính xác")
    ax.legend(fontsize=9.5)
    save(fig, "ch7_q10_regret")


def main():
    fig_ch4_delta()
    fig_ch6_baseline()
    base = raw("C")
    fig_sweep("C", ["0.01", "0.10", "0.50"], ["α = 0,01", "α = 0,10", "α = 0,50"], "ch6_c_alpha")
    fig_sweep("D", ["khong", "thap", "co_so", "cao"], ["ε = 0", "thấp", "cơ sở", "cao"], "ch6_d_explore",
              extra=[dict(r, key="co_so") for r in base if r["key"] == "0.10"])
    fig_ch6_e()
    fig_ch6_f()
    fig_ch6_h()
    fig_ch7_q10()


if __name__ == "__main__":
    main()
