"""Chương 4: hình kết quả từ ch4_*.json và dự báo đã lưu; chạy sau ch4_collect.py và ch4_extras.py."""
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from tlf.ch4_data import OUT_DIR, TICKERS, Prepared
from tlf.ch4_torch import ARCH_NAMES, ARCHS, config_of
from tlf.common import load_json, model_dir
from tlf.plot_style import C, W, save

COLOR = {"RNN": C["blue"], "LSTM": C["orange"], "GRU": C["green"], "BiLSTM": C["purple"],
         "Persistence": C["gray"], "Mùa vụ 24 giờ": C["brown"]}
DASH = {"Persistence": "--", "Mùa vụ 24 giờ": ":"}
SHOW = (("FPT", "FPT (phiên)", "VNĐ"), ("weather", "Nhiệt độ (giờ)", "°C"), ("power", "Công suất (giờ)", "kW"))
RUNS = OUT_DIR / "runs"
ZOOM = {"FPT": 120, "weather": 168, "power": 168}


def _best_arch(summary, s):
    return next(a for a in ARCHS if ARCH_NAMES[a] == summary[s]["best"])


def _best_seed_level(prep, arch):
    """Dự báo của hạt giống có MSE validation thấp nhất, đơn vị gốc."""
    metas = sorted(RUNS.glob(f"{prep.spec.id}__{arch}__s*.json"))
    best = min(metas, key=lambda f: json.loads(f.read_text(encoding="utf-8"))["val_mse"])
    return prep.to_level(np.load(best.with_suffix(".npz"))["test"], prep.last_index("test"))


def fig_skill(rows):
    """Một trục cho mỗi nhóm chuỗi vì độ lớn khác nhau hẳn; 0 là ngang Persistence."""
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [2, 1]})
    groups = (list(TICKERS), ["weather", "power"])
    for a, names in zip(ax, groups):
        for i, arch in enumerate(ARCH_NAMES.values()):
            vals = [next(100 * r["skill"] for r in rows if r["dataset"] == s and r["model"] == arch
                         and r["framework"] == "pytorch") for s in names]
            a.bar(np.arange(len(names)) + (i - 1.5) * 0.2, vals, 0.18, color=COLOR[arch], label=arch)
        a.set_xticks(range(len(names)), [{"weather": "Nhiệt\nđộ", "power": "Công\nsuất"}.get(s, s) for s in names])
        a.axhline(0, color="black", lw=1)
    ax[0].set_ylabel("giảm RMSE so với\nPersistence (%)")
    ax[0].legend(ncols=4, loc="upper center", bbox_to_anchor=(0.75, 1.22), fontsize=11, columnspacing=1)
    fig.tight_layout()
    save(fig, "ch4_skill")


def fig_forecast(summary):
    fig, ax = plt.subplots(3, 1, figsize=(W, 6.0))
    for a, (s, title, unit) in zip(ax, SHOW):
        prep = Prepared(s)
        tt, lv = prep.last_index("test"), _best_seed_level(prep, _best_arch(summary, s))
        n = ZOOM[s]
        x = prep.index[tt[-n:] + 1]
        a.plot(x, prep.truth(tt)[-n:, 0], color="black", lw=1.5, label="thật")
        a.plot(x, lv[-n:, 0], color=C["orange"], lw=1.5, label=f"{summary[s]['best']}, một bước")
        a.plot(x, prep.level[tt[-n:]], color=C["gray"], lw=1, ls="--", label="Persistence")
        a.set_ylabel(unit)
        a.set_title(title, loc="left", fontsize=13)
        a.tick_params(axis="x", labelsize=10)
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncols=3, loc="upper center", fontsize=11, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, "ch4_forecast")


def fig_horizon(curves):
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.75))
    for a, (s, title, unit) in zip(ax, SHOW):
        for name, v in curves[s].items():
            a.plot(np.arange(1, len(v) + 1), v, color=COLOR[name], ls=DASH.get(name, "-"), lw=1.6, label=name)
        a.set_title(title, fontsize=12)
        a.set_xlabel("bước $k$")
        a.set_ylabel(f"RMSE ({unit})", labelpad=1)
        a.tick_params(labelsize=10)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, ncols=3, loc="upper center", fontsize=11, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, "ch4_horizon")


def fig_fan(summary):
    """Một lần dự báo trực tiếp ở điểm giữa tập test, kèm khoảng 90 % từ phân vị sai số validation."""
    import torch
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6))
    for a, (s, title, unit) in zip(ax, SHOW):
        prep = Prepared(s)
        arch = _best_arch(summary, s)
        tt, lv = prep.last_index("test"), _best_seed_level(prep, arch)
        i, h = len(tt) // 2, prep.spec.horizon
        q = torch.load(model_dir("ch4") / f"{s}__{arch}.pt", map_location="cpu", weights_only=True)["q"].numpy()
        z = np.log(lv[i]) if prep.spec.log_target else lv[i]
        lo, hi = (np.exp(z + q[0]), np.exp(z + q[1])) if prep.spec.log_target else (z + q[0], z + q[1])
        past = np.arange(-2 * h + 1, 1)
        fut = np.arange(1, h + 1)
        a.plot(past, prep.level[tt[i] + past], color="black", lw=1.3)
        a.plot(fut, prep.truth(tt)[i], color="black", lw=1.3, label="thật")
        a.fill_between(fut, lo, hi, color=C["orange"], alpha=0.25, label="khoảng 90 %")
        a.plot(fut, lv[i], color=C["orange"], lw=1.8, label="dự báo")
        a.axvline(0, color=C["gray"], lw=1)
        a.set_title(title, fontsize=12)
        a.set_xlabel("bước")
        a.tick_params(labelsize=10)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncols=3, loc="upper center", fontsize=11, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "ch4_fan")


def fig_curves():
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6))
    for a, (s, title, _) in zip(ax, SHOW):
        for arch, name in ARCH_NAMES.items():
            m = json.loads((RUNS / f"{s}__{arch}__s42.json").read_text(encoding="utf-8"))
            a.plot(np.arange(1, m["epochs"] + 1), m["hist"]["val_loss"], color=COLOR[name], lw=1.5, label=name)
            a.plot(np.arange(1, m["epochs"] + 1), m["hist"]["loss"], color=COLOR[name], lw=0.9, ls=":")
        a.set_title(title, fontsize=12)
        a.set_xlabel("epoch")
        a.tick_params(labelsize=10)
    ax[0].set_ylabel("MSE chuẩn hóa")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, ncols=4, loc="upper center", fontsize=11, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save(fig, "ch4_curves")


def fig_sensitivity(sens):
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for a, (s, title, _) in zip(ax, SHOW[1:]):
        for arch, name in ARCH_NAMES.items():
            a.plot(np.arange(len(sens[s][arch])), sens[s][arch], color=COLOR[name], lw=1.7, label=name)
        a.set_yscale("log")
        a.set_title(title, fontsize=12)
        a.set_xlabel("độ trễ $k$ (giờ)")
    ax[0].set_ylabel("độ nhạy tương đối")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncols=4, loc="upper center", fontsize=11, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save(fig, "ch4_sensitivity")


def fig_errors(errors):
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.5), gridspec_kw={"width_ratios": [0.8, 1.1, 1.5]})
    titles = {"FPT": ("FPT, biến động", "VNĐ"), "weather": ("Nhiệt độ, tháng", "°C"),
              "power": ("Công suất, giờ", "kW")}
    for a, s in zip(ax, ("FPT", "weather", "power")):
        rows = errors[s] if s == "FPT" else sorted(errors[s], key=lambda r: r["group"])
        if s == "FPT":
            rows = sorted(rows, key=lambda r: ["thấp", "vừa", "cao"].index(r["group"]))
        x = np.arange(len(rows))
        a.bar(x - 0.2, [r["base"] for r in rows], 0.38, color=C["gray"], label="Persistence")
        a.bar(x + 0.2, [r["model"] for r in rows], 0.38, color=C["orange"], label="mô hình chọn")
        a.set_title(titles[s][0], fontsize=12)
        a.set_ylabel(f"MAE ({titles[s][1]})", labelpad=1)
        a.set_xticks(x[::1 if len(x) < 5 else 3], [rows[i]["group"] for i in x[::1 if len(x) < 5 else 3]])
        a.tick_params(labelsize=10)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncols=2, loc="upper center", fontsize=11, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save(fig, "ch4_errors")


def fig_norms():
    """Chuẩn δ_t của RNN NumPy khi lùi về đầu cửa sổ, đo trực tiếp trong hàm backward."""
    m = json.loads((RUNS / "FPT__rnn__numpy.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(W, 2.3))
    v = np.array(m["delta_norms"])
    ax.plot(np.arange(len(v)), v / v[0], color=C["blue"])
    ax.set_yscale("log")
    ax.set_xlabel("số bước lùi từ $T$")
    ax.set_ylabel("$\\|\\delta_t\\|/\\|\\delta_T\\|$")
    fig.tight_layout()
    save(fig, "ch4_norms")


def main():
    summary, rows = load_json("ch4_summary"), load_json("ch4_compare")["rows"]
    fig_skill(rows)
    fig_forecast(summary)
    fig_horizon(load_json("ch4_horizon")["series"])
    fig_fan(summary)
    fig_curves()
    fig_sensitivity(load_json("ch4_sensitivity")["series"])
    fig_errors(load_json("ch4_errors")["rows"])
    fig_norms()


if __name__ == "__main__":
    main()
