"""Hình ablation, thời gian và sơ đồ vết thực của một lượt chạy cấu hình full, đọc từ metrics."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from dnx.common import METRICS, load_json
from dnx.plot_style import H1, SEQ, W, save, vn_axis

CFG = {"full": "Đầy đủ", "no_critic": "Bỏ Critic", "vector_only": "Chỉ vector", "no_rules": "Bỏ luật", "single_agent": "Một tác tử"}
MET = {"route_ok": "Đúng tuyến", "numbers_ok": "Số trung thực", "citations_ok": "Trích dẫn hợp lệ", "safe": "An toàn"}
LANE = ["Supervisor", "Intake", "RuleGuard", "Risk", "Retina", "Knowledge", "Advisor", "Critic"]


def ablation(r):
    cfgs = [c for c in CFG if c in r["configs"]]
    fig, ax = plt.subplots(figsize=(W, H1 + 0.6))
    w = 0.8 / len(MET)
    for i, (k, lb) in enumerate(MET.items()):
        vals = [100 * (r["configs"][c][k] if r["configs"][c][k] is not None else np.nan) for c in cfgs]
        ax.bar(np.arange(len(cfgs)) + i * w, vals, w, label=lb, color=SEQ[i])
    ax.set_xticks(np.arange(len(cfgs)) + 1.5 * w, [CFG[c] for c in cfgs])
    ax.set(ylabel="Tỉ lệ đạt (%)", ylim=(0, 108))
    ax.legend(ncol=4, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.14), columnspacing=1.0, handlelength=1.2)
    ax.grid(axis="x", visible=False)
    save(fig, "agent_ablation")


def seconds(r):
    cfgs = [c for c in CFG if c in r["configs"]]
    fig, ax = plt.subplots(figsize=(W, H1))
    ax.bar([CFG[c] for c in cfgs], [r["configs"][c]["seconds_mean"] for c in cfgs], color=SEQ[0])
    ax.set(ylabel="Giây mỗi kịch bản (trung bình)")
    vn_axis(ax, "y", 0)
    ax.grid(axis="x", visible=False)
    save(fig, "agent_seconds")


def trace():
    """Chọn lượt thật dài nhất của cấu hình full; mỗi đoạn là thời gian thực của một tác tử."""
    rows = [json.loads(x) for x in (METRICS / "agent_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    run = max((x for x in rows if x["config"] == "full"), key=lambda x: len(x["path"]))
    fig, ax = plt.subplots(figsize=(W, H1 + 0.5))
    open_at = {}
    for e in run["trace"]:
        if e["kind"] == "start":
            open_at[e["agent"]] = e["t"]
        elif e["kind"] == "end" and e["agent"] in open_at:
            y, t0 = LANE.index(e["agent"]), open_at.pop(e["agent"])
            ax.barh(y, max(e["t"] - t0, run["seconds"] / 150), left=t0, color=SEQ[y % len(SEQ)])
    ax.set_yticks(range(len(LANE)), LANE)
    ax.invert_yaxis()
    ax.set(xlabel="Thời gian (giây)")
    vn_axis(ax, "x", 0)
    ax.grid(axis="y", visible=False)
    save(fig, "agent_trace")
    print("vết:", run["id"], run["path"], flush=True)


if __name__ == "__main__":
    res = load_json("agent_ablation")
    ablation(res)
    seconds(res)
    trace()
