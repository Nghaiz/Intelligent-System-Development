"""Hình vector và bảng LaTeX cho Chương 8 từ outputs/metrics/ch8_qlearning.json (do notebook ghi).

    python scripts/report_assets.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ag1 import plot_style as ps  # noqa: E402
from ag1.common import load_json  # noqa: E402
from ag1.maze import ACTIONS, MazeEnv  # noqa: E402
from ag1.texfmt import table  # noqa: E402
from ag1.viz import ACTION_NAMES, draw_maze, draw_rewards  # noqa: E402

plt = ps.plt


def blocked(env, s):
    """Các hướng đi từ ô s đâm vào tường lửa hoặc ra khỏi mê cung."""
    out = []
    for name, dr, dc in ACTIONS:
        r, c = s[0] + dr, s[1] + dc
        if not (0 <= r < env.n_rows and 0 <= c < env.n_cols) or env.is_fire(r, c):
            out.append(name)
    return out


def meaning(env, s, best, nxt):
    walls = blocked(env, s)
    wall_txt = f"; {', '.join(walls)} là tường/biên" if walls else ""
    if nxt == env.goal:
        return f"Bước quyết định: {best} vào đích G{wall_txt}"
    if s == env.start:
        return f"Xuất phát: chọn {best}{wall_txt}"
    return f"Điểm rẽ: {best} sang {nxt}{wall_txt}"


def main():
    m = load_json("ch8_qlearning")
    env = MazeEnv()
    path = [tuple(p) for p in m["path"]]
    hist = m["history"]

    fig, ax = plt.subplots(figsize=(ps.W, ps.W * 0.98))
    draw_maze(ax, env, path)
    ps.save(fig, "ch8_route")

    fig, ax = plt.subplots(figsize=(ps.W, 4.2))
    draw_rewards(ax, hist)
    ps.save(fig, "ch8_rewards")

    rows = []
    for rec in m["qtable"]:
        s = tuple(int(v) for v in rec["Trạng thái"].strip("()").split(","))
        q = [rec[f"Q({n})"] for n in ACTION_NAMES]
        best = rec["π*(s)"]
        nxt = path[path.index(s) + 1]
        cells = [f"{v:.2f}".replace(".", ",") for v in q]
        cells[int(np.argmax(q))] = f"\\textbf{{{cells[int(np.argmax(q))]}}}"
        rows.append([f"({s[0]}, {s[1]})", *cells, best, meaning(env, s, best, nxt)])
    table("ch8_qtable", "crrrrcL{6.2cm}", ["Trạng thái $s$", "Q(Trái)", "Q(Phải)", "Q(Lên)", "Q(Xuống)",
                                             "$\\pi^*(s)$", "Ý nghĩa điều hướng"], rows)

    rows = []
    for ep in range(250, len(hist["reward"]) + 1, 250):
        w = slice(ep - 250, ep)
        rows.append([f"{ep:,}".replace(",", "."), f"{hist['epsilon'][ep - 1]:.3f}".replace(".", ","),
                     f"{np.mean(hist['reward'][w]):.2f}".replace(".", ",").replace("-", "−"),
                     f"{np.mean(hist['steps'][w]):.1f}".replace(".", ","),
                     f"{100 * np.mean(hist['success'][w]):.1f}".replace(".", ",")])
    table("ch8_log", "rrrrr", ["Episode", "$\\varepsilon$ cuối khối", "Thưởng TB 250 ep.", "Số bước TB",
                               "Về đích (\\%)"], rows)


if __name__ == "__main__":
    main()
