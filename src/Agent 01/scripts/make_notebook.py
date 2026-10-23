"""Sinh và chạy notebooks/agent01_rl_qlearning.ipynb.

Ô mã của notebook là nguyên văn ag1/maze.py, ag1/qlearning.py, ag1/viz.py (bỏ dòng import) nên ảnh chụp màn hình
notebook và đoạn mã trích trong báo cáo là cùng một mã. Notebook tự ghi outputs/metrics/ch8_qlearning.json.

    python scripts/make_notebook.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "agent01_rl_qlearning.ipynb"


def body(module):
    """Mã của mô-đun không kèm các dòng import ở đầu tệp."""
    lines = (ROOT / "ag1" / f"{module}.py").read_text(encoding="utf-8").splitlines()
    keep = [l for l in lines if not l.startswith(("import ", "from "))]
    return "\n".join(keep).strip("\n")


IMPORTS = """\
import json
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

plt.rcParams.update({"font.size": 12, "figure.dpi": 110})"""

TRAIN = """\
env = MazeEnv()
agent = QLearningAgent(env.n_states, env.n_actions)
print(f"Mê cung {env.n_rows}x{env.n_cols}: {env.n_states} trạng thái, {int((env.grid == '#').sum())} ô tường lửa, "
      f"S={env.start}, G={env.goal}")
print(f"alpha={ALPHA}, gamma={GAMMA}, epsilon: {EPS_START} -> {EPS_MIN} (x{EPS_DECAY}/episode), "
      f"{EPISODES} episodes, tối đa {MAX_STEPS} bước/episode")
hist = train(env, agent)
print(f"Huấn luyện xong trong {hist['seconds']:.2f} giây")"""

PATH = """\
path, total = greedy_path(env, agent)
print(f"Lộ trình học được: {len(path) - 1} bước, tổng phần thưởng {total:+.1f}")
print(f"Đường ngắn nhất (BFS kiểm chứng): {env.shortest_path_length()} bước")
print(" -> ".join(str(p) for p in path))"""

PLOT = """\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1, 1.35]})
draw_maze(ax1, env, path)
draw_rewards(ax2, hist)
fig.tight_layout()
plt.show()"""

QTAB = """\
qtab = q_table_excerpt(env, agent, path)
qtab"""

SAVE = """\
def first_episode(cond):
    return next((i + 1 for i, ok in enumerate(cond) if ok), None)

rewards = np.array(hist["reward"])
ma = np.convolve(rewards, np.ones(100) / 100, mode="valid")
metrics = {
    "params": {"alpha": ALPHA, "gamma": GAMMA, "eps_start": EPS_START, "eps_min": EPS_MIN, "eps_decay": EPS_DECAY,
               "episodes": EPISODES, "max_steps": MAX_STEPS, "r_fire": R_FIRE, "r_goal": R_GOAL, "r_step": R_STEP,
               "n_rows": env.n_rows, "n_cols": env.n_cols, "n_states": env.n_states,
               "n_fire": int((env.grid == "#").sum()), "start": str(env.start), "goal": str(env.goal)},
    "result": {"path_len": len(path) - 1, "bfs_len": env.shortest_path_length(), "greedy_reward": total,
               "seconds": hist["seconds"], "first_success": first_episode(hist["success"]),
               "ma_first_positive": first_episode(ma > 0) + 99,
               "eps_below_005": first_episode(np.array(hist["epsilon"]) < 0.05),
               "reward_first100": rewards[:100].mean(), "reward_min": rewards.min(),
               "reward_last100": rewards[-100:].mean(), "steps_last100": np.mean(hist["steps"][-100:]),
               "success_first250": np.mean(hist["success"][:250]), "success_last100": np.mean(hist["success"][-100:])},
    "qtable": qtab.to_dict(orient="records"),
    "history": {k: hist[k] for k in ("reward", "steps", "epsilon", "success")},
    "path": [list(p) for p in path],
}
out = Path("../outputs/metrics/ch8_qlearning.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=lambda o: o.item()), encoding="utf-8")
print("Đã ghi", out, "-", {k: metrics["result"][k] for k in ("path_len", "bfs_len", "greedy_reward")})"""


def main():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        md("# Bài tập Agent 01 — Q-Learning Agent điều hướng mê cung\n"
           "Nguyễn Duy Nghĩa · B23DCCN600 · Nhóm lớp CT — Nhóm 07\n\n"
           "Tác tử học tăng cường tự khám phá mê cung 12×12, tránh tường lửa và tìm đường ngắn nhất từ **S** về **G**."),
        md("## 1. Thư viện"), code(IMPORTS),
        md("## 2. Môi trường mê cung (MDP: S, A, P, R)"), code(body("maze")),
        md("## 3. Q-Learning Agent và vòng huấn luyện"), code(body("qlearning")),
        md("## 4. Huấn luyện tác tử"), code(TRAIN),
        md("## 5. Lộ trình tối ưu theo chính sách tham lam"), code(PATH),
        md("## 6. Trực quan hóa kết quả"), code(body("viz")), code(PLOT),
        md("## 7. Trích đoạn Q-Table tại các ô then chốt"), code(QTAB),
        md("## 8. Lưu số liệu cho báo cáo"), code(SAVE),
    ]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
    NB.parent.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbformat.write(nb, NB)
    print("->", NB)


if __name__ == "__main__":
    main()
