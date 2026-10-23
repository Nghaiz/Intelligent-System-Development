import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ACTION_NAMES = ["Trái", "Phải", "Lên", "Xuống"]


def draw_maze(ax, env, path):
    """Vẽ mê cung (tường lửa đỏ cam), lộ trình tối ưu (đường xanh) và hai ô S, G."""
    walls = (env.grid == "#").astype(int)
    ax.imshow(walls, cmap=ListedColormap(["#F8FAFC", "#EA580C"]), vmin=0, vmax=1)
    ys, xs = zip(*path)
    ax.plot(xs, ys, color="#2563EB", lw=4, marker="o", ms=5, label=f"Lộ trình tối ưu ({len(path) - 1} bước)")
    ax.text(*env.start[::-1], "S", ha="center", va="center", fontsize=15, weight="bold", color="#15803D")
    ax.text(*env.goal[::-1], "G", ha="center", va="center", fontsize=15, weight="bold", color="#7C3AED")
    ax.set_xticks(np.arange(-0.5, env.n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n_rows, 1), minor=True)
    ax.grid(which="minor", color="#94A3B8", lw=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.set_title(f"Mê cung {env.n_rows}×{env.n_cols}: lộ trình của tác tử")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False)


def draw_rewards(ax, hist, window=100):
    """Phần thưởng mỗi episode, trung bình trượt và epsilon (trục phải)."""
    rewards = np.array(hist["reward"])
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ep = np.arange(1, len(rewards) + 1)
    ax.plot(ep, rewards, color="#93C5FD", lw=0.8, alpha=0.7, label="Phần thưởng mỗi episode")
    ax.plot(ep[window - 1:], ma, color="#1D4ED8", lw=2.6, label=f"Trung bình trượt {window} episode")
    ax.set_ylim(max(rewards.min(), -400), rewards.max() + 20)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tổng phần thưởng")
    ax.set_title("Đường cong hội tụ phần thưởng")
    ax2 = ax.twinx()
    ax2.plot(ep, hist["epsilon"], color="#DC2626", lw=2, ls="--", label="epsilon")
    ax2.set_ylabel("epsilon", color="#DC2626")
    ax2.set_ylim(0, 1.05)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right")


def q_table_excerpt(env, agent, path):
    """Trích Q-Table tại các ô then chốt: xuất phát, các điểm rẽ của lộ trình và ô ngay trước đích."""
    keys = [path[0]]
    for prev, cur, nxt in zip(path, path[1:], path[2:]):
        if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
            keys.append(cur)
    keys.append(path[-2])
    rows = []
    for s in dict.fromkeys(keys):
        q = agent.q[env.to_index(s)]
        rows.append({"Trạng thái": str(s), **{f"Q({n})": round(float(v), 2) for n, v in zip(ACTION_NAMES, q)},
                     "π*(s)": ACTION_NAMES[int(np.argmax(q))]})
    return pd.DataFrame(rows)
