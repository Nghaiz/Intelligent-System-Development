"""Hình RL từ rl_compare.json và rl_history.json, rồi chọn trọng số DQN cho policy.py theo mục tiêu trên kiểm định."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from dnx.common import load_json
from dnx.plot_style import H1, SEQ, W, save, vn_axis
from dnx.rl import predictor
from dnx.rl.features import COSTS, FEATURES, KEYS, N

MAIN = "0.002"
LABELS = {"fixed_importance": "Thứ tự cố định", "random": "Ngẫu nhiên", "greedy_info": "Tham lam thông tin",
          "greedy_info_per_cost": "Thông tin/chi phí"}
SHORT = {"high_bp": "Huyết áp cao", "high_chol": "Cholesterol cao", "chol_check": "Kiểm tra cholesterol", "bmi": "BMI",
         "smoker": "Hút thuốc", "stroke": "Đột quỵ", "heart_disease_or_attack": "Bệnh tim", "phys_activity": "Vận động",
         "fruits": "Trái cây", "veggies": "Rau", "hvy_alcohol_consump": "Rượu bia", "any_healthcare": "Bảo hiểm",
         "no_docbc_cost": "Ngại chi phí", "gen_hlth": "Sức khỏe chung", "ment_hlth": "Ngày tinh thần kém",
         "phys_hlth": "Ngày thể chất kém", "diff_walk": "Khó đi lại", "sex": "Giới", "age": "Tuổi", "education": "Học vấn",
         "income": "Thu nhập"}


def learning(hist):
    fig, ax = plt.subplots(figsize=(W, H1))
    runs = [v for k, v in hist.items() if k.endswith(f"_l{MAIN}")]
    x = [r["step"] for r in runs[0]]
    y = np.array([[r["objective"] for r in v] for v in runs])
    ax.plot(x, y.mean(0), label=f"Trung bình {len(runs)} seed")
    ax.fill_between(x, y.min(0), y.max(0), alpha=0.2, label="Khoảng min–max")
    ax.set(xlabel="Bước môi trường", ylabel="Mục tiêu trên kiểm định")
    vn_axis(ax, "y", 3)
    vn_axis(ax, "x")
    ax.legend()
    save(fig, "rl_learning_curve")


def by_questions(c):
    fig, ax = plt.subplots(figsize=(W, H1))
    for k, lb in LABELS.items():
        ax.plot([r["k"] for r in c["policies"][k]["curve"]], [r["auc"] for r in c["policies"][k]["curve"]], label=lb)
    d = c["dqn_curves"][MAIN]["curve"]
    ax.plot([r["k"] for r in d], [r["auc"] for r in d], color="black", label=f'DQN ({c["dqn_curves"][MAIN]["n_seeds"]} seed)')
    ax.fill_between([r["k"] for r in d], [r["auc"] - r["auc_std"] for r in d], [r["auc"] + r["auc_std"] for r in d],
                    color="black", alpha=0.15)
    ax.set(xlabel="Số câu đã hỏi", ylabel="AUC", xlim=(1, 21), ylim=(0.6, None))
    vn_axis(ax, "y", 2)
    ax.legend(ncol=2, fontsize=11)
    save(fig, "rl_auc_by_questions")


def tradeoff(c):
    fig, ax = plt.subplots(figsize=(W, H1))
    for k, lb in LABELS.items():
        rows = c["policies"][k]["curve"][1:]
        ax.plot([r["cost"] for r in rows], [r["auc"] for r in rows], label=lb, alpha=0.8)
    pts = sorted(((float(l), np.mean([r["cost"] for r in v]), np.mean([r["auc"] for r in v])) for l, v in c["dqn_adaptive"].items()),
                 key=lambda t: t[1])
    ax.plot([p[1] for p in pts], [p[2] for p in pts], "o-", color="black", label="DQN dừng thích ứng (theo λ)")
    q = sorted(((np.mean([r["cost"] for r in v]), np.mean([r["auc"] for r in v])) for v in c["tabular_q"].values()))
    ax.plot([p[0] for p in q], [p[1] for p in q], "s--", color=SEQ[3], label="Q-learning bảng")
    ax.set(xlabel="Chi phí hỏi trung bình", ylabel="AUC", ylim=(0.6, None))
    vn_axis(ax, "y", 2)
    vn_axis(ax, "x", 1)
    ax.legend(ncol=1, fontsize=10, loc="lower right")
    save(fig, "rl_cost_tradeoff")


def frequency(c):
    freq = np.mean([r["freq"] for r in c["dqn_adaptive"][MAIN]], axis=0)
    order = np.argsort(freq)
    fig, ax = plt.subplots(figsize=(W, 6.0))
    ax.barh([SHORT[KEYS[i]] for i in order], freq[order] * 100, color=SEQ[0])
    ax.set(xlabel="Tỉ lệ người được hỏi (%)")
    vn_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    save(fig, "rl_question_frequency")


def q_heatmap():
    import torch

    from dnx.rl import dqn
    from dnx.rl.env import make_state
    meta = json.loads((predictor.DIR / "policy.json").read_text(encoding="utf-8"))
    model, mean, std = predictor.load("cpu")
    q = dqn.QNet()
    q.load_state_dict(torch.load(predictor.DIR / "dqn" / meta["weights"], map_location="cpu", weights_only=True))
    ps = np.array([0.03, 0.08, 0.15, 0.3, 0.5, 0.7])
    x, m = torch.zeros(len(ps), N), torch.zeros(len(ps), N)
    with torch.no_grad():
        v = q(make_state(x, m, torch.tensor(ps, dtype=torch.float32))).numpy()
    top = np.argsort(-v.mean(0))[:12]
    names = [SHORT[KEYS[i]] if i < N else "Dừng" for i in top]
    fig, ax = plt.subplots(figsize=(W, 5.2))
    im = ax.imshow(v[:, top].T, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ps)), [f"{p * 100:.0f}%".replace(".", ",") for p in ps])
    ax.set_yticks(range(len(top)), names)
    ax.set(xlabel="Xác suất hiện tại, chưa có đáp án nào")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Giá trị Q")
    save(fig, "rl_q_heatmap")


def choose_policy(c):
    """Chọn λ chính, seed có mục tiêu kiểm định cao nhất; không nhìn tập test."""
    hist = load_json("rl_history")
    best = max((k for k in hist if k.endswith(f"_l{MAIN}")), key=lambda k: max(r["objective"] for r in hist[k]))
    meta = {"weights": f"{best}.pt", "lam": float(MAIN), "seed": int(re.match(r"s(\d+)", best)[1]),
            "selected_by": "objective trên tập kiểm định"}
    (predictor.DIR / "policy.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(meta, flush=True)


if __name__ == "__main__":
    assert len(COSTS) == len(FEATURES)
    cmp_, hist_ = load_json("rl_compare"), load_json("rl_history")
    choose_policy(cmp_)
    learning(hist_)
    by_questions(cmp_)
    tradeoff(cmp_)
    frequency(cmp_)
    q_heatmap()
