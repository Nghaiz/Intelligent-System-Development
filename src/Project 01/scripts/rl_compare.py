"""So sánh chính sách hỏi trên tập test: đối chứng + Q-learning bảng + mọi DQN đã huấn luyện. Ghi rl_compare, rl_history."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from dnx.common import METRICS, device, limit_threads, save_json
from dnx.gpu import cap, heavy
from dnx.mem import slot
from dnx.rl import baselines as B
from dnx.rl import data, dqn, predictor
from dnx.rl.env import log_score
from dnx.rl.features import COSTS, FEATURES, KEYS, N
from dnx.runlog import log_run

QL_LAMS = (0.0005, 0.002, 0.005)


def np_(t):
    return t.detach().cpu().numpy()


def best_threshold(y, p):
    grid = np.quantile(p, np.linspace(0.3, 0.97, 60))
    return float(max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def score(yv, pv, yt, pt, cost=None, questions=None, lam=None):
    th = best_threshold(yv, pv)
    row = {"auc": float(roc_auc_score(yt, pt)), "f1": float(f1_score(yt, pt >= th, zero_division=0)), "threshold": th,
           "log_score": float(log_score(torch.tensor(yt), torch.tensor(pt)).mean())}
    if cost is not None:
        row.update(cost=float(cost), questions=float(questions))
        if lam is not None:
            row["objective"] = row["log_score"] - lam * row["cost"]
    return row


def curve(select, model, Xv, yv, Xt, yt, name):
    """Đường AUC/F1 theo số câu cho chính sách hỏi đủ 21 câu."""
    pv, ov = B.trace(select, model, Xv)
    pt, ot = B.trace(select, model, Xt)
    y1, y2 = np_(yv), np_(yt)
    costs = torch.tensor(COSTS, device=ot.device)[ot].cumsum(0).mean(1)
    rows = [{"k": 0, "cost": 0.0, "auc": 0.5, "f1": None, "log_score": float(log_score(yt, pt[0]).mean())}]
    for k in range(1, N + 1):
        s = score(y1, np_(pv[k]), y2, np_(pt[k]))
        rows.append({"k": k, "cost": float(costs[k - 1]), **s})
    freq = np.bincount(np_(ot[:5]).ravel(), minlength=N) / len(Xt)
    print(name, "k=5 auc", round(rows[5]["auc"], 4), "k=21", round(rows[-1]["auc"], 4), flush=True)
    return {"curve": rows, "first5_freq": freq.tolist(), "first": np.bincount(np_(ot[0]), minlength=N).tolist()}


def greedy_curves(model, Xtr, ytr, Xv, yv, Xt, yt):
    """Đối chứng tham lam chạy trên đủ tập test như mọi chính sách khác, nếu không thì các đường không so được."""
    out = {}
    for name, per_cost in (("greedy_info", False), ("greedy_info_per_cost", True)):
        g = B.GreedyInfo(Xtr, None, ytr, per_cost)
        out[name] = curve(g.selector(model), model, Xv, yv, Xt, yt, name)
    return out


def best_objectives(res):
    """Cùng thước đo với DQN: mỗi đối chứng hỏi đúng k câu đầu, chọn k tốt nhất cho từng λ."""
    out = {}
    for lam in res["dqn_adaptive"]:
        row = {}
        for name in ("fixed_importance", "greedy_info", "greedy_info_per_cost"):
            best = max(res["policies"][name]["curve"], key=lambda r: r["log_score"] - float(lam) * r["cost"])
            row[name] = {"k": best["k"], "cost": best["cost"], "auc": best["auc"], "f1": best["f1"],
                         "log_score": best["log_score"], "objective": best["log_score"] - float(lam) * best["cost"]}
        out[lam] = row
    return out


def main():
    limit_threads(4)
    cap(0.15)
    t0, dev = time.time(), device()
    model, mean, std = predictor.load(dev)
    splits, _, _ = data.load()
    mu, sd = mean.cpu().numpy(), std.cpu().numpy()
    tens = {k: (torch.tensor((x - mu) / sd, device=dev), torch.tensor(y, device=dev)) for k, (x, y) in splits.items()}
    Xtr, ytr = tens["train"]
    Xv, yv = tens["val"]
    Xt, yt = tens["test"]
    rank = B.importance_rank(model, Xv, yv)
    print("rank", [KEYS[i] for i in rank], flush=True)
    res = {"rank": [KEYS[i] for i in rank], "n_test": len(Xt), "n_val": len(Xv),
           "costs": dict(zip(KEYS, COSTS)), "features_vi": {f.key: f.question_vi for f in FEATURES}, "policies": {}}
    pol = res["policies"]
    pol["fixed_importance"] = curve(B.fixed_order(rank), model, Xv, yv, Xt, yt, "fixed")
    rnd = [curve(B.random_order(s), model, Xv, yv, Xt, yt, f"random{s}") for s in range(3)]
    pol["random"] = {"curve": [{"k": r0["k"], "cost": float(np.mean([r[i]["cost"] for r in [x["curve"] for x in rnd]])),
                                "auc": float(np.mean([x["curve"][i]["auc"] for x in rnd])),
                                "f1": r0["f1"] if i == 0 else float(np.mean([x["curve"][i]["f1"] for x in rnd]))}
                               for i, r0 in enumerate(rnd[0]["curve"])]}
    pol.update(greedy_curves(model, Xtr, ytr, Xv, yv, Xt, yt))
    # DQN: gom theo λ, mỗi seed một đường theo số câu và một điểm dừng thích ứng
    runs = sorted(METRICS.glob("rl_dqn_s*.json"))
    history, adaptive, curves = {}, {}, {}
    for f in runs:
        meta = json.loads(f.read_text(encoding="utf-8"))
        seed, lam = meta["seed"], meta["lam"]
        history[f"s{seed}_l{lam:g}"] = meta["history"]
        q = dqn.QNet().to(dev)
        q.load_state_dict(torch.load(predictor.DIR / "dqn" / f"s{seed}_l{lam:g}.pt", map_location=dev, weights_only=True))
        q.eval()
        pv = dqn.greedy_episode(q, model, Xv)[0][-1]
        pr, ask, mask = dqn.greedy_episode(q, model, Xt)
        cost = (mask * torch.tensor(COSTS, device=dev)).sum(1).mean().item()
        row = score(np_(yv), np_(pv), np_(yt), np_(pr[-1]), cost, mask.sum(1).mean().item(), lam)
        row.update(seed=seed, hist_q=np.bincount(np_(mask.sum(1)).astype(int), minlength=N + 1).tolist(),
                   freq=np_(mask.mean(0)).tolist())
        adaptive.setdefault(f"{lam:g}", []).append(row)

        def sel(x, m, p):
            from dnx.rl.env import make_state
            s = make_state(x, m, p)
            qq = dqn.masked_q(q(s), s)
            qq[:, N] = -1e9
            return qq.argmax(1)
        curves.setdefault(f"{lam:g}", []).append(curve(sel, model, Xv, yv, Xt, yt, f"dqn {seed} {lam:g}"))
    res["dqn_adaptive"] = adaptive
    res["dqn_seeds"] = {lam: sorted(r["seed"] for r in v) for lam, v in adaptive.items()}
    res["dqn_curves"] = {lam: {"curve": [{"k": c["k"], "cost": float(np.mean([x["curve"][c["k"]]["cost"] for x in v])),
                                          "auc": float(np.mean([x["curve"][c["k"]]["auc"] for x in v])),
                                          "auc_std": float(np.std([x["curve"][c["k"]]["auc"] for x in v])),
                                          "f1": c["f1"] if c["k"] == 0 else
                                          float(np.mean([x["curve"][c["k"]]["f1"] for x in v]))}
                                         for c in v[0]["curve"]],
                                "n_seeds": len(v)} for lam, v in curves.items()}
    # Q-learning bảng cùng λ với DQN
    ql = {}
    for lam in QL_LAMS:
        rows = []
        for seed in range(3):
            t = B.TabularQ(rank, dev)
            t.fit(model, Xtr, lam, seed)
            pv, _ = t.episode(model, Xv)
            pt, mt = t.episode(model, Xt)
            cost = (mt * torch.tensor(COSTS, device=dev)).sum(1).mean().item()
            rows.append({**score(np_(yv), np_(pv), np_(yt), np_(pt), cost, mt.sum(1).mean().item(), lam), "seed": seed})
        ql[f"{lam:g}"] = rows
        print("ql", lam, rows[0], flush=True)
    res["tabular_q"] = ql
    res["baseline_best_objective"] = best_objectives(res)
    res["seconds"] = time.time() - t0
    save_json("rl_compare", res)
    save_json("rl_history", history)
    log_run("rl", "cdc_brfss", "compare", "pytorch", {"auc_21": pol["fixed_importance"]["curve"][-1]["auc"]},
            res["seconds"], dev, {"runs": len(runs)})


def refresh_greedy():
    """Chỉ tính lại hai đường tham lam trên đủ test và bảng mục tiêu, giữ nguyên kết quả DQN đã có."""
    limit_threads(4)
    cap(0.15)
    dev = device()
    model, mean, std = predictor.load(dev)
    splits, _, _ = data.load()
    mu, sd = mean.cpu().numpy(), std.cpu().numpy()
    tens = {k: (torch.tensor((x - mu) / sd, device=dev), torch.tensor(y, device=dev)) for k, (x, y) in splits.items()}
    res = json.loads((METRICS / "rl_compare.json").read_text(encoding="utf-8"))
    res.pop("sub_greedy", None)
    res["policies"].update(greedy_curves(model, tens["train"][0], tens["train"][1], *tens["val"], *tens["test"]))
    res["baseline_best_objective"] = best_objectives(res)
    save_json("rl_compare", res)


if __name__ == "__main__":
    with slot(3, "rl compare"), heavy("rl compare"):
        refresh_greedy() if "--greedy-only" in sys.argv else main()
