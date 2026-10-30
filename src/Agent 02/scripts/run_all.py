"""Chạy toàn bộ thí nghiệm A–H và câu 10 song song trên CPU (mỗi lượt huấn luyện một tiến trình).

    python scripts/run_all.py            -> outputs/raw/*.json, outputs/raw/baseline.npz, outputs/logs/run_all.log

Q-table chỉ có 8 ô nên GPU không giúp gì; song song theo lượt chạy (seed × cấu hình) dùng hết các nhân CPU.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ag2.agents import FixedAgent, QLearningAgent, RuleBasedAgent, observe_base, observe_ext  # noqa: E402
from ag2.common import OUT  # noqa: E402
from ag2.exact import best_stationary, exact_value, optimal_value, policy_of  # noqa: E402
from ag2.market import MarketConfig  # noqa: E402
from ag2.model_agent import ModelBasedAgent  # noqa: E402
from ag2.training import evaluate, train  # noqa: E402

RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
BASE_SEED = 2026
SEEDS = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]
EPISODES, EVAL_EPISODES = 5000, 500
AGENT_SEED_OFFSET, EVAL_SEED_OFFSET = 10_000, 500_000
WINDOW = 100
SCHEDULES = {"thap": (0.05, 0.01, 0.995), "co_so": (0.20, 0.02, 0.995),
             "cao": (0.50, 0.05, 0.999), "khong": (0.0, 0.0, 1.0)}


def make_agent(kind, seed, **kw):
    """Tạo tác tử theo tên: q (đề bài), q_ext (thêm a_i(t−1)), model (câu 10), rule hoặc một đối thủ cố định."""
    if kind == "q":
        return QLearningAgent(seed=seed, **kw)
    if kind == "q_ext":
        return QLearningAgent(seed=seed, extended=True, **kw)
    if kind == "model":
        return ModelBasedAgent(seed=seed)
    if kind == "rule":
        return RuleBasedAgent()
    return FixedAgent(kind, seed=seed)


def run_job(job):
    """Một lượt: huấn luyện tác tử ghế 0 với đối thủ, đánh giá đóng băng, so cặp với luật trên cùng seed."""
    cfg = MarketConfig().with_(**job.get("cfg", {}))
    eval_cfg = MarketConfig().with_(**job.get("eval_cfg", job.get("cfg", {})))
    seed = job["seed"]
    me = make_agent(job["agent"], seed + AGENT_SEED_OFFSET, **job.get("agent_kw", {}))
    opp = make_agent(job["opp"], seed + 2 * AGENT_SEED_OFFSET, **job.get("opp_kw", {}))
    t0 = time.perf_counter()
    hist = train(cfg, [me, opp], EPISODES, seed)
    train_s = time.perf_counter() - t0
    ev = evaluate(eval_cfg, [me, opp], EVAL_EPISODES, seed + EVAL_SEED_OFFSET)
    # đối thủ học được thì giữ nguyên bản đã huấn luyện (đóng băng); đối thủ cố định thì tạo lại cùng seed
    learner = job["opp"] in ("q", "q_ext", "model")
    rule_opp = opp if learner else make_agent(job["opp"], seed + 2 * AGENT_SEED_OFFSET, **job.get("opp_kw", {}))
    ev_rule = evaluate(eval_cfg, [RuleBasedAgent(), rule_opp], EVAL_EPISODES, seed + EVAL_SEED_OFFSET)
    diff = ev["episodes"][:, 0] - ev_rule["episodes"][:, 0]
    opp_pol = policy_of(opp, 1)
    out = {
        "exp": job["exp"], "key": job["key"], "seed": seed, "agent": job["agent"], "opp": job["opp"],
        "eval_mean": ev["mean"][0], "eval_opp_mean": ev["mean"][1], "eval_disc": ev["discount_rate"][0],
        "eval_opp_disc": ev["discount_rate"][1],
        "rule_mean": ev_rule["mean"][0], "rule_disc": ev_rule["discount_rate"][0],
        "paired_diff": float(diff.mean()), "paired_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "train_mean": float(hist["reward"][:, 0].mean()), "train_last": float(hist["reward"][-500:, 0].mean()),
        "train_disc": float(hist["discount"][:, 0].mean()),
        "curve": hist["reward"][:, 0].reshape(-1, WINDOW).mean(axis=1).tolist(),
        "curve_disc": hist["discount"][:, 0].reshape(-1, WINDOW).mean(axis=1).tolist(),
        "exact": exact_value(eval_cfg, policy_of(me, 0), opp_pol)[0],
        "exact_rule": exact_value(eval_cfg, policy_of(RuleBasedAgent(), 0), opp_pol)[0],
        "optimal": optimal_value(eval_cfg, opp_pol), "train_s": train_s,
    }
    if job["opp"] in ("q", "q_ext"):   # hai tác tử cùng học: lưu cả đường giảm giá của đối thủ
        out["curve_opp_disc"] = hist["discount"][:, 1].reshape(-1, WINDOW).mean(axis=1).tolist()
        out["curve_opp"] = hist["reward"][:, 1].reshape(-1, WINDOW).mean(axis=1).tolist()
        out["opp_policy"] = opp.policy_string()
    if hasattr(me, "policy_string"):
        out["policy"] = me.policy_string()
        out["switches"] = int(sum(a != b for a, b in zip(hist["policy"], hist["policy"][1:])))
        out["q"] = {"".join(map(str, s)): me.q[s].tolist() for s in me.states}
        obs, n_obs = (observe_ext, 3) if job["agent"] == "q_ext" else (observe_base, 2)
        out["best_stationary"] = best_stationary(eval_cfg, opp_pol, obs, n_obs)["value"]
    return out


def jobs():
    """Danh sách lượt chạy của mọi thí nghiệm (khoá `exp/key`)."""
    js = []
    for alpha in (0.01, 0.10, 0.50):                                # C (α = 0,10 cũng là E)
        js += [dict(exp="C", key=f"{alpha:.2f}", seed=s, agent="q", opp="rule", agent_kw={"alpha": alpha})
               for s in SEEDS]
    for name, (e0, emin, rho) in SCHEDULES.items():                 # D
        if name == "co_so":
            continue
        kw = {"eps_start": e0, "eps_min": emin, "eps_decay": rho}
        js += [dict(exp="D", key=name, seed=s, agent="q", opp="rule", agent_kw=kw) for s in SEEDS]
    for delta in (0.05, 0.10, 0.15, 0.25, 0.30):                    # F (0,20 lấy từ E)
        js += [dict(exp="F", key=f"{delta:.2f}", seed=s, agent="q", opp="rule", cfg={"discount_rate": delta})
               for s in SEEDS]
    js += [dict(exp="G", key="ben_vung", seed=s, agent="q", opp="rule", cfg={"demand_persist": 0.8})
           for s in SEEDS]
    js += [dict(exp="G", key="doanh_thu", seed=s, agent="q", opp="rule", cfg={"reward_kind": "revenue"},
                eval_cfg={}) for s in SEEDS]
    js += [dict(exp="H", key="hai_q", seed=s, agent="q", opp="q") for s in SEEDS]
    scen = {"tham_chieu": ({}, "rule"), "giam_5": ({"discount_rate": 0.05}, "rule"),
            "an_mieng": ({}, "tit_for_tat"), "luon_giam": ({}, "always_discount"),
            "ben_vung": ({"demand_persist": 0.8}, "rule")}
    for sk, (cfg, opp) in scen.items():                             # câu 10
        for ag in ("q", "q_ext", "model"):
            js += [dict(exp="Q10", key=f"{sk}/{ag}", seed=s, agent=ag, opp=opp, cfg=cfg) for s in SEEDS]
    return js


def baseline():
    """A–B: một lượt tham chiếu đầy đủ lịch sử (Q từng episode), các cập nhật đầu tiên cho câu 3."""
    cfg = MarketConfig()
    me, opp = QLearningAgent(seed=BASE_SEED + AGENT_SEED_OFFSET), RuleBasedAgent()
    first = []

    def on_update(o, a, r, nxt, done, td):
        if len(first) < 40:
            q_new = float(me.q[o][a])
            first.append({"s": list(o), "a": int(a), "r": float(r), "s2": list(nxt), "done": bool(done),
                          "td": float(td), "q_new": q_new, "q_old": q_new - me.alpha * float(td),
                          "max_next": float(me.q[nxt].max())})
    hist = train(cfg, [me, opp], EPISODES, BASE_SEED, track_q=True, on_update=on_update)
    ev = evaluate(cfg, [me, opp], EVAL_EPISODES, BASE_SEED + EVAL_SEED_OFFSET)
    ev_rule = evaluate(cfg, [RuleBasedAgent(), RuleBasedAgent()], EVAL_EPISODES, BASE_SEED + EVAL_SEED_OFFSET)
    np.savez_compressed(RAW / "baseline.npz", reward=hist["reward"], discount=hist["discount"],
                        epsilon=hist["epsilon"], q=hist["q"], policy=np.array(hist["policy"]),
                        eval_ep=ev["episodes"], eval_rule_ep=ev_rule["episodes"])
    res = {"eval": {k: ev[k] for k in ("total", "mean", "discount_rate", "steps")},
           "eval_rule": {k: ev_rule[k] for k in ("total", "mean", "discount_rate", "steps")},
           "q": {"".join(map(str, s)): me.q[s].tolist() for s in me.states},
           "visits": {"".join(map(str, s)): me.visits[s].tolist() for s in me.states},
           "policy": me.policy_string(), "first_updates": first,
           "exact": exact_value(cfg, policy_of(me, 0), policy_of(opp, 1))}
    (RAW / "baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main():
    t0 = time.perf_counter()
    js = jobs()
    workers = min(24, os.cpu_count() or 4)
    with Pool(workers) as pool:
        res_async = pool.map_async(run_job, js, chunksize=1)
        base = baseline()
        results = res_async.get()
    by_exp = {}
    for r in results:
        by_exp.setdefault(r["exp"], []).append(r)
    for exp, rs in by_exp.items():
        (RAW / f"{exp}.json").write_text(json.dumps(rs, ensure_ascii=False), encoding="utf-8")
    msg = (f"{len(js)} lượt + baseline, {workers} tiến trình, {time.perf_counter() - t0:.1f} s; "
           f"baseline {base['eval']['mean'][0]:.4f} chính sách {base['policy']}")
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs" / "run_all.log").write_text(msg + "\n", encoding="utf-8")
    print(msg)


if __name__ == "__main__":
    main()
