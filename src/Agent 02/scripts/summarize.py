"""Tổng hợp outputs/raw/*.json thành số liệu báo cáo outputs/metrics/ch4_*, ch6_*, ch7_*.json (nguồn của \\SL)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from ag2.agents import QLearningAgent, RuleBasedAgent, observe_base  # noqa: E402
from ag2.common import OUT, save_json  # noqa: E402
from ag2.exact import best_stationary, exact_value, optimal_value, policy_of  # noqa: E402
from ag2.market import MarketConfig, conversion_probability, expected_reward, unit_reward  # noqa: E402

RAW = OUT / "raw"
T95 = 2.262   # t_{0,975; 9} cho 10 seed
TOL = 1e-6


def load(exp):
    return json.loads((RAW / f"{exp}.json").read_text(encoding="utf-8"))


def stats(rs):
    """Trung bình, độ lệch chuẩn mẫu và số seed đạt tối ưu (giá trị chính xác bằng tối ưu dừng)."""
    ev = np.array([r["eval_mean"] for r in rs])
    ex = np.array([r["exact"] for r in rs])
    best = np.array([r.get("best_stationary", r["optimal"]) for r in rs])
    out = {"eval_mean": ev.mean(), "eval_sd": ev.std(ddof=1), "exact_mean": ex.mean(), "exact_sd": ex.std(ddof=1),
           "n_opt": int((ex >= best - TOL).sum()), "n": len(rs),
           "eval_disc": float(np.mean([r["eval_disc"] for r in rs])),
           "train_mean": float(np.mean([r["train_mean"] for r in rs])),
           "train_last": float(np.mean([r["train_last"] for r in rs])),
           "train_disc": float(np.mean([r["train_disc"] for r in rs])),
           "rule_mean": float(np.mean([r["rule_mean"] for r in rs])),
           "exact_rule": float(np.mean([r["exact_rule"] for r in rs])),
           "optimal": float(np.mean([r["optimal"] for r in rs])), "best": float(best.mean()),
           "regret": float((np.array([r["optimal"] for r in rs]) - ex).mean())}
    if "switches" in rs[0]:
        out["switches"] = float(np.mean([r["switches"] for r in rs]))
    d = np.array([r["paired_diff"] for r in rs])
    out.update(diff_mean=d.mean(), diff_sd=d.std(ddof=1), ci_lo=d.mean() - T95 * d.std(ddof=1) / np.sqrt(len(d)),
               ci_hi=d.mean() + T95 * d.std(ddof=1) / np.sqrt(len(d)))
    return out


def group(rs):
    g = {}
    for r in rs:
        g.setdefault(r["key"], []).append(r)
    return g


def model_tables():
    """Ch.4: bảng thưởng kỳ vọng một bước và giá trị tối ưu của các chính sách mốc."""
    out = {}
    for name, delta in (("d20", 0.20), ("d05", 0.05)):
        cfg = MarketConfig().with_(discount_rate=delta)
        rows = []
        for d in (0, 1):
            for b in (0, 1):
                keep, disc = expected_reward(cfg, d, 0, b), expected_reward(cfg, d, 1, b)
                rows.append({"d": d, "b": b, "p0": conversion_probability(cfg, d, 0, b),
                             "p1": conversion_probability(cfg, d, 1, b), "keep": keep, "disc": disc,
                             "better": int(disc > keep)})
        rule = policy_of(RuleBasedAgent(), 1)
        bs = best_stationary(cfg, rule, observe_base, 2)
        out[name] = {"rows": rows, "margin_keep": unit_reward(cfg, 0), "margin_disc": unit_reward(cfg, 1),
                     "rule_rule": exact_value(cfg, policy_of(RuleBasedAgent(), 0), rule)[0],
                     "best_value": bs["value"], "best_policy": bs["policy"], "optimal": optimal_value(cfg, rule)}
    return out


def baseline_metrics():
    """Ch.6 A–B: lượt tham chiếu, đường học và giá trị chính xác của chính sách tham lam theo episode."""
    b = json.loads((RAW / "baseline.json").read_text(encoding="utf-8"))
    z = np.load(RAW / "baseline.npz")
    cfg, rule = MarketConfig(), policy_of(RuleBasedAgent(), 1)
    probe = QLearningAgent()
    checkpoints, values = list(range(0, 5000, 25)) + [4999], []
    for e in checkpoints:
        for k, s in enumerate(probe.states):
            probe.q[s] = z["q"][e, 2 * k:2 * k + 2].copy()
        values.append(exact_value(cfg, policy_of(probe, 0), rule)[0])
    pol = list(z["policy"])
    stable = next(i for i in range(len(pol)) if all(p == pol[-1] for p in pol[i:]))
    r = z["reward"][:, 0]
    a = {"eval_mean": b["eval"]["mean"][0], "eval_total": b["eval"]["total"][0], "eval_disc": b["eval"]["discount_rate"][0],
         "rule_mean": b["eval_rule"]["mean"][0], "rule_total": b["eval_rule"]["total"][0],
         "diff": b["eval"]["mean"][0] - b["eval_rule"]["mean"][0], "steps": b["eval"]["steps"],
         "policy": b["policy"], "exact": b["exact"][0], "q": b["q"], "visits": b["visits"]}
    bb = {"first500": float(r[:500].mean()), "last500": float(r[-500:].mean()), "eps_end": float(z["epsilon"][-1]),
          "disc_first500": float(z["discount"][:500, 0].mean()), "disc_last500": float(z["discount"][-500:, 0].mean()),
          "stable_ep": int(stable), "n_policies": len(set(pol)),
          "exact_curve": {"ep": checkpoints, "v": values}}
    return a, bb, b["first_updates"]


def q3_example(first):
    """Câu 3: một bước cập nhật thật lấy từ nhật ký huấn luyện tham chiếu (bước đầu tiên có Q cũ và max Q(s') > 0)."""
    u = next(x for x in first if x["q_old"] > 0 and x["max_next"] > 0 and not x["done"])
    alpha, gamma = 0.10, 0.90
    target = u["r"] + gamma * u["max_next"]
    return {"s": "".join(map(str, u["s"])), "a": u["a"], "r": u["r"], "s2": "".join(map(str, u["s2"])),
            "alpha": alpha, "gamma": gamma, "q_old": u["q_old"], "max_next": u["max_next"], "target": target,
            "td": u["td"], "q_new": u["q_new"], "check": u["q_old"] + alpha * (target - u["q_old"])}


def run_meta():
    """Ch.5: số lượt, số tiến trình, thời gian tường và thời gian huấn luyện TB mỗi lượt (từ log và raw)."""
    import os
    import platform
    import re

    import matplotlib
    log = (OUT / "logs" / "run_all.log").read_text(encoding="utf-8")
    n, workers, wall = re.match(r"(\d+) lượt \+ baseline, (\d+) tiến trình, ([\d.]+) s", log).groups()
    ts = {exp: [r["train_s"] for r in load(exp)] for exp in ("C", "D", "F", "G", "H", "Q10")}
    q10 = load("Q10")
    return {"runs": int(n) + 1, "workers": int(workers), "wall_s": float(wall), "cpus": os.cpu_count(),
            "train_s_q": float(np.mean(ts["C"])),
            "train_s_model": float(np.mean([r["train_s"] for r in q10 if r["agent"] == "model"])),
            "serial_s": float(sum(sum(v) for v in ts.values())),
            "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__}


def main():
    save_json("ch5_run", run_meta())
    save_json("ch4_model", model_tables())
    a, bb, first = baseline_metrics()
    save_json("ch6_a", a)
    save_json("ch6_b", bb)
    save_json("ch7_q3", q3_example(first))
    c = group(load("C"))
    save_json("ch6_c", {k: stats(v) for k, v in c.items()})
    d = group(load("D"))
    d["co_so"] = c["0.10"]
    save_json("ch6_d", {k: stats(v) for k, v in d.items()})
    e = c["0.10"]
    save_json("ch6_e", {"agg": stats(e), "seeds": [{k: r[k] for k in ("seed", "eval_mean", "rule_mean", "paired_diff",
                                                                    "paired_se", "exact", "policy", "eval_disc")}
                                                  for r in e]})
    f = group(load("F"))
    f["0.20"] = e
    save_json("ch6_f", {k.replace(".", ""): stats(v) for k, v in sorted(f.items())})
    save_json("ch6_g", {k: stats(v) for k, v in group(load("G")).items()})
    h = load("H")
    save_json("ch6_h", {"agg": stats(h), "opp_eval": float(np.mean([r["eval_opp_mean"] for r in h])),
                        "opp_disc": float(np.mean([r["eval_opp_disc"] for r in h])),
                        "policies": [r["policy"] for r in h], "opp_policies": [r["opp_policy"] for r in h],
                        "both_0000": int(sum(r["policy"] == "0000" and r["opp_policy"] == "0000" for r in h))})
    q10 = {}
    for k, v in group(load("Q10")).items():
        sc, ag = k.split("/")
        q10.setdefault(sc, {})[ag] = stats(v)
    save_json("ch7_q10", q10)
    print("đã ghi metrics")


if __name__ == "__main__":
    main()
