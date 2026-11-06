"""Ch.8: nhập số liệu của bài Agent 02 (đã chạy) sang outputs/metrics/ch8_*.json, không thí nghiệm lại.

Ngoài bản sao số liệu, kịch bản sinh ch8_datasets.json, ch8_compare.json, nhật ký log_run và bảng Q cho trang
mô phỏng trên Web App (models/ch8/q_tables.json: huấn luyện lại đúng cấu hình tham chiếu, chỉ phục vụ minh họa).
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from tlf.ch8_agents import QLearningAgent, RuleBasedAgent  # noqa: E402
from tlf.ch8_market import ECommerceMarket, MarketConfig  # noqa: E402
from tlf.ch8_training import run_episode, train  # noqa: E402
from tlf.common import ROOT, data_dir, load_json, model_dir, save_json  # noqa: E402
from tlf.runlog import RUNS, log_run  # noqa: E402

AG2 = ROOT.parent / "Agent 02" / "outputs"
# tên trong bài Agent 02 -> tên trong Tiểu luận Final
METRIC_MAP = {"ch4_model": "ch8_model", "ch5_run": "ch8_run", "ch6_a": "ch8_a", "ch6_b": "ch8_b", "ch6_c": "ch8_c",
              "ch6_d": "ch8_d", "ch6_e": "ch8_e", "ch6_f": "ch8_f", "ch6_g": "ch8_g", "ch6_h": "ch8_h",
              "ch7_q3": "ch8_q3", "ch7_q10": "ch8_q10"}
SCENARIOS = {"tham_chieu": "Tham chiếu", "giam_5": "δ = 5%", "an_mieng": "Đối thủ ăn miếng trả miếng",
             "luon_giam": "Đối thủ luôn giảm giá", "ben_vung": "Nhu cầu bền"}
AGENTS = {"q": "Q-learning (d, c)", "q_ext": "Q-learning mở rộng", "model": "Dựa trên mô hình"}
DEMO_DELTAS = (0.05, 0.10, 0.20, 0.30)
DEMO_SEED = 2026
DEMO_EPISODES = 5000


def copy_results():
    """Chép nguyên số liệu đã chốt của Agent 02; dữ liệu thô dùng để vẽ lại hình cỡ chữ của tiểu luận."""
    for src, dst in METRIC_MAP.items():
        shutil.copy2(AG2 / "metrics" / f"{src}.json", ROOT / "outputs" / "metrics" / f"{dst}.json")
    raw = data_dir("ch8") / "raw"
    raw.mkdir(exist_ok=True)
    for p in (AG2 / "raw").iterdir():
        shutil.copy2(p, raw / p.name)
    return raw


def sample_rows():
    """Sáu bước đầu của một episode: bảng Q tham chiếu (đóng băng) đối đầu tác tử luật."""
    agent = QLearningAgent(seed=DEMO_SEED)
    agent.q = {s: np.array(v, dtype=float) for s, v in
               ((tuple(int(c) for c in k), v) for k, v in load_json("ch8_a")["q"].items())}
    env = ECommerceMarket(MarketConfig(), seed=DEMO_SEED)
    rule = RuleBasedAgent()
    joint, rows = env.reset(), []
    for t in range(6):
        d, c = agent.observe(joint, 0)
        acts = [agent.act((d, c), explore=False), rule.act(rule.observe(joint, 1))]
        r, sales, _ = env.step(acts)
        rows.append({"t": t + 1, "d": d, "c": c, "a_rl": acts[0], "a_luat": acts[1], "r_rl": r[0], "r_luat": r[1]})
        joint = env.joint_state()
    return rows


def write_datasets(raw):
    a = load_json("ch8_a")
    labels = [f"({k[0]}, {k[1]})" for k in sorted(a["visits"])]
    values = [int(sum(a["visits"][k])) for k in sorted(a["visits"])]
    size = sum(p.stat().st_size for p in raw.iterdir()) / 1e6
    save_json("ch8_datasets", {"datasets": [{
        "id": "market", "name": "Thị trường giảm giá hai cửa hàng (mô phỏng)",
        "source": "Bài Agent 02 của chính tác giả (src/Agent 02, Report/Agent 02)", "url": "",
        "task": "Học tăng cường: chọn giữ giá hoặc giảm giá để tối đa lợi nhuận",
        "n_samples": sum(values), "n_features": 2, "size_mb": round(size, 2), "license": "Tự sinh, dữ liệu tổng hợp",
        "note": "Dữ liệu tổng hợp: mọi quan sát do bộ mô phỏng sinh; không có dữ liệu bán hàng thật. "
                "n_samples là số bước huấn luyện của lượt tham chiếu, phân bố là số lần thăm từng trạng thái (d, c).",
        "sample_rows": sample_rows(), "distribution": {"labels": labels, "values": values}}]})


def compare_rows(q10, run):
    rows = []
    for key, name in SCENARIOS.items():
        ref = q10[key]["q"]
        rows.append({"dataset": name, "model": "Luật", "framework": "NumPy", "exact_value": ref["exact_rule"],
                     "regret": ref["optimal"] - ref["exact_rule"], "n_opt": None, "seconds": 0.0})
        for ag, label in AGENTS.items():
            s = q10[key][ag]
            sec = run["train_s_model"] if ag == "model" else run["train_s_q"]
            rows.append({"dataset": name, "model": label, "framework": "NumPy", "exact_value": s["exact_mean"],
                         "regret": s["regret"], "n_opt": s["n_opt"], "seconds": sec})
    return rows


def already_logged():
    return RUNS.exists() and any('"chapter": 8' in line for line in RUNS.read_text(encoding="utf-8").splitlines())


def write_compare():
    rows = compare_rows(load_json("ch8_q10"), load_json("ch8_run"))
    save_json("ch8_compare", {"rows": rows, "primary": "exact_value"})
    if already_logged():
        return
    for r in rows:
        if r["model"] != "Luật":
            log_run(8, r["dataset"], r["model"], r["framework"], {"exact_value": r["exact_value"], "regret": r["regret"]},
                    r["seconds"], "cpu", {"seeds": 10, "episodes": 5000, "source": "Agent 02"})


def demo_q_tables():
    """Bảng Q cho Web App: mỗi mức giảm giá một lượt huấn luyện với đối thủ luật."""
    out = {}
    for delta in DEMO_DELTAS:
        t0 = time.perf_counter()
        agent = QLearningAgent(seed=DEMO_SEED + 10_000)
        train(MarketConfig().with_(discount_rate=delta), [agent, RuleBasedAgent()], DEMO_EPISODES, DEMO_SEED)
        out[f"{delta:.2f}"] = {"".join(map(str, s)): agent.q[s].tolist() for s in agent.states}
        print(f"δ = {delta:.2f}: {agent.policy_string()} ({time.perf_counter() - t0:.1f}s)", flush=True)
    (model_dir("ch8") / "q_tables.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    raw = copy_results()
    write_datasets(raw)
    write_compare()
    demo_q_tables()


if __name__ == "__main__":
    main()
