"""Web App, Ch.8: mô phỏng vài vòng thị trường giữa tác tử Q-learning đã học và tác tử luật.

Bảng Q nằm ở models/ch8/q_tables.json (scripts/ch8_import.py); nạp lười, giữ lại, chạy CPU, chỉ cần NumPy.
Cùng một seed, tác tử luật được đặt vào ghế của RL và chạy lại trên đúng chuỗi nhu cầu để so sánh cặp.
"""
import json

import numpy as np

from .ch8_agents import FixedAgent, QLearningAgent, RuleBasedAgent
from .ch8_market import ECommerceMarket, MarketConfig
from .common import MODELS

STATE_NAMES = {"demand": ("thấp", "cao"), "action": ("giữ giá", "giảm giá")}
RIVALS = {"Tác tử luật": "rule", "Ăn miếng trả miếng": "tit_for_tat", "Luôn giảm giá": "always_discount"}
MAX_ROUNDS = 200

SPECS = [{
    "id": "market_sim", "title": "Mô phỏng thị trường hai cửa hàng", "dataset": "Thị trường giảm giá (mô phỏng)",
    "inputs": [
        {"name": "delta", "type": "select", "label": "Mức giảm giá δ", "default": "0.20",
         "options": ["0.05", "0.10", "0.20", "0.30"]},
        {"name": "rival", "type": "select", "label": "Đối thủ", "default": "Tác tử luật", "options": list(RIVALS)},
        {"name": "rounds", "type": "number", "label": "Số vòng", "default": 20, "min": 5, "max": MAX_ROUNDS},
        {"name": "seed", "type": "number", "label": "Seed", "default": 7, "min": 0, "max": 100000},
    ],
    "output": "series",
}]

_tables = None


def _load_tables():
    global _tables
    if _tables is None:
        _tables = json.loads((MODELS / "ch8" / "q_tables.json").read_text(encoding="utf-8"))
    return _tables


def _q_agent(delta_key):
    tables = _load_tables()
    if delta_key not in tables:
        raise ValueError(f"Không có bảng Q cho δ = {delta_key}; chọn một trong {sorted(tables)}")
    agent = QLearningAgent()
    agent.q = {tuple(int(c) for c in s): np.array(v, dtype=float) for s, v in tables[delta_key].items()}
    return agent


def _rival(kind, seed):
    return RuleBasedAgent() if kind == "rule" else FixedAgent(kind, seed=seed)


def _play(cfg, me, opp, rounds, seed):
    """Chạy `rounds` vòng; hết 20 bước thì sang episode mới. Trả danh sách từng vòng."""
    env = ECommerceMarket(cfg, seed)
    joint, rows = env.reset(), []
    for t in range(rounds):
        d = joint[0]
        acts = [me.act(me.observe(joint, 0), explore=False), opp.act(opp.observe(joint, 1), explore=False)]
        rewards, sales, done = env.step(acts)
        rows.append({"round": t + 1, "demand": STATE_NAMES["demand"][d], "a1": acts[0], "a2": acts[1],
                     "sales1": sales[0], "sales2": sales[1], "r1": rewards[0], "r2": rewards[1]})
        joint = env.reset() if done else env.joint_state()
    return rows


def _simulate(delta, rival_label, rounds, seed):
    if rival_label not in RIVALS:
        raise ValueError(f"Đối thủ không hợp lệ: {rival_label}")
    if not 5 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"Số vòng phải trong khoảng 5–{MAX_ROUNDS}")
    cfg = MarketConfig().with_(discount_rate=float(delta))
    kind = RIVALS[rival_label]
    rows = _play(cfg, _q_agent(delta), _rival(kind, seed), rounds, seed)
    rule_rows = _play(cfg, RuleBasedAgent(), _rival(kind, seed), rounds, seed)
    cum1, cum2 = np.cumsum([r["r1"] for r in rows]), np.cumsum([r["r2"] for r in rows])
    return {
        "label": f"Q-learning {cum1[-1]:.0f} so với đối thủ {cum2[-1]:.0f} sau {rounds} vòng",
        "series": {"x": [r["round"] for r in rows],
                   "lines": [{"name": "Q-learning (lợi nhuận cộng dồn)", "y": cum1.tolist()},
                             {"name": "Đối thủ (lợi nhuận cộng dồn)", "y": cum2.tolist()}]},
        "rows": [dict(r, a1=STATE_NAMES["action"][r["a1"]], a2=STATE_NAMES["action"][r["a2"]]) for r in rows],
        "summary": {"profit_rl": float(cum1[-1]), "profit_rival": float(cum2[-1]),
                    "profit_rule_in_seat": float(sum(r["r1"] for r in rule_rows)),
                    "discount_rate_rl": float(np.mean([r["a1"] for r in rows])),
                    "discount_rate_rival": float(np.mean([r["a2"] for r in rows]))},
        "note": "Mô phỏng tổng hợp, chỉ minh họa cách hai tác tử tương tác; không phải dữ liệu bán hàng thật.",
    }


def predict(spec_id, payload):
    if spec_id != "market_sim":
        raise KeyError(f"Không có mô hình {spec_id}")
    p = SPECS[0]["inputs"]
    d = {i["name"]: payload.get(i["name"], i["default"]) for i in p}
    return _simulate(str(d["delta"]), d["rival"], int(d["rounds"]), int(d["seed"]))
