"""Đánh giá chính xác (không lấy mẫu) và chính sách tối ưu.

Trạng thái chung (d, a_1(t−1), a_2(t−1)) chỉ có 8 giá trị nên có thể lan truyền phân phối của nó qua 20 bước
và cộng phần thưởng kỳ vọng. Kết quả là giá trị thật của một chính sách, tách khỏi nhiễu Binomial.
Một chính sách ở đây là hàm (joint, t) -> P(giảm giá).
"""
import itertools

import numpy as np

from .market import expected_reward


def policy_of(agent, seat):
    """Chính sách tham lam (đóng băng) của một tác tử dưới dạng hàm (joint, t) -> P(giảm giá)."""
    if hasattr(agent, "q"):
        def pol(joint, t):
            q = agent.q[agent.observe(joint, seat)]
            return 0.5 if q[0] == q[1] else float(np.argmax(q))
    elif hasattr(agent, "plan"):
        def pol(joint, t):
            return float(agent.greedy(agent.observe(joint, seat), t))
    else:
        def pol(joint, t):
            return float(agent.act(agent.observe(joint, seat), explore=False))
    return pol


def table_policy(bits, observe, seat):
    """Chính sách tất định từ chuỗi bit trên các quan sát theo thứ tự itertools.product."""
    n = len(bits).bit_length() - 1
    states = list(itertools.product((0, 1), repeat=n))
    lookup = dict(zip(states, bits))
    return lambda joint, t: float(lookup[observe(joint, seat)])


def _next_states(cfg, d, a1, a2):
    """Các trạng thái chung kế tiếp và xác suất: d' theo quá trình nhu cầu, hành động trước = (a1, a2)."""
    keep = cfg.demand_persist
    return [((d, a1, a2), keep), ((1 - d, a1, a2), 1.0 - keep)]


def exact_value(cfg, pol1, pol2):
    """Thưởng kỳ vọng mỗi bước (ghế 0, ghế 1) của cặp chính sách trên một episode."""
    dist = {(0, 0, 0): 0.5, (1, 0, 0): 0.5}
    total = np.zeros(2)
    for t in range(cfg.steps):
        nxt = {}
        for joint, w in dist.items():
            d = joint[0]
            p1, p2 = pol1(joint, t), pol2(joint, t)
            for a1, a2 in itertools.product((0, 1), repeat=2):
                pr = w * (p1 if a1 else 1 - p1) * (p2 if a2 else 1 - p2)
                if pr == 0:
                    continue
                total += pr * np.array([expected_reward(cfg, d, a1, a2), expected_reward(cfg, d, a2, a1)])
                for s2, q in _next_states(cfg, d, a1, a2):
                    nxt[s2] = nxt.get(s2, 0.0) + pr * q
        dist = nxt
    return (total / cfg.steps).tolist()


def best_stationary(cfg, opp_pol, observe, n_obs):
    """Liệt kê mọi chính sách tất định dừng trên quan sát (16 với (d,c), 256 với (d,c,a_i(t−1)))."""
    best = (-np.inf, None)
    for bits in itertools.product((0, 1), repeat=2 ** n_obs):
        v = exact_value(cfg, table_policy(bits, observe, 0), opp_pol)[0]
        if v > best[0] + 1e-12:
            best = (v, "".join(map(str, bits)))
    return {"value": best[0], "policy": best[1]}


def optimal_value(cfg, opp_pol):
    """Tối ưu thật khi thấy trạng thái chung và biết bước t: quy hoạch động ngược trên 8 × 20 trạng thái."""
    states = list(itertools.product((0, 1), repeat=3))
    v = {s: 0.0 for s in states}
    for t in reversed(range(cfg.steps)):
        v_new = {}
        for s in states:
            p2 = opp_pol(s, t)
            best = -np.inf
            for a1 in (0, 1):
                q = 0.0
                for a2, pa in ((1, p2), (0, 1 - p2)):
                    if pa == 0:
                        continue
                    future = sum(w * v[s2] for s2, w in _next_states(cfg, s[0], a1, a2))
                    q += pa * (expected_reward(cfg, s[0], a1, a2) + future)
                best = max(best, q)
            v_new[s] = best
        v = v_new
    return (0.5 * v[(0, 0, 0)] + 0.5 * v[(1, 0, 0)]) / cfg.steps
