# ag2/exact.py
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
