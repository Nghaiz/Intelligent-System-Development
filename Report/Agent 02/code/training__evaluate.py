# ag2/training.py
def evaluate(cfg, agents, episodes=500, seed=0):
    """Đánh giá đóng băng: tổng thưởng, thưởng TB mỗi bước, tỉ lệ giảm giá, thưởng TB từng episode."""
    env = ECommerceMarket(cfg, seed)
    per_ep, n_disc = [], np.zeros(2)
    for e in range(episodes):
        r, a = run_episode(env, agents, explore=False, learn=False, episode=e)
        per_ep.append(r.mean(axis=0))
        n_disc += a.sum(axis=0)
    per_ep = np.array(per_ep)
    steps = episodes * cfg.steps
    return {"total": (per_ep.sum(axis=0) * cfg.steps).tolist(), "mean": per_ep.mean(axis=0).tolist(),
            "discount_rate": (n_disc / steps).tolist(), "episodes": per_ep, "steps": steps}
