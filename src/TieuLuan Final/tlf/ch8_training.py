"""Vòng huấn luyện và đánh giá chung cho mọi cặp tác tử (RL–luật, RL–RL, tác tử mới).

Huấn luyện: ε > 0, gọi learn() sau mỗi bước. Đánh giá: môi trường mới với seed riêng, explore=False,
không gọi learn() — bảng Q được đóng băng.
"""
import numpy as np

from .ch8_market import ECommerceMarket


def run_episode(env, agents, explore, learn, episode=0, on_update=None):
    """Một episode 20 bước; trả mảng phần thưởng (T, 2) và hành động (T, 2)."""
    for ag in agents:
        if hasattr(ag, "start_episode"):
            ag.start_episode(episode)
    joint = env.reset()
    rewards, actions = [], []
    done = False
    while not done:
        obs = [ag.observe(joint, i) for i, ag in enumerate(agents)]
        acts = [ag.act(o, explore=explore) for ag, o in zip(agents, obs)]
        r, _, done = env.step(acts)
        joint = env.joint_state()
        if learn:
            for i, ag in enumerate(agents):
                nxt = ag.observe(joint, i)
                td = ag.learn(obs[i], acts[i], r[i], nxt, done)
                if on_update is not None and i == 0:
                    on_update(obs[i], acts[i], r[i], nxt, done, td)
        rewards.append(r)
        actions.append(acts)
    return np.array(rewards, dtype=float), np.array(actions, dtype=int)


def train(cfg, agents, episodes=5000, seed=0, track_q=False, on_update=None):
    """Huấn luyện; lịch sử theo episode: thưởng TB/bước, tỉ lệ giảm giá, ε, (tùy chọn) 8 giá trị Q của ghế 0."""
    env = ECommerceMarket(cfg, seed)
    hist = {"reward": np.zeros((episodes, 2)), "discount": np.zeros((episodes, 2)), "epsilon": np.zeros(episodes),
            "policy": []}
    q_traj = []
    for e in range(episodes):
        r, a = run_episode(env, agents, explore=True, learn=True, episode=e, on_update=on_update)
        hist["reward"][e] = r.mean(axis=0)
        hist["discount"][e] = a.mean(axis=0)
        hist["epsilon"][e] = getattr(agents[0], "epsilon", 0.0)
        if hasattr(agents[0], "policy_string"):
            hist["policy"].append(agents[0].policy_string())
        if track_q:
            q_traj.append(np.concatenate([agents[0].q[s] for s in agents[0].states]))
    if track_q:
        hist["q"] = np.array(q_traj)
    return hist


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
