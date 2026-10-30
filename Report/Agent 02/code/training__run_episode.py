# ag2/training.py
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
