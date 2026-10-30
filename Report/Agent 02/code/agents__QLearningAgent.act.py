# ag2/agents.py
def act(self, obs, explore=True):
    if explore and self.rng.random() < self.epsilon:
        return int(self.rng.integers(2))
    return self.greedy(obs)
