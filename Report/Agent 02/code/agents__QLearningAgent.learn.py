# ag2/agents.py
def learn(self, obs, action, reward, next_obs, done):
    """Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]; ở bước cuối mục tiêu chỉ còn r."""
    target = reward if done else reward + self.gamma * self.q[next_obs].max()
    td_error = target - self.q[obs][action]
    self.q[obs][action] += self.alpha * td_error
    self.visits[obs][action] += 1
    return td_error
