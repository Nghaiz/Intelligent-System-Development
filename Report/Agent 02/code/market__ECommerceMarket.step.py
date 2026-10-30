# ag2/market.py
def step(self, actions):
    """Thực hiện cặp hành động đồng thời; trả (phần thưởng, doanh số, kết thúc episode)."""
    a1, a2 = actions
    n = customers(self.cfg, self.demand)
    sales, rewards = [], []
    for i, (own, rival) in enumerate(((a1, a2), (a2, a1))):
        p = conversion_probability(self.cfg, self.demand, own, rival)
        s = int(self.rng_sales[i].binomial(n, p))
        sales.append(s)
        rewards.append(s * unit_reward(self.cfg, own))
    self.prev = (a1, a2)
    self.demand = self._next_demand()
    self.t += 1
    return rewards, sales, self.t >= self.cfg.steps
