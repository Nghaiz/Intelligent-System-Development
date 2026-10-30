# ag2/model_agent.py
def backward_induction(self):
    """V_T = 0; Q_t(s,a) = Σ_a' π̂(a'|s) [R̂(d,a,a') + Σ_d' P̂(d'|d) V_{t+1}(d', a', a)]."""
    r_hat = self.reward_model()
    p_d = self.d_counts / self.d_counts.sum(axis=1, keepdims=True)
    v_next = {s: 0.0 for s in self.states}
    plan = []
    for _ in range(self.horizon):
        q_t, v_t = {}, {}
        for s in self.states:
            d = s[0]
            pi = self.opp_counts[s] / self.opp_counts[s].sum()
            q = np.zeros(2)
            for a in (0, 1):
                for b in (0, 1):
                    future = sum(p_d[d, d2] * v_next[(d2, b, a)] for d2 in (0, 1))
                    q[a] += pi[b] * (r_hat[d, a, b] + future)
            q_t[s], v_t[s] = q, q.max()
        plan.append(q_t)
        v_next = v_t
    return plan[::-1]
