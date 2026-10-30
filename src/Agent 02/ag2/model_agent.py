"""Tác tử dựa trên mô hình đề xuất cho câu 10: học mô hình thị trường và đối thủ từ kinh nghiệm, rồi lập kế hoạch.

Tác tử quan sát trạng thái chung s = (d, a_i(t−1), a_−i(t−1)) và ước lượng ba thành phần:
  π̂(a_−i | s)        phản ứng của đối thủ (đếm tần suất, tiên nghiệm Laplace),
  R̂(d, a_i, a_−i)    phần thưởng trung bình của từng tổ hợp,
  P̂(d' | d)          quá trình nhu cầu.
Mỗi episode, quy hoạch động ngược trên đúng 20 bước còn lại (không chiết khấu, đúng thước đo đánh giá).
"""
import itertools

import numpy as np

from .agents import observe_ext


class ModelBasedAgent:
    name = "Dựa trên mô hình"
    observe = staticmethod(observe_ext)

    def __init__(self, horizon=20, eps_start=0.20, eps_min=0.02, eps_decay=0.995, seed=0):
        self.horizon = horizon
        self.eps_start, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.states = list(itertools.product((0, 1), repeat=3))   # (d, đối thủ trước, mình trước)
        self.opp_counts = {s: np.ones(2) for s in self.states}
        self.r_sum = np.zeros((2, 2, 2))
        self.r_n = np.zeros((2, 2, 2))
        self.d_counts = np.ones((2, 2))
        self.rng = np.random.default_rng(seed)
        self.epsilon, self.t = eps_start, 0
        self.plan = None

    def start_episode(self, episode):
        self.epsilon = max(self.eps_min, self.eps_start * self.eps_decay ** episode)
        self.t = 0
        self.plan = self.backward_induction()

    def reward_model(self):
        """R̂; tổ hợp chưa gặp lấy trung bình chung để không bị bỏ qua mãi."""
        mean = self.r_sum.sum() / max(self.r_n.sum(), 1.0)
        return np.where(self.r_n > 0, self.r_sum / np.maximum(self.r_n, 1), mean)

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

    def greedy(self, obs, t=None):
        q = self.plan[min(self.t if t is None else t, self.horizon - 1)][obs]
        return int(np.argmax(q))

    def act(self, obs, explore=True):
        if explore and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(2))
        else:
            a = self.greedy(obs)
        self.t += 1
        return a

    def learn(self, obs, action, reward, next_obs, done):
        """Cập nhật mô hình: hành động của đối thủ ở bước t chính là thành phần thứ hai của quan sát kế tiếp."""
        d, opp_action, d_next = obs[0], next_obs[1], next_obs[0]
        self.opp_counts[obs][opp_action] += 1
        self.r_sum[d, action, opp_action] += reward
        self.r_n[d, action, opp_action] += 1
        self.d_counts[d, d_next] += 1
