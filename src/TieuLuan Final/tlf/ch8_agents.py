"""Các tác tử: Q-learning dạng bảng, tác tử theo luật của đề bài và các đối thủ cố định dùng cho câu 9–10.

Mọi tác tử có cùng giao diện: observe(joint, seat) -> quan sát, act(obs, explore) -> hành động,
learn(obs, a, r, next_obs, done). Tác tử không học thì learn() không làm gì.
"""
import itertools

import numpy as np

from .ch8_market import DISCOUNT, KEEP


def observe_base(joint, seat):
    """Quan sát của đề bài: s_i = (d_t, a_−i(t−1))."""
    d, p1, p2 = joint
    return d, (p2 if seat == 0 else p1)


def observe_ext(joint, seat):
    """Quan sát mở rộng (câu 10): thêm hành động trước của chính mình, s_i = (d_t, a_−i(t−1), a_i(t−1))."""
    d, p1, p2 = joint
    return (d, p2, p1) if seat == 0 else (d, p1, p2)


class RuleBasedAgent:
    """Luật của đề bài: nhu cầu thấp → giữ giá; nhu cầu cao → giảm giá chỉ khi đối thủ vừa giảm giá."""
    name = "Luật"
    observe = staticmethod(observe_base)

    def act(self, obs, explore=False):
        demand, rival_prev = obs[0], obs[1]
        if demand == 0:
            return KEEP
        return DISCOUNT if rival_prev == DISCOUNT else KEEP

    def learn(self, *args):
        pass


class FixedAgent:
    """Đối thủ cố định cho câu 9–10: luôn giữ giá, luôn giảm giá, ăn miếng trả miếng hoặc ngẫu nhiên."""
    observe = staticmethod(observe_base)

    def __init__(self, kind, seed=0):
        self.kind, self.name = kind, kind
        self.rng = np.random.default_rng(seed)

    def act(self, obs, explore=False):
        if self.kind == "always_keep":
            return KEEP
        if self.kind == "always_discount":
            return DISCOUNT
        if self.kind == "tit_for_tat":
            return obs[1]
        return int(self.rng.integers(2))

    def learn(self, *args):
        pass


class QLearningAgent:
    """Q-learning dạng bảng với ε-greedy, phá hòa ngẫu nhiên; ε giảm theo episode."""
    name = "Q-learning"

    def __init__(self, alpha=0.10, gamma=0.90, eps_start=0.20, eps_min=0.02, eps_decay=0.995,
                 extended=False, seed=0):
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.observe = observe_ext if extended else observe_base
        n_obs = 3 if extended else 2
        self.states = list(itertools.product((0, 1), repeat=n_obs))
        self.q = {s: np.zeros(2) for s in self.states}
        self.visits = {s: np.zeros(2, dtype=int) for s in self.states}
        self.rng = np.random.default_rng(seed)
        self.epsilon = eps_start

    def start_episode(self, episode):
        """ε_e = max(ε_min, ε_0 · ρ^e)."""
        self.epsilon = max(self.eps_min, self.eps_start * self.eps_decay ** episode)

    def greedy(self, obs):
        q = self.q[obs]
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def act(self, obs, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        return self.greedy(obs)

    def learn(self, obs, action, reward, next_obs, done):
        """Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]; ở bước cuối mục tiêu chỉ còn r."""
        target = reward if done else reward + self.gamma * self.q[next_obs].max()
        td_error = target - self.q[obs][action]
        self.q[obs][action] += self.alpha * td_error
        self.visits[obs][action] += 1
        return td_error

    def policy_string(self):
        """Chuỗi hành động tham lam theo thứ tự trạng thái, ví dụ '0000' = không bao giờ giảm giá."""
        return "".join(str(int(np.argmax(self.q[s]))) for s in self.states)
