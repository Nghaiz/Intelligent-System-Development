import time

import numpy as np

# Siêu tham số của Q-Learning Agent
ALPHA = 0.1          # tốc độ học
GAMMA = 0.95         # hệ số chiết khấu
EPS_START = 1.0      # epsilon ban đầu: thăm dò hoàn toàn
EPS_MIN = 0.01       # epsilon tối thiểu
EPS_DECAY = 0.998    # epsilon nhân với hệ số này sau mỗi episode
EPISODES = 3000
MAX_STEPS = 300      # giới hạn bước mỗi episode (guardrail chống vòng lặp vô hạn)
LOG_EVERY = 250
SEED = 42


class QLearningAgent:
    """Learning Agent dạng bảng: Q-Table là tri thức, epsilon-greedy là Problem Generator,
    cập nhật Bellman là Learning Element."""

    def __init__(self, n_states, n_actions, alpha=ALPHA, gamma=GAMMA, epsilon=EPS_START, seed=SEED):
        self.q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def choose_action(self, s):
        """epsilon-greedy: thăm dò với xác suất epsilon, ngược lại khai thác argmax Q(s, a)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        best = np.flatnonzero(self.q[s] == self.q[s].max())
        return int(self.rng.choice(best))  # phá hòa ngẫu nhiên giữa các hành động bằng nhau

    def update(self, s, a, r, s_next, done):
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]"""
        target = r if done else r + self.gamma * self.q[s_next].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)


def train(env, agent, episodes=EPISODES, max_steps=MAX_STEPS, log_every=LOG_EVERY):
    """Huấn luyện qua nhiều episode; trả về lịch sử phần thưởng, số bước, epsilon, cờ về đích."""
    hist = {"reward": [], "steps": [], "epsilon": [], "success": []}
    t0 = time.perf_counter()
    for ep in range(1, episodes + 1):
        state, total, done = env.reset(), 0.0, False
        s = env.to_index(state)
        for step in range(1, max_steps + 1):
            a = agent.choose_action(s)
            state, r, done = env.step(a)
            s_next = env.to_index(state)
            agent.update(s, a, r, s_next, done)
            s, total = s_next, total + r
            if done:
                break
        hist["reward"].append(total)
        hist["steps"].append(step)
        hist["epsilon"].append(agent.epsilon)
        hist["success"].append(done)
        agent.decay_epsilon()
        if ep % log_every == 0:
            w = slice(ep - log_every, ep)
            print(f"Episode {ep:5d} | epsilon={hist['epsilon'][-1]:.3f} | "
                  f"thưởng TB={np.mean(hist['reward'][w]):8.2f} | bước TB={np.mean(hist['steps'][w]):6.1f} | "
                  f"về đích={100 * np.mean(hist['success'][w]):5.1f}%")
    hist["seconds"] = time.perf_counter() - t0
    return hist


def greedy_path(env, agent, max_steps=MAX_STEPS):
    """Chạy chính sách tham lam pi*(s) = argmax_a Q(s, a) từ S; trả về lộ trình và tổng phần thưởng."""
    state, total, path = env.reset(), 0.0, [env.start]
    for _ in range(max_steps):
        a = int(np.argmax(agent.q[env.to_index(state)]))
        state, r, done = env.step(a)
        total += r
        path.append(state)
        if done:
            break
    return path, total
