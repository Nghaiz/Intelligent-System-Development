"""Trò chơi "Mèo bắt chuột": môi trường lưới + chuột học bằng Q-Learning dạng bảng + mèo theo luật.

Cùng quy ước với ag1/qlearning.py: epsilon-greedy (phá hòa ngẫu nhiên), cập nhật Bellman
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)], 4 hành động như ag1/maze.py.
Q-Table lưu bằng list Python (nhanh hơn numpy với mảng 4 phần tử trong vòng lặp thuần Python).
"""
import random
import sys
import time
from collections import deque
from pathlib import Path

AG_ROOT = Path(__file__).resolve().parents[1]
if str(AG_ROOT) not in sys.path:
    sys.path.insert(0, str(AG_ROOT))

from ag1.maze import ACTIONS  # noqa: E402  (Trái, Phải, Lên, Xuống)

WALL, HOLE, CHEESE, MOUSE, CAT = "#", "H", "P", "M", "C"

# Phần thưởng của chuột
R_STEP = -0.2     # mỗi bước: phạt nhẹ để chuột không lang thang
R_BUMP = -1.0     # đâm tường / ra ngoài: đứng yên, phạt nhẹ
R_CHEESE = 10.0   # ăn một miếng phô mai
R_WIN = 50.0      # ăn hết phô mai: thắng, kết thúc ván
R_CAUGHT = -50.0  # bị mèo bắt: thua, kết thúc ván

EPS_MIN = 0.02
GREEDY_NOISE = 0.2  # mèo "Đuổi tham lam" đi ngẫu nhiên với xác suất này

CAT_MODES = {"random": "Ngẫu nhiên", "greedy": "Đuổi tham lam", "bfs": "Săn BFS"}

# Bản đồ: # tường, . trống, M chuột, C mèo, P phô mai, H hang (chỉ chuột vào được)
SCENARIOS = {
    "Cánh đồng trống 6×6": dict(
        layout=["M.....",
                "......",
                "......",
                "......",
                "......",
                "C....P"],
        max_steps=60, episodes=3000,
        note="Không vật cản: chuột phải vòng tránh mèo để tới miếng phô mai ở góc."),
    "Mê cung 8×8": dict(
        layout=["M..#....",
                ".#.#.##.",
                ".#...#..",
                ".##.##.#",
                "........",
                ".#.##.#.",
                ".#..#.#P",
                "...C#..."],
        max_steps=80, episodes=5000,
        note="Hành lang hẹp: mèo tham lam dễ kẹt sau tường, chuột học đi đường vòng."),
    "Các căn phòng 9×9": dict(
        layout=["M...#....",
                "....#....",
                ".........",
                "....#....",
                "##.###.##",
                "....#....",
                "....#..P.",
                ".........",
                "C...#...."],
        max_steps=100, episodes=6000,
        note="Bốn phòng nối bằng cửa: chuột chọn cửa xa mèo để sang phòng có phô mai."),
    "Hang chuột 7×7": dict(
        layout=["M......",
                ".H...H.",
                ".......",
                "...C...",
                ".......",
                ".H...H.",
                "......P"],
        max_steps=80, episodes=5000,
        note="Ô hang H chỉ chuột vào được: chuột học trốn vào hang khi mèo áp sát."),
    "Nhiều miếng phô mai 7×7": dict(
        layout=["P.....M",
                ".#.#.#.",
                ".......",
                ".#.#.#.",
                "P......",
                ".#.#.#.",
                "C.....P"],
        max_steps=120, episodes=12000,
        note="Ba miếng phô mai: trạng thái gồm cả tập phô mai còn lại (2^3 tổ hợp)."),
}


class CatMouseEnv:
    """MDP của chuột: trạng thái = (ô chuột, ô mèo, tập phô mai còn lại), 4 hành động."""

    def __init__(self, layout, cat_mode="greedy", cat_every=1, max_steps=80, seed=0):
        self.layout = list(layout)
        self.n_rows, self.n_cols = len(layout), len(layout[0])
        self.cat_mode, self.cat_every, self.max_steps = cat_mode, cat_every, max_steps
        self.rng = random.Random(seed)
        self.cells = [(r, c) for r in range(self.n_rows) for c in range(self.n_cols) if layout[r][c] != WALL]
        self.idx = {rc: i for i, rc in enumerate(self.cells)}
        self.n = len(self.cells)
        self.is_hole = [layout[r][c] == HOLE for r, c in self.cells]
        find = lambda ch: [self.idx[(r, c)] for r in range(self.n_rows) for c in range(self.n_cols) if layout[r][c] == ch]
        self.mouse_start, self.cat_start = find(MOUSE)[0], find(CAT)[0]
        self.cheeses = find(CHEESE)
        self.cheese_bit = {cell: 1 << k for k, cell in enumerate(self.cheeses)}
        self.n_mask = 1 << len(self.cheeses)
        self.full_mask = self.n_mask - 1
        self.n_states = self.n * self.n * self.n_mask
        self.n_actions = len(ACTIONS)
        # Ô kế tiếp của chuột theo từng hành động (-1 = đâm tường/ra ngoài)
        self.mouse_next = []
        for r, c in self.cells:
            row = []
            for _, dr, dc in ACTIONS:
                row.append(self.idx.get((r + dr, c + dc), -1))
            self.mouse_next.append(row)
        # Ô mèo đi được (không vào hang)
        self.cat_nbrs = [[j for j in nxt if j >= 0 and not self.is_hole[j]] for nxt in self.mouse_next]
        self.dist_to = [self._bfs_from(t) for t in range(self.n)]  # dist_to[đích][ô mèo]
        self.reset()

    def _bfs_from(self, t):
        """Khoảng cách đường đi của mèo từ mọi ô tới ô t (mèo không đi qua hang, trừ chính ô t)."""
        dist = [None] * self.n
        dist[t] = 0
        q = deque([t])
        while q:
            u = q.popleft()
            for v in self.cat_nbrs[u]:
                if dist[v] is None:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def state_index(self, m, c, mask):
        return (m * self.n + c) * self.n_mask + mask

    def reset(self):
        self.m, self.c, self.mask, self.t = self.mouse_start, self.cat_start, self.full_mask, 0
        return self.state_index(self.m, self.c, self.mask)

    def manhattan(self, a, b):
        (r1, c1), (r2, c2) = self.cells[a], self.cells[b]
        return abs(r1 - r2) + abs(c1 - c2)

    def _cat_move(self):
        nbrs = self.cat_nbrs[self.c]
        if not nbrs:
            return
        mode = self.cat_mode
        if mode == "random" or (mode == "greedy" and self.rng.random() < GREEDY_NOISE):
            self.c = self.rng.choice(nbrs)
            return
        dist = self.dist_to[self.m] if mode == "bfs" else None
        if dist is not None and dist[self.c] is None:
            dist = None  # chuột ở chỗ mèo không tới được: quay về đuổi theo Manhattan
        score = [dist[j] if dist is not None else self.manhattan(j, self.m) for j in nbrs]
        best = min(score)
        self.c = self.rng.choice([j for j, s in zip(nbrs, score) if s == best])

    def step(self, a):
        """Chuột đi trước, rồi mèo đi (nếu đến lượt). Trả về (s', r, done, kết cục)."""
        self.t += 1
        nxt = self.mouse_next[self.m][a]
        r = R_STEP
        if nxt < 0:
            r += R_BUMP
        else:
            self.m = nxt
        if self.m == self.c:
            return self.state_index(self.m, self.c, self.mask), R_CAUGHT, True, "caught"
        bit = self.cheese_bit.get(self.m, 0)
        if self.mask & bit:
            self.mask &= ~bit
            r += R_CHEESE
            if self.mask == 0:
                return self.state_index(self.m, self.c, self.mask), r + R_WIN, True, "win"
        if self.t % self.cat_every == 0:
            self._cat_move()
            if self.c == self.m:
                return self.state_index(self.m, self.c, self.mask), R_CAUGHT, True, "caught"
        s = self.state_index(self.m, self.c, self.mask)
        if self.t >= self.max_steps:
            return s, r, True, "timeout"
        return s, r, False, ""

    def snapshot(self):
        return self.m, self.c, self.mask


class MouseAgent:
    """Chuột Q-Learning dạng bảng: epsilon-greedy + cập nhật Bellman (như ag1.qlearning.QLearningAgent)."""

    def __init__(self, n_states, n_actions, alpha=0.2, gamma=0.95, epsilon=1.0, seed=42):
        self.q = [[0.0] * n_actions for _ in range(n_states)]
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def greedy(self, s, q=None):
        row = (q or self.q)[s]
        best = max(row)
        return self.rng.choice([a for a, v in enumerate(row) if v == best])  # phá hòa ngẫu nhiên

    def choose_action(self, s):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self.greedy(s)

    def update(self, s, a, r, s_next, terminal):
        target = r if terminal else r + self.gamma * max(self.q[s_next])
        self.q[s][a] += self.alpha * (target - self.q[s][a])


def train(env, agent, episodes, eps_min=EPS_MIN, progress=None, snapshot_frac=0.1):
    """Huấn luyện chuột; epsilon giảm mũ để chạm eps_min ở 80% số episode.
    Trả về lịch sử (thưởng, số bước, kết cục, epsilon) và bản sao Q-Table lúc mới học (snapshot_frac)."""
    decay = (eps_min / agent.epsilon) ** (1 / max(1, 0.8 * episodes)) if agent.epsilon > eps_min else 1.0
    hist = {"reward": [], "steps": [], "outcome": [], "epsilon": []}
    snap_ep = max(1, int(episodes * snapshot_frac))
    snapshot = None
    tick = max(1, episodes // 100)
    t0 = time.perf_counter()
    for ep in range(1, episodes + 1):
        s, total, done, outcome = env.reset(), 0.0, False, ""
        while not done:
            a = agent.choose_action(s)
            s2, r, done, outcome = env.step(a)
            agent.update(s, a, r, s2, outcome in ("win", "caught"))  # hết giờ không phải trạng thái kết thúc thật
            s, total = s2, total + r
        hist["reward"].append(total)
        hist["steps"].append(env.t)
        hist["outcome"].append(outcome)
        hist["epsilon"].append(agent.epsilon)
        agent.epsilon = max(eps_min, agent.epsilon * decay)
        if ep == snap_ep:
            snapshot = [row[:] for row in agent.q]
        if progress and ep % tick == 0:
            progress(ep / episodes)
    hist["seconds"] = time.perf_counter() - t0
    return hist, snapshot


def make_policy(kind, agent, q=None, seed=0):
    """'learned' = tham lam theo Q; 'snapshot' = tham lam theo Q lúc mới học; 'random' = chuột ngẫu nhiên."""
    rng = random.Random(seed)
    if kind == "random":
        return lambda s: rng.randrange(agent.n_actions)
    table = q if q is not None else agent.q

    def pol(s):
        row = table[s]
        best = max(row)
        return rng.choice([a for a, v in enumerate(row) if v == best])
    return pol


def play(env, policy, record=False):
    """Chơi một ván; trả về (kết cục, tổng thưởng, số bước, danh sách khung (m, c, mask, hành động))."""
    s, total, done, outcome = env.reset(), 0.0, False, ""
    frames = [(*env.snapshot(), None)] if record else None
    while not done:
        a = policy(s)
        s, r, done, outcome = env.step(a)
        total += r
        if record:
            frames.append((*env.snapshot(), a))
    return outcome, total, env.t, frames


def evaluate(make_env, policy, n_games=100):
    """Chạy n_games ván; trả về tỉ lệ thắng / bị bắt / hết giờ, thưởng và số bước trung bình."""
    env = make_env()
    res = {"win": 0, "caught": 0, "timeout": 0}
    rewards, steps = [], []
    for _ in range(n_games):
        outcome, total, t, _ = play(env, policy)
        res[outcome] += 1
        rewards.append(total)
        steps.append(t)
    return {"Thắng": res["win"] / n_games, "Bị bắt": res["caught"] / n_games, "Hết giờ": res["timeout"] / n_games,
            "Thưởng TB": sum(rewards) / n_games, "Số bước TB": sum(steps) / n_games}


def run_experiment(scenario, cat_mode="greedy", cat_every=1, alpha=0.2, gamma=0.95, eps_start=1.0,
                   episodes=None, seed=42, n_eval=100, progress=None):
    """Huấn luyện + đánh giá 3 loại chuột trên cùng kịch bản. Dùng cho app và script kiểm tra."""
    cfg = SCENARIOS[scenario]
    episodes = episodes or cfg["episodes"]

    def make_env(s=seed):
        return CatMouseEnv(cfg["layout"], cat_mode, cat_every, cfg["max_steps"], seed=s)

    env = make_env()
    agent = MouseAgent(env.n_states, env.n_actions, alpha, gamma, eps_start, seed)
    hist, snap = train(env, agent, episodes, progress=progress)
    ev = {
        "Chuột đã học": evaluate(lambda: make_env(seed + 1), make_policy("learned", agent, seed=seed + 1), n_eval),
        "Chuột mới học (10% episode)": evaluate(lambda: make_env(seed + 2),
                                                 make_policy("snapshot", agent, snap, seed=seed + 2), n_eval),
        "Chuột ngẫu nhiên": evaluate(lambda: make_env(seed + 3), make_policy("random", agent, seed=seed + 3), n_eval),
    }
    return {"scenario": scenario, "cat_mode": cat_mode, "cat_every": cat_every, "episodes": episodes,
            "params": dict(alpha=alpha, gamma=gamma, eps_start=eps_start, seed=seed),
            "agent": agent, "snapshot": snap, "hist": hist, "eval": ev, "make_env": make_env}
