import numpy as np
from collections import deque

# Mê cung 12 x 12: S xuất phát, G đích, # ô tường lửa (Fire), . ô trống
LAYOUT = [
    "S..#.....#..",
    ".#.#.###.#..",
    ".#...#.....#",
    ".###.#.###..",
    "...#...#....",
    "##.###.#.##.",
    "....#..#..#.",
    ".##.#.##.##.",
    "...#.....#..",
    "##.#.##.##.#",
    "....#.#.....",
    ".##...#.##.G",
]

# Bốn hành động rời rạc: (tên, dịch chuyển hàng, dịch chuyển cột)
ACTIONS = [("Trái", 0, -1), ("Phải", 0, 1), ("Lên", -1, 0), ("Xuống", 1, 0)]

R_FIRE = -10.0   # đâm vào tường lửa hoặc ra khỏi mê cung: bị phạt và đứng yên
R_GOAL = +100.0  # về đích: kết thúc episode
R_STEP = -1.0    # mỗi bước đi hợp lệ: phạt nhẹ để tác tử chọn đường ngắn


class MazeEnv:
    """Môi trường mê cung lưới: trạng thái là tọa độ (r, c), 4 hành động, phần thưởng theo ô đích đến."""

    def __init__(self, layout=LAYOUT):
        self.grid = np.array([list(row) for row in layout])
        self.n_rows, self.n_cols = self.grid.shape
        self.start = tuple(int(v) for v in np.argwhere(self.grid == "S")[0])
        self.goal = tuple(int(v) for v in np.argwhere(self.grid == "G")[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(ACTIONS)
        self.state = self.start

    def is_fire(self, r, c):
        return self.grid[r, c] == "#"

    def to_index(self, state):
        return state[0] * self.n_cols + state[1]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        """Thực hiện hành động; trả về (trạng thái mới, phần thưởng, kết thúc?)."""
        _, dr, dc = ACTIONS[action]
        r, c = self.state[0] + dr, self.state[1] + dc
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or self.is_fire(r, c):
            return self.state, R_FIRE, False
        self.state = (r, c)
        if self.state == self.goal:
            return self.state, R_GOAL, True
        return self.state, R_STEP, False

    def shortest_path_length(self):
        """BFS trên mê cung (tác tử không biết hàm này) để kiểm chứng lộ trình học được có ngắn nhất không."""
        dist = {self.start: 0}
        queue = deque([self.start])
        while queue:
            r, c = queue.popleft()
            for _, dr, dc in ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and not self.is_fire(nr, nc) \
                        and (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    queue.append((nr, nc))
        return dist.get(self.goal)
