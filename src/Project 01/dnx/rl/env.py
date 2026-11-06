"""MDP sàng lọc thích ứng, chạy song song nhiều bản ghi thật trên GPU.

Trạng thái: đáp án đã có, mặt nạ, logit xác suất hiện tại. Hành động: hỏi đặc trưng k chưa hỏi, hoặc dừng.
Thưởng: tổng thưởng mỗi tập bằng log-score cuối trừ log-score lúc đầu trừ chi phí hỏi. Phần log-score được chia theo
từng bước (r_t = Δ log-score − λ·c_k), đây là biến đổi theo thế năng nên không đổi chính sách tối ưu, và tính kỳ vọng
theo nhãn với xác suất của mô hình đủ 21 đặc trưng để giảm phương sai; đánh giá cuối cùng vẫn dùng nhãn thật.
"""
import torch

from .features import COSTS, N, STOP

EPS = 1e-6
LOGIT_SCALE = 0.25
REWARD_SCALE = 100.0  # log-score trung bình rất nhỏ, nhân lên cho mạng dễ học; không đổi chính sách


def make_state(x, mask, p):
    logit = torch.logit(p.clamp(EPS, 1 - EPS)) * LOGIT_SCALE
    return torch.cat([x * mask, mask, logit.unsqueeze(-1)], dim=-1)


def valid_actions(mask):
    """Mặt nạ hành động hợp lệ: chưa hỏi hoặc dừng."""
    stop = torch.ones(mask.shape[0], 1, device=mask.device, dtype=torch.bool)
    return torch.cat([mask < 0.5, stop], dim=-1)


def log_score(p, q):
    """Kỳ vọng log q(y) khi y ~ Bernoulli(p); p là nhãn thật 0/1 khi chấm điểm."""
    q = q.clamp(EPS, 1 - EPS)
    return p * torch.log(q) + (1 - p) * torch.log(1 - q)


class ScreenEnv:
    @torch.no_grad()
    def __init__(self, model, X, n_env, lam, generator=None):
        self.model, self.X, self.n, self.lam = model, X, n_env, lam
        self.dev = X.device
        self.costs = torch.tensor(COSTS, device=self.dev)
        self.gen = generator
        full = torch.ones_like(X)
        self.p_full = torch.cat([model.prob(b, m) for b, m in zip(X.split(65536), full.split(65536))])
        self.p0 = model.prob(X[:1], torch.zeros_like(X[:1])).item()
        self.idx = torch.zeros(n_env, dtype=torch.long, device=self.dev)
        self.mask = torch.zeros(n_env, N, device=self.dev)
        self.p = torch.full((n_env,), self.p0, device=self.dev)
        self.reset_where(torch.ones(n_env, dtype=torch.bool, device=self.dev))

    def reset_where(self, done):
        k = int(done.sum())
        if k:
            self.idx[done] = torch.randint(len(self.X), (k,), device=self.dev, generator=self.gen)
            self.mask[done] = 0.0
            self.p[done] = self.p0

    def state(self):
        return make_state(self.X[self.idx], self.mask, self.p)

    @torch.no_grad()
    def step(self, action):
        """Trả (thưởng, kết thúc); hành động dừng có thưởng 0 vì log-score đã được cộng dần."""
        ask = action != STOP
        row = torch.nonzero(ask, as_tuple=True)[0]
        r = torch.zeros(self.n, device=self.dev)
        if len(row):
            a = action[row]
            self.mask[row, a] = 1.0
            x = self.X[self.idx[row]]
            q = self.model.prob(x, self.mask[row])
            target = self.p_full[self.idx[row]]
            r[row] = (log_score(target, q) - log_score(target, self.p[row]) - self.lam * self.costs[a]) * REWARD_SCALE
            self.p[row] = q
        done = (~ask) | (self.mask.sum(1) >= N)
        return r, done
