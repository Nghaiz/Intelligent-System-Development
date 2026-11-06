"""Các đối chứng của chính sách hỏi: thứ tự cố định, ngẫu nhiên, tham lam theo thông tin, Q-learning bảng."""
import numpy as np
import torch

from .env import ScreenEnv, valid_actions
from .features import COSTS, N, STOP

MAX_VALUES = 6
EPS = 1e-6


def entropy(p):
    p = p.clamp(EPS, 1 - EPS)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


@torch.no_grad()
def trace(select, model, x):
    """Chạy một chính sách tới đủ 21 câu; select(x, mask, p) trả chỉ số đặc trưng. Trả (K+1, n) xác suất và thứ tự."""
    mask = torch.zeros_like(x)
    p = _prob(model, x, mask)
    probs, order = [p], []
    rows = torch.arange(len(x), device=x.device)
    for _ in range(N):
        a = select(x, mask, p)
        mask[rows, a] = 1.0
        p = _prob(model, x, mask)
        probs.append(p)
        order.append(a)
    return torch.stack(probs), torch.stack(order)


def _prob(model, x, mask, chunk=200000):
    return torch.cat([model.prob(a, b) for a, b in zip(x.split(chunk), mask.split(chunk))])


def fixed_order(rank):
    """Luôn hỏi đặc trưng chưa hỏi đứng đầu danh sách rank."""
    score = torch.zeros(N)
    score[torch.as_tensor(rank)] = torch.arange(N, 0, -1).float()

    def select(x, mask, p):
        return (score.to(x.device) * (1 - mask) - 1e6 * mask).argmax(1)
    return select


def random_order(seed):
    gen = torch.Generator().manual_seed(seed)

    def select(x, mask, p):
        noise = torch.rand(x.shape, generator=gen).to(x.device)
        return noise.masked_fill(mask > 0.5, -1).argmax(1)
    return select


@torch.no_grad()
def importance_rank(model, x, y):
    """Xếp hạng toàn cục bằng chọn tiến: mỗi bước thêm đặc trưng làm tăng log-score trên tập kiểm định nhiều nhất."""
    from .env import log_score
    chosen, mask = [], torch.zeros_like(x)
    for _ in range(N):
        best, best_v = None, -1e9
        for k in range(N):
            if k in chosen:
                continue
            m = mask.clone()
            m[:, k] = 1.0
            v = log_score(y, _prob(model, x, m)).mean().item()
            if v > best_v:
                best, best_v = k, v
        chosen.append(best)
        mask[:, best] = 1.0
    return chosen


class GreedyInfo:
    """Chọn đặc trưng có độ giảm entropy kỳ vọng lớn nhất. Phân phối đáp án xấp xỉ bằng hỗn hợp P(v|y) theo p hiện tại,
    các giá trị liên tục rút về tối đa MAX_VALUES phân vị. per_cost=True chia lợi ích cho chi phí."""

    def __init__(self, x_std, x_raw, y, per_cost=False):
        self.per_cost = per_cost
        self.costs = torch.tensor(COSTS, device=x_std.device)
        self.values, self.p1, self.p0 = [], [], []
        pos = y > 0.5
        for k in range(N):
            uniq = torch.unique(x_std[:, k])
            if len(uniq) > MAX_VALUES:
                uniq = torch.unique(torch.quantile(x_std[:, k], torch.linspace(0.05, 0.95, MAX_VALUES, device=x_std.device)))
            bins = (x_std[:, k, None] - uniq).abs().argmin(1)
            count = lambda b: (torch.bincount(b, minlength=len(uniq)).float() + 1) / (len(b) + len(uniq))
            self.values.append(uniq)
            self.p1.append(count(bins[pos]))
            self.p0.append(count(bins[~pos]))

    @torch.no_grad()
    def gains(self, model, x, mask, p):
        out = torch.full(x.shape, -1e9, device=x.device)
        base = entropy(p)
        for k in range(N):
            todo = torch.nonzero(mask[:, k] < 0.5, as_tuple=True)[0]
            if not len(todo):
                continue
            exp_h = torch.zeros(len(todo), device=x.device)
            for j, v in enumerate(self.values[k]):
                xx, mm = x[todo].clone(), mask[todo].clone()
                xx[:, k], mm[:, k] = v, 1.0
                w = p[todo] * self.p1[k][j] + (1 - p[todo]) * self.p0[k][j]
                exp_h += w * entropy(_prob(model, xx, mm))
            out[todo, k] = base[todo] - exp_h
        return out

    def selector(self, model):
        def select(x, mask, p):
            g = self.gains(model, x, mask, p)
            if self.per_cost:
                g = torch.where(g > -1e8, g / self.costs, g)
            return g.argmax(1)
        return select


class TabularQ:
    """Q-learning bảng trên trạng thái rời rạc: BINS mức xác suất × đã hỏi hay chưa TOP đặc trưng quan trọng nhất."""
    BINS, TOP = 12, 6

    def __init__(self, rank, dev):
        self.top = torch.as_tensor(rank[: self.TOP], device=dev)
        self.pow = 2 ** torch.arange(self.TOP, device=dev)
        self.q = torch.zeros(self.BINS * 2 ** self.TOP, N + 1, device=dev)

    def index(self, mask, p):
        logit = torch.logit(p.clamp(EPS, 1 - EPS))
        b = ((logit + 5) / 6 * self.BINS).long().clamp(0, self.BINS - 1)  # logit −5…+1 chia đều, phần lớn dữ liệu ở đây
        return b + self.BINS * (mask[:, self.top].long() * self.pow).sum(1)

    def act(self, mask, p):
        return self.q[self.index(mask, p)].masked_fill(~valid_actions(mask), -1e9).argmax(1)

    @torch.no_grad()
    def fit(self, model, X, lam, seed, steps=20000, n_env=512, alpha=0.05, eps_frac=0.7):
        gen = torch.Generator(device=X.device).manual_seed(seed)
        env = ScreenEnv(model, X, n_env, lam, gen)
        for t in range(steps):
            eps = max(0.02, 1 - t / (eps_frac * steps))
            s, vm = self.index(env.mask, env.p), valid_actions(env.mask)
            a = self.q[s].masked_fill(~vm, -1e9).argmax(1)
            rand = torch.rand(n_env, device=X.device, generator=gen) < eps
            noise = torch.rand(n_env, N + 1, device=X.device, generator=gen).masked_fill(~vm, -1)
            a[rand] = noise.argmax(1)[rand]
            r, done = env.step(a)
            s2 = self.index(env.mask, env.p)
            nxt = self.q[s2].masked_fill(~valid_actions(env.mask), -1e9).max(1).values
            y = r + (1 - done.float()) * nxt
            # cập nhật gộp theo ô (s,a): nhiều môi trường trùng ô thì lấy trung bình để alpha giữ đúng ý nghĩa
            cell = s * (N + 1) + a
            total = torch.zeros(self.q.numel(), device=X.device).index_add_(0, cell, y)
            count = torch.zeros(self.q.numel(), device=X.device).index_add_(0, cell, torch.ones_like(y))
            hit = count > 0
            flat = self.q.view(-1)
            flat[hit] += alpha * (total[hit] / count[hit] - flat[hit])
            env.reset_where(done)

    @torch.no_grad()
    def episode(self, model, x):
        n, mask = len(x), torch.zeros_like(x)
        p, active = _prob(model, x, mask), torch.ones(n, dtype=torch.bool, device=x.device)
        for _ in range(N):
            a = self.act(mask, p)
            ask = active & (a != STOP)
            row = torch.nonzero(ask, as_tuple=True)[0]
            if not len(row):
                break
            mask[row, a[row]] = 1.0
            p = p.clone()
            p[row] = model.prob(x[row], mask[row])
            active = ask
        return p, mask


def np_float(t):
    return t.detach().cpu().numpy().astype(np.float64)
