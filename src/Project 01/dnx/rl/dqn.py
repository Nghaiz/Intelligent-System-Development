"""Double DQN: replay trên GPU, mạng đích cập nhật mềm, ε giảm tuyến tính, che hành động không hợp lệ."""
import copy
import time

import torch
import torch.nn.functional as F
from torch import nn

from .env import ScreenEnv, make_state, valid_actions
from .features import N, STOP

STATE_DIM = 2 * N + 1
CFG = {"steps": 8000, "n_env": 512, "replay": 400000, "batch": 1024, "lr": 5e-4, "gamma": 1.0, "tau": 0.005,
       "eps_start": 1.0, "eps_end": 0.02, "eps_frac": 0.6, "warmup": 200, "eval_every": 1000, "eval_rows": 20000}


class QNet(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, N + 1))

    def forward(self, s):
        return self.net(s)


class Replay:
    """Bộ đệm vòng trên GPU; mỗi bước ghi nguyên một lô chuyển trạng thái của n_env môi trường."""

    def __init__(self, cap, dev):
        self.cap, self.i, self.full = cap, 0, False
        self.s = torch.zeros(cap, STATE_DIM, device=dev)
        self.s2 = torch.zeros(cap, STATE_DIM, device=dev)
        self.a = torch.zeros(cap, dtype=torch.long, device=dev)
        self.r = torch.zeros(cap, device=dev)
        self.done = torch.zeros(cap, device=dev)

    def add(self, s, a, r, s2, done):
        n = len(a)
        idx = (torch.arange(n, device=a.device) + self.i) % self.cap
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx] = s, a, r, s2, done.float()
        self.full = self.full or self.i + n >= self.cap
        self.i = (self.i + n) % self.cap

    def sample(self, n, gen):
        hi = self.cap if self.full else self.i
        idx = torch.randint(hi, (n,), device=self.a.device, generator=gen)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


def masked_q(q, state):
    """Gán −∞ cho hành động đã hỏi; mặt nạ nằm ngay trong trạng thái."""
    return q.masked_fill(~valid_actions(state[:, N:2 * N]), -1e9)


@torch.no_grad()
def greedy_episode(qnet, model, x, ask_all=False):
    """Chạy chính sách tham lam tới khi dừng; ask_all cấm dừng để đo đường cong theo số câu."""
    n = len(x)
    mask = torch.zeros_like(x)
    p = model.prob(x, mask)
    probs, active, asked = [p.clone()], torch.ones(n, dtype=torch.bool, device=x.device), []
    for _ in range(N):
        q = masked_q(qnet(make_state(x, mask, p)), make_state(x, mask, p))
        if ask_all:
            q[:, STOP] = -1e9
        a = q.argmax(1)
        ask = active & (a != STOP)
        row = torch.nonzero(ask, as_tuple=True)[0]
        mask[row, a[row]] = 1.0
        asked.append(a.masked_fill(~ask, -1))
        p = p.clone()
        if len(row):
            p[row] = model.prob(x[row], mask[row])
        active = ask
        probs.append(p.clone())
        if not active.any():
            break
    return torch.stack(probs), torch.stack(asked), mask


def train(model, X_tr, X_va, y_va, lam, seed, cfg=CFG, on_eval=None):
    """Huấn luyện một seed; trả trọng số tốt nhất theo mục tiêu trên tập kiểm định và lịch sử học."""
    dev = X_tr.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    env = ScreenEnv(model, X_tr, cfg["n_env"], lam, gen)
    qnet, target = QNet().to(dev), None
    target = copy.deepcopy(qnet)
    opt = torch.optim.Adam(qnet.parameters(), lr=cfg["lr"])
    buf = Replay(cfg["replay"], dev)
    hist, best, best_j, losses, t0 = [], None, -1e9, [], time.time()
    decay = cfg["eps_frac"] * cfg["steps"]
    for t in range(1, cfg["steps"] + 1):
        eps = max(cfg["eps_end"], cfg["eps_start"] + (cfg["eps_end"] - cfg["eps_start"]) * t / decay)
        s = env.state()
        with torch.no_grad():
            a = masked_q(qnet(s), s).argmax(1)
        rand = torch.rand(env.n, device=dev, generator=gen) < eps
        if rand.any():
            noise = torch.rand(env.n, N + 1, device=dev, generator=gen).masked_fill(~valid_actions(s[:, N:2 * N]), -1)
            a[rand] = noise.argmax(1)[rand]
        r, done = env.step(a)
        buf.add(s, a, r, env.state(), done)
        env.reset_where(done)
        if t < cfg["warmup"]:
            continue
        bs, ba, br, bs2, bd = buf.sample(cfg["batch"], gen)
        with torch.no_grad():
            next_a = masked_q(qnet(bs2), bs2).argmax(1)
            next_q = masked_q(target(bs2), bs2).gather(1, next_a[:, None]).squeeze(1)
            y = br + cfg["gamma"] * (1 - bd) * next_q
        loss = F.smooth_l1_loss(qnet(bs).gather(1, ba[:, None]).squeeze(1), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
        opt.step()
        losses.append(loss.detach())
        with torch.no_grad():
            for p, q in zip(target.parameters(), qnet.parameters()):
                p.lerp_(q, cfg["tau"])
        if t % cfg["eval_every"] == 0 or t == cfg["steps"]:
            row = evaluate_objective(qnet, model, X_va, y_va, lam)
            row.update(step=t, eps=eps, loss=float(torch.stack(losses).mean()), seconds=round(time.time() - t0, 1))
            losses = []
            hist.append(row)
            if row["objective"] > best_j:
                best_j, best = row["objective"], copy.deepcopy(qnet.state_dict())
            if on_eval:
                on_eval(row)
    return best, hist


@torch.no_grad()
def evaluate_objective(qnet, model, X, y, lam):
    """Mục tiêu thật trên tập kiểm định: log-score theo nhãn thật trừ chi phí, không dùng thưởng mềm."""
    from .env import log_score
    from .features import COSTS
    n = min(len(X), CFG["eval_rows"])
    x, yy = X[:n], y[:n]
    probs, _, mask = greedy_episode(qnet, model, x)
    cost = (mask * torch.tensor(COSTS, device=x.device)).sum(1).mean().item()
    ll = log_score(yy, probs[-1]).mean().item()
    return {"log_score": ll, "cost": cost, "questions": mask.sum(1).mean().item(), "objective": ll - lam * cost}
