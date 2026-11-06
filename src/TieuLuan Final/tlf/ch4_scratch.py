"""Chương 4: RNN viết bằng NumPy theo lô, BPTT tự tính, kiểm tra gradient bằng sai phân và Adam.

h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h),   ŷ = W_hy h_T + b_y,   L = trung bình (ŷ − y)²
"""
import numpy as np


class RNNScratch:
    """RNN một lớp, đầu ra nhiều bước từ trạng thái ẩn cuối."""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(n_hidden)
        self.p = {"Wxh": rng.uniform(-s, s, (n_in, n_hidden)), "Whh": rng.uniform(-s, s, (n_hidden, n_hidden)),
                  "bh": np.zeros(n_hidden), "Why": rng.uniform(-s, s, (n_hidden, n_out)), "by": np.zeros(n_out)}

    def forward(self, x):
        """x: (lô, T, D). Giữ mọi h_t vì BPTT cần chúng khi đi ngược."""
        p = self.p
        hs = [np.zeros((x.shape[0], p["Whh"].shape[0]))]
        for t in range(x.shape[1]):
            hs.append(np.tanh(x[:, t] @ p["Wxh"] + hs[-1] @ p["Whh"] + p["bh"]))
        self.x, self.hs = x, hs
        return hs[-1] @ p["Why"] + p["by"]

    def backward(self, yhat, y):
        """BPTT: δ_T = ∂L/∂h_T, δ_{t−1} = [(1 − h_t²) ⊙ δ_t] Whhᵀ; cộng dồn gradient của trọng số dùng chung."""
        p, hs, x = self.p, self.hs, self.x
        b, k = y.shape
        g = {name: np.zeros_like(v) for name, v in p.items()}
        dy = 2 * (yhat - y) / (b * k)
        g["Why"], g["by"] = hs[-1].T @ dy, dy.sum(0)
        delta = dy @ p["Why"].T
        self.delta_norms = []
        for t in range(x.shape[1], 0, -1):
            self.delta_norms.append(float(np.linalg.norm(delta) / b))
            da = (1.0 - hs[t] ** 2) * delta
            g["Wxh"] += x[:, t - 1].T @ da
            g["Whh"] += hs[t - 1].T @ da
            g["bh"] += da.sum(0)
            delta = da @ p["Whh"].T
        return g

    def loss(self, x, y):
        return float(np.mean((self.forward(x) - y) ** 2))


def grad_check(model, x, y, eps=1e-5, n_checks=10, seed=0):
    """So gradient BPTT với sai phân trung tâm trên các phần tử ngẫu nhiên; trả sai số tương đối lớn nhất mỗi khối."""
    rng = np.random.default_rng(seed)
    g = model.backward(model.forward(x), y)
    rel = {}
    for name, w in model.p.items():
        errs = []
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + eps
            hi = model.loss(x, y)
            w[idx] = old - eps
            lo = model.loss(x, y)
            w[idx] = old
            num = (hi - lo) / (2 * eps)
            errs.append(abs(num - g[name][idx]) / max(1e-12, abs(num) + abs(g[name][idx])))
        rel[name] = float(np.max(errs))
    return rel


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.lr, self.b, self.eps, self.t = params, lr, betas, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, g, clip=1.0):
        """Cắt chuẩn toàn cục rồi cập nhật Adam tại chỗ."""
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in self.p:
            self.m[k] = self.b[0] * self.m[k] + (1 - self.b[0]) * g[k] * scale
            self.v[k] = self.b[1] * self.v[k] + (1 - self.b[1]) * (g[k] * scale) ** 2
            mh, vh = self.m[k] / (1 - self.b[0] ** self.t), self.v[k] / (1 - self.b[1] ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(model, X, Y, Xv, Yv, epochs=40, batch=64, lr=1e-3, patience=8, seed=0):
    """Huấn luyện theo lô, dừng sớm theo MSE validation; khôi phục tham số tốt nhất."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr)
    hist = {"loss": [], "val_loss": []}
    best, keep, bad = np.inf, None, 0
    for _ in range(epochs):
        order, total = rng.permutation(len(X)), 0.0
        for i in range(0, len(X), batch):
            idx = order[i:i + batch]
            yhat = model.forward(X[idx])
            total += float(np.mean((yhat - Y[idx]) ** 2)) * len(idx)
            opt.step(model.backward(yhat, Y[idx]))
        va = model.loss(Xv, Yv)
        hist["loss"].append(total / len(X))
        hist["val_loss"].append(va)
        if va < best:
            best, keep, bad = va, {k: v.copy() for k, v in model.p.items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.p.update(keep)
    return hist
