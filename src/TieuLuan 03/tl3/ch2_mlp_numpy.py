"""Chương 2 — MLP cài đặt từ đầu bằng NumPy (lan truyền ngược viết tay, tối ưu Adam, dừng sớm)."""
import numpy as np

from .common import SEED


def bce_logits(z, y, pos_weight=1.0):
    """Entropy chéo nhị phân trên logit z; pos_weight bù mất cân bằng lớp."""
    return np.mean(pos_weight * y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))


class NumpyMLP:
    """Mạng Linear + ReLU nhiều tầng, đầu ra một logit (phân loại) hoặc một giá trị (hồi quy)."""

    def __init__(self, n_in, hidden=(128, 64, 32), task="clf", pos_weight=1.0, seed=SEED):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]  # khởi tạo He
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.task, self.pos_weight = task, pos_weight

    def forward(self, X):
        """Lan truyền thuận; giữ lại đầu vào của từng tầng để lan truyền ngược."""
        cache, h = [], X
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            cache.append(h)
            h = h @ W + b
            if k < len(self.W) - 1:
                h = np.maximum(h, 0)
        return h[:, 0], cache

    def loss(self, X, y):
        z = self.forward(X)[0]
        return bce_logits(z, y, self.pos_weight) if self.task == "clf" else float(np.mean((z - y) ** 2))

    def backward(self, z, cache, y):
        """Lan truyền ngược: từ đạo hàm mất mát theo đầu ra, nhân dần qua từng tầng về đầu vào."""
        if self.task == "clf":
            p = 1 / (1 + np.exp(-z))
            g = (self.pos_weight * y * (p - 1) + (1 - y) * p)[:, None]
        else:
            g = 2 * (z - y)[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for k in reversed(range(len(self.W))):
            gW[k] = cache[k].T @ g / len(y)
            gb[k] = g.mean(axis=0)
            g = g @ self.W[k].T * (cache[k] > 0) if k else g
        return gW, gb

    def fit(self, X, y, Xv, yv, lr=1e-3, epochs=60, bs=512, patience=8):
        """Adam theo mini-batch, dừng sớm theo mất mát Validation; trả về số epoch đã chạy."""
        rng = np.random.default_rng(SEED)
        params = self.W + self.b
        m1, m2 = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        best, best_params, wait, t = np.inf, None, 0, 0
        for ep in range(1, epochs + 1):
            idx = rng.permutation(len(X))
            for i in range(0, len(X), bs):
                b = idx[i:i + bs]
                z, cache = self.forward(X[b])
                gW, gb = self.backward(z, cache, y[b])
                t += 1
                for j, (p, g) in enumerate(zip(params, gW + gb)):
                    m1[j] = 0.9 * m1[j] + 0.1 * g
                    m2[j] = 0.999 * m2[j] + 0.001 * g ** 2
                    p -= lr * (m1[j] / (1 - 0.9 ** t)) / (np.sqrt(m2[j] / (1 - 0.999 ** t)) + 1e-8)
            va = self.loss(Xv, yv)
            if va < best - 1e-5:
                best, best_params, wait = va, [p.copy() for p in params], 0
            else:
                wait += 1
                if wait >= patience:
                    break
        for p, q in zip(params, best_params):
            p[...] = q
        return ep

    def predict_raw(self, X):
        return self.forward(X)[0]
