"""Chương 2: MLP viết tay bằng NumPy, lan truyền ngược và Adam không dựa vào thư viện học sâu.

Kiến trúc trùng với bản Keras và PyTorch: Linear -> ReLU -> Dropout lặp lại, tầng cuối tuyến tính. Đầu ra k
giá trị với softmax cho phân loại, một giá trị cho hồi quy.
"""
import numpy as np


def softmax_ce(z, y, w):
    """Mất mát entropy chéo có trọng số w trên logit z, và đạo hàm theo z."""
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = np.arange(len(y))
    loss = np.mean(w * -np.log(np.maximum(p[n, y], 1e-7)))
    g = p.copy()
    g[n, y] -= 1
    return loss, g * w[:, None]


class NumpyMLP:
    def __init__(self, n_in, n_out, hidden=(128, 64), drop=0.2, seed=0, dtype=np.float32):
        self.rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [self.rng.normal(0, np.sqrt(2 / a), (a, b)).astype(dtype) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, dtype) for b in sizes[1:]]
        self.drop = drop

    def forward(self, X, train=False):
        """Lan truyền thuận; cache giữ đầu vào tầng và mặt nạ ReLU + dropout để đi ngược."""
        cache, h = [], X
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            a = h @ W + b
            if k == len(self.W) - 1:
                cache.append((h, None))
                return a, cache
            mask = (a > 0).astype(a.dtype)
            if train and self.drop > 0:
                mask = mask * (self.rng.random(a.shape) >= self.drop) / (1 - self.drop)
            cache.append((h, mask))
            h = a * mask

    def backward(self, g, cache):
        """Lan truyền ngược: g = dL/da của tầng cuối, nhân chuỗi về từng tầng trước."""
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        for k in reversed(range(len(self.W))):
            h, mask = cache[k]
            if mask is not None:
                g = g * mask
            grads_W[k] = h.T @ g / len(h)
            grads_b[k] = g.mean(axis=0)
            g = g @ self.W[k].T
        return grads_W, grads_b

    def loss_grad(self, X, y, w, train=True):
        a, cache = self.forward(X, train)
        if y.dtype.kind in "iu":
            loss, g = softmax_ce(a, y, w)
        else:
            e = a[:, 0] - y
            loss, g = float(np.mean(e ** 2)), 2 * e[:, None]
        return (loss, *self.backward(g, cache))

    def predict_raw(self, X, bs=8192):
        return np.concatenate([self.forward(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def fit(self, X, y, Xv, yv, w=None, lr=1e-3, epochs=60, bs=512, patience=6):
        """Adam theo mini-batch, dừng sớm theo mất mát validation; trả lịch sử từng epoch."""
        w = (np.ones(len(y)) if w is None else w).astype(self.W[0].dtype)
        params = self.W + self.b
        m1, m2 = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        best, keep, wait, t, hist = np.inf, None, 0, 0, []
        for ep in range(1, epochs + 1):
            order, total = self.rng.permutation(len(X)), 0.0
            for i in range(0, len(X), bs):
                b = order[i:i + bs]
                loss, gW, gb = self.loss_grad(X[b], y[b], w[b])
                total += loss * len(b)
                t += 1
                for j, (p, g) in enumerate(zip(params, gW + gb)):
                    # Cập nhật tại chỗ: không cấp phát mảng mới ở mỗi bước khi bộ nhớ commit eo hẹp.
                    m1[j] *= 0.9
                    m1[j] += 0.1 * g
                    m2[j] *= 0.999
                    m2[j] += 0.001 * g ** 2
                    p -= lr * (m1[j] / (1 - 0.9 ** t)) / (np.sqrt(m2[j] / (1 - 0.999 ** t)) + 1e-8)
            va = self.val_loss(Xv, yv)
            hist.append({"epoch": ep, "train_loss": total / len(X), "val_loss": va})
            if va < best - 1e-5:
                best, keep, wait = va, [p.copy() for p in params], 0
            else:
                wait += 1
                if wait >= patience:
                    break
        for p, q in zip(params, keep):
            p[...] = q
        return hist

    def val_loss(self, X, y):
        a = self.predict_raw(X)
        if y.dtype.kind in "iu":
            return float(softmax_ce(a, y, np.ones(len(y)))[0])
        return float(np.mean((a[:, 0] - y) ** 2))


def grad_check(seed=0, eps=1e-6):
    """So đạo hàm lan truyền ngược với sai phân trung tâm trên mạng nhỏ: ||a-b|| / (||a||+||b||), số tham số."""
    rng = np.random.default_rng(seed)
    X, y, w = rng.normal(size=(40, 6)), rng.integers(0, 3, 40), np.ones(40)
    net = NumpyMLP(6, 3, hidden=(8, 5), drop=0.0, seed=seed, dtype=np.float64)
    # Bias khác 0 để sai phân không rơi đúng vào điểm gãy của ReLU tại a = 0.
    net.b = [p + rng.normal(0, 0.1, p.shape) for p in net.b]
    _, gW, gb = net.loss_grad(X, y, w, train=False)
    numeric, exact = [], []
    for P, G in zip(net.W + net.b, gW + gb):
        for i in np.ndindex(P.shape):
            old = P[i]
            P[i] = old + eps
            up = net.loss_grad(X, y, w, train=False)[0]
            P[i] = old - eps
            down = net.loss_grad(X, y, w, train=False)[0]
            P[i] = old
            numeric.append((up - down) / (2 * eps))
            exact.append(G[i])
    a, b = np.array(numeric), np.array(exact)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))), len(a)
