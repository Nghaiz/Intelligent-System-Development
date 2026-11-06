"""Chương 7: GCN hai lớp viết từ đầu bằng NumPy. Không autograd, không ma trận thưa: đạo hàm tính tay."""
import numpy as np


def normalized_adj(edge_index, n):
    """Â = D̃^(-1/2) (A + I) D̃^(-1/2), dạng dày n × n."""
    a = np.eye(n, dtype=np.float32)
    a[edge_index[0], edge_index[1]] = 1.0
    d = a.sum(1) ** -0.5
    return d[:, None] * a * d[None, :]


def glorot(d_in, d_out, rng):
    lim = np.sqrt(6.0 / (d_in + d_out))
    return rng.uniform(-lim, lim, (d_in, d_out)).astype(np.float32)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class GCNNumpy:
    """H¹ = ReLU(Â X W₀); Z = Â H¹ W₁. Dropout ở X và H¹, L2 chỉ ở W₀ như Kipf & Welling."""

    def __init__(self, d_in, d_hid, n_cls, seed=0, dropout=0.5, wd=5e-4):
        self.rng = np.random.default_rng(seed)
        self.W = [glorot(d_in, d_hid, self.rng), glorot(d_hid, n_cls, self.rng)]
        self.p, self.wd = dropout, wd

    def _drop(self, h, on):
        if not on or self.p == 0:
            return h, 1.0
        keep = (self.rng.random(h.shape) >= self.p).astype(np.float32) / (1 - self.p)
        return h * keep, keep

    def forward(self, A, X, train=False):
        h0, k0 = self._drop(X, train)
        z1 = A @ (h0 @ self.W[0])
        h1, k1 = self._drop(np.maximum(z1, 0), train)
        return A @ (h1 @ self.W[1]), (h0, z1, h1, k1)

    def loss_grads(self, A, X, y, mask, train=True):
        """Lan truyền ngược bằng tay: G = ∂L/∂Z, rồi đi ngược từng phép nhân ma trận."""
        logits, (h0, z1, h1, k1) = self.forward(A, X, train)
        p = softmax(logits)
        n = mask.sum()
        loss = -np.log(p[mask, y[mask]] + 1e-12).mean() + 0.5 * self.wd * (self.W[0] ** 2).sum()
        one_hot = np.eye(p.shape[1], dtype=np.float32)[y]
        g = (p - one_hot) * mask[:, None] / n        # ∂L/∂Z, n × c
        at_g = A.T @ g
        grad1 = h1.T @ at_g                          # ∂L/∂W₁ = H¹ᵀ Âᵀ G
        g_h1 = (at_g @ self.W[1].T) * k1             # ngược qua dropout H¹
        g_z1 = g_h1 * (z1 > 0)                       # ngược qua ReLU
        grad0 = h0.T @ (A.T @ g_z1) + self.wd * self.W[0]   # ∂L/∂W₀
        return loss, [grad0, grad1]


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train_numpy(A, X, y, tr, va, te, d_hid=64, seed=0, epochs=200, lr=0.01):
    """Huấn luyện toàn bộ đồ thị, giữ trọng số tốt nhất trên val, chỉ chạm test một lần ở cuối."""
    net = GCNNumpy(X.shape[1], d_hid, int(y.max()) + 1, seed)
    opt = Adam(net.W, lr)
    best, keep, hist = -1.0, None, []
    for ep in range(epochs):
        loss, grads = net.loss_grads(A, X, y, tr)
        opt.step(grads)
        pred = net.forward(A, X)[0].argmax(1)
        v = float((pred[va] == y[va]).mean())
        hist.append({"epoch": ep + 1, "loss": float(loss), "val": v})
        if v > best:
            best, keep = v, [w.copy() for w in net.W]
    net.W = keep
    pred = net.forward(A, X)[0].argmax(1)
    return {"pred": pred, "val": best, "test": float((pred[te] == y[te]).mean()), "history": hist}


def grad_check(seed=0, n=12, d=6, h=4, c=3, eps=1e-3):
    """So đạo hàm tính tay với sai phân trung tâm trên đồ thị ngẫu nhiên nhỏ, dropout tắt, float64."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n)) < 0.25
    ei = np.stack(np.nonzero(a | a.T))
    A, X = normalized_adj(ei, n).astype(np.float64), rng.random((n, d))
    y, mask = rng.integers(0, c, n), np.ones(n, dtype=bool)
    net = GCNNumpy(d, h, c, seed, dropout=0.0)
    net.W = [w.astype(np.float64) for w in net.W]
    _, grads = net.loss_grads(A, X, y, mask, train=False)
    worst = 0.0
    for w, g in zip(net.W, grads):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i, j] += eps
                up = net.loss_grads(A, X, y, mask, train=False)[0]
                w[i, j] -= 2 * eps
                dn = net.loss_grads(A, X, y, mask, train=False)[0]
                w[i, j] += eps
                worst = max(worst, abs((up - dn) / (2 * eps) - g[i, j]))
    return worst
