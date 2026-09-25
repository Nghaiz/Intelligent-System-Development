"""Mục 1 — các khái niệm cơ bản của CNN viết dưới dạng hàm NumPy.

Mỗi tầng của mạng là một hàm f_l. Một CNN là hợp của các hàm đó:
    f = f_L ∘ ... ∘ f_2 ∘ f_1
Tệp này chỉ dùng NumPy để mọi phép tính đều nhìn thấy được; notebook 00
đối chiếu từng hàm với PyTorch bằng np.allclose.
"""
from functools import reduce

import numpy as np


# ---------------------------------------------------------------- neuron, hợp hàm
def neuron(x, w, b):
    """Neuron tuyến tính: z = w · x + b (x, w là vectơ, b là số)."""
    return np.dot(w, x) + b


def compose(*fs):
    """Hợp hàm theo thứ tự áp dụng: compose(f1, f2, f3)(x) = f3(f2(f1(x)))."""
    return lambda x: reduce(lambda acc, f: f(acc), fs, x)


def affine(W, b):
    """Trả về hàm affine x -> W x + b."""
    return lambda x: W @ x + b


def collapse_affine(W1, b1, W2, b2):
    """Hai tầng affine liên tiếp gộp lại vẫn là một tầng affine:
    W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)."""
    return W2 @ W1, W2 @ b1 + b2


# ---------------------------------------------------------------- hàm kích hoạt
def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


# ---------------------------------------------------------------- tích chập
def out_size(H, K, P, S):
    """Kích thước đầu ra của tích chập/pooling: floor((H + 2P - K) / S) + 1."""
    return (H + 2 * P - K) // S + 1


def conv2d(X, K, b=None, stride=1, pad=0):
    """Tích chập 2D (thực chất là cross-correlation, giống nn.Conv2d).

    X: (C, H, W) · K: (F, C, kh, kw) · b: (F,) · trả về (F, Ho, Wo).
    Mỗi ô đầu ra là tích vô hướng giữa bộ lọc và một cửa sổ của đầu vào.
    """
    F, C, kh, kw = K.shape
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad)))
    Ho = out_size(X.shape[1], kh, pad, stride)
    Wo = out_size(X.shape[2], kw, pad, stride)
    Y = np.zeros((F, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            patch = Xp[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
            Y[:, i, j] = np.tensordot(K, patch, axes=([1, 2, 3], [0, 1, 2]))
    if b is not None:
        Y += b[:, None, None]
    return Y


def conv2d_backward(X, K, dY, stride=1, pad=0):
    """Lan truyền ngược của conv2d theo quy tắc chuỗi.

    Mỗi ô đầu ra Y[f,i,j] phụ thuộc tuyến tính vào một cửa sổ của X và vào K[f],
    nên gradient chỉ là cộng dồn dY[f,i,j] nhân với phần tử tương ứng.
    Trả về (dX, dK, db).
    """
    F, C, kh, kw = K.shape
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad)))
    dXp = np.zeros_like(Xp)
    dK = np.zeros_like(K)
    _, Ho, Wo = dY.shape
    for i in range(Ho):
        for j in range(Wo):
            rs, cs = i * stride, j * stride
            patch = Xp[:, rs:rs + kh, cs:cs + kw]
            dK += dY[:, i, j][:, None, None, None] * patch[None]
            dXp[:, rs:rs + kh, cs:cs + kw] += np.tensordot(dY[:, i, j], K, axes=(0, 0))
    dX = dXp[:, pad:pad + X.shape[1], pad:pad + X.shape[2]]
    return dX, dK, dY.sum(axis=(1, 2))


def count_params_dense(n_in, n_out):
    """Tầng Dense nối đầy đủ: n_in · n_out trọng số + n_out bias."""
    return n_in * n_out + n_out


def count_params_conv(c_in, c_out, k):
    """Tầng Conv k×k: số tham số không phụ thuộc kích thước ảnh."""
    return c_in * c_out * k * k + c_out


# ---------------------------------------------------------------- pooling, đầu ra
def maxpool2d(X, size=2, stride=2):
    """Max pooling trên từng kênh: X (C, H, W) -> (C, Ho, Wo)."""
    C, H, W = X.shape
    Ho, Wo = out_size(H, size, 0, stride), out_size(W, size, 0, stride)
    Y = np.empty((C, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            win = X[:, i * stride:i * stride + size, j * stride:j * stride + size]
            Y[:, i, j] = win.max(axis=(1, 2))
    return Y


def flatten(X):
    return X.reshape(-1)


def softmax(z):
    """Softmax ổn định số: trừ max trước khi lấy mũ, kết quả không đổi."""
    e = np.exp(z - np.max(z))
    return e / e.sum()


def cross_entropy(z, y):
    """Cross-entropy từ logits z với nhãn nguyên y, dùng log-sum-exp ổn định."""
    m = np.max(z)
    return m + np.log(np.exp(z - m).sum()) - z[y]


# ---------------------------------------------------------------- CNN là hợp hàm
def tiny_cnn_forward(X, p):
    """CNN nhỏ viết thẳng thành hợp hàm, ghi lại shape sau từng hàm.

    f = softmax ∘ dense ∘ flatten ∘ maxpool ∘ relu ∘ conv
    p: dict có 'K', 'b' (conv) và 'W', 'c' (dense). Trả về (logits, trace).
    """
    layers = [
        ("conv", lambda x: conv2d(x, p["K"], p["b"], stride=1, pad=1)),
        ("relu", relu),
        ("maxpool", maxpool2d),
        ("flatten", flatten),
        ("dense", lambda x: p["W"] @ x + p["c"]),
    ]
    trace = [("input", X.shape)]
    for name, f in layers:
        X = f(X)
        trace.append((name, X.shape))
    return X, trace


# ---------------------------------------------------------------- kiểm tra đạo hàm
def grad_check(f, x, grad, eps=1e-6):
    """So gradient giải tích với sai phân trung tâm (f(x+e) - f(x-e)) / 2e.

    Trả về sai số tương đối lớn nhất ||g_số - g_giải_tích|| / ||g_số + g_giải_tích||.
    """
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        num[idx] = (fp - fm) / (2 * eps)
    return np.linalg.norm(num - grad) / max(np.linalg.norm(num + grad), 1e-12)
