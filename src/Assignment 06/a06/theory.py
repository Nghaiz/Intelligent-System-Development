"""Hạng mục 1 (Ch.1): các khái niệm của mạng hồi quy viết thành hàm NumPy.

Mỗi hàm có một phép đối chiếu với lớp PyTorch tương ứng trong notebook 00 (np.allclose).
Quy ước thứ tự cổng theo PyTorch: LSTM (i, f, g, o), GRU (r, z, n).
"""
import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------- vì sao cần RNN: đếm tham số
def count_params_mlp(T, D, H):
    """MLP nhận cả chuỗi làm phẳng T·D: số trọng số tăng tuyến tính theo độ dài chuỗi."""
    return T * D * H + H


def count_params_conv1d(K, D_in, D_out):
    """Conv1D: dùng chung kernel theo thời gian nhưng chỉ nhìn K bước một lúc."""
    return K * D_in * D_out + D_out


def count_params_rnn(D, H):
    """Simple RNN: W_xh (H×D), W_hh (H×H), b (H), không phụ thuộc T."""
    return H * (D + H + 1)


def receptive_field(L, K, dilated=False):
    """Số bước quá khứ một nơ-ron ở tầng L của Conv1D nhìn thấy.

    Thường: 1 + L(K-1), tăng tuyến tính theo số tầng. Giãn nở 2^l (TCN): 1 + (K-1)(2^L - 1).
    Dù loại nào cũng là một hằng số chốt lúc thiết kế; RNN không có giới hạn này.
    """
    return 1 + (K - 1) * (2 ** L - 1) if dilated else 1 + L * (K - 1)


def future_dependence(conv, T=10, t=4):
    """Đầu ra ở bước t có phụ thuộc đầu vào ở các bước SAU t không? Trả về chuẩn gradient theo x_{t+1..T}.

    Khác 0 nghĩa là lớp tích chập nhìn thấy tương lai (rò rỉ khi dự báo chuỗi thời gian).
    """
    x = torch.randn(1, conv.in_channels, T, requires_grad=True)
    conv(x)[0, :, t].sum().backward()
    return x.grad[0, :, t + 1:].norm().item()


class CausalConv1d(nn.Module):
    """Tích chập nhân quả: đệm K-1 giá trị bên TRÁI, không đệm bên phải, nên y_t chỉ dùng x_{t-K+1..t}."""

    def __init__(self, c_in, c_out, K):
        super().__init__()
        self.pad = K - 1
        self.in_channels = c_in
        self.conv = nn.Conv1d(c_in, c_out, K)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


# ---------------------------------------------------------------- Simple RNN
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rnn_forward(X, W_xh, W_hh, b, h0=None):
    """h_t = tanh(W_xh x_t + W_hh h_{t-1} + b). X: (T, D) -> (T, H), cùng W ở mọi bước."""
    h = np.zeros(W_hh.shape[0]) if h0 is None else h0
    out = []
    for x_t in X:
        h = np.tanh(W_xh @ x_t + W_hh @ h + b)
        out.append(h)
    return np.stack(out)


# ---------------------------------------------------------------- BPTT: gradient chảy ngược theo thời gian
def jacobian_product_norm(rho, T, n=32, seed=0):
    """‖(W_hh)^t v‖ / ‖v‖ với W_hh ngẫu nhiên có bán kính phổ rho (bỏ qua tanh để thấy riêng vai trò của W).

    rho < 1: suy giảm theo hàm mũ (triệt tiêu); rho > 1: tăng theo hàm mũ (bùng nổ).
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    norms = []
    for _ in range(T):
        norms.append(np.linalg.norm(v))
        v = W.T @ v
    return np.array(norms)


def grad_norms_through_time(cell="rnn", T=100, D=8, H=32, seed=0, forget_bias=0.0):
    """‖∂L/∂h_t‖ với L chỉ phụ thuộc h_T, lấy bằng autograd trên một chuỗi ngẫu nhiên.

    Mở vòng lặp bằng RNNCell/LSTMCell để giữ gradient của từng h_t. forget_bias > 0 đẩy cổng quên
    f_t về gần 1 (Jozefowicz và cộng sự, 2015), để đường cộng của cell state thực sự giữ gradient.
    """
    torch.manual_seed(seed)
    layer = nn.RNNCell(D, H) if cell == "rnn" else nn.LSTMCell(D, H)
    if cell == "lstm":
        with torch.no_grad():
            layer.bias_ih[H:2 * H] = forget_bias   # khối thứ hai (i, f, g, o) là cổng quên
            layer.bias_hh[H:2 * H] = 0.0
    x = torch.randn(T, 1, D)
    h = torch.zeros(1, H)
    c = torch.zeros(1, H)
    hs = []
    for t in range(T):
        if cell == "rnn":
            h = layer(x[t], h)
        else:
            h, c = layer(x[t], (h, c))
        h.retain_grad()
        hs.append(h)
    hs[-1].sum().backward()
    return np.array([hh.grad.norm().item() for hh in hs])


def clip_grad_norm(grads, max_norm):
    """Co toàn bộ gradient về chuẩn max_norm nếu vượt, GIỮ NGUYÊN HƯỚNG. Trả về (grads mới, chuẩn cũ)."""
    total = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    return [g * scale for g in grads], total


# ---------------------------------------------------------------- LSTM và GRU
def lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh):
    """Một bước LSTM. W_ih (4H, D), W_hh (4H, H) xếp khối theo thứ tự cổng i, f, g, o như PyTorch."""
    z = W_ih @ x + b_ih + W_hh @ h + b_hh
    i, f, g, o = np.split(z, 4)
    i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_new = f * c + i * g           # cell state: cộng có cổng, gradient đi thẳng qua f
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    """Một bước GRU, thứ tự khối r, z, n như PyTorch; h' là tổ hợp lồi của h cũ và ứng viên n."""
    gi = W_ih @ x + b_ih
    gh = W_hh @ h + b_hh
    ir, iz, in_ = np.split(gi, 3)
    hr, hz, hn = np.split(gh, 3)
    r = sigmoid(ir + hr)
    z = sigmoid(iz + hz)
    n = np.tanh(in_ + r * hn)
    return (1 - z) * n + z * h


# ---------------------------------------------------------------- điều chuẩn
def recurrent_dropout_masks(T, H, p, seed=0):
    """Dropout thường: mặt nạ mới ở mỗi bước. Recurrent dropout (Gal & Ghahramani): một mặt nạ cho cả chuỗi."""
    rng = np.random.default_rng(seed)
    standard = (rng.random((T, H)) >= p).astype(float)
    fixed = np.tile((rng.random(H) >= p).astype(float), (T, 1))
    return standard, fixed


def layer_norm(x, gamma, beta, eps=1e-5):
    """Chuẩn hoá theo chiều đặc trưng của TỪNG mẫu (không phụ thuộc lô, hợp với RNN)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
