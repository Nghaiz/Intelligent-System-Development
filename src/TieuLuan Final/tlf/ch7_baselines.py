"""Chương 7: Label Propagation (Zhou et al.) chỉ dùng cạnh, không dùng đặc trưng nút."""
import numpy as np
import scipy.sparse as sp


def sym_norm(edge_index, n):
    """S = D^(-1/2) A D^(-1/2) dạng thưa; không thêm tự khuyên vì nhãn gốc đã giữ lại bằng hạng (1-α)Y."""
    s, d = edge_index
    a = sp.coo_matrix((np.ones(s.size, dtype=np.float32), (s, d)), shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(1)).ravel()
    inv = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
    return sp.diags(inv) @ a @ sp.diags(inv)


def label_propagation(s_mat, y, train, n_cls, alpha=0.9, iters=50):
    """F ← α S F + (1-α) Y. Trả F; hàng toàn 0 là nút không có đường đi tới nút có nhãn nào."""
    y0 = np.zeros((y.size, n_cls), dtype=np.float32)
    y0[train, y[train]] = 1.0
    f = y0.copy()
    for _ in range(iters):
        f = alpha * (s_mat @ f) + (1 - alpha) * y0
    return f


def predict_lp(f, y, train):
    """argmax F; nút không được lan tới nhận lớp đa số của train, và báo tỷ lệ này ra ngoài."""
    reached = f.sum(1) > 0
    pred = f.argmax(1)
    pred[~reached] = np.bincount(y[train]).argmax()
    return pred, reached
