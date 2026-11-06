"""Chương 7: nạp Cora, Amazon Photo, Elliptic; phép chia chống rò rỉ; thống kê mô tả đồ thị.

PyG chỉ dùng để tải và đọc tệp gốc (nạp lười). Phần còn lại là PyTorch thuần trên lớp Graph nhỏ dưới đây.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .common import data_dir

ROOT = data_dir("ch7")
CORA_CLASSES = ["Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
                "Reinforcement_Learning", "Rule_Learning", "Theory"]
ELLIPTIC_CLASSES = ["licit", "illicit"]
UNKNOWN = -1   # nhãn "chưa biết" của Elliptic (gốc mã 2), không bao giờ vào loss hay phép đo
VAL_STEP, TEST_STEP = 30, 35   # Elliptic: train 1-29, val 30-34, test 35-49


@dataclass
class Graph:
    name: str
    x: torch.Tensor
    edge_index: torch.Tensor   # đã đối xứng hóa, 2 × |E|
    y: torch.Tensor
    classes: list
    time: torch.Tensor = None  # bước thời gian, chỉ Elliptic

    @property
    def n(self):
        return self.x.size(0)


@dataclass
class Split:
    train: torch.Tensor        # nút có nhãn dùng tính loss
    val: torch.Tensor
    test: torch.Tensor
    hold_val: torch.Tensor     # nút phải vắng khỏi đồ thị huấn luyện khi học quy nạp
    hold_test: torch.Tensor    # nút phải vắng cả ở lượt chọn mô hình trên val


def row_normalize(x):
    """Chia mỗi hàng cho tổng của nó: biến đổi theo từng nút, không học gì từ test."""
    return x / x.sum(1, keepdim=True).clamp(min=1)


def symmetrize(edge_index):
    """Bỏ tự khuyên, thêm chiều ngược, bỏ cạnh trùng."""
    ei = edge_index[:, edge_index[0] != edge_index[1]]
    return torch.unique(torch.cat([ei, ei.flip(0)], 1), dim=1)


def load_cora():
    from torch_geometric.datasets import Planetoid
    d = Planetoid(str(ROOT / "planetoid"), "Cora")[0]
    g = Graph("cora", row_normalize(d.x), symmetrize(d.edge_index), d.y, CORA_CLASSES)
    return g, Split(d.train_mask, d.val_mask, d.test_mask, d.val_mask, d.test_mask)


def load_photo():
    from torch_geometric.datasets import Amazon
    d = Amazon(str(ROOT / "amazon"), "Photo")[0]
    classes = [f"Lớp {i}" for i in range(int(d.y.max()) + 1)]   # bộ gốc không công bố tên lớp
    return Graph("photo", row_normalize(d.x), symmetrize(d.edge_index), d.y, classes), None


def shchur_split(y, seed, n_train=20, n_val=30):
    """Phép chia của Shchur et al.: 20 nút/lớp train, 30 nút/lớp val, còn lại test."""
    g = torch.Generator().manual_seed(seed)
    train, val = torch.zeros_like(y, dtype=torch.bool), torch.zeros_like(y, dtype=torch.bool)
    for c in y.unique():
        idx = (y == c).nonzero().flatten()
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        train[idx[:n_train]] = True
        val[idx[n_train:n_train + n_val]] = True
    test = ~(train | val)
    return Split(train, val, test, val, test)


def elliptic_ids():
    """txId và bước thời gian đọc thẳng từ CSV gốc (PyG bỏ hai cột này khi xử lý)."""
    import pandas as pd
    raw = ROOT / "elliptic" / "raw"
    ids = pd.read_csv(raw / "elliptic_txs_features.csv", header=None, usecols=[0, 1], names=["tx", "t"])
    cls = pd.read_csv(raw / "elliptic_txs_classes.csv")
    if not (ids["tx"].values == cls["txId"].values).all():
        raise ValueError("Thứ tự giao dịch trong hai tệp CSV Elliptic không khớp, nhãn sẽ lệch")
    return ids["tx"].values, ids["t"].values


def load_elliptic():
    from torch_geometric.datasets import EllipticBitcoinDataset
    d = EllipticBitcoinDataset(str(ROOT / "elliptic"))[0]
    _, t = elliptic_ids()
    time = torch.from_numpy(t.astype(np.int64))
    y = d.y.masked_fill(d.y == 2, UNKNOWN)
    g = Graph("elliptic", d.x, symmetrize(d.edge_index), y, ELLIPTIC_CLASSES, time)
    g.x = standardize_on(g.x, time < VAL_STEP)
    return g, temporal_split(g)


def temporal_split(g):
    """Chia theo thời gian: mô hình chỉ thấy quá khứ, đúng như khi triển khai thật."""
    known, t = g.y != UNKNOWN, g.time
    tr, va, te = t < VAL_STEP, (t >= VAL_STEP) & (t < TEST_STEP), t >= TEST_STEP
    return Split(tr & known, va & known, te & known, va, te)


def random_split(g, ref, seed):
    """Giữ nguyên cỡ ba tập của `ref` nhưng trộn ngẫu nhiên: cố ý rò rỉ thời gian để đo độ lạc quan giả."""
    gen = torch.Generator().manual_seed(seed)
    idx = (g.y != UNKNOWN).nonzero().flatten()
    idx = idx[torch.randperm(idx.numel(), generator=gen)]
    a, b = int(ref.train.sum()), int(ref.train.sum() + ref.val.sum())
    masks = [torch.zeros(g.n, dtype=torch.bool) for _ in range(3)]
    for m, part in zip(masks, (idx[:a], idx[a:b], idx[b:])):
        m[part] = True
    return Split(masks[0], masks[1], masks[2], masks[1], masks[2])


def standardize_on(x, mask):
    """z-score với trung bình, độ lệch tính CHỈ trên `mask` (quá khứ), rồi áp cho toàn bộ."""
    mu, sd = x[mask].mean(0), x[mask].std(0).clamp(min=1e-6)
    return (x - mu) / sd


def load_graph(name):
    return {"cora": load_cora, "photo": load_photo, "elliptic": load_elliptic}[name]()


def splits_for(g, fixed, seed):
    """Cora, Elliptic có phép chia cố định; Photo đổi phép chia theo seed."""
    return shchur_split(g.y, seed) if fixed is None else fixed


# ---------------------------------------------------------------- thống kê
def degrees(edge_index, n):
    return torch.bincount(edge_index[1], minlength=n)


def known_edges(edge_index, y):
    s, d = edge_index
    ok = (y[s] != UNKNOWN) & (y[d] != UNKNOWN)
    return s[ok], d[ok]


def edge_homophily(edge_index, y):
    """Tỷ lệ cạnh nối hai nút cùng nhãn, chỉ tính cạnh có đủ nhãn ở cả hai đầu."""
    s, d = known_edges(edge_index, y)
    return float((y[s] == y[d]).float().mean())


def node_homophily(edge_index, y, n):
    """Với mỗi nút: tỷ lệ láng giềng cùng nhãn; nút không có láng giềng có nhãn nhận NaN."""
    s, d = known_edges(edge_index, y)
    same = torch.zeros(n).index_add_(0, d, (y[s] == y[d]).float())
    total = torch.zeros(n).index_add_(0, d, torch.ones(d.numel()))
    return (same / total).numpy()


def adjusted_homophily(edge_index, y):
    """Homophily trừ mức ngẫu nhiên do mất cân bằng lớp: 0 là không hơn ngẫu nhiên."""
    h = edge_homophily(edge_index, y)
    _, d = known_edges(edge_index, y)
    p = torch.bincount(y[d]).float()
    p = p / p.sum()
    chance = float((p * p).sum())
    return (h - chance) / (1 - chance)


def components(edge_index, n):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    s, d = edge_index.numpy()
    m = coo_matrix((np.ones_like(s, dtype=np.int8), (s, d)), shape=(n, n))
    return connected_components(m, directed=False)
