"""Chương 7: GCN, GraphSAGE, GAT, MLP viết bằng PyTorch thuần và vòng huấn luyện chống rò rỉ.

Đồ thị là edge_index 2 × |E|; mọi phép gộp láng giềng viết bằng index_add_ hoặc ma trận thưa để lộ rõ cơ chế
truyền thông điệp. Không import PyG, nên Web App chạy được chỉ với torch.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "cora": dict(hidden=64, lr=0.01, wd=5e-4, dropout=0.5, epochs=200, patience=30, heads=8, sample=10, every=1),
    "photo": dict(hidden=64, lr=0.01, wd=5e-5, dropout=0.5, epochs=600, patience=120, heads=8, sample=25, every=1),
    "elliptic": dict(hidden=128, lr=0.005, wd=1e-5, dropout=0.2, epochs=400, patience=12, heads=8, sample=10,
                     every=5),
}
ACT = {"gat": F.elu}


def self_loops(edge_index, n):
    loops = torch.arange(n, device=edge_index.device)
    return torch.cat([edge_index, torch.stack([loops, loops])], 1)


def gcn_norm_adj(edge_index, n):
    """Â = D̃^(-1/2)(A + I)D̃^(-1/2), thưa."""
    ei = self_loops(edge_index, n)
    one = torch.ones(ei.size(1), device=ei.device)
    deg = torch.zeros(n, device=ei.device).index_add_(0, ei[1], one)
    w = deg.pow(-0.5)
    w = w[ei[0]] * w[ei[1]]
    return torch.sparse_coo_tensor(ei.flip(0), w, (n, n)).coalesce()


def scatter_softmax(score, index, n):
    """Softmax điểm cạnh theo từng nút đích: mỗi nút tự chia 1 cho các láng giềng."""
    idx = index.view(-1, 1).expand_as(score)
    top = torch.full((n, score.size(1)), -1e9, device=score.device)
    top = top.scatter_reduce(0, idx, score, "amax")
    e = (score - top[index]).exp()
    total = torch.zeros_like(top).index_add_(0, index, e)
    return e / total[index].clamp(min=1e-16)


def sample_neighbors(edge_index, k):
    """Giữ ngẫu nhiên tối đa k cạnh vào mỗi nút đích (GraphSAGE, chỉ dùng lúc huấn luyện)."""
    dst = edge_index[1]
    order = torch.rand(dst.numel(), device=dst.device).argsort()
    order = order[dst[order].argsort(stable=True)]
    d = dst[order]
    first = torch.cat([torch.ones(1, dtype=torch.bool, device=d.device), d[1:] != d[:-1]])
    start = first.nonzero().flatten()
    rank = torch.arange(d.numel(), device=d.device) - start[first.long().cumsum(0) - 1]
    return edge_index[:, order[rank < k]]


class GCNLayer(nn.Module):
    """H' = Â H W: lấy trung bình có trọng số theo bậc của chính nút và láng giềng."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.lin(x)) + self.bias


class SAGELayer(nn.Module):
    """GraphSAGE-mean: h' = W₁ h + W₂ mean{h_u : u ∈ N(v)}; tách riêng trọng số của chính nút."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_neigh = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, edge_index):
        src, dst = edge_index
        n = x.size(0)
        agg = torch.zeros(n, x.size(1), device=x.device).index_add_(0, dst, x[src])
        one = torch.ones_like(dst, dtype=x.dtype)
        deg = torch.zeros(n, device=x.device).index_add_(0, dst, one)
        return self.lin_self(x) + self.lin_neigh(agg / deg.clamp(min=1).unsqueeze(1))


class GATLayer(nn.Module):
    """GAT: α_vu = softmax_u LeakyReLU(aᵀ[Wh_v ‖ Wh_u]), h_v' = ‖_k Σ_u α^k_vu W^k h_u."""

    def __init__(self, d_in, d_out, heads=8, concat=True, dropout=0.6):
        super().__init__()
        self.heads, self.d_out, self.concat, self.dropout = heads, d_out, concat, dropout
        self.lin = nn.Linear(d_in, heads * d_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, d_out))
        self.a_dst = nn.Parameter(torch.empty(heads, d_out))
        self.bias = nn.Parameter(torch.zeros(heads * d_out if concat else d_out))
        for p in (self.lin.weight, self.a_src, self.a_dst):
            nn.init.xavier_uniform_(p)
        self.last_alpha = None

    def forward(self, x, edge_index):
        n = x.size(0)
        ei = self_loops(edge_index, n)
        src, dst = ei
        h = self.lin(x).view(n, self.heads, self.d_out)
        e = (h * self.a_src).sum(-1)[src] + (h * self.a_dst).sum(-1)[dst]
        alpha = scatter_softmax(F.leaky_relu(e, 0.2), dst, n)
        self.last_alpha = (ei.detach(), alpha.detach())
        drop = F.dropout(alpha, self.dropout, self.training)
        out = torch.zeros_like(h).index_add_(0, dst, h[src] * drop.unsqueeze(-1))
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        return out + self.bias


def make_layer(kind, d_in, d_out, last, heads, dropout):
    if kind == "gcn":
        return GCNLayer(d_in, d_out)
    if kind == "sage":
        return SAGELayer(d_in, d_out)
    if kind == "gat":
        d = d_out if last else d_out // heads
        return GATLayer(d_in, d, 1 if last else heads, not last, dropout)
    return nn.Linear(d_in, d_out)


class GNN(nn.Module):
    """Chồng L lớp cùng loại (gcn | sage | gat | mlp), dropout giữa các lớp, tùy chọn nối tắt."""

    def __init__(self, kind, d_in, d_hid, n_cls, layers=2, dropout=0.5, heads=8, residual=False,
                 sample=None):
        super().__init__()
        self.kind, self.dropout, self.residual, self.sample = kind, dropout, residual, sample
        dims = [d_in] + [d_hid] * (layers - 1) + [n_cls]
        self.layers = nn.ModuleList(
            make_layer(kind, dims[i], dims[i + 1], i == layers - 1, heads, dropout) for i in range(layers))
        self.hidden = None

    def forward(self, x, graph):
        if self.kind == "sage" and self.training and self.sample:
            graph = sample_neighbors(graph, self.sample)
        act = ACT.get(self.kind, F.relu)
        for i, layer in enumerate(self.layers):
            x_in = x
            x = F.dropout(x, self.dropout, self.training)
            x = layer(x) if self.kind == "mlp" else layer(x, graph)
            if i < len(self.layers) - 1:
                x = act(x)
                if self.residual and x.shape == x_in.shape:
                    x = x + x_in
                self.hidden = x
        return x


def graph_input(kind, edge_index, n):
    """GCN cần Â thưa; SAGE, GAT dùng edge_index; MLP không dùng đồ thị."""
    return gcn_norm_adj(edge_index, n) if kind == "gcn" else edge_index


def keep_nodes(edge_index, drop):
    """Bỏ mọi cạnh chạm nút trong tập `drop`: đây là chỗ chặn rò rỉ cấu trúc."""
    return edge_index[:, ~(drop[edge_index[0]] | drop[edge_index[1]])]


def graph_views(kind, g, split, inductive, device):
    """Ba góc nhìn đồ thị. Quy nạp: train không thấy val, test; val không thấy test; test thấy tất cả."""
    ei = g.edge_index.to(device)
    full = graph_input(kind, ei, g.n)
    if not inductive:
        return full, full, full
    hv, ht = split.hold_val.to(device), split.hold_test.to(device)
    train = graph_input(kind, keep_nodes(ei, hv | ht), g.n)
    return train, graph_input(kind, keep_nodes(ei, ht), g.n), full


def macro_f1(pred, y, n_cls):
    f1 = []
    for c in range(n_cls):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return sum(f1) / n_cls


def scores(pred, y, n_cls, positive=None):
    """accuracy, f1_macro; nếu có lớp dương thì thêm precision, recall, F1 của riêng lớp đó."""
    out = {"accuracy": (pred == y).float().mean().item(), "f1_macro": macro_f1(pred, y, n_cls)}
    if positive is not None:
        tp = ((pred == positive) & (y == positive)).sum().item()
        fp = ((pred == positive) & (y != positive)).sum().item()
        fn = ((pred != positive) & (y == positive)).sum().item()
        out["illicit_precision"] = tp / max(tp + fp, 1)
        out["illicit_recall"] = tp / max(tp + fn, 1)
        out["illicit_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return out


def class_weight(y, n_cls):
    """Trọng số lớp nghịch đảo tần suất trên train, dùng cho đồ thị lệch lớp nặng."""
    w = 1.0 / torch.bincount(y, minlength=n_cls).float().clamp(min=1)
    return w / w.sum() * n_cls


def build(kind, g, cfg, n_cls, layers=2, residual=False):
    drop = 0.6 if kind == "gat" and g.name != "elliptic" else cfg["dropout"]
    return GNN(kind, g.x.size(1), cfg["hidden"], n_cls, layers, drop, cfg["heads"], residual, cfg["sample"])


def train_epochs(model, x, y, split, views, cfg, weight, n_cls):
    """Vòng lặp: loss trên nút train, chọn trọng số theo f1_macro của val, dừng sớm."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    g_tr, g_va, _ = views
    best, keep, bad, hist = -1.0, None, 0, []
    for ep in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, g_tr)[split.train], y[split.train], weight=weight)
        loss.backward()
        opt.step()
        if (ep + 1) % cfg["every"]:
            continue
        model.eval()
        with torch.no_grad():
            pred = model(x, g_va).argmax(1)
        v = macro_f1(pred[split.val], y[split.val], n_cls)
        hist.append({"epoch": ep + 1, "loss": loss.item(), "val_f1": v})
        bad = 0 if v > best else bad + 1
        if v > best:
            best = v
            keep = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(keep)
    return best, hist


def fit(kind, g, split, cfg, seed, inductive=False, device="cpu", layers=2, residual=False):
    """Huấn luyện một lượt, chỉ đo test đúng một lần ở cuối, với trọng số đã chọn trên val."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_cls = len(g.classes)
    positive = 1 if g.name == "elliptic" else None
    model = build(kind, g, cfg, n_cls, layers, residual).to(device)
    views = graph_views(kind, g, split, inductive, device)
    x, y = g.x.to(device), g.y.to(device)
    masks = type(split)(*(m.to(device) for m in (split.train, split.val, split.test,
                                                 split.hold_val, split.hold_test)))
    weight = class_weight(y[masks.train], n_cls) if positive is not None else None
    t0 = time.perf_counter()
    best_val, hist = train_epochs(model, x, y, masks, views, cfg, weight, n_cls)
    seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        logits = model(x, views[2])
    res = scores(logits[masks.test].argmax(1), y[masks.test], n_cls, positive)
    res.update(val_f1=best_val, seconds=seconds, epochs=len(hist), history=hist, model=model,
               params=sum(p.numel() for p in model.parameters()), logits=logits.cpu(),
               hidden=None if model.hidden is None else model.hidden.detach().cpu())
    return res
