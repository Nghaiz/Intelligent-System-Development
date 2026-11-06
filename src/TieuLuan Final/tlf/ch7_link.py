"""Chương 7: dự đoán liên kết trên đồ thị Hetionet đọc từ Neo4j (làn Ch.5 đã nạp).

Nút có nhiều kiểu, nên mẫu âm được tráo đuôi trong CÙNG kiểu nút: dễ hơn sẽ làm AUC đẹp giả. Cạnh chia theo CẶP nút
không thứ tự: mọi quan hệ của một cặp đi chung một tập, nếu không cạnh song song sẽ lộ đáp án qua đồ thị truyền tin.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Q_EDGES = """MATCH (a:Entity {dataset:'hetionet'})-[r]->(b:Entity {dataset:'hetionet'})
WHERE r.metaedge IS NOT NULL
RETURN a.id AS s, [l IN labels(a) WHERE l <> 'Entity'][0] AS ks, r.metaedge AS m,
       b.id AS t, [l IN labels(b) WHERE l <> 'Entity'][0] AS kt, a.name AS ns, b.name AS nt"""


def read_neo4j():
    from .ch5_neo import cypher
    return [(r["s"], r["ks"], r["m"], r["t"], r["kt"], r["ns"], r["nt"]) for r in cypher(Q_EDGES)]


def read_file():
    """Dự phòng khi Neo4j không chạy: đúng luật chọn tiểu đồ thị của Ch.5, đọc từ tệp Hetionet."""
    from .ch5_kg import hetionet_edges
    import pandas as pd
    from .ch5_data import RAW
    names = pd.read_csv(RAW / "hetionet-nodes.tsv", sep="\t").set_index("id")["name"].astype(str).to_dict()
    kept, _, _ = hetionet_edges()
    kind = lambda x: x.split("::")[0].replace(" ", "")   # noqa: E731
    return [(f"H:{s}", kind(s), m, f"H:{t}", kind(t), names[s], names[t]) for s, m, t in kept]


def load_triples():
    """Trả (mảng triples, mô tả nguồn). Thử Neo4j trước, hỏng thì đọc tệp và ghi rõ đã dùng đường nào."""
    try:
        rows, source = read_neo4j(), "neo4j"
        if not rows:
            raise ValueError("Neo4j chưa có cạnh Hetionet")
    except Exception as e:  # noqa: BLE001 - mọi lỗi kết nối đều dẫn về đường tệp, và ghi lý do ra ngoài
        rows, source = read_file(), f"file ({type(e).__name__})"
    return encode(rows), source


def encode(rows):
    nodes, kinds, names, rels = {}, [], [], {}
    triples = []
    for s, ks, m, t, kt, ns, nt in rows:
        for v, k, nm in ((s, ks, ns), (t, kt, nt)):
            if v not in nodes:
                nodes[v] = len(nodes)
                kinds.append(k)
                names.append(nm)
        triples.append((nodes[s], rels.setdefault(m, len(rels)), nodes[t]))
    uniq = sorted(set(kinds))
    return {"triples": np.array(sorted(set(triples)), dtype=np.int64), "kinds": uniq,
            "kind": np.array([uniq.index(k) for k in kinds]), "names": names, "rels": list(rels),
            "ids": list(nodes)}


def split_pairs(triples, seed, val=0.1, test=0.1):
    """Chia 80/10/10 theo cặp nút không thứ tự, chống rò rỉ qua cạnh song song và cạnh ngược."""
    pair = np.minimum(triples[:, 0], triples[:, 2]) * 100000 + np.maximum(triples[:, 0], triples[:, 2])
    uniq, inv = np.unique(pair, return_inverse=True)
    order = np.random.default_rng(seed).permutation(uniq.size)
    fold = np.zeros(uniq.size, dtype=np.int8)
    fold[order[:int(test * uniq.size)]] = 2
    fold[order[int(test * uniq.size):int((test + val) * uniq.size)]] = 1
    f = fold[inv]
    return triples[f == 0], triples[f == 1], triples[f == 2]


def message_graph(train, n_rels):
    """Thêm chiều ngược với mã quan hệ r + n_rels để tin nhắn đi được hai chiều."""
    s, r, t = train.T
    src = np.concatenate([s, t])
    dst = np.concatenate([t, s])
    rel = np.concatenate([r, r + n_rels])
    return (torch.from_numpy(src), torch.from_numpy(dst), torch.from_numpy(rel))


class RGCNLayer(nn.Module):
    """h'_v = W₀ h_v + Σ_(u,r) c_uv (h_u ⊙ d_r): mỗi quan hệ một vector đường chéo d_r, c_uv chuẩn hóa theo bậc."""

    def __init__(self, dim, n_rel):
        super().__init__()
        self.rel = nn.Embedding(n_rel, dim)
        self.self_lin = nn.Linear(dim, dim)
        nn.init.ones_(self.rel.weight)

    def forward(self, h, graph):
        src, dst, rel, c = graph
        msg = h[src] * self.rel(rel) * c.unsqueeze(1)
        return self.self_lin(h) + torch.zeros_like(h).index_add_(0, dst, msg)


class GCNEncoder(nn.Module):
    """Bỏ kiểu quan hệ, chỉ giữ cấu trúc: bản đối chứng cho thấy kiểu cạnh có đáng giá không."""

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, h, graph):
        src, dst, _, c = graph
        msg = self.lin(h)[src] * c.unsqueeze(1)
        return h + torch.zeros_like(h).index_add_(0, dst, msg)


class LinkModel(nn.Module):
    """Vector nhúng nút học được, 0 hoặc 2 lớp truyền tin, giải mã DistMult: σ(zₛᵀ diag(r) zₜ)."""

    def __init__(self, kind, n, n_kinds, n_rel, node_kind, dim=128, dropout=0.2):
        super().__init__()
        self.kind_name, self.dropout = kind, dropout
        self.emb = nn.Embedding(n, dim)
        self.kind_emb = nn.Embedding(n_kinds, dim)
        self.register_buffer("node_kind", torch.as_tensor(node_kind))
        self.decoder = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.emb.weight)
        make = {"rgcn": lambda: RGCNLayer(dim, 2 * n_rel), "gcn": lambda: GCNEncoder(dim)}.get(kind)
        self.layers = nn.ModuleList([make() for _ in range(2)] if make else [])

    def encode(self, graph):
        h = self.emb.weight + self.kind_emb(self.node_kind)
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, self.dropout, self.training)
            h = layer(h, graph)
            if i == 0:
                h = F.relu(h)
        return h

    def score(self, z, t):
        return (z[t[:, 0]] * self.decoder(t[:, 1]) * z[t[:, 2]]).sum(1)


def degree_weights(graph, n):
    src, dst, rel = graph
    one = torch.ones_like(src, dtype=torch.float)
    deg = torch.zeros(n).index_add_(0, dst, one)
    c = (deg[src] * deg[dst]).clamp(min=1).pow(-0.5)
    return src, dst, rel, c


class Negatives:
    """Mẫu âm tráo đuôi (hoặc đầu) sang nút CÙNG KIỂU; lọc mọi bộ ba đã biết ở cả ba tập."""

    def __init__(self, triples, kind):
        self.kind = kind
        self.pool = {k: np.nonzero(kind == k)[0] for k in np.unique(kind)}
        self.known = {tuple(t) for t in triples.tolist()}

    def sample(self, pos, rng, per_pos=1):
        out = []
        for s, r, t in np.repeat(pos, per_pos, axis=0).tolist():
            head = rng.random() < 0.5
            pool = self.pool[self.kind[s if head else t]]
            for _ in range(20):
                c = int(pool[rng.integers(pool.size)])
                cand = (c, r, t) if head else (s, r, c)
                if cand not in self.known:
                    break
            out.append(cand)
        return np.array(out, dtype=np.int64)


def adamic_adar(train, n, pairs):
    """Heuristic không học: tổng 1/log(bậc) của láng giềng chung, trên đồ thị train vô hướng."""
    import scipy.sparse as sp
    s, t = train[:, 0], train[:, 2]
    a = sp.coo_matrix((np.ones(2 * s.size), (np.r_[s, t], np.r_[t, s])), shape=(n, n)).tocsr()
    a.data[:] = 1
    deg = np.asarray(a.sum(1)).ravel()
    w = 1 / np.log(np.maximum(deg, 2))
    b = a @ sp.diags(w)
    return np.asarray(b[pairs[:, 0]].multiply(a[pairs[:, 2]]).sum(1)).ravel()


def preferential(train, n, pairs):
    """Heuristic bậc: tích bậc hai đầu. Cao nghĩa là chỉ cần biết nút nào là nút trung tâm."""
    deg = np.bincount(np.r_[train[:, 0], train[:, 2]], minlength=n).astype(float)
    return deg[pairs[:, 0]] * deg[pairs[:, 2]]


def evaluate(scores_pos, scores_neg):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.r_[np.ones(scores_pos.size), np.zeros(scores_neg.size)]
    p = np.r_[scores_pos, scores_neg]
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def scored(model, graph, t, device):
    model.eval()
    with torch.no_grad():
        return model.score(model.encode(graph), torch.as_tensor(t, device=device)).cpu().numpy()


def train_link(model, graph, data, neg, rng, device, epochs=200, lr=0.01, wd=1e-5, every=20, patience=3):
    """BCE trên dương và âm cùng kiểu; dừng theo AUC val, test không tham gia."""
    train, val_pos, val_neg = data
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos = torch.as_tensor(train, device=device)
    best, keep, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(graph)
        nt = torch.as_tensor(neg.sample(train, rng), device=device)
        logit = torch.cat([model.score(z, pos), model.score(z, nt)])
        lab = torch.cat([torch.ones(pos.size(0)), torch.zeros(nt.size(0))]).to(device)
        loss = F.binary_cross_entropy_with_logits(logit, lab)
        loss.backward()
        opt.step()
        if (ep + 1) % every:
            continue
        auc, _ = evaluate(scored(model, graph, val_pos, device), scored(model, graph, val_neg, device))
        hist.append({"epoch": ep + 1, "loss": loss.item(), "val_auc": auc})
        bad = 0 if auc > best else bad + 1
        if auc > best:
            best, keep = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if bad >= patience:
            break
    model.load_state_dict(keep)
    return best, hist
