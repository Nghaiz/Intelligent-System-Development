"""Web App, Ch.7: đồ thị con quanh một nút kèm dự đoán và trọng số truyền thông điệp.

Trọng số Cora, Amazon Photo do scripts/ch7_train.py lưu ở models/ch7/. Nạp lười, giữ lại, chạy CPU, không cần PyG.
Với GAT trọng số cạnh là attention trung bình các đầu ở lớp 1; với GCN là phần tử Â_vu cố định theo bậc.
"""
from collections import deque

import numpy as np

from .ch7_data import CORA_CLASSES
from .common import MODELS

DATASETS = {"Cora": "cora", "Amazon Photo": "photo"}
MODEL_KINDS = {"GAT": "gat", "GCN": "gcn"}
MAX_NODES = 120
_cache = {}

SPECS = [{
    "id": "gnn_subgraph", "title": "Truyền thông điệp quanh một nút", "dataset": "Cora, Amazon Photo",
    "inputs": [
        {"name": "dataset", "type": "select", "label": "Đồ thị", "default": "Cora", "options": list(DATASETS)},
        {"name": "model", "type": "select", "label": "Mô hình", "default": "GAT", "options": list(MODEL_KINDS)},
        {"name": "node", "type": "number", "label": "Mã nút", "default": 35, "min": 0, "max": 7649},
        {"name": "hops", "type": "number", "label": "Số bước láng giềng", "default": 2, "min": 1, "max": 2},
    ],
    "output": "probs",
}]


def _load(key, kind):
    """Nạp đồ thị và mô hình một lần; lần sau lấy từ _cache."""
    import torch
    from .ch7_gnn import CONFIG, GNN, gcn_norm_adj, graph_input
    if key not in _cache:
        d = np.load(MODELS / "ch7" / f"{key}_graph.npz")
        ei = torch.from_numpy(d["edge_index"].astype(np.int64))
        _cache[key] = {"x": torch.from_numpy(d["x"]), "ei": ei, "y": d["y"].astype(np.int64),
                       "adj": gcn_norm_adj(ei, d["x"].shape[0]).coalesce()}
    g = _cache[key]
    tag = (key, kind)
    if tag not in _cache:
        ckpt = torch.load(MODELS / "ch7" / f"{key}_{kind}.pt", map_location="cpu", weights_only=True)
        cfg = CONFIG[key]
        n_cls = int(g["y"].max()) + 1
        drop = 0.6 if kind == "gat" else cfg["dropout"]
        net = GNN(kind, g["x"].size(1), cfg["hidden"], n_cls, 2, drop, cfg["heads"])
        net.load_state_dict(ckpt["state"])
        net.eval()
        with torch.no_grad():
            logits = net(g["x"], graph_input(kind, g["ei"], g["x"].size(0)))
        _cache[tag] = {"probs": torch.softmax(logits, 1).numpy(), "weights": _edge_weights(net, kind, g)}
    return g, _cache[tag]


def _edge_weights(net, kind, g):
    """Từ điển (nguồn, đích) -> trọng số cạnh, gồm cả tự khuyên."""
    if kind == "gat":
        ei, alpha = net.layers[0].last_alpha
        return dict(zip(zip(ei[0].tolist(), ei[1].tolist()), alpha.mean(1).tolist()))
    idx, val = g["adj"].indices(), g["adj"].values()
    return dict(zip(zip(idx[1].tolist(), idx[0].tolist()), val.tolist()))


def _neighbors(ei):
    src, dst = ei.numpy()
    order = np.argsort(dst, kind="stable")
    dst_sorted = dst[order]
    return src[order], dst_sorted


def _ball(g, node, hops):
    """Tìm theo chiều rộng tới `hops` bước, cắt ở MAX_NODES nút gần nhất."""
    if "nb" not in g:
        g["nb"] = _neighbors(g["ei"])
    src, dst = g["nb"]
    hop = {node: 0}
    q = deque([node])
    while q and len(hop) < MAX_NODES:
        v = q.popleft()
        if hop[v] == hops:
            continue
        a, b = np.searchsorted(dst, [v, v + 1])
        for u in src[a:b].tolist():
            if u not in hop and len(hop) < MAX_NODES:
                hop[u] = hop[v] + 1
                q.append(u)
    return hop


def _class_name(key, c):
    return CORA_CLASSES[c] if key == "cora" else f"Lớp {c}"


def _subgraph(key, g, m, node, hops):
    hop = _ball(g, node, hops)
    pred = m["probs"].argmax(1)
    y = g["y"]
    nodes = [{"id": v, "hop": h, "label": _class_name(key, int(y[v])), "pred": _class_name(key, int(pred[v])),
              "correct": bool(pred[v] == y[v]), "confidence": float(m["probs"][v].max())}
             for v, h in sorted(hop.items(), key=lambda kv: kv[1])]
    links = [{"source": u, "target": v, "weight": round(w, 5), "self": u == v}
             for (u, v), w in ((e, m["weights"].get(e)) for e in _pairs(g, hop)) if w is not None]
    return nodes, links


def _pairs(g, hop):
    src, dst = g["nb"]
    for v in hop:
        yield (v, v)
        a, b = np.searchsorted(dst, [v, v + 1])
        for u in src[a:b].tolist():
            if u in hop:
                yield (u, v)


def predict(spec_id, payload):
    if spec_id != "gnn_subgraph":
        raise ValueError(f"Không có mô hình {spec_id}")
    ds, kind = payload.get("dataset", "Cora"), payload.get("model", "GAT")
    if ds not in DATASETS or kind not in MODEL_KINDS:
        raise ValueError("Đồ thị hoặc mô hình không hợp lệ")
    key = DATASETS[ds]
    g, m = _load(key, MODEL_KINDS[kind])
    n = g["x"].size(0)
    node, hops = int(payload.get("node", 35)), int(payload.get("hops", 2))
    if not 0 <= node < n:
        raise ValueError(f"Mã nút phải trong khoảng 0 đến {n - 1}")
    if hops not in (1, 2):
        raise ValueError("Số bước láng giềng là 1 hoặc 2")
    probs = m["probs"][node]
    nodes, links = _subgraph(key, g, m, node, hops)
    into = sorted((x for x in links if x["target"] == node), key=lambda x: -x["weight"])
    names = {i: _class_name(key, i) for i in range(probs.size)}
    return {"label": names[int(probs.argmax())], "truth": names[int(g["y"][node])],
            "probs": {names[i]: float(p) for i, p in enumerate(probs)},
            "graph": {"nodes": nodes, "links": links}, "messages": into[:12],
            "weight_kind": "attention học được" if kind == "GAT" else "Â_vu cố định theo bậc",
            "truncated": len(nodes) >= MAX_NODES}
