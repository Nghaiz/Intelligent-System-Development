"""Suy luận Chương 6 cho Web App: phân loại một bài báo Cora bằng GCN / GraphSAGE / GAT / MLP đã huấn luyện."""
import time

import torch

from .ch6_gnn import CLASSES, GNN, graph_input, load_planetoid
from .common import MODELS

LABELS_VI = {"Case_Based": "Lập luận theo tình huống", "Genetic_Algorithms": "Thuật toán di truyền",
             "Neural_Networks": "Mạng nơ-ron", "Probabilistic_Methods": "Phương pháp xác suất",
             "Reinforcement_Learning": "Học tăng cường", "Rule_Learning": "Học luật", "Theory": "Lý thuyết"}
MODEL_IDS = ["gcn", "sage", "gat", "mlp"]

SPECS = {"ch6_cora": {
    "title": "Phân loại chủ đề bài báo trên đồ thị trích dẫn Cora (GNN)",
    "kind": "graph",
    "inputs": [{"name": "node_id", "label": "Mã nút (bài báo) 0–2707", "type": "number", "min": 0, "max": 2707,
                "default": 1701},
               {"name": "model", "label": "Mô hình", "type": "select", "options": MODEL_IDS, "default": "gcn"}],
    "models": MODEL_IDS,
    "example": {"node_id": 1701, "model": "gcn"},
}}

_STATE = {}


def _load(model_id):
    """Nạp đồ thị Cora một lần và mô hình theo yêu cầu; chạy CPU (đồ thị nhỏ, độ trễ vài chục ms)."""
    if "data" not in _STATE:
        d = load_planetoid("Cora")
        _STATE["data"] = d
        _STATE["deg"] = torch.bincount(d.edge_index[1], minlength=d.num_nodes)
    if model_id not in _STATE:
        ck = torch.load(MODELS / "ch6" / f"cora_{model_id}.pt", map_location="cpu")
        d = _STATE["data"]
        m = GNN(model_id, d.num_features, ck["hidden"], len(CLASSES["Cora"]))
        m.load_state_dict(ck["state"])
        m.eval()
        with torch.no_grad():
            probs = torch.softmax(m(d.x, graph_input(model_id, d, "cpu")), 1)
        _STATE[model_id] = {"probs": probs, "test": ck["test"]}
    return _STATE["data"], _STATE[model_id]


def predict(exp_id, payload, model_id=None):
    """payload {"node_id": int, "model": gcn|sage|gat|mlp} -> nhãn, xác suất 7 lớp, láng giềng và nhãn thật."""
    if exp_id != "ch6_cora":
        raise KeyError(exp_id)
    t0 = time.perf_counter()
    model_id = model_id or payload.get("model", "gcn")
    if model_id not in MODEL_IDS:
        raise ValueError(f"mô hình không hợp lệ: {model_id}")
    data, st = _load(model_id)
    v = int(payload.get("node_id", 0))
    if not 0 <= v < data.num_nodes:
        raise ValueError(f"node_id phải trong 0..{data.num_nodes - 1}")
    names = CLASSES["Cora"]
    p = st["probs"][v]
    k = int(p.argmax())
    nbrs = data.edge_index[0, data.edge_index[1] == v].tolist()
    split = "train" if data.train_mask[v] else "validation" if data.val_mask[v] else "test" if data.test_mask[v] else "không gán nhãn"
    return {"label": LABELS_VI[names[k]],
            "proba": {LABELS_VI[n]: round(float(p[i]), 4) for i, n in enumerate(names)},
            "details": {"node_id": v, "true_label": LABELS_VI[names[int(data.y[v])]], "correct": k == int(data.y[v]),
                        "degree": int(_STATE["deg"][v]), "split": split, "model_test_acc": round(st["test"], 4),
                        "neighbors": [{"id": int(u), "label": LABELS_VI[names[int(data.y[u])]]} for u in nbrs[:20]]},
            "model": model_id, "latency_ms": round(1000 * (time.perf_counter() - t0), 2)}
