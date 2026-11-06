"""Chương 7: tải Photo, đo thống kê và nút mẫu thật, ghi ch7_datasets.json và ch7_data.json."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.common import limit_threads, save_json, seed_all  # noqa: E402
from tlf.mem import require_disk, slot  # noqa: E402

META = {
    "cora": dict(name="Cora", source="Planetoid (Yang et al., 2016)",
                 url="https://github.com/kimiyoung/planetoid", license="MIT (kho Planetoid)",
                 task="Phân loại chủ đề bài báo", files=["planetoid/Cora/raw"],
                 note="Nút là bài báo, cạnh là trích dẫn, đặc trưng là túi từ nhị phân."),
    "photo": dict(name="Amazon Photo", source="Shchur et al., 2018 (gnn-benchmark)",
                  url="https://github.com/shchur/gnn-benchmark", license="MIT (kho gnn-benchmark)",
                  task="Phân loại nhóm sản phẩm", files=["amazon/Photo/raw"],
                  note="Nút là sản phẩm, cạnh là thường được mua cùng nhau. Bộ gốc không công bố tên lớp."),
    "elliptic": dict(name="Elliptic Bitcoin", source="Weber et al., 2019, bản sao PyG",
                     url="https://data.pyg.org/datasets/elliptic/elliptic_txs_features.csv.zip",
                     license="Không ghi rõ, Elliptic công bố cho nghiên cứu",
                     task="Phát hiện giao dịch phi pháp theo thời gian", files=["elliptic/raw"],
                     note="Chọn thay ogbn-arxiv vì có chia theo thời gian thật và lệch lớp nặng. Dùng toàn bộ đồ thị; "
                          "nút là giao dịch, cạnh là dòng tiền, phần lớn nút không có nhãn."),
}


def folder_mb(p):
    from tlf.ch7_data import ROOT
    return sum(f.stat().st_size for f in (ROOT / p).glob("*") if f.is_file()) / 2**20


def degree_curve(deg):
    """Phân bố bậc gộp theo thang log: số nút mỗi ô, để vẽ log-log."""
    d = deg[deg > 0].astype(float)
    edges = np.unique(np.round(np.logspace(0, np.log10(d.max() + 1), 14)))
    hist, _ = np.histogram(d, bins=edges)
    return [{"deg": float(edges[i]), "count": int(c)} for i, c in enumerate(hist) if c > 0]


def graph_stats(g, split):
    import torch
    from tlf.ch7_data import UNKNOWN, adjusted_homophily, components, degrees, edge_homophily, node_homophily
    ei = g.edge_index
    deg = degrees(ei, g.n).numpy()
    n_comp, labels = components(ei, g.n)
    counts = torch.bincount(g.y[g.y != UNKNOWN]).tolist()
    out = {"nodes": g.n, "edges": ei.size(1) // 2, "features": g.x.size(1), "classes": len(g.classes),
           "deg_mean": float(deg.mean()), "deg_median": float(np.median(deg)), "deg_max": int(deg.max()),
           "isolated": int((deg == 0).sum()), "density": float(ei.size(1) / (g.n * (g.n - 1))),
           "components": int(n_comp), "largest_cc": int(np.bincount(labels).max()),
           "edge_homophily": edge_homophily(ei, g.y),
           "node_homophily": float(np.nanmean(node_homophily(ei, g.y, g.n))),
           "adj_homophily": adjusted_homophily(ei, g.y), "class_counts": counts, "class_names": g.classes,
           "degree_curve": {"rows": degree_curve(deg)}, "imbalance": max(counts) / min(counts),
           "labeled": sum(counts)}
    if split is not None:
        out.update(train=int(split.train.sum()), val=int(split.val.sum()), test=int(split.test.sum()))
    return out


def sample_nodes(g, n=2):
    """Nút thật đầu mỗi lớp: chỉ số, nhãn, bậc, số từ khác 0, láng giềng đầu và nhãn của chúng."""
    from tlf.ch7_data import degrees
    rows = []
    deg = degrees(g.edge_index, g.n)
    for c in range(n):
        v = int((g.y == c).nonzero()[0])
        nb = g.edge_index[0][g.edge_index[1] == v][:2].tolist()
        rows.append({"node": v, "label": g.classes[c], "degree": int(deg[v]), "nnz": int((g.x[v] > 0).sum()),
                     "neighbors": nb, "neighbor_labels": [g.classes[int(g.y[u])] for u in nb]})
    return rows


def sample_elliptic(g):
    """Giao dịch thật: txId, bước thời gian, nhãn, bậc, hai đặc trưng đầu đã chuẩn hóa."""
    from tlf.ch7_data import UNKNOWN, degrees, elliptic_ids
    ids, _ = elliptic_ids()
    deg = degrees(g.edge_index, g.n)
    names = {0: "licit", 1: "illicit", UNKNOWN: "chưa biết"}
    rows = []
    for c in (1, 0, UNKNOWN):
        v = int((g.y == c).nonzero()[0])
        rows.append({"node": int(ids[v]), "label": names[c], "step": int(g.time[v]), "degree": int(deg[v]),
                     "f1": round(float(g.x[v, 0]), 3), "f2": round(float(g.x[v, 1]), 3)})
    return rows


def elliptic_time(g):
    """Số giao dịch có nhãn theo bước thời gian; đếm cạnh vắt qua hai bước."""
    from tlf.ch7_data import UNKNOWN
    s, d = g.edge_index
    rows = []
    for t in range(1, 50):
        m = g.time == t
        rows.append({"step": t, "illicit": int((g.y[m] == 1).sum()), "licit": int((g.y[m] == 0).sum()),
                     "unknown": int((g.y[m] == UNKNOWN).sum())})
    return {"rows": rows, "cross_step_edges": int((g.time[s] != g.time[d]).sum()), "steps": 49}


def main():
    limit_threads(4)
    seed_all()
    from tlf.ch7_data import load_graph
    stats, entries = {}, []
    for key, meta in META.items():
        g, split = load_graph(key)
        st = graph_stats(g, split)
        stats[key] = st
        rows = sample_elliptic(g) if key == "elliptic" else sample_nodes(g)
        if key == "elliptic":
            stats["elliptic_time"] = elliptic_time(g)
        entries.append({"id": key, "name": meta["name"], "source": meta["source"], "url": meta["url"],
                        "task": meta["task"], "n_samples": st["nodes"], "n_features": st["features"],
                        "size_mb": round(sum(folder_mb(p) for p in meta["files"]), 2), "license": meta["license"],
                        "note": meta["note"], "sample_rows": rows,
                        "distribution": {"labels": st["class_names"], "values": st["class_counts"]}})
        print(key, {k: v for k, v in st.items() if not isinstance(v, (dict, list))}, flush=True)
    save_json("ch7_data", stats)
    save_json("ch7_datasets", {"datasets": entries})
    print("đã ghi ch7_datasets, ch7_data", flush=True)


if __name__ == "__main__":
    require_disk(2, "ch7 tải Photo, Elliptic")
    with slot(3.5, "ch7 datasets"):
        main()
