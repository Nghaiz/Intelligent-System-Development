"""Chương 7: thực nghiệm phân loại nút trên CPU. Mỗi giai đoạn ghi JSON riêng.

    python scripts/ch7_train.py node cora photo | node elliptic | numpy | leak | tsne | compare
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.common import METRICS, limit_threads, load_json, model_dir, save_json  # noqa: E402
from tlf.gpu import cap  # noqa: E402
from tlf.mem import require_disk, slot  # noqa: E402
from tlf.runlog import log_run  # noqa: E402

KEYS = ("cora", "photo", "elliptic")
SEEDS = {"cora": 5, "photo": 5, "elliptic": 5}
NEURAL = ("mlp", "gcn", "sage", "gat")
LABEL = {"mlp": "MLP", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "lp": "Label Propagation"}
MODELS_DIR = model_dir("ch7")
DEV = "cuda"


def existing(name):
    return load_json(name) if (METRICS / f"{name}.json").exists() else {}


def summarize(rs):
    out = {"n_seeds": len(rs), "seconds": float(np.mean([r["seconds"] for r in rs])),
           "params": int(rs[0].get("params", 0))}
    for k in ("accuracy", "f1_macro", "illicit_f1", "illicit_precision", "illicit_recall"):
        if k not in rs[0]:
            continue
        v = np.array([r[k] for r in rs])
        out[k], out[f"{k}_std"], out[f"{k}_runs"] = float(v.mean()), float(v.std()), v.tolist()
    return out


def homophily_bins(g, split, logits_by_model):
    """Phân tích lỗi: độ đúng trên test theo tỷ lệ láng giềng cùng nhãn (chỉ phân tích sau khi đo)."""
    from tlf.ch7_data import node_homophily
    h = node_homophily(g.edge_index, g.y, g.n)
    te, y = split.test.numpy(), g.y.numpy()
    rows = []
    for lo, hi, name in ((0, .25, "0-25"), (.25, .5, "25-50"), (.5, .75, "50-75"), (.75, 1.01, "75-100")):
        m = te & (h >= lo) & (h < hi)
        row = {"bin": name, "n": int(m.sum())}
        for kind, logits in logits_by_model.items():
            row[kind] = float((logits.argmax(1).numpy()[m] == y[m]).mean()) if m.any() else None
        rows.append(row)
    return rows


def run_lp(key, g, fixed):
    """Label Propagation: α chọn trên val trong ba giá trị; báo tỷ lệ nút test nhãn lan tới được."""
    import torch
    from tlf.ch7_baselines import label_propagation, predict_lp, sym_norm
    from tlf.ch7_data import splits_for
    from tlf.ch7_gnn import scores
    s_mat = sym_norm(g.edge_index.numpy(), g.n)
    y = g.y.numpy()
    positive = 1 if key == "elliptic" else None
    rs = []
    for seed in range(SEEDS[key]):
        sp = splits_for(g, fixed, seed)
        tr, va, te = sp.train.numpy(), sp.val.numpy(), sp.test.numpy()
        t0 = time.perf_counter()
        fs = {a: label_propagation(s_mat, y, tr, len(g.classes), a) for a in (0.5, 0.9, 0.99)}
        val = {a: (predict_lp(f, y, tr)[0][va] == y[va]).mean() for a, f in fs.items()}
        alpha = max(val, key=val.get)
        pred, reached = predict_lp(fs[alpha], y, tr)
        r = scores(torch.from_numpy(pred[te]), g.y[sp.test], len(g.classes), positive)
        r.update(seconds=time.perf_counter() - t0, reached=float(reached[te].mean()), alpha=alpha)
        rs.append(r)
        log_run("ch7", key, "lp", "numpy", {k: r[k] for k in ("accuracy", "f1_macro")}, r["seconds"], "cpu",
                {"alpha": alpha})
    out = summarize(rs)
    out["reached"], out["alpha"] = float(np.mean([r["reached"] for r in rs])), float(rs[0]["alpha"])
    return out


def step_f1(g, split, logits):
    """F1 lớp illicit theo từng bước thời gian của tập test."""
    pred = logits.argmax(1).numpy()
    y, t, te = g.y.numpy(), g.time.numpy(), split.test.numpy()
    out = []
    for step in range(35, 50):
        m = te & (t == step)
        tp = int(((pred == 1) & (y == 1) & m).sum())
        fp = int(((pred == 1) & (y == 0) & m).sum())
        fn = int(((pred == 0) & (y == 1) & m).sum())
        out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return out


def run_neural(key, kind, g, fixed, cfg):
    from tlf.ch7_data import splits_for
    from tlf.ch7_gnn import fit
    rs, steps = [], []
    for seed in range(SEEDS[key]):
        sp = splits_for(g, fixed, seed)
        r = fit(kind, g, sp, cfg, seed, inductive=key == "elliptic", device=DEV)
        if key == "elliptic":
            steps.append(step_f1(g, sp, r["logits"]))
        rs.append(r)
        log_run("ch7", key, kind, "pytorch", {k: r[k] for k in ("accuracy", "f1_macro")}, r["seconds"], DEV,
                {k: cfg[k] for k in ("hidden", "lr", "wd")} | {"seed": seed, "epochs": r["epochs"]})
        print(key, kind, seed, round(r["accuracy"], 4), round(r["f1_macro"], 4), round(r["seconds"], 1), flush=True)
    return summarize(rs), rs[0], (np.mean(steps, 0) if steps else None)


def save_serving(key, g, keep):
    """Lưu đồ thị và trọng số seed 0 cho Web App (ghi đè, không giữ phiên bản cũ)."""
    import torch
    for kind in ("gcn", "gat"):
        torch.save({"state": keep[kind]["model"].state_dict(), "kind": kind}, MODELS_DIR / f"{key}_{kind}.pt")
    np.savez_compressed(MODELS_DIR / f"{key}_graph.npz", x=g.x.numpy().astype(np.float32),
                        edge_index=g.edge_index.numpy().astype(np.int32), y=g.y.numpy().astype(np.int16))


def stage_node(keys):
    import torch
    from tlf.ch7_data import load_graph, splits_for
    from tlf.ch7_gnn import CONFIG
    res = existing("ch7_node")
    for key in keys:
        if "homophily_bins" in res.get(key, {}):
            continue
        g, fixed = load_graph(key)
        entry, keep, steps = {"lp": run_lp(key, g, fixed)}, {}, {}
        for kind in NEURAL:
            entry[kind], keep[kind], st = run_neural(key, kind, g, fixed, CONFIG[key])
            if st is not None:
                steps[kind] = st
            res[key] = entry
            save_json("ch7_node", res)
        sp0 = splits_for(g, fixed, 0)
        entry["homophily_bins"] = {"rows": homophily_bins(g, sp0, {k: keep[k]["logits"] for k in NEURAL})}
        if steps:
            entry["steps"] = {"rows": [{"step": 35 + i, **{k: float(v[i]) for k, v in steps.items()}}
                                       for i in range(15)]}
        save_json("ch7_node", res)
        idx = sp0.test.nonzero().flatten() if key == "elliptic" else torch.arange(g.n)
        if key != "cora":
            np.savez_compressed(MODELS_DIR / f"{key}_hidden.npz", y=g.y[idx].numpy(), raw=g.x[idx].numpy(),
                                gcn=keep["gcn"]["hidden"][idx].numpy())
        if key != "elliptic":
            save_serving(key, g, keep)


def stage_numpy():
    """GCN NumPy trên Cora, 5 seed, cùng cấu hình bản PyTorch; kèm kiểm tra đạo hàm."""
    import torch
    from tlf.ch7_data import load_graph
    from tlf.ch7_gnn import macro_f1
    from tlf.ch7_numpy import grad_check, normalized_adj, train_numpy
    g, sp = load_graph("cora")
    A = normalized_adj(g.edge_index.numpy(), g.n)
    X, y = g.x.numpy().astype(np.float32), g.y.numpy()
    tr, va, te = sp.train.numpy(), sp.val.numpy(), sp.test.numpy()
    worst, rs = grad_check(), []
    for seed in range(SEEDS["cora"]):
        t0 = time.perf_counter()
        r = train_numpy(A, X, y, tr, va, te, seed=seed)
        r["seconds"] = time.perf_counter() - t0
        r["accuracy"] = r["test"]
        r["f1_macro"] = macro_f1(torch.from_numpy(r["pred"][te]), g.y[sp.test], 7)
        rs.append(r)
        log_run("ch7", "cora", "gcn", "numpy", {k: r[k] for k in ("accuracy", "f1_macro")}, r["seconds"], "cpu",
                {"seed": seed, "hidden": 64})
        print("numpy", seed, round(r["accuracy"], 4), round(r["seconds"], 1), flush=True)
    out = summarize(rs)
    out.update(grad_check_max_error=worst, history={"rows": rs[0]["history"]})
    save_json("ch7_numpy", out)


def stage_leak():
    """Hai thí nghiệm rò rỉ: xuyên đồ thị so với quy nạp; chia ngẫu nhiên so với chia thời gian."""
    from tlf.ch7_data import load_graph, random_split, splits_for
    from tlf.ch7_gnn import CONFIG, fit
    node = load_json("ch7_node")
    rows = []
    for key in ("cora", "photo"):
        g, fixed = load_graph(key)
        for kind in ("gcn", "sage"):
            f1 = [fit(kind, g, splits_for(g, fixed, s), CONFIG[key], s, True, DEV)["f1_macro"]
                  for s in range(SEEDS[key])]
            rows.append({"dataset": key, "model": kind, "transductive": node[key][kind]["f1_macro"],
                         "inductive": float(np.mean(f1)), "inductive_std": float(np.std(f1))})
            print(rows[-1], flush=True)
            save_json("ch7_leak", {"inductive": {"rows": rows}})
    g, fixed = load_graph("elliptic")
    temporal = []
    for kind in ("mlp", "gcn"):
        rs = [fit(kind, g, random_split(g, fixed, s), CONFIG["elliptic"], s, False, DEV)
              for s in range(SEEDS["elliptic"])]
        ours = node["elliptic"][kind]
        temporal.append({"model": kind, "random_f1": float(np.mean([r["illicit_f1"] for r in rs])),
                         "random_std": float(np.std([r["illicit_f1"] for r in rs])),
                         "temporal_f1": ours["illicit_f1"], "temporal_std": ours["illicit_f1_std"]})
        print(temporal[-1], flush=True)
    save_json("ch7_leak", {"inductive": {"rows": rows}, "temporal": {"rows": temporal}})


def stage_tsne():
    """t-SNE, silhouette trên 3.000 nút mỗi đồ thị; Cora lấy lại từ Tiểu luận 03."""
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    out = {}
    for key in ("photo", "elliptic"):
        d = np.load(MODELS_DIR / f"{key}_hidden.npz")
        idx = np.random.default_rng(0).permutation(d["y"].size)[:3000]
        y = d["y"][idx]
        out[key] = {"labels": {"series": y.tolist()}}
        for tag in ("raw", "gcn"):
            mat = d[tag][idx].astype(np.float32)
            z = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(mat)
            out[key][tag] = {"silhouette": float(silhouette_score(mat, y, metric="cosine")),
                             "series": np.round(z, 2).tolist()}
            print(key, tag, out[key][tag]["silhouette"], flush=True)
    save_json("ch7_tsne", out)


def stage_compare():
    """Gộp mọi kết quả về ch7_compare.json theo hợp đồng chung."""
    node, num = load_json("ch7_node"), load_json("ch7_numpy")
    name = {"cora": "Cora", "photo": "Amazon Photo", "elliptic": "Elliptic Bitcoin"}
    keys = ("accuracy", "accuracy_std", "f1_macro", "f1_macro_std", "seconds", "n_seeds")
    rows = []
    for key in KEYS:
        for kind in ("lp",) + NEURAL:
            rows.append({"dataset": name[key], "model": LABEL[kind], "framework": "numpy" if kind == "lp" else "pytorch",
                         **{k: node[key][kind][k] for k in keys + ("params",)},
                         **({"illicit_f1": node[key][kind]["illicit_f1"]} if key == "elliptic" else {})})
    rows.append({"dataset": "Cora", "model": "GCN", "framework": "numpy", **{k: num[k] for k in keys}})
    for m in existing("ch7_link").get("models", {"rows": []})["rows"]:
        rows.append({"dataset": "Hetionet (Neo4j)", "model": m["model"], "framework": "pytorch", "auc": m["auc"],
                     "auc_std": m["auc_std"], "ap": m["ap"], "ap_std": m["ap_std"], "seconds": m["seconds"],
                     "n_seeds": m["n_seeds"]})
    save_json("ch7_compare", {"rows": rows, "primary": "f1_macro"})


def main():
    stage, args = sys.argv[1], sys.argv[2:]
    limit_threads(4)
    require_disk(0.2, f"ch7 {stage}")
    if stage == "compare":
        return stage_compare()
    gb = 4 if stage in ("leak", "tsne") or "elliptic" in args else 3
    with slot(gb, f"ch7 {stage}"):
        if stage != "numpy":
            cap(0.4)
        if stage == "node":
            return stage_node(args)
        {"numpy": stage_numpy, "leak": stage_leak, "tsne": stage_tsne}[stage]()


if __name__ == "__main__":
    main()
