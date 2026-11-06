"""Chương 7: dự đoán liên kết Hetionet: mô hình, đường cơ sở, thí nghiệm rò rỉ, ứng viên cho đái tháo đường type 2."""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf import ch7_link as L  # noqa: E402
from tlf.common import limit_threads, save_json  # noqa: E402
from tlf.gpu import cap  # noqa: E402
from tlf.mem import require_disk, slot  # noqa: E402
from tlf.runlog import log_run  # noqa: E402

DEV = "cuda"
SEEDS = 3
T2D = "H:Disease::DOID:9352"
TOP_RELS = ("GiG", "DaG", "CbG", "CcSE", "GpPW", "CtD")
LABELS = {"distmult": "DistMult (không truyền tin)", "gcn": "GCN + DistMult", "rgcn": "R-GCN + DistMult"}


def prepare(enc, seed):
    tr, va, te = L.split_pairs(enc["triples"], seed)
    neg = L.Negatives(enc["triples"], enc["kind"])
    rng = np.random.default_rng(seed)
    return dict(train=tr, val=va, test=te, neg=neg, rng=rng, val_neg=neg.sample(va, rng),
                test_neg=neg.sample(te, rng))


def build_graph(triples, enc):
    g = L.degree_weights(L.message_graph(triples, len(enc["rels"])), len(enc["names"]))
    return tuple(x.to(DEV) for x in g)


def one_run(kind, enc, seed, d):
    """Một lượt trên đồ thị truyền tin chỉ gồm cạnh train."""
    torch.manual_seed(seed)
    n, r = len(enc["names"]), len(enc["rels"])
    model = L.LinkModel(kind, n, len(enc["kinds"]), r, enc["kind"]).to(DEV)
    graph = build_graph(d["train"], enc)
    t0 = time.perf_counter()
    best, hist = L.train_link(model, graph, (d["train"], d["val"], d["val_neg"]), d["neg"], d["rng"], DEV)
    seconds = time.perf_counter() - t0
    pos, neg = L.scored(model, graph, d["test"], DEV), L.scored(model, graph, d["test_neg"], DEV)
    auc, ap = L.evaluate(pos, neg)
    return {"auc": auc, "ap": ap, "val_auc": best, "seconds": seconds, "history": hist, "model": model,
            "graph": graph, "pos": pos, "neg": neg, "params": sum(p.numel() for p in model.parameters())}


def leaky_eval(res, enc, d):
    """Cùng mô hình đã học, nhưng cạnh test lọt vào đồ thị truyền tin lúc đo: mô phỏng lỗi tách cạnh."""
    g = build_graph(np.r_[d["train"], d["test"]], enc)
    auc, ap = L.evaluate(L.scored(res["model"], g, d["test"], DEV), L.scored(res["model"], g, d["test_neg"], DEV))
    return {"auc": auc, "ap": ap, "seconds": 0.0}


def heuristic(fn, enc, d):
    n = len(enc["names"])
    t0 = time.perf_counter()
    pos, neg = fn(d["train"], n, d["test"]), fn(d["train"], n, d["test_neg"])
    auc, ap = L.evaluate(pos, neg)
    return {"auc": auc, "ap": ap, "seconds": time.perf_counter() - t0}


def per_relation(enc, d, pos, neg):
    """AUC riêng các quan hệ lớn; mẫu âm cùng thứ tự với dương nên ghép được theo chỉ số."""
    rows = {}
    for name in TOP_RELS:
        if name in enc["rels"]:
            m = d["test"][:, 1] == enc["rels"].index(name)
            if m.sum() >= 5:
                rows[name] = (L.evaluate(pos[m], neg[m])[0], int(m.sum()))
    return rows


def summary(label, rs):
    a, p = [r["auc"] for r in rs], [r["ap"] for r in rs]
    return {"model": label, "auc": float(np.mean(a)), "auc_std": float(np.std(a)), "ap": float(np.mean(p)),
            "ap_std": float(np.std(p)), "seconds": float(np.mean([r["seconds"] for r in rs])), "n_seeds": len(rs)}


def candidates(enc, res, d, k=8):
    """Xếp hạng hợp chất cho 'điều trị đái tháo đường type 2' bằng mô hình seed 0, loại cạnh train."""
    if T2D not in enc["ids"] or "CtD" not in enc["rels"]:
        return []
    v, r = enc["ids"].index(T2D), enc["rels"].index("CtD")
    comp = np.nonzero(np.array(enc["kinds"])[enc["kind"]] == "Compound")[0]
    t = np.stack([comp, np.full_like(comp, r), np.full_like(comp, v)], 1)
    s = L.scored(res["model"], res["graph"], t, DEV)
    train = {tuple(x) for x in d["train"].tolist()}
    held = {tuple(x) for x in np.r_[d["val"], d["test"]].tolist()}
    rows = []
    for i in np.argsort(-s):
        key = tuple(t[i].tolist())
        if key not in train:
            rows.append({"name": enc["names"][int(t[i, 0])], "score": float(torch.sigmoid(torch.tensor(s[i]))),
                         "held_out_true": key in held})
        if len(rows) == k:
            break
    return rows


def run_seeds(enc, source):
    runs = {k: [] for k in LABELS}
    out = {"aa": [], "pa": [], "leak": [], "rel": {}}
    for seed in range(SEEDS):
        d = prepare(enc, seed)
        out["aa"].append(heuristic(L.adamic_adar, enc, d))
        out["pa"].append(heuristic(L.preferential, enc, d))
        for kind in LABELS:
            r = one_run(kind, enc, seed, d)
            runs[kind].append(r)
            log_run("ch7", "hetionet", kind, "pytorch", {"auc": r["auc"], "ap": r["ap"]}, r["seconds"], DEV,
                    {"seed": seed, "source": source})
            print(seed, kind, round(r["auc"], 4), round(r["ap"], 4), flush=True)
        out["leak"].append(leaky_eval(runs["rgcn"][-1], enc, d))
        print(seed, "leaky", round(out["leak"][-1]["auc"], 4), flush=True)
        for name, val in per_relation(enc, d, runs["rgcn"][-1]["pos"], runs["rgcn"][-1]["neg"]).items():
            out["rel"].setdefault(name, []).append(val)
        if seed == 0:
            out["keep"], out["d0"] = runs["rgcn"][0], d
    return runs, out


def main():
    limit_threads(4)
    cap(0.15)
    enc, source = L.load_triples()
    print("nguồn", source, len(enc["names"]), len(enc["triples"]), flush=True)
    runs, o = run_seeds(enc, source)
    rows = [summary("Adamic-Adar", o["aa"]), summary("Tích bậc", o["pa"])]
    rows += [summary(LABELS[k], rs) | {"params": rs[0]["params"]} for k, rs in runs.items()]
    kinds = Counter(np.array(enc["kinds"])[enc["kind"]].tolist())
    rels = Counter(np.array(enc["rels"])[enc["triples"][:, 1]].tolist())
    d = o["d0"]
    save_json("ch7_link", {
        "source": source, "nodes": len(enc["names"]), "triples": len(enc["triples"]),
        "relations": len(enc["rels"]), "node_kinds": len(enc["kinds"]),
        "kinds": {"rows": [{"kind": k, "n": v} for k, v in kinds.most_common()]},
        "rels": {"rows": [{"rel": k, "n": v} for k, v in rels.most_common()]},
        "train": len(d["train"]), "val": len(d["val"]), "test": len(d["test"]), "models": {"rows": rows},
        "clean": rows[-1], "leaky": summary("R-GCN, cạnh test lọt vào đồ thị", o["leak"]),
        "per_relation": {"rows": [{"rel": k, "auc": float(np.mean([x[0] for x in v])),
                                   "n": int(np.mean([x[1] for x in v]))} for k, v in o["rel"].items()]},
        "history": o["keep"]["history"], "candidates": {"rows": candidates(enc, o["keep"], d)}})
    print("XONG", flush=True)


if __name__ == "__main__":
    require_disk(0.1, "ch7 link")
    with slot(3, "ch7 link"):
        main()
