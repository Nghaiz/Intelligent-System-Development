"""Chương 7: sinh bảng LaTeX và ch7_summary.json từ outputs/metrics/ch7_*.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.common import load_json, save_json  # noqa: E402
from tlf.texfmt import best, num, table, tex_escape  # noqa: E402

KEYS = (("cora", "Cora"), ("photo", "Amazon Photo"), ("elliptic", "Elliptic"))
MODELS = (("lp", "Label Propagation"), ("mlp", "MLP"), ("gcn", "GCN"), ("sage", "GraphSAGE"), ("gat", "GAT"))


def pm(v, s):
    return f"{num(v, 3)} $\\pm$ {num(s, 3)}"


def tab_datasets():
    d = load_json("ch7_data")
    meta = {x["id"]: x for x in load_json("ch7_datasets")["datasets"]}
    rows_def = [("Số nút", "nodes", 0), ("Số cạnh vô hướng", "edges", 0), ("Đặc trưng mỗi nút", "features", 0),
                ("Số lớp", "classes", 0), ("Nút có nhãn", "labeled", 0), ("Bậc trung bình", "deg_mean", 2),
                ("Bậc lớn nhất", "deg_max", 0), ("Nút cô lập", "isolated", 0), ("Thành phần liên thông", "components", 0),
                ("Đồng dạng cạnh", "edge_homophily", 3), ("Đồng dạng hiệu chỉnh", "adj_homophily", 3),
                ("Lệch lớp (lớn/nhỏ)", "imbalance", 1)]
    rows = [[lab] + [num(d[k][f], dec) for k, _ in KEYS] for lab, f, dec in rows_def]
    rows.append(["Dung lượng tệp gốc (MB)"] + [num(meta[k]["size_mb"], 1) for k, _ in KEYS])
    table("ch7_datasets", "lrrr", ["Thuộc tính"] + [n for _, n in KEYS], rows)


def tab_samples():
    meta = {x["id"]: x for x in load_json("ch7_datasets")["datasets"]}
    rows = []
    for key, name in KEYS[:2]:
        for r in meta[key]["sample_rows"]:
            nb = ", ".join(str(n) for n in r["neighbors"])
            rows.append([name, str(r["node"]), tex_escape(r["label"]), str(r["degree"]), str(r["nnz"]), nb,
                         tex_escape(r["neighbor_labels"][0]) if r["neighbor_labels"] else "--"])
    table("ch7_samples", "llp{3.6cm}rrlp{3.4cm}",
          ["Đồ thị", "Nút", "Nhãn", "Bậc", "Số từ", "Láng giềng", "Nhãn láng giềng đầu"], rows)
    rows = [[str(r["node"]), r["label"], str(r["step"]), str(r["degree"]), num(r["f1"], 3), num(r["f2"], 3)]
            for r in meta["elliptic"]["sample_rows"]]
    table("ch7_samples_ell", "rlrrrr", ["txId", "Nhãn", "Bước", "Bậc", "Đặc trưng 1", "Đặc trưng 2"], rows)


def tab_compare():
    node, numpy_gcn = load_json("ch7_node"), load_json("ch7_numpy")
    rows = []
    for key, name in KEYS:
        e = node[key]
        f = best([e[k]["f1_macro"] for k, _ in MODELS], [pm(e[k]["f1_macro"], e[k]["f1_macro_std"]) for k, _ in MODELS])
        for (k, lab), cell in zip(MODELS, f):
            extra = num(e[k]["illicit_f1"], 3) if key == "elliptic" else "--"
            rows.append([name, lab, pm(e[k]["accuracy"], e[k]["accuracy_std"]), cell, extra,
                         str(e[k]["n_seeds"]), num(e[k]["seconds"], 1)])
        if key == "cora":
            rows.append([name, "GCN NumPy", pm(numpy_gcn["accuracy"], numpy_gcn["accuracy_std"]),
                         pm(numpy_gcn["f1_macro"], numpy_gcn["f1_macro_std"]), "--", str(numpy_gcn["n_seeds"]),
                         num(numpy_gcn["seconds"], 1)])
        rows.append("\\midrule")
    table("ch7_compare", "llccrrr", ["Đồ thị", "Mô hình", "Độ đúng", "F1 macro", "F1 illicit", "Seed", "Giây/lượt"],
          rows[:-1], note="Trung bình $\\pm$ độ lệch chuẩn. Giây là trên GPU RTX 4060 dùng chung, GCN NumPy và LP chạy CPU.")


def tab_leak():
    leak = load_json("ch7_leak")
    label = {"gcn": "GCN", "sage": "GraphSAGE", "mlp": "MLP", "cora": "Cora", "photo": "Amazon Photo"}
    rows = [[label[r["dataset"]], label[r["model"]], "F1 macro: xuyên đồ thị / quy nạp", num(r["transductive"], 3),
             pm(r["inductive"], r["inductive_std"])] for r in leak["inductive"]["rows"]]
    rows += [["Elliptic", label[r["model"]], "F1 illicit: chia thời gian / ngẫu nhiên", num(r["temporal_f1"], 3),
              pm(r["random_f1"], r["random_std"])] for r in leak["temporal"]["rows"]]
    table("ch7_leak", "llp{5.6cm}rr", ["Đồ thị", "Mô hình", "Phép so", "Mức 1", "Mức 2"], rows)


def tab_link():
    d = load_json("ch7_link")
    ms = d["models"]["rows"]
    rows = [[tex_escape(m["model"]), pm(m["auc"], m["auc_std"]), pm(m["ap"], m["ap_std"]), num(m["seconds"], 1)]
            for m in ms]
    rows.append([tex_escape(d["leaky"]["model"]), pm(d["leaky"]["auc"], d["leaky"]["auc_std"]),
                 pm(d["leaky"]["ap"], d["leaky"]["ap_std"]), "--"])
    table("ch7_link", "lccr", ["Phương pháp", "AUC", "AP", "Giây/lượt"], rows)
    rows = [[r["rel"], str(r["n"]), num(r["auc"], 3)] for r in d["per_relation"]["rows"]]
    table("ch7_linkrel", "lrr", ["Quan hệ", "Cạnh test", "AUC"], rows)


def summary():
    node, link, leak = load_json("ch7_node"), load_json("ch7_link"), load_json("ch7_leak")
    tsne, rel = load_json("ch7_tsne"), load_json("ch7_reused")
    out = {}
    for key, _ in KEYS:
        e = node[key]
        order = sorted(("mlp", "gcn", "sage", "gat"), key=lambda k: -e[k]["f1_macro"])
        out[key] = {"best": dict(MODELS)[order[0]], "best_f1": e[order[0]]["f1_macro"],
                    "gain_vs_mlp": e[order[0]]["f1_macro"] - e["mlp"]["f1_macro"],
                    "gcn_vs_mlp": e["gcn"]["f1_macro"] - e["mlp"]["f1_macro"]}
    gaps = {f'{r["dataset"]}_{r["model"]}': r["transductive"] - r["inductive"] for r in leak["inductive"]["rows"]}
    tmp = {r["model"]: r["random_f1"] - r["temporal_f1"] for r in leak["temporal"]["rows"]}
    plain = [x for x in rel["cora_depth"]["rows"] if not x["residual"]]
    res = [x for x in rel["cora_depth"]["rows"] if x["residual"]]
    by_name = {m["model"]: m["auc"] for m in link["models"]["rows"]}
    out.update(leak_gap=gaps, temporal_gap=tmp, leak_auc_gain=link["leaky"]["auc"] - link["clean"]["auc"],
               link_distmult=by_name["DistMult (không truyền tin)"], link_pa=by_name["Tích bậc"],
               depth2=plain[1]["test_mean"], depth8=plain[-1]["test_mean"], depth8_res=res[-1]["test_mean"],
               mad0=rel["cora_depth"]["propagation"]["rows"][0]["mad"],
               mad64=rel["cora_depth"]["propagation"]["rows"][-1]["mad"],
               sil={"cora_raw": rel["cora_tsne"]["raw"]["silhouette"], "cora_gcn": rel["cora_tsne"]["gcn"]["silhouette"],
                    "photo_raw": tsne["photo"]["raw"]["silhouette"], "photo_gcn": tsne["photo"]["gcn"]["silhouette"],
                    "ell_raw": tsne["elliptic"]["raw"]["silhouette"], "ell_gcn": tsne["elliptic"]["gcn"]["silhouette"]})
    for key in ("cora", "photo"):
        low, high = node[key]["homophily_bins"]["rows"][0], node[key]["homophily_bins"]["rows"][-1]
        out[key].update(low_n=low["n"], low_mlp=low["mlp"], low_gcn=low["gcn"], high_mlp=high["mlp"], high_gcn=high["gcn"])
    steps = node["elliptic"]["steps"]["rows"]
    out["elliptic"].update(before43={k: sum(r[k] for r in steps if r["step"] < 43) / 8 for k in ("mlp", "gcn", "sage", "gat")},
                           after43={k: sum(r[k] for r in steps if r["step"] >= 43) / 7 for k in ("mlp", "gcn", "sage", "gat")})
    save_json("ch7_summary", out)


def leak_summary():
    """Mức giảm khi siết phép chia, để phần lời dùng \SL thay vì gõ tay."""
    leak = load_json("ch7_leak")
    out = {f"{r['dataset']}_{r['model']}_drop": r["transductive"] - r["inductive"] for r in leak["inductive"]["rows"]}
    for r in leak["temporal"]["rows"]:
        out[f"elliptic_{r['model']}_random"] = r["random_f1"]
        out[f"elliptic_{r['model']}_temporal"] = r["temporal_f1"]
    save_json("ch7_leaksum", out)


def main():
    for f in (tab_datasets, tab_samples, tab_compare, tab_leak, tab_link, summary, leak_summary):
        f()
    print("xong bảng ch7")


if __name__ == "__main__":
    main()
