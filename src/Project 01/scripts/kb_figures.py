"""Hình KB, KG, RAG cho báo cáo; mọi số đọc từ outputs/metrics và Neo4j, không viết tay."""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import load_json  # noqa: E402
from dnx.kb import search as K, store  # noqa: E402
from dnx.plot_style import C, SEQ, plt, save, vn  # noqa: E402

W = 6.3
SRC_VI = {"europepmc": "Europe PMC", "medlineplus": "MedlinePlus", "niddk": "NIDDK", "cdc": "CDC", "who": "WHO", "wikipedia_vi": "Wikipedia vi",
          "bo_y_te": "Bộ Y tế, lâm sàng VN", "medlineplus_vi": "MedlinePlus vi"}
MODE_VI = {"bm25": "BM25", "dense": "Dense", "graph": "Graph", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + rerank"}


def bars(ax, names, vals, color=C["blue"], fmt=lambda v: vn(v)):
    y = np.arange(len(names))
    ax.barh(y, vals, color=color, height=.7)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(v, i, " " + fmt(v), va="center", fontsize=12)


def fig_sources():
    c = load_json("kb_chunks")["by_source"]
    s = load_json("kb_sources")["sources"]
    names = sorted(c, key=lambda k: -c[k])
    fig, ax = plt.subplots(figsize=(W, 3.9))
    bars(ax, [SRC_VI.get(n, n) for n in names], [c[n] for n in names])
    ax.set_xscale("log")
    ax.set_xlim(1, max(c.values()) * 8)
    ax.set_xlabel("Số chunk (thang log)")
    ax.set_title(f"Kho tri thức: {vn(sum(v['docs'] for v in s.values()))} tài liệu, {vn(sum(c.values()))} chunk")
    save(fig, "kb_sources")


def fig_chunk_len():
    rows, _ = store.chunks()
    w = {"vi": [r["words"] for r in rows if r["lang"] == "vi"], "en": [r["words"] for r in rows if r["lang"] == "en"]}
    fig, ax = plt.subplots(figsize=(W, 3.4))
    bins = np.arange(0, 215, 10)
    ax.hist(w["en"], bins=bins, color=C["blue"], alpha=.75, label=f"Tiếng Anh ({vn(len(w['en']))})")
    ax.hist(w["vi"], bins=bins, color=C["orange"], alpha=.9, label=f"Tiếng Việt ({vn(len(w['vi']))})")
    ax.set_yscale("log")
    ax.set(xlabel="Số từ mỗi chunk", ylabel="Số chunk (thang log)", title="Độ dài chunk")
    ax.legend()
    save(fig, "kb_chunk_len")


def fig_kg():
    s = load_json("kg_stats")
    lab = {k: v for k, v in s["by_label"].items() if k != "Entity"}
    n = sorted(lab, key=lambda k: -lab[k])
    fig, ax = plt.subplots(figsize=(W, 6.2))
    bars(ax, n, [lab[k] for k in n], C["teal"])
    ax.set_xscale("log")
    ax.set_xlim(1, max(lab.values()) * 10)
    ax.set_xlabel("Số nút (thang log)")
    ax.set_title(f"Nhãn nút: tổng {vn(s['nodes'])} nút")
    save(fig, "kg_labels")
    rel = s["by_type"]
    n = sorted(rel, key=lambda k: -rel[k])
    fig, ax = plt.subplots(figsize=(W, 7.2))
    bars(ax, n, [rel[k] for k in n], C["purple"])
    ax.set_xscale("log")
    ax.set_xlim(1, max(rel.values()) * 10)
    ax.set_xlabel("Số cạnh (thang log)")
    ax.set_title(f"Loại quan hệ: tổng {vn(s['relationships'])} cạnh")
    save(fig, "kg_relations")


def fig_subgraph():
    """Đồ thị con thật lấy từ Neo4j quanh đái tháo đường típ 2 và HbA1c."""
    seeds = [e["id"] for e in K.entities("type 2 diabetes mellitus") if e["label"] in ("Disease", "MeshTerm")][:1] + ["T:hba1c", "X:retinopathy"]
    sg = K.subgraph(seeds, hops=1, limit=44)
    g = nx.Graph()
    labels = {n["id"]: n for n in sg["nodes"]}
    g.add_nodes_from(labels)
    g.add_edges_from((l["source"], l["target"]) for l in sg["links"])
    g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
    kinds = sorted({labels[n]["label"] for n in g})
    color = {k: SEQ[i % len(SEQ)] for i, k in enumerate(kinds)}
    pos = nx.spring_layout(g, seed=7, k=2.4 / max(1, np.sqrt(len(g))), iterations=300)
    fig, ax = plt.subplots(figsize=(W, 6.0))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#9CA3AF", width=.7)
    for k in kinds:
        nodes = [n for n in g if labels[n]["label"] == k]
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=color[k], node_size=[260 if n in seeds else 90 for n in nodes], ax=ax, label=k)
    top = sorted(g, key=lambda n: -g.degree(n))[:7]
    named = [n for n in set(top) | (set(seeds) & set(g)) if labels[n]["label"] != "MeshTerm" or "Type 2" in labels[n]["name"]]
    for n in named:
        x, y = pos[n]
        ax.annotate(labels[n]["name"][:26], (x, y), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=12, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D1D5DB", lw=.6, alpha=.92))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.01), ncol=4, fontsize=11)
    ax.set_title(f"Đồ thị con Neo4j: {g.number_of_nodes()} nút, {g.number_of_edges()} cạnh")
    ax.axis("off")
    save(fig, "kg_subgraph")


def fig_rag():
    r = load_json("rag_eval")["modes"]
    modes = [m for m in MODE_VI if m in r]
    fig, ax = plt.subplots(figsize=(W, 3.9))
    keys = [("hit@5", "Hit@5"), ("hit@10", "Hit@10"), ("mrr", "MRR"), ("ndcg", "nDCG@10")]
    x = np.arange(len(modes))
    wd = .2
    for i, (k, name) in enumerate(keys):
        ax.bar(x + (i - 1.5) * wd, [r[m]["all"][k] for m in modes], wd, label=name, color=SEQ[i])
    ax.set_xticks(x, [MODE_VI[m].replace(" + ", "\n+ ") for m in modes])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Điểm")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.17))
    save(fig, "rag_retrieval")
    fig, ax = plt.subplots(figsize=(W, 3.9))
    kinds = [("kind=same_en", "Sinh, Anh-Anh"), ("kind=same_vi", "Sinh, Việt-Việt"), ("kind=cross_vi_on_en", "Sinh, hỏi Việt nguồn Anh"), ("kind=hand", "Tự viết")]
    for i, (k, name) in enumerate(kinds):
        ax.bar(x + (i - 1.5) * wd, [r[m].get(k, {}).get("mrr", 0) for m in modes], wd, label=name, color=SEQ[i])
    ax.set_xticks(x, [MODE_VI[m].replace(" + ", "\n+ ") for m in modes])
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.27))
    save(fig, "rag_by_group")
    fig, ax = plt.subplots(figsize=(W, 3.2))
    bars(ax, [MODE_VI[m] for m in modes], [r[m]["latency_ms"]["p50"] for m in modes], C["orange"], lambda v: f"{vn(v)} ms")
    p95 = [r[m]["latency_ms"]["p95"] for m in modes]
    ax.scatter(p95, range(len(modes)), color=C["navy"], zorder=3, label="p95", marker="D")
    ax.set_xscale("log")
    ax.set_xlim(1, max(p95) * 6)
    ax.set_xlabel("Độ trễ một truy vấn (ms, thang log), CPU; cột: trung vị")
    ax.legend(loc="lower right")
    save(fig, "rag_latency")


if __name__ == "__main__":
    for f in (fig_sources, fig_chunk_len, fig_kg, fig_subgraph, fig_rag):
        try:
            f()
        except FileNotFoundError as e:
            print("chưa đủ dữ liệu:", f.__name__, e, flush=True)
    print(json.dumps(sorted(p.name for p in __import__("dnx.common", fromlist=["FIGS"]).FIGS.glob("*") if p.name.startswith(("kb_", "kg_", "rag_")))), flush=True)
