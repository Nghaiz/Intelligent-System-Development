"""Hình dữ liệu Chương 5 -> Report/TieuLuan Final/figures/ch5_*.pdf. Thiếu kết quả nào thì bỏ qua hình đó và in cảnh báo."""
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tlf import ch5_neo as N
from tlf.ch5_data import RAW, read_jsonl
from tlf.common import METRICS, load_json
from tlf.plot_style import C, H1, SEQ, W, W2, save, vn_axis

DS_VN = {"hetionet": "Hetionet", "vn30": "VN30", "hotpot": "HotpotQA"}
LABEL_COLOR = {"Disease": C["green"], "Compound": C["blue"], "Gene": C["teal"], "Symptom": C["orange"], "Anatomy": C["pink"],
               "Company": C["orange"], "Sector": C["brown"], "MarketIndex": C["red"], "Pathway": C["purple"]}


def fig_degree():
    g = load_json("ch5_graph")
    fig, ax = plt.subplots(figsize=(W, H1))
    for (ds, name), col in zip(DS_VN.items(), SEQ):
        d = np.array(g[ds]["degree_series"]["series"])
        bins = np.unique(np.logspace(0, np.log10(d.max() + 1), 22).astype(int))
        h, e = np.histogram(d, bins=bins)
        ax.loglog(e[:-1], np.maximum(h, 0.5), "o-", color=col, label=name, ms=4)
    ax.set(xlabel="Bậc của nút", ylabel="Số nút")
    ax.legend()
    save(fig, "ch5_degree")


def _short(n):
    return n["id"].split(":")[1] if n["label"] == "Company" else n["name"][:16]


def draw_sub(seed, name, depth=1, limit=46):
    sub = N.subgraph(seed, depth, limit)
    G = nx.Graph()
    # nút chỉ số kỳ dày đặc che mất cấu trúc nên không vẽ
    kind = {n["id"]: n for n in sub["nodes"] if n["label"] not in ("YearStat", "MonthStat")}
    G.add_nodes_from(kind)
    G.add_edges_from((e["source"], e["target"]) for e in sub["links"] if e["source"] in kind and e["target"] in kind)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(W2 + 0.5, W2 + 0.3))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9CA3AF", width=0.7)
    colors = [LABEL_COLOR.get(kind[n]["label"], C["gray"]) for n in G]
    sizes = [min(420, 50 + 9 * G.degree(n)) for n in G]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, linewidths=0.4, edgecolors="white")
    top = sorted(G, key=G.degree, reverse=True)[:4]
    nx.draw_networkx_labels(G, pos, {n: _short(kind[n]) for n in top}, ax=ax, font_size=9, font_family="DejaVu Sans")
    ax.axis("off")
    save(fig, name)


def fig_hetionet_edges():
    rows = load_json("ch5_graph")["hetionet"]["by_type"]["rows"][:9]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.barh([r["type"] for r in rows][::-1], [r["n"] for r in rows][::-1], color=C["teal"])
    vn_axis(ax, "x")
    ax.set_xlabel("Số cạnh")
    ax.grid(axis="y", visible=False)
    save(fig, "ch5_het_edges")


def fig_vn30_corr():
    from tlf.ch5_kg import company_frame, correlations
    rho, _, _ = correlations(company_frame())
    order = rho.mean().sort_values(ascending=False).index
    m = rho.loc[order, order]
    fig, ax = plt.subplots(figsize=(W, 5.0))
    im = ax.imshow(m, vmin=0, vmax=0.8, cmap="YlOrRd")
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=9)
    ax.set_yticks(range(len(order)), order, fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Hệ số tương quan")
    save(fig, "ch5_vn30_corr")


def fig_hotpot():
    rows = read_jsonl(RAW / "hotpot_val.jsonl")[:1000]
    q = [len(r["question"].split()) for r in rows]
    p = [len(x["text"].split()) for r in rows for x in r["paragraphs"]]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7))
    ax[0].hist(q, bins=25, color=C["purple"])
    ax[0].set(xlabel="Số từ của câu hỏi", ylabel="Số câu")
    ax[1].hist(np.minimum(p, 300), bins=30, color=C["sky"])
    ax[1].set(xlabel="Số từ của đoạn (cắt ở 300)", ylabel="Số đoạn")
    vn_axis(ax[1])
    fig.tight_layout()
    save(fig, "ch5_hotpot_len")


def grouped(ax, groups, series, values, legend=True):
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    for i, s in enumerate(series):
        ax.bar(x + i * width - 0.4 + width / 2, [values[(g, s)] for g in groups], width, label=s, color=SEQ[i])
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)
    if legend:
        ax.legend(ncol=len(series), fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.2))


def fig_retrieval():
    rows = load_json("ch5_retrieval")["rows"]
    series = ["BM25", "Dense", "GraphRAG", "Hybrid"]
    fig, ax = plt.subplots(figsize=(W, 3.1))
    grouped(ax, list(DS_VN.values()), series, {(DS_VN[r["dataset"]], r["model"]): r["recall@5"] for r in rows})
    ax.set(ylabel="Recall@5", ylim=(0, 1))
    save(fig, "ch5_ret_bars")


def fig_compare():
    rows = load_json("ch5_compare")["rows"]
    gen = [r for r in rows if r.get("kind") == "llm"]
    models = sorted({r["model"] for r in gen})
    fig, ax = plt.subplots(figsize=(W, 3.0))
    grouped(ax, list(DS_VN.values()), models, {(DS_VN[r["dataset"]], r["model"]): r["em"] for r in gen})
    ax.set(ylabel="EM", ylim=(0, 1))
    save(fig, "ch5_gen_bars")
    sys_rows = [r for r in rows if r.get("kind") == "system"]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    grouped(ax, list(DS_VN.values()), ["Naive RAG", "GraphRAG", "Agent"],
            {(DS_VN[r["dataset"]], r["model"]): r["em"] for r in sys_rows})
    ax.set(ylabel="EM", ylim=(0, 1))
    save(fig, "ch5_agent_bars")


if __name__ == "__main__":
    fig_degree()
    fig_hetionet_edges()
    fig_vn30_corr()
    fig_hotpot()
    draw_sub("Metformin", "ch5_sub_het")
    draw_sub("VN30:FPT", "ch5_sub_vn30", limit=120)
    if (METRICS / "ch5_retrieval.json").exists():
        fig_retrieval()
    if (METRICS / "ch5_compare.json").exists():
        fig_compare()
