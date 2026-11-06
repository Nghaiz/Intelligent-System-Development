"""Bảng, khối Cypher và vết suy luận Chương 5 -> Report/TieuLuan Final/generated/ch5_*. Đọc từ outputs/metrics."""
from tlf import ch5_neo as N
from tlf import ch5_schema as S
from tlf.ch5_data import DATA5, read_jsonl
from tlf.common import GEN, METRICS, load_json
from tlf.texfmt import best, num, table, tex_escape

DS_VN = {"hetionet": "Hetionet", "vn30": "VN30", "hotpot": "HotpotQA"}
RET_KEYS = [("recall@1", "R@1"), ("recall@3", "R@3"), ("recall@5", "R@5"), ("recall@10", "R@10"), ("mrr", "MRR"), ("ndcg", "nDCG")]
TYPE_VN = {"count_1hop": "Đếm 1 bước", "lookup_1hop": "Tra 1 bước", "stat_1hop": "Số liệu kỳ", "intersect_2hop": "Giao 2 bước",
           "filter_2hop": "Lọc 2 bước", "aggregate_2hop": "Tổng hợp 2 bước", "compare_2hop": "So sánh 2 bước",
           "bridge": "Cầu nối (bridge)", "comparison": "So sánh (comparison)"}
ONTOLOGY = [
    ("Hetionet", "Disease", "Bệnh", "name, hid", "ASSOCIATES, LOCALIZES, PRESENTS, RESEMBLES"),
    ("Hetionet", "Compound", "Hợp chất, thuốc", "name, hid", "TREATS, PALLIATES, BINDS, CAUSES"),
    ("Hetionet", "Gene", "Gen", "name, hid", "INTERACTS, PARTICIPATES"),
    ("Hetionet", "Pathway", "Đường truyền", "name, hid", "(từ Gene) PARTICIPATES"),
    ("Hetionet", "SideEffect", "Tác dụng phụ", "name, hid", "(từ Compound) CAUSES"),
    ("VN30", "Company", "Công ty niêm yết", "ticker, name", "IN\\_SECTOR, MEMBER\\_OF, CORRELATED"),
    ("VN30", "Sector", "Ngành", "name", "(từ Company) IN\\_SECTOR"),
    ("VN30", "YearStat", "Chỉ số năm", "ret\\_pct, vol\\_pct, high, low", "OF\\_YEAR, (từ Company) HAS\\_YEAR"),
    ("VN30", "MonthStat", "Chỉ số tháng", "month, ret\\_pct", "OF\\_MONTH, (từ Company) HAS\\_MONTH"),
    ("HotpotQA", "TextEntity", "Thực thể văn bản", "name", "LINKS\\_TO"),
    ("Cả ba", "Chunk", "Đoạn tri thức", "text, source, embedding", "ABOUT, MENTIONS"),
]
CYPHER = {"schema": lambda: ";\n".join(S.CONSTRAINTS[:3] + S.INDEXES[-1:]) + ";\n" + S.VECTOR_INDEX.format(
              name="vec_hetionet", label=S.DATASETS["hetionet"], dim=S.EMBED_DIM) + ";",
          "load": lambda: S.LOAD_EDGES, "drug": lambda: S.Q_DRUG_TARGET_ASSOC, "sector": lambda: S.Q_TOP_YEAR_IN_SECTOR,
          "corr": lambda: S.Q_TOP_CORRELATED, "bridge": lambda: S.Q_BRIDGE, "link": lambda: S.LINK_ENTITIES,
          "vector": lambda: S.VECTOR_SEARCH, "retrieve": lambda: S.GRAPH_RETRIEVE}
STEP = {"thought": ("cPurple", "THOUGHT"), "action": ("cOrange", "ACTION"), "observation": ("cTeal", "OBSERV."),
        "final": ("cGreen", "FINAL")}


def t_ontology():
    labels = {r["label"]: r["n"] for r in load_json("ch5_graph")["by_label"]["rows"]}
    rows = [[d, f"\\texttt{{{lab}}}", vn, num(labels[lab], 0), f"\\footnotesize {props}", f"\\footnotesize {rel}"]
            for d, lab, vn, props, rel in ONTOLOGY]
    table("ch5_ontology", "llLr" .replace("L", "L{2.3cm}") + "L{3.0cm}L{3.9cm}",
          ["Miền", "Nhãn", "Nghĩa", "Số nút", "Thuộc tính", "Quan hệ đi ra"], rows)


def t_graph():
    g = load_json("ch5_graph")
    rows = []
    for ds, name in DS_VN.items():
        d = g[ds]
        rows.append([name, num(d["nodes"], 0), num(d["edges"], 0), num(d["chunks"], 0), num(d["degree"]["mean"], 2),
                     num(d["degree"]["median"], 0), num(d["degree"]["max"], 0), tex_escape(d["hubs"]["rows"][0]["name"])])
    table("ch5_graph", "lrrrrrrl", ["Miền", "Thực thể", "Cạnh thực thể", "Chunk", "Bậc TB", "Trung vị", "Bậc lớn nhất", "Nút trung tâm"], rows)


def t_questions():
    q = load_json("ch5_questions")
    rows = []
    for ds, name in DS_VN.items():
        ex = {}
        for r in read_jsonl(DATA5 / f"qa_{ds}.jsonl"):
            ex.setdefault(r["type"], r)
        for t, n in sorted(q[ds]["by_type"].items()):
            rows.append([name, TYPE_VN[t], num(n, 0), f"\\footnotesize {tex_escape(ex[t]['question'][:74])}",
                         f"\\footnotesize {tex_escape(str(ex[t]['answer'])[:18])}"])
    table("ch5_questions", "lL{2.7cm}rL{6.6cm}L{2.0cm}", ["Miền", "Loại", "Số câu", "Ví dụ", "Đáp án"], rows)


def t_retrieval():
    rows = load_json("ch5_retrieval")["rows"]
    out = []
    for i, (ds, name) in enumerate(DS_VN.items()):
        g = [r for r in rows if r["dataset"] == ds]
        cols = {k: best([r[k] for r in g], [num(r[k], 3) for r in g]) for k, _ in RET_KEYS}
        if i:
            out.append("\\midrule")
        for j, r in enumerate(g):
            out.append([f"{name} (n={g[0]['n']})" if j == 0 else "", r["model"], *[cols[k][j] for k, _ in RET_KEYS], num(r["ms_p50"], 1)])
    table("ch5_ret", "llrrrrrrr", ["Miền", "Bộ truy xuất", *[h for _, h in RET_KEYS], "ms"], out,
          note="ms: trung vị thời gian một truy vấn, chưa tính thời gian mã hóa câu hỏi. In đậm: tốt nhất trong miền.")


def t_ret_type():
    rows = load_json("ch5_retrieval")["by_type"]["rows"]
    keys = sorted({(r["dataset"], r["type"]) for r in rows}, key=lambda x: (list(DS_VN).index(x[0]), x[1]))
    out = []
    for ds, t in keys:
        g = {r["model"]: r for r in rows if r["dataset"] == ds and r["type"] == t}
        vals = [g[m]["recall@5"] for m in ("BM25", "Dense", "GraphRAG", "Hybrid")]
        out.append([DS_VN[ds], TYPE_VN[t], num(g["BM25"]["n"], 0), *best(vals, [num(v, 3) for v in vals])])
    table("ch5_ret_type", "llrrrrr", ["Miền", "Loại câu hỏi", "n", "BM25", "Dense", "GraphRAG", "Hybrid"], out)


def t_compare():
    rows = load_json("ch5_compare")["rows"]
    for kind, name, head in (("llm", "ch5_gen", "LLM"), ("system", "ch5_sys", "Hệ thống")):
        out = []
        for i, (ds, dn) in enumerate(DS_VN.items()):
            g = [r for r in rows if r["kind"] == kind and r["dataset"] == ds]
            if not g:
                continue
            if i:
                out.append("\\midrule")
            em, f1, fa = (best([r[k] for r in g], [num(r[k], 3) for r in g]) for k in ("em", "f1", "faith"))
            for j, r in enumerate(g):
                out.append([f"{dn} (n={r['n']})" if j == 0 else "", tex_escape(r["model"]), em[j], f1[j], fa[j],
                            num(r["latency_p50"], 2), *([num(r["steps"], 2)] if kind == "system" else [])])
        heads = ["Miền", head, "EM", "F1", "Độ bám", "Trễ p50 (s)"] + (["Số bước"] if kind == "system" else [])
        table(name, "ll" + "r" * (len(heads) - 2), heads, out)


def t_traces():
    tr = load_json("ch5_traces")["traces"]
    for i, t in enumerate(tr):
        lines = [f"\\vetBuoc{{cSlate}}{{CÂU HỎI}}{{{tex_escape(t['question'])}}}"]
        for s in t["trace"]:
            col, lab = STEP[s["kind"]]
            lines.append(f"\\vetBuoc{{{col}}}{{{lab}}}{{{tex_escape(s['text'][:240])}}}")
        (GEN / f"ch5_trace_{t['dataset']}_{int(t['em'])}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("vết:", [(t["dataset"], t["em"]) for t in tr])


def t_samples():
    """Ba chunk đầu của mỗi miền có độ dài vừa phải, in nguyên văn đã cắt."""
    picks = {"hetionet": ("Hợp chất Metformin", "Bệnh type 2 diabetes mellitus có triệu"), "vn30": ("Năm 2023, cổ phiếu FPT", "FPT là mã"),
             "hotpot": ("Scott Derrickson.", "Ed Wood.")}
    for ds, heads in picks.items():
        rows = read_jsonl(DATA5 / f"chunks_{ds}.jsonl")
        got = [next(r for r in rows if r["text"].startswith(h)) for h in heads]
        lines = [tex_escape(f"{r['id']}: {r['text'][:210]} ...") + "\\par" for r in got]
        (GEN / f"ch5_sample_{ds}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def query_outputs():
    """Chạy lại các truy vấn trong báo cáo và ghi vài dòng đầu của kết quả thật."""
    runs = {"drug": (S.Q_DRUG_TARGET_ASSOC, {"disease": "H:Disease::DOID:9352"}), "sector": (S.Q_TOP_YEAR_IN_SECTOR, {"year": 2023}),
            "bridge": (S.Q_BRIDGE, {"k": 3})}
    for name, (q, params) in runs.items():
        rows = N.cypher(q, params)[:5]
        text = "\n".join(tex_escape(", ".join(f"{k}={v}" for k, v in r.items())) + "\\par" for r in rows)
        (GEN / f"ch5_out_{name}.tex").write_text(text + "\n", encoding="utf-8", newline="\n")


def errors():
    """Liệt kê câu sai của LLM tốt nhất để đọc thủ công lúc viết phân tích lỗi."""
    for m in ("qwen2.5-7b",):
        path = DATA5 / f"gen_{m}.jsonl"
        if path.exists():
            bad = [r for r in read_jsonl(path) if r["em"] == 0]
            for r in bad[::6][:12]:
                print(r["dataset"], r["type"], r["hit"], "|", r["question"][:70], "|", r["gold"], "|", r["answer"][:50])


def cypher_files():
    for name, fn in CYPHER.items():
        (GEN / f"ch5_cy_{name}.cypher").write_text(fn() + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    cypher_files()
    t_ontology()
    t_graph()
    t_questions()
    t_samples()
    query_outputs()
    if (METRICS / "ch5_retrieval.json").exists():
        t_retrieval()
        t_ret_type()
    if (METRICS / "ch5_compare.json").exists():
        t_compare()
    if (METRICS / "ch5_traces.json").exists():
        t_traces()
    errors()
