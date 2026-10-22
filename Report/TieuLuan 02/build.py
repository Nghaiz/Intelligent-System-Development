"""Dựng báo cáo tiểu luận bằng một lệnh:  python build.py [--no-pdf]

1. Sinh macro số liệu và bảng  -> generated/      (từ src/TieuLuan 02/outputs/metrics/*.json và data/)
2. Trích mã nguồn theo tên     -> code/           (mỗi \\maNguon{module__ten} trong chapters/*.tex; truy vấn SPARQL -> .rq)
3. Xóa cache của hình TikZ nào có mã hoặc số liệu đổi
4. latexmk -lualatex -shell-escape main.tex, rồi chép PDF thành tieuluan02_CT_07_nghiand.pdf

Chương không gõ tay con số thực nghiệm nào: chỉ gọi \\SL{tệp/khoá/...} hoặc \\input một bảng sinh ra ở đây.
"""
import ast
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src" / "TieuLuan 02"
METRICS = SRC / "outputs" / "metrics"
GEN = HERE / "generated"
CODE = HERE / "code"
PDF_NAME = "tieuluan02_CT_07_nghiand.pdf"
CODE_DIRS = ("tl2", "scripts", "app")
SKIP_FILES = {"retrieval_perq"}                      # kết quả từng câu hỏi: chỉ dùng cho hình, không sinh macro
# mảng dữ liệu của hình và bảng, không phải số liệu chữ; so khớp cả đoạn (key + "/")
SKIP_KEYS = ("/rows/", "/traces/", "/per_q/", "/reports/", "/examples/", "/layout/", "/history/", "/test_series/",
             "/sample_docs/", "/sample_triples/", "/query/")
MAX_LIST = 13                                        # danh sách dài hơn là chuỗi thời gian / mẫu vẽ hình
MAX_STR = 120

STRAT = {"fixed": "Cố định 400 ký tự", "recursive": "Đệ quy (recursive)", "structure": "Theo cấu trúc tiêu đề",
         "semantic": "Ngữ nghĩa (semantic)", "parent_child": "Cha -- con"}
RET = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense e5-base", "hybrid": "Hybrid RRF", "hybrid_rerank": "Hybrid + rerank"}
EMB = {"minilm": "paraphrase-multilingual-MiniLM-L12-v2", "e5-small": "multilingual-e5-small",
       "e5-base": "multilingual-e5-base"}
ADV = {"dense": "Dense e5-base (gốc)", "hyde": "HyDE (chỉ đoạn giả định)", "hyde_plus_q": "HyDE + câu hỏi (RRF)",
       "multiquery": "Multi-Query (4 truy vấn, RRF)"}
GRAG = {"naive": "Naive RAG (cố định, dense)", "advanced": "Advanced RAG (hybrid + rerank)",
        "graph_local": "GraphRAG cục bộ", "graph_local_global": "GraphRAG cục bộ + toàn cục"}
ASSET = {"FPT": ("FPT", "đồng"), "HPG": ("HPG", "đồng"), "VCB": ("VCB", "đồng"), "VIC": ("VIC", "đồng"),
         "VN30ETF": ("ETF E1VFVN30", "đồng"), "GOLD_VND": ("Vàng TG quy đổi", "triệu đ/lượng"),
         "XAU": ("XAU/USD", "USD/ounce"), "USDVND": ("USD/VND", "đồng"), "DXY": ("DXY", "điểm")}
FC = {"bilstm": "BiLSTM", "lstm": "LSTM", "ridge": "Ridge (30 trễ)", "naive": "Giữ nguyên giá"}
TOOL_SHORT = {"chan_doan_nguy_co": "MLP", "kg_y_te": "KG-YT", "kg_tai_chinh": "KG-TC", "du_bao_bilstm": "BiLSTM",
              "quy_doi_vang": "Vàng", "du_lieu_thi_truong": "Giá", "truy_xuat_rag": "RAG"}
SPARQL_VN = {"q1_tuong_tac_ruou": "Thuốc tương tác nguy hiểm với rượu bia, đọc mức độ qua reification",
             "q2_thuc_pham_gi": "Món ăn và chỉ số GI theo lớp con của Food",
             "q3_tieu_chuan": "Tiêu chuẩn chẩn đoán ĐTĐ típ 2 và ngưỡng",
             "q4_thuoc_theo_lop": "Đếm thuốc theo nhóm (lớp con của Drug)",
             "q5_lan_truyen_fed": "Mọi biến chịu tác động từ lãi suất Fed (đường thuộc tính +)",
             "q6_co_phieu_hose": "Cổ phiếu niêm yết trên HOSE và ngành",
             "q7_bat_loi_ty_gia": "Doanh nghiệp bất lợi khi USD/VND tăng, kèm cơ chế"}


# ---------------------------------------------------------------- định dạng số kiểu Việt
def num(v, decimals=2, sign=False):
    s = f"{v:+,.{decimals}f}" if sign else f"{v:,.{decimals}f}"
    s = s.replace(",", "\0").replace(".", "{,}").replace("\0", "{.}")
    s = s.replace("-", "\\ensuremath{-}")
    return s.replace("+", "\\ensuremath{+}") if sign else s


def sci(v):
    if v == 0:
        return "\\ensuremath{< 10^{-300}}"
    m, e = f"{v:.2e}".split("e")
    return f"\\ensuremath{{{m.replace('.', '{,}')} \\times 10^{{{int(e)}}}}}"


def fmt(v):
    if isinstance(v, bool):
        return "đúng" if v else "sai"
    if isinstance(v, int):
        return num(v, 0)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if v != 0 and abs(v) < 1e-3:
            return sci(v)
        d = 4 if abs(v) < 1 else 3 if abs(v) < 10 else 2 if abs(v) < 1000 else 0
        return num(v, d)
    return tex_escape(str(v))


def tex_escape(s):
    s = unicodedata.normalize("NFC", str(s))
    s = (s.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}").replace("_", "\\_")
          .replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("$", "\\$")
          .replace("^", "\\^{}").replace("~", "\\textasciitilde{}"))
    # nháy kép ASCII in thành ” ở cả hai đầu với font Times; cặp "..." đổi sang ``...'' của LaTeX
    return re.sub(r'"([^"]*)"', r"``\1''", s)


def short(s, n):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def flatten(obj, prefix):
    if isinstance(obj, list) and len(obj) > MAX_LIST:
        return
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    else:
        yield prefix, obj
        return
    for k, v in items:
        yield from flatten(v, f"{prefix}/{k}")


def gen_macros(metrics):
    lines = ["% Sinh tự động bởi build.py từ outputs/metrics/*.json. Không sửa tay."]
    for name, data in metrics.items():
        if name in SKIP_FILES:
            continue
        for key, v in flatten(data, name):
            if any(s in key + "/" for s in SKIP_KEYS) or any(c in key for c in "\\%#{}~$&"):
                continue
            if isinstance(v, str) and len(v) > MAX_STR or v is None:
                continue
            lines.append(f"\\defSL{{{key}}}{{{fmt(v)}}}")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                lines.append(f"\\defSL{{{key}Raw}}{{{v:.6g}}}")
                if abs(v) <= 1.5:
                    lines.append(f"\\defSL{{{key}Pct}}{{{num(100 * v, 2)}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- bảng
def wrap_head(h, align):
    """Ngắt tiêu đề cột dài thành hai dòng (\\makecell) để bảng nhiều cột không bị co nhỏ chữ."""
    if len(h) <= 9 or " " not in h or "\\\\" in h or "$" in h:
        return h
    spaces = [i for i, ch in enumerate(h) if ch == " "]
    cut = min(spaces, key=lambda i: abs(i - len(h) / 2))
    return f"\\makecell[{align}]{{{h[:cut]}\\\\{h[cut + 1:]}}}"


def table(spec, header, rows, note=None):
    aligns = [c for c in re.sub(r"\{[^}]*\}", "", spec) if c in "lrcpLCR"]
    header = [wrap_head(h, "l" if a in "lpL" else "r" if a in "rR" else "c") for h, a in zip(header, aligns)]
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [r if isinstance(r, str) else " & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    body = "\n".join(out) + "\n"
    # ghi chú ra tệp riêng <tên>_note.tex: \bang co bảng theo bề rộng trang, ghi chú thì không được co
    return (body, note) if note else body


def best(values, formatted, higher=True):
    b = max(values) if higher else min(values)
    return [f"\\textbf{{{f}}}" if v == b else f for v, f in zip(values, formatted)]


def p4(v):
    return num(v, 4)


def pct(v, d=1):
    return num(100 * v, d)


# ---------------------------------------------------------------- dữ liệu: kho tài liệu và bộ hỏi đáp
def corpus_tables():
    rows, total = [], [0, 0]
    for dom, lab in (("health", "Y tế"), ("finance", "Tài chính")):
        for p in sorted((SRC / "data" / "corpus" / dom).glob("*.md")):
            text = unicodedata.normalize("NFC", p.read_text(encoding="utf-8"))
            title = re.search(r"^#\s+(.+)$", text, re.M).group(1).strip()
            n_sec = len(re.findall(r"^##\s", text, re.M))
            rows.append([f"\\texttt{{{tex_escape(p.stem)}}}", lab, tex_escape(title), str(n_sec), num(len(text), 0)])
            total[0] += n_sec
            total[1] += len(text)
    rows += ["\\midrule", ["\\textbf{Tổng}", "", f"{len(rows)} tài liệu", str(total[0]), num(total[1], 0)]]
    out = {"tab_corpus": table("llL{7.6cm}rr", ["Mã tài liệu", "Miền", "Tiêu đề", "Số mục", "Ký tự"], rows,
                               "Mỗi tài liệu Markdown ghi rõ văn bản gốc (Quyết định 5481/QĐ-BYT, ADA, Nghị định, "
                               "báo cáo thường niên) ở dòng đầu; kho được viết lại bằng lời của tác giả để học tập.")}
    qa = json.loads((SRC / "data" / "qa" / "qa_eval.json").read_text(encoding="utf-8"))
    pick = [q for q in qa if q["domain"] == "health"][:4] + [q for q in qa if q["domain"] == "finance"][:4]
    out["tab_qa_sample"] = table("lL{6.4cm}L{6.0cm}", ["Mã", "Câu hỏi", "Đáp án tham chiếu"],
                                 [[q["id"], tex_escape(q["q"]), tex_escape(q["a"])] for q in pick],
                                 f"Trích 8 trong {len(qa)} cặp hỏi đáp; mỗi câu gắn tài liệu chứa đáp án để chấm hit@k.")
    return out


# ---------------------------------------------------------------- chương 3: truy xuất
def retrieval_tables(m):
    g = m["retrieval_grid"]
    out = {}
    cs = g["chunk_stats"]
    rows = [[STRAT[s], num(c["n_chunks"], 0), "--" if c["n_parents"] is None else num(c["n_parents"], 0),
             num(c["mean_len"], 1), num(c["min_len"], 0), num(c["max_len"], 0), pct(c["intact"])] for s, c in cs.items()]
    out["tab_chunk_stats"] = table("lrrrrrr", ["Chiến lược", "Số đoạn", "Đoạn cha", "Dài TB", "Ngắn nhất", "Dài nhất",
                                               "Giữ trọn đáp án (\\%)"], rows,
                                   "Độ dài tính bằng ký tự. Giữ trọn đáp án: tỉ lệ câu hỏi có ít nhất một đoạn chứa trọn "
                                   "câu đáp án gốc. Với cha -- con, đoạn con dùng để tìm, đoạn cha để đưa vào ngữ cảnh.")
    for metric, lab in (("hit@1", "hit1"), ("mrr", "mrr")):
        rows = []
        for s in STRAT:
            vals = [g["grid"][s][r][metric] for r in RET]
            rows.append([STRAT[s]] + best(vals, [p4(v) for v in vals]))
        out[f"tab_grid_{lab}"] = table("l" + "r" * len(RET), ["Chiến lược chia đoạn"] + list(RET.values()), rows,
                                       f"{g['n_questions']} câu hỏi trên {g['n_docs']} tài liệu; chữ đậm: bộ truy xuất "
                                       "tốt nhất của mỗi chiến lược.")
    s = g["grid"]["structure"]
    keys = ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@10", "health_hit@1", "finance_hit@1"]
    cols = {k: best([s[r][k] for r in RET], [p4(s[r][k]) for r in RET]) for k in keys}
    lat = best([s[r]["latency_ms"] for r in RET], [num(s[r]["latency_ms"], 2) for r in RET], higher=False)
    rows = [[RET[r]] + [cols[k][i] for k in keys] + [lat[i]] for i, r in enumerate(RET)]
    out["tab_ret_detail"] = table("lrrrrrrrr", ["Bộ truy xuất", "hit@1", "hit@3", "hit@5", "MRR", "nDCG@10", "hit@1 y tế",
                                                "hit@1 tài chính", "Trễ (ms)"], rows,
                                  "Chia đoạn theo cấu trúc tiêu đề. Trễ: trung vị mỗi câu hỏi, gồm mã hóa câu hỏi trên GPU; "
                                  "rerank chấm lại 30 ứng viên của hybrid.")
    rows = [[num(r["size"], 0), num(r["n_chunks"], 0), num(r["mean_len"], 1), pct(r["intact"]), p4(r["bm25_hit@1"]),
             p4(r["bm25_mrr"]), p4(r["dense_hit@1"]), p4(r["dense_mrr"]), num(r["ctx_tokens_top5"], 0)]
            for r in m["retrieval_sweep"]]
    out["tab_sweep"] = table("rrrrrrrrr", ["Cỡ đoạn", "Số đoạn", "Dài TB", "Giữ trọn (\\%)", "BM25 hit@1", "BM25 MRR",
                                           "Dense hit@1", "Dense MRR", "Token top-5"], rows,
                             "Chia đệ quy, chồng lấn 15\\% cỡ đoạn; token top-5: ước lượng 1 token $\\approx$ 4 ký tự cho "
                             "năm đoạn đầu đưa vào LLM.")
    e = m["retrieval_embed"]
    keys = ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@10", "health_hit@1", "finance_hit@1"]
    cols = {k: best([e[x][k] for x in EMB], [p4(e[x][k]) for x in EMB]) for k in keys}
    out["tab_embed"] = table("lrrrrrrr", ["Mô hình nhúng", "hit@1", "hit@3", "hit@5", "MRR", "nDCG@10", "hit@1 y tế",
                                          "hit@1 tài chính"],
                             [[f"\\texttt{{{EMB[x]}}}"] + [cols[k][i] for k in keys] for i, x in enumerate(EMB)],
                             "Cùng chia đoạn theo cấu trúc, truy xuất dense thuần; e5 dùng tiền tố query:/passage:.")
    a = m["advanced"]
    keys = ["hit@1", "hit@5", "mrr", "ndcg@10"]
    cols = {k: best([a["agg"][x][k] for x in ADV], [p4(a["agg"][x][k]) for x in ADV]) for k in keys}
    rows = []
    for i, x in enumerate(ADV):
        r = a["agg"][x]
        gen = a["gen_s"].get("hyde" if x.startswith("hyde") else x)
        rows.append([ADV[x]] + [cols[k][i] for k in keys] +
                    (["--", "--"] if x == "dense" else [str(r["wins_vs_dense"]), str(r["losses_vs_dense"])]) +
                    ["--" if gen is None else num(gen, 2)])
    out["tab_advanced"] = table("lrrrrrrr", ["Phương pháp", "hit@1", "hit@5", "MRR", "nDCG@10", "Thắng", "Thua",
                                             "Sinh (s)"], rows,
                                "Thắng/thua: số câu hỏi có MRR cao/thấp hơn dense gốc. Sinh: trung vị thời gian Qwen2.5-1.5B "
                                "viết đoạn giả định hoặc ba câu hỏi biến thể.")
    rows = []
    for ex in a["examples"][:3]:
        rows.append([ex["id"], tex_escape(short(ex["q"], 90)), tex_escape(short(ex["hyde"], 260)),
                     tex_escape(short(ex["variants"][0].lstrip("0123456789. ").strip('"'), 110))])
    out["tab_hyde_ex"] = table("lL{3.6cm}L{6.0cm}L{3.6cm}", ["Mã", "Câu hỏi", "Đoạn giả định HyDE", "Biến thể 1"], rows,
                               "Văn bản sinh nguyên văn (cắt bớt); đoạn HyDE chứa cả chi tiết đúng lẫn chi tiết bịa.")
    return out


def graphrag_tables(m):
    g = m["graphrag"]
    out = {}
    A = g["agg"]
    keys = ["ctx_cov", "ans_cov"]
    cols = {k: best([A[x][k] for x in GRAG], [p4(A[x][k]) for x in GRAG]) for k in keys}
    rows = [[GRAG[x], cols["ctx_cov"][i], cols["ans_cov"][i], num(A[x]["ctx_chars"], 0), num(A[x]["gen_s"], 2)]
            for i, x in enumerate(GRAG)]
    out["tab_graphrag"] = table("lrrrr", ["Phương pháp", "Độ phủ ngữ cảnh", "Độ phủ câu trả lời", "Ký tự ngữ cảnh",
                                          "Sinh (s)"], rows,
                                f"Trung bình trên {len(g['rows'])} câu hỏi đa bước. Độ phủ: tỉ lệ dữ kiện bắt buộc (nhóm từ "
                                "khóa) có mặt; cùng LLM Qwen2.5-1.5B, cùng prompt có trích dẫn.")
    rows = []
    for r in g["reports"]:
        kg = "Y tế" if any(n["comm"] == r["id"] and n["kg"] == "VietnamDiabetesKG" for n in g["layout"]["nodes"]) else "Tài chính"
        rows.append([f"C{r['id']}", kg, str(r["size"]), tex_escape(short(r["title"], 70)), str(len(r["facts"])),
                     str(len(r["sources"])), str(r["tokens"])])
    out["tab_communities"] = table("llrL{6.8cm}rrr", ["Cụm", "Đồ thị", "Nút", "Ba thực thể bậc cao nhất", "Dữ kiện",
                                                      "Nguồn", "Token tóm tắt"], rows,
                                   f"Louvain trên đồ thị hợp nhất, modularity {num(g['modularity'], 3)}; tóm tắt cộng đồng do "
                                   f"LLM viết từ các dữ kiện, tổng {num(g['summary_s'], 1)} s.")
    rows = []
    for i, r in enumerate(g["rows"], 1):
        rows.append([str(i), tex_escape(short(r["q"], 95))] +
                    [f"{num(r[x]['ctx_cov'], 2)} / {num(r[x]['ans_cov'], 2)}" for x in GRAG])
    out["tab_grag_perq"] = table("rL{6.0cm}cccc", ["STT", "Câu hỏi", "Naive", "Advanced", "Graph cục bộ",
                                                   "Graph + toàn cục"], rows,
                                 "Mỗi ô: độ phủ ngữ cảnh / độ phủ câu trả lời.")
    return out


# ---------------------------------------------------------------- chương 2, 4, 5: đồ thị tri thức
def kg_tables(m):
    k = m["kg"]
    out = {}
    lab = [("nodes", "Số nút (thực thể)"), ("edges", "Số cạnh (quan hệ)"), ("classes", "Số lớp trong ontology"),
           ("predicates", "Số vị từ (predicate)"), ("density", "Mật độ đồ thị"), ("avg_degree", "Bậc trung bình"),
           ("components", "Thành phần liên thông"), ("with_source", "Cạnh có trích nguồn"),
           ("rdf_triples", "Bộ ba RDF xuất ra"), ("inferred", "Bộ ba suy diễn thêm")]
    rows = [[t, fmt(k["health"][x]), fmt(k["finance"][x])] for x, t in lab]
    out["tab_kg_stats"] = table("lrr", ["Chỉ số", "VietnamDiabetesKG", "VietnamFinanceKG"], rows,
                                "Bộ ba RDF gồm cả nhãn, kiểu và thuộc tính cạnh (reification); suy diễn: bao đóng RDFS và "
                                "các luật đối xứng, bắc cầu, nghịch đảo.")
    for part in ("node_types", "relations"):
        h = sorted(k["health"][part].items(), key=lambda x: -x[1])
        f = sorted(k["finance"][part].items(), key=lambda x: -x[1])
        rows = []
        for i in range(max(len(h), len(f))):
            a = h[i] if i < len(h) else ("", "")
            b = f[i] if i < len(f) else ("", "")
            rows.append([f"\\texttt{{{tex_escape(a[0])}}}" if a[0] else "", str(a[1]),
                         f"\\texttt{{{tex_escape(b[0])}}}" if b[0] else "", str(b[1])])
        head = ["Lớp (y tế)", "Nút", "Lớp (tài chính)", "Nút"] if part == "node_types" else \
               ["Quan hệ (y tế)", "Cạnh", "Quan hệ (tài chính)", "Cạnh"]
        out[f"tab_kg_{part}"] = table("lrlr", head, rows)
    rows = [[f"Q{i}", "Y tế" if v["domain"] == "health" else "Tài chính", tex_escape(SPARQL_VN.get(q, q)), str(len(v["rows"])),
             str(v["rows_before_inference"]), num(v["ms"], 1)] for i, (q, v) in enumerate(k["sparql"].items(), 1)]
    out["tab_sparql"] = table("llL{7.2cm}rrr", ["Mã", "Miền", "Nội dung truy vấn", "Kết quả", "Trước suy diễn", "ms"], rows,
                              "Kết quả: số dòng trên đồ thị đã suy diễn; trước suy diễn: cùng truy vấn trên bộ ba gốc. "
                              "Các truy vấn theo lớp cha (Food, Drug, Stock) chỉ có kết quả sau khi suy diễn kiểu.")
    rows = [[f"\\texttt{{{tex_escape(' '.join(v['triple']))}}}", tex_escape(v["error"].split(": ", 1)[-1])]
            for v in k["violations"]]
    out["tab_violations"] = table("L{6.0cm}L{8.6cm}", ["Bộ ba cố tình sai", "Thông báo của bộ kiểm tra ràng buộc"], rows)
    t = k["tabular"]["diabetes"]
    out["tab_tabular"] = table("lr", ["Chỉ số", "Giá trị"],
                               [["Bản ghi CSV", num(t["records"], 0)], ["Văn bản hồ sơ sinh ra", num(t["documents"], 0)],
                                ["Bộ ba (triple)", num(t["triples"], 0)], ["Nút", num(t["nodes"], 0)],
                                ["Cạnh", num(t["edges"], 0)], ["Thời gian sinh văn bản (s)", num(t["doc_s"], 2)],
                                ["Thời gian dựng đồ thị (s)", num(t["kg_s"], 2)]])
    rows = []
    for pair, v in k["demo_finance"].items():
        a, b = pair.split("->")
        rows.append([f"\\texttt{{{tex_escape(a)}}} $\\to$ \\texttt{{{tex_escape(b)}}}", str(len(v["paths"])),
                     num(v["net"], 2, sign=True), tex_escape(v["verdict"])])
    out["tab_multihop"] = table("lrrL{6.4cm}", ["Câu hỏi tác động", "Số đường", "Ròng", "Kết luận của bộ suy luận"], rows,
                                "Dấu một đường $\\pi$ là tích dấu các cạnh; ròng $= \\sum_\\pi \\mathrm{dấu}(\\pi)/|\\pi|$ "
                                "với $|\\pi|$ là số bước, nên đường ngắn có trọng số lớn hơn.")
    rows = [[tex_escape(n), "tăng" if s > 0 else "giảm"] for n, s in k["shock_dxy"].items()]
    out["tab_shock"] = table("ll", ["Biến bị ảnh hưởng", "Chiều khi DXY tăng"], rows)
    h = k["demo_health"]
    groups = [("Metformin + rượu bia: cạnh trực tiếp", h["warning"]),
              ("Metformin + rượu bia: nguy cơ chung (2 bước)", h["shared_risk"]),
              ("Thịt gà (nhóm thực phẩm)", h["chicken"]), ("Tinh bột", h["rice"])]
    rows = []
    for name, items in groups:
        for i, it in enumerate(items):
            rows.append([tex_escape(name) if i == 0 else "", tex_escape(it)])
        rows.append("\\midrule")
    rows.pop()
    out["tab_demo_health"] = table("L{3.8cm}L{10.6cm}", ["Truy vấn trên KG", "Kết quả (bộ ba đã diễn đạt)"], rows,
                                   "Kết quả nguyên văn của \\texttt{drug\\_warnings} và \\texttt{food\\_advice} trên "
                                   "\\texttt{VietnamDiabetesKG}; mỗi bộ ba mang mã tài liệu nguồn.")
    return out


def market_tables(m):
    mk = m["market"]
    out = {}
    rows = []
    for s, (lab, unit) in ASSET.items():
        d = mk["describe"][s]
        rows.append([lab, num(d["last"], 1 if d["last"] < 1000 else 0), unit, pct(d["ann_ret"]), pct(d["ann_vol"]),
                     pct(d["max_dd"]), num(d["total_ret"], 2)])
    out["tab_market_desc"] = table("lllrrrr", ["Tài sản", "Giá cuối", "Đơn vị", "Lợi suất năm (\\%)", "Biến động năm (\\%)",
                                               "Sụt giảm lớn nhất (\\%)", "Tăng cả kỳ (lần)"], rows,
                                   f"Dữ liệu ngày {mk['range'][0]} -- {mk['range'][1]} ({num(mk['n_days'], 0)} phiên chung); "
                                   "giá cổ phiếu đã điều chỉnh; lợi suất năm tính từ lợi suất log trung bình $\\times 252$.")
    rows = []
    for e in mk["kg_edges_evidence"]:
        rows.append([f"\\texttt{{{tex_escape(e['source'])}}} $\\to$ \\texttt{{{tex_escape(e['target'])}}}",
                     "+" if e["theory_sign"] > 0 else "$-$", num(e["corr_20d"], 3, sign=True),
                     "\\textcolor{ForestGreen}{khớp}" if e["agree"] else "\\textcolor{BrickRed}{không khớp}"])
    out["tab_edges_evidence"] = table("lccc", ["Cạnh trong KG", "Dấu lý thuyết", "Tương quan 20 phiên", "Đối chiếu"], rows,
                                      "Tương quan Pearson của lợi suất cộng dồn 20 phiên trên toàn bộ giai đoạn.")
    return out


def forecast_tables(m):
    f = m["forecast"]
    out = {}
    rows = []
    for s in [x for x in f if x != "config"]:
        ratios = [f[s][x]["rmse_ratio"] for x in FC]
        rb = best(ratios, [num(v, 4) for v in ratios], higher=False)
        for i, x in enumerate(FC):
            r = f[s][x]
            rows.append([ASSET[s][0] if i == 0 else "", FC[x], num(1000 * r["rmse_ret"], 2), rb[i],
                         num(r["mae_price"], 1 if s == "GOLD_VND" else 0), "--" if x == "naive" else pct(r["dir_acc"]),
                         num(r["r2_ret"], 4, sign=True), str(r.get("epochs", "--")),
                         "--" if "train_s" not in r else num(r["train_s"], 2)])
        rows.append("\\midrule")
    rows.pop()
    out["tab_forecast"] = table("llrrrrrrr", ["Tài sản", "Mô hình", "RMSE ($\\times 10^{-3}$)", "RMSE / ngây thơ",
                                              "MAE giá", "Đúng chiều (\\%)", "$R^2$ lợi suất", "Epoch", "Huấn luyện (s)"],
                              rows, "Kiểm thử từ " + f["config"]["test_start"] + "; RMSE trên lợi suất log phiên kế tiếp; "
                              "giữ nguyên giá = dự báo lợi suất 0. MAE giá: đồng/cổ phiếu, vàng: triệu đồng/lượng. "
                              "Chữ đậm: tỉ số nhỏ nhất của mỗi tài sản (dưới 1 mới là thắng mô hình ngây thơ).")
    n = f["FPT"]["n"]
    c = f["config"]
    out["tab_fc_config"] = table("lr", ["Siêu tham số", "Giá trị"],
                                 [["Cửa sổ quan sát (phiên)", str(c["window"])], ["Kích thước ẩn mỗi chiều", str(c["hidden"])],
                                  ["Số lớp LSTM", str(c["layers"])], ["Dropout", num(c["dropout"], 1)],
                                  ["Đặc trưng đầu vào (cổ phiếu / vàng)", f"{n['features']} / {f['GOLD_VND']['n']['features']}"],
                                  ["Mẫu huấn luyện / kiểm định / kiểm thử (FPT)", f"{n['train']} / {n['val']} / {n['test']}"],
                                  ["Bắt đầu kiểm định / kiểm thử", f"{c['val_start']} / {c['test_start']}"],
                                  ["Số mẫu MC Dropout", "200"]])
    live = m["forecast_live"]
    rows = [[ASSET[s][0], r["last_date"], num(r["last_price"], 1 if s == "GOLD_VND" else 0),
             num(r["pred_return_pct"], 2, sign=True), num(r["mc_std_pct"], 3), num(r["rmse_pct"], 2),
             f"{num(r['interval90'][0], 1 if s == 'GOLD_VND' else 0)} -- {num(r['interval90'][1], 1 if s == 'GOLD_VND' else 0)}",
             pct(r["prob_up"])] for s, r in live.items()]
    out["tab_fc_live"] = table("llrrrrcr", ["Tài sản", "Phiên gần nhất", "Giá gần nhất", "Lợi suất dự báo (\\%)",
                                            "$\\sigma$ MC Dropout (\\%)", "RMSE kiểm thử (\\%)", "Khoảng dự báo 90\\%",
                                            "P(tăng) (\\%)"],
                               rows, "Mỗi tài sản 200 mẫu MC Dropout, mỗi mẫu cộng nhiễu $\\mathcal{N}(0, \\mathrm{RMSE}"
                                     "_{\\text{kiểm thử}})$; giá: đồng/cổ phiếu, vàng: triệu đồng/lượng.")
    return out


def diabetes_tables(m):
    d = m["diabetes"]
    out = {}
    D = d["data"]
    out["tab_dia_data"] = table("lr", ["Thuộc tính", "Giá trị"],
                                [["Bản ghi sau khi bỏ trùng", num(D["n"], 0)], ["Tỉ lệ mắc bệnh", pct(D["pos_rate"], 2) + "\\%"],
                                 ["Huấn luyện / kiểm định / kiểm thử", f"{num(D['n_train'], 0)} / {num(D['n_val'], 0)} / "
                                                                       f"{num(D['n_test'], 0)}"],
                                 ["Số đặc trưng sau mã hóa", str(D["n_features"])]])
    keys = ["acc", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    names = [("mlp", "MLP PyTorch (GPU)"), ("logreg", "Logistic Regression")]
    cols = {k: best([d[x][k] for x, _ in names], [p4(d[x][k]) for x, _ in names]) for k in keys}
    rows = [[n] + [cols[k][i] for k in keys] + [num(d[x]["threshold"], 3), num(d[x]["train_s"], 2)]
            for i, (x, n) in enumerate(names)]
    out["tab_dia_models"] = table("lrrrrrrrr", ["Mô hình", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
                                                "Ngưỡng", "Huấn luyện (s)"], rows,
                                  f"Tập kiểm thử {num(D['n_test'], 0)} bản ghi; ngưỡng chọn để F1 lớn nhất trên tập kiểm định. "
                                  f"MLP có {num(d['mlp']['params'], 0)} tham số, {d['mlp']['epochs']} epoch.")
    for x, _ in names:
        cm = d[x]["confusion"]
        out[f"tab_dia_cm_{x}"] = table("lrr", ["", "Dự đoán âm", "Dự đoán dương"],
                                       [["Thực tế âm", num(cm[0][0], 0), num(cm[0][1], 0)],
                                        ["Thực tế dương", num(cm[1][0], 0), num(cm[1][1], 0)]])
    return out


# ---------------------------------------------------------------- chương 6: tác tử
def tool_specs():
    src = (SRC / "tl2" / "agent.py").read_text(encoding="utf-8")
    return re.findall(r'Tool\("(\w+)", "(\w+)", "([^"]+)"', src)


def agent_tables(m):
    a = m["agent"]
    G = a["agg"]
    out = {}
    rows = [[f"\\texttt{{{tex_escape(n)}}}", k, tex_escape(desc), str(G["tool_calls"].get(n, 0)),
             num(G["tool_ms"][n], 1) if n in G["tool_ms"] else "--"] for n, k, desc in tool_specs()]
    out["tab_agent_tools"] = table("llL{7.0cm}rr", ["Công cụ", "Loại", "Chức năng", "Lần gọi", "Trễ TV (ms)"], rows,
                                   f"Lần gọi và trung vị độ trễ đo trên {len(a['rows'])} kịch bản đánh giá; DL: học sâu, KG: đồ thị "
                                   "tri thức, RAG: truy xuất văn bản, CALC: tính toán trên dữ liệu thị trường.")
    rows = []
    for r in a["rows"]:
        rows.append([str(r["id"] + 1), "Y tế" if r["domain"] == "health" else "TC", tex_escape(short(r["q"], 76)),
                     ", ".join(TOOL_SHORT[t] for t in r["tools"]), num(r["tool_recall"], 2), num(r["coverage"], 2),
                     num(r["coverage_llm_only"], 2), "LLM" if r["used"] == "llm" else "khuôn"])
    # 16 dòng phải nằm trên một trang: cột câu hỏi rộng để mỗi dòng chỉ 2 hàng chữ, tiêu đề cột số xếp hai tầng
    out["tab_agent_scen"] = table("rlL{7.6cm}L{2.6cm}rrrl", ["STT", "Miền", "Câu hỏi", "Công cụ đã gọi", "Recall",
                                                              r"\makecell[r]{Độ\\phủ}", r"\makecell[r]{Chỉ\\LLM}",
                                                              r"\makecell[l]{Trả\\lời}"], rows,
                                  "Recall: tỉ lệ công cụ kỳ vọng đã được gọi. Độ phủ: tỉ lệ dữ kiện bắt buộc có trong câu trả "
                                  "cuối; chỉ LLM: cùng Qwen2.5-1.5B trả lời không có công cụ. Trả lời: bản LLM qua kiểm "
                                  "chứng hay bản tổng hợp theo khuôn.")
    lab = [("tool_recall", "Recall công cụ"), ("tool_precision", "Precision công cụ"), ("domain_acc", "Nhận đúng miền"),
           ("coverage", "Độ phủ dữ kiện (agent)"), ("coverage_llm_only", "Độ phủ dữ kiện (chỉ LLM)"),
           ("faith", "Trung thực số học bản nháp LLM (agent)"), ("faith_llm_only", "Trung thực số học (chỉ LLM)"),
           ("llm_pass", "Tỉ lệ bản LLM qua kiểm chứng")]
    rows = [[t] + [num(G[d][x], 3) for d in ("health", "finance", "all")] for x, t in lab]
    rows += ["\\midrule", ["Tổng thời gian TB (s)"] + [num(G[d]["total_ms"] / 1000, 2) for d in ("health", "finance", "all")],
             ["Thời gian sinh của LLM TB (s)"] + [num(G[d]["llm_s"], 2) for d in ("health", "finance", "all")]]
    out["tab_agent_agg"] = table("lrrr", ["Chỉ số", "Y tế", "Tài chính", "Chung"], rows,
                                 "Trung thực số học: tỉ lệ con số trong văn bản có mặt trong bằng chứng hoặc câu hỏi.")
    return out


def trace_files(m):
    """Vết suy luận Thought/Action/Observation của hai kịch bản tiêu biểu -> generated/trace_<miền>.tex."""
    for dom, t in m["agent"]["traces"].items():
        lines = ["% Sinh bởi build.py từ outputs/metrics/agent.json", f"\\vetCauHoi{{{tex_escape(t['question'])}}}"]
        for s in t["trace"]:
            if s["type"] == "THOUGHT":
                lines.append(f"\\vetThought{{{tex_escape(s['text'])}}}")
            elif s["type"] == "ACTION":
                args = s["args"]
                args = ", ".join(args) if isinstance(args, list) else ("" if args is None else
                                                                        "hồ sơ" if isinstance(args, dict) else short(args, 80))
                lines.append(f"\\vetAction{{{tex_escape(s['tool'])}}}{{{s['kind']}}}{{{num(s['ms'], 1)}}}{{{tex_escape(args)}}}")
            else:
                lines.append(f"\\vetObs{{{tex_escape(short(s['text'], 330))}}}")
        v = t["verification"]
        lines.append(f"\\vetVerify{{{v['n_citations']}}}{{{num(v['numeric_faithfulness'], 2)}}}"
                     f"{{{'bản LLM' if v['used'] == 'llm' else 'bản theo khuôn'}}}{{{num(t['total_ms'] / 1000, 2)}}}")
        (GEN / f"trace_{dom}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        ans = t["answer"].split("\n")
        body = "\n".join(f"\\vetDong{{{tex_escape(short(x, 300))}}}" for x in ans if x.strip())
        (GEN / f"answer_{dom}.tex").write_text("% Sinh bởi build.py\n" + body + "\n", encoding="utf-8")
        llm = v.get("llm_text", "")
        body = "\n".join(f"\\vetDong{{{tex_escape(short(x, 300))}}}" for x in llm.split("\n") if x.strip())
        (GEN / f"llmdraft_{dom}.tex").write_text("% Sinh bởi build.py\n" + body + "\n", encoding="utf-8")
    return 2


def gen_tables(m):
    GEN.mkdir(exist_ok=True)
    builders = [corpus_tables, lambda: retrieval_tables(m), lambda: graphrag_tables(m), lambda: kg_tables(m),
                lambda: market_tables(m), lambda: forecast_tables(m), lambda: diabetes_tables(m), lambda: agent_tables(m)]
    n = 0
    for build in builders:
        try:
            for name, body in build().items():
                note = None
                if isinstance(body, tuple):
                    body, note = body
                (GEN / f"{name}.tex").write_text("% Sinh bởi build.py\n" + body, encoding="utf-8")
                note_file = GEN / f"{name}_note.tex"
                if note:
                    note_file.write_text("% Sinh bởi build.py\n" + note + "\n", encoding="utf-8")
                else:
                    note_file.unlink(missing_ok=True)
                n += 1
        except KeyError as e:
            print("CẢNH BÁO: chưa sinh được nhóm bảng, thiếu khóa", e)
    if "agent" in m:
        trace_files(m)
    return n


# ---------------------------------------------------------------- trích mã nguồn
def find_module(module):
    for d in CODE_DIRS:
        p = SRC / d / f"{module}.py"
        if p.exists():
            return p
    raise KeyError(f"Không thấy mô-đun {module} trong {', '.join(CODE_DIRS)}")


def extract(module, dotted):
    """Trích nguyên văn hàm, lớp, phương thức (A.b) hoặc phép gán cấp mô-đun (TÊN) từ mã nguồn đã chạy."""
    path = find_module(module)
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    nodes = ast.parse(src).body
    target = None
    for part in dotted.split("."):
        target = None
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part:
                target = n
            elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == part for t in n.targets):
                target = n
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == part:
                target = n
            if target is not None:
                break
        if target is None:
            raise KeyError(f"Không thấy '{dotted}' trong {path.name}")
        nodes = getattr(target, "body", [])
    decos = getattr(target, "decorator_list", [])
    start = min([target.lineno] + [d.lineno for d in decos]) - 1
    block = lines[start:target.end_lineno]
    indent = len(block[0]) - len(block[0].lstrip())
    rel = path.relative_to(SRC).as_posix()
    return f"# {rel}\n" + "\n".join(l[indent:] for l in block) + "\n"


def gen_code(metrics):
    CODE.mkdir(exist_ok=True)
    refs = set()
    for tex in (HERE / "chapters").glob("*.tex"):
        refs |= set(re.findall(r"\\ma(?:Nguon|Dai)\{([A-Za-z0-9_.]+)\}", tex.read_text(encoding="utf-8")))
    for ref in sorted(refs):
        module, name = ref.split("__", 1)
        (CODE / f"{ref}.py").write_text(extract(module, name), encoding="utf-8", newline="\n")
    for i, (q, v) in enumerate(metrics["kg"]["sparql"].items(), 1):
        (CODE / f"sparql_q{i}.rq").write_text(f"# Q{i} -- {q}\n" + v["query"].strip() + "\n", encoding="utf-8", newline="\n")
    return len(refs)


def invalidate_figures():
    """tikz external chỉ so md5 của mã hình; băm thêm các macro \\SL mà hình dùng để vẽ lại khi số liệu đổi."""
    cache = HERE / "tikzcache"
    cache.mkdir(exist_ok=True)
    so_lieu = (GEN / "so_lieu.tex").read_text(encoding="utf-8")
    stale = []
    for tex in sorted((HERE / "tikz").glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        h = hashlib.sha256(text.encode("utf-8"))
        h.update((HERE / "preamble" / "tikzstyle.tex").read_bytes())
        keys = re.findall(r"\\SL\{([^}]+)\}", text)
        if keys:
            want = re.compile(r"\\defSL\{(" + "|".join(re.escape(k) for k in keys) + r")(Raw|Pct)?\}")
            h.update("\n".join(l for l in so_lieu.splitlines() if want.match(l)).encode("utf-8"))
        stamp = cache / f"{tex.stem}.srchash"
        if stamp.exists() and stamp.read_text() == h.hexdigest() and (cache / f"{tex.stem}.pdf").exists():
            continue
        for ext in (".pdf", ".md5", ".dpth", ".log", ".dep"):
            (cache / f"{tex.stem}{ext}").unlink(missing_ok=True)
        stamp.write_text(h.hexdigest())
        stale.append(tex.stem)
    return stale


def counts(m):
    """Các số đếm suy ra từ JSON (số kịch bản, số câu hỏi theo miền...) để chương gọi \\SL{dem/...} thay vì gõ tay."""
    qa = json.loads((SRC / "data" / "qa" / "qa_eval.json").read_text(encoding="utf-8"))
    rows = m["agent"]["rows"]
    fc = [x for x in m["forecast"] if x != "config"]
    beat = [f"{s}/{x}" for s in fc for x in ("bilstm", "lstm", "ridge") if m["forecast"][s][x]["rmse_ratio"] < 1]
    a = m["advanced"]["agg"]
    return {"qa_health": sum(q["domain"] == "health" for q in qa), "qa_finance": sum(q["domain"] == "finance" for q in qa),
            "agent_scen": len(rows), "agent_health": sum(r["domain"] == "health" for r in rows),
            "agent_finance": sum(r["domain"] == "finance" for r in rows),
            "agent_llm": sum(r["used"] == "llm" for r in rows), "agent_nocite": sum(not r["n_citations"] for r in rows),
            "grag_q": len(m["graphrag"]["rows"]), "sparql": len(m["kg"]["sparql"]), "fc_targets": len(fc),
            "fc_models": 3 * len(fc), "fc_beat": len(beat), "violations": len(m["kg"]["violations"]),
            "fc_best_gain": max(1 - m["forecast"][s][x]["rmse_ratio"] for s in fc for x in ("bilstm", "lstm", "ridge")),
            "fc_worst_ratio": max(m["forecast"][s][x]["rmse_ratio"] for s in fc for x in ("bilstm", "lstm", "ridge")),
            "edges_agree": sum(e["agree"] for e in m["market"]["kg_edges_evidence"]),
            "edges_checked": len(m["market"]["kg_edges_evidence"]),
            "hyde_drop": a["dense"]["hit@1"] - a["hyde"]["hit@1"], "mq_drop": a["dense"]["hit@1"] - a["multiquery"]["hit@1"],
            "agent_cov_gain": m["agent"]["agg"]["all"]["coverage"] - m["agent"]["agg"]["all"]["coverage_llm_only"],
            "grag_gain": m["graphrag"]["agg"]["graph_local"]["ans_cov"] - m["graphrag"]["agg"]["naive"]["ans_cov"],
            "tools": len(tool_specs()), "comms": m["graphrag"]["n_comms"], **hba1c_cross(m["diabetes"]),
            **lag_peak(m["market"]["lag_corr"]), **fc_spread(m["forecast"])}


def fc_spread(f):
    """Độ phân tán lợi suất dự báo của BiLSTM so với thực tế trên tập kiểm thử (đơn vị %)."""
    out = {}
    for t in ("FPT", "GOLD_VND"):
        s = f[t]["test_series"]
        rt, rb = [100 * v for v in s["r_true"]], [100 * v for v in s["r_bilstm"]]
        out[f"fc_std_true_{t}"], out[f"fc_std_pred_{t}"] = statistics.pstdev(rt), statistics.pstdev(rb)
        out[f"fc_mean_pred_{t}"] = statistics.fmean(rb)
    return out


def lag_peak(lc):
    """Độ trễ (phiên) mà tương quan USD/VND -> cổ phiếu đạt cực đại dương, cho từng mã."""
    out = {}
    for s, v in lc["series"].items():
        i = max(range(len(v)), key=lambda j: v[j])
        out[f"lag_peak_{s}"], out[f"lag_corr_{s}"] = lc["lags"][i], v[i]
    return out


def hba1c_cross(d):
    """HbA1c nhỏ nhất mà xác suất MLP vượt ngưỡng quyết định, cho từng mức glucose của hình dia_hba1c_curve."""
    h, thr, out = d["hba1c_curve"], d["mlp"]["threshold"], {}
    for g, p in zip(h["glucose"], h["prob"]):
        hit = [x for x, v in zip(h["hba1c"], p) if v >= thr]
        out[f"dia_cross_{g}"] = f"{min(hit):.1f}".replace(".", ",") if hit else "không vượt"
    return out


def check_errors():
    """Một lượt lualatex nonstopmode, hình TikZ vẽ tại chỗ: in mọi lỗi LaTeX kèm dòng nguồn, không dừng ở lỗi đầu."""
    cmd = ["lualatex", "-shell-escape", "-interaction=nonstopmode", "-jobname=kiemtra", r"\def\kiemtranhanh{}\input{main}"]
    subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL)
    log = (HERE / "kiemtra.log").read_text(encoding="utf-8", errors="replace").splitlines()
    errs = [i for i, s in enumerate(log) if s.startswith("!")]
    for i in errs:
        print("\n".join(log[i:i + 4]), "\n")
    print(f"{len(errs)} lỗi LaTeX")
    sys.exit(1 if errs else 0)


def main():
    metrics ={p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS.glob("*.json"))}
    metrics["dem"] = counts(metrics)
    GEN.mkdir(exist_ok=True)
    (GEN / "so_lieu.tex").write_text(gen_macros(metrics), encoding="utf-8")
    n_tab = gen_tables(metrics)
    n_code = gen_code(metrics)
    print(f"macro từ {len(metrics)} JSON · {n_tab} bảng · {n_code} đoạn mã")
    stale = invalidate_figures()
    print(f"hình TikZ phải vẽ lại ({len(stale)}):", " ".join(stale) or "không")
    if "--no-pdf" in sys.argv:
        return
    if "--kiem-tra" in sys.argv:
        return check_errors()
    cmd = ["latexmk", "-g", "-lualatex", "-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"latexmk lỗi (mã {r.returncode}), xem main.log")
    log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
    missing = sorted(set(re.findall(r"Thieu so lieu '([^']+)'", log)))
    if missing:
        sys.exit("PDF có số liệu thiếu (in ra ??): " + ", ".join(missing))
    undefined = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
    if undefined:
        sys.exit("Còn tham chiếu chưa định nghĩa: " + ", ".join(undefined))
    shutil.copy2(HERE / "main.pdf", HERE / PDF_NAME)
    print("Đã ghi", PDF_NAME)


if __name__ == "__main__":
    main()
