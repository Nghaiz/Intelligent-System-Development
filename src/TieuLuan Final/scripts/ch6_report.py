"""Tổng hợp Chương 6 từ JSONL từng câu: thống kê, ch6_compare.json, ch6_datasets.json, runlog, bảng và
hình."""
import json
import re
from collections import Counter

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from tlf import plot_style as ps
from tlf.ch6_domain import DOMAINS, is_correct, load_items
from tlf.ch6_llm import MODEL
from tlf.ch6_rules import GUIDELINES
from tlf.common import ROOT, SEED, data_dir, save_json
from tlf.runlog import RUNS as RUNS_LOG, log_run
from tlf.texfmt import num, table, tex_escape

RUNS = data_dir("ch6") / "runs"
CJK = re.compile(r"[\u4e00-\u9fff]")
DATASETS = ["gsm8k", "pubmed", "clinical"]
ARCHS = ["direct", "single", "supervisor", "debate"]
LABEL = {"direct": "Một lượt gọi", "single": "Single ReAct", "supervisor": "Supervisor", "debate": "Debate"}
DS_NAME = {"gsm8k": "GSM8K", "pubmed": "PubMedQA", "clinical": "Ca lâm sàng"}


def load(ds, arch):
    path = RUNS / f"{ds}_{arch}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split(chr(10)) if line.strip()]


def merged(ds, arch):
    return {r["id"]: r for r in load(ds, arch)}


def bootstrap_diff(a, b, n=4000):
    """Khoảng tin cậy 95% cho hiệu độ đúng theo cặp (b - a), bootstrap trên câu hỏi."""
    rng = np.random.default_rng(SEED)
    d = np.asarray(b, float) - np.asarray(a, float)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def f1(rows):
    labels = sorted({r["gold"] for r in rows})
    y = [r["gold"] for r in rows]
    p = [r["pred"] if r["pred"] in labels else "none" for r in rows]
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def summarize(rows):
    """Các chỉ số của một kiến trúc trên một dataset; độ trễ lấy trung vị để bớt nhiễu của lần nạp mô hình."""
    n, k = len(rows), sum(r["correct"] for r in rows)
    lo, hi = proportion_confint(k, n, method="wilson")
    tok = [r["tokens_in"] + r["tokens_out"] for r in rows]
    return {"n": n, "correct": k, "accuracy": k / n, "ci_lo": lo, "ci_hi": hi,
            "tokens": float(np.mean(tok)), "tokens_out": float(np.mean([r["tokens_out"] for r in rows])),
            "calls": float(np.mean([r["llm_calls"] for r in rows])),
            "messages": float(np.mean([len(r["messages"]) for r in rows])),
            "rounds": float(np.mean([r["rounds"] for r in rows])),
            "no_answer": sum(r["pred"] is None for r in rows) / n,
            "latency_s": float(np.median([r["wall_s"] for r in rows]))}


def mcnemar(base, other, ids):
    """Kiểm định McNemar chính xác: chỉ các câu hai bên bất đồng mới mang thông tin."""
    a = [base[i]["correct"] for i in ids]
    b = [other[i]["correct"] for i in ids]
    wins = sum(y and not x for x, y in zip(a, b))
    losses = sum(x and not y for x, y in zip(a, b))
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    lo, hi = bootstrap_diff(a, b)
    return {"wins": wins, "losses": losses, "p": float(p), "diff": float(np.mean(b) - np.mean(a)),
            "diff_lo": lo, "diff_hi": hi}


def logged(ds, arch, n):
    """Chạy lại tổng hợp không ghi trùng vào nhật ký chung."""
    if not RUNS_LOG.exists():
        return False
    rows = [json.loads(x) for x in RUNS_LOG.read_text(encoding="utf-8").split(chr(10))
            if '"chapter": "ch6"' in x]
    return any(r["dataset"] == ds and r["model"] == arch and r["params"].get("n") == n for r in rows)


def compare():
    rows, summary, pairs = [], {}, []
    for ds in DATASETS:
        data = {a: merged(ds, a) for a in ARCHS}
        ids = sorted(set.intersection(*[set(v) for v in data.values()]), key=str)
        summary[ds] = {}
        for a in ARCHS:
            sel = [data[a][i] for i in ids]
            s = summarize(sel)
            s["f1_macro"] = f1(sel) if ds != "gsm8k" else None   # đáp án số: không có khái niệm lớp
            s["seconds"] = sum(r["wall_s"] for r in sel)
            base = [data["single"][i]["tokens_in"] + data["single"][i]["tokens_out"] for i in ids]
            s["tokens_x"] = s["tokens"] / float(np.mean(base))
            summary[ds][a] = s
            rows.append({"dataset": ds, "model": a, "framework": f"tự cài trên {MODEL}", **s})
            s["tokens_x_s"] = f"{s['tokens_x']:.1f}".replace(".", ",")
            if logged(ds, a, s["n"]):
                continue
            log_run("ch6", ds, a, "ollama", {k: s[k] for k in ("accuracy", "f1_macro", "tokens", "calls")},
                    s["seconds"], "cuda", {"n": s["n"], "llm": MODEL})
        for a, b in [("single", "supervisor"), ("single", "debate"), ("supervisor", "debate")]:
            pairs.append({"dataset": ds, "a": a, "b": b, **mcnemar(data[a], data[b], ids)})
    adjusted = multipletests([p["p"] for p in pairs], method="holm")[1]
    for p, q in zip(pairs, adjusted):
        p["p_holm"] = float(q)
    return rows, summary, pairs


def fail_rate(rows):
    """Tỉ lệ lượt kiểm chứng trả FAIL trên mọi lượt kiểm chứng."""
    v = [m["content"].strip().upper().startswith("PASS") for r in rows for m in r["messages"]
         if m["from"] == "Verifier"]
    return 1 - float(np.mean(v)) if v else 0.0


def foreign_rate(rows):
    """Tỉ lệ lượt LLM chứa chữ Hán: mô hình trôi sang tiếng Trung giữa chừng."""
    v = [bool(CJK.search(m["content"])) for r in rows for m in r["messages"] if m["tokens"]]
    return float(np.mean(v)) if v else 0.0


def revision_effect(ds, rows):
    """Lượt làm lại sau FAIL giúp hay hại: so nhãn bản nháp đầu trình Verifier với nhãn cuối cùng."""
    fixed = broken = 0
    for r in rows:
        if r["rounds"] < 2:
            continue
        msgs = [m for m in r["messages"] if m["tokens"]]
        first = next(i for i, m in enumerate(msgs) if m["from"] == "Verifier")
        before = is_correct(ds, DOMAINS[ds].parse(msgs[first - 1]["content"]), r["gold"])
        fixed, broken = fixed + (r["correct"] and not before), broken + (before and not r["correct"])
    return {"rev_fixed": int(fixed), "rev_broken": int(broken)}


def before_fix():
    """Lượt trước khi sửa cài đặt, giữ ở data/ch6/runs_n30, chỉ để nêu tác dụng của bản sửa."""
    path = data_dir("ch6") / "runs_n30"
    rows = {a: [json.loads(x) for x in (path / f"clinical_{a}.jsonl").read_text(encoding="utf-8").split(chr(10))
                if x.strip()] for a in ("supervisor", "debate")}
    parse = DOMAINS["clinical"].parse
    return {"foreign_rate": foreign_rate(rows["supervisor"] + rows["debate"]),
            "supervisor_no_answer": float(np.mean([parse(r["text"]) is None for r in rows["supervisor"]]))}


def diagnostics():
    """Cơ chế phản biện thực sự làm gì: tỉ lệ bất đồng, số lần kiểm chứng bác bỏ, đáp án trước và sau khi
    xử."""
    out = {}
    for ds in DATASETS:
        sup, deb = list(merged(ds, "supervisor").values()), list(merged(ds, "debate").values())
        disp = [r for r in deb if r["consensus"] is False]
        revised = [r for r in sup if r["rounds"] > 1]
        tools = sum(any(m["from"].startswith("Công cụ") for m in r["messages"])
                    for r in merged(ds, "single").values())
        out[ds] = {"disputed": len(disp), "disputed_rate": len(disp) / len(deb),
                   "disputed_correct": sum(r["correct"] for r in disp),
                   "consensus_acc": float(np.mean([r["correct"] for r in deb if r["consensus"]] or [0])),
                   "revised": len(revised), "revised_correct": sum(r["correct"] for r in revised),
                   "single_tool_rate": tools / max(1, len(merged(ds, "single"))),
                   "both_unreadable": sum(r["consensus"] is True and r["pred"] is None for r in deb),
                   **revision_effect(ds, sup), "verifier_fail_rate": fail_rate(sup), "foreign_rate": foreign_rate(sup + deb)}
    return out


def dist(ds, n):
    items = load_items(ds, n)
    if ds == "gsm8k":
        return items, [len(i["solution"].split("\n")) - 1 for i in items]
    return items, [i["gold"] for i in items]


def describe(ds, n):
    items, values = dist(ds, n)
    size = sum(p.stat().st_size for p in [data_dir("ch6") / f"{ds}.jsonl"]) / 2**20
    total = sum(1 for _ in open(data_dir("ch6") / f"{ds}.jsonl", encoding="utf-8"))
    if ds == "gsm8k":
        c = Counter(values)
        keys = sorted(c)
        labels, vals = [f"{k} bước" for k in keys], [c[k] for k in keys]
    else:
        c = Counter(values)
        labels, vals = list(c), list(c.values())
    meta = {
        "gsm8k": ("GSM8K (test)", "Hugging Face openai/gsm8k", "https://huggingface.co/datasets/openai/gsm8k",
                  "toán lời văn, đáp án số", "MIT", "Dùng tập test, lấy mẫu ngẫu nhiên có hạt giống."),
        "pubmed": ("PubMedQA pqa_labeled", "Hugging Face qiaojin/PubMedQA",
                   "https://huggingface.co/datasets/qiaojin/PubMedQA", "yes/no/maybe từ tóm tắt", "MIT",
                   "Dùng tập gán nhãn thủ công, lấy mẫu ngẫu nhiên có hạt giống."),
        "clinical": ("Ca lâm sàng tổng hợp", "Tự sinh bằng tlf/ch6_rules.py", "", "đơn thuốc đúng/sai theo luật",
                     "tự sinh", "Dữ liệu tổng hợp, chỉ để đo tác tử, không phải tư vấn y tế."),
    }[ds]
    return {"id": ds, "name": meta[0], "source": meta[1], "url": meta[2], "task": meta[3], "n_samples": total,
            "n_eval": len(items), "n_features": 1, "size_mb": round(size, 2), "license": meta[4], "note": meta[5],
            "sample_rows": [{"question": i["question"][:220], "gold": i["gold"]} for i in items[:3]],
            "distribution": {"labels": labels, "values": vals}}


def fig_dist(sets):
    fig, axes = ps.plt.subplots(1, 3, figsize=(ps.W, 2.6))
    for ax, d in zip(axes, sets):
        ax.bar(d["distribution"]["labels"], d["distribution"]["values"], color=ps.SEQ[:len(d["distribution"]["labels"])])
        ax.set_title(DS_NAME[d["id"]], fontsize=12)
        ax.tick_params(axis="x", labelsize=9, rotation=45 if d["id"] == "gsm8k" else 0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("số câu")
    fig.tight_layout()
    ps.save(fig, "ch6_dist")


def fig_acc(summary):
    fig, axes = ps.plt.subplots(1, 3, figsize=(ps.W, 3.0), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        s = summary[ds]
        acc = [s[a]["accuracy"] for a in ARCHS]
        err = [[s[a]["accuracy"] - s[a]["ci_lo"] for a in ARCHS], [s[a]["ci_hi"] - s[a]["accuracy"] for a in ARCHS]]
        ax.bar(range(4), acc, yerr=err, capsize=4, color=ps.SEQ[:4], ecolor="#333333")
        ax.set_xticks(range(4), ["1 gọi", "ReAct", "Sup.", "Deb."], fontsize=10, rotation=40)
        ax.set_title(DS_NAME[ds], fontsize=12)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("độ đúng")
    fig.tight_layout()
    ps.save(fig, "ch6_acc")


def fig_cost(summary):
    fig, ax = ps.plt.subplots(figsize=(ps.W, 3.3))
    marks = {"gsm8k": "o", "pubmed": "s", "clinical": "^"}
    for ds in DATASETS:
        for i, a in enumerate(ARCHS):
            s = summary[ds][a]
            ax.scatter(s["tokens"], s["accuracy"], marker=marks[ds], s=95, color=ps.SEQ[i], edgecolor="black")
    ax.set_xscale("log")
    ax.set_xlabel("token trung bình mỗi câu (thang log)")
    ax.set_ylabel("độ đúng")
    h = [ps.plt.Line2D([], [], marker="o", ls="", color=ps.SEQ[i], label=LABEL[a]) for i, a in enumerate(ARCHS)]
    h += [ps.plt.Line2D([], [], marker=m, ls="", color="#555555", label=DS_NAME[d]) for d, m in marks.items()]
    ax.legend(handles=h, ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    ps.save(fig, "ch6_cost")


def f1_cell(row):
    return "--" if row["f1_macro"] is None else num(row["f1_macro"], 3)


def prior_gsm8k():
    """Số gốc của Tiểu luận 03 (GSM8K, Qwen2.5-1.5B), chỉ để đối chiếu, không gộp vào so sánh chính."""
    d = json.loads((ROOT.parent / "TieuLuan 03/outputs/metrics/ch7_gsm8k.json").read_text(encoding="utf-8"))
    return {"model": "Qwen2.5-1.5B", "n": d["full"]["multi"]["n"], "multi_acc": d["full"]["multi"]["acc"],
            "cot_acc": d["full"]["cot"]["acc"], "cot_n": d["full"]["cot"]["n"]}


def tables(rows, pairs, sets, majority):
    body, last = [], None
    for r in rows:
        if last and r["dataset"] != last:
            body.append("\\midrule")
        last = r["dataset"]
        body.append([DS_NAME[r["dataset"]], LABEL[r["model"]], num(100 * r["accuracy"], 1),
                     f"{num(100 * r['ci_lo'], 1)}--{num(100 * r['ci_hi'], 1)}", f1_cell(r),
                     num(r["tokens"], 0), num(r["calls"], 1), num(r["rounds"], 2), num(r["latency_s"], 1)])
    table("ch6_compare", "llrrrrrrr", ["Dataset", "Kiến trúc", "Đúng \\%", "KTC 95\\%", "F1", "Token", "Gọi LLM",
                                      "Vòng", "Trễ (s)"], body,
          "Lớp đa số: " + ", ".join(f"{DS_NAME[d]} {num(100 * v, 1)}\\%" for d, v in majority.items())
          + ". Độ trễ là trung vị mỗi câu, các lượt gọi chạy tuần tự.")
    table("ch6_pairs", "lllrrrrr", ["Dataset", "So sánh", "", "Thắng", "Thua", "Hiệu (\\%)", "$p$", "$p$ Holm"],
          [[DS_NAME[p["dataset"]], LABEL[p["a"]], "so với " + LABEL[p["b"]], p["wins"], p["losses"],
            f"{num(100 * p['diff'], 1, True)} [{num(100 * p['diff_lo'], 1)}; {num(100 * p['diff_hi'], 1)}]",
            num(p["p"], 3), num(p["p_holm"], 3)] for p in pairs])
    table("ch6_datasets", "lrrL{3.1cm}L{4.6cm}", ["Dataset", "Tổng mẫu", "Dùng đo", "Nhiệm vụ", "Nguồn"],
          [[tex_escape(d["name"]), d["n_samples"], d["n_eval"], tex_escape(d["task"]), tex_escape(d["source"])] for d in sets])
    table("ch6_samples", "lL{10.9cm}l", ["Dataset", "Mẫu đầu ra thực", "Nhãn"],
          [[DS_NAME[d["id"]], tex_escape(d["sample_rows"][0]["question"][:190]) + "...",
            tex_escape(d["sample_rows"][0]["gold"])] for d in sets])


def main():
    rows, summary, pairs = compare()
    n = summary["gsm8k"]["single"]["n"]
    sets = [describe(ds, summary[ds]["single"]["n"]) for ds in DATASETS]
    save_json("ch6_datasets", {"datasets": sets})
    majority = {}
    for d in sets:
        if d["id"] == "gsm8k":
            majority[d["id"]] = 0.0
        else:
            majority[d["id"]] = max(d["distribution"]["values"]) / sum(d["distribution"]["values"])
    save_json("ch6_compare", {"rows": rows, "primary": "accuracy", "summary": summary, "n_questions": n, "majority": majority, "diagnostics": diagnostics(),
                              "n_rules": len(GUIDELINES), "model": MODEL,
                              "prior_gsm8k": prior_gsm8k(), "best_pair": min(pairs, key=lambda p: p["p"]),
                              "before_fix": before_fix()})
    save_json("ch6_pairs", {"rows": pairs})
    fig_dist(sets)
    fig_acc(summary)
    fig_cost(summary)
    tables(rows, pairs, sets, majority)


if __name__ == "__main__":
    main()
