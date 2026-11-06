"""Sinh bảng LaTeX của báo cáo từ outputs/metrics: chạy bằng python scripts/report_tables.py."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
METRICS = HERE.parents[1] / "src" / "Project 01" / "outputs" / "metrics"
GEN = HERE / "generated"
GEN.mkdir(exist_ok=True)
MODELS = (("logreg", "Hồi quy logistic"), ("random_forest", "Rừng ngẫu nhiên"), ("xgboost", "XGBoost"),
          ("mlp", "MLP"), ("resnet", "ResNet bảng"), ("ft_transformer", "FT-Transformer"))
TASKS = (("full", "Có xét nghiệm"), ("screening", "Sàng lọc"), ("readmission", "Tái nhập viện"))


def load(name):
    return json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8"))


def n(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def models_table():
    s = load("ml_summary")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{t}}}}}" for _, t in TASKS)
    sub = " & ".join(["PR-AUC & F1"] * len(TASKS))
    lines = [r"\begin{tabular}{@{}lcccccc@{}}\toprule", rf"\textbf{{Mô hình}} & {head}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}", rf" & {sub}\\\midrule"]
    best = {t: (max(s[t]["models"].values(), key=lambda m: m["test_pr_auc"])["test_pr_auc"],
                max(s[t]["models"].values(), key=lambda m: m["test_f1"])["test_f1"]) for t, _ in TASKS}
    for key, name in MODELS:
        row = [name]
        for t, _ in TASKS:
            m = s[t]["models"][key]
            for v, b in zip((m["test_pr_auc"], m["test_f1"]), best[t]):
                row.append(rf"\textbf{{{n(v)}}}" if v == b else n(v))
        lines.append(" & ".join(row) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (GEN / "ch4_models.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rules_table():
    rows = json.loads((METRICS.parents[1] / "dnx" / "rules.json").read_text(encoding="utf-8"))["rules"]
    names = (("urgent", "Khẩn cấp", "Đường huyết rất cao, rất thấp, triệu chứng cấp"),
             ("medication", "An toàn thuốc", "Metformin với eGFR thấp, rượu, thuốc cản quang"),
             ("diagnostic", "Ngưỡng chẩn đoán", "HbA1c, đường huyết lúc đói"),
             ("risk", "Yếu tố nguy cơ", "BMI, huyết áp, tuổi"),
             ("policy", "Chính sách", "Không nêu liều, không kê đơn"))
    lines = [r"\begin{table}[!htbp]\centering\caption{Nhóm luật của RuleGuard}\label{tab:luat}",
             r"\small\begin{tabular}{@{}lcl@{}}\toprule", r"\textbf{Nhóm} & \textbf{Số luật} & \textbf{Ví dụ}\\\midrule"]
    for g, name, ex in names:
        lines.append(rf"{name} & {sum(r['group'] == g for r in rows)} & {ex}\\")
    lines.append(rf"\midrule\textbf{{Tổng}} & \textbf{{{len(rows)}}} & \\\bottomrule\end{{tabular}}\end{{table}}")
    (GEN / "ch6_rules.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def _at_cost(curve, cost):
    """AUC nội suy tuyến tính của một chính sách thứ tự cố định tại đúng mức chi phí của DQN."""
    pts = [(p["cost"], p["auc"]) for p in curve]
    for (c0, a0), (c1, a1) in zip(pts, pts[1:]):
        if c0 <= cost <= c1:
            return a0 + (a1 - a0) * (cost - c0) / max(c1 - c0, 1e-9)
    return pts[-1][1]


def rl_table():
    d = load("rl_compare")
    fixed = d["policies"]["fixed_importance"]["curve"]
    greedy = d["policies"]["greedy_info_per_cost"]["curve"]
    lines = [r"\begin{tabular}{@{}cccccccc@{}}\toprule",
             r"\textbf{$\lambda$} & \textbf{Seed} & \textbf{Số câu} & \textbf{Chi phí} & \textbf{AUC DQN} & \textbf{F1} & "
             r"\textbf{Cố định} & \textbf{Thông tin/phí}\\\midrule"]
    out = {}
    for lam, rows in d["dqn_adaptive"].items():
        cost, auc = _mean(rows, "cost"), _mean(rows, "auc")
        f, g = _at_cost(fixed, cost), _at_cost(greedy, cost)
        out[lam] = {"questions": _mean(rows, "questions"), "auc": auc, "cost": cost}
        lines.append(rf"{lam.replace('.', ',')} & {len(rows)} & {n(out[lam]['questions'], 1)} & {n(cost, 2)} & "
                     rf"{n(auc, 4)} & {n(_mean(rows, 'f1'))} & {n(f, 4)} & {n(g, 4)}\\")
    lines.append(r"\bottomrule\end{tabular}")
    (GEN / "ch6_rl.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def rag_table():
    d = load("rag_eval")
    names = (("bm25", "BM25"), ("dense", "Vector e5"), ("graph", "Đồ thị"), ("hybrid", "Lai (RRF)"), ("hybrid_rerank", "Lai + rerank"))
    cols = ("hit@1", "hit@5", "mrr", "ndcg")
    rows = {m: d["modes"][m]["all"] for m, _ in names}
    best = {c: max(r[c] for r in rows.values()) for c in cols}
    lines = [r"\begin{tabular}{@{}lccccc@{}}\toprule",
             r"\textbf{Chế độ} & \textbf{Hit@1} & \textbf{Hit@5} & \textbf{MRR} & \textbf{nDCG@10} & \textbf{Trung vị (ms)}\\\midrule"]
    for m, name in names:
        cells = [rf"\textbf{{{n(rows[m][c])}}}" if rows[m][c] == best[c] else n(rows[m][c]) for c in cols]
        lines.append(" & ".join([name, *cells, f"{d['modes'][m]['latency_ms']['p50']:.0f}"]) + r"\\")
    lines.append(r"\midrule")
    for k, name in (("no_bm25", "Bỏ BM25"), ("no_dense", "Bỏ vector"), ("no_graph", "Bỏ đồ thị")):
        if k in d.get("ablation", {}):
            lines.append(" & ".join([name, *[n(d["ablation"][k][c]) for c in cols], "--"]) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (GEN / "ch5_rag.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ablation_table():
    d = load("agent_ablation")["configs"]
    names = (("full", "Đầy đủ"), ("no_critic", "Bỏ Critic"), ("vector_only", "Chỉ vector"), ("no_rules", "Bỏ luật"),
             ("single_agent", "Một tác tử"))
    lines = [r"\begin{tabular}{@{}lccccc@{}}\toprule",
             r"\textbf{Cấu hình} & \textbf{Định tuyến} & \textbf{Số đúng} & \textbf{Trích dẫn} & \textbf{An toàn} & \textbf{Giây}\\\midrule"]
    for key, name in names:
        c = d[key]
        route = "--" if key == "single_agent" else n(c["route_ok"])
        lines.append(rf"{name} & {route} & {n(c['numbers_ok'])} & {n(c['citations_ok'])} & {n(c['safe'])} & {n(c['seconds_mean'], 1)}\\")
    lines.append(r"\bottomrule\end{tabular}")
    (GEN / "ch6_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ablation_table()
    models_table()
    rules_table()
    print(rl_table())
    rag_table()
    print("xong")
