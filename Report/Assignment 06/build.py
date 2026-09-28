"""Dựng báo cáo Assignment 06 bằng một lệnh:  python build.py [--no-pdf]

1. Chép outputs/figdata/*.dat  -> figdata/        (dữ liệu cho pgfplots)
2. Trích mã nguồn theo tên     -> code/           (mỗi \\maNguon{mod__ten} và \\maNguonJS{ten} trong chapters/*.tex)
3. Sinh macro số liệu và bảng  -> generated/      (từ outputs/metrics/*.json)
4. latexmk -lualatex -shell-escape main.tex, rồi chép PDF thành A06_CT_nghiand.600.pdf

Không có con số nào gõ tay trong chương: chương chỉ gọi \\SL{tệp/khoá/...} hoặc \\input một bảng sinh ra ở đây.
"""
import ast
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src" / "Assignment 06"
METRICS = SRC / "outputs" / "metrics"
GEN = HERE / "generated"
PDF_NAME = "A06_CT_nghiand.600.pdf"
# \maNguon{module__ten}: module nằm ở a06/ trừ hai tệp của Web App.
MODULE_FILES = {"app": SRC / "app" / "app.py", "smoke_test": SRC / "app" / "smoke_test.py"}
JS_FILE = SRC / "app" / "static" / "app.js"

CELLS = ["rnn", "lstm", "gru", "bilstm"]
LABELS = {"rnn": "Simple RNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}
FW = {"torch": "PyTorch", "keras": "Keras"}
# Lá JSON lưu dạng tỉ lệ 0..1 nhưng báo cáo đọc theo phần trăm.
PERCENT_KEYS = {"acc", "precision", "recall", "f1", "dir_acc", "churn_rate_labeled", "churn_rate_with_log",
                "churn_rate_sample", "pos_rate_test", "tail4", "tail4_gauss", "mask_p", "mask_keep_std",
                "mask_keep_rec", "prob", "prob_best", "loyal_act", "churn_act"}


# ---------------------------------------------------------------- định dạng số kiểu Việt
def num(v, decimals=2, sign=False):
    s = f"{v:+,.{decimals}f}" if sign else f"{v:,.{decimals}f}"
    s = s.replace(",", "\0").replace(".", "{,}").replace("\0", "{.}")
    # \ensuremath: macro số đặt được cả trong văn bản lẫn trong $...$ mà không lồng dấu $
    s = s.replace("-", "\\ensuremath{-}")
    return s.replace("+", "\\ensuremath{+}") if sign else s


def sci(v):
    if v == 0:  # p-value nhỏ hơn số thực nhỏ nhất của float64 nên SciPy trả về đúng 0
        return "\\ensuremath{< 10^{-300}}"
    m, e = f"{v:.2e}".split("e")
    return f"\\ensuremath{{{m.replace('.', '{,}')} \\times 10^{{{int(e)}}}}}"


def fmt(v, key=""):
    if isinstance(v, bool):
        return "đúng" if v else "sai"
    if isinstance(v, int):
        return num(v, 0)
    if isinstance(v, float):
        if key in PERCENT_KEYS:
            return num(100 * v)
        if math.isnan(v):
            return "NaN"
        if v != 0 and abs(v) < 1e-3:
            return sci(v)
        s = f"{v:.{4 if abs(v) < 1 else 2 if abs(v) < 1000 else 0}f}"
        if "." in s:  # 0.1000 -> 0.1, 2.00 -> 2: cấu hình và tỉ số không cần số 0 thừa
            s = s.rstrip("0").rstrip(".")
        return num(float(s), len(s.split(".")[1]) if "." in s else 0)
    return tex_escape(str(v))


def tex_escape(s):
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
             .replace("&", "\\&").replace("#", "\\#").replace("$", "\\$"))


def flatten(obj, prefix):
    """{'a': {'b': 1}} -> {'prefix/a/b': 1}; danh sách đánh chỉ số 0, 1, 2..."""
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    else:
        yield prefix, obj
        return
    for k, v in items:
        yield from flatten(v, f"{prefix}/{k}")


def model_label(v):
    m = re.fullmatch(r"(?:amzn|kkbox)_(torch|keras)_(\w+)", v)
    if m and m.group(2) in LABELS:
        return f"{LABELS[m.group(2)]} ({FW[m.group(1)]})"
    return LABELS.get(v)


def gen_macros(metrics):
    lines = ["% Sinh tự động bởi build.py từ src/Assignment 06/outputs/metrics/*.json. Không sửa tay."]
    for name, data in metrics.items():
        for key, v in flatten(data, name):
            if "/sequence/" in key:            # chuỗi 31×8 của người dùng mẫu: dữ liệu hình, không phải số liệu chữ
                continue
            leaf = key.rsplit("/", 1)[-1]
            is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
            text = (sci(v) if v < 1e-3 else num(v, 4)) if leaf == "p" and is_num else fmt(v, leaf)
            lines.append(f"\\defSL{{{key}}}{{{text}}}")
            label = model_label(v) if isinstance(v, str) else None
            if label:  # tên mô hình đọc được thay cho khoá thô: "gru" -> GRU, "amzn_torch_bilstm" -> BiLSTM (PyTorch)
                lines.append(f"\\defSL{{{key}Label}}{{{label}}}")
            if is_num:
                lines.append(f"\\defSL{{{key}Raw}}{{{v:.6g}}}")     # dạng số thô cho toạ độ pgfplots
                if leaf in ("roc_auc", "pr_auc", "prob", "threshold", "r2_ret", "diff", "prob_best"):
                    lines.append(f"\\defSL{{{key}Pct}}{{{num(100 * v, 2, sign=leaf in ('r2_ret', 'diff'))}}}")
                if leaf in ("stat", "t", "diff", "change_pct", "trend"):
                    lines.append(f"\\defSL{{{key}Sgn}}{{{num(v, 2 if abs(v) >= 1 else 4, sign=True)}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- bảng
def table(spec, header, rows, note=None):
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [r if isinstance(r, str) else " & ".join(r) + " \\\\" for r in rows]
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    if note:
        out.append(f"\\par\\smallskip{{\\footnotesize {note}}}")
    return "\n".join(out) + "\n"


def bold_best(values, formatted, higher=True):
    best = max(values) if higher else min(values)
    return [f"\\textbf{{{f}}}" if v == best else f for v, f in zip(values, formatted)]


def tab_amzn_desc(m):
    d = m["amzn"]["desc"]
    names = {"Close": "Close (USD)", "Open": "Open (USD)", "High": "High (USD)", "Low": "Low (USD)",
             "Volume": "Volume (triệu cp)", "Return": "Return (\\%)", "MA20": "MA20 (USD)", "MA50": "MA50 (USD)",
             "Volatility20": "Volatility20 (năm)", "RSI14": "RSI14"}
    scale = {"Volume": 1e-6, "Return": 100}
    keys = ("mean", "std", "min", "q25", "median", "q75", "max")
    rows = [[n, *(num(d[c][k] * scale.get(c, 1), 2) for k in keys), num(d[c]["kurt"], 2)] for c, n in names.items()]
    return table("l" + "r" * 8, ["Đặc trưng", "TB", "ĐLC", "Min", "Q25", "Trung vị", "Q75", "Max", "Kurtosis"], rows,
                 "TB: trung bình; ĐLC: độ lệch chuẩn; kurtosis là độ nhọn dư (phân phối chuẩn bằng 0).")


def tab_ttest(m):
    t = m["kkbox"]["ttest"]
    rows = []
    for f, r in t.items():
        p = sci(r["p"]) if r["p"] < 1e-3 else num(r["p"], 4)
        rows.append([f"\\texttt{{{tex_escape(f)}}}", num(r["loyal"], 3), num(r["churn"], 3),
                     num(r["diff_pct"], 1, sign=True), num(r["t"], 2, sign=True), p])
    return table("lrrrrr", ["Đặc trưng (TB 31 ngày)", "Loyal", "Churn", "Chênh (\\%)", "$t$ Welch", "$p$"], rows,
                 "Đặc trưng đếm và số giây đã lấy $\\log(1+x)$; \\texttt{active} là tỉ lệ ngày có nghe.")


def tab_torch_amzn(m):
    mm, nv = m["torch_amzn"]["models"], m["torch_amzn"]["naive"]
    rmse_f = bold_best([mm[c]["rmse"] for c in CELLS] + [nv["rmse"]],
                       [num(mm[c]["rmse"], 3) for c in CELLS] + [num(nv["rmse"], 3)], higher=False)
    rows = [[LABELS[c], rmse_f[i], num(mm[c]["mae"], 3), num(mm[c]["r2"], 4), num(100 * mm[c]["r2_ret"], 2, sign=True),
             num(100 * mm[c]["dir_acc"]), num(mm[c]["params"], 0), f"{mm[c]['best_epoch']}/{mm[c]['epochs_run']}",
             num(mm[c]["epoch_s"], 2)] for i, c in enumerate(CELLS)]
    rows.append("\\midrule")
    rows.append(["Mốc ngây thơ $\\hat P_{t+1}=P_t$", rmse_f[-1], num(nv["mae"], 3), num(nv["r2"], 4),
                 num(100 * nv["r2_ret"], 2, sign=True), "--", "0", "--", "--"])
    return table("lrrrrrrrr", ["Mô hình", "RMSE", "MAE", "$R^2$ giá", "$R^2$ return (\\%)", "Đúng chiều (\\%)",
                               "Tham số", "Epoch tốt/chạy", "s/epoch"], rows,
                 "RMSE, MAE tính bằng USD trên tập test; chữ đậm: nhỏ nhất.")


def tab_torch_kkbox(m):
    mm = m["torch_kkbox"]["models"]
    keys = ("acc", "precision", "recall", "f1", "roc_auc", "pr_auc")
    f = {k: bold_best([mm[c][k] for c in CELLS], [num(100 * mm[c][k]) if k in PERCENT_KEYS else num(mm[c][k], 4)
                                                  for c in CELLS]) for k in keys}
    rows = [[LABELS[c]] + [f[k][i] for k in keys] + [num(mm[c]["threshold"], 3),
                                                   f"{mm[c]['best_epoch']}/{mm[c]['epochs_run']}", num(mm[c]["epoch_s"], 2)]
            for i, c in enumerate(CELLS)]
    return table("lrrrrrrrrr", ["Mô hình", "Acc", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Ngưỡng",
                                "Epoch tốt/chạy", "s/epoch"], rows,
                 "Acc, Precision, Recall, F1 theo \\%, tại ngưỡng chọn trên validation; chữ đậm: tốt nhất.")


def tab_bench(m, ds):
    b = m["bench"]["models"]
    keys = [f"{ds}_{fw}_{c}" for fw in FW for c in CELLS]
    if ds == "amzn":
        mv = [b[k]["rmse"] for k in keys]
        main = bold_best(mv, [num(v, 3) for v in mv], higher=False)
        head = ["Framework", "Mô hình", "RMSE", "MAE", "$R^2$ ret. (\\%)"]
        extra = lambda k: [num(b[k]["mae"], 3), num(100 * b[k]["r2_ret"], 2, sign=True)]  # noqa: E731
    else:
        mv = [b[k]["f1"] for k in keys]
        main = bold_best(mv, [num(100 * v) for v in mv])
        head = ["Framework", "Mô hình", "F1 (\\%)", "Recall (\\%)", "ROC-AUC"]
        extra = lambda k: [num(100 * b[k]["recall"]), num(b[k]["roc_auc"], 4)]  # noqa: E731
    lat = bold_best([b[k]["latency_ms"] for k in keys], [num(b[k]["latency_ms"], 3) for k in keys], higher=False)
    rows = []
    for i, k in enumerate(keys):
        fw, c = k.split("_")[1:]
        if i == 4:
            rows.append("\\midrule")
        dev = "GPU" if b[k]["device"].startswith("cuda") else "CPU"
        rows.append([FW[fw] if i % 4 == 0 else "", LABELS[c], main[i], *extra(k), num(b[k]["params"], 0),
                     num(b[k]["size_kb"], 1), f"{num(b[k]['epoch_s'], 2)} ({dev})", lat[i],
                     num(b[k]["throughput"], 0)])
    if ds == "amzn":
        nv = m["bench"]["naive"]
        rows += ["\\midrule", ["", "Mốc ngây thơ", num(nv["rmse"], 3), num(nv["mae"], 3),
                               num(100 * nv["r2_ret"], 2, sign=True), "0", "--", "--", "--", "--"]]
    head += ["Tham số", "KB", "s/epoch", "ms/mẫu", "mẫu/s"]
    return table("ll" + "r" * (len(head) - 2), head, rows,
                 "Độ trễ (lô 1, trung vị 200 lần) và thông lượng (lô 256) đo trên CPU, 1 luồng, cả hai framework. "
                 "s/epoch đo lúc huấn luyện, trên thiết bị ghi trong ngoặc.")


def gen_tables(m):
    GEN.mkdir(exist_ok=True)
    builders = {"tab_amzn_desc": lambda: tab_amzn_desc(m), "tab_ttest": lambda: tab_ttest(m),
                "tab_torch_amzn": lambda: tab_torch_amzn(m), "tab_torch_kkbox": lambda: tab_torch_kkbox(m),
                "tab_bench_amzn": lambda: tab_bench(m, "amzn"), "tab_bench_kkbox": lambda: tab_bench(m, "kkbox")}
    missing = []
    for name, build in builders.items():
        try:
            body = build()
        except KeyError as e:  # notebook tương ứng chưa chạy: báo rõ, LaTeX sẽ dừng nếu chương cần bảng này
            missing.append(f"{name} (thiếu {e})")
            continue
        (GEN / f"{name}.tex").write_text("% Sinh bởi build.py\n" + body, encoding="utf-8")
    for x in missing:
        print("CẢNH BÁO: chưa sinh được bảng", x)
    return len(builders) - len(missing)


# ---------------------------------------------------------------- trích mã nguồn
def extract(module, dotted):
    """Cắt đúng thân hàm/lớp (kể cả decorator) theo tên, hỗ trợ Lớp.phương_thức."""
    path = MODULE_FILES.get(module, SRC / "a06" / f"{module}.py")
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    nodes = ast.parse(src).body
    target = None
    for part in dotted.split("."):
        target = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                       and n.name == part), None)
        if target is None:
            raise KeyError(f"Không thấy '{dotted}' trong {path.name}")
        nodes = target.body
    start = min([target.lineno] + [d.lineno for d in target.decorator_list]) - 1
    block = lines[start:target.end_lineno]
    indent = len(block[0]) - len(block[0].lstrip())
    rel = path.relative_to(SRC).as_posix()
    return f"# {rel}\n" + "\n".join(l[indent:] for l in block) + "\n"


def extract_js(name):
    """Cắt một hàm JS theo tên bằng cách đếm ngoặc nhọn (app.js không có ngoặc trong chuỗi ký tự)."""
    src = JS_FILE.read_text(encoding="utf-8")
    m = re.search(rf"^(async\s+)?function\s+{name}\s*\(", src, re.M)
    if not m:
        raise KeyError(f"Không thấy hàm '{name}' trong app.js")
    i = src.index("{", m.end())
    depth = 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return "// app/static/app.js\n" + src[m.start():j + 1] + "\n"
    raise ValueError(f"Hàm '{name}' thiếu ngoặc đóng")


def gen_code():
    out = HERE / "code"
    out.mkdir(exist_ok=True)
    refs, js = set(), set()
    for tex in (HERE / "chapters").glob("*.tex"):
        text = tex.read_text(encoding="utf-8")
        refs |= set(re.findall(r"\\maNguon\{([A-Za-z0-9_.]+)\}", text))
        js |= set(re.findall(r"\\maNguonJS\{([A-Za-z0-9_]+)\}", text))
    for ref in sorted(refs):
        module, name = ref.split("__", 1)
        (out / f"{ref}.py").write_text(extract(module, name), encoding="utf-8")
    for name in sorted(js):
        (out / f"js__{name}.js").write_text(extract_js(name), encoding="utf-8")
    return len(refs) + len(js)


# ---------------------------------------------------------------- làm mới cache hình
def invalidate_figures():
    """tikz external chỉ so md5 của mã hình, không biết tệp .dat đã đổi. Ở đây băm mã hình
    (kể cả tệp nó \\input), các tệp figdata nó đọc và các macro \\SL nó dùng; hình nào có
    hash mới thì xoá bản cache của đúng hình đó để external vẽ lại. Trả về danh sách hình phải vẽ lại."""
    cache = HERE / "tikzcache"
    cache.mkdir(exist_ok=True)
    so_lieu = (GEN / "so_lieu.tex").read_text(encoding="utf-8")
    stale = []
    for tex in sorted((HERE / "tikz").glob("*.tex")):
        if tex.name.startswith("_"):  # mã dùng chung được \input từ hình khác, không phải một hình riêng
            continue
        text = tex.read_text(encoding="utf-8")
        for inc in re.findall(r"\\input\{(tikz/[^}]+)\}", text):
            text += (HERE / (inc if inc.endswith(".tex") else inc + ".tex")).read_text(encoding="utf-8")
        h = hashlib.sha256(text.encode("utf-8"))
        for pat in sorted(set(re.findall(r"figdata/([^}\s]+?\.dat)", text))):
            for f in sorted((HERE / "figdata").glob(re.sub(r"\\[A-Za-z]+", "*", pat))):
                h.update(f.read_bytes())
        # Chỉ băm các macro số liệu mà hình này dùng; khoá chứa biến vòng lặp (\k, \m) thành mẫu regex.
        pats = [re.sub(r"\\\\[A-Za-z]+", "[^/}]+", re.escape(k)) for k in re.findall(r"\\SL\{([^}]+)\}", text)]
        if pats:
            want = re.compile(r"\\defSL\{(" + "|".join(pats) + r")(Raw|Pct|Sgn|Label)?\}")
            h.update("\n".join(l for l in so_lieu.splitlines() if want.match(l)).encode("utf-8"))
        stamp = cache / f"{tex.stem}.srchash"
        if stamp.exists() and stamp.read_text() == h.hexdigest() and (cache / f"{tex.stem}.pdf").exists():
            continue
        for ext in (".pdf", ".md5", ".dpth", ".log", ".dep"):
            (cache / f"{tex.stem}{ext}").unlink(missing_ok=True)
        stamp.write_text(h.hexdigest())
        stale.append(tex.stem)
    return stale


# ---------------------------------------------------------------- dựng
def main():
    fd = HERE / "figdata"
    fd.mkdir(exist_ok=True)
    n_dat = 0
    for f in (SRC / "outputs" / "figdata").glob("*.dat"):
        shutil.copy2(f, fd / f.name)
        n_dat += 1
    metrics = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS.glob("*.json"))}
    GEN.mkdir(exist_ok=True)
    (GEN / "so_lieu.tex").write_text(gen_macros(metrics), encoding="utf-8")
    n_tab = gen_tables(metrics)
    n_code = gen_code()
    print(f"figdata: {n_dat} tệp · macro từ {len(metrics)} JSON · {n_tab} bảng · {n_code} đoạn mã")
    stale = invalidate_figures()
    print(f"hình phải vẽ lại ({len(stale)}):", " ".join(stale) or "không")
    if "--no-pdf" in sys.argv:
        return
    cmd = ["latexmk", "-g", "-lualatex", "-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"latexmk lỗi (mã {r.returncode}), xem main.log")
    log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
    missing = sorted(set(re.findall(r"Thieu so lieu '([^']+)'", log)))
    if missing:
        sys.exit("PDF có số liệu thiếu (in ra ??), chưa ghi bản cuối: " + ", ".join(missing))
    undefined = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
    if undefined:
        sys.exit("Còn tham chiếu chưa định nghĩa: " + ", ".join(undefined))
    shutil.copy2(HERE / "main.pdf", HERE / PDF_NAME)
    print("Đã ghi", PDF_NAME)


if __name__ == "__main__":
    main()
