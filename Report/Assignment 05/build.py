"""Dựng báo cáo Assignment 05 bằng một lệnh:  python build.py [--no-pdf]

1. Chép outputs/figdata/*.dat  -> figdata/        (dữ liệu cho pgfplots)
2. Trích mã nguồn theo tên     -> code/           (mỗi \\maNguon{mod__ten} trong chapters/*.tex)
3. Sinh macro số liệu và bảng  -> generated/      (từ outputs/metrics/*.json)
4. latexmk -lualatex -shell-escape main.tex, rồi chép PDF thành A05_CT_nghiand.600.pdf

Không có con số nào gõ tay trong chương: chương chỉ gọi \\SL{tệp/khoá/...} hoặc \\input một bảng sinh ra ở đây.
"""
import ast
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src" / "Assignment 05"
PKG = SRC / "a05"
METRICS = SRC / "outputs" / "metrics"
GEN = HERE / "generated"
PDF_NAME = "A05_CT_nghiand.600.pdf"

MODELS = ["basic", "vgg", "resnet", "seresnet"]
LABELS = {"basic": "BasicCNN", "vgg": "VGGNet", "resnet": "ResNet", "seresnet": "SE-ResNet"}
# Lá JSON lưu dạng tỉ lệ 0..1 nhưng báo cáo đọc theo phần trăm.
PERCENT_KEYS = {"acc", "top5", "macro_f1", "f1", "precision", "recall", "pos_rate", "no_info_rate",
                "best_val_acc", "coarse_acc", "majority_acc", "vgg_saving", "pos_covered"}


# ---------------------------------------------------------------- định dạng số kiểu Việt
def num(v, decimals=2, sign=False):
    s = f"{v:+,.{decimals}f}" if sign else f"{v:,.{decimals}f}"
    s = s.replace(",", "\0").replace(".", "{,}").replace("\0", "{.}")
    return s.replace("-", "$-$").replace("+", "$+$") if sign else s.replace("-", "$-$")


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
            m, e = f"{v:.2e}".split("e")
            return f"${m.replace('.', '{,}')} \\times 10^{{{int(e)}}}$"
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


def gen_macros(metrics):
    lines = ["% Sinh tự động bởi build.py từ src/Assignment 05/outputs/metrics/*.json. Không sửa tay."]
    for name, data in metrics.items():
        for key, v in flatten(data, name):
            leaf = key.rsplit("/", 1)[-1]
            if "_names/" in key or "/sample_fine/" in key:  # tên lớp dùng làm nhãn hình: "aquatic_mammals" -> "aquatic mammals"
                v = v.replace("_", " ")
            lines.append(f"\\defSL{{{key}}}{{{fmt(v, leaf)}}}")
            if isinstance(v, (int, float)) and not isinstance(v, bool):  # dạng số thô cho toạ độ pgfplots
                lines.append(f"\\defSL{{{key}Raw}}{{{v:.6g}}}")
            if leaf in ("params", "macs") and isinstance(v, int):
                lines.append(f"\\defSL{{{key}M}}{{{num(v / 1e6, 2)}}}")
            if leaf in ("roc_auc", "pr_auc") and isinstance(v, float):
                lines.append(f"\\defSL{{{key}Pct}}{{{num(100 * v)}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- bảng
def table(spec, header, rows, note=None):
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    if note:
        out.append(f"\\par\\smallskip{{\\footnotesize {note}}}")
    return "\n".join(out) + "\n"


def bold_best(values, formatted, higher=True):
    best = max(values) if higher else min(values)
    return [f"\\textbf{{{f}}}" if v == best else f for v, f in zip(values, formatted)]


def tab_image(m, ds):
    mm = m[ds]["models"]
    accs = [mm[n]["acc"] for n in MODELS]
    acc_f = bold_best(accs, [num(100 * a) for a in accs])
    has_top5 = ds == "cifar100"
    head = ["Mô hình", "Acc (\\%)"] + (["Top-5 (\\%)"] if has_top5 else []) + \
           ["Macro-F1 (\\%)", "Tham số", "MACs (tr.)", "s/epoch", "ms/ảnh", "Epoch tốt"]
    rows = []
    for n, a in zip(MODELS, acc_f):
        r = mm[n]
        rows.append([LABELS[n], a] + ([num(100 * r["top5"])] if has_top5 else []) +
                    [num(100 * r["macro_f1"]), num(r["params"], 0), num(r["macs"] / 1e6, 1),
                     num(r["epoch_s"], 1), num(r["infer_ms"], 3), str(r["best_epoch"])])
    return table("l" + "r" * (len(head) - 1), head, rows)


def tab_diabetes(m):
    mm = m["diabetes"]["models"]
    head = ["Mô hình", "Acc", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Ngưỡng", "Tham số", "s/epoch"]
    cols = {k: [mm[n][k] for n in MODELS] for k in ("acc", "precision", "recall", "f1", "roc_auc", "pr_auc")}
    f = {k: bold_best(v, [num(100 * x) if k in PERCENT_KEYS else num(x, 4) for x in v]) for k, v in cols.items()}
    rows = [[LABELS[n]] + [f[k][i] for k in cols] +
            [num(mm[n]["threshold"], 3), num(mm[n]["params"], 0), num(mm[n]["epoch_s"], 2)]
            for i, n in enumerate(MODELS)]
    return table("l" + "r" * 9, head, rows,
                 "Acc, Precision, Recall, F1 tính theo \\%, tại ngưỡng chọn trên tập validation.")


def tab_summary(m):
    names = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "diabetes": "Diabetes"}
    out = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Tập & Mô hình & Acc (\%) & F1 (\%) & Tham số & s/epoch & Epoch tốt/chạy \\"]
    for ds, label in names.items():
        mm = m[ds]["models"]
        key = "f1" if ds == "diabetes" else "macro_f1"
        accs = [mm[n]["acc"] for n in MODELS]
        acc_f = bold_best(accs, [num(100 * a) for a in accs])
        out.append(r"\midrule")
        for i, n in enumerate(MODELS):
            r = mm[n]
            cells = [label if i == 0 else "", LABELS[n], acc_f[i], num(100 * r[key]), num(r["params"], 0),
                     num(r["epoch_s"], 2), f"{r['best_epoch']}/{r['epochs_run']}"]
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\par\smallskip{\footnotesize F1 là macro-F1 với hai tập ảnh, F1 của lớp dương với Diabetes.}"]
    return "\n".join(out) + "\n"


def tab_blocks(m):
    b = m["blocks"]["blocks"]
    spec = [("lenet", "LeNet-5", "1998"), ("alexnet", "AlexNet (thu nhỏ)", "2012"), ("vgg", "Khối VGG", "2014"),
            ("inception", "Khối Inception", "2014"), ("residual", "Khối Residual", "2015"),
            ("dense", "Khối Dense (4 tầng)", "2017"), ("depthwise", "Depthwise separable", "2017"),
            ("se", "SE", "2017"), ("cbam", "CBAM", "2018"), ("patch", "ViT: nhúng ô", "2020"),
            ("vit", "ViT: khối encoder", "2020")]
    rows = [[lab, yr, b[k]["in"], b[k]["out"], num(b[k]["params"], 0)] for k, lab, yr in spec]
    return table("llllr", ["Khối minh hoạ", "Năm", "Vào (C×H×W)", "Ra", "Tham số"], rows)


def tab_delta(m):
    """Chênh lệch giữa hai mô hình liền kề (điểm %): acc với ảnh, F1 với Diabetes; kèm chênh số tham số."""
    d = m["compare"]["delta"]
    steps = [("basic", "vgg", "M0 $\\to$ M1: sâu hơn + BN + GAP"), ("vgg", "resnet", "M1 $\\to$ M2: + đường tắt"),
             ("resnet", "seresnet", "M2 $\\to$ M3: + SE")]
    rows = []
    for a, b, lab in steps:
        k = f"{a}_{b}"
        cells = [lab]
        for ds in ("cifar10", "cifar100", "diabetes"):
            v = d[ds][k]
            s = num(v, 2, sign=True)
            cells.append(f"\\textit{{{s}}}" if abs(v) < 0.5 else s)
        cells.append(num(d["cifar10"][f"{k}_params"], 0) if d["cifar10"][f"{k}_params"] < 0
                     else "$+$" + num(d["cifar10"][f"{k}_params"], 0))
        rows.append(cells)
    return table("lrrrr", ["Bước", "CIFAR-10 (acc)", "CIFAR-100 (acc)", "Diabetes (F1)", "Tham số (CIFAR-10)"],
                 rows, "Đơn vị: điểm phần trăm. Chữ nghiêng: chênh lệch dưới 0,5 điểm, trong mức dao động của một lần chạy.")


def tab_trace(m, key):
    rows = [[f"{i}", tex_escape(r[0]), r[1], num(r[2], 0) if r[2] else "--"]
            for i, r in enumerate(m["compare"]["trace"][key])]
    return table("rlrr", ["\\#", "Tầng", "Shape đầu ra", "Tham số"], rows)


def tab_tiny_trace(m):
    rows = [[f"$f_{{{i}}}$" if i else "--", r[0], r[1]] for i, r in enumerate(m["concepts"]["trace"])]
    return table("cll", ["Hàm", "Tên", "Shape"], rows)


def tab_diab_desc(m):
    d = m["data"]["diabetes"]["desc"]
    rows = [[tex_escape(c), num(v["mean"]), num(v["std"]), num(v["min"]), num(v["max"])] for c, v in d.items()]
    return table("lrrrr", ["Đặc trưng", "Trung bình", "Độ lệch chuẩn", "Nhỏ nhất", "Lớn nhất"], rows)


def gen_tables(m):
    GEN.mkdir(exist_ok=True)
    builders = {"tab_cifar10": lambda: tab_image(m, "cifar10"), "tab_cifar100": lambda: tab_image(m, "cifar100"),
                "tab_diabetes": lambda: tab_diabetes(m), "tab_summary": lambda: tab_summary(m),
                "tab_blocks": lambda: tab_blocks(m), "tab_trace_basic": lambda: tab_trace(m, "basic_2d"),
                "tab_trace_seresnet": lambda: tab_trace(m, "seresnet_2d"),
                "tab_trace_basic1d": lambda: tab_trace(m, "basic_1d"), "tab_tiny_trace": lambda: tab_tiny_trace(m),
                "tab_diab_desc": lambda: tab_diab_desc(m), "tab_delta": lambda: tab_delta(m)}
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
    src = (PKG / f"{module}.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    nodes = ast.parse(src).body
    target = None
    for part in dotted.split("."):
        target = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                       and n.name == part), None)
        if target is None:
            raise KeyError(f"Không thấy '{dotted}' trong a05/{module}.py")
        nodes = target.body
    start = min([target.lineno] + [d.lineno for d in target.decorator_list]) - 1
    block = lines[start:target.end_lineno]
    indent = len(block[0]) - len(block[0].lstrip())
    return "\n".join(l[indent:] for l in block) + "\n"


def gen_code():
    out = HERE / "code"
    out.mkdir(exist_ok=True)
    refs = set()
    for tex in (HERE / "chapters").glob("*.tex"):
        refs |= set(re.findall(r"\\maNguon\{([A-Za-z0-9_.]+)\}", tex.read_text(encoding="utf-8")))
    for ref in sorted(refs):
        module, name = ref.split("__", 1)
        header = f"# a05/{module}.py\n"
        (out / f"{ref}.py").write_text(header + extract(module, name), encoding="utf-8")
    return len(refs)


# ---------------------------------------------------------------- làm mới cache hình
def invalidate_figures():
    """tikz external chỉ so md5 của mã hình, không biết tệp .dat đã đổi. Ở đây băm mã hình
    (kể cả tệp nó \\input), các tệp figdata nó đọc và so_lieu.tex nếu hình dùng \\SL; hình nào có
    hash mới thì xoá bản cache của đúng hình đó để external vẽ lại. Trả về danh sách hình phải vẽ lại."""
    import hashlib
    cache = HERE / "tikzcache"
    cache.mkdir(exist_ok=True)
    so_lieu = (GEN / "so_lieu.tex").read_bytes()
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
            want = re.compile(r"\\defSL\{(" + "|".join(pats) + r")(Raw|M|Pct)?\}")
            h.update("\n".join(l for l in so_lieu.decode("utf-8").splitlines() if want.match(l)).encode("utf-8"))
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
    shutil.copy2(HERE / "main.pdf", HERE / PDF_NAME)
    print("Đã ghi", PDF_NAME)


if __name__ == "__main__":
    main()
