"""Dựng báo cáo tiểu luận bằng một lệnh:  python build.py [--no-pdf]

1. Sinh macro số liệu và bảng  -> generated/      (từ src/TieuLuan 01/outputs/metrics/*.json)
2. Trích mã nguồn theo tên     -> code/           (mỗi \\maNguon{module__ten} trong chapters/*.tex)
3. Xóa cache của hình TikZ nào có mã hoặc số liệu đổi
4. latexmk -lualatex -shell-escape main.tex, rồi chép PDF thành tieuluan01_CT_nghiand.600.pdf

Chương không gõ tay con số thực nghiệm nào: chỉ gọi \\SL{tệp/khoá/...} hoặc \\input một bảng sinh ra ở đây.
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
SRC = HERE.parents[1] / "src" / "TieuLuan 01"
METRICS = SRC / "outputs" / "metrics"
GEN = HERE / "generated"
PDF_NAME = "tieuluan01_CT_nghiand.600.pdf"
SKIP_KEYS = ("/hist/", "/pred", "/confusion", "/sample")       # mảng dữ liệu hình và bảng, không phải số liệu chữ
CELLS = {"rnn": "Simple RNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}
ARCHS = {"custom": "Custom CNN", "vgg11": "VGG-11", "mobilenetv2": "MobileNetV2", "resnet18": "ResNet-18"}


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
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
             .replace("&", "\\&").replace("#", "\\#").replace("$", "\\$"))


def flatten(obj, prefix):
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
        for key, v in flatten(data, name):
            if any(s in key + "/" for s in SKIP_KEYS):
                continue
            lines.append(f"\\defSL{{{key}}}{{{fmt(v)}}}")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                lines.append(f"\\defSL{{{key}Raw}}{{{v:.6g}}}")
                if abs(v) <= 1.5:
                    lines.append(f"\\defSL{{{key}Pct}}{{{num(100 * v, 2)}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- bảng
def table(spec, header, rows, note=None):
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [r if isinstance(r, str) else " & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    body = "\n".join(out) + "\n"
    # ghi chú ra tệp riêng <tên>_note.tex: \bang co bảng theo bề rộng trang, ghi chú thì không được co
    return (body, note) if note else body


def best(values, formatted, higher=True):
    b = max(values) if higher else min(values)
    return [f"\\textbf{{{f}}}" if v == b else f for v, f in zip(values, formatted)]


def size(kb):
    return f"{num(kb, 1)} KB" if kb < 1024 else f"{num(kb / 1024, 1)} MB"


def secs(s):
    return num(s, 1 if s < 100 else 0)


def tab_cls(rows_in, note, binary=True):
    """rows_in: [(tên, dict kết quả)]."""
    keys = ["acc", "precision", "recall", "f1"] + (["auc"] if binary else [])
    cols = {k: best([r[k] for _, r in rows_in], [num(100 * r[k]) if k == "acc" else num(r[k], 4) for _, r in rows_in])
            for k in keys}
    rows = [[n] + [cols[k][i] for k in keys] + [secs(r["train_s"]), size(r["size_kb"])] for i, (n, r) in enumerate(rows_in)]
    head = ["Mô hình", "Acc (\\%)", "Precision", "Recall", "F1"] + (["ROC-AUC"] if binary else []) + ["Huấn luyện (s)", "Dung lượng"]
    return table("l" + "r" * (len(head) - 1), head, rows, note)


def tab_reg(rows_in, note):
    mae = best([r["mae"] for _, r in rows_in], [num(r["mae"], 3) for _, r in rows_in], higher=False)
    rmse = best([r["rmse"] for _, r in rows_in], [num(r["rmse"], 3) for _, r in rows_in], higher=False)
    r2 = best([r["r2"] for _, r in rows_in], [num(r["r2"], 4) for _, r in rows_in])
    rows = [[n, mae[i], rmse[i], r2[i], secs(r["train_s"]), size(r["size_kb"])] for i, (n, r) in enumerate(rows_in)]
    return table("lrrrrr", ["Mô hình", "MAE (tỷ đ)", "RMSE (tỷ đ)", "$R^2$", "Huấn luyện (s)", "Dung lượng"], rows, note)


def ch2_tables(m):
    sk = m["ch2_sklearn"]
    mlp = [("MLP NumPy (scratch)", "ch2_mlp_scratch"), ("MLP Keras", "ch2_mlp_keras"), ("MLP PyTorch", "ch2_mlp_torch")]
    d_cls = [("Logistic Regression (balanced)", sk["diabetes/logreg"]), ("SVM nhân RBF (balanced)", sk["diabetes/svm"]),
             ("Random Forest (200 cây)", sk["diabetes/rf"])]
    h_cls = [("Linear Regression (OLS)", sk["house/ols"]), ("Ridge ($\\alpha = 1$)", sk["house/ridge"]),
             ("SVR nhân RBF", sk["house/svr"]), ("Random Forest (200 cây)", sk["house/rf"])]
    r_cls = [("Logistic Regression + TF-IDF", sk["reviews/logreg"]), ("Linear SVM (LinearSVC)", sk["reviews/linsvm"]),
             ("Random Forest + TF-IDF", sk["reviews/rf"])]
    note_d = "Precision, Recall, F1 tính cho lớp dương (mắc bệnh), ngưỡng 0,5; chữ đậm: tốt nhất cột."
    note_r = "Precision, Recall, F1 lấy trung bình macro trên ba lớp; chữ đậm: tốt nhất cột."
    out = {"tab_ch2_diab_sk": tab_cls(d_cls, note_d), "tab_ch2_house_sk": tab_reg(h_cls, "Chữ đậm: tốt nhất cột."),
           "tab_ch2_rev_sk": tab_cls(r_cls, note_r, binary=False)}
    out["tab_ch2_diab_all"] = tab_cls(d_cls + [(n, m[f]["diabetes"]) for n, f in mlp], note_d)
    out["tab_ch2_house_all"] = tab_reg(h_cls + [(n, m[f]["house"]) for n, f in mlp],
                                       "MLP học nhãn đã chuẩn hóa z-score rồi đổi ngược về tỷ đồng; chữ đậm: tốt nhất cột.")
    out["tab_ch2_rev_all"] = tab_cls(r_cls + [(n, m[f]["reviews"]) for n, f in mlp], note_r, binary=False)
    # bảng ba nền tảng MLP
    rows = []
    for ds, lab in (("diabetes", "Tiểu đường"), ("house", "Giá nhà"), ("reviews", "Cảm xúc")):
        for i, (n, f) in enumerate(mlp):
            r = m[f][ds]
            main = num(r["f1"], 4) if ds != "house" else num(r["r2"], 4)
            rows.append([lab if i == 0 else "", n.replace("MLP ", ""), main, num(r["params"], 0), str(r["epochs"]),
                         secs(r["train_s"]), size(r["size_kb"])])
        rows.append("\\midrule") if ds != "reviews" else None
    rows = [r for r in rows if r is not None]
    out["tab_ch2_fw"] = table("llrrrrr", ["Tập dữ liệu", "Nền tảng", "F1 / $R^2$", "Tham số", "Epoch", "Huấn luyện (s)",
                                          "Dung lượng"], rows,
                              "F1 cho tiểu đường (lớp dương) và cảm xúc (macro); $R^2$ cho giá nhà. NumPy và Keras chạy CPU, "
                              "PyTorch chạy GPU; số epoch là số epoch thực chạy trước khi dừng sớm.")
    return out


def ch3_tables(m):
    b = m["bench"]
    rows = []
    for ds, lab in (("svhn", "SVHN"), ("eurosat", "EuroSAT")):
        r = m[f"ch3_{ds}"]
        keys = list(ARCHS)
        acc = best([r[a]["acc"] for a in keys], [num(100 * r[a]["acc"]) for a in keys])
        for i, a in enumerate(keys):
            bb = b[f"ch3/{ds}/{a}"]
            rows.append([lab if i == 0 else "", ARCHS[a], acc[i], num(r[a]["f1"], 4), num(r[a]["params"] / 1e6, 2),
                         num(r[a]["gflops"], 3), num(bb["gpu_ms"], 2), num(bb["cpu_ms"], 2), num(bb["fps"], 0),
                         secs(r[a]["train_s"]), size(r[a]["size_kb"])])
        if "custom_keras" in r:
            k = r["custom_keras"]
            rows.append(["", "Custom CNN (Keras)", num(100 * k["acc"]), num(k["f1"], 4), num(k["params"] / 1e6, 2),
                         num(r["custom"]["gflops"], 3), "--", num(b[f"ch3/{ds}/custom_keras"]["cpu_ms"], 2), "--",
                         secs(k["train_s"]) + "*", size(k["size_kb"])])
        rows.append("\\midrule")
    c = m["ch3_diabetes1d"]["cnn1d"]
    bb = b["ch3/diabetes1d/cnn1d"]
    rows.append(["Tiểu đường", "1D CNN (Conv1D)", num(100 * c["acc"]), num(c["f1"], 4), num(c["params"] / 1e6, 3),
                 num(c["gflops"], 4), num(bb["gpu_ms"], 2), num(bb["cpu_ms"], 2), num(bb["fps"], 0), secs(c["train_s"]),
                 size(c["size_kb"])])
    note = ("F1 là macro trên 10 lớp (1D CNN: lớp dương). FLOPs $= 2 \\times$ số phép nhân-cộng của một ảnh. Độ trễ đo lô 1 "
            "(trung vị), CPU 4 luồng; FPS đo trên GPU với lô 256. Huấn luyện trên GPU RTX 4060 Laptop, "
            "* Keras chạy CPU (TensorFlow trên Windows không dùng GPU).")
    return {"tab_ch3": table("llrrrrrrrrr", ["Dữ liệu", "Kiến trúc", "Acc (\\%)", "F1", "Tham số (M)", "GFLOPs",
                                             "GPU (ms)", "CPU (ms)", "FPS", "Huấn luyện (s)", "Dung lượng"], rows, note)}


def ch4_tables(m):
    t, k, b = m["ch4_torch"], m["ch4_keras"], m["bench"]
    keys = list(CELLS)
    rm = best([t[f"stock/{c}"]["rmse"] for c in keys] + [t["stock/naive"]["rmse"]],
              [num(t[f"stock/{c}"]["rmse"], 3) for c in keys] + [num(t["stock/naive"]["rmse"], 3)], higher=False)
    rows = []
    for i, c in enumerate(keys):
        r = t[f"stock/{c}"]
        rows.append([CELLS[c], rm[i], num(r["mae"], 3), num(r["r2"], 4), num(100 * r["r2_ret"], 2, sign=True),
                     num(100 * r["dir_acc"], 1), num(r["params"], 0), num(b[f"ch4/stock/{c}_torch"]["cpu_ms"], 3),
                     size(r["size_kb"])])
    nv = t["stock/naive"]
    rows += ["\\midrule", ["Giữ nguyên giá ($\\hat P_{t+1} = P_t$)", rm[-1], num(nv["mae"], 3), num(nv["r2"], 4),
                           num(0.0, 2), "--", "0", "--", "--"]]
    out = {"tab_ch4_stock": table("lrrrrrrrr", ["Mô hình (PyTorch)", "RMSE", "MAE", "$R^2$ giá", "$R^2$ lợi suất (\\%)",
                                                "Đúng chiều (\\%)", "Tham số", "Trễ CPU (ms)", "Dung lượng"], rows,
                                  "RMSE, MAE tính bằng nghìn đồng trên giá đóng cửa của 706 phiên kiểm thử; chữ đậm: nhỏ nhất.")}
    ks = ["acc", "precision", "recall", "f1", "auc"]
    cols = {q: best([t[f"churn/{c}"][q] for c in keys], [num(100 * t[f"churn/{c}"][q]) if q == "acc" else
                                                      num(t[f"churn/{c}"][q], 4) for c in keys]) for q in ks}
    rows = [[CELLS[c]] + [cols[q][i] for q in ks] + [num(t[f"churn/{c}"]["threshold"], 2), num(t[f"churn/{c}"]["params"], 0),
                                                     num(b[f"ch4/churn/{c}_torch"]["cpu_ms"], 3), size(t[f"churn/{c}"]["size_kb"])]
            for i, c in enumerate(keys)]
    out["tab_ch4_churn"] = table("lrrrrrrrrr", ["Mô hình (PyTorch)", "Acc (\\%)", "Precision", "Recall", "F1", "ROC-AUC",
                                               "Ngưỡng", "Tham số", "Trễ CPU (ms)", "Dung lượng"], rows,
                                 "Precision, Recall, F1 cho lớp rời bỏ tại ngưỡng chọn trên tập kiểm định; chữ đậm: tốt nhất cột.")
    rows = []
    for c in keys:
        for fw, r in (("PyTorch", t), ("Keras", k)):
            rows.append([CELLS[c] if fw == "PyTorch" else "", fw, num(r[f"stock/{c}"]["rmse"], 3), num(r[f"churn/{c}"]["f1"], 4),
                         num(r[f"churn/{c}"]["auc"], 4), num(r[f"stock/{c}"]["params"], 0),
                         size(r[f"stock/{c}"]["size_kb"]),
                         num(b[f"ch4/stock/{c}_{'torch' if fw == 'PyTorch' else 'keras'}"]["cpu_ms"], 3),
                         secs(r[f"stock/{c}"]["train_s"]) + " / " + secs(r[f"churn/{c}"]["train_s"])])
        if c != "bilstm":
            rows.append("\\midrule")
    out["tab_ch4_fw"] = table("llrrrrrrr", ["Kiến trúc", "Nền tảng", "RMSE FPT", "F1 KKBox", "AUC KKBox", "Tham số (FPT)",
                                            "Dung lượng", "Trễ CPU (ms)", "Huấn luyện (s)"], rows,
                              "Độ trễ: mô hình FPT, lô 1, CPU 4 luồng cho cả hai nền tảng. Thời gian huấn luyện FPT / KKBox: "
                              "PyTorch trên GPU, Keras trên CPU.")
    return out


def sample_tables(e):
    d = e["diabetes"]
    cols = d["sample_cols"]
    rows = []
    for i, r in enumerate(d["sample"], 1):
        x = dict(zip(cols, r))
        rows.append([str(i), "Nữ" if x["gender"] == "Female" else "Nam", num(float(x["age"]), 0), x["hypertension"],
                     x["heart_disease"], tex_escape(x["smoking_history"]), num(float(x["bmi"]), 2),
                     num(float(x["HbA1c_level"]), 1), x["blood_glucose_level"], x["diabetes"]])
    out = {"tab_sample_diab": table("rlrcclrrrc", ["STT", "Giới", "Tuổi", "Huyết áp", "Tim", "Hút thuốc", "BMI", "HbA1c",
                                                  "Glucose", "Nhãn $y$"], rows)}
    rows = []
    for i, r in enumerate(e["house"]["sample"], 1):
        f = lambda v, d=1: "--" if v is None or (isinstance(v, float) and math.isnan(v)) else num(float(v), d)  # noqa: E731
        rows.append([str(i), f(r[0]), f(r[1], 0), f(r[2], 0), f(r[3]), f(r[4]), tex_escape(r[5]), tex_escape(r[6]), f(r[7], 2)])
    out["tab_sample_house"] = table("rrrrrrllr", ["STT", "Diện tích", "Tầng", "Phòng ngủ", "Mặt tiền", "Đường vào",
                                                  "Quận/huyện", "Tỉnh/thành", "Giá (tỷ)"], rows,
                                    "Dấu -- là ô trống trong dữ liệu gốc (được điền trung vị và đánh dấu khi tiền xử lý).")
    rows = [[str(i), f"{r[0]} sao", tex_escape(r[1]), r[2]] for i, r in enumerate(e["reviews"]["sample"], 1)]
    out["tab_sample_rev"] = table("rlp{10.2cm}l", ["STT", "Số sao", "Nội dung nhận xét (nguyên văn tiếng Anh)", "Nhãn"], rows)
    rows = [[r[0], *(num(v / 1000, 2) for v in r[1:5]), num(r[5], 0), num(r[6] / 1000, 2), num(r[7], 2)]
            for r in e["stock"]["sample"]]
    out["tab_sample_stock"] = table("lrrrrrrr", ["Ngày", "Open", "High", "Low", "Close", "Volume (cp)", "SMA20", "RSI-14"], rows,
                                    "Giá tính bằng nghìn đồng, đã điều chỉnh theo chia tách và cổ tức (auto\\_adjust của Yahoo Finance).")
    return out


def gen_tables(m):
    GEN.mkdir(exist_ok=True)
    builders = [lambda: sample_tables(m["eda"]), lambda: ch2_tables(m), lambda: ch3_tables(m), lambda: ch4_tables(m)]
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
    return n


# ---------------------------------------------------------------- trích mã nguồn
def extract(module, dotted):
    path = SRC / "app" / "app.py" if module == "app" else SRC / "tl" / f"{module}.py"
    if not path.exists():                                   # kịch bản thí nghiệm nằm ở scripts/
        path = SRC / "scripts" / f"{module}.py"
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    nodes = ast.parse(src).body
    target = None
    for part in dotted.split("."):
        target = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part), None)
        if target is None:
            raise KeyError(f"Không thấy '{dotted}' trong {path.name}")
        nodes = target.body
    start = min([target.lineno] + [d.lineno for d in target.decorator_list]) - 1
    block = lines[start:target.end_lineno]
    indent = len(block[0]) - len(block[0].lstrip())
    rel = path.relative_to(SRC).as_posix()
    return f"# {rel}\n" + "\n".join(l[indent:] for l in block) + "\n"


def gen_code():
    out = HERE / "code"
    out.mkdir(exist_ok=True)
    refs = set()
    for tex in (HERE / "chapters").glob("*.tex"):
        refs |= set(re.findall(r"\\maNguon\{([A-Za-z0-9_.]+)\}", tex.read_text(encoding="utf-8")))
    for ref in sorted(refs):
        module, name = ref.split("__", 1)
        (out / f"{ref}.py").write_text(extract(module, name), encoding="utf-8")
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


def main():
    metrics = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS.glob("*.json"))}
    GEN.mkdir(exist_ok=True)
    (GEN / "so_lieu.tex").write_text(gen_macros(metrics), encoding="utf-8")
    n_tab = gen_tables(metrics)
    n_code = gen_code()
    print(f"macro từ {len(metrics)} JSON · {n_tab} bảng · {n_code} đoạn mã")
    stale = invalidate_figures()
    print(f"hình TikZ phải vẽ lại ({len(stale)}):", " ".join(stale) or "không")
    if "--no-pdf" in sys.argv:
        return
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
