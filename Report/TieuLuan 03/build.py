"""Dựng báo cáo Tiểu luận 03.

    python build.py               toàn bộ báo cáo -> tieuluan03_CT_07_nghiand.pdf
    python build.py --chapter N   chỉ chương N (TikZ vẽ tại chỗ) -> kiemtra_chN.pdf, in mọi lỗi LaTeX
    python build.py --no-pdf      chỉ sinh macro và mã

1. Sinh macro số liệu  -> generated/so_lieu.tex   (từ src/TieuLuan 03/outputs/metrics/*.json)
2. Trích mã nguồn      -> code/<module>__<tên>.py (mỗi \\maNguon / \\maDai trong chapters/*.tex)
3. Xóa cache của hình TikZ nào có mã hoặc số liệu đổi
4. latexmk -lualatex -shell-escape main.tex

Bảng (generated/chN_*.tex) do scripts/chN_tables.py của từng chương sinh, không sinh ở đây.
"""
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# build.py chỉ cần numpy để nạp tl3.texfmt; một luồng BLAS là đủ và tránh lỗi cấp phát bộ nhớ của OpenBLAS
# khi các làn huấn luyện đang chiếm RAM.
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src" / "TieuLuan 03"
sys.path.insert(0, str(SRC))
from tl3.texfmt import fmt, num  # noqa: E402

METRICS = SRC / "outputs" / "metrics"
GEN = HERE / "generated"
CODE = HERE / "code"
PDF_NAME = "tieuluan03_CT_07_nghiand.pdf"
CODE_DIRS = ("tl3", "scripts", "app")
# mảng dữ liệu của hình và bảng, không phải số liệu chữ; so khớp cả đoạn (key + "/")
SKIP_KEYS = ("/rows/", "/history/", "/series/", "/curve/", "/samples/", "/per_q/", "/traces/", "/examples/")
MAX_LIST = 13
MAX_STR = 160


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
        for key, v in flatten(data, name):
            if any(s in key + "/" for s in SKIP_KEYS) or any(c in key for c in "\\%#{}~$& "):
                continue
            if v is None or isinstance(v, str) and len(v) > MAX_STR:
                continue
            # năm in liền (2021), không tách nghìn như số đếm (2.021)
            is_year = isinstance(v, int) and 1000 <= v <= 2999 and "year" in key.rsplit("/", 1)[-1]
            lines.append(f"\\defSL{{{key}}}{{{v if is_year else fmt(v)}}}")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v)):
                lines.append(f"\\defSL{{{key}Raw}}{{{v:.6g}}}")
                if abs(v) <= 1.5:
                    lines.append(f"\\defSL{{{key}Pct}}{{{num(100 * v, 2)}}}")
    return "\n".join(lines) + "\n"


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
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part:
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


def gen_code(chapters):
    CODE.mkdir(exist_ok=True)
    refs, errors = set(), []
    for tex in chapters:
        refs |= set(re.findall(r"\\ma(?:Nguon|Dai)\{([A-Za-z0-9_.]+)\}", tex.read_text(encoding="utf-8")))
    for ref in sorted(refs):
        module, name = ref.split("__", 1)
        try:
            (CODE / f"{ref}.py").write_text(extract(module, name), encoding="utf-8", newline="\n")
        except (KeyError, SyntaxError) as e:
            errors.append(f"{ref}: {e}")
            (CODE / f"{ref}.py").write_text(f"# THIẾU MÃ: {e}\n", encoding="utf-8", newline="\n")
    for e in errors:
        print("!! mã nguồn:", e)
    return len(refs), errors


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


def report_log(log_path):
    """In lỗi LaTeX (kèm 4 dòng ngữ cảnh), số liệu thiếu, tham chiếu chưa định nghĩa và hộp tràn lớn."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    errs = [i for i, s in enumerate(lines) if s.startswith("!")]
    for i in errs[:40]:
        print("\n".join(lines[i:i + 4]), "\n")
    missing = sorted(set(re.findall(r"Thieu so lieu '([^']+)'", text)))
    undefined = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)'", text)))
    # cả \hbox (tràn lề phải) lẫn \vbox (hình/trang cao quá khổ giấy)
    overfull = [float(x) for x in re.findall(r"Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)", text)]
    big = sum(1 for x in overfull if x > 20)
    print(f"{len(errs)} lỗi LaTeX · {len(missing)} số liệu/hình thiếu · {len(undefined)} tham chiếu/trích dẫn chưa có · "
          f"{big} hộp tràn > 20pt")
    if missing:
        print("  thiếu:", ", ".join(missing[:60]))
    if undefined:
        print("  chưa định nghĩa:", ", ".join(undefined[:60]))
    return errs, missing, undefined


def lualatex_pass(job, tex, tries=3):
    """Một lượt lualatex; khi máy cạn bộ nhớ lualatex có thể bị giết giữa chừng mà không in lỗi '!' nào,
    nên coi lượt là hỏng nếu log không có 'Output written on' và chạy lại."""
    cmd = ["lualatex", "-shell-escape", "-interaction=nonstopmode", f"-jobname={job}", tex]
    for i in range(1, tries + 1):
        subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL)
        log = (HERE / f"{job}.log").read_text(encoding="utf-8", errors="replace")
        if "Output written on" in log:
            return True
        print(f"!! lượt lualatex {job} dừng giữa chừng (lần {i}/{tries}), chạy lại")
    return False


def biber_pass(job):
    r = subprocess.run(["biber", job], cwd=HERE, stdout=subprocess.DEVNULL)
    if r.returncode != 0:
        print(f"!! biber {job} lỗi, xem {job}.blg")


def check_chapter(n):
    """Một chương: lualatex + biber + lualatex, TikZ vẽ tại chỗ, in mọi lỗi một lượt."""
    job = f"kiemtra_ch{n}"
    tex = r"\def\kiemtranhanh{}\def\chuongkiemtra{" + str(n) + r"}\input{main}"
    ok = lualatex_pass(job, tex)
    biber_pass(job)
    ok = lualatex_pass(job, tex) and ok
    if not ok:
        sys.exit(f"lualatex không chạy trọn được {job}")
    errs, _, _ = report_log(HERE / f"{job}.log")
    print("->", HERE / f"{job}.pdf")
    sys.exit(1 if errs else 0)


def build_inline():
    """Toàn bộ báo cáo, TikZ vẽ tại chỗ (không externalize: ~150 hình × một lượt dịch cả tài liệu là quá chậm).
    lualatex → biber → lualatex × 2, rồi chép PDF thành PDF_NAME."""
    job = "toanbo"
    tex = r"\def\kiemtranhanh{}\input{main}"
    for step in ("tex", "biber", "tex", "tex"):
        if step == "biber":
            biber_pass(job)
        elif not lualatex_pass(job, tex):
            sys.exit(f"lualatex không chạy trọn được {job}; xem {job}.log")
    errs, missing, undefined = report_log(HERE / f"{job}.log")
    shutil.copy2(HERE / f"{job}.pdf", HERE / PDF_NAME)
    print("Đã ghi", PDF_NAME, "(có lỗi)" if errs else "")
    sys.exit(1 if errs else 0)


def main():
    metrics = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS.glob("*.json"))}
    GEN.mkdir(exist_ok=True)
    # nhiều làn chạy --chapter song song: ghi tệp tạm rồi thay nguyên khối để lualatex không đọc phải tệp dở
    tmp = GEN / f"so_lieu.{os.getpid()}.tmp"
    tmp.write_text(gen_macros(metrics), encoding="utf-8", newline="\n")
    os.replace(tmp, GEN / "so_lieu.tex")
    chapter = None
    if "--chapter" in sys.argv:
        chapter = int(sys.argv[sys.argv.index("--chapter") + 1])
    chapters = sorted((HERE / "chapters").glob("*.tex"))
    if chapter is not None:
        chapters = [p for p in chapters if p.stem == f"ch{chapter}"]
    n_code, code_err = gen_code(chapters)
    print(f"macro từ {len(metrics)} JSON · {n_code} đoạn mã ({len(code_err)} lỗi)")
    if chapter is not None:
        return check_chapter(chapter)
    if "--inline" in sys.argv:
        return build_inline()
    stale = invalidate_figures()
    print(f"hình TikZ phải vẽ lại ({len(stale)}):", " ".join(stale) or "không")
    if "--no-pdf" in sys.argv:
        return
    cmd = ["latexmk", "-g", "-lualatex", "-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    r = subprocess.run(cmd, cwd=HERE)
    errs, missing, undefined = report_log(HERE / "main.log")
    if r.returncode != 0 or errs:
        sys.exit(f"latexmk lỗi (mã {r.returncode}), xem main.log")
    if missing or undefined or code_err:
        print("!! PDF còn chỗ thiếu — xem danh sách trên")
    shutil.copy2(HERE / "main.pdf", HERE / PDF_NAME)
    print("Đã ghi", PDF_NAME)


if __name__ == "__main__":
    main()
