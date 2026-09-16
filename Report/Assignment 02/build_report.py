"""Dựng báo cáo Assignment 02 — HTML tiếng Việt rồi in ra PDF bằng Chrome headless.

Chạy:  python report/build_report.py
Kết quả: report/Assignment_02.html  và  report/A02_CT_nghiand.600.pdf

Mọi con số trong báo cáo được ĐỌC TỪ model/metadata.json của ba ứng dụng, không
gõ tay. Sửa notebook rồi chạy lại là báo cáo tự cập nhật theo — không có chỗ nào
để một con số cũ nằm lại trong văn bản.
"""

from __future__ import annotations

import base64
import html as html_mod
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1] / "src" / "Assignment 02"
FIG = REPORT / "figures"
ASSETS = REPORT / "assets"

GITHUB_URL = "https://github.com/Nghaiz/Assignment-02-Intelligent-System"

# Ba tập dữ liệu gốc — in ngay trang đầu cạnh liên kết mã nguồn.
DATASETS = [
    ("Ứng dụng 1 — Tiểu đường", "Pima Indians Diabetes Database",
     "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"),
    ("Ứng dụng 2 — Giá nhà", "Vietnam Housing Dataset 2024",
     "https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024"),
    ("Ứng dụng 3 — Hành vi khách hàng", "Women's E-Commerce Clothing Reviews",
     "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews"),
]

STUDENT = {
    "lop": "D23CTPM01",
    "ten": "Nguyễn Duy Nghĩa",
    "mssv": "B23DCCN600",
    "gvhd": "PGS.TS Trần Đình Quế",
    "hocky": "Học kỳ 1 năm học 2026 – 2027",
}

APPS = ["diabetes", "house_price", "customer_behavior"]


# ----------------------------------------------------------------------------
# Nạp dữ liệu thật
# ----------------------------------------------------------------------------

def load_meta(app: str) -> dict:
    p = ROOT / app / "model" / "metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"Thiếu {p} — hãy chạy notebook của ứng dụng '{app}' trước.")
    return json.loads(p.read_text(encoding="utf-8"))


META = {a: load_meta(a) for a in APPS}


def notebook_code(app: str, needle: str, *, drop_comment_header: bool = False) -> str:
    """Lấy nguyên văn nội dung ô code đầu tiên của notebook có chứa `needle`.

    Mã in trong báo cáo phải là mã ĐÃ CHẠY THẬT, nên nó được đọc ngược từ file
    .ipynb chứ không chép tay sang đây — chép tay thì sớm muộn hai bản sẽ lệch
    nhau mà không có gì báo lỗi.
    """
    nbs = {
        "diabetes": "diabetes/notebook/Diabetes_A02.ipynb",
        "house_price": "house_price/notebook/HousePrice_A02.ipynb",
        "customer_behavior": "customer_behavior/notebook/CustomerBehavior_A02.ipynb",
    }
    nb = json.loads((ROOT / nbs[app]).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if needle in src:
            lines = src.splitlines()
            if drop_comment_header:
                while lines and (lines[0].startswith("#") or not lines[0].strip()):
                    lines.pop(0)
            return "\n".join(lines).strip()
    raise LookupError(f"Không tìm thấy ô code chứa {needle!r} trong notebook {app}")


def notebook_output(app: str, needle: str, max_lines: int = 40) -> str:
    """Lấy kết quả in ra (stdout) của ô code đầu tiên chứa `needle`."""
    nbs = {
        "diabetes": "diabetes/notebook/Diabetes_A02.ipynb",
        "house_price": "house_price/notebook/HousePrice_A02.ipynb",
        "customer_behavior": "customer_behavior/notebook/CustomerBehavior_A02.ipynb",
    }
    nb = json.loads((ROOT / nbs[app]).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if needle not in "".join(cell["source"]):
            continue
        chunks = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append("".join(out.get("text", [])))
            elif out.get("output_type") == "execute_result":
                chunks.append("".join(out.get("data", {}).get("text/plain", [])))
        text = "".join(chunks).rstrip()
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... (rút gọn, còn {len(lines) - max_lines} dòng)"]
        return "\n".join(lines)
    raise LookupError(f"Không tìm thấy kết quả của ô chứa {needle!r} trong notebook {app}")


# ----------------------------------------------------------------------------
# Thành phần HTML
# ----------------------------------------------------------------------------

_FIG_N = {"count": 0}
_TBL_N = {"count": 0}


def esc(s) -> str:
    return html_mod.escape(str(s))


def data_uri(path: Path) -> str:
    """Nhúng ảnh thành data URI.

    Chrome headless khi in PDF từ file:// đôi khi bỏ qua ảnh tham chiếu tương
    đối tuỳ thời điểm layout; nhúng thẳng vào HTML thì không còn phụ thuộc nữa.
    """
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def figure(name: str, caption: str, width: str = "100%") -> str:
    p = FIG / name
    if not p.exists():
        return (f'<div class="missing">[Thiếu hình: {esc(name)} — '
                f'chạy lại notebook hoặc bộ dựng hình minh hoạ]</div>')
    _FIG_N["count"] += 1
    return (f'<figure><img src="{data_uri(p)}" style="width:{width}">'
            f'<figcaption>Hình {_FIG_N["count"]}. {caption}</figcaption></figure>')


PY_KW = (r"\b(and|as|assert|async|await|break|class|continue|def|del|elif|else|except|"
         r"False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|"
         r"pass|raise|return|True|try|while|with|yield)\b")


def highlight(code: str) -> str:
    """Tô màu cú pháp Python tối giản, đủ để mã đọc được trên trang in."""
    out, i, n = [], 0, len(code)
    while i < n:
        ch = code[i]
        if ch == "#":
            j = code.find("\n", i)
            j = n if j == -1 else j
            out.append(f'<span class="c">{esc(code[i:j])}</span>')
            i = j
        elif ch in "\"'":
            q3 = code[i:i + 3]
            if q3 in ('"""', "'''"):
                j = code.find(q3, i + 3)
                j = n if j == -1 else j + 3
            else:
                j = i + 1
                while j < n and code[j] != ch:
                    j += 2 if code[j] == "\\" else 1
                j = min(j + 1, n)
            out.append(f'<span class="s">{esc(code[i:j])}</span>')
            i = j
        else:
            j = i
            while j < n and code[j] not in "#\"'":
                j += 1
            seg = esc(code[i:j])
            seg = re.sub(PY_KW, r'<span class="k">\1</span>', seg)
            seg = re.sub(r"\b(\d+\.?\d*)\b", r'<span class="n">\1</span>', seg)
            out.append(seg)
            i = j
    return "".join(out)


def code(src: str, caption: str = "", lang: str = "Python") -> str:
    cap = f'<div class="codecap">{esc(caption)}</div>' if caption else ""
    return (f'<div class="codebox"><div class="codehdr">{esc(lang)}</div>'
            f'<pre class="code">{highlight(src)}</pre></div>{cap}')


def output(text: str, caption: str = "") -> str:
    cap = f'<div class="codecap">{esc(caption)}</div>' if caption else ""
    return (f'<div class="outbox"><div class="outhdr">Kết quả chạy</div>'
            f'<pre class="out">{esc(text)}</pre></div>{cap}')


def table(headers, rows, caption: str = "", cls: str = "") -> str:
    _TBL_N["count"] += 1
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = (f'<div class="tblcap">Bảng {_TBL_N["count"]}. {caption}</div>'
           if caption else "")
    return f'{cap}<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def note(title: str, body: str, kind: str = "info") -> str:
    return f'<div class="note {kind}"><strong>{title}</strong> {body}</div>'


def oim(observation: str, interpretation: str, implication: str) -> str:
    """Khối Quan sát / Diễn giải / Ý nghĩa — định dạng đề bài yêu cầu cho mỗi hình."""
    return (f'<div class="oim">'
            f'<p><span class="oim-l">Quan sát.</span> {observation}</p>'
            f'<p><span class="oim-l">Diễn giải.</span> {interpretation}</p>'
            f'<p><span class="oim-l">Ý nghĩa với học máy.</span> {implication}</p>'
            f'</div>')


def fmt(x, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def metric_rows(app: str, keys, order_by: str, reverse: bool = True):
    """Sinh các dòng bảng so sánh mô hình từ metadata, đã sắp xếp."""
    m = META[app]["test_metrics"]
    labels = META[app].get("model_labels") or {}
    items = sorted(m.items(), key=lambda kv: kv[1].get(order_by, 0), reverse=reverse)
    rows = []
    for key, vals in items:
        label = labels.get(key) or key.replace("_", " ").title()
        rows.append([label] + [fmt(vals.get(k, float("nan"))) for k in keys])
    return rows, items[0][0]


CSS = r"""
@page { size: A4; margin: 22mm 20mm 20mm 25mm; }
@page :first { margin: 0; }
* { box-sizing: border-box; }
body {
  font-family: "Times New Roman", "Tinos", Times, serif;
  font-size: 13pt; line-height: 1.62; color: #111; margin: 0;
  text-align: justify; hyphens: auto;
}
h1, h2, h3, h4 { font-family: "Times New Roman", Times, serif; text-align: left; }
h1 { font-size: 19pt; margin: 0 0 14pt; page-break-before: always; page-break-after: avoid; }
h1.nobreak { page-break-before: avoid; }
h2 { font-size: 15pt; margin: 18pt 0 8pt; page-break-after: avoid; }
h3 { font-size: 13.5pt; margin: 13pt 0 6pt; page-break-after: avoid; }
h4 { font-size: 13pt; font-style: italic; margin: 11pt 0 5pt; page-break-after: avoid; }
p { margin: 0 0 8pt; }
ul, ol { margin: 0 0 9pt; padding-left: 22pt; }
li { margin-bottom: 3pt; }
a { color: #1a4f9c; }

/* ---- Trang bìa ---- */
.cover { position: relative; width: 209mm; height: 296mm; page-break-after: always;
         overflow: hidden; background: #fff; }
/* Khung viền của ảnh nằm sát mép ảnh, nên phải chừa lề trắng — nếu vẽ tràn ra
   mép giấy thì Chrome cắt mất đường viền phải và dưới (bìa trông "lệch khung"). */
.cover img.frame { position: absolute; top: 6.5mm; left: 6.5mm; width: 197mm; height: 284mm; }
.cover .inner { position: absolute; top: 6.5mm; left: 6.5mm; width: 197mm; height: 284mm;
                padding: 20mm 18mm 10mm; text-align: center;
                display: flex; flex-direction: column; align-items: center; }
.cover .u1 { font-size: 15pt; margin-top: 6mm; font-weight: bold; line-height: 1.4; }
.cover .u2 { font-size: 14pt; font-weight: bold; margin-top: 4mm; }
.cover .u3 { font-size: 13.5pt; font-weight: bold; margin-top: 4mm; }
.cover .logo { height: 40mm; margin: 12mm 0 10mm; }
.cover .title { font-size: 22pt; font-weight: bold; letter-spacing: .5pt; }
.cover .subject { font-size: 13pt; font-weight: bold; margin-top: 5mm; line-height: 1.5; }
.cover .info { margin-top: 13mm; font-size: 13pt; text-align: left; }
.cover .info table { border: 0; width: auto; margin: 0 auto; }
.cover .info td { border: 0; padding: 2.4mm 4mm; font-size: 13pt; }
.cover .info td.k { font-weight: bold; white-space: nowrap; }
.cover .info tr { background: #fff !important; }
.cover .city { margin-top: auto; padding-bottom: 10mm; width: 100%;
               font-size: 13pt; font-weight: bold; font-style: italic; text-align: center; }

/* ---- Mục lục ---- */
.toc { page-break-after: always; }
.toc h1 { text-align: center; page-break-before: avoid; margin-bottom: 16pt; }
.toc ul { list-style: none; padding: 0; margin: 0; }
.toc li { margin: 0; }
.toc a { text-decoration: none; color: #000; display: flex; align-items: baseline; }
.toc .dots { flex: 1; border-bottom: 1px dotted #999; margin: 0 5px 3px; }
.toc .pg { min-width: 20pt; text-align: right; font-variant-numeric: tabular-nums; }
.toc .lv1 > a { font-weight: bold; margin-top: 5pt; }
.toc .lv2 { padding-left: 16pt; font-size: 12.5pt; }
.toc .lv3 { padding-left: 34pt; font-size: 12pt; }

/* ---- Hình ---- */
figure { margin: 12pt 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #d0d0d0; }
figcaption { font-size: 11.5pt; font-style: italic; margin-top: 5pt; text-align: center; color: #333; }

/* ---- Bảng ---- */
/* Ô bảng phải ngắt được chuỗi dài (đường dẫn, slug Kaggle): nếu min-content
   width của bảng vượt vùng in, Chrome thu nhỏ TOÀN BỘ tài liệu (shrink-to-fit)
   và trang bìa bị lệch khung. */
table { width: 100%; border-collapse: collapse; margin: 8pt 0 12pt;
        font-size: 11.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 4.5pt 6pt; text-align: left; vertical-align: top;
          word-break: break-word; overflow-wrap: anywhere; }
th { background: #e8eef7; font-weight: bold; }
tbody tr:nth-child(even) { background: #f7f9fc; }
table.num td:not(:first-child) { text-align: right; font-family: Consolas, monospace; font-size: 11pt; }
.tblcap { font-size: 11.5pt; font-style: italic; text-align: center; margin-top: 10pt; color: #333; }

/* ---- Mã nguồn ---- */
.codebox { border: 1px solid #c3cad6; border-radius: 4px; margin: 9pt 0 2pt;
           page-break-inside: avoid; overflow: hidden; background: #fbfcfe; }
.codehdr { background: #e8eef7; border-bottom: 1px solid #c3cad6; padding: 2.5pt 7pt;
           font-family: Consolas, monospace; font-size: 9.5pt; color: #44546a;
           text-transform: uppercase; letter-spacing: .5pt; }
pre.code { margin: 0; padding: 7pt 9pt; font-family: Consolas, "Cascadia Mono", monospace;
           font-size: 9.6pt; line-height: 1.42; white-space: pre-wrap; word-break: break-word;
           text-align: left; color: #1e293b; }
pre.code .k { color: #0b5cad; font-weight: bold; }
pre.code .s { color: #0a7a44; }
pre.code .c { color: #7a8493; font-style: italic; }
pre.code .n { color: #a03000; }
.outbox { border: 1px solid #d5d5d5; border-radius: 4px; margin: 7pt 0 2pt;
          page-break-inside: avoid; overflow: hidden; background: #fafafa; }
.outhdr { background: #ededed; border-bottom: 1px solid #d5d5d5; padding: 2.5pt 7pt;
          font-family: Consolas, monospace; font-size: 9.5pt; color: #555;
          text-transform: uppercase; letter-spacing: .5pt; }
pre.out { margin: 0; padding: 7pt 9pt; font-family: Consolas, monospace; font-size: 9.2pt;
          line-height: 1.38; white-space: pre-wrap; word-break: break-word;
          text-align: left; color: #222; }
.codecap { font-size: 11pt; font-style: italic; text-align: center; margin: 4pt 0 11pt; color: #333; }
code.inl { font-family: Consolas, monospace; font-size: 11pt; background: #f0f2f5;
           padding: 0 3px; border-radius: 3px;
           word-break: break-word; overflow-wrap: anywhere; }

/* ---- Khối nhấn ---- */
.note { border-left: 3.5px solid #2563eb; background: #eff6ff; padding: 7pt 11pt;
        margin: 9pt 0; page-break-inside: avoid; font-size: 12.5pt; }
.note.warn { border-color: #d97706; background: #fffbeb; }
.note.bad { border-color: #dc2626; background: #fef2f2; }
.note.good { border-color: #059669; background: #ecfdf5; }
.oim { border-left: 3px solid #94a3b8; padding: 2pt 0 2pt 11pt; margin: 8pt 0 12pt; font-size: 12.5pt; }
.oim p { margin: 0 0 5pt; }
.oim-l { font-weight: bold; }
.formula { text-align: center; font-size: 13.5pt; margin: 10pt 0; font-style: italic; }
.flow { text-align: center; font-weight: bold; margin: 10pt 0; font-size: 13pt;
        border: 1px solid #333; padding: 6pt 9pt; page-break-inside: avoid; }
.missing { border: 1.5px dashed #dc2626; color: #dc2626; padding: 9pt;
           text-align: center; font-size: 11.5pt; margin: 9pt 0; }
.srcline { font-size: 13pt; margin: 0 0 14pt; }
table.links td { font-size: 12.5pt; vertical-align: top; }
table.links td.lab { font-weight: bold; white-space: nowrap; width: 34%; }
table.links a { word-break: break-all; }
"""


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    return re.sub(r"-+", "-", s).strip("-") or "sec"


class Report:
    """Gom nội dung và tự dựng mục lục từ các tiêu đề đã thêm."""

    def __init__(self):
        self.parts: list[str] = []
        self.toc: list[tuple[int, str, str]] = []

    def h(self, level: int, text: str, *, nobreak: bool = False) -> None:
        anchor = f"s{len(self.toc)}-{slug(text)[:40]}"
        self.toc.append((level, text, anchor))
        cls = ' class="nobreak"' if (level == 1 and nobreak) else ""
        self.parts.append(f'<h{level} id="{anchor}"{cls}>{text}</h{level}>')

    def p(self, *chunks: str) -> None:
        for c in chunks:
            self.parts.append(c if c.lstrip().startswith("<") else f"<p>{c}</p>")

    def render_toc(self, page_numbers: dict[str, int] | None = None) -> str:
        items = []
        for level, text, anchor in self.toc:
            num = (page_numbers or {}).get(text)
            pg = f'<span class="pg">{num}</span>' if num else '<span class="pg"></span>'
            items.append(
                f'<li class="lv{level}"><a href="#{anchor}">'
                f'<span>{text}</span><span class="dots"></span>{pg}</a></li>')
        return ('<div class="toc"><h1 class="nobreak">Mục lục</h1><ul>'
                + "".join(items) + "</ul></div>")

    def cover(self) -> str:
        frame = ASSETS / "cover-frame.png"
        logo = ASSETS / "ptit-logo.png"
        frame_img = (f'<img class="frame" src="{data_uri(frame)}">' if frame.exists() else "")
        logo_img = (f'<img class="logo" src="{data_uri(logo)}">' if logo.exists() else "")
        rows = [
            ("Lớp:", STUDENT["lop"]),
            ("Họ và tên:", STUDENT["ten"]),
            ("Mã sinh viên:", STUDENT["mssv"]),
            ("Giảng viên hướng dẫn:", STUDENT["gvhd"]),
            ("Học kỳ:", STUDENT["hocky"]),
        ]
        info = "".join(f'<tr><td class="k">{k}</td><td>{v}</td></tr>' for k, v in rows)
        return f"""
<div class="cover">{frame_img}
  <div class="inner">
    <div class="u1">HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG</div>
    <div class="u2">KHOA CÔNG NGHỆ THÔNG TIN 1</div>
    <div class="u3">HỌC PHẦN PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH</div>
    {logo_img}
    <div class="title">BÁO CÁO ASSIGNMENT 02</div>
    <div class="subject">ĐỀ TÀI: TỪ BIỂU DIỄN DỮ LIỆU ĐẾN<br>HỆ THỐNG THÔNG MINH CÓ THỂ TRIỂN KHAI</div>
    <div class="info"><table>{info}</table></div>
    <div class="city">Hà Nội, 2026</div>
  </div>
</div>"""

    def build(self, page_numbers: dict[str, int] | None = None) -> str:
        return (f"<!doctype html><html lang=\"vi\"><head><meta charset=\"utf-8\">"
                f"<title>Báo cáo Assignment 02 — {STUDENT['ten']}</title>"
                f"<style>{CSS}</style></head><body>"
                + self.cover() + self.render_toc(page_numbers) + "".join(self.parts)
                + "</body></html>")


def to_pdf(html_path: Path, pdf_path: Path) -> bool:
    chrome = None
    for c in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]:
        if Path(c).exists():
            chrome = c
            break
    if chrome is None:
        chrome = shutil.which("chrome") or shutil.which("msedge")
    if chrome is None:
        print("! Không tìm thấy Chrome/Edge — bỏ qua bước xuất PDF.")
        return False

    cmd = [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           "--run-all-compositor-stages-before-draw", "--virtual-time-budget=20000",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if pdf_path.exists() and pdf_path.stat().st_size > 20_000:
        return True
    print("! Xuất PDF thất bại:", r.stderr[-800:])
    return False


def main() -> int:
    import chapters_apps
    import chapters_final
    import chapters_theory

    r = Report()
    chapters_theory.write(r)
    chapters_apps.write(r)
    chapters_final.write(r)

    import paginate

    html_path = REPORT / "Assignment_02.html"
    pdf_path = REPORT / "A02_CT_nghiand.600.pdf"
    headings = [text for _, text, _ in r.toc]

    # Lượt 1 — in để biết mục nào rơi vào trang nào.
    html_path.write_text(r.build(), encoding="utf-8")
    print(f"  lượt 1: {html_path.stat().st_size / 1e6:.1f} MB HTML …")
    if not to_pdf(html_path, pdf_path):
        return 1

    first = paginate.find_first_content_page(pdf_path)
    pages = paginate.heading_pages(pdf_path, headings, first)
    print(f"  dò được số trang cho {len(pages)}/{len(headings)} mục "
          f"(nội dung bắt đầu ở trang PDF thứ {first + 1})")

    # Lượt 2 — in lại kèm số trang trong mục lục. Số trang có thể xê dịch nếu mục
    # lục dài thêm một trang, nên kiểm tra lại và in lượt 3 khi cần.
    html_path.write_text(r.build(pages), encoding="utf-8")
    if not to_pdf(html_path, pdf_path):
        return 1
    first2 = paginate.find_first_content_page(pdf_path)
    if first2 != first:
        pages = paginate.heading_pages(pdf_path, headings, first2)
        html_path.write_text(r.build(pages), encoding="utf-8")
        if not to_pdf(html_path, pdf_path):
            return 1
        first2 = paginate.find_first_content_page(pdf_path)
        print(f"  mục lục đổi độ dài → in lại lượt 3")

    numbered = paginate.add_page_numbers(pdf_path, first2)
    print(f"✓ {html_path.name}  ({html_path.stat().st_size / 1e6:.1f} MB, {len(r.toc)} mục)")
    print(f"✓ {pdf_path.name}  ({pdf_path.stat().st_size / 1e6:.1f} MB, "
          f"{numbered} trang nội dung được đánh số)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPORT))
    raise SystemExit(main())
