"""Dựng báo cáo Assignment 04 — HTML tiếng Việt rồi in ra PDF bằng Chrome headless.

Chạy:  python "Report/Assignment 04/build_report.py"
Kết quả: Report/Assignment 04/Assignment_04.html  và  A04_CT_nghiand.600.pdf

Nguyên tắc bất di bất dịch: **mọi con số trong báo cáo được ĐỌC TỪ các tệp JSON
mà notebook sinh ra, không gõ tay**. Sửa notebook rồi chạy lại là báo cáo tự cập
nhật theo — không có chỗ nào để một con số cũ nằm lại trong văn bản.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1] / "src" / "Assignment 04"
sys.path.insert(0, str(REPORT))

import report_lib as R  # noqa: E402

FIG = REPORT / "figures"


# ---------------------------------------------------------------------------
# Gom hình từ sáu hệ thống + thư mục hình lý thuyết về một chỗ
# ---------------------------------------------------------------------------
FIG_SOURCES = [
    ("th", ROOT / "theory_figures"),
    ("cm", ROOT / "customer_comments" / "reports" / "figures"),
    ("db", ROOT / "diabetes" / "reports" / "figures"),
    ("hp", ROOT / "house_price" / "reports" / "figures"),
    ("mn", ROOT / "mnist" / "reports" / "figures"),
    ("cf", ROOT / "cifar10" / "reports" / "figures"),
    ("mv", ROOT / "mlp_vs_cnn" / "reports" / "figures"),
    ("an", ROOT / "analysis" / "reports" / "figures"),
]


def collect_figures() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    n = 0
    for prefix, src in FIG_SOURCES:
        if not src.exists():
            print(f"  ! thiếu thư mục hình: {src}")
            continue
        for png in sorted(src.glob("*.png")):
            shutil.copy2(png, FIG / f"{prefix}_{png.name}")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Nạp toàn bộ số liệu thực nghiệm
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Thiếu {path}\n→ Hãy chạy notebook tương ứng trước khi dựng báo cáo.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional(path: Path):
    """Doc tep neu co, tra None neu chua co. Dung cho ba thuc nghiem bo sung."""
    if not path.exists():
        print(f"  ! chua co {path.name}, chuong tuong ung se in o canh bao")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_all() -> dict:
    return {
        "comments": load_json(ROOT / "customer_comments" / "reports" / "metrics_customer_comments.json"),
        "diabetes": load_json(ROOT / "diabetes" / "reports" / "metrics_diabetes.json"),
        "house": load_json(ROOT / "house_price" / "reports" / "metrics_house_price.json"),
        "mnist": load_json(ROOT / "mnist" / "reports" / "metrics_mnist.json"),
        "cifar": load_json(ROOT / "cifar10" / "reports" / "metrics_cifar10.json"),
        "mlp": load_json(ROOT / "mlp_vs_cnn" / "reports" / "metrics_mlp_vs_cnn.json"),
        # Ba thuc nghiem goc o Muc 8 hop dong. Chua co thi bao cao van dung duoc,
        # chuong tuong ung se in mot o canh bao thay vi lam hong ca ban build.
        "statistical": load_optional(ROOT / "analysis" / "reports" / "metrics_statistical.json"),
        "anatomy": load_optional(ROOT / "analysis" / "reports" / "metrics_anatomy.json"),
        "ablation": load_optional(ROOT / "analysis" / "reports" / "metrics_ablation.json"),
    }


# ---------------------------------------------------------------------------
# Xuất PDF
# ---------------------------------------------------------------------------
def to_pdf(html_path: Path, pdf_path: Path) -> bool:
    chrome = None
    for c in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]:
        if Path(c).exists():
            chrome = c
            break
    if chrome is None:
        chrome = shutil.which("chrome") or shutil.which("msedge")
    if chrome is None:
        print("! Không tìm thấy Chrome/Edge — bỏ qua bước xuất PDF.")
        return False

    cmd = [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           "--run-all-compositor-stages-before-draw", "--virtual-time-budget=30000",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if pdf_path.exists() and pdf_path.stat().st_size > 20_000:
        return True
    print("! Xuất PDF thất bại:", (r.stderr or "")[-800:])
    return False


def main() -> int:
    import chapters_theory
    import chapters_1d
    import chapters_2d
    import chapters_analysis
    import chapters_final
    import paginate

    n_fig = collect_figures()
    print(f"  gom được {n_fig} hình từ bảy nguồn")

    data = load_all()
    print("  nạp số liệu: comments · diabetes · house · mnist · cifar10 · mlp_vs_cnn")

    rep = R.Report()
    chapters_theory.write(rep, data)
    chapters_1d.write(rep, data)
    chapters_2d.write(rep, data)
    chapters_analysis.write(rep, data)
    chapters_final.write(rep, data)

    html_path = REPORT / "Assignment_04.html"
    pdf_path = REPORT / "A04_CT_nghiand.600.pdf"
    headings = [text for _, text, _ in rep.toc]

    # Lượt 1 — in để biết mục nào rơi vào trang nào.
    html_path.write_text(rep.build(), encoding="utf-8")
    print(f"  lượt 1: {html_path.stat().st_size / 1e6:.1f} MB HTML …")
    if not to_pdf(html_path, pdf_path):
        return 1

    first = paginate.find_first_content_page(pdf_path, headings[0] if headings else None)
    pages = paginate.heading_pages(pdf_path, headings, first)
    print(f"  dò được số trang cho {len(pages)}/{len(headings)} mục "
          f"(nội dung bắt đầu ở trang PDF thứ {first + 1})")

    # Lượt 2 — in lại kèm số trang trong mục lục.
    html_path.write_text(rep.build(pages), encoding="utf-8")
    if not to_pdf(html_path, pdf_path):
        return 1
    first2 = paginate.find_first_content_page(pdf_path, headings[0] if headings else None)
    if first2 != first:
        pages = paginate.heading_pages(pdf_path, headings, first2)
        html_path.write_text(rep.build(pages), encoding="utf-8")
        if not to_pdf(html_path, pdf_path):
            return 1
        first2 = paginate.find_first_content_page(pdf_path, headings[0] if headings else None)
        print("  mục lục đổi độ dài → in lại lượt 3")

    numbered = paginate.add_page_numbers(pdf_path, first2)
    print(f"✓ {html_path.name}  ({html_path.stat().st_size / 1e6:.1f} MB, {len(rep.toc)} mục)")
    print(f"✓ {pdf_path.name}  ({pdf_path.stat().st_size / 1e6:.1f} MB, "
          f"{numbered} trang nội dung được đánh số)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
