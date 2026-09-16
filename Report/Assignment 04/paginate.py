"""Đánh số trang cho PDF và tìm số trang của từng mục để điền vào mục lục.

Chrome headless in PDF không hỗ trợ hộp lề `@page` nên không thể sinh số trang
bằng CSS. Vì vậy số trang được dập vào PDF sau khi in.

Số trang của từng mục được tìm bằng cách **đọc lại văn bản trong PDF đã in** rồi
dò tiêu đề, chứ không dựa vào đích đến của liên kết nội bộ — Chrome không phải
lúc nào cũng ghi liên kết nội bộ vào PDF, còn văn bản thì luôn có. Cách này chậm
hơn một chút nhưng không im lặng trả về sai.

Quy ước: trang bìa và các trang mục lục không mang số; nội dung bắt đầu từ 1.
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

# Chuỗi này chỉ xuất hiện ở thân báo cáo, không xuất hiện trong mục lục, nên nó
# đánh dấu đúng trang nội dung đầu tiên.
FIRST_CONTENT_MARKER = "Kho lưu trữ mã nguồn dự án"


def _norm(text: str) -> str:
    """Chuẩn hoá để so khớp: NFC, gộp khoảng trắng, bỏ dấu câu trang trí."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "").replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def _page_texts(reader: PdfReader) -> list[str]:
    out = []
    for page in reader.pages:
        try:
            out.append(_norm(page.extract_text() or ""))
        except Exception:
            out.append("")
    return out


def find_first_content_page(pdf_path: Path, first_heading: str | None = None) -> int:
    """Chỉ số 0-based của trang nội dung đầu tiên (ngay sau mục lục).

    Cách nhận biết đáng tin nhất: trang nội dung đầu tiên **mở đầu bằng chính
    tiêu đề chương 1**, trong khi các trang mục lục tuy cũng chứa tiêu đề đó
    nhưng luôn kèm dãy chấm dẫn và số trang, và trang mục lục đầu tiên còn mở
    đầu bằng chữ "Mục lục".

    Chỉ khi không truyền `first_heading` mới quay về cách dò theo chuỗi mốc cũ.
    """
    texts = _page_texts(PdfReader(str(pdf_path)))

    if first_heading:
        needle = _norm(re.sub(r"<[^>]+>", "", first_heading))
        if needle:
            for i, t in enumerate(texts):
                if i == 0:
                    continue
                if t.startswith(needle):      # tiêu đề nằm ngay đầu trang → thân báo cáo
                    return i

    needle = _norm(FIRST_CONTENT_MARKER)
    toc_needle = _norm("Mục lục")
    for i, t in enumerate(texts):
        if i == 0 or toc_needle in t[:200]:
            continue
        if needle in t:
            return i
    return 1


def heading_pages(pdf_path: Path, headings: list[str], first_content: int) -> dict[str, int]:
    """Số trang hiển thị của từng tiêu đề, dò từ văn bản của các trang nội dung."""
    texts = _page_texts(PdfReader(str(pdf_path)))
    result: dict[str, int] = {}
    for raw in headings:
        # Bỏ thẻ HTML còn sót trong tiêu đề trước khi so khớp.
        needle = _norm(re.sub(r"<[^>]+>", "", raw))
        if not needle:
            continue
        for i in range(first_content, len(texts)):
            if needle in texts[i]:
                result[raw] = i - first_content + 1
                break
    return result


def add_page_numbers(pdf_path: Path, first_content: int) -> int:
    """Dập số trang vào giữa lề dưới. Trả về số trang đã được đánh số."""
    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    writer = PdfWriter()

    for i, page in enumerate(pages):
        if i >= first_content:
            width = float(page.mediabox.width)
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=(width, float(page.mediabox.height)))
            c.setFont("Times-Roman", 10.5)
            c.drawCentredString(width / 2.0, 34, str(i - first_content + 1))
            c.showPage()
            c.save()
            buf.seek(0)
            page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)

    with pdf_path.open("wb") as fh:
        writer.write(fh)
    return len(pages) - first_content
