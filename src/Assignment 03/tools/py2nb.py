"""Chuyển mã nguồn dạng "percent script" thành Jupyter Notebook (.ipynb).

Định dạng đầu vào (jupytext percent-format rút gọn):

    # %% [markdown]
    # # Tiêu đề
    # Đoạn văn ...

    # %%
    print("mã Python bình thường")

Cách dùng:
    python tools/py2nb.py <nguồn.py> <đích.ipynb>

Lý do tự viết thay vì dùng jupytext: chỉ cần đúng hai loại cell, không muốn thêm
một phụ thuộc ngoài chỉ để làm một việc mười dòng.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

MARK_MD = "# %% [markdown]"
MARK_CODE = "# %%"


def split_cells(text: str) -> list[tuple[str, str]]:
    """Tách văn bản nguồn thành danh sách (loại_cell, nội_dung)."""
    cells: list[tuple[str, str]] = []
    kind: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped == MARK_MD:
            flush()
            kind, buf = "markdown", []
        elif stripped == MARK_CODE:
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def strip_comment_prefix(body: str) -> str:
    """Bỏ tiền tố '# ' của mỗi dòng trong cell markdown."""
    out = []
    for line in body.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def build(src: Path, dst: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(strip_comment_prefix(body))
        if kind == "markdown"
        else nbf.v4.new_code_cell(body)
        for kind, body in split_cells(src.read_text(encoding="utf-8"))
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.13"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, dst)
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(f"{dst.name}: {len(nb.cells)} cell ({n_md} markdown, {len(nb.cells) - n_md} code)")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
