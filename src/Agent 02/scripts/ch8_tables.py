"""Bảng Chương 8: chỉ số phân loại ý định trên tập test và nhật ký thông điệp demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ag2.common import GEN, load_json  # noqa: E402
from ag2.texfmt import num, table, tex_escape  # noqa: E402

NAMES = {"tim_san_pham": "Tìm SP", "so_sanh": "So sánh", "hoi_ton_kho": "Tồn kho",
         "hoi_gia": "Giá/KM", "chao_hoi": "Chào hỏi", "khac": "Khác"}


def report(d):
    """P/R/F1 từng ý định (nhãn ngắn trùng trục ma trận nhầm lẫn) và dòng tổng accuracy / macro-F1."""
    t = d["test"]
    rows = [[NAMES[k], num(v["precision"], 3), num(v["recall"], 3), num(v["f1"], 3)]
            for k, v in t["per_class"].items()]
    rows += ["\\midrule", ["\\textbf{Acc / Macro-F1}", "", num(t["accuracy"], 3), num(t["macro_f1"], 3)]]
    table("ch8_report", "lrrr", ["Ý định", "P", "R", "F1"], rows)


def demo(d, which=(0,)):
    """Nhật ký thông điệp của yêu cầu demo -> generated/ch8_demo.tex (chèn trong hộp ketqua)."""
    out = []
    for i in which:
        r = d["demo"]["rows"][i]
        out.append(f"\\textbf{{>>> {tex_escape(r['query'])}}}")
        out += [tex_escape(line[:86]).replace(" ", "~") for line in r["log"]]
        out.append(f"\\textit{{=> {tex_escape(r['status'])} {tex_escape(', '.join(r['items']))} "
                   f"({r['messages']} thông điệp, {num(r['latency_ms'], 1)} ms)}}")
    (GEN / "ch8_demo.tex").write_text("\\par\n".join(out) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    data = load_json("ch8_shop")
    report(data)

    demo(data)
