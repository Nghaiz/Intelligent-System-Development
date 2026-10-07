"""Kiểu vẽ matplotlib dùng chung cho mọi hình dữ liệu của báo cáo (xuất PDF vector)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path(__file__).resolve().parents[3] / "Report" / "TieuLuan 01" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Bảng màu an toàn cho người mù màu (Okabe-Ito), cố định một màu cho mỗi mô hình ở mọi hình
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "yellow": "#F0E442", "gray": "#7F7F7F", "navy": "#123A75"}
SEQ = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"], C["gray"]]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.fontsize": 8, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(color=SEQ), "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("->", name)


def vn(x, d=0):
    """Định dạng số kiểu Việt: 12.345,6"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")
