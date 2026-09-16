# -*- coding: utf-8 -*-
"""
Sinh 9 hinh ly thuyet cho Chuong 1 cua bao cao Assignment 04 (CNN).

Cac hinh la so do minh hoa thuan matplotlib: khong huan luyen, khong doc du lieu.
Moi con so xuat hien trong hinh deu duoc tinh truc tiep trong script nay, nen
nguoi doc co the kiem tra lai bang tay va luon khop.

Quy uoc mau (dung thong nhat cho ca 9 hinh):
    xanh lam   -> tensor dau vao / dac trung dau vao
    ho phach   -> nhan tich chap, trong so, cua so truot
    xanh la    -> ban do dac trung dau ra / ket qua
    xam        -> chu thich, mui ten, cong thuc

Chay:  python make_theory_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------------
# Cau hinh chung theo CONTRACT.md muc 5.2
# ----------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["mathtext.fontset"] = "dejavusans"

OUT_DIR = Path(__file__).resolve().parent
DPI = 150

# Bang mau han che, dung xuyen suot ca chuong
C_IN = "#3F6493"        # xanh lam dam  - vien tensor dau vao
C_IN_L = "#DCE5F1"      # xanh lam nhat - nen o dau vao
C_K = "#C8871B"         # ho phach dam  - vien nhan / trong so
C_K_L = "#FAE7C6"       # ho phach nhat - nen o nhan
C_OUT = "#47795A"       # xanh la dam   - vien dau ra
C_OUT_L = "#DCEBE1"     # xanh la nhat  - nen o dau ra
C_TXT = "#2B2B2B"       # chu chinh
C_GREY = "#7A7A7A"      # chu phu, mui ten
C_PAD = "#F2F2F2"       # o dem 0 (padding)
C_SKIP = "#EAEAEA"      # o bi bo qua (dilation)


# ----------------------------------------------------------------------------
# Ham tien ich ve
# ----------------------------------------------------------------------------
def fmt(v):
    """Dinh dang so gon gang: so nguyen khong co .0, so thuc toi da 2 chu so."""
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return ("%.2f" % v).rstrip("0").rstrip(".")
    return str(v)


def blank(ax, xlim, ylim, equal=True):
    """Tat truc / khung: day la hinh minh hoa chu khong phai do thi."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")
    ax.set_axis_off()


def panel_tag(ax, tag, x=0.0, y=1.0):
    ax.text(x, y, tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold", color=C_TXT)


def draw_grid(ax, nrow, ncol, x0, y0, s=1.0, vals=None, face=C_IN_L, edge=C_IN,
              lw=1.1, fs=11, tc=C_TXT, face_fn=None, text_fn=None, alpha=1.0, z=2):
    """Ve luoi o vuong. (x0, y0) la goc TREN-TRAI; hang tang xuong duoi."""
    for r in range(nrow):
        for c in range(ncol):
            fc = face_fn(r, c) if face_fn is not None else face
            ax.add_patch(Rectangle((x0 + c * s, y0 - (r + 1) * s), s, s,
                                   facecolor=fc, edgecolor=edge, lw=lw,
                                   alpha=alpha, zorder=z))
            txt = None
            if text_fn is not None:
                txt = text_fn(r, c)
            elif vals is not None:
                txt = fmt(vals[r][c])
            if txt:
                ax.text(x0 + (c + 0.5) * s, y0 - (r + 0.5) * s, txt,
                        ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1)


def outline(ax, r, c, nr, nc, x0, y0, s=1.0, color=C_K, lw=2.6, fc="none",
            alpha=1.0, ls="-", z=5):
    """Khung bao quanh vung (r, c) kich thuoc nr x nc tren luoi."""
    ax.add_patch(Rectangle((x0 + c * s, y0 - (r + nr) * s), nc * s, nr * s,
                           facecolor=fc, edgecolor=color, lw=lw, alpha=alpha,
                           linestyle=ls, zorder=z))


def arrow(ax, p0, p1, color=C_GREY, lw=1.8, rad=0.0, ls="-", mut=15, z=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle="arc3,rad=%s" % rad,
                                 shrinkA=3, shrinkB=3, zorder=z))


def caption(ax, x, y, text, fs=11, color=C_TXT, weight="normal", ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            fontweight=weight, zorder=7)


def stage_box(ax, x, y, w, h, title, sub=None, fc=C_IN_L, ec=C_IN,
              fs=11, fs_sub=9, lw=1.6):
    """Hop chu nhat bo goc cho so do khoi (im2col / kien truc / lan truyen)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.10",
                                facecolor=fc, edgecolor=ec, lw=lw, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=fs, color=C_TXT, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=fs_sub, color=C_GREY, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=C_TXT, fontweight="bold", zorder=4)


def note_box(ax, x, y, text, fs=10, fc="#FBFBFB", ec=C_GREY, ha="left", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=C_TXT, zorder=8,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=fc, edgecolor=ec, lw=1.0))


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  da luu: %s" % name)


# ============================================================================
# Hình 1 — Phép tích chập 2 chiều
# ============================================================================
def fig_convolution():
    X = [[1, 2, 0, 1, 3],
         [0, 1, 2, 3, 1],
         [2, 1, 0, 1, 2],
         [1, 0, 3, 2, 0],
         [3, 2, 1, 0, 1]]
    W = [[1, 0, -1],
         [1, 0, -1],
         [1, 0, -1]]
    b = 1

    def conv_at(i, j):
        return sum(X[i + p][j + q] * W[p][q] for p in range(3) for q in range(3)) + b

    Z = [[conv_at(i, j) for j in range(3)] for i in range(3)]

    fig, ax = plt.subplots(figsize=(11.6, 8.0))
    blank(ax, (-0.7, 15.6), (-9.4, 2.2))

    # --- đầu vào 5x5, làm nổi cửa sổ 3x3 ở góc trên trái ---
    draw_grid(ax, 5, 5, 0, 0, vals=X, face_fn=lambda r, c: C_K_L if (r < 3 and c < 3) else C_IN_L)
    outline(ax, 0, 0, 3, 3, 0, 0, color=C_K, lw=3.0)
    caption(ax, 2.5, 0.45, "Đầu vào $X$ (5×5)", fs=12, weight="bold", color=C_IN)
    caption(ax, 2.5, -5.45, "vùng hổ phách: cửa sổ nhận thức hiện tại", fs=10, color=C_GREY)

    # --- nhân 3x3 ---
    caption(ax, 5.75, -2.5, "$\\circledast$", fs=22, color=C_GREY)
    draw_grid(ax, 3, 3, 6.5, -1, vals=W, face=C_K_L, edge=C_K)
    caption(ax, 8.0, -0.55, "Nhân $W$ (3×3)", fs=12, weight="bold", color=C_K)
    caption(ax, 8.0, -4.45, "hệ số lệch $b = %d$" % b, fs=10, color=C_GREY)

    caption(ax, 10.3, -2.5, "=", fs=20, color=C_GREY)

    # --- bản đồ đặc trưng đầu ra 3x3 ---
    draw_grid(ax, 3, 3, 11.4, -1, vals=Z, face=C_OUT_L, edge=C_OUT)
    outline(ax, 0, 0, 1, 1, 11.4, -1, color=C_OUT, lw=3.0)
    caption(ax, 12.9, -0.55, "Đầu ra $Z$ (3×3)", fs=12, weight="bold", color=C_OUT)
    caption(ax, 12.9, -4.45, "$H_{out} = H - K + 1 = 5 - 3 + 1 = 3$", fs=10, color=C_GREY)

    # --- mũi tên ánh xạ cửa sổ -> một ô đầu ra ---
    arrow(ax, (3.05, -0.05), (11.9, -1.35), color=C_K, lw=2.0, rad=-0.30, mut=17)
    caption(ax, 7.5, 1.85, "một cửa sổ 3×3 của $X$ → đúng một ô của $Z$",
            fs=11, color=C_K, weight="bold")

    # --- phần số học viết tường minh ---
    caption(ax, 0.0, -6.15, "Nhân từng phần tử rồi cộng dồn:", fs=11.5,
            weight="bold", color=C_TXT, ha="left")
    y = -6.85
    total = 0
    for r in range(3):
        terms = []
        for c in range(3):
            w = W[r][c]
            wtxt = str(w) if w >= 0 else "(%d)" % w
            terms.append("%d·%s" % (X[r][c], wtxt))
            total += X[r][c] * w
        prefix = "$z$  =  " if r == 0 else "       +  "
        caption(ax, 0.0, y, prefix + "  +  ".join(terms), fs=11.5, color=C_TXT, ha="left")
        y -= 0.62
    caption(ax, 0.0, y, "       +  $b$   =   %d + %d  =  %d" % (total, b, total + b),
            fs=11.5, color=C_OUT, ha="left", weight="bold")

    note_box(ax, 8.2, -7.6,
             "$z \\;=\\; \\sum_{p}\\sum_{q} X_{p,q}\\,W_{p,q} \\;+\\; b$\n"
             "Nhân trượt trên toàn ảnh với cùng một bộ trọng số\n"
             "→ chia sẻ tham số và bất biến tịnh tiến.", fs=11)

    fig.suptitle("Phép tích chập 2 chiều: cửa sổ trượt, nhân từng phần tử rồi cộng dồn",
                 fontsize=14, fontweight="bold", color=C_TXT, y=0.99)
    save(fig, "fig_th_convolution.png")


# ============================================================================
# Hình 2 — Đệm biên (padding)
# ============================================================================
def fig_padding():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))

    # ---------------- (a) valid, P = 0 ----------------
    ax = axes[0]
    blank(ax, (-0.6, 13.6), (-9.2, 1.9))
    panel_tag(ax, "(a)")
    ox = 1.5   # dịch phải để hai bảng con cùng tỉ lệ hiển thị
    draw_grid(ax, 5, 5, ox, 0, face=C_IN_L)
    outline(ax, 0, 0, 3, 3, ox, 0, color=C_K, lw=2.6)
    outline(ax, 2, 2, 3, 3, ox, 0, color=C_K, lw=2.6, ls="--")
    caption(ax, ox + 2.5, 0.5, "Đầu vào 5×5 (không đệm)", fs=12, weight="bold", color=C_IN)
    caption(ax, ox + 2.5, -5.45, "nét liền / nét đứt: vị trí cửa sổ đầu và cuối",
            fs=10, color=C_GREY)

    arrow(ax, (ox + 5.4, -2.5), (ox + 7.2, -2.5), color=C_GREY, lw=2.0)
    caption(ax, ox + 6.3, -1.95, "Conv 3×3", fs=10.5, color=C_GREY)

    draw_grid(ax, 3, 3, ox + 7.6, -1, face=C_OUT_L, edge=C_OUT)
    caption(ax, ox + 9.1, -0.5, "Đầu ra 3×3", fs=12, weight="bold", color=C_OUT)

    caption(ax, ox + 5.5, -7.4,
            "Đệm hợp lệ: $P = 0$\n$H_{out} = H + 2P - K + 1 = 5 + 0 - 3 + 1 = 3$\n"
            "Biên bị bào mòn 2 ô sau mỗi tầng.", fs=11.5, color=C_TXT)

    # ---------------- (b) same, P = 1 ----------------
    ax = axes[1]
    blank(ax, (-0.6, 13.6), (-9.2, 1.9))
    panel_tag(ax, "(b)")
    draw_grid(ax, 7, 7, 0, 0,
              face_fn=lambda r, c: C_IN_L if (1 <= r <= 5 and 1 <= c <= 5) else C_PAD,
              edge=C_IN,
              text_fn=lambda r, c: "" if (1 <= r <= 5 and 1 <= c <= 5) else "0")
    outline(ax, 1, 1, 5, 5, 0, 0, color=C_IN, lw=2.4)
    outline(ax, 0, 0, 3, 3, 0, 0, color=C_K, lw=2.6)
    caption(ax, 3.5, 0.5, "Đầu vào 5×5 + viền 0 → 7×7", fs=12, weight="bold", color=C_IN)
    caption(ax, 3.5, -7.45, "ô xám nhạt: giá trị 0 được đệm thêm", fs=10, color=C_GREY)

    arrow(ax, (7.4, -3.5), (9.0, -3.5), color=C_GREY, lw=2.0)
    caption(ax, 8.2, -2.95, "Conv 3×3", fs=10.5, color=C_GREY)

    draw_grid(ax, 5, 5, 9.4, -1, face=C_OUT_L, edge=C_OUT)
    caption(ax, 11.9, -0.5, "Đầu ra 5×5", fs=12, weight="bold", color=C_OUT)

    caption(ax, 7.0, -8.3,
            "Đệm đồng dạng: $P = 1$    ·    $H_{out} = 5 + 2 - 3 + 1 = 5$\n"
            "Kích thước không đổi → xếp sâu tuỳ ý.", fs=11.5, color=C_TXT)

    fig.suptitle("Đệm biên quyết định kích thước bản đồ đặc trưng đầu ra",
                 fontsize=14, fontweight="bold", color=C_TXT, y=1.0)
    fig.tight_layout()
    save(fig, "fig_th_padding.png")


# ============================================================================
# Hình 3 — Bước trượt (stride)
# ============================================================================
def _stride_panel(ax, stride, tag):
    positions = list(range(0, 5 - 3 + 1, stride))     # vị trí cột hợp lệ
    n_out = (5 - 3) // stride + 1

    blank(ax, (-0.6, 11.8), (-8.4, 1.9))
    panel_tag(ax, tag)

    draw_grid(ax, 5, 5, 0, 0, face=C_IN_L)
    caption(ax, 2.5, 0.5, "Đầu vào 5×5", fs=12, weight="bold", color=C_IN)

    # các vị trí liên tiếp của nhân trên hàng đầu, vẽ chồng mờ
    for k, c in enumerate(positions):
        outline(ax, 0, c, 3, 3, 0, 0, color=C_K, lw=2.2, fc=C_K, alpha=0.20, z=4)
        outline(ax, 0, c, 3, 3, 0, 0, color=C_K, lw=1.8, alpha=0.85, z=5)
        ax.text(c + 1.5, -1.5, str(k + 1), ha="center", va="center",
                fontsize=15, fontweight="bold", color=C_K, zorder=7)

    caption(ax, 2.5, -5.55, "vị trí nhân liên tiếp trên hàng đầu (%d vị trí)" % len(positions),
            fs=10.5, color=C_GREY)

    arrow(ax, (5.4, -2.5), (7.0, -2.5), color=C_GREY, lw=2.0)
    caption(ax, 6.2, -1.95, "$S = %d$" % stride, fs=11, color=C_GREY)

    draw_grid(ax, n_out, n_out, 7.4, -1, face=C_OUT_L, edge=C_OUT)
    caption(ax, 7.4 + n_out / 2, -0.5, "Đầu ra %d×%d" % (n_out, n_out),
            fs=12, weight="bold", color=C_OUT)

    caption(ax, 5.5, -7.1,
            "Bước trượt $S = %d$:  $H_{out} = \\left\\lfloor \\dfrac{H - K}{S} \\right\\rfloor + 1"
            " = \\left\\lfloor \\dfrac{5 - 3}{%d} \\right\\rfloor + 1 = %d$"
            % (stride, stride, n_out), fs=12, color=C_TXT)


def fig_stride():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.4))
    _stride_panel(axes[0], 1, "(a)")
    _stride_panel(axes[1], 2, "(b)")
    axes[0].text(5.5, -8.1, "Trượt sát nhau → chồng lấn nhiều, giữ chi tiết.",
                 ha="center", va="center", fontsize=11, color=C_GREY)
    axes[1].text(5.5, -8.1, "Trượt cách quãng → giảm mẫu, chi phí giảm ~4 lần.",
                 ha="center", va="center", fontsize=11, color=C_GREY)
    fig.suptitle("Bước trượt điều khiển mức chồng lấn và kích thước đầu ra",
                 fontsize=14, fontweight="bold", color=C_TXT, y=1.0)
    fig.tight_layout()
    save(fig, "fig_th_stride.png")


# ============================================================================
# Hình 4 — Giãn nở nhân (dilation)
# ============================================================================
def _dilation_panel(ax, d, tag):
    rows = [0 + d * i for i in range(3)]
    cols = [0 + d * j for j in range(3)]
    span = d * (3 - 1) + 1
    active = {(r, c): (i, j) for i, r in enumerate(rows) for j, c in enumerate(cols)}

    blank(ax, (-0.6, 5.8), (-8.2, 1.9))
    panel_tag(ax, tag)

    def face_fn(r, c):
        if (r, c) in active:
            return C_K_L
        if r <= max(rows) and c <= max(cols):
            return C_SKIP
        return C_IN_L

    def text_fn(r, c):
        if (r, c) in active:
            i, j = active[(r, c)]
            return "$w_{%d%d}$" % (i + 1, j + 1)
        if r <= max(rows) and c <= max(cols):
            return "×"
        return ""

    draw_grid(ax, 5, 5, 0, 0, face_fn=face_fn, text_fn=text_fn, fs=10, edge=C_IN)
    outline(ax, 0, 0, span, span, 0, 0, color=C_K, lw=2.6, ls="--")

    caption(ax, 2.5, 0.5, "Giãn nở $d = %d$  →  tầm phủ %d×%d" % (d, span, span),
            fs=12, weight="bold", color=C_K)
    legend = "ô hổ phách: 9 trọng số" if d == 1 else "ô hổ phách: 9 trọng số  ·  dấu × : ô bị bỏ qua"
    caption(ax, 2.5, -5.55, legend, fs=10.5, color=C_GREY)
    caption(ax, 2.5, -6.7,
            "$K' = d\\,(K-1) + 1 = %d\\cdot 2 + 1 = %d$\nSố trọng số vẫn là $3\\times 3 = 9$"
            % (d, span), fs=11.5, color=C_TXT)


def fig_dilation():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.4))
    _dilation_panel(axes[0], 1, "(a)")
    _dilation_panel(axes[1], 2, "(b)")
    axes[0].text(2.5, -7.9, "Nhân 3×3 chuẩn: các ô liền kề.",
                 ha="center", va="center", fontsize=11, color=C_GREY)
    axes[1].text(2.5, -7.9, "Nhân giãn nở: phủ rộng hơn mà không thêm tham số.",
                 ha="center", va="center", fontsize=11, color=C_GREY)
    fig.suptitle("Giãn nở mở rộng vùng nhận thức với cùng số trọng số",
                 fontsize=14, fontweight="bold", color=C_TXT, y=1.0)
    fig.tight_layout()
    save(fig, "fig_th_dilation.png")


# ============================================================================
# Hình 5 — Vùng nhận thức của hai tầng 3×3 xếp chồng
# ============================================================================
def fig_receptive_field():
    fig, ax = plt.subplots(figsize=(12.4, 7.4))
    blank(ax, (-0.8, 17.2), (-9.4, 2.9))

    # tầng 0: ảnh vào 5x5 — toàn bộ là vùng nhận thức hiệu dụng
    draw_grid(ax, 5, 5, 0, 0, face=C_IN_L, edge=C_IN)
    outline(ax, 0, 0, 5, 5, 0, 0, color=C_IN, lw=3.0)
    outline(ax, 1, 1, 3, 3, 0, 0, color=C_K, lw=2.4, ls="--")
    caption(ax, 2.5, 0.6, "Ảnh vào (5×5)", fs=12.5, weight="bold", color=C_IN)
    caption(ax, 2.5, -5.55, "vùng nhận thức hiệu dụng 5×5", fs=10.5, color=C_IN)

    arrow(ax, (5.4, -2.5), (7.4, -2.5), color=C_GREY, lw=2.2)
    caption(ax, 6.4, -1.9, "Conv 3×3", fs=11, color=C_GREY, weight="bold")
    caption(ax, 6.4, -3.15, "$P=0$", fs=10, color=C_GREY)

    # tầng 1: bản đồ đặc trưng 3x3
    draw_grid(ax, 3, 3, 7.8, -1, face=C_OUT_L, edge=C_OUT)
    outline(ax, 0, 0, 3, 3, 7.8, -1, color=C_K, lw=2.6)
    caption(ax, 9.3, -0.4, "Đặc trưng tầng 1 (3×3)", fs=12.5, weight="bold", color=C_OUT)
    caption(ax, 9.3, -4.55, "mỗi ô nhìn thấy 3×3 của ảnh vào", fs=10.5, color=C_GREY)

    arrow(ax, (11.0, -2.5), (13.0, -2.5), color=C_GREY, lw=2.2)
    caption(ax, 12.0, -1.9, "Conv 3×3", fs=11, color=C_GREY, weight="bold")
    caption(ax, 12.0, -3.15, "$P=0$", fs=10, color=C_GREY)

    # tầng 2: một ô duy nhất
    draw_grid(ax, 1, 1, 13.4, -2, face=C_OUT_L, edge=C_OUT, lw=2.4)
    caption(ax, 13.9, -3.8, "Tầng 2 (1×1)\nmột ô duy nhất", fs=12, weight="bold", color=C_OUT)

    # đường nối thể hiện gộp vùng nhận thức
    arrow(ax, (13.9, -1.95), (2.5, 0.95), color=C_K, lw=2.0, rad=0.26, mut=16)
    caption(ax, 8.4, 2.5, "một ô ở tầng 2 tổng hợp trọn vẹn một vùng 5×5 của ảnh vào",
            fs=11.5, color=C_K, weight="bold")

    note_box(ax, 0.0, -7.6,
             "Vùng nhận thức:  $R_2 = R_1 + (K-1) = 3 + 2 = 5$\n"
             "Tham số hai tầng 3×3:  $2\\cdot(3\\cdot 3\\cdot C^2) = 18C^2$\n"
             "Tham số một tầng 5×5:  $5\\cdot 5\\cdot C^2 = 25C^2$\n"
             "Tiết kiệm:  $(25-18)/25 = 28\\%$  — cùng tầm nhìn, ít tham số hơn "
             "và thêm một lần phi tuyến.", fs=11.5, ha="left")

    fig.suptitle("Hai tầng 3×3 xếp chồng nhìn xa bằng một tầng 5×5 nhưng rẻ hơn 28%",
                 fontsize=14, fontweight="bold", color=C_TXT, y=1.0)
    save(fig, "fig_th_receptive_field.png")


# ============================================================================
# Hình 6 — Các phép gộp (pooling)
# ============================================================================
def fig_pooling():
    M = [[1, 3, 2, 4],
         [5, 6, 1, 2],
         [7, 2, 3, 8],
         [1, 4, 6, 5]]

    def block(i, j):
        return [M[2 * i][2 * j], M[2 * i][2 * j + 1],
                M[2 * i + 1][2 * j], M[2 * i + 1][2 * j + 1]]

    mx = [[max(block(i, j)) for j in range(2)] for i in range(2)]
    av = [[sum(block(i, j)) / 4.0 for j in range(2)] for i in range(2)]
    gap = sum(sum(row) for row in M) / 16.0

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.2))

    # ---------------- (a) gộp cực đại ----------------
    ax = axes[0]
    blank(ax, (-0.6, 9.4), (-8.6, 1.9))
    panel_tag(ax, "(a)")
    draw_grid(ax, 4, 4, 0, 0, vals=M, face=C_IN_L, edge=C_IN)
    for i in range(2):
        for j in range(2):
            outline(ax, 2 * i, 2 * j, 2, 2, 0, 0, color=C_K, lw=2.4)
    caption(ax, 2.0, 0.5, "Gộp cực đại 2×2, $S=2$", fs=12.5, weight="bold", color=C_K)
    arrow(ax, (4.4, -2.0), (5.8, -2.0), color=C_GREY, lw=2.0)
    draw_grid(ax, 2, 2, 6.2, -1.0, vals=mx, face=C_OUT_L, edge=C_OUT, fs=13)
    caption(ax, 7.2, -0.45, "4×4 → 2×2", fs=11.5, weight="bold", color=C_OUT)
    y = -5.2
    for i in range(2):
        for j in range(2):
            caption(ax, 0.0, y, "max(%s) = %s" % (", ".join(fmt(v) for v in block(i, j)),
                                                  fmt(mx[i][j])),
                    fs=11.5, color=C_TXT, ha="left")
            y -= 0.68
    caption(ax, 0.0, y - 0.25, "Giữ kích thích mạnh nhất → bất biến dịch chuyển nhỏ.",
            fs=10.5, color=C_GREY, ha="left")

    # ---------------- (b) gộp trung bình ----------------
    ax = axes[1]
    blank(ax, (-0.6, 9.4), (-8.6, 1.9))
    panel_tag(ax, "(b)")
    draw_grid(ax, 4, 4, 0, 0, vals=M, face=C_IN_L, edge=C_IN)
    for i in range(2):
        for j in range(2):
            outline(ax, 2 * i, 2 * j, 2, 2, 0, 0, color=C_K, lw=2.4)
    caption(ax, 2.0, 0.5, "Gộp trung bình 2×2, $S=2$", fs=12.5, weight="bold", color=C_K)
    arrow(ax, (4.4, -2.0), (5.8, -2.0), color=C_GREY, lw=2.0)
    draw_grid(ax, 2, 2, 6.2, -1.0, vals=av, face=C_OUT_L, edge=C_OUT, fs=12)
    caption(ax, 7.2, -0.45, "4×4 → 2×2", fs=11.5, weight="bold", color=C_OUT)
    y = -5.2
    for i in range(2):
        for j in range(2):
            bl = block(i, j)
            caption(ax, 0.0, y, "(%s) / 4 = %s" % (" + ".join(fmt(v) for v in bl),
                                                   fmt(av[i][j])),
                    fs=11.5, color=C_TXT, ha="left")
            y -= 0.68
    caption(ax, 0.0, y - 0.25, "Làm trơn toàn vùng → giữ nền, giảm nhiễu.",
            fs=10.5, color=C_GREY, ha="left")

    # ---------------- (c) gộp trung bình toàn cục ----------------
    ax = axes[2]
    blank(ax, (-0.6, 9.4), (-8.6, 1.9))
    panel_tag(ax, "(c)")
    draw_grid(ax, 4, 4, 0, 0, vals=M, face=C_IN_L, edge=C_IN)
    outline(ax, 0, 0, 4, 4, 0, 0, color=C_K, lw=2.8)
    caption(ax, 2.0, 0.5, "Gộp trung bình toàn cục", fs=12.5, weight="bold", color=C_K)
    caption(ax, 2.0, -4.55, "bản đồ $H\\times W$ (ở đây 4×4)", fs=10.5, color=C_GREY)
    arrow(ax, (4.4, -2.0), (6.0, -2.0), color=C_GREY, lw=2.0)
    draw_grid(ax, 1, 1, 6.4, -1.5, vals=[[gap]], face=C_OUT_L, edge=C_OUT, lw=2.4, fs=14)
    caption(ax, 6.9, -0.95, "một số vô hướng", fs=11.5, weight="bold", color=C_OUT)
    caption(ax, 0.0, -5.85,
            "$\\dfrac{1}{H\\,W}\\sum_{h}\\sum_{w} A_{h,w} = \\dfrac{%d}{16} = %s$"
            % (int(sum(sum(r) for r in M)), fmt(gap)), fs=12.5, color=C_TXT, ha="left")
    caption(ax, 0.0, -7.35,
            "Mỗi kênh thu về một số → thay lớp Flatten\nkhổng lồ, giảm mạnh số tham số.",
            fs=10.5, color=C_GREY, ha="left")

    fig.suptitle("Ba phép gộp: cực đại, trung bình và trung bình toàn cục",
                 fontsize=14, fontweight="bold", color=C_TXT, y=1.0)
    fig.tight_layout()
    save(fig, "fig_th_pooling.png")


# ============================================================================
# Hình 7 — Đường ống im2col
# ============================================================================
def fig_im2col():
    N, C, H, Wd = 1, 3, 32, 32
    K, P, S, F = 3, 1, 1, 8
    H_out = (H + 2 * P - K) // S + 1
    W_out = (Wd + 2 * P - K) // S + 1
    rows = C * K * K            # 27
    cols = N * H_out * W_out    # 1024

    fig, ax = plt.subplots(figsize=(13.4, 5.6))
    blank(ax, (-0.6, 16.4), (-2.9, 5.0), equal=False)

    w, h, pitch = 2.5, 1.9, 3.35
    xs = [i * pitch for i in range(5)]

    stage_box(ax, xs[0], 0, w, h, "Tensor vào $X$",
              "$(N, C, H, W)$\n$(%d,\\,%d,\\,%d,\\,%d)$" % (N, C, H, Wd),
              fc=C_IN_L, ec=C_IN, fs=11.5, fs_sub=10)
    stage_box(ax, xs[1], 0, w, h, "Đệm & trải cửa sổ",
              "im2col,  $P=%d$,  $S=%d$" % (P, S),
              fc="#FFFFFF", ec=C_GREY, fs=11.5, fs_sub=10)
    stage_box(ax, xs[2], 0, w, h, "Ma trận $X_{col}$",
              "$(C\\!\\cdot\\!K^2,\\; N\\!\\cdot\\!H_{out}\\!\\cdot\\!W_{out})$\n"
              "$(%d,\\, %d)$" % (rows, cols),
              fc=C_IN_L, ec=C_IN, fs=11.5, fs_sub=10)
    stage_box(ax, xs[3], 0, w, h, "GEMM",
              "$Z_{col} = W_{row} X_{col} + b$\n$(%d,\\, %d)$" % (F, cols),
              fc=C_OUT_L, ec=C_OUT, fs=11.5, fs_sub=10)
    stage_box(ax, xs[4], 0, w, h, "Định dạng lại $Z$",
              "$(N, F, H_{out}, W_{out})$\n$(%d,\\,%d,\\,%d,\\,%d)$"
              % (N, F, H_out, W_out),
              fc=C_OUT_L, ec=C_OUT, fs=11.5, fs_sub=10)

    for i in range(4):
        arrow(ax, (xs[i] + w, h / 2), (xs[i + 1], h / 2), color=C_GREY, lw=2.0, mut=16)

    # nhánh trọng số đi từ trên xuống khối GEMM
    stage_box(ax, xs[3], 2.9, w, h, "Trọng số $W_{row}$",
              "$(F,\\; C\\!\\cdot\\!K^2)$\n$(%d,\\, %d)$" % (F, rows),
              fc=C_K_L, ec=C_K, fs=11.5, fs_sub=10)
    arrow(ax, (xs[3] + w / 2, 2.9), (xs[3] + w / 2, h), color=C_K, lw=2.0, mut=16)

    caption(ax, xs[1] + w / 2, 2.55,
            "mỗi cột của $X_{col}$ = một cửa sổ $C\\times K\\times K$ đã duỗi thẳng",
            fs=10.5, color=C_GREY)
    arrow(ax, (xs[1] + w / 2, 2.25), (xs[2] + w / 2, h + 0.1),
          color=C_GREY, lw=1.3, ls="--", mut=12)

    note_box(ax, 0.0, -1.85,
             "im2col quy tích chập về đúng MỘT phép nhân ma trận: vector hoá 100%%, "
             "không còn vòng lặp theo từng điểm ảnh, tận dụng trọn vẹn BLAS của NumPy.\n"
             "Cái giá phải trả là bộ nhớ — mỗi điểm ảnh bị sao chép tối đa $K^2 = %d$ lần "
             "trong $X_{col}$." % (K * K), fs=11, ha="left")

    fig.suptitle("Đường ống im2col: đưa tích chập về một phép nhân ma trận",
                 fontsize=14, fontweight="bold", color=C_TXT, y=1.0)
    save(fig, "fig_th_im2col.png")


# ============================================================================
# Hình 8 — Kiến trúc CNN đầu-cuối như một hợp thành hàm
# ============================================================================
def fig_architecture():
    # Cấu hình minh hoạ: ảnh MNIST 28×28, Conv(32) rồi Conv(64), P = 1
    stages_top = [
        ("$X$ — ảnh vào", "$(1,\\,1,\\,28,\\,28)$", C_IN_L, C_IN),
        ("Conv 3×3 · 32", "$P=1$\n$(1,\\,32,\\,28,\\,28)$", C_K_L, C_K),
        ("ReLU", "$(1,\\,32,\\,28,\\,28)$", "#FFFFFF", C_GREY),
        ("Conv 3×3 · 64", "$P=1$\n$(1,\\,64,\\,28,\\,28)$", C_K_L, C_K),
        ("ReLU", "$(1,\\,64,\\,28,\\,28)$", "#FFFFFF", C_GREY),
    ]
    stages_bot = [
        ("MaxPool 2×2", "$(1,\\,64,\\,14,\\,14)$", C_K_L, C_K),
        ("Flatten", "$(1,\\,12544)$", "#FFFFFF", C_GREY),
        ("Dense 10", "$(1,\\,10)$", C_K_L, C_K),
        ("$\\hat{y}$ — Softmax", "$(1,\\,10)$", C_OUT_L, C_OUT),
    ]

    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    blank(ax, (-1.0, 13.6), (-3.4, 4.3))

    w, h, pitch = 1.95, 1.5, 2.5
    y_top, y_bot = 1.7, -1.5

    for i, (t, s, fc, ec) in enumerate(stages_top):
        stage_box(ax, i * pitch, y_top, w, h, t, s, fc=fc, ec=ec, fs=11, fs_sub=9)
        if i:
            arrow(ax, ((i - 1) * pitch + w, y_top + h / 2), (i * pitch, y_top + h / 2),
                  color=C_GREY, lw=1.8, mut=14)
    for i, (t, s, fc, ec) in enumerate(stages_bot):
        stage_box(ax, i * pitch, y_bot, w, h, t, s, fc=fc, ec=ec, fs=11, fs_sub=9)
        if i:
            arrow(ax, ((i - 1) * pitch + w, y_bot + h / 2), (i * pitch, y_bot + h / 2),
                  color=C_GREY, lw=1.8, mut=14)

    # khuỷu nối cuối hàng trên xuống đầu hàng dưới
    x_end = 4 * pitch + w
    elbow_x, elbow_y = x_end + 0.75, y_bot + h + 0.85
    ax.plot([x_end, elbow_x, elbow_x, w / 2], [y_top + h / 2, y_top + h / 2, elbow_y, elbow_y],
            color=C_GREY, lw=1.8, solid_joinstyle="round", zorder=2)
    arrow(ax, (w / 2, elbow_y), (w / 2, y_bot + h), color=C_GREY, lw=1.8, mut=14)

    caption(ax, 6.0, 3.9,
            "$\\hat{y} \\;=\\; f_L \\circ f_{L-1} \\circ \\cdots \\circ f_1 (X)$",
            fs=16, color=C_TXT)
    caption(ax, 6.0, 3.35,
            "mỗi khối là một hàm khả vi; chồng chúng lại thành một hợp thành duy nhất",
            fs=11, color=C_GREY)

    note_box(ax, -0.6, -3.0,
             "Đặc trưng nông (cạnh, góc) → đặc trưng sâu (bộ phận, vật thể).  "
             "Conv giữ cấu trúc không gian, Pool giảm mẫu, Flatten + Dense mới ra quyết định.\n"
             "Số tham số tập trung ở tầng Dense: $12544 \\times 10 + 10 = 125\\,450$ trọng số.",
             fs=10.5, ha="left")

    fig.suptitle("Kiến trúc CNN đầu-cuối và hình dạng tensor sau từng tầng",
                 fontsize=14, fontweight="bold", color=C_TXT, y=1.0)
    save(fig, "fig_th_architecture.png")


# ============================================================================
# Hình 9 — Lan truyền thuận và lan truyền ngược
# ============================================================================
def fig_backprop():
    stages = [
        ("$X$", "ảnh vào", C_IN_L, C_IN),
        ("Conv", "$Z = W_{row}X_{col}+b$", C_K_L, C_K),
        ("ReLU", "$A = \\max(0, Z)$", "#FFFFFF", C_GREY),
        ("MaxPool", "$2\\times 2$", C_K_L, C_K),
        ("Flatten\n+ Dense", "$\\hat{y}$", C_K_L, C_K),
        ("Mất mát $L$", "Cross-Entropy", C_OUT_L, C_OUT),
    ]
    grads = ["$\\partial L/\\partial X$", "$\\partial L/\\partial Z$",
             "$\\partial L/\\partial A$", "$\\partial L/\\partial P$",
             "$\\partial L/\\partial \\hat{y}$"]

    fig, ax = plt.subplots(figsize=(13.2, 9.6))
    blank(ax, (-0.6, 14.2), (-7.6, 3.2))

    w, h, pitch = 1.8, 1.5, 2.3
    y0 = 0.0
    for i, (t, s, fc, ec) in enumerate(stages):
        stage_box(ax, i * pitch, y0, w, h, t, s, fc=fc, ec=ec, fs=11.5, fs_sub=9)

    # --- lan truyền thuận (trên) ---
    for i in range(len(stages) - 1):
        arrow(ax, (i * pitch + w, y0 + h * 0.62), ((i + 1) * pitch, y0 + h * 0.62),
              color=C_OUT, lw=2.0, mut=15)
    caption(ax, 6.2, 2.75, "Lan truyền thuận  →  tính $\\hat{y}$ và $L$",
            fs=13, color=C_OUT, weight="bold")

    # --- lan truyền ngược (dưới) ---
    for i in range(len(stages) - 1):
        arrow(ax, ((i + 1) * pitch, -0.95), (i * pitch + w, -0.95),
              color=C_K, lw=2.0, mut=15)
        caption(ax, i * pitch + w / 2 + 0.55, -1.5, grads[i], fs=10, color=C_K)
    caption(ax, 6.2, -0.35, "←  Lan truyền ngược: quy tắc chuỗi truyền gradient về đầu vào",
            fs=13, color=C_K, weight="bold")

    # --- chú thích 1: gradient của trọng số tầng tích chập ---
    note_box(ax, -0.4, -3.3,
             "Tầng tích chập (dạng im2col)\n"
             "$\\dfrac{\\partial L}{\\partial W} = X_{col}^{\\top}\\,"
             "\\dfrac{\\partial L}{\\partial Z_{col}}$\n"
             "$\\dfrac{\\partial L}{\\partial X_{col}} = W_{row}^{\\top}\\,"
             "\\dfrac{\\partial L}{\\partial Z_{col}}$\n"
             "rồi col2im để gộp về đúng hình dạng ảnh.", fs=11, ha="left")

    # --- chú thích 2: định tuyến argmax của gộp cực đại ---
    cx = 6.1
    caption(ax, cx + 1.65, -2.85, "Gộp cực đại: gradient chỉ về ô thắng",
            fs=11.5, color=C_TXT, weight="bold")
    fwd = [[1, 3], [5, 6]]
    draw_grid(ax, 2, 2, cx, -3.3, s=0.62, vals=fwd, face=C_IN_L, edge=C_IN, fs=11)
    outline(ax, 1, 1, 1, 1, cx, -3.3, s=0.62, color=C_K, lw=2.4)
    caption(ax, cx + 0.62, -4.85, "thuận: max = 6\n(ghi nhớ argmax)", fs=10, color=C_GREY)
    arrow(ax, (cx + 1.45, -3.92), (cx + 2.25, -3.92), color=C_K, lw=1.8, mut=14)
    draw_grid(ax, 2, 2, cx + 2.35, -3.3, s=0.62, face=C_OUT_L, edge=C_OUT, fs=11,
              text_fn=lambda r, c: "$\\delta$" if (r, c) == (1, 1) else "0")
    caption(ax, cx + 2.97, -4.85, "ngược: $\\delta$ về ô thắng,\ncác ô khác nhận 0",
            fs=10, color=C_GREY)

    # --- chú thích 3: bước cập nhật tham số ---
    note_box(ax, 10.6, -3.5,
             "Cập nhật tham số\n"
             "$\\theta \\;\\leftarrow\\; \\theta - \\eta\\,\\nabla_{\\theta} L$\n"
             "$\\eta$: tốc độ học\n"
             "Dùng chung cho mọi $W$, $b$ của mạng.", fs=11, ha="left")

    caption(ax, 6.2, -6.6,
            "ReLU chặn gradient ở các vị trí $Z \\leq 0$; gộp cực đại chặn ở mọi ô không thắng — "
            "nhờ vậy gradient đi đúng đường đã kích hoạt lúc lan truyền thuận.",
            fs=11, color=C_GREY)

    fig.suptitle("Lan truyền thuận và lan truyền ngược qua cùng một chồng tầng",
                 fontsize=14, fontweight="bold", color=C_TXT, y=1.0)
    save(fig, "fig_th_backprop.png")


# ============================================================================
def main():
    print("Đang dựng 9 hình lý thuyết vào: %s" % OUT_DIR)
    fig_convolution()
    fig_padding()
    fig_stride()
    fig_dilation()
    fig_receptive_field()
    fig_pooling()
    fig_im2col()
    fig_architecture()
    fig_backprop()
    print("Hoàn tất 9/9 hình.")


if __name__ == "__main__":
    main()
