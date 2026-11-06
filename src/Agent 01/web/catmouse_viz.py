"""Vẽ trò chơi Mèo bắt chuột: khung hình Pillow (emoji màu), GIF, đường cong học, bản đồ nhiệt max Q."""
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from catmouse import ACTIONS, HOLE, WALL

CELL = 64
BAR = 44
EMOJI_FONT = Path(r"C:\Windows\Fonts\seguiemj.ttf")
TEXT_FONTS = [Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")]
COL = {"floor": "#F8FAFC", "floor2": "#EEF2F7", "wall": "#334155", "hole": "#D6B98C", "hole_in": "#3F2A14",
       "grid": "#CBD5E1", "mouse": "#9CA3AF", "cat": "#F97316", "cheese": "#FACC15", "bar": "#0F172A"}
OUTCOME_VI = {"win": "THẮNG: chuột ăn hết phô mai!", "caught": "THUA: chuột bị mèo bắt!",
              "timeout": "HẾT GIỜ", "": ""}


def _font(paths, size):
    for p in paths:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


_EMOJI = _font([EMOJI_FONT], int(CELL * 0.62)) if EMOJI_FONT.exists() else None
_TEXT = _font(TEXT_FONTS, 18)


def _emoji(d, xy, ch, fallback_color):
    x, y = xy
    if _EMOJI is not None:
        d.text((x + CELL / 2, y + CELL / 2), ch, font=_EMOJI, embedded_color=True, anchor="mm")
    else:
        d.ellipse([x + 12, y + 12, x + CELL - 12, y + CELL - 12], fill=fallback_color, outline="#111827")


def render_frame(env, m, c, mask, caption=""):
    """Một khung hình: lưới, tường, hang, phô mai còn lại, chuột, mèo và dòng trạng thái."""
    w, h = env.n_cols * CELL, env.n_rows * CELL + BAR
    img = Image.new("RGB", (w, h), COL["floor"])
    d = ImageDraw.Draw(img)
    for r in range(env.n_rows):
        for cc in range(env.n_cols):
            x, y = cc * CELL, r * CELL
            ch = env.layout[r][cc]
            fill = COL["wall"] if ch == WALL else (COL["floor"] if (r + cc) % 2 == 0 else COL["floor2"])
            if ch == HOLE:
                fill = COL["hole"]
            d.rectangle([x, y, x + CELL, y + CELL], fill=fill, outline=COL["grid"])
            if ch == HOLE:
                d.ellipse([x + 10, y + 20, x + CELL - 10, y + CELL - 8], fill=COL["hole_in"])
    for k, cell in enumerate(env.cheeses):
        if mask & (1 << k):
            r, cc = env.cells[cell]
            _emoji(d, (cc * CELL, r * CELL), "🧀", COL["cheese"])
    (mr, mc), (cr, ccat) = env.cells[m], env.cells[c]
    if m == c:
        d.rectangle([mc * CELL + 2, mr * CELL + 2, (mc + 1) * CELL - 2, (mr + 1) * CELL - 2], outline="#DC2626", width=4)
    _emoji(d, (mc * CELL, mr * CELL), "🐭", COL["mouse"])
    _emoji(d, (ccat * CELL, cr * CELL), "🐱", COL["cat"])  # bị bắt: mèo đè lên chuột, ô viền đỏ
    d.rectangle([0, h - BAR, w, h], fill=COL["bar"])
    d.text((10, h - BAR / 2), caption, font=_TEXT, fill="white", anchor="lm")
    return img


def episode_frames(env, frames, outcome):
    """Danh sách ảnh PIL cho từng bước của ván đã ghi (frames từ catmouse.play(record=True))."""
    total = len(env.cheeses)
    out = []
    for i, (m, c, mask, a) in enumerate(frames):
        left = bin(mask).count("1")
        move = f" · chuột: {ACTIONS[a][0]}" if a is not None else ""
        cap = f"Bước {i}/{len(frames) - 1} · phô mai {total - left}/{total}{move}"
        if i == len(frames) - 1 and outcome:
            cap = f"Bước {i} · {OUTCOME_VI[outcome]}"
        out.append(render_frame(env, m, c, mask, cap))
    return out


def to_gif(images, ms_per_frame=350, hold_last=6):
    """Ghép ảnh thành GIF động (giữ khung cuối lâu hơn để thấy kết cục)."""
    seq = images + [images[-1]] * hold_last
    buf = io.BytesIO()
    seq[0].save(buf, format="GIF", save_all=True, append_images=seq[1:], duration=ms_per_frame, loop=0)
    return buf.getvalue()


def _rolling(x, w):
    x = np.asarray(x, dtype=float)
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="valid"), w


def fig_learning(hist, window=200):
    """Trái: phần thưởng mỗi ván + trung bình trượt. Phải: tỉ lệ thắng / bị bắt / hết giờ theo cửa sổ trượt."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    rew = np.array(hist["reward"])
    ep = np.arange(1, len(rew) + 1)
    ma, w = _rolling(rew, window)
    ax1.plot(ep, rew, color="#93C5FD", lw=0.6, alpha=0.6, label="Phần thưởng mỗi ván")
    ax1.plot(ep[w - 1:], ma, color="#1D4ED8", lw=2.4, label=f"Trung bình trượt {w} ván")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Tổng phần thưởng")
    ax1.set_title("Phần thưởng của chuột")
    ax1b = ax1.twinx()
    ax1b.plot(ep, hist["epsilon"], color="#DC2626", lw=1.6, ls="--", label="epsilon")
    ax1b.set_ylim(0, 1.05)
    ax1b.set_ylabel("epsilon", color="#DC2626")
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=9)
    oc = np.array(hist["outcome"])
    for key, label, color in (("win", "Thắng (ăn hết phô mai)", "#16A34A"), ("caught", "Bị bắt", "#DC2626"),
                              ("timeout", "Hết giờ", "#F59E0B")):
        r, w = _rolling(oc == key, window)
        ax2.plot(ep[w - 1:], 100 * r, color=color, lw=2.2, label=label)
    ax2.set_ylim(-2, 102)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Tỉ lệ (%)")
    ax2.set_title(f"Kết cục theo cửa sổ trượt {w} ván")
    ax2.legend(loc="center right", fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_eval(ev):
    """Cột chồng: thắng / bị bắt / hết giờ của từng loại chuột."""
    names = list(ev)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    left = np.zeros(len(names))
    for key, color in (("Thắng", "#16A34A"), ("Bị bắt", "#DC2626"), ("Hết giờ", "#F59E0B")):
        vals = np.array([100 * ev[n][key] for n in names])
        ax.barh(names, vals, left=left, color=color, label=key)
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", weight="bold")
        left += vals
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("% số ván")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    fig.tight_layout()
    return fig


def fig_heatmap(env, q, cat_cell, mask=None):
    """Bản đồ nhiệt max_a Q((chuột, mèo cố định, phô mai), a) theo vị trí chuột."""
    mask = env.full_mask if mask is None else mask
    grid = np.full((env.n_rows, env.n_cols), np.nan)
    unseen = np.zeros((env.n_rows, env.n_cols), dtype=bool)
    for i, (r, c) in enumerate(env.cells):
        if i != cat_cell:
            row = q[env.state_index(i, cat_cell, mask)]
            if any(row):
                grid[r, c] = max(row)
            else:
                unseen[r, c] = True  # Q còn nguyên 0: chuột chưa từng gặp trạng thái này
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(np.where(unseen, 1.0, np.nan), cmap="Greys", vmin=0, vmax=4)
    ax.set_facecolor("#334155")
    cmap = plt.get_cmap("RdYlGn")
    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-vmax, vmax=vmax)
    for i, (r, c) in enumerate(env.cells):
        if i == cat_cell:
            ax.text(c, r, "MÈO", ha="center", va="center", fontsize=10, weight="bold", color="white",
                    bbox=dict(boxstyle="round", fc="#F97316", ec="none"))
        elif unseen[r, c]:
            ax.text(c, r, "?", ha="center", va="center", fontsize=10, color="#475569")
        elif not np.isnan(grid[r, c]):
            v = int(round(grid[r, c])) or 0
            ax.text(c, r - 0.2, f"{v}", ha="center", va="center", fontsize=8, color="#0F172A")
            row = q[env.state_index(i, cat_cell, mask)]
            _, dr, dc = ACTIONS[row.index(max(row))]  # hành động tham lam pi*(s)
            if abs(grid[r, c]) >= 0.5:  # bỏ mũi tên ở trạng thái gần như chưa học
                ax.arrow(c - dc * 0.2, r + 0.2 - dr * 0.2, dc * 0.4, dr * 0.4, head_width=0.14, head_length=0.12,
                     length_includes_head=True, color="#1E293B", lw=1.2)
        if env.layout[r][c] == HOLE:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="#3F2A14", lw=2.5))
    for k, cell in enumerate(env.cheeses):
        if mask & (1 << k):
            r, c = env.cells[cell]
            ax.plot(c, r - 0.3, marker="*", ms=14, color="#FACC15", mec="#92400E")
    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.set_title("max Q và hành động tham lam của chuột (mèo cố định)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="max_a Q(s, a)")
    fig.tight_layout()
    return fig
