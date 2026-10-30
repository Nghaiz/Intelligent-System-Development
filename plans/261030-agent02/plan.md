# Kế hoạch Agent 02 — hợp đồng chung cho mọi làn

**Trạng thái:** XONG 2026-10-31 — `Report/Agent 02/agent02_CT_07_nghiand.pdf`, 58 trang (người dùng nới giới hạn thành < 60), 0 lỗi LaTeX. Chưa commit.
Thiết kế: `plans/reports/brainstorm-261030-agent02.md`. Văn bản bài mẫu đã trích (UTF-8):
`C:/Users/Nghaiz/AppData/Local/Temp/claude/d--Python-Intelligent-System-Development/90d2873a-46a5-4394-a9f6-a9a2e6545341/scratchpad/txt/*.txt`.

## 1. Thư mục và quyền sở hữu
```
src/Agent 02/ag2/common.py plot_style.py texfmt.py      lead (chỉ đọc với làn)
src/Agent 02/ag2/market.py agents.py training.py exact.py   lead (RL)
src/Agent 02/scripts/run_all.py ch4_*.py ch5_*.py ch6_*.py ch7_*.py   lead
src/Agent 02/ag2/shop*.py, scripts/ch8_*.py              L2
src/Agent 02/outputs/metrics/chN_*.json                  chủ của chương N
Report/Agent 02/main.tex preamble/ build.py refs/agent02.bib assets/   lead
Report/Agent 02/chapters/chN.tex, tikz/chN_*.tex, figures/chN_*, generated/chN_*   chủ của chương N
Report/Agent 02/refs/chN.bib                             làn (khoá tiền tố chN:)
```
Chủ chương: L1 = ch1, ch2, ch3 · L2 = ch8 · lead = ch0, ch4–ch7, ch9. Làn không `git add/commit`.

## 2. Quy ước (giống Agent 01)
- Số liệu: `save_json("chN_x", obj)` → `\SL{chN_x/khoá/lá}` (biến thể `Raw`, `Pct`). Không gõ tay số thực nghiệm.
- Bảng: `ag2.texfmt.table("chN_x", spec, header, rows)` → `\bang{chN_x}{chú thích}{nhãn}`.
- Hình matplotlib: `from ag2.plot_style import plt, save, W, W2, C, SEQ` → `save(fig,"chN_x")`; `\hinhpdf{chN_x}{1}{..}{..}`.
- TikZ: `tikz/chN_x.tex` chỉ chứa `tikzpicture`; `\hinhtikz{chN_x}{..}{..}`. Nhiều màu, khối 3D (`\khoibaD`), bóng, rộng ≤ 16 cm, chữ ≥ `\small`.
- Mã: `\maNguon{module__Ten}{..}` / `\maDai{..}`, module trong `ag2/` hoặc `scripts/`.
- Kiểm một chương: `python build.py --chapter N`. Dựng toàn bộ: `python build.py --inline`.

## 3. Ngân sách trang (cứng)
ch0 1,5 · ch1 4,5 · ch2 5,5 · ch3 4 · ch4 5 · ch5 7 · ch6 10 · ch7 7 · ch8 4 · ch9 2 · bìa+mục lục+TLTK 4 → ≤ 55.

## 4. Mô hình (SSOT: `ag2/market.py`)
p=clip(0,30+0,25d+0,15a_i−0,10a_−i; 0,05; 0,95); N=10/20; S~Bin(N,p); r=S[P(1−δa)−C]; P=10, C=5, δ=0,2;
T=20; d_{t+1}~U{0,1}; đầu episode a_{t−1}=(0,0). Q-learning α=0,1, γ=0,9, ε_e=max(0,02; 0,2·0,995^e), 5.000 ep.
Đánh giá: 500 ep, ε=0, không học, seed riêng, số ngẫu nhiên chung. Seed baseline 2026; 10 seed 7,17,…,97.
