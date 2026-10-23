# Kế hoạch Tiểu luận 03 — hợp đồng chung cho mọi làn

**Trạng thái:** Pha 0 (khung) xong → Pha 1 (6 làn song song) → Pha 2 (Web App + ảnh chụp) → Pha 3 (ghép, Kết luận, dựng PDF, rà soát).
Thiết kế: `plans/reports/brainstorm-261023-tieuluan03.md`. Bài mẫu (văn bản đã trích): thư mục `ref/` trong scratchpad của phiên — đường dẫn tuyệt đối:
`C:/Users/Nghaiz/AppData/Local/Temp/claude/d--Python-Intelligent-System-Development/7405c60a-58bc-4428-a9cd-c1d39ef1d0ad/scratchpad/ref/*.txt`
(PDF gốc ở `tmp/`). dungvt = mẫu chính; ba bài kia bổ sung.

## 1. Thư mục và quyền sở hữu tệp (BẮT BUỘC)
```
src/TieuLuan 03/                      SRC
  tl3/common.py gpu.py plot_style.py texfmt.py llm.py   (lead — chỉ đọc với các làn)
  tl3/chN_*.py                        thư viện của làn chương N
  tl3/serve_chN.py                    hàm suy luận cho Web App (hợp đồng mục 6)
  scripts/chN_*.py                    kịch bản chạy thực nghiệm, vẽ hình, sinh bảng của làn N
  data/chN/...                        dữ liệu làn N (tải bằng scripts/chN_download.py)
  models/chN/...                      mô hình đã huấn luyện của làn N
  outputs/metrics/chN_*.json          số liệu (nguồn của \SL)
  outputs/logs/chN_*.log
  app/                                Pha 2 (làn Web)
Report/TieuLuan 03/                   REP
  main.tex preamble/ build.py refs/refs.bib assets/   (lead)
  chapters/chN.tex                    làn N (ch0 = Mở đầu, ch1..ch7, ch8 = Kết luận do lead)
  tikz/chN_*.tex                      hình TikZ của làn N
  figures/chN_*.pdf                   hình matplotlib của làn N
  generated/chN_*.tex                 bảng do scripts của làn N sinh
  refs/chN.bib                        tài liệu tham khảo của làn N (khoá có tiền tố chN: ví dụ ch3:lecun1998)
  code/                               build.py tự sinh — KHÔNG ghi tay
  assets/ui/chN_*.png                 Pha 2 sinh
```
Không làn nào sửa tệp ngoài tiền tố của mình. Không `git add/commit/checkout` — lead commit ở cuối.

## 2. Số liệu và bảng
- Mỗi kịch bản ghi `save_json("chN_<tên>", obj)` (tl3.common). Trong .tex chỉ dùng `\SL{chN_<tên>/khoá/.../lá}`; số thực ≤1,5 có thêm biến thể `...Pct` (phần trăm). KHÔNG gõ tay con số thực nghiệm.
- Bảng: kịch bản `scripts/chN_tables.py` dùng `tl3.texfmt.table(...)` ghi `REP/generated/chN_<tên>.tex` (+ `_note.tex` tùy chọn). Trong .tex: `\bang{chN_<tên>}{chú thích}{nhãn}`. Số kiểu Việt (phẩy thập phân) qua `texfmt.num/fmt`.
- Bảng tĩnh mô tả (không phải kết quả) được viết trực tiếp trong chương.

## 3. Hình
- matplotlib: `from tl3.plot_style import plt, save, W, C, SEQ` → `save(fig, "chN_<tên>")` ghi `REP/figures/chN_<tên>.pdf`. Vẽ đúng bề rộng W = 6,3 in, chữ ≥ 11 pt. Chèn: `\hinhpdf{chN_<tên>}{1}{chú thích}{nhãn}` hoặc `\haihinh{...}`.
- TikZ: `REP/tikz/chN_<tên>.tex` chỉ chứa `\begin{tikzpicture}...\end{tikzpicture}`; chèn `\hinhtikz{chN_<tên>}{chú thích}{nhãn}`. Dùng style trong `preamble/tikzstyle.tex` (khoi, khoiC/G/P/R/T/X, cau, mui, `\khoibaD` hộp 3D, bảng màu cBlue…cNavy). Yêu cầu người dùng: **nhiều màu, nhiều khối, bóng đổ, 2D–3D, đẹp**; vẽ ở bề rộng ≤ 16 cm để không bị co, chữ không nhỏ hơn `\small`.
- Ảnh (mẫu dữ liệu, Grad-CAM…) xuất PDF/PNG vào `REP/figures/chN_*.{pdf,png}`.
- Ảnh giao diện: `\anhUI{chN_<exp>_web}{...}{...}` và `\anhMobile{chN_<exp>_mobile}{...}{...}` — Pha 2 sinh tệp; trước đó build hiện khung "chưa chụp".

## 4. Mã nguồn trong báo cáo
`\maNguon{module__ten}{chú thích}` (≤ 1 trang) hoặc `\maDai{...}` (dài). `module` = tên tệp (không .py) trong `tl3/`, `scripts/` hoặc `app/`; `ten` = hàm, lớp, `Lop.phuong_thuc` hoặc hằng. build.py trích nguyên văn → mã trong báo cáo luôn là mã đã chạy. Mỗi chương thực nghiệm ≥ 25 đoạn mã, phủ: tải/làm sạch dữ liệu, EDA, mô hình (mỗi framework), huấn luyện, đánh giá, suy luận/triển khai. Viết hàm ngắn, có docstring tiếng Việt, để trích đẹp.

## 5. Văn phong & quy mô
Tiếng Việt học thuật, rõ, không sáo rỗng; chú thích hình/bảng đầy đủ; mỗi hình/bảng có đoạn `\nhanxet{}` diễn giải. Thuật ngữ Anh để trong ngoặc lần đầu. Trích dẫn `\cite{chN:key}`. Công thức toán đầy đủ cho lý thuyết. Quy mô mục tiêu (14pt): Ch1 ≥ 25 tr; Ch2 ≥ 55 tr; Ch3 ≥ 65 tr; Ch4 ≥ 45 tr; Ch5 ≥ 45 tr; Ch6 ≥ 20 tr; Ch7 ≥ 20 tr. Đề mục bám dungvt (xem TOC trong `ref/tieuluan03_05_07_dungvt.txt`), bổ sung từ 3 bài kia.

## 6. Hợp đồng triển khai (Web App Pha 2)
Mỗi làn thực nghiệm cung cấp `tl3/serve_chN.py`:
```python
SPECS: dict[str, dict]   # exp_id -> {"title": str, "kind": "tabular"|"image"|"series"|"chat"|"text"|"graph",
                         #            "inputs": [ {name, label, type: number|select|text|image, options?, min?, max?, default} ],
                         #            "models": [model_id,...], "example": {...}}
def predict(exp_id: str, payload: dict, model_id: str | None = None) -> dict
    # trả {"label"/"value": ..., "proba"?: {...}, "details": {...}, "model": model_id, "latency_ms": float}
```
Nạp mô hình lười (lazy) từ `models/chN/`; ảnh nhận PIL.Image; chuỗi thời gian nhận `{"history": [...]} ` hoặc dùng dữ liệu cuối. Không phụ thuộc Flask. exp_id: `ch2_diabetes, ch2_house, ch2_comments, ch3_mnist, ch3_fashion, ch3_rps, ch3_cifar10, ch4_ecommerce, ch4_aapl, ch4_gold, ch5_diabetes, ch5_stock, ch5_gold, ch6_cora, ch7_math`.
Tên ảnh UI: `chN_<exp>_web.png`, `chN_<exp>_mobile.png` (exp không có tiền tố chN_, ví dụ `ch3_mnist_web.png`).

## 7. Tài nguyên máy
- GPU RTX 4060 8 GB (≈2 GB đã dùng bởi hệ thống). Việc GPU nặng (CNN PyTorch, LLM, embedding hàng loạt) bọc `with tl3.gpu.heavy("mô tả"):` (khoá tệp, chờ lượt). Việc GPU nhẹ (MLP, RNN, GNN, ≤1,5 GB) không cần khoá nhưng gọi `tl3.gpu.cap(0.2)`.
- CPU 32 luồng: gọi `tl3.common.limit_threads(n)` đầu kịch bản. Hạn mức: ch2 8, ch3 NumPy 6 + TF 8, ch4 TF 6, ch5 4, ch6/7 3, ch1 0. RAM ≤ 4 GB/tiến trình.
- Huấn luyện dài chạy nền (`run_in_background`), log vào `outputs/logs/`. Không để phiên đứng chờ lệnh tiền cảnh > 2 phút.
- CUDA hay sập lúc thoát (0xC0000409/127) SAU khi đã ghi kết quả → kiểm tra mtime tệp đầu ra, không tin mã thoát.
- Bash heredoc làm hỏng dấu `\` → mọi tệp có `\` (LaTeX, regex) viết bằng Write/Edit.
- Dữ liệu Kaggle: `kagglehub.dataset_download("owner/slug")` tải ẩn danh được; chép tệp cần dùng vào `data/chN/`.
- Python: `D:/Python/Intelligent-System-Development/.venv/Scripts/python`, chạy từ `src/TieuLuan 03` (để `import tl3`).

## 7b. Kiểm tra LaTeX của riêng chương
`python build.py --chapter N` (trong REP): sinh macro/mã/bảng rồi dựng `kiemtra_chN.pdf` chỉ gồm chương N (TikZ vẽ tại chỗ, không cache). Phải hết lỗi LaTeX, hết `Thieu so lieu`, hết overfull nghiêm trọng trước khi báo xong. Mở PDF bằng Read (pages) để tự soát hình/bảng to, rõ.

## 8. Các làn Pha 1
| Làn | Phạm vi | Tệp chương |
|---|---|---|
| L1 | Mở đầu + Ch1 Lịch sử AI (dungvt 1.1–1.15 + linhvtm 2025–2026 + lucnvm 1.9 dữ liệu/đo lường + AnhNT 1.7) | ch0.tex, ch1.tex |
| L2 | Ch2 ML cơ bản: lý thuyết 2.1–2.12 (+PLA, Softmax, SVR, KNN, K-means, PCA, GD); TN1 Diabetes, TN2 House Price, TN3 Customer Comments; K-means/PCA minh hoạ | ch2.tex |
| L3 | Ch3 CNN: lý thuyết 3.1–3.6; TN MNIST, Fashion-MNIST, Rock-Paper-Scissors, CIFAR-10; Basic scratch/PyTorch/TF + Improved PyTorch/TF; Grad-CAM | ch3.tex |
| L4 | Ch4 RNN: lý thuyết 4.1–4.21; TN E-commerce, AAPL, Gold; RNN/LSTM/GRU(+BiLSTM) PyTorch & Keras; RNN NumPy BPTT | ch4.tex |
| L5 | Ch5 KG & RAG: lý thuyết 5.1–5.3 (+linhvtm); TN Diabetes, Google stock, Gold (FINAL_USO); truy xuất 5 kiểu + đánh giá + LLM + agent định tuyến | ch5.tex |
| L6 | Ch6 GNN (GCN/SAGE/GAT, Cora/CiteSeer) + Ch7 Hệ đa tác tử (GSM8K tập con, 3–4 hệ so sánh) | ch6.tex, ch7.tex |
Pha 2: L7 Web App (Flask, `src/TieuLuan 03/app/`) + `app/screenshots.py` (desktop 1440×900, mobile 390×844). Pha 3: lead.
