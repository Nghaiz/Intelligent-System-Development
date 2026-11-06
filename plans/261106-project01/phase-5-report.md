# Phase 5 — Báo cáo LaTeX (≤ 55 trang)

Khung: chép từ `Report/TieuLuan Final/` (`main.tex`, `build.py`, `preamble/tikzstyle.tex`, bìa khuôn A04, logo, `\SL{}`, `\maNguon`, `\hinhpdf`, `\hinhtikz`), đổi `SRC`, `tlf`→`dnx`, tên PDF `project01_CT_07_nghiand.pdf`, đầu trang "Project 01: Phát triển Hệ thống Thông minh", bỏ các addbibresource thừa.
Bìa: "BÁO CÁO PROJECT 01", đề tài DiaNexus, Nguyễn Duy Nghĩa, D23CTPM01, B23DCCN600, nhóm lớp CT, nhóm 07, PGS.TS Trần Đình Quế, Hà Nội, tháng 11/2026.

## Quy cách
- `extreport` 14pt, mã ≥ 11pt, hình matplotlib chữ ≥ 12pt rộng 6,3 in, bảng không co quá mức. Ảnh chụp web to rộng hết khổ.
- TikZ nhiều màu, khối 3D, 2D + 3D, đẹp; ≥ 10 hình. Không chồng chữ. Nhãn TikZ ≥ 10pt.
- Mọi số qua `\SL{}` từ metrics. Không số chép tay. Không nhắc "bài mẫu"/tên bạn. Không em-dash lặp "câu—câu". Tiếng Việt tự nhiên, không sáo.

## Ngân sách trang (cứng, tổng ≤ 55)
| Phần | Trang |
|---|---|
| Bìa, mục lục, danh mục hình/bảng, viết tắt | 6 |
| Ch.1 Tổng quan, đề bài, ánh xạ sơ đồ | 3 |
| Ch.2 Cơ sở lý thuyết | 7 |
| Ch.3 Dữ liệu, CSDL, EDA | 6 |
| Ch.4 Học máy, học sâu, võng mạc | 8 |
| Ch.5 KB, Neo4j, RAG | 8 |
| Ch.6 Rule, RL, đa tác tử | 9 |
| Ch.7 Website | 5 |
| Ch.8 Đánh giá tổng, kết luận | 2 |
| Tham khảo | 1–2 |

## Làn
R1 Ch.1–2 · R2 Ch.3–4 · R3 Ch.5–6 · R4 Ch.7–8. Mỗi làn chỉ sửa `chapters/chN.tex`, `tikz/chN_*`, `refs/chN.bib`, `scripts/report_*`. Mỗi chương có 2–4 đoạn mã thật trích từ `src/Project 01` bằng `\maNguon`.
Lead ghép, biên dịch, kiểm số trang, đọc lại các trang bằng ảnh, sửa tràn và trang trống.
