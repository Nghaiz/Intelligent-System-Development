# Phase 8 — Báo cáo LaTeX

**Mục tiêu:** báo cáo kỹ thuật khoảng 45 trang, hợp nhất phong cách của cả hai
bài mẫu. Đáp ứng yêu cầu R2, R5, R10, R13.

**Tệp sở hữu:** `report/main.tex`, `report/chapters/*.tex`, `report/refs.bib`

## Yêu cầu về hình thức

- Biên dịch bằng **XeLaTeX**, bắt buộc, để hiển thị đúng dấu tiếng Việt.
- Trang bìa sao theo `A1_CT_tupv.879.pdf`: khung viền, logo học viện, thông tin
  sinh viên Nguyễn Duy Nghĩa, lớp D23CTPM01, mã B23DCCN600, giảng viên hướng dẫn
  PGS.TS Trần Đình Quế.
- Có Mục lục, Danh mục Hình vẽ, Danh mục Bảng biểu.
- Công thức toán đánh số: Accuracy, Precision, Recall, F1, MAE, RMSE, R bình
  phương, hàm sigmoid, hàm mất mát có điều chuẩn, và KG = (V, E, R).

## Cấu trúc chương

1. Cơ sở lý thuyết và Nguyên lý hệ thống thông minh
2. Hệ thống Chẩn đoán Nguy cơ Tiểu đường
3. Hệ thống Định giá Bất động sản Việt Nam
4. Kết luận và Hướng phát triển

## Phong cách nội dung, hợp nhất hai mẫu

Khung chương và độ sâu lý thuyết lấy theo `tupv.879`. Cách diễn giải từng biểu
đồ lấy theo `dungvt.194`: mỗi hình kèm ba mục — Loại biểu đồ, Dạng phân phối,
Chi tiết phân bố.

## Tiêu chí hoàn thành

- [x] `xelatex` biên dịch không lỗi, sinh ra `report/build/main.pdf` (72 trang)
- [x] Mọi hình và bảng đều được tham chiếu chéo trong phần thân bài
- [x] Không còn cảnh báo về tham chiếu hoặc trích dẫn chưa xác định

## Đã mở rộng so với kế hoạch ban đầu

Phiên làm việc bổ sung bốn nhóm hình mà bài mẫu `tupv.879` có còn kế hoạch gốc
thiếu, cùng hai tầng tri thức miền mới cho cả đồ thị lẫn ứng dụng web:

| Bổ sung | Tệp sinh ra |
|---------|-------------|
| Đường cong ROC 5 mô hình | `figures/dia_roc_curves.png` |
| Lưới ma trận nhầm lẫn 5 mô hình | `figures/dia_confusion_grid.png` |
| Sơ đồ Kiến trúc Đồ thị Tri thức đa tầng, hai miền | `figures/fig_kg_architecture_*.png` |
| Khám phá liên hợp Giá – Diện tích – Số phòng ngủ | `figures/hou_price_area_bedrooms.png` |
| Phân tích phần dư 4 bảng (thay ROC cho bài hồi quy) | `figures/hou_residual_analysis.png` |
| Danh mục bán lẻ dược phẩm FPT Long Châu | `src/kg/catalog_pharmacy.py` |
| Hạ tầng đô thị và phép tính tài chính niên kim | `src/kg/catalog_urban.py` |

Đồ thị Tri thức tăng từ 95 nút / 96 cạnh / 21 quan hệ lên **117 / 133 / 28**,
tổ chức lại thành **năm tầng** ở cả hai miền. Ứng dụng web thêm một tab cho mỗi
trang công cụ: *Gói chăm sóc* (giải pháp dược phẩm kèm chi phí) và
*Hạ tầng & Tài chính* (trả góp niên kim, lịch trả nợ, cụm tiện ích).

Bộ kiểm thử tăng từ 289 lên **355 bài**, trong đó `tests/test_catalogs.py` canh
tính toàn vẹn tham chiếu của danh mục và đối chiếu công thức niên kim với một
phép tính độc lập viết lại tay.
