# Kế hoạch triển khai — Assignment 01: Hệ thống Thông minh

Mỗi phase là một tệp độc lập, đọc được mà không cần ngữ cảnh phiên làm việc.

| Phase | Tên | Tệp | Trạng thái |
|-------|-----|-----|-----------|
| P0 | Khởi tạo dự án | [phase-0-khoi-tao.md](phase-0-khoi-tao.md) | ✅ Xong |
| P1 | Dữ liệu & Phân tích khám phá | [phase-1-du-lieu-eda.md](phase-1-du-lieu-eda.md) | ✅ Xong |
| P2 | Huấn luyện mô hình | [phase-2-mo-hinh.md](phase-2-mo-hinh.md) | ✅ Xong |
| P3 | Thực nghiệm có kiểm soát | [phase-3-thuc-nghiem.md](phase-3-thuc-nghiem.md) | ✅ Xong |
| P4 | Jupyter Notebook | [phase-4-notebook.md](phase-4-notebook.md) | ✅ Xong |
| P5 | Đồ thị Tri thức Neo4j | [phase-5-knowledge-graph.md](phase-5-knowledge-graph.md) | ✅ Xong |
| P6 | Ứng dụng web Streamlit | [phase-6-web-app.md](phase-6-web-app.md) | ✅ Xong |
| P7 | Deploy lên internet | [phase-7-deploy.md](phase-7-deploy.md) | ✅ Xong |
| P8 | Báo cáo LaTeX | [phase-8-bao-cao-latex.md](phase-8-bao-cao-latex.md) | ✅ Xong |
| P9 | Hoàn thiện & nộp bài | [phase-9-hoan-thien.md](phase-9-hoan-thien.md) | ⬜ Chưa bắt đầu |

## Kho mã

https://github.com/Nghaiz/Assignment-01-Intelligent-System

## Bản deploy

https://httm-nguyen-duy-nghia-assignment-01-intelligent-system.streamlit.app

## Quy ước chung cho mọi phase

- **Ngôn ngữ báo cáo:** tiếng Việt, thuật ngữ kỹ thuật giữ tiếng Anh trong ngoặc.
- **Hai bài toán song song:** Tiểu đường (phân lớp) và Bất động sản (hồi quy).
- **Nguồn tham chiếu bắt buộc:** `slide-assign_01/intel_sys_dev_assignment_01_2.pdf`
  (đề bài), `A01_05_dungvt.194.pdf` và `A1_CT_tupv.879.pdf` (hai bài mẫu).
- **Tính tái lập:** mọi con số sinh ra từ `python run_pipeline.py`, hạt giống cố
  định `RANDOM_STATE = 42` khai báo tại `src/config.py`.

## Đối chiếu với 14 yêu cầu bắt buộc của đề bài

| Mã | Yêu cầu | Phase đáp ứng |
|----|---------|---------------|
| R1 | Dữ liệu thật | P0 |
| R2 | Định nghĩa hệ thống | P4, P8 |
| R3 | Biểu diễn dữ liệu | P1 |
| R4 | Phân tích đặc trưng | P1 |
| R5 | Phát biểu bài toán | P4, P8 |
| R6 | Mô hình cơ sở | P2 |
| R7 | Tối thiểu 4 mô hình | P2 (làm 5) |
| R8 | Tối thiểu 3 thực nghiệm | P3 (làm 4) |
| R9 | Độ đo phù hợp | P2, P3 |
| R10 | Phân tích khoa học | P8 |
| R11 | Ứng dụng nhỏ | P6 |
| R12 | Minh hoạ toàn pipeline | P6, P7 |
| R13 | Phản chiếu về biểu diễn | P8 |
| R14 | Chạy lại được từ đầu | P4, P9 |
