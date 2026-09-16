# Phát triển các Hệ thống Thông minh

Bài tập lớn học phần **Phát triển các Hệ thống Thông minh** — Học viện Công nghệ Bưu chính Viễn thông.

**Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · Lớp D23CTPM01
**Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
**Học kỳ:** Học kỳ 1 năm học 2026 – 2027

---

## Các bài đã nộp

| # | Chủ đề | Nội dung chính | Mã nguồn | Báo cáo |
|---|---|---|---|---|
| **01** | Từ Biểu diễn Dữ liệu đến Hệ thống Thông minh Đầu tiên | 2 hệ thống (chẩn đoán tiểu đường · định giá bất động sản Việt Nam), 5 mô hình ML mỗi bài, 4 thực nghiệm có kiểm soát, Đồ thị Tri thức Neo4j, web app Streamlit | [`src/Assignment 01`](src/Assignment%2001) | [PDF](Report/Assignment%2001/A01_CT_nghiand.600.pdf) |
| **02** | Hệ thống Thông minh có thể Triển khai | 3 ứng dụng (tiểu đường · giá nhà · hành vi khách hàng TMĐT) với REST API, giao diện Web và Mobile, chạy đồng thời trên 3 cổng | [`src/Assignment 02`](src/Assignment%2002) | [PDF](Report/Assignment%2002/A02_CT_nghiand.600.pdf) |
| **03** | Mạng Nơ-ron và Học Biểu diễn | Mạng nơ-ron sâu viết tay hoàn toàn bằng NumPy (không TensorFlow/PyTorch), mở rộng lên 3 hệ thống quy mô lớn kèm Web + Mobile | [`src/Assignment 03`](src/Assignment%2003) | [PDF](Report/Assignment%2003/A03_CT_nghiand.600.pdf) |

Mỗi bài có README riêng trong thư mục mã nguồn, hướng dẫn dựng lại từ đầu.

---

## Bố cục kho lưu trữ

```
Intelligent-System-Development/
├── src/                        Mã nguồn từng bài
│   ├── Assignment 01/            src/ app/ data/ models/ notebooks/ outputs/ figures/ tests/
│   ├── Assignment 02/            diabetes/ house_price/ customer_behavior/
│   └── Assignment 03/            diabetes_baseline/ diabetes_large/ house_price_large/
│                                 customer_comments/ knowledge_graph/
│
├── Report/                     Báo cáo từng bài (PDF + mã nguồn dựng báo cáo)
│   ├── Assignment 01/            LaTeX: main.tex, chapters/, refs.bib, figures/
│   ├── Assignment 02/            Python: build_report.py, chapters_*.py, figures/
│   └── Assignment 03/            Python: build_report.py, chapters_*.py, figures/
│
└── TAILIEU/                    Tài liệu môn học
    ├── 1. BASIC MACHINE LEARNING ... .pdf
    ├── 2. Deep learning with python ... .pdf
    └── slide-assign_01 … 04/     Slide bài giảng và đề bài từng tuần
```

Mỗi bài trong `src/` và `Report/` mang cùng một tên thư mục, nên tra chéo mã nguồn với báo cáo chỉ cần đổi thư mục gốc.

### Thêm bài mới

Bài tiếp theo chỉ cần ba thư mục cùng tên, không phải sửa gì ở nơi khác:

```
src/Assignment 04/            mã nguồn
Report/Assignment 04/         báo cáo
TAILIEU/slide-assign_04/      slide và đề bài
```

---

## Quy ước chung

**Tên thư mục con của mỗi hệ thống** trong `src/` thống nhất qua các bài:

| Thư mục | Chứa gì |
|---|---|
| `data/` | Dữ liệu thô và dữ liệu kiểm định ngoài |
| `model/` hoặc `models/` | Mô hình đã huấn luyện (`.joblib`, `.npz`) kèm `metadata.json` |
| `notebook/` hoặc `notebooks/` | Notebook phân tích và huấn luyện |
| `reports/figures/` | Biểu đồ sinh ra từ notebook |
| `api/` | Dịch vụ REST |
| `web/`, `mobile/` | Giao diện người dùng |

**Tên file báo cáo:** `A<số bài>_CT_nghiand.600.pdf`

**Môi trường:** mỗi bài có `requirements.txt` riêng. Tạo môi trường ảo trong chính thư mục bài đó:

```bash
cd "src/Assignment 03"
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Biến môi trường:** các bài dùng Neo4j đọc thông tin kết nối từ `.env`. Sao chép `.env.example` thành `.env` rồi điền giá trị của bạn. File `.env` đã được `.gitignore` chặn, không bao giờ được commit.

---

## Ghi chú

Toàn bộ hệ thống chạy trên máy cá nhân, không có bản triển khai công khai trên Internet. Hướng dẫn dựng lại nằm trong README của từng bài.
