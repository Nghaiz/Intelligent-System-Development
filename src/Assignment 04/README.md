# Assignment 04 — Convolutional Neural Networks

Từ phép tích chập toán học đến NumPy, PyTorch và TensorFlow.

**Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · Lớp D23CTPM01
**Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
**Học kỳ:** Học kỳ 1 năm học 2026 – 2027

Báo cáo: [`Report/Assignment 04/A04_CT_nghiand.600.pdf`](../../Report/Assignment%2004/)

---

## Bài này làm gì

Cùng một kiến trúc tích chập được hiện thực **ba lần** — NumPy thuần từ đầu, PyTorch,
TensorFlow/Keras — rồi chạy trên **năm miền dữ liệu** có hình học khác nhau, để trả lời một
câu hỏi duy nhất: *khi tầng hiện thực thay đổi mà mô hình toán học giữ nguyên, cái gì đổi và
cái gì không?*

| Miền | Dạng dữ liệu | Bài toán | Kiến trúc |
|---|---|---|---|
| Customer Comments | Chuỗi token | Phân loại nhị phân | 1D CNN + Global Max Pooling |
| Diabetes | Bảng số (8 cột) | Phân loại nhị phân | 1D CNN 8 → 16 → 1 |
| House Price | Bảng số (8 cột) | Hồi quy liên tục | 1D CNN, đầu ra tuyến tính |
| MNIST | Ảnh xám 28×28×1 | Phân loại 10 lớp | 2D CNN (cơ sở và cải tiến) |
| CIFAR-10 | Ảnh màu 32×32×3 | Phân loại 10 lớp | 2D CNN 3 khối phân cấp |

Kèm hai thực nghiệm phụ trợ: **hoán vị cột đặc trưng** trên dữ liệu bảng (kiểm chứng luận
điểm "dữ liệu bảng không có tô-pô"), và **đối kháng MLP với CNN** kèm phân tích không gian
biểu diễn ẩn bằng PCA.

Toàn bộ lan truyền ngược viết tay đều được xác nhận bằng **kiểm chứng sai phân hữu hạn**
trước khi bất kỳ con số huấn luyện nào được báo cáo.

---

## Bố cục

```
src/Assignment 04/
├── CONTRACT.md              Hợp đồng tích hợp: tên file, khóa JSON, danh mục hình
├── requirements.txt
├── theory_figures/          9 hình minh hoạ lý thuyết cho Chương 1 của báo cáo
│
├── customer_comments/       ┐
├── diabetes/                │ data/ notebooks/ models/ reports/figures/
├── house_price/             │ mỗi miền có 3 notebook: NumPy · PyTorch · TensorFlow
├── mnist/                   │
├── cifar10/                 ┘
└── mlp_vs_cnn/              Đối kháng MLP–CNN và phân tích PCA không gian ẩn
```

Mỗi miền tự chứa: notebook đọc dữ liệu từ `data/` của chính nó, ghi hình vào
`reports/figures/` và ghi số liệu vào `reports/metrics_<miền>.json`. Bộ dựng báo cáo chỉ
đọc lại các tệp JSON đó, nên **không có con số nào được chép tay vào văn bản**.

---

## Chạy lại từ đầu

```bash
cd "src/Assignment 04"
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Sau đó chạy notebook theo thứ tự số trong tên tệp, từng miền một:

```bash
cd customer_comments/notebooks
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 01_comments_1d_cnn_numpy.ipynb
```

Thứ tự bắt buộc trong mỗi miền: notebook `01` (NumPy) → `02` (PyTorch) → `03` (TensorFlow).
Notebook cuối cùng của mỗi miền là nơi gộp kết quả ba khung lại thành hình so sánh và tệp
`metrics_<miền>.json`, nên chạy lẻ notebook `03` khi chưa có `01` và `02` sẽ báo thiếu dữ liệu.

`mlp_vs_cnn` cần MNIST và CIFAR-10 đã chạy xong, vì nó nạp lại mô hình PyTorch đã lưu để
trích vector biểu diễn ẩn.

Cuối cùng dựng báo cáo:

```bash
python "../../Report/Assignment 04/build_report.py"
```

---

## Dữ liệu

Ba tập bảng và văn bản được tái sử dụng từ các bài trước, đã nằm sẵn trong `data/` của từng
miền. Hai tập ảnh tải về dưới dạng `.npz`:

| Tập | Nguồn |
|---|---|
| Women's Clothing E-Commerce Reviews | [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) |
| Diabetes Prediction Dataset | [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| USA Real Estate Dataset | [Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) |
| MNIST | [Yann LeCun / Kaggle mirror](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) |
| CIFAR-10 | [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) |

---

## Ghi chú về môi trường chạy

Toàn bộ thực nghiệm chạy trên **CPU, không có GPU**. Điều này ảnh hưởng tới hai điều, và cả
hai đều được nêu rõ trong báo cáo thay vì giấu đi:

- Hai mô hình 2D CNN thuần NumPy được huấn luyện trên **tập con lấy mẫu phân tầng** để giữ
  thời gian chạy trong ngân sách bài tập, trong khi PyTorch và Keras dùng toàn bộ tập
  huấn luyện. Cả bốn mô hình vẫn được đánh giá trên cùng một tập kiểm thử đầy đủ.
- Các con số thời gian huấn luyện phản ánh **một lần chạy trên một máy cụ thể**, không phải
  một phép đo hiệu năng tuyệt đối của khung thư viện.
