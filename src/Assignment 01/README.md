# Assignment 01 — Từ Biểu diễn Dữ liệu đến Hệ thống Thông minh Đầu tiên

Môn **Phát triển các Hệ thống Thông minh** · Học viện Công nghệ Bưu chính Viễn thông
Sinh viên **Nguyễn Duy Nghĩa** · Lớp D23CTPM01 · Mã B23DCCN600
Giảng viên hướng dẫn: PGS.TS Trần Đình Quế

---

## Hệ thống này làm gì

Hai hệ thống thông minh cỡ nhỏ, cùng đi trọn một chuỗi xử lý:

```
Thế giới thực → Dữ liệu → Vector đặc trưng → Mô hình ML → Dự đoán → Ứng dụng
```

| | Bài toán 1 | Bài toán 2 |
|---|---|---|
| **Miền** | Y tế | Bất động sản |
| **Nhiệm vụ** | Phân lớp — chẩn đoán nguy cơ tiểu đường | Hồi quy — định giá nhà đất Việt Nam |
| **Dữ liệu** | Pima Indians Diabetes, 768 quan sát | Vietnam Housing, 30.229 tin đăng |
| **Mục tiêu** | `Outcome` nhị phân | `Price`, đơn vị tỷ VNĐ |
| **Người dùng** | Bác sĩ sàng lọc | Người mua, nhà đầu tư |

Mỗi bài toán được giải bằng **5 mô hình Học máy truyền thống**, so sánh dưới cùng
một giao thức đánh giá, cộng thêm **4 thực nghiệm có kiểm soát** và một **Đồ thị
Tri thức** biến con số dự báo thành khuyến nghị hành động được.

---

## Chạy lại từ đầu

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  nguồn .venv/bin/activate trên Linux
pip install -r requirements.txt

python run_pipeline.py           # Sinh lại TẤT CẢ: mô hình, biểu đồ, bảng, đồ thị tri thức
jupyter lab notebooks/           # Mở hai notebook, chạy Restart & Run All
streamlit run app/app.py         # Mở ứng dụng web
```

Bản chạy sẵn trên internet, không cần cài gì:
**https://httm-nguyen-duy-nghia-assignment-01-intelligent-system.streamlit.app**

Một lệnh dựng lại mọi thứ, hết khoảng 100 giây. Hai cờ khi cần:

```bash
python run_pipeline.py --kg-offline   # Dựng đồ thị nhưng không kết nối Neo4j
python run_pipeline.py --skip-kg      # Bỏ hẳn bước đồ thị
python -m src.kg.build_graph          # Chỉ dựng lại đồ thị, không huấn luyện lại
```

Bước đồ thị là bước duy nhất có thể chạm tới mạng, và nó không bao giờ được
phép làm hỏng cả lệnh: thiếu tài khoản Neo4j thì nó in cảnh báo rồi đi tiếp
bằng đồ thị offline.

Kiểm tra hai notebook chạy sạch từ đầu đến cuối mà không cần mở giao diện:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

`run_pipeline.py` là nguồn duy nhất sinh ra mọi con số trong báo cáo. Hạt giống
ngẫu nhiên cố định tại `src/config.py`, nên chạy lại cho kết quả trùng khít.

---

## Kiểm thử

```bash
python -m pytest                 # Toàn bộ, khoảng 15 giây
python -m pytest -m "not slow"   # Bỏ phần cần nạp mô hình và dữ liệu thật
python -m pytest -m "not neo4j"  # Bỏ phần cần máy chủ Neo4j
```

355 bài kiểm thử. Phần cần máy chủ Neo4j **tự bỏ qua** khi `.env` chưa cấu
hình, nên bộ kiểm thử vẫn xanh trên máy không có tài khoản Aura — đúng tinh
thần hai đường của Phase 5. Phần ấy chỉ ghi vào đồ thị mang định danh
`pytest_tmp` và tự dọn sau mỗi bài, không đụng tới dữ liệu thật.

| Tệp | Canh điều gì |
|-----|--------------|
| `tests/test_ontology.py` | Ngưỡng phân tầng tại đúng giá trị biên, tính toàn vẹn đồ thị |
| `tests/test_build_graph.py` | Ranh giới kiểu pandas ↔ Python, tính tái lập của việc chọn ca |
| `tests/test_neo4j_client.py` | Thoát chuỗi Cypher, rào chắn tiêm mã, không rò rỉ mật khẩu |
| `tests/test_artifacts.py` | Hợp đồng dữ liệu `graph.json` mà Phase 6 và Phase 8 sẽ đọc |
| `tests/test_catalogs.py` | Toàn vẹn tham chiếu danh mục, công thức niên kim, kiến trúc đa tầng |
| `tests/test_neo4j_live.py` | Nạp lên máy chủ thật rồi đọc ngược ra đối chiếu |

---

## Cấu trúc kho mã

| Đường dẫn | Nội dung |
|-----------|----------|
| `src/config.py` | Đường dẫn, hằng số, hạt giống ngẫu nhiên, kiểu vẽ biểu đồ |
| `src/preprocess.py` | Làm sạch dữ liệu và xây dựng biểu diễn đặc trưng |
| `src/train.py` | Định nghĩa và huấn luyện 5 + 5 mô hình |
| `src/evaluate.py` | Bộ độ đo và các thực nghiệm có kiểm soát |
| `src/viz.py` | Sinh toàn bộ biểu đồ |
| `src/kg/` | Bản thể học, hai danh mục tri thức miền và Đồ thị Tri thức Neo4j |
| `notebooks/` | Hai Jupyter Notebook, đủ 22 mục theo đề bài |
| `app/` | Ứng dụng web Streamlit — ba trang, bốn tab mỗi trang công cụ |
| [`Report/Assignment 01/`](../../Report/Assignment%2001) | Mã nguồn LaTeX của báo cáo (ở thư mục gốc kho) |
| `tests/` | 355 bài kiểm thử, chạy bằng `python -m pytest` |
| `plans/` | Kế hoạch triển khai, chia theo phase |
| `setup-guides/` | Hướng dẫn tạo tài khoản dịch vụ bên ngoài |
| `figures/`, `outputs/`, `models/` | Kết quả sinh ra tự động |

---

## Kết quả chính

**Chẩn đoán tiểu đường** — mô hình cơ sở đoán lớp đa số đạt Accuracy 0.6547
nhưng Recall bằng 0, tức bỏ sót toàn bộ người bệnh. Mô hình tốt nhất đạt
Accuracy 0.7914, khớp với mức 0.77–0.79 mà các nghiên cứu công bố trên cùng tập
dữ liệu. Hạ ngưỡng quyết định từ 0.50 xuống 0.15 nâng Recall từ 0.458 lên 0.854.

**Định giá bất động sản** — XGBoost đạt R² = 0.6141 và MAE = 1.05 tỷ VNĐ, so với
mô hình cơ sở đoán trung vị có R² xấp xỉ 0. Loại bỏ các đặc trưng định danh
(tỉnh, quận, pháp lý) làm R² tụt xuống 0.3340, cho thấy vị trí hành chính là
nhóm thông tin quyết định trong định giá.

**Đồ thị Tri thức** — 117 thực thể, 133 cạnh, 28 loại quan hệ, nạp lên Neo4j Aura.
Sáu ca mẫu lấy từ chính hai tập dữ liệu, mỗi tầng nguy cơ và mỗi phân khúc thị
trường một ca. Đồ thị dựng theo **năm tầng** ở cả hai miền, và hai tầng cuối mới
là chỗ con số biến thành việc làm được:

- **Y sinh** — bệnh nhân #192, Glucose 159 mg/dL → KNN cho xác suất 54.5% →
  ICD-10 E11 → Nhóm Nguy cơ Cao → *Gói Theo dõi Chuyên sâu*: 5 mặt hàng có thật
  tại Nhà thuốc FPT Long Châu, chi phí ban đầu 1.99 – 2.75 triệu đ, duy trì
  1.20 – 1.66 triệu đ mỗi tháng, kèm dịch vụ đo đường huyết miễn phí tại quầy.
- **Đô thị – Tài chính** — tin đăng #23933 ở Nhà Bè, 100 m², 4 tầng → XGBoost
  định giá 8.42 tỷ → Phân khúc Cao cấp → hạn mức vay LTV 70% là 5.89 tỷ →
  **trả góp 47.5 triệu đ mỗi tháng** trong 25 năm (công thức niên kim), tổng
  tiền lãi 8.34 tỷ, **thu nhập tối thiểu 118.7 triệu đ/tháng** ở ngưỡng DTI 40%,
  cùng cụm bốn tiện ích hạ tầng trong bán kính 2 km.

Giá dược phẩm là **khoảng tham khảo có ghi ngày chốt**, không phải bản ghi giá
trực tiếp — mỗi mặt hàng mang một đường dẫn tra cứu giá hiện hành. Điều kiện vay
là minh hoạ theo mặt bằng lãi suất công bố, nhưng phép tính trả góp thì là toán
học kiểm chứng được. Thiếu tài khoản Neo4j thì hệ thống tự chuyển sang đồ thị
offline, không phần nào bị chặn.

**Phát hiện xuyên suốt hai bài toán** — cùng một thiết kế thực nghiệm về biểu
diễn dữ liệu cho hai kết luận trái ngược: ở bài toán bất động sản, đổi biểu diễn
tạo biên độ ngang với đổi thuật toán (0.26 R²); ở bài toán tiểu đường, biên độ do
biểu diễn chỉ bằng một phần sáu biên độ do thuật toán (0.03 so với 0.18 F1). Điều
phân biệt là biểu diễn có làm thay đổi **lượng thông tin** hay chỉ đổi **dạng**
thông tin đã có.

---

## Nguồn dữ liệu

- **Pima Indians Diabetes** — https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Vietnam Housing Dataset** — https://www.kaggle.com/datasets/huutri148/vietnam-housing-dataset
