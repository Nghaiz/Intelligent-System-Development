# Assignment 03 — Neural Networks and Representation Learning

**Học phần:** Phát triển các Hệ thống Thông minh — Học viện Công nghệ Bưu chính Viễn thông
**Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · D23CTPM01
**Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
**Học kỳ:** Học kỳ 1 năm học 2026 – 2027

Xây dựng mạng nơ-ron sâu **hoàn toàn bằng NumPy** — không TensorFlow, không PyTorch, không `sklearn.neural_network`. Mọi phép nhân ma trận, mọi đạo hàm riêng, mọi bước cập nhật trọng số đều được viết tay từ đầu.

Nghiên cứu đi từ một mạng 289 tham số trên 768 mẫu, mở rộng lên **ba hệ thống thông minh quy mô lớn**, rồi triển khai thành ba dịch vụ chạy được thật có giao diện Web và Mobile.

**Báo cáo đầy đủ:** [`Report/Assignment 03/A03_CT_nghiand.600.pdf`](../../Report/Assignment%2003/A03_CT_nghiand.600.pdf)

---

## Ba hệ thống thông minh

| Hệ thống | Bài toán | Quy mô dữ liệu | Kiến trúc DL tốt nhất | Cổng API |
|---|---|---|---|---|
| **Sàng lọc tiểu đường** | Phân loại nhị phân mất cân bằng | 100.000 hồ sơ | Deeper MLP `14→64→32→16→1` | `5001` |
| **Định giá bất động sản** | Hồi quy giá trị liên tục | 150.000 giao dịch | Wide Regressor `13→128→64→1` | `5002` |
| **Phân loại nhận xét** | Phân loại văn bản NLP | 23.486 đánh giá | Standard Text MLP `1000→32→16→1` | `5003` |

Ba dịch vụ dùng ba cổng khác nhau nên **chạy đồng thời được** — không phải tắt cái này để chạy cái kia.

---

## Phát hiện chính

**1. Không tồn tại một kiến trúc tối ưu cho mọi bài toán.** Cùng một lớp mạng, cùng cách khởi tạo He Normal, cùng thuật toán tối ưu — nhưng cấu hình tầng thắng cuộc lại khác nhau ở cả ba bài toán, và khác theo hướng giải thích được:

- **Phân loại → chiều sâu thắng.** Bài toán cần bẻ cong một mặt phân tách; mỗi tầng ẩn cho phép gấp không gian thêm một lần.
- **Hồi quy → chiều rộng thắng.** Bài toán cần phủ một mặt giá trị liên tục; mỗi nơ-ron ReLU là một mảnh phẳng, càng nhiều mảnh song song thì bề mặt càng mượt.
- **Văn bản thưa chiều cao → vừa đủ thắng.** Với độ thưa 97,8%, phần lớn tham số tầng đầu không được cập nhật trong mỗi lô; tăng kích thước chỉ tạo thêm tham số chết.

**2. Biểu diễn quan trọng hơn thuật toán.** Ở hệ thống NLP, bốn mô hình chênh nhau từ 1.001 đến 51.050 tham số nhưng Macro F1 chỉ dao động trong khoảng hơn 2 điểm phần trăm. Trong khi đó, việc thêm bigram vào TF-IDF, thêm hai đặc trưng tương tác lâm sàng, hay áp dụng Smooth Bayesian Target Encoding — mỗi bước đều tạo khác biệt lớn hơn nhiều so với đổi thuật toán.

**3. Tinh chỉnh ngưỡng là bước rẻ nhất nhưng hay bị bỏ quên nhất.** Chỉ dịch chuyển ngưỡng quyết định, không đổi một trọng số nào, đã cứu được hàng trăm ca bệnh khỏi bị bỏ sót trên tập kiểm thử. Chi phí tính toán gần bằng 0.

**4. R² trên thang log có thể đánh lừa nghiêm trọng.** Ở bài toán định giá nhà, Linear Regression đạt R²(log) = 0,786 nhưng R²(raw) chỉ còn 0,656. Hàm mũ khuếch đại sai số ở đuôi giá cao theo cấp số nhân, nên phải luôn báo cáo song song cả hai chỉ số.

**5. Mạng viết tay chạy được thật ở quy mô sản xuất.** Ba REST API tự thực hiện forward pass bằng 9 dòng NumPy đọc từ tệp `.npz`. Sai khác so với mô hình gốc trong notebook nhỏ hơn 10⁻¹⁰.

---

## Năm notebook

Notebook là sản phẩm trọng tâm của Assignment. Mỗi notebook tự chứa đầy đủ: lý thuyết toán học, mã nguồn, kết quả chạy thật và phân tích.

| # | Notebook | Nội dung |
|---|---|---|
| 01 | [`diabetes_baseline/notebook/01_diabetes_dl_from_scratch_numpy.ipynb`](diabetes_baseline/notebook/01_diabetes_dl_from_scratch_numpy.ipynb) | Mạng cơ sở `8→16→8→1` (289 tham số) với **bốn khuyết tật cố ý**: rò rỉ dữ liệu, chia ngẫu nhiên không phân tầng, giữ giá trị 0 phi lý sinh học, ngưỡng cứng 0,50 |
| 02 | [`diabetes_baseline/notebook/02_diabetes_dl_from_scratch_improvements.ipynb`](diabetes_baseline/notebook/02_diabetes_dl_from_scratch_improvements.ipynb) | Sửa cả bốn khuyết tật + He Normal + dò ngưỡng tối ưu, kèm **6 nghiên cứu bóc tách kiến trúc** và trực quan hoá PCA |
| 03 | [`diabetes_large/notebook/03_diabetes_large_ml_vs_dl_scratch.ipynb`](diabetes_large/notebook/03_diabetes_large_ml_vs_dl_scratch.ipynb) | Hệ thống 1 — EDA, ColumnTransformer 14 chiều, 4 kiến trúc, 3 mô hình ML, tinh chỉnh ngưỡng, PCA 4 tầng |
| 04 | [`house_price_large/notebook/04_house_price_large_ml_vs_dl_scratch.ipynb`](house_price_large/notebook/04_house_price_large_ml_vs_dl_scratch.ipynb) | Hệ thống 2 — biến đổi log, Smooth Bayesian Target Encoding 4 cấp địa lý, 13 đặc trưng, phân tích phần dư |
| 05 | [`customer_comments/notebook/05_comments_large_ml_vs_dl_scratch.ipynb`](customer_comments/notebook/05_comments_large_ml_vs_dl_scratch.ipynb) | Hệ thống 3 — TF-IDF 1.000 chiều ngram(1,2), 4 kiến trúc văn bản, tối ưu Macro F1, PCA ngữ nghĩa |

Notebook được sinh từ tệp nguồn `_src_*.py` qua [`tools/py2nb.py`](tools/py2nb.py), nên diff được bằng Git thay vì phải so sánh JSON của `.ipynb`.

---

## Cấu trúc thư mục

```text
Assignment-03-Intelligent-System/
├── README.md
├── requirements.txt
├── .env.example                       # mẫu cấu hình Neo4j (.env không commit)
│
├── diabetes_baseline/                 # Chương 2 — Pima 768 mẫu
│   ├── data/pima_diabetes.csv
│   ├── notebook/                      # notebook 01 + 02 và tệp nguồn _src_*.py
│   └── reports/figures/               # 4 hình + 3 tệp metrics JSON
│
├── diabetes_large/                    # Chương 3 — 100.000 hồ sơ
│   ├── data/diabetes_prediction_dataset.csv
│   ├── notebook/03_diabetes_large_ml_vs_dl_scratch.ipynb
│   ├── model/
│   │   ├── dl_scratch_weights.npz     # trọng số mạng nơ-ron thuần NumPy
│   │   ├── random_forest.joblib
│   │   ├── logistic_regression.joblib
│   │   └── metadata.json              # nguồn số liệu duy nhất cho API và báo cáo
│   ├── api/rest_api.py                # Flask, cổng 5001
│   ├── web/index.html
│   ├── mobile/index.html + lib/main.dart + pubspec.yaml
│   └── reports/figures/               # 7 hình
│
├── house_price_large/                 # Chương 4 — 150.000 giao dịch (cổng 5002)
├── customer_comments/                 # Chương 5 — 23.486 nhận xét (cổng 5003)
│
├── knowledge_graph/                   # tầng tri thức Neo4j dùng chung
│   ├── knowledge.py                   # truy vấn Cypher, suy biến an toàn khi mất kết nối
│   └── seed_knowledge_graph.py        # nạp dữ liệu tư vấn vào đồ thị
│
├── tools/py2nb.py                     # chuyển _src_*.py thành .ipynb
│
(Báo cáo nằm ngoài thư mục này, tại ../../Report/Assignment 03/)
    ├── A03_CT_nghiand.600.pdf         # báo cáo cuối
    ├── build_report.py                # dựng HTML rồi in PDF bằng Chrome headless
    ├── chapters_theory.py             # Chương 1 – 2
    ├── chapters_systems.py            # Chương 3 – 5
    ├── chapters_final.py              # Chương 6 – 8
    ├── capture_screenshots.py         # tự động chụp giao diện Web/Mobile
    ├── figures/                       # hình gom từ ba hệ thống
    └── screenshots/                   # ảnh giao diện đưa vào Chương 7
```

---

## Chạy lại từ đầu

### 1. Cài phụ thuộc

```bash
pip install -r requirements.txt
playwright install chromium      # chỉ cần nếu muốn chụp lại ảnh giao diện
```

### 2. Chạy lại notebook

```bash
# Sinh .ipynb từ tệp nguồn rồi chạy
python tools/py2nb.py diabetes_baseline/notebook/_src_01_baseline.py \
                     diabetes_baseline/notebook/01_diabetes_dl_from_scratch_numpy.ipynb
jupyter nbconvert --to notebook --execute --inplace \
                  diabetes_baseline/notebook/01_diabetes_dl_from_scratch_numpy.ipynb
```

Lặp lại cho bốn notebook còn lại. Thứ tự bắt buộc: notebook 02 đọc `metrics_baseline.json` do notebook 01 sinh ra.

### 3. Nạp đồ thị tri thức (tuỳ chọn)

```bash
cp .env.example .env        # điền NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
python knowledge_graph/seed_knowledge_graph.py
```

Không cấu hình Neo4j thì API vẫn dự đoán bình thường, chỉ thiếu phần tư vấn kèm theo.

### 4. Khởi động ba dịch vụ

```bash
python diabetes_large/api/rest_api.py         # http://127.0.0.1:5001
python house_price_large/api/rest_api.py      # http://127.0.0.1:5002
python customer_comments/api/rest_api.py      # http://127.0.0.1:5003
```

Mỗi dịch vụ tự phục vụ luôn giao diện Web tại `/` và giao diện Mobile tại `/mobile`.

### 5. Dựng lại báo cáo

```bash
python "../../Report/Assignment 03/capture_screenshots.py"   # tự khởi động API, chụp ảnh, tắt API
python "../../Report/Assignment 03/build_report.py"          # → Report/Assignment 03/A03_CT_nghiand.600.pdf
```

---

## REST API

| Hệ thống | Endpoint | Ví dụ payload |
|---|---|---|
| Tiểu đường | `POST /diabetes/v1/predict` | `{"gender":"Female","age":54,"hypertension":1,"heart_disease":0,"smoking_history":"former","bmi":32.1,"HbA1c_level":6.8,"blood_glucose_level":180}` |
| Bất động sản | `POST /house-price/v1/predict` | `{"bed":3,"bath":2,"house_size":1850,"acre_lot":0.25,"city":"Austin","state":"Texas","zip_code":"78704"}` |
| Nhận xét | `POST /comments/v1/predict` | `{"text":"I wanted to love this dress but the fabric feels cheap."}` |

Kiểm tra tình trạng dịch vụ: `GET /diabetes/v1/health`, `GET /house-price/v1/health`, `GET /comments/v1/health`.

Phản hồi gồm ba phần: kết quả dự đoán, thông tin mô hình đã dùng, và khối `knowledge` chứa tư vấn lấy từ đồ thị Neo4j (rỗng nếu chưa cấu hình).

---

## Nguồn dữ liệu

| Bộ dữ liệu | Quy mô | Nguồn |
|---|---|---|
| Pima Indians Diabetes | 768 | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Diabetes Prediction Dataset | 100.000 | [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| USA Real Estate Dataset | 150.000 | [Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) |
| Women's Clothing E-Commerce Reviews | 23.486 | [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) |

---

## Ghi chú về tính tái lập

Toàn bộ thực nghiệm cố định `SEED = 42`. Mọi tham số tiền xử lý — trung bình, độ lệch chuẩn, trung vị, từ điển TF-IDF, bảng Target Encoding — đều được học **duy nhất trên tập Train** rồi áp cố định sang tập Test. Ngưỡng quyết định cũng được dò trên tập Train.

**Về artifact được lưu.** Kho chỉ giữ những tệp mà khâu triển khai thật sự cần: `dl_scratch_weights.npz` của cả ba hệ thống, cộng `tfidf_vectorizer.joblib` cho hệ thống NLP. Riêng Random Forest của bài toán định giá nhà **không được lưu** — rừng 100 cây với `max_depth=22` trên 118.026 mẫu sinh ra hơn 3,5 triệu nút, tương đương một tệp joblib khoảng 244 MB, vượt giới hạn 100 MB mỗi tệp của GitHub trong khi REST API không hề dùng đến nó. Mọi chỉ số đối chuẩn của mô hình này đã nằm trong `metadata.json`; ai cần chính mô hình đó chỉ việc chạy lại ô huấn luyện ở mục 11 của notebook 04, mất khoảng 5 giây.

Mọi con số trong báo cáo được đọc tự động từ các tệp JSON do notebook sinh ra. Sửa notebook rồi chạy lại `build_report.py` là báo cáo tự cập nhật theo — không có chỗ nào để một con số cũ nằm lại trong văn bản.
