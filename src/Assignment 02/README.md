# Assignment 02 — Từ Biểu diễn Dữ liệu đến Hệ thống Thông minh có thể Triển khai

**Học phần:** Phát triển các Hệ thống Thông minh — Học viện Công nghệ Bưu chính Viễn thông
**Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · D23CTPM01
**Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế

Ba ứng dụng học máy hoàn chỉnh, mỗi ứng dụng đi trọn chuỗi:

```
Dữ liệu thô → Tìm hiểu → Làm sạch → Biểu diễn → Học → Đánh giá → Lưu trữ → Triển khai
```

Toàn bộ hệ thống chạy **hoàn toàn trên máy cá nhân** (local) — không có bản deploy trên Internet. Tài liệu này hướng dẫn dựng lại từ đầu: từ notebook đến ba REST API, giao diện Web, giao diện Mobile và báo cáo PDF.

| Ứng dụng | Bài toán | Mô hình được chọn | Kết quả chính | Cổng | Endpoint dự đoán |
|---|---|---|---|---|---|
| **Diabetes** | Phân loại nhị phân | Random Forest | ROC-AUC **0,853** · Recall **0,775** | `5001` | `POST /diabetes/v1/predict` |
| **House Price** | Hồi quy | Gradient Boosting Regressor | R² **0,638** · MAE **1,004** tỷ VNĐ | `5002` | `POST /house-price/v1/predict` |
| **Customer Behavior** | Phân loại nhị phân | Logistic Regression (bảng + văn bản) | ROC-AUC **0,943** · F1 **0,935** | `5003` | `POST /customer-behavior/v1/predict` |

Cả ba dịch vụ dùng **cổng khác nhau** nên chạy đồng thời được — không phải tắt dịch vụ này để chạy dịch vụ kia. Chi tiết ở mục 6.

**Báo cáo đầy đủ:** [`Report/Assignment 02/A02_CT_nghiand.600.pdf`](../../Report/Assignment%2002/A02_CT_nghiand.600.pdf) (Học kỳ 1 năm học 2026 – 2027).

---

## Phát hiện chính

Ở cả ba ứng dụng, **thay đổi cách biểu diễn dữ liệu tạo ra khác biệt lớn hơn thay đổi thuật toán**:

- **Customer Behavior** — cùng một cách chia dữ liệu, mô hình chỉ dùng đặc trưng bảng đạt ROC-AUC 0,552; mô hình được đọc thêm văn bản đánh giá đạt 0,943 (cải thiện tương đối 71%). Multinomial Naive Bayes — mô hình đơn giản nhất trong sáu mô hình — khi đọc văn bản (ROC-AUC 0,935) vẫn vượt xa Random Forest 250 cây chạy trên đặc trưng bảng (ROC-AUC 0,530).
- **House Price** — năm đặc trưng phái sinh (`Total_Area`, `Room_Density`, `Frontage_Ratio`, `Area_per_Bedroom`, `Log_Area`) cải thiện R² ở mọi mô hình; rút tỉnh/huyện từ cột địa chỉ dạng văn bản tự do giải phóng hai trong số các biến giải thích mạnh nhất của bài toán (`Province`, `District` chiếm nhiều trọng số trong `feature_importance`).
- **Diabetes** — ba đặc trưng đắt tiền về mặt thu thập (Insulin, SkinThickness, BloodPressure) gần như không đóng góp thêm thông tin dự báo so với dùng 5 đặc trưng còn lại (chênh lệch ROC-AUC khi thêm ba cột này dao động từ -0,019 đến +0,006 tuỳ mô hình), nên giao diện chỉ cần hỏi 5 con số: Glucose, BMI, Age, Pregnancies, DiabetesPedigreeFunction.

Cả ba tập dữ liệu đều có **một vấn đề chất lượng mà công cụ kiểm tra tiêu chuẩn không phát hiện được**, và mỗi lần là một dạng khác nhau: giá trị thiếu mã hoá thành số 0 ở Glucose/BMI (Diabetes), 2.716 bản ghi trùng lặp gây rò rỉ giữa train và test (House Price, 30.229 dòng thô còn lại 27.513 dòng sạch), và rò rỉ nhãn qua cột `Rating` — riêng quy tắc "Rating ≥ 4" đã đạt accuracy 0,9365 mà không cần học gì (Customer Behavior, nên cột này bị loại khỏi đặc trưng).

---

## Sản phẩm bàn giao

- Ba notebook Jupyter (EDA → làm sạch → biểu diễn → huấn luyện → đánh giá → lưu artifact) cho Diabetes, House Price, Customer Behavior.
- Artifact tiền xử lý (`preprocessor.joblib`, riêng Customer Behavior có thêm bộ vector hoá TF-IDF) và toàn bộ mô hình `.joblib` đã huấn luyện, kèm `metadata.json` — nguồn dữ liệu duy nhất cho cả API lẫn báo cáo.
- Ba Flask REST API độc lập (cổng 5001/5002/5003), mỗi API tự phục vụ luôn giao diện Web và Mobile của ứng dụng đó.
- Ba giao diện Web (`web/index.html`) và ba giao diện Mobile — gồm trang mô phỏng khung điện thoại chạy trong trình duyệt (`mobile/index.html`) và mã nguồn Flutter thật (`mobile/lib/main.dart`).
- Báo cáo cuối dạng PDF dựng tự động từ dữ liệu thật: [`Report/Assignment 02/A02_CT_nghiand.600.pdf`](../../Report/Assignment%2002/A02_CT_nghiand.600.pdf), cùng script tái tạo báo cáo, tái tạo hình đồ thị tri thức Neo4j, và script tự động chụp ảnh giao diện.

---

## Cấu trúc thư mục

```text
Assignment-02-Intelligent-System/
├── README.md
├── requirements.txt                         # phụ thuộc dùng chung cho cả ba ứng dụng + notebook
├── .env.example                             # mẫu cấu hình Neo4j (.env không được commit)
├── .gitignore
│
├── diabetes/
│   ├── data/diabetes.csv
│   ├── notebook/Diabetes_A02.ipynb          # EDA, huấn luyện, đánh giá, lưu artifact
│   ├── model/
│   │   ├── preprocessor.joblib              # xử lý zero-as-missing (Glucose, BMI) + chuẩn hoá
│   │   ├── metadata.json                    # cột đặc trưng, nhãn lớp, toàn bộ điểm số, feature_importance
│   │   ├── logistic_regression.joblib
│   │   ├── knn.joblib
│   │   ├── decision_tree.joblib
│   │   ├── random_forest.joblib             # mô hình tốt nhất
│   │   └── svm_rbf.joblib
│   ├── api/REST_API.py                      # Flask, cổng 5001, kèm tầng tri thức Neo4j tuỳ chọn
│   ├── web/index.html                       # giao diện Web
│   ├── mobile/
│   │   ├── index.html                       # khung điện thoại mô phỏng, chạy thẳng trong trình duyệt
│   │   ├── lib/main.dart                    # mã nguồn Flutter thật
│   │   └── pubspec.yaml
│   └── requirements.txt                     # phụ thuộc riêng để chỉ chạy API Diabetes
│
├── house_price/
│   ├── data/house_prices.csv
│   ├── notebook/HousePrice_A02.ipynb
│   ├── model/
│   │   ├── preprocessor.joblib              # one-hot cho 6 cột phân loại + chuẩn hoá 11 cột số
│   │   ├── metadata.json                    # feature_columns, categorical_options, mae_by_price_band...
│   │   ├── linear_regression.joblib
│   │   ├── ridge_regression.joblib
│   │   ├── decision_tree_regressor.joblib
│   │   ├── random_forest_regressor.joblib
│   │   └── gradient_boosting_regressor.joblib  # mô hình tốt nhất
│   ├── api/REST_API.py                      # Flask, cổng 5002, tự tính lại 5 đặc trưng phái sinh
│   ├── web/index.html
│   └── mobile/
│       ├── index.html
│       ├── lib/main.dart
│       └── pubspec.yaml
│   (không có requirements.txt riêng ở cấp này — dùng requirements.txt cùng cấp house_price/)
│
├── customer_behavior/
│   ├── data/womens_ecommerce_reviews.csv
│   ├── notebook/CustomerBehavior_A02.ipynb
│   ├── model/
│   │   ├── preprocessor.joblib              # bộ tiền xử lý cho biểu diễn hybrid (bảng + TF-IDF)
│   │   ├── preprocessor_tabular.joblib      # bộ tiền xử lý riêng cho 3 mô hình chỉ-bảng
│   │   ├── tfidf_text_only.joblib           # vector hoá cho mô hình chỉ-văn-bản
│   │   ├── metadata.json                    # categorical_options, top_positive/negative_terms...
│   │   ├── logistic_regression_tabular.joblib
│   │   ├── decision_tree_tabular.joblib
│   │   ├── random_forest_tabular.joblib
│   │   ├── logistic_regression_hybrid.joblib   # mô hình tốt nhất
│   │   ├── linear_svm_hybrid.joblib
│   │   └── multinomial_nb_text.joblib
│   ├── api/REST_API.py                      # Flask, cổng 5003, chọn bộ tiền xử lý theo biểu diễn của mô hình
│   ├── web/index.html
│   └── mobile/
│       ├── index.html
│       ├── lib/main.dart
│       └── pubspec.yaml
│
├── requirements.txt (đã liệt kê ở trên) — mỗi thư mục diabetes/, house_price/, customer_behavior/
│   đều có requirements.txt riêng ghi rõ "chỉ cần khi chạy đúng API của ứng dụng đó"
│
(Báo cáo nằm ngoài thư mục này, tại ../../Report/Assignment 02/)
    ├── A02_CT_nghiand.600.pdf               # BÁO CÁO CUỐI
    ├── Assignment_02.html                   # nguồn HTML của báo cáo (build_report.py sinh ra)
    ├── build_report.py                      # bộ dựng báo cáo — đọc metadata.json + notebook đã chạy
    ├── chapters_theory.py, chapters_apps.py, chapters_final.py  # nội dung từng phần của báo cáo
    ├── paginate.py                          # đánh số trang cho bản in PDF
    ├── capture_screenshots.py               # tự động chạy 3 API + chụp ảnh giao diện Web/Mobile
    ├── make_neo4j_figures.py                # dựng lại 4 hình đồ thị tri thức từ dữ liệu Neo4j thật
    ├── assets/                              # cover-frame.png, ptit-logo.png — dùng cho trang bìa báo cáo
    ├── diagrams/                            # 9 file HTML nguồn của các hình minh hoạ lý thuyết
    └── figures/                             # toàn bộ hình đã chụp: 9 hình lý thuyết, 4 hình đồ thị tri
                                              # thức Neo4j, 12 ảnh chụp màn hình giao diện Web/Mobile
```

`HUONG-DAN-CHO-BAN/` (nếu bạn thấy thư mục này khi mở dự án cục bộ) là ghi chú vận hành cá nhân của tác giả, bị loại khỏi kho mã nguồn qua `.gitignore` — không phải một phần sản phẩm bàn giao.

---

## Yêu cầu môi trường

- Python 3.13 (đã kiểm thử trên 3.13.7), Windows 11.
- Trình duyệt hiện đại (Chrome, Edge...) để mở giao diện Web và Mobile.
- *(Tuỳ chọn)* Tài khoản Neo4j Aura — chỉ dùng cho tầng tri thức bổ sung của ứng dụng Diabetes. Không có cũng chạy được toàn bộ hệ thống.
- *(Tuỳ chọn)* Flutter SDK bản mới (kênh stable, Dart SDK ≥ 3.5.0) — chỉ cần nếu muốn chạy app trên thiết bị Android/iOS thật thay vì dùng trang mô phỏng trong trình duyệt.

Mở PowerShell tại thư mục gốc dự án (`Assignment-02-Intelligent-System/`) rồi tạo môi trường ảo:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`(.venv)` xuất hiện ở đầu dấu nhắc lệnh nghĩa là môi trường ảo đã được kích hoạt. `requirements.txt` ở gốc dự án cài đủ mọi thứ (Jupyter, scikit-learn, Flask, Neo4j driver...) để chạy cả notebook lẫn cả ba API. Nếu chỉ muốn chạy API của một ứng dụng mà không đụng tới notebook, có thể cài gọn hơn, ví dụ:

```powershell
pip install -r diabetes\requirements.txt
```

---

## Tái lập notebook

Chạy lại ba notebook sẽ sinh lại toàn bộ mô hình (`.joblib`), bộ tiền xử lý và `metadata.json` — đây là bước bắt buộc trước khi chạy REST API lần đầu, vì các file trong `model/` không tự sinh ra nếu chưa chạy notebook.

```powershell
jupyter nbconvert --to notebook --execute --inplace diabetes\notebook\Diabetes_A02.ipynb
jupyter nbconvert --to notebook --execute --inplace house_price\notebook\HousePrice_A02.ipynb
jupyter nbconvert --to notebook --execute --inplace customer_behavior\notebook\CustomerBehavior_A02.ipynb
```

Chạy từ thư mục gốc dự án, sau khi đã kích hoạt `.venv`. `--inplace` ghi kết quả (bảng, biểu đồ, output) ngược lại vào chính file `.ipynb` — đây cũng là nguồn mà `Report/Assignment 02/build_report.py` đọc lại code và kết quả để đưa vào báo cáo.

**Thời gian chạy khác nhau rõ rệt giữa ba notebook:** Diabetes nhanh nhất vì chỉ có 768 dòng dữ liệu và 5 mô hình đơn giản. House Price chậm hơn hẳn vì phải làm sạch ~30 nghìn dòng, mã hoá one-hot ra 111 cột rồi huấn luyện 5 mô hình (trong đó có Random Forest và Gradient Boosting) trên gần 19.258 dòng train. Customer Behavior thường lâu nhất vì có thêm bước vector hoá TF-IDF ra hàng chục nghìn chiều và huấn luyện 6 mô hình, hai trong số đó chạy trên ma trận hybrid (bảng + văn bản) hơn 20 nghìn cột. Không chạy song song ba lệnh trên cùng một máy nếu cấu hình yếu — Jupyter sẽ tranh CPU và làm cả ba chậm đi.

Mọi mô hình dùng chung `RANDOM_SEED = 42` (đặt cả `np.random.seed` lẫn `random.seed`, và truyền `random_state=RANDOM_SEED` vào từng bước chia tập/huấn luyện). Các mô hình chạy song song (`n_jobs=-1`, ví dụ Random Forest) có thể lệch ở chữ số thập phân thứ tư do thứ tự cộng dồn dấu phẩy động giữa các luồng; sai khác này không ảnh hưởng tới thứ hạng mô hình hay `best_model` được chọn.

**Về rò rỉ dữ liệu:** không bao giờ `fit` lại scaler, encoder, imputer hay TF-IDF trên dữ liệu người dùng lúc dự đoán. REST API chỉ nạp artifact đã fit sẵn trên tập train (bằng `joblib.load`) rồi gọi `transform` — không có bước `fit` nào chạy trong lúc phục vụ request. Mỗi notebook kết thúc bằng một phép khẳng định (`assert np.array_equal(...)`) rằng mô hình + bộ tiền xử lý nạp lại từ đĩa cho kết quả dự đoán trùng khớp hoàn toàn với mô hình đang nằm trong bộ nhớ, đo trên toàn bộ tập test (ví dụ notebook Diabetes kiểm tra trên 116 mẫu test). Nếu phép khẳng định này thất bại, artifact đã hỏng và REST API sẽ trả kết quả khác với thực nghiệm trong notebook.

---

## Chạy 3 REST API

Ba API độc lập, mỗi API dùng một cổng riêng (5001/5002/5003) nên **chạy đồng thời được cả ba** — khác với việc phải tắt dịch vụ này để bật dịch vụ kia. Mở ba cửa sổ PowerShell riêng biệt, mỗi cửa sổ đứng ở thư mục gốc dự án:

```powershell
# Cửa sổ 1
python diabetes\api\REST_API.py            # cổng 5001

# Cửa sổ 2
python house_price\api\REST_API.py         # cổng 5002

# Cửa sổ 3
python customer_behavior\api\REST_API.py   # cổng 5003
```

Mỗi lệnh sẽ giữ cửa sổ chạy liên tục (log Flask hiện `Running on http://0.0.0.0:500x`), Ctrl+C để dừng. Nếu artifact chưa được sinh (chưa chạy notebook ở mục 5), lệnh trên sẽ dừng ngay với `FileNotFoundError`.

---

## Danh sách endpoint

Cả ba dịch vụ dùng chung một bộ route, chỉ khác đường dẫn `predict` và cổng:

| Method & Path | Có ở cả 3 dịch vụ? | Trả về |
|---|---|---|
| `GET /` hoặc `GET /web` | Có | Trang giao diện Web (`web/index.html`) |
| `GET /mobile` | Có | Trang mô phỏng khung điện thoại (`mobile/index.html`) |
| `GET /health` | Có | `status`, `application`, danh sách mã mô hình đã nạp, `feature_columns`, `best_model` |
| `GET /metadata` | Có | Toàn bộ nội dung `model/metadata.json` — nguồn dữ liệu duy nhất mà Web/Mobile dùng để dựng form, bảng so sánh mô hình, mức độ quan trọng đặc trưng |
| `GET /models` | Có | Danh sách mô hình kèm điểm số đo trên tập test, đọc trực tiếp từ metadata (không hardcode số liệu trong code) |
| `POST /diabetes/v1/predict` | Chỉ Diabetes (5001) | Dự đoán nguy cơ tiểu đường |
| `POST /house-price/v1/predict` | Chỉ House Price (5002) | Dự đoán giá nhà |
| `POST /customer-behavior/v1/predict` | Chỉ Customer Behavior (5003) | Dự đoán khả năng khuyến nghị sản phẩm |

Cả ba endpoint `predict` đều nhận thêm khoá **tuỳ chọn** `"model"` trong JSON body — bỏ qua thì API tự dùng `best_model` (mô hình tốt nhất theo `metadata.json`); truyền một mã không hợp lệ sẽ trả lỗi 400 kèm danh sách mã hợp lệ.

> **Lưu ý PowerShell:** trong PowerShell, `curl` là alias của `Invoke-WebRequest` (cú pháp `-H`/`-d` khác hẳn cURL thật). Các ví dụ dưới đây dùng `curl.exe` (cURL thật, có sẵn từ Windows 10 trở lên) để chạy đúng như minh hoạ.

### 1. Diabetes — `POST /diabetes/v1/predict`

Yêu cầu đủ 5 trường số, không trường nào được để trống:

| Trường | Ràng buộc |
|---|---|
| `Glucose` | số, > 0 |
| `BMI` | số, > 0 |
| `Age` | số, 1–120 |
| `Pregnancies` | số, ≥ 0 |
| `DiabetesPedigreeFunction` | số, ≥ 0 |
| `model` *(tuỳ chọn)* | một trong `logistic_regression`, `knn`, `decision_tree`, `random_forest`, `svm_rbf` — mặc định `random_forest` |

```powershell
curl.exe -X POST http://127.0.0.1:5001/diabetes/v1/predict -H "Content-Type: application/json" -d '{"Glucose":180,"BMI":40,"Age":55,"Pregnancies":7,"DiabetesPedigreeFunction":1.1}'
```

```json
{
  "application": "diabetes",
  "model": "random_forest",
  "model_label": "Random Forest",
  "prediction_class": 1,
  "prediction": "Dương tính (nguy cơ cao)",
  "risk_level": "high",
  "confidence": 88.94,
  "probability_positive": 88.94,
  "interpretation": "Mô hình phát hiện các chỉ số sức khỏe tương đồng với nhóm có nguy cơ mắc tiểu đường cao. Nên tham khảo ý kiến bác sĩ để được xét nghiệm và tư vấn chính xác.",
  "risk_tier": "Nhóm Nguy cơ Cao",
  "knowledge": [
    { "title": "Hướng dẫn cần làm", "items": [ { "title": "...", "content": "...", "price": null, "url": null } ] }
  ],
  "knowledge_error": null
}
```

`knowledge` và `risk_tier` chỉ có nội dung khi Neo4j đã được cấu hình và kết nối được (mục 11); `knowledge_error` sẽ khác `null` nếu không.

### 2. House Price — `POST /house-price/v1/predict`

Chỉ `Area` bắt buộc; các trường số còn lại để trống thì API coi là chưa biết. Các trường phân loại phải khớp đúng một giá trị trong `categorical_options` (lấy từ `GET /metadata`), để trống sẽ tự dùng `"Không rõ"` nếu đó là một lựa chọn hợp lệ của trường đó:

| Trường | Bắt buộc | Ghi chú |
|---|---|---|
| `Area` | Có | số > 0, đơn vị m² |
| `Frontage`, `Access Road`, `Floors`, `Bedrooms`, `Bathrooms` | Không | số ≥ 0 nếu có nhập |
| `House direction`, `Balcony direction`, `Legal status`, `Furniture state`, `Province`, `District` | Không | chuỗi, phải khớp `categorical_options` |
| `model` *(tuỳ chọn)* | Không | một trong `linear_regression`, `ridge_regression`, `decision_tree_regressor`, `random_forest_regressor`, `gradient_boosting_regressor` — mặc định `gradient_boosting_regressor` |

API tự tính lại 5 đặc trưng phái sinh (`Total_Area`, `Room_Density`, `Frontage_Ratio`, `Area_per_Bedroom`, `Log_Area`) từ các trường cơ bản phía trên — client không cần gửi chúng.

```powershell
curl.exe -X POST http://127.0.0.1:5002/house-price/v1/predict -H "Content-Type: application/json" -d '{"Area":85,"Frontage":5,"Access Road":8,"Floors":4,"Bedrooms":4,"Bathrooms":3,"Legal status":"Have certificate","Furniture state":"Full","Province":"Hà Nội"}'
```

```json
{
  "application": "house_price",
  "model": "gradient_boosting_regressor",
  "model_label": "Gradient Boosting Regressor",
  "predicted_price": 7.42,
  "unit": "tỷ VNĐ",
  "price_low": 6.08,
  "price_high": 8.76,
  "interpretation": "Dựa trên các đặc điểm bất động sản đã nhập, mô hình Gradient Boosting Regressor ước tính giá trị khoảng 7.42 tỷ VNĐ, dao động trong khoảng 6.08 - 8.76 tỷ VNĐ."
}
```

`price_low`/`price_high` là khoảng `predicted_price ± residual_std` (độ lệch chuẩn phần dư đo trên tập test, 1,3447 tỷ VNĐ) — giao diện Web hiển thị khoảng này dưới dạng thước đo trực quan thay vì chỉ một con số, vì mô hình kém tin cậy hơn ở hai đầu phổ giá.

### 3. Customer Behavior — `POST /customer-behavior/v1/predict`

`Review Text` bắt buộc và không được rỗng; các trường phân loại để trống hoặc không hợp lệ sẽ tự chuyển thành `"Unknown"`:

| Trường | Bắt buộc | Ghi chú |
|---|---|---|
| `Review Text` | Có | chuỗi, không được rỗng |
| `Age` | Có | số, 1–120 |
| `Title` | Không | chuỗi, mặc định rỗng |
| `Positive Feedback Count` | Không | số ≥ 0, mặc định 0 |
| `Division Name`, `Department Name`, `Class Name` | Không | chuỗi, phải khớp `categorical_options` hoặc để trống |
| `model` *(tuỳ chọn)* | Không | một trong `logistic_regression_tabular`, `decision_tree_tabular`, `random_forest_tabular`, `logistic_regression_hybrid`, `linear_svm_hybrid`, `multinomial_nb_text` — mặc định `logistic_regression_hybrid` |

```powershell
curl.exe -X POST http://127.0.0.1:5003/customer-behavior/v1/predict -H "Content-Type: application/json" -d '{"Review Text":"Very disappointed. The material feels cheap and thin, it runs two sizes too small and looks nothing like the picture. Returned it.","Age":45,"Positive Feedback Count":0}'
```

```json
{
  "application": "customer_behavior",
  "model": "logistic_regression_hybrid",
  "model_label": "Logistic Regression (bảng + văn bản)",
  "prediction_class": 0,
  "prediction": "Không khuyến nghị sản phẩm",
  "confidence": 91.35,
  "representation": "Bảng + văn bản",
  "interpretation": "Dựa trên nội dung đánh giá và thông tin sản phẩm, mô hình dự đoán khách hàng khó có khả năng khuyến nghị sản phẩm này."
}
```

`representation` cho biết mô hình đang dùng loại đặc trưng nào (`Chỉ bảng` / `Bảng + văn bản` / `Chỉ văn bản`) — ba mô hình `*_tabular` không đọc `Review Text`, `multinomial_nb_text` chỉ đọc `Review Text` (bỏ qua Age/Division/Department/Class), chỉ hai mô hình `*_hybrid` dùng cả hai loại đặc trưng cùng lúc.

---

## Giao diện Web

| | Diabetes | House Price | Customer Behavior |
|---|---|---|---|
| Địa chỉ | http://127.0.0.1:5001/web | http://127.0.0.1:5002/web | http://127.0.0.1:5003/web |

Cả ba giao diện Web dùng chung một bộ tính năng, chỉ khác nội dung form và màu chủ đạo:

1. **Chọn mô hình** — ô chọn liệt kê toàn bộ mô hình đã huấn luyện (đọc từ `GET /metadata`), mặc định chọn sẵn `best_model` và đánh dấu "(khuyến nghị)".
2. **Bảng so sánh mô hình** — bảng chỉ số đo trên tập test cho tất cả mô hình cùng lúc (Accuracy/Precision/Recall/F1/ROC-AUC cho hai bài toán phân loại; MAE/RMSE/R² cho House Price), dòng ứng với mô hình đang chọn được tô đậm, mô hình tốt nhất có dấu ★.
3. **Thanh/thước đo độ tin cậy** — Diabetes và Customer Behavior hiển thị thanh phần trăm độ tin cậy (`confidence`) đổi màu theo kết quả (xanh/đỏ); House Price thay bằng một thước đo trực quan cho biết giá dự đoán nằm ở đâu trong khoảng `price_low`–`price_high`.
4. **Mức độ quan trọng của đặc trưng** — Diabetes và House Price vẽ thanh trọng số cho từng đặc trưng (từ `feature_importance` trong metadata), kèm đối chiếu giá trị người dùng vừa nhập; Customer Behavior thay bằng danh sách "từ khoá quan trọng nhất" (từ `top_positive_terms`/`top_negative_terms`), tô đậm những từ thực sự xuất hiện trong đoạn đánh giá vừa nhập.
5. **Nút điền ví dụ mẫu** — mỗi form có 2–3 hồ sơ mẫu (ví dụ Diabetes có "Nguy cơ cao / Trung bình / Nguy cơ thấp") để bấm điền nhanh toàn bộ form, kèm nút "Xoá form".
6. **Diễn giải bằng lời** — mỗi kết quả luôn kèm một đoạn văn tiếng Việt giải thích ý nghĩa dự đoán (trường `interpretation` trả về từ API), không chỉ hiện con số thô.

Diabetes còn hiển thị thêm khối "Kiến thức tham khảo" lấy từ Neo4j (mục 11) khi có cấu hình.

---

## Giao diện Mobile

`/mobile` phục vụ một trang **mô phỏng thiết bị 390×844 chạy thẳng trong trình duyệt** — có thanh trạng thái, thanh tiêu đề ứng dụng và bố cục cuộn riêng theo đúng kích thước một điện thoại thật. Trang này gọi **đúng REST endpoint** mà giao diện Web gọi (cùng `/metadata`, `/models`, `POST .../predict`) và có đầy đủ 6 tính năng ở mục trên (thu gọn vào các khối `<details>` để vừa màn hình), nên nó minh hoạ trọn vẹn luồng `Mobile → REST API → mô hình` **mà không cần cài trình giả lập Android**. Mở trực tiếp:

| | Diabetes | House Price | Customer Behavior |
|---|---|---|---|
| Địa chỉ | http://127.0.0.1:5001/mobile | http://127.0.0.1:5002/mobile | http://127.0.0.1:5003/mobile |

Trang này tự lấy `window.location.origin` làm địa chỉ API (có ô "Cấu hình API" để đổi thủ công nếu cần trỏ sang máy khác).

### Chạy ứng dụng Flutter thật

Mã nguồn Flutter thật nằm ở `<ứng-dụng>/mobile/lib/main.dart`, dùng package `http` để gọi REST API — giao diện và logic tương đương bản mô phỏng ở trên. Để chạy trên máy giả lập hoặc thiết bị thật:

```powershell
cd diabetes\mobile
flutter create .
flutter pub get
flutter run
```

Lặp lại tương tự cho `house_price\mobile` và `customer_behavior\mobile`. `flutter create .` chỉ cần chạy một lần đầu (sinh thư mục nền tảng `android/`, `ios/`... đã bị `.gitignore` loại khỏi kho mã nguồn).

Mỗi ứng dụng Flutter đã đặt sẵn địa chỉ API mặc định đúng cổng của nó:

| Ứng dụng | Địa chỉ mặc định trong `main.dart` |
|---|---|
| Diabetes | `http://10.0.2.2:5001` |
| House Price | `http://10.0.2.2:5002` |
| Customer Behavior | `http://10.0.2.2:5003` |

`10.0.2.2` là địa chỉ đặc biệt của **Android Emulator**, trỏ tới `localhost` của chính máy tính đang chạy emulator — giữ nguyên giá trị này nếu chạy bằng emulator. Chạy trên **điện thoại thật**, phải đổi ô cấu hình API trong app thành `http://<IPv4-LAN-của-máy-tính>:<cổng>` (xem cách lấy địa chỉ IPv4 ở mục dưới), và Windows Firewall trên máy tính phải cho phép Python nhận kết nối đến ở cổng đó.

---

## Truy cập từ điện thoại thật cùng mạng Wi-Fi

Không cần cài Flutter cũng kiểm thử được bằng điện thoại thật, qua chính giao diện Web/Mobile ở trình duyệt điện thoại.

**Bước 1 — lấy địa chỉ IPv4 của máy tính** (PowerShell, trên máy tính đang chạy API):

```powershell
ipconfig
```

Tìm dòng `IPv4 Address` trong phần card mạng đang dùng (Wi-Fi hoặc Ethernet), ví dụ `172.18.15.93`.

**Bước 2 — API vẫn phải đang chạy** (mục 6), ví dụ Diabetes ở cổng 5001.

**Bước 3 — trên điện thoại**, mở trình duyệt và truy cập:

```
http://172.18.15.93:5001/health
```

(thay `172.18.15.93` bằng IPv4 thật của máy bạn). Nhận được JSON `{"status": "ok", ...}` nghĩa là điện thoại đã kết nối được — tiếp tục mở `http://172.18.15.93:5001/web` hoặc `/mobile` để dùng giao diện đầy đủ.

Nếu không kết nối được:

- Điện thoại và máy tính phải **cùng một mạng Wi-Fi/LAN**, mạng không được bật chế độ cách ly thiết bị (client isolation — thường gặp ở Wi-Fi công cộng, ký túc xá, hoặc mạng khách của router).
- Windows Firewall phải cho phép `python.exe` nhận kết nối đến trên mạng **Private** — lần đầu chạy `python REST_API.py`, Windows thường tự hiện hộp thoại xin cấp quyền, chọn "Cho phép truy cập" (Allow access) cho cả mạng riêng tư.

> **Lưu ý:** `127.0.0.1` (hay `localhost`) trong trình duyệt trên điện thoại luôn là **chính điện thoại đó** — không bao giờ trỏ về máy tính. `10.0.2.2` cũng chỉ có nghĩa bên trong Android Emulator. Điện thoại thật bắt buộc phải dùng địa chỉ IPv4 LAN thật của máy tính, lấy từ `ipconfig`.

---

## Tầng tri thức Neo4j *(tuỳ chọn — chỉ cho Diabetes)*

Sau khi mô hình dự đoán xong xác suất dương tính, hệ thống ánh xạ xác suất đó sang một trong ba tầng nguy cơ (`med_high` ≥ 0,25, `med_moderate` ≥ 0,15, `med_low` còn lại) rồi tra đồ thị tri thức Neo4j để lấy nội dung tư vấn tương ứng (hướng dẫn cần làm, thiết bị theo dõi, dinh dưỡng hỗ trợ, dịch vụ đi kèm).

**Neo4j chỉ bổ sung nội dung, không bao giờ tham gia vào việc tính nhãn hay xác suất dự đoán.** Nếu chưa cấu hình hoặc mất kết nối, API vẫn trả về đúng `prediction`, `prediction_class` và `probability_positive` như bình thường — chỉ khác là `knowledge` rỗng và `knowledge_error` chứa một thông báo giải thích lý do.

Sao chép `.env.example` thành `.env` ở thư mục gốc dự án rồi điền thông tin kết nối thật:

```powershell
Copy-Item .env.example .env
notepad .env
```

```dotenv
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<mật khẩu của bạn>
NEO4J_DATABASE=neo4j
```

`.env` đã nằm trong `.gitignore` và không bao giờ được commit. Khởi động lại `python diabetes\api\REST_API.py` sau khi sửa `.env` để cấu hình mới có hiệu lực.

---

## Dựng lại báo cáo

```powershell
python report\build_report.py
```

Chạy từ thư mục gốc dự án. Lệnh trên sinh ra `report\Assignment_02.html`, sau đó tự tìm Chrome hoặc Edge trên máy để in file HTML đó ra `report\A02_CT_nghiand.600.pdf`. Mọi con số trong báo cáo được **đọc trực tiếp** từ `model/metadata.json` của ba ứng dụng, và mọi đoạn mã in trong báo cáo được **đọc ngược** từ file `.ipynb` đã chạy (mục 5) — không có con số hay dòng mã nào gõ tay vào văn bản báo cáo, nên sửa notebook rồi chạy lại notebook + lệnh trên là báo cáo tự cập nhật theo, không có chỗ nào để một con số cũ nằm lại.

Dựng lại 4 hình minh hoạ đồ thị tri thức Neo4j (sơ đồ bản thể học, toàn bộ đồ thị thật, một ca nguy cơ cao, sơ đồ luồng xử lý):

```powershell
python report\make_neo4j_figures.py
```

**Khác với REST API, script này bắt buộc phải kết nối được Neo4j thật** — nó đọc `.env` ở thư mục gốc, và nếu mất kết nối thì dừng lại và báo lỗi ngay, không tự bịa dữ liệu để vẽ hình. Cần hoàn tất mục 11 (tạo `.env`, instance Neo4j đang chạy) trước khi chạy lệnh này.

---

## Chụp lại ảnh giao diện

```powershell
pip install playwright
python report\capture_screenshots.py
```

Script này tự khởi động cả ba Flask API (`diabetes`, `house_price`, `customer_behavior`) làm tiến trình nền, chờ từng `/health` trả về 200, rồi dùng Chrome headless (thông qua thư viện Playwright, chạy thẳng bản Chrome đã cài trên máy — không tải thêm trình duyệt riêng) để điền sẵn dữ liệu mẫu vào từng form, bấm nút gửi, và chụp lại cả trạng thái "đã điền" lẫn "kết quả" cho cả hai giao diện Web và Mobile của ba ứng dụng, cộng thêm một ảnh chụp riêng bảng so sánh mô hình cho mỗi ứng dụng — tổng cộng 15 ảnh, lưu vào `report\figures\`.

`playwright` (gói Python) không nằm trong `requirements.txt` gốc vì chỉ script này cần — cài thêm bằng lệnh ở trên. Không cần chạy `playwright install` vì script trỏ thẳng tới Chrome hệ thống tại `C:\Program Files\Google\Chrome\Application\chrome.exe`; script sẽ báo lỗi nếu không tìm thấy Chrome ở đúng đường dẫn đó.

---

## Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Chạy `python .../REST_API.py` báo lỗi liên quan tới địa chỉ đã được dùng (`WinError 10048`), hoặc trình duyệt báo không kết nối được | Cổng đã bị một tiến trình khác chiếm — thường là một lần chạy trước chưa tắt hẳn | `netstat -ano \| findstr :5001` để tìm PID đang giữ cổng, rồi `taskkill /PID <PID> /F`, sau đó chạy lại API |
| `FileNotFoundError: Không tìm thấy file mô hình` hoặc `Không tìm thấy file metadata` khi khởi động API | Chưa chạy notebook để sinh artifact, hoặc notebook chạy lỗi giữa chừng nên `model/` thiếu file | Chạy lại đúng lệnh `jupyter nbconvert` tương ứng ở mục "Tái lập notebook", kiểm tra thư mục `model/` có đủ các file `.joblib` và `metadata.json` |
| `ModuleNotFoundError: No module named 'flask'` (hoặc `pandas`, `sklearn`...) | Chưa kích hoạt đúng `.venv`, hoặc cài thiếu gói | Kiểm tra dấu nhắc lệnh có tiền tố `(.venv)`; nếu không, chạy `.venv\Scripts\activate`; rồi `pip install -r requirements.txt` |
| Diabetes API vẫn chạy, dự đoán vẫn ra kết quả, nhưng `knowledge_error` luôn có nội dung | Chưa tạo `.env`, sai `NEO4J_PASSWORD`, hoặc instance Neo4j Aura miễn phí đã tự tạm dừng do không hoạt động lâu ngày | Đăng nhập console Neo4j Aura, resume instance nếu đang tạm dừng, cập nhật lại `.env` nếu URI/mật khẩu thay đổi, khởi động lại API. Đây **không phải lỗi nghiêm trọng** — nhãn và xác suất dự đoán vẫn đúng như bình thường |
| `python report\build_report.py` chạy xong, có file HTML nhưng không có PDF, log in "Không tìm thấy Chrome/Edge — bỏ qua bước xuất PDF" | Máy không cài Chrome/Edge ở đường dẫn chuẩn | Cài Google Chrome (hoặc dùng Edge có sẵn trên Windows 11 nếu đủ điều kiện); hoặc mở thủ công `report\Assignment_02.html` bằng trình duyệt rồi in ra PDF bằng Ctrl+P |
| `python report\make_neo4j_figures.py` dừng ngay và báo lỗi, không sinh hình nào | Script này bắt buộc kết nối Neo4j thật (không giống REST API — không tự bịa dữ liệu khi mất kết nối) | Hoàn tất mục "Tầng tri thức Neo4j" trước: `.env` đã điền đúng và instance Neo4j Aura đang chạy, không bị tạm dừng |
| `python report\capture_screenshots.py` báo lỗi liên quan tới `playwright` hoặc không mở được Chrome | Thiếu gói `playwright` (không có sẵn trong `requirements.txt` gốc), hoặc Chrome không cài đúng đường dẫn mặc định | `pip install playwright` (không cần `playwright install` vì script dùng Chrome hệ thống); cài Google Chrome vào đường dẫn mặc định nếu chưa có |
| Gõ `curl.exe` với dữ liệu tiếng Việt (ví dụ `"Hà Nội"`) nhưng PowerShell hiển thị hoặc gửi đi ký tự sai | PowerShell console cổ điển không mặc định dùng UTF-8 | Chạy `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` trước khi gọi `curl.exe`, hoặc dùng Windows Terminal, hoặc kiểm thử qua giao diện Web thay vì gõ tay JSON |
| Điện thoại không mở được `http://<IPv4>:500x/health` dù cùng Wi-Fi với máy tính | Windows Firewall chặn Python, hoặc mạng đang bật chế độ cách ly thiết bị (client isolation) | Cho phép `python.exe` qua Windows Firewall ở mạng Private (hộp thoại thường tự hiện lần đầu chạy API); nếu ở mạng công cộng/ký túc xá có cách ly thiết bị, thử phát Wi-Fi từ chính điện thoại rồi nối máy tính vào mạng đó |

---

## Tham chiếu bộ dữ liệu

| Ứng dụng | File dữ liệu | Tập dữ liệu | Nguồn Kaggle |
|---|---|---|---|
| Diabetes | `diabetes/data/diabetes.csv` | Pima Indians Diabetes Database | [uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| House Price | `house_price/data/house_prices.csv` | Vietnam Housing Dataset 2024 | [nguyentiennhan/vietnam-housing-dataset-2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024) |
| Customer Behavior | `customer_behavior/data/womens_ecommerce_reviews.csv` | Women's E-Commerce Clothing Reviews | [nicapotato/womens-ecommerce-clothing-reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) |

---

## Lưu ý và hạn chế

- **Diabetes** — dữ liệu chỉ gồm phụ nữ gốc Pima từ 21 tuổi trở lên; ngưỡng quyết định của mô hình không suy rộng thẳng cho quần thể khác (nam giới, trẻ em, dân tộc khác). Dự đoán chỉ hỗ trợ sàng lọc tham khảo, **không thay thế chẩn đoán y khoa**.
- **House Price** — `Price` trong dữ liệu là giá chào bán trên tin đăng, không phải giá giao dịch thành công thực tế; tập dữ liệu bị chặn trên khoảng 11,5 tỷ VNĐ nên mô hình không đáng tin cậy cho phân khúc siêu cao cấp. Mô hình có xu hướng co dự đoán về giá trị trung bình nên kém chính xác hơn ở hai đầu phổ giá (dưới 3 tỷ và trên 9 tỷ) — vì vậy giao diện luôn hiển thị khoảng dao động thay vì chỉ một con số duy nhất.
- **Customer Behavior** — tập dữ liệu không có `Customer ID`, nên mọi kết luận là ở **cấp lượt đánh giá** (mỗi review độc lập), không phải cấp khách hàng — không thể dùng để suy ra hành vi dài hạn của một khách hàng cụ thể.
