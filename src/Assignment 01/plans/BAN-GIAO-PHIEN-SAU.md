# Bàn giao — Bắt đầu phiên mới từ đâu

Cập nhật: 26/08/2026 · Kho mã: https://github.com/Nghaiz/Assignment-01-Intelligent-System

---

## Câu lệnh mở đầu phiên mới

Dán nguyên đoạn này:

> Đọc `plans/BAN-GIAO-PHIEN-SAU.md` và `plans/README.md`, rồi làm tiếp Phase 9.

---

## Đã xong (P0 → P8)

| Phase | Sản phẩm | Bằng chứng |
|-------|----------|-----------|
| P0 | Bộ khung dự án, môi trường, kho GitHub công khai | 13 mô hình đã lên GitHub, repo `main` |
| P1 | Làm sạch dữ liệu, biểu diễn đặc trưng, 36 biểu đồ | `figures/*.png` |
| P2 | 5 + 5 mô hình, đều vượt mô hình cơ sở | `models/*.joblib`, 17 MB |
| P3 | 4 thực nghiệm có kiểm soát | `outputs/*.csv`, 9 bảng |
| P4 | Hai notebook, đủ 22 mục, chạy sạch | `notebooks/*.ipynb` |
| P5 | Đồ thị Tri thức, đã nạp lên Neo4j Aura thật | `outputs/graph.json`, 117 nút / 133 cạnh trên máy chủ |
| P6 | Ứng dụng web Streamlit ba trang, bốn tab mỗi trang công cụ | `app/`, chạy `streamlit run app/app.py` |
| P7 | Deploy công khai, không cần cài gì để xem | Bản chạy trên Streamlit Cloud |
| P8 | Báo cáo LaTeX 72 trang, biên dịch bằng XeLaTeX | `report/build/main.pdf` |

Một lệnh tái tạo **toàn bộ**, kể cả đồ thị tri thức: `python run_pipeline.py`
(khoảng 100 giây). Cờ `--kg-offline` để không đụng mạng, `--skip-kg` để bỏ hẳn
bước đồ thị. Chỉ dựng lại đồ thị mà không huấn luyện lại: `python -m src.kg.build_graph`.
Bộ kiểm thử: `python -m pytest` — 289 bài, khoảng 15 giây, phải xanh hết.
Kiểm tra hai notebook: `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`

### Số liệu chốt để viết báo cáo

**Tiểu đường** — 768 quan sát, 8 đặc trưng gốc + 4 đặc trưng thiết kế.
Chia 552 / 139 / 77 (huấn luyện / kiểm tra / ngoại kiểm).

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|---------:|----------:|-------:|---:|--------:|
| Cơ sở (đoán lớp đa số) | 0.6547 | 0.0000 | 0.0000 | 0.0000 | — |
| K-Nearest Neighbors | 0.7842 | 0.6957 | 0.6667 | **0.6809** | 0.8462 |
| Logistic Regression | 0.7914 | 0.7436 | 0.6042 | 0.6667 | 0.8709 |
| XGBoost | 0.7194 | 0.5957 | 0.5833 | 0.5895 | 0.7905 |
| Random Forest | 0.7194 | 0.6154 | 0.5000 | 0.5517 | 0.8191 |
| Decision Tree | 0.7266 | 0.6786 | 0.3958 | 0.5000 | 0.7446 |

Ngưỡng tối ưu θ = 0.15 trên XGBoost: Recall 0.4583 → **0.8542**, F1 đạt đỉnh
0.6508. Số láng giềng tối ưu k = 5.

**Bất động sản** — 30.229 tin đăng, 9 đặc trưng định lượng + 5 định danh.
Chia 21.764 / 5.442 / 3.023.

| Mô hình | MAE | RMSE | R² |
|---------|----:|-----:|---:|
| Cơ sở (đoán trung vị) | 1.8587 | 2.2265 | −0.0008 |
| XGBoost | **1.0493** | **1.3826** | **0.6141** |
| Random Forest | 1.0818 | 1.4207 | 0.5925 |
| Support Vector Regression | 1.0743 | 1.4451 | 0.5784 |
| Decision Tree | 1.2412 | 1.6589 | 0.4444 |
| Linear Regression | 1.2677 | 1.7865 | 0.3557 |

Độ sâu tối ưu của rừng: 24. Bỏ hết đặc trưng định danh làm R² tụt còn 0.3340 —
vị trí hành chính là nhóm thông tin quyết định.

---

## Sáu phát hiện phải viết vào phần Phân tích khoa học của báo cáo

Ba phát hiện đầu có từ P3; ba phát hiện sau phát sinh khi viết notebook ở P4 và
**bác bỏ giả định ban đầu**, nên cần đọc kỹ trước khi viết báo cáo.

**1. Rò rỉ nhãn qua bước điền khuyết.** Điền khuyết bằng trung vị nhóm theo
`Outcome` đẩy Accuracy lên 0.8705 — vượt xa trần 0.77–0.79 mà tài liệu công bố
trên tập Pima. Nguyên nhân: `Insulin` khuyết 374/768 và `SkinThickness` khuyết
227/768, nên gần một nửa dữ liệu mang dấu vết của chính nhãn cần dự đoán. Đưa
`SimpleImputer` vào trong Pipeline thì còn 0.7914.

**2. Quét ngưỡng cần mô hình cho xác suất liên tục.** Chạy trên Decision Tree
sâu 6 tầng, bảy ngưỡng từ 0.50 xuống 0.20 cho ra đúng một kết quả vì cây nông
chỉ trả về vài giá trị xác suất rời rạc. Chuyển sang XGBoost mới thấy được hiệu
ứng đánh đổi Precision–Recall.

**3. Cực trị chạm biên dải khảo sát thì chưa phải cực trị.** Xảy ra hai lần —
độ sâu rừng dừng ở 24 và ngưỡng dừng ở 0.15. Phải nới dải mới xác nhận được.

**4. "Biểu diễn quan trọng hơn thuật toán" là SAI nếu phát biểu trần.** Cùng một
thiết kế Thực nghiệm 3 cho hai kết luận trái ngược trên hai bài toán:

| | Tiểu đường | Bất động sản |
|---|---|---|
| Biên độ do đổi **biểu diễn** | 0.0300 F1 | 0.2563 R² |
| Biên độ do đổi **thuật toán** | 0.1809 F1 | 0.2584 R² |
| Bên nào chi phối | **Thuật toán** (gấp 6 lần) | **Biểu diễn** (ngang bằng, ít công hơn) |

Điều phân biệt: biểu diễn quyết định khi nó **thêm hoặc bớt thông tin** (biến định
danh ở bài 2 không suy ra được từ cột nào khác), không phải khi nó chỉ **đổi dạng**
thông tin đã có (chuẩn hoá, rời rạc hoá, nhân hai cột — bài 1). Bốn phương án biểu
diễn ở bài 1 chỉ phân loại khác nhau 1–9 ca trên 139, tức nằm trong nhiễu lấy mẫu.

Hệ quả phụ đáng chú ý: `Glucose_BMI_Risk` vừa là đặc trưng tương quan mạnh nhất với
nhãn (0.5199, vượt cả `Glucose` gốc) vừa là đặc trưng quan trọng nhất của XGBoost
(0.1874), **nhưng lại làm Logistic Regression kém đi** (F1 0.6818 → 0.6667) vì đa
cộng tuyến dưới phạt L2. "Đặc trưng tốt" là thuộc tính của **cặp đặc trưng – thuật
toán**, không phải của riêng đặc trưng.

**5. `Price` của tập bất động sản đã bị cắt biên từ nguồn.** Giá nằm gọn trong
1 – 11.5 tỷ VNĐ, skewness −0.03, kurtosis −0.85, trung bình 5.872 ≈ trung vị 5.900.
Giả định "giá nhà lệch phải mạnh" đúng với thị trường thật nhưng **sai** với tập
này. Hệ quả: R² ở đây khắt khe hơn (phương sai mục tiêu bị thu hẹp), và hệ thống
chỉ áp dụng được trong đúng dải 1 – 11.5 tỷ vì mô hình cây không ngoại suy.

**6. Mô hình bất động sản có độ chệch đơn điệu theo phân khúc giá.**

| Phân khúc | Giá thật TB | Giá dự đoán TB | Độ chệch | MAE |
|---|---:|---:|---:|---:|
| Q1 rẻ nhất | 2.685 | 3.734 | **+1.049** | 1.158 |
| Q3 giữa | 5.919 | 6.161 | +0.242 | 0.789 |
| Q5 đắt nhất | 8.977 | 7.493 | **−1.484** | 1.534 |

Hồi quy về trung bình cộng với việc cây quyết định không ngoại suy. Cần cảnh báo
người dùng: với bất động sản cao cấp, con số ước lượng là **cận dưới**.

---

## Số liệu Đồ thị Tri thức để viết báo cáo

Sáu ca mẫu — mỗi tầng nguy cơ và mỗi phân khúc thị trường một ca — chọn từ dữ
liệu thật, xác suất và giá do mô hình trong `models/` dự báo. Ca được lấy ở
**trung vị** của dải nên tái lập được.

| Đại lượng | Giá trị |
|---|---|
| Số thực thể trên máy chủ (\|V\|) | 95 |
| Số cạnh (\|E\|) | 96 |
| Số loại quan hệ (\|R\|) | 21 |
| Số loại thực thể | 17 |
| Máy chủ | Neo4j Kernel 5.27-aura, enterprise |

Chuỗi suy luận đầy đủ, kiểm chứng bằng truy vấn Cypher chạy thật:

- **Y sinh** — bệnh nhân #192, Glucose 159 mg/dL, BMI 30.4 → K-Nearest Neighbors
  cho xác suất 54.5% → ICD-10 E11 → Nhóm Nguy cơ Cao → ba hướng dẫn lâm sàng,
  hai thiết bị, hai sản phẩm dinh dưỡng, chuyển khám Nội tiết.
- **Bất động sản** — tin #23933, 100 m², 4 tầng, Nhà Bè TP.HCM → XGBoost định
  giá 8.42 tỷ → Phân khúc Cao cấp → Gói vay Cao cấp, LTV 70%, **hạn mức 5.89 tỷ**.

Đây là câu trả lời cho câu hỏi "đồ thị để làm gì": mô hình trả về một con số,
đồ thị biến con số ấy thành một hạn mức vay cụ thể và một danh sách việc phải làm.

---

## Việc còn lại (P9)

### Phase 9 — Hoàn thiện và nộp bài
Chỉ còn đối chiếu chéo trước khi nộp. Danh sách kiểm tra nằm ở
`plans/phase-9-hoan-thien.md`. Ba việc đáng chú ý:

1. **Bản PDF của báo cáo không nằm trong git** — `.gitignore` chặn
   `report/build/`. Dựng lại bằng `latexmk -xelatex -outdir=build main.tex`
   trong thư mục `report/` (khoảng 40 giây, cần MiKTeX). Nếu muốn nộp kèm PDF
   thì phải quyết định có bỏ dòng chặn ấy đi hay không.
2. **Đối chiếu R1–R14** đã có sẵn thành bảng trong báo cáo, Mục 4.8 — chỉ cần
   đọc lại chứ không phải làm mới.
3. **Kiểm tra notebook trên máy sạch** bằng
   `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.

---

## Đã bổ sung ở phiên Phase 8

Ngoài bản báo cáo, phiên này mở rộng cả pipeline lẫn ứng dụng web:

| Hạng mục | Trước | Sau |
|---|---:|---:|
| Biểu đồ | 36 | **44** |
| Bài kiểm thử | 289 | **355** |
| Đồ thị Tri thức | 95 nút / 96 cạnh / 21 quan hệ | **117 / 133 / 28** |
| Số tầng của đồ thị | 4 | **5** ở cả hai miền |
| Tab mỗi trang công cụ | 3 | **4** |

Năm hình mới: `dia_roc_curves`, `dia_confusion_grid`, `hou_residual_analysis`,
`hou_price_area_bedrooms`, và hai sơ đồ `fig_kg_architecture_*`.

Hai mô-đun tri thức miền mới, cả hai đều là **danh mục biên soạn offline** chứ
không phải trình cào dữ liệu — giá bán lẻ đổi theo ngày nên cào trực tiếp sẽ phá
vỡ cam kết tái lập của R14:

- `src/kg/catalog_pharmacy.py` — 10 mặt hàng có thật tại FPT Long Châu, mã SKU,
  khoảng giá **có ghi ngày chốt** và đường dẫn tra cứu giá hiện hành; ba gói
  chăm sóc theo tầng nguy cơ, tách chi phí ban đầu khỏi chi phí duy trì.
- `src/kg/catalog_urban.py` — cụm tiện ích hạ tầng theo phân khúc, và phép tính
  tài chính bằng công thức niên kim (trả góp, tổng lãi, thu nhập tối thiểu theo
  ngưỡng DTI). Phần này là toán học kiểm chứng được, khác với phần điều kiện vay
  vốn chỉ là minh hoạ theo mặt bằng lãi suất công bố.

**Nguyên tắc SSOT đã áp:** `RISK_TIERS` và `PROPERTY_SEGMENTS` không còn khai
lại tên mặt hàng, lãi suất hay danh sách tiện ích — chúng suy ra từ hai danh mục
trên lúc nạp mô-đun. Nhờ vậy con số hiện trên giao diện web không thể lệch con
số dùng để dựng đồ thị.

---

## Cái gì cần bạn chuẩn bị

| Việc | Khi nào | Hướng dẫn | Bắt buộc? |
|------|---------|-----------|-----------|
| Tài khoản Neo4j Aura Free | Xong rồi | `setup-guides/01-neo4j-aura.md` | — |
| Tài khoản Streamlit Cloud | Trước P7 | `setup-guides/02-streamlit-cloud.md` | Không — chạy local được |
| GitHub | Xong rồi | — | — |

**Một việc nhỏ trước P7:** Streamlit Cloud không đọc được `.env` (tệp đó bị
`.gitignore` chặn, đúng như phải thế). Ba biến Neo4j phải chép sang mục
**Settings → Secrets** của ứng dụng trên Streamlit Cloud. Mã đã chuẩn bị sẵn cho
việc này — biến môi trường thật luôn thắng giá trị trong tệp.

---

## Bẫy đã biết, tránh giẫm lại

- `vietnam_housing_dataset.csv` có BOM ở đầu, phải đọc bằng `utf-8-sig`.
- Vẽ biểu đồ thì đặt nhãn chú thích theo toạ độ hệ trục, không theo giá trị dữ
  liệu, nếu không matplotlib sẽ giãn khung hình ra một khoảng trắng lớn.
- Mô hình lưu bằng `joblib` phải bật `compress=3`, nếu không tệp vượt giới hạn
  100 MB của GitHub.
- Trước khi kết luận một siêu tham số là tối ưu, kiểm tra nó không nằm ở đầu
  mút dải khảo sát.
- Notebook và `run_pipeline.py` cùng ghi ra `figures/*_representation.png`. Nhãn
  phương án trong Thực nghiệm 3 phải trùng khít giữa hai nơi, nếu không hình sẽ
  phụ thuộc vào việc chạy cái nào sau cùng.
- Notebook không được ghi đè `models/`. Mỗi lần ghi là 17 MB thay đổi trong git
  mà nội dung không khác gì. Mục 18 của notebook **nạp lại** mô hình đã lưu rồi
  đối chiếu dự đoán — vừa gọn vừa chứng minh được tính tái lập.
- `src/viz.py` dùng backend `Agg` nên gọi hàm vẽ trong notebook sẽ không hiện
  hình. Gọi hàm rồi hiển thị lại tệp PNG bằng `IPython.display.Image`.
- `Pipeline.get_feature_names_out()` **ném lỗi** trên bộ tiền xử lý của bài toán
  tiểu đường vì `FunctionTransformer` không khai báo `feature_names_out`. Lấy tên
  cột từ `named_steps["scale"].feature_names_in_` thay thế.
- **`python run_pipeline.py` chạy lại KHÔNG cho ra tệp trùng byte.** `models/`
  thì trùng khít, nhưng `hou_depth_sweep.csv` và `hou_representation.csv` lệch
  ở chữ số cuối của số thực (bậc 1e-15) do phép cộng dồn song song trong rừng
  ngẫu nhiên không có tính kết hợp. Mọi con số ở độ chính xác báo cáo (4 chữ số
  thập phân) đều trùng. Đừng commit phần nhiễu ấy, và khi đối chiếu R14 ở
  Phase 9 thì so ở độ chính xác báo cáo chứ đừng so `git diff` trống.
- **Hàm đọc `.env` không được ghi vào `os.environ`.** Bản đầu dùng
  `os.environ.setdefault` cho tiện; hậu quả là một bài kiểm thử nạp tệp `.env`
  giả làm mọi bài chạy sau nó thừa hưởng giá trị giả rồi **tự bỏ qua trong im
  lặng** — bộ kiểm thử vẫn xanh mà bảy bài quan trọng nhất không hề chạy. Thứ
  tự ưu tiên giữa tệp và môi trường quyết định ở một chỗ nhìn thấy cả hai nguồn.
- Neo4j Aura bản hiện hành dùng **mã instance** làm tên đăng nhập và làm luôn
  tên cơ sở dữ liệu. Sửa thành `neo4j` cho "đúng tài liệu" sẽ nhận `AuthError`
  hoặc `DatabaseNotFound`. Cứ chép nguyên tệp Aura tải về vào `.env`.
- `NaN` của pandas **không phải** `None` của Python. Chỗ nào kiểm tra thiếu dữ
  liệu bằng `is None` mà nhận thẳng giá trị từ DataFrame là chỗ đó sẽ sinh ra
  những nhãn "= nan". Trình điều khiển Neo4j cũng không nuốt được `numpy.int64`.
  Đổi kiểu ngay tại ranh giới, đừng đổi rải rác.
- Bản thể học đặt tên nút cố định (`patient`, `model`, `prediction`) cho mọi ca,
  nên nạp nhiều ca lên chung một cơ sở dữ liệu phải gắn tiền tố định danh theo
  ca, nếu không sáu ca sẽ gộp thành một.
- **Streamlit Cloud nạp lại tệp trang nhưng KHÔNG nạp lại module đã import.** Sau
  khi push một hàm mới vào `app/lib.py`, bản deploy chạy tệp trang mới (tab mới
  hiện ra) nhưng vẫn giữ đối tượng `lib` cũ trong bộ nhớ, nên báo
  `AttributeError` ở đúng hàm vừa thêm. Chờ bao lâu cũng không tự khỏi. Cách
  buộc dựng lại cả môi trường: đổi `requirements.txt` rồi push — hoặc bấm
  Reboot trong bảng Manage app. Mất khoảng 5–10 phút.
- **Playwright không chụp được ảnh trang deploy qua vỏ bọc streamlit.app** —
  khung nhúng trạng thái của Streamlit Cloud không bao giờ ngừng tải nên lệnh
  chụp luôn hết giờ. Mở thẳng URL bên trong (`/~/+/<ten_trang>`) thì chụp được.
- Kiểm tra giả định về phân bố dữ liệu **trước khi** viện dẫn nó làm lý do thiết
  kế. Giả định "giá nhà lệch phải" nghe hiển nhiên tới mức suýt trôi qua mà không
  ai đo (phát hiện số 5).
