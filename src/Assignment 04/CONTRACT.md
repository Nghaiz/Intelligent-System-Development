# Assignment 04 — Hợp đồng tích hợp (đọc kỹ trước khi viết code)

Tài liệu này là **nguồn chân lý duy nhất** cho mọi agent làm việc song song trên Assignment 04.
Mọi tên file, tên khóa JSON, tên figure phải khớp **chính xác từng ký tự** với đặc tả dưới đây,
vì trình dựng báo cáo (`Report/Assignment 04/build_report.py`) đọc chúng theo tên.

---

## 0. Bối cảnh môn học

- Học phần: Phát triển các Hệ thống Thông minh — Học viện Công nghệ Bưu chính Viễn thông
- Sinh viên: **Nguyễn Duy Nghĩa** · **B23DCCN600** · Lớp **D23CTPM01**
- Giảng viên: **PGS.TS Trần Đình Quế**
- Học kỳ: Học kỳ 1 năm học 2026 – 2027
- Đề bài Assignment 04: **Convolutional Neural Networks — From Mathematical Convolution to
  NumPy, PyTorch and TensorFlow.** Sinh viên phải hiện thực CNN bằng ba cách:
  (1) NumPy thuần từ đầu, (2) PyTorch, (3) TensorFlow/Keras — trên dữ liệu bảng, văn bản và ảnh.

## 1. Môi trường

Toàn bộ chạy trong **một venv duy nhất**: `D:/Python/Intelligent-System-Development/.venv`.
Gọi bằng `python` hoặc `python -m pip` từ thư mục gốc dự án. **Không dùng lệnh `pip` trần**, vì nó
trỏ sang một venv khác ở thư mục cha và đã gây ra một giờ chẩn đoán nhầm.

| Thư viện | Phiên bản | Thiết bị |
|---|---|---|
| Python | 3.13 | |
| NumPy | 2.5.1 | CPU (theo định nghĩa) |
| pandas | 3.0.5 | |
| scikit-learn | 1.9.0 | |
| SciPy | 1.18.0 | |
| **PyTorch** | **2.13.0+cu126** | **GPU: RTX 4060 Laptop, 8,6 GB, CUDA 12.6** |
| TensorFlow / Keras | 2.21.0 / 3.15.1 | **CPU**, vì TF từ 2.11 bỏ hỗ trợ GPU native trên Windows |

### Quy tắc thiết bị, bắt buộc tuân thủ

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "Phai chay tren GPU; kiem tra lai venv"
```

Mọi notebook PyTorch **phải** dùng GPU và **phải in ra tên thiết bị** trong output. Đo thời gian
thì bọc `torch.cuda.synchronize()` trước và sau, nếu không sẽ đo nhầm thời gian xếp hàng lệnh
chứ không phải thời gian tính toán.

**Ba mô hình chạy trên ba thiết bị khác nhau, và đây là điều phải nói thẳng trong báo cáo:**
PyTorch trên GPU, TensorFlow trên CPU, NumPy trên CPU. Hệ quả là **cột thời gian không còn là
phép so sánh khung thư viện mà là phép so sánh phần cứng**. Mọi bảng có cột thời gian phải ghi
rõ thiết bị bên cạnh, và phần diễn giải không được kết luận khung nào nhanh hơn khung nào. Các
chỉ số chất lượng (accuracy, F1, R²) vẫn so sánh được bình thường vì không phụ thuộc thiết bị.

Ngân sách thời gian mỗi notebook: 20 phút. Với GPU, một epoch CNN trên 40.000 ảnh CIFAR-10 mất
khoảng 2,3 giây, nên **không còn lý do gì để lấy mẫu con cho PyTorch**. Chỉ hai mô hình NumPy
thuần vẫn phải dùng tập con, vì NumPy chạy CPU là bản chất của bài tập chứ không phải hạn chế
phần cứng.

`RANDOM_SEED = 42` ở mọi nơi, trừ các thực nghiệm đa hạt giống ở Mục 8.

## 2. Cây thư mục

```
src/Assignment 04/
├── CONTRACT.md                   ← file này
├── customer_comments/   data/ notebooks/ models/ reports/figures/
├── diabetes/            data/ notebooks/ models/ reports/figures/
├── house_price/         data/ notebooks/ models/ reports/figures/
├── mnist/               data/ notebooks/ models/ reports/figures/
├── cifar10/             data/ notebooks/ models/ reports/figures/
└── mlp_vs_cnn/          notebooks/ models/ reports/figures/
```

Đường dẫn trong notebook luôn **tương đối từ thư mục `notebooks/`**, dùng `../data/...`
và `../reports/figures/...`. Không hard-code đường dẫn tuyệt đối.

## 3. Dữ liệu (đã có sẵn trên đĩa — KHÔNG tải lại)

| Miền | File | Kích thước | Ghi chú |
|---|---|---|---|
| Diabetes | `diabetes/data/diabetes_prediction_dataset.csv` | 100 000 dòng | 8 đặc trưng + nhãn `diabetes` |
| Comments | `customer_comments/data/womens_ecommerce_reviews.csv` | 23 486 review | nhãn `Recommended IND` |
| House Price | `house_price/data/usa_real_estate_150k.csv` | 150 000 dòng | hồi quy `price` |
| MNIST | `mnist/data/mnist.npz` | 60k train / 10k test | khóa `x_train,y_train,x_test,y_test` |
| CIFAR-10 | `cifar10/data/cifar10.npz` | 50k train / 10k test | khóa như trên, `y` đã `ravel()` |

### Tiền xử lý bắt buộc (giữ nguyên để mọi miền so sánh được với nhau)

**Diabetes** — khử trùng lặp; loại `gender == 'Other'`; `gender → {Male:1, Female:0}`;
`smoking_history → {never:0, 'No Info':0, former:1, 'not current':1, current:2, ever:2}`.
8 đặc trưng theo đúng thứ tự:
`[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]`.
Chuẩn hóa `StandardScaler` **fit trên train**, transform cho val/test.

**Comments** — `dropna(subset=['Review Text','Recommended IND'])`; hạ chữ thường;
tách token bằng `re.findall(r'[a-zA-Z]+', text)`; từ điển `VOCAB_SIZE = 5000`
(`<PAD>`=0, `<UNK>`=1); cắt/đệm về `MAX_LEN = 50`; chiều nhúng `EMBED_DIM = 100`.

**House Price** — `dropna` trên `price, house_size, bed, bath`; lọc
`10 000 ≤ price ≤ 5 000 000` và `200 ≤ house_size ≤ 20 000`. 8 đặc trưng theo thứ tự:
`[log_house_size, bed, bath, total_rooms, bed_bath_prod, sqft_per_room, bath_bed_ratio, log_acre_lot]`,
mục tiêu `log_price = log(price)`. Báo cáo RMSE/MAE **quy đổi ngược về đơn vị USD**
(`np.exp`), R² tính trên thang log.

**MNIST / CIFAR-10** — chia `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
để có train/validation; test giữ nguyên 10 000 ảnh gốc, **không bao giờ** dùng để chọn epoch.
Chuẩn hóa: `x/255.0` rồi trừ mean / chia std **học từ train**.

## 4. Kiến trúc mô hình (bám sát slide bài giảng)

**1D CNN cho dữ liệu bảng (Diabetes, House Price)** — coi 8 đặc trưng là chuỗi dài 8:

```
8 → Conv1D(16, K=3, same) → ReLU → Conv1D(16, K=3, same) → ReLU
  → MaxPool1D(2) → Flatten → Dense(8) → ReLU → Dense(1)
```

Đầu ra: Diabetes dùng **Sigmoid + BCE**; House Price dùng **Linear + MSE**.

**1D CNN cho văn bản (Comments)**:

```
tokens(50) → Embedding(5000, 100) → Conv1D(32, K=3) → ReLU
           → GlobalMaxPool1D → Dense(1, Sigmoid)
```

**2D CNN thuần NumPy (MNIST, CIFAR-10)** — hai biến thể bắt buộc:

- *Baseline*: `Conv2D(8/16, K=3, padding=0)` → ReLU → MaxPool(2) → `Conv2D(16/32, K=3, padding=0)`
  → ReLU → MaxPool(2) → Flatten → Dense(64/128) → ReLU → Dense(10) → Softmax.
  Khởi tạo ngẫu nhiên `randn * 0.01`, learning rate cố định.
- *Improved*: y hệt nhưng `padding=1` (same), khởi tạo **He Normal**
  `σ = sqrt(2 / (C_in·K·K))`, và **lịch giảm learning rate**.

Bắt buộc hiện thực bằng `im2col` / `col2im` (vector hóa 100%, không vòng lặp pixel).

**2D CNN framework (PyTorch / Keras)** — 3 khối phân cấp:

```
Conv-BN-ReLU-MaxPool-Dropout ×3  →  Flatten → Dense(128) → ReLU → Dropout → Dense(10)
```

Kênh: MNIST `32→64`(2 khối); CIFAR-10 `32→64→64`(3 khối).
Model PyTorch **phải có** phương thức `extract_features(x)` trả về vector ẩn 128 chiều
(dùng cho PCA ở notebook `mlp_vs_cnn`).

**MLP đối kháng** — `Flatten → Dense(512) → Dense(256) → Dense(128) → Dense(10)`,
Dropout(0.2), Adam, Cross-Entropy. Cùng split, cùng chuẩn hóa với CNN.

## 5. Đầu ra bắt buộc

### 5.1 Notebook

- Mỗi notebook **phải được chạy hết và lưu kèm output** (`nbconvert --execute --inplace`).
- Ngôn ngữ trình bày: **tiếng Việt**, văn phong học thuật, xưng "báo cáo"/"chúng tôi" nhất quán.
- Cấu trúc markdown bắt buộc của mỗi notebook:
  1. Tiêu đề + thông tin sinh viên + mục tiêu notebook
  2. Nhập thư viện & cấu hình seed
  3. Nạp dữ liệu & khảo sát (EDA có số liệu cụ thể)
  4. Tiền xử lý (giải thích **tại sao**, không chỉ *làm gì*)
  5. Cơ sở toán học của tầng sắp cài (công thức LaTeX trong markdown)
  6. Hiện thực mô hình (code có chú thích tiếng Việt)
  7. Huấn luyện (in log từng epoch)
  8. Đánh giá + vẽ hình
  9. Lưu metrics JSON + nhận xét kết quả (đọc số thật, không chung chung)
- Sau **mỗi** hình và **mỗi** bảng số phải có một đoạn markdown **diễn giải** kết quả.
- Mọi công thức toán viết bằng `$...$` / `$$...$$`.

### 5.2 Figure

- PNG, `dpi=150`, `bbox_inches='tight'`, nền trắng.
- Tiêu đề, nhãn trục, legend bằng **tiếng Việt**.
- Font: `plt.rcParams['font.sans-serif'] = ['Segoe UI','DejaVu Sans']`,
  `plt.rcParams['axes.unicode_minus'] = False`.
- Lưu vào `<domain>/reports/figures/` theo đúng tên ở Mục 6.

### 5.3 Metrics JSON

Mỗi miền ghi **một** file `<domain>/reports/metrics_<domain>.json`. Schema:

```json
{
  "domain": "diabetes",
  "task": "classification",
  "dataset": {"file": "...", "n_raw": 100000, "n_clean": 96146,
              "n_train": 0, "n_val": 0, "n_test": 0, "n_features": 8},
  "models": {
    "numpy":      {"framework": "NumPy From Scratch", "params": 0,
                   "train_time_s": 0.0, "epochs": 0, "best_epoch": 0,
                   "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                   "roc_auc": 0.0, "loss": 0.0,
                   "history": {"train_loss": [], "val_loss": [],
                               "train_acc": [], "val_acc": []},
                   "confusion_matrix": [[0,0],[0,0]]},
    "pytorch":    { ... same keys ... },
    "tensorflow": { ... same keys ... }
  },
  "notes": "ghi chú về mọi sai lệch so với hợp đồng"
}
```

Với **hồi quy** (`house_price`) thay bộ `accuracy/precision/recall/f1/roc_auc/confusion_matrix`
bằng `rmse_usd`, `mae_usd`, `r2`, `rmse_log`, `mae_log`, và thêm
`scatter_sample`: `{"y_true": [...200 giá trị...], "y_pred": [...]}`.

Với **phân loại 10 lớp** (`mnist`, `cifar10`) dùng `macro_precision`, `macro_recall`,
`macro_f1`, `confusion_matrix` 10×10, và thêm
`per_class_accuracy`: mảng 10 số, `high_conf_errors`: danh sách ≤ 8 phần tử
`{"index": i, "true": k, "pred": j, "confidence": p}`.
Tên model key cho ảnh: `numpy_baseline`, `numpy_improved`, `pytorch`, `tensorflow`.

**Số học:** ghi số thực đầy đủ (không làm tròn chuỗi), tỉ lệ ở thang `[0,1]` chứ không phải phần trăm.

## 6. Danh mục figure bắt buộc (KHÔNG được thiếu bất kỳ file nào)

### customer_comments/reports/figures/
| File | Nội dung |
|---|---|
| `fig_comments_eda.png` | 2 panel: phân phối nhãn + histogram độ dài review |
| `fig_comments_loss_curves.png` | 3 panel cạnh nhau: BCE loss theo epoch (NumPy / PyTorch / TF) |
| `fig_comments_confusion.png` | 3 panel: ma trận nhầm lẫn của ba framework |
| `fig_comments_benchmark.png` | Bar chart nhóm: Accuracy/Precision/Recall/F1 × 3 framework |

### diabetes/reports/figures/
| File | Nội dung |
|---|---|
| `fig_diabetes_eda.png` | 2 panel: phân phối nhãn + tương quan 8 đặc trưng |
| `fig_diabetes_loss_curves.png` | 3 panel: train/val loss theo epoch |
| `fig_diabetes_confusion.png` | 3 panel: ma trận nhầm lẫn |
| `fig_diabetes_benchmark.png` | Bar chart nhóm 4 chỉ số × 3 framework |

### house_price/reports/figures/
| File | Nội dung |
|---|---|
| `fig_house_eda.png` | 2 panel: phân phối `price` và `log_price` |
| `fig_house_loss_curves.png` | 3 panel: MSE loss theo epoch |
| `fig_house_scatter.png` | 3 panel: y_thực vs y_dự_đoán + đường chéo, chú thích R² |
| `fig_house_benchmark.png` | Bar chart: RMSE / MAE / R² × 3 framework |

### mnist/reports/figures/
| File | Nội dung |
|---|---|
| `fig_mnist_class_distribution.png` | Phân phối lớp train vs test |
| `fig_mnist_sample_grid.png` | Lưới 10 lớp × 10 mẫu |
| `fig_mnist_scratch_curves.png` | 2 panel: đường cong Baseline vs Improved (loss + acc) |
| `fig_mnist_scratch_confusion.png` | 2 panel: CFM Baseline vs Improved |
| `fig_mnist_scratch_comparison.png` | Bar chart Baseline vs Improved (4 chỉ số) |
| `fig_mnist_framework_curves.png` | 2 panel: đường cong PyTorch vs Keras |
| `fig_mnist_framework_confusion.png` | 2 panel: CFM PyTorch vs Keras |
| `fig_mnist_3way_benchmark.png` | Bar chart 4 chỉ số × 3 cách cài đặt |
| `fig_mnist_per_class_accuracy.png` | Bar chart accuracy theo từng lớp (mô hình tốt nhất) |
| `fig_mnist_high_conf_errors.png` | Lưới 8 ảnh sai với confidence cao nhất |

### cifar10/reports/figures/
Y hệt danh sách MNIST, đổi tiền tố `fig_mnist_` → `fig_cifar10_`. **10 file.**

### mlp_vs_cnn/reports/figures/
| File | Nội dung |
|---|---|
| `fig_mlp_curves.png` | 2 panel: đường cong MLP trên MNIST và CIFAR-10 |
| `fig_mlp_confusion.png` | 2 panel: CFM MLP trên MNIST và CIFAR-10 |
| `fig_mlp_vs_cnn_gap.png` | Bar chart đối đầu MLP vs CNN + nhãn chênh lệch |
| `fig_latent_pca_mnist.png` | Scatter PCA 2D không gian ẩn 128D, tô màu theo 10 lớp |
| `fig_latent_pca_cifar10.png` | Scatter PCA 2D, tô màu 10 lớp, chú thích nhóm phương tiện/động vật |

### Hình lý thuyết — do agent riêng dựng, lưu ở `src/Assignment 04/theory_figures/`
`fig_th_convolution.png`, `fig_th_padding.png`, `fig_th_stride.png`, `fig_th_dilation.png`,
`fig_th_receptive_field.png`, `fig_th_pooling.png`, `fig_th_im2col.png`,
`fig_th_architecture.png`, `fig_th_backprop.png`.

## 7. Điều tuyệt đối không được làm

- Không bịa số liệu. Mọi con số trong JSON phải đến từ một lần chạy thật.
- Không sửa file ngoài thư mục miền được giao (tránh tranh chấp khi chạy song song).
- Không `git add .` / `git commit -a`. Chỉ commit theo đường dẫn tường minh:
  `git commit -m "..." -- "src/Assignment 04/<domain>/..."`.
- Không đổi tên figure hay khóa JSON vì "thấy hợp lý hơn". Báo cáo đọc theo tên ở đây.
- Không dùng `plt.show()` mà quên `plt.savefig()` — hình phải nằm trên đĩa.

---

## 8. Ba thực nghiệm gốc (miền `analysis/`)

Phần này là **đóng góp riêng của báo cáo**, không có trong bất kỳ bài tham khảo nào. Mục tiêu là
lấp đúng những lỗ hổng mà các bài khác hoặc bỏ qua, hoặc chỉ nêu ở phần hướng phát triển.

Ba notebook, đặt ở `analysis/notebooks/`, mỗi notebook ghi một tệp JSON riêng.

### 8.1 `04_statistical_rigor.ipynb` → `reports/metrics_statistical.json`

Trả lời câu hỏi mà mọi bảng đối chuẩn trong báo cáo đang né: **chênh lệch 0,26 điểm phần trăm
giữa hai khung có thật sự là khác biệt, hay chỉ là nhiễu khởi tạo?**

- **Đa hạt giống**: huấn luyện lại mỗi cấu hình với `seed ∈ {42, 43, 44, 45, 46}`, báo cáo
  `mean ± std` thay cho một con số trần. Làm cho Comments, Diabetes, House Price và MNIST.
- **Kiểm định McNemar** cho từng cặp khung trên cùng tập kiểm thử. Đây là phép kiểm đúng cho
  hai bộ phân loại chạy trên **cùng** các mẫu, vì nó chỉ nhìn vào số mẫu mà hai bên **bất đồng**
  (`b` và `c` trong bảng 2×2), chứ không coi hai dãy dự đoán là độc lập. Báo cáo `statistic`,
  `p_value`, và kết luận ở mức ý nghĩa 0,05. Dùng `statsmodels` nếu có, nếu không thì tự cài
  bằng `scipy.stats.binomtest(b, b+c, 0.5)` (phiên bản chính xác, đúng khi `b+c` nhỏ).
- **Khoảng tin cậy Wilson 95%** cho từng accuracy. Ưu điểm so với khoảng Wald là không tràn ra
  ngoài đoạn [0,1] và vẫn đúng khi tỉ lệ gần 0 hoặc gần 1.
- Kết luận phải nói thẳng: những cặp nào **không** khác biệt có ý nghĩa thống kê. Nếu hoá ra
  phần lớn các cặp đều không khác biệt, đó là một kết quả mạnh chứ không phải thất bại, vì nó
  chứng minh đúng luận điểm trung tâm rằng nền toán học quyết định kết quả.

Hình: `fig_seed_variance.png` (thanh lỗi mean ± std theo miền),
`fig_mcnemar_matrix.png` (ma trận p-value từng cặp), `fig_wilson_ci.png` (khoảng tin cậy).

### 8.2 `05_model_anatomy.ipynb` → `reports/metrics_anatomy.json`

Mổ xẻ mô hình đã huấn luyện, thay vì chỉ báo cáo điểm số.

- **Hiệu chỉnh xác suất**: tính **ECE** (Expected Calibration Error, 15 bin) và vẽ **biểu đồ độ
  tin cậy**. Một trong hai bài tham khảo chỉ ghi "có thể bổ sung calibration" vào hướng phát
  triển; báo cáo này làm thật và đo được. Kèm **nhiệt độ scaling**: tối ưu một tham số `T` duy
  nhất trên tập validation rồi báo cáo ECE trước và sau. Nếu ECE giảm rõ thì đó là bằng chứng
  mô hình *quá tự tin*, đúng như hiện tượng "sai với độ tin cậy cao" mà Chương 6 và 7 mô tả.
- **Trực quan bộ lọc tầng một**: vẽ toàn bộ 32 kernel của tầng Conv đầu tiên trên CIFAR-10
  (3×3×3 nên hiển thị được thành ảnh RGB), và trên MNIST (3×3×1, thang xám). Nhận xét xem có
  bộ lọc nào học được cạnh, gradient màu, hay đốm.
- **Bản đồ đặc trưng**: chọn một ảnh test, cho đi qua từng khối và vẽ các feature map, để thấy
  biểu diễn trừu tượng dần theo độ sâu.
- **Độ nhạy che khuất**: trượt một ô vuông xám 8×8 trên ảnh, ghi lại xác suất lớp đúng tại mỗi
  vị trí, rồi vẽ bản đồ nhiệt. Bản đồ này cho biết **vùng ảnh nào thực sự chi phối quyết định**,
  và là cách kiểm tra mô hình có nhìn vào vật thể hay đang bám vào nền.

Hình: `fig_reliability_diagram.png`, `fig_temperature_scaling.png`, `fig_conv1_filters.png`,
`fig_feature_maps.png`, `fig_occlusion_sensitivity.png`.

### 8.3 `06_ablation_scaling.ipynb` → `reports/metrics_ablation.json`

Gỡ hai chỗ mà phần còn lại của báo cáo đang phải thừa nhận là bị trộn lẫn.

- **Bóc tách yếu tố cải tiến**: hiện tại "Improved" gộp ba thay đổi (đệm viền, He Normal, lịch
  giảm tốc độ học) nên không biết cái nào đóng góp bao nhiêu. Chạy đủ **8 tổ hợp 2³** trên tập
  con MNIST, báo cáo accuracy từng tổ hợp, rồi tính **hiệu ứng chính** của từng yếu tố bằng
  trung bình chênh lệch khi bật so với khi tắt. Đây là thiết kế giai thừa đầy đủ, không phải
  thử từng cái một.
- **Đường cong theo cỡ dữ liệu**: huấn luyện cùng một kiến trúc trên
  `n ∈ {500, 1000, 2000, 5000, 10000, 20000, 40000}` ảnh, vẽ accuracy theo `n` ở thang log.
  Mục đích rất cụ thể: Chương 6 và 7 thừa nhận khoảng cách giữa NumPy và framework **trộn lẫn**
  ba nguyên nhân (ít dữ liệu hơn, kiến trúc nông hơn, tầng hiện thực khác). Đường cong này tách
  được phần do **dữ liệu**: đọc giá trị của mô hình framework tại đúng `n` mà NumPy đã dùng, rồi
  so với giá trị tại `n` đầy đủ. Phần chênh còn lại mới là do kiến trúc.
  Báo cáo phải nêu con số tách bạch đó, ví dụ "trong 20,3 điểm chênh lệch thì 12,1 điểm là do
  dữ liệu và 8,2 điểm là do kiến trúc".

Hình: `fig_ablation_factorial.png` (8 tổ hợp + hiệu ứng chính),
`fig_learning_curve.png` (accuracy theo cỡ dữ liệu, có đánh dấu điểm NumPy dùng),
`fig_gap_decomposition.png` (thanh xếp chồng tách phần dữ liệu và phần kiến trúc).

### Quy tắc chung cho cả ba notebook

Không được chạy lại toàn bộ pipeline nặng nếu có thể nạp lại mô hình đã lưu. Mọi con số vào JSON
phải từ một lần chạy thật. Nếu một phép kiểm cho kết quả **trái** với kỳ vọng nêu trong phần mở
đầu notebook, giữ nguyên kết quả và sửa phần diễn giải, tuyệt đối không sửa thực nghiệm cho khớp
câu chuyện.
