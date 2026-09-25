# Assignment 05 — Kế hoạch chi tiết (một phase duy nhất)

**Đề tài:** Mạng nơ-ron tích chập nhìn như hợp của các hàm, từ CNN cơ bản đến các kiến trúc phát triển
**Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · D23CTPM01 · GVHD PGS.TS Trần Đình Quế
**Sản phẩm cuối:** `Report/Assignment 05/A05_CT_nghiand.600.pdf` và mã nguồn trong `src/Assignment 05/`
**Trạng thái:** hoàn thành 2026-09-25. Sai khác so với kế hoạch: báo cáo dựng bằng LuaLaTeX thay XeLaTeX (ảnh vẽ từng điểm ảnh vượt bộ nhớ XeTeX, đúng rủi ro ở Mục 7); bỏ `channels_last` (đo được chậm hơn 3,6 lần); thêm cắt gradient cho cả bốn mô hình ảnh vì M0 phân kỳ ở lr 0,1; M1 dùng 2 khối × 2 conv mỗi tầng để M1 và M2 chỉ khác đúng đường tắt.

---

## 0. Nguyên tắc chốt (không bàn lại)

| # | Nguyên tắc | Nguồn |
|---|---|---|
| 1 | Chỉ làm đúng 5 mục của đề. Mọi hạng mục bên dưới phải chỉ ra được nó phục vụ mục nào. Không chạy đa seed, không ablation, không kiểm định thống kê, không pretrained, không API/web/mobile | người dùng |
| 2 | **Mọi hình trong báo cáo vẽ bằng TikZ/pgfplots.** Không có tệp PNG nào. Biểu đồ đọc dữ liệu từ `.dat` do notebook xuất ra. Ảnh mẫu, feature map, bộ lọc được xuất thành ma trận pixel rồi vẽ bằng `matrix plot*` | người dùng |
| 3 | Code xuất hiện ở **mọi chương**, lấy thẳng từ mã nguồn thật (trích tự động theo tên hàm), không chép tay vào `.tex` | người dùng |
| 4 | Không có con số nào gõ tay trong báo cáo. Mọi số liệu là macro LaTeX sinh từ JSON | quy ước A04 |
| 5 | Làm đơn luồng, không sub-agent | người dùng |
| 6 | Dữ liệu: CIFAR-10 + CIFAR-100 + Diabetes Prediction Dataset (tái dùng từ A04). "MNIST-100" trong đề được hiểu là CIFAR-100, và báo cáo ghi rõ cách hiểu này. Không dùng bộ `mohith2409` (thực chất là Diabetes 130-US Hospitals, bài toán tái nhập viện, tốn công tiền xử lý) | người dùng chốt |
| 7 | Mô hình: Basic CNN → VGG → ResNet → SE-ResNet, tự xây từ đầu, cùng ngân sách huấn luyện. Bản Conv1D cho Diabetes | người dùng chốt |

## 1. Đối chiếu 5 mục của đề với sản phẩm

| Mục đề | Notebook | Chương báo cáo | Hình TikZ chính |
|---|---|---|---|
| 1. Khái niệm cơ bản qua hàm và hợp hàm, kèm code | `00_ham_va_hop_ham.ipynb` | Ch.1 | chuỗi hợp hàm, hàm kích hoạt, tích chập từng bước, pooling |
| 2. Các mô hình phát triển của CNN, hàm và code | `01_kien_truc_phat_trien.ipynb` | Ch.2 | dòng thời gian, sơ đồ 8 khối kiến trúc |
| 3. Hai tập ảnh + một tập diabetes, có link và mô tả | `02_du_lieu.ipynb` | Ch.3 | lưới ảnh mẫu (vẽ từng pixel), phân bố nhãn, tương quan |
| 4. Code một CNN cơ bản + ba mô hình phát triển | `03_cifar10.ipynb`, `04_cifar100.ipynb`, `05_diabetes.ipynb` | Ch.4 | sơ đồ 4 kiến trúc, bảng shape, feature map |
| 5. So sánh và đánh giá | `06_so_sanh.ipynb` | Ch.5 | đường học, cột chỉ số, ma trận nhầm lẫn, ROC/PR, bộ lọc lớp 1 |

## 2. Phân bổ tài nguyên (GPU RTX 4060 8 GB · CPU 32 luồng)

| Tác vụ | Thiết bị | Lý do |
|---|---|---|
| Huấn luyện 4 mô hình 2D trên CIFAR-10 / CIFAR-100 | **GPU**, AMP fp16, `channels_last`, `cudnn.benchmark=True` | Tích chập 2D trên 50k ảnh, phần tính toán áp đảo chi phí truyền |
| Nạp dữ liệu ảnh | **GPU**: toàn bộ tập dưới dạng tensor `uint8` nằm sẵn trên VRAM (~150 MB mỗi tập), augmentation (crop có padding 4 + lật ngang) làm trên GPU | Trên Windows, DataLoader nhiều worker chậm và dễ lỗi. Cách này bỏ hẳn DataLoader |
| Trích feature map, bộ lọc, suy luận trên tập test | GPU | Chạy một lượt, nhanh |
| 4 mô hình Conv1D trên Diabetes | **CPU**, `torch.set_num_threads(8)` | Mô hình dưới 50k tham số. A04 đo được GPU chậm hơn CPU gần 10 lần với loại mô hình này |
| NumPy thuần ở mục 1, tiền xử lý, xuất `.dat` cho hình | CPU | Bản chất tác vụ |
| Biên dịch LaTeX, externalize TikZ | CPU | Có thể mất vài phút lần đầu, sau đó dùng cache |

Mọi notebook in ra thiết bị đã dùng. Mọi bảng có cột thời gian đều ghi kèm thiết bị. Đo thời gian trên GPU phải bọc `torch.cuda.synchronize()`.

## 3. Bố cục thư mục

```
src/Assignment 05/
├── README.md                  hướng dẫn chạy lại (theo mẫu A04)
├── requirements.txt
├── plans/                     kế hoạch này
├── a05/                       gói dùng chung, mỗi trách nhiệm một file (SSOT)
│   ├── concepts.py            mục 1: neuron, kích hoạt, conv2d, pooling, softmax+CE, hợp hàm, kiểm tra đạo hàm (NumPy)
│   ├── blocks.py              mục 2: LeNet, AlexNet-mini, VGG block, Inception, Residual, Dense, Depthwise, SE, CBAM, ViT patch
│   ├── data.py                mục 3: nạp CIFAR-10/100 lên GPU, augmentation GPU, tiền xử lý Diabetes (fit trên train)
│   ├── models.py              mục 4: BasicCNN, VGGNet, ResNet, SEResNet, mỗi họ có bản 2D và 1D, build_model(name, ...)
│   ├── train.py               mục 4: vòng lặp huấn luyện chung, checkpoint theo val, ghi lịch sử
│   ├── metrics.py             mục 5: accuracy, top-5, macro-F1, precision/recall, ROC/PR, ma trận nhầm lẫn, đếm tham số/MACs, đo thời gian
│   └── export.py              ghi .dat cho pgfplots và JSON số liệu (một định dạng, một chỗ)
├── data/                      cifar10/, cifar100/, diabetes/ (tệp lớn bị .gitignore chặn)
├── notebooks/                 00 → 06 như bảng Mục 1
├── models/                    <tập>_<mô hình>.pt + metadata.json
└── outputs/
    ├── figdata/               *.dat cho mọi hình pgfplots
    └── metrics/               *.json cho mọi con số trong báo cáo

Report/Assignment 05/
├── main.tex                   dựa trên khuôn A01 (XeLaTeX, Times New Roman, bìa PTIT)
├── preamble/tikzstyle.tex     bảng màu, style khối kiến trúc, style pgfplots dùng chung
├── chapters/                  00-mo-dau, 01…06, phu-luc
├── tikz/                      mỗi hình một file .tex
├── figdata/                   build.py chép từ outputs/figdata
├── code/                      build.py trích từ a05/*.py theo tên hàm/lớp
├── generated/so_lieu.tex      build.py sinh macro \SoLieu... từ outputs/metrics/*.json
├── refs.bib
└── build.py                   chép dữ liệu → trích code → sinh macro → latexmk -xelatex -shell-escape
```

## 4. Nội dung chi tiết từng mục

### Mục 1: khái niệm cơ bản qua hàm và hợp hàm (`a05/concepts.py`, notebook 00, Ch.1)

Mỗi khái niệm gồm công thức, một hàm NumPy, một ví dụ tính tay và một dòng so khớp với PyTorch (`np.allclose`).

| Khái niệm | Hàm | Kiểm chứng |
|---|---|---|
| Neuron tuyến tính `z = w·x + b` | `neuron(x, w, b)` | khớp `nn.Linear` |
| Vì sao cần phi tuyến: hai tầng affine gộp lại vẫn là affine | `compose(f, g)` | so hai ma trận |
| ReLU, sigmoid, tanh và đạo hàm | `relu`, `sigmoid`, `tanh` | hình pgfplots vẽ từ biểu thức |
| Tích chập 2D (cross-correlation), padding, stride | `conv2d(X, K, stride, pad)`, `out_size(H,K,P,S)` | khớp `F.conv2d` |
| Số tham số Dense so với Conv | `count_params_dense`, `count_params_conv` | khớp `numel()` |
| Max pooling | `maxpool2d` | khớp `F.max_pool2d` |
| Flatten + Dense, softmax + cross-entropy ổn định số | `softmax`, `cross_entropy` | khớp `F.cross_entropy` |
| CNN nhỏ là một hợp hàm `f = f_L ∘ … ∘ f_1`, in shape từng bước | `tiny_cnn_forward` | logits khớp `nn.Sequential` |
| Lan truyền ngược là quy tắc chuỗi, cập nhật `θ ← θ − η∇L` | `conv2d_backward`, `grad_check` | sai số sai phân hữu hạn < 1e-6 |

Hình TikZ (5): chuỗi hợp hàm Conv→ReLU→Pool→Flatten→Dense→Softmax có ghi shape · 4 hàm kích hoạt và đạo hàm · kernel trượt trên lưới số, tính tay một ô · padding/stride và công thức kích thước · max pooling 4×4→2×2.

### Mục 2: các mô hình phát triển của CNN (`a05/blocks.py`, notebook 01, Ch.2)

Mỗi kiến trúc có: vấn đề của kiến trúc trước, cơ chế mới viết dưới dạng hàm, lớp PyTorch ngắn, số tham số và một lượt forward trên tensor giả kiểm tra shape. **Không huấn luyện** ở mục này.

| Kiến trúc | Năm | Cơ chế viết dưới dạng hàm | Lớp trong `blocks.py` |
|---|---|---|---|
| LeNet-5 | 1998 | Conv→Pool→FC | `LeNet5` |
| AlexNet | 2012 | ReLU + Dropout, sâu hơn | `AlexNetMini` |
| VGG | 2014 | chồng conv 3×3 thay conv lớn | `VGGBlock` |
| Inception | 2014 | nhánh song song 1×1/3×3/5×5, nối kênh | `InceptionBlock` |
| ResNet | 2015 | `y = F(x) + x` | `ResidualBlock` |
| DenseNet | 2017 | `x_l = H_l([x_0,…,x_{l−1}])` | `DenseBlock` |
| MobileNet | 2017 | depthwise + pointwise | `DepthwiseSeparable` |
| SE-Net / CBAM | 2017–18 | `y = x ⊙ σ(MLP(GAP(x)))` | `SEBlock`, `CBAM` |
| ViT | 2020 | ảnh → patch → self-attention | `PatchEmbed` + `nn.MultiheadAttention` |

Hình TikZ (2): dòng thời gian tiến hoá · một hình nhiều ô, mỗi ô là sơ đồ khối của một cơ chế (VGG, Inception, Residual, Dense, Depthwise, SE).
Bảng: kiến trúc, vấn đề giải quyết, cơ chế, số tham số của khối minh hoạ.

### Mục 3: dữ liệu (`a05/data.py`, notebook 02, Ch.3)

| Tập | Nguồn (kiểm tra link bằng `curl -I` trước khi đưa vào báo cáo) | Quy mô | Chia |
|---|---|---|---|
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html (tái dùng tệp đã tải ở A04) | 60k ảnh 32×32×3, 10 lớp | 45k train / 5k val (phân tầng, seed 42) / 10k test chính thức |
| CIFAR-100 | cùng trang trên, `cifar-100-python.tar.gz` | 60k ảnh 32×32×3, 100 lớp mịn, 20 siêu lớp | như trên |
| Diabetes | Diabetes Prediction Dataset, https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset (chép `diabetes_prediction_dataset.csv` từ `src/Assignment 04/diabetes/data/`) | 100.000 dòng × 8 đặc trưng + nhãn `diabetes`, lớp dương 8,5%, 3.854 dòng trùng | bỏ dòng trùng rồi chia 70/15/15 phân tầng. One-hot `gender`, `smoking_history`; StandardScaler cho cột số; chỉ fit trên train |

Đặc trưng Diabetes: `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history` (có giá trị "No Info" 35,8%, giữ như một nhóm riêng), `bmi`, `HbA1c_level`, `blood_glucose_level`. Mất cân bằng 8,5% nên dùng `pos_weight` và báo recall, F1, ROC-AUC, PR-AUC cạnh accuracy.

Mô tả tối thiểu mỗi tập: kích thước, số lớp, phân bố nhãn, mean/std theo kênh (ảnh), thống kê mô tả và tỉ lệ lớp dương (Diabetes), dạng tensor đưa vào mạng (`[B,3,32,32]`, `[B,1,d]`).

Hình TikZ (4): lưới 10 ảnh CIFAR-10 (mỗi lớp một ảnh, vẽ từng pixel RGB) · lưới 20 ảnh CIFAR-100 (mỗi siêu lớp một ảnh) · histogram pixel theo 3 kênh cho cả hai tập · Diabetes: phân bố nhãn, histogram 3 đặc trưng quan trọng nhất, heatmap tương quan.

### Mục 4: một CNN cơ bản và ba mô hình phát triển (`a05/models.py`, `a05/train.py`, notebook 03/04/05, Ch.4)

Mỗi bước tiến hoá thêm **đúng một** cơ chế, để phần so sánh ở mục 5 quy được khác biệt về cơ chế đó.

| Mô hình | Cơ chế thêm vào | Cấu trúc 2D (ảnh 32×32) |
|---|---|---|
| M0 `BasicCNN` | gốc, kiểu LeNet/AlexNet | [Conv3×3 → ReLU → MaxPool] × 3 (32/64/128) → Flatten → FC256 → FC K |
| M1 `VGGNet` | sâu hơn, conv 3×3 đôi, BatchNorm | 3 tầng × 2 conv 3×3 + BN (64/128/256), MaxPool, GAP → FC K |
| M2 `ResNet` | đường tắt `F(x)+x` | cùng độ rộng và độ sâu với M1, 2 BasicBlock mỗi tầng, shortcut chiếu 1×1 khi đổi kích thước |
| M3 `SEResNet` | chú ý theo kênh | M2 + `SEBlock` (r = 16) trong mỗi block |

Bản 1D cho Diabetes giữ nguyên cơ chế, đổi `Conv2d`→`Conv1d`, độ rộng nhỏ (16/32/64). Đầu ra là một logit với `BCEWithLogitsLoss(pos_weight)`.

Cấu hình huấn luyện, giống nhau cho cả 4 mô hình trong cùng một tập:

| | CIFAR-10 / CIFAR-100 (GPU) | Diabetes (CPU) |
|---|---|---|
| Optimizer | SGD momentum 0.9, nesterov, wd 5e-4 | AdamW lr 1e-3, wd 1e-4 |
| Lịch lr | OneCycle, max lr 0.1 | cố định |
| Epoch · batch | 30 · 256 | 30 · 512, early stop patience 5 |
| Checkpoint | val loss nhỏ nhất, nạp lại trước khi test | như bên trái |
| Seed | 42 (một lần chạy) | 42 |
| Ngưỡng | argmax | chọn trên val theo F1, áp dụng cho test |

Ước lượng thời gian: mô hình 0,3–3 triệu tham số trên RTX 4060 với dữ liệu nằm sẵn trên GPU chạy khoảng 4–10 s/epoch, tức 2–5 phút mỗi mô hình. **8 lượt GPU mất tổng khoảng 30–45 phút.** Diabetes chạy 4 mô hình trên CPU, khoảng 5–10 phút.

Lưu: `models/<tập>_<mô hình>.pt`, `metadata.json` (tham số, cấu hình, thiết bị, epoch tốt nhất), `outputs/figdata/<tập>_<mô hình>_history.dat`.

Hình TikZ (3): sơ đồ 4 kiến trúc đặt cạnh nhau, ghi shape tensor · bảng/hình dò shape tầng-theo-tầng của BasicCNN · feature map sau conv đầu của M0 cho một ảnh (8 kênh, vẽ từng pixel).

### Mục 5: so sánh và đánh giá (`a05/metrics.py`, notebook 06, Ch.5)

Tiêu chí, lấy theo tinh thần tutorial (không chỉ so accuracy):

| Tiêu chí | CIFAR-10 | CIFAR-100 | Diabetes |
|---|---|---|---|
| Chất lượng | accuracy, macro-F1 | top-1, top-5, macro-F1 | accuracy, precision, recall, F1, ROC-AUC, PR-AUC, so với mốc đoán lớp đa số |
| Chi phí | số tham số, MACs, thời gian/epoch, ms/ảnh khi suy luận | như bên trái | tham số, thời gian |
| Hội tụ | đường loss/acc train–val, epoch tốt nhất | như bên trái | như bên trái |
| Lỗi | ma trận nhầm lẫn 10×10 của mô hình tốt nhất | ma trận nhầm lẫn 20×20 theo siêu lớp | ma trận nhầm lẫn 2×2 tại ngưỡng đã chọn |
| Biểu diễn | 16 bộ lọc lớp 1 và feature map của cùng một ảnh qua 4 mô hình | không làm lại | không áp dụng |

Hình TikZ (8): đường học CIFAR-10 (4 mô hình, 2 ô loss/acc) · đường học CIFAR-100 · đường học Diabetes · cột chỉ số chất lượng + chi phí cho 3 tập · accuracy theo số tham số (scatter) · ma trận nhầm lẫn CIFAR-10 · ma trận siêu lớp CIFAR-100 · ROC + PR Diabetes (4 đường) · bộ lọc lớp 1 và feature map của 4 mô hình.
Bảng: kết quả từng tập, và một bảng tổng hợp 12 lượt chạy (3 tập × 4 mô hình).

Phần diễn giải chỉ nói điều số liệu cho thấy (ví dụ cơ chế nào mang lại bao nhiêu điểm, đổi lại bao nhiêu tham số và thời gian). Khác biệt dưới khoảng 0,5 điểm được ghi là "trong mức dao động của một lần chạy", vì chỉ chạy một seed.

## 5. Báo cáo LaTeX

**Khuôn:** chép `Report/Assignment 01/main.tex` (XeLaTeX, bìa PTIT, header/footer, màu `ptitblue`), thêm `pgfplots` (compat 1.18), `pgfplotstable`, thư viện TikZ `external, positioning, arrows.meta, matrix, fit, calc, shapes`, `minted` (hoặc `listings` nếu minted lỗi), `tcolorbox` cho hộp code.

**Mục lục báo cáo** (khoảng 50–60 trang):

| Phần | Nội dung | Code trong chương |
|---|---|---|
| Mở đầu | Bìa, lời mở đầu, mục lục, danh mục hình/bảng/code, bảng đối chiếu 5 mục của đề | — |
| Ch.1 Khái niệm cơ bản qua hàm và hợp hàm | theo bảng Mục 4 phần 1 | 8–10 đoạn (NumPy + PyTorch đối chiếu) |
| Ch.2 Các mô hình phát triển của CNN | theo bảng mục 2 | 9 đoạn, mỗi kiến trúc một lớp |
| Ch.3 Dữ liệu | 3 tập, link, mô tả, tiền xử lý, dạng tensor | 3–4 đoạn (nạp, augmentation GPU, pipeline Diabetes) |
| Ch.4 CNN cơ bản và ba mô hình phát triển | thiết kế, dò shape, cấu hình, vòng lặp huấn luyện | 6–7 đoạn (4 lớp mô hình, `build_model`, `train_one_epoch`, checkpoint) |
| Ch.5 So sánh và đánh giá | kết quả 3 tập, chi phí, lỗi, biểu diễn, tổng hợp | 3–4 đoạn (hàm đánh giá, đếm MACs, đo thời gian, chọn ngưỡng) |
| Ch.6 Kết luận | trả lời 5 mục, hạn chế (một seed, CIFAR-100 được hiểu thay cho MNIST-100) | — |
| Tài liệu tham khảo, Phụ lục tái lập | lệnh chạy, phiên bản thư viện, thiết bị | 1 đoạn lệnh |

Tổng cộng khoảng **30 đoạn code** trải qua cả 5 chương nội dung. Mỗi đoạn có caption và 2–4 dòng giải thích ngay sau.

**Cơ chế đưa code vào báo cáo:** `build.py` dùng `ast` để cắt đúng thân hàm/lớp theo tên (ví dụ `concepts.conv2d`, `models.ResidualBlock2d`) ra `code/<module>__<tên>.py`, rồi chương gọi `\inputminted{python}{code/...}`. Sửa mã nguồn thì báo cáo tự đổi theo.

**Cơ chế đưa số vào báo cáo:** `build.py` đọc `outputs/metrics/*.json`, sinh `\newcommand{\SoLieuCifTenResNetAcc}{93{,}12}` (định dạng dấu phẩy thập phân kiểu Việt), chương chỉ gọi macro.

**Hình ảnh pixel bằng pgfplots:** `export.py` ghi ảnh thành `.dat` 3 cột `x y màu` với màu dạng `rgb=r,g,b` (tách cột bằng dấu cách để không đụng dấu phẩy), vẽ bằng `\addplot[matrix plot*, mesh/cols=32, mesh/color input=explicit, point meta=explicit symbolic]`. Ảnh xám và feature map dùng colormap thường. Bật `\tikzexternalize[prefix=tikzcache/]` để mỗi hình chỉ biên dịch một lần.

## 6. Trình tự thực hiện (đơn luồng)

| Bước | Việc | Thiết bị | Ước lượng | Kiểm chứng |
|---|---|---|---|---|
| 1 | Tạo khung thư mục, `requirements.txt`, `.gitignore` cho data/models lớn, chép CIFAR-10 và CSV Diabetes từ A04, tải CIFAR-100 | CPU | 10 phút | `data/` có đủ 2 tệp tar.gz và 1 CSV, đọc được |
| 2 | **Thử rủi ro TikZ trước:** 1 ảnh CIFAR 32×32 vẽ bằng `matrix plot*`, một listing minted, một macro số liệu, biên dịch với `-shell-escape` | CPU | 20 phút | PDF thử hiển thị đúng màu ảnh, code tô màu, số in ra. Nếu XeLaTeX tràn bộ nhớ thì đổi sang LuaLaTeX |
| 3 | `concepts.py` + notebook 00 (mục 1), xuất `.dat` cho hình ch.1 | CPU | 45 phút | mọi `np.allclose` in True, sai số gradient < 1e-6 |
| 4 | `blocks.py` + notebook 01 (mục 2) | CPU | 40 phút | mỗi khối chạy forward đúng shape, số tham số khớp công thức |
| 5 | `data.py` + notebook 02 (mục 3): nạp 3 tập, thống kê, xuất `.dat` ảnh mẫu/histogram/tương quan | CPU → GPU | 40 phút | tổng số mẫu đúng, phân tầng đúng, không rò rỉ (scaler fit trên train) |
| 6 | `models.py`, `train.py`, `metrics.py`, `export.py`; chạy thử mỗi mô hình 1 epoch trên 2k ảnh | GPU | 45 phút | loss giảm, file history/metadata sinh ra đủ khoá |
| 7 | Notebook 03: CIFAR-10, 4 mô hình × 30 epoch | GPU | ~20 phút | 4 checkpoint + 4 history + metrics JSON |
| 8 | Notebook 04: CIFAR-100, 4 mô hình × 30 epoch | GPU | ~25 phút | như trên, có top-5 |
| 9 | Notebook 05: Diabetes, 4 mô hình Conv1D (có thể chạy song song với bước 7–8 như một tiến trình CPU nền) | CPU | ~10 phút | ROC-AUC, ngưỡng chọn trên val |
| 10 | Notebook 06: tổng hợp, ma trận nhầm lẫn, bộ lọc, feature map, xuất toàn bộ `.dat` và JSON | GPU + CPU | 30 phút | mỗi hình trong danh mục ở Mục 4 có đủ tệp dữ liệu |
| 11 | Viết `tikzstyle.tex` và các file `tikz/*.tex` (khoảng 22 hình) | CPU | 2,5 giờ | mỗi hình biên dịch độc lập được |
| 12 | Viết 6 chương + `build.py`, biên dịch toàn bộ | CPU | 3 giờ | PDF không còn `??`, không lỗi overfull nghiêm trọng, mục lục đủ |
| 13 | Rà soát cuối: 5 mục đề đều có mặt, mọi số là macro, mọi hình là TikZ (`grep includegraphics` chỉ còn logo bìa), link dữ liệu trả HTTP 200. Cập nhật README gốc và README bài | CPU | 30 phút | checklist ở Mục 8 qua hết |

**Tổng ước lượng: khoảng 10–11 giờ làm việc**, trong đó chỉ khoảng 1 giờ là GPU chạy máy. Phần tốn nhất là vẽ TikZ và viết báo cáo.

## 7. Rủi ro và cách xử lý

| Rủi ro | Xử lý |
|---|---|
| Hình pixel làm TeX tràn bộ nhớ hoặc biên dịch rất chậm | Bước 2 thử trước. Dùng externalize. Nếu vẫn quá tải thì chuyển LuaLaTeX (cấp phát bộ nhớ động) |
| `minted` lỗi với `-shell-escape` trên MiKTeX | Chuyển sang `listings` với style tự định nghĩa, cơ chế trích code giữ nguyên |
| Ma trận nhầm lẫn 100×100 quá dày | Chỉ vẽ theo 20 siêu lớp (đã chốt ở Mục 5) |
| Một seed nên chênh lệch nhỏ không đáng tin | Ghi rõ trong phần diễn giải, không kết luận khi chênh dưới 0,5 điểm |

## 8. Checklist hoàn thành

- [x] Đủ 7 notebook đã chạy từ đầu đến cuối, có output
- [x] 8 checkpoint ảnh + 4 checkpoint Diabetes, mỗi cái kèm `metadata.json`
- [x] Ch.1–Ch.5 mỗi chương có ít nhất 3 đoạn code trích từ `a05/`
- [x] Mọi hình là TikZ/pgfplots, không có `\includegraphics` ngoài logo trên bìa
- [x] Không có con số gõ tay: mọi số liệu là macro `\SL{tệp/khoá}` sinh từ JSON
- [x] Link 3 tập dữ liệu đã kiểm tra trả HTTP 200
- [x] PDF đặt tên `A05_CT_nghiand.600.pdf`, README gốc thêm dòng bài 05
