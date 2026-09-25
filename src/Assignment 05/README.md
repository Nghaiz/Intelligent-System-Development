# Assignment 05 — Mạng nơ-ron tích chập nhìn như hợp của các hàm

Từ CNN cơ bản đến các kiến trúc phát triển.

**Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · Lớp D23CTPM01
**Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
**Học kỳ:** Học kỳ 1 năm học 2026 – 2027

Báo cáo: [`Report/Assignment 05/A05_CT_nghiand.600.pdf`](../../Report/Assignment%2005/A05_CT_nghiand.600.pdf)

---

## Bài này làm gì

Đề có năm mục. Mỗi mục ứng với một module trong gói `a05/`, một notebook và một chương báo cáo:

| Mục của đề | Module | Notebook | Chương |
|---|---|---|---|
| 1. Khái niệm cơ bản qua hàm và hợp hàm | `a05/concepts.py` | `00_ham_va_hop_ham` | 1 |
| 2. Các mô hình phát triển của CNN | `a05/blocks.py` | `01_kien_truc_phat_trien` | 2 |
| 3. Hai tập ảnh và một tập tiểu đường | `a05/data.py` | `02_du_lieu` | 3 |
| 4. Một CNN cơ bản và ba mô hình phát triển | `a05/models.py`, `a05/train.py` | `03_cifar10`, `04_cifar100`, `05_diabetes` | 4 |
| 5. So sánh và đánh giá | `a05/metrics.py`, `a05/experiment.py` | `06_so_sanh` | 5 |

Bốn mô hình ở mục 4, mỗi bước thêm đúng một cơ chế:

| | Mô hình | Cơ chế thêm vào |
|---|---|---|
| M0 | BasicCNN | gốc: [Conv → ReLU → MaxPool] × 3 → FC → FC |
| M1 | VGGNet | chồng conv 3×3, BatchNorm, Global Average Pooling |
| M2 | ResNet | đường tắt `y = ReLU(F(x) + x)` |
| M3 | SE-ResNet | chú ý theo kênh `x ⊙ σ(MLP(GAP(x)))` |

Mỗi mô hình có bản 2D (ảnh) và bản 1D (Diabetes) sinh từ cùng một định nghĩa, `build_model(name, dim, ...)`.

---

## Bố cục

```
src/Assignment 05/
├── a05/                 gói dùng chung, mỗi trách nhiệm một tệp
│   ├── concepts.py        mục 1: neuron, kích hoạt, conv2d + lan truyền ngược, pooling, softmax (NumPy)
│   ├── blocks.py          mục 2: LeNet, AlexNet, VGG, Inception, Residual, Dense, Depthwise, SE, CBAM, ViT
│   ├── data.py            mục 3: CIFAR nạp lên GPU + augmentation trên GPU; pipeline Diabetes
│   ├── models.py          mục 4: BasicCNN, StagedCNN (VGG / ResNet / SE-ResNet), build_model
│   ├── train.py           mục 4: vòng lặp huấn luyện chung, checkpoint theo val loss
│   ├── metrics.py         mục 5: top-k, macro-F1, ROC/PR, ma trận nhầm lẫn, MACs, thời gian suy luận
│   ├── experiment.py      một lượt thí nghiệm đầy đủ cho một tập
│   └── export.py          nơi duy nhất ghi .dat (cho pgfplots) và .json (số liệu)
├── notebooks/           00 → 06, đã chạy, có output
├── data/                cifar10/, cifar100/ (không commit), diabetes/
├── models/              <tập>_<mô hình>.pt + metadata.json (12 checkpoint)
└── outputs/
    ├── figdata/           *.dat, dữ liệu của mọi hình trong báo cáo
    └── metrics/           *.json, mọi con số trong báo cáo
```

Báo cáo không có con số nào gõ tay: `Report/Assignment 05/build.py` biến `outputs/metrics/*.json` thành
macro LaTeX, và trích mã nguồn trong báo cáo thẳng từ `a05/*.py` theo tên hàm. Mọi hình vẽ bằng TikZ/pgfplots
từ `outputs/figdata/*.dat`, kể cả ảnh mẫu và feature map (vẽ từng điểm ảnh).

---

## Chạy lại từ đầu

```bash
cd "src/Assignment 05"
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m ipykernel install --user --name a05
```

Tải hai tập CIFAR (không nằm trong kho vì mỗi tệp khoảng 170 MB, vượt giới hạn 100 MB của GitHub):

```bash
curl -L -o data/cifar10/cifar-10-python.tar.gz   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
curl -L -o data/cifar100/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

Chạy các notebook theo thứ tự số (06 đọc kết quả của 03, 04, 05):

```bash
cd notebooks
for nb in 00 01 02 03 04 05 06; do
  python -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=a05 --ExecutePreprocessor.timeout=-1 ${nb}_*.ipynb
done
```

Cuối cùng dựng báo cáo (cần MiKTeX hoặc TeX Live có LuaLaTeX, `latexmk`, `biber`, gói `minted`):

```bash
python "../../../Report/Assignment 05/build.py"
```

Lần dựng đầu mất khoảng 15–20 phút vì mọi hình được biên dịch riêng và cache vào `tikzcache/`; các lần sau chỉ
vẽ lại hình nào có mã hoặc dữ liệu thay đổi.

---

## Dữ liệu

| Tập | Nguồn |
|---|---|
| CIFAR-10 | [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) |
| CIFAR-100 | [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Diabetes Prediction Dataset | [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (tệp CSV nằm sẵn trong `data/diabetes/`) |

Đề ghi "MNIST-100". Không có bộ dữ liệu chuẩn nào mang tên này, nên bài hiểu là **CIFAR-100**, tập 100 lớp
cùng họ với CIFAR-10. Báo cáo ghi rõ cách hiểu này ở Chương 3.

---

## Ghi chú về môi trường chạy

- Hai tập ảnh huấn luyện trên **GPU** (RTX 4060 Laptop, AMP fp16). Toàn bộ tập ảnh nằm sẵn trên VRAM và
  augmentation chạy trên GPU, nên không dùng `DataLoader`. Bố cục `channels_last` đo được chậm hơn 3,6 lần trên máy
  này nên không dùng.
- Diabetes huấn luyện trên **CPU** 8 luồng: mô hình dưới 70 nghìn tham số, chuyển dữ liệu lên GPU tốn hơn phần tính.
- GPU laptop tự hạ xung vì nhiệt trong các lượt huấn luyện ảnh, nên số giây mỗi epoch trong báo cáo chỉ dùng để so
  sánh các mô hình với nhau.
- Mỗi mô hình chạy **một seed** (42). Báo cáo coi chênh lệch dưới 0,5 điểm là nằm trong mức dao động của một lần chạy.
