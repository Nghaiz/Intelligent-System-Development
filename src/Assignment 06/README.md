# Assignment 06: Mạng nơ-ron hồi quy trên hai tập dữ liệu chuỗi thực tế

RNN, LSTM, GRU và BiLSTM bằng PyTorch và Keras, triển khai Web App.

**Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · Lớp D23CTPM01
**Giảng viên hướng dẫn:** PGS.TS Trần Đình Quế
**Học kỳ:** Học kỳ 1 năm học 2026 – 2027

Báo cáo: [`Report/Assignment 06/A06_CT_nghiand.600.pdf`](../../Report/Assignment%2006/A06_CT_nghiand.600.pdf)

---

## Bài này làm gì

Sáu hạng mục, mỗi hạng mục một module trong `a06/` (hoặc `app/`), một notebook và một chương báo cáo:

| Hạng mục | Module | Notebook | Chương |
|---|---|---|---|
| 1. Cơ sở lý thuyết RNN kèm code | `a06/theory.py` | `00_ly_thuyet_rnn` | 1 |
| 2. Phân tích và tiền xử lý hai tập dữ liệu | `a06/data_stock.py`, `a06/data_churn.py` | `01_du_lieu_amzn`, `02_du_lieu_kkbox` | 2 |
| 3. Hiện thực và huấn luyện bằng PyTorch | `a06/models_torch.py`, `a06/train_torch.py` | `03_pytorch_amzn`, `04_pytorch_kkbox` | 3 |
| 4. Keras và đối chuẩn PyTorch với Keras | `a06/models_keras.py`, `a06/bench.py` | `05_keras`, `06_doi_chuan` | 4 |
| 5. Web App | `app/` | | 5 |
| 6. Kết luận | | | 6 |

4 kiến trúc × 2 framework × 2 tập = **16 mô hình**, cùng cấu hình ở hai framework: tầng hồi quy 64 đơn vị →
Linear(32) → ReLU → Linear(1), AdamW, clip gradient 1,0, dừng sớm theo val loss.

| Tập | Bài toán | Quy mô | Chia |
|---|---|---|---|
| AMZN (Yahoo Finance) | hồi quy: giá đóng cửa phiên kế tiếp | 6.885 phiên, 1997-05-15 → 2024-09-24, cửa sổ 30 phiên × 8 đặc trưng | theo thời gian 70/15/15 |
| KKBox Churn (Kaggle, WSDM 2018) | phân loại: người dùng có rời bỏ không | 100.000 người lấy mẫu phân tầng, chuỗi 31 ngày × 8 đặc trưng | phân tầng 70/15/15 |

Hai điểm tiền xử lý đáng chú ý:

- AMZN dự báo **log-return** rồi quy ra giá, vì giá không dừng: giá lớn nhất ở tập test gấp 5 lần giá lớn nhất ở train.
  Mọi scaler chỉ fit trên train. Kết quả được so với mốc ngây thơ "giá ngày mai = giá hôm nay", và không mô hình nào
  thắng được mốc đó. Báo cáo phân tích vì sao.
- Tệp log KKBox (1,4 GB, 18,4 triệu dòng) được đọc theo khối bằng pyarrow, không nạp trọn vào RAM, và gom thành
  `data/kkbox/kkbox_seq.npz` trong khoảng một phút.

---

## Bố cục

```
src/Assignment 06/
├── a06/                 gói dùng chung, mỗi trách nhiệm một tệp
│   ├── theory.py          Ch.1: RNN forward, tích Jacobi, gradient theo thời gian, clipping, tế bào LSTM/GRU, dropout, LayerNorm
│   ├── data_stock.py      Ch.2: AMZN, đặc trưng tài chính, ADF, cửa sổ trượt, chia theo thời gian, scaler fit trên train
│   ├── data_churn.py      Ch.2: KKBox, lấy mẫu phân tầng, gom log thành (N, 31, 8), chia 70/15/15
│   ├── models_torch.py    Ch.3: PyTorchRNN (rnn | lstm | gru | bilstm)
│   ├── train_torch.py     Ch.3: vòng lặp huấn luyện, chọn CPU/GPU bằng đo
│   ├── models_keras.py    Ch.4: build_keras, fit_keras (TensorFlow chỉ import ở tệp này)
│   ├── bench.py           Ch.4: chỉ số, ngưỡng từ val, độ trễ suy luận, lưu kết quả
│   └── export.py          nơi duy nhất ghi .dat (pgfplots) và .json (số liệu)
├── app/                 Ch.5: app.py (Flask), templates/, static/app.js, smoke_test.py
├── notebooks/           00 → 06, đã chạy, có output
├── data/amzn/AMZN.csv   dữ liệu giá (có trong kho để tái lập)
├── data/kkbox/          dữ liệu KKBox (không commit, tải theo hướng dẫn dưới)
├── models/              16 mô hình (.pth, .keras) + metadata.json cho Web App
└── outputs/
    ├── figdata/           *.dat, dữ liệu của mọi hình trong báo cáo
    ├── metrics/           *.json, mọi con số trong báo cáo
    └── preds/             dự báo val/test của 16 mô hình
```

Báo cáo không có con số nào gõ tay: `Report/Assignment 06/build.py` biến `outputs/metrics/*.json` thành macro LaTeX và
trích mã nguồn thẳng từ `a06/`, `app/` theo tên hàm. Mọi hình vẽ bằng TikZ/pgfplots, kể cả hai khung giao diện của Web App
(vẽ lại từ phản hồi thật của API).

---

## Chạy lại từ đầu

```bash
cd "src/Assignment 06"
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m ipykernel install --user --name a06 --display-name "Python (A06)"
```

Tải dữ liệu KKBox (cần tài khoản Kaggle và bấm chấp nhận luật cuộc thi): vào
[trang dữ liệu của cuộc thi](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data), tải
`train_v2.csv.7z` và `user_logs_v2.csv.7z` vào `data/kkbox/` rồi giải nén:

```bash
cd data/kkbox
7z e train_v2.csv.7z
7z e user_logs_v2.csv.7z
```

Chạy notebook theo thứ tự số. Notebook 03–04 (GPU) chạy song song được với 05 (Keras trên CPU). Đặt
`OPENBLAS_NUM_THREADS=8` trước khi chạy: trên máy 32 luồng, OpenBLAS mặc định cấp bộ đệm cho 32 luồng và có lúc làm
kernel chết (`OpenBLAS error: Memory allocation still failed`).

```bash
cd notebooks
set OPENBLAS_NUM_THREADS=8
for %n in (00 01 02 03 04 05 06) do python -m jupyter nbconvert --to notebook --execute --inplace ^
    --ExecutePreprocessor.kernel_name=a06 --ExecutePreprocessor.timeout=-1 %n_*.ipynb
```

Web App:

```bash
python app/smoke_test.py      # kiểm chứng 3 endpoint bằng test client
python app/app.py             # rồi mở http://localhost:8080
```

Báo cáo (cần MiKTeX hoặc TeX Live có LuaLaTeX, `latexmk`, `biber`, gói `minted`):

```bash
python "../../Report/Assignment 06/build.py"
```

---

## Ghi chú về môi trường chạy

- PyTorch huấn luyện trên **GPU** (RTX 4060 Laptop) cho cả hai tập. Với AMZN, notebook 03 đo một epoch trên CPU và GPU
  rồi mới chọn; GPU nhanh hơn khoảng 3 lần nhờ nhân cuDNN của LSTM.
- TensorFlow ≥ 2.11 không hỗ trợ GPU trên Windows, nên Keras chạy trên **CPU** 16 luồng, song song với PyTorch trên GPU.
- Độ trễ suy luận của cả 16 mô hình đo trên **CPU, 1 luồng**. Mô hình Keras chạy qua `tf.function`, vì gọi eager chậm hơn
  khoảng 100 lần. Đo hết PyTorch rồi mới đo Keras: các luồng của TensorFlow còn quay chờ sau mỗi lần gọi và làm chậm
  PyTorch nếu đo xen kẽ.
- Mỗi mô hình chạy **một seed** (42). Chênh lệch nhỏ giữa các mô hình và giữa hai framework nằm trong mức dao động của một
  lần chạy.
