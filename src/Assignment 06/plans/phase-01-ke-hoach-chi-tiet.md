# Assignment 06 — Kế hoạch chi tiết (một phase duy nhất)

**Đề tài:** Mạng nơ-ron hồi quy (RNN, LSTM, GRU, BiLSTM) trên hai tập dữ liệu chuỗi thực tế, hiện thực bằng PyTorch và Keras, triển khai Web App
**Sinh viên:** Nguyễn Duy Nghĩa · B23DCCN600 · D23CTPM01 · GVHD PGS.TS Trần Đình Quế
**Sản phẩm cuối:** `Report/Assignment 06/A06_CT_nghiand.600.pdf` và mã nguồn trong `src/Assignment 06/`
**Trạng thái:** hoàn thành (2026-09-28). 7 notebook đã chạy, 16 mô hình, Web App qua smoke test, báo cáo 44 trang `A06_CT_nghiand.600.pdf` (bìa theo mẫu A04). Khác kế hoạch: AMZN cũng huấn luyện trên GPU (đo được nhanh hơn CPU khoảng 3 lần); độ trễ Keras đo qua `tf.function`; Chương 1 thêm đường LSTM có bias cổng quên b_f = 3 vì LSTM khởi tạo mặc định suy giảm gradient gần như RNN. Rà soát lại theo báo cáo mẫu (2026-09-28): nâng trần KKBox 20 → 60 epoch (LSTM từng chạm trần), thêm 5 dạng bài toán chuỗi, batch-first, trường thụ cảm/nhân quả Conv1D, bảng thống kê 10 đặc trưng, tương quan với nhãn theo xu hướng, khuyến nghị/RSI trên Web App, mục Hướng phát triển; báo cáo 49 trang

**Nguồn yêu cầu:** trong repo không có đề gốc bài 06 (`TAILIEU/` dừng ở `slide-assign_04`). Yêu cầu được suy ra từ báo cáo mẫu `tmp/A06_CT_tupv.879.pdf` (49 trang, 6 chương). Bảng ở Mục 1 liệt kê 6 hạng mục rút ra từ báo cáo đó; nếu sau này có đề gốc thì đối chiếu lại bảng này trước tiên.

---

## 0. Nguyên tắc chốt (không bàn lại)

| # | Nguyên tắc | Nguồn |
|---|---|---|
| 1 | Chỉ làm đúng 6 hạng mục ở Mục 1. Không đa seed, không kiểm định McNemar, không ablation, không Attention/Transformer, không mobile | người dùng (làm nhanh, đúng đề) |
| 2 | **Mọi hình trong báo cáo vẽ bằng TikZ/pgfplots.** Không có PNG nào ngoài logo bìa. Biểu đồ đọc `.dat` do notebook xuất. Giao diện Web App cũng vẽ lại bằng TikZ (khung giao diện + biểu đồ pgfplots từ dữ liệu thật do API trả về), không chụp màn hình | người dùng |
| 3 | Code xuất hiện ở **mọi chương** (Ch.1–Ch.5), trích tự động từ mã nguồn thật theo tên hàm/lớp, không chép tay vào `.tex` | người dùng |
| 4 | Không có con số nào gõ tay trong báo cáo: mọi số liệu là macro `\SL{tệp/khoá}` sinh từ `outputs/metrics/*.json` | quy ước A04, A05 |
| 5 | Làm đơn luồng, không sub-agent. Tác vụ dài chạy nền song song (huấn luyện GPU cùng lúc huấn luyện CPU, tiền xử lý, biên dịch LaTeX) | người dùng |
| 6 | Dữ liệu: **AMZN** (giá cổ phiếu thật 1997-05-15 → 2024-09-24, bài hồi quy) + **KKBox Churn** (WSDM 2018, log nghe nhạc theo ngày của người dùng thật, bài phân loại) | người dùng chốt 2026-09-28 |
| 7 | 4 kiến trúc × 2 framework × 2 tập = **16 mô hình**, cấu hình giống nhau giữa hai framework | suy từ báo cáo mẫu |
| 8 | Keras chạy backend **TensorFlow trên CPU** (TF 2.21 trên Windows không có GPU). PyTorch huấn luyện trên GPU khi có lợi. Độ trễ suy luận của cả hai framework đo **trên CPU** để so công bằng | người dùng chốt |
| 9 | Web App bản gọn: Flask + một trang HTML/Chart.js, 2 phân hệ, chạy cục bộ | người dùng chốt |
| 10 | Không lặp lỗi của báo cáo mẫu: (a) scaler fit trên toàn bộ dữ liệu → chỉ fit trên train; (b) dự báo mức giá bằng Min-Max fit trên train, trong khi giá test (2019–2024) vượt xa max của train → dự báo log-return, quy đổi ngược ra USD; (c) tập churn mô phỏng cho 100% ở mọi mô hình → dùng dữ liệu thật; (d) không có mốc so sánh → thêm một dòng mốc ngây thơ | rút ra khi đọc báo cáo mẫu |

## 1. Đối chiếu 6 hạng mục với sản phẩm

| Hạng mục (theo báo cáo mẫu) | Module | Notebook | Chương | Hình TikZ chính |
|---|---|---|---|---|
| 1. Cơ sở lý thuyết RNN kèm code | `a06/theory.py` | `00_ly_thuyet_rnn` | Ch.1 | tensor (B,T,D), RNN mở theo thời gian, dòng gradient BPTT, clipping, tế bào LSTM, GRU, BiLSTM, dropout thường vs recurrent dropout |
| 2. Phân tích + tiền xử lý 2 tập lớn thực tế | `a06/data_stock.py`, `a06/data_churn.py` | `01_du_lieu_amzn`, `02_du_lieu_kkbox` | Ch.2 | giá + SMA + volume, return/phân phối/biến động/tương quan, ACF/PACF, cửa sổ trượt, quỹ đạo churn vs loyal, phân bố nhãn |
| 3. Hiện thực + huấn luyện PyTorch | `a06/models_torch.py`, `a06/train_torch.py` | `03_pytorch_amzn`, `04_pytorch_kkbox` | Ch.3 | đường loss, giá thật vs dự báo, ma trận nhầm lẫn, ROC/PR |
| 4. Hiện thực Keras + đối chuẩn PyTorch vs Keras | `a06/models_keras.py`, `a06/bench.py` | `05_keras`, `06_doi_chuan` | Ch.4 | đường loss Keras, bảng đối chuẩn 16 mô hình, scatter sai số–độ trễ |
| 5. Đóng gói triển khai Web App | `app/` | — | Ch.5 | sơ đồ kiến trúc, 2 khung giao diện vẽ bằng TikZ |
| 6. Kết luận | — | — | Ch.6 | — |

## 2. Phân bổ tài nguyên (GPU RTX 4060 8 GB · CPU 32 luồng)

| Tác vụ | Thiết bị | Lý do |
|---|---|---|
| Đọc `user_logs_v2.csv` (~1,3 GB, ~18 triệu dòng), gom thành tensor 31 ngày | **CPU**, pandas `engine="pyarrow"`, đọc theo khối, lưu `.npz` một lần | I/O + groupby, GPU không giúp gì |
| PyTorch trên KKBox (≈100k chuỗi × 31 × 8) | **GPU**, cuDNN RNN, tensor dữ liệu nằm sẵn trên VRAM, không DataLoader | Nhiều mẫu, batch 512, cuDNN LSTM/GRU nhanh hơn CPU rõ rệt |
| PyTorch trên AMZN (≈4,8k cửa sổ × 30 × 8) | **Chọn bằng đo**: bước smoke chạy 1 epoch trên CPU và GPU, lấy bên nhanh hơn, ghi thiết bị vào `metadata.json` | Mô hình ~20k tham số, tập nhỏ; A04/A05 từng đo GPU chậm hơn CPU với loại mô hình này |
| Keras/TF, cả 2 tập | **CPU**, tiến trình nền riêng, `tf.config.threading` 16 luồng | TF Windows không có GPU. Chạy song song với PyTorch-GPU để không ai chờ ai |
| Đo độ trễ suy luận + thông lượng, cả 16 mô hình | **CPU**, 1 luồng cố định, warm-up 20 lần, lấy trung vị 200 lần, batch 1 và batch 256 | So hai framework trên cùng điều kiện |
| NumPy lý thuyết (Ch.1), ADF, ACF/PACF, xuất `.dat` | CPU | Bản chất tác vụ |
| Flask Web App | CPU | Suy luận một mẫu, không cần GPU |
| Biên dịch LaTeX (LuaLaTeX, externalize) | CPU, chạy nền khi đang viết chương | A05 đã xác nhận LuaLaTeX tránh tràn bộ nhớ |

Mọi notebook in thiết bị đã dùng; đo thời gian trên GPU bọc `torch.cuda.synchronize()`.

**Môi trường:** PyTorch hiện có là `2.9.1+cpu` (không thấy GPU). Tạo `src/Assignment 06/.venv` (Python 3.13), cài `torch` bản CUDA cu126, `tensorflow==2.21`, `keras`, `statsmodels`, `yfinance`, `pyarrow`, `flask`, `scikit-learn`, `nbconvert`, `ipykernel`. Kiểm tra `torch.cuda.is_available() == True` trước khi làm bất cứ gì khác.

## 3. Bố cục thư mục

```
src/Assignment 06/
├── README.md                  hướng dẫn chạy lại (theo mẫu A05)
├── requirements.txt
├── plans/                     kế hoạch này
├── a06/                       gói dùng chung, mỗi trách nhiệm một tệp
│   ├── theory.py              Ch.1: forward RNN, tế bào LSTM/GRU NumPy, BPTT, chuẩn gradient theo độ dài, clip_grad_norm
│   ├── data_stock.py          Ch.2: tải AMZN, đặc trưng tài chính, ADF, cửa sổ trượt, chia theo thời gian, scaler fit trên train
│   ├── data_churn.py          Ch.2: đọc KKBox, gom 31 ngày × 8 đặc trưng, lấy mẫu phân tầng, chia 70/15/15
│   ├── models_torch.py        Ch.3: PyTorchRNN (rnn | lstm | gru | bilstm), build_torch(cell, task)
│   ├── train_torch.py         Ch.3: vòng lặp AdamW + clipping + early stop theo val, checkpoint .pth
│   ├── models_keras.py        Ch.4: build_keras(cell, task) — import TF chỉ ở tệp này
│   ├── bench.py               Ch.4: RMSE/MAE/R², Acc/P/R/F1/ROC-AUC/PR-AUC, tham số, kích thước tệp, độ trễ, FPS
│   └── export.py              nơi duy nhất ghi .dat (pgfplots) và .json (số liệu)
├── app/                       Ch.5: app.py (Flask), templates/index.html, static/app.js
├── data/                      amzn/AMZN.csv (commit), kkbox/ (tệp gốc không commit; .npz đã gom không commit)
├── notebooks/                 00 → 06 như bảng Mục 1
├── models/                    <tập>_<framework>_<cell>.pth|.keras + metadata.json (16 mô hình + scaler)
└── outputs/
    ├── figdata/               *.dat cho mọi hình pgfplots
    └── metrics/               *.json cho mọi con số trong báo cáo

Report/Assignment 06/          chép khung A05: build.py, main.tex, preamble/, chapters/, tikz/, figdata/, code/, generated/, refs.bib
```

## 4. Nội dung chi tiết từng hạng mục

### Hạng mục 1: cơ sở lý thuyết (`a06/theory.py`, notebook 00, Ch.1)

Mỗi khái niệm: công thức, một hàm NumPy, một dòng so khớp với PyTorch (`np.allclose`) khi có lớp tương ứng.

| Khái niệm | Hàm | Kiểm chứng |
|---|---|---|
| Dữ liệu chuỗi, vi phạm i.i.d., tensor `X ∈ R^{B×T×D}` | — | in shape tensor thật của 2 tập |
| Vì sao MLP và Conv1D chưa đủ: số tham số MLP `(T·D)·H + H`, Conv1D `K·D_in·D_out + D_out`, RNN `H(D+H+1)` | `count_params_mlp/conv1d/rnn` | khớp `numel()` |
| Simple RNN forward `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)` | `rnn_forward` | khớp `nn.RNN` |
| BPTT, tích Jacobi `∏ W_hhᵀ diag(1−h²)` | `bptt_grad_norms` | vẽ `‖∂L/∂h_t‖` theo t cho RNN vs LSTM (lấy bằng autograd), thấy suy giảm hàm mũ |
| Triệt tiêu / bùng nổ theo bán kính phổ của `W_hh` | `jacobian_product_norm(rho, T)` | 3 đường ρ = 0,9 / 1,0 / 1,1 |
| Gradient clipping theo chuẩn | `clip_grad_norm` | khớp `nn.utils.clip_grad_norm_` |
| Tế bào LSTM (f, i, o, c̃, cell state) | `lstm_cell` | khớp `nn.LSTMCell` |
| Tế bào GRU (z, r, h̃, hòa trộn lồi) | `gru_cell` | khớp `nn.GRUCell` |
| BiLSTM: nối `[h→_T ; h←_1]` | — | shape đầu ra ×2 |
| Recurrent dropout (mặt nạ cố định theo t) vs dropout thường; LayerNorm | `recurrent_dropout_masks`, `layer_norm` | khớp `nn.LayerNorm` |

Hình TikZ (9): tensor 3D (B,T,D) · RNN mở theo thời gian với tham số dùng chung · đường chuẩn gradient theo bước (pgfplots, dữ liệu thật) · 3 đường ρ · hình học clipping (vector bị co, giữ hướng) · tế bào LSTM · tế bào GRU · BiLSTM hai dòng · mặt nạ dropout thường vs recurrent.
Bảng: MLP vs Conv1D vs RNN (tham số, chia sẻ trọng số, độ dài chuỗi thay đổi, trạng thái nhớ). LSTM vs GRU.

### Hạng mục 2: dữ liệu (`a06/data_stock.py`, `a06/data_churn.py`, notebook 01–02, Ch.2)

| Tập | Nguồn (kiểm `curl -I` trước khi đưa link vào báo cáo) | Quy mô | Chia |
|---|---|---|---|
| AMZN | Yahoo Finance qua `yfinance`, https://finance.yahoo.com/quote/AMZN/history/ ; tải một lần, lưu `data/amzn/AMZN.csv` và commit để tái lập | ≈6.900 phiên 1997-05-15 → 2024-09-24, OHLCV | theo thời gian 70/15/15 (train/val/test), không xáo trộn |
| KKBox Churn | https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data : `train_v2.csv` (nhãn `is_churn`), `user_logs_v2.csv` (log theo ngày tháng 3/2017) | ≈970k người dùng có nhãn; giữ người có log, lấy mẫu phân tầng **100.000 người** | phân tầng 70/15/15, seed 42 |

**AMZN.** Đặc trưng 8 chiều: Close, Open, High, Low, Volume, Return (log), MA20, RSI-14 (thêm MA50, Volatility-20 cho EDA). Tiền xử lý:
- Giá (O/H/L/C, MA20) chia cho Close ngày cuối cửa sổ → bất biến theo mức giá, không cần extrapolate ngoài khoảng train. Volume lấy log. Các cột còn lại z-score **fit trên train**.
- Cửa sổ T = 30, nhãn `r_{t+1} = ln(P_{t+1}/P_t)`. Dự báo quy đổi `P̂_{t+1} = P_t · exp(r̂)` để báo RMSE/MAE bằng USD.
- ADF trên giá và trên return (statsmodels), ACF/PACF 40 trễ.
- Mốc ngây thơ `P̂_{t+1} = P_t` (một dòng trong bảng kết quả), để thấy mô hình có hơn "giữ nguyên giá" hay không.

**KKBox.** Mỗi người dùng một chuỗi 31 ngày × 8 đặc trưng: `num_25, num_50, num_75, num_985, num_100, num_unq, total_secs` (log1p) và cờ `active` (ngày không có log = 0). StandardScaler fit trên train. Nhãn mất cân bằng (≈ vài % đến ~10% churn, số thật in từ dữ liệu) → `pos_weight`, chọn ngưỡng trên val theo F1.
- EDA: phân bố nhãn, quỹ đạo trung bình 31 ngày loyal vs churn cho 3 đặc trưng, t-test Welch từng đặc trưng (một bảng, như báo cáo mẫu), tương quan đặc trưng.

**Tải KKBox cần bạn làm một lần:** đăng nhập Kaggle, bấm chấp nhận luật cuộc thi, tải `train_v2.csv.7z` và `user_logs_v2.csv.7z` vào `src/Assignment 06/data/kkbox/` (hoặc tạo API token để tôi dùng `kaggle competitions download`). Tôi không tự chấp nhận điều khoản thay bạn.

Hình TikZ (8): giá + SMA20/50 + volume · 4 ô (return theo thời gian, histogram vs Gauss, volatility + volume, heatmap tương quan) · ACF/PACF giá vs return · cửa sổ trượt + chia theo thời gian · phân bố nhãn KKBox · quỹ đạo 31 ngày loyal vs churn · histogram `total_secs` 2 nhóm · heatmap tương quan KKBox.
Bảng: thống kê mô tả AMZN · t-test loyal vs churn.

### Hạng mục 3: PyTorch (`a06/models_torch.py`, `a06/train_torch.py`, notebook 03–04, Ch.3)

| Mô hình | Lớp PyTorch | Ghi chú |
|---|---|---|
| Simple RNN | `nn.RNN(8, 64)` | mốc gốc |
| LSTM | `nn.LSTM(8, 64)` | thêm cổng + cell state |
| GRU | `nn.GRU(8, 64)` | 2 cổng, ít tham số hơn LSTM |
| BiLSTM | `nn.LSTM(8, 64, bidirectional=True)` | ghi rõ: với AMZN, chiều ngược chỉ đọc lại quá khứ trong cửa sổ, không nhìn tương lai ngoài cửa sổ |

Đầu: `h_T → Linear(·,32) → ReLU → Linear(32,1)`. Hồi quy dùng MSE trên return; phân loại dùng một logit + `BCEWithLogitsLoss(pos_weight)`.

| | AMZN | KKBox |
|---|---|---|
| Optimizer | AdamW lr 1e-3, wd 1e-4 | như bên trái |
| Clipping | `clip_grad_norm_(…, 1.0)` | như bên trái |
| Epoch · batch | tối đa 60 · 64, early stop patience 8 theo val loss | tối đa 20 · 512, patience 4 |
| Checkpoint | val loss nhỏ nhất → `.pth` (state_dict) | như bên trái |
| Chỉ số | RMSE, MAE (USD), R², so mốc ngây thơ | Acc, Precision, Recall, F1, ROC-AUC, PR-AUC, ngưỡng từ val |
| Seed | 42 | 42 |

Ước lượng: AMZN 4 mô hình ≈ 5 phút tổng; KKBox trên GPU ≈ 3–6 s/epoch → ≈ 10 phút tổng.

Hình TikZ (5): đường loss train/val 4 mô hình AMZN · giá thật vs dự báo 4 mô hình trên 150 phiên cuối test · đường loss KKBox · 4 ma trận nhầm lẫn KKBox · ROC + PR KKBox.

### Hạng mục 4: Keras + đối chuẩn (`a06/models_keras.py`, `a06/bench.py`, notebook 05–06, Ch.4)

- `build_keras(cell, task)`: `Input(T,8)` → `SimpleRNN | LSTM | GRU | Bidirectional(LSTM)` 64 đơn vị → `Dense(32, relu)` → `Dense(1)`. Cùng tối ưu (AdamW, `clipnorm=1.0`), cùng batch, cùng early stop (`EarlyStopping(restore_best_weights=True)`), cùng dữ liệu `.npz` đã chuẩn hóa như PyTorch. Lưu `.keras`.
- Notebook 05 chạy nền trên CPU **cùng lúc** notebook 03–04 chạy GPU.
- Số tham số hai framework chênh nhau ở LSTM/GRU (PyTorch có 2 vector bias, Keras GRU `reset_after`): ghi công thức, không ép bằng.

Bảng đối chuẩn (16 dòng + 1 dòng mốc ngây thơ): framework, mô hình, RMSE/MAE/R² (AMZN) hoặc Acc/Recall/F1/ROC-AUC (KKBox), số tham số, kích thước tệp (KB), thời gian huấn luyện/epoch kèm thiết bị, độ trễ CPU (ms/mẫu, batch 1), thông lượng CPU (mẫu/s, batch 256).

Hình TikZ (3): đường loss Keras (2 ô) · scatter RMSE–độ trễ (AMZN) và F1–độ trễ (KKBox), màu theo framework · cột kích thước tệp/tham số.
Mục thảo luận giới hạn mô hình tài chính: trễ pha tại điểm đảo chiều (so trực tiếp với mốc ngây thơ), nhiễu, sự kiện ngoài dữ liệu. Chỉ nói điều số liệu cho thấy; một seed nên chênh lệch nhỏ ghi là trong mức dao động.

### Hạng mục 5: Web App (`app/`, Ch.5)

- `app.py` (Flask, cổng 8080): nạp sẵn mô hình tốt nhất mỗi tập (cả `.pth` và `.keras`, chọn qua tham số `framework`) và scaler lúc khởi động.
  - `GET /api/stock/history` → 100 phiên cuối (Close, MA20).
  - `POST /api/stock/predict {model, framework}` → giá dự báo ngày kế, % thay đổi, độ trễ ms.
  - `POST /api/churn/predict {sequence 31×8 | preset}` → xác suất churn, mức rủi ro theo ngưỡng từ val, độ trễ.
  - Hai preset lấy từ **người dùng thật** trong tập test (một churn, một loyal có xác suất cao nhất).
- `templates/index.html` + `static/app.js`: 2 tab, Chart.js vẽ giá/MA20 và quỹ đạo 31 ngày.
- Kiểm chứng: script `app/smoke_test.py` gọi 3 endpoint bằng test client, lưu JSON phản hồi vào `outputs/metrics/app.json` và `.dat` cho hình.
- Báo cáo: sơ đồ kiến trúc (TikZ) + 2 khung giao diện vẽ bằng TikZ, biểu đồ bên trong là pgfplots đọc đúng dữ liệu API trả về.

Hình TikZ (3). Code: route predict, hàm nạp mô hình, một đoạn JS gọi API.

## 5. Báo cáo LaTeX

**Khuôn:** chép `Report/Assignment 05/` (`main.tex`, `preamble/`, `build.py`), đổi hằng số `SRC`, `PDF_NAME`, danh sách mô hình. LuaLaTeX + `-shell-escape`, minted, tikz externalize.

| Phần | Nội dung | Code trong chương |
|---|---|---|
| Mở đầu | bìa, lời mở đầu, mục lục, danh mục hình/bảng/code, bảng đối chiếu 6 hạng mục | — |
| Ch.1 Cơ sở lý thuyết RNN | theo bảng hạng mục 1 | 8–9 đoạn (NumPy + đối chiếu PyTorch) |
| Ch.2 Hai tập dữ liệu | nguồn, EDA, ADF/ACF, cửa sổ trượt, chống rò rỉ, KKBox | 5–6 đoạn (tải, đặc trưng, cửa sổ, scaler, gom log KKBox) |
| Ch.3 PyTorch | lớp mô hình, vòng lặp, kết quả 2 tập | 4–5 đoạn |
| Ch.4 Keras + đối chuẩn | mô hình Keras, callbacks, đo độ trễ, bảng 16 mô hình, thảo luận | 4–5 đoạn |
| Ch.5 Web App | kiến trúc, API, giao diện | 3 đoạn |
| Ch.6 Kết luận | trả lời 6 hạng mục, hạn chế (một seed, lấy mẫu 100k người, 31 ngày) | — |
| Tài liệu tham khảo, Phụ lục tái lập | lệnh chạy, phiên bản thư viện, thiết bị | 1 đoạn |

Khoảng **25–28 đoạn code** trải qua Ch.1–Ch.5, **≈28 hình TikZ**, ≈6 bảng sinh tự động. Ước tính 50–60 trang.

## 6. Trình tự thực hiện (đơn luồng, việc dài chạy nền)

| Bước | Việc | Thiết bị | Ước lượng | Kiểm chứng |
|---|---|---|---|---|
| 1 | Tạo `.venv`, cài torch CUDA + TF + phụ thuộc; khung thư mục, `.gitignore` (data/kkbox, `.npz`, `.venv`) | CPU | 20 phút | `torch.cuda.is_available()` True; `import tensorflow` chạy |
| 2 | Tải AMZN (yfinance) → CSV. **Bạn tải KKBox** (Mục 4, hạng mục 2); tôi giải nén và gom thành `.npz` chạy nền | CPU | 15 phút + nền 10–20 phút | số phiên AMZN khớp khoảng ngày; `.npz` có đúng shape (N,31,8), tỉ lệ nhãn in ra |
| 3 | Chép khung báo cáo A05; thử 1 hình pgfplots + 1 listing + 1 macro | CPU | 20 phút | PDF thử đúng |
| 4 | `theory.py` + notebook 00 | CPU | 45 phút | mọi `allclose` True |
| 5 | `data_stock.py`, `data_churn.py` + notebook 01, 02 (EDA, ADF, xuất `.dat`) | CPU | 1 giờ | scaler chỉ fit train; không có ngày test trong train |
| 6 | `models_torch.py`, `train_torch.py`, `models_keras.py`, `bench.py`, `export.py`; smoke 1 epoch mỗi mô hình, đo CPU vs GPU cho AMZN để chốt thiết bị | GPU + CPU | 45 phút | loss giảm; đủ khoá metadata |
| 7 | Chạy song song: notebook 03 + 04 (PyTorch, GPU) ‖ notebook 05 (Keras, CPU nền) | GPU ‖ CPU | ≈20–30 phút đồng hồ | 16 mô hình + history + metrics JSON |
| 8 | Notebook 06: đo độ trễ CPU 16 mô hình, bảng đối chuẩn, xuất `.dat` | CPU | 20 phút | bảng đủ 17 dòng |
| 9 | `app/` + `smoke_test.py` | CPU | 1 giờ | 3 endpoint trả 200, JSON đúng khoá |
| 10 | Vẽ ≈28 hình `tikz/*.tex` (biên dịch nền theo lô) | CPU | 3 giờ | từng hình biên dịch được |
| 11 | Viết 6 chương + mở đầu, chạy `build.py` | CPU | 3 giờ | không còn `??`, mục lục đủ |
| 12 | Rà soát cuối theo checklist Mục 8, cập nhật README bài và README gốc (dòng 06), commit | CPU | 30 phút | checklist qua hết |

**Tổng ước lượng: ≈11–12 giờ làm việc**, máy chạy huấn luyện chỉ ≈30 phút đồng hồ nhờ chạy song song GPU/CPU. Tốn nhất là vẽ TikZ và viết chương.

## 7. Rủi ro và cách xử lý

| Rủi ro | Xử lý |
|---|---|
| Không tải được KKBox (chưa chấp nhận luật cuộc thi, không có token) | Dừng ở bước 2 và báo bạn; không tự đổi sang dữ liệu mô phỏng |
| `user_logs_v2.csv` lớn, đọc chậm/tràn RAM | đọc theo khối với pyarrow, chỉ giữ user thuộc mẫu 100k, ép kiểu float32/int32 |
| Cài torch CUDA đụng bản CPU đang có ở Python toàn cục | cài trong `.venv` riêng của bài, không động môi trường toàn cục |
| TF 2.21 + Python 3.13 cài lỗi trong venv | dùng bản TF đang chạy được ở Python toàn cục (2.21) làm tham chiếu phiên bản; nếu vẫn lỗi thì báo bạn trước khi đổi phiên bản |
| Mô hình AMZN không hơn mốc ngây thơ | ghi đúng như số liệu, phân tích ở Ch.4 (đây là kết quả thường gặp với giá cổ phiếu, không phải lỗi) |
| Keras và PyTorch lệch số tham số | giải thích bằng công thức bias, không ép bằng |
| Hình nhiều điểm (6.900 phiên) làm TeX chậm | tỉa điểm khi xuất `.dat` cho hình toàn giai đoạn (1 điểm/tuần), giữ đủ điểm cho hình 150 phiên test |

## 8. Checklist hoàn thành

- [ ] 7 notebook (00–06) đã chạy từ đầu đến cuối, có output
- [ ] 16 mô hình: 8 `.pth` + 8 `.keras`, kèm scaler và `metadata.json` (cấu hình, thiết bị, epoch tốt nhất)
- [ ] Bảng đối chuẩn đủ 16 dòng + mốc ngây thơ, độ trễ đo trên CPU
- [ ] Web App chạy, `smoke_test.py` qua 3 endpoint
- [ ] Ch.1–Ch.5 mỗi chương có ≥3 đoạn code trích từ `a06/` hoặc `app/`
- [ ] Mọi hình là TikZ/pgfplots, `grep includegraphics` chỉ còn logo bìa
- [ ] Không có con số gõ tay: mọi số liệu là macro `\SL{…}` sinh từ JSON
- [ ] Link AMZN và KKBox kiểm tra trả HTTP 200
- [ ] PDF tên `A06_CT_nghiand.600.pdf`, README gốc thêm dòng bài 06
- [ ] Không đẩy gì ra dịch vụ ngoài (không Plane, không Artifact); chỉ `git push` repo khi bạn yêu cầu
