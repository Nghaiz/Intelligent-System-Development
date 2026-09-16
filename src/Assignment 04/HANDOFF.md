# Assignment 04 — Điểm dừng ngày 2026-09-16

Phiên làm việc dừng theo yêu cầu tắt máy. Tài liệu này ghi lại chính xác trạng thái để phiên sau
tiếp tục được ngay, không phải dò lại.

Đọc kèm: [`CONTRACT.md`](CONTRACT.md) là nguồn chân lý cho mọi tên file, khoá JSON và danh mục hình.

---

## Tiến độ theo miền

| Miền | Notebook | Hình | metrics JSON | Trạng thái |
|---|---|---|---|---|
| `theory_figures` | — | 9/9 | — | **Xong**, đã commit |
| `customer_comments` | 3/3 chạy hết | 4/4 | có | **Xong**, đã commit |
| `diabetes` | 3/3 chạy hết | 4/4 | có | **Xong**, đã commit |
| `mnist` | 3/3 chạy hết | 10/10 | có | **Xong**, đã commit |
| `house_price` | 01 chạy hết · 02, 03 đã viết chưa chạy | 1/4 | chưa | Dở dang |
| `cifar10` | chưa có | 0/10 | chưa | Chưa bắt đầu |
| `mlp_vs_cnn` | chưa có | 0/5 | chưa | Chưa bắt đầu |

Tổng: **18/46 hình** đã có. Báo cáo dựng được ngay nhưng 28 chỗ hình sẽ hiện ô báo thiếu.

## Kết quả đã đo được

**Customer Comments** (1D CNN văn bản, cùng 509.665 tham số cả ba khung)

| Khung | Accuracy | F1 | Thời gian |
|---|---|---|---|
| NumPy thuần | 88,31% | 93,01% | 353,6 s |
| PyTorch | 88,05% | 92,95% | 23,9 s |
| TensorFlow | 87,81% | 92,71% | 21,3 s |

**Diabetes** (1D CNN bảng, cùng 1.377 tham số cả ba khung)

| Khung | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| NumPy thuần | 97,00% | 79,87% | 0,9735 |
| PyTorch | 97,02% | 80,02% | 0,9747 |
| TensorFlow | 96,76% | 78,61% | 0,9723 |

Thực nghiệm hoán vị cột: F1 chỉ đổi 0,68% khi xáo trộn thứ tự 8 cột. Đây là bằng chứng định
lượng cho luận điểm dữ liệu bảng không có tô-pô, và là đóng góp phương pháp luận của Chương 4.

**MNIST** (2D CNN)

| Mô hình | Accuracy | Macro-F1 | Tham số |
|---|---|---|---|
| NumPy cơ sở | 97,03% | 97,01% | 27.562 |
| NumPy cải tiến | 98,23% | 98,22% | 52.138 |
| PyTorch | 98,98% | 98,98% | 421.738 |
| TensorFlow | 99,02% | 99,01% | 421.738 |

Kiểm chứng gradient bằng sai phân hữu hạn: 16 toạ độ, sai số tương đối lớn nhất 1,86e-10.

**House Price** (mới có NumPy): R² = 0,399 trên thang log, RMSE 559.142 USD.
Con số này đã được kiểm tra chéo: độ lệch chuẩn của `log_price` là 0,8825 và `rmse_log` là
0,6845, nên 1 − (0,6845/0,8825)² = 0,398, khớp. Mô hình thiếu khớp vì 8 đặc trưng theo hợp đồng
thuần về cấu trúc nhà và không mang tín hiệu địa lý, trong khi giá bất động sản bị chi phối bởi
vị trí. **Đây là kết quả trung thực, không phải lỗi.** Không thêm đặc trưng địa lý để kéo số lên,
vì làm vậy sẽ phá tính so sánh được với các miền khác.

---

## Việc còn lại, theo đúng thứ tự

### 1. Hoàn tất `house_price`

```bash
cd "src/Assignment 04/house_price/notebooks"
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 02_house_price_1d_cnn_pytorch.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 03_house_price_1d_cnn_tensorflow.ipynb
```

Notebook 03 phải gộp `fig_house_loss_curves.png`, `fig_house_scatter.png`,
`fig_house_benchmark.png` và ghi `reports/metrics_house_price.json` theo lược đồ hồi quy ở
CONTRACT.md §5.3.

### 2. Lấy nốt dữ liệu CIFAR-10

Bản tải dở **đã được giữ lại** ở `cifar10/data/cifar-10-python.tar.gz`, được
148.815.680 trên 170.498.071 byte (87%). Script dưới đây tải tiếp từ đúng chỗ đó bằng HTTP Range,
không tải lại từ đầu:

```bash
python "src/Assignment 04/download_cifar10.py"
```

Mạng lúc làm việc rất chậm, khoảng 3–4 MB mỗi phút, nên phần còn lại mất chừng 6–8 phút.

### 3. Dựng miền `cifar10` và `mlp_vs_cnn`

Hai miền này chưa có notebook nào. Brief đầy đủ nằm trong CONTRACT.md §4, §5, §6.
Điểm cần nhớ:

- `cifar10` tái sử dụng ngăn xếp NumPy 2D CNN đã kiểm chứng trong
  `mnist/notebooks/01_mnist_cnn_scratch.ipynb`, chỉnh cho 3 kênh màu, và **phải chạy lại kiểm
  chứng gradient** vì lỗi xử lý kênh là đúng loại lỗi mà phép kiểm đó bắt được.
- Model PyTorch của `cifar10` phải có `extract_features(x)` trả vector ẩn 128 chiều, lưu
  `models/cifar10_cnn_pytorch.pt`, `models/cifar10_cnn_def.py`, `models/cifar10_preproc.json`.
  `mlp_vs_cnn` nạp lại đúng ba tệp này (MNIST đã có sẵn bộ tương ứng).
- `mlp_vs_cnn` ghi `reports/metrics_mlp_vs_cnn.json` với khoá `pca.cifar10.vehicle_animal_separation`.
  Chương 8 và mục 9.5 của báo cáo **đọc khoá này rồi tự đổi cách diễn đạt**: nếu mức tách yếu,
  báo cáo nói đúng là yếu thay vì kể câu chuyện gọn gàng hơn dữ liệu cho phép.

### 4. Dựng báo cáo

```bash
python "Report/Assignment 04/build_report.py"
```

Cần đủ **cả sáu** tệp `metrics_*.json` thì mới chạy được, vì `load_all()` báo lỗi nếu thiếu.

---

## Ghi chú kỹ thuật cho phiên sau

**Ngân sách lượt của sub-agent.** Bốn trong sáu agent chạm trần 40 lượt trước khi xong việc. Khi
giao lại, hoặc chia nhỏ phạm vi, hoặc nâng `maxTurns`, hoặc dặn agent commit sớm và thường xuyên.

**Số luồng của PyTorch.** Agent `house_price` đo được `torch.set_num_threads(32)` cho 40 giây mỗi
epoch, còn 4 luồng chỉ mất 2,4 giây mỗi epoch, nhanh hơn 16 lần. Trên máy này nên đặt
`torch.set_num_threads(4)` trong mọi notebook PyTorch. Đây cũng là lý do PyTorch đo được 292 giây
ở miền diabetes trong khi NumPy chỉ mất 67 giây, một kết quả ngược với trực giác thông thường.

**Heredoc và tiếng Việt.** Hai agent bị hỏng lệnh khi viết nội dung tiếng Việt qua heredoc của
shell. Dùng thẳng công cụ ghi tệp thay cho `cat > file <<'EOF'`.

**Thư mục hình của báo cáo.** `Report/Assignment 04/figures/` là bản sao sinh tự động, đã được
`.gitignore` chặn. `build_report.py` tự gom lại từ sáu miền mỗi lần chạy.

**Một lỗi đã sửa, đáng nhớ.** Notebook MNIST ghi `numpy_subset` lồng trong `dataset` và
`gradient_check` theo dạng `{rows, max_rel_error}` thay vì dạng phẳng mà hợp đồng mô tả. Bộ dựng
báo cáo tra theo tên, không thấy, và **bỏ qua trong im lặng**: ghi chú về tập con cùng toàn bộ
bảng kiểm chứng gradient sẽ biến mất khỏi bản PDF mà không có một cảnh báo nào. Nay
`chapters_2d.py` đọc được cả hai cách bố trí. Khi thêm miền mới, hãy kiểm tra mục tương ứng có
thực sự hiện ra trong HTML chứ đừng tin là nó hiện.
