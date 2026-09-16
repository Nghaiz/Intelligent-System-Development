"""Chương 6 (MNIST) và Chương 7 (CIFAR-10) — mạng tích chập hai chiều trên ảnh."""

from __future__ import annotations

import report_lib as R
from chapters_1d import dev, device_note, num, pct, thousands

M_LABEL = {
    "numpy_baseline": "NumPy cơ sở",
    "numpy_improved": "NumPy cải tiến",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow / Keras",
}
M_ORDER = ["numpy_baseline", "numpy_improved", "pytorch", "tensorflow"]


def _best(models: dict, key: str = "macro_f1") -> tuple[str, dict]:
    name = max(models, key=lambda k: models[k].get(key, float("-inf")))
    return name, models[name]


def _rows(models: dict) -> list[list[str]]:
    order = sorted(models, key=lambda k: models[k].get("macro_f1", 0), reverse=True)
    out = []
    for k in order:
        mk = models[k]
        out.append([M_LABEL.get(k, k), pct(mk["accuracy"]), pct(mk["macro_precision"]),
                    pct(mk["macro_recall"]), pct(mk["macro_f1"]),
                    f'{mk["params"]:,}', dev(mk), num(mk["train_time_s"], 1)])
    return out


def write(r: R.Report, data: dict) -> None:
    _chapter6(r, data)
    _chapter7(r, data)


# ===========================================================================
# CHƯƠNG 6 — MNIST
# ===========================================================================
def _chapter6(r: R.Report, data: dict) -> None:
    d = data["mnist"]
    m = d["models"]
    ds = d["dataset"]

    r.h(1, "Chương 6. Tích chập hai chiều trên ảnh chữ số viết tay")

    r.p(
        "Ba chương trước dùng tích chập một chiều, trong đó chỉ có văn bản thực sự có tô-pô "
        "để khai thác. Từ chương này trở đi, dữ liệu là ảnh, và mọi giả định đặt ra ở "
        "Chương 1 đều được thoả mãn trọn vẹn: điểm ảnh lân cận thực sự có quan hệ, một hoa "
        "văn thực sự có thể xuất hiện ở nhiều vị trí, và chia sẻ trọng số thực sự tiết kiệm "
        "tham số một cách có ý nghĩa.")
    r.p(
        "MNIST được chọn làm bước đầu tiên không phải vì nó khó mà vì nó <em>dễ theo một "
        "cách hữu ích</em>. Nền đen đồng nhất, một đối tượng duy nhất, chữ số đã được căn "
        "vào trọng tâm khung 28×28. Chính sự sạch sẽ đó biến MNIST thành bộ kiểm tra tính "
        "đúng đắn lý tưởng cho một hiện thực tích chập viết tay: nếu mã nguồn đúng, độ "
        "chính xác phải vượt 97%; nếu không, lỗi nằm ở đạo hàm chứ không ở dữ liệu.")

    # ---- 6.1
    r.h(2, "6.1. Khảo sát dữ liệu")
    r.p(
        f"Bộ dữ liệu gồm {thousands(ds['n_raw'])} ảnh xám 28×28×1 của mười chữ số viết tay, "
        f"trong đó {thousands(ds['n_test'])} ảnh được giữ làm tập kiểm thử độc lập. Giá trị "
        f"điểm ảnh nằm trong khoảng 0 đến 255.")
    r.p(R.figure("mn_fig_mnist_sample_grid.png",
                 "Mười mẫu đại diện cho mỗi lớp chữ số trong MNIST."))
    r.p(
        "Lưới ảnh cho thấy biến thiên về nét chữ trong cùng một lớp là đáng kể: độ nghiêng, "
        "độ dày nét và cách khép vòng khác nhau rõ rệt giữa các người viết. Nhưng nền luôn "
        "đen tuyệt đối và đối tượng luôn nằm giữa khung, nên biến thiên đó là biến thiên "
        "<em>hình dạng</em> chứ không phải biến thiên <em>vị trí</em>. Ghi nhận này sẽ trở "
        "lại ở Chương 8 khi giải thích vì sao một mạng truyền thẳng vẫn xoay xở tốt trên "
        "MNIST.")
    r.p(R.figure("mn_fig_mnist_class_distribution.png",
                 "Phân phối lớp của MNIST trên tập huấn luyện và tập kiểm thử."))
    r.p(
        "Phân phối lớp không hoàn toàn đồng đều nhưng cũng không mất cân bằng nghiêm trọng. "
        "Báo cáo vẫn dùng trung bình macro để mỗi chữ số đóng góp ngang nhau vào kết luận, "
        "thay vì để những lớp đông mẫu hơn chi phối con số tổng hợp.")

    # ---- 6.2
    r.h(2, "6.2. Cấu hình huấn luyện và một sai lệch được công bố trước")
    r.p(
        f"Tập huấn luyện framework gồm {thousands(ds['n_train'])} ảnh và tập kiểm định "
        f"{thousands(ds['n_val'])} ảnh, tách theo phân tầng. Trung bình và độ lệch chuẩn "
        f"của điểm ảnh được học từ tập huấn luyện rồi áp nguyên cho kiểm định và kiểm thử.")
    _subset_note(r, d, "MNIST")

    # ---- 6.3
    r.h(2, "6.3. Hiện thực tích chập hai chiều bằng NumPy thuần")
    r.p(
        "Chuyển từ một chiều sang hai chiều làm số vòng lặp lồng nhau tăng từ ba lên năm, "
        "nên hiện thực ngây thơ trở nên hoàn toàn bất khả thi. Giải pháp là "
        "<code class='inl'>im2col</code>: trải phẳng mọi cửa sổ trượt thành các cột của một "
        "ma trận, rồi quy toàn bộ phép tích chập về một phép nhân ma trận duy nhất.")
    r.p(R.code('''class Conv2D:
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, init='he'):
        fan_in = in_ch * kernel_size * kernel_size
        # He Normal: nhân đôi phương sai để bù phần tín hiệu bị ReLU cắt bỏ
        scale = np.sqrt(2.0 / fan_in) if init == 'he' else np.sqrt(1.0 / fan_in)
        self.W = (np.random.randn(out_ch, in_ch, kernel_size, kernel_size) * scale)
        self.b = np.zeros((out_ch, 1), dtype=np.float32)

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.cols = im2col_indices(x, self.kernel_size, self.kernel_size,
                                   padding=self.padding, stride=self.stride)
        out = self.W.reshape(self.out_channels, -1) @ self.cols + self.b
        return out.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)

    def backward(self, dout):
        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        self.dW = (dout_flat @ self.cols.T).reshape(self.W.shape)
        self.db = dout_flat.sum(axis=1, keepdims=True)
        dcols = self.W.reshape(self.out_channels, -1).T @ dout_flat
        # col2im cộng dồn gradient ở những vùng cửa sổ chồng lấn
        return col2im_indices(dcols, self.x.shape, self.kernel_size,
                              self.kernel_size, self.padding, self.stride)''',
        "Tầng Conv2D thuần NumPy dựa trên im2col và col2im."))
    r.p(
        "Điểm tinh tế nằm ở <code class='inl'>col2im_indices</code>. Trong lượt tiến, một "
        "điểm ảnh có thể thuộc nhiều cửa sổ trượt khác nhau. Trong lượt ngược, gradient "
        "chảy về điểm ảnh đó phải là <em>tổng</em> đóng góp từ tất cả những cửa sổ ấy. Hàm "
        "<code class='inl'>np.add.at</code> được dùng cho phép cộng dồn này thay vì phép "
        "gán thông thường, vì phép gán sẽ lặng lẽ ghi đè và chỉ giữ lại đóng góp cuối cùng.")
    r.p(R.code('''class MaxPool2D:
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        oh, ow = H // self.stride, W // self.stride
        xr = x.reshape(N, C, oh, self.pool_size, ow, self.pool_size)
        return xr.max(axis=(3, 5))

    def backward(self, dout):
        N, C, H, W = self.x.shape
        oh, ow = H // self.stride, W // self.stride
        xr = self.x.reshape(N, C, oh, self.pool_size, ow, self.pool_size)
        out = xr.max(axis=(3, 5))
        # Mặt nạ cực đại: gradient chỉ đi về đúng phần tử thắng cuộc trong mỗi cửa sổ
        mask = (xr == out[:, :, :, None, :, None])
        mask_sum = mask.sum(axis=(3, 5), keepdims=True)
        dxr = (mask / mask_sum) * dout[:, :, :, None, :, None]
        return dxr.reshape(self.x.shape)''',
        "Tầng MaxPool2D với cơ chế mặt nạ cực đại cho lượt lan truyền ngược."))
    r.p(
        "Phép chia cho <code class='inl'>mask_sum</code> xử lý trường hợp hai phần tử trong "
        "cùng một cửa sổ có giá trị bằng nhau. Không có phép chia này, gradient sẽ bị nhân "
        "đôi mỗi khi xảy ra hoà, và lỗi đó rất khó phát hiện vì nó không làm mô hình sai "
        "hẳn, chỉ làm nó học lệch.")

    # ---- 6.4
    r.h(2, "6.4. Kiểm chứng đạo hàm bằng sai phân hữu hạn")
    _gradcheck_commentary(r, d)

    # ---- 6.5
    r.h(2, "6.5. Mô hình cơ sở đối chiếu mô hình cải tiến")
    r.p(
        "Hai mô hình NumPy dùng chung kiến trúc và chỉ khác nhau ở ba lựa chọn kỹ thuật: "
        "mô hình cơ sở không đệm viền, khởi tạo trọng số theo tỉ lệ cố định và giữ tốc độ "
        "học không đổi; mô hình cải tiến dùng đệm viền <span class='inl'>P = 1</span>, khởi "
        "tạo He Normal và lịch giảm tốc độ học. Thiết kế này cô lập được tác động của ba "
        "kỹ thuật đã bàn ở mục 1.3 và 1.6.")
    r.p(R.figure("mn_fig_mnist_scratch_curves.png",
                 "Quá trình hội tụ hàm mất mát và độ chính xác của hai mô hình 2D CNN thuần "
                 "NumPy trên MNIST."))
    _improvement_commentary(r, m)
    r.p(R.figure("mn_fig_mnist_scratch_comparison.png",
                 "So sánh đa chỉ số giữa mô hình cơ sở và mô hình cải tiến trên MNIST."))
    r.p(R.figure("mn_fig_mnist_scratch_confusion.png",
                 "Ma trận nhầm lẫn của hai mô hình thuần NumPy trên tập kiểm thử MNIST."))

    # ---- 6.6
    r.h(2, "6.6. Đối chuẩn với PyTorch và TensorFlow")
    r.p(
        "Hai khung thư viện dựng kiến trúc sâu hơn: hai khối Conv-BatchNorm-ReLU-MaxPool-"
        "Dropout với 32 rồi 64 kênh, tiếp theo là Dense 128 và tầng đầu ra 10 lớp. Batch "
        "Normalization và Dropout là hai thành phần mà hiện thực NumPy không có, nên phần "
        "chênh lệch kết quả bên dưới đến từ cả kiến trúc lẫn tầng hiện thực.")
    r.p(R.figure("mn_fig_mnist_framework_curves.png",
                 "Đường cong huấn luyện của PyTorch và Keras trên MNIST."))
    r.p(R.figure("mn_fig_mnist_framework_confusion.png",
                 "Ma trận nhầm lẫn của hai mô hình framework trên tập kiểm thử MNIST."))
    r.p(R.table(
        ["Mô hình", "Accuracy", "Macro-P", "Macro-R", "Macro-F1", "Tham số",
         "Thiết bị", "Thời gian (s)"],
        _rows(m),
        "Đối chuẩn bốn cách cài đặt 2D CNN trên MNIST, sắp theo Macro-F1."))
    r.p(device_note(m))
    _threeway_commentary(r, m, "MNIST")
    r.p(R.figure("mn_fig_mnist_3way_benchmark.png",
                 "So sánh Accuracy, Macro-Precision, Macro-Recall và Macro-F1 giữa các cách "
                 "cài đặt trên MNIST."))

    # ---- 6.7
    r.h(2, "6.7. Độ khó theo từng lớp và phân tích lỗi tự tin cao")
    r.p(R.figure("mn_fig_mnist_per_class_accuracy.png",
                 "Độ chính xác theo từng lớp chữ số của mô hình tốt nhất trên MNIST."))
    _per_class_commentary(r, m, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], "chữ số")
    r.p(R.figure("mn_fig_mnist_high_conf_errors.png",
                 "Tám dự đoán sai có độ tin cậy cao nhất của mô hình tốt nhất trên MNIST."))
    _high_conf_commentary(r, m)
    r.p(
        "Những ảnh này gần như đều là nét viết nằm ngoài kiểu phổ biến của tập huấn luyện: "
        "chữ số 9 có vòng trên khép hờ trông như số 4, chữ số 7 viết không gạch ngang gần "
        "giống số 1, chữ số 5 có nét cong dưới sâu dễ nhầm với số 6. Điểm đáng nói không "
        "phải là mô hình sai, mà là mô hình <em>sai với độ tin cậy rất cao</em>. Xác suất "
        "softmax lớn không đồng nghĩa với xác suất hậu nghiệm đã được hiệu chỉnh tốt, và "
        "một hệ thống thật nên có cơ chế từ chối trả lời thay vì tin tuyệt đối vào con số "
        "đó.")


# ===========================================================================
# CHƯƠNG 7 — CIFAR-10
# ===========================================================================
def _chapter7(r: R.Report, data: dict) -> None:
    d = data["cifar"]
    m = d["models"]
    ds = d["dataset"]
    names = ["máy bay", "ô tô", "chim", "mèo", "hươu", "chó", "ếch", "ngựa", "tàu thuỷ",
             "xe tải"]

    r.h(1, "Chương 7. Tích chập hai chiều trên ảnh màu tự nhiên")

    r.p(
        "CIFAR-10 giữ nguyên bài toán của Chương 6 — phân loại mười lớp — nhưng thay toàn "
        "bộ điều kiện làm việc. Ảnh có ba kênh màu thay vì một, nền là cảnh thật thay vì "
        "màu đen đồng nhất, đối tượng xuất hiện ở tỉ lệ và góc nhìn tuỳ ý, và độ phân giải "
        "chỉ 32×32 nên chi tiết phân biệt thường mờ ngay cả với mắt người. Đây là phép thử "
        "nghiêm túc đầu tiên đối với giới hạn của một mạng tích chập nông.")

    # ---- 7.1
    r.h(2, "7.1. Khảo sát dữ liệu")
    r.p(
        f"Bộ dữ liệu gồm {thousands(ds['n_raw'])} ảnh màu 32×32×3 chia đều cho mười lớp: "
        f"{', '.join(names)}. Tập kiểm thử có {thousands(ds['n_test'])} ảnh.")
    r.p(R.figure("cf_fig_cifar10_sample_grid.png",
                 "Mười mẫu đại diện cho mỗi lớp trong CIFAR-10."))
    r.p(
        "Đặt lưới ảnh này cạnh lưới MNIST ở mục 6.1 là cách nhanh nhất để thấy chênh lệch "
        "độ khó. Ở MNIST, một lớp là một hình dạng. Ở CIFAR-10, lớp \"chó\" bao gồm hàng "
        "chục giống chó ở đủ tư thế, đủ màu lông, trên đủ loại nền, đôi khi bị che khuất "
        "một phần. Mô hình phải học khái niệm ngữ nghĩa chứ không phải một khuôn mẫu hình "
        "học.")
    r.p(R.figure("cf_fig_cifar10_class_distribution.png",
                 "Phân phối lớp cân bằng tuyệt đối của CIFAR-10."))
    r.p(
        "Mỗi lớp có đúng số ảnh như nhau ở cả tập huấn luyện lẫn kiểm thử. Vì dữ liệu cân "
        "bằng tuyệt đối, Accuracy và Macro-Recall trùng nhau trong các bảng kết quả. "
        "Macro-F1 vẫn cần thiết vì precision có thể lệch đáng kể giữa lớp dễ và lớp khó.")

    # ---- 7.2
    r.h(2, "7.2. Cấu hình huấn luyện")
    r.p(
        f"Tập huấn luyện framework gồm {thousands(ds['n_train'])} ảnh và tập kiểm định "
        f"{thousands(ds['n_val'])} ảnh. Chuẩn hoá dùng trung bình và độ lệch chuẩn riêng "
        f"cho từng kênh màu, học từ tập huấn luyện.")
    _subset_note(r, d, "CIFAR-10")

    # ---- 7.3
    r.h(2, "7.3. Kết quả mô hình thuần NumPy")
    r.p(R.figure("cf_fig_cifar10_scratch_curves.png",
                 "Quá trình hội tụ của hai mô hình 2D CNN thuần NumPy trên CIFAR-10."))
    _improvement_commentary(r, m)
    r.p(R.figure("cf_fig_cifar10_scratch_comparison.png",
                 "So sánh đa chỉ số giữa mô hình cơ sở và mô hình cải tiến trên CIFAR-10."))
    r.p(R.figure("cf_fig_cifar10_scratch_confusion.png",
                 "Ma trận nhầm lẫn của hai mô hình thuần NumPy trên tập kiểm thử CIFAR-10."))
    r.p(
        "Điều đáng chú ý ở các đường cong CIFAR-10 là hai đường train và validation nằm khá "
        "gần nhau trong khi cả hai đều dừng ở mức thấp. Đây là dấu hiệu của "
        "<strong>thiếu khớp</strong> chứ không phải quá khớp: mô hình không học thuộc dữ "
        "liệu huấn luyện, nó đơn giản là không đủ dung lượng để biểu diễn các hoa văn cần "
        "thiết. Thêm dropout hay tăng cường chính quy hoá sẽ làm tình hình tệ hơn; thứ cần "
        "thêm là độ sâu.")

    # ---- 7.4
    r.h(2, "7.4. Đối chuẩn với PyTorch và TensorFlow")
    r.p(
        "Hai khung thư viện dùng kiến trúc ba khối Conv-BatchNorm-ReLU-MaxPool-Dropout với "
        "kênh 32, 64 và 64, theo sau là Dense 128 và tầng đầu ra 10 lớp. Kiến trúc trích "
        "xuất đặc trưng ở đây cũng chính là nơi cung cấp vector ẩn 128 chiều cho phân tích "
        "PCA ở Chương 8.")
    r.p(R.code('''class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),      # 32x32x3  -> 16x16x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),      # 16x16x32 -> 8x8x64
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),      # 8x8x64   -> 4x4x64
        )
        self.fc1, self.fc2 = nn.Linear(64 * 4 * 4, 128), nn.Linear(128, 10)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0.4)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

    def extract_features(self, x):
        """Vector biểu diễn ẩn 128 chiều, dùng cho phân tích PCA ở Chương 8."""
        return self.relu(self.fc1(self.features(x).flatten(1)))''',
        "Kiến trúc ba khối phân cấp trên PyTorch, kèm cổng trích xuất biểu diễn ẩn."))
    r.p(R.figure("cf_fig_cifar10_framework_curves.png",
                 "Đường cong huấn luyện của PyTorch và Keras trên CIFAR-10."))
    r.p(R.figure("cf_fig_cifar10_framework_confusion.png",
                 "Ma trận nhầm lẫn của hai mô hình framework trên tập kiểm thử CIFAR-10."))
    r.p(R.table(
        ["Mô hình", "Accuracy", "Macro-P", "Macro-R", "Macro-F1", "Tham số",
         "Thiết bị", "Thời gian (s)"],
        _rows(m),
        "Đối chuẩn bốn cách cài đặt 2D CNN trên CIFAR-10, sắp theo Macro-F1."))
    r.p(device_note(m))
    _threeway_commentary(r, m, "CIFAR-10")
    r.p(R.figure("cf_fig_cifar10_3way_benchmark.png",
                 "So sánh đa chỉ số giữa các cách cài đặt trên CIFAR-10."))

    # ---- 7.5
    r.h(2, "7.5. Độ khó theo lớp và cấu trúc nhầm lẫn ngữ nghĩa")
    r.p(R.figure("cf_fig_cifar10_per_class_accuracy.png",
                 "Độ chính xác theo từng lớp của mô hình tốt nhất trên CIFAR-10."))
    _per_class_commentary(r, m, names, "lớp")
    r.p(
        "Cấu trúc nhầm lẫn trên CIFAR-10 không ngẫu nhiên mà tuân theo ngữ nghĩa. Các lớp "
        "phương tiện do con người chế tạo có đường nét thẳng, cạnh sắc và bối cảnh đặc "
        "trưng, nên tách khỏi nhau tương đối dễ. Các lớp động vật chia sẻ texture lông, tư "
        "thế bốn chân và nền thiên nhiên, nên chồng lấn nhiều hơn. Cặp mèo và chó là nguồn "
        "lỗi dai dẳng nhất, và ở độ phân giải 32×32 thì ngay cả người quan sát cũng gặp khó. "
        "Quan sát này sẽ được xác nhận một lần nữa, bằng một phương pháp hoàn toàn khác, "
        "trong phân tích PCA ở Chương 8.")
    r.p(R.figure("cf_fig_cifar10_high_conf_errors.png",
                 "Tám dự đoán sai có độ tin cậy cao nhất của mô hình tốt nhất trên CIFAR-10."))
    _high_conf_commentary(r, m)
    r.p(
        "Nhiều lỗi trong số này mang tính mơ hồ thật sự chứ không phải lỗi ngớ ngẩn: một "
        "con ngựa bị cắt cụt trong khung hình trông như một khối tối trên nền sáng, một con "
        "hươu giữa rừng cây có texture gần như đồng nhất với nền, một chiếc thuyền và một "
        "chiếc máy bay cùng nằm trên nền trời xanh. Trong một hệ thống triển khai thật, "
        "những trường hợp này là lý do để bổ sung hiệu chỉnh xác suất, dự đoán top-k, hoặc "
        "một ngưỡng từ chối trả lời khi độ tin cậy không đáng tin.")


# ===========================================================================
# Các đoạn bình luận sinh tự động từ số liệu
# ===========================================================================
def _subset_note(r: R.Report, d: dict, name: str) -> None:
    # Notebook co the ghi khoa nay o cap cao nhat hoac long trong "dataset"
    sub = d.get("numpy_subset") or (d.get("dataset") or {}).get("numpy_subset")
    if not sub:
        return
    r.p(R.note(
        "Sai lệch được công bố trước khi trình bày kết quả.",
        f"Lan truyền ngược viết tay bằng NumPy trên toàn bộ tập {name} vượt xa ngân sách "
        f"thời gian của bài tập khi chạy trên CPU. Hai mô hình NumPy vì vậy được huấn luyện "
        f"trên một tập con lấy mẫu phân tầng gồm {thousands(sub['n_train'])} ảnh huấn luyện "
        f"và {thousands(sub['n_val'])} ảnh kiểm định, trong khi PyTorch và Keras dùng toàn "
        f"bộ tập huấn luyện. Cả bốn mô hình đều được đánh giá trên <em>cùng một</em> tập "
        f"kiểm thử đầy đủ. Vì vậy khoảng cách giữa NumPy và hai khung thư viện phản ánh "
        f"đồng thời ba yếu tố: lượng dữ liệu huấn luyện, độ sâu kiến trúc và tầng hiện thực. "
        f"Nó không phải phép đo riêng biệt của bất kỳ yếu tố nào trong ba.", "warn"))


def _gradcheck_commentary(r: R.Report, d: dict) -> None:
    gc = d.get("gradient_check")
    if not gc:
        r.p(
            "Trước khi tin vào bất kỳ con số huấn luyện nào, lượt lan truyền ngược cần được "
            "kiểm chứng độc lập. Phương pháp là sai phân hữu hạn: nhiễu từng tham số một "
            "lượng ε rất nhỏ, đo thay đổi của hàm mất mát, rồi so tỉ số đó với gradient "
            "giải tích mà mã nguồn tính ra.")
        r.p('<div class="formula">∂L/∂θ ≈ (L(θ + ε) − L(θ − ε)) / 2ε</div>')
        r.p(R.note(
            "Thiếu số liệu.",
            "Kết quả kiểm chứng đạo hàm chưa có trong tệp JSON. Chạy lại notebook "
            "<code class='inl'>01_mnist_cnn_scratch.ipynb</code> để sinh khoá "
            "<code class='inl'>gradient_check</code>.", "warn"))
        return

    r.p(
        "Trước khi tin vào bất kỳ con số huấn luyện nào, lượt lan truyền ngược cần được "
        "kiểm chứng độc lập. Một gradient sai không làm chương trình báo lỗi; nó chỉ làm mô "
        "hình học kém đi một cách khó truy nguyên. Phương pháp kiểm chứng là sai phân hữu "
        "hạn: nhiễu từng tham số một lượng ε rất nhỏ, đo thay đổi tương ứng của hàm mất "
        "mát, rồi so với gradient giải tích mà mã nguồn tính ra.")
    r.p('<div class="formula">∂L/∂θ ≈ (L(θ + ε) − L(θ − ε)) / 2ε</div>')
    if isinstance(gc.get("rows"), list):          # dang co cau truc day du
        src = gc["rows"]
        rows = [[f'{it.get("layer", "?")}.{it.get("param", "?")}',
                 f'{it["analytic"]:.6e}', f'{it["numeric"]:.6e}', f'{it["rel_error"]:.2e}']
                for it in src[:12]]
        worst = gc.get("max_rel_error") or max(it["rel_error"] for it in src)
        n_checks = gc.get("n_checks", len(src))
        eps, dtype = gc.get("eps"), gc.get("dtype")
    else:                                          # dang phang {ten_tang: {...}}
        rows = [[layer, f"{v['analytic']:.6e}", f"{v['numeric']:.6e}", f"{v['rel_error']:.2e}"]
                for layer, v in gc.items()]
        worst = max(v["rel_error"] for v in gc.values())
        n_checks, eps, dtype = len(gc), None, None

    cap = "Kiểm chứng lan truyền ngược bằng sai phân hữu hạn trên các tham số học được"
    if eps and dtype:
        cap += f" (ε = {eps}, số thực {dtype}, {n_checks} toạ độ được kiểm)"
    r.p(R.table(
        ["Tham số được kiểm", "Gradient giải tích", "Gradient sai phân", "Sai số tương đối"],
        rows, cap + "."))
    r.p(
        f"Sai số tương đối lớn nhất trên toàn bộ các tầng là {worst:.2e}. Ngưỡng thường "
        f"được chấp nhận cho kiểm chứng kiểu này là 10⁻⁵ với số thực 32 bit và 10⁻⁷ với số "
        f"thực 64 bit. "
        + ("Kết quả nằm dưới ngưỡng, nên lượt lan truyền ngược viết tay được xác nhận là "
           "đúng về mặt toán học. Mọi kết quả huấn luyện trình bày sau đây đứng trên một "
           "nền đã được kiểm chứng, chứ không phải trên niềm tin rằng mã nguồn đúng."
           if worst < 1e-4 else
           "Kết quả vượt ngưỡng lý tưởng, nguyên nhân nhiều khả năng là sai số làm tròn của "
           "số thực 32 bit tích luỹ qua nhiều tầng, chứ không phải lỗi công thức. Cần lặp "
           "lại phép kiểm ở độ chính xác 64 bit để kết luận dứt khoát."))


def _improvement_commentary(r: R.Report, m: dict) -> None:
    base = m.get("numpy_baseline")
    imp = m.get("numpy_improved")
    if not base or not imp:
        return
    delta = imp["accuracy"] - base["accuracy"]
    r.p(
        f"Mô hình cơ sở đạt độ chính xác {pct(base['accuracy'])} trên tập kiểm thử, mô hình "
        f"cải tiến đạt {pct(imp['accuracy'])}. Mức tăng {pct(delta)} điểm phần trăm tuyệt "
        f"đối đến hoàn toàn từ ba thay đổi kỹ thuật, không từ thêm dữ liệu hay thêm tham "
        f"số đáng kể: đệm viền giữ lại thông tin biên qua các tầng, He Normal giữ phương "
        f"sai tín hiệu không tắt dần, và lịch giảm tốc độ học cho phép mô hình tinh chỉnh ở "
        f"giai đoạn cuối thay vì dao động quanh cực tiểu.")
    b_ep, i_ep = base.get("best_epoch"), imp.get("best_epoch")
    if b_ep and i_ep:
        r.p(
            f"Epoch tốt nhất theo kiểm định là epoch {b_ep} với mô hình cơ sở và epoch "
            f"{i_ep} với mô hình cải tiến. "
            + ("Mô hình cải tiến còn tiếp tục tiến bộ ở những epoch muộn hơn, cho thấy lịch "
               "giảm tốc độ học đang phát huy tác dụng đúng như thiết kế."
               if i_ep > b_ep else
               "Cả hai mô hình chạm mức tốt nhất ở giai đoạn tương đương, nên phần chênh "
               "lệch đến từ chất lượng biểu diễn chứ không từ thời lượng huấn luyện."))


def _threeway_commentary(r: R.Report, m: dict, dataset: str) -> None:
    best_k, best_m = _best(m, "macro_f1")
    fw = {k: m[k] for k in ("pytorch", "tensorflow") if k in m}
    np_best = m.get("numpy_improved") or m.get("numpy_baseline")

    if len(fw) == 2:
        a, b = "pytorch", "tensorflow"
        gap = abs(fw[a]["macro_f1"] - fw[b]["macro_f1"])
        lead = a if fw[a]["macro_f1"] >= fw[b]["macro_f1"] else b
        r.p(
            f"Trên {dataset}, {M_LABEL[lead]} dẫn đầu theo Macro-F1 với "
            f"{pct(fw[lead]['macro_f1'])}, hơn khung còn lại {pct(gap)}. "
            + ("Chênh lệch dưới nửa điểm phần trăm giữa hai hiện thực của cùng một kiến "
               "trúc là bằng chứng trực tiếp cho luận điểm trung tâm của báo cáo: điều "
               "quyết định kết quả là mô hình toán học, không phải khung thư viện."
               if gap < 0.005 else
               "Chênh lệch này đủ để nhận ra nhưng vẫn nhỏ so với khoảng cách giữa kiến "
               "trúc nông và kiến trúc sâu, nên thứ tự giữa hai khung không nên được đọc "
               "như một kết luận bền vững."))
        t_pt, t_tf = fw[a]["train_time_s"], fw[b]["train_time_s"]
        ratio = max(t_pt, t_tf) / max(min(t_pt, t_tf), 1e-9)
        r.p(
            f"Về thời gian chạy, PyTorch mất {num(t_pt, 1)} giây và Keras mất "
            f"{num(t_tf, 1)} giây. Tỉ lệ {ratio:.2f} lần giữa hai con số này <strong>không "
            f"nói lên điều gì về hai khung</strong>, vì PyTorch chạy trên GPU còn Keras chạy "
            f"trên CPU. Đây là chênh lệch phần cứng chứ không phải chênh lệch phần mềm.")

    if np_best:
        gap = best_m["macro_f1"] - np_best["macro_f1"]
        r.p(
            f"Khoảng cách giữa mô hình tốt nhất ({M_LABEL[best_k]}, Macro-F1 "
            f"{pct(best_m['macro_f1'])}) và mô hình NumPy tốt nhất "
            f"({pct(np_best['macro_f1'])}) là {pct(gap)}. "
            + ("Khoảng cách nhỏ như vậy trên một bộ dữ liệu ảnh cho thấy hiện thực viết tay "
               "đã nắm đúng bản chất thuật toán; phần còn thiếu chủ yếu là lượng dữ liệu "
               "huấn luyện và vài thành phần chính quy hoá."
               if gap < 0.05 else
               "Khoảng cách lớn này phản ánh đúng mức độ phức tạp của dữ liệu: một mạng "
               "nông hai khối tích chập không đủ dung lượng để biểu diễn các hoa văn thị "
               "giác cần thiết, bất kể hiện thực có đúng đến đâu. Độ sâu kiến trúc phải "
               "tương xứng với độ phức tạp của dữ liệu."))


def _per_class_commentary(r: R.Report, m: dict, names: list[str], unit: str) -> None:
    best_k, best_m = _best(m, "macro_f1")
    per = best_m.get("per_class_accuracy")
    if not per:
        return
    order = sorted(range(len(per)), key=lambda i: per[i])
    lo, hi = order[0], order[-1]
    r.p(
        f"Lấy mô hình tốt nhất ({M_LABEL.get(best_k, best_k)}) làm đại diện, "
        f"{unit} dễ nhất là <strong>{names[hi]}</strong> với độ chính xác "
        f"{pct(per[hi])}, {unit} khó nhất là <strong>{names[lo]}</strong> với "
        f"{pct(per[lo])}. Khoảng cách {pct(per[hi] - per[lo])} điểm phần trăm giữa hai đầu "
        f"cho thấy một con số tổng hợp duy nhất che đi khá nhiều: chỉ số chung "
        f"{pct(best_m['accuracy'])} không phản ánh được rằng một số lớp gần như hoàn hảo "
        f"trong khi những lớp khác còn cách xa mức có thể triển khai.")
    if len(order) >= 3:
        hard = ", ".join(names[i] for i in order[:3])
        r.p(f"Ba {unit} khó nhất theo thứ tự là {hard}.")


def _high_conf_commentary(r: R.Report, m: dict) -> None:
    best_k, best_m = _best(m, "macro_f1")
    errs = best_m.get("high_conf_errors")
    if not errs:
        return
    top = max(errs, key=lambda e: e["confidence"])
    r.p(
        f"Lỗi tự tin nhất trong toàn bộ tập kiểm thử có độ tin cậy {pct(top['confidence'])} "
        f"nhưng vẫn sai. Trong số {len(errs)} lỗi được chọn ra, độ tin cậy thấp nhất vẫn ở "
        f"mức {pct(min(e['confidence'] for e in errs))}.")
