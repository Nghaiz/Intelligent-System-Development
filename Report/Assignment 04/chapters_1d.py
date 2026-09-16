"""Chương 2 (thiết kế thực nghiệm) và các Chương 3–5 về mạng tích chập một chiều."""

from __future__ import annotations

import report_lib as R

FW_LABEL = {
    "numpy": "NumPy thuần",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow / Keras",
}
FW_ORDER = ["numpy", "pytorch", "tensorflow"]

DEVICE_LABEL = {"cuda": "GPU", "cpu": "CPU"}


def dev(m: dict) -> str:
    """Nhan thiet bi cho cot thoi gian. Chua ghi thi tra dau gach."""
    return DEVICE_LABEL.get(m.get("device"), "—")


# ---------------------------------------------------------------------------
# Tiện ích định dạng — mọi con số đều đi qua đây, không gõ tay vào văn bản
# ---------------------------------------------------------------------------
def pct(x, nd: int = 2) -> str:
    return f"{100 * float(x):.{nd}f}%"


def num(x, nd: int = 4) -> str:
    return f"{float(x):.{nd}f}"


def usd(x) -> str:
    return f"${float(x):,.0f}"


def thousands(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def best_of(models: dict, key: str) -> tuple[str, dict]:
    """Trả về (tên khung, dict chỉ số) của mô hình tốt nhất theo một chỉ số."""
    name = max(models, key=lambda k: models[k].get(key, float("-inf")))
    return name, models[name]


def worst_of(models: dict, key: str) -> tuple[str, dict]:
    name = min(models, key=lambda k: models[k].get(key, float("inf")))
    return name, models[name]


def spread(models: dict, key: str) -> float:
    vals = [m[key] for m in models.values() if key in m]
    return max(vals) - min(vals)


def write(r: R.Report, data: dict) -> None:
    _chapter2(r, data)
    _chapter3(r, data)
    _chapter4(r, data)
    _chapter5(r, data)


# ===========================================================================
# CHƯƠNG 2 — THIẾT KẾ THỰC NGHIỆM
# ===========================================================================
def _chapter2(r: R.Report, data: dict) -> None:
    r.h(1, "Chương 2. Thiết kế thực nghiệm và giao thức đối chuẩn")

    r.p(
        "Bài tập này không đặt mục tiêu săn một con số độ chính xác cao. Mục tiêu là trả "
        "lời một câu hỏi có cấu trúc: <em>khi cùng một ý tưởng toán học được hiện thực ở "
        "ba mức độ kiểm soát khác nhau, trên năm miền dữ liệu có hình học khác nhau, thì "
        "cái gì bất biến và cái gì thay đổi?</em> Một câu hỏi như vậy chỉ có nghĩa nếu mọi "
        "yếu tố ngoài biến khảo sát được giữ cố định. Chương này nêu rõ những gì đã được "
        "giữ cố định, và thành thật về những gì không giữ được.")

    # ---- 2.1
    r.h(2, "2.1. Ba mức độ kiểm soát khi xây dựng cùng một kiến trúc")
    r.p(
        '<ol class="tight">'
        '<li><strong>NumPy thuần từ đầu.</strong> Tự viết tích chập, gộp, Dense, softmax '
        'hoặc sigmoid, lan truyền ngược và Adam. Không autograd, không một dòng gọi thư '
        'viện học sâu nào. Đây là mức kiểm soát cao nhất và cũng là mức chậm nhất.</li>'
        '<li><strong>PyTorch.</strong> Định nghĩa module, optimizer và vòng lặp huấn luyện '
        'tường minh. Autograd lo phần đạo hàm; phần còn lại vẫn nằm trong tay người viết.</li>'
        '<li><strong>TensorFlow / Keras.</strong> Mô tả kiến trúc bằng API cấp cao, giao '
        'phần lớn vòng lặp cho <code class="inl">fit()</code>. Mức kiểm soát thấp nhất và '
        'mã nguồn ngắn nhất.</li>'
        '</ol>')
    r.p(
        "Ba mức này không cạnh tranh nhau. Chúng trả lời ba câu hỏi khác nhau: mức một trả "
        "lời \"thuật toán hoạt động ra sao\", mức hai trả lời \"làm sao để linh hoạt và gỡ "
        "lỗi được\", mức ba trả lời \"làm sao để viết nhanh và chuẩn hoá\".")

    # ---- 2.2
    r.h(2, "2.2. Năm miền dữ liệu và ánh xạ sang toán tử tích chập")
    r.p(
        "Năm bộ dữ liệu được chọn để phủ ba dạng hình học dữ liệu khác nhau, chứ không "
        "phải để phủ nhiều lĩnh vực ứng dụng.")
    d = data
    r.p(R.table(
        ["Miền dữ liệu", "Dạng dữ liệu", "Bài toán", "Quy mô thô", "Diễn giải tích chập"],
        [["Customer Comments", "Chuỗi token", "Phân loại nhị phân",
          f'{thousands(d["comments"]["dataset"]["n_raw"])} đánh giá',
          "Cửa sổ trượt bắt cụm n-gram"],
         ["Diabetes", "Bảng số", "Phân loại nhị phân",
          f'{thousands(d["diabetes"]["dataset"]["n_raw"])} hồ sơ',
          "Chuỗi 1D nhân tạo dài 8"],
         ["House Price", "Bảng số", "Hồi quy liên tục",
          f'{thousands(d["house"]["dataset"]["n_raw"])} giao dịch',
          "Chuỗi 1D nhân tạo dài 8"],
         ["MNIST", "Ảnh xám 28×28×1", "Phân loại 10 lớp",
          f'{thousands(d["mnist"]["dataset"]["n_raw"])} ảnh',
          "Cửa sổ trượt 2D trên lưới điểm ảnh"],
         ["CIFAR-10", "Ảnh màu 32×32×3", "Phân loại 10 lớp",
          f'{thousands(d["cifar"]["dataset"]["n_raw"])} ảnh',
          "Cửa sổ trượt 2D trên ba kênh màu"]],
        "Năm miền dữ liệu của bài tập số 4 và cách mỗi miền được ánh xạ sang toán tử tích chập."))

    r.p(R.note(
        "Cảnh báo phương pháp luận, nêu trước khi trình bày bất cứ kết quả nào.",
        "Dữ liệu bảng <em>không</em> có tô-pô không gian. Việc cột "
        "<code class='inl'>Glucose</code> nằm cạnh cột <code class='inl'>BloodPressure</code> "
        "trong tệp CSV là một quy ước trình bày, không phải một quan hệ lân cận. Trượt bộ "
        "lọc dọc theo tám cột vì vậy là một <strong>nghiên cứu đối sánh có kiểm soát</strong> "
        "về năng lực biểu diễn của khung tích chập, chứ không phải một khẳng định rằng CNN "
        "là kiến trúc tối ưu cho dữ liệu bảng. Chương 4 kiểm chứng luận điểm này bằng một "
        "thực nghiệm hoán vị cột, thay vì chỉ tuyên bố nó.",
        "warn"))

    # ---- 2.3
    r.h(2, "2.3. Tách dữ liệu và phòng rò rỉ")
    r.p(
        "Mọi miền dùng chung một giao thức. Tập kiểm thử được tách ra trước tiên và "
        "<strong>không tham gia</strong> vào bất kỳ quyết định nào: không chọn epoch, không "
        "chọn tốc độ học, không chọn ngưỡng phân loại. Phần còn lại được tách tiếp thành "
        "train và validation theo phân tầng với <code class='inl'>random_state = 42</code>.")
    r.p(
        "Tham số chuẩn hoá là điểm rò rỉ dễ mắc nhất và thường bị bỏ qua. Trung bình và độ "
        "lệch chuẩn chỉ được học từ tập train, sau đó cùng bộ tham số đó được áp cho "
        "validation và test:")
    r.p('<div class="formula">X<sub>chuẩn hoá</sub> = (X/255 − μ<sub>train</sub>) / (σ<sub>train</sub> + ε)</div>')
    r.p(
        "Tương tự với văn bản, từ điển được xây dựng từ tập huấn luyện; từ nào chỉ xuất "
        "hiện ở tập kiểm thử được ánh xạ về token <code class='inl'>&lt;UNK&gt;</code>. "
        "Nếu xây từ điển trên toàn bộ dữ liệu, mô hình đã biết trước sự tồn tại của những "
        "từ mà lẽ ra nó chưa từng gặp.")
    r.p(R.table(
        ["Miền dữ liệu", "Train", "Validation", "Test", "Số đặc trưng đầu vào"],
        [[name,
          thousands(d[k]["dataset"]["n_train"]),
          thousands(d[k]["dataset"]["n_val"]),
          thousands(d[k]["dataset"]["n_test"]),
          str(d[k]["dataset"]["n_features"])]
         for k, name in [("comments", "Customer Comments"), ("diabetes", "Diabetes"),
                         ("house", "House Price"), ("mnist", "MNIST"),
                         ("cifar", "CIFAR-10")]],
        "Kích thước thực tế của các tập dữ liệu sau khi tách, đọc từ tệp JSON của notebook."))

    # ---- 2.4
    r.h(2, "2.4. Cấu hình kiến trúc giữ cố định giữa ba khung")
    r.p(
        "Điều kiện tiên quyết để so sánh có nghĩa là ba khung phải dựng <em>cùng một</em> "
        "kiến trúc. Hai sơ đồ dưới đây được áp dụng nguyên văn cho cả NumPy, PyTorch và "
        "TensorFlow.")
    r.p('<div class="flow">Bảng số: 8 → Conv1D(16, K=3) → ReLU → Conv1D(16, K=3) → ReLU '
        '→ MaxPool1D(2) → Flatten → Dense(8) → ReLU → Dense(1)</div>')
    r.p('<div class="flow">Văn bản: tokens(50) → Embedding(5000, 100) → Conv1D(32, K=3) '
        '→ ReLU → GlobalMaxPool1D → Dense(1, Sigmoid)</div>')
    r.p(
        "Với ảnh, mô hình NumPy giữ ở mức hai khối tích chập để lan truyền ngược viết tay "
        "còn chạy được trong vài phút, trong khi PyTorch và Keras dùng kiến trúc sâu hơn "
        "có Batch Normalization và Dropout. Đây là một sai lệch có chủ đích và được báo cáo "
        "công khai ở mục 9.3, chứ không bị che đi bằng cách chỉ trích dẫn con số cuối cùng.")

    # ---- 2.5
    r.h(2, "2.5. Vòng lặp huấn luyện viết tay")
    r.p(
        "Đoạn mã dưới tóm tắt khung vòng lặp dùng chung cho mọi mô hình NumPy trong báo "
        "cáo. Khác biệt giữa các miền chỉ nằm ở số bộ lọc, số nơ-ron ẩn, số epoch và hàm "
        "mất mát.")
    r.p(R.code('''for epoch in range(epochs):
    lr = lr_schedule(initial_lr, epoch, epochs)       # lịch giảm tốc độ học
    order = rng.permutation(len(X_train))             # xáo trộn lại mỗi epoch

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        logits = model.forward(X_train[idx])          # lượt tiến
        loss   = criterion.forward(logits, y_train[idx])
        grad   = criterion.backward()                 # dL/dz
        model.backward(grad)                          # lan truyền ngược viết tay
        optimizer.step(model.parameters(), lr)        # Adam viết tay

    val_loss, val_acc = evaluate(model, X_val, y_val)
    if val_loss < best_loss:                          # chỉ dựa trên validation
        best_loss, best_state = val_loss, model.state_dict()

model.load_state_dict(best_state)                     # khôi phục trước khi chạm test''',
        "Khung vòng lặp huấn luyện dùng chung cho mọi hiện thực NumPy trong báo cáo."))
    r.p(R.note(
        "Một ghi chú về cột thời gian trong mọi bảng đối chuẩn.",
        "Ba cách cài đặt không chạy trên cùng một thiết bị. Các mô hình tích chập hai chiều "
        "trên ảnh dùng GPU, còn các mô hình một chiều trên bảng và văn bản dùng CPU, vì ở quy "
        "mô dưới mười nghìn tham số thì chi phí khởi chạy kernel và truyền dữ liệu lớn hơn chính "
        "phần tính toán. Ngoài ra TensorFlow từ bản 2.11 không còn hỗ trợ GPU native trên Windows, "
        "và các mô hình NumPy thuần chạy CPU là bản chất của bài tập. Vì vậy <strong>cột thời gian "
        "chỉ cho biết chi phí tuyệt đối của từng cấu hình, không dùng để so sánh khung thư viện "
        "với nhau</strong>. Mỗi bảng đều ghi kèm thiết bị ở cột bên cạnh. Các chỉ số chất lượng "
        "không phụ thuộc thiết bị nên vẫn so sánh được bình thường."))
    r.p(
        "Hai chi tiết đáng chú ý. Thứ nhất, optimizer chỉ nhìn thấy train loss; validation "
        "chỉ dùng để theo dõi và chọn checkpoint. Thứ hai, trạng thái tốt nhất được khôi "
        "phục <em>trước khi</em> chạm vào tập kiểm thử, nên vài epoch cuối dao động không "
        "làm hỏng con số báo cáo.")

    # ---- 2.6
    r.h(2, "2.6. Nguyên tắc đọc kết quả")
    r.p(
        "Các bảng đối chuẩn trong báo cáo được sắp theo chỉ số chính của từng bài toán: "
        "F1 cho phân loại nhị phân mất cân bằng, Macro-F1 cho phân loại mười lớp, R² cho "
        "hồi quy. Nhưng một bảng xếp hạng không phải là kết luận. Ba nguyên tắc đọc:")
    r.p(
        '<ol class="tight">'
        '<li>Một khoảng cách nhỏ hơn một điểm phần trăm giữa hai khung, với cùng kiến trúc '
        'và cùng dữ liệu, nên được đọc là <em>tương đương</em>, không phải là thắng thua. '
        'Nó đến từ khởi tạo ngẫu nhiên và thứ tự mini-batch nhiều hơn là từ chất lượng khung.</li>'
        '<li>Thời gian huấn luyện là số đo của <em>một lần chạy trên một máy cụ thể</em>. '
        'Toàn bộ thực nghiệm chạy trên CPU không có GPU, nên các con số thời gian nói về '
        'chi phí tương đối giữa ba khung, không phải về hiệu năng tuyệt đối của khung.</li>'
        '<li>Chỉ một hạt giống ngẫu nhiên duy nhất được báo cáo. Báo cáo không đưa ra trung '
        'bình và độ lệch chuẩn qua nhiều lần chạy, nên mọi khoảng cách nhỏ đều mang tính '
        'gợi ý chứ chưa phải bằng chứng thống kê.</li>'
        '</ol>')


# ===========================================================================
# CHƯƠNG 3 — CUSTOMER COMMENTS (1D CNN trên chuỗi văn bản)
# ===========================================================================
def _chapter3(r: R.Report, data: dict) -> None:
    d = data["comments"]
    m = d["models"]
    ds = d["dataset"]

    r.h(1, "Chương 3. Tích chập một chiều trên chuỗi văn bản")

    r.p(
        "Trong năm miền dữ liệu của báo cáo, văn bản là miền duy nhất mà tích chập một "
        "chiều có biện minh lý thuyết trọn vẹn. Thứ tự các từ trong câu không phải quy ước: "
        "đảo hai từ cạnh nhau có thể lật ngược nghĩa của cả câu. Một bộ lọc rộng ba token "
        "vì vậy đang học đúng thứ nó nên học, những cụm như \"rất đáng tiền\" hay \"không "
        "vừa vặn\".")

    # ---- 3.1
    r.h(2, "3.1. Dữ liệu và quy trình tiền xử lý chuỗi")
    r.p(
        f"Tập dữ liệu gồm {thousands(ds['n_raw'])} đánh giá sản phẩm thời trang trên một "
        f"sàn thương mại điện tử. Sau khi loại các bản ghi thiếu nội dung hoặc thiếu nhãn, "
        f"còn {thousands(ds['n_clean'])} mẫu hợp lệ. Nhãn mục tiêu là "
        f"<code class='inl'>Recommended IND</code>, nhận giá trị 1 khi khách hàng khuyến "
        f"nghị sản phẩm.")
    r.p(
        "Quy trình xử lý chuỗi gồm bốn bước, mỗi bước đều có lý do chứ không phải thói quen:")
    r.p(
        '<ol class="tight">'
        '<li><strong>Chuẩn hoá về chữ thường và tách token bằng biểu thức chính quy</strong> '
        '<code class="inl">[a-zA-Z]+</code>. Thao tác này gộp Good và good thành một '
        'token, giảm kích thước từ điển mà không mất thông tin cảm xúc.</li>'
        '<li><strong>Xây từ điển 5.000 từ phổ biến nhất</strong>, với chỉ số 0 dành cho '
        '<code class="inl">&lt;PAD&gt;</code> và 1 cho <code class="inl">&lt;UNK&gt;</code>. '
        'Ngưỡng 5.000 là điểm cân bằng: đủ rộng để phủ từ vựng cảm xúc thông dụng, đủ hẹp '
        'để bảng nhúng không chiếm phần lớn tham số của mô hình.</li>'
        '<li><strong>Cắt hoặc đệm về đúng 50 token.</strong> Chuỗi cố định cho phép xử lý '
        'theo lô; token đệm mang chỉ số 0 và vector nhúng của nó được học như mọi token khác.</li>'
        '<li><strong>Tầng nhúng 100 chiều</strong> ánh xạ mỗi chỉ số nguyên thành một vector '
        'liên tục. Đây là tầng đầu tiên trong mạng và cũng là tầng nhiều tham số nhất.</li>'
        '</ol>')
    r.p(R.figure("cm_fig_comments_eda.png",
                 "Khảo sát tập đánh giá khách hàng: phân phối nhãn khuyến nghị và phân bố "
                 "độ dài chuỗi sau khi tách token."))

    # ---- 3.2
    r.h(2, "3.2. Kiến trúc và vai trò của gộp cực đại toàn cục")
    r.p('<div class="flow">tokens(50) → Embedding(5000×100) → Conv1D(32, K=3) → ReLU '
        '→ GlobalMaxPool1D → Dense(1) → Sigmoid</div>')
    r.p(
        "Tầng đáng bàn nhất ở đây là gộp cực đại toàn cục. Sau tích chập, ta có 32 bản đồ "
        "đặc trưng, mỗi bản đồ dài 48. Gộp cực đại toàn cục lấy giá trị lớn nhất trên toàn "
        "chiều dài của từng bản đồ, cho ra đúng 32 số.")
    r.p(
        "Ý nghĩa của phép rút gọn này rất cụ thể: mỗi bộ lọc trả lời câu hỏi <em>\"hoa văn "
        "mà tôi phụ trách có xuất hiện ở đâu đó trong câu này không, và mạnh đến mức nào?\"</em> "
        "Vị trí xuất hiện bị bỏ đi một cách có chủ ý. Với phân loại cảm xúc, biết câu có "
        "chứa cụm phủ định mạnh là đủ; biết cụm đó nằm ở token thứ 7 hay thứ 31 thì không "
        "thêm thông tin. Đây cũng là lý do kiến trúc này xử lý được câu dài ngắn khác nhau "
        "mà không cần thay đổi gì.")

    # ---- 3.3
    r.h(2, "3.3. Hiện thực tích chập một chiều bằng NumPy thuần")
    r.p(
        "Hiện thực ngây thơ sẽ dùng một vòng lặp Python chạy qua 48 vị trí cửa sổ. Cách làm "
        "đó đúng nhưng chậm hơn hai bậc độ lớn so với cần thiết. Báo cáo dùng "
        "<code class='inl'>sliding_window_view</code> để tạo <em>view</em> của mọi cửa sổ "
        "cùng lúc mà không sao chép bộ nhớ, rồi gom toàn bộ phép nhân cộng vào một lời gọi "
        "<code class='inl'>einsum</code> duy nhất.")
    r.p(R.code('''def forward(self, X):
    self.X = X
    X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding,) * 2)) if self.padding else X
    # n: mẫu trong lô · c: kênh vào · l: vị trí cửa sổ · k: chỉ số trong kernel
    self.windows = np.lib.stride_tricks.sliding_window_view(
        X_pad, window_shape=self.kernel_size, axis=-1)[:, :, ::self.stride, :]
    return np.einsum('nclk,fck->nfl', self.windows, self.W) + self.b[:, None]

def backward(self, dZ):
    self.db = dZ.sum(axis=(0, 2))
    self.dW = np.einsum('nfl,nclk->fck', dZ, self.windows)   # cộng dồn mọi vị trí
    dX, L_out = np.zeros_like(self.X), dZ.shape[2]
    for k in range(self.kernel_size):                        # rải ngược về đầu vào
        dX[:, :, k:k + L_out] += np.einsum('nfl,fc->ncl', dZ, self.W[:, :, k])
    return dX''',
        "Tầng Conv1D thuần NumPy, vector hoá bằng sliding_window_view và einsum."))
    r.p(
        "Dòng tính <code class='inl'>self.dW</code> là nơi nguyên lý chia sẻ trọng số hiện "
        "ra rõ nhất. Phép <code class='inl'>einsum</code> cộng dồn trên cả chỉ số lô "
        "<code class='inl'>n</code> lẫn chỉ số vị trí <code class='inl'>l</code>, vì một "
        "trọng số duy nhất đã tham gia vào mọi vị trí cửa sổ của mọi mẫu trong lô. Bỏ sót "
        "phép cộng dồn này là lỗi kinh điển khi tự viết tích chập, và nó không gây lỗi "
        "chạy mà chỉ khiến mô hình học chậm một cách khó hiểu.")

    # ---- 3.4
    r.h(2, "3.4. Diễn biến huấn luyện")
    r.p(R.figure("cm_fig_comments_loss_curves.png",
                 "Hàm mất mát Binary Cross-Entropy theo epoch của ba cách cài đặt trên tập "
                 "đánh giá khách hàng."))
    _fw_loss_commentary(r, m, "phân loại cảm xúc")

    # ---- 3.5
    r.h(2, "3.5. Kết quả đối chuẩn ba khung")
    r.p(R.table(
        ["Khung hiện thực", "Accuracy", "Precision", "Recall", "F1", "Tham số",
         "Thiết bị", "Thời gian (s)"],
        [[FW_LABEL[k], pct(m[k]["accuracy"]), pct(m[k]["precision"]),
          pct(m[k]["recall"]), pct(m[k]["f1"]),
          f'{m[k]["params"]:,}', dev(m[k]), num(m[k]["train_time_s"], 1)] for k in FW_ORDER],
        "Đối chuẩn 1D CNN trên tập đánh giá khách hàng, đo trên cùng một tập kiểm thử."))
    _benchmark_commentary(r, m, "f1", "F1")
    r.p(R.figure("cm_fig_comments_benchmark.png",
                 "So sánh Accuracy, Precision, Recall và F1 giữa ba cách cài đặt trên tập "
                 "đánh giá khách hàng."))

    # ---- 3.6
    r.h(2, "3.6. Ma trận nhầm lẫn và cấu trúc lỗi")
    r.p(R.figure("cm_fig_comments_confusion.png",
                 "Ma trận nhầm lẫn trên tập kiểm thử của ba cách cài đặt."))
    _binary_cm_commentary(r, m, pos_label="đánh giá tích cực", neg_label="đánh giá tiêu cực")
    r.p(
        "Cấu trúc lỗi ở đây phản ánh đúng đặc thù của dữ liệu đánh giá sản phẩm: phần lớn "
        "khách hàng để lại nhận xét tích cực, nên lớp tiêu cực vừa ít mẫu hơn vừa đa dạng "
        "hơn về cách diễn đạt. Những câu mang sắc thái trung tính hoặc chứa lời khen kèm "
        "một điểm chê là nhóm khó nhất, vì một bộ lọc rộng ba token khó nắm được quan hệ "
        "nhượng bộ trải dài cả câu.")


# ===========================================================================
# CHƯƠNG 4 — DIABETES (1D CNN trên bảng số, phân loại)
# ===========================================================================
def _chapter4(r: R.Report, data: dict) -> None:
    d = data["diabetes"]
    m = d["models"]
    ds = d["dataset"]

    r.h(1, "Chương 4. Tích chập một chiều trên dữ liệu bảng y tế")

    r.p(
        "Chương này là một thực nghiệm đối sánh có chủ đích. Slide bài giảng nêu thẳng rằng "
        "áp dụng tích chập một chiều lên dữ liệu bảng là <em>một thí nghiệm sư phạm</em>, "
        "không phải một khẳng định về kiến trúc tối ưu. Báo cáo không chỉ nhắc lại cảnh "
        "báo đó mà tìm cách đo nó.")

    # ---- 4.1
    r.h(2, "4.1. Dữ liệu và tám đặc trưng lâm sàng")
    r.p(
        f"Tập dữ liệu gồm {thousands(ds['n_raw'])} hồ sơ bệnh nhân. Sau khi khử trùng lặp "
        f"và loại nhóm giới tính không xác định, còn {thousands(ds['n_clean'])} bản ghi. "
        f"Tám đặc trưng được giữ đúng theo thứ tự nêu trong slide bài giảng, tương ứng với "
        f"chuỗi đầu vào dài 8 mà tầng tích chập sẽ trượt qua.")
    r.p(R.figure("db_fig_diabetes_eda.png",
                 "Khảo sát tập dữ liệu tiểu đường: phân phối nhãn và tương quan giữa tám "
                 "đặc trưng lâm sàng."))
    r.p(
        "Điểm cần chú ý ngay từ khâu khảo sát là mất cân bằng lớp. Tỉ lệ bệnh nhân dương "
        "tính chỉ ở mức một chữ số phần trăm, nghĩa là một mô hình luôn dự đoán \"không mắc "
        "bệnh\" đã đạt độ chính xác rất cao mà hoàn toàn vô dụng về mặt lâm sàng. Vì vậy "
        "mọi bảng kết quả trong chương này đặt recall ngang hàng với accuracy.")

    # ---- 4.2
    r.h(2, "4.2. Kiến trúc và luồng biến đổi kích thước tensor")
    r.p('<div class="flow">8 → Conv1D(16, K=3, same) → ReLU → Conv1D(16, K=3, same) → ReLU '
        '→ MaxPool1D(2) → Flatten(64) → Dense(8) → ReLU → Dense(1) → Sigmoid</div>')
    r.p(
        "Tầng tích chập thứ nhất học các tổ hợp cục bộ giữa ba chỉ số lâm sàng liền kề. "
        "Tầng thứ hai học tổ hợp của chính những hoa văn đó, nên trường tiếp nhận mở rộng "
        "lên 5 trong 8 đặc trưng. Câu hỏi là: <em>những tổ hợp \"liền kề\" ấy có mang ý "
        "nghĩa gì không, khi thứ tự cột vốn tuỳ ý?</em>")

    # ---- 4.3
    r.h(2, "4.3. Diễn biến huấn luyện")
    r.p(R.figure("db_fig_diabetes_loss_curves.png",
                 "Hàm mất mát huấn luyện và kiểm định theo epoch của ba cách cài đặt trên "
                 "bài toán chẩn đoán tiểu đường."))
    _fw_loss_commentary(r, m, "chẩn đoán tiểu đường")

    # ---- 4.4
    r.h(2, "4.4. Kết quả đối chuẩn ba khung")
    r.p(R.table(
        ["Khung hiện thực", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
         "Thiết bị", "Thời gian (s)"],
        [[FW_LABEL[k], pct(m[k]["accuracy"]), pct(m[k]["precision"]),
          pct(m[k]["recall"]), pct(m[k]["f1"]), num(m[k].get("roc_auc", 0)),
          dev(m[k]), num(m[k]["train_time_s"], 1)] for k in FW_ORDER],
        "Hiệu năng 1D CNN trên bài toán chẩn đoán tiểu đường, chỉ số tính trên lớp dương."))
    _benchmark_commentary(r, m, "f1", "F1")
    r.p(R.figure("db_fig_diabetes_benchmark.png",
                 "So sánh bốn chỉ số phân loại giữa ba cách cài đặt trên bài toán tiểu đường."))

    # ---- 4.5
    r.h(2, "4.5. Ma trận nhầm lẫn và cái giá của âm tính giả")
    r.p(R.figure("db_fig_diabetes_confusion.png",
                 "Ma trận nhầm lẫn trên tập kiểm thử chẩn đoán tiểu đường."))
    _binary_cm_commentary(r, m, pos_label="bệnh nhân dương tính", neg_label="bệnh nhân âm tính")
    r.p(
        "Trong bối cảnh sàng lọc y tế, hai loại lỗi không có cùng giá trị. Một dương tính "
        "giả dẫn tới một lần xét nghiệm bổ sung; một âm tính giả là một ca bệnh bị bỏ lọt. "
        "Nếu triển khai thật, ngưỡng phân loại nên được hạ xuống dưới 0,5 để đổi precision "
        "lấy recall, và điểm làm việc phải do bác sĩ chọn trên đường cong ROC chứ không "
        "phải do mặc định của thư viện.")

    # ---- 4.6
    r.h(2, "4.6. Thực nghiệm hoán vị cột: kiểm chứng cảnh báo phương pháp luận")
    r.p(
        "Luận điểm \"dữ liệu bảng không có tô-pô\" dễ phát biểu nhưng cần được đo. Thực "
        "nghiệm kiểm chứng rất đơn giản: <strong>hoán vị ngẫu nhiên thứ tự tám cột đặc "
        "trưng rồi huấn luyện lại từ đầu với đúng mọi siêu tham số cũ</strong>.")
    r.p(
        "Suy luận là thế này. Nếu tích chập thật sự khai thác quan hệ lân cận giữa các cột, "
        "hoán vị phải phá vỡ quan hệ đó và làm hiệu năng sụt rõ rệt, giống hệt như xáo trộn "
        "thứ tự từ trong một câu sẽ phá huỷ mô hình phân loại văn bản. Nếu hiệu năng gần "
        "như không đổi, kết luận là toán tử tích chập ở đây chỉ đang hoạt động như một mạng "
        "truyền thẳng có ràng buộc chia sẻ trọng số, chứ không khai thác cấu trúc nào cả.")
    _permutation_commentary(r, d)

    r.p(R.note(
        "Kết luận phương pháp luận của Chương 4.",
        "Con số accuracy cao ở chương này <em>không</em> chứng minh tích chập phù hợp với "
        "dữ liệu bảng. Nó chứng minh rằng bộ dữ liệu này đủ tách được bằng một mô hình phi "
        "tuyến bất kỳ có dung lượng vừa phải. Đóng góp thực sự của chương là phép đo hoán "
        "vị cột, vì nó biến một cảnh báo lý thuyết thành một con số kiểm chứng được."))


# ===========================================================================
# CHƯƠNG 5 — HOUSE PRICE (1D CNN hồi quy)
# ===========================================================================
def _chapter5(r: R.Report, data: dict) -> None:
    d = data["house"]
    m = d["models"]
    ds = d["dataset"]

    r.h(1, "Chương 5. Tích chập một chiều cho bài toán hồi quy")

    r.p(
        "Ba chương trước đều là phân loại. Chương này đổi bài toán chứ không đổi kiến trúc, "
        "để làm nổi bật một nguyên tắc đã nêu ở mục 1.7: <strong>tầng cuối phải theo bài "
        "toán</strong>. Toàn bộ phần trích xuất đặc trưng giữ nguyên; chỉ nơ-ron đầu ra "
        "chuyển từ sigmoid sang tuyến tính và hàm mất mát chuyển từ Binary Cross-Entropy "
        "sang sai số toàn phương trung bình.")
    r.p('<div class="formula">L<sub>MSE</sub> = (1/N) Σ<sub>i</sub> (y<sub>i</sub> − ŷ<sub>i</sub>)²</div>')
    r.p(
        "Điều gì xảy ra nếu dùng nhầm? Sigmoid ép đầu ra vào khoảng (0, 1), nên mô hình "
        "không thể biểu diễn một mức giá vượt quá 1 dù có huấn luyện bao lâu. Gradient "
        "cũng bão hoà về 0 ở hai đầu, khiến việc học dừng hẳn. Đây không phải lỗi tinh "
        "vi mà là lỗi làm hỏng toàn bộ mô hình ngay từ epoch đầu tiên.")

    # ---- 5.1
    r.h(2, "5.1. Dữ liệu, lọc ngoại lai và tám đặc trưng kỹ nghệ")
    r.p(
        f"Tập dữ liệu gồm {thousands(ds['n_raw'])} bản ghi giao dịch bất động sản. Sau khi "
        f"loại bản ghi thiếu và cắt ngoại lai (giá ngoài khoảng 10 nghìn đến 5 triệu đô-la, "
        f"diện tích ngoài khoảng 200 đến 20.000 foot vuông), còn "
        f"{thousands(ds['n_clean'])} mẫu. Việc cắt ngoại lai không phải để làm đẹp số liệu: "
        f"một biệt thự 50 triệu đô-la trong tập huấn luyện tạo ra gradient lớn tới mức nó "
        f"chi phối toàn bộ một mini-batch.")
    r.p(
        "Tám đặc trưng được tạo ra từ bốn cột gốc, trong đó có các đặc trưng tương tác như "
        "tích số phòng ngủ với số phòng tắm và diện tích trung bình trên mỗi phòng. Những "
        "đặc trưng này mang thông tin mà mô hình khó tự học từ dữ liệu thô với kích thước "
        "mạng hiện tại.")
    r.p(R.figure("hp_fig_house_eda.png",
                 "Phân phối giá bất động sản trước và sau phép biến đổi logarit."))

    # ---- 5.2
    r.h(2, "5.2. Vì sao huấn luyện trên thang logarit")
    r.p(
        "Phân phối giá nhà lệch phải rất mạnh: phần lớn giao dịch tập trung ở vùng giá "
        "thấp, một đuôi dài kéo lên vùng giá cao. Huấn luyện trực tiếp trên thang đô-la "
        "khiến hàm MSE bị chi phối bởi vài mẫu đắt tiền, vì sai số được bình phương. Phép "
        "biến đổi <span class='inl'>log_price = log(price)</span> đưa phân phối về gần "
        "chuẩn và, quan trọng hơn, biến sai số tuyệt đối thành sai số <em>tương đối</em>: "
        "nhầm 20 nghìn đô-la trên một căn 100 nghìn bị phạt nặng hơn nhầm 20 nghìn trên một "
        "căn 2 triệu, đúng như trực giác định giá.")
    r.p(
        "Cái giá phải trả nằm ở khâu quy đổi ngược. Vì hàm mũ lồi, bất đẳng thức Jensen cho "
        "biết <span class='inl'>E[exp(ŷ)] ≥ exp(E[ŷ])</span>, nghĩa là lấy "
        "<span class='inl'>exp</span> của giá trị dự đoán trung bình sẽ <em>thiên lệch "
        "thấp</em> so với kỳ vọng thật của giá. Báo cáo vì vậy trình bày cả RMSE trên thang "
        "logarit lẫn RMSE quy đổi về đô-la, để người đọc thấy được cả hai mặt.")

    # ---- 5.3
    r.h(2, "5.3. Diễn biến huấn luyện")
    r.p(R.figure("hp_fig_house_loss_curves.png",
                 "Hàm mất mát MSE theo epoch của ba cách cài đặt trên bài toán định giá "
                 "bất động sản."))
    _fw_loss_commentary(r, m, "định giá bất động sản", metric_key="r2", metric_name="R²")

    # ---- 5.4
    r.h(2, "5.4. Kết quả đối chuẩn ba khung")
    r.p(R.table(
        ["Khung hiện thực", "RMSE (USD)", "MAE (USD)", "RMSE (log)", "MAE (log)", "R²",
         "Thiết bị", "Thời gian (s)"],
        [[FW_LABEL[k], usd(m[k]["rmse_usd"]), usd(m[k]["mae_usd"]),
          num(m[k]["rmse_log"]), num(m[k]["mae_log"]), num(m[k]["r2"]),
          dev(m[k]), num(m[k]["train_time_s"], 1)] for k in FW_ORDER],
        "Các chỉ số hồi quy của mô hình 1D CNN trên bài toán định giá bất động sản."))

    best_k, best_m = best_of(m, "r2")
    worst_k, worst_m = worst_of(m, "r2")
    r.p(
        f"Khung {FW_LABEL[best_k]} đạt R² cao nhất với {num(best_m['r2'])}, nghĩa là mô "
        f"hình giải thích được {pct(best_m['r2'], 1)} phương sai của giá trên thang "
        f"logarit. Khoảng cách so với {FW_LABEL[worst_k]} là "
        f"{num(best_m['r2'] - worst_m['r2'])} đơn vị R². "
        + ("Khoảng cách này đủ nhỏ để quy về dao động khởi tạo chứ không phản ánh ưu thế "
           "kiến trúc." if best_m["r2"] - worst_m["r2"] < 0.02 else
           "Khoảng cách này đủ lớn để đáng truy nguyên, và nguyên nhân khả dĩ nhất là khác "
           "biệt ở lịch tốc độ học giữa các khung."))
    r.p(
        f"Về sai số tuyệt đối, mô hình tốt nhất cho MAE khoảng {usd(best_m['mae_usd'])}. "
        f"Con số này cần được đọc cùng với dải giá của thị trường: sai số vài chục nghìn "
        f"đô-la là chấp nhận được với một căn nhà giá trên nửa triệu, nhưng là sai số lớn "
        f"với một căn giá một trăm nghìn. Đây chính là điều mà một chỉ số duy nhất không "
        f"nói ra, và là lý do RMSE, MAE và R² luôn đi cùng nhau trong bảng trên.")
    r.p(R.figure("hp_fig_house_benchmark.png",
                 "So sánh RMSE, MAE và R² giữa ba cách cài đặt trên bài toán định giá."))

    # ---- 5.5
    r.h(2, "5.5. Đồ thị đối chiếu giá trị thực và giá trị dự đoán")
    r.p(R.figure("hp_fig_house_scatter.png",
                 "Giá trị thực tế đối chiếu giá trị dự đoán trên tập kiểm thử, kèm đường "
                 "chéo lý tưởng."))
    r.p(
        "Đường chéo là quỹ tích của dự đoán hoàn hảo. Đám mây điểm bám sát đường chéo ở "
        "vùng giá trung bình và loe rộng dần về hai đầu. Hiện tượng loe này có một cách "
        "giải thích thẳng thắn: vùng giá trung bình chiếm phần lớn dữ liệu huấn luyện nên "
        "mô hình được học kỹ nhất ở đó, còn vùng giá rất thấp và rất cao vừa thưa mẫu vừa "
        "chịu ảnh hưởng nặng của các yếu tố không có trong tám đặc trưng, đặc biệt là vị "
        "trí địa lý.")
    r.p(
        "Một mô hình chỉ biết diện tích, số phòng và diện tích đất thì về nguyên tắc không "
        "thể phân biệt hai căn giống hệt nhau ở hai khu vực có mặt bằng giá khác nhau. "
        "Phần phương sai không giải thích được vì vậy phần lớn là phương sai địa lý, chứ "
        "không phải hạn chế của toán tử tích chập.")


# ===========================================================================
# Các đoạn bình luận sinh tự động từ số liệu
# ===========================================================================
def _fw_loss_commentary(r: R.Report, m: dict, task: str,
                        metric_key: str = "accuracy", metric_name: str = "độ chính xác") -> None:
    parts = []
    for k in FW_ORDER:
        h = m[k].get("history", {})
        tl = h.get("train_loss") or []
        vl = h.get("val_loss") or []
        if tl and vl:
            parts.append(
                f"{FW_LABEL[k]} hạ mất mát huấn luyện từ {num(tl[0])} xuống {num(tl[-1])} "
                f"sau {len(tl)} epoch, mất mát kiểm định tương ứng đi từ {num(vl[0])} "
                f"xuống {num(vl[-1])}")
    if parts:
        r.p("Ba đường cong cho thấy cùng một hình dạng hội tụ. " + "; ".join(parts) + ".")

    gaps = []
    for k in FW_ORDER:
        h = m[k].get("history", {})
        tl, vl = h.get("train_loss") or [], h.get("val_loss") or []
        if tl and vl:
            gaps.append((k, vl[-1] - tl[-1]))
    if gaps:
        k_max, g_max = max(gaps, key=lambda t: abs(t[1]))
        if abs(g_max) < 0.05:
            r.p(
                f"Khoảng cách giữa hai đường train và validation ở cuối quá trình huấn "
                f"luyện lớn nhất là {num(abs(g_max))} (khung {FW_LABEL[k_max]}). Mức chênh "
                f"nhỏ như vậy cho biết mô hình chưa quá khớp; hạn chế nằm ở dung lượng "
                f"biểu diễn chứ không ở việc học thuộc dữ liệu huấn luyện.")
        else:
            r.p(
                f"Khoảng cách train và validation lớn nhất là {num(abs(g_max))} ở khung "
                f"{FW_LABEL[k_max]}. Mức chênh này cho thấy mô hình bắt đầu học thuộc dữ "
                f"liệu huấn luyện, và chính là lý do quy trình khôi phục checkpoint tốt "
                f"nhất theo validation ở mục 2.5 là cần thiết chứ không phải thủ tục hình thức.")


def _benchmark_commentary(r: R.Report, m: dict, key: str, label: str) -> None:
    best_k, best_m = best_of(m, key)
    worst_k, worst_m = worst_of(m, key)
    gap = best_m[key] - worst_m[key]
    r.p(
        f"Khung {FW_LABEL[best_k]} dẫn đầu theo {label} với {pct(best_m[key])}, "
        f"{FW_LABEL[worst_k]} xếp sau với {pct(worst_m[key])}. Khoảng cách "
        f"{pct(gap)} " +
        ("nằm trong biên độ dao động do khởi tạo ngẫu nhiên và thứ tự mini-batch, nên kết "
         "luận đúng ở đây là ba khung <em>tương đương</em> khi được cấp cùng kiến trúc và "
         "cùng dữ liệu. Đó chính là điều báo cáo muốn chứng minh: nền toán học quyết định "
         "kết quả, tầng hiện thực chỉ quyết định tốc độ và công sức viết mã."
         if gap < 0.02 else
         "đủ lớn để không quy về nhiễu. Nguyên nhân khả dĩ nhất nằm ở khác biệt mặc định "
         "giữa các khung: quy tắc khởi tạo trọng số, tham số beta của Adam và cách xử lý "
         "epsilon trong phép chuẩn hoá đều không giống nhau giữa ba thư viện."))

    t_fast, t_slow = worst_of(m, "train_time_s"), best_of(m, "train_time_s")
    r.p(
        f"Về thời gian chạy, {FW_LABEL[t_fast[0]]} mất "
        f"{num(t_fast[1]['train_time_s'], 1)} giây và {FW_LABEL[t_slow[0]]} mất "
        f"{num(t_slow[1]['train_time_s'], 1)} giây. Hai con số này <strong>không</strong> so "
        f"sánh được với nhau, vì chúng đo trên hai loại thiết bị khác nhau như ô cảnh báo bên "
        f"trên đã nêu. Điều duy nhất rút ra được là chi phí tuyệt đối của từng cấu hình trong "
        f"đúng thiết lập này, đủ để biết một lần chạy lại tốn bao lâu.")


def _binary_cm_commentary(r: R.Report, m: dict, pos_label: str, neg_label: str) -> None:
    best_k, _ = best_of(m, "f1")
    cm = m[best_k].get("confusion_matrix")
    if not cm or len(cm) != 2:
        return
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    tot = tn + fp + fn + tp
    r.p(
        f"Lấy mô hình tốt nhất ({FW_LABEL[best_k]}) làm đại diện: trên {tot:,} mẫu kiểm "
        f"thử, mô hình nhận đúng {tp:,} {pos_label} và {tn:,} {neg_label}, trong khi bỏ "
        f"lọt {fn:,} {pos_label} và báo nhầm {fp:,} {neg_label}. "
        + (f"Số âm tính giả ({fn:,}) lớn hơn số dương tính giả ({fp:,}), nghĩa là mô hình "
           f"đang nghiêng về phía thận trọng khi khẳng định lớp dương."
           if fn > fp else
           f"Số dương tính giả ({fp:,}) lớn hơn số âm tính giả ({fn:,}), nghĩa là mô hình "
           f"đang nghiêng về phía sẵn sàng gán nhãn dương."))


def _permutation_commentary(r: R.Report, d: dict) -> None:
    perm = d.get("permutation_experiment")
    if not perm:
        r.p(R.note(
            "Thiếu số liệu.",
            "Thực nghiệm hoán vị cột chưa có trong tệp JSON kết quả. Chạy lại notebook "
            "<code class='inl'>diabetes/notebooks/01_diabetes_1d_cnn_numpy.ipynb</code> "
            "để sinh khoá <code class='inl'>permutation_experiment</code>.", "warn"))
        return

    base, permuted = perm["baseline_f1"], perm["permuted_f1"]
    delta = permuted - base
    r.p(R.table(
        ["Cấu hình thứ tự cột", "Accuracy", "F1 lớp dương"],
        [["Thứ tự gốc theo slide bài giảng", pct(perm["baseline_accuracy"]), pct(base)],
         ["Thứ tự hoán vị ngẫu nhiên", pct(perm["permuted_accuracy"]), pct(permuted)]],
        "Ảnh hưởng của việc hoán vị ngẫu nhiên thứ tự tám cột đặc trưng lên hiệu năng 1D CNN."))
    if abs(delta) < 0.02:
        r.p(
            f"Chênh lệch F1 giữa hai cấu hình chỉ là {pct(abs(delta))}. Kết quả này xác "
            f"nhận cảnh báo ở mục 2.2 một cách định lượng: toán tử tích chập ở đây "
            f"<strong>không</strong> khai thác được quan hệ lân cận nào giữa các cột, vì "
            f"đơn giản là không có quan hệ nào để khai thác. Thứ mô hình thực sự học được "
            f"là một hàm phi tuyến trên tám biến, và ràng buộc chia sẻ trọng số chỉ đóng "
            f"vai trò một dạng chính quy hoá nhẹ.")
        r.p(
            "Phép so sánh làm rõ ý này: nếu lặp lại đúng thực nghiệm hoán vị trên dữ liệu "
            "văn bản ở Chương 3, tức xáo trộn thứ tự token trong mỗi câu, hiệu năng sẽ sụt "
            "rõ rệt, vì cụm \"không vừa ý\" và \"ý không vừa\" là hai chuỗi khác nhau về "
            "nghĩa. Sự bất đối xứng giữa hai miền chính là bằng chứng cho luận điểm về "
            "tô-pô dữ liệu.")
    else:
        r.p(
            f"Chênh lệch F1 giữa hai cấu hình là {pct(abs(delta))}, lớn hơn mức nhiễu dự "
            f"kiến. Kết quả này đáng được điều tra thêm trước khi kết luận, vì nó mâu "
            f"thuẫn với kỳ vọng lý thuyết rằng thứ tự cột không mang thông tin.")
