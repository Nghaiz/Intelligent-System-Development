"""Chương 1 (cơ sở lý thuyết) và Chương 2 (baseline + bóc tách kiến trúc)."""

from __future__ import annotations

import report_lib as R

GITHUB = "https://github.com/Nghaiz/Assignment-03-Intelligent-System"

DATASETS = [
    ("Chương 2 — Mô hình cơ sở & Bóc tách", "Pima Indians Diabetes Database (768 bản ghi)",
     "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"),
    ("Chương 3 — Hệ thống 1", "Diabetes Prediction Dataset (100,000 bản ghi)",
     "https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset"),
    ("Chương 4 — Hệ thống 2", "USA Real Estate Dataset (150,000 bản ghi)",
     "https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset"),
    ("Chương 5 — Hệ thống 3", "Women's Clothing E-Commerce Reviews (23,486 bản ghi)",
     "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews"),
]

NOTEBOOKS = [
    ("Notebook 01 — Mạng cơ sở viết tay (768 mẫu Pima)",
     "diabetes_baseline/notebook/01_diabetes_dl_from_scratch_numpy.ipynb"),
    ("Notebook 02 — Cải tiến mô hình & 6 nghiên cứu bóc tách",
     "diabetes_baseline/notebook/02_diabetes_dl_from_scratch_improvements.ipynb"),
    ("Notebook 03 — Hệ thống 1: Sàng lọc tiểu đường 100,000 mẫu",
     "diabetes_large/notebook/03_diabetes_large_ml_vs_dl_scratch.ipynb"),
    ("Notebook 04 — Hệ thống 2: Định giá bất động sản 150,000 mẫu",
     "house_price_large/notebook/04_house_price_large_ml_vs_dl_scratch.ipynb"),
    ("Notebook 05 — Hệ thống 3: Phân loại nhận xét NLP 23,486 mẫu",
     "customer_comments/notebook/05_comments_large_ml_vs_dl_scratch.ipynb"),
]


APIS = [
    ("Hệ thống 1 — Sàng lọc tiểu đường", "5001", "/diabetes/v1/predict",
     "diabetes_large/api/rest_api.py"),
    ("Hệ thống 2 — Định giá bất động sản", "5002", "/house-price/v1/predict",
     "house_price_large/api/rest_api.py"),
    ("Hệ thống 3 — Phân loại nhận xét", "5003", "/comments/v1/predict",
     "customer_comments/api/rest_api.py"),
]


def write(r: R.Report, data: dict) -> None:
    _front(r, data)
    _chapter1(r, data)
    _chapter2(r, data)


# ===========================================================================
# TRANG TÀI NGUYÊN — nằm ngay sau mục lục, trước Chương 1
# ===========================================================================
def _front(r: R.Report, data: dict) -> None:
    total = (data["baseline"]["dataset"]["n_samples"]
             + data["s1"]["dataset"]["n_raw"]
             + data["s2"]["dataset"]["n_raw"]
             + data["s3"]["dataset"]["n_raw"])

    r.front(
        '<h1 class="nobreak">Tài nguyên và mã nguồn dự án</h1>',
        '<p class="lead">Toàn bộ mã nguồn, dữ liệu, notebook thực nghiệm và báo cáo này '
        'được quản lý phiên bản công khai.</p>')

    r.front("<h3>Kho lưu trữ mã nguồn</h3>")
    r.front(R.plain_table(
        ["Hạng mục", "Đường dẫn"],
        [["Kho lưu trữ GitHub của dự án", f'<a href="{GITHUB}">{GITHUB}</a>'],
         ["Báo cáo nghiên cứu (tệp này)",
          '<code class="inl">report/A03_CT_nghiand.600.pdf</code>'],
         ["Hướng dẫn cài đặt và chạy lại", '<code class="inl">README.md</code>'],
         ["Danh sách phụ thuộc", '<code class="inl">requirements.txt</code>']],
        cls="links"))

    r.front("<h3>Bốn bộ dữ liệu thực nghiệm</h3>")
    r.front(R.plain_table(
        ["Sử dụng ở", "Bộ dữ liệu", "Nguồn Kaggle"],
        [[where, name, f'<a href="{url}">{url}</a>'] for where, name, url in DATASETS],
        cls="links"))
    r.front(
        f'<p style="text-align:center;font-size:11.5pt;color:#444">Tổng cộng '
        f'<strong>{total:,} bản ghi</strong> dữ liệu thực tế.</p>')

    r.front("<h3>Năm notebook thực nghiệm</h3>")
    r.front(R.plain_table(
        ["Notebook", "Đường dẫn trong kho mã nguồn"],
        [[label, f'<code class="inl">{path}</code>'] for label, path in NOTEBOOKS],
        cls="links"))

    r.front("<h3>Ba dịch vụ REST API kèm giao diện Web và Mobile</h3>")
    r.front(R.plain_table(
        ["Dịch vụ", "Cổng", "Endpoint dự đoán"],
        [[name, f'<code class="inl nb">{port}</code>',
          f'<code class="inl nb">POST {ep}</code>'] for name, port, ep, _ in APIS],
        cls="links api"))
    r.front(
        '<p style="font-size:11.5pt;color:#444">Mã nguồn ba dịch vụ nằm ở '
        '<code class="inl">&lt;tên_hệ_thống&gt;/api/rest_api.py</code>. Mỗi dịch vụ tự phục '
        'vụ luôn giao diện Web tại <code class="inl">/</code> và giao diện Mobile tại '
        '<code class="inl">/mobile</code>; ba cổng khác nhau nên chạy đồng thời được.</p>')


# ===========================================================================
# CHƯƠNG 1
# ===========================================================================
def _chapter1(r: R.Report, data: dict) -> None:
    r.h(1, "Chương 1. Cơ sở lý thuyết về mạng nơ-ron và học biểu diễn đặc trưng")

    r.p(
        "Báo cáo này trình bày một chuỗi thực nghiệm về mạng nơ-ron sâu, trong đó ràng "
        "buộc trung tâm là: <strong>toàn bộ mô hình học sâu được hiện thực hoá 100% bằng "
        "NumPy</strong>. Không TensorFlow, không PyTorch, không <code class='inl'>sklearn."
        "neural_network</code>. Mọi phép nhân ma trận, mọi đạo hàm riêng, mọi bước cập "
        "nhật trọng số đều được viết tay.",
        "Ràng buộc này không nhằm gây khó. Nó phục vụ một mục đích cụ thể: khi tự tay "
        "lập trình lan truyền tiến và lan truyền ngược, ta buộc phải hiểu tường minh bản "
        "chất dòng chảy tensor, cơ chế cập nhật gradient, và vai trò của từng hàm toán "
        "học. Điều này loại bỏ hoàn toàn tư duy xem mạng nơ-ron là một chiếc hộp đen bí ẩn.",
    )

    # ---- 1.1
    r.h(2, "1.1. Từ học máy truyền thống đến học biểu diễn sâu")
    r.p(
        "Trong quy trình học máy truyền thống, chu trình phát triển hệ thống thông minh "
        "thường tuân theo mô hình tuần tự:",
        '<div class="flow">X → [Kỹ nghệ đặc trưng thủ công] → X<sub>engineered</sub> → '
        '[Thuật toán học máy] → ŷ</div>',
        "Hiệu năng của các mô hình tuyến tính, cây quyết định hay Naive Bayes phụ thuộc "
        "rất lớn vào chất lượng của các thuộc tính do chuyên gia con người trích xuất. "
        "Cách tiếp cận này bộc lộ giới hạn nghiêm trọng khi dữ liệu có cấu trúc phi tuyến "
        "cao, tương tác phức tạp, hoặc số chiều lớn.",
        "Câu hỏi nghiên cứu trung tâm được đặt ra: <em>liệu mô hình có thể tự động học "
        "các không gian biểu diễn hữu ích trực tiếp từ dữ liệu hay không?</em>",
        '<div class="flow">X → [Biến đổi tham số hoá h<sub>θ</sub>(X)] → H → '
        '[Tầng dự đoán] → ŷ</div>',
        "Điểm cốt lõi là không gian ẩn <strong>H</strong> không do con người định nghĩa "
        "trước, mà các tham số θ = {W, b} của nó được tối ưu hoá <em>đồng thời</em> cùng "
        "hàm mục tiêu thông qua giải thuật lan truyền ngược.",
    )

    # ---- 1.2
    r.h(2, "1.2. Bản chất toán học: mạng nơ-ron là phép hợp hàm tham số hoá")
    r.p(
        "Một mạng nơ-ron đa tầng (Multi-Layer Perceptron) không phải chiếc hộp ma thuật. "
        "Về bản chất nó là một hàm toán học tham số hoá được xây dựng từ phép hợp của "
        "nhiều phép biến đổi liên tiếp:",
        '<div class="formula">ŷ = f<sub>θ</sub>(x) = (f<sub>L</sub> ∘ f<sub>L−1</sub> '
        '∘ ⋯ ∘ f<sub>2</sub> ∘ f<sub>1</sub>)(x)</div>',
        "Trong đó mỗi tầng nơ-ron ℓ ∈ {1, 2, …, L} thực hiện đúng hai thao tác tuần tự:",
    )
    r.p(
        "<ol>"
        "<li><strong>Biến đổi affine.</strong> Chiếu vector đầu vào từ không gian "
        "ℝ<sup>d<sub>ℓ−1</sub></sup> sang không gian ℝ<sup>d<sub>ℓ</sub></sup>:"
        '<div class="formula">Z<sup>[ℓ]</sup> = A<sup>[ℓ−1]</sup> W<sup>[ℓ]</sup> '
        '+ b<sup>[ℓ]</sup></div>'
        "với W<sup>[ℓ]</sup> ∈ ℝ<sup>d<sub>ℓ−1</sub>×d<sub>ℓ</sub></sup> là ma trận trọng "
        "số và b<sup>[ℓ]</sup> ∈ ℝ<sup>1×d<sub>ℓ</sub></sup> là vector độ chệch.</li>"
        "<li><strong>Kích hoạt phi tuyến.</strong> Áp dụng hàm φ(·) trên từng phần tử để "
        "sinh ra biểu diễn ẩn:"
        '<div class="formula">A<sup>[ℓ]</sup> = φ(Z<sup>[ℓ]</sup>)</div></li>'
        "</ol>"
    )
    r.p(
        R.note("Vai trò của ma trận trọng số W.",
               "Định hình độ dốc, co giãn và xoay không gian đặc trưng — xác định mức độ "
               "ảnh hưởng của từng nơ-ron tầng trước lên tầng sau."),
        R.note("Vai trò của vector độ chệch b.",
               "Dịch chuyển siêu phẳng quyết định trong không gian mà không bị gò bó phải "
               "đi qua gốc toạ độ."),
    )

    # ---- 1.3
    r.h(2, "1.3. Chứng minh toán học: vì sao bắt buộc phải có hàm kích hoạt phi tuyến?")
    r.p(
        "Đây là mệnh đề nền tảng của toàn bộ Deep Learning, và nó chứng minh được bằng "
        "vài dòng đại số. Giả sử ta loại bỏ hàm kích hoạt phi tuyến (tức chọn φ(z) = z là "
        "hàm đồng nhất). Xét mạng hai tầng ẩn:",
        '<div class="formula">h<sub>1</sub> = xW<sub>1</sub> + b<sub>1</sub></div>',
        '<div class="formula">h<sub>2</sub> = h<sub>1</sub>W<sub>2</sub> + b<sub>2</sub> '
        '= (xW<sub>1</sub> + b<sub>1</sub>)W<sub>2</sub> + b<sub>2</sub> '
        '= x(W<sub>1</sub>W<sub>2</sub>) + (b<sub>1</sub>W<sub>2</sub> + b<sub>2</sub>)</div>',
        "Đặt W<sub>composite</sub> = W<sub>1</sub>W<sub>2</sub> và b<sub>composite</sub> "
        "= b<sub>1</sub>W<sub>2</sub> + b<sub>2</sub>, ta thu được:",
        '<div class="formula">h<sub>2</sub> = xW<sub>composite</sub> + b<sub>composite</sub></div>',
    )
    r.p(
        R.note("Kết luận toán học.",
               "Phép hợp của bất kỳ số lượng tầng tuyến tính nào cũng chỉ tương đương với "
               "<strong>một</strong> phép biến đổi affine duy nhất. Việc xếp chồng nhiều "
               "tầng tuyến tính hoàn toàn không làm tăng khả năng biểu diễn "
               "(Expressive Power) của mạng so với mô hình hồi quy tuyến tính ban đầu. "
               "Do đó, tính phi tuyến là điều kiện tiên quyết tạo nên sức mạnh của "
               "Deep Learning.", "good"),
        "Mệnh đề này không dừng lại ở lý thuyết. Ở mục 2.3, tôi sẽ <strong>kiểm chứng nó "
        "bằng thực nghiệm có kiểm soát</strong>: huấn luyện một mạng y hệt nhưng bỏ hẳn "
        "hàm ReLU, rồi đo xem hiệu năng tụt bao nhiêu.",
    )

    # ---- 1.4
    r.h(2, "1.4. Hàm kích hoạt ReLU và khởi tạo trọng số He Normal")
    r.p(
        "Trong toàn bộ các kiến trúc mạng nơ-ron sâu được xây dựng trong báo cáo, hàm "
        "kích hoạt <strong>ReLU (Rectified Linear Unit)</strong> được lựa chọn cho tất cả "
        "các tầng ẩn:",
        '<div class="formula">ReLU(z) = max(0, z) &nbsp;&nbsp;·&nbsp;&nbsp; '
        'd/dz ReLU(z) = 1 nếu z &gt; 0, và = 0 nếu z ≤ 0</div>',
    )
    r.p(
        "<strong>Hai ưu thế quyết định của ReLU:</strong>"
        "<ul>"
        "<li><strong>Triệt tiêu hiện tượng bão hoà đạo hàm (Vanishing Gradient).</strong> "
        "Khi z &gt; 0, đạo hàm luôn bằng 1, cho phép gradient truyền ngược qua nhiều tầng "
        "mà không bị suy giảm theo cấp số nhân như Sigmoid hay Tanh.</li>"
        "<li><strong>Tạo tính thưa thớt (Sparsity).</strong> Các nơ-ron có giá trị z ≤ 0 "
        "bị khoá về đúng 0, giúp mạng học biểu diễn đặc trưng phân tách rời rạc và tăng "
        "tốc độ tính toán ma trận.</li>"
        "</ul>"
    )
    r.p(
        "<strong>Khởi tạo trọng số He Normal (Kaiming Normal).</strong> Để ngăn hiện tượng "
        "nổ gradient (Exploding) hoặc biến mất gradient ngay từ epoch đầu tiên, trọng số "
        "của mỗi tầng được khởi tạo theo phân phối Gauss có phương sai chuẩn hoá theo số "
        "nút vào d<sub>in</sub>:",
        '<div class="formula">W<sup>[ℓ]</sup> ~ 𝒩(0, √(2 / d<sub>in</sub>)) '
        '&nbsp;&nbsp;·&nbsp;&nbsp; b<sup>[ℓ]</sup> = 0</div>',
        "Hệ số √2 được thiết kế riêng cho ReLU: vì ReLU khoá khoảng một nửa số nơ-ron về "
        "0, phương sai tín hiệu bị giảm một nửa qua mỗi tầng; nhân bù hệ số 2 giữ cho "
        "phương sai ổn định xuyên suốt độ sâu của mạng.",
    )

    # ---- 1.5
    r.h(2, "1.5. Lan truyền tiến và lan truyền ngược thuần NumPy")
    r.p(
        "Quá trình tối ưu hoá mạng nơ-ron được thực hiện thông qua Gradient Descent. Phần "
        "này ghi lại đầy đủ hệ phương trình mà toàn bộ mã nguồn trong năm notebook hiện "
        "thực hoá nguyên văn."
    )

    r.h(3, "1.5.1. Lan truyền tiến (Forward Pass)")
    r.p(
        "Với các tầng ẩn ℓ = 1, …, L−1:",
        '<div class="formula">Z<sup>[ℓ]</sup> = A<sup>[ℓ−1]</sup>W<sup>[ℓ]</sup> + '
        'b<sup>[ℓ]</sup> &nbsp;&nbsp;·&nbsp;&nbsp; A<sup>[ℓ]</sup> = max(0, Z<sup>[ℓ]</sup>)</div>',
        "Tại tầng ra L, hình thức phụ thuộc vào loại bài toán:",
    )
    r.p(
        "<strong>Bài toán phân loại nhị phân</strong> (Chương 2, 3 và 5):",
        '<div class="formula">ŷ = σ(Z<sup>[L]</sup>) = 1 / (1 + e<sup>−Z</sup>) ∈ (0, 1)</div>',
        "với hàm mất mát Binary Cross-Entropy trên batch gồm m mẫu:",
        '<div class="formula">ℒ<sub>BCE</sub> = −(1/m) Σ [ y<sup>(i)</sup> log(ŷ<sup>(i)</sup>) '
        '+ (1 − y<sup>(i)</sup>) log(1 − ŷ<sup>(i)</sup>) ]</div>',
        "<strong>Bài toán hồi quy</strong> (Chương 4) — tầng ra <em>tuyến tính</em>:",
        '<div class="formula">ŷ = Z<sup>[L]</sup> &nbsp;&nbsp;·&nbsp;&nbsp; '
        'ℒ<sub>MSE</sub> = (1/2m) Σ (ŷ<sup>(i)</sup> − y<sup>(i)</sup>)²</div>',
    )

    r.h(3, "1.5.2. Lan truyền ngược (Backward Pass)")
    r.p(
        "Tại tầng ra L, gradient sai số được rút gọn kỳ diệu nhờ sự triệt tiêu giữa đạo "
        "hàm hàm mất mát và đạo hàm hàm kích hoạt:",
        '<div class="formula">dZ<sup>[L]</sup> = (1/m)(ŷ − y)</div>',
        R.note("Một điểm thú vị đáng ghi nhận.",
               "Công thức gradient tầng ra của <em>BCE + Sigmoid</em> và của <em>MSE + "
               "Linear</em> hoàn toàn giống nhau. Đây không phải trùng hợp ngẫu nhiên — cả "
               "hai cặp đều thuộc họ hàm mất mát chính tắc (canonical link) của phân phối "
               "mũ tương ứng. Hệ quả thực tiễn: toàn bộ phần lan truyền ngược qua các tầng "
               "ẩn giữ nguyên không đổi giữa bài toán phân loại và bài toán hồi quy, chỉ "
               "cần đổi hàm kích hoạt tầng cuối."),
        "Gradient của trọng số và độ chệch tại tầng L:",
        '<div class="formula">dW<sup>[L]</sup> = (A<sup>[L−1]</sup>)<sup>T</sup>dZ<sup>[L]</sup> '
        '&nbsp;·&nbsp; db<sup>[L]</sup> = Σ<sub>i</sub> dZ<sup>[L]</sup></div>',
        "Truyền ngược gradient về các tầng ẩn ℓ = L−1, L−2, …, 1:",
        '<div class="formula">dA<sup>[ℓ]</sup> = dZ<sup>[ℓ+1]</sup>(W<sup>[ℓ+1]</sup>)<sup>T</sup> '
        '&nbsp;·&nbsp; dZ<sup>[ℓ]</sup> = dA<sup>[ℓ]</sup> ⊙ 𝕀(Z<sup>[ℓ]</sup> &gt; 0)</div>',
        '<div class="formula">dW<sup>[ℓ]</sup> = (A<sup>[ℓ−1]</sup>)<sup>T</sup>dZ<sup>[ℓ]</sup> '
        '&nbsp;·&nbsp; db<sup>[ℓ]</sup> = Σ<sub>i</sub> dZ<sup>[ℓ]</sup></div>',
        "Cập nhật trọng số theo luật Gradient Descent với tốc độ học η:",
        '<div class="formula">W<sup>[ℓ]</sup> ← W<sup>[ℓ]</sup> − η·dW<sup>[ℓ]</sup> '
        '&nbsp;·&nbsp; b<sup>[ℓ]</sup> ← b<sup>[ℓ]</sup> − η·db<sup>[ℓ]</sup></div>',
    )

    r.p(R.code('''def _backward(self, y):
    """Lan truyền ngược — bản dịch trực tiếp hệ phương trình ở mục 1.5.2 sang code."""
    dZ = (self.As[-1] - y) / y.shape[0]          # dZ^[L] = (ŷ − y) / m
    for i in range(self.L - 1, -1, -1):
        dW = self.As[i].T @ dZ                    # dW^[ℓ]
        db = dZ.sum(axis=0, keepdims=True)        # db^[ℓ]
        if i > 0:
            dZ = (dZ @ self.W[i].T) * (self.Zs[i - 1] > 0)   # dA ⊙ 𝕀(Z > 0)
        self.W[i] -= self.lr * dW                 # cập nhật Gradient Descent
        self.b[i] -= self.lr * db''',
                "Toàn bộ thuật toán lan truyền ngược trong 9 dòng NumPy — dùng chung cho "
                "cả năm notebook."))

    # ---- 1.6
    r.h(2, "1.6. Mười sáu khái niệm nền tảng trong mạng nơ-ron và học biểu diễn")
    r.p(
        "Để củng cố cơ sở lý luận trước khi đi vào các bài toán thực nghiệm đối đầu, phần "
        "này tổng hợp có hệ thống 16 vấn đề khoa học cốt lõi định hình nguyên lý vận hành "
        "của mạng nơ-ron sâu."
    )
    concepts = [
        ("Khác biệt giữa ML truyền thống và Deep Learning",
         "ML truyền thống đòi hỏi con người thiết kế đặc trưng thủ công rồi đưa vào thuật "
         "toán tuyến tính. Deep Learning học đồng thời cả không gian biểu diễn đặc trưng "
         "ẩn lẫn hàm phân loại/hồi quy."),
        ("Ý nghĩa của phương trình ŷ = f₃(f₂(f₁(X)))",
         "Mạng nơ-ron là phép hợp của ba biến đổi liên tiếp: f₁ học biểu diễn sơ cấp, f₂ "
         "kết hợp thành biểu diễn trừu tượng, f₃ ánh xạ thành giá trị dự đoán."),
        ("Mục đích của ma trận trọng số W",
         "Định hình độ dốc, co giãn và xoay không gian đặc trưng; xác định mức độ ảnh "
         "hưởng của từng nơ-ron tầng trước lên tầng sau."),
        ("Mục đích của vector độ chệch b",
         "Dịch chuyển siêu phẳng quyết định trong không gian mà không bị gò bó phải đi qua "
         "gốc toạ độ."),
        ("Vì sao cần hàm kích hoạt phi tuyến",
         "Không có phi tuyến, phép hợp của mọi tầng chỉ là một hàm tuyến tính đơn lẻ, làm "
         "mất toàn bộ khả năng xấp xỉ các quan hệ phức tạp (chứng minh ở mục 1.3)."),
        ("Hàm ReLU thực hiện thao tác gì",
         "Lọc bỏ toàn bộ giá trị âm (gán về 0) và giữ nguyên giá trị dương, đưa tính phi "
         "tuyến và tính thưa thớt vào mạng."),
        ("Vì sao dùng Sigmoid ở tầng cuối bài toán phân loại",
         "Sigmoid nén giá trị số thực (−∞, +∞) về khoảng xác suất (0, 1), phù hợp với phân "
         "phối Bernoulli của biến cố nhị phân."),
        ("Vì sao cần hàm mất mát",
         "Cung cấp thước đo định lượng về sai lệch giữa dự đoán và thực tế, đóng vai trò "
         "hàm mục tiêu để thuật toán tối ưu hoá dẫn dắt việc học."),
        ("Binary Cross-Entropy đo lường điều gì",
         "Đo khoảng cách phân phối xác suất (KL divergence) giữa phân phối thực tế "
         "y ∈ {0, 1} và phân phối dự báo ŷ ∈ (0, 1)."),
        ("Gradient là gì",
         "Vector chứa các đạo hàm riêng của hàm mất mát đối với từng trọng số, chỉ hướng "
         "tăng nhanh nhất của hàm mất mát trong không gian tham số."),
        ("Thuật toán Backpropagation tính toán điều gì",
         "Áp dụng quy tắc dây chuyền (Chain Rule) để tính chính xác gradient ∇<sub>θ</sub>ℒ "
         "từ tầng ra ngược về tầng vào một cách hiệu quả."),
        ("Giải thuật Gradient Descent làm gì",
         "Cập nhật trọng số theo hướng ngược chiều gradient (−∇<sub>θ</sub>ℒ) để từng bước "
         "tiến dần về cực tiểu của hàm mất mát."),
        ("Vì sao trọng số được cập nhật lặp lại nhiều lần (Epochs)",
         "Vì bề mặt hàm mất mát là phi tuyến đa chiều, mỗi bước cập nhật chỉ di chuyển một "
         "bước nhỏ η; cần nhiều bước để mạng hội tụ về trạng thái tối ưu."),
        ("Vì sao phải chia tách dữ liệu Train và Test",
         "Để kiểm tra khả năng tổng quát hoá (Generalization) của mô hình trên dữ liệu "
         "chưa từng thấy, phát hiện và ngăn chặn hiện tượng học vẹt (Overfitting)."),
        ("Vì sao phải chuẩn hoá đặc trưng",
         "Giúp mặt đẳng mức hàm mất mát có dạng tròn đều đối xứng, giúp gradient descent "
         "hội tụ nhanh và ổn định, tránh dao động hình dích-dắc."),
        ("Khác biệt giữa xác suất và phân lớp dự đoán",
         "Xác suất ŷ ∈ [0, 1] biểu thị mức độ tự tin liên tục của mô hình; phân lớp dự "
         "đoán ŷ<sub>class</sub> ∈ {0, 1} là quyết định rời rạc thu được sau khi áp dụng "
         "ngưỡng quyết định (Threshold cut-off)."),
    ]
    r.p(R.table(
        ["#", "Khái niệm", "Nội dung cốt lõi"],
        [[str(i), f"<strong>{name}</strong>", body] for i, (name, body) in enumerate(concepts, 1)],
        "Mười sáu khái niệm nền tảng định hình nguyên lý vận hành của mạng nơ-ron sâu"))

    # Danh mục kho mã nguồn, bộ dữ liệu, notebook và REST API đã được đặt ở trang
    # "Tài nguyên và mã nguồn dự án" ngay sau mục lục, nên không lặp lại ở đây.


# ===========================================================================
# CHƯƠNG 2
# ===========================================================================
def _chapter2(r: R.Report, data: dict) -> None:
    base, imp, abl = data["baseline"], data["improved"], data["ablation"]
    bs = base["test_scores"]
    io_ = imp["test_scores_optimal"]
    cmp_ = imp["comparison_vs_baseline"]
    tau = imp["threshold"]["optimal"]

    r.h(1, "Chương 2. Thực nghiệm cơ sở và phân tích bóc tách kiến trúc")

    r.p(
        "Bước đầu tiên của chuỗi nghiên cứu là xây dựng một mạng nơ-ron ba tầng thuần "
        "NumPy trên tập dữ liệu chuẩn đoán tiểu đường gốc <em>Pima Indians Diabetes</em> "
        f"gồm {base['dataset']['n_samples']} bệnh nhân với "
        f"{base['dataset']['n_features']} đặc trưng y sinh.",
        "Mục tiêu của chương này <strong>không phải đạt điểm số cao</strong>. Mục tiêu là "
        "dựng một mô hình cơ sở có khuyết tật cố ý, phơi bày trọn vẹn các lỗi phương pháp "
        "luận thường gặp, rồi sửa từng lỗi một và đo chính xác mức cải thiện thu được.",
    )

    # ---- 2.1
    r.h(2, "2.1. Mô hình cơ sở và bốn khuyết tật phương pháp luận cố ý")
    arch = base["architecture"]
    r.p(R.table(
        ["Hạng mục cấu hình", "Giá trị"],
        [["Cấu hình mạng", f"<code class='inl'>{arch['layers']}</code> "
          "(2 tầng ẩn ReLU, tầng ra Sigmoid)"],
         ["Số tham số học được", f"<strong>{arch['n_params']}</strong> tham số"],
         ["Khởi tạo trọng số", arch["init"]],
         ["Tốc độ học η", str(arch["lr"])],
         ["Số chu kỳ huấn luyện", f"{arch['epochs']:,} epochs"],
         ["Phương pháp cập nhật", arch["optimizer"]],
         ["Ngưỡng quyết định", "0.50 (cố định, cứng nhắc)"]],
        "Cấu hình kiến trúc và siêu tham số của mô hình cơ sở"))

    r.p(
        "Bốn khuyết tật được cài vào một cách có chủ đích, mỗi khuyết tật đại diện cho một "
        "lỗi phương pháp luận rất phổ biến:"
    )
    zero = base["zero_audit"]
    zero_rows = sorted(zero.items(), key=lambda kv: -kv[1]["count"])
    r.p(
        "<ol>"
        "<li><strong>Rò rỉ dữ liệu (Data Leakage).</strong> Phép chuẩn hoá Z-Score được "
        "áp dụng trên <em>toàn bộ</em> tập dữ liệu <em>trước khi</em> phân chia Train/Test. "
        "Vì μ<sub>global</sub> và σ<sub>global</sub> chứa thông tin của tập kiểm thử, mô "
        "hình đã 'nhìn trộm' phân phối dữ liệu mà lẽ ra nó chưa từng thấy. Mọi điểm số thu "
        "được đều lạc quan giả tạo và mất tính khách quan khoa học.</li>"
        "<li><strong>Phân chia ngẫu nhiên không phân tầng.</strong> Việc chia 80/20 bằng "
        "hoán vị ngẫu nhiên thuần làm sai lệch tỷ lệ bệnh nhân mắc bệnh giữa hai tập: tập "
        f"Test chỉ có {base['split']['test_pos_rate']:.1%} dương tính trong khi quần thể là "
        f"{base['dataset']['positive_rate']:.1%}.</li>"
        "<li><strong>Bỏ qua các giá trị 0 phi lý sinh học.</strong> Trong y học, người sống "
        "không thể có nồng độ glucose huyết, huyết áp tâm trương hay chỉ số BMI bằng 0. "
        "Các số 0 này thực chất là giá trị thiếu bị mã hoá nhầm.</li>"
        "<li><strong>Áp đặt ngưỡng quyết định cứng nhắc 0.50.</strong> Khiến mô hình thiên "
        "lệch về nhãn đa số, dẫn đến tỷ lệ bỏ sót ca bệnh rất cao.</li>"
        "</ol>"
    )
    r.p(R.table(
        ["Chỉ số y sinh", "Số bản ghi bằng 0", "Tỷ lệ trên tổng mẫu"],
        [[f"<code class='inl'>{k}</code>", f"{v['count']}", f"{v['pct']:.2f}%"]
         for k, v in zero_rows],
        "Kiểm toán các giá trị 0 phi lý sinh học trong tập Pima gốc — mô hình cơ sở giữ "
        "nguyên toàn bộ", "num"))

    r.p(R.figure("nb_fig01_baseline_loss_curve.png",
                 "Đường cong suy giảm hàm mất mát Binary Cross-Entropy của mô hình cơ sở"))
    r.p(R.oim(
        f"Hàm mất mát giảm đều và trơn từ {base['loss']['start']:.4f} xuống "
        f"{base['loss']['end']:.4f} sau {arch['epochs']:,} epochs, không có dao động.",
        "Đường cong mượt chứng tỏ các phương trình lan truyền ngược viết tay hoạt động "
        "đúng về mặt toán học: gradient được tính chính xác và truyền qua ba tầng ổn định.",
        "Loss huấn luyện giảm <strong>không</strong> đồng nghĩa với năng lực sàng lọc lâm "
        "sàng. Một đường cong đẹp có thể che giấu một mô hình vô dụng trên tập kiểm thử — "
        "đó chính xác là điều xảy ra ở đây."))

    r.h(3, "2.1.1. Kết quả định lượng của mô hình cơ sở")
    r.p(R.table(
        ["Chỉ số đánh giá", "Giá trị", "Diễn giải"],
        [["Accuracy (Độ chính xác)", f"{bs['accuracy']:.2%}", "Trên tập kiểm thử độc lập"],
         ["Precision (Độ chuẩn xác)", f"{bs['precision']:.2%}",
          "Trong số ca báo dương, bao nhiêu là đúng"],
         ["<strong>Recall (Độ nhạy phát hiện bệnh)</strong>",
          f"<strong>{bs['recall']:.2%}</strong>",
          "Trong số ca bệnh thực tế, phát hiện được bao nhiêu"],
         ["F1-Score", f"{bs['f1']:.2%}", "Trung bình điều hoà Precision và Recall"],
         ["<strong>Số ca bệnh bị bỏ sót (FN)</strong>",
          f"<strong>{bs['FN']} ca</strong>",
          f"Trên tổng {bs['FN'] + bs['TP']} ca dương tính thực tế"]],
        f"Kết quả mô hình cơ sở trên {base['split']['n_test']} mẫu kiểm thử "
        f"(ngưỡng cứng 0.50)", "num"))
    r.p(R.note(
        "Vấn đề nghiêm trọng nhất.",
        f"Recall chỉ đạt {bs['recall']:.2%} — nghĩa là mô hình <strong>bỏ sót "
        f"{bs['FN']} trong tổng số {bs['FN'] + bs['TP']} bệnh nhân thực sự mắc bệnh</strong>. "
        "Trong bối cảnh sàng lọc y tế, đây là một thất bại nghiêm trọng: mỗi ca bỏ sót là "
        "một bệnh nhân mất cơ hội can thiệp điều trị sớm.", "bad"))

    # ---- 2.2
    r.h(2, "2.2. Quy trình cải tiến chuẩn hoá y sinh")
    r.p(
        "Nhằm khắc phục triệt để các hạn chế trên, tôi thiết lập quy trình thực nghiệm cải "
        "tiến với năm thay đổi. Điều quan trọng cần nhấn mạnh: <strong>kiến trúc mạng được "
        "giữ nguyên hoàn toàn</strong> "
        f"(<code class='inl'>{imp['architecture']['layers']}</code>, "
        f"{imp['architecture']['n_params']} tham số) để đảm bảo so sánh công bằng. Mọi "
        "chênh lệch điểm số đo được đều đến từ <em>phương pháp luận</em>, không phải từ "
        "việc làm mô hình to hơn."
    )

    r.h(3, "2.2.1. Làm sạch dữ liệu lâm sàng bằng trung vị")
    r.p(
        "Toàn bộ giá trị 0 phi lý được chuyển thành NaN và điền khuyết bằng trung vị của "
        "các mẫu dương trên <em>tập huấn luyện</em>:",
        '<div class="formula">x<sub>imputed</sub> = x nếu x &gt; 0 ; '
        'ngược lại = median({x<sub>i</sub> ∈ 𝒟<sub>train</sub> | x<sub>i</sub> &gt; 0})</div>',
        "<strong>Vì sao trung vị mà không phải trung bình?</strong> Phân phối của các chỉ "
        "số y sinh — đặc biệt Insulin và SkinThickness — lệch phải rất nặng. Giá trị trung "
        "bình sẽ bị kéo lệch bởi nhóm ca bệnh cực đoan, trong khi trung vị phản ánh chính "
        "xác xu thế trung tâm của quần thể.",
    )
    med = imp["preprocessing"]["median_imputation"]
    r.p(R.table(
        ["Chỉ số y sinh", "Trung vị tính trên tập Train"],
        [[f"<code class='inl'>{k}</code>", f"{v:.2f}"] for k, v in med.items()],
        "Giá trị trung vị lâm sàng dùng để điền khuyết — tính duy nhất trên tập Train", "num"))

    r.h(3, "2.2.2. Triệt tiêu hoàn toàn rò rỉ dữ liệu")
    r.p(
        "Bộ chuẩn hoá Z-Score chỉ tính μ<sub>train</sub> và σ<sub>train</sub> trên tập huấn "
        "luyện (<code class='inl'>fit_transform</code>), sau đó áp dụng cố định các tham số "
        "này để chuẩn hoá tập kiểm thử (<code class='inl'>transform</code>):",
        '<div class="formula">x̃<sub>train</sub> = (x<sub>train</sub> − μ<sub>train</sub>) / '
        'σ<sub>train</sub> &nbsp;·&nbsp; x̃<sub>test</sub> = (x<sub>test</sub> − '
        'μ<sub>train</sub>) / σ<sub>train</sub></div>',
        R.note("Cách kiểm chứng thực nghiệm rằng rò rỉ đã bị triệt tiêu.",
               "Sau chuẩn hoá, tập Train phải cho trung bình đúng bằng 0 và độ lệch chuẩn "
               "đúng bằng 1. Tập Test thì <strong>không</strong> — vì nó được biến đổi bằng "
               "thống kê của tập khác. Nếu tập Test cũng cho ra 0 và 1, đó chính là dấu "
               "hiệu rò rỉ."),
    )

    r.h(3, "2.2.3. Phân chia phân tầng và dò tìm ngưỡng quyết định tối ưu")
    pre = imp["preprocessing"]
    r.p(
        f"Phép chia phân tầng (Stratified 80/20) bảo toàn chính xác tỷ lệ nhãn bệnh: tập "
        f"Train {pre['train_pos_rate']:.2%} và tập Test {pre['test_pos_rate']:.2%}, khớp với "
        f"tỷ lệ quần thể {base['dataset']['positive_rate']:.2%}.",
        f"Ngưỡng xác suất được quét từ 0.10 đến 0.90 <strong>trên tập Train</strong>, tìm ra "
        f"giá trị tối ưu hoá chỉ số F1 là τ<sub>opt</sub> = <strong>{tau:.2f}</strong>, sau "
        f"đó áp cố định sang tập Test.",
        R.note("Nguyên tắc bất di bất dịch.",
               "Ngưỡng phải được dò trên tập Train. Nếu dò trực tiếp trên tập Test, ta lại "
               "rơi vào rò rỉ dữ liệu — chỉ là ở một dạng tinh vi hơn.", "warn"),
    )

    r.p(R.figure("nb_fig02_improved_convergence_threshold.png",
                 "Phân tích hội tụ và tối ưu hoá ngưỡng quyết định trong mô hình cải tiến"))
    lr_study = abl["learning_rate_study"]
    lr_txt = " · ".join(f"η={k}: {v['final_loss']:.4f}" for k, v in lr_study.items())
    r.p(R.oim(
        f"Bảng (a) so sánh bốn tốc độ học ({lr_txt}). Bảng (b) cho thấy mô hình được chọn "
        f"hội tụ về {imp['loss']['end']:.4f}. Bảng (c) là điểm giao của ba đường cong "
        f"Precision, Recall và F1 quanh τ = {tau:.2f}. Bảng (d) là ma trận nhầm lẫn tại "
        f"ngưỡng tối ưu với {io_['FN']} ca bỏ sót.",
        "η quá nhỏ khiến mạng dừng lại ở sườn dốc (chưa học đủ); η quá lớn khiến bước nhảy "
        "vượt qua đáy cực tiểu và dao động. Đường cong Precision và Recall cắt nhau tại "
        "vùng ngưỡng tối ưu — đó chính là điểm cân bằng giữa hai thái cực.",
        "Việc chọn siêu tham số không phải cảm tính mà là một bài toán khảo sát có kiểm "
        "soát. Và ngưỡng quyết định — thứ hoàn toàn nằm ngoài quá trình huấn luyện — lại "
        "là đòn bẩy mạnh nhất để cải thiện năng lực lâm sàng."))

    # ---- 2.3 bảng đối chứng
    r.h(2, "2.3. Bảng đối chứng trực diện: Cơ sở so với Cải tiến")
    r.p(R.table(
        ["Tiêu chí phương pháp luận", "Mô hình Cơ sở", "Mô hình Cải tiến", "Ý nghĩa"],
        [["Tiền xử lý dữ liệu", "Z-Score toàn tập (rò rỉ)",
          "Làm sạch 0 y tế + fit chỉ trên Train", "Triệt tiêu Data Leakage"],
         ["Phân chia Train/Test", "80/20 ngẫu nhiên thuần", "Phân tầng (Stratified)",
          f"Bảo toàn {base['dataset']['positive_rate']:.1%} nhãn bệnh"],
         ["Khởi tạo trọng số", "𝒩(0, 1) chuẩn", "He Normal √(2/d<sub>in</sub>)",
          "Ổn định gradient"],
         ["Ngưỡng quyết định", "0.50 cố định",
          f"{tau:.2f} (tối ưu F1 trên Train)", "Tối ưu hoá lâm sàng"]],
        "So sánh phương pháp luận giữa mô hình cơ sở và mô hình cải tiến"))

    def d(v):
        return f'<span style="color:#0a7a44"><strong>+{v:.2f}%</strong></span>' if v > 0 \
            else f'<span style="color:#c0392b"><strong>{v:.2f}%</strong></span>'

    r.p(R.table(
        ["Chỉ số định lượng", "Cơ sở (Baseline)", "Cải tiến (Improved)", "Mức cải thiện"],
        [["Final Train Loss", f"{base['loss']['end']:.4f}", f"{imp['loss']['end']:.4f}",
          f"Giảm {cmp_['loss_reduction_pct']:.1f}%"],
         ["Accuracy", f"{bs['accuracy']:.2%}", f"{io_['accuracy']:.2%}",
          d(cmp_["accuracy_delta_pp"])],
         ["Precision", f"{bs['precision']:.2%}", f"{io_['precision']:.2%}",
          d(cmp_["precision_delta_pp"])],
         ["<strong>Recall (Độ nhạy)</strong>", f"{bs['recall']:.2%}",
          f"<strong>{io_['recall']:.2%}</strong>", d(cmp_["recall_delta_pp"])],
         ["F1-Score", f"{bs['f1']:.2%}", f"{io_['f1']:.2%}", d(cmp_["f1_delta_pp"])],
         ["<strong>Số ca bỏ sót (FN)</strong>", f"{bs['FN']} ca",
          f"<strong>{io_['FN']} ca</strong>",
          f'<span style="color:#0a7a44"><strong>Giảm {cmp_["fn_reduction_pct"]:.1f}%'
          f'</strong></span>']],
        "Đối chứng định lượng toàn diện giữa mô hình cơ sở và mô hình cải tiến", "num"))

    r.p(R.note(
        "Kết quả cải tiến.",
        f"Recall tăng vọt từ {bs['recall']:.2%} lên <strong>{io_['recall']:.2%}</strong> "
        f"(+{cmp_['recall_delta_pp']:.2f} điểm phần trăm), đưa F1-Score từ mức "
        f"{bs['f1']:.2%} lên {io_['f1']:.2%}. Số ca bệnh bị bỏ sót giảm từ {bs['FN']} xuống "
        f"còn <strong>{io_['FN']} ca</strong> — cứu vãn được "
        f"{bs['FN'] - io_['FN']} bệnh nhân thực tế. Toàn bộ mức cải thiện này đạt được "
        "<strong>mà không thay đổi một nơ-ron nào</strong> trong kiến trúc mạng.", "good"))

    # ---- 2.4 ablation
    r.h(2, "2.4. Sáu nghiên cứu bóc tách kiến trúc (Ablation Studies)")
    r.p(
        "Bóc tách nghĩa là: giữ nguyên mọi thứ, chỉ thay đổi <strong>đúng một</strong> "
        "thành phần, rồi đo xem chỉ số tụt hay tăng bao nhiêu. Đó là cách duy nhất để "
        "khẳng định một thành phần thật sự có vai trò, chứ không phải chỉ 'có mặt cho đủ bộ'."
    )
    archs = abl["architectures"]
    r.p(R.table(
        ["Cấu hình Kiến trúc", "Tham số", "Hàm ẩn", "Final Loss", "Ngưỡng",
         "Test Acc", "Recall", "F1-Score", "Số ca FN"],
        [[a["Cấu hình Kiến trúc"], f"{a['Tham số']:,}", a["Hàm ẩn"],
          f"{a['Final Loss']:.4f}", f"{a['Ngưỡng']:.2f}",
          a["Test Acc"], a["Recall"], a["F1-Score"], str(a["Số ca FN"])]
         for a in archs],
        "Bảng đối chuẩn nghiên cứu bóc tách kiến trúc trên bài toán chuẩn đoán tiểu đường",
        "num"))

    r.p(R.figure("nb_fig03_ablation_architecture.png",
                 "Khảo sát bóc tách kiến trúc và độ nhạy tốc độ học trên mô hình NumPy "
                 "thuần từ đầu"))

    std, wide, shallow, lin = (a["scores"] for a in archs)
    r.p(R.oim(
        f"Mạng rộng ({archs[1]['Tham số']:,} tham số, gấp "
        f"{archs[1]['Tham số'] / archs[0]['Tham số']:.1f} lần mạng chuẩn) cho F1 "
        f"{wide['f1']:.2%} so với {std['f1']:.2%} của mạng chuẩn. Mạng nông cho F1 "
        f"{shallow['f1']:.2%}. Mạng bỏ ReLU kẹt ở loss {archs[3]['Final Loss']:.4f} với "
        f"Recall chỉ {lin['recall']:.2%} và {lin['FN']} ca bỏ sót.",
        "Trên tập dữ liệu nhỏ 768 mẫu, mạng quá rộng học vẹt thay vì học quy luật tổng "
        "quát. Mạng nông thiếu tầng kết hợp đặc trưng trung gian nên không dựng được biểu "
        "diễn phân cấp. Mạng bỏ ReLU suy biến thành một phép biến đổi tuyến tính đơn lẻ.",
        "Thực nghiệm số 4 là <strong>minh chứng thực nghiệm trực tiếp</strong> cho chứng "
        "minh toán học ở mục 1.3: xếp chồng các biến đổi tuyến tính không mang lại bất kỳ "
        "ưu thế biểu diễn nào. Đây là điểm mà lý thuyết và thực nghiệm gặp nhau."))

    r.h(3, "2.4.1. Phân tích chi tiết từng nghiên cứu bóc tách")
    r.p(
        "<ol>"
        f"<li><strong>Thực nghiệm 1 — Chiều rộng mạng.</strong> Mở rộng "
        f"{archs[0]['Tham số']:,} → {archs[1]['Tham số']:,} tham số làm F1 thay đổi từ "
        f"{std['f1']:.2%} xuống {wide['f1']:.2%}. Kết luận: trên dữ liệu nhỏ, nhiều tham số "
        f"hơn <em>không</em> đồng nghĩa tốt hơn.</li>"
        f"<li><strong>Thực nghiệm 2 — Độ nhạy tốc độ học.</strong> {lr_txt}. Giá trị η quá "
        f"nhỏ khiến mạng dừng ở sườn dốc (Underfitting); η quá lớn gây dao động quanh cực "
        f"tiểu; η = 0.02 cho đường cong suy giảm mượt mà và ổn định nhất.</li>"
        f"<li><strong>Thực nghiệm 3 — Chiều sâu mô hình.</strong> Loại bỏ tầng ẩn thứ hai "
        f"giảm số tham số còn {archs[2]['Tham số']:,}, kéo F1 xuống {shallow['f1']:.2%} "
        f"({(shallow['f1'] - std['f1']) * 100:+.2f} điểm). Mạng nông không có tầng kết hợp "
        f"đặc trưng trung gian.</li>"
        f"<li><strong>Thực nghiệm 4 — Vai trò của phi tuyến.</strong> Bỏ toàn bộ ReLU khiến "
        f"hàm mất mát kẹt ở {archs[3]['Final Loss']:.4f}, Recall rơi xuống {lin['recall']:.2%} "
        f"và số ca bỏ sót vọt lên {lin['FN']} ca.</li>"
        "<li><strong>Thực nghiệm 5 — Đánh đổi ngưỡng quyết định.</strong> Chi tiết ở bảng "
        "dưới.</li>"
        "<li><strong>Thực nghiệm 6 — Trực quan hoá không gian biểu diễn ẩn qua PCA 2D.</strong> "
        "Chi tiết ở mục 2.5.</li>"
        "</ol>"
    )
    tr = abl["threshold_tradeoff"]
    r.p(R.table(
        ["Ngưỡng τ", "Precision", "Recall", "F1-Score", "Số ca bỏ sót (FN)",
         "Số báo động giả (FP)"],
        [[t["Ngưỡng τ"], t["Precision"], t["Recall"], t["F1-Score"],
          str(t["Số ca bỏ sót (FN)"]), str(t["Số báo động giả (FP)"])] for t in tr],
        "Thực nghiệm 5 — Đánh đổi ngưỡng quyết định phân loại lâm sàng", "num"))

    # ---- 2.5 PCA
    r.h(2, "2.5. Minh chứng hình học về cơ chế học biểu diễn qua PCA 2D")
    r.p(
        "Đây là thực nghiệm trả lời câu hỏi cốt lõi: <em>mạng nơ-ron thực sự học được gì ở "
        "các tầng ẩn?</em> Tôi trích xuất vector kích hoạt tại từng tầng ẩn của mô hình cải "
        "tiến, rồi chiếu độc lập từng không gian xuống 2 chiều bằng PCA (tự cài bằng phân "
        "rã SVD, không dùng thư viện):",
        '<div class="formula">X ∈ ℝ<sup>8</sup> → H<sub>1</sub> ∈ ℝ<sup>16</sup> → '
        'H<sub>2</sub> ∈ ℝ<sup>8</sup> → ŷ ∈ ℝ<sup>1</sup></div>',
    )
    r.p(R.figure("nb_fig04_representation_pca.png",
                 "Trực quan hoá không gian biểu diễn ẩn X → H₁ → H₂ bằng PCA 2D trên tập "
                 "dữ liệu tiểu đường gốc"))
    r.p(R.oim(
        "Ở không gian đầu vào X (bảng A), hai đám mây dữ liệu — bệnh nhân tiểu đường (đỏ) "
        "và người khoẻ mạnh (xanh) — nằm đan xen hỗn độn, chồng lấn dày đặc. Qua tầng ẩn "
        "H₁ (bảng B) các điểm bắt đầu dãn nở. Đến tầng H₂ (bảng C), hai nhóm tách rời rõ rệt.",
        "Mỗi phép biến đổi affine kết hợp kích hoạt ReLU thực hiện một thao tác hình học: "
        "xoay, co giãn và bẻ gập không gian. Chuỗi các thao tác này dần dần biến một bài "
        "toán phân tách phi tuyến khó thành một bài toán phân tách tuyến tính dễ.",
        "Đây là <strong>định nghĩa thực nghiệm của Representation Learning</strong>: mạng "
        "<em>tự</em> tìm ra phép ánh xạ khiến tầng ra chỉ cần một phép biến đổi tuyến tính "
        "đơn giản kết hợp Sigmoid là đủ để phân loại chính xác."))

    r.h(2, "2.6. Tổng kết bài học thực nghiệm nền tảng")
    r.p(
        "<ol>"
        "<li>Mô hình cơ sở đã làm sáng tỏ các lỗi cố hữu về rò rỉ dữ liệu và mất cân bằng "
        "lớp khi áp dụng các kỹ thuật xử lý dữ liệu một cách máy móc.</li>"
        f"<li>Bằng các kỹ thuật tiền xử lý y sinh chính xác, phân chia phân tầng và dò "
        f"ngưỡng tối ưu, hiệu năng mô hình đã có bước nhảy vọt "
        f"(+{cmp_['recall_delta_pp']:.1f}% Recall) mà không đổi kiến trúc.</li>"
        "<li>Các nghiên cứu bóc tách đã chứng minh một cách khoa học vai trò không thể thay "
        "thế của tính phi tuyến ReLU và sự cân bằng cần thiết giữa chiều sâu và chiều rộng.</li>"
        "</ol>",
        "Những bài học nền tảng này chính là kim chỉ nam để mở rộng quy mô nghiên cứu lên "
        "<strong>ba bộ dữ liệu lớn trong thực tế</strong> — 100,000 hồ sơ y tế, 150,000 "
        "giao dịch bất động sản và 23,486 nhận xét văn bản — trong các chương tiếp theo.",
    )
