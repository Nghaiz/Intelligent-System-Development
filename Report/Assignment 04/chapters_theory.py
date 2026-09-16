"""Trang tài nguyên (sau mục lục) và Chương 1 — cơ sở lý thuyết mạng tích chập."""

from __future__ import annotations

import report_lib as R

GITHUB = "https://github.com/Nghaiz/Intelligent-System-Development"
GITHUB_A04 = GITHUB + "/tree/main/src/Assignment%2004"

DATASETS = [
    ("Chương 3 — Chuỗi văn bản",
     "Women's Clothing E-Commerce Reviews (23.486 đánh giá)",
     "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews"),
    ("Chương 4 — Bảng phân loại",
     "Diabetes Prediction Dataset (100.000 hồ sơ lâm sàng)",
     "https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset"),
    ("Chương 5 — Bảng hồi quy",
     "USA Real Estate Dataset (150.000 giao dịch)",
     "https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset"),
    ("Chương 6 — Ảnh xám",
     "MNIST Handwritten Digits (70.000 ảnh 28×28×1)",
     "https://www.kaggle.com/datasets/hojjatk/mnist-dataset"),
    ("Chương 7 — Ảnh màu",
     "CIFAR-10 Natural Images (60.000 ảnh 32×32×3)",
     "https://www.cs.toronto.edu/~kriz/cifar.html"),
]

NOTEBOOKS = [
    ("Chuỗi văn bản — 1D CNN thuần NumPy",
     "customer_comments/notebooks/01_comments_1d_cnn_numpy.ipynb"),
    ("Chuỗi văn bản — 1D CNN PyTorch",
     "customer_comments/notebooks/02_comments_1d_cnn_pytorch.ipynb"),
    ("Chuỗi văn bản — 1D CNN TensorFlow/Keras",
     "customer_comments/notebooks/03_comments_1d_cnn_tensorflow.ipynb"),
    ("Bảng y tế — 1D CNN thuần NumPy",
     "diabetes/notebooks/01_diabetes_1d_cnn_numpy.ipynb"),
    ("Bảng y tế — 1D CNN PyTorch",
     "diabetes/notebooks/02_diabetes_1d_cnn_pytorch.ipynb"),
    ("Bảng y tế — 1D CNN TensorFlow/Keras",
     "diabetes/notebooks/03_diabetes_1d_cnn_tensorflow.ipynb"),
    ("Bất động sản — 1D CNN hồi quy thuần NumPy",
     "house_price/notebooks/01_house_price_1d_cnn_numpy.ipynb"),
    ("Bất động sản — 1D CNN hồi quy PyTorch",
     "house_price/notebooks/02_house_price_1d_cnn_pytorch.ipynb"),
    ("Bất động sản — 1D CNN hồi quy TensorFlow/Keras",
     "house_price/notebooks/03_house_price_1d_cnn_tensorflow.ipynb"),
    ("MNIST — Khảo sát dữ liệu",
     "mnist/notebooks/00_mnist_visualization.ipynb"),
    ("MNIST — 2D CNN thuần NumPy (cơ sở và cải tiến)",
     "mnist/notebooks/01_mnist_cnn_scratch.ipynb"),
    ("MNIST — Đối chuẩn PyTorch và Keras",
     "mnist/notebooks/02_mnist_frameworks_comparison.ipynb"),
    ("CIFAR-10 — Khảo sát dữ liệu",
     "cifar10/notebooks/00_cifar10_visualization.ipynb"),
    ("CIFAR-10 — 2D CNN thuần NumPy (cơ sở và cải tiến)",
     "cifar10/notebooks/01_cifar10_cnn_scratch.ipynb"),
    ("CIFAR-10 — Đối chuẩn PyTorch và Keras",
     "cifar10/notebooks/02_cifar10_frameworks_comparison.ipynb"),
    ("Đối kháng MLP với CNN và phân tích không gian ẩn PCA",
     "mlp_vs_cnn/notebooks/03_mlp_vs_cnn_pca.ipynb"),
]


def write(r: R.Report, data: dict) -> None:
    _front(r, data)
    _chapter1(r, data)


# ===========================================================================
# TRANG TÀI NGUYÊN — nằm ngay sau mục lục, trước Chương 1
# ===========================================================================
def _front(r: R.Report, data: dict) -> None:
    n_records = (data["comments"]["dataset"]["n_raw"]
                 + data["diabetes"]["dataset"]["n_raw"]
                 + data["house"]["dataset"]["n_raw"]
                 + data["mnist"]["dataset"]["n_raw"]
                 + data["cifar"]["dataset"]["n_raw"])

    r.front(
        '<h1 class="nobreak">Tài nguyên và mã nguồn dự án</h1>',
        '<p class="lead">Mọi con số, bảng biểu và hình vẽ trong báo cáo này đều được sinh '
        'ra từ một lần chạy thật của các notebook liệt kê bên dưới. Không có giá trị nào '
        'được chép tay vào văn bản. Toàn bộ mã nguồn, dữ liệu và bản báo cáo này được '
        'quản lý phiên bản công khai.</p>')

    r.front("<h3>Kho lưu trữ mã nguồn</h3>")
    r.front(R.plain_table(
        ["Hạng mục", "Đường dẫn"],
        [["Kho lưu trữ GitHub của dự án", f'<a href="{GITHUB}">{GITHUB}</a>'],
         ["Thư mục bài tập số 4", f'<a href="{GITHUB_A04}">{GITHUB_A04}</a>'],
         ["Báo cáo nghiên cứu (tệp này)",
          '<code class="inl">Report/Assignment 04/A04_CT_nghiand.600.pdf</code>'],
         ["Hợp đồng tích hợp giữa các notebook",
          '<code class="inl">src/Assignment 04/CONTRACT.md</code>'],
         ["Hướng dẫn cài đặt và chạy lại",
          '<code class="inl">src/Assignment 04/README.md</code>'],
         ["Danh sách phụ thuộc",
          '<code class="inl">src/Assignment 04/requirements.txt</code>']],
        cls="links"))

    r.front("<h3>Năm bộ dữ liệu thực nghiệm</h3>")
    r.front(R.plain_table(
        ["Sử dụng ở", "Bộ dữ liệu", "Nguồn công khai"],
        [[where, name, f'<a href="{url}">{url}</a>'] for where, name, url in DATASETS],
        cls="links"))
    r.front(
        f'<p style="text-align:center;font-size:11.5pt;color:#444">Tổng cộng '
        f'<strong>{n_records:,} bản ghi</strong> trải trên ba dạng biểu diễn: '
        f'chuỗi ký tự, bảng số và ảnh raster.</p>')

    r.front("<h3>Mười sáu notebook thực nghiệm</h3>")
    r.front(R.plain_table(
        ["Notebook", "Đường dẫn trong kho mã nguồn"],
        [[label, f'<code class="inl">{path}</code>'] for label, path in NOTEBOOKS],
        cls="links"))
    r.front(
        '<p style="font-size:11.5pt;color:#444">Đường dẫn tính từ thư mục '
        '<code class="inl">src/Assignment 04/</code>. Mỗi notebook tự nạp dữ liệu, tự '
        'huấn luyện, tự vẽ hình và tự ghi tệp JSON kết quả vào '
        '<code class="inl">&lt;miền&gt;/reports/</code>. Bộ dựng báo cáo '
        '<code class="inl">Report/Assignment 04/build_report.py</code> chỉ đọc lại các '
        'tệp JSON đó.</p>')


# ===========================================================================
# CHƯƠNG 1
# ===========================================================================
def _chapter1(r: R.Report, data: dict) -> None:
    r.h(1, "Chương 1. Cơ sở lý thuyết về mạng nơ-ron tích chập")

    r.p(
        "Bài tập số 3 dừng lại ở mạng truyền thẳng: một chuỗi phép nhân ma trận xen kẽ "
        "hàm phi tuyến, trong đó mỗi nơ-ron nhìn thấy toàn bộ vector đầu vào. Bài tập số 4 "
        "đặt ra một câu hỏi hẹp hơn nhưng sâu hơn: <em>có thể thiết kế một tầng mạng "
        "chuyên đi tìm những hoa văn cục bộ hay không?</em> Câu trả lời là phép tích chập.",
        "Ràng buộc trung tâm của báo cáo vẫn giữ nguyên tinh thần của bài trước: "
        "<strong>mọi kiến trúc tích chập đều được hiện thực trước bằng NumPy thuần</strong> "
        "— tự viết cửa sổ trượt, tự viết đạo hàm theo bộ lọc, tự viết định tuyến gradient "
        "qua tầng gộp, tự viết Adam — rồi mới dựng lại đúng kiến trúc đó bằng PyTorch và "
        "TensorFlow để đối chiếu. Cách làm này biến khung thư viện từ một chiếc hộp đen "
        "thành một lớp tăng tốc cho công thức mà ta đã tự tay kiểm chứng.",
    )

    # ---- 1.1 ------------------------------------------------------------
    r.h(2, "1.1. Vì sao tầng liên kết đầy đủ không đủ cho dữ liệu có cấu trúc")
    r.p(
        "Một tầng Dense tính <span class='inl'>H = σ(XW + b)</span>. Công thức này rất "
        "tổng quát, và chính sự tổng quát ấy là vấn đề khi dữ liệu mang cấu trúc hình học.",
        "Xét một bức ảnh màu 224×224×3. Làm phẳng nó thành vector cho ra 150.528 số. Nối "
        "vector đó với một tầng ẩn 1.000 nơ-ron cần hơn 150 triệu trọng số chỉ cho riêng "
        "tầng đầu tiên. Ba hệ quả lập tức xuất hiện:",
    )
    r.p(
        '<ol class="tight">'
        '<li><strong>Bùng nổ tham số.</strong> Số trọng số tỉ lệ thuận với độ phân giải. '
        'Tăng ảnh lên gấp đôi mỗi chiều làm tham số tăng gấp bốn, kéo theo quá khớp và '
        'chi phí bộ nhớ.</li>'
        '<li><strong>Phá huỷ quan hệ lân cận.</strong> Thao tác làm phẳng coi mọi điểm ảnh '
        'là độc lập và hoán vị được. Nhưng điểm ảnh tại (i, j) rõ ràng có quan hệ với '
        '(i±1, j) và (i, j±1); mạng buộc phải học lại quan hệ đó từ đầu thay vì được cho '
        'sẵn.</li>'
        '<li><strong>Không có tính bất biến vị trí.</strong> Nếu một vật thể dịch sang phải '
        '5 điểm ảnh, toàn bộ tập trọng số đã học cho vị trí cũ trở nên vô dụng. Mạng phải '
        'học lại cùng một hoa văn ở từng vị trí khả dĩ.</li>'
        '</ol>')
    r.p(
        "Mạng tích chập giải quyết cả ba bằng hai nguyên lý duy nhất: <strong>liên kết cục "
        "bộ</strong> (mỗi nơ-ron chỉ nhìn một cửa sổ nhỏ) và <strong>chia sẻ trọng số</strong> "
        "(cùng một bộ lọc được dùng lại ở mọi vị trí). Số tham số khi đó chỉ phụ thuộc kích "
        "thước bộ lọc và số kênh, hoàn toàn độc lập với kích thước ảnh."
    )
    r.p('<div class="flow">CNN = Cục bộ + Chia sẻ trọng số + Phi tuyến</div>')

    # ---- 1.2 ------------------------------------------------------------
    r.h(2, "1.2. Phép tích chập: định nghĩa, ví dụ số và bản đồ đặc trưng")
    r.p(
        "Với bộ lọc kích thước <span class='inl'>K<sub>h</sub> × K<sub>w</sub></span>, bước "
        "trượt <span class='inl'>s</span> và đệm viền <span class='inl'>P</span>, phần tử "
        "tại vị trí (i, j) của bản đồ đặc trưng thứ k được tính bằng một tổng có trọng số "
        "trên một khối con của đầu vào:")
    r.p('<div class="formula">Y<sub>i,j,k</sub> = b<sub>k</sub> + Σ<sub>u</sub> Σ<sub>v</sub> '
        'Σ<sub>c</sub> X<sub>is+u−P, js+v−P, c</sub> · W<sub>u,v,c,k</sub></div>')
    r.p("Kích thước không gian đầu ra suy ra trực tiếp từ hình học cửa sổ trượt:")
    r.p('<div class="formula">H<sub>out</sub> = ⌊(H + 2P − d(K−1) − 1) / s⌋ + 1</div>')
    r.p(
        "trong đó <span class='inl'>d</span> là độ giãn nở, sẽ bàn ở mục 1.4. Với cấu hình "
        "thông dụng <span class='inl'>d = 1</span>, công thức rút gọn thành "
        "<span class='inl'>⌊(H + 2P − K)/s⌋ + 1</span>.")
    r.p(R.figure("th_fig_th_convolution.png",
                 "Bản chất tính toán của phép tích chập hai chiều: tích vô hướng từng phần "
                 "tử giữa cửa sổ tiếp nhận cục bộ và bộ lọc trọng số, cộng thêm độ chệch."))
    r.p(
        "Một ví dụ số một chiều làm rõ cơ chế. Cho "
        "<span class='inl'>x = [2, 1, 3, 4, 2]</span> và bộ lọc "
        "<span class='inl'>k = [0,5; −1; 0,5]</span> với <span class='inl'>b = 0</span>. "
        "Tại vị trí đầu tiên, cửa sổ là [2, 1, 3]:")
    r.p('<div class="formula">z<sub>1</sub> = 2(0,5) + 1(−1) + 3(0,5) = 1</div>')
    r.p(
        "Trượt sang phải một bước, cửa sổ thành [1, 3, 4] và "
        "<span class='inl'>z<sub>2</sub> = 1(0,5) + 3(−1) + 4(0,5) = −0,5</span>. Tiếp tục "
        "cho <span class='inl'>z<sub>3</sub> = −1,5</span>. Kết quả đầy đủ là "
        "<span class='inl'>[1; −0,5; −1,5]</span>. Điểm mấu chốt: cả ba giá trị dùng "
        "<em>cùng một</em> bộ ba trọng số. Đó chính là chia sẻ trọng số, và nó là lý do "
        "một bộ lọc học được khái niệm \"cạnh dọc\" chứ không phải \"cạnh dọc ở cột thứ 12\".",
        "Ví dụ số này được lặp lại bên trong mọi notebook NumPy của báo cáo như một phép "
        "kiểm tra tính đúng đắn: mã nguồn phải tái tạo đúng ba con số trên trước khi được "
        "phép huấn luyện bất cứ thứ gì.",
    )
    r.p(
        "Một lưu ý thuật ngữ đáng nhắc. Phép tích chập trong giải tích lật ngược bộ lọc "
        "trước khi nhân chập để bảo đảm tính giao hoán. Các thư viện học sâu bỏ bước lật "
        "đó, tức là thực chất tính <em>tương quan chéo</em>. Vì trọng số được học tự do từ "
        "dữ liệu, một bộ lọc lật ngược cũng học được không kém, nên năng lực biểu diễn "
        "không đổi và cộng đồng quy ước gọi chung là tích chập.")

    # ---- 1.3 ------------------------------------------------------------
    r.h(2, "1.3. Đệm viền và bước trượt")
    r.p(
        "Không đệm viền, mỗi tầng tích chập 3×3 làm ảnh co lại hai điểm ảnh mỗi chiều. "
        "Xếp chồng mười tầng là mất hai mươi điểm ảnh, đủ để xoá sạch một ảnh 28×28. "
        "Đệm viền bằng 0 với <span class='inl'>P = 1</span> giữ nguyên kích thước và cho "
        "phép dựng mạng sâu tuỳ ý. Nó còn sửa một khiếm khuyết tinh tế hơn: khi không đệm, "
        "điểm ảnh ở góc chỉ tham gia đúng một cửa sổ, trong khi điểm ảnh ở giữa tham gia "
        "chín cửa sổ, nên thông tin biên bị đánh giá thấp một cách có hệ thống.")
    r.p(R.figure("th_fig_th_padding.png",
                 "Đệm viền: (a) không đệm làm suy giảm độ phân giải sau mỗi tầng; "
                 "(b) đệm đối xứng P = 1 với bộ lọc 3×3 bảo toàn nguyên vẹn kích thước."))
    r.p(
        "Bước trượt đóng vai trò ngược lại. Với <span class='inl'>s = 1</span>, bộ lọc quét "
        "từng điểm ảnh và cho bản đồ đặc trưng mịn nhất. Với <span class='inl'>s = 2</span>, "
        "bộ lọc nhảy cách một điểm ảnh, giảm một nửa độ phân giải. So với tầng gộp, bước "
        "trượt lớn thực hiện cùng chức năng nén nhưng phép nén đó <em>có tham số học được</em>.")
    r.p(R.figure("th_fig_th_stride.png",
                 "Bước trượt: (a) s = 1 quét liên tục từng điểm ảnh; (b) s = 2 nhảy hai "
                 "điểm ảnh mỗi lần và thực hiện giảm độ phân giải."))

    # ---- 1.4 ------------------------------------------------------------
    r.h(2, "1.4. Tích chập giãn nở và trường tiếp nhận")
    r.p(
        "Tích chập giãn nở chèn khoảng trống vào giữa các phần tử của bộ lọc. Một bộ lọc "
        "3×3 với <span class='inl'>d = 2</span> bao phủ một vùng rộng 5×5 trên đầu vào "
        "nhưng vẫn chỉ mang đúng chín trọng số. Đây là cách mở rộng tầm nhìn theo cấp số "
        "nhân mà không trả thêm một tham số nào.")
    r.p(R.figure("th_fig_th_dilation.png",
                 "Tích chập giãn nở: (a) bộ lọc chuẩn 3×3 với d = 1; (b) bộ lọc giãn nở "
                 "d = 2 mở rộng vùng bao phủ lên 5×5 với đúng 9 trọng số."))
    r.p(
        "Trường tiếp nhận là vùng ảnh gốc chi phối một nơ-ron ở tầng sâu. Tầng tích chập "
        "3×3 đầu tiên cho trường tiếp nhận bằng 3. Tầng 3×3 thứ hai tổng hợp một vùng 3×3 "
        "của tầng một, nên gián tiếp bao phủ 5×5 trên ảnh gốc. Điều này dẫn tới một lựa "
        "chọn thiết kế đã thành chuẩn mực kể từ VGG: <strong>thay một tầng 5×5 bằng hai "
        "tầng 3×3 xếp chồng</strong>.")
    r.p(
        "Lý do là hai lợi ích cùng lúc. Về tham số, một tầng 5×5 với C kênh cần "
        "<span class='inl'>25C²</span> trọng số, trong khi hai tầng 3×3 chỉ cần "
        "<span class='inl'>18C²</span>, tiết kiệm 28%. Về biểu diễn, hai tầng cho phép chèn "
        "thêm một hàm ReLU ở giữa, nên hàm tổng hợp phi tuyến hơn hẳn một tầng đơn lẻ có "
        "cùng trường tiếp nhận.")
    r.p(R.figure("th_fig_th_receptive_field.png",
                 "Trường tiếp nhận hiệu dụng: hai tầng tích chập 3×3 liên tiếp bao phủ "
                 "vùng 5×5 trên ảnh gốc với số tham số ít hơn 28% so với một tầng 5×5."))
    r.p(
        "Cần phân biệt hai tính chất thường bị nhập làm một. <strong>Tương đương tịnh "
        "tiến</strong> là tính chất của tầng tích chập: ảnh dịch đi bao nhiêu thì bản đồ "
        "đặc trưng dịch đi đúng bấy nhiêu. <strong>Bất biến tịnh tiến</strong> là tính chất "
        "của tầng gộp: ảnh xê dịch nhẹ nhưng giá trị sau khi gộp không đổi. Mạng cần cả "
        "hai, và chúng đến từ hai tầng khác nhau.")

    # ---- 1.5 ------------------------------------------------------------
    r.h(2, "1.5. Tầng gộp và định tuyến gradient")
    r.p(
        "Gộp cực đại chọn giá trị kích hoạt lớn nhất trong mỗi cửa sổ, chẳng hạn "
        "<span class='inl'>[1, 5, 2, 4]</span> với cửa sổ 2 cho ra "
        "<span class='inl'>[5, 4]</span>. Gộp trung bình lấy trung bình cộng và làm mịn tín "
        "hiệu. Gộp trung bình toàn cục nén trọn một bản đồ đặc trưng H×W thành một số vô "
        "hướng, qua đó thay thế hoàn toàn tầng Flatten và cắt bỏ phần lớn tham số ở cuối "
        "mạng.")
    r.p(R.figure("th_fig_th_pooling.png",
                 "Ba cơ chế gộp đặc trưng: (a) gộp cực đại giữ phản hồi mạnh nhất; "
                 "(b) gộp trung bình làm mịn tín hiệu; (c) gộp trung bình toàn cục nén "
                 "mỗi kênh thành một giá trị vô hướng."))
    r.p(
        "Phần thú vị nằm ở chiều ngược. Trong lan truyền tiến, gộp cực đại ghi nhớ chỉ số "
        "của phần tử thắng cuộc. Trong lan truyền ngược, gradient từ tầng sau đi trọn vẹn "
        "về đúng phần tử đó, mọi phần tử còn lại trong cửa sổ nhận gradient bằng 0. Gộp "
        "trung bình thì ngược lại: mọi phần tử đóng góp như nhau nên gradient được chia "
        "đều. Hiện thực NumPy trong báo cáo lưu một mặt nạ cực đại ở lượt tiến để thực "
        "hiện chính xác phép định tuyến này.")

    # ---- 1.6 ------------------------------------------------------------
    r.h(2, "1.6. Vector hoá bằng im2col và khởi tạo He Normal")
    r.p(
        "Viết tích chập bằng bốn vòng lặp Python lồng nhau thì đúng về toán nhưng chậm tới "
        "mức không thể huấn luyện nổi. Giải pháp kinh điển là <span class='inl'>im2col</span>: "
        "trải phẳng từng cửa sổ trượt thành một hàng của ma trận "
        "<span class='inl'>X<sub>col</sub></span>, dàn phẳng bộ lọc thành "
        "<span class='inl'>W<sub>row</sub></span>, rồi quy toàn bộ phép tích chập trên cả "
        "mini-batch về một phép nhân ma trận duy nhất:")
    r.p('<div class="formula">Z<sub>col</sub> = X<sub>col</sub> W<sub>row</sub> + b</div>')
    r.p(R.figure("th_fig_th_im2col.png",
                 "Quy trình vector hoá phép tích chập bằng im2col và phép nhân ma trận "
                 "tổng quát GEMM."))
    r.p(
        "Cấu trúc ma trận này khiến lan truyền ngược trở nên gọn gàng. Đạo hàm theo trọng "
        "số và theo đầu vào đều là phép nhân ma trận chuyển vị:")
    r.p('<div class="formula">∂L/∂W<sub>row</sub> = X<sub>col</sub><sup>T</sup> ∂L/∂Z<sub>col</sub>'
        '&nbsp;&nbsp;&nbsp;&nbsp;'
        '∂L/∂X<sub>col</sub> = ∂L/∂Z<sub>col</sub> W<sub>row</sub><sup>T</sup></div>')
    r.p(
        "Hàm <span class='inl'>col2im</span> đảo ngược thao tác trải phẳng và "
        "<em>cộng dồn</em> gradient tại những vùng cửa sổ chồng lấn lên nhau. Phép cộng dồn "
        "này không phải chi tiết kỹ thuật tuỳ chọn mà là hệ quả trực tiếp của chia sẻ trọng "
        "số: một trọng số đã tham gia vào nhiều vị trí thì phải nhận gradient từ tất cả "
        "những vị trí đó.")
    r.p(
        "Khởi tạo trọng số là mắt xích cuối cùng. Vì ReLU đặt một nửa tín hiệu về 0, phương "
        "sai kích hoạt bị giảm khoảng một nửa sau mỗi tầng. Dùng khởi tạo chuẩn thông "
        "thường thì tín hiệu tắt dần về 0 chỉ sau vài tầng. He Normal bù chính xác phần "
        "hụt đó bằng cách nhân đôi phương sai khởi tạo:")
    r.p('<div class="formula">Var(W) = 2 / n<sub>in</sub> = 2 / (C<sub>in</sub> · K<sub>h</sub> · K<sub>w</sub>)</div>')
    r.p(
        "Chương 6 và Chương 7 sẽ đo trực tiếp tác động của lựa chọn này: mô hình cơ sở dùng "
        "khởi tạo ngẫu nhiên tỉ lệ cố định, mô hình cải tiến dùng He Normal cùng đệm viền "
        "và lịch giảm tốc độ học, và khoảng cách giữa hai bên được báo cáo bằng điểm phần "
        "trăm tuyệt đối.")

    # ---- 1.7 ------------------------------------------------------------
    r.h(2, "1.7. Kiến trúc như một phép hợp hàm và bài toán quyết định tầng cuối")
    r.p(
        "Điểm gắn kết bài tập số 4 với bài tập số 3 nằm ở chỗ mạng tích chập không phải "
        "một ý tưởng khác biệt hoàn toàn. Nó vẫn là một phép hợp hàm:")
    r.p('<div class="formula">ŷ = f<sub>L</sub> ∘ f<sub>L−1</sub> ∘ … ∘ f<sub>1</sub>(X)</div>')
    r.p(
        "Khác biệt duy nhất là một số hàm <span class='inl'>f<sub>i</sub></span> giờ đây là "
        "tích chập thay vì phép nhân ma trận đầy đủ. Toàn bộ bộ máy lan truyền ngược, quy "
        "tắc chuỗi và cập nhật gradient <span class='inl'>θ ← θ − η∇<sub>θ</sub>L</span> "
        "giữ nguyên.")
    r.p(R.figure("th_fig_th_architecture.png",
                 "Mạng tích chập như một phép hợp hàm tham số hoá, kèm hình dạng tensor "
                 "tại mỗi chặng biến đổi."))
    r.p(R.figure("th_fig_th_backprop.png",
                 "Lượt tiến và lượt ngược trên cùng một ngăn xếp tầng: đạo hàm theo trọng "
                 "số, định tuyến gradient qua mặt nạ cực đại và bước cập nhật tham số."))
    r.p(
        "Tầng cuối cùng thì không được phép tuỳ tiện: nó phải khớp với bài toán. Đây là "
        "nguyên tắc chi phối toàn bộ phần thực nghiệm của báo cáo.")
    r.p(R.table(
        ["Loại bài toán", "Kích hoạt đầu ra", "Hàm mất mát", "Áp dụng ở chương"],
        [["Phân loại nhị phân", "Sigmoid", "Binary Cross-Entropy", "Chương 3, Chương 4"],
         ["Phân loại đa lớp", "Softmax", "Categorical Cross-Entropy", "Chương 6, Chương 7"],
         ["Hồi quy giá trị liên tục", "Tuyến tính", "MSE hoặc MAE", "Chương 5"]],
        "Kiến trúc tầng cuối được quyết định bởi bài toán, không phải bởi sở thích."))
    r.p('<div class="flow">Kiến trúc = Dữ liệu + Bài toán + Biểu diễn</div>')

    # ---- 1.8 ------------------------------------------------------------
    r.h(2, "1.8. Softmax, Cross-Entropy và một mẹo số học bắt buộc")
    r.p(
        "Với vector logits <span class='inl'>z</span>, softmax cho xác suất của lớp k bằng "
        "<span class='inl'>ŷ<sub>k</sub> = exp(z<sub>k</sub>) / Σ<sub>j</sub> exp(z<sub>j</sub>)</span>. "
        "Hiện thực ngây thơ của công thức này tràn số ngay khi một logit vượt quá khoảng "
        "700, vì <span class='inl'>exp(700)</span> đã vượt giới hạn số thực 64 bit. Mọi "
        "notebook trong báo cáo trừ <span class='inl'>max<sub>j</sub> z<sub>j</sub></span> "
        "trước khi lấy hàm mũ; phép trừ này không đổi giá trị softmax nhưng loại bỏ hoàn "
        "toàn khả năng tràn.")
    r.p("Hàm mất mát Cross-Entropy trên một lô m mẫu với nhãn one-hot y là:")
    r.p('<div class="formula">L = − (1/m) Σ<sub>i</sub> Σ<sub>k</sub> y<sub>ik</sub> log(ŷ<sub>ik</sub> + ε)</div>')
    r.p(
        "Gộp softmax và Cross-Entropy lại rồi lấy đạo hàm theo logits cho một kết quả đẹp "
        "tới mức đáng ngờ:")
    r.p('<div class="formula">∂L / ∂z = ŷ − y</div>')
    r.p(
        "Đây là điểm khởi đầu của toàn bộ chuỗi lan truyền ngược qua Dense, gộp và tích "
        "chập trong các hiện thực NumPy. Sự gọn gàng của nó cũng là lý do hai hàm này gần "
        "như luôn được cài đặt chung thành một khối thay vì hai tầng riêng biệt.")

    # ---- 1.9 ------------------------------------------------------------
    r.h(2, "1.9. Ba cách cài đặt: điều gì đổi, điều gì không")
    r.p(
        "Báo cáo hiện thực mỗi kiến trúc ba lần. Câu hỏi đáng hỏi không phải là bên nào "
        "thắng, mà là <em>thứ gì thay đổi và thứ gì bất biến</em> giữa ba cách cài đặt.")
    r.p(R.table(
        ["Thành phần", "NumPy thuần", "PyTorch", "TensorFlow / Keras"],
        [["Tích chập", "<code class='inl'>conv1d/2d()</code> tự viết",
          "<code class='inl'>nn.Conv1d/2d</code>", "<code class='inl'>layers.Conv1D/2D</code>"],
         ["Kích hoạt", "<code class='inl'>relu()</code> tự viết",
          "<code class='inl'>nn.ReLU</code>", "<code class='inl'>activation='relu'</code>"],
         ["Gộp", "Mặt nạ cực đại tự viết",
          "<code class='inl'>nn.MaxPool1d/2d</code>", "<code class='inl'>MaxPooling1D/2D</code>"],
         ["Gradient", "Đạo hàm giải tích viết tay", "Autograd",
          "<code class='inl'>GradientTape</code> / tự động"],
         ["Cập nhật tham số", "Adam tự viết", "<code class='inl'>optimizer.step()</code>",
          "<code class='inl'>model.fit()</code>"],
         ["Vòng lặp huấn luyện", "Viết tay hoàn toàn", "Viết tay tường minh",
          "<code class='inl'>fit()</code> hoặc vòng lặp tuỳ biến"]],
        "Điều thay đổi giữa ba cách cài đặt là tầng hiện thực, không phải mô hình toán học."))
    r.p(
        "Điều <em>không</em> đổi: dữ liệu đầu vào, cách tách tập, quy trình tiền xử lý, ý "
        "tưởng kiến trúc, hàm mất mát, bộ chỉ số đánh giá và mục tiêu tối ưu. Chính vì vậy "
        "một phép so sánh công bằng bắt buộc phải kiểm soát tất cả những yếu tố đó, và "
        "Chương 2 mở đầu bằng việc nêu rõ thiết kế kiểm soát ấy.")
    r.p(R.note(
        "Bài học kỹ nghệ phần mềm.",
        "Cùng một mô hình toán học không có nghĩa là cùng một hiện thực. Khi hai khung thư "
        "viện cho ra số liệu lệch nhau dưới một điểm phần trăm, cái ta chứng minh được là "
        "nền toán học vững, chứ không phải khung nào ưu việt hơn."))

    # ---- 1.10 -----------------------------------------------------------
    r.h(2, "1.10. Các hướng kiến trúc phát triển từ tích chập cơ bản")
    r.p(
        "Những kiến trúc hiện đại đều xây trên đúng các khối vừa mô tả. Bảng dưới tóm tắt "
        "cải tiến cốt lõi của từng mốc, để đặt các thực nghiệm của báo cáo vào đúng vị trí "
        "trên bản đồ chung.")
    r.p(R.table(
        ["Kiến trúc", "Cải tiến chính", "Ý nghĩa"],
        [["LeNet-5", "Tích chập, gộp và liên kết đầy đủ nối tiếp",
          "Nền móng cho nhận dạng chữ số"],
         ["AlexNet", "ReLU, dropout, tăng cường dữ liệu, huấn luyện trên GPU",
          "Đưa CNN sâu lên quy mô ImageNet"],
         ["VGG", "Xếp chồng thuần bộ lọc 3×3",
          "Thiết kế đồng nhất nhưng nhiều tham số"],
         ["Inception", "Nhiều nhánh kernel và tích chập 1×1",
          "Học đặc trưng đa tỉ lệ trong cùng một khối"],
         ["ResNet", "Kết nối tắt F(x) + x",
          "Cho phép mạng rất sâu nhờ dòng gradient không suy giảm"],
         ["DenseNet", "Nối đặc trưng từ mọi tầng trước",
          "Tái sử dụng đặc trưng, giảm số tham số cần thiết"],
         ["MobileNet", "Tích chập tách theo chiều sâu",
          "Cắt mạnh FLOPs cho thiết bị biên"],
         ["EfficientNet", "Scale đồng thời độ sâu, độ rộng và độ phân giải",
          "Cân bằng chất lượng với chi phí theo một quy luật"],
         ["Vision Transformer", "Tự chú ý toàn cục trên các mảnh ảnh",
          "Thay cục bộ bằng toàn cục, đòi hỏi dữ liệu lớn hơn"]],
        "Một số hướng phát triển tiêu biểu đi ra từ khối tích chập cơ bản."))
    r.p(
        "Các thực nghiệm trong báo cáo dừng ở mức LeNet và VGG thu nhỏ. Lựa chọn này là có "
        "chủ đích: mục tiêu của bài tập là hiểu tường tận cơ chế, và một kiến trúc đủ nhỏ "
        "để lan truyền ngược viết tay chạy được trong vài phút có giá trị sư phạm cao hơn "
        "một mạng sâu mà ta chỉ gọi hàm.")

    # ---- 1.11 -----------------------------------------------------------
    r.h(2, "1.11. Bộ chỉ số đánh giá dùng xuyên suốt báo cáo")
    r.p(
        "Với bài toán phân loại, độ chính xác là tỉ lệ dự đoán đúng trên toàn tập kiểm thử. "
        "Chỉ số này đủ khi các lớp cân bằng, nhưng dễ đánh lừa khi không. Tập Diabetes ở "
        "Chương 4 có khoảng 8,5% mẫu dương; một mô hình luôn trả lời \"không mắc bệnh\" đạt "
        "trên 91% độ chính xác mà không có chút giá trị lâm sàng nào. Vì vậy mọi bảng kết "
        "quả phân loại trong báo cáo đều kèm precision, recall và F1 trên lớp dương.")
    r.p(
        "Với phân loại mười lớp ở Chương 6 và Chương 7, báo cáo dùng trung bình macro để "
        "mỗi lớp đóng góp ngang nhau vào kết luận:")
    r.p('<div class="formula">Macro-F1 = (1/K) Σ<sub>k</sub> F1<sub>k</sub></div>')
    r.p(
        "Với bài toán hồi quy ở Chương 5, ba chỉ số được báo cáo cùng lúc: RMSE phạt nặng "
        "sai số lớn, MAE phản ánh sai số điển hình bằng chính đơn vị đô-la, và R² cho biết "
        "tỉ lệ phương sai được giải thích. Ba con số này trả lời ba câu hỏi khác nhau, nên "
        "chỉ trích dẫn một trong ba là che mất hai phần ba bức tranh.")
    r.p(
        "Cuối cùng, mọi bảng đối chuẩn đều kèm số tham số và thời gian huấn luyện. Một mô "
        "hình hơn đối thủ 0,3 điểm phần trăm nhưng tốn gấp mười lần thời gian không phải "
        "là mô hình tốt hơn trong mọi ngữ cảnh, và báo cáo cố gắng không giấu đi sự đánh "
        "đổi đó.")
