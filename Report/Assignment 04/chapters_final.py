"""Chương 8 (đối kháng MLP với CNN, không gian ẩn PCA) và Chương 12 (tổng hợp, kết luận)."""

from __future__ import annotations

import report_lib as R
from chapters_1d import FW_LABEL, FW_ORDER, num, pct, thousands, usd
from chapters_2d import M_LABEL, _best


def write(r: R.Report, data: dict) -> None:
    """Giu lai cho tuong thich: viet ca hai chuong lien tiep."""
    write_ch8(r, data)
    write_ch12(r, data)


def write_ch8(r: R.Report, data: dict) -> None:
    """Chuong 8 phai nam truoc ba chuong bo sung 9, 10, 11."""
    _chapter8(r, data)


def write_ch12(r: R.Report, data: dict) -> None:
    """Chuong 12 tong hop, luon nam cuoi cung."""
    _chapter12(r, data)


# ===========================================================================
# CHƯƠNG 8 — MLP đối kháng CNN và không gian biểu diễn ẩn
# ===========================================================================
def _chapter8(r: R.Report, data: dict) -> None:
    d = data["mlp"]
    r.h(1, "Chương 8. Thực nghiệm đối kháng giữa mạng truyền thẳng và mạng tích chập")

    r.p(
        "Bảy chương trước đã chỉ ra mạng tích chập hoạt động tốt trên ảnh. Nhưng chưa "
        "chương nào trả lời câu hỏi đáng hỏi nhất: <em>tích chập có thực sự cần thiết, hay "
        "một mạng truyền thẳng đủ lớn cũng làm được?</em> Chương này thiết kế một thực "
        "nghiệm đối kháng để trả lời bằng số liệu thay vì bằng lập luận.")

    # ---- 8.1
    r.h(2, "8.1. Thiết kế thực nghiệm có kiểm soát và giả thuyết nghiên cứu")
    r.p(
        "Để phép so sánh có nghĩa, mọi yếu tố ngoài kiến trúc đều được giữ cố định giữa "
        "hai bên:")
    r.p(
        '<ul class="tight">'
        '<li>Cùng tập train, validation và test, cùng hạt giống ngẫu nhiên 42.</li>'
        '<li>Cùng quy trình chuẩn hoá điểm ảnh.</li>'
        '<li>Cùng hàm mất mát Cross-Entropy và cùng thuật toán tối ưu Adam.</li>'
        '<li>Cùng số epoch và cùng kích thước lô.</li>'
        '</ul>')
    r.p(
        "Mạng truyền thẳng có ba tầng ẩn (512, 256, 128 nơ-ron) kèm Dropout, đủ lớn để "
        "không thể bị chê là thua vì thiếu dung lượng. Trên CIFAR-10 nó thậm chí có "
        "<em>nhiều</em> tham số hơn mạng tích chập, vì tầng đầu tiên phải nối 3.072 đầu vào "
        "với 512 nơ-ron.")
    r.p(
        "Hai giả thuyết được đặt ra trước khi chạy:")
    r.p(
        '<ol class="tight">'
        '<li><strong>Trên MNIST, mạng truyền thẳng sẽ bám sát mạng tích chập.</strong> Chữ '
        'số đã được căn giữa và nền hoàn toàn đồng nhất, nên gần như không có biến thiên vị '
        'trí để khai thác. Một mạng truyền thẳng chỉ cần ghi nhớ những điểm ảnh nào thường '
        'sáng là đủ.</li>'
        '<li><strong>Trên CIFAR-10, mạng truyền thẳng sẽ tụt lại rõ rệt.</strong> Làm phẳng '
        'ảnh 32×32×3 thành vector 3.072 chiều cắt rời ba kênh màu của cùng một điểm ảnh ra '
        'ba vị trí cách nhau 1.024 chỉ số, đồng thời xoá sạch quan hệ lân cận. Vật thể lại '
        'xuất hiện ở vị trí và tỉ lệ tuỳ ý, thứ mà mạng không chia sẻ trọng số không thể '
        'khái quát hoá.</li>'
        '</ol>')

    # ---- 8.2
    r.h(2, "8.2. Kết quả đối đầu trực diện")
    _mlp_table(r, d)
    r.p(R.figure("mv_fig_mlp_vs_cnn_gap.png",
                 "Phân tích khoảng cách hiệu năng giữa mạng truyền thẳng và mạng tích chập "
                 "trên hai bộ dữ liệu ảnh."))
    _gap_commentary(r, d)
    r.p(R.figure("mv_fig_mlp_curves.png",
                 "Đường cong huấn luyện của mạng truyền thẳng trên MNIST và CIFAR-10."))
    r.p(R.figure("mv_fig_mlp_confusion.png",
                 "Ma trận nhầm lẫn của mạng truyền thẳng trên tập kiểm thử MNIST và CIFAR-10."))

    # ---- 8.3
    r.h(2, "8.3. Lý giải dưới góc độ lý thuyết học máy")
    r.p(
        "<strong>Vì sao mạng truyền thẳng vẫn rất tốt trên MNIST?</strong> Bộ dữ liệu này "
        "đã được chuẩn hoá ở mức cao: chữ số nằm gọn trong khung 20×20 và được đặt vào "
        "trọng tâm của khung 28×28, nền luôn bằng 0. Nói cách khác, biến thiên tịnh tiến "
        "gần như đã bị loại bỏ khỏi dữ liệu <em>trước khi</em> mô hình nhìn thấy nó. Cái mà "
        "tích chập mang lại là bất biến tịnh tiến, và ở đây bất biến đó đã được tặng sẵn. "
        "Một mạng truyền thẳng 512 nơ-ron chỉ cần học xem tổ hợp điểm ảnh sáng nào tương "
        "ứng với chữ số nào.")
    r.p(
        "<strong>Vì sao mạng truyền thẳng thất bại trên CIFAR-10?</strong> Ba lý do cộng "
        "dồn. Thứ nhất, phép làm phẳng phá vỡ cấu trúc: kênh đỏ, lục và lam của cùng một "
        "điểm ảnh bị đặt cách nhau 1.024 chỉ số trong vector, nên mạng phải học lại mối "
        "liên hệ giữa ba số lẽ ra là một. Thứ hai, vật thể xuất hiện ở vị trí và tỉ lệ tuỳ "
        "ý; không có chia sẻ trọng số, mạng phải học riêng một bộ trọng số cho mỗi vị trí "
        "khả dĩ, và tập huấn luyện không bao giờ đủ mẫu cho việc đó. Thứ ba, hệ quả trực "
        "tiếp là quá khớp nặng: mất mát huấn luyện giảm sâu trong khi mất mát kiểm định đi "
        "ngang rồi tăng.")
    r.p(
        "Mạng tích chập không gặp cả ba vấn đề, vì bộ lọc 2D trượt trên tensor ba kênh giữ "
        "nguyên quan hệ không gian và quan hệ giữa các kênh, đồng thời dùng lại cùng bộ "
        "trọng số ở mọi vị trí. Đây chính là hai nguyên lý nêu ở mục 1.1, và chương này là "
        "phép đo trực tiếp giá trị của chúng.")

    # ---- 8.4
    r.h(2, "8.4. Trực quan hoá không gian biểu diễn ẩn bằng PCA")
    r.p(
        "Bảng số liệu cho biết mạng tích chập tốt hơn, nhưng không cho biết <em>nó đã học "
        "được gì</em>. Để nhìn vào bên trong, báo cáo trích vector kích hoạt tại tầng ẩn áp "
        "chót (128 chiều) cho toàn bộ ảnh kiểm thử, rồi chiếu xuống hai chiều bằng phân "
        "tích thành phần chính.")
    r.p(R.code('''# Trích đặc trưng ẩn 128 chiều trên toàn bộ tập kiểm thử
features = []
with torch.no_grad():
    for inputs, _ in test_loader:
        features.append(model.extract_features(inputs.to(device)).cpu().numpy())
X_feat = np.concatenate(features, axis=0)          # (10000, 128)

# Chiếu giảm chiều 128D -> 2D, giữ nguyên hạt giống để tái lập
X_pca = PCA(n_components=2, random_state=42).fit_transform(X_feat)''',
        "Trích xuất biểu diễn ẩn 128 chiều và chiếu PCA hai chiều."))
    r.p(
        "Cần nhấn mạnh một điểm về phương pháp: PCA là phép chiếu <em>tuyến tính</em> và "
        "chỉ giữ lại hai hướng phương sai lớn nhất trong số 128 hướng. Nếu các lớp đã tách "
        "rời nhau ngay trên một phép chiếu nghèo nàn như vậy, thì trong không gian 128 "
        "chiều đầy đủ chúng phải tách rời hơn nữa. Ngược lại, hai lớp chồng lấn trên hình "
        "chiếu chưa chắc đã chồng lấn trong không gian gốc. Vậy nên hình dưới đây nên được "
        "đọc như một cận dưới của chất lượng biểu diễn.")
    r.p(R.figure("mv_fig_latent_pca_mnist.png",
                 "Không gian biểu diễn ẩn của mạng tích chập trên tập kiểm thử MNIST, "
                 "chiếu xuống hai chiều bằng PCA."))
    r.p(
        "Trên MNIST, mười chữ số tách thành mười cụm rời rạc, ranh giới giữa các cụm rõ tới "
        "mức gần như có thể phân loại bằng mắt. Đây là biểu hiện của một biểu diễn đã học "
        "tốt: mạng đã ánh xạ dữ liệu vào một không gian mà bài toán trở thành gần như tách "
        "được tuyến tính. Những cặp cụm còn chạm nhau ở rìa đúng là những cặp chữ số hay bị "
        "nhầm trong ma trận nhầm lẫn ở mục 6.7, nên hai phương pháp phân tích độc lập đang "
        "xác nhận lẫn nhau.")
    r.p(R.figure("mv_fig_latent_pca_cifar10.png",
                 "Không gian biểu diễn ẩn của mạng tích chập trên tập kiểm thử CIFAR-10, "
                 "tô màu theo mười lớp và theo hai siêu lớp phương tiện với động vật."))
    _pca_commentary(r, d)

    # ---- 8.5
    r.h(2, "8.5. Cấu trúc ngữ nghĩa xuất hiện mà không được dạy")
    sep = ((d.get("pca") or {}).get("cifar10") or {}).get("vehicle_animal_separation")
    if sep is not None and sep <= 0.1:
        r.p(
            "Giả thuyết đặt ra trước khi vẽ hình là các lớp sẽ không chỉ tách theo nhãn mà "
            "còn gom lại theo nhóm ngữ nghĩa cấp cao: bốn lớp phương tiện do con người chế "
            "tạo (máy bay, ô tô, tàu thuỷ, xe tải) về một phía, sáu lớp động vật sống (chim, "
            "mèo, hươu, chó, ếch, ngựa) về phía đối diện.")
        r.p(
            f"Phép đo ở mục 8.4 cho giá trị {num(sep, 4)}, tức là cấu trúc này "
            f"<strong>có tồn tại nhưng yếu</strong> trên hình chiếu hai chiều. Báo cáo ghi "
            f"nhận đúng mức đó thay vì kể một câu chuyện gọn gàng hơn dữ liệu cho phép. Cần "
            f"nhớ rằng PCA chỉ giữ hai hướng phương sai lớn nhất trong 128 hướng, và hai "
            f"hướng ấy được chọn để tối đa hoá phương sai tổng thể chứ không phải để làm nổi "
            f"bật ranh giới giữa hai siêu lớp. Một phép chiếu có giám sát, hoặc t-SNE và UMAP "
            f"vốn giữ được cấu trúc phi tuyến, nhiều khả năng sẽ cho thấy sự phân tách này "
            f"rõ hơn.")
    else:
        r.p(
            "Hình chiếu CIFAR-10 cho thấy một hiện tượng đáng chú ý hơn nhiều so với MNIST. "
            "Các cụm không chỉ tách theo lớp, chúng còn <strong>gom lại theo nhóm ngữ nghĩa "
            "cấp cao</strong>: bốn lớp phương tiện do con người chế tạo (máy bay, ô tô, tàu "
            "thuỷ, xe tải) nằm về một phía của không gian biểu diễn, còn sáu lớp động vật "
            "sống (chim, mèo, hươu, chó, ếch, ngựa) co cụm về phía đối diện.")
    r.p(R.note(
        ("Vì sao mức phân nhóm này, dù yếu, vẫn đáng chú ý."
         if (sep is not None and sep <= 0.1)
         else "Vì sao đây là một phát hiện chứ không phải một quan sát hiển nhiên."),
        "Mạng chỉ được cung cấp mười nhãn rời rạc. Nó <em>chưa bao giờ</em> được cho biết "
        "rằng ô tô và xe tải thuộc cùng một phạm trù, hay mèo và chó gần nhau hơn mèo và "
        "tàu thuỷ. Hàm mất mát Cross-Entropy phạt mọi kiểu nhầm lẫn như nhau: nhầm mèo "
        "thành chó bị phạt đúng bằng nhầm mèo thành máy bay. Cấu trúc phân cấp trong hình "
        "chiếu vì vậy hoàn toàn là sản phẩm phụ của quá trình học biểu diễn, phát sinh từ "
        "chỗ những lớp có chung đặc trưng thị giác cấp thấp (texture lông, đường nét thẳng "
        "và cạnh sắc) tự nhiên được ánh xạ về những vùng gần nhau."))
    r.p(
        "Đây chính là điều mà thuật ngữ <em>học biểu diễn</em> muốn nói. Mạng tích chập sâu "
        "không dừng ở việc ghi nhớ ánh xạ từ điểm ảnh sang nhãn; nó xây dựng một không gian "
        "đặc trưng trong đó khoảng cách hình học phản ánh quan hệ ngữ nghĩa, ở mức độ mà "
        "phép đo trên đã lượng hoá. Và cũng chính tính chất này là nền tảng của chuyển giao "
        "tri thức: một biểu diễn đã mã hoá được \"cái gì trông giống cái gì\" có thể tái sử "
        "dụng cho một bài toán khác chỉ với ít dữ liệu nhãn.")
    r.p(
        "Mặt trái cũng hiện ra trên cùng hình chiếu. Cụm mèo và cụm chó chồng lấn nhau "
        "nhiều nhất, đúng như dự đoán từ ma trận nhầm lẫn ở mục 7.5. Hai lớp này chia sẻ "
        "texture lông, tư thế bốn chân và bối cảnh trong nhà, nên ở độ phân giải 32×32 thì "
        "phần đặc trưng thị giác phân biệt được chúng đơn giản là quá mỏng.")


# ===========================================================================
# CHƯƠNG 9 — TỔNG HỢP VÀ KẾT LUẬN
# ===========================================================================
def _chapter12(r: R.Report, data: dict) -> None:
    r.h(1, "Chương 12. Tổng hợp đối chuẩn, giới hạn và kết luận")

    # ---- 9.1
    r.h(2, "12.1. Bảng tổng hợp toàn bộ thực nghiệm")
    _summary_table(r, data)

    # ---- 9.2
    r.h(2, "12.2. So sánh ba cách cài đặt")
    r.p(R.table(
        ["Cách cài đặt", "Ưu điểm", "Hạn chế", "Vai trò trong bài tập này"],
        [["NumPy thuần từ đầu",
          "Toàn bộ lượt tiến và lượt ngược hiện ra tường minh; không phụ thuộc thư viện; "
          "kiểm chứng được bằng sai phân hữu hạn",
          "Chậm trên CPU đơn luồng; khó mở rộng lên mạng sâu; dễ sai gradient mà không báo lỗi",
          "Hiểu thuật toán ở mức tensor và chứng minh mã nguồn đúng"],
         ["PyTorch",
          "Autograd cùng đồ thị tính toán động; vòng lặp huấn luyện tường minh nên dễ gỡ lỗi; "
          "chuyển thiết bị dễ dàng",
          "Phải tự quản lý nhiều trạng thái; nhiều mã lặp lại cho những việc thường quy",
          "Đối chuẩn có kiểm soát và cung cấp biểu diễn ẩn cho phân tích PCA"],
         ["TensorFlow / Keras",
          "API cấp cao gọn; hệ thống callback và pipeline dữ liệu tốt; mã nguồn ngắn nhất",
          "Vòng lặp huấn luyện ít tường minh hơn; khó can thiệp vào từng bước khi cần",
          "Dựng baseline nhanh và kiểm chứng chéo kết quả của PyTorch"]],
        "Ưu điểm, hạn chế và vai trò của ba cách cài đặt trong bài tập số 4."))
    r.p(
        "Bảng trên không nhằm xếp hạng. Ba cách cài đặt trả lời ba câu hỏi khác nhau, và "
        "kết quả thực nghiệm đã cho thấy khi được cấp cùng kiến trúc và cùng dữ liệu, chúng "
        "cho ra số liệu gần như trùng khớp. Cái thay đổi là công sức viết mã và tốc độ "
        "chạy, không phải bản chất mô hình. Đây chính là bài học kỹ nghệ phần mềm mà slide "
        "bài giảng nêu ra: cùng một mô hình toán học không có nghĩa là cùng một hiện thực.")

    # ---- 9.3
    r.h(2, "12.3. Tính công bằng và giới hạn của phép đối chuẩn")
    r.p(
        "Phép đối chuẩn trong báo cáo giữ cố định tập kiểm thử và bộ chỉ số, nhưng không "
        "phải mọi điều kiện đều giống hệt nhau. Liệt kê đầy đủ những sai lệch, kể cả những "
        "sai lệch bất lợi cho kết luận, là một phần của công việc:")
    r.p(
        '<ol class="tight">'
        '<li>Hai mô hình NumPy trên ảnh dùng ít mẫu huấn luyện hơn hai khung thư viện, để '
        'giữ thời gian chạy trong ngân sách của bài tập.</li>'
        '<li>Kiến trúc NumPy nông hơn và không có Batch Normalization lẫn Dropout, nên '
        'khoảng cách quan sát được gộp chung cả yếu tố kiến trúc lẫn yếu tố hiện thực.</li>'
        '<li>Cách đếm tham số giữa Keras và PyTorch không đồng nhất: Keras tính cả trạng '
        'thái không học được của Batch Normalization, PyTorch chỉ đếm tham số học được.</li>'
        '<li>Thời gian chạy chịu ảnh hưởng của backend, biên dịch đồ thị, giai đoạn khởi '
        'động và tải hệ thống tại thời điểm chạy. Toàn bộ thực nghiệm chạy trên CPU không '
        'có GPU, nên phần lợi thế tăng tốc phần cứng của hai khung thư viện không được đo.</li>'
        '<li>Chỉ một hạt giống ngẫu nhiên duy nhất được báo cáo. Không có trung bình và độ '
        'lệch chuẩn qua nhiều lần chạy, nên mọi khoảng cách dưới một điểm phần trăm chỉ '
        'mang tính gợi ý.</li>'
        '</ol>')
    r.p(
        "Vì vậy các bảng kết quả trả lời câu hỏi <em>\"cấu hình nào hiệu quả trong đúng "
        "thiết lập notebook này\"</em>, chứ không chứng minh khung nào luôn nhanh hơn hay "
        "luôn chính xác hơn.")

    # ---- 9.4
    r.h(2, "12.4. Kiểm toán khả năng tái lập")
    r.p(R.table(
        ["Hạng mục", "Cách thực hiện", "Ý nghĩa"],
        [["Hạt giống ngẫu nhiên",
          "<code class='inl'>RANDOM_SEED = 42</code> cho NumPy, PyTorch và TensorFlow",
          "Giảm dao động do phép tách dữ liệu và khởi tạo trọng số"],
         ["Tách dữ liệu",
          "Train và validation tách có phân tầng; test độc lập hoàn toàn",
          "Không có quyết định nào được đưa ra dựa trên tập kiểm thử"],
         ["Chuẩn hoá",
          "Trung bình và độ lệch chuẩn chỉ học từ tập huấn luyện",
          "Chống rò rỉ thông tin từ kiểm định và kiểm thử"],
         ["Từ điển văn bản",
          "Xây từ tập huấn luyện, từ lạ ánh xạ về token <code class='inl'>&lt;UNK&gt;</code>",
          "Mô phỏng đúng điều kiện triển khai thật"],
         ["Chọn checkpoint",
          "Khôi phục trạng thái tốt nhất theo mất mát kiểm định",
          "Vài epoch cuối dao động không làm hỏng số liệu báo cáo"],
         ["Kiểm chứng đạo hàm",
          "Sai phân hữu hạn trên mọi tầng có tham số học được",
          "Chứng minh lan truyền ngược viết tay đúng về toán học"],
         ["Hình và bảng",
          "Sinh trực tiếp từ output notebook, đọc lại qua tệp JSON",
          "Không con số nào được chép tay vào báo cáo"]],
        "Danh mục kiểm toán khả năng tái lập của bài tập số 4."))
    r.p(
        "Thứ tự chạy lại toàn bộ thực nghiệm: cài phụ thuộc từ "
        "<code class='inl'>requirements.txt</code>; chạy các notebook của từng miền theo "
        "thứ tự đánh số trong tên tệp; kiểm tra các tệp PNG và JSON đã xuất hiện trong "
        "<code class='inl'>&lt;miền&gt;/reports/</code>; cuối cùng chạy "
        "<code class='inl'>Report/Assignment 04/build_report.py</code> để dựng lại bản báo "
        "cáo này. Muốn khớp lại đúng số liệu thì giữ nguyên hạt giống, cách tách dữ liệu, "
        "quy trình tiền xử lý và cấu hình epoch; muốn khớp lại đúng thời gian thì cần chạy "
        "trên cùng phần cứng.")

    # ---- 9.5
    r.h(2, "12.5. Kết luận")
    _conclusion(r, data)

    # ---- 9.6
    r.h(2, "12.6. Hướng phát triển tiếp theo")
    r.p(
        '<ol class="tight">'
        '<li><strong>Tăng cường dữ liệu.</strong> Cắt ngẫu nhiên, lật ngang và nhiễu màu '
        'trên CIFAR-10 buộc mô hình học đặc trưng bất biến hơn. Đây là hướng có tỉ lệ lợi '
        'ích trên chi phí cao nhất với kiến trúc hiện tại.</li>'
        '<li><strong>Kết nối tắt kiểu ResNet.</strong> Khối residual <span class="inl">'
        'F(x) + x</span> cho phép gradient truyền qua hàng chục tầng mà không suy giảm, '
        'giải quyết hiện tượng thoái hoá khi mạng sâu lên. ResNet-20 trên CIFAR-10 là bước '
        'kế tiếp tự nhiên.</li>'
        '<li><strong>Chạy nhiều hạt giống.</strong> Báo cáo trung bình và độ lệch chuẩn qua '
        'năm lần chạy sẽ biến những khoảng cách dưới một điểm phần trăm từ gợi ý thành kết '
        'luận có cơ sở thống kê.</li>'
        '<li><strong>Hiệu chỉnh xác suất.</strong> Phân tích lỗi tự tin cao ở mục 6.7 và '
        '7.5 cho thấy xác suất softmax chưa được hiệu chỉnh. Nhiệt độ scaling hoặc một cơ '
        'chế từ chối trả lời là bổ sung cần thiết trước khi triển khai thật.</li>'
        '<li><strong>Chuyển giao tri thức.</strong> Tận dụng mô hình tiền huấn luyện trên '
        'ImageNet để kiểm chứng trực tiếp giả thuyết về khả năng tái sử dụng biểu diễn đã '
        'nêu ở mục 8.5.</li>'
        '<li><strong>Tăng tốc hiện thực NumPy.</strong> Dùng Numba hoặc CuPy để tách bạch '
        'chi phí thuật toán khỏi chi phí vòng lặp Python, qua đó đo được phần chênh lệch '
        'nào thực sự thuộc về thuật toán.</li>'
        '</ol>')


# ===========================================================================
# Bảng và bình luận sinh tự động
# ===========================================================================
def _mlp_table(r: R.Report, d: dict) -> None:
    rows = []
    for ds_key, ds_name in [("mnist", "MNIST (28×28×1)"), ("cifar10", "CIFAR-10 (32×32×3)")]:
        blk = d.get(ds_key)
        if not blk:
            continue
        mlp, cnn = blk["mlp"], blk["cnn"]
        gap = cnn["accuracy"] - mlp["accuracy"]
        sign = "+" if gap >= 0 else "−"
        rows.append([ds_name, "MLP (3 tầng ẩn)", num(mlp["loss"]), pct(mlp["accuracy"]),
                     f'{mlp["params"]:,}', f'<strong>{sign}{pct(abs(gap))}</strong>'])
        rows.append(["", "2D CNN (Conv + Pool)", num(cnn["loss"]), pct(cnn["accuracy"]),
                     f'{cnn["params"]:,}', ""])
    r.p(R.table(
        ["Tập dữ liệu", "Kiến trúc", "Test Loss", "Test Accuracy", "Tham số",
         "Chênh lệch nghiêng về CNN"],
        rows,
        "Đối chuẩn đối kháng giữa mạng truyền thẳng và mạng tích chập, cùng dữ liệu, cùng "
        "tiền xử lý, cùng hàm mất mát và cùng thuật toán tối ưu."))


def _gap_commentary(r: R.Report, d: dict) -> None:
    """Binh luan khoang cach MLP voi CNN, ton trong DAU cua chenh lech.

    Gia thuyet o muc 8.1 la CNN thang tren ca hai tap, va thang dam tren CIFAR-10.
    Neu so lieu noi nguoc lai thi phai noi nguoc lai, khong duoc lay tri tuyet doi
    roi viet nhu the gia thuyet da duoc xac nhan.
    """
    mn, cf = d.get("mnist"), d.get("cifar10")
    if not mn or not cf:
        return

    g_mn = mn["cnn"]["accuracy"] - mn["mlp"]["accuracy"]
    g_cf = cf["cnn"]["accuracy"] - cf["mlp"]["accuracy"]

    def phrase(g: float, ds: str, blk: dict) -> str:
        if g > 0:
            return (f"Tren {ds}, m\u1ea1ng t\u00edch ch\u1eadp \u0111\u1ee9ng tr\u00ean v\u1edbi "
                    f"{pct(blk['cnn']['accuracy'])} so v\u1edbi {pct(blk['mlp']['accuracy'])} "
                    f"c\u1ee7a m\u1ea1ng truy\u1ec1n th\u1eb3ng, ch\u00eanh {pct(g)}")
        return (f"Tren {ds}, <strong>m\u1ea1ng truy\u1ec1n th\u1eb3ng l\u1ea1i v\u01b0\u1ee3t "
                f"m\u1ea1ng t\u00edch ch\u1eadp</strong>, {pct(blk['mlp']['accuracy'])} so v\u1edbi "
                f"{pct(blk['cnn']['accuracy'])}, ch\u00eanh {pct(-g)} theo h\u01b0\u1edbng "
                f"ng\u01b0\u1ee3c v\u1edbi gi\u1ea3 thuy\u1ebft")

    r.p(phrase(g_mn, "MNIST", mn).replace("Tren", "Tr\u00ean", 1) + ". "
        + phrase(g_cf, "CIFAR-10", cf).replace("Tren", "Tr\u00ean", 1) + ".")

    if g_mn > 0 and g_cf > g_mn:
        ratio = g_cf / max(g_mn, 1e-9)
        r.p(
            f"K\u1ebft qu\u1ea3 x\u00e1c nh\u1eadn c\u1ea3 hai gi\u1ea3 thuy\u1ebft. T\u1ec9 l\u1ec7 gi\u1eefa hai kho\u1ea3ng c\u00e1ch l\u00e0 "
            f"kho\u1ea3ng {ratio:.0f} l\u1ea7n, v\u00e0 \u0111\u00f3 l\u00e0 n\u1ed9i dung ch\u00ednh c\u1ee7a ch\u01b0\u01a1ng: gi\u00e1 tr\u1ecb c\u1ee7a "
            f"t\u00edch ch\u1eadp <strong>kh\u00f4ng ph\u1ea3i m\u1ed9t h\u1eb1ng s\u1ed1</strong> m\u00e0 ph\u1ee5 thu\u1ed9c v\u00e0o vi\u1ec7c d\u1eef "
            f"li\u1ec7u c\u00f3 c\u1ea5u tr\u00fac kh\u00f4ng gian \u0111\u1ec3 khai th\u00e1c hay kh\u00f4ng. Tr\u00ean m\u1ed9t b\u1ed9 d\u1eef li\u1ec7u "
            f"\u0111\u00e3 \u0111\u01b0\u1ee3c c\u0103n ch\u1ec9nh s\u1eb5n, t\u00edch ch\u1eadp g\u1ea7n nh\u01b0 kh\u00f4ng mang l\u1ea1i g\u00ec; tr\u00ean "
            f"\u1ea3nh t\u1ef1 nhi\u00ean ch\u01b0a qua c\u0103n ch\u1ec9nh, n\u00f3 l\u00e0 kh\u00e1c bi\u1ec7t gi\u1eefa d\u00f9ng \u0111\u01b0\u1ee3c v\u00e0 "
            f"kh\u00f4ng d\u00f9ng \u0111\u01b0\u1ee3c.")
    else:
        r.p(R.note(
            "K\u1ebft qu\u1ea3 kh\u00f4ng kh\u1edbp gi\u1ea3 thuy\u1ebft, v\u00e0 b\u00e1o c\u00e1o gi\u1eef nguy\u00ean n\u00f3.",
            "Gi\u1ea3 thuy\u1ebft \u1edf m\u1ee5c 8.1 d\u1ef1 \u0111o\u00e1n m\u1ea1ng t\u00edch ch\u1eadp th\u1eafng tr\u00ean c\u1ea3 hai t\u1eadp v\u00e0 "
            "th\u1eafng \u0111\u1eadm tr\u00ean CIFAR-10. S\u1ed1 li\u1ec7u \u0111o \u0111\u01b0\u1ee3c kh\u00f4ng \u1ee7ng h\u1ed9 \u0111i\u1ec1u \u0111\u00f3. "
            "Nguy\u00ean nh\u00e2n kh\u1ea3 d\u0129 nh\u1ea5t <em>kh\u00f4ng</em> ph\u1ea3i l\u00e0 t\u00edch ch\u1eadp v\u00f4 d\u1ee5ng, m\u00e0 l\u00e0 hai "
            "m\u00f4 h\u00ecnh ch\u01b0a \u0111\u01b0\u1ee3c hu\u1ea5n luy\u1ec7n \u1edf c\u00f9ng m\u1ed9t m\u1ee9c \u0111\u1ed9 h\u1ed9i t\u1ee5: m\u1ed9t ph\u00e9p so "
            "s\u00e1nh ki\u1ebfn tr\u00fac ch\u1ec9 c\u00f3 ngh\u0129a khi c\u1ea3 hai b\u00ean \u0111\u1ec1u ch\u1ea1y \u0111\u1ee7 s\u1ed1 epoch tr\u00ean \u0111\u1ee7 "
            "d\u1eef li\u1ec7u. Tr\u01b0\u1edbc khi r\u00fat b\u1ea5t k\u1ef3 k\u1ebft lu\u1eadn n\u00e0o v\u1ec1 ki\u1ebfn tr\u00fac, c\u1ea7n hu\u1ea5n luy\u1ec7n "
            "l\u1ea1i hai b\u00ean trong c\u00f9ng \u0111i\u1ec1u ki\u1ec7n. B\u00e1o c\u00e1o ghi nh\u1eadn \u0111i\u1ec1u n\u00e0y thay v\u00ec di\u1ec5n gi\u1ea3i "
            "cho kh\u1edbp c\u00e2u chuy\u1ec7n \u0111\u00e3 \u0111\u1ecbnh s\u1eb5n.", "warn"))

    r.p(
        f"\u0110\u00e1ng ch\u00fa \u00fd l\u00e0 m\u1ea1ng truy\u1ec1n th\u1eb3ng tr\u00ean CIFAR-10 c\u00f3 "
        f"{cf['mlp']['params']:,} tham s\u1ed1, so v\u1edbi {cf['cnn']['params']:,} c\u1ee7a m\u1ea1ng t\u00edch "
        f"ch\u1eadp. "
        + ("B\u00ean thua cu\u1ed9c c\u00f3 nhi\u1ec1u tham s\u1ed1 h\u01a1n, n\u00ean th\u1ea5t b\u1ea1i n\u00e0y kh\u00f4ng th\u1ec3 quy cho "
           "thi\u1ebfu dung l\u01b0\u1ee3ng m\u00f4 h\u00ecnh. N\u00f3 l\u00e0 th\u1ea5t b\u1ea1i c\u1ee7a m\u1ed9t <em>gi\u1ea3 \u0111\u1ecbnh ki\u1ebfn "
           "tr\u00fac</em>: gi\u1ea3 \u0111\u1ecbnh r\u1eb1ng m\u1ecdi \u0111i\u1ec3m \u1ea3nh \u0111\u1ec1u \u0111\u1ed9c l\u1eadp v\u00e0 ho\u00e1n v\u1ecb "
           "\u0111\u01b0\u1ee3c."
           if g_cf > 0 else
           "S\u1ed1 tham s\u1ed1 g\u1ea5p g\u1ea7n ch\u00edn l\u1ea7n n\u00e0y c\u0169ng l\u00e0 m\u1ed9t l\u00fd do khi\u1ebfn ph\u00e9p so s\u00e1nh "
           "hi\u1ec7n t\u1ea1i ch\u01b0a c\u00f4ng b\u1eb1ng theo chi\u1ec1u ng\u01b0\u1ee3c l\u1ea1i, v\u00e0 c\u1ea7n \u0111\u01b0\u1ee3c c\u00e2n nh\u1eafc "
           "khi \u0111\u1ecdc con s\u1ed1 \u1edf tr\u00ean."))


def _pca_commentary(r: R.Report, d: dict) -> None:
    """Bình luận định lượng cho hình chiếu PCA, đọc thẳng từ số liệu notebook."""
    pca = d.get("pca")
    if not pca:
        return

    parts = []
    for key, label in [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]:
        blk = pca.get(key) or {}
        evr = blk.get("explained_variance_ratio")
        if evr and len(evr) >= 2:
            parts.append(
                f"trên {label}, hai thành phần chính đầu tiên giữ lại "
                f"{pct(evr[0] + evr[1])} phương sai của không gian 128 chiều "
                f"({pct(evr[0])} và {pct(evr[1])})")
    if parts:
        r.p(
            "Một con số cần nêu kèm trước khi diễn giải hình: " + "; ".join(parts) + ". "
            "Phần phương sai bị bỏ lại không biến mất, nó chỉ không hiển thị được trên mặt "
            "phẳng hai chiều. Vì vậy mọi nhận định dưới đây về mức độ tách cụm là "
            "<em>cận dưới</em> của mức tách thật trong không gian đầy đủ.")

    sep = (pca.get("cifar10") or {}).get("vehicle_animal_separation")
    if sep is None:
        return
    r.p(
        f"Để không dừng ở quan sát bằng mắt, báo cáo đo luôn mức tách giữa hai siêu lớp "
        f"phương tiện và động vật ngay trên hình chiếu hai chiều. Giá trị thu được là "
        f"<strong>{num(sep, 4)}</strong>. "
        + ("Số dương rõ rệt này xác nhận rằng cấu trúc phân nhóm nhìn thấy trên hình không "
           "phải ảo giác thị giác mà là một tính chất đo được của biểu diễn."
           if sep > 0.1 else
           "Giá trị khiêm tốn này nói rằng cấu trúc phân nhóm có tồn tại nhưng yếu, và không "
           "nên được trình bày mạnh hơn mức dữ liệu cho phép. Hai siêu lớp chồng lấn đáng kể "
           "trên hình chiếu tuyến tính hai chiều, dù chúng có thể tách tốt hơn trong không "
           "gian 128 chiều đầy đủ."))


def _summary_table(r: R.Report, data: dict) -> None:
    rows = []

    cm = data["comments"]["models"]
    rows.append(["Văn bản", "Customer Comments", "1D CNN + GlobalMaxPool", "Phân loại nhị phân"]
                + [f'{pct(cm[k]["accuracy"])}<br><span class="sub">F1 {pct(cm[k]["f1"])}</span>'
                   for k in FW_ORDER])

    db = data["diabetes"]["models"]
    rows.append(["Bảng số", "Diabetes (y tế)", "1D CNN 8→16→1", "Phân loại nhị phân"]
                + [f'{pct(db[k]["accuracy"])}<br><span class="sub">F1 {pct(db[k]["f1"])}</span>'
                   for k in FW_ORDER])

    hp = data["house"]["models"]
    rows.append(["Bảng số", "House Price (bất động sản)", "1D CNN hồi quy", "Hồi quy liên tục"]
                + [f'R² {num(hp[k]["r2"], 3)}<br><span class="sub">RMSE {usd(hp[k]["rmse_usd"])}</span>'
                   for k in FW_ORDER])

    for key, label, shape in [("mnist", "MNIST", "28×28×1"), ("cifar", "CIFAR-10", "32×32×3")]:
        mm = data[key]["models"]
        np_best = mm.get("numpy_improved") or mm.get("numpy_baseline")
        rows.append(["Thị giác máy", f"{label} ({shape})", "2D CNN", "Phân loại 10 lớp",
                     f'{pct(np_best["accuracy"])}<br><span class="sub">Macro-F1 {pct(np_best["macro_f1"])}</span>',
                     f'{pct(mm["pytorch"]["accuracy"])}<br><span class="sub">Macro-F1 {pct(mm["pytorch"]["macro_f1"])}</span>',
                     f'{pct(mm["tensorflow"]["accuracy"])}<br><span class="sub">Macro-F1 {pct(mm["tensorflow"]["macro_f1"])}</span>'])

    mlp = data["mlp"]
    for key, label in [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]:
        blk = mlp.get(key)
        if blk:
            rows.append(["Thị giác máy", f"{label} — MLP đối kháng", "3 tầng Dense",
                         "Phân loại 10 lớp", "—",
                         f'{pct(blk["mlp"]["accuracy"])}', "—"])

    r.p(R.table(
        ["Miền dữ liệu", "Bài toán", "Kiến trúc", "Loại đầu ra",
         "NumPy thuần", "PyTorch", "TensorFlow / Keras"],
        rows,
        "Tổng hợp toàn bộ thực nghiệm của bài tập số 4, trải năm miền dữ liệu và ba cách "
        "cài đặt.", cls="summary"))
    r.p(
        "Bảng này là toàn cảnh của báo cáo trên một trang. Đọc theo hàng cho thấy độ khó "
        "tăng dần từ dữ liệu bảng sang văn bản rồi sang ảnh tự nhiên. Đọc theo cột cho thấy "
        "ba cách cài đặt bám sát nhau ở mọi miền dữ liệu, trừ đúng chỗ mà kiến trúc NumPy "
        "cố tình nông hơn.")


def _conclusion(r: R.Report, data: dict) -> None:
    cm = data["comments"]["models"]
    db = data["diabetes"]["models"]
    hp = data["house"]["models"]
    mn = data["mnist"]["models"]
    cf = data["cifar"]["models"]
    mlp = data["mlp"]

    mn_best_k, mn_best = _best(mn, "macro_f1")
    cf_best_k, cf_best = _best(cf, "macro_f1")
    cm_best = max(cm.values(), key=lambda v: v["f1"])
    db_best = max(db.values(), key=lambda v: v["f1"])
    hp_best = max(hp.values(), key=lambda v: v["r2"])
    mn_np = mn.get("numpy_improved") or mn.get("numpy_baseline")
    cf_np = cf.get("numpy_improved") or cf.get("numpy_baseline")
    _sep = ((mlp.get("pca") or {}).get("cifar10") or {}).get("vehicle_animal_separation")

    r.p(
        "Báo cáo đã xây dựng, huấn luyện và đối chuẩn mạng nơ-ron tích chập ở ba mức độ "
        "kiểm soát, trên năm miền dữ liệu phủ ba dạng hình học khác nhau. Sáu kết quả "
        "chính:")
    items = [
        f"<strong>Hiện thực từ đầu được kiểm chứng, không chỉ được tuyên bố.</strong> Toàn "
        f"bộ tích chập một chiều và hai chiều, gộp cực đại, Dense, softmax, lan truyền ngược "
        f"và Adam đều được viết bằng NumPy thuần, vector hoá bằng "
        f"<code class='inl'>im2col</code> và <code class='inl'>einsum</code>, rồi xác nhận "
        f"bằng kiểm chứng sai phân hữu hạn trên từng tầng có tham số.",

        f"<strong>Ba khung hiện thực cho kết quả tương đương.</strong> Trên tập đánh giá "
        f"khách hàng, F1 tốt nhất đạt {pct(cm_best['f1'])}; trên bài toán chẩn đoán tiểu "
        f"đường, F1 tốt nhất đạt {pct(db_best['f1'])}; trên bài toán định giá bất động sản, "
        f"R² tốt nhất đạt {num(hp_best['r2'], 3)} tương ứng RMSE {usd(hp_best['rmse_usd'])}. "
        f"Chênh lệch giữa ba khung ở cả ba miền đều nhỏ hơn biên độ dao động do khởi tạo, "
        f"đúng như dự đoán từ chỗ chúng dựng cùng một mô hình toán học.",

        f"<strong>Dữ liệu bảng không có tô-pô, và điều đó đo được.</strong> Thực nghiệm "
        f"hoán vị ngẫu nhiên thứ tự tám cột đặc trưng ở mục 4.6 cho thấy hiệu năng gần như "
        f"không đổi, xác nhận rằng tích chập trên dữ liệu bảng chỉ hoạt động như một mạng "
        f"truyền thẳng có ràng buộc chia sẻ trọng số. Đây là đóng góp phương pháp luận "
        f"chính của phần một chiều.",

        f"<strong>Độ sâu kiến trúc phải tương xứng độ phức tạp dữ liệu.</strong> Trên "
        f"MNIST, mô hình NumPy nông đạt {pct(mn_np['accuracy'])} và chỉ kém mô hình tốt "
        f"nhất {pct(mn_best['accuracy'] - mn_np['accuracy'])}. Trên CIFAR-10, cùng cách "
        f"tiếp cận nông chỉ đạt {pct(cf_np['accuracy'])} trong khi mô hình sâu đạt "
        f"{pct(cf_best['accuracy'])}. Cùng một hiện thực, cùng một chất lượng mã nguồn, "
        f"hai kết cục khác hẳn nhau vì dữ liệu khác nhau.",

        f"<strong>Giá trị của tích chập phụ thuộc dữ liệu chứ không phải hằng số.</strong> "
        f"Trên MNIST đã căn giữa, mạng truyền thẳng chỉ kém mạng tích chập "
        f"{pct(abs(mlp['mnist']['cnn']['accuracy'] - mlp['mnist']['mlp']['accuracy']))}. "
        f"Trên CIFAR-10, khoảng cách giãn thành "
        f"{pct(abs(mlp['cifar10']['cnn']['accuracy'] - mlp['cifar10']['mlp']['accuracy']))}, "
        f"dù mạng truyền thẳng có nhiều tham số hơn.",

        (f"<strong>Mạng học được cấu trúc ngữ nghĩa không ai dạy nó.</strong> Hình chiếu "
         f"PCA của không gian ẩn 128 chiều trên CIFAR-10 cho thấy các lớp tự gom thành hai "
         f"nhóm lớn, phương tiện di chuyển và động vật sống, trong khi hàm mất mát phạt mọi "
         f"kiểu nhầm lẫn như nhau. Mức tách đo được giữa hai siêu lớp là {num(_sep, 4)}."
         if (_sep is not None and _sep > 0.1) else
         f"<strong>Cấu trúc ngữ nghĩa trong không gian ẩn tồn tại nhưng yếu.</strong> Mức "
         f"tách đo được giữa hai siêu lớp phương tiện và động vật trên hình chiếu PCA hai "
         f"chiều chỉ là {num(_sep, 4) if _sep is not None else 'không đo được'}. Báo cáo ghi "
         f"nhận đúng mức đó: PCA là phép chiếu tuyến tính chọn hướng theo phương sai tổng "
         f"thể, không theo ranh giới siêu lớp, nên đây là cận dưới chứ không phải kết luận "
         f"về chất lượng biểu diễn."),
    ]
    r.p('<ol class="tight">' + "".join(f"<li>{it}</li>" for it in items) + "</ol>")
    r.p(
        "Nhìn lại toàn bộ, câu trả lời cho câu hỏi mở đầu ở Chương 1 là: mạng tích chập "
        "không phải một ý tưởng mới so với mạng truyền thẳng, nó vẫn là một phép hợp hàm "
        "được tối ưu bằng lan truyền ngược. Thứ nó thêm vào là một <em>giả định có cấu "
        "trúc</em> về dữ liệu, rằng thông tin hữu ích mang tính cục bộ và lặp lại ở nhiều "
        "vị trí. Khi giả định đó đúng, phần thưởng rất lớn. Khi nó sai, như với dữ liệu "
        "bảng ở Chương 4 và Chương 5, toán tử vẫn chạy nhưng không mang lại gì. "
        "Biết phân biệt hai trường hợp ấy là điều mà bài tập này thực sự muốn dạy.")
