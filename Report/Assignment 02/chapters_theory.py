"""Chương I–VI: phần lý thuyết chung của báo cáo Assignment 02."""

from build_report import (APPS, DATASETS, GITHUB_URL, META, code, figure, fmt,
                          note, notebook_code, table)

D = META["diabetes"]


def write(r) -> None:
    _ch1(r)
    _ch2(r)
    _ch3(r)
    _ch4(r)
    _ch5(r)
    _ch6(r)


# ---------------------------------------------------------------- Chương I
def _ch1(r) -> None:
    r.h(1, "Mã nguồn dự án và nguồn dữ liệu", nobreak=True)
    r.p("Toàn bộ mã nguồn, notebook, mô hình đã huấn luyện và bản báo cáo này được "
        "công bố công khai tại kho mã nguồn dưới đây.")
    r.p(f'<div class="srcline"><b>Kho mã nguồn:</b> '
        f'<a href="{GITHUB_URL}">{GITHUB_URL}</a></div>')
    r.p("Ba tập dữ liệu gốc được tải từ Kaggle và giữ nguyên trong thư mục "
        "<code class=\"inl\">&lt;ứng-dụng&gt;/data/</code> của kho mã nguồn:")
    rows = [(lab, f'{name}<br><a href="{url}">{url}</a>') for lab, name, url in DATASETS]
    r.p('<table class="links"><tbody>'
        + "".join(f'<tr><td class="lab">{a}</td><td>{b}</td></tr>' for a, b in rows)
        + '</tbody></table>')

    r.h(1, "Chương I. Giới thiệu bài toán")

    r.h(2, "1.1. Bối cảnh")
    r.p(
        "Dữ liệu thực tế được sinh ra dưới nhiều hình thức rất khác nhau: bảng số "
        "liệu, giao dịch, văn bản tự do, ảnh, âm thanh. Nhưng hầu hết mô hình học "
        "máy chỉ nhận được một thứ duy nhất — một mảng số có kích thước xác định. "
        "Khoảng cách giữa hai điều đó là nơi phần lớn công sức của một hệ thống "
        "thông minh thực sự được tiêu tốn.",
        "Assignment 02 với chủ đề <em>From Data Representation to a Deployable "
        "Intelligent System</em> yêu cầu đi trọn con đường ấy, chứ không dừng ở "
        "việc huấn luyện một mô hình:",
        '<div class="flow">Dữ liệu thô → Tìm hiểu → Làm sạch → Biểu diễn → Học '
        "→ Đánh giá → Lưu trữ → Triển khai</div>",
        "Điểm nhấn nằm ở chỗ chất lượng của một hệ thống thông minh không do thuật "
        "toán quyết định một mình. Nó phụ thuộc vào chất lượng dữ liệu, vào cách "
        "dữ liệu được biểu diễn thành số, và vào việc phép đánh giá có trung thực "
        "hay không. Báo cáo này sẽ đưa ra bằng chứng định lượng cho cả ba mệnh đề "
        "đó, ở ba ứng dụng khác nhau.",
        "Ngoài phần xây dựng mô hình, hệ thống sau khi huấn luyện phải dùng lại "
        "được mà không cần huấn luyện lại. Vì vậy mô hình và quy trình tiền xử lý "
        "đều được lưu trữ và nạp lại lúc dự đoán. Cả ba ứng dụng đều được triển "
        "khai thành một dịch vụ Web kèm giao diện Mobile.",
    )

    r.h(2, "1.2. Mục tiêu của bài tập")
    r.p("Assignment 02 hướng tới bảy mục tiêu, và báo cáo này bám theo đúng thứ tự ấy.")
    r.p(table(
        ["#", "Mục tiêu", "Được chứng minh ở đâu trong báo cáo"],
        [
            ["1", "Hiểu và phân tích dữ liệu thực tế",
             "Mục 7.3, 8.3, 9.3 — khảo sát cấu trúc và chất lượng từng tập"],
            ["2", "Xây dựng biểu diễn dữ liệu phù hợp cho mô hình",
             "Chương II, và mục 7.5, 8.5, 9.5"],
            ["3", "Tiền xử lý dữ liệu một cách có hệ thống",
             "Chương III, và mục 7.4, 8.4, 9.4"],
            ["4", "Xây dựng và so sánh nhiều mô hình học máy",
             "Chương V; 5 + 5 + 6 mô hình ở mục 7.7, 8.7, 9.7"],
            ["5", "Đánh giá và lựa chọn mô hình phù hợp",
             "Chương IV, và mục 7.8–7.9, 8.8–8.9, 9.8–9.9"],
            ["6", "Lưu trữ mô hình và quy trình tiền xử lý",
             "Chương VI, mục 6.2"],
            ["7", "Triển khai thành hệ thống có khả năng sử dụng",
             "Mục 7.10, 8.10, 9.11 — REST API + Web + Mobile"],
        ]))
    r.p(
        "Mục tiêu thứ hai là mục tiêu trung tâm. Ba ứng dụng được chọn có chủ đích "
        "để chúng đòi hỏi ba kiểu biểu diễn khác nhau: dữ liệu bảng thuần số, dữ "
        "liệu bảng có biến phân loại, và dữ liệu có văn bản. Nhờ vậy Chương X mới "
        "so sánh được chúng một cách có ý nghĩa.",
    )

    r.h(2, "1.3. Quy trình phát triển hệ thống thông minh")
    r.p("Cả ba ứng dụng đều đi theo đúng một quy trình tám bước.")
    r.p(figure("fig_pipeline.png",
               "Quy trình phát triển hệ thống thông minh áp dụng cho cả ba ứng dụng."))
    r.p(table(
        ["Bước", "Nội dung", "Kết quả tạo ra"],
        [
            ["Dữ liệu thô", "Thu thập bộ dữ liệu thực tế từ Kaggle", "Tệp CSV trong <code class='inl'>data/</code>"],
            ["Tìm hiểu", "Kiểm tra cấu trúc, kiểu dữ liệu, chất lượng", "Bảng khảo sát, biểu đồ phân bố"],
            ["Làm sạch", "Xử lý thiếu, trùng lặp, giá trị không hợp lệ, ngoại lệ", "Bảng dữ liệu sạch"],
            ["Biểu diễn", "Chuyển thành vectơ, ma trận hoặc tensor", "Ma trận đặc trưng <em>X</em>"],
            ["Học", "Huấn luyện nhiều mô hình trên cùng một cách chia dữ liệu", "Các mô hình đã huấn luyện"],
            ["Đánh giá", "Đo bằng độ đo phù hợp với loại bài toán", "Bảng so sánh mô hình"],
            ["Lưu trữ", "Lưu pipeline tiền xử lý và mô hình được chọn", "Tệp <code class='inl'>.joblib</code>"],
            ["Triển khai", "REST API + giao diện Web + giao diện Mobile", "Dịch vụ chạy được"],
        ]))
    r.p(note(
        "Điểm mấu chốt.",
        "Bốn bước đầu — trước khi bất kỳ mô hình nào được huấn luyện — quyết định "
        "trần hiệu năng của toàn hệ thống. Chương X sẽ chỉ ra rằng ở cả ba ứng dụng, "
        "thay đổi cách biểu diễn dữ liệu tạo ra khác biệt lớn hơn nhiều so với thay "
        "đổi thuật toán."))

    r.h(2, "1.4. Tổng quan về ba ứng dụng")
    d, h, e = META["diabetes"], META["house_price"], META["customer_behavior"]
    r.p(table(
        ["", "Ứng dụng 1", "Ứng dụng 2", "Ứng dụng 3"],
        [
            ["Tên", "Dự đoán bệnh tiểu đường", "Dự đoán giá nhà",
             "Hành vi và sở thích khách hàng"],
            ["Loại bài toán", "Phân loại nhị phân", "Hồi quy", "Phân loại nhị phân"],
            ["Dữ liệu thô", "CSV — bảng thuần số", "CSV — bảng có biến phân loại",
             "CSV — bảng + văn bản đánh giá"],
            ["Một quan sát", "Một lần khám bệnh nhân", "Một tin rao bán bất động sản",
             "Một lượt đánh giá sản phẩm"],
            ["Số mẫu", f"{d['n_samples']:,}",
             f"{h['n_samples_raw']:,} → {h['n_samples_clean']:,} sau làm sạch",
             f"{e['n_samples_raw']:,} → {e['n_samples_clean']:,} sau làm sạch"],
            ["Biến mục tiêu", "<code class='inl'>Outcome</code>",
             "<code class='inl'>Price</code> (tỷ VNĐ)",
             "<code class='inl'>Recommended IND</code>"],
            ["Số mô hình so sánh", "5", "5", "6"],
            ["Mô hình được chọn", d["best_model_label"], h["best_model_label"],
             e["best_model_label"]],
            ["Độ đo chính", "Recall", "MAE", "F1 lớp thiểu số / ROC-AUC"],
        ]))

    r.h(3, "1.4.1. Ứng dụng dự đoán bệnh tiểu đường")
    r.p(
        "Dự đoán một bệnh nhân có thuộc lớp dương tính tiểu đường hay không, từ các "
        "chỉ số lâm sàng thu được trong một lần khám. Toàn bộ tám đặc trưng đều là "
        "số, nên đây là biểu diễn dữ liệu đơn giản nhất trong ba ứng dụng: không "
        "cần mã hoá, chỉ cần điền khuyết và chuẩn hoá thang đo.",
        "Thách thức thật của ứng dụng này không nằm ở mô hình mà nằm ở chất lượng "
        "dữ liệu: giá trị thiếu được mã hoá bằng số 0, nên mọi hàm kiểm tra tiêu "
        "chuẩn đều báo dữ liệu sạch trong khi gần một nửa một cột là dữ liệu bịa.",
    )

    r.h(3, "1.4.2. Ứng dụng dự đoán giá nhà")
    r.p(
        "Ước lượng giá rao bán của một bất động sản nhà ở tại Việt Nam. Khác ứng "
        "dụng 1 ở hai điểm: đầu ra là số thực liên tục nên bộ độ đo phải đổi hoàn "
        "toàn, và dữ liệu có biến phân loại nên bắt buộc phải mã hoá one-hot.",
        "Ứng dụng này còn có một cột địa chỉ ở dạng văn bản tự do. Một phép rút "
        "chuỗi đơn giản biến nó thành hai biến vị trí có sức dự báo mạnh — minh hoạ "
        "trực tiếp cho luận điểm rằng biểu diễn dữ liệu là một phần của lời giải.",
    )

    r.h(3, "1.4.3. Ứng dụng phân tích hành vi và khám phá sở thích khách hàng thương mại điện tử")
    r.p(
        "Từ thông tin khách hàng và nội dung đánh giá họ viết, dự đoán khách hàng "
        "có khuyến nghị sản phẩm hay không. Đây là ứng dụng duy nhất có dữ liệu văn "
        "bản, nên là nơi trình bày đầy đủ chuỗi biến đổi mà Bài giảng 02 yêu cầu: "
        "bình luận → token → token ID → vectơ / embedding.",
        "Ứng dụng này cũng cho bằng chứng mạnh nhất trong cả báo cáo: cùng một cách "
        "chia dữ liệu, mô hình chỉ dùng đặc trưng dạng bảng đạt ROC-AUC khoảng "
        f"{fmt(max(v['ROC-AUC'] for k, v in e['test_metrics'].items() if e['model_repr'][k] == 'tabular'), 3)}, "
        "trong khi mô hình được đọc thêm văn bản đạt "
        f"{fmt(max(v['ROC-AUC'] for v in e['test_metrics'].values()), 3)}.",
    )


# --------------------------------------------------------------- Chương II
def _ch2(r) -> None:
    r.h(1, "Chương II. Cơ sở lý thuyết về biểu diễn dữ liệu")

    r.h(2, "2.1. Tổng quan về học máy và hệ thống thông minh")
    r.p(
        "Học máy có giám sát tìm một ánh xạ <em>f</em> từ không gian đặc trưng sang "
        "không gian mục tiêu, dựa trên một tập ví dụ đã biết đáp án:",
        '<div class="formula">f : X → y, &nbsp;&nbsp; X ∈ ℝ<sup>N×d</sup>, &nbsp; '
        "y ∈ ℝ<sup>N</sup> hoặc {0, 1}<sup>N</sup></div>",
        "Một <strong>hệ thống thông minh</strong> thì rộng hơn một mô hình. Nó gồm "
        "sáu thành phần, và thiếu bất kỳ thành phần nào thì hệ thống không dùng "
        "được: dữ liệu, biểu diễn, mô hình đã học, phép đánh giá, phần mềm triển "
        "khai, và giao diện tương tác với người dùng. Assignment 02 yêu cầu đủ cả sáu.",
    )

    r.h(2, "2.2. Bài toán phân loại và hồi quy")
    r.h(3, "2.2.1. Bài toán phân loại")
    r.p(
        "Biến mục tiêu nhận giá trị rời rạc. Với phân loại nhị phân, "
        "<em>y</em> ∈ {0, 1}. Mô hình thường trả về một xác suất "
        "<em>P(y = 1 | x)</em> rồi so với một ngưỡng (mặc định 0,5) để ra nhãn. "
        "Sai lầm được <strong>đếm</strong>: một dự đoán hoặc đúng, hoặc sai, không "
        "có mức độ ở giữa. Vì vậy công cụ đánh giá là ma trận nhầm lẫn.",
        "Ứng dụng 1 và Ứng dụng 3 thuộc loại này.",
    )
    r.h(3, "2.2.2. Bài toán hồi quy")
    r.p(
        "Biến mục tiêu nhận giá trị liên tục, <em>y</em> ∈ ℝ. Sai lầm được "
        "<strong>đo bằng khoảng cách</strong>: dự đoán 5,0 tỷ cho một căn 5,2 tỷ là "
        "gần đúng, dự đoán 9,0 tỷ là sai nặng. Không có khái niệm ma trận nhầm lẫn, "
        "không có accuracy, không có ROC-AUC.",
        "Ứng dụng 2 thuộc loại này. Sự khác biệt này là lý do Chương IV phải trình "
        "bày hai bộ độ đo tách biệt.",
    )

    r.h(2, "2.3. Khái niệm biểu diễn dữ liệu")
    r.p(
        "Bài giảng 02 giới thiệu nguyên tắc:",
        '<div class="flow">Dữ liệu thực tế → Biểu diễn số → Mô hình tính toán</div>',
        "Biểu diễn dữ liệu không đơn giản là cách dữ liệu được lưu trong một tệp. "
        "Cùng một tệp CSV có thể sinh ra nhiều biểu diễn số khác nhau, và mỗi biểu "
        "diễn dẫn tới một kết quả khác nhau. Khi xây dựng một biểu diễn, cần xác "
        "định rõ bảy yếu tố:",
        "<ul>"
        "<li><strong>Shape</strong> — kích thước dữ liệu;</li>"
        "<li><strong>Dtype</strong> — kiểu dữ liệu;</li>"
        "<li><strong>Range</strong> — miền giá trị;</li>"
        "<li><strong>Ý nghĩa của từng chiều</strong>;</li>"
        "<li><strong>Cách mã hoá</strong> biến phân loại;</li>"
        "<li><strong>Cách xử lý giá trị thiếu</strong>;</li>"
        "<li><strong>Thứ tự cột</strong> đưa vào mô hình.</li></ul>",
    )
    r.p(note(
        "Vì sao thứ tự cột thuộc về biểu diễn.",
        "Hai ma trận cùng kích thước nhưng thứ tự cột khác nhau là hai biểu diễn "
        "khác nhau. Mô hình nhận đầu vào theo <em>vị trí</em> chứ không theo tên "
        "cột — nếu dịch vụ Web dựng khung dữ liệu sai thứ tự, giá trị BMI sẽ được "
        "chuẩn hoá bằng tham số của cột tuổi. Không lỗi nào được ném ra, mô hình "
        "vẫn trả về một con số, chỉ là con số sai.", "warn"))

    r.h(2, "2.4. Vectơ đặc trưng")
    r.p(
        "Một mẫu dữ liệu được biểu diễn bằng một vectơ đặc trưng gồm nhiều giá trị số:",
        '<div class="formula">x = [x<sub>1</sub>, x<sub>2</sub>, …, x<sub>d</sub>]'
        "<sup>T</sup> ∈ ℝ<sup>d</sup></div>",
        "trong đó <em>x</em> là vectơ đặc trưng của một mẫu, <em>x<sub>i</sub></em> "
        "là giá trị của đặc trưng thứ <em>i</em>, <em>d</em> là số lượng đặc trưng, "
        "và <em>T</em> biểu thị phép chuyển vị.",
        "Ví dụ, một bệnh nhân trong Ứng dụng 1 được biểu diễn bởi "
        "<em>x</em> = [Glucose, BMI, Age, Pregnancies, DiabetesPedigreeFunction]<sup>T</sup>, "
        "còn một căn nhà trong Ứng dụng 2 bởi "
        "<em>x</em> = [Area, Floors, Bedrooms, Bathrooms, …]<sup>T</sup>.",
    )

    r.h(2, "2.5. Ma trận đặc trưng")
    r.p(
        "Khi ghép nhiều vectơ đặc trưng lại theo hàng, ta được ma trận đặc trưng:",
        '<div class="formula">X ∈ ℝ<sup>N×d</sup></div>',
        "với <em>N</em> là số lượng mẫu, <em>d</em> là số lượng đặc trưng; mỗi hàng "
        "tương ứng một mẫu, mỗi cột tương ứng một đặc trưng. Nếu tập dữ liệu có "
        "1.000 mẫu và 10 đặc trưng thì <em>X</em> ∈ ℝ<sup>1000×10</sup>.",
    )
    r.p(figure("fig_vector_matrix.png",
               "Minh hoạ biểu diễn dữ liệu dưới dạng vectơ và ma trận đặc trưng."))

    r.h(2, "2.6. Tensor và biểu diễn dữ liệu nhiều chiều")
    r.p(
        "Tensor là dạng biểu diễn có thể chứa nhiều hơn hai chiều, dùng khi cấu "
        "trúc dữ liệu cần nhiều chiều mới thể hiện đủ thông tin.",
        "Với dữ liệu văn bản, nếu có <em>B</em> văn bản trong một lô, mỗi văn bản "
        "có <em>T</em> token, và mỗi token được biểu diễn bằng một vectơ embedding "
        "<em>d</em> chiều, thì dữ liệu embedding có dạng:",
        '<div class="formula">E ∈ ℝ<sup>B×T×d</sup></div>',
        "Ứng dụng 3 dựng thật tensor này với <em>B</em> = "
        f"{META['customer_behavior']['tensor_demo']['B']}, <em>T</em> = "
        f"{META['customer_behavior']['tensor_demo']['T']}, <em>d</em> = "
        f"{META['customer_behavior']['tensor_demo']['d']} — xem mục 9.5.",
    )

    r.h(2, "2.7. Biểu diễn dữ liệu văn bản")
    r.p(
        "Khi bộ dữ liệu chứa bình luận hoặc đánh giá của khách hàng, văn bản phải "
        "đi qua chuỗi biến đổi sau trước khi mô hình dùng được:",
        '<div class="flow">Văn bản → Token → Token ID → Vectơ / Embedding</div>',
        "<strong>Token</strong> là các đơn vị được tạo ra từ văn bản qua quá trình "
        "tách token. Tuỳ phương pháp, một token có thể là một từ, một phần của từ, "
        "hoặc một ký hiệu.",
        "<strong>Token ID</strong> là số nguyên mà mỗi token được ánh xạ tới. Bước "
        "này đưa văn bản về dạng số, nhưng bản thân ID chưa mang ngữ nghĩa: ID 5 "
        "không “gần” ID 4 hơn ID 900 về mặt ý nghĩa.",
        "<strong>Embedding</strong> là vectơ số nhiều chiều biểu diễn token trong "
        "không gian số, trong đó khoảng cách giữa hai vectơ mới thực sự phản ánh "
        "quan hệ ngữ nghĩa.",
    )
    r.p(figure("fig_text_pipeline.png",
               "Quy trình chuyển đổi dữ liệu văn bản thành biểu diễn số."))
    r.p(
        "Ứng dụng 3 sử dụng <strong>TF-IDF</strong> để triển khai thay vì embedding "
        "học từ đầu. TF-IDF cho mỗi văn bản một vectơ duy nhất "
        "<em>X</em> ∈ ℝ<sup>N×d</sup>, mất thứ tự từ nhưng bù lại mỗi chiều tương "
        "ứng một từ cụ thể nên <strong>đọc được</strong>. Lý do chọn và cái giá "
        "phải trả được phân tích ở mục 9.5.",
    )

    r.h(2, "2.8. Mã hoá dữ liệu phân loại")
    r.p(
        "Các đặc trưng dạng phân loại không thể đưa trực tiếp vào phần lớn mô hình "
        "học máy, nên phải chuyển thành số. Hai phương pháp phổ biến:",
        "<ul>"
        "<li><strong>Label Encoding</strong> — ánh xạ mỗi hạng mục thành một số nguyên;</li>"
        "<li><strong>One-Hot Encoding</strong> — mỗi hạng mục thành một cột nhị phân riêng.</li>"
        "</ul>",
        "Ví dụ, tập hạng mục {Urban, Suburban, Rural} qua one-hot trở thành "
        "Urban → [1, 0, 0], Suburban → [0, 1, 0], Rural → [0, 0, 1].",
    )
    r.p(figure("fig_encoding.png", "So sánh Label Encoding và One-Hot Encoding."))
    r.p(note(
        "Vì sao báo cáo này dùng one-hot chứ không dùng label encoding.",
        "Label encoding áp đặt một thứ tự không có thật lên dữ liệu. Nếu gán "
        "Đông = 0, Tây = 1, Nam = 2, mô hình sẽ hiểu rằng Nam “lớn hơn” Tây và "
        "khoảng cách Đông → Nam gấp đôi khoảng cách Đông → Tây. Với hướng nhà, "
        "điều đó vô nghĩa. One-hot không giả định gì về thứ tự, đổi lại làm tăng "
        "số chiều — ở Ứng dụng 2, từ "
        f"{META['house_price']['n_features_before_encoding']} cột lên "
        f"{META['house_price']['n_features_after_encoding']} cột."))

    r.h(2, "2.9. Kiểu dữ liệu và miền giá trị")
    r.p(
        "Biểu diễn dữ liệu không chỉ phụ thuộc kích thước mà còn phụ thuộc kiểu dữ "
        "liệu và miền giá trị. Sáu yếu tố cần kiểm tra: Shape, Dtype, Range, "
        "Encoding, Ordering (thứ tự đặc trưng), và Preprocessing.",
        "Miền giá trị đặc biệt quan trọng với các mô hình dựa trên khoảng cách. "
        "Trong Ứng dụng 1, <code class='inl'>Glucose</code> có thang 0–200 còn "
        "<code class='inl'>DiabetesPedigreeFunction</code> có thang 0–2,4. Nếu "
        "không chuẩn hoá, khoảng cách Euclid giữa hai bệnh nhân gần như chỉ còn "
        "phản ánh mỗi chênh lệch glucose — bốn đặc trưng còn lại bị vô hiệu hoá "
        "trên thực tế mà không có cảnh báo nào.",
    )

    r.h(2, "2.10. Tính nhất quán của biểu diễn dữ liệu")
    r.p(
        "Biểu diễn dữ liệu phải được duy trì nhất quán từ quá trình huấn luyện đến "
        "quá trình triển khai. Các bước như mã hoá, chuẩn hoá hoặc xử lý giá trị "
        "thiếu dùng lúc huấn luyện phải được áp dụng <em>y hệt</em> khi mô hình "
        "nhận dữ liệu mới.",
        "Nếu quá trình triển khai dùng một cách tiền xử lý khác với lúc huấn luyện, "
        "dữ liệu đầu vào không còn cùng ý nghĩa với dữ liệu mà mô hình đã học. Đây "
        "là lý do cả ba ứng dụng đều đóng gói tiền xử lý thành một đối tượng "
        "<code class='inl'>Pipeline</code> duy nhất, lưu ra đĩa, và REST API chỉ "
        "nạp lại rồi gọi <code class='inl'>transform</code> — không bao giờ "
        "<code class='inl'>fit</code> lại trên dữ liệu người dùng.",
    )

    r.h(2, "2.11. Rò rỉ dữ liệu")
    r.p(
        "Rò rỉ dữ liệu xảy ra khi thông tin không được phép — từ tập kiểm tra, hoặc "
        "từ tương lai — ảnh hưởng đến quá trình huấn luyện hoặc tiền xử lý.",
        "Ví dụ kinh điển: dùng toàn bộ dữ liệu để tính tham số của bộ chuẩn hoá "
        "<em>trước khi</em> chia tập. Khi ấy trung bình và độ lệch chuẩn dùng lúc "
        "huấn luyện đã chứa thông tin từ tập kiểm tra, và điểm số cuối cùng lạc "
        "quan giả tạo.",
        "Quy trình đúng là:",
        '<div class="flow">Chia dữ liệu → Fit pipeline CHỈ trên tập train → '
        "Áp dụng transform cho validation/test → Huấn luyện → Đánh giá</div>",
    )
    r.p(figure("fig_leakage.png", "Rò rỉ dữ liệu và quy trình phòng tránh."))
    r.p("Assignment 02 gặp cả ba dạng rò rỉ, mỗi ứng dụng một dạng khác nhau:")
    r.p(table(
        ["Dạng rò rỉ", "Gặp ở đâu", "Vì sao nguy hiểm", "Cách xử lý"],
        [
            ["Rò rỉ qua tiền xử lý", "Cả ba ứng dụng",
             "Tham số chuẩn hoá nhìn thấy tập test",
             "Đưa mọi phép biến đổi vào <code class='inl'>Pipeline</code>, chỉ fit trên train"],
            ["Rò rỉ qua bản ghi trùng lặp", "Ứng dụng 2",
             "Cùng một căn nhà nằm ở cả train lẫn test; mô hình được chấm điểm trên "
             "chính dòng nó đã học thuộc",
             "Khử trùng lặp <em>trước</em> khi chia tập — mục 8.4"],
            ["Rò rỉ nhãn", "Ứng dụng 3",
             "Cột <code class='inl'>Rating</code> gần như chính là đáp án; điểm số "
             "đẹp nhưng mô hình không học được gì",
             "Loại bỏ cột khỏi tập đặc trưng — mục 9.4"],
        ]))
    r.p(note(
        "Dạng thứ ba khó phát hiện nhất.",
        "Rò rỉ qua tiền xử lý bị chặn bằng một quy tắc kỹ thuật đơn giản. Rò rỉ "
        "nhãn thì không: nó không vi phạm bất kỳ quy tắc pipeline nào, không có "
        "công cụ nào cảnh báo, và biểu hiện duy nhất là điểm số <em>quá đẹp</em>. "
        "Chỉ có hiểu biết về ý nghĩa từng cột mới phát hiện được.", "bad"))


# -------------------------------------------------------------- Chương III
def _ch3(r) -> None:
    r.h(1, "Chương III. Phương pháp tiền xử lý dữ liệu")
    r.p(figure("fig_preprocess.png", "Quy trình tiền xử lý dữ liệu tám bước.", "62%"))

    r.h(2, "3.1. Tìm hiểu dữ liệu")
    r.p(
        "Trước khi xây dựng mô hình, dữ liệu cần được kiểm tra để xác định cấu trúc "
        "và chất lượng. Sáu lệnh khảo sát chuẩn được dùng thống nhất ở cả ba ứng dụng:",
    )
    r.p(code(
        "df.shape          # kích thước tập dữ liệu\n"
        "df.head()         # xem vài bản ghi đầu\n"
        "df.info()         # kiểu dữ liệu từng cột, số giá trị không rỗng\n"
        "df.describe()     # thống kê mô tả các cột số\n"
        "df.isna().sum()   # số giá trị thiếu theo cột\n"
        "df.duplicated().sum()   # số bản ghi trùng lặp",
        "Sáu lệnh khảo sát bắt buộc theo yêu cầu đề bài."))
    r.p(note(
        "Cảnh báo quan trọng.",
        "<code class='inl'>isna().sum()</code> chỉ đếm giá trị <code class='inl'>NaN</code>. "
        "Nó <em>không</em> phát hiện giá trị thiếu được mã hoá bằng một con số hợp "
        "lệ về kiểu — chẳng hạn số 0. Ứng dụng 1 là ví dụ trực tiếp: hàm này trả về "
        "0 ở mọi cột, trong khi 48,7% cột <code class='inl'>Insulin</code> là dữ "
        "liệu thiếu. Chất lượng dữ liệu là việc của người đọc dữ liệu, không phải "
        "của một hàm.", "warn"))

    r.h(2, "3.2. Làm sạch dữ liệu")
    r.h(3, "3.2.1. Xử lý giá trị thiếu")
    r.p(table(
        ["Cách xử lý", "Khi nào dùng", "Rủi ro", "Dùng ở ứng dụng"],
        [
            ["Xoá dòng", "Tỷ lệ thiếu rất thấp và thiếu ngẫu nhiên",
             "Mất dữ liệu; lệch mẫu nếu thiếu có hệ thống", "Ứng dụng 3 (dòng không có bình luận)"],
            ["Xoá cột", "Thiếu trên 80% và cột ít giá trị dự báo",
             "Mất thông tin nếu bản thân sự vắng mặt mang tín hiệu", "Ứng dụng 1 (3 cột bị loại)"],
            ["Điền bằng trung bình", "Phân phối gần đối xứng", "Bị đuôi dài kéo lệch", "Không dùng"],
            ["Điền bằng trung vị", "Phân phối lệch, có ngoại lệ",
             "Làm giảm phương sai thật của cột", "Ứng dụng 1, 2, 3 — mọi cột số"],
            ["Coi là một hạng mục riêng", "Biến phân loại; sự vắng mặt mang thông tin",
             "Tăng số chiều one-hot", "Ứng dụng 2 (hạng mục “Không rõ”)"],
        ]))
    r.p(
        "Báo cáo này chọn <strong>trung vị</strong> chứ không phải trung bình cho "
        "mọi cột số, vì cả ba tập dữ liệu đều có các cột lệch phải mạnh — trung "
        "bình sẽ bị đuôi dài kéo lệch trong khi trung vị bền với ngoại lệ.",
    )
    r.h(3, "3.2.2. Xử lý dữ liệu trùng lặp")
    r.p(
        "Bản ghi trùng lặp phải được khử <strong>trước khi chia tập</strong>. Nếu "
        "không, một bản ghi có thể nằm cả ở tập train lẫn tập test; mô hình được "
        "chấm điểm trên chính dòng nó đã học thuộc, và điểm test bị thổi phồng mà "
        "không có dấu hiệu nào lộ ra. Mục 8.4 đo trực tiếp mức thổi phồng này trên "
        "dữ liệu thật.",
    )
    r.h(3, "3.2.3. Xử lý giá trị không hợp lệ")
    r.p(
        "Giá trị không hợp lệ là giá trị đúng kiểu dữ liệu nhưng vô nghĩa trong "
        "miền ứng dụng: huyết áp bằng 0, diện tích âm, giá bằng 0. Chúng nguy hiểm "
        "hơn giá trị thiếu vì không công cụ nào tự phát hiện — chỉ hiểu biết về "
        "miền ứng dụng mới nhận ra.",
    )
    r.h(3, "3.2.4. Phân tích ngoại lệ")
    r.p(
        "Ngoại lệ được phát hiện bằng quy tắc IQR: một giá trị nằm ngoài khoảng "
        "[Q1 − 1,5·IQR, Q3 + 1,5·IQR] được coi là ngoại lệ, với IQR = Q3 − Q1.",
        "Phát hiện xong <em>không</em> đồng nghĩa với việc phải xoá. Câu hỏi đúng "
        "là: đây là lỗi nhập liệu, hay là một quan sát thật mang đúng tín hiệu ta "
        "muốn mô hình học? Trong Ứng dụng 1, insulin rất cao là biểu hiện kháng "
        "insulin thật — xoá đi là xoá mất chính tín hiệu bệnh lý. Báo cáo này giữ "
        "lại ngoại lệ thật và phòng thủ bằng cách điền khuyết theo trung vị và "
        "chuẩn hoá thang đo.",
    )

    r.h(2, "3.3. Mã hoá biến phân loại")
    r.p(
        "Hai phương pháp đã trình bày ở mục 2.8. Báo cáo này dùng "
        "<strong>one-hot</strong> cho toàn bộ biến phân loại, với tham số "
        "<code class='inl'>handle_unknown=\"ignore\"</code>.",
        "Tham số ấy quan trọng cho khâu triển khai: nếu người dùng Web nhập một "
        "tỉnh không có trong tập huấn luyện, bộ mã hoá trả về vectơ toàn 0 thay vì "
        "ném lỗi và làm sập API.",
    )

    r.h(2, "3.4. Chuẩn hoá dữ liệu số")
    r.p(
        "<code class='inl'>StandardScaler</code> đưa mỗi cột về trung bình 0 và độ "
        "lệch chuẩn 1:",
        '<div class="formula">z = (x − μ) / σ</div>',
        "trong đó μ và σ được tính <strong>chỉ trên tập train</strong>.",
        "Chuẩn hoá là bắt buộc với KNN và SVM vì chúng dựa trên khoảng cách. Với mô "
        "hình dạng cây thì không cần — cây chia theo ngưỡng trên từng biến nên miễn "
        "nhiễm với thang đo — nhưng để trong pipeline chung vẫn vô hại và giữ cho "
        "mọi mô hình dùng đúng một biểu diễn, nhờ đó bảng so sánh mới công bằng.",
    )

    r.h(2, "3.5. Chia tập dữ liệu")
    r.p(
        "Đề bài yêu cầu ba tập độc lập:",
        '<div class="formula">D = D<sub>train</sub> ∪ D<sub>validation</sub> ∪ '
        "D<sub>test</sub></div>",
        "Báo cáo dùng tỷ lệ <strong>70 / 15 / 15</strong> cho cả ba ứng dụng. Vai "
        "trò của từng tập khác nhau và không được lẫn lộn:",
        "<ul>"
        "<li><strong>Train</strong> — học tham số mô hình và tham số tiền xử lý;</li>"
        "<li><strong>Validation</strong> — so sánh và chọn mô hình;</li>"
        "<li><strong>Test</strong> — chỉ dùng <em>một lần</em>, để ước lượng sai số "
        "cuối cùng trên dữ liệu chưa từng thấy.</li></ul>",
        "Với hai bài toán phân loại, phép chia được <strong>phân tầng</strong> "
        "(<code class='inl'>stratify=y</code>) để giữ tỷ lệ hai lớp gần như đồng "
        "nhất ở cả ba tập. Không phân tầng thì tập test có thể có tỷ lệ lớp khác "
        "hẳn train, và khi ấy điểm test đo lẫn cả sự lệch phân bố lẫn chất lượng mô "
        "hình — không tách được hai thứ.",
    )
    r.p(note(
        "Vì sao cần tập validation riêng.",
        "Nếu chọn mô hình dựa trên chính tập test, tập test trở thành một tập "
        "validation trá hình: ta đã dùng nó để ra quyết định, nên nó không còn “chưa "
        "từng thấy” nữa và ước lượng cuối cùng lại lạc quan thêm một lần. Đây là "
        "một trong bốn cải tiến của Assignment 02 so với Assignment 01 — xem mục 8.4."))

    r.h(2, "3.6. Ngăn ngừa rò rỉ dữ liệu")
    r.p(
        "Ba biện pháp được áp dụng thống nhất ở cả ba ứng dụng: khử trùng lặp trước "
        "khi chia tập; đóng gói toàn bộ tiền xử lý vào "
        "<code class='inl'>Pipeline</code> và chỉ <code class='inl'>fit</code> trên "
        "train; và loại bỏ mọi đặc trưng chứa sẵn thông tin về nhãn.",
    )

    r.h(2, "3.7. Pipeline tiền xử lý")
    r.p(
        "<code class='inl'>ColumnTransformer</code> cho phép áp dụng các phép biến "
        "đổi khác nhau cho các nhóm cột khác nhau, rồi ghép kết quả thành một ma "
        "trận duy nhất. Đoạn mã dưới là pipeline thật của Ứng dụng 2, lấy nguyên "
        "văn từ notebook đã chạy:",
    )
    r.p(code(notebook_code("house_price", "ColumnTransformer([", drop_comment_header=True),
             "Pipeline hai nhánh của Ứng dụng 2 — nhánh số và nhánh phân loại."))
    r.p(
        "Lợi ích không chỉ là gọn mã. Khi gọi "
        "<code class='inl'>preprocessor.fit(X_train)</code>, mọi bộ điền khuyết, "
        "bộ chuẩn hoá và bộ mã hoá bên trong <strong>chỉ nhìn thấy tập train</strong>. "
        "Đây là biện pháp chống rò rỉ đáng tin cậy nhất vì nó đúng theo cấu trúc "
        "chứ không phụ thuộc vào việc người viết có nhớ hay không.",
    )

    r.h(2, "3.8. Nguyên tắc tái lập quá trình tiền xử lý")
    r.p(
        "Toàn bộ pipeline được lưu ra đĩa bằng <code class='inl'>joblib</code> cùng "
        "với mô hình. Lúc triển khai, dịch vụ nạp lại chính đối tượng đó và gọi "
        "<code class='inl'>transform</code>. Ba nguyên tắc bắt buộc:",
        "<ol>"
        "<li><strong>Không bao giờ <code class='inl'>fit</code> lại</strong> trên dữ liệu người dùng.</li>"
        "<li><strong>Giữ đúng thứ tự cột</strong> — thứ tự được lưu trong "
        "<code class='inl'>metadata.json</code> và dịch vụ dựng khung dữ liệu theo đúng nó.</li>"
        "<li><strong>Kiểm chứng sau khi nạp lại</strong> — cả ba notebook đều có một "
        "phép khẳng định rằng artifact nạp từ đĩa cho kết quả trùng khớp hoàn toàn "
        "với mô hình trong bộ nhớ trên toàn bộ tập test.</li></ol>",
    )


# --------------------------------------------------------------- Chương IV
def _ch4(r) -> None:
    r.h(1, "Chương IV. Các độ đo đánh giá mô hình")
    r.p(
        "Loại bài toán quyết định bộ độ đo. Dùng nhầm bộ độ đo là một lỗi nghiêm "
        "trọng hơn dùng nhầm thuật toán, vì nó khiến ta chọn sai mô hình mà vẫn "
        "tưởng mình đang tối ưu đúng thứ.",
    )
    r.p(figure("fig_metrics.png", "Các độ đo đánh giá cho bài toán phân loại và hồi quy."))

    r.h(2, "4.1. Đối với bài toán Hồi quy")
    r.p(table(
        ["Độ đo", "Công thức", "Đơn vị", "Ý nghĩa trong bài toán giá nhà"],
        [
            ["MAE", "(1/N) Σ |y<sub>i</sub> − ŷ<sub>i</sub>|", "tỷ VNĐ",
             "Sai lệch trung bình. Con số dễ giải thích nhất cho người dùng cuối."],
            ["MSE", "(1/N) Σ (y<sub>i</sub> − ŷ<sub>i</sub>)²", "(tỷ VNĐ)²",
             "Phạt nặng các cú trượt lớn. Không đọc trực tiếp được vì đơn vị bị bình phương."],
            ["RMSE", "√MSE", "tỷ VNĐ",
             "Đưa MSE về lại đơn vị gốc. So RMSE với MAE cho biết sai số rải đều hay "
             "tập trung ở vài cú trượt lớn."],
            ["R²", "1 − SS<sub>res</sub> / SS<sub>tot</sub>", "không đơn vị",
             "Tỷ lệ phương sai giá được giải thích. 1 là hoàn hảo, 0 ngang mô hình "
             "cơ sở, âm là tệ hơn cả việc luôn đoán giá trung bình."],
        ]))
    r.p(note(
        "Cách đọc cặp MAE và RMSE.",
        "RMSE luôn lớn hơn hoặc bằng MAE. Khoảng cách giữa hai con số cho biết "
        "phân bố sai số: nếu RMSE xấp xỉ MAE thì sai số rải khá đều; nếu RMSE lớn "
        "hơn MAE nhiều thì có một nhóm nhỏ dự đoán sai rất nặng đang kéo chỉ số "
        "lên. Ứng dụng 2 rơi vào trường hợp thứ hai, và mục 8.8 truy ra nhóm ấy "
        "nằm ở hai đầu phổ giá."))

    r.h(2, "4.2. Đối với bài toán Phân lớp")
    r.p(
        "Mọi độ đo phân lớp đều bắt nguồn từ bốn ô của ma trận nhầm lẫn: "
        "TP (dương tính thật), TN (âm tính thật), FP (dương tính giả) và "
        "FN (âm tính giả).",
    )
    r.p(table(
        ["Độ đo", "Công thức", "Trả lời câu hỏi gì", "Khi nào gây hiểu lầm"],
        [
            ["Accuracy", "(TP + TN) / (TP + TN + FP + FN)",
             "Tỷ lệ dự đoán đúng nói chung",
             "<strong>Khi hai lớp mất cân bằng.</strong> Ở Ứng dụng 3, luôn đoán "
             "“khuyến nghị” đã đạt "
             f"{fmt(META['customer_behavior']['baseline_accuracy'] * 100, 1)}% mà không học gì."],
            ["Precision", "TP / (TP + FP)",
             "Trong các trường hợp bị báo động, bao nhiêu phần trăm là báo động thật",
             "Khi FN mới là loại sai lầm đắt tiền"],
            ["Recall", "TP / (TP + FN)",
             "Trong các trường hợp thật sự dương tính, bắt được bao nhiêu phần trăm",
             "Khi FP đắt; có thể nâng Recall lên 1 bằng cách đoán tất cả là dương tính"],
            ["F1", "2·(P·R) / (P + R)",
             "Trung bình điều hoà của Precision và Recall",
             "Che giấu việc một trong hai thành phần rất kém"],
            ["ROC-AUC", "diện tích dưới đường cong ROC",
             "Khả năng phân tách hai lớp ở mọi ngưỡng",
             "Có thể trông đẹp ngay cả khi ngưỡng 0,5 cho kết quả thực tế tệ"],
        ]))
    r.p(
        "<strong>Ma trận nhầm lẫn phải được diễn giải, không chỉ hiển thị.</strong> "
        "Ý nghĩa của bốn ô phụ thuộc hoàn toàn vào bài toán, và chính điều đó quyết "
        "định độ đo nào là quan trọng nhất:",
    )
    r.p(table(
        ["Ứng dụng", "FN nghĩa là gì", "FP nghĩa là gì", "Cái nào đắt hơn", "Độ đo chính"],
        [
            ["1 — Tiểu đường",
             "Bệnh nhân <strong>có bệnh</strong> bị bỏ sót",
             "Người khoẻ bị cảnh báo nhầm, phải làm thêm một xét nghiệm",
             "FN đắt hơn nhiều", "<strong>Recall</strong>"],
            ["3 — Thương mại điện tử",
             "Khách hài lòng bị đoán là không hài lòng — chỉ gây rà soát thừa",
             "Phản hồi xấu bị bỏ lọt, sản phẩm lỗi tiếp tục được bán",
             "FP đắt hơn", "<strong>F1 lớp thiểu số</strong>"],
        ]))
    r.p(
        "Hai ứng dụng cùng là phân loại nhị phân nhưng ưu tiên ngược nhau. Đây là lý "
        "do báo cáo không dùng một độ đo duy nhất cho mọi bài toán, và là lý do "
        "<code class='inl'>class_weight=\"balanced\"</code> được bật ở những mô hình "
        "hỗ trợ — nó nhân trọng số lỗi của lớp thiểu số lên, đẩy mô hình về phía ưu "
        "tiên đúng với bài toán.",
    )


# ---------------------------------------------------------------- Chương V
def _ch5(r) -> None:
    r.h(1, "Chương V. Phương pháp xây dựng mô hình")

    r.h(2, "5.1. Tổng quan về xây dựng mô hình")
    r.p(
        "Assignment 02 yêu cầu so sánh năm mô hình cho bài toán tiểu đường, năm mô "
        "hình cho bài toán giá nhà và sáu mô hình cho ứng dụng thương mại điện tử — "
        "tổng cộng 16 mô hình.",
        "Nguyên tắc so sánh: mọi mô hình trong cùng một ứng dụng được huấn luyện "
        "trên <strong>cùng một cách chia dữ liệu</strong> và <strong>cùng một biểu "
        "diễn</strong> (trừ khi bản thân biểu diễn mới là thứ đang được so sánh, "
        "như ở Ứng dụng 3). Chỉ khi đó chênh lệch điểm số mới quy được về chất lượng "
        "mô hình chứ không lẫn yếu tố nào khác.",
    )
    r.p(figure("fig_models.png", "Các nhóm mô hình học máy được sử dụng."))

    r.h(2, "5.2. Mô hình Baseline")
    r.p(
        "Trước khi huấn luyện bất kỳ mô hình nào, phải biết ngưỡng vô nghĩa nằm ở đâu.",
        "<ul>"
        "<li><strong>Phân loại</strong> — <code class='inl'>DummyClassifier"
        "(strategy=\"most_frequent\")</code>: luôn đoán lớp đa số.</li>"
        "<li><strong>Hồi quy</strong> — <code class='inl'>DummyRegressor"
        "(strategy=\"mean\")</code>: luôn đoán giá trị trung bình; theo định nghĩa "
        "có R² = 0.</li></ul>",
        "Mô hình nào không vượt được baseline thì không có giá trị, dù điểm số "
        "trông cao. Baseline của Ứng dụng 3 đạt tới "
        f"{fmt(META['customer_behavior']['baseline_accuracy'] * 100, 1)}% accuracy — "
        "cao hơn nhiều mô hình thật ở các bài toán khác, mà vẫn hoàn toàn vô dụng.",
    )

    r.h(2, "5.3. Các mô hình cho bài toán hồi quy")
    r.p(table(
        ["Mô hình", "Nguyên lý", "Ưu điểm", "Hạn chế"],
        [
            ["Linear Regression", "Tìm siêu phẳng cực tiểu tổng bình phương sai số",
             "Hệ số đọc được trực tiếp thành “mỗi m² thêm bao nhiêu tỷ”",
             "Chỉ bắt được quan hệ tuyến tính"],
            ["Ridge Regression", "Hồi quy tuyến tính có phạt L2 trên hệ số",
             "Ổn định hoá hệ số khi one-hot sinh ra nhiều cột tương quan",
             "Vẫn là mô hình tuyến tính"],
            ["Decision Tree Regressor", "Chia không gian đặc trưng bằng các ngưỡng liên tiếp",
             "Bắt được phi tuyến và tương tác giữa các biến",
             "Dễ quá khớp; dự đoán bậc thang, không mượt"],
            ["Random Forest Regressor", "Trung bình hoá nhiều cây trên các mẫu bootstrap",
             "Giảm mạnh phương sai; bền với ngoại lệ", "Khó diễn giải; tệp mô hình lớn"],
            ["Gradient Boosting Regressor", "Các cây học tuần tự trên phần dư của cây trước",
             "Thường chính xác nhất trên dữ liệu bảng", "Huấn luyện lâu; nhạy với siêu tham số"],
        ]))
    r.p(note(
        "Giới hạn chung của mô hình dạng cây trong bài toán hồi quy.",
        "Mỗi lá trả về trung bình của các mẫu rơi vào lá đó, nên mô hình "
        "<strong>không bao giờ dự đoán vượt ra ngoài khoảng giá đã thấy lúc huấn "
        "luyện</strong>. Hệ quả là hiện tượng co về trung bình: căn rẻ bị đoán đắt "
        "lên, căn đắt bị đoán rẻ đi. Mục 8.8 chỉ ra hiện tượng này trên dữ liệu "
        "thật và giải thích vì sao giao diện Web phải hiển thị khoảng dao động thay "
        "vì một con số duy nhất."))

    r.h(2, "5.4. Các mô hình cho bài toán phân loại")
    r.p(table(
        ["Mô hình", "Nguyên lý", "Ưu điểm", "Hạn chế"],
        [
            ["Logistic Regression", "Ánh xạ tổ hợp tuyến tính qua hàm sigmoid thành xác suất",
             "Chuẩn tham chiếu trong y học; hệ số diễn giải được thành tỷ số chênh",
             "Chỉ tạo ra ranh giới tuyến tính"],
            ["K-Nearest Neighbors", "Bỏ phiếu theo <em>k</em> mẫu gần nhất",
             "Không tham số; không giả định dạng phân phối",
             "<strong>Bắt buộc chuẩn hoá</strong>; chậm khi dữ liệu lớn"],
            ["Decision Tree", "Chia dữ liệu theo ngưỡng trên từng đặc trưng",
             "Sinh ra luật đọc được, chuyên gia kiểm tra được", "Dễ quá khớp nếu không giới hạn độ sâu"],
            ["Random Forest", "Tập hợp nhiều cây trên mẫu bootstrap",
             "Giảm phương sai; cho tầm quan trọng đặc trưng", "Hộp đen so với cây đơn"],
            ["SVM (RBF)", "Tìm siêu phẳng biên lớn trong không gian ánh xạ phi tuyến",
             "Bắt được ranh giới phi tuyến phức tạp",
             "<strong>Bắt buộc chuẩn hoá</strong>; chi phí bậc hai theo số mẫu"],
            ["Linear SVM", "Siêu phẳng biên lớn trong không gian gốc",
             "Rất mạnh trên dữ liệu văn bản thưa nhiều chiều", "Không cho xác suất trực tiếp"],
            ["Multinomial Naive Bayes", "Xác suất Bayes với giả định độc lập giữa các đặc trưng",
             "Chuẩn cổ điển cho phân loại văn bản; huấn luyện gần như tức thì",
             "Yêu cầu đặc trưng không âm; giả định độc lập không đúng thực tế"],
        ]))
    r.p(
        "<strong>Xử lý mất cân bằng lớp.</strong> Cả Ứng dụng 1 (tỷ lệ 1,87 : 1) và "
        "Ứng dụng 3 (4,5 : 1) đều mất cân bằng. Tham số "
        "<code class='inl'>class_weight=\"balanced\"</code> được bật ở Logistic "
        "Regression, Decision Tree, Random Forest và SVM. Nó nhân trọng số lỗi của "
        "lớp thiểu số theo tỷ lệ nghịch với tần suất, khiến mô hình chịu thiệt nhiều "
        "hơn khi bỏ sót lớp hiếm — đúng với ưu tiên đã phân tích ở mục 4.2.",
    )


# --------------------------------------------------------------- Chương VI
def _ch6(r) -> None:
    r.h(1, "Chương VI. Phương pháp lưu trữ và triển khai mô hình")

    r.h(2, "6.1. Tổng quan")
    r.p(
        "Giai đoạn triển khai nối thực nghiệm học máy với một hệ thống phần mềm "
        "dùng được. Bốn thành phần bắt buộc:",
        '<div class="flow">Bộ tiền xử lý đã lưu + Mô hình đã lưu + REST API + '
        "Giao diện Web và Mobile</div>",
        "Phân biệt quan trọng: <strong>huấn luyện ≠ suy luận</strong>. Mô hình được "
        "huấn luyện một lần trong giai đoạn thực nghiệm rồi lưu lại. Dịch vụ Web và "
        "ứng dụng Mobile chỉ thực hiện suy luận trên mô hình đã lưu, không bao giờ "
        "huấn luyện lại.",
    )

    r.h(2, "6.2. Lưu trữ mô hình")
    r.p("Bốn thứ phải được lưu, theo đúng yêu cầu đề bài:")
    r.p(table(
        ["Thành phần", "Tệp", "Nội dung"],
        [
            ["Tiền xử lý + biến đổi đặc trưng", "<code class='inl'>preprocessor.joblib</code>",
             "Toàn bộ <code class='inl'>Pipeline</code>: điền khuyết, chuẩn hoá, "
             "one-hot, TF-IDF — kèm mọi tham số đã học từ tập train"],
            ["Mô hình đã huấn luyện", "<code class='inl'>&lt;tên_mô_hình&gt;.joblib</code>",
             "Cả bộ mô hình được lưu, để giao diện Web cho phép người dùng đổi mô hình "
             "và so sánh trực tiếp"],
            ["Cấu hình mô hình", "<code class='inl'>metadata.json</code>",
             "Thứ tự cột, danh sách hạng mục hợp lệ, nhãn lớp, mô hình được chọn, "
             "toàn bộ điểm số trên tập test"],
        ]))
    r.p(code(
        "import joblib\n\n"
        "joblib.dump(preprocessor, MODEL_DIR / \"preprocessor.joblib\")\n"
        "joblib.dump(best_model, MODEL_DIR / \"random_forest.joblib\")",
        "Lưu artifact bằng joblib."))
    r.p(note(
        "Vì sao thứ tự cột phải được lưu.",
        "<code class='inl'>preprocessor.transform</code> nhận dữ liệu theo "
        "<strong>vị trí</strong>, không theo tên cột. Nếu REST API dựng khung dữ "
        "liệu theo thứ tự khác lúc huấn luyện, giá trị BMI sẽ được chuẩn hoá bằng "
        "trung bình và độ lệch chuẩn của tuổi. Không lỗi nào được ném ra và mô hình "
        "vẫn trả về một con số — chỉ là con số sai. "
        "<code class='inl'>metadata.json</code> tồn tại để chặn đúng lỗi này.", "warn"))

    r.h(2, "6.3. Tính nhất quán giữa huấn luyện và dự đoán")
    r.p(
        "Quy trình tiền xử lý lúc dự đoán phải trùng khớp quy trình lúc huấn luyện. "
        "Cụ thể, dịch vụ triển khai <strong>không được</strong> tạo mới scaler, "
        "encoder, imputer hay bộ vector hoá từ dữ liệu người dùng, mà phải nạp lại "
        "pipeline đã học từ tập train.",
        "Cả ba notebook đều kết thúc bằng một phép kiểm chứng: nạp artifact từ đĩa, "
        "chạy trên toàn bộ tập test, và khẳng định kết quả trùng khớp hoàn toàn với "
        "mô hình đang nằm trong bộ nhớ. Nếu phép khẳng định này sai thì artifact đã "
        "hỏng và dịch vụ Web sẽ trả về kết quả khác với thực nghiệm.",
    )
    r.p(code(notebook_code("diabetes", "Artifact nạp lại KHÔNG khớp"),
             "Kiểm thử suy luận của Ứng dụng 1 — chứng minh artifact tự nó chạy được."))

    r.h(2, "6.4. Triển khai mô hình dưới dạng Web Service")
    r.p(
        "Cả ba ứng dụng dùng <strong>Flask</strong>. Mỗi ứng dụng là một dịch vụ "
        "độc lập chạy trên cổng riêng, phơi ra các điểm cuối sau:",
    )
    r.p(table(
        ["Điểm cuối", "Phương thức", "Chức năng"],
        [
            ["<code class='inl'>/</code>, <code class='inl'>/web</code>", "GET", "Trả về giao diện Web"],
            ["<code class='inl'>/mobile</code>", "GET", "Trả về giao diện Mobile"],
            ["<code class='inl'>/health</code>", "GET", "Kiểm tra dịch vụ sống và liệt kê mô hình khả dụng"],
            ["<code class='inl'>/metadata</code>", "GET",
             "Trả về <code class='inl'>metadata.json</code> để giao diện tự dựng biểu mẫu"],
            ["<code class='inl'>/models</code>", "GET",
             "Liệt kê từng mô hình đã nạp kèm nhãn hiển thị, điểm số trên tập test và cờ "
             "đánh dấu mô hình tốt nhất — đọc thẳng từ <code class='inl'>metadata.json</code>, "
             "không mô hình nào bị liệt kê cứng trong mã"],
            ["<code class='inl'>/&lt;ứng-dụng&gt;/v1/predict</code>", "POST", "Nhận JSON, trả về dự đoán"],
        ]))
    r.p(
        "Luồng xử lý của một yêu cầu dự đoán:",
        '<div class="flow">JSON đầu vào → Kiểm tra hợp lệ → Tiền xử lý đã lưu → '
        "Mô hình → JSON đầu ra</div>",
        "Bước kiểm tra hợp lệ trả về mã 400 kèm thông báo tiếng Việt khi dữ liệu "
        "sai, thay vì để ngoại lệ lan xuống tầng mô hình và sinh ra mã 500.",
        "Phần thân JSON của <code class='inl'>POST /&lt;ứng-dụng&gt;/v1/predict</code> "
        "còn nhận thêm một trường tuỳ chọn, <code class='inl'>model</code>, cho phép "
        "người gọi chỉ định mô hình nào sẽ chạy dự đoán thay vì mặc định. Quy tắc chọn "
        "mô hình ở cả ba dịch vụ đều là <code class='inl'>model_id = payload.get('model') "
        "or BEST_MODEL</code> — một yêu cầu không kèm trường này, đúng hình dạng JSON từ "
        "trước khi tính năng chọn mô hình tồn tại, vẫn chạy đúng mô hình tốt nhất như cũ. "
        "Tham số mới vì vậy <strong>tương thích ngược hoàn toàn</strong>: không client "
        "cũ nào cần sửa để tiếp tục hoạt động đúng.",
    )
    r.p(figure("fig_deployment.png", "Kiến trúc triển khai hoàn chỉnh của hệ thống."))

    r.h(2, "6.5. Web Application")
    r.p(
        "Giao diện Web là một trang HTML tự chứa, nạp "
        "<code class='inl'>/metadata</code> khi khởi động để tự dựng danh sách mô "
        "hình và các danh sách hạng mục — nhờ vậy khi notebook được chạy lại và tập "
        "hạng mục thay đổi, giao diện tự cập nhật theo mà không cần sửa mã.",
        "Theo yêu cầu đề bài, mỗi giao diện đều có: biểu mẫu nhập liệu với kiểu điều "
        "khiển phù hợp, kiểm tra dữ liệu phía trình duyệt, nút dự đoán, kết quả dự "
        "đoán, độ tin cậy hoặc xác suất, và một câu diễn giải ngắn.",
    )

    r.h(3, "6.5.1. Sáu thành phần giao diện phục vụ khả năng diễn giải")
    r.p(
        "Trả về đúng một nhãn hoặc một con số là chưa đủ cho một hệ thống được gọi "
        "là “thông minh”: người dùng cần thấy dự đoán ấy dựa trên điều gì và đáng "
        "tin tới đâu, nếu không giao diện chỉ là một lớp vỏ bọc quanh một hộp đen. "
        "Vì vậy cả ba giao diện Web đều bổ sung sáu thành phần dưới đây, tất cả đều "
        "dựng động từ <code class='inl'>/metadata</code> — không mô hình, điểm số "
        "hay danh sách hạng mục nào bị gõ cứng trong HTML.",
    )
    r.p(table(
        ["#", "Thành phần", "Cách hoạt động", "Vì sao cần thiết"],
        [
            ["1", "Hộp chọn mô hình",
             "Nạp danh sách mô hình và mô hình tốt nhất từ "
             "<code class='inl'>/metadata</code>, chọn sẵn mô hình tốt nhất kèm nhãn "
             "“khuyến nghị”; người dùng có thể đổi sang mô hình khác rồi dự đoán lại "
             "trên cùng dữ liệu vừa nhập.",
             "Biến bảng so sánh mô hình ở Chương VII–IX từ một bảng số liệu tĩnh "
             "thành một thực nghiệm người dùng tự làm được: đổi mô hình, thấy kết "
             "quả đổi theo."],
            ["2", "Bảng so sánh mô hình",
             "Đọc trường <code class='inl'>test_metrics</code> trong "
             "<code class='inl'>/metadata</code>, hiển thị đúng bộ độ đo theo loại "
             "bài toán (Accuracy/Precision/Recall/F1/ROC-AUC cho phân loại; "
             "MAE/RMSE/R² cho hồi quy) và tô đậm hàng của mô hình đang chọn.",
             "Đưa đúng con số đã dùng để chọn mô hình ở Chương VII–IX ra giao diện, "
             "để người dùng tự đọc bằng chứng thay vì tin lời khẳng định “mô hình "
             "này tốt nhất”."],
            ["3", "Thanh độ tin cậy / thước đo khoảng giá",
             "Hai bài toán phân loại hiển thị một thanh ngang theo phần trăm xác "
             "suất mô hình gán cho nhãn dự đoán, đổi màu khi rủi ro cao hoặc không "
             "khuyến nghị. Ứng dụng giá nhà — hồi quy, không có xác suất — thay "
             "bằng một vạch định vị giá dự đoán trong đoạn [giá thấp, giá cao].",
             "Một con số trần trụi ngụ ý một sự chắc chắn không có thật. Hiển thị "
             "độ tin cậy hoặc khoảng dao động buộc người xem đọc kết quả đúng bản "
             "chất: một ước lượng, không phải một sự thật tuyệt đối."],
            ["4", "Biểu đồ mức độ quan trọng đặc trưng",
             "Ứng dụng 1 vẽ thanh ngang cho từng đặc trưng, kèm giá trị người dùng "
             "vừa nhập để đối chiếu trực tiếp. Ứng dụng 2 vẽ các đặc trưng trọng số "
             "cao nhất nhưng không đối chiếu giá trị nhập, vì phần lớn cột trọng số "
             "cao là cột sinh ra từ one-hot. Ứng dụng 3 thay biểu đồ bằng danh sách "
             "từ khoá tích cực/tiêu cực, tô đậm những từ thật sự xuất hiện trong "
             "bình luận vừa nhập.",
             "Giải thích <em>vì sao</em> mô hình ra kết quả đó, không chỉ ra kết "
             "quả — đúng yêu cầu diễn giải được của một hệ thống thông minh."],
            ["5", "Hồ sơ mẫu và nút xoá form",
             "Mỗi ứng dụng có 2–3 nút điền nhanh một hồ sơ mẫu hợp lý về mặt lâm "
             "sàng / thị trường / văn phong đánh giá (không lấy từ bản ghi thật "
             "nào), cộng một nút xoá trắng form.",
             "Hạ thấp rào cản kiểm thử thủ công và chụp ảnh minh hoạ — không phải "
             "gõ tay từng trường mỗi lần muốn thử một kịch bản khác."],
            ["6", "Câu diễn giải bằng lời",
             "REST API soạn sẵn một câu tiếng Việt theo lớp dự đoán (hoặc theo giá "
             "và khoảng dao động ở Ứng dụng 2) và trả về qua trường "
             "<code class='inl'>interpretation</code>; giao diện chỉ hiển thị "
             "nguyên văn, không tự suy diễn thêm.",
             "Cầu nối giữa con số kỹ thuật (xác suất, nhãn lớp, giá dự đoán) và "
             "người đọc không chuyên — đúng tinh thần “hệ thống thông minh” chứ "
             "không chỉ là một API trả JSON."],
        ]))
    r.p(note(
        "Vì sao cả sáu đều đọc dữ liệu, không gõ cứng.",
        "Không thành phần nào trong sáu thành phần trên chứa một con số hay một "
        "tên mô hình gõ tay trong HTML — tất cả đều dựng động từ "
        "<code class='inl'>/metadata</code> lúc trang tải. Notebook chạy lại sinh "
        "ra điểm số mới, tập hạng mục mới hay tầm quan trọng đặc trưng mới thì "
        "giao diện tự phản ánh đúng, không cần sửa một dòng HTML/JavaScript nào — "
        "đúng nguyên tắc một nguồn sự thật duy nhất đã áp dụng cho "
        "<code class='inl'>metadata.json</code> ở mục 6.2."))

    r.h(2, "6.6. Triển khai trên ứng dụng mobile")
    r.p(
        "Ứng dụng Mobile đóng vai trò <strong>máy khách của dịch vụ dự đoán</strong>, "
        "không chứa mô hình và không huấn luyện. Kiến trúc:",
        '<div class="flow">Giao diện Mobile → REST API → Tiền xử lý → Mô hình học '
        "máy → Kết quả dự đoán → Giao diện Mobile</div>",
        "Báo cáo này minh hoạ giao diện Mobile bằng một trang mô phỏng thiết bị "
        "chạy trong trình duyệt tại điểm cuối <code class='inl'>/mobile</code>: một "
        "khung điện thoại 390×844 với thanh trạng thái, thanh tiêu đề ứng dụng và "
        "biểu mẫu bố cục theo chuẩn di động. Trang này gọi <strong>đúng điểm cuối "
        "REST</strong> mà giao diện Web gọi, nên nó chứng minh được đầy đủ luồng "
        "Mobile → REST API → mô hình mà không cần trình giả lập Android.",
        "Toàn bộ sáu thành phần giao diện ở mục 6.5.1 — hộp chọn mô hình, bảng so "
        "sánh mô hình, thanh độ tin cậy hoặc thước đo khoảng giá, biểu đồ hay danh "
        "sách từ khoá mức độ quan trọng, hồ sơ mẫu kèm nút xoá, và câu diễn giải — "
        "đều có mặt trên bản Mobile, chỉ khác nhau ở cách bố trí cho một màn hình "
        "390×844 hẹp hơn nhiều so với màn hình Web. Riêng bảng so sánh mô hình "
        "được đặt trong một khối <code class='inl'>&lt;details&gt;</code> thu gọn "
        "(nhấn vào tiêu đề để mở), thay vì hiển thị thường trực như trên Web, để "
        "không chiếm hết không gian màn hình vốn đã eo hẹp của thiết bị di động.",
        "Mã nguồn Flutter (<code class='inl'>mobile/lib/main.dart</code> và "
        "<code class='inl'>mobile/pubspec.yaml</code>) cũng được kèm trong kho mã "
        "cho mỗi ứng dụng.",
    )

    r.h(2, "6.7. Kiến trúc hoàn chỉnh của hệ thống")
    r.p(
        "Ghép mọi thành phần lại, mỗi ứng dụng có đường đi trọn vẹn sau:",
        '<div class="flow">Dữ liệu thực → Làm sạch → Biểu diễn (vectơ / ma trận / '
        "tensor) → Mô hình học máy → Đánh giá → Lưu trữ → Web API → Ứng dụng Mobile</div>",
        "Ba chương tiếp theo trình bày từng ứng dụng theo đúng khung này, để Chương X "
        "so sánh được chúng trên cùng một hệ quy chiếu.",
    )

    # Khối này trước đây là “Phụ lục B” tách rời ở cuối báo cáo. Nó thuộc về
    # chương triển khai: người đọc cần biết cách tái lập ngay sau khi đọc phần
    # lưu trữ và triển khai, chứ không phải sau ba chương ứng dụng.
    r.h(2, "6.8. Thông tin tái lập thực nghiệm")
    r.p(
        "Đề bài yêu cầu một sinh viên khác phải tái lập được thí nghiệm chính và chạy "
        "được ứng dụng đã triển khai. Bảng dưới liệt kê toàn bộ thông tin cần thiết.",
    )
    r.p(table(
        ["Hạng mục", "Giá trị"],
        [
            ["Ngôn ngữ", "Python 3.13.7"],
            ["Hệ điều hành", "Windows 11"],
            ["Hạt giống ngẫu nhiên", f"<code class='inl'>RANDOM_SEED = {D['random_seed']}</code> "
             "— đặt cho <code class='inl'>numpy</code>, <code class='inl'>random</code> "
             "và mọi mô hình có tham số <code class='inl'>random_state</code>"],
            ["Thư viện chính", "scikit-learn 1.9, pandas 3.0, numpy 2.5, "
             "matplotlib 3.11, seaborn 0.13, joblib 1.5, Flask 3.1, neo4j 6.2"],
            ["Chia tập", "70 / 15 / 15; phân tầng theo nhãn ở hai bài toán phân loại"],
            ["Kho mã nguồn", f"<a href='{GITHUB_URL}'>{GITHUB_URL}</a>"],
        ]))
    r.p(table(
        ["Ứng dụng", "Nguồn dữ liệu Kaggle", "Tệp cục bộ"],
        [
            ["Tiểu đường", "uciml/pima-indians-diabetes-database",
             "<code class='inl'>diabetes/data/diabetes.csv</code>"],
            ["Giá nhà", "nguyentiennhan/vietnam-housing-dataset-2024",
             "<code class='inl'>house_price/data/house_prices.csv</code>"],
            ["Thương mại điện tử", "nicapotato/womens-ecommerce-clothing-reviews",
             "<code class='inl'>customer_behavior/data/womens_ecommerce_reviews.csv</code>"],
        ]))
    r.h(3, "6.8.1. Các bước tái lập")
    r.p(code(
        "# 1. Cài môi trường\n"
        "python -m venv .venv\n"
        ".venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n\n"
        "# 2. Chạy lại ba notebook — sinh lại toàn bộ mô hình, hình vẽ và metadata\n"
        "jupyter nbconvert --to notebook --execute --inplace \\\n"
        "    diabetes/notebook/Diabetes_A02.ipynb\n"
        "jupyter nbconvert --to notebook --execute --inplace \\\n"
        "    house_price/notebook/HousePrice_A02.ipynb\n"
        "jupyter nbconvert --to notebook --execute --inplace \\\n"
        "    customer_behavior/notebook/CustomerBehavior_A02.ipynb\n\n"
        "# 3. Chạy dịch vụ (mỗi lệnh một cửa sổ terminal)\n"
        "python diabetes/api/REST_API.py            # cổng 5001\n"
        "python house_price/api/REST_API.py         # cổng 5002\n"
        "python customer_behavior/api/REST_API.py   # cổng 5003\n\n"
        "# 4. Mở giao diện\n"
        "#    Web    : http://127.0.0.1:5001/web\n"
        "#    Mobile : http://127.0.0.1:5001/mobile\n\n"
        "# 5. Dựng lại báo cáo này\n"
        "python report/build_report.py",
        "Quy trình tái lập đầy đủ.", "Bash"))
    r.p(note(
        "Về tính xác định của kết quả.",
        "Toàn bộ mô hình dùng chung một hạt giống ngẫu nhiên nên chạy lại cho kết quả "
        "trùng khớp. Riêng các mô hình chạy song song "
        "(<code class='inl'>n_jobs=-1</code>) có thể lệch ở chữ số thập phân thứ tư "
        "do thứ tự cộng dồn dấu phẩy động khác nhau giữa các luồng; sai khác này "
        "không ảnh hưởng tới thứ hạng mô hình hay bất kỳ kết luận nào trong báo cáo."))
