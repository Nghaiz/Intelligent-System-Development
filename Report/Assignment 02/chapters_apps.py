"""Chương VII–IX: ba ứng dụng, mỗi ứng dụng theo cùng một khung mười mục."""

from build_report import (META, code, figure, fmt, metric_rows, note,
                          notebook_code, notebook_output, oim, output, table)


def write(r) -> None:
    _ch7(r)
    _ch8(r)
    _ch9(r)


def _ui_screenshots(r, sec: str, key: str, port: int, path: str, name: str) -> None:
    """Ảnh chụp giao diện Web + Mobile, đặt ngay trong mục Triển khai của từng chương.

    Trước đây khối này nằm ở một phụ lục tách rời, khiến người đọc phải lật đi lật
    lại giữa chương ứng dụng và cuối báo cáo. Đặt tại chỗ thì mô tả kỹ thuật và ảnh
    minh hoạ của cùng một hệ thống nằm cạnh nhau.
    """
    r.h(3, f"{sec}.1. Giao diện Web")
    r.p(
        "Ảnh dưới đây được chụp tự động từ dịch vụ đang chạy thật, bằng trình duyệt "
        "điều khiển qua kịch bản "
        "(<code class='inl'>report/capture_screenshots.py</code>) — không phải ảnh "
        "dựng lại bằng công cụ thiết kế.",
    )
    r.p(figure(f"shot_{key}_web_input.png", f"{name} — giao diện Web, màn hình nhập liệu."))
    r.p(figure(f"shot_{key}_web_result.png", f"{name} — giao diện Web, màn hình kết quả dự đoán."))

    r.h(3, f"{sec}.2. Giao diện Mobile")
    r.p(
        "Giao diện Mobile là trang mô phỏng thiết bị 390×844 phục vụ tại điểm cuối "
        f"<code class='inl'>http://127.0.0.1:{port}/mobile</code>. Trang này gọi "
        f"<strong>đúng điểm cuối</strong> <code class='inl'>POST /{path}/v1/predict</code> "
        "mà giao diện Web gọi, nên nó minh hoạ đầy đủ luồng "
        "<em>Mobile → REST API → mô hình</em> mà không cần trình giả lập Android. "
        "Mã nguồn Flutter tương ứng nằm ở "
        f"<code class='inl'>{path.replace('-', '_')}/mobile/lib/main.dart</code>.",
    )
    r.p(figure(f"shot_{key}_mobile_input.png",
               f"{name} — giao diện Mobile, màn hình nhập liệu.", "58%"))
    r.p(figure(f"shot_{key}_mobile_result.png",
               f"{name} — giao diện Mobile, màn hình kết quả dự đoán.", "58%"))


# -------------------------------------------------------------- Chương VII
def _ch7(r) -> None:
    m = META["diabetes"]
    r.h(1, "Chương VII. Ứng dụng 1 — Dự đoán bệnh tiểu đường")

    r.h(2, "7.1. Mô tả bài toán")
    r.p(
        "Mục tiêu của ứng dụng này là dự đoán một bệnh nhân có thuộc lớp dương tính "
        "tiểu đường hay không, dựa trên các chỉ số lâm sàng thu được trong một lần "
        "khám. Kết quả dự đoán có thể hỗ trợ việc quyết định bệnh nhân nào cần được "
        "chỉ định làm nghiệm pháp dung nạp glucose chuyên sâu.",
        "<ul><li><em>X</em> = đặc trưng bệnh nhân (các chỉ số lâm sàng và nhân trắc học);</li>"
        "<li><em>y</em> = lớp tiểu đường, <em>y</em> ∈ {0, 1}.</li></ul>",
        "Đây là bài toán <strong>phân loại nhị phân có giám sát</strong>.",
    )
    r.p(note(
        "Mô hình này làm gì và không làm gì.",
        "Tiểu đường type 2 tiến triển âm thầm nhiều năm trước khi có triệu chứng. "
        "Một mô hình sàng lọc chạy trên các chỉ số xét nghiệm thường quy không thay "
        "thế chẩn đoán của bác sĩ — nó <strong>phân bổ nguồn lực chẩn đoán</strong>. "
        "Nhận định này quyết định trực tiếp việc chọn độ đo ở mục 7.8: bỏ sót một ca "
        "bệnh đắt hơn nhiều so với cảnh báo nhầm một người khoẻ."))

    r.h(2, "7.2. Giới thiệu tập dữ liệu")
    r.p(table(
        ["Mục", "Giá trị"],
        [
            ["Tên tập dữ liệu", "Pima Indians Diabetes Database"],
            ["Nguồn Kaggle",
             "<a href='https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database'>"
             "kaggle.com/datasets/uciml/pima-indians-diabetes-database</a>"],
            ["Nguồn gốc", "National Institute of Diabetes and Digestive and Kidney Diseases (UCI)"],
            ["Số quan sát", f"{m['n_samples']:,}"],
            ["Số thuộc tính", "9 (8 đặc trưng + 1 biến mục tiêu)"],
            ["Biến mục tiêu", "<code class='inl'>Outcome</code> (0 = âm tính, 1 = dương tính)"],
            ["Tệp cục bộ", "<code class='inl'>diabetes/data/diabetes.csv</code>"],
        ]))
    r.p(
        "<strong>Một quan sát biểu diễn cái gì.</strong> Mỗi dòng là một lần khám "
        "của một bệnh nhân nữ, từ 21 tuổi trở lên, thuộc cộng đồng người Pima ở "
        "bang Arizona, Hoa Kỳ. Ràng buộc nhân khẩu học này là một giới hạn thật của "
        "mô hình: cộng đồng Pima có tỷ lệ mắc tiểu đường type 2 cao bậc nhất thế "
        "giới, nên ngưỡng quyết định học được ở đây không thể suy rộng thẳng cho "
        "quần thể khác.",
    )
    r.p(table(
        ["Đặc trưng", "Kiểu", "Ý nghĩa"],
        [
            ["Pregnancies", "Số nguyên", "Số lần mang thai"],
            ["Glucose", "Số thực", "Nồng độ glucose huyết tương sau 2 giờ (mg/dL)"],
            ["BloodPressure", "Số thực", "Huyết áp tâm trương (mm Hg)"],
            ["SkinThickness", "Số thực", "Độ dày nếp gấp da cơ tam đầu (mm)"],
            ["Insulin", "Số thực", "Insulin huyết thanh sau 2 giờ (μU/mL)"],
            ["BMI", "Số thực", "Chỉ số khối cơ thể (kg/m²)"],
            ["DiabetesPedigreeFunction", "Số thực", "Điểm số tiền sử gia đình về tiểu đường"],
            ["Age", "Số nguyên", "Tuổi (năm)"],
            ["<strong>Outcome</strong>", "Nhị phân", "<strong>Biến mục tiêu</strong>: 0 âm tính, 1 dương tính"],
        ]))
    r.p(
        "Toàn bộ tám đặc trưng đều là <strong>số</strong>. Không có biến phân loại, "
        "không có cột văn bản. Vì vậy bước mã hoá trong pipeline chuẩn là rỗng ở "
        "ứng dụng này, và toàn bộ công sức tiền xử lý dồn vào điền khuyết và chuẩn "
        "hoá thang đo. Đây là điểm khác biệt lớn so với hai ứng dụng sau.",
    )

    r.h(2, "7.3. Khảo sát và tìm hiểu dữ liệu")
    r.p(code(notebook_code("diabetes", "df.isna().sum()"),
             "Khảo sát giá trị thiếu, trùng lặp và phân bố biến mục tiêu."))
    r.p(output(notebook_output("diabetes", "df.isna().sum()", 22)))
    r.p(note(
        "Kết quả này gây hiểu lầm nghiêm trọng.",
        "<code class='inl'>isna().sum()</code> trả về 0 ở mọi cột, và không có bản "
        "ghi trùng lặp nào. Nếu dừng ở đây, kết luận sẽ là “dữ liệu hoàn toàn sạch”. "
        "<strong>Kết luận đó sai.</strong> Mục 7.4 chỉ ra vì sao.", "bad"))

    r.h(2, "7.4. Làm sạch dữ liệu")
    r.h(3, "7.4.1. Giá trị thiếu bị mã hoá thành số 0")
    r.p(
        "Tập Pima mã hoá giá trị thiếu bằng số <strong>0</strong> chứ không bằng "
        "<code class='inl'>NaN</code>. Về mặt sinh lý học, năm cột dưới đây không "
        "thể bằng 0 trên một người còn sống: <code class='inl'>Glucose</code>, "
        "<code class='inl'>BloodPressure</code>, <code class='inl'>SkinThickness</code>, "
        "<code class='inl'>Insulin</code>, <code class='inl'>BMI</code>. "
        "Ngược lại <code class='inl'>Pregnancies = 0</code> là hoàn toàn hợp lệ.",
    )
    r.p(output(notebook_output("diabetes", "INVALID_ZERO = [", 14),
               "Tỷ lệ giá trị thiếu thật, sau khi nhận ra số 0 là mã của giá trị thiếu."))
    r.p(figure("dia_missing.png", "Ứng dụng 1 — Giá trị thiếu bị mã hoá thành số 0.", "72%"))
    r.p(oim(
        "<code class='inl'>Insulin</code> thiếu 48,7% và "
        "<code class='inl'>SkinThickness</code> thiếu 29,6% — gần một nửa cột "
        "<code class='inl'>Insulin</code> là dữ liệu bịa. "
        "<code class='inl'>Glucose</code> và <code class='inl'>BMI</code> chỉ thiếu "
        "dưới 1,5%.",
        "Insulin huyết thanh 2 giờ đòi hỏi nghiệm pháp dung nạp glucose đường uống "
        "kéo dài, tốn thời gian và chi phí, nên nhiều bệnh nhân không được làm. Đây "
        "là dạng thiếu <strong>có hệ thống</strong>, không phải ngẫu nhiên.",
        "Không được xoá dòng — xoá theo <code class='inl'>Insulin</code> sẽ mất gần "
        "một nửa tập dữ liệu vốn chỉ có 768 dòng. Cũng không được điền bằng trung "
        "bình tính trên toàn bộ dữ liệu, vì trung bình ấy đã nhìn thấy tập test. "
        "Cách xử lý đúng là đưa việc điền khuyết vào bên trong "
        "<code class='inl'>Pipeline</code> (mục 7.5)."))
    r.p(note(
        "Đây là bài học quan trọng nhất của Ứng dụng 1.",
        "Một giá trị không hợp lệ có thể đội lốt giá trị hợp lệ. "
        "<code class='inl'>describe()</code> vẫn tính trung bình bình thường, không "
        "hàm nào báo lỗi, và mô hình vẫn huấn luyện xong — chỉ là học sai. "
        "<strong>Chất lượng dữ liệu là việc của người đọc dữ liệu, không phải của "
        "<code class='inl'>isna()</code>.</strong>", "warn"))

    r.h(3, "7.4.2. Phân tích ngoại lệ")
    r.p(figure("dia_boxplot.png",
               "Ứng dụng 1 — Hộp râu tám đặc trưng, sau khi loại giá trị 0 không hợp lệ."))
    r.p(oim(
        "<code class='inl'>Insulin</code> và "
        "<code class='inl'>DiabetesPedigreeFunction</code> có tỷ lệ ngoại lệ cao "
        "nhất, phần lớn nằm ở đuôi phải.",
        "Insulin huyết thanh có phân phối lệch phải rất mạnh trong sinh lý người: "
        "người kháng insulin nặng có thể cao gấp nhiều lần trung vị. "
        "<code class='inl'>DiabetesPedigreeFunction</code> là điểm số di truyền tổng "
        "hợp, cũng lệch phải theo thiết kế.",
        "Đây là <strong>ngoại lệ thật, không phải lỗi nhập liệu</strong> — chúng "
        "mang đúng tín hiệu bệnh lý mà ta muốn mô hình học, nên không cắt bỏ. Thay "
        "vào đó chọn hai biện pháp phòng thủ: điền khuyết bằng trung vị (bền với "
        "đuôi dài) và chuẩn hoá bằng <code class='inl'>StandardScaler</code> để KNN "
        "và SVM không bị một cột đuôi dài chi phối."))

    r.h(2, "7.5. Biểu diễn dữ liệu")
    r.p(
        "Chuỗi biến đổi của ứng dụng này:",
        '<div class="flow">CSV → DataFrame → ma trận đặc trưng sạch → ma trận đã '
        "điền khuyết và chuẩn hoá → đầu vào mô hình</div>",
        "Một bệnh nhân trở thành một vectơ đặc trưng "
        "<em>x<sub>i</sub></em> ∈ ℝ<sup>d</sup>, và toàn bộ tập dữ liệu trở thành "
        "ma trận <em>X</em> ∈ ℝ<sup>N×d</sup>.",
    )
    r.p(code(notebook_code("diabetes", "BƯỚC 1 — MỘT BẢN GHI CSV GỐC"),
             "Mã in ra bản ghi CSV gốc, vectơ đặc trưng tương ứng và hình dạng ma trận."))
    r.p(output(notebook_output("diabetes", "BƯỚC 1 — MỘT BẢN GHI CSV GỐC", 34),
               "Bản ghi CSV gốc, vectơ đặc trưng và hình dạng ma trận — ba thứ đề bài bắt buộc trình bày."))
    r.p(table(
        ["Hạng mục", "Giá trị"],
        [
            ["Hình dạng DataFrame gốc", f"({m['n_samples']}, 9)"],
            ["Hình dạng ma trận đặc trưng", f"<em>X</em> ∈ ℝ<sup>{m['feature_matrix_shape'][0]}×"
             f"{m['feature_matrix_shape'][1]}</sup>"],
            ["<em>N</em> (số mẫu)", str(m["feature_matrix_shape"][0])],
            ["<em>d</em> (số đặc trưng)", str(m["feature_matrix_shape"][1])],
            ["Kiểu dữ liệu", "<code class='inl'>float64</code>"],
            ["Mã hoá đặc trưng", "Không cần — toàn bộ đặc trưng đều là số"],
            ["Chuẩn hoá", "<code class='inl'>StandardScaler</code> — trung bình 0, độ lệch chuẩn 1"],
            ["Hình dạng đầu vào mô hình cuối cùng",
             f"({m['split']['train']}, {m['feature_matrix_shape'][1]}) cho tập train"],
        ]))

    r.h(3, "7.5.1. Vì sao chỉ dùng 5 trong 8 đặc trưng")
    r.p(
        "<code class='inl'>Insulin</code> thiếu 48,7% và "
        "<code class='inl'>SkinThickness</code> thiếu 29,6% — điền khuyết cho gần "
        "một nửa số dòng nghĩa là bịa ra gần một nửa cột đó. "
        "<code class='inl'>BloodPressure</code> thì gần như không tách được hai lớp "
        "trong khi vẫn thiếu 4,6%. Ba cột bị loại vì cùng một lý do thực tế: "
        "<strong>chi phí thu thập cao, đóng góp thông tin thấp</strong>.",
        "Nhưng đây là một quyết định, không phải một sự thật hiển nhiên, nên nó phải "
        "được đo:",
    )
    r.p(output(notebook_output("diabetes", "CÁI GIÁ CỦA VIỆC RÚT GỌN", 14),
               "Đo cái giá của việc rút gọn từ 8 xuống 5 đặc trưng."))
    r.p(
        "Chênh lệch ROC-AUC trung bình giữa hai phương án rất nhỏ. Ba đặc trưng bị "
        "loại gần như không mang thêm thông tin dự báo, trong khi chúng chiếm phần "
        "lớn giá trị thiếu và đòi hỏi xét nghiệm tốn kém. Nhờ vậy giao diện Web ở "
        "mục 7.10 chỉ hỏi năm con số — người dùng điền được trong 20 giây, thay vì "
        "phải có sẵn kết quả xét nghiệm insulin 2 giờ.",
    )

    r.h(3, "7.5.2. Pipeline tiền xử lý")
    r.p(code(notebook_code("diabetes", "preprocessor = Pipeline(["),
             "Pipeline ba bước của Ứng dụng 1, chỉ fit trên tập train."))
    r.p(output(notebook_output("diabetes", "Trung vị học được từ tập train", 18)))
    r.p(
        "Ba bước và lý do của từng bước: (1) "
        "<code class='inl'>FunctionTransformer</code> thay giá trị 0 không hợp lệ "
        "bằng <code class='inl'>NaN</code>; (2) "
        "<code class='inl'>SimpleImputer(strategy=\"median\")</code> điền khuyết "
        "bằng trung vị của tập train — chọn trung vị vì các cột đều lệch phải; "
        "(3) <code class='inl'>StandardScaler</code> đưa mỗi cột về trung bình 0 và "
        "độ lệch chuẩn 1, bắt buộc với KNN và SVM.",
    )

    r.h(2, "7.6. Phân tích khám phá dữ liệu (EDA)")
    r.p(figure("dia_target.png", "Ứng dụng 1 — Phân bố biến mục tiêu và mức mất cân bằng lớp."))
    r.p(oim(
        "500 ca âm tính so với 268 ca dương tính — tỷ lệ khoảng 1,87 : 1 "
        "(65,1% và 34,9%).",
        "Mất cân bằng nhẹ. Không nghiêm trọng như bài toán phát hiện gian lận "
        "(thường 1000 : 1), nhưng đủ để làm hỏng một độ đo.",
        "Một mô hình luôn đoán “âm tính” đạt "
        f"{fmt(m['baseline_accuracy'] * 100, 1)}% accuracy mà không học được gì. Đó "
        "chính là baseline ở mục 7.7, và mọi mô hình phải vượt mốc ấy mới được coi "
        "là có giá trị. Hệ quả thứ hai: accuracy không đủ để xếp hạng mô hình ở bài "
        "toán này."))

    r.p(figure("dia_dist_by_class.png",
               "Ứng dụng 1 — Phân phối từng đặc trưng theo hai lớp."))
    r.p(oim(
        "<code class='inl'>Glucose</code> tách hai lớp rõ rệt nhất: đỉnh nhóm âm "
        "tính quanh 110 mg/dL, nhóm dương tính quanh 140 mg/dL. "
        "<code class='inl'>BMI</code> và <code class='inl'>Age</code> tách vừa phải. "
        "<code class='inl'>BloodPressure</code> và "
        "<code class='inl'>SkinThickness</code> gần như chồng lên nhau.",
        "Kết quả khớp với y văn — glucose huyết tương chính là tiêu chuẩn chẩn đoán "
        "tiểu đường, nên nó phải là biến tách lớp mạnh nhất. Việc mô hình “phát hiện "
        "lại” điều đã biết là một tín hiệu tốt: dữ liệu và pipeline không bị hỏng ở "
        "đâu đó.",
        "<code class='inl'>Glucose</code> sẽ chi phối mọi mô hình, còn "
        "<code class='inl'>BloodPressure</code> và "
        "<code class='inl'>SkinThickness</code> đóng góp rất ít — điều này được xác "
        "nhận lại ở biểu đồ tầm quan trọng đặc trưng (mục 7.8) và là căn cứ cho "
        "quyết định rút gọn ở mục 7.5.1."))

    r.p(figure("dia_corr.png", "Ứng dụng 1 — Ma trận tương quan Pearson.", "68%"))
    r.p(oim(
        "Tương quan với <code class='inl'>Outcome</code> mạnh nhất là "
        "<code class='inl'>Glucose</code> (≈ 0,49), sau đó "
        "<code class='inl'>BMI</code> (≈ 0,31) và <code class='inl'>Age</code> "
        "(≈ 0,24). Giữa các đặc trưng, cặp mạnh nhất là "
        "<code class='inl'>Age</code>–<code class='inl'>Pregnancies</code> (≈ 0,54).",
        "Không cặp nào vượt 0,8, tức là <strong>không có đa cộng tuyến nghiêm trọng</strong>. "
        "Cặp <code class='inl'>Age</code>–<code class='inl'>Pregnancies</code> cao là "
        "hiển nhiên về mặt nhân khẩu học; cặp "
        "<code class='inl'>SkinThickness</code>–<code class='inl'>BMI</code> cao vì "
        "cả hai đều đo lượng mỡ cơ thể.",
        "Logistic Regression dùng được mà không cần loại cột, hệ số vẫn diễn giải "
        "được. Nếu có cặp vượt 0,9 thì hệ số hồi quy sẽ mất ổn định và ta đã phải "
        "bỏ bớt một cột."))

    r.h(2, "7.7. Xây dựng mô hình")
    r.h(3, "7.7.1. Chia tập Train, Validation và Test")
    r.p(output(notebook_output("diabetes", "split_tbl = pd.DataFrame", 14)))
    r.p(
        "Tỷ lệ dương tính ở cả ba tập đều xấp xỉ 34,9% — phép phân tầng "
        "(<code class='inl'>stratify=y</code>) đã làm đúng việc. Không phân tầng, "
        "tập test chỉ có 116 mẫu có thể lệch tỷ lệ lớp vài phần trăm so với train, "
        "và khi ấy điểm test đo lẫn cả sự lệch phân bố lẫn chất lượng mô hình.",
    )

    r.h(3, "7.7.2. Mô hình cơ sở")
    r.p(output(notebook_output("diabetes", "BASELINE — luôn đoán", 10)))
    r.p(
        f"Baseline đạt accuracy {fmt(m['baseline_accuracy'] * 100, 1)}% nhưng "
        "<strong>Recall = 0</strong> — nó bỏ sót 100% bệnh nhân tiểu đường. Con số "
        "này minh hoạ chính xác vì sao accuracy một mình là độ đo gây hiểu lầm ở bài "
        "toán mất cân bằng.",
    )

    r.h(3, "7.7.3. Năm mô hình phân loại")
    r.p(code(notebook_code("diabetes", "MODELS = {", drop_comment_header=True),
             "Định nghĩa năm mô hình phân loại của Ứng dụng 1."))
    r.p(table(
        ["Mô hình", "Họ", "Vì sao đưa vào"],
        [
            ["Logistic Regression", "Tuyến tính",
             "Chuẩn tham chiếu trong y học; hệ số diễn giải được thành tỷ số chênh"],
            ["K-Nearest Neighbors", "Dựa trên khoảng cách",
             "Không tham số; kiểm chứng xem bệnh nhân giống nhau có cùng nhãn không"],
            ["Decision Tree", "Dạng cây", "Sinh ra luật đọc được, bác sĩ kiểm tra được"],
            ["Random Forest", "Tập hợp cây", "Giảm phương sai của cây đơn; cho tầm quan trọng đặc trưng"],
            ["SVM (RBF)", "Biên lớn", "Bắt ranh giới phi tuyến; cần dữ liệu đã chuẩn hoá"],
        ]))
    r.p(
        "<code class='inl'>class_weight=\"balanced\"</code> được bật ở bốn trong năm "
        "mô hình: nó nhân trọng số lỗi của lớp thiểu số lên, đẩy mô hình ưu tiên "
        "<strong>không bỏ sót ca bệnh</strong> — đúng với ưu tiên lâm sàng đã nêu ở "
        "mục 7.1.",
    )

    r.h(2, "7.8. Đánh giá mô hình")
    r.p(
        "<strong>Độ đo nào quan trọng nhất cho bài toán này? — Recall của lớp dương "
        "tính.</strong> Bốn ô của ma trận nhầm lẫn có ý nghĩa lâm sàng rất khác nhau:",
    )
    r.p(table(
        ["Ô", "Ý nghĩa lâm sàng", "Cái giá"],
        [
            ["TP", "Có bệnh, được cảnh báo đúng", "Đúng ý đồ hệ thống"],
            ["TN", "Không bệnh, không bị cảnh báo", "Đúng"],
            ["FP", "Không bệnh nhưng bị cảnh báo nhầm", "Một lần xét nghiệm xác nhận thừa — phiền, rẻ"],
            ["<strong>FN</strong>", "<strong>Có bệnh nhưng bị bỏ sót</strong>",
             "<strong>Bệnh tiến triển âm thầm nhiều năm — đắt, có khi không hồi phục</strong>"],
        ]))
    keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rows, _ = metric_rows("diabetes", keys, "ROC-AUC")
    r.p(table(["Mô hình"] + keys, rows,
              "Ứng dụng 1 — So sánh năm mô hình trên tập test, sắp theo ROC-AUC.", "num"))
    r.p(figure("dia_model_comparison.png", "Ứng dụng 1 — So sánh năm mô hình trên tập test."))
    r.p(figure("dia_confusion.png", "Ứng dụng 1 — Ma trận nhầm lẫn của năm mô hình trên tập test."))
    r.p(figure("dia_roc.png", "Ứng dụng 1 — Đường cong ROC trên tập test.", "62%"))

    r.h(3, "7.8.1. Diễn giải ma trận nhầm lẫn")
    r.p(output(notebook_output("diabetes", "bệnh nhân có bệnh, được cảnh báo đúng", 16),
               "Diễn giải bốn ô của ma trận nhầm lẫn sang ngôn ngữ lâm sàng."))
    cm = m["confusion_matrix_best"]
    r.p(
        f"Trên 116 mẫu test, mô hình {m['best_model_label']} bắt đúng {cm['tp']} ca "
        f"bệnh và bỏ sót {cm['fn']} ca; đồng thời cảnh báo nhầm {cm['fp']} người "
        f"khoẻ. Recall đạt {fmt(cm['tp'] / (cm['tp'] + cm['fn']), 4)}, nghĩa là hệ "
        f"thống bắt được {fmt(cm['tp'] / (cm['tp'] + cm['fn']) * 100, 1)}% số ca "
        "bệnh thật.",
    )

    r.h(3, "7.8.2. Phân tích sai số")
    r.p(figure("dia_importance.png",
               "Ứng dụng 1 — Tầm quan trọng đặc trưng theo Random Forest.", "68%"))
    r.p(output(notebook_output("diabetes", "Glucose trung bình theo nhóm", 8)))
    r.p(oim(
        "Các ca bị bỏ sót (FN) có <code class='inl'>Glucose</code> trung bình thấp "
        "hơn hẳn các ca được bắt đúng, và nằm gần vùng giá trị của nhóm âm tính.",
        "Đây là những bệnh nhân đã mắc bệnh nhưng chưa biểu hiện tăng đường huyết rõ "
        "— giai đoạn sớm, hoặc đang được kiểm soát bằng chế độ ăn. Với chỉ năm chỉ số "
        "tại một thời điểm, không mô hình nào phân biệt được họ với người khoẻ: "
        "thông tin cần thiết (HbA1c, tiền sử theo thời gian) không nằm trong dữ liệu.",
        "Đây là <strong>giới hạn của biểu diễn dữ liệu, không phải của thuật toán</strong>. "
        "Đổi sang mô hình mạnh hơn sẽ không sửa được; chỉ có thêm đặc trưng mới sửa "
        "được. Đúng luận điểm trung tâm của Assignment 02."))

    r.h(2, "7.9. Lựa chọn mô hình")
    r.p(table(
        ["Tiêu chí", "Nhận định"],
        [
            ["Hiệu năng dự báo", "Xếp theo ROC-AUC trên tập test — bảng ở mục 7.8"],
            ["Khả năng diễn giải", "Logistic Regression và Decision Tree đọc được; SVM RBF là hộp đen"],
            ["Chi phí tính toán", "Mọi mô hình huấn luyện dưới 1 giây với <em>N</em> = 768"],
            ["Độ bền", "Random Forest bền nhất với ngoại lệ nhờ trung bình hoá nhiều cây"],
            ["Ràng buộc triển khai", "Cả năm đều tuần tự hoá được bằng joblib, kích thước nhỏ"],
        ]))
    best = m["test_metrics"][m["best_model"]]
    r.p(note(
        f"Mô hình được chọn: {m['best_model_label']}.",
        f"ROC-AUC = {fmt(best['ROC-AUC'])}, Recall = {fmt(best['Recall'])}, "
        f"F1 = {fmt(best['F1'])}, Accuracy = {fmt(best['Accuracy'])} — vượt xa "
        f"baseline {fmt(m['baseline_accuracy'])}. Ở quy mô dữ liệu này chi phí tính "
        "toán không phân biệt được các phương án, nên quyết định thực chất nằm ở "
        "hiệu năng và độ bền, và Random Forest thắng ở cả hai.", "good"))

    r.h(2, "7.10. Triển khai hệ thống")
    r.p(
        "Mô hình được triển khai dưới dạng <strong>Web Service + ứng dụng Mobile</strong>. "
        "Luồng xử lý:",
        '<div class="flow">Nhập thông tin bệnh nhân → Kiểm tra hợp lệ → Tiền xử lý → '
        "Mô hình → Dự đoán + độ tin cậy</div>",
    )
    r.p(table(
        ["Hạng mục", "Giá trị"],
        [
            ["Khung web", "Flask 3.1"],
            ["Điểm cuối", "<code class='inl'>POST /diabetes/v1/predict</code>"],
            ["Cổng", "5001"],
            ["Biến đầu vào", ", ".join(f"<code class='inl'>{c}</code>" for c in m["feature_columns"])],
            ["Quy tắc kiểm tra",
             "Đủ trường bắt buộc; giá trị phải là số hữu hạn; Glucose và BMI phải lớn "
             "hơn 0; Age lớn hơn 0; Pregnancies không âm"],
            ["Tiền xử lý sử dụng",
             "<code class='inl'>preprocessor.joblib</code> — nạp lại, chỉ gọi "
             "<code class='inl'>transform</code>"],
            ["Mô hình nạp", "Cả năm mô hình; mặc định "
             f"<code class='inl'>{m['best_model']}</code>"],
        ]))
    r.p(code(
        '{\n'
        '  "Glucose": 168,\n'
        '  "BMI": 38.2,\n'
        '  "Age": 52,\n'
        '  "Pregnancies": 6,\n'
        '  "DiabetesPedigreeFunction": 0.85,\n'
        '  "model": "' + m["best_model"] + '"\n'
        '}', "Ví dụ yêu cầu gửi tới API.", "JSON"))
    r.p(output(notebook_output("diabetes", "Bệnh nhân", 20),
               "Kiểm thử suy luận từ artifact đã lưu — chính là điều REST API thực hiện."))
    r.p(
        "Giao diện Web đặt biểu đồ mức độ quan trọng đặc trưng ngay dưới kết quả "
        "dự đoán, mỗi thanh kèm theo giá trị người dùng vừa nhập cho đúng đặc "
        "trưng đó — nhờ vậy người xem thấy trực tiếp đặc trưng nào kéo dự đoán về "
        "phía nguy cơ cao, thay vì chỉ nhận một nhãn không kèm lời giải thích. Hộp "
        f"chọn mô hình mặc định là <strong>{m['best_model_label']}</strong> — mô "
        "hình tốt nhất theo mục 7.9 — nhưng có thể đổi sang bốn mô hình còn lại và "
        "dự đoán lại trên cùng hồ sơ để so sánh trực tiếp. Ba hồ sơ mẫu dựng sẵn "
        "(🔴 nguy cơ cao, 🟡 trung bình, 🟢 nguy cơ thấp) cho phép bấm điền nhanh ba "
        "kịch bản đại diện ba mức nguy cơ khác nhau, thay vì phải gõ tay từng chỉ "
        "số mỗi lần kiểm thử hoặc chụp ảnh minh hoạ.",
    )
    r.p(figure("shot_dia_web_models.png",
               "Bảng so sánh năm mô hình phân loại hiển thị ngay trên giao diện Web. "
               "Số liệu đọc trực tiếp từ điểm cuối /metadata nên luôn trùng khớp với "
               "bảng ở mục 7.9 — không có đường nào để hai bên lệch nhau; hàng tô nền, "
               "in đậm kèm ★ là mô hình đang được chọn ở hộp chọn phía trên."))
    _ui_screenshots(r, "7.10", "dia", 5001, "diabetes",
                    "Ứng dụng 1 — Dự đoán bệnh tiểu đường")


# ------------------------------------------------------------- Chương VIII
def _ch8(r) -> None:
    m = META["house_price"]
    r.h(1, "Chương VIII. Ứng dụng 2 — Dự đoán giá nhà")

    r.h(2, "8.1. Mô tả bài toán")
    r.p(
        "Ước lượng giá rao bán của một bất động sản nhà ở tại Việt Nam từ các đặc "
        "điểm của chính bất động sản đó.",
        "<ul><li><em>X</em> = đặc điểm căn nhà (diện tích, số tầng, số phòng, hướng, "
        "tình trạng pháp lý, vị trí…);</li>"
        "<li><em>y</em> = giá nhà tính bằng tỷ VNĐ, <em>y</em> ∈ ℝ<sup>+</sup>.</li></ul>",
        "Đây là bài toán <strong>hồi quy có giám sát</strong>.",
    )
    r.p(note(
        "Vì sao khác Ứng dụng 1.",
        "Ở tiểu đường, đầu ra là một trong hai nhãn rời rạc và sai lầm được đếm bằng "
        "số ca. Ở đây đầu ra là một <strong>số thực liên tục</strong> và sai lầm được "
        "đo bằng <strong>khoảng cách</strong>: đoán 5,0 tỷ cho căn 5,2 tỷ là gần "
        "đúng, đoán 9,0 tỷ là sai nặng. Hệ quả trực tiếp: không có ma trận nhầm lẫn, "
        "không có accuracy, không có ROC-AUC. Bộ độ đo đổi hoàn toàn sang MAE, MSE, "
        "RMSE và R²."))

    r.h(2, "8.2. Giới thiệu tập dữ liệu")
    r.p(table(
        ["Mục", "Giá trị"],
        [
            ["Tên tập dữ liệu", "Vietnam Housing Dataset 2024"],
            ["Nguồn Kaggle",
             "<a href='https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024'>"
             "kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024</a>"],
            ["Số quan sát", f"{m['n_samples_raw']:,}"],
            ["Số thuộc tính", "12 (11 đặc trưng + 1 biến mục tiêu)"],
            ["Biến mục tiêu", "<code class='inl'>Price</code> (tỷ VNĐ)"],
            ["Tệp cục bộ", "<code class='inl'>house_price/data/house_prices.csv</code>"],
        ]))
    r.p(
        "<strong>Một quan sát biểu diễn cái gì.</strong> Mỗi dòng là một tin rao bán "
        "bất động sản trên sàn giao dịch trực tuyến tại Việt Nam năm 2024. Điểm này "
        "cần nói rõ: <code class='inl'>Price</code> là <strong>giá chào bán</strong>, "
        "không phải giá giao dịch thành công. Mô hình vì vậy học “thị trường đang hỏi "
        "bao nhiêu”, chứ không phải “thị trường trả bao nhiêu”.",
    )
    r.p(table(
        ["Nhóm", "Cột"],
        [
            ["Số (numerical)", "<code class='inl'>Area</code>, <code class='inl'>Frontage</code>, "
             "<code class='inl'>Access Road</code>, <code class='inl'>Floors</code>, "
             "<code class='inl'>Bedrooms</code>, <code class='inl'>Bathrooms</code>"],
            ["Phân loại (categorical)", "<code class='inl'>House direction</code>, "
             "<code class='inl'>Balcony direction</code>, <code class='inl'>Legal status</code>, "
             "<code class='inl'>Furniture state</code>"],
            ["Văn bản (text)", "<code class='inl'>Address</code> — địa chỉ tự do"],
            ["Mục tiêu", "<code class='inl'>Price</code>"],
        ]))
    r.p(
        "Khác hẳn Ứng dụng 1: ở đây có <strong>bốn biến phân loại</strong>, nên bước "
        "mã hoá one-hot là bắt buộc — đây chính là nội dung mục 2.8. Ngoài ra còn một "
        "cột văn bản tự do sẽ được khai thác ở mục 8.5.",
    )

    r.h(2, "8.3. Khảo sát và tìm hiểu dữ liệu")
    r.p(output(notebook_output("house_price", "Giá trị thiếu theo cột", 20)))
    r.p(figure("hou_missing.png", "Ứng dụng 2 — Tỷ lệ giá trị thiếu theo cột.", "72%"))
    r.p(oim(
        "<code class='inl'>Balcony direction</code> thiếu 82,6%, "
        "<code class='inl'>House direction</code> thiếu 70,3%, "
        "<code class='inl'>Furniture state</code> 46,7%, "
        "<code class='inl'>Access Road</code> 44,0%, "
        "<code class='inl'>Frontage</code> 38,3%. "
        "<code class='inl'>Area</code> và <code class='inl'>Price</code> không thiếu "
        "dòng nào.",
        "Đây <strong>không phải lỗi thu thập</strong> mà là hành vi người đăng tin: "
        "biểu mẫu đăng tin chỉ bắt buộc diện tích và giá, các trường còn lại tuỳ chọn "
        "nên người bán chỉ điền khi thấy có lợi. Nghĩa là bản thân việc thiếu cũng "
        "mang thông tin.",
        "Không thể xoá dòng theo <code class='inl'>Balcony direction</code> — sẽ mất "
        "82,6% dữ liệu. Với biến phân loại có một lựa chọn sạch hơn: coi “không khai "
        "báo” là <strong>một hạng mục riêng</strong> (<code class='inl'>Không rõ</code>) "
        "rồi one-hot nó. Cách này giữ nguyên số dòng và biến sự vắng mặt thành tín hiệu."))

    r.p(figure("hou_boxplot.png", "Ứng dụng 2 — Hộp râu các biến số."))
    r.p(oim(
        "<code class='inl'>Area</code> có đuôi phải cực dài (tối đa hàng nghìn m²) "
        "trong khi trung vị chỉ vài chục m². <code class='inl'>Price</code> bị chặn "
        "ở 11,5 tỷ.",
        "Đuôi <code class='inl'>Area</code> là các lô đất lớn hoặc nhà xưởng lẫn vào "
        "tập dữ liệu nhà ở. Việc <code class='inl'>Price</code> chặn ở 11,5 tỷ cho "
        "thấy tập dữ liệu đã được lọc sẵn ở khoảng phổ thông.",
        "Ta <strong>cắt ngưỡng có kiểm soát</strong>, giữ các căn có diện tích trong "
        "khoảng hợp lý cho nhà ở đô thị. Đây không phải “xoá dữ liệu xấu” mà là thu "
        "hẹp phạm vi bài toán một cách minh bạch — mô hình dự đoán nhà ở đô thị, và "
        "ta khai báo đúng như vậy."))

    r.h(2, "8.4. Làm sạch dữ liệu")
    r.p(output(notebook_output("house_price", "Sau khi khử trùng lặp", 8)))
    r.p(
        f"Quá trình làm sạch giữ lại {m['n_samples_clean']:,} trên "
        f"{m['n_samples_raw']:,} dòng ban đầu "
        f"({m['n_samples_clean'] / m['n_samples_raw'] * 100:.1f}%).",
    )

    r.h(3, "8.4.1. Vì sao khử trùng lặp là bước quan trọng nhất")
    r.p(
        "Một bất động sản thường được đăng lại nhiều lần, hoặc do nhiều môi giới "
        "cùng đăng một căn. Nếu không khử trùng lặp <strong>trước khi chia tập</strong>, "
        "cùng một căn nhà xuất hiện ở cả tập train lẫn tập test; mô hình được chấm "
        "điểm trên đúng căn nó đã học thuộc, và R² trên test bị thổi phồng.",
        "Đây là dạng rò rỉ tinh vi nhất trong ba ứng dụng vì nó <strong>không lộ ra "
        "ở bất kỳ biểu đồ nào</strong>. Vì vậy nó phải được đo trực tiếp:",
    )
    r.p(output(notebook_output("house_price", "ĐO MỨC THỔI PHỒNG", 20),
               "Đo mức thổi phồng R² khi không khử trùng lặp."))
    r.p(note(
        "Kết luận.",
        "Bỏ bước khử trùng lặp làm R² tăng đáng kể ở mọi mô hình — đúng bằng phần "
        "điểm mà mô hình “kiếm được” nhờ nhìn thấy trước đáp án. "
        "<strong>Điểm số cao hơn không có nghĩa là mô hình tốt hơn; nó chỉ có nghĩa "
        "là phép đo đã bị nhiễm dữ liệu huấn luyện.</strong>", "bad"))

    r.h(3, "8.4.2. Bốn cải tiến so với Assignment 01")
    r.p(
        "Assignment 01 đã làm việc trên cùng bộ dữ liệu này. Assignment 02 sửa bốn "
        "điểm về mặt phương pháp:",
    )
    r.p(table(
        ["#", "Assignment 01", "Assignment 02", "Hệ quả"],
        [
            ["1", "Không khử trùng lặp", "Khử theo <code class='inl'>(Address, Area, Price)</code> "
             "trước khi chia tập", "Loại bỏ rò rỉ; điểm số thấp hơn nhưng trung thực"],
            ["2", "Điền khuyết bằng trung vị tính trên toàn bộ dữ liệu",
             "Điền khuyết bên trong <code class='inl'>Pipeline</code>, chỉ học từ tập train",
             "Loại bỏ rò rỉ qua tiền xử lý"],
            ["3", "Ba đặc trưng phái sinh", "Năm đặc trưng phái sinh, kèm phép đo hiệu quả",
             "Quyết định có bằng chứng thay vì phỏng đoán"],
            ["4", "Chia train/test; chọn mô hình trên chính tập test",
             "Chia train/validation/test; chọn trên validation, chấm điểm cuối trên test",
             "Ước lượng cuối cùng không bị lạc quan hoá"],
        ]))
    r.p(
        "Điểm 2 và 4 đáng chú ý ngang điểm 1: chọn mô hình dựa trên tập test biến tập "
        "test thành một tập validation trá hình, khiến ước lượng cuối cùng lạc quan "
        "thêm một lần nữa.",
    )

    r.h(2, "8.5. Biểu diễn dữ liệu")
    r.h(3, "8.5.1. Rút biến vị trí từ cột địa chỉ")
    r.p(
        "<code class='inl'>Address</code> là chuỗi tự do dạng "
        "<em>“Dự án ..., Xã Long Hưng, Văn Giang, Hưng Yên”</em>. Địa chỉ hành chính "
        "Việt Nam viết từ nhỏ đến lớn, nên thành phần cuối là cấp tỉnh và áp chót là "
        "cấp huyện — chính là biến vị trí mà EDA chỉ ra là còn thiếu.",
        "Ta <strong>không</strong> one-hot toàn bộ chuỗi địa chỉ: mỗi địa chỉ gần như "
        "duy nhất, sẽ sinh ra hàng chục nghìn cột và mô hình học thuộc lòng thay vì "
        "khái quát. Thay vào đó rút ra tỉnh và huyện, giữ những giá trị có đủ số "
        "lượng tin đăng, gộp phần còn lại thành <code class='inl'>Khác</code>.",
    )
    r.p(code(notebook_code("house_price", "def parse_address(address):"),
             "Rút tỉnh/thành và quận/huyện từ chuỗi địa chỉ tự do."))
    r.p(figure("hou_province.png", "Ứng dụng 2 — Giá trung vị theo tỉnh/thành phố.", "80%"))
    r.p(oim(
        "Giá trung vị chênh lệch rõ rệt giữa các tỉnh/thành phố.",
        "Xác nhận giả thuyết từ EDA: vị trí là biến giải thích mạnh, và nó vốn bị "
        "chôn trong một cột văn bản mà mô hình không đọc được.",
        "Chỉ một phép rút chuỗi đơn giản đã biến dữ liệu văn bản không dùng được "
        "thành đặc trưng phân loại có sức dự báo. Đây là minh hoạ trực tiếp cho luận "
        "điểm của Assignment 02: <strong>biểu diễn dữ liệu là một phần của lời giải, "
        "không phải công đoạn phụ trợ</strong>."))

    r.h(3, "8.5.2. Năm đặc trưng phái sinh")
    r.p(table(
        ["Đặc trưng", "Công thức", "Mã hoá điều gì"],
        [
            ["Total_Area", "Area × max(Floors, 1)",
             "Tổng diện tích sàn sử dụng — thứ người mua thực sự trả tiền, khác diện tích đất"],
            ["Room_Density", "(Bedrooms + Bathrooms) / Area",
             "Mức độ chia nhỏ; mật độ cao gợi ý nhà cho thuê chứ không phải nhà ở gia đình"],
            ["Frontage_Ratio", "Frontage / Area",
             "Hình dạng lô đất; tỷ lệ thấp là nhà ống sâu, tỷ lệ cao là lô vuông vắn"],
            ["Area_per_Bedroom", "Area / max(Bedrooms, 1)",
             "Độ rộng rãi mỗi phòng, phân biệt nhà cao cấp với nhà chia nhỏ"],
            ["Log_Area", "log(1 + Area)",
             "Nén đuôi phải của diện tích, giúp mô hình tuyến tính bắt quan hệ dưới tuyến tính"],
        ]))
    r.p(
        "<strong>Vì sao ba tỷ số lại có ích ngay cả với mô hình dạng cây.</strong> "
        "Cây quyết định chia theo một biến mỗi lần, nên nó không thể biểu diễn "
        "<em>Bedrooms / Area</em> trừ khi được cho sẵn — nó sẽ phải xấp xỉ bằng rất "
        "nhiều nhát cắt bậc thang trên hai biến riêng lẻ, tốn độ sâu và dễ quá khớp. "
        "Đưa sẵn tỷ số vào là cách giảm gánh nặng biểu diễn cho mô hình.",
        "Nhưng đây cũng là một quyết định cần bằng chứng, nên nó được đo:",
    )
    r.p(output(notebook_output("house_price", "ĐO HIỆU QUẢ CỦA ĐẶC TRƯNG PHÁI SINH", 14),
               "Đo hiệu quả thật của năm đặc trưng phái sinh trên tập test."))
    r.p(
        "Năm đặc trưng phái sinh cải thiện cả R² lẫn MAE ở mọi mô hình. Mức cải thiện "
        "lớn nhất rơi vào <strong>Linear Regression</strong> — hợp lý, vì "
        "<code class='inl'>Log_Area</code> và các tỷ số chính là những quan hệ phi "
        "tuyến mà mô hình tuyến tính không thể tự tạo ra. Đây là bằng chứng định "
        "lượng cho luận điểm: <strong>cùng một dữ liệu, cùng một thuật toán, chỉ đổi "
        "cách biểu diễn thì kết quả đã khác</strong>.",
    )

    r.h(3, "8.5.3. Ma trận đặc trưng cuối cùng")
    r.p(output(notebook_output("house_price", "Số chiều TRƯỚC mã hoá", 14)))
    r.p(table(
        ["Hạng mục", "Giá trị"],
        [
            ["Số dòng sau làm sạch", f"{m['n_samples_clean']:,}"],
            ["Số cột trước mã hoá", str(m["n_features_before_encoding"])],
            ["Số cột sau one-hot", str(m["n_features_after_encoding"])],
            ["Hình dạng ma trận đặc trưng",
             f"<em>X</em> ∈ ℝ<sup>{m['feature_matrix_shape'][0]}×{m['feature_matrix_shape'][1]}</sup>"],
            ["Vectơ mục tiêu", f"<em>y</em> ∈ ℝ<sup>{m['feature_matrix_shape'][0]}</sup>, đơn vị tỷ VNĐ"],
            ["Kiểu dữ liệu", "<code class='inl'>float64</code>"],
        ]))
    r.p(
        f"One-hot làm số chiều tăng từ {m['n_features_before_encoding']} lên "
        f"{m['n_features_after_encoding']} cột — cái giá phải trả để không áp đặt "
        "thứ tự giả lên các biến phân loại.",
    )

    r.h(2, "8.6. Phân tích khám phá dữ liệu (EDA)")
    r.p(figure("hou_target.png", "Ứng dụng 2 — Phân phối biến mục tiêu."))
    r.p(oim(
        "Giá tập trung quanh trung vị 5,9 tỷ, phân phối gần đối xứng, độ lệch nhỏ. "
        "Không có đuôi phải cực đoan như thường thấy ở dữ liệu bất động sản.",
        "Do tập dữ liệu đã bị chặn trên ở 11,5 tỷ. Ở một tập không bị chặn, giá bất "
        "động sản luôn lệch phải mạnh và ta sẽ phải huấn luyện trên log(1 + y) rồi mũ "
        "hoá ngược khi dự đoán.",
        "Vì phân phối đã gần đối xứng, ta huấn luyện thẳng trên <em>y</em> mà không "
        "cần biến đổi log. Quyết định này giữ cho MAE và RMSE đọc được trực tiếp bằng "
        "tỷ VNĐ — người dùng cuối hiểu ngay “sai trung bình 1 tỷ” mà không cần quy đổi."))

    r.p(figure("hou_relations.png", "Ứng dụng 2 — Quan hệ giữa các đặc trưng và giá."))
    r.p(oim(
        "Giá tăng theo diện tích nhưng độ tán rất lớn: hai căn cùng 80 m² có thể "
        "chênh nhau vài tỷ. Giá tăng đều theo số phòng ngủ và số tầng.",
        "Diện tích một mình không quyết định giá — <strong>vị trí</strong> mới quyết "
        "định, và vị trí đang nằm ẩn trong cột <code class='inl'>Address</code> dạng "
        "văn bản tự do mà mô hình chưa đọc được.",
        "Đây là căn cứ trực tiếp cho kỹ thuật đặc trưng ở mục 8.5.1. Nếu bỏ qua bước "
        "rút vị trí, mô hình sẽ mãi không giải thích được phần lớn phương sai của giá."))

    r.p(figure("hou_corr.png", "Ứng dụng 2 — Ma trận tương quan Pearson.", "62%"))
    r.p(oim(
        "Tương quan tuyến tính giữa từng biến số và <code class='inl'>Price</code> "
        "đều thấp, dưới 0,3.",
        "Tương quan Pearson chỉ đo quan hệ <strong>tuyến tính</strong>. Giá bất động "
        "sản phụ thuộc vị trí theo cách phi tuyến và theo tương tác — 80 m² ở Hà Nội "
        "khác hẳn 80 m² ở tỉnh lẻ — nên hệ số tuyến tính thấp là điều được dự đoán trước.",
        "Đây là dự báo mạnh rằng <strong>Linear Regression sẽ kém hơn các mô hình dạng "
        "cây</strong> ở bài toán này, và bảng kết quả mục 8.8 xác nhận. Ở Ứng dụng 1 "
        "thì ngược lại: <code class='inl'>Glucose</code> có tương quan tuyến tính 0,49 "
        "với nhãn nên Logistic Regression cạnh tranh tốt."))

    r.h(2, "8.7. Xây dựng mô hình")
    r.h(3, "8.7.1. Chia tập và mô hình cơ sở")
    r.p(output(notebook_output("house_price", "BASELINE — luôn đoán giá trung bình", 8)))
    r.p(
        "Không phân tầng vì đây là bài toán hồi quy, biến mục tiêu liên tục nên không "
        f"có “lớp” để giữ tỷ lệ. Với <em>N</em> > {m['n_samples_clean'] // 1000}.000, "
        "phép chia ngẫu nhiên đủ để ba tập có phân phối giá tương đương.",
    )
    r.h(3, "8.7.2. Năm mô hình hồi quy")
    r.p(code(notebook_code("house_price", "MODELS = {", drop_comment_header=True),
             "Định nghĩa năm mô hình hồi quy của Ứng dụng 2."))

    r.h(2, "8.8. Đánh giá mô hình")
    keys = ["MAE", "MSE", "RMSE", "R2"]
    rows, _ = metric_rows("house_price", keys, "R2")
    r.p(table(["Mô hình", "MAE (tỷ)", "MSE", "RMSE (tỷ)", "R²"], rows,
              "Ứng dụng 2 — So sánh năm mô hình hồi quy trên tập test, sắp theo R².", "num"))
    r.p(figure("hou_model_comparison.png", "Ứng dụng 2 — So sánh năm mô hình trên tập test."))
    best = m["test_metrics"][m["best_model"]]
    r.p(
        f"Mô hình tốt nhất là <strong>{m['best_model_label']}</strong> với MAE = "
        f"{fmt(best['MAE'])} tỷ VNĐ và R² = {fmt(best['R2'])}, so với baseline "
        f"MAE = {fmt(m['baseline']['MAE'])} tỷ và R² = 0. Nghĩa là hệ thống thường "
        f"lệch khoảng {fmt(best['MAE'], 2)} tỷ VNĐ và giải thích được "
        f"{fmt(best['R2'] * 100, 1)}% phương sai giá.",
        "Ba mô hình dạng cây đều vượt hai mô hình tuyến tính, đúng như phân tích "
        "tương quan ở mục 8.6 đã dự báo.",
    )

    r.h(3, "8.8.1. Phân tích phần dư")
    r.p(figure("hou_residual.png", "Ứng dụng 2 — Chẩn đoán chất lượng dự đoán."))
    r.p(oim(
        "Phần dư phân bố quanh 0 gần đối xứng, nhưng biểu đồ thực-tế-so-với-dự-đoán "
        "cho thấy mô hình <strong>kéo các dự đoán về phía trung tâm</strong>: căn rẻ "
        "bị đoán đắt lên, căn đắt bị đoán rẻ đi.",
        "Đây là hiện tượng <strong>co về trung bình</strong>, đặc trưng của mô hình "
        "dạng cây: mỗi lá trả về trung bình của các mẫu rơi vào lá đó, nên không bao "
        "giờ dự đoán vượt quá khoảng giá đã thấy trong huấn luyện.",
        "Hệ thống đáng tin ở phân khúc phổ thông (nơi có nhiều dữ liệu) và kém tin "
        "cậy ở hai đầu. Giao diện Web vì vậy phải trình bày kết quả kèm "
        "<strong>khoảng dao động</strong> thay vì một con số duy nhất — nói “khoảng "
        "5,2 tỷ ± 1,3 tỷ” trung thực hơn hẳn “5,234 tỷ”."))
    r.p(output(notebook_output("house_price", "Sai số tuyệt đối trung bình theo khoảng giá", 12)))
    r.p(
        "MAE tăng rõ rệt ở hai đầu phổ giá và thấp nhất ở khoảng giữa. Một con số MAE "
        "tổng thể <strong>che giấu</strong> sự chênh lệch này; báo cáo MAE theo từng "
        "phân khúc là cách trung thực hơn để mô tả năng lực hệ thống.",
    )
    r.p(figure("hou_importance.png",
               "Ứng dụng 2 — Các đặc trưng quan trọng nhất theo mô hình được chọn.", "78%"))

    r.h(2, "8.9. Lựa chọn mô hình")
    r.p(table(
        ["Tiêu chí", "Nhận định"],
        [
            ["Hiệu năng dự báo", "Xếp theo R² và MAE trên tập test"],
            ["Khả năng diễn giải", "Linear/Ridge cho hệ số đọc được; rừng cây chỉ cho "
             "tầm quan trọng tương đối"],
            ["Chi phí tính toán", "Gradient Boosting huấn luyện lâu nhất; Linear gần như tức thì"],
            ["Độ bền", "Rừng cây bền với ngoại lệ nhờ trung bình hoá nhiều cây"],
            ["Ràng buộc triển khai", "Random Forest cho tệp mô hình lớn nhất"],
        ]))
    r.p(note(
        f"Mô hình được chọn: {m['best_model_label']}.",
        f"MAE = {fmt(best['MAE'])} tỷ VNĐ, RMSE = {fmt(best['RMSE'])} tỷ, "
        f"R² = {fmt(best['R2'])}. Ở bài toán này hai tiêu chí đầu <strong>xung đột "
        "nhau</strong>: mô hình chính xác nhất lại khó giải thích nhất. Ta chọn theo "
        "hiệu năng, vì người dùng cuối cần một con số giá đáng tin hơn là một công "
        "thức đọc được.", "good"))

    r.h(2, "8.10. Triển khai hệ thống")
    r.p(table(
        ["Hạng mục", "Giá trị"],
        [
            ["Khung web", "Flask 3.1"],
            ["Điểm cuối", "<code class='inl'>POST /house-price/v1/predict</code>"],
            ["Cổng", "5002"],
            ["Biến đầu vào số", ", ".join(f"<code class='inl'>{c}</code>"
                                          for c in ["Area", "Frontage", "Access Road",
                                                    "Floors", "Bedrooms", "Bathrooms"])],
            ["Biến đầu vào phân loại", ", ".join(f"<code class='inl'>{c}</code>"
                                                 for c in m["categorical_features"])],
            ["Đặc trưng phái sinh", "API tự tính từ các trường cơ bản, đúng công thức đã "
             "dùng lúc huấn luyện"],
            ["Quy tắc kiểm tra", "Area phải lớn hơn 0; các trường số tuỳ chọn có thể để "
             "trống và sẽ được pipeline điền khuyết; hạng mục lạ được xử lý bằng "
             "<code class='inl'>handle_unknown=\"ignore\"</code>"],
            ["Đầu ra", "Giá dự đoán kèm khoảng dao động ± "
             f"{fmt(m['residual_std'], 2)} tỷ VNĐ"],
        ]))
    r.p(code(
        '{\n'
        '  "Area": 85, "Frontage": 5, "Access Road": 8,\n'
        '  "Floors": 4, "Bedrooms": 4, "Bathrooms": 3,\n'
        '  "House direction": "East",\n'
        '  "Legal status": "Have certificate",\n'
        '  "Furniture state": "Full",\n'
        '  "Province": "Hà Nội",\n'
        '  "model": "' + m["best_model"] + '"\n'
        '}', "Ví dụ yêu cầu gửi tới API.", "JSON"))
    r.p(output(notebook_output("house_price", "Căn nhà", 16),
               "Kiểm thử suy luận từ artifact đã lưu."))
    r.p(note(
        "Vì sao API phải tự tính đặc trưng phái sinh.",
        "Người dùng chỉ nhập sáu con số cơ bản, nhưng mô hình được huấn luyện trên "
        f"{m['n_features_after_encoding']} chiều bao gồm cả năm đặc trưng phái sinh. "
        "Nếu API không tính lại chúng theo <em>đúng công thức đã dùng lúc huấn luyện</em>, "
        "biểu diễn lúc dự đoán sẽ lệch khỏi biểu diễn lúc học — chính là vi phạm "
        "nguyên tắc nhất quán ở mục 2.10.", "warn"))
    r.p(
        "Vì mô hình hồi quy không sinh ra xác suất, giao diện Web thay thanh độ "
        "tin cậy quen thuộc bằng một thước đo khoảng giá: một vạch định vị giá dự "
        "đoán bên trong đoạn [giá thấp, giá cao] rộng ± "
        f"{fmt(m['residual_std'], 2)} tỷ VNĐ quanh nó. Đây là một lựa chọn thiết "
        "kế có chủ đích, không phải một hạn chế bị bỏ qua — mục 8.8.1 đã chỉ ra mô "
        "hình dạng cây co dự đoán về trung bình, nên một con số đơn lẻ như "
        "“5,234 tỷ” ngụ ý một độ chính xác không có thật, trong khi khoảng dao "
        "động buộc người xem đọc kết quả đúng bản chất của nó: một ước lượng, "
        "không phải một câu trả lời tuyệt đối. Biểu đồ mức độ quan trọng đặc "
        "trưng bên dưới liệt kê các đặc trưng ảnh hưởng nhiều nhất tới giá nhưng "
        "— khác Ứng dụng 1 — không đối chiếu với giá trị người dùng vừa nhập, vì "
        "phần lớn cột trọng số cao nhất là cột sinh ra từ one-hot (từng tỉnh/"
        "thành, từng tình trạng pháp lý riêng lẻ) không còn map một-một về ô nhập "
        "gốc trên biểu mẫu. Hai hồ sơ mẫu — nhà cao cấp trung tâm và nhà bình dân "
        "ngoại thành — cố ý đặt ở hai đầu phổ giá, đúng vùng mà mục 8.8.1 vừa chỉ "
        "ra là mô hình kém tin cậy nhất.",
    )
    r.p(figure("shot_hou_web_models.png",
               "Bảng so sánh năm mô hình hồi quy hiển thị ngay trên giao diện Web. Vì "
               "bảng lấy số trực tiếp từ /metadata — cùng một metadata.json đã dùng để "
               "dựng bảng ở mục 8.8 — con số người dùng nhìn thấy trên trình duyệt luôn "
               "trùng khớp với con số trong báo cáo; hàng tô nền, in đậm kèm ★ là mô "
               "hình đang chạy."))
    _ui_screenshots(r, "8.10", "hou", 5002, "house-price",
                    "Ứng dụng 2 — Dự đoán giá nhà")


# --------------------------------------------------------------- Chương IX
def _ch9(r) -> None:
    m = META["customer_behavior"]
    r.h(1, "Chương IX. Ứng dụng 3 — Phân tích hành vi khách hàng và khám phá sở thích")

    r.h(2, "9.1. Mô tả bài toán")
    r.p(
        "Từ thông tin khách hàng và <strong>nội dung đánh giá họ viết</strong>, dự "
        "đoán khách hàng có khuyến nghị sản phẩm cho người khác hay không.",
        "<ul><li><em>X</em> = đặc trưng hành vi khách hàng + biểu diễn văn bản của bình luận;</li>"
        "<li><em>y</em> = <code class='inl'>Recommended IND</code> ∈ {0, 1}.</li></ul>",
        "Đây là bài toán <strong>phân loại nhị phân</strong>.",
    )
    r.p(
        "<strong>Vì sao chọn đúng mục tiêu này.</strong> Đề bài liệt kê nhiều lựa "
        "chọn và yêu cầu chọn một mục tiêu được xác định rõ ràng. "
        "<code class='inl'>Recommended IND</code> là lựa chọn đúng vì ba lý do: nó "
        "<strong>có sẵn nhãn thật</strong> trong dữ liệu chứ không phải nhãn tự bịa "
        "bằng quy tắc; nó là <strong>tín hiệu thương mại trực tiếp</strong> vận hành "
        "xếp hạng sản phẩm và gợi ý cá nhân hoá; và nó <strong>phụ thuộc mạnh vào nội "
        "dung bình luận</strong>, nên là bài toán lý tưởng để đo giá trị thật của "
        "biểu diễn văn bản — đúng trọng tâm đề bài đặt ra.",
    )
    r.p(note(
        "Giới hạn phải nói rõ.",
        "Tập dữ liệu <strong>không có <code class='inl'>Customer ID</code></strong>. "
        "Mỗi dòng là một lượt đánh giá, không phải một khách hàng. Vì vậy mọi kết "
        "luận ở đây là ở <em>cấp lượt đánh giá</em>, và không thể khẳng định hành vi "
        "dài hạn của từng khách hàng.", "warn"))

    r.h(2, "9.2. Giới thiệu tập dữ liệu")
    r.p(table(
        ["Mục", "Giá trị"],
        [
            ["Tên tập dữ liệu", "Women's E-Commerce Clothing Reviews"],
            ["Nguồn Kaggle",
             "<a href='https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews'>"
             "kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews</a>"],
            ["Số quan sát", f"{m['n_samples_raw']:,}"],
            ["Số thuộc tính", "11 (10 đặc trưng + 1 biến mục tiêu)"],
            ["Biến mục tiêu", "<code class='inl'>Recommended IND</code>"],
            ["Tệp cục bộ", "<code class='inl'>customer_behavior/data/womens_ecommerce_reviews.csv</code>"],
        ]))
    r.p(table(
        ["Nhóm", "Cột"],
        [
            ["Số", "<code class='inl'>Age</code>, <code class='inl'>Rating</code>, "
             "<code class='inl'>Positive Feedback Count</code>, <code class='inl'>Clothing ID</code>"],
            ["Phân loại", "<code class='inl'>Division Name</code>, "
             "<code class='inl'>Department Name</code>, <code class='inl'>Class Name</code>"],
            ["<strong>Văn bản</strong>", "<code class='inl'>Title</code>, "
             "<code class='inl'>Review Text</code>"],
            ["Mục tiêu", "<code class='inl'>Recommended IND</code>"],
        ]))
    r.p(
        "Đây là ứng dụng <strong>duy nhất có đủ cả ba loại dữ liệu</strong>: số, "
        "phân loại và văn bản. Chính vì vậy nó đòi hỏi biểu diễn phức tạp nhất trong "
        "ba ứng dụng.",
    )

    r.h(2, "9.3. Biểu diễn khách hàng")
    r.p(
        "Một lượt đánh giá được biểu diễn bằng hai nhóm thành phần ghép song song:",
        '<div class="formula">x<sub>i</sub> = [x<sup>bảng</sup><sub>i</sub> ; '
        "x<sup>văn bản</sup><sub>i</sub>]</div>",
        "<ul>"
        "<li><strong>Nhánh bảng</strong> — tuổi, số lượt phản hồi hữu ích, độ dài bình "
        "luận, ba cấp phân loại sản phẩm;</li>"
        "<li><strong>Nhánh văn bản</strong> — vectơ TF-IDF của tiêu đề ghép với nội "
        "dung đánh giá.</li></ul>",
        "Mục 9.5 trình bày chi tiết chuỗi biến đổi của nhánh văn bản, và mục 9.8 đo "
        "xem nhánh ấy đóng góp bao nhiêu.",
    )

    r.h(2, "9.4. Làm sạch dữ liệu")
    r.h(3, "9.4.1. Vấn đề nghiêm trọng nhất là rò rỉ nhãn")
    r.p(
        "<code class='inl'>Rating</code> (số sao 1–5) và "
        "<code class='inl'>Recommended IND</code> gần như là <strong>cùng một thông "
        "tin</strong>: khách cho 5 sao thì hầu như chắc chắn khuyến nghị, cho 1 sao "
        "thì hầu như chắc chắn không.",
    )
    r.p(output(notebook_output("customer_behavior", "Bảng chéo Rating", 26),
               "Đo mức trùng khớp giữa Rating và biến mục tiêu."))
    r.p(note(
        "Vì sao phải loại bỏ cột Rating.",
        "Một quy tắc thô “Rating ≥ 4 thì khuyến nghị” đã đạt accuracy trên 90% mà "
        "không học được gì. Nếu giữ <code class='inl'>Rating</code> làm đặc trưng, "
        "mọi mô hình sẽ đạt điểm rất cao nhưng <strong>không mô hình nào học được gì "
        "về hành vi khách hàng</strong> — chúng chỉ đọc lại số sao. Tệ hơn, khi triển "
        "khai thật, ta muốn dự đoán khuyến nghị <em>từ nội dung khách viết</em>, mà "
        "lúc ấy <code class='inl'>Rating</code> cũng chưa có.", "bad"))
    r.p(
        "Đây là <strong>rò rỉ nhãn</strong> — dạng rò rỉ tinh vi hơn hẳn việc "
        "<code class='inl'>fit</code> scaler trên tập test, vì nó không vi phạm bất "
        "kỳ quy tắc pipeline nào và không có công cụ nào cảnh báo. Biểu hiện duy nhất "
        "là điểm số <em>quá đẹp</em>. <code class='inl'>Clothing ID</code> cũng bị "
        "loại, vì nó là định danh sản phẩm và không khái quát sang mã hàng mới.",
    )
    r.h(3, "9.4.2. Trùng lặp và bình luận rỗng")
    r.p(output(notebook_output("customer_behavior", "Sau khử trùng lặp", 8)))
    r.p(
        f"Quá trình làm sạch giữ lại {m['n_samples_clean']:,} trên "
        f"{m['n_samples_raw']:,} dòng. Dòng không có bình luận bị loại vì ứng dụng "
        "này lấy văn bản làm trung tâm — một dòng không có văn bản thì nhánh biểu "
        "diễn quan trọng nhất là rỗng. <code class='inl'>Title</code> thiếu thì thay "
        "bằng chuỗi rỗng rồi ghép vào <code class='inl'>Review Text</code>, vì tiêu "
        "đề thường cô đọng đúng cảm xúc chính.",
    )

    r.h(2, "9.5. Biểu diễn dữ liệu văn bản")
    r.p(
        "Đây là phần trung tâm của ứng dụng và là nơi duy nhất trong Assignment 02 "
        "trình bày đầy đủ chuỗi biến đổi mà Bài giảng 02 yêu cầu:",
        '<div class="flow">Bình luận → Token → Token ID → Vectơ → '
        "E ∈ ℝ<sup>B×T×d</sup></div>",
    )
    r.h(3, "9.5.1. Bốn bước trên một bình luận thật")
    r.p(code(notebook_code("customer_behavior", "BƯỚC 2 — TÁCH TOKEN"),
             "Tách token từ bình luận gốc."))
    r.p(output(notebook_output("customer_behavior", "BƯỚC 1 — BÌNH LUẬN GỐC", 12),
               "Bước 1 và 2 — từ chuỗi ký tự sang danh sách token."))
    r.p(output(notebook_output("customer_behavior", "BƯỚC 3 — TOKEN ID", 22),
               "Bước 3 — ánh xạ mỗi token thành một số nguyên."))
    r.p(output(notebook_output("customer_behavior", "BƯỚC 4 — VECTƠ TF-IDF", 20),
               "Bước 4 — vectơ TF-IDF, mỗi chiều là một từ cụ thể."))
    r.p(
        "Điểm cần nhấn mạnh ở bước 3: token ID đã là số, nhưng <strong>ID chỉ là "
        "nhãn, không mang ngữ nghĩa</strong>. ID 5 không “gần” ID 4 hơn ID 900 về mặt "
        "ý nghĩa. Phải sang bước 4 hoặc bước 5 thì giá trị số mới thực sự phản ánh "
        "nội dung.",
    )

    r.h(3, "9.5.2. Tensor embedding E ∈ ℝ^(B×T×d)")
    r.p(
        "TF-IDF cho mỗi văn bản <em>một vectơ duy nhất</em>, tức "
        "<em>X</em> ∈ ℝ<sup>N×d</sup> — mất hoàn toàn thứ tự từ. Mô hình mạng nơ-ron "
        "dùng cách khác: mỗi <em>token</em> được ánh xạ thành một vectơ <em>d</em> "
        "chiều, nên một văn bản thành ma trận <em>T</em> × <em>d</em>, và một lô "
        "<em>B</em> văn bản thành tensor ba chiều.",
    )
    r.p(code(notebook_code("customer_behavior", "B, T, D_EMB ="),
             "Dựng thật tensor embedding ba chiều."))
    r.p(output(notebook_output("customer_behavior", "Kích thước từ điển V", 20),
               "Ba chiều B, T, d với giá trị cụ thể."))
    t = m["tensor_demo"]
    r.p(table(
        ["Chiều", "Giá trị", "Ý nghĩa"],
        [
            ["<em>B</em>", str(t["B"]), "Số văn bản trong một lô (batch size)"],
            ["<em>T</em>", str(t["T"]), "Số token mỗi văn bản, đã cắt hoặc đệm về cùng độ dài"],
            ["<em>d</em>", str(t["d"]), "Số chiều embedding của mỗi token"],
            ["<em>V</em>", f"{t['vocab_size']:,}", "Kích thước từ điển (số token duy nhất)"],
            ["Hình dạng tensor", "×".join(str(x) for x in t["shape"]),
             f"E ∈ ℝ<sup>{t['B']}×{t['T']}×{t['d']}</sup>"],
        ]))

    r.h(3, "9.5.3. Vì sao triển khai bằng TF-IDF chứ không bằng embedding")
    r.p(table(
        ["", "TF-IDF (dùng để triển khai)", "Embedding (minh hoạ khái niệm)"],
        [
            ["Hình dạng", "ℝ<sup>N×d</sup> — hai chiều", "ℝ<sup>B×T×d</sup> — ba chiều"],
            ["Giữ thứ tự từ", "Không", "Có"],
            ["Mỗi chiều nghĩa là gì", "Một từ cụ thể — <strong>đọc được</strong>",
             "Chiều ẩn học được — không đọc được"],
            ["Chi phí huấn luyện", "Giây", "Giờ, cần GPU"],
            ["Cần bao nhiêu dữ liệu", "Vài nghìn dòng là đủ", "Hàng trăm nghìn dòng trở lên"],
        ]))
    r.p(
        f"Với {m['n_samples_clean']:,} dòng, embedding học từ đầu sẽ quá khớp. Hơn "
        "nữa TF-IDF cho phép <strong>chỉ ra từ nào đẩy dự đoán về phía nào</strong> — "
        "điều mà mục 9.9 khai thác và là thứ một hệ thống thương mại điện tử cần để "
        "giải thích kết quả cho người vận hành.",
        "<strong>Thông tin nào được giữ:</strong> từ nào xuất hiện, mức độ hiếm của "
        "từ đó, cường độ tín hiệu cảm xúc. <strong>Thông tin nào bị mất:</strong> thứ "
        "tự từ, phủ định tầm xa (“not good” bị tách thành hai token độc lập — một "
        "phần được bù bằng n-gram bậc 2), mỉa mai, và ngữ cảnh vượt câu.",
    )

    r.h(3, "9.5.4. Hai biểu diễn song song")
    r.p(code(notebook_code("customer_behavior", "# Biểu diễn 1 — CHỈ đặc trưng dạng bảng"),
             "Hai biểu diễn được dựng song song để so sánh trực tiếp."))
    r.p(output(notebook_output("customer_behavior", "BIỂU DIỄN 1 — chỉ đặc trưng dạng bảng", 14)))
    s = m["representation_shapes"]
    r.p(table(
        ["Biểu diễn", "Hình dạng ma trận train", "Số chiều", "Ghi chú"],
        [
            ["Chỉ bảng", f"({s['tabular'][0]:,}, {s['tabular'][1]})", str(s["tabular"][1]),
             "3 cột số + one-hot của 3 biến phân loại"],
            ["Bảng + văn bản", f"({s['hybrid'][0]:,}, {s['hybrid'][1]:,})", f"{s['hybrid'][1]:,}",
             f"văn bản đóng góp thêm {s['hybrid'][1] - s['tabular'][1]:,} chiều; ma trận thưa"],
            ["Chỉ văn bản", f"({s['text_only'][0]:,}, {s['text_only'][1]:,})",
             f"{s['text_only'][1]:,}", "TF-IDF 1–2 gram, dùng cho Naive Bayes"],
        ]))

    r.h(2, "9.6. Khám phá dữ liệu (EDA)")
    r.p(figure("ecom_overview.png", "Ứng dụng 3 — Tổng quan dữ liệu."))
    r.p(oim(
        f"Khoảng {fmt(m['baseline_accuracy'] * 100, 0)}% lượt đánh giá là khuyến "
        "nghị — tỷ lệ mất cân bằng khoảng 4,5 : 1, <strong>nặng hơn hẳn</strong> Ứng "
        "dụng 1 (1,87 : 1).",
        "Đây là <strong>thiên lệch tự chọn</strong>: người mua hài lòng có xu hướng "
        "viết đánh giá nhiều hơn người thất vọng, vì người thất vọng thường chỉ trả "
        "hàng rồi thôi.",
        f"Baseline “luôn đoán khuyến nghị” đạt tới {fmt(m['baseline_accuracy'] * 100, 1)}% "
        "accuracy. Với mức mất cân bằng này, accuracy gần như vô dụng để xếp hạng mô "
        "hình — phải dùng <strong>F1 của lớp thiểu số</strong> và ROC-AUC, đồng thời "
        "bật <code class='inl'>class_weight=\"balanced\"</code>."))

    r.p(figure("ecom_text_behavior.png", "Ứng dụng 3 — Đặc điểm bình luận và hành vi."))
    r.p(oim(
        "Bình luận của nhóm <strong>không</strong> khuyến nghị <strong>dài hơn</strong> "
        "nhóm khuyến nghị.",
        "Khách hài lòng viết ngắn (“Love this dress!”); khách thất vọng viết dài để "
        "giải thích cái gì sai — sai size, vải mỏng, màu khác ảnh.",
        "Bản thân <strong>độ dài văn bản đã là một đặc trưng dự báo</strong>, độc lập "
        "với nội dung từ ngữ, nên <code class='inl'>word_count</code> được đưa vào "
        "nhánh bảng. Đây là ví dụ về việc một thuộc tính <em>siêu dữ liệu</em> của "
        "văn bản mang thông tin mà mô hình túi-từ không tự nắm được."))

    r.p(figure("ecom_terms.png", "Ứng dụng 3 — Từ khoá xuất hiện nhiều nhất theo từng nhãn."))
    r.p(oim(
        "Hai nhóm chia sẻ nhiều từ chung (<em>dress</em>, <em>fabric</em>, "
        "<em>size</em>, <em>fit</em>) nhưng khác nhau ở các từ mang cảm xúc và từ chỉ "
        "vấn đề.",
        "Danh từ sản phẩm xuất hiện ở cả hai nhóm nên <strong>không phân biệt được</strong>; "
        "thứ phân biệt là tính từ đánh giá và từ chỉ lỗi.",
        "Đây chính xác là lý do phải dùng <strong>TF-IDF chứ không phải đếm tần suất "
        "thô</strong>: TF-IDF hạ trọng số những từ xuất hiện ở khắp nơi và nâng trọng "
        "số những từ mang tính phân biệt."))

    r.h(2, "9.7. Xây dựng mô hình")
    r.p(
        "Bộ sáu mô hình được thiết kế để trả lời câu hỏi trung tâm — văn bản có giá "
        "trị bao nhiêu. Ba mô hình đầu chỉ dùng đặc trưng bảng, ba mô hình sau được "
        "đọc thêm văn bản; tất cả dùng <strong>cùng một cách chia dữ liệu</strong>, "
        "nên chênh lệch điểm số giữa hai nhóm đo đúng đóng góp của biểu diễn văn bản.",
    )
    r.p(table(
        ["#", "Mô hình", "Biểu diễn", "Vai trò"],
        [
            ["1", "Logistic Regression", "Chỉ bảng", "Chuẩn tham chiếu không dùng văn bản"],
            ["2", "Decision Tree", "Chỉ bảng", "Kiểm tra xem phi tuyến trên đặc trưng bảng có cứu vãn được không"],
            ["3", "Random Forest", "Chỉ bảng", "Trần hiệu năng của biểu diễn chỉ-bảng"],
            ["4", "Logistic Regression", "Bảng + văn bản",
             "Phân loại tuyến tính trên văn bản — mô hình đề bài chỉ đích danh"],
            ["5", "Linear SVM", "Bảng + văn bản", "Biên lớn; chuẩn mạnh cho dữ liệu văn bản thưa"],
            ["6", "Multinomial Naive Bayes", "Chỉ văn bản", "Mô hình xác suất cổ điển cho phân loại văn bản"],
        ]))
    r.p(code(notebook_code("customer_behavior", "SPECS = {", drop_comment_header=True),
             "Định nghĩa sáu mô hình và biểu diễn tương ứng."))

    r.h(2, "9.8. Đánh giá mô hình")
    keys = ["Accuracy", "Precision", "Recall", "F1", "F1 (lớp 0)", "ROC-AUC"]
    rows, _ = metric_rows("customer_behavior", keys, "ROC-AUC")
    repr_vn = {"tabular": "Chỉ bảng", "hybrid": "Bảng + văn bản", "text": "Chỉ văn bản"}
    labels_to_key = {v: k for k, v in m["model_labels"].items()}
    rows2 = [[rw[0], repr_vn[m["model_repr"][labels_to_key[rw[0]]]]] + rw[1:] for rw in rows]
    r.p(table(["Mô hình", "Biểu diễn"] + keys, rows2,
              "Ứng dụng 3 — So sánh sáu mô hình trên tập test, sắp theo ROC-AUC.", "num"))
    r.p(figure("ecom_model_comparison.png",
               "Ứng dụng 3 — Biểu diễn văn bản đóng góp bao nhiêu?"))

    r.h(3, "9.8.1. Câu hỏi trung tâm — văn bản có cải thiện dự đoán không?")
    r.p(output(notebook_output("customer_behavior", "TÁC ĐỘNG CỦA BIỂU DIỄN VĂN BẢN", 18),
               "So sánh trực tiếp biểu diễn chỉ-bảng với biểu diễn có kèm văn bản."))
    tab_auc = max(v["ROC-AUC"] for k, v in m["test_metrics"].items()
                  if m["model_repr"][k] == "tabular")
    txt_auc = max(v["ROC-AUC"] for k, v in m["test_metrics"].items()
                  if m["model_repr"][k] != "tabular")
    r.p(oim(
        f"Mô hình chỉ dùng đặc trưng bảng đạt ROC-AUC cao nhất {fmt(tab_auc)}; mô "
        f"hình có đọc văn bản đạt {fmt(txt_auc)} — chênh lệch "
        f"{fmt(txt_auc - tab_auc)}. Khoảng cách lớn nhất nằm ở F1 của lớp thiểu số.",
        "Đặc trưng bảng (tuổi, nhóm hàng, số lượt hữu ích) gần như <strong>không mang "
        "tín hiệu</strong> về việc khách có hài lòng hay không — chúng mô tả "
        "<em>ai đánh giá</em> và <em>đánh giá cái gì</em>, chứ không mô tả "
        "<em>khách nghĩ gì</em>. Ý kiến nằm trong văn bản, và chỉ nằm ở đó.",
        "Đây là bằng chứng mạnh nhất trong cả ba ứng dụng cho luận điểm trung tâm của "
        "Assignment 02: <strong>chọn biểu diễn đúng có tác động lớn hơn hẳn chọn "
        "thuật toán mạnh</strong>. Random Forest với 250 cây trên đặc trưng bảng vẫn "
        "thua Multinomial Naive Bayes — mô hình đơn giản nhất trong sáu — khi Naive "
        "Bayes được đọc văn bản."))

    r.p(figure("ecom_confusion.png", "Ứng dụng 3 — Ma trận nhầm lẫn của sáu mô hình trên tập test."))
    r.p(output(notebook_output("customer_behavior", "khách khuyến nghị, dự đoán đúng", 14),
               "Diễn giải ma trận nhầm lẫn theo nghĩa kinh doanh."))
    r.p(
        "Đáng chú ý: <strong>ưu tiên ở đây ngược với Ứng dụng 1</strong>. Ở bài toán "
        "tiểu đường, FN (bỏ sót ca bệnh) là sai lầm đắt nhất. Ở đây, FP — phản hồi "
        "xấu bị đoán nhầm thành hài lòng — mới nguy hiểm, vì nó khiến sản phẩm lỗi "
        "tiếp tục được bán mà không ai biết. Vì vậy chỉ số vận hành của ứng dụng này "
        "là <strong>Recall của lớp 0</strong>.",
    )

    r.h(2, "9.9. Khám phá sở thích và hành vi khách hàng")
    r.p(figure("ecom_coefficients.png",
               "Ứng dụng 3 — Cụm từ có ảnh hưởng mạnh nhất tới dự đoán."))
    r.p(oim(
        "Các cụm từ tiêu cực mạnh nhất chỉ vào <strong>vấn đề cụ thể của sản phẩm</strong>: "
        "sai kích cỡ, chất liệu mỏng, khác ảnh, phải trả hàng. Các cụm từ tích cực "
        "chỉ vào cảm xúc và độ vừa vặn.",
        "Đây chính là “khám phá sở thích khách hàng” mà đề bài yêu cầu. Mô hình không "
        "chỉ dự đoán một nhãn; hệ số của nó là <strong>một danh sách xếp hạng những "
        "điều khách hàng quan tâm nhất</strong>, rút tự động từ hơn hai mươi nghìn "
        "bình luận mà không cần ai đọc tay.",
        "Đây cũng là lý do tiêu chí “khả năng diễn giải” có sức nặng bất thường ở ứng "
        "dụng này: một mô hình hộp đen dù nhỉnh hơn vài phần nghìn ROC-AUC cũng không "
        "cho ra được bảng này."))
    r.h(3, "9.9.1. Ứng dụng trong thương mại điện tử")
    r.p(
        "Doanh nghiệp dùng được ngay ba việc từ kết quả trên:",
        "<ol>"
        "<li><strong>Cảnh báo sớm sản phẩm có vấn đề</strong> khi tần suất các cụm từ "
        "tiêu cực tăng đột biến ở một mã hàng;</li>"
        "<li><strong>Sửa bảng thông số kích cỡ</strong> ở đúng những mã hàng bị than "
        "phiền về size — đây là nhóm từ tiêu cực xuất hiện nhiều nhất;</li>"
        "<li><strong>Ưu tiên hiển thị</strong> những sản phẩm được khen đúng các thuộc "
        "tính mà khách coi trọng.</li></ol>",
    )
    r.h(3, "9.9.2. Phân tích sai số")
    r.p(output(notebook_output("customer_behavior", "Số dự đoán sai", 22)))
    r.p(oim(
        "Các trường hợp sai thường là bình luận <strong>mâu thuẫn nội tại</strong>: "
        "khen kiểu dáng nhưng chê chất liệu, hoặc khen sản phẩm nhưng vẫn trả hàng vì "
        "sai size.",
        "Với biểu diễn túi-từ, một bình luận chứa cả từ tích cực lẫn tiêu cực sẽ có "
        "hai nhóm tín hiệu triệt tiêu nhau. Mô hình không nắm được <strong>cấu trúc "
        "nhượng bộ</strong> kiểu “đẹp <em>nhưng</em> không vừa” — mà chính vế sau mới "
        "quyết định nhãn.",
        "Đây là <strong>giới hạn cố hữu của biểu diễn túi-từ, không phải của thuật "
        "toán</strong>. Muốn vượt qua thì phải đổi biểu diễn — sang embedding có trật "
        "tự và mô hình đọc được ngữ cảnh, đúng như tensor B × T × d minh hoạ ở mục "
        "9.5.2. Một lần nữa: rào cản nằm ở biểu diễn, không nằm ở mô hình."))

    r.h(2, "9.10. Lựa chọn mô hình")
    best = m["test_metrics"][m["best_model"]]
    r.p(note(
        f"Mô hình được chọn: {m['best_model_label']}.",
        f"ROC-AUC = {fmt(best['ROC-AUC'])}, F1 = {fmt(best['F1'])}, "
        f"F1 lớp 0 = {fmt(best['F1 (lớp 0)'])}, Accuracy = {fmt(best['Accuracy'])}. "
        "Ngoài hiệu năng dẫn đầu, mô hình này còn cho <strong>hệ số đọc được từng "
        "từ</strong> — chính là sản phẩm “khám phá sở thích khách hàng” ở mục 9.9, "
        "thứ mà một mô hình hộp đen không đánh đổi được.", "good"))

    r.h(2, "9.11. Triển khai hệ thống")
    r.p(table(
        ["Hạng mục", "Giá trị"],
        [
            ["Khung web", "Flask 3.1"],
            ["Điểm cuối", "<code class='inl'>POST /customer-behavior/v1/predict</code>"],
            ["Cổng", "5003"],
            ["Biến đầu vào", "<code class='inl'>Age</code>, "
             "<code class='inl'>Positive Feedback Count</code>, "
             "<code class='inl'>Title</code>, <code class='inl'>Review Text</code>, "
             "<code class='inl'>Division Name</code>, "
             "<code class='inl'>Department Name</code>, <code class='inl'>Class Name</code>"],
            ["Đặc trưng API tự tính", "<code class='inl'>full_text</code> = Title + Review Text; "
             "<code class='inl'>word_count</code> = số từ của full_text"],
            ["Quy tắc kiểm tra", "Bình luận rỗng bị từ chối với mã 400 — đây là nhánh "
             "biểu diễn quan trọng nhất, không có nó thì dự đoán vô nghĩa"],
            ["Ba bộ tiền xử lý", "<code class='inl'>preprocessor.joblib</code> (lai), "
             "<code class='inl'>preprocessor_tabular.joblib</code> (chỉ bảng), "
             "<code class='inl'>tfidf_text_only.joblib</code> (chỉ văn bản) — chọn theo "
             "biểu diễn của mô hình người dùng chọn"],
        ]))
    r.p(code(
        '{\n'
        '  "Age": 34,\n'
        '  "Positive Feedback Count": 3,\n'
        '  "Title": "",\n'
        '  "Review Text": "Very disappointed. The material feels cheap and thin,\\n'
        '                  it runs two sizes too small. Returned it.",\n'
        '  "Division Name": "General",\n'
        '  "Department Name": "Tops",\n'
        '  "Class Name": "Blouses",\n'
        '  "model": "' + m["best_model"] + '"\n'
        '}', "Ví dụ yêu cầu gửi tới API.", "JSON"))
    r.p(output(notebook_output("customer_behavior", "Đánh giá", 18),
               "Kiểm thử suy luận từ artifact đã lưu."))
    r.p(
        "Giao diện Web của ứng dụng này lấy ô nhập bình luận làm trọng tâm, và "
        "ngay dưới kết quả là danh sách từ khoá tích cực/tiêu cực có trọng số cao "
        "nhất từ mô hình tuyến tính, trong đó những từ thật sự xuất hiện trong "
        "bình luận vừa nhập được tô viền đậm để phân biệt với phần còn lại của "
        "danh sách. Hộp chọn cho phép đổi giữa sáu mô hình; khi người dùng đổi mô "
        "hình rồi bấm dự đoán lại trên cùng một bình luận, nhãn “Biểu diễn: …” "
        "cạnh kết quả cùng đổi theo — chỉ bảng, bảng + văn bản, hoặc chỉ văn bản "
        "— để thấy tận mắt cùng một bình luận cho ra dự đoán khác nhau tuỳ biểu "
        "diễn nào được mô hình sử dụng. Đây chính là phát hiện trung tâm của toàn "
        "báo cáo (mục 9.5, 9.8.1): thêm văn bản vào biểu diễn thay đổi kết quả "
        "nhiều hơn thay đổi thuật toán, và giao diện để phát hiện đó tự chứng "
        "minh chứ không chỉ được phát biểu bằng lời.",
    )
    r.p(figure("shot_ecom_web_models.png",
               "Bảng so sánh sáu mô hình hiển thị ngay trên giao diện Web, hàng tô "
               "nền và in đậm kèm ★ là mô hình đang chọn. Bảng đọc thẳng từ /metadata "
               "— cùng nguồn với bảng ở mục 9.8 — nên số liệu trên giao diện và trong "
               "báo cáo không bao giờ lệch nhau."))
    _ui_screenshots(r, "9.11", "ecom", 5003, "customer-behavior",
                    "Ứng dụng 3 — Hành vi khách hàng thương mại điện tử")
