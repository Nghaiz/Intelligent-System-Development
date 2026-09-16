"""Chương 9, 10, 11 — ba thực nghiệm gốc không có trong bất kỳ bài tham khảo nào.

Chương 9  : độ tin cậy thống kê (đa hạt giống, McNemar, khoảng tin cậy Wilson)
Chương 10 : giải phẫu mô hình (hiệu chỉnh xác suất, bộ lọc, bản đồ đặc trưng, che khuất)
Chương 11 : bóc tách giai thừa và đường cong theo cỡ dữ liệu
"""

from __future__ import annotations

import report_lib as R
from chapters_1d import num, pct


def write(r: R.Report, data: dict) -> None:
    _chapter9(r, data)
    _chapter10(r, data)
    _chapter11(r, data)


def _missing(r: R.Report, nb: str, key: str) -> None:
    r.p(R.note(
        "Thiếu số liệu.",
        f"Chưa có tệp kết quả cho phần này. Chạy notebook "
        f"<code class='inl'>analysis/notebooks/{nb}</code> để sinh "
        f"<code class='inl'>{key}</code>.", "warn"))


# ===========================================================================
# CHƯƠNG 9 — ĐỘ TIN CẬY THỐNG KÊ
# ===========================================================================
def _chapter9(r: R.Report, data: dict) -> None:
    d = data.get("statistical")
    r.h(1, "Chương 9. Chênh lệch giữa ba khung có thật sự tồn tại không")

    r.p(
        "Bảy chương đầu đều kết thúc bằng một bảng đối chuẩn, và bảng nào cũng cho những khoảng "
        "cách rất nhỏ: 0,26 điểm phần trăm giữa NumPy và TensorFlow trên bài tiểu đường, 0,50 "
        "điểm trên bài đánh giá khách hàng. Ở mỗi chỗ báo cáo đều nói rằng khoảng cách đó nằm "
        "trong biên độ dao động. Nhưng nói như vậy là một <em>khẳng định chưa được kiểm chứng</em>. "
        "Chương này kiểm chứng nó.")
    r.p(
        "Đây cũng là chỗ mà cả hai bài tham khảo cùng dừng lại. Một trong hai bài liệt kê thẳng "
        "vào phần hạn chế rằng chỉ một hạt giống ngẫu nhiên được báo cáo và chưa có trung bình "
        "cùng độ lệch chuẩn qua nhiều lần chạy. Báo cáo này không để hạn chế đó nằm yên ở mục "
        "hạn chế mà đem đi đo.")

    if not d:
        _missing(r, "04_statistical_rigor.ipynb", "reports/metrics_statistical.json")
        return

    # ---- 9.1
    r.h(2, "9.1. Phương pháp")
    r.p(
        '<ol class="tight">'
        '<li><strong>Đa hạt giống.</strong> Huấn luyện lại mỗi cấu hình với năm hạt giống khác '
        'nhau, giữ nguyên tuyệt đối kiến trúc, cách chia dữ liệu và siêu tham số. Hạt giống là '
        'biến duy nhất thay đổi, nên mọi chênh lệch quan sát được đều quy về dao động khởi tạo '
        'và thứ tự mini-batch.</li>'
        '<li><strong>Kiểm định McNemar.</strong> Hai mô hình được đánh giá trên <em>cùng</em> tập '
        'kiểm thử nên hai dãy dự đoán tương quan rất mạnh. Dùng kiểm định t hai mẫu ở đây là sai '
        'phương pháp, vì nó giả định hai mẫu độc lập. McNemar điều kiện hoá trên đúng những mẫu '
        'mà hai mô hình <em>bất đồng</em>, bỏ qua phần chúng đồng ý, nên là phép kiểm đúng cho '
        'tình huống này.</li>'
        '<li><strong>Khoảng tin cậy Wilson.</strong> Ưu thế so với khoảng Wald là không bao giờ '
        'tràn ra ngoài đoạn [0, 1] và vẫn chuẩn khi tỉ lệ tiến sát 1, đúng vùng làm việc của một '
        'mô hình MNIST đạt trên 99%.</li>'
        '</ol>')

    # ---- 9.2
    r.h(2, "9.2. Dao động theo hạt giống")
    ms = d.get("multi_seed") or {}
    if ms:
        rows = []
        for dom, fws in ms.items():
            for fw, v in fws.items():
                metric = v.get("metric", "accuracy")
                fmt = (lambda x: num(x, 4)) if metric == "r2" else (lambda x: pct(x))
                rows.append([dom, fw, metric, fmt(v["mean"]),
                             f'± {num(v["std"], 4)}', str(len(v.get("values", [])))])
        r.p(R.table(
            ["Miền", "Khung", "Chỉ số", "Trung bình", "Độ lệch chuẩn", "Số lần chạy"],
            rows,
            "Kết quả qua năm hạt giống ngẫu nhiên. Cột độ lệch chuẩn chính là thước đo biên độ "
            "nhiễu mà mọi so sánh trong báo cáo phải vượt qua mới đáng tin."))
        r.p(R.figure("an_fig_seed_variance.png",
                     "Trung bình kèm thanh lỗi một độ lệch chuẩn qua năm hạt giống."))
        _seed_commentary(r, ms)

    # ---- 9.3
    r.h(2, "9.3. Kiểm định McNemar theo từng cặp khung")
    mc = d.get("mcnemar") or {}
    if mc:
        rows, n_sig, n_tot = [], 0, 0
        for dom, tests in mc.items():
            for t in tests:
                n_tot += 1
                sig = bool(t.get("significant"))
                n_sig += sig
                rows.append([dom, " với ".join(t["pair"]), str(t.get("b", "")),
                             str(t.get("c", "")), num(t.get("statistic", 0), 3),
                             num(t.get("p_value", 0), 4),
                             "<strong>Có</strong>" if sig else "Không"])
        r.p(R.table(
            ["Miền", "Cặp so sánh", "b", "c", "Thống kê", "p-value", "Khác biệt có ý nghĩa"],
            rows,
            "Kiểm định McNemar ở mức ý nghĩa 0,05. Cột b và c là số mẫu mà đúng một trong hai "
            "mô hình dự đoán sai, tức phần thông tin duy nhất mà phép kiểm sử dụng."))
        r.p(R.figure("an_fig_mcnemar_matrix.png",
                     "Ma trận p-value của kiểm định McNemar cho từng cặp khung."))
        _mcnemar_commentary(r, n_sig, n_tot)

    # ---- 9.4
    r.h(2, "9.4. Khoảng tin cậy Wilson")
    wl = d.get("wilson") or {}
    if wl:
        rows = []
        for dom, fws in wl.items():
            for fw, v in fws.items():
                rows.append([dom, fw, pct(v["accuracy"]), f'{v.get("n", 0):,}',
                             f'[{pct(v["lo"])}, {pct(v["hi"])}]',
                             pct(v["hi"] - v["lo"])])
        r.p(R.table(
            ["Miền", "Khung", "Accuracy", "Cỡ mẫu kiểm thử", "Khoảng tin cậy 95%", "Độ rộng"],
            rows,
            "Khoảng tin cậy Wilson 95% cho độ chính xác trên tập kiểm thử."))
        r.p(R.figure("an_fig_wilson_ci.png",
                     "Độ chính xác kèm khoảng tin cậy Wilson 95%. Các khoảng chồng lấn nhau là "
                     "dấu hiệu trực quan của việc không thể kết luận khác biệt."))
        r.p(
            "Cách đọc hình này rất trực tiếp: khi khoảng tin cậy của hai mô hình chồng lấn nhau, "
            "dữ liệu hiện có không đủ để nói mô hình nào tốt hơn. Bề rộng khoảng tin cậy còn cho "
            "biết giới hạn phân giải của phép đo, nghĩa là mọi khác biệt nhỏ hơn bề rộng đó đều "
            "nằm dưới ngưỡng mà tập kiểm thử này có thể phát hiện.")

    # ---- 9.5
    r.h(2, "9.5. Ý nghĩa đối với các chương trước")
    _stat_conclusion(r, d)


# ===========================================================================
# CHƯƠNG 10 — GIẢI PHẪU MÔ HÌNH
# ===========================================================================
def _chapter10(r: R.Report, data: dict) -> None:
    d = data.get("anatomy")
    r.h(1, "Chương 10. Mô hình đã học được gì và có đáng tin không")

    r.p(
        "Chương 6 và Chương 7 đều kết thúc bằng một lưới ảnh các dự đoán sai có độ tin cậy cao, "
        "kèm nhận xét rằng xác suất softmax lớn không đồng nghĩa với xác suất đã được hiệu chỉnh "
        "tốt. Nhận xét đó đúng nhưng vẫn chỉ là nhận xét. Chương này biến nó thành một con số, "
        "rồi sửa nó.")
    r.p(
        "Phần sau của chương đi xa hơn một bước, mở mô hình ra xem bên trong: bộ lọc tầng đầu "
        "trông như thế nào, biểu diễn thay đổi ra sao qua từng khối, và quan trọng nhất, mô hình "
        "thực sự nhìn vào vùng nào của bức ảnh khi ra quyết định.")

    if not d:
        _missing(r, "05_model_anatomy.ipynb", "reports/metrics_anatomy.json")
        return

    # ---- 10.1
    r.h(2, "10.1. Hiệu chỉnh xác suất và sai số hiệu chỉnh kỳ vọng")
    r.p(
        "Một mô hình được hiệu chỉnh tốt là mô hình mà trong số những lần nó tuyên bố tin cậy "
        "90%, đúng khoảng 90% là chính xác. Sai số hiệu chỉnh kỳ vọng đo độ lệch khỏi lý tưởng "
        "đó bằng cách chia các dự đoán thành 15 khoảng theo độ tin cậy rồi lấy trung bình có "
        "trọng số của chênh lệch giữa độ chính xác thực tế và độ tin cậy khai báo:")
    r.p('<div class="formula">ECE = Σ<sub>b</sub> (n<sub>b</sub> / N) · '
        '| acc(b) − conf(b) |</div>')
    _calibration_commentary(r, d)
    r.p(R.figure("an_fig_reliability_diagram.png",
                 "Biểu đồ độ tin cậy. Đường chéo là mô hình hiệu chỉnh hoàn hảo; cột nằm dưới "
                 "đường chéo nghĩa là mô hình tự tin quá mức so với năng lực thực."))
    r.p(
        "Phép sửa được dùng ở đây là <strong>nhiệt độ scaling</strong>, cách hiệu chỉnh tối "
        "giản nhất có thể: chia toàn bộ logits cho đúng <em>một</em> tham số vô hướng T trước "
        "khi qua softmax. Vì phép chia không làm đổi thứ tự các logits, <em>độ chính xác giữ "
        "nguyên tuyệt đối</em>; thứ duy nhất thay đổi là độ sắc của phân phối xác suất. Giá trị "
        "T lớn hơn 1 làm phân phối mềm đi, và việc tối ưu chọn ra T lớn hơn 1 chính là bằng "
        "chứng mô hình vốn quá tự tin.")
    r.p(R.note(
        "Một chi tiết quyết định tính hợp lệ.",
        "Tham số T được tối ưu trên <strong>tập kiểm định</strong>, tuyệt đối không dùng tập "
        "kiểm thử. Nếu tinh chỉnh T trên chính tập dùng để báo cáo ECE thì con số thu được là "
        "kết quả của việc khớp vào tập đó, không còn là phép đo khả năng tổng quát hoá."))
    r.p(R.figure("an_fig_temperature_scaling.png",
                 "Sai số hiệu chỉnh trước và sau khi áp dụng nhiệt độ scaling."))

    # ---- 10.2
    r.h(2, "10.2. Bộ lọc tầng tích chập thứ nhất")
    r.p(
        "Tầng tích chập đầu tiên nhìn thẳng vào điểm ảnh thô, nên trọng số của nó có thể hiển "
        "thị trực tiếp thành ảnh. Với CIFAR-10, mỗi bộ lọc có kích thước 3×3×3 nên vẽ được "
        "thành một ô màu RGB; với MNIST, bộ lọc 3×3×1 hiển thị ở thang xám.")
    r.p(R.figure("an_fig_conv1_filters.png",
                 "Toàn bộ bộ lọc của tầng tích chập thứ nhất, mỗi bộ lọc được chuẩn hoá độ "
                 "sáng độc lập để nhìn rõ cấu trúc."))
    r.p(
        "Cần đọc hình này với một mức kỳ vọng hợp lý. Những bộ lọc Gabor sắc nét thường thấy "
        "trong sách giáo khoa đến từ các mạng có tầng đầu 7×7 hoặc 11×11 huấn luyện trên "
        "ImageNet. Ở đây bộ lọc chỉ 3×3, tức chỉ chín trọng số cho mỗi kênh, nên cấu trúc học "
        "được nhất thiết phải thô hơn nhiều. Điều đáng tìm là tính <em>đối lập</em>: một phía "
        "sáng và phía kia tối thì đó là bộ dò cạnh, còn các kênh màu lệch nhau thì đó là bộ dò "
        "đối lập màu.")

    # ---- 10.3
    r.h(2, "10.3. Biểu diễn thay đổi qua chiều sâu")
    r.p(R.figure("an_fig_feature_maps.png",
                 "Bản đồ đặc trưng của một ảnh kiểm thử sau từng khối tích chập, kèm kích "
                 "thước tensor tại mỗi chặng."))
    r.p(
        "Hai xu hướng chạy ngược nhau qua các khối. Chiều không gian co lại dần do gộp cực đại, "
        "trong khi số kênh tăng lên. Nói cách khác mạng đánh đổi <em>độ phân giải vị trí</em> "
        "lấy <em>độ phong phú ngữ nghĩa</em>: các tầng đầu còn giữ được hình dáng nhận ra được "
        "của vật thể, các tầng sau chỉ còn những mảng kích hoạt rời rạc mà từng kênh phản ứng "
        "với một loại hoa văn riêng. Đây chính là cơ chế đứng sau không gian biểu diễn ẩn mà "
        "Chương 8 đã chiếu bằng PCA.")

    # ---- 10.4
    r.h(2, "10.4. Mô hình thực sự nhìn vào đâu")
    r.p(
        "Ba phần trên mô tả cái mô hình học được, nhưng chưa trả lời câu hỏi quan trọng nhất về "
        "độ tin cậy: khi phân loại một bức ảnh, mô hình dựa vào vùng nào? Phép thử che khuất trả "
        "lời bằng cách trượt một ô vuông xám 8×8 khắp bức ảnh và ghi lại xác suất của lớp đúng "
        "tại mỗi vị trí che.")
    r.p(
        "Logic rất rõ ràng. Nếu che một vùng làm xác suất lớp đúng tụt mạnh, vùng đó mang thông "
        "tin quyết định. Nếu che vùng nào cũng không ảnh hưởng, mô hình đang dựa vào tín hiệu "
        "phân tán khắp ảnh. Còn nếu xác suất chỉ tụt khi che <em>nền</em> chứ không phải vật "
        "thể, đó là dấu hiệu mô hình bắt được một mối tương quan giả trong dữ liệu huấn luyện "
        "thay vì học đặc trưng của vật thể.")
    r.p(R.figure("an_fig_occlusion_sensitivity.png",
                 "Bản đồ nhiệt độ nhạy che khuất. Vùng càng tối nghĩa là che vào đó càng làm "
                 "xác suất lớp đúng giảm mạnh, tức vùng đó càng quan trọng với quyết định."))
    _occlusion_commentary(r, d)


# ===========================================================================
# CHƯƠNG 11 — BÓC TÁCH GIAI THỪA VÀ TỈ LỆ DỮ LIỆU
# ===========================================================================
def _chapter11(r: R.Report, data: dict) -> None:
    d = data.get("ablation")
    r.h(1, "Chương 11. Bóc tách những gì các chương trước phải gộp chung")

    r.p(
        "Hai chỗ trong báo cáo này buộc phải thừa nhận là chưa tách bạch được nguyên nhân. "
        "Chương 6 và Chương 7 gọi mô hình cải tiến là cải tiến, nhưng thực chất bật cùng lúc ba "
        "thay đổi nên không biết cái nào mới là cái có tác dụng. Và cả hai chương đều nói rằng "
        "khoảng cách giữa NumPy với framework trộn lẫn ba nguyên nhân: ít dữ liệu hơn, kiến trúc "
        "nông hơn, tầng hiện thực khác. Chương này gỡ cả hai.")

    if not d:
        _missing(r, "06_ablation_scaling.ipynb", "reports/metrics_ablation.json")
        return

    # ---- 11.1
    r.h(2, "11.1. Thiết kế giai thừa đầy đủ cho gói cải tiến")
    r.p(
        "Cách làm thông thường là tắt từng yếu tố một rồi xem hiệu năng tụt bao nhiêu. Cách đó "
        "có một điểm mù: nó không phát hiện được <em>tương tác</em>, tức trường hợp hai yếu tố "
        "chỉ phát huy khi đi cùng nhau trong khi từng cái riêng lẻ không có tác dụng gì. Vì chỉ "
        "có ba yếu tố nên chạy đủ cả tám tổ hợp là khả thi, và thiết kế giai thừa đầy đủ cho "
        "phép ước lượng cả hiệu ứng chính lẫn hiệu ứng tương tác.")
    fac = d.get("factorial") or {}
    combos = fac.get("combinations") or []
    if combos:
        rows = [[("bật" if c.get("padding") else "tắt"),
                 ("bật" if c.get("he_init") else "tắt"),
                 ("bật" if c.get("lr_decay") else "tắt"),
                 pct(c["accuracy"])] for c in combos]
        r.p(R.table(
            ["Đệm viền", "He Normal", "Giảm tốc độ học", "Accuracy kiểm thử"],
            rows,
            f"Tám tổ hợp của thiết kế giai thừa 2³, huấn luyện trên "
            f"{fac.get('n_train', 0):,} ảnh trong {fac.get('epochs', 0)} epoch."))
    me = fac.get("main_effects") or {}
    if me:
        label = {"padding": "Đệm viền", "he_init": "Khởi tạo He Normal",
                 "lr_decay": "Lịch giảm tốc độ học"}
        order = sorted(me, key=lambda k: abs(me[k]), reverse=True)
        r.p(R.table(
            ["Yếu tố", "Hiệu ứng chính (điểm phần trăm)", "Xếp hạng theo độ lớn"],
            [[label.get(k, k), f'{100 * me[k]:+.2f}', str(i + 1)]
             for i, k in enumerate(order)],
            "Hiệu ứng chính của từng yếu tố, tính bằng chênh lệch trung bình giữa nhóm bật và "
            "nhóm tắt trên cả bốn tổ hợp của hai yếu tố còn lại."))
        _factorial_commentary(r, me, label, order, fac.get("interactions") or {})
    r.p(R.figure("an_fig_ablation_factorial.png",
                 "Tám tổ hợp giai thừa và hiệu ứng chính của từng yếu tố."))

    # ---- 11.2
    r.h(2, "11.2. Đường cong hiệu năng theo cỡ dữ liệu huấn luyện")
    r.p(
        "Phần này trả lời một câu hỏi cụ thể: trong khoảng cách giữa mô hình NumPy và mô hình "
        "framework, bao nhiêu phần là do NumPy được huấn luyện trên ít dữ liệu hơn? Cách đo là "
        "giữ nguyên kiến trúc rồi thay đổi duy nhất cỡ tập huấn luyện, từ 500 ảnh lên toàn bộ.")
    r.p(R.figure("an_fig_learning_curve.png",
                 "Độ chính xác theo cỡ tập huấn luyện ở thang log, mỗi điểm lấy trung bình ba "
                 "hạt giống kèm dải một độ lệch chuẩn. Đường dọc đánh dấu cỡ tập con mà hai mô "
                 "hình NumPy đã dùng."))
    r.p(
        "Việc lặp ba hạt giống ở đây không phải cho đủ thủ tục. Ở vùng dữ liệu ít, dao động giữa "
        "các lần chạy lớn tới mức một điểm đơn lẻ có thể lệch vài điểm phần trăm, nên một đường "
        "cong vẽ từ một hạt giống duy nhất sẽ gợi ra những chỗ gấp khúc hoàn toàn không có thật.")

    # ---- 11.3
    r.h(2, "11.3. Tách phần do dữ liệu khỏi phần còn lại")
    _gap_commentary(r, d)
    r.p(R.figure("an_fig_gap_decomposition.png",
                 "Tách tổng khoảng cách thành phần do lượng dữ liệu và phần dư còn lại."))
    r.p(R.note(
        "Giới hạn của phép tách này, nói trước khi người đọc tự nhận ra.",
        "Phép tách chỉ bóc được <strong>một</strong> lớp nguyên nhân là lượng dữ liệu. Phần dư "
        "vẫn gộp chung kiến trúc với tầng hiện thực, vì mô hình NumPy đồng thời nông hơn "
        "<em>và</em> được viết bằng một công cụ khác. Muốn tách nốt hai thứ đó thì phải dựng "
        "thêm một mô hình framework có kiến trúc y hệt mô hình NumPy, và đó là việc nằm ngoài "
        "phạm vi bài tập này. Vì vậy con số phần dư nên được đọc là <em>cận trên</em> của tác "
        "động kiến trúc, chứ không phải phép đo trực tiếp."))


# ===========================================================================
# Bình luận sinh tự động từ số liệu
# ===========================================================================
def _seed_commentary(r: R.Report, ms: dict) -> None:
    worst = None
    for dom, fws in ms.items():
        for fw, v in fws.items():
            if worst is None or v["std"] > worst[2]:
                worst = (dom, fw, v["std"])
    if not worst:
        return
    r.p(
        f"Độ lệch chuẩn lớn nhất quan sát được là {num(worst[2], 4)}, ở khung {worst[1]} trên "
        f"miền {worst[0]}. Con số này đặt ra một ngưỡng đọc cho toàn bộ báo cáo: "
        f"<strong>mọi chênh lệch nhỏ hơn khoảng hai lần độ lệch chuẩn đều không nên được diễn "
        f"giải như một khác biệt thật</strong>. Nhiều khoảng cách giữa các khung trong Chương 3 "
        f"đến Chương 7 rơi đúng vào vùng đó, và đây là cơ sở định lượng cho những câu trong các "
        f"chương ấy vốn chỉ mới nói rằng chênh lệch nằm trong biên độ dao động.")


def _mcnemar_commentary(r: R.Report, n_sig: int, n_tot: int) -> None:
    if not n_tot:
        return
    if n_sig == 0:
        r.p(
            f"Không một cặp nào trong {n_tot} cặp được kiểm cho kết quả khác biệt có ý nghĩa "
            f"thống kê ở mức 0,05. Đây <strong>không</strong> phải kết quả rỗng mà là kết quả "
            f"mạnh, và nó xác nhận đúng luận điểm trung tâm của báo cáo: khi ba khung được cấp "
            f"cùng một kiến trúc, cùng dữ liệu và cùng quy trình, chúng cho ra những bộ phân "
            f"loại mà phép kiểm thống kê không phân biệt nổi. Thứ quyết định kết quả là mô hình "
            f"toán học, còn tầng hiện thực chỉ quyết định công sức viết mã và tốc độ chạy.")
    else:
        r.p(
            f"Có {n_sig} trong {n_tot} cặp cho khác biệt có ý nghĩa thống kê ở mức 0,05. Với "
            f"những cặp này, chênh lệch không quy về nhiễu khởi tạo được và cần truy nguyên. "
            f"Các nghi phạm thông thường là khác biệt mặc định giữa các thư viện về quy tắc khởi "
            f"tạo trọng số, tham số beta của Adam, cách xử lý epsilon trong chuẩn hoá, và thứ tự "
            f"cộng dồn số thực trên các thiết bị khác nhau. Những cặp còn lại vẫn nằm trong vùng "
            f"không phân biệt được.")
    r.p(
        "Một lưu ý khi đọc: không bác bỏ được giả thuyết không <em>không</em> chứng minh hai mô "
        "hình bằng nhau. Nó chỉ nói rằng tập kiểm thử hiện có không đủ bằng chứng để khẳng định "
        "chúng khác nhau. Muốn phát hiện khác biệt nhỏ hơn thì cần tập kiểm thử lớn hơn.")


def _stat_conclusion(r: R.Report, d: dict) -> None:
    mc = d.get("mcnemar") or {}
    n_sig = sum(1 for tests in mc.values() for t in tests if t.get("significant"))
    n_tot = sum(len(tests) for tests in mc.values())
    r.p(
        "Chương này thay đổi cách nên đọc mọi bảng đối chuẩn ở các chương trước. Trước đây mỗi "
        "bảng đều có một dòng dẫn đầu, và cách trình bày ấy ngầm gợi ý rằng có một khung thắng "
        "cuộc.")
    if n_tot and n_sig == 0:
        r.p(
            "Kết quả kiểm định cho thấy cách đọc đó là sai. Thứ tự trong bảng phản ánh dao động "
            "ngẫu nhiên chứ không phản ánh chất lượng, nên cách diễn đạt đúng là ba khung "
            "<em>tương đương trong giới hạn đo được</em>. Báo cáo giữ nguyên các bảng vì chúng "
            "vẫn cho biết độ lớn tuyệt đối của hiệu năng, nhưng người đọc nên bỏ qua thứ hạng.")
    else:
        r.p(
            "Kết quả kiểm định cho thấy cách đọc đó chỉ đúng một phần. Một số cặp thực sự khác "
            "biệt, phần lớn còn lại thì không, nên thứ hạng trong bảng chỉ có nghĩa ở những chỗ "
            "phép kiểm xác nhận.")
    r.p(
        "Đóng góp phương pháp luận của chương nằm ở chỗ này: một bảng số liệu không kèm thước đo "
        "bất định thì không thể phân biệt phát hiện thật với nhiễu, và việc bổ sung thước đo đó "
        "tốn thêm bốn lần chạy cho mỗi cấu hình chứ không đòi hỏi kỹ thuật nào mới.")


def _calibration_commentary(r: R.Report, d: dict) -> None:
    cal = d.get("calibration") or {}
    if not cal:
        return
    rows = []
    for ds, v in cal.items():
        before, after = v.get("ece_before"), v.get("ece_after")
        red = (1 - after / before) * 100 if before else 0
        rows.append([ds, num(before, 4), num(after, 4), num(v.get("temperature", 0), 3),
                     f'{red:.1f}%'])
    r.p(R.table(
        ["Tập dữ liệu", "ECE trước", "ECE sau", "Nhiệt độ T", "Mức giảm"],
        rows,
        "Sai số hiệu chỉnh kỳ vọng trước và sau nhiệt độ scaling, với 15 khoảng độ tin cậy."))
    for ds, v in cal.items():
        T = v.get("temperature", 1)
        if T > 1.05:
            r.p(
                f"Trên {ds}, nhiệt độ tối ưu là T = {num(T, 3)}, lớn hơn 1 một cách rõ rệt. Đây "
                f"là bằng chứng định lượng cho hiện tượng đã mô tả định tính ở Chương 6 và "
                f"Chương 7: mô hình <strong>tự tin quá mức</strong>. Phép chia logits cho T làm "
                f"mềm phân phối xác suất và kéo ECE từ {num(v['ece_before'], 4)} xuống "
                f"{num(v['ece_after'], 4)}, trong khi độ chính xác không đổi một chữ số nào vì "
                f"phép chia không hoán đổi thứ tự các logits.")
        elif T < 0.95:
            r.p(
                f"Trên {ds}, nhiệt độ tối ưu là T = {num(T, 3)}, nhỏ hơn 1. Trái với kỳ vọng "
                f"thông thường, mô hình ở đây <em>thiếu</em> tự tin chứ không thừa, nên phép "
                f"hiệu chỉnh làm phân phối sắc lại.")
        else:
            r.p(
                f"Trên {ds}, nhiệt độ tối ưu T = {num(T, 3)} gần như bằng 1, nghĩa là mô hình "
                f"vốn đã được hiệu chỉnh khá tốt và gần như không còn gì để sửa.")


def _occlusion_commentary(r: R.Report, d: dict) -> None:
    occ = d.get("occlusion") or {}
    for ds, items in occ.items():
        if not items:
            continue
        drops = [it.get("prob_drop_max", 0) for it in items]
        mx = max(drops)
        r.p(
            f"Trên {ds}, mức tụt xác suất lớn nhất khi che là {pct(mx)} trên {len(items)} ảnh "
            f"được khảo sát. "
            + ("Mức tụt lớn như vậy cho thấy quyết định của mô hình tập trung vào một vùng "
               "không gian hẹp và xác định, chứ không rải đều khắp ảnh."
               if mx > 0.3 else
               "Mức tụt khiêm tốn này cho thấy mô hình dựa vào tín hiệu phân tán nhiều nơi, nên "
               "che một ô vuông đơn lẻ không đủ phá vỡ quyết định."))


def _factorial_commentary(r: R.Report, me: dict, label: dict, order: list,
                          inter: dict) -> None:
    top = order[0]
    r.p(
        f"Yếu tố có ảnh hưởng lớn nhất là <strong>{label.get(top, top)}</strong> với hiệu ứng "
        f"chính {100 * me[top]:+.2f} điểm phần trăm.")
    weak = [k for k in order if abs(me[k]) < 0.005]
    if weak:
        r.p(
            "Đáng chú ý hơn là " + ", ".join(label.get(k, k) for k in weak) +
            " có hiệu ứng chính gần như bằng không. Phát hiện này trực tiếp giải thích vì sao "
            "gói cải tiến ở một số miền không thắng nổi mô hình cơ sở: gói đó gộp cả yếu tố có "
            "tác dụng lẫn yếu tố không có tác dụng, nên hiệu quả tổng bị pha loãng. Nếu chỉ "
            "quan sát ở mức gói, không cách nào thấy được điều này.")
    neg = [k for k in order if me[k] < -0.005]
    if neg:
        r.p(
            "Có yếu tố cho hiệu ứng <strong>âm</strong>: " +
            ", ".join(f"{label.get(k, k)} ({100 * me[k]:+.2f} điểm)" for k in neg) +
            ". Nghĩa là bật yếu tố đó lên trung bình làm hiệu năng giảm. Đây là loại kết quả mà "
            "một nghiên cứu bóc tách sinh ra để tìm, và nó chỉ lộ ra khi các yếu tố được tách "
            "riêng thay vì bật cùng lúc.")
    if inter:
        big = max(inter, key=lambda k: abs(inter[k]))
        if abs(inter[big]) > 0.005:
            r.p(
                f"Tương tác đáng kể nhất là {big} với độ lớn {100 * inter[big]:+.2f} điểm phần "
                f"trăm. Tương tác khác không có nghĩa là tác động của hai yếu tố này không cộng "
                f"tuyến tính, nên kết luận rút ra từ việc thử từng yếu tố một sẽ không còn đúng "
                f"khi chúng được dùng chung.")
        else:
            r.p(
                "Cả ba tương tác đều nhỏ, nên trong phạm vi thực nghiệm này ba yếu tố tác động "
                "gần như độc lập và cộng tuyến tính với nhau.")


def _gap_commentary(r: R.Report, d: dict) -> None:
    g = d.get("gap_decomposition") or {}
    if not g:
        return
    total = g.get("total_gap", 0)
    dc, rc = g.get("data_component", 0), g.get("residual_component", 0)
    r.p(R.table(
        ["Thành phần", "Giá trị", "Điểm phần trăm", "Tỉ lệ trong tổng khoảng cách"],
        [["Mô hình NumPy cải tiến", pct(g.get("a_numpy", 0)),
          "—", "—"],
         [f"Framework tại cùng cỡ dữ liệu ({g.get('n_small', 0):,} ảnh)",
          pct(g.get("a_small", 0)), f"{100 * rc:+.2f}",
          f"{100 * rc / total:.1f}%" if total else "—"],
         [f"Framework tại dữ liệu đầy đủ ({g.get('n_full', 0):,} ảnh)",
          pct(g.get("a_full", 0)), f"{100 * dc:+.2f}",
          f"{100 * dc / total:.1f}%" if total else "—"],
         ["<strong>Tổng khoảng cách</strong>", "—",
          f"<strong>{100 * total:+.2f}</strong>", "<strong>100%</strong>"]],
        "Tách khoảng cách giữa mô hình NumPy và mô hình framework thành phần do lượng dữ liệu "
        "và phần dư."))
    if total:
        r.p(
            f"Trong tổng khoảng cách {100 * total:.2f} điểm phần trăm, có "
            f"<strong>{100 * dc:.2f} điểm là do lượng dữ liệu huấn luyện</strong> "
            f"({100 * dc / total:.0f}%) và {100 * rc:.2f} điểm còn lại đến từ kiến trúc cùng "
            f"tầng hiện thực ({100 * rc / total:.0f}%).")
        if dc > rc:
            r.p(
                "Phần do dữ liệu chiếm ưu thế. Kết luận rút ra là hiện thực NumPy viết tay "
                "<em>không</em> kém về mặt thuật toán như con số thô gợi ra; nó chủ yếu bị thiệt "
                "vì được cho ăn ít dữ liệu hơn, mà nguyên nhân của việc đó lại là ràng buộc thời "
                "gian chạy trên CPU chứ không phải khiếm khuyết của mã nguồn.")
        else:
            r.p(
                "Phần dư chiếm ưu thế, nghĩa là ngay cả khi được cấp cùng lượng dữ liệu thì mô "
                "hình NumPy vẫn còn cách framework một khoảng đáng kể. Điều này chỉ về phía kiến "
                "trúc: mạng nông hai khối tích chập không có Batch Normalization và Dropout thì "
                "thiếu dung lượng biểu diễn so với mạng sâu, bất kể dữ liệu nhiều đến đâu.")
