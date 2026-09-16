"""Chương 3, 4, 5 — ba hệ thống thông minh quy mô lớn.

Chương 3: Sàng lọc nguy cơ tiểu đường (100,000 hồ sơ, phân loại nhị phân).
Chương 4: Định giá bất động sản (150,000 giao dịch, hồi quy).
Chương 5: Phân loại nhận xét thương mại điện tử (23,486 đánh giá, NLP).

Mọi con số đều đọc từ metadata.json do notebook sinh ra.
"""

from __future__ import annotations

import report_lib as R


def write(r: R.Report, data: dict) -> None:
    _chapter3(r, data["s1"])
    _chapter4(r, data["s2"])
    _chapter5(r, data["s3"])


def pct(x, nd=2) -> str:
    return f"{x * 100:.{nd}f}%"


def usd(x, nd=0) -> str:
    return f"${x:,.{nd}f}"


# ===========================================================================
# CHƯƠNG 3 — HỆ THỐNG 1: SÀNG LỌC TIỂU ĐƯỜNG 100,000 MẪU
# ===========================================================================
def _chapter3(r: R.Report, s1: dict) -> None:
    ds = s1["dataset"]
    sp = s1["split"]
    sc = s1["scores"]
    dl = s1["dl_model"]
    corr = s1["target_correlation"]

    r.h(1, "Chương 3. Hệ thống 1 — Sàng lọc nguy cơ tiểu đường quy mô lớn "
           "(100,000 mẫu dữ liệu)")

    r.p(
        "Hai chương đầu làm việc trên 768 mẫu — đủ để hiểu cơ chế vận hành của mạng "
        "nơ-ron, nhưng quá nhỏ để nói bất cứ điều gì chắc chắn về năng lực thực tế. Từ "
        "chương này trở đi, quy mô nhảy lên <strong>100,000 hồ sơ bệnh nhân</strong>, và "
        "câu hỏi nghiên cứu thay đổi hoàn toàn:",
        '<div class="flow">Trên dữ liệu bảng quy mô lớn, một mạng nơ-ron sâu viết tay bằng '
        'NumPy có đủ sức cạnh tranh với các mô hình Machine Learning cổ điển đã được tối ưu '
        'hàng chục năm hay không?</div>',
        "Để trả lời, chương này dựng một cuộc đối đầu bốn bên trên <strong>cùng một tập dữ "
        "liệu, cùng một phép chia phân tầng, cùng một ma trận đặc trưng 14 chiều</strong>. "
        "Không mô hình nào được ưu ái thêm bất kỳ thông tin nào.",
    )

    r.p(R.table(
        ["Nhóm", "Mô hình", "Bản chất thuật toán"],
        [
            ["Machine Learning", "Logistic Regression", "Ranh giới quyết định tuyến tính"],
            ["Machine Learning", "Decision Tree", "Lát cắt trực giao theo trục đặc trưng"],
            ["Machine Learning", "Random Forest (100 cây)", "Ensemble Bagging giảm phương sai"],
            ["<strong>Deep Learning</strong>",
             f"<strong>{dl['name']} thuần NumPy</strong>",
             "<strong>Biểu diễn phân cấp phi tuyến</strong>"],
        ],
        caption="Bốn mô hình tham gia đối đầu trong bài toán sàng lọc tiểu đường quy mô lớn.",
    ))

    # ---------------- 3.1 EDA ----------------
    r.h(2, "3.1. Bối cảnh y tế lâm sàng và khám phá dữ liệu (EDA)")
    r.p(
        f"Tập dữ liệu <code class=\"inl\">{ds['name']}</code> gồm "
        f"<strong>{ds['n_raw']:,} hồ sơ bệnh nhân</strong> với các thông số lâm sàng và "
        "nhân khẩu học: Tuổi, Giới tính, Chỉ số khối cơ thể BMI, Tiền sử hút thuốc, Tăng "
        "huyết áp, Bệnh tim mạch, HbA1c (đường huyết trung bình ba tháng) và Glucose lúc đói.",
    )

    # Hình 1 được vẽ trên dữ liệu THÔ (trước bước làm sạch ở mục 3.2), nên phần
    # bình luận phải dùng đúng thống kê thô — không lẫn với tỷ lệ sau làm sạch.
    n_pos = ds.get("n_raw_positive", round(ds["n_raw"] * ds["positive_rate"]))
    n_neg = ds.get("n_raw_negative", ds["n_raw"] - n_pos)
    raw_rate = ds.get("raw_positive_rate", n_pos / ds["n_raw"])

    r.p(R.figure(
        "s1_fig01_target_distribution.png",
        "Phân phối nhãn mục tiêu sàng lọc tiểu đường trên dữ liệu thô — thách thức mất cân "
        "bằng lớp lâm sàng.",
        width="72%"))

    r.p(R.oim(
        f"Chỉ <strong>{n_pos:,} bệnh nhân ({pct(raw_rate, 2)})</strong> trong "
        f"{ds['n_raw']:,} hồ sơ thực sự mắc tiểu đường (nhãn 1), còn lại {n_neg:,} người "
        f"({pct(1 - raw_rate, 2)}) khoẻ mạnh (nhãn 0). Tỷ lệ mất cân bằng xấp xỉ "
        f"1 : {(1 - raw_rate) / raw_rate:.1f}.",
        "Đây không phải lỗi thu thập dữ liệu mà phản ánh đúng tỷ lệ mắc bệnh trong quần thể "
        "dân cư. Tuy nhiên nó tạo ra một cái bẫy đánh giá cực kỳ nguy hiểm.",
        f"Một mô hình \"ngu ngốc\" luôn dự đoán ÂM TÍNH cho mọi bệnh nhân vẫn đạt Accuracy "
        f"<strong>{pct(1 - raw_rate, 2)}</strong> — con số nghe rất ấn tượng. "
        f"Nhưng toàn bộ <strong>{n_pos:,} bệnh nhân mắc bệnh đều bị bỏ sót</strong>, mất "
        f"hoàn toàn cơ hội can thiệp điều trị sớm. Đây chính là lý do Accuracy "
        f"<strong>không bao giờ</strong> là chỉ số đủ tin cậy trên dữ liệu y tế mất cân "
        f"bằng — phải nhìn vào Recall và F1-Score."))

    r.p(
        f"Sau bước làm sạch trình bày ở mục 3.2, tập dữ liệu còn {ds['n_clean']:,} hồ sơ và "
        f"tỷ lệ nhãn bệnh nhích lên <strong>{pct(ds['positive_rate'], 2)}</strong> — đây mới "
        f"là tỷ lệ được dùng cho toàn bộ phần huấn luyện và đánh giá phía sau.")

    r.p(R.figure(
        "s1_fig02_feature_distributions.png",
        "Phân phối các chỉ số sinh hoá và nhân khẩu học theo nhãn bệnh. Hai đường ngưỡng "
        "chẩn đoán chuẩn của y học được vẽ kèm để đối chiếu: HbA1c ≥ 6,5% (tiêu chuẩn ADA) "
        "và Glucose > 200 mg/dL."))

    r.p(R.figure(
        "s1_fig03_correlation_heatmap.png",
        "Ma trận tương quan Pearson đầy đủ giữa các đặc trưng và nhãn mục tiêu.",
        width="80%"))

    top_corr = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:5]
    r.p(R.table(
        ["Đặc trưng", "Hệ số tương quan Pearson r", "Diễn giải y học"],
        [
            ["<code class=\"inl\">blood_glucose_level</code>", f"{corr['blood_glucose_level']:+.4f}",
             "Đường huyết lúc đói — chỉ số chẩn đoán trực tiếp"],
            ["<code class=\"inl\">HbA1c_level</code>", f"{corr['HbA1c_level']:+.4f}",
             "Đường huyết trung bình 3 tháng — tiêu chuẩn ADA"],
            ["<code class=\"inl\">age</code>", f"{corr['age']:+.4f}",
             "Tuổi tác — yếu tố nguy cơ tích luỹ theo thời gian"],
            ["<code class=\"inl\">bmi</code>", f"{corr['bmi']:+.4f}",
             "Béo phì làm giảm độ nhạy insulin"],
            ["<code class=\"inl\">hypertension</code>", f"{corr['hypertension']:+.4f}",
             "Tăng huyết áp đi kèm hội chứng chuyển hoá"],
        ],
        caption="Năm đặc trưng tương quan mạnh nhất với nhãn bệnh, xếp theo độ lớn.",
        cls="tight"))

    r.p(
        f"Hai chỉ số đường huyết dẫn đầu tuyệt đối: "
        f"<code class=\"inl\">blood_glucose_level</code> (r = {corr['blood_glucose_level']:.2f}) "
        f"và <code class=\"inl\">HbA1c_level</code> (r = {corr['HbA1c_level']:.2f}). Kết quả "
        "này khớp chính xác với y văn, xác nhận dữ liệu phản ánh đúng cơ chế bệnh học chứ "
        "không phải nhiễu ngẫu nhiên.")

    # ---------------- 3.2 Làm sạch + FE ----------------
    r.h(2, "3.2. Kiểm toán làm sạch dữ liệu và kỹ nghệ đặc trưng lâm sàng")
    r.p(
        "Để đảm bảo chất lượng dữ liệu đầu vào cho cả bốn mô hình, một quy trình kiểm toán "
        "làm sạch nghiêm ngặt qua bốn giai đoạn được thiết lập. Nguyên tắc bất di bất dịch: "
        "<strong>mọi bản ghi bị loại đều phải có lý do lâm sàng ghi rõ</strong>, không được "
        "im lặng vứt dữ liệu.")

    r.p(R.table(
        ["Bước xử lý", "Tiêu chí sàng lọc lâm sàng", "Số loại bỏ", "Số còn lại", "Tỷ lệ giữ"],
        [[a["Bước Xử lý"], a["Tiêu chí Sàng lọc Lâm sàng"],
          f"{a['Số lượng Loại bỏ']:,}", f"{a['Số lượng Còn lại']:,}",
          f"{a['Tỷ lệ Giữ lại (%)']:.2f}%"] for a in s1["cleaning_audit"]],
        caption=f"Bảng kiểm toán quy trình làm sạch dữ liệu y tế sàng lọc tiểu đường "
                f"({ds['n_raw']:,} mẫu ban đầu)."))

    r.h(3, "3.2.1. Kỹ nghệ đặc trưng tương tác lâm sàng")
    r.p(
        "Mạng nơ-ron về lý thuyết có thể tự học các tương tác phi tuyến. Nhưng cung cấp sẵn "
        "tín hiệu tương tác dựa trên kiến thức y sinh chuyên môn giúp mạng hội tụ nhanh hơn, "
        "và quan trọng hơn — giúp <strong>cả các mô hình tuyến tính cũng được hưởng lợi</strong>, "
        "điều kiện cần cho một cuộc đối đầu công bằng. Hai đặc trưng tương tác được xây dựng:")

    r.p(
        "<p><strong>1. Tải lượng chuyển hoá kết hợp</strong> "
        "(<code class=\"inl\">glucose_hba1c_interaction</code>):</p>"
        '<div class="formula">glucose_hba1c_interaction = '
        "(blood_glucose_level × HbA1c_level) / 100</div>",
        "Đặc trưng này mô tả tác động <strong>cộng hưởng</strong> khi cả đường huyết tức thời "
        "và đường huyết tích luỹ ba tháng cùng ở mức cao. Một bệnh nhân có Glucose cao nhưng "
        "HbA1c bình thường có thể chỉ vừa ăn xong; nhưng cả hai cùng cao là dấu hiệu rối loạn "
        "chuyển hoá mạn tính.",
        "<p><strong>2. Tương tác tuổi tác và tăng huyết áp</strong> "
        "(<code class=\"inl\">age_hypertension_risk</code>):</p>"
        '<div class="formula">age_hypertension_risk = age × hypertension</div>',
        "Phản ánh gia tốc nguy cơ biến chứng tim mạch ở nhóm bệnh nhân lớn tuổi có tiền sử "
        "huyết áp cao. Với người không tăng huyết áp, đặc trưng này bằng 0.")

    r.h(3, "3.2.2. Kiến trúc tiền xử lý ColumnTransformer chống rò rỉ dữ liệu")
    r.p(
        f"Toàn bộ {ds['n_clean']:,} mẫu sạch được phân chia phân tầng 80/20 "
        f"(<code class=\"inl\">stratify=y</code>) thành tập Train ({sp['n_train']:,} mẫu) và "
        f"tập Test ({sp['n_test']:,} mẫu), bảo toàn chính xác tỷ lệ "
        f"{pct(sp['train_pos_rate'], 2)} nhãn bệnh ở cả hai tập. Quy trình mã hoá và chuẩn "
        f"hoá được đóng gói qua đường ống mô-đun hoá "
        f"<code class=\"inl\">ColumnTransformerScratch</code> với cấu trúc "
        f"<strong>{s1['n_features']} chiều đặc trưng minh bạch</strong>:")

    r.p(R.table(
        ["Nhóm biến", "Cách xử lý", "Số chiều"],
        [
            ["<code class=\"inl\">gender</code> (2 giá trị sau lọc)",
             "One-Hot Encoding, <code class=\"inl\">drop='first'</code>", "1"],
            ["<code class=\"inl\">smoking_history</code> (6 danh mục)",
             "One-Hot Encoding, <code class=\"inl\">drop='first'</code>", "5"],
            ["4 biến lâm sàng + 2 biến tương tác mới tạo",
             "Chuẩn hoá Z-Score (fit chỉ trên Train)", "6"],
            ["<code class=\"inl\">hypertension</code>, <code class=\"inl\">heart_disease</code>",
             "Giữ nguyên Passthrough (đã là nhị phân)", "2"],
            ["<strong>Tổng cộng</strong>", "", f"<strong>{s1['n_features']}</strong>"],
        ],
        caption="Cấu trúc không gian đặc trưng 14 chiều sau tiền xử lý."))

    r.p(R.note(
        "Vì sao dùng <code class=\"inl\">drop='first'</code>?",
        "Với k danh mục, One-Hot đầy đủ tạo ra k cột nhưng chỉ có k−1 cột độc lập tuyến tính "
        "(vì tổng các cột luôn bằng 1). Cột dư thừa gây hiện tượng <strong>đa cộng tuyến hoàn "
        "hảo</strong>, làm ma trận thiết kế suy biến và phá hỏng nghiệm của Logistic Regression. "
        "Bỏ một cột làm mốc tham chiếu triệt tiêu hoàn toàn vấn đề này."))

    # ---------------- 3.3 Kiến trúc ----------------
    r.h(2, "3.3. Khảo sát đánh đổi kiến trúc: Chiều sâu và Chiều rộng")
    r.p(
        "Để đánh giá tương quan giữa năng lực biểu diễn và độ phức tạp kiến trúc "
        "(Depth vs Width Trade-off), lớp mạng đa tầng tuỳ biến "
        "<code class=\"inl\">DeepMLPScratch</code> thuần NumPy được huấn luyện trên bốn cấu "
        "hình có kiểm soát — <strong>cùng dữ liệu, cùng seed, cùng tốc độ học η = "
        f"{dl['lr']}, cùng {dl['epochs']} epochs, cùng kích thước lô {dl['batch_size']}</strong>. "
        "Chỉ hình dạng mạng thay đổi.")

    r.p(R.note(
        "Chuyển sang Mini-Batch Gradient Descent.",
        f"Ở quy mô {sp['n_train']:,} mẫu, Full-Batch Gradient Descent (dùng ở Chương 2) trở nên "
        f"không hiệu quả: mỗi epoch chỉ cập nhật trọng số đúng một lần. Chương này chuyển sang "
        f"Mini-Batch với kích thước lô {dl['batch_size']}, tức khoảng "
        f"{sp['n_train'] // dl['batch_size']:,} bước cập nhật mỗi epoch. Nhiễu ngẫu nhiên từ "
        f"việc lấy mẫu lô còn giúp mạng thoát khỏi các cực tiểu địa phương nông."))

    arch = s1["architecture_study"]
    r.p(R.table(
        ["Kiến trúc", "Cấu hình tầng", "Tham số", "Thời gian", "Loss cuối",
         "Accuracy", "Recall", "F1-Score"],
        [[(f"<strong>{a['Kiến trúc']}</strong>"
           if a["Kiến trúc"] == dl["name"] else a["Kiến trúc"]),
          a["Cấu hình tầng"], f"{a['Tham số']:,}", a["Thời gian"],
          f"{a['Loss cuối']:.4f}", a["Accuracy"], a["Recall"], a["F1-Score"]]
         for a in arch],
        caption="Bảng đối chứng thực nghiệm khảo sát kiến trúc mạng nơ-ron sâu thuần NumPy "
                f"trên {ds['n_raw']:,} mẫu.",
        cls="tight"))

    r.p(R.figure(
        "s1_fig04_architecture_study.png",
        f"Khảo sát hội tụ Loss và đánh đổi hiệu năng giữa 4 kiến trúc mạng nơ-ron sâu "
        f"trên {ds['n_raw']:,} mẫu."))

    best = max(arch, key=lambda a: a["scores"]["f1"])
    shallow = next(a for a in arch if a["Kiến trúc"] == "Shallow MLP")
    wide = next(a for a in arch if a["Kiến trúc"] == "Wide MLP")

    r.p(R.oim(
        f"Mạng <strong>{best['Kiến trúc']}</strong> ({best['Cấu hình tầng']}, "
        f"{best['Tham số']:,} tham số) giành chiến thắng với Loss huấn luyện thấp nhất "
        f"({best['Loss cuối']:.4f}) và F1-Score cao nhất ({best['F1-Score']}). Mạng "
        f"{wide['Kiến trúc']} dù có tới {wide['Tham số']:,} tham số — gấp "
        f"{wide['Tham số'] / best['Tham số']:.1f} lần — nhưng F1 chỉ đạt {wide['F1-Score']}.",
        "Việc gia tăng chiều sâu cho phép mạng hình thành cơ chế <strong>biểu diễn phân cấp "
        "(Hierarchical Abstraction)</strong>: tầng H₁ mã hoá các tương tác tuyến tính cục bộ "
        "giữa các chỉ số; tầng H₂ tích hợp chúng thành tổ hợp nguy cơ (Glucose × HbA1c × Tuổi); "
        "tầng H₃ phân tách ranh giới phi tuyến cuối cùng.",
        "Mạng rộng chỉ dàn trải các bộ dò đặc trưng <strong>song song ở cùng một mức trừu "
        "tượng</strong>, nên không tái sử dụng được đặc trưng sơ cấp để xây đặc trưng bậc cao "
        "hơn. Đây là bằng chứng thực nghiệm cho quy luật: <strong>tăng số nơ-ron chiều rộng "
        "không thay thế được cấu trúc phân cấp tuần tự của chiều sâu</strong> trong bài toán "
        "phân loại."))

    # ---------------- 3.4 Ngưỡng ----------------
    r.h(2, "3.4. Tinh chỉnh ngưỡng quyết định và giảm thiểu bỏ sót ca bệnh")
    r.p(
        "Trong bài toán y tế, mục tiêu tối thượng là <strong>giảm thiểu ca bỏ sót "
        "(False Negative)</strong>. Ngưỡng xác suất mặc định 0,50 chỉ tối ưu khi hai lớp cân "
        f"bằng 50:50 và chi phí của False Positive bằng chi phí của False Negative. Ở đây cả "
        f"hai giả định đều sai: lớp bệnh chỉ chiếm {pct(ds['positive_rate'], 1)}, và một ca bỏ "
        "sót nguy hiểm hơn nhiều một lần xét nghiệm lại.",
        "Ngưỡng tối ưu được dò trên <strong>tập Train</strong> rồi áp cố định sang tập Test. "
        "Dò ngưỡng trực tiếp trên tập Test cũng là một dạng rò rỉ dữ liệu, chỉ tinh vi hơn.")

    r.p(R.figure(
        "s1_fig05_threshold_tuning.png",
        f"Đường cong đánh đổi Precision, Recall, F1 theo dải ngưỡng quyết định "
        f"({ds['n_raw']:,} mẫu)."))

    rows = []
    for name, v in sc.items():
        d, o = v["default"], v["optimal"]
        rows.append([
            name, f"{v['threshold']:.2f}", f"{d['FN']:,}", f"{o['FN']:,}",
            f"<strong>{d['FN'] - o['FN']:,}</strong>",
            f"{(o['recall'] - d['recall']) * 100:+.2f}%",
        ])
    r.p(R.table(
        ["Mô hình", "Ngưỡng τ tối ưu", "FN tại τ = 0,50", "FN tại τ tối ưu",
         "Số ca cứu được", "Δ Recall"],
        rows,
        caption="Hiệu quả cứu vãn ca bệnh nhờ tinh chỉnh ngưỡng quyết định.",
        cls="tight"))

    saves = {n: v["default"]["FN"] - v["optimal"]["FN"] for n, v in sc.items()}
    top_save = max(saves, key=saves.get)
    r.p(R.oim(
        f"Việc dịch chuyển ngưỡng quyết định về vùng tối ưu thực nghiệm giúp cả bốn mô hình "
        f"giảm số ca bỏ sót. Mô hình {top_save} cứu được nhiều nhất — "
        f"<strong>{saves[top_save]:,} bệnh nhân</strong> từ chỗ bị bỏ sót trở thành được phát hiện.",
        "Ngưỡng tối ưu của mọi mô hình đều nằm <em>thấp hơn</em> 0,50. Đây là hệ quả trực tiếp "
        "của mất cân bằng lớp: mô hình học được rằng \"đoán âm tính\" là chiến lược an toàn về "
        "mặt xác suất, nên phân phối xác suất dự đoán bị dồn về phía 0.",
        "Trong y học cộng đồng, một ca báo động giả chỉ gây tốn kém một khoản chi phí nhỏ; "
        "ngược lại, một ca bỏ sót có thể dẫn tới biến chứng suy thận, mù loà hoặc hoại tử chi. "
        "Việc chấp nhận đánh đổi một lượng nhỏ Precision để đổi lấy sự gia tăng Recall là "
        "<strong>yêu cầu bắt buộc và nhân văn</strong> trong y tế số."))

    # ---------------- 3.5 PCA ----------------
    r.h(2, "3.5. Trực quan hoá quá trình học biểu diễn qua các tầng ẩn bằng PCA 2D")
    r.p(
        "Nhằm trả lời câu hỏi cốt lõi về bản chất học biểu diễn — <em>mạng nơ-ron thực sự học "
        "được gì ở các tầng ẩn?</em> — vector kích hoạt tại từng tầng của mô hình "
        f"<strong>{dl['name']}</strong> được trích xuất trên 2.500 mẫu kiểm thử:",
        f'<div class="formula">X ∈ ℝ<sup>14</sup> → H₁ ∈ ℝ<sup>64</sup> → H₂ ∈ ℝ<sup>32</sup> '
        f'→ H₃ ∈ ℝ<sup>16</sup> → ŷ ∈ ℝ<sup>1</sup></div>',
        "Sau đó thuật toán PCA 2D (tự cài bằng phân rã SVD, không dùng thư viện) được áp dụng "
        "<strong>độc lập</strong> trên từng không gian để trực quan hoá quá trình chuyển dịch "
        "hình học.")

    r.p(R.figure(
        "s1_fig06_representation_pca_spaces.png",
        "Trực quan hoá sự xuất hiện của ranh giới phân tách tuyến tính qua các tầng ẩn bằng PCA 2D."))

    r.p(
        "<p><strong>Minh chứng hình học về cơ chế Representation Learning:</strong></p>"
        "<ol>"
        "<li><strong>Không gian đầu vào X (14D — bảng A).</strong> Hai đám mây dữ liệu (bệnh "
        "nhân tiểu đường màu đỏ và người khoẻ mạnh màu xanh) nằm đan xen hỗn độn, chồng lấn "
        "với mật độ dày đặc. Không thể dùng một siêu phẳng tuyến tính nào để phân tách.</li>"
        "<li><strong>Tầng ẩn H₁ (64D — bảng B).</strong> Qua phép biến đổi tuyến tính và kích "
        "hoạt ReLU, các điểm dữ liệu bắt đầu dãn nở không gian; cụm màu đỏ dịch chuyển dần về "
        "một góc phần tư riêng.</li>"
        "<li><strong>Tầng ẩn H₂ (32D — bảng C).</strong> Các tổ hợp phi tuyến tiếp tục tinh "
        "lọc đặc trưng, làm giảm phương sai nội cụm của nhóm người khoẻ mạnh.</li>"
        "<li><strong>Tầng ẩn H₃ (16D — bảng D).</strong> Tính phân tách tuyến tính "
        "(Linear Separability) <strong>xuất hiện rõ rệt</strong> — hai nhóm bệnh nhân bị đẩy "
        "hẳn về hai phía đối lập với khoảng phân cách rộng mở. Lúc này tầng ngõ ra chỉ cần một "
        f"phép biến đổi tuyến tính kết hợp Sigmoid với ngưỡng τ = {dl['threshold']:.2f} là đủ "
        "để phân loại cực kỳ chính xác.</li>"
        "</ol>")

    # ---------------- 3.6 Benchmark ----------------
    r.h(2, "3.6. Tổng hợp đối chuẩn 4 mô hình (trước và sau tinh chỉnh ngưỡng)")

    rows = []
    champ = s1["champion"]
    for name, v in sc.items():
        d, o = v["default"], v["optimal"]
        nm = f"<strong>{name}</strong>" if name == champ else name
        rows.append([
            nm, v["kind"], f"{v['threshold']:.2f}", pct(o["accuracy"]),
            pct(d["recall"]), pct(o["recall"]), pct(d["f1"]), pct(o["f1"]),
            f"{v['roc_auc']:.4f}", f"−{d['FN'] - o['FN']} ca",
        ])
    r.p(R.table(
        ["Mô hình", "Loại mô hình", "Ngưỡng", "Acc", "Rec (0,50)", "Rec (Opt)",
         "F1 (0,50)", "F1 (Opt)", "ROC-AUC", "FN giảm"],
        rows,
        caption="Bảng đối chuẩn tổng hợp đầy đủ chỉ số định lượng ở cả hai mức ngưỡng "
                "(mặc định 0,50 và tối ưu).",
        cls="tight"))

    r.p(R.figure(
        "s1_fig07_model_comparison.png",
        f"Trực quan hoá đối chuẩn hiệu năng hệ thống sàng lọc tiểu đường {ds['n_raw']:,} mẫu. "
        "Hàng trên: so sánh đa chỉ số và mức giảm ca bỏ sót. Hai hàng dưới: hệ thống ma trận "
        "nhầm lẫn 2×4 minh chứng mức độ giảm thiểu ca bỏ sót."))

    r.h(3, "3.6.1. Nhận xét chuyên sâu và đánh giá kết quả thực nghiệm y tế")

    rf, dl_s, lr_s, dt_s = (sc["Random Forest"], sc["Deeper DL Scratch"],
                            sc["Logistic Regression"], sc["Decision Tree"])
    best_auc = max(sc.items(), key=lambda kv: kv[1]["roc_auc"])
    best_rec = max(sc.items(), key=lambda kv: kv[1]["optimal"]["recall"])

    r.p(
        "<p><strong>1. Vì sao các mô hình cây thống trị trên dữ liệu bảng dạng ngưỡng sinh học.</strong></p>"
        f"Mô hình {champ} đạt F1-Score {pct(sc[champ]['optimal']['f1'])} và "
        f"{best_auc[0]} đạt ROC-AUC cao nhất {best_auc[1]['roc_auc']:.4f}. Đặc thù của dữ liệu "
        "y tế là ranh giới bệnh/thường được xác lập bởi các <strong>ngưỡng giới hạn sinh hoá "
        "nghiêm ngặt</strong> — chẳng hạn HbA1c ≥ 6,5% hoặc Glucose ≥ 126 mg/dL đại diện cho "
        "ngưỡng chẩn đoán đái tháo đường của WHO. Cây quyết định chia không gian bằng các lát "
        "cắt trực giao (Axis-aligned splits) bắt trọn các bước nhảy bậc thang này mà không cần "
        "tối ưu hoá liên tục như đạo hàm gradient descent.")

    r.p(
        "<p><strong>2. Năng lực biểu diễn của mạng nơ-ron thuần NumPy.</strong></p>"
        f"Mô hình mạng nơ-ron sâu {dl['name']} ({dl['layers']}, {dl['n_params']:,} tham số) đạt "
        f"Accuracy {pct(dl_s['optimal']['accuracy'])}, Recall {pct(dl_s['optimal']['recall'])}, "
        f"F1-Score {pct(dl_s['optimal']['f1'])} và ROC-AUC {dl_s['roc_auc']:.4f}. Mô hình vượt "
        f"Logistic Regression <strong>{(dl_s['optimal']['f1'] - lr_s['optimal']['f1']) * 100:+.2f} "
        f"điểm F1</strong> và cạnh tranh sòng phẳng với các mô hình cây phức tạp. Kết quả này "
        "chứng minh rằng dù không có sự hỗ trợ của các framework tự động vi phân, các phương "
        f"trình lan truyền ngược thuần tuý từ đầu vẫn hội tụ ổn định và học được các biểu diễn "
        f"trừu tượng hiệu quả từ {ds['n_raw']:,} mẫu dữ liệu thực tế.")

    r.p(
        "<p><strong>3. Ý nghĩa lâm sàng sống còn của việc tinh chỉnh ngưỡng quyết định.</strong></p>"
        "<ul>"
        + "".join(
            f"<li>{n}: bỏ sót {v['default']['FN']:,} ca ở ngưỡng mặc định, giảm còn "
            f"<strong>{v['optimal']['FN']:,} ca</strong> ở ngưỡng tối ưu "
            f"τ = {v['threshold']:.2f} — cứu vãn {v['default']['FN'] - v['optimal']['FN']:,} "
            f"bệnh nhân thực tế.</li>"
            for n, v in sc.items())
        + "</ul>"
        f"Xét trên toàn hệ thống, mô hình có độ nhạy phát hiện bệnh cao nhất là "
        f"<strong>{best_rec[0]}</strong> với Recall {pct(best_rec[1]['optimal']['recall'])}. "
        "Trong bối cảnh sàng lọc cộng đồng, đây mới là chỉ số quyết định giá trị thực tiễn của "
        "hệ thống, chứ không phải Accuracy.")


# ===========================================================================
# CHƯƠNG 4 — HỆ THỐNG 2: ĐỊNH GIÁ BẤT ĐỘNG SẢN 150,000 MẪU
# ===========================================================================
def _chapter4(r: R.Report, s2: dict) -> None:
    ds = s2["dataset"]
    sp = s2["split"]
    sc = s2["scores"]
    dl = s2["dl_model"]
    pre = s2["preprocess"]

    r.h(1, "Chương 4. Hệ thống 2 — Định giá bất động sản thực nghiệm "
           "(150,000 mẫu dữ liệu)")

    r.p(
        "Chương 3 giải bài toán <strong>phân loại nhị phân</strong>. Chương này chuyển sang "
        "một họ bài toán khác hẳn: <strong>hồi quy giá trị liên tục</strong> trên "
        f"{ds['n_raw']:,} giao dịch bất động sản. Sự chuyển đổi này kéo theo bốn thay đổi kỹ "
        "thuật căn bản trong chính kiến trúc mạng nơ-ron.")

    r.p(R.table(
        ["Thành phần", "Phân loại (Chương 3)", "Hồi quy (Chương 4)"],
        [
            ["Hàm kích hoạt tầng ra", "Sigmoid σ(z)",
             "<strong>Tuyến tính</strong> φ(z) = z"],
            ["Hàm mất mát", "Binary Cross-Entropy",
             "<strong>Mean Squared Error</strong>"],
            ["Chỉ số đánh giá", "Accuracy, Recall, F1, ROC-AUC",
             "<strong>R², MAE, RMSE, MAPE</strong>"],
            ["Xử lý biến mục tiêu", "Nhãn nhị phân {0, 1}",
             "<strong>Biến đổi Logarit</strong>"],
        ],
        caption="Bốn thay đổi kỹ thuật khi chuyển từ bài toán phân loại sang hồi quy."))

    r.p(
        "Câu hỏi nghiên cứu trung tâm của chương:",
        '<div class="flow">Quy luật "chiều sâu thắng chiều rộng" tìm được ở Chương 3 có còn '
        'đúng khi bài toán chuyển từ phân loại sang hồi quy hay không?</div>')

    # ---------------- 4.1 EDA ----------------
    r.h(2, "4.1. Bối cảnh thị trường bất động sản và phân tích EDA")
    r.p(
        f"Tập dữ liệu <code class=\"inl\">{ds['name']}</code> bao gồm {ds['n_raw']:,} tin đăng "
        "bất động sản toàn nước Mỹ với các đặc trưng không gian, diện tích và kết cấu công "
        "trình. Đặc tính cốt lõi của giá nhà là tính không âm và phân phối "
        f"<strong>lệch phải cực kỳ nặng</strong> (Right-Skewed, hệ số lệch "
        f"{ds['skew_raw']:.3f}): đa số nhà có giá từ 100.000 USD đến 600.000 USD, trong khi "
        "một số ít biệt thự siêu sang vượt ngưỡng 5.000.000 USD kéo dài cái đuôi bên phải.")

    r.p(R.note(
        "Vì sao độ lệch phải là vấn đề nghiêm trọng với hồi quy?",
        "Hàm mất mát MSE bình phương sai số, nên một căn biệt thự dự đoán sai 2 triệu USD đóng "
        "góp vào loss gấp <strong>một triệu lần</strong> một căn nhà phổ thông sai 2 nghìn USD. "
        "Mô hình sẽ dồn toàn bộ năng lực vào nhóm cực đoan và bỏ mặc 95% thị trường còn lại.",
        kind="warn"))

    r.p(
        "Áp dụng lý thuyết kinh tế lượng Hedonic, biến mục tiêu được chuyển đổi qua hàm logarit:",
        '<div class="formula">y<sub>log</sub> = log(1 + price)</div>',
        f"Phép biến đổi đưa hệ số lệch từ <strong>{ds['skew_raw']:.3f}</strong> xuống "
        f"<strong>{ds['skew_log']:.3f}</strong> — gần như đối xứng chuẩn hoàn hảo. Nó cũng "
        "chuyển bài toán từ <em>sai số tuyệt đối</em> sang <em>sai số tương đối</em>, đúng với "
        "cách thị trường bất động sản vận hành: người ta nói \"căn nhà này đắt hơn 20%\", "
        "không nói \"đắt hơn 80.000 USD\".")

    r.p(R.figure(
        "s2_fig01_price_distribution.png",
        "Phân phối giá nhà thô (lệch phải nặng) và biến đổi Logarit đối xứng Gauss."))

    r.p(R.figure(
        "s2_fig02_features_vs_price.png",
        "Khám phá đặc trưng không gian và cấu trúc nhà ở: quan hệ diện tích – giá, phân bố giá "
        "theo số phòng ngủ, và ma trận tương quan Pearson."))

    r.p(R.oim(
        "Đồ thị phân tán cho thấy quan hệ giữa diện tích sàn và giá bán mang hình dạng hình "
        "nón mở rộng chứ không phải một đường thẳng. Hệ số tương quan của diện tích với "
        "ln(price) cao hơn với price thô.",
        "Quan hệ diện tích – giá mang bản chất <strong>nhân tính</strong> (giá tăng theo tỷ lệ "
        "phần trăm) chứ không phải cộng tính. Thêm 500 sqft cho một căn hộ nhỏ và cho một biệt "
        "thự tạo ra mức tăng giá tuyệt đối hoàn toàn khác nhau.",
        "Đây là bằng chứng định lượng ủng hộ quyết định biến đổi logarit ở trên: sau khi lấy "
        "log, quan hệ nhân tính trở thành quan hệ cộng tính, và mô hình tuyến tính mới có cơ "
        "hội khớp được dữ liệu."))

    # ---------------- 4.2 Làm sạch + FE ----------------
    r.h(2, "4.2. Kiểm toán làm sạch dữ liệu và kỹ nghệ đặc trưng vị trí địa lý")
    r.p(
        "Nhằm xây dựng tập dữ liệu huấn luyện tin cậy cho bài toán định giá quy mô lớn, một "
        "quy trình kiểm toán làm sạch được áp dụng tuần tự qua ba bước sàng lọc.")

    r.p(R.table(
        ["Bước xử lý", "Tiêu chí kiểm toán thực nghiệm", "Số loại bỏ", "Số còn lại", "Tỷ lệ giữ"],
        [[a["Bước Xử lý"], a["Tiêu chí Kiểm toán Thực nghiệm"],
          f"{a['Số lượng Loại bỏ']:,}", f"{a['Số lượng Còn lại']:,}",
          f"{a['Tỷ lệ Giữ lại (%)']:.2f}%"] for a in s2["cleaning_audit"]],
        caption=f"Bảng kiểm toán quy trình làm sạch dữ liệu bất động sản "
                f"({ds['n_raw']:,} mẫu ban đầu)."))

    r.h(3, "4.2.1. Kỹ nghệ đặc trưng hình học và tỷ lệ không gian")
    r.p(
        "Dựa trên nguyên lý định giá bất động sản Hedonic, các đặc trưng tỷ lệ hình học và "
        "công năng được trích xuất nhằm tăng cường tín hiệu phi tuyến:")

    r.p(
        "<ul>"
        "<li><strong>Xử lý khuyết thiếu diện tích lô đất</strong> "
        "(<code class=\"inl\">acre_lot</code>): trong tin đăng thực tế, nhiều căn hộ chung cư "
        "hoặc nhà phố liền kề không ghi nhận diện tích lô đất. Các giá trị khuyết thiếu này "
        f"được điền bằng trung vị tính duy nhất trên tập huấn luyện "
        f"({pre['acre_lot_median']:.4f} acre).</li>"
        "<li><strong>Biến đổi logarit co giãn diện tích:</strong> "
        "log_house_size = ln(house_size) và log_acre_lot = ln(1 + acre_lot).</li>"
        "<li><strong>Cấu trúc công năng xây dựng:</strong> total_rooms = bed + bath, "
        "sqft_per_room = house_size / total_rooms, bath_bed_ratio = bath / bed, và "
        "bed_bath_prod = bed × bath.</li>"
        "</ul>")

    r.p(
        "<p><strong>Chỉ số quy mô tương đối cục bộ</strong> "
        "(<code class=\"inl\">relative_sqft</code>):</p>"
        '<div class="formula">relative_sqft = house_size / mean(house_size | zip_code)</div>',
        "Đo lường mức độ bề thế <strong>tương đối</strong> của ngôi nhà so với mặt bằng quy mô "
        "dân cư lân cận cùng mã bưu chính. Một căn 2.000 sqft là bình thường ở ngoại ô nhưng "
        "là biệt thự ở trung tâm thành phố.")

    r.h(3, "4.2.2. Mã hoá đích Bayes làm mượt (Smooth Bayesian Target Encoding) chống rò rỉ")
    r.p(
        "Vị trí địa lý là nhân tố định đoạt giá trị bất động sản. Tuy nhiên, việc One-Hot "
        "Encoding hàng nghìn mã bưu chính sẽ làm bùng nổ chiều dữ liệu và gây thưa thớt nghiêm "
        "trọng. Để khắc phục, kỹ thuật Smooth Target Encoding có suy biến Bayes "
        "(Bayesian Shrinkage Smoothing) được triển khai cho bốn cấp độ địa lý:",
        f'<div class="formula">S<sub>i</sub> = (n<sub>i</sub> · ȳ<sub>i</sub> + m · '
        f'ȳ<sub>global</sub>) / (n<sub>i</sub> + m)</div>',
        "Trong đó n<sub>i</sub> và ȳ<sub>i</sub> là số lượng tin đăng và giá log trung bình của "
        "đơn vị địa lý i; ȳ<sub>global</sub> là giá trị trung bình toàn cục trên tập huấn "
        "luyện; tham số làm mượt m đóng vai trò tiên nghiệm giả định "
        "(Prior Pseudo-counts) nhằm co kéo (shrink) các khu vực có quá ít giao dịch về mức giá "
        "trung bình toàn quốc, ngăn ngừa hiện tượng học vẹt cục bộ.")

    lv = pre["target_encoding"]["levels"]
    r.p(R.table(
        ["Cấp độ địa lý", "Tham số làm mượt m", "Số đơn vị học được từ tập Train"],
        [
            ["Bang (<code class=\"inl\">state</code>)", lv["state"]["m"],
             f"{lv['state']['n_units']:,}"],
            ["Vùng bưu chính 3 số (<code class=\"inl\">zip3</code>)", lv["zip3"]["m"],
             f"{lv['zip3']['n_units']:,}"],
            ["Mã bưu chính 5 số (<code class=\"inl\">zip_code</code>)", lv["zip_code"]["m"],
             f"{lv['zip_code']['n_units']:,}"],
            ["Thành phố (<code class=\"inl\">city</code>)", lv["city"]["m"],
             f"{lv['city']['n_units']:,}"],
        ],
        caption="Bốn cấp độ mã hoá đích Bayes với tham số làm mượt tương ứng.",
        cls="tight"))

    r.p(R.note(
        "Trực giác của công thức làm mượt.",
        "Một mã bưu chính chỉ có 2 tin đăng (n<sub>i</sub> = 2) thì giá trung bình của nó gần "
        "như vô nghĩa về mặt thống kê — công thức sẽ kéo mạnh nó về giá trung bình toàn quốc. "
        "Ngược lại, một thành phố có 5.000 tin đăng (n<sub>i</sub> ≫ m) thì giá trung bình rất "
        "đáng tin — công thức giữ gần như nguyên giá trị đó."))

    r.p(
        f"<strong>Quy chuẩn chống rò rỉ dữ liệu.</strong> Toàn bộ bảng tra cứu ánh xạ "
        f"S<sub>i</sub> được tính <strong>duy nhất trên tập Train</strong> ({sp['n_train']:,} "
        f"mẫu) và ánh xạ cố định sang tập Test ({sp['n_test']:,} mẫu). Các khu vực mới xuất "
        "hiện ở tập Test được gán giá trị trung bình toàn cục ȳ<sub>global</sub>. Không gian "
        f"đặc trưng sau cùng gồm đúng <strong>{s2['n_features']} biến đầu vào</strong>, toàn "
        "bộ được chuẩn hoá Z-Score dựa trên thống kê của tập Train.")

    # ---------------- 4.3 Kiến trúc ----------------
    r.h(2, "4.3. Khảo sát kiến trúc mạng hồi quy sâu: đột phá của chiều rộng")
    r.p(
        "Để huấn luyện mạng hồi quy sâu thuần NumPy "
        "(<code class=\"inl\">MLPRegressorScratch</code>), tầng ngõ ra sử dụng hàm kích hoạt "
        "tuyến tính φ(z) = z và hàm mất mát sai số toàn phương trung bình (MSE). Do "
        "y<sub>log</sub> tiếp tục được chuẩn hoá Z-Score về phân phối có kỳ vọng bằng 0 và "
        "phương sai bằng 1, hàm mất mát huấn luyện là một đại lượng "
        "<strong>không thứ nguyên</strong> (Dimensionless), thuận tiện để so sánh giữa các "
        "kiến trúc.")

    r.p(R.note(
        "Một quan sát toán học thú vị.",
        "Công thức gradient tầng ra của cặp <em>MSE + Linear</em> hoàn toàn giống với cặp "
        "<em>BCE + Sigmoid</em> ở Chương 3: dZ<sup>[L]</sup> = (ŷ − y) / m. Đây không phải "
        "trùng hợp ngẫu nhiên — cả hai cặp đều thuộc họ hàm mất mát chính tắc (canonical link) "
        "của phân phối mũ tương ứng. Nhờ vậy toàn bộ phần lan truyền ngược qua các tầng ẩn giữ "
        "nguyên không đổi, chỉ cần thay hàm kích hoạt tầng cuối."))

    arch = s2["architecture_study"]
    r.p(R.table(
        ["Kiến trúc", "Cấu hình tầng", "Tham số", "Thời gian", "MSE Loss",
         "R² Score", "MAE (USD)", "RMSE (USD)"],
        [[(f"<strong>{a['Kiến trúc']}</strong>"
           if a["Kiến trúc"] == dl["name"] else a["Kiến trúc"]),
          a["Cấu hình tầng"], f"{a['Tham số']:,}", a["Thời gian"],
          f"{a['MSE Loss']:.4f}", f"{a['R² Score']:.4f}",
          a["MAE (USD)"], a["RMSE (USD)"]] for a in arch],
        caption="Bảng đối chứng thực nghiệm 4 cấu hình kiến trúc mạng hồi quy sâu thuần NumPy.",
        cls="tight"))

    r.p(R.figure(
        "s2_fig03_architecture_study.png",
        "Đường cong suy giảm hàm mất mát MSE chuẩn hoá và so sánh chỉ số hồi quy trên Dollar thực."))

    best_a = max(arch, key=lambda a: a["scores"]["r2_log"])
    shal = next(a for a in arch if "Shallow" in a["Kiến trúc"])
    deep = next(a for a in arch if "Deeper" in a["Kiến trúc"])

    r.p(R.oim(
        f"Khác với Chương 3, ở đây kiến trúc <strong>{best_a['Kiến trúc']}</strong> "
        f"({best_a['Cấu hình tầng']}) mới là mô hình tốt nhất: MSE Loss thấp nhất "
        f"{best_a['MSE Loss']:.4f}, R² cao nhất {best_a['R² Score']:.4f} và MAE thấp nhất "
        f"{best_a['MAE (USD)']}. Mạng {deep['Kiến trúc']} — ưu tiên chiều sâu — chỉ đạt R² "
        f"{deep['R² Score']:.4f}.",
        "Bài toán phân loại cần <strong>bẻ cong một siêu phẳng phân tách</strong> để bao lấy "
        "các cụm dữ liệu, và việc bẻ cong phức tạp đòi hỏi tổ hợp phân cấp nhiều bậc — thế mạnh "
        "của chiều sâu. Bài toán hồi quy thì khác hẳn: nó cần <strong>xấp xỉ một mặt phẳng giá "
        "trị liên tục, trơn và mấp mô</strong> trải trên nhiều chiều đặc trưng.",
        "Mỗi nơ-ron ReLU tạo ra một \"nếp gấp\" tuyến tính từng đoạn "
        "(Piecewise Linear Approximator). Tăng chiều rộng tầng ẩn (128 → 64) cung cấp một "
        "<strong>hệ cơ sở hàm ReLU phong phú hơn</strong>, cho phép ghép nối nhiều mặt phẳng "
        "con cục bộ mượt mà hơn — giống như dùng nhiều mảnh vải nhỏ để phủ một bề mặt cong "
        "thay vì ít mảnh lớn. <strong>Quy luật kiến trúc của Chương 3 bị đảo ngược hoàn toàn.</strong>"))

    # ---------------- 4.4 Residual ----------------
    r.h(2, "4.4. Phân tích chẩn đoán phần dư và sai số thị trường")
    r.p(
        "Phần dư là hiệu giữa giá thực tế và giá dự đoán: e<sub>i</sub> = y<sub>i</sub> − "
        "ŷ<sub>i</sub>. Đồ thị phần dư là công cụ chẩn đoán mạnh nhất của kinh tế lượng — nó "
        "tiết lộ những gì mà một con số R² đơn lẻ che giấu.")

    r.p(R.figure(
        "s2_fig04_residual_analysis.png",
        "Chẩn đoán phần dư, kiểm tra phân phối chuẩn và phân tích sai số theo ba phân khúc thị trường."))

    r.p(
        "<p><strong>Các phát hiện kinh tế lượng từ đồ thị phần dư:</strong></p>"
        "<ol>"
        f"<li><strong>Phần dư đối xứng quanh 0.</strong> Trung bình phần dư xấp xỉ "
        f"{usd(s2['residual_mean_usd'])} trên mức giá trung vị {usd(ds['price_median'])}, "
        f"tương đương tỷ lệ thiên lệch chỉ "
        f"{abs(s2['residual_mean_usd']) / ds['price_median'] * 100:.2f}%. Điều này chứng minh "
        "mô hình <strong>không bị thiên lệch hệ thống</strong> (No Systematic Bias).</li>"
        "<li><strong>Hiện tượng phương sai thay đổi (Heteroscedasticity).</strong> Phễu phần "
        "dư mở rộng dần khi giá nhà dự đoán tăng lên. Điều này hoàn toàn phù hợp với thực tế "
        "kinh tế và được định lượng ở bảng dưới đây.</li>"
        "</ol>")

    r.p(R.table(
        ["Phân khúc thị trường", "Số giao dịch", "Tỷ trọng", "MAE", "MAPE"],
        [[m["Phân khúc thị trường"], m["Số giao dịch"], m["Tỷ trọng"], m["MAE"], m["MAPE"]]
         for m in s2["market_segments"]],
        caption="Phân tích sai số định giá theo ba phân khúc thị trường bất động sản.",
        cls="tight"))

    seg = s2["market_segments"]
    r.p(
        f"Ở phân khúc <strong>{seg[1]['Phân khúc thị trường']}</strong> — chiếm "
        f"{seg[1]['Tỷ trọng']} thị trường — mô hình định giá chuẩn xác nhất với MAPE chỉ "
        f"{seg[1]['MAPE']}. Ở phân khúc {seg[2]['Phân khúc thị trường']}, sai số tuyệt đối vọt "
        f"lên {seg[2]['MAE']}: giá cả phụ thuộc nhiều vào yếu tố độc bản (nội thất xa xỉ, tầm "
        "nhìn cảnh quan, uy tín kiến trúc sư) mà dữ liệu thuộc tính thông thường không nắm bắt "
        f"hết được. Đáng chú ý, phân khúc {seg[0]['Phân khúc thị trường']} có MAE thấp nhất "
        f"({seg[0]['MAE']}) nhưng MAPE lại cao nhất ({seg[0]['MAPE']}) — vì mẫu số nhỏ khiến "
        "cùng một sai số tuyệt đối trở thành tỷ lệ phần trăm lớn.")

    # ---------------- 4.5 Benchmark ----------------
    r.h(2, "4.5. Tổng hợp đối chuẩn 4 mô hình định giá bất động sản")

    champ = s2["champion"]
    r.p(R.table(
        ["Mô hình", "Loại mô hình", "Tham số", "Thời gian", "R² (Log)", "R² (Raw)",
         "MAE (USD)", "MAPE"],
        [[(f"<strong>{n}</strong>" if n == champ else n), v["kind"],
          f"{v['n_params']:,}", f"{v['time_sec']:.3f}s",
          f"{v['r2_log']:.4f}", f"{v['r2_raw']:.4f}",
          f"{v['mae_usd']:,.2f}", f"{v['mape']:.2f}%"] for n, v in sc.items()],
        caption="Bảng đối chuẩn toàn diện 4 mô hình định giá nhà trên không gian giá trị "
                "Dollar thực.",
        cls="tight"))

    r.p(R.figure(
        "s2_fig05_model_comparison.png",
        f"Đối chuẩn trực quan hiệu năng hệ thống định giá bất động sản ({ds['n_raw']:,} giao dịch)."))

    r.h(3, "4.5.1. Nhận xét chuyên sâu và đánh giá kết quả thực nghiệm định giá nhà")

    lin = sc["Linear Regression"]
    dls = sc["Wide DL Scratch"]
    rf = sc["Random Forest"]

    r.p(
        "<p><strong>1. Sự phân kỳ giữa không gian Logarit và không gian Dollar thực.</strong></p>"
        f"Linear Regression đạt R²(log) = {lin['r2_log']:.4f} — nghe khá tốt — nhưng R²(raw) "
        f"tụt xuống chỉ còn <strong>{lin['r2_raw']:.4f}</strong>, và sai số MAE vọt lên "
        f"{usd(lin['mae_usd'], 2)}. Nguyên nhân nằm ở bản chất hàm mũ. Nếu "
        "log(y) = w<sup>T</sup>x + b + ε thì khi nghịch đảo:"
        '<div class="formula">y = exp(w<sup>T</sup>x + b) · exp(ε) ≈ '
        'exp(w<sup>T</sup>x + b) · (1 + ε + ε²/2)</div>'
        "Một sai số dư nhỏ ε = 0,30 trên thang log của căn nhà trị giá 3.000.000 USD sẽ bị "
        "khuếch đại thành mức chênh lệch vượt quá 1.050.000 USD trong giá trị thực. "
        "<strong>Bài học: phải luôn báo cáo song song cả hai chỉ số R²</strong>.")

    r.p(
        "<p><strong>2. Ưu thế của mạng rộng trong xấp xỉ mặt phẳng liên tục.</strong></p>"
        f"Mô hình {dl['name']} ({dl['layers']}, {dl['n_params']:,} tham số) đạt "
        f"R²(log) = {dls['r2_log']:.4f}, MAE = {usd(dls['mae_usd'], 2)} và "
        f"MAPE = {dls['mape']:.2f}%. Mô hình vượt Linear Regression tới "
        f"<strong>{usd(lin['mae_usd'] - dls['mae_usd'])} sai số trên mỗi căn nhà</strong>. "
        "128 nơ-ron tại tầng ẩn thứ nhất cung cấp một hệ cơ sở hàm ReLU phong phú, giúp mô hình "
        "bẻ gập và khớp nối các bề mặt giá trị đa chiều mấp mô một cách trơn tru.")

    r.p(
        "<p><strong>3. Random Forest và bản chất phân mảnh của thị trường bất động sản.</strong></p>"
        f"Random Forest giành vị trí dẫn đầu với R²(raw) = {rf['r2_raw']:.4f}, "
        f"MAE = {usd(rf['mae_usd'], 2)} và MAPE = {rf['mape']:.2f}%. Thị trường bất động sản "
        "không phải một mặt phẳng liên tục toàn cục mà là tập hợp của hàng nghìn thị trường con "
        "địa phương với quy luật giá riêng biệt. Các bề mặt xấp xỉ từng đoạn cục bộ "
        "(Piecewise Approximation) của rừng cây kìm hãm sự bùng nổ sai số ở đuôi giá cao — đúng "
        f"nơi mà mô hình tuyến tính sụp đổ (R²(raw) chỉ {lin['r2_raw']:.4f}).")


# ===========================================================================
# CHƯƠNG 5 — HỆ THỐNG 3: PHÂN LOẠI NHẬN XÉT NLP
# ===========================================================================
def _chapter5(r: R.Report, s3: dict) -> None:
    ds = s3["dataset"]
    sp = s3["split"]
    sc = s3["scores"]
    dl = s3["dl_model"]
    vec = s3["vectorizer"]

    r.h(1, "Chương 5. Hệ thống 3 — Phân loại nhận xét thương mại điện tử "
           f"({ds['n_clean']:,} đánh giá NLP)")

    r.p(
        "Hai chương trước làm việc trên <strong>dữ liệu bảng có cấu trúc</strong>: mỗi cột là "
        "một đại lượng đo được, mỗi hàng là một quan sát. Chương này bước sang miền dữ liệu "
        "<strong>phi cấu trúc</strong>: văn bản tự do do khách hàng viết ra.",
        "Sự chuyển đổi này đặt ra một thách thức hoàn toàn mới. Một mạng nơ-ron chỉ nhận vào "
        "các vector số thực. Vậy làm thế nào biến câu \"chiếc váy này rất đẹp nhưng size hơi "
        "nhỏ\" thành một vector? Đó chính là bài toán <strong>biểu diễn văn bản</strong> "
        "(Text Representation) mà chương này giải quyết bằng TF-IDF.")

    r.p(R.table(
        ["Đặc điểm", "Dữ liệu bảng (Chương 3, 4)", "Dữ liệu văn bản (Chương 5)"],
        [
            ["Nguồn đặc trưng", "Đo lường trực tiếp (tuổi, BMI, diện tích)",
             "<strong>Trích xuất từ chuỗi ký tự</strong>"],
            ["Số chiều", "13 – 14 chiều", f"<strong>{vec['max_features']:,} chiều</strong>"],
            ["Mật độ ma trận", "Đặc (dense) gần 100%",
             f"<strong>Thưa {vec['sparsity_pct']:.1f}% giá trị bằng 0</strong>"],
            ["Ý nghĩa mỗi chiều", "Một đại lượng vật lý/sinh học",
             "Trọng số của một từ hoặc cụm hai từ"],
        ],
        caption="Đối chiếu bản chất dữ liệu bảng và dữ liệu văn bản phi cấu trúc."))

    # ---------------- 5.1 EDA ----------------
    r.h(2, "5.1. Bối cảnh văn bản e-commerce và không gian TF-IDF")
    r.p(
        f"Tập dữ liệu <code class=\"inl\">{ds['name']}</code> gồm {ds['n_raw']:,} đánh giá sản "
        "phẩm thời trang nữ. Mỗi bản ghi chứa tiêu đề, nội dung nhận xét, điểm đánh giá sao và "
        "nhãn nhị phân <code class=\"inl\">Recommended IND</code> — biến mục tiêu cần dự đoán: "
        "khách hàng có giới thiệu sản phẩm này cho người khác hay không.")

    r.p(R.figure(
        "s3_fig01_target_and_length.png",
        "Phân tích phân phối mục tiêu và độ dài văn bản nhận xét thương mại điện tử."))

    r.p(R.oim(
        f"Tỷ lệ nhãn tích cực chiếm <strong>{pct(ds['positive_rate'], 1)}</strong> — mất cân "
        f"bằng theo hướng ngược lại so với Chương 3. Độ dài nhận xét có trung vị "
        f"{ds['median_words_pos']:.0f} từ ở nhóm tích cực và {ds['median_words_neg']:.0f} từ ở "
        "nhóm tiêu cực.",
        "Khách hàng không hài lòng viết dài hơn khách hàng hài lòng. Người hài lòng thường chỉ "
        "cần vài từ (\"love it\", \"perfect fit\"), trong khi người thất vọng có xu hướng giải "
        "thích cặn kẽ lý do trả hàng.",
        "Chênh lệch độ dài này là một tín hiệu phân biệt tiềm năng, nhưng nó <em>không</em> "
        "được đưa vào mô hình như một đặc trưng riêng — mục tiêu của chương là kiểm tra xem "
        "mô hình học được gì <strong>chỉ từ nội dung ngữ nghĩa</strong> của văn bản."))

    r.p(R.figure(
        "s3_fig02_top_keywords.png",
        "Top 15 từ khoá xuất hiện nhiều nhất trong đánh giá của khách hàng.",
        width="82%"))

    kws = ", ".join(f"<code class=\"inl\">{k['word']}</code>" for k in s3["top_keywords"][:8])
    r.p(
        f"Tám từ khoá xuất hiện dày đặc nhất là {kws}. Đây chủ yếu là danh từ chỉ sản phẩm và "
        "tính từ đánh giá — đúng như kỳ vọng với một tập dữ liệu nhận xét thời trang.")

    r.h(3, "5.1.1. Quy trình làm sạch dữ liệu văn bản (NLP Text Cleaning)")
    r.p(R.table(
        ["Bước xử lý", "Tiêu chí tiền xử lý văn bản", "Số loại bỏ", "Số còn lại", "Tỷ lệ giữ"],
        [[a["Bước Xử lý"], a["Tiêu chí Tiền xử lý Văn bản"],
          f"{a['Số lượng Loại bỏ']:,}", f"{a['Số lượng Còn lại']:,}",
          f"{a['Tỷ lệ Giữ lại (%)']:.2f}%"] for a in s3["cleaning_audit"]],
        caption=f"Bảng kiểm toán quy trình làm sạch dữ liệu văn bản NLP thương mại điện tử "
                f"({ds['n_raw']:,} nhận xét ban đầu)."))

    r.h(3, "5.1.2. Phân chia phân tầng và trích xuất TF-IDF chống rò rỉ")
    r.p(
        "TF-IDF (Term Frequency – Inverse Document Frequency) gán cho mỗi từ một trọng số phản "
        "ánh mức độ <em>đặc trưng</em> của nó với một văn bản cụ thể:",
        '<div class="formula">tfidf(t, d) = tf(t, d) × log(N / df(t))</div>',
        "Một từ xuất hiện nhiều trong một nhận xét (tf cao) nhưng hiếm gặp trong toàn bộ tập dữ "
        "liệu (df thấp) sẽ nhận trọng số lớn — đó là từ mang tính phân biệt. Ngược lại, từ xuất "
        "hiện ở mọi văn bản (như \"the\", \"and\") nhận trọng số gần 0.")

    r.p(R.table(
        ["Tham số cấu hình", "Giá trị", "Lý do lựa chọn"],
        [
            ["<code class=\"inl\">max_features</code>", f"{vec['max_features']:,}",
             "Giữ 1.000 từ/cụm từ có trọng số cao nhất, cân bằng biểu diễn và chi phí tính toán"],
            ["<code class=\"inl\">ngram_range</code>", f"({vec['ngram_range'][0]}, {vec['ngram_range'][1]})",
             f"Bắt cả từ đơn lẫn cụm hai từ — {vec['n_bigrams']} bigram được giữ lại, "
             "nắm được cụm phủ định như \"not worth\""],
            ["<code class=\"inl\">stop_words</code>", vec["stop_words"],
             "Loại bỏ hư từ tiếng Anh không mang thông tin phân loại"],
            ["<code class=\"inl\">sublinear_tf</code>", str(vec["sublinear_tf"]),
             "Dùng 1 + log(tf) thay tf thô, giảm ảnh hưởng của từ lặp quá nhiều lần"],
        ],
        caption="Cấu hình bộ vector hoá TF-IDF."))

    r.p(
        f"Bộ vector hoá được <code class=\"inl\">fit</code> <strong>duy nhất trên tập Train</strong> "
        f"({sp['n_train']:,} mẫu) rồi <code class=\"inl\">transform</code> sang tập Test "
        f"({sp['n_test']:,} mẫu). Nếu học từ điển trên toàn bộ dữ liệu, các từ chỉ xuất hiện ở "
        "tập Test sẽ tham gia vào từ điển — một dạng rò rỉ dữ liệu. Ma trận kết quả có độ thưa "
        f"<strong>{vec['sparsity_pct']:.2f}%</strong>: trung bình mỗi nhận xét chỉ kích hoạt "
        f"khoảng {vec['max_features'] * (1 - vec['sparsity_pct'] / 100):.0f} trong số "
        f"{vec['max_features']:,} chiều.")

    # ---------------- 5.2 Kiến trúc ----------------
    r.h(2, "5.2. Khảo sát kiến trúc NLP: thích ứng trên dữ liệu thưa chiều cao")
    r.p(
        f"Bốn kiến trúc mạng được huấn luyện trên không gian TF-IDF {vec['max_features']:,} "
        f"chiều với tốc độ học η = {dl['lr']}, kích thước lô {dl['batch_size']} và "
        f"{dl['epochs']} epochs.")

    arch = s3["architecture_study"]
    r.p(R.table(
        ["Kiến trúc", "Cấu hình tầng", "Tham số", "Thời gian", "Loss cuối",
         "Accuracy", "Recall lớp tiêu cực", "Macro F1"],
        [[(f"<strong>{a['Kiến trúc']}</strong>"
           if a["Kiến trúc"] == dl["name"] else a["Kiến trúc"]),
          a["Cấu hình tầng"], f"{a['Tham số']:,}", a["Thời gian"],
          f"{a['Loss cuối']:.4f}", a["Accuracy"], a["Rec0 (Tiêu cực)"], a["Macro F1"]]
         for a in arch],
        caption="Bảng đối chứng thực nghiệm khảo sát kiến trúc mạng văn bản thuần NumPy.",
        cls="tight"))

    r.p(R.figure(
        "s3_fig03_architecture_study.png",
        "Khảo sát hội tụ Loss và so sánh Macro F1 giữa 4 kiến trúc mạng nơ-ron phân loại văn bản."))

    wide_t = next(a for a in arch if "Wide" in a["Kiến trúc"])
    best_t = max(arch, key=lambda a: a["scores"]["macro_f1"])
    r.p(R.oim(
        f"Kiến trúc <strong>{best_t['Kiến trúc']}</strong> ({best_t['Cấu hình tầng']}, "
        f"{best_t['Tham số']:,} tham số) đạt Macro F1 cao nhất {best_t['Macro F1']}. Mạng "
        f"{wide_t['Kiến trúc']} có tới {wide_t['Tham số']:,} tham số — gấp "
        f"{wide_t['Tham số'] / best_t['Tham số']:.1f} lần — nhưng Macro F1 chỉ "
        f"{wide_t['Macro F1']}.",
        "Trên không gian thưa chiều cao, phần lớn tham số của tầng đầu tiên không bao giờ được "
        f"cập nhật trong một lô dữ liệu bất kỳ: với độ thưa {vec['sparsity_pct']:.1f}%, mỗi mẫu "
        "chỉ kích hoạt vài chục chiều đầu vào. Tăng chiều rộng chỉ làm tăng số tham số \"chết\" "
        "và tạo cơ hội học vẹt các từ hiếm.",
        "Bài toán văn bản có quy luật kiến trúc riêng, khác cả Chương 3 lẫn Chương 4: "
        "<strong>không phải sâu hơn, cũng không phải rộng hơn, mà là vừa đủ</strong>. Điều "
        "quyết định chất lượng ở đây là chất lượng phép biểu diễn TF-IDF, không phải độ phức "
        "tạp của mạng."))

    # ---------------- 5.3 Ngưỡng Macro F1 ----------------
    r.h(2, "5.3. Tinh chỉnh ngưỡng cân bằng lớp (Macro F1 Optimization)")
    r.p(
        f"Với {pct(ds['positive_rate'], 1)} nhãn tích cực, việc tối ưu theo F1 của riêng lớp "
        "đa số sẽ che giấu hoàn toàn năng lực phát hiện nhận xét tiêu cực — vốn mới là thứ "
        "doanh nghiệp cần. Vì vậy chương này chuyển sang tối ưu <strong>Macro F1</strong>:",
        '<div class="formula">Macro F1 = (F1<sub>lớp 0</sub> + F1<sub>lớp 1</sub>) / 2</div>',
        "Chỉ số này đối xử với hai lớp <em>bình đẳng</em> bất kể số lượng mẫu, buộc mô hình "
        "phải làm tốt ở cả hai phía.")

    r.p(R.figure(
        "s3_fig04_threshold_curves.png",
        "Đường cong cân bằng lớp và tinh chỉnh ngưỡng quyết định trên văn bản NLP."))

    rows = []
    for n, v in sc.items():
        d, o = v["default"], v["optimal"]
        rows.append([
            n, f"{v['threshold']:.2f}",
            pct(d["macro_f1"]), pct(o["macro_f1"]),
            pct(d["recall_0"]), pct(o["recall_0"]),
            f"{(o['recall_0'] - d['recall_0']) * 100:+.2f}%",
        ])
    r.p(R.table(
        ["Mô hình", "Ngưỡng τ", "Macro F1 (0,50)", "Macro F1 (Opt)",
         "Recall lớp 0 (0,50)", "Recall lớp 0 (Opt)", "Δ Recall lớp 0"],
        rows,
        caption="Hiệu quả tinh chỉnh ngưỡng lên năng lực phát hiện nhận xét tiêu cực.",
        cls="tight"))

    r.p(
        "Điểm chung của cả bốn mô hình: ngưỡng tối ưu đều nằm <strong>cao hơn</strong> 0,50 "
        "(ngược hẳn với Chương 3). Nguyên nhân là hướng mất cân bằng đảo chiều — ở đây lớp đa "
        "số là lớp tích cực, nên xác suất dự đoán bị dồn về phía 1, và phải nâng ngưỡng lên mới "
        "phát hiện được nhận xét tiêu cực.")

    # ---------------- 5.4 PCA ----------------
    r.h(2, "5.4. Trực quan hoá quá trình học biểu diễn ngữ nghĩa bằng PCA 2D")
    r.p(
        f"Không gian TF-IDF ban đầu có {vec['max_features']:,} chiều với độ thưa "
        f"{vec['sparsity_pct']:.1f}%. Câu hỏi đặt ra: mạng nơ-ron biến đổi không gian thưa thớt "
        "này thành cái gì ở các tầng ẩn?")

    r.p(R.figure(
        "s3_fig05_representation_pca_spaces.png",
        "Trực quan hoá biến đổi không gian biểu diễn ngữ nghĩa bằng PCA 2D."))

    r.p(R.oim(
        "Ở không gian TF-IDF đầu vào, hai lớp nhận xét chồng lấn gần như hoàn toàn — điều dễ "
        "hiểu vì phần lớn từ vựng (tên sản phẩm, chất liệu, màu sắc) xuất hiện ở cả nhận xét "
        "tích cực lẫn tiêu cực. Qua các tầng ẩn, cụm màu đỏ (không giới thiệu) tách dần ra một "
        "phía.",
        "Mạng đã học được cách <strong>lọc bỏ từ vựng trung tính</strong> và khuếch đại trọng "
        "số của các từ mang cảm xúc. Phép chiếu từ 1.000 chiều thưa xuống 32 chiều đặc chính là "
        "một dạng nén ngữ nghĩa (Semantic Compression) do mạng tự khám phá.",
        "Đây là bằng chứng cho thấy Representation Learning hoạt động không chỉ trên dữ liệu số "
        "mà cả trên văn bản: mạng tự tìm ra một không gian ngữ nghĩa nơi \"khen\" và \"chê\" "
        "nằm ở hai vùng tách biệt, dù chưa từng được dạy khái niệm cảm xúc nào."))

    neg = s3["sentiment_terms"]["negative"][:8]
    pos = s3["sentiment_terms"]["positive"][:8]
    n = min(len(neg), len(pos))
    r.p(R.table(
        ["Hạng", "Từ khoá tiêu cực", "Trọng số", "Từ khoá tích cực", "Trọng số"],
        [[i + 1, f"<code class=\"inl\">{neg[i][0]}</code>", f"{neg[i][1]:.3f}",
          f"<code class=\"inl\">{pos[i][0]}</code>", f"{pos[i][1]:+.3f}"] for i in range(n)],
        caption="Các từ khoá mang trọng số phân cực mạnh nhất do mô hình tuyến tính học được.",
        cls="tight"))

    r.p(
        "Bảng trọng số này là minh chứng cho thấy mô hình học được ngữ nghĩa thật chứ không "
        "phải nhiễu ngẫu nhiên. Đặc biệt đáng chú ý là cụm bigram "
        "<code class=\"inl\">wanted love</code> nằm trong nhóm tiêu cực mạnh nhất — cụm này đến "
        "từ mẫu câu \"I wanted to love this but...\", một cách diễn đạt thất vọng lịch sự mà "
        "mô hình chỉ có thể bắt được nhờ cấu hình <code class=\"inl\">ngram_range = (1, 2)</code>. "
        "Nếu chỉ dùng từ đơn, cả hai từ <em>wanted</em> và <em>love</em> đều mang sắc thái "
        "trung tính hoặc tích cực.")

    # ---------------- 5.5 Benchmark ----------------
    r.h(2, "5.5. Tổng hợp đối chuẩn 4 mô hình xử lý văn bản NLP")

    champ = s3["champion"]
    r.p(R.table(
        ["Mô hình", "Loại mô hình", "Tham số", "Thời gian", "Ngưỡng",
         "Accuracy", "Macro F1", "Recall lớp 0", "ROC-AUC"],
        [[(f"<strong>{n}</strong>" if n == champ else n), v["kind"],
          f"{v['n_params']:,}", f"{v['time_sec']:.3f}s", f"{v['threshold']:.2f}",
          pct(v["optimal"]["accuracy"]), pct(v["optimal"]["macro_f1"]),
          pct(v["optimal"]["recall_0"]), f"{v['roc_auc']:.4f}"]
         for n, v in sc.items()],
        caption="Bảng đối chuẩn khoa học: Machine Learning và Deep Learning thuần NumPy "
                f"({ds['n_clean']:,} nhận xét).",
        cls="tight"))

    r.p(R.figure(
        "s3_fig06_model_comparison.png",
        "Đối chuẩn trực quan hiệu năng hệ thống phân loại đánh giá văn bản NLP."))

    r.h(3, "5.5.1. Nhận xét chuyên sâu và đánh giá kết quả phân loại văn bản NLP")

    lr_s, nb_s, rf_s, dl_s = (sc["Logistic Regression"], sc["Naive Bayes"],
                              sc["Random Forest"], sc["DL Text Scratch"])

    r.p(
        "<p><strong>1. Biểu diễn quan trọng hơn thuật toán.</strong></p>"
        f"Bốn mô hình có độ phức tạp chênh nhau rất xa — từ Naive Bayes với "
        f"{nb_s['n_params']:,} tham số đến mạng nơ-ron {dl_s['n_params']:,} tham số — nhưng "
        f"Macro F1 chỉ dao động trong khoảng "
        f"{min(v['optimal']['macro_f1'] for v in sc.values()) * 100:.2f}% – "
        f"{max(v['optimal']['macro_f1'] for v in sc.values()) * 100:.2f}%. Khoảng cách "
        f"{(max(v['optimal']['macro_f1'] for v in sc.values()) - min(v['optimal']['macro_f1'] for v in sc.values())) * 100:.2f} "
        "điểm là rất nhỏ so với khoảng cách giữa các mô hình ở Chương 3 và 4. "
        "<strong>Chất lượng phép biểu diễn TF-IDF mới là yếu tố quyết định</strong>, không phải "
        "độ tinh vi của thuật toán phân loại phía sau.")

    r.p(
        "<p><strong>2. Naive Bayes — mô hình đơn giản nhất, tốc độ vượt trội.</strong></p>"
        f"Naive Bayes huấn luyện trong {nb_s['time_sec']:.4f} giây — nhanh gấp "
        f"{dl_s['time_sec'] / max(nb_s['time_sec'], 1e-9):.0f} lần mạng nơ-ron — mà vẫn đạt "
        f"Macro F1 {pct(nb_s['optimal']['macro_f1'])} và ROC-AUC {nb_s['roc_auc']:.4f}. "
        "Giả định độc lập điều kiện giữa các từ (naive) tuy sai về mặt ngôn ngữ học nhưng lại "
        "rất hiệu quả trên ma trận đếm từ thưa. Trong một hệ thống thực tế cần xử lý hàng triệu "
        "nhận xét mỗi ngày, đây là đánh đổi rất đáng cân nhắc.")

    r.p(
        "<p><strong>3. Random Forest và giới hạn của cây quyết định trên dữ liệu thưa.</strong></p>"
        f"Random Forest — mô hình thống trị ở Chương 3 và 4 — lại đứng cuối bảng ở đây với "
        f"ROC-AUC {rf_s['roc_auc']:.4f}, thấp hơn Logistic Regression "
        f"({lr_s['roc_auc']:.4f}). Cây quyết định chọn <em>một</em> đặc trưng tại mỗi nút để "
        f"chia; trên không gian {vec['max_features']:,} chiều với độ thưa "
        f"{vec['sparsity_pct']:.1f}%, mỗi phép chia chỉ dựa trên sự có mặt của một từ đơn lẻ. "
        "Trong khi đó, cảm xúc của một câu là <strong>tổng hợp tuyến tính</strong> của nhiều từ "
        "cùng lúc — đúng là thứ mà Logistic Regression và mạng nơ-ron làm tự nhiên.")

    r.p(
        "<p><strong>4. Mạng nơ-ron thuần NumPy trên không gian 1.000 chiều.</strong></p>"
        f"Mô hình {dl['name']} ({dl['layers']}) đạt Macro F1 "
        f"{pct(dl_s['optimal']['macro_f1'])} và Recall lớp tiêu cực "
        f"{pct(dl_s['optimal']['recall_0'])} — dẫn đầu bảng đối chuẩn. Điều đáng ghi nhận là "
        f"toàn bộ {dl_s['n_params']:,} tham số này được huấn luyện bằng các phương trình lan "
        f"truyền ngược viết tay, hội tụ ổn định chỉ trong {dl_s['time_sec']:.2f} giây trên "
        f"{sp['n_train']:,} mẫu văn bản.")
