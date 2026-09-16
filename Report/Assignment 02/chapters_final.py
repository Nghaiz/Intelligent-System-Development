"""Chương X (so sánh + tổng hợp) và Chương XI (đồ thị tri thức Neo4j)."""

from build_report import GITHUB_URL, META, code, figure, fmt, note, table

D = META["diabetes"]
H = META["house_price"]
E = META["customer_behavior"]

DB = D["test_metrics"][D["best_model"]]
HB = H["test_metrics"][H["best_model"]]
EB = E["test_metrics"][E["best_model"]]


def write(r) -> None:
    _ch10(r)
    _wrapup(r)
    _ch11(r)


# ---------------------------------------------------------------- Chương X
def _ch10(r) -> None:
    r.h(1, "Chương X. So sánh ba hệ thống thông minh")
    r.p(
        "Ba ứng dụng được xây dựng theo cùng một quy trình và cùng một khung trình "
        "bày, nên chúng so sánh được với nhau. Chương này rút ra những gì chỉ nhìn "
        "thấy khi đặt cả ba cạnh nhau.",
    )

    r.h(2, "10.1. So sánh dữ liệu và bài toán")
    r.p(table(
        ["Khía cạnh", "Ứng dụng 1 — Tiểu đường", "Ứng dụng 2 — Giá nhà",
         "Ứng dụng 3 — Thương mại điện tử"],
        [
            ["Loại bài toán", "Phân loại nhị phân", "Hồi quy", "Phân loại nhị phân"],
            ["Dữ liệu thô", "CSV, bảng thuần số", "CSV, bảng có biến phân loại",
             "CSV, bảng + văn bản tự do"],
            ["Một quan sát", "Một lần khám của một bệnh nhân", "Một tin rao bán bất động sản",
             "Một lượt đánh giá sản phẩm"],
            ["Số mẫu thô", f"{D['n_samples']:,}", f"{H['n_samples_raw']:,}",
             f"{E['n_samples_raw']:,}"],
            ["Số mẫu sau làm sạch", f"{D['n_samples']:,}", f"{H['n_samples_clean']:,}",
             f"{E['n_samples_clean']:,}"],
            ["Biến mục tiêu", "<code class='inl'>Outcome</code> ∈ {0,1}",
             "<code class='inl'>Price</code> ∈ ℝ⁺ (tỷ VNĐ)",
             "<code class='inl'>Recommended IND</code> ∈ {0,1}"],
            ["Mất cân bằng lớp", "1,87 : 1", "Không áp dụng (hồi quy)", "≈ 4,5 : 1"],
        ]))
    r.p(
        "<strong>Ba tập dữ liệu khác nhau ở đâu.</strong> Ứng dụng 1 nhỏ nhất "
        f"({D['n_samples']} dòng) nhưng sạch nhất về cấu trúc — mọi cột đều là số. "
        f"Ứng dụng 2 lớn nhất ({H['n_samples_raw']:,} dòng) và bẩn nhất về giá trị "
        "thiếu. Ứng dụng 3 ở giữa về kích thước nhưng phức tạp nhất về kiểu dữ liệu.",
    )

    r.h(2, "10.2. So sánh biểu diễn dữ liệu")
    r.p(table(
        ["Ứng dụng", "Dạng thô", "Biểu diễn số", "Đầu vào mô hình", "Số chiều"],
        [
            ["Tiểu đường", "CSV / bảng", "Vectơ và ma trận đặc trưng",
             f"<em>B</em> × <em>d</em>", str(D["feature_matrix_shape"][1])],
            ["Giá nhà", "CSV / bảng", "Ma trận đặc trưng đã mã hoá",
             f"<em>B</em> × <em>d</em>", str(H["n_features_after_encoding"])],
            ["Thương mại điện tử", "CSV + bình luận",
             "Đặc trưng bảng + vectơ / embedding văn bản",
             "<em>B</em> × <em>d</em> và/hoặc <em>B</em> × <em>T</em> × <em>d</em>",
             f"{E['representation_shapes']['hybrid'][1]:,}"],
        ], "Bảng tóm tắt biểu diễn dữ liệu bắt buộc theo yêu cầu đề bài."))
    r.p("<strong>Giải thích từng chiều trong các hình dạng đã báo cáo:</strong>")
    t = E["tensor_demo"]
    r.p(table(
        ["Ký hiệu", "Nghĩa", "Giá trị cụ thể trong báo cáo này"],
        [
            ["<em>N</em>", "Số quan sát trong toàn tập",
             f"{D['feature_matrix_shape'][0]:,} / {H['feature_matrix_shape'][0]:,} / "
             f"{E['n_samples_clean']:,}"],
            ["<em>B</em>", "Số quan sát trong một lô đưa vào mô hình",
             f"Ứng dụng 3 minh hoạ với <em>B</em> = {t['B']}"],
            ["<em>d</em>", "Số chiều của một vectơ đặc trưng",
             f"{D['feature_matrix_shape'][1]} / {H['n_features_after_encoding']} / "
             f"{E['representation_shapes']['hybrid'][1]:,}"],
            ["<em>T</em>", "Số token của một văn bản, sau khi cắt hoặc đệm",
             f"Chỉ có ở Ứng dụng 3: <em>T</em> = {t['T']}"],
            ["<em>V</em>", "Kích thước từ điển token",
             f"Chỉ có ở Ứng dụng 3: <em>V</em> = {t['vocab_size']:,}"],
        ]))
    r.p(note(
        "Điểm khác biệt cốt lõi.",
        f"Ứng dụng 1 dùng {D['feature_matrix_shape'][1]} chiều, Ứng dụng 2 dùng "
        f"{H['n_features_after_encoding']} chiều, Ứng dụng 3 dùng "
        f"{E['representation_shapes']['hybrid'][1]:,} chiều — chênh nhau hơn ba bậc "
        "độ lớn. Nguyên nhân không phải bài toán khó hơn, mà là <strong>loại dữ liệu "
        "khác nhau</strong>: mỗi từ trong từ vựng trở thành một chiều riêng, nên văn "
        "bản luôn sinh ra không gian rất nhiều chiều và rất thưa."))

    r.h(2, "10.3. So sánh chất lượng dữ liệu và tiền xử lý")
    r.h(3, "10.3.1. Các bước tiền xử lý chung")
    r.p(
        "Bốn bước có mặt ở cả ba ứng dụng: khử bản ghi trùng lặp trước khi chia tập; "
        "điền khuyết bằng trung vị bên trong <code class='inl'>Pipeline</code>; chia "
        "70/15/15; và đóng gói toàn bộ tiền xử lý thành một artifact duy nhất để "
        "triển khai.",
    )
    r.h(3, "10.3.2. Các bước tiền xử lý riêng")
    r.p(table(
        ["Ứng dụng", "Vấn đề chất lượng đặc thù", "Xử lý riêng", "Nếu bỏ qua thì sao"],
        [
            ["Tiểu đường",
             "Giá trị thiếu mã hoá thành số 0 (Insulin 48,7%)",
             "<code class='inl'>FunctionTransformer</code> chuyển 0 thành "
             "<code class='inl'>NaN</code> trước khi điền khuyết",
             "Mô hình học rằng “glucose = 0” là một tình trạng sức khoẻ hợp lệ"],
            ["Giá nhà",
             "Thiếu tới 82,6%; bản ghi trùng lặp; địa chỉ dạng văn bản tự do",
             "Coi “không khai báo” là một hạng mục; khử trùng lặp; rút tỉnh và huyện "
             "từ địa chỉ; thêm 5 đặc trưng phái sinh",
             "Mất 82,6% dữ liệu nếu xoá dòng; R² bị thổi phồng nếu không khử trùng; "
             "mất hoàn toàn biến vị trí"],
            ["Thương mại điện tử",
             "Rò rỉ nhãn qua cột <code class='inl'>Rating</code>; mất cân bằng 4,5 : 1; "
             "bình luận rỗng",
             "Loại bỏ <code class='inl'>Rating</code> và <code class='inl'>Clothing ID</code>; "
             "<code class='inl'>class_weight=\"balanced\"</code>; loại dòng không có bình luận",
             "Accuracy trên 90% nhưng mô hình chỉ đọc lại số sao, vô dụng khi triển khai"],
        ]))
    r.p(
        "<strong>Cả ba ứng dụng đều có một vấn đề chất lượng mà công cụ tiêu chuẩn "
        "không phát hiện được</strong>, và mỗi lần là một dạng khác nhau. Đây là "
        "phát hiện đáng chú ý nhất khi đặt ba ứng dụng cạnh nhau: sự cố về dữ liệu "
        "không lặp lại theo khuôn mẫu, nên không có một danh sách kiểm tra cố định "
        "nào thay được việc thực sự đọc và hiểu dữ liệu.",
    )

    r.h(2, "10.4. So sánh mô hình và đánh giá")
    r.p(table(
        ["Ứng dụng", "Số mô hình", "Mô hình được chọn", "Độ đo chính", "Kết quả chính",
         "So với baseline"],
        [
            ["Tiểu đường", "5", D["best_model_label"], "Recall",
             f"ROC-AUC {fmt(DB['ROC-AUC'])}, Recall {fmt(DB['Recall'])}",
             f"Accuracy {fmt(DB['Accuracy'])} so với {fmt(D['baseline_accuracy'])}"],
            ["Giá nhà", "5", H["best_model_label"], "MAE",
             f"MAE {fmt(HB['MAE'])} tỷ, R² {fmt(HB['R2'])}",
             f"MAE giảm {fmt((1 - HB['MAE'] / H['baseline']['MAE']) * 100, 1)}% so với baseline"],
            ["Thương mại điện tử", "6", E["best_model_label"], "F1 lớp thiểu số",
             f"ROC-AUC {fmt(EB['ROC-AUC'])}, F1 lớp 0 {fmt(EB['F1 (lớp 0)'])}",
             f"Accuracy {fmt(EB['Accuracy'])} so với {fmt(E['baseline_accuracy'])}"],
        ], "So sánh kết quả cuối cùng của ba hệ thống."))
    r.p(
        "<strong>Vì sao độ đo chính khác nhau ở cả ba.</strong> Ứng dụng 1 ưu tiên "
        "Recall vì bỏ sót ca bệnh đắt hơn cảnh báo nhầm. Ứng dụng 3 ưu tiên F1 lớp "
        "thiểu số vì ngược lại — bỏ lọt phản hồi xấu mới là sai lầm đắt. Ứng dụng 2 "
        "dùng MAE vì nó là con số duy nhất người dùng cuối đọc được trực tiếp bằng "
        "tỷ VNĐ. Không có một độ đo nào đúng cho cả ba.",
    )
    r.p(note(
        "Phát hiện nhất quán ở cả ba ứng dụng.",
        "Ở mỗi ứng dụng, khoảng cách điểm số do <strong>thay đổi biểu diễn dữ liệu</strong> "
        "lớn hơn khoảng cách do <strong>thay đổi thuật toán</strong>:"
        "<ul>"
        "<li><strong>Ứng dụng 1</strong> — chênh lệch ROC-AUC giữa mô hình tốt nhất "
        f"({fmt(DB['ROC-AUC'], 3)}) và kém nhất "
        f"({fmt(min(v['ROC-AUC'] for v in D['test_metrics'].values()), 3)}) là "
        f"{fmt(DB['ROC-AUC'] - min(v['ROC-AUC'] for v in D['test_metrics'].values()), 3)}.</li>"
        "<li><strong>Ứng dụng 2</strong> — năm đặc trưng phái sinh cải thiện R² ở mọi "
        "mô hình; khử trùng lặp thay đổi R² nhiều hơn cả việc đổi thuật toán.</li>"
        "<li><strong>Ứng dụng 3</strong> — chênh lệch ROC-AUC do biểu diễn là "
        f"{fmt(max(v['ROC-AUC'] for k, v in E['test_metrics'].items() if E['model_repr'][k] != 'tabular') - max(v['ROC-AUC'] for k, v in E['test_metrics'].items() if E['model_repr'][k] == 'tabular'), 3)}, "
        "lớn hơn hẳn chênh lệch giữa các thuật toán trong cùng một biểu diễn.</li>"
        "</ul>", "good"))

    r.h(2, "10.5. So sánh triển khai")
    r.p(table(
        ["Khía cạnh", "Tiểu đường", "Giá nhà", "Thương mại điện tử"],
        [
            ["Cổng", "5001", "5002", "5003"],
            ["Điểm cuối",
             "<code class='inl'>/diabetes/v1/predict</code>",
             "<code class='inl'>/house-price/v1/predict</code>",
             "<code class='inl'>/customer-behavior/v1/predict</code>"],
            ["Số trường người dùng nhập", str(len(D["feature_columns"])), "11", "7"],
            ["API tự tính thêm", "Không",
             "5 đặc trưng phái sinh", "<code class='inl'>full_text</code>, "
             "<code class='inl'>word_count</code>"],
            ["Số artifact phải nạp", "1 preprocessor + 5 mô hình",
             "1 preprocessor + 5 mô hình", "3 preprocessor + 6 mô hình"],
            ["Đầu ra", "Nhãn + độ tin cậy + tri thức Neo4j",
             "Giá + khoảng dao động", "Nhãn + độ tin cậy + tên biểu diễn"],
            ["Triển khai Web", "Có", "Có", "Có"],
            ["Triển khai Mobile", "Có", "Có", "Có"],
        ]))
    r.p(
        "<strong>Ứng dụng nào dễ triển khai nhất.</strong> Ứng dụng 1 — người dùng "
        "nhập năm con số, API không phải tính thêm gì, artifact nhỏ. "
        "<strong>Ứng dụng nào khó nhất.</strong> Ứng dụng 3 — phải nạp ba bộ tiền xử "
        "lý khác nhau và chọn đúng bộ theo mô hình người dùng chọn, cộng với từ điển "
        f"TF-IDF {E['representation_shapes']['hybrid'][1]:,} chiều khiến artifact lớn "
        "hơn hẳn. Ứng dụng 2 nằm giữa nhưng có rủi ro riêng: API phải tính lại năm "
        "đặc trưng phái sinh đúng <em>y hệt</em> công thức lúc huấn luyện, nếu lệch "
        "thì không có gì báo lỗi.",
    )
    r.p(
        "<strong>Ứng dụng nào tốn tài nguyên tính toán nhất.</strong> Ứng dụng 2 khi "
        f"huấn luyện (Gradient Boosting trên {H['n_samples_clean']:,} dòng × "
        f"{H['n_features_after_encoding']} chiều). Khi suy luận thì cả ba đều dưới "
        "một phần trăm giây.",
    )


# --------------------------------------------------------------- Chương XI
def _ch11(r) -> None:
    r.h(1, "Chương XI. Mở rộng — Tích hợp đồ thị tri thức Neo4j")
    r.p(
        "Phần mở rộng này bổ sung một tầng <strong>đồ thị tri thức</strong> vào Ứng "
        "dụng 1. Mục tiêu: sau khi mô hình đã dự đoán, hệ thống trả về thêm nội dung "
        "tư vấn phù hợp với tầng nguy cơ của bệnh nhân, thay vì chỉ trả về một con số "
        "xác suất.",
    )
    r.p(note(
        "Nguyên tắc thiết kế bắt buộc.",
        "Đồ thị tri thức <strong>chỉ bổ sung nội dung, không bao giờ tham gia vào "
        "việc dự đoán</strong>. Nhãn và xác suất do mô hình học máy quyết định hoàn "
        "toàn. Nếu Neo4j hỏng, mất kết nối hoặc chưa cấu hình, API vẫn phải trả về dự "
        "đoán bình thường kèm một thông báo ở trường "
        "<code class='inl'>knowledge_error</code>. Đây là một ràng buộc kiến trúc, "
        "không phải một tuỳ chọn."))

    r.h(2, "11.1. Luồng hoạt động")
    r.p(
        '<div class="flow">Đầu vào người dùng → Tiền xử lý → Mô hình → Xác suất → '
        "Phân tầng nguy cơ → Truy vấn Neo4j → Kết quả + Tư vấn</div>",
        "Điểm nối duy nhất giữa mô hình và đồ thị là <strong>xác suất dương tính</strong>. "
        "Xác suất ấy được ánh xạ sang một trong ba tầng nguy cơ, và tầng nguy cơ là "
        "khoá tra cứu vào đồ thị.",
    )
    r.p(figure("fig_kg_pipeline.png",
               "Vị trí của Neo4j trong hệ thống. Nhánh dự đoán (trên) và nhánh tri "
               "thức (dưới) tách rời nhau: đồ thị nhận vào tầng nguy cơ đã được mô "
               "hình quyết định, và không có mũi tên nào đi ngược lại vào mô hình."))
    r.p(table(
        ["Tầng nguy cơ", "Ngưỡng xác suất", "Nút <code class='inl'>RiskTier</code> trong đồ thị"],
        [
            ["Nhóm Nguy cơ Cao", "≥ 0,25", "<code class='inl'>med_high</code>"],
            ["Nhóm Tiền Đái tháo đường", "0,15 – 0,25", "<code class='inl'>med_moderate</code>"],
            ["Nhóm Khoẻ mạnh", "&lt; 0,15", "<code class='inl'>med_low</code>"],
        ]))
    r.p(
        "Ngưỡng đặt thấp hơn 0,5 có chủ đích: với bài toán sàng lọc, một bệnh nhân có "
        "xác suất 0,3 chưa bị mô hình gán nhãn dương tính nhưng vẫn đáng được nhận "
        "khuyến nghị theo dõi. Đây chính là ưu tiên Recall đã phân tích ở mục 7.8, "
        "được áp dụng ở tầng tư vấn thay vì tầng phân loại.",
    )

    r.h(3, "11.1.1. Lược đồ đồ thị")
    r.p(
        "Đồ thị được xây dựng lại từ phần mở rộng của Assignment 01 và nạp lên một "
        "phiên bản Neo4j Aura đang chạy. Lược đồ dưới đây liệt kê các loại thực thể "
        "và loại quan hệ đọc trực tiếp từ cơ sở dữ liệu bằng "
        "<code class='inl'>CALL db.labels()</code> và "
        "<code class='inl'>CALL db.relationshipTypes()</code>.",
    )
    r.p(figure("fig_kg_schema.png",
               "Lược đồ bản thể học của đồ thị tri thức: các loại thực thể, các loại "
               "quan hệ nối chúng, và số lượng thực tế của từng loại."))
    r.p(figure("fig_kg_full.png",
               "Ba nhánh tri thức y tế trên Neo4j Aura, mỗi nhánh ứng với một nút "
               "<code class='inl'>RiskTier</code> và chính là một bộ tư vấn mà API "
               "có thể trả về. Màu nút mã hoá loại thực thể. Hình chỉ vẽ phần miền "
               "y tế vì đó là phần duy nhất Ứng dụng 1 truy vấn tới; miền bất động "
               "sản của đồ thị xuất hiện đầy đủ ở hình lược đồ phía trên."))
    r.p(table(
        ["Nhãn nút", "Ý nghĩa", "Số lượng"],
        [
            ["<code class='inl'>RiskTier</code>", "Ba tầng nguy cơ", "3"],
            ["<code class='inl'>Action</code>", "Hướng dẫn lâm sàng cần thực hiện", "9"],
            ["<code class='inl'>CarePackage</code>", "Gói chăm sóc gắn với từng tầng", "3"],
            ["<code class='inl'>Device</code>", "Thiết bị theo dõi (máy đo, que thử…)", "4"],
            ["<code class='inl'>Nutrition</code>", "Sản phẩm dinh dưỡng hỗ trợ", "6"],
            ["<code class='inl'>Service</code>", "Dịch vụ nhà thuốc đi kèm", "9"],
        ]))
    r.p(
        "Quan hệ chính: "
        "<code class='inl'>(RiskTier)-[:ADVISES]-&gt;(Action)</code> và "
        "<code class='inl'>(RiskTier)-[:RECOMMENDS_PACKAGE]-&gt;(CarePackage)"
        "-[:INCLUDES_PRODUCT]-&gt;(Device | Nutrition | Service)</code>.",
    )

    r.h(3, "11.1.2. Truy vấn Cypher")
    r.p(code(
        "MATCH (tier:RiskTier {case: $case})\n"
        "OPTIONAL MATCH (tier)-[:ADVISES]->(action:Action)\n"
        "OPTIONAL MATCH (tier)-[:RECOMMENDS_PACKAGE]->(:CarePackage)\n"
        "              -[:INCLUDES_PRODUCT]->(item)\n"
        "RETURN tier.name AS tier_name,\n"
        "       collect(DISTINCT {kind: 'Action', name: action.name,\n"
        "                         detail: action.`Nhóm`}) AS actions,\n"
        "       collect(DISTINCT {kind: item.kind, name: item.name,\n"
        "                         detail: item.`Công dụng`,\n"
        "                         price: item.`Khoảng giá`,\n"
        "                         url:   item.`Tra cứu giá`}) AS products",
        "Truy vấn Cypher lấy tư vấn theo tầng nguy cơ.", "Cypher"))
    r.p(
        "Hai mệnh đề <code class='inl'>OPTIONAL MATCH</code> là chi tiết quan trọng: "
        "tầng “Nhóm Khoẻ mạnh” không có gói thiết bị theo dõi, và nếu dùng "
        "<code class='inl'>MATCH</code> thường thì toàn bộ kết quả — kể cả các hướng "
        "dẫn lối sống — sẽ biến mất, khiến người khoẻ mạnh không nhận được lời khuyên nào.",
    )
    r.p(code(
        "def resolve_risk_case(probability_positive):\n"
        "    \"\"\"Chọn tầng nguy cơ theo xác suất dương tính mà mô hình trả về.\"\"\"\n"
        "    for threshold, case in RISK_TIERS:\n"
        "        if probability_positive >= threshold:\n"
        "            return case\n"
        "    return \"med_low\"",
        "Ánh xạ xác suất của mô hình sang tầng nguy cơ."))

    r.h(2, "11.2. Kết quả thực nghiệm")
    r.p(
        "Hệ thống được kiểm thử với hai hồ sơ bệnh nhân trái ngược nhau, gửi tới cùng "
        "một điểm cuối <code class='inl'>POST /diabetes/v1/predict</code>.",
    )
    r.p(figure("fig_kg_case_high.png",
               "Đồ thị con được kích hoạt cho một ca nguy cơ cao: từ xác suất mô hình "
               "trả về, hệ thống đi tới nút tầng nguy cơ tương ứng rồi thu thập toàn "
               "bộ hướng dẫn, thiết bị, dinh dưỡng và dịch vụ nối vào nút đó. Đây "
               "chính là nội dung trường <code class='inl'>knowledge</code> trong "
               "phản hồi JSON."))
    r.p(code(
        "$ curl -X POST localhost:5001/diabetes/v1/predict \\\n"
        "    -H \"Content-Type: application/json\" \\\n"
        "    -d '{\"Glucose\":180,\"BMI\":40,\"Age\":55,\n"
        "         \"Pregnancies\":7,\"DiabetesPedigreeFunction\":1.1}'",
        "", "Bash"))
    r.p("""<div class="outbox"><div class="outhdr">Kết quả chạy</div><pre class="out">class: 1 | p+: 88.94 | tier: Nhóm Nguy cơ Cao | err: None
  Hướng dẫn cần làm -> ['Khám chuyên khoa Nội tiết để xét nghiệm HbA1c tĩnh mạch',
                        'Đo đường huyết đói và sau ăn 2 giờ mỗi ngày',
                        'Tuyệt đối không tự dùng Metformin khi chưa có chỉ định bác sĩ']
  Thiết bị theo dõi -> ['Máy đo đường huyết Accu-Chek Instant',
                        'Que thử đường huyết Accu-Chek Instant (hộp 50 que)',
                        'Kim chích máu Accu-Chek Softclix (hộp 25 kim)']
  Dinh dưỡng hỗ trợ -> ['Sữa Abbott Glucerna 850g',
                        'Viên uống Dây thìa canh Diabetna (hộp 40 viên)']</pre></div>
<div class="codecap">Bệnh nhân nguy cơ cao — xác suất 88,94%, đồ thị trả về ba nhóm tư vấn.</div>""")
    r.p("""<div class="outbox"><div class="outhdr">Kết quả chạy</div><pre class="out">class: 0 | p+: 2.06 | tier: Nhóm Khoẻ mạnh
  Hướng dẫn cần làm -> ['Duy trì chế độ dinh dưỡng cân bằng',
                        'Giữ chỉ số BMI trong khoảng 18.5 đến 22.9',
                        'Khám sức khoẻ định kỳ hàng năm']
  Dinh dưỡng hỗ trợ -> ['Viên uống Vitamin tổng hợp &amp; Khoáng chất']</pre></div>
<div class="codecap">Bệnh nhân nguy cơ thấp — xác suất 2,06%, nội dung tư vấn hoàn toàn khác.</div>""")
    r.p(
        "Hai kết quả cho thấy tầng tri thức <strong>phản ứng đúng theo đầu ra của mô "
        "hình</strong>: cùng một điểm cuối, cùng một truy vấn, nhưng nội dung trả về "
        "khác hẳn nhau vì xác suất khác nhau dẫn tới tầng nguy cơ khác nhau.",
    )

    r.h(3, "11.2.1. Kiểm chứng nguyên tắc “không ảnh hưởng tới dự đoán”")
    r.p(
        "Khi ngắt biến môi trường <code class='inl'>NEO4J_PASSWORD</code>, API vẫn "
        "khởi động và vẫn dự đoán bình thường; trường "
        "<code class='inl'>knowledge</code> trả về mảng rỗng và "
        "<code class='inl'>knowledge_error</code> chứa thông báo “Chưa cấu hình kết "
        "nối Neo4j”. Nhãn và xác suất <strong>không thay đổi một chữ số nào</strong>.",
        "Đây là bằng chứng cho ràng buộc kiến trúc đã nêu ở đầu chương: tầng tri thức "
        "là một lớp bổ sung có thể tháo rời, không phải một thành phần của mô hình.",
    )

    r.h(3, "11.2.2. Giá trị mang lại và hạn chế")
    r.p(
        "<strong>Giá trị.</strong> Một con số xác suất, tự nó, không cho người dùng "
        "biết phải làm gì tiếp theo. Tầng tri thức biến đầu ra của mô hình thành một "
        "khuyến nghị hành động cụ thể, và vì tri thức nằm trong đồ thị chứ không nằm "
        "trong mã nguồn, nội dung tư vấn có thể được cập nhật mà không cần huấn luyện "
        "lại hay triển khai lại mô hình.",
        "<strong>Hạn chế.</strong> Nội dung trong đồ thị được nhập thủ công, nên phạm "
        "vi còn hẹp và cần chuyên gia y tế thẩm định trước khi dùng thật. Ngoài ra "
        "tư vấn hiện chỉ phụ thuộc tầng nguy cơ, chưa cá nhân hoá theo từng chỉ số cụ "
        "thể của bệnh nhân — một bệnh nhân nguy cơ cao vì BMI và một bệnh nhân nguy "
        "cơ cao vì glucose hiện nhận cùng một bộ khuyến nghị.",
    )


# ------------------------------------------- Phần tổng hợp cuối Chương X
def _wrapup(r) -> None:
    """Phần tổng hợp cuối Chương X.

    Trước đây đây là một chương "Kết luận" riêng. Gộp vào Chương X vì nó
    tổng hợp đúng những gì Chương X vừa so sánh — tách ra thành chương
    riêng chỉ khiến người đọc gặp lại cùng một nội dung hai lần.
    """
    r.h(2, "10.6. Tổng hợp và bài học rút ra")
    r.p(
        "Năm mục so sánh ở trên đối chiếu ba hệ thống trên từng chiều riêng lẻ. Mục "
        "này tổng hợp lại: điều gì đúng ở cả ba, và điều đó nói lên gì về việc xây "
        "dựng một hệ thống thông minh.",
        '<div class="flow">Dữ liệu thô → Làm sạch → Biểu diễn → Học → Đánh giá → '
        "Lưu trữ → Triển khai</div>",
    )

    r.h(3, "10.6.1. Sáu bài học chính")
    r.p(note(
        "Một hệ thống thông minh không phải là một mô hình học máy đã huấn luyện.",
        "Nó là sự kết hợp của dữ liệu, biểu diễn, quá trình học, phép đánh giá, phần "
        "mềm triển khai và tương tác với người dùng. Thiếu bất kỳ thành phần nào thì "
        "kết quả không dùng được — và trong sáu thành phần ấy, mô hình chỉ là một."))
    r.p(table(
        ["#", "Bài học", "Bằng chứng trong báo cáo"],
        [
            ["1", "<strong>Biểu diễn dữ liệu quan trọng hơn thuật toán.</strong> Ở cả "
             "ba ứng dụng, thay đổi cách biểu diễn tạo ra khác biệt lớn hơn thay đổi mô hình.",
             "Mục 9.8 — ROC-AUC "
             f"{fmt(max(v['ROC-AUC'] for k, v in E['test_metrics'].items() if E['model_repr'][k] == 'tabular'), 3)} "
             f"→ {fmt(EB['ROC-AUC'], 3)}; mục 8.5.2 — năm đặc trưng phái sinh"],
            ["2", "<strong>Công cụ kiểm tra chất lượng dữ liệu có thể báo sạch khi dữ "
             "liệu bẩn.</strong> Cả ba ứng dụng đều có một vấn đề mà hàm tiêu chuẩn "
             "không phát hiện, và mỗi lần là một dạng khác nhau.",
             "Mục 7.4 (số 0 giả), mục 8.4 (trùng lặp), mục 9.4 (rò rỉ nhãn)"],
            ["3", "<strong>Điểm số cao hơn không đồng nghĩa với mô hình tốt hơn.</strong> "
             "Nó có thể chỉ có nghĩa là phép đo đã bị nhiễm.",
             "Mục 8.4.1 — đo trực tiếp mức thổi phồng R² do không khử trùng lặp"],
            ["4", "<strong>Độ đo phải chọn theo cái giá của từng loại sai lầm</strong>, "
             "không chọn theo thói quen. Hai bài toán cùng là phân loại nhị phân nhưng "
             "ưu tiên ngược nhau.",
             "Mục 4.2 — FN đắt ở Ứng dụng 1, FP đắt ở Ứng dụng 3"],
            ["5", "<strong>Có những giới hạn không mô hình nào vượt qua được</strong>, "
             "vì thông tin cần thiết không nằm trong dữ liệu.",
             "Mục 7.8.2 (ca bệnh sớm), mục 9.9.2 (bình luận mâu thuẫn nội tại)"],
            ["6", "<strong>Nhất quán giữa huấn luyện và triển khai là một ràng buộc "
             "kỹ thuật, không phải một lời khuyên.</strong> Sai thứ tự cột không sinh "
             "ra lỗi nào — chỉ sinh ra kết quả sai.",
             "Mục 6.2 và 6.3 — <code class='inl'>metadata.json</code> và phép kiểm "
             "chứng artifact"],
        ]))

    r.h(2, "10.7. Thách thức kỹ thuật lớn nhất")
    r.p(
        "Không phải việc huấn luyện mô hình, mà là <strong>phát hiện ra ba dạng rò rỉ "
        "dữ liệu</strong> — mỗi ứng dụng một dạng, và không dạng nào tự lộ ra. Rò rỉ "
        "qua tiền xử lý được chặn bằng một quy tắc kỹ thuật đơn giản; hai dạng còn "
        "lại chỉ có hiểu biết về ý nghĩa dữ liệu mới nhận ra được. Đáng chú ý là cả "
        "ba dạng đều làm điểm số <em>đẹp hơn</em>, nên không có động lực tự nhiên nào "
        "để đi tìm chúng.",
    )

    r.h(2, "10.8. Vấn đề biểu diễn dữ liệu quan trọng nhất")
    r.p(
        "Ở Ứng dụng 3, cột dễ dùng nhất lại là cột phải vứt đi "
        "(<code class='inl'>Rating</code> cho điểm số đẹp nhưng vô giá trị), còn cột "
        "trông khó dùng nhất lại mang gần như toàn bộ tín hiệu (văn bản tự do). Ở Ứng "
        "dụng 2, biến giải thích mạnh nhất — vị trí — nằm chôn trong một cột địa chỉ "
        "mà mô hình không đọc được, và một hàm rút chuỗi năm dòng đã giải phóng nó. "
        "Cả hai trường hợp đều nói cùng một điều: <strong>quyết định đưa gì vào biểu "
        "diễn quan trọng hơn quyết định dùng thuật toán nào</strong>.",
    )

    r.h(2, "10.9. Bài học về triển khai")
    r.p(
        "Khoảng cách giữa “mô hình chạy trong notebook” và “dịch vụ chạy được” lớn "
        "hơn dự kiến, và phần lớn khoảng cách ấy nằm ở việc giữ cho biểu diễn lúc dự "
        "đoán trùng khớp biểu diễn lúc huấn luyện. Ứng dụng 2 là ví dụ rõ nhất: API "
        "phải tính lại năm đặc trưng phái sinh đúng y hệt công thức lúc huấn luyện, "
        "và nếu lệch thì không có gì báo lỗi — mô hình vẫn trả về một con số hợp lý "
        "về mặt hình thức.",
    )

    r.h(2, "10.10. Hướng cải tiến")
    r.p(
        "<ul>"
        "<li><strong>Ứng dụng 1</strong> — bổ sung HbA1c và dữ liệu theo thời gian. "
        "Giới hạn hiện tại nằm ở dữ liệu, không ở mô hình, nên đây là hướng duy nhất "
        "thực sự cải thiện được các ca bệnh giai đoạn sớm.</li>"
        "<li><strong>Ứng dụng 2</strong> — giữ chi tiết địa chỉ tới cấp phường và bổ "
        "sung giá giao dịch thực thay vì giá chào bán.</li>"
        "<li><strong>Ứng dụng 3</strong> — thay TF-IDF bằng mô hình ngôn ngữ đọc được "
        "ngữ cảnh, để xử lý cấu trúc nhượng bộ kiểu “đẹp nhưng không vừa” — nhóm sai "
        "sót lớn nhất hiện nay.</li>"
        "<li><strong>Chung cho cả ba</strong> — bổ sung giám sát trôi dữ liệu sau khi "
        "triển khai. Một mô hình đúng lúc huấn luyện sẽ dần sai khi phân bố dữ liệu "
        "thực tế thay đổi, và hiện chưa có cơ chế nào phát hiện điều đó.</li></ul>",
    )
    r.p(note(
        "Kết lại.",
        "Một hệ thống thông minh có thể triển khai là sự kết hợp của <strong>dữ liệu, "
        "biểu diễn, quá trình học, đánh giá, phần mềm, triển khai và tương tác người "
        "dùng</strong>. Assignment 02 cho thấy phần khó nhất không nằm ở chỗ người ta "
        "thường nghĩ."))
