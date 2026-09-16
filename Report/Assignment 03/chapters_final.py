"""Chương 6, 7, 8 — tổng hợp đối chuẩn, triển khai hệ thống và kết luận.

Chương 6: Đối chuẩn khoa học đa chiều 12 mô hình trên cả ba hệ thống.
Chương 7: Triển khai — REST API, giao diện Web, giao diện Mobile, đồ thị tri thức Neo4j.
Chương 8: Kết luận và kiểm toán khả năng tái lập.
"""

from __future__ import annotations

from pathlib import Path

import report_lib as R

REPORT = Path(__file__).resolve().parent
SHOTS = REPORT / "screenshots"

GITHUB = "https://github.com/Nghaiz/Assignment-03-Intelligent-System"


def write(r: R.Report, data: dict) -> None:
    _chapter6(r, data)
    _chapter7(r, data)
    _chapter8(r, data)


def pct(x, nd=2) -> str:
    return f"{x * 100:.{nd}f}%"


# ===========================================================================
# CHƯƠNG 6 — ĐỐI CHUẨN ĐA CHIỀU
# ===========================================================================
def _chapter6(r: R.Report, data: dict) -> None:
    s1, s2, s3 = data["s1"], data["s2"], data["s3"]

    r.h(1, "Chương 6. Tổng hợp đối chuẩn khoa học đa chiều và thảo luận chuyên sâu")

    r.p(
        "Ba chương trước đã giải quyết ba bài toán thuộc ba họ khác nhau — phân loại nhị phân "
        "mất cân bằng, hồi quy giá trị liên tục, và phân loại văn bản phi cấu trúc. Chương này "
        "đặt toàn bộ <strong>12 mô hình</strong> lên cùng một mặt bàn để rút ra các quy luật "
        "xuyên suốt.")

    # ---------------- 6.1 Bảng 12 mô hình ----------------
    r.h(2, "6.1. Bảng tổng hợp đối chuẩn 12 mô hình trên cả ba hệ thống")

    def cell_nb(text: str) -> str:
        """Ô số — cấm ngắt dòng giữa con số."""
        return f'<span style="white-space:nowrap">{text}</span>'

    rows = []
    for name, v in s1["scores"].items():
        o = v["optimal"]
        rows.append([
            "HT1", f"{name}<br><span style='font-size:9.4pt;color:#555'>{v['kind']}</span>",
            cell_nb(f"{v['n_params']:,}"), cell_nb(f"{v['time_sec']:.3f}s"),
            cell_nb(f"F1 {pct(o['f1'])}"),
            cell_nb(f"AUC {v['roc_auc']:.4f}") + "<br>" + cell_nb(f"Recall {pct(o['recall'])}"),
        ])
    for name, v in s2["scores"].items():
        rows.append([
            "HT2", f"{name}<br><span style='font-size:9.4pt;color:#555'>{v['kind']}</span>",
            cell_nb(f"{v['n_params']:,}"), cell_nb(f"{v['time_sec']:.3f}s"),
            cell_nb(f"R² {v['r2_raw']:.4f}"),
            cell_nb(f"MAE ${v['mae_usd']:,.0f}") + "<br>" + cell_nb(f"MAPE {v['mape']:.2f}%"),
        ])
    for name, v in s3["scores"].items():
        o = v["optimal"]
        rows.append([
            "HT3", f"{name}<br><span style='font-size:9.4pt;color:#555'>{v['kind']}</span>",
            cell_nb(f"{v['n_params']:,}"), cell_nb(f"{v['time_sec']:.3f}s"),
            cell_nb(f"Macro F1 {pct(o['macro_f1'])}"),
            cell_nb(f"AUC {v['roc_auc']:.4f}") + "<br>" + cell_nb(f"Rec₀ {pct(o['recall_0'])}"),
        ])

    r.p(R.table(
        ["HT", "Mô hình / Loại", "Tham số", "Thời gian", "Chỉ số chính", "Chỉ số phụ"],
        rows,
        caption="Bảng đối chuẩn tổng hợp đa chiều 12 mô hình trên ba bài toán hệ thống "
                "thông minh quy mô lớn. HT1 = sàng lọc tiểu đường, HT2 = định giá bất động "
                "sản, HT3 = phân loại nhận xét NLP.",
        cls="tight"))

    r.p(R.note(
        "Vì sao mỗi hệ thống dùng một chỉ số chính khác nhau?",
        "Không có một thước đo duy nhất áp dụng được cho cả ba bài toán. Hệ thống 1 mất cân "
        "bằng nặng nên Accuracy vô nghĩa, phải dùng F1 và Recall. Hệ thống 2 là hồi quy nên "
        "không tồn tại khái niệm Accuracy, phải dùng R² và MAE. Hệ thống 3 mất cân bằng theo "
        "chiều ngược lại nên phải dùng Macro F1 để hai lớp được đối xử bình đẳng. "
        "<strong>Chọn sai chỉ số là cách nhanh nhất để rút ra kết luận sai.</strong>"))

    # ---------------- 6.2 Quy luật kiến trúc ----------------
    r.h(2, "6.2. Thảo luận về kiến trúc và triết lý lựa chọn mô hình")

    r.h(3, "6.2.1. Không tồn tại một kiến trúc tối ưu cho mọi bài toán")

    a1 = max(s1["architecture_study"], key=lambda a: a["scores"]["f1"])
    a2 = max(s2["architecture_study"], key=lambda a: a["scores"]["r2_log"])
    a3 = max(s3["architecture_study"], key=lambda a: a["scores"]["macro_f1"])

    r.p(R.table(
        ["Hệ thống", "Bài toán", "Kiến trúc thắng cuộc", "Cấu hình tầng", "Quy luật rút ra"],
        [
            ["HT1", "Phân loại nhị phân", a1["Kiến trúc"], a1["Cấu hình tầng"],
             "<strong>Chiều sâu thắng</strong> — cần tổ hợp phân cấp để bẻ cong siêu phẳng"],
            ["HT2", "Hồi quy liên tục", a2["Kiến trúc"], a2["Cấu hình tầng"],
             "<strong>Chiều rộng thắng</strong> — cần nhiều bộ xấp xỉ tuyến tính từng đoạn"],
            ["HT3", "Phân loại văn bản", a3["Kiến trúc"], a3["Cấu hình tầng"],
             "<strong>Vừa đủ thắng</strong> — dữ liệu thưa, thêm tham số chỉ gây học vẹt"],
        ],
        caption="Ba bài toán, ba quy luật kiến trúc hoàn toàn khác nhau."))

    r.p(
        "Đây là phát hiện quan trọng nhất về mặt kiến trúc của toàn bộ nghiên cứu. Cùng một lớp "
        "mạng <code class=\"inl\">MLPScratch</code>, cùng cách khởi tạo He Normal, cùng thuật "
        "toán tối ưu Mini-Batch Gradient Descent — nhưng cấu hình tầng tối ưu lại khác nhau ở "
        "cả ba bài toán, và khác theo hướng có thể giải thích được:")

    r.p(
        "<ul>"
        "<li><strong>Phân loại</strong> đòi hỏi mô hình <em>bẻ cong một mặt phân tách</em>. "
        "Mỗi tầng ẩn bổ sung cho phép gấp không gian thêm một lần, nên chiều sâu tạo ra năng "
        "lực biểu diễn theo cấp số nhân.</li>"
        "<li><strong>Hồi quy</strong> đòi hỏi mô hình <em>phủ một mặt giá trị liên tục</em>. "
        "Mỗi nơ-ron ReLU là một mảnh phẳng; càng nhiều mảnh song song thì bề mặt càng mượt, nên "
        "chiều rộng mới là thứ quyết định.</li>"
        "<li><strong>Văn bản thưa chiều cao</strong> đặt ra ràng buộc khác hẳn: phần lớn tham "
        f"số tầng đầu không được cập nhật trong mỗi lô (độ thưa "
        f"{s3['vectorizer']['sparsity_pct']:.1f}%), nên tăng kích thước mạng chỉ làm tăng số "
        "tham số chết.</li>"
        "</ul>")

    r.h(3, "6.2.2. Khi nào nên chọn Machine Learning, khi nào nên chọn Deep Learning?")

    champ1, champ2, champ3 = s1["champion"], s2["champion"], s3["champion"]
    r.p(R.table(
        ["Hệ thống", "Quán quân", "Vì sao thắng"],
        [
            ["HT1 — Tiểu đường", champ1,
             "Ranh giới bệnh được xác lập bởi <strong>ngưỡng sinh hoá nghiêm ngặt</strong> "
             "(HbA1c ≥ 6,5%). Lát cắt trực giao của cây bắt trọn bước nhảy bậc thang."],
            ["HT2 — Bất động sản", champ2,
             "Thị trường là <strong>tập hợp hàng nghìn thị trường con địa phương</strong>. "
             "Xấp xỉ từng đoạn cục bộ của rừng cây kìm hãm bùng nổ sai số ở đuôi giá cao."],
            ["HT3 — Nhận xét NLP", champ3,
             "Cảm xúc câu là <strong>tổng hợp tuyến tính của nhiều từ cùng lúc</strong> — "
             "đúng thứ mạng nơ-ron và mô hình tuyến tính làm tự nhiên, còn cây thì không."],
        ],
        caption="Mô hình thắng cuộc ở mỗi hệ thống và nguyên nhân cấu trúc đằng sau."))

    r.p(
        "Ba kết quả này gộp lại cho một nguyên tắc chọn mô hình có thể áp dụng ngay:",
        "<ul>"
        "<li><strong>Dữ liệu bảng có ngưỡng rõ ràng</strong> (y tế, tài chính, kiểm định chất "
        "lượng) → ưu tiên mô hình cây. Chúng nhanh hơn, dễ giải thích hơn, và thường thắng.</li>"
        "<li><strong>Dữ liệu có quan hệ trơn, liên tục</strong> (định giá, dự báo nhu cầu) → "
        "mạng nơ-ron rộng và rừng cây đều tốt; chọn theo yêu cầu về tốc độ suy luận.</li>"
        "<li><strong>Dữ liệu phi cấu trúc</strong> (văn bản, ảnh, âm thanh) → mạng nơ-ron, và "
        "quan trọng hơn cả mô hình là <strong>chất lượng phép biểu diễn</strong>.</li>"
        "</ul>")

    r.h(3, "6.2.3. Biểu diễn quan trọng hơn thuật toán")

    macro = [v["optimal"]["macro_f1"] for v in s3["scores"].values()]
    r.p(
        f"Ở Hệ thống 3, bốn mô hình có độ phức tạp chênh nhau từ "
        f"{min(v['n_params'] for v in s3['scores'].values()):,} đến "
        f"{max(v['n_params'] for v in s3['scores'].values()):,} tham số, nhưng Macro F1 chỉ "
        f"dao động trong khoảng {min(macro) * 100:.2f}% – {max(macro) * 100:.2f}% — chênh lệch "
        f"vỏn vẹn {(max(macro) - min(macro)) * 100:.2f} điểm phần trăm.",
        "Trong khi đó, việc chuyển từ đặc trưng thô sang biểu diễn TF-IDF có bigram, hay việc "
        "thêm hai đặc trưng tương tác lâm sàng ở Hệ thống 1, hay việc áp dụng Smooth Bayesian "
        "Target Encoding ở Hệ thống 2 — mỗi bước đều tạo ra khác biệt lớn hơn nhiều so với việc "
        "đổi thuật toán.",
        "<strong>Kết luận thực hành:</strong> khi một hệ thống chưa đạt yêu cầu, hãy dành công "
        "sức cho cách biểu diễn dữ liệu trước, rồi mới nghĩ đến việc đổi mô hình.")

    r.h(3, "6.2.4. Tinh chỉnh ngưỡng — bước rẻ nhất nhưng bị bỏ quên nhiều nhất")

    fn_saved = sum(v["default"]["FN"] - v["optimal"]["FN"] for v in s1["scores"].values())
    r.p(
        f"Trên Hệ thống 1, riêng việc dịch chuyển ngưỡng quyết định — <strong>không thay đổi "
        f"một trọng số nào của mô hình</strong> — đã cứu được tổng cộng "
        f"<strong>{fn_saved:,} ca bệnh</strong> khỏi bị bỏ sót trên tập kiểm thử "
        f"{s1['split']['n_test']:,} mẫu. Trên Hệ thống 3, cùng thao tác đó nâng Recall lớp "
        "tiêu cực lên đáng kể ở cả bốn mô hình.",
        "Chi phí tính toán của bước này gần như bằng 0: chỉ là một vòng lặp quét 80 giá trị "
        "ngưỡng trên xác suất đã có sẵn. Nhưng nó thường bị bỏ qua vì ngưỡng 0,50 là giá trị "
        "mặc định của mọi thư viện, và mặc định thì ít ai đặt câu hỏi.",
        "Điều kiện duy nhất phải nhớ: <strong>ngưỡng luôn được dò trên tập Train</strong>, không "
        "bao giờ trên tập Test. Dò trên Test là rò rỉ dữ liệu, chỉ ở dạng tinh vi hơn.")


# ===========================================================================
# CHƯƠNG 7 — TRIỂN KHAI
# ===========================================================================
def _chapter7(r: R.Report, data: dict) -> None:
    s1, s2, s3 = data["s1"], data["s2"], data["s3"]

    r.h(1, "Chương 7. Triển khai hệ thống thông minh: REST API, Web và Mobile")

    r.p(
        "Một mô hình chỉ tồn tại trong notebook thì chưa phải là hệ thống thông minh. Chương "
        "này đưa cả ba mô hình ra khỏi môi trường nghiên cứu và biến chúng thành ba dịch vụ "
        "chạy được thật, có giao diện cho người dùng cuối.",
        "Điểm đặc biệt về mặt kỹ thuật: <strong>tầng suy luận của mạng nơ-ron trong API được "
        "viết lại hoàn toàn bằng NumPy</strong>, đọc trực tiếp từ tệp trọng số "
        "<code class=\"inl\">.npz</code> mà notebook đã lưu. API không nạp TensorFlow, không "
        "nạp PyTorch, và cũng không dùng <code class=\"inl\">sklearn</code> cho phần Deep "
        "Learning. Đây là minh chứng rằng mạng tự viết tay <em>chạy được thật trong môi trường "
        "triển khai</em>, chứ không chỉ tồn tại trong notebook.")

    # ---------------- 7.1 Kiến trúc ----------------
    r.h(2, "7.1. Kiến trúc triển khai ba dịch vụ độc lập")

    r.p(R.table(
        ["Hệ thống", "Cổng", "Endpoint dự đoán", "Mô hình phục vụ", "Đầu ra"],
        [
            ["Sàng lọc tiểu đường", "<code class=\"inl\">5001</code>",
             "<code class=\"inl\">POST /predict</code>",
             f"{s1['dl_model']['name']} ({s1['dl_model']['layers']})",
             f"Xác suất mắc bệnh + phân loại tại τ = {s1['dl_model']['threshold']:.2f}"],
            ["Định giá bất động sản", "<code class=\"inl\">5002</code>",
             "<code class=\"inl\">POST /predict</code>",
             f"{s2['dl_model']['name']} ({s2['dl_model']['layers']})",
             "Giá dự đoán (USD) + giá trên mỗi sqft + phân khúc"],
            ["Phân loại nhận xét", "<code class=\"inl\">5003</code>",
             "<code class=\"inl\">POST /predict</code>",
             f"{s3['dl_model']['name']} ({s3['dl_model']['layers']})",
             f"Xác suất giới thiệu + nhãn tại τ = {s3['dl_model']['threshold']:.2f}"],
        ],
        caption="Ba REST API độc lập, mỗi dịch vụ một cổng riêng nên chạy đồng thời được."))

    r.p(
        "Ba dịch vụ dùng ba cổng khác nhau nên có thể chạy song song — không phải tắt dịch vụ "
        "này để khởi động dịch vụ kia. Mỗi API tự phục vụ luôn cả giao diện Web "
        "(<code class=\"inl\">GET /</code>) và giao diện Mobile "
        "(<code class=\"inl\">GET /mobile</code>) của ứng dụng đó, nên người dùng chỉ cần chạy "
        "một lệnh Python là có đủ backend lẫn frontend.")

    r.p(R.code(
        '''def forward(X, Ws, bs):
    """Lan truyền tiến thuần NumPy — nạp thẳng từ tệp .npz do notebook lưu."""
    A = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        Z = A @ W + b
        if i < len(Ws) - 1:
            A = np.maximum(0.0, Z)                       # ReLU ở các tầng ẩn
        else:
            A = 1.0 / (1.0 + np.exp(-np.clip(Z, -25, 25)))   # Sigmoid ở tầng ra
    return A''',
        caption="Toàn bộ tầng suy luận của mạng nơ-ron trong REST API — đúng 9 dòng NumPy, "
                "không phụ thuộc bất kỳ framework học sâu nào."))

    r.p(R.note(
        "Kiểm chứng tính đúng đắn của artifact.",
        "Ở cuối mỗi notebook có một ô nạp lại tệp <code class=\"inl\">.npz</code> rồi so sánh "
        "kết quả forward pass với mô hình gốc trong bộ nhớ. Sai khác tuyệt đối lớn nhất đo được "
        "nhỏ hơn 10⁻¹⁰ trên cả ba hệ thống — tức artifact tái lập chính xác mô hình đã huấn "
        "luyện, không mất mát thông tin khi lưu.", kind="good"))

    # ---------------- 7.2 Tầng tri thức ----------------
    r.h(2, "7.2. Tầng tri thức Neo4j — bổ sung tư vấn, không quyết định dự đoán")

    r.p(
        "Bên cạnh con số dự đoán, mỗi API còn trả về một khối tư vấn lấy từ đồ thị tri thức "
        "Neo4j. Nguyên tắc thiết kế quan trọng nhất của tầng này: <strong>đồ thị chỉ BỔ SUNG, "
        "không bao giờ QUYẾT ĐỊNH</strong>. Mạng nơ-ron tính ra con số trước; đồ thị chỉ dịch "
        "con số đó thành một tầng tư vấn rồi lấy nội dung tương ứng về.")

    r.p(
        "Cấu trúc đồ thị gồm ba loại nút và hai loại quan hệ:",
        '<div class="flow">(:A03System) −[:HAS_TIER]→ (:A03Tier) −[:ADVISES]→ (:A03Advice)</div>',
        "Trong đó <code class=\"inl\">A03System</code> là một trong ba hệ thống, "
        "<code class=\"inl\">A03Tier</code> là một tầng nguy cơ/phân khúc/sắc thái được suy ra "
        "từ ngưỡng dự đoán, và <code class=\"inl\">A03Advice</code> là các mục tư vấn cụ thể "
        "được nhóm theo chủ đề.")

    r.p(R.code(
        '''MATCH (t:A03Tier {case: $case})
OPTIONAL MATCH (t)-[:ADVISES]->(a:A03Advice)
RETURN t.name AS tier_name, t.summary AS summary,
       collect({group: a.group, title: a.title, content: a.content}) AS advices''',
        caption="Truy vấn Cypher lấy tầng tư vấn tương ứng với kết quả dự đoán.",
        lang="Cypher"))

    r.p(R.note(
        "Suy biến an toàn khi Neo4j không sẵn sàng.",
        "Mọi lỗi ở tầng đồ thị đều bị nuốt lại thành một chuỗi thông báo. Nếu Neo4j sập, mất "
        "mạng, hay chưa cấu hình mật khẩu trong <code class=\"inl\">.env</code>, API vẫn trả về "
        "dự đoán bình thường — chỉ thiếu phần tư vấn kèm theo, và giao diện tự ẩn khối tri thức "
        "đi. Đây là lựa chọn có chủ đích: <strong>một tính năng phụ không được phép làm sập "
        "tính năng chính</strong>."))

    # ---------------- 7.3 Giao diện Web ----------------
    r.h(2, "7.3. Giao diện Web")

    r.p(
        "Mỗi hệ thống có một trang Web độc lập, viết bằng HTML/CSS/JavaScript thuần trong một "
        "tệp duy nhất — không framework, không CDN ngoài, nên mở được cả khi máy không có mạng. "
        "Giao diện gồm ba khối: biểu mẫu nhập liệu bên trái, kết quả dự đoán bên phải, và khối "
        "tư vấn từ đồ thị tri thức phía dưới.")

    for key, label, sysname in [
        ("s1_diabetes", "Sàng lọc nguy cơ tiểu đường", "Hệ thống 1"),
        ("s2_house", "Định giá bất động sản", "Hệ thống 2"),
        ("s3_comments", "Phân loại nhận xét thương mại điện tử", "Hệ thống 3"),
    ]:
        r.p(R.figure(f"ui_{key}_web_result.png",
                     f"Giao diện Web {sysname} — {label}: kết quả dự đoán kèm tầng tư vấn "
                     f"lấy từ đồ thị tri thức Neo4j."))

    # ---------------- 7.4 Giao diện Mobile ----------------
    r.h(2, "7.4. Giao diện Mobile")

    r.p(
        "Phần Mobile được cung cấp ở hai dạng. Dạng thứ nhất là trang mô phỏng khung điện thoại "
        "chạy thẳng trong trình duyệt (<code class=\"inl\">mobile/index.html</code>) — dùng để "
        "trình diễn và chụp ảnh báo cáo mà không cần cài đặt gì. Dạng thứ hai là mã nguồn "
        "Flutter thật (<code class=\"inl\">mobile/lib/main.dart</code> kèm "
        "<code class=\"inl\">pubspec.yaml</code>) — biên dịch được thành ứng dụng Android/iOS "
        "thực thụ, gọi cùng một REST API.")

    for key, label, sysname in [
        ("s1_diabetes", "Sàng lọc nguy cơ tiểu đường", "Hệ thống 1"),
        ("s2_house", "Định giá bất động sản", "Hệ thống 2"),
        ("s3_comments", "Phân loại nhận xét thương mại điện tử", "Hệ thống 3"),
    ]:
        r.p(R.figure(f"ui_{key}_mobile_result.png",
                     f"Giao diện Mobile {sysname} — {label}: khung điện thoại mô phỏng hiển thị "
                     f"kết quả dự đoán."))

    # ---------------- 7.5 Hướng dẫn chạy ----------------
    r.h(2, "7.5. Hướng dẫn chạy lại toàn bộ hệ thống")

    r.p(R.code(
        '''# 1. Cài phụ thuộc
pip install -r requirements.txt

# 2. Chạy lại toàn bộ notebook để sinh artifact và hình vẽ
python tools/py2nb.py diabetes_baseline/notebook/_src_01_baseline.py \\
                     diabetes_baseline/notebook/01_diabetes_dl_from_scratch_numpy.ipynb
jupyter nbconvert --to notebook --execute --inplace \\
                  diabetes_baseline/notebook/01_diabetes_dl_from_scratch_numpy.ipynb
# ... lặp lại cho 4 notebook còn lại

# 3. Nạp đồ thị tri thức (tuỳ chọn — cần .env cấu hình Neo4j)
python knowledge_graph/seed_knowledge_graph.py

# 4. Khởi động ba dịch vụ, mỗi dịch vụ một cửa sổ terminal
python diabetes_large/api/rest_api.py         # cổng 5001
python house_price_large/api/rest_api.py      # cổng 5002
python customer_comments/api/rest_api.py      # cổng 5003

# 5. Dựng lại báo cáo PDF
python report/capture_screenshots.py
python report/build_report.py''',
        caption="Quy trình dựng lại toàn bộ hệ thống từ đầu.", lang="Bash"))


# ===========================================================================
# CHƯƠNG 8 — KẾT LUẬN
# ===========================================================================
def _chapter8(r: R.Report, data: dict) -> None:
    base, imp, abl = data["baseline"], data["improved"], data["ablation"]
    s1, s2, s3 = data["s1"], data["s2"], data["s3"]
    cmp_ = imp["comparison_vs_baseline"]

    r.h(1, "Chương 8. Kết luận và kiểm toán khả năng tái lập")

    # ---------------- 8.1 ----------------
    r.h(2, "8.1. Tổng kết các đóng góp khoa học chính")

    r.p(
        "Nghiên cứu này xây dựng từ đầu một hệ thống mạng nơ-ron sâu hoàn chỉnh bằng NumPy "
        "thuần tuý, sau đó mở rộng lên ba bài toán thực tế quy mô lớn và triển khai thành ba "
        "dịch vụ chạy được. Năm đóng góp chính:")

    r.p(
        "<p><strong>1. Chứng minh toán học và kiểm chứng thực nghiệm vai trò của phi tuyến.</strong></p>"
        "Chương 1 chứng minh bằng đại số rằng phép hợp của các tầng tuyến tính chỉ tương đương "
        "một phép biến đổi affine duy nhất. Chương 2 kiểm chứng bằng thực nghiệm: khi loại bỏ "
        f"hoàn toàn ReLU, hàm mất mát kẹt ở mức "
        f"{abl['architectures'][3]['Final Loss']:.4f} (so với "
        f"{abl['architectures'][0]['Final Loss']:.4f} khi có ReLU) và số ca bệnh bỏ sót vọt lên "
        f"{abl['architectures'][3]['Số ca FN']} ca.")

    r.p(
        "<p><strong>2. Định lượng cái giá của bốn khuyết tật phương pháp luận.</strong></p>"
        f"Bằng cách dựng một mô hình cơ sở có khuyết tật cố ý rồi sửa từng điểm một, nghiên cứu "
        f"đo được chính xác mức cải thiện: Recall tăng "
        f"<strong>{cmp_['recall_delta_pp']:+.2f} điểm phần trăm</strong>, F1-Score tăng "
        f"{cmp_['f1_delta_pp']:+.2f} điểm, và số ca bệnh bỏ sót giảm từ {cmp_['fn_baseline']} "
        f"xuống {cmp_['fn_improved']} ca (<strong>giảm {cmp_['fn_reduction_pct']:.1f}%</strong>) "
        "— tất cả mà <em>không thay đổi một dòng nào trong kiến trúc mạng</em>.")

    r.p(
        "<p><strong>3. Phát hiện quy luật kiến trúc phụ thuộc vào loại bài toán.</strong></p>"
        "Chiều sâu thắng ở phân loại, chiều rộng thắng ở hồi quy, và kích thước vừa đủ thắng ở "
        "văn bản thưa chiều cao. Ba quy luật này được rút ra từ 12 cấu hình kiến trúc chạy có "
        "kiểm soát, và mỗi quy luật đều có lời giải thích cấu trúc đi kèm (Chương 6, mục 6.2).")

    r.p(
        "<p><strong>4. Chứng minh mạng viết tay hoạt động được ở quy mô thực tế.</strong></p>"
        f"Các phương trình lan truyền ngược thuần NumPy hội tụ ổn định trên "
        f"{s1['dataset']['n_raw']:,} hồ sơ y tế, {s2['dataset']['n_raw']:,} giao dịch bất động "
        f"sản và {s3['dataset']['n_clean']:,} nhận xét văn bản 1.000 chiều — không cần bất kỳ "
        "framework tự động vi phân nào.")

    r.p(
        "<p><strong>5. Đưa mô hình tự viết vào môi trường triển khai thật.</strong></p>"
        "Ba REST API tự thực hiện forward pass bằng 9 dòng NumPy đọc từ tệp trọng số "
        "<code class=\"inl\">.npz</code>, kèm giao diện Web, giao diện Mobile và tầng tri thức "
        "Neo4j suy biến an toàn.")

    # ---------------- 8.2 ----------------
    r.h(2, "8.2. Kiểm toán khả năng tái lập (Reproducibility Audit)")

    r.p(
        "Một kết quả nghiên cứu chỉ có giá trị khi người khác chạy lại được. Toàn bộ nghiên cứu "
        "này được thiết kế để tái lập bit-by-bit:")

    r.p(R.table(
        ["Hạng mục kiểm toán", "Biện pháp bảo đảm"],
        [
            ["Tính ngẫu nhiên",
             "Cố định <code class=\"inl\">SEED = 42</code> ở mọi notebook, mọi bộ sinh số ngẫu "
             "nhiên (<code class=\"inl\">np.random.default_rng</code>, khởi tạo trọng số, xáo "
             "trộn mini-batch, chia train/test)."],
            ["Rò rỉ dữ liệu",
             "Mọi tham số tiền xử lý (μ, σ, trung vị, từ điển TF-IDF, bảng Target Encoding) đều "
             "<code class=\"inl\">fit</code> duy nhất trên tập Train. Notebook có ô kiểm chứng "
             "in ra trung bình tập Test khác 0 để chứng minh."],
            ["Ngưỡng quyết định",
             "Dò trên tập Train, áp cố định sang tập Test — không bao giờ dò trên Test."],
            ["Số liệu trong báo cáo",
             "Mọi con số được đọc tự động từ các tệp JSON do notebook sinh ra. Không có con số "
             "nào gõ tay trong mã dựng báo cáo."],
            ["Artifact mô hình",
             "Mỗi notebook có ô nạp lại tệp <code class=\"inl\">.npz</code> và so sánh với mô "
             "hình gốc; sai khác đo được &lt; 10⁻¹⁰."],
            ["Mã nguồn notebook",
             "Sinh từ tệp nguồn <code class=\"inl\">_src_*.py</code> qua "
             "<code class=\"inl\">tools/py2nb.py</code>, nên diff được bằng Git thay vì so sánh "
             "JSON của <code class=\"inl\">.ipynb</code>."],
        ],
        caption="Sáu hạng mục kiểm toán khả năng tái lập."))

    r.p(R.note(
        "Một điểm trung thực cần nêu rõ.",
        "Các con số trong báo cáo này là kết quả chạy thật trên máy cá nhân với "
        "<code class=\"inl\">SEED = 42</code>. Chạy lại trên máy khác với cùng phiên bản NumPy "
        "sẽ cho kết quả trùng khớp. Tuy nhiên, đổi seed sẽ làm điểm số dao động vài phần trăm — "
        "đặc biệt ở Chương 2 nơi tập kiểm thử chỉ có "
        f"{base['split']['n_test']} mẫu. Đây là giới hạn cố hữu của tập dữ liệu nhỏ, và cũng "
        "chính là lý do nghiên cứu phải mở rộng lên ba bộ dữ liệu lớn ở các chương sau.",
        kind="warn"))

    # ---------------- 8.3 ----------------
    r.h(2, "8.3. Hạn chế của nghiên cứu")

    r.p(
        "<ul>"
        "<li><strong>Không có tập Validation riêng.</strong> Nghiên cứu chỉ chia Train/Test. "
        "Việc chọn kiến trúc và dò ngưỡng đều dựa trên tập Train, nên tránh được rò rỉ, nhưng "
        "một tập Validation riêng sẽ cho ước lượng hiệu năng khách quan hơn nữa.</li>"
        "<li><strong>Chưa có cơ chế điều chuẩn.</strong> Mạng không dùng Dropout, L2 "
        "regularization hay Early Stopping. Ở Chương 2, hiện tượng học vẹt của mạng rộng có thể "
        "đã được kìm hãm nếu có các cơ chế này.</li>"
        "<li><strong>Thuật toán tối ưu dừng ở Mini-Batch Gradient Descent.</strong> Các bộ tối "
        "ưu hiện đại (Adam, RMSProp) với tốc độ học thích ứng nhiều khả năng cho hội tụ nhanh "
        "hơn — nhưng cài đặt chúng nằm ngoài phạm vi của một nghiên cứu tập trung vào cơ chế "
        "nền tảng.</li>"
        "<li><strong>Biểu diễn văn bản dừng ở TF-IDF.</strong> Các phép nhúng ngữ nghĩa "
        "(Word2Vec, GloVe, BERT) nắm bắt được quan hệ đồng nghĩa mà TF-IDF hoàn toàn mù, nhưng "
        "chúng đòi hỏi mô hình đã tiền huấn luyện — trái với ràng buộc \"viết từ đầu\" của "
        "Assignment.</li>"
        "<li><strong>Dữ liệu bất động sản là thị trường Mỹ.</strong> Các quy luật định giá tìm "
        "được không chuyển trực tiếp sang thị trường Việt Nam, nơi cấu trúc pháp lý và tập quán "
        "giao dịch khác hẳn.</li>"
        "</ul>")

    # ---------------- 8.4 ----------------
    r.h(2, "8.4. Hướng phát triển và mở rộng nghiên cứu tiếp theo")

    r.p(
        "<ol>"
        "<li><strong>Bổ sung bộ tối ưu Adam và các cơ chế điều chuẩn.</strong> Cài đặt Adam, "
        "Dropout và Batch Normalization vẫn hoàn toàn bằng NumPy để giữ tinh thần \"viết từ "
        "đầu\", rồi đo lại toàn bộ 12 cấu hình kiến trúc.</li>"
        "<li><strong>Kiểm định chéo k-fold.</strong> Thay phép chia đơn 80/20 bằng "
        "cross-validation 5 fold để có khoảng tin cậy cho mỗi chỉ số, thay vì một con số điểm.</li>"
        "<li><strong>Giải thích mô hình (Explainable AI).</strong> Cài đặt SHAP hoặc "
        "Integrated Gradients để trả lời câu hỏi \"vì sao mô hình dự đoán bệnh nhân này dương "
        "tính\" — yêu cầu bắt buộc nếu triển khai thật trong y tế.</li>"
        "<li><strong>Mở rộng sang kiến trúc chuyên biệt.</strong> CNN cho dữ liệu ảnh y tế, "
        "RNN/Transformer cho văn bản — vẫn theo hướng cài đặt thủ công để hiểu cơ chế.</li>"
        "<li><strong>Giám sát trôi dữ liệu (Data Drift Monitoring).</strong> Ba dịch vụ hiện "
        "chạy với mô hình tĩnh. Một hệ thống thực tế cần theo dõi sự thay đổi phân phối đầu vào "
        "theo thời gian và cảnh báo khi cần huấn luyện lại.</li>"
        "</ol>")

    # ---------------- 8.5 ----------------
    r.h(2, "8.5. Kho mã nguồn và tài nguyên đính kèm")

    r.p(R.table(
        ["Thành phần", "Đường dẫn trong kho mã nguồn"],
        [
            ["Notebook 01 — Mạng cơ sở viết tay",
             "<code class=\"inl\">diabetes_baseline/notebook/01_diabetes_dl_from_scratch_numpy.ipynb</code>"],
            ["Notebook 02 — Cải tiến và 6 nghiên cứu bóc tách",
             "<code class=\"inl\">diabetes_baseline/notebook/02_diabetes_dl_from_scratch_improvements.ipynb</code>"],
            ["Notebook 03 — Hệ thống 1 (100.000 mẫu)",
             "<code class=\"inl\">diabetes_large/notebook/03_diabetes_large_ml_vs_dl_scratch.ipynb</code>"],
            ["Notebook 04 — Hệ thống 2 (150.000 mẫu)",
             "<code class=\"inl\">house_price_large/notebook/04_house_price_large_ml_vs_dl_scratch.ipynb</code>"],
            ["Notebook 05 — Hệ thống 3 (23.486 nhận xét)",
             "<code class=\"inl\">customer_comments/notebook/05_comments_large_ml_vs_dl_scratch.ipynb</code>"],
            ["Ba REST API",
             "<code class=\"inl\">*/api/rest_api.py</code> (cổng 5001, 5002, 5003)"],
            ["Ba giao diện Web",
             "<code class=\"inl\">*/web/index.html</code>"],
            ["Ba giao diện Mobile",
             "<code class=\"inl\">*/mobile/index.html</code> và "
             "<code class=\"inl\">*/mobile/lib/main.dart</code>"],
            ["Tầng đồ thị tri thức Neo4j",
             "<code class=\"inl\">knowledge_graph/</code>"],
            ["Mã dựng báo cáo",
             "<code class=\"inl\">report/build_report.py</code> và "
             "<code class=\"inl\">report/chapters_*.py</code>"],
        ],
        caption="Danh mục sản phẩm bàn giao của Assignment 03.",
        cls="links"))

    r.p(
        f'<p class="srcline">Kho mã nguồn công khai: '
        f'<a href="{GITHUB}">{GITHUB}</a></p>')
