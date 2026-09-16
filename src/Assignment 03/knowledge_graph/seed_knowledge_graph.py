"""Nạp đồ thị tri thức Assignment 03 vào Neo4j Aura.

Đồ thị này KHÔNG tham gia vào việc dự đoán. Mô hình nơ-ron tính ra xác suất
(hoặc giá trị) trước; sau đó API dùng con số ấy để chọn một tầng (tier) trong đồ
thị, rồi lấy về nội dung tư vấn gắn với tầng đó. Nói cách khác: mô hình quyết
định "bao nhiêu", đồ thị trả lời "vậy thì nên làm gì".

Toàn bộ node của Assignment 03 mang nhãn có tiền tố `A03` nên không đụng chạm
tới dữ liệu của các assignment trước trên cùng một instance.

Chạy:  python knowledge_graph/seed_knowledge_graph.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Chưa cài package neo4j.  pip install neo4j")
    sys.exit(1)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ---------------------------------------------------------------------------
# Nội dung tri thức — ba miền nghiệp vụ, mỗi miền ba tầng
# ---------------------------------------------------------------------------

DIABETES_TIERS = [
    {
        "case": "dia_high", "name": "Nguy cơ cao",
        "range": "Xác suất mô hình ≥ 0.60",
        "advice": [
            ("Hành động", "Đặt lịch khám nội tiết trong vòng 7 ngày",
             "Xét nghiệm HbA1c và đường huyết lúc đói để xác nhận chẩn đoán"),
            ("Hành động", "Ghi nhật ký đường huyết hai lần mỗi ngày",
             "Đo lúc đói buổi sáng và sau bữa tối hai giờ"),
            ("Thiết bị", "Máy đo đường huyết cá nhân",
             "Theo dõi tại nhà giữa các lần tái khám · khoảng 600.000 – 1.200.000 đ"),
            ("Dinh dưỡng", "Cắt giảm tinh bột tinh chế và đồ uống có đường",
             "Thay cơm trắng bằng gạo lứt, yến mạch; bỏ hoàn toàn nước ngọt"),
            ("Vận động", "Đi bộ nhanh 30 phút mỗi ngày, 5 ngày một tuần",
             "Vận động aerobic đều đặn cải thiện độ nhạy insulin rõ rệt"),
        ],
    },
    {
        "case": "dia_moderate", "name": "Nguy cơ trung bình",
        "range": "Xác suất mô hình 0.30 – 0.60",
        "advice": [
            ("Hành động", "Xét nghiệm HbA1c định kỳ 6 tháng một lần",
             "Phát hiện sớm giai đoạn tiền đái tháo đường"),
            ("Hành động", "Kiểm soát cân nặng về ngưỡng BMI 18.5 – 24.9",
             "Giảm 5–7% cân nặng đã hạ đáng kể nguy cơ tiến triển thành bệnh"),
            ("Thiết bị", "Cân điện tử đo thành phần cơ thể",
             "Theo dõi tỷ lệ mỡ và khối cơ · khoảng 400.000 – 900.000 đ"),
            ("Dinh dưỡng", "Tăng chất xơ lên 25–30 gam mỗi ngày",
             "Rau xanh, đậu, ngũ cốc nguyên hạt làm chậm hấp thu đường"),
        ],
    },
    {
        "case": "dia_low", "name": "Nguy cơ thấp",
        "range": "Xác suất mô hình < 0.30",
        "advice": [
            ("Hành động", "Khám sức khoẻ tổng quát mỗi năm một lần",
             "Duy trì theo dõi định kỳ ngay cả khi chưa có dấu hiệu bất thường"),
            ("Dinh dưỡng", "Duy trì chế độ ăn cân bằng hiện tại",
             "Hạn chế đường tinh luyện, ưu tiên thực phẩm tươi"),
            ("Vận động", "Duy trì tối thiểu 150 phút vận động mỗi tuần",
             "Theo khuyến cáo của Tổ chức Y tế Thế giới"),
        ],
    },
]

HOUSE_TIERS = [
    {
        "case": "house_luxury", "name": "Phân khúc Cao cấp & Biệt thự",
        "range": "Giá dự đoán > 800.000 USD",
        "advice": [
            ("Định giá", "Sai số mô hình ở phân khúc này cao nhất",
             "Giá phụ thuộc nặng vào yếu tố ngoại sinh: nội thất, tầm nhìn, uy tín kiến trúc sư"),
            ("Khuyến nghị", "Cần thẩm định độc lập tại chỗ trước khi giao dịch",
             "Không nên dùng giá mô hình làm căn cứ đàm phán duy nhất"),
            ("Yếu tố giá", "Chất lượng hoàn thiện và tầm nhìn cảnh quan",
             "Hai yếu tố không xuất hiện trong dữ liệu thuộc tính thông thường"),
        ],
    },
    {
        "case": "house_mid", "name": "Phân khúc Trung cấp",
        "range": "Giá dự đoán 300.000 – 800.000 USD",
        "advice": [
            ("Định giá", "Sai số tương đối ở mức trung bình",
             "Mô hình bắt tốt quan hệ giữa diện tích, số phòng và vị trí bưu chính"),
            ("Khuyến nghị", "Đối chiếu với 3–5 giao dịch tương đương cùng mã bưu chính",
             "Chỉ số relative_sqft cho biết nhà rộng hay hẹp so với mặt bằng khu vực"),
            ("Yếu tố giá", "Tỷ lệ phòng tắm trên phòng ngủ ảnh hưởng đáng kể",
             "Đặc trưng bath_bed_ratio nằm trong nhóm có trọng số cao"),
        ],
    },
    {
        "case": "house_standard", "name": "Phân khúc Phổ thông",
        "range": "Giá dự đoán < 300.000 USD",
        "advice": [
            ("Định giá", "Đây là phân khúc mô hình định giá chuẩn xác nhất",
             "Giá gắn chặt với công năng vật lý: diện tích, số phòng, vị trí"),
            ("Khuyến nghị", "Có thể dùng giá mô hình làm mốc tham chiếu đàm phán",
             "Sai số tương đối thấp nhất trong ba phân khúc"),
            ("Yếu tố giá", "Diện tích sàn là biến giải thích mạnh nhất",
             "Đặc trưng log_size chiếm trọng số lớn trong mô hình"),
        ],
    },
]

COMMENT_TIERS = [
    {
        "case": "cmt_negative", "name": "Nhận xét Tiêu cực — cần xử lý ngay",
        "range": "Xác suất khuyến nghị < ngưỡng tối ưu",
        "advice": [
            ("Xử lý", "Chuyển ticket sang đội chăm sóc khách hàng trong 2 giờ",
             "Bỏ sót một khách bức xúc có thể lan truyền dư luận tiêu cực"),
            ("Xử lý", "Liên hệ trực tiếp đề nghị đổi trả hoặc hoàn tiền",
             "Phản hồi chủ động làm giảm đáng kể nguy cơ đánh giá công khai 1 sao"),
            ("Phân tích", "Đối chiếu với cụm từ khoá phàn nàn phổ biến",
             "Bốn cụm chính: chất liệu kém, sai kích cỡ, màu khác ảnh, giao hàng chậm"),
            ("Vận hành", "Gắn cờ mã sản phẩm để rà soát chất lượng lô hàng",
             "Nhiều phàn nàn cùng một mã sản phẩm là dấu hiệu lỗi hệ thống"),
        ],
    },
    {
        "case": "cmt_neutral", "name": "Nhận xét Trung tính — theo dõi",
        "range": "Xác suất khuyến nghị quanh vùng ngưỡng",
        "advice": [
            ("Xử lý", "Đưa vào hàng đợi rà soát thủ công",
             "Mô hình chưa đủ tự tin, cần con người xác nhận"),
            ("Phân tích", "Trích xuất khía cạnh được nhắc tới trong nhận xét",
             "Thường là khen một mặt và chê một mặt khác trong cùng đánh giá"),
        ],
    },
    {
        "case": "cmt_positive", "name": "Nhận xét Tích cực — khai thác",
        "range": "Xác suất khuyến nghị cao hơn hẳn ngưỡng",
        "advice": [
            ("Marketing", "Đề nghị khách hàng cho phép trích dẫn làm đánh giá nổi bật",
             "Nội dung do khách hàng tạo có sức thuyết phục cao hơn quảng cáo"),
            ("Vận hành", "Ghi nhận mã sản phẩm vào nhóm bán chạy",
             "Ưu tiên bổ sung tồn kho và mở rộng dải kích cỡ"),
            ("Phân tích", "Trích xuất từ khoá khen để tối ưu mô tả sản phẩm",
             "Dùng chính ngôn ngữ của khách hàng trong trang bán hàng"),
        ],
    },
]

SYSTEMS = [
    ("diabetes_large", "Hệ thống 1 — Sàng lọc nguy cơ tiểu đường", DIABETES_TIERS),
    ("house_price_large", "Hệ thống 2 — Định giá bất động sản", HOUSE_TIERS),
    ("customer_comments", "Hệ thống 3 — Phân loại nhận xét khách hàng", COMMENT_TIERS),
]


CLEAR = """
MATCH (n)
WHERE n:A03System OR n:A03Tier OR n:A03Advice
DETACH DELETE n
"""

CREATE_SYSTEM = "MERGE (s:A03System {key: $key}) SET s.name = $name"

CREATE_TIER = """
MATCH (s:A03System {key: $sys})
MERGE (t:A03Tier {case: $case})
SET t.name = $name, t.range = $range, t.system = $sys
MERGE (s)-[:HAS_TIER]->(t)
"""

CREATE_ADVICE = """
MATCH (t:A03Tier {case: $case})
MERGE (a:A03Advice {case: $case, title: $title})
SET a.group = $group, a.detail = $detail
MERGE (t)-[:ADVISES]->(a)
"""


def main() -> int:
    if not PASSWORD:
        print("Thiếu NEO4J_PASSWORD trong .env — bỏ qua bước nạp đồ thị.")
        return 1

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session(database=DATABASE) as session:
        session.run(CLEAR)
        print("Đã dọn sạch các node A03 cũ.")

        n_tier = n_adv = 0
        for key, name, tiers in SYSTEMS:
            session.run(CREATE_SYSTEM, key=key, name=name)
            for tier in tiers:
                session.run(CREATE_TIER, sys=key, case=tier["case"],
                            name=tier["name"], range=tier["range"])
                n_tier += 1
                for group, title, detail in tier["advice"]:
                    session.run(CREATE_ADVICE, case=tier["case"], group=group,
                                title=title, detail=detail)
                    n_adv += 1
            print(f"  {key:<20} {len(tiers)} tầng")

        counts = session.run(
            "MATCH (n) WHERE n:A03System OR n:A03Tier OR n:A03Advice "
            "RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l"
        ).data()

    driver.close()
    print(f"\nĐã nạp {len(SYSTEMS)} hệ thống · {n_tier} tầng · {n_adv} mục tư vấn")
    for row in counts:
        print(f"  {row['l']:<12} {row['c']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
