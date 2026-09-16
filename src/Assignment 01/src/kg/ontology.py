"""Bản thể học (Ontology) và sinh bộ ba tri thức cho hai miền ứng dụng.

Mô hình Học máy chỉ trả về một con số — xác suất 83.1% hay giá 8.13 tỷ. Con số
đó không tự nó nói cho người dùng biết **phải làm gì tiếp theo**. Đồ thị Tri
thức lấp khoảng trống ấy: nó nối kết quả dự báo với tri thức miền đã chuẩn hoá
(mã bệnh ICD-10, thiết bị y tế, gói vay ngân hàng) để biến một dự báo thống kê
thành một khuyến nghị hành động được.

Đồ thị được biểu diễn dưới dạng KG = (V, E, R), trong đó V là tập thực thể,
E là tập cạnh và R là tập quan hệ ngữ nghĩa.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from . import catalog_pharmacy as pharm
from . import catalog_urban as urban


# ==========================================================================
#  CẤU TRÚC DỮ LIỆU CƠ SỞ
# ==========================================================================

@dataclass
class Node:
    """Một thực thể trong đồ thị."""

    id: str
    label: str          # Nhãn hiển thị cho người đọc
    type: str           # Loại thực thể — quyết định màu sắc khi vẽ
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Một quan hệ có hướng giữa hai thực thể."""

    source: str
    relation: str       # Tên quan hệ theo quy ước RDF, viết HOA_CÓ_GẠCH_DƯỚI
    target: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Đồ thị tri thức hoàn chỉnh KG = (V, E, R)."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> Node:
        """Thêm thực thể, bỏ qua nếu định danh đã tồn tại (bảo đảm tính duy nhất)."""
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)
        return node

    def add_edge(self, edge: Edge) -> Edge:
        self.edges.append(edge)
        return edge

    def to_dict(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    def triplets(self) -> list[tuple[str, str, str]]:
        """Kết xuất đồ thị thành danh sách bộ ba (Chủ thể, Quan hệ, Đối tượng)."""
        names = {n.id: n.label for n in self.nodes}
        return [(names.get(e.source, e.source), e.relation, names.get(e.target, e.target))
                for e in self.edges]

    def stats(self) -> dict[str, int]:
        return {
            "Số thực thể (|V|)": len(self.nodes),
            "Số cạnh (|E|)": len(self.edges),
            "Số loại quan hệ (|R|)": len({e.relation for e in self.edges}),
            "Số loại thực thể": len({n.type for n in self.nodes}),
        }


# ==========================================================================
#  MIỀN 1 — ĐỒ THỊ TRI THỨC Y SINH
# ==========================================================================

# Phân tầng nguy cơ lâm sàng. Ngưỡng lấy từ thực nghiệm quét θ ở Chương 2:
# hạ ngưỡng xuống 0.25 làm tăng mạnh Recall, phù hợp mục tiêu sàng lọc.
RISK_TIERS = [
    {
        "id": "tier_high",
        "label": "Nhóm Nguy cơ Cao",
        "min_prob": 0.25,
        "icd10": "E11",
        "icd10_name": "Đái tháo đường Type 2",
        "actions": [
            "Khám chuyên khoa Nội tiết để xét nghiệm HbA1c tĩnh mạch",
            "Đo đường huyết đói và sau ăn 2 giờ mỗi ngày",
            "Tuyệt đối không tự dùng Metformin khi chưa có chỉ định bác sĩ",
        ],
    },
    {
        "id": "tier_moderate",
        "label": "Nhóm Tiền Đái tháo đường",
        "min_prob": 0.15,
        "icd10": "R73.0",
        "icd10_name": "Tăng glucose máu bất thường",
        "actions": [
            "Cắt giảm đường tinh luyện và tinh bột hấp thu nhanh (GI > 70)",
            "Vận động tối thiểu 150 phút mỗi tuần để giảm kháng insulin",
            "Kiểm tra lại đường huyết sau 3 tháng",
        ],
    },
    {
        "id": "tier_low",
        "label": "Nhóm Khoẻ mạnh",
        "min_prob": 0.0,
        "icd10": "Z13.1",
        "icd10_name": "Khám sàng lọc đái tháo đường",
        "actions": [
            "Duy trì chế độ dinh dưỡng cân bằng",
            "Giữ chỉ số BMI trong khoảng 18.5 đến 22.9",
            "Khám sức khoẻ định kỳ hàng năm",
        ],
    },
]


# Danh sách thiết bị và dinh dưỡng của mỗi tầng KHÔNG được khai ở đây mà suy ra
# từ danh mục bán lẻ (:mod:`src.kg.catalog_pharmacy`) — nơi duy nhất giữ mã SKU,
# khoảng giá và đường dẫn tra cứu. Khai lại tên mặt hàng ở hai chỗ là cách chắc
# chắn nhất để hai chỗ ấy lệch nhau sau vài lần sửa danh mục.
for _tier in RISK_TIERS:
    _goi = pharm.CARE_PACKAGES[_tier["id"]]
    _mat_hang = [pharm.san_pham(sku) for sku in _goi["skus"]]
    _tier["package_id"] = _goi["id"]
    _tier["package_title"] = _goi["title"]
    _tier["devices"] = [p["name"] for p in _mat_hang if p["group"] == "Device"]
    _tier["nutrition"] = [p["name"] for p in _mat_hang if p["group"] == "Nutrition"]
del _tier, _goi, _mat_hang


def classify_risk(probability: float) -> dict:
    """Ánh xạ xác suất dự báo sang tầng nguy cơ lâm sàng tương ứng."""
    for tier in RISK_TIERS:
        if probability >= tier["min_prob"]:
            return tier
    return RISK_TIERS[-1]


def build_medical_kg(patient: dict, probability: float,
                     model_name: str = "XGBoost") -> KnowledgeGraph:
    """Dựng đồ thị tri thức y sinh năm lớp cho một ca bệnh cụ thể.

    Lớp 1 — Hồ sơ sinh học cá thể: các chỉ số sinh tồn đo được.
    Lớp 2 — Suy luận AI và chuẩn hoá quốc tế: nối dự báo với mã bệnh ICD-10.
    Lớp 3 — Danh mục can thiệp: thiết bị, dinh dưỡng, hướng dẫn hành động.
    Lớp 4 — Gói chăm sóc sức khoẻ: gom danh mục thành một đơn vị có chi phí.
    Lớp 5 — Hạ tầng bán lẻ: nhà thuốc cung ứng và dịch vụ đi kèm.
    """
    kg = KnowledgeGraph()
    tier = classify_risk(probability)

    # ---- Lớp 1: hồ sơ sinh học ----
    kg.add_node(Node("patient", "Bệnh nhân", "Patient", {
        "Tuổi": patient.get("Age"),
        "Glucose": patient.get("Glucose"),
        "BMI": patient.get("BMI"),
    }))

    for key, unit in [("Glucose", "mg/dL"), ("BMI", "kg/m²"),
                      ("BloodPressure", "mmHg"), ("Insulin", "µU/mL")]:
        if patient.get(key) is None:
            continue
        node_id = f"bio_{key.lower()}"
        kg.add_node(Node(node_id, f"{key} = {patient[key]:g} {unit}", "Biomarker",
                         {"Giá trị": patient[key], "Đơn vị": unit}))
        kg.add_edge(Edge("patient", "HAS_BIOMARKER", node_id))

    # Đặc trưng tương tác do con người thiết kế, đúng tinh thần Học máy truyền thống.
    # Kiểm ``is not None`` chứ không kiểm tính đúng-sai: giá trị 0 là số đo thật
    # (form nhập liệu ở Phase 6 trả về 0 khi người dùng chưa động vào ô, và
    # prepare_diabetes(clean=False) giữ nguyên số 0 quy ước của tập Pima). Kiểm
    # bằng tính đúng-sai thì nút Glucose = 0 vẫn hiện mà đặc trưng tương tác
    # biến mất, không lỗi nào được ném — đồ thị tự mâu thuẫn với chính nó.
    if patient.get("Glucose") is not None and patient.get("BMI") is not None:
        risk_value = patient["Glucose"] * patient["BMI"]
        kg.add_node(Node("bio_interaction", f"Glucose × BMI = {risk_value:,.0f}",
                         "Biomarker", {"Loại": "Đặc trưng thiết kế"}))
        kg.add_edge(Edge("patient", "CALCULATED_RISK", "bio_interaction"))

    # ---- Lớp 2: suy luận AI và mã bệnh quốc tế ----
    kg.add_node(Node("model", f"Mô hình {model_name}", "Model",
                     {"Loại": "Học máy truyền thống"}))
    kg.add_edge(Edge("patient", "EVALUATED_BY", "model"))

    kg.add_node(Node("prediction", f"Xác suất rủi ro {probability:.1%}", "Prediction",
                     {"Xác suất": round(probability, 4), "Ngưỡng": 0.25}))
    kg.add_edge(Edge("model", "PRODUCES", "prediction"))

    kg.add_node(Node("icd10", f"ICD-10 {tier['icd10']}: {tier['icd10_name']}",
                     "Disease", {"Hệ mã": "ICD-10", "Mã": tier["icd10"]}))
    kg.add_edge(Edge("prediction", "MAPS_TO_DISEASE", "icd10"))

    kg.add_node(Node(tier["id"], tier["label"], "RiskTier",
                     {"Ngưỡng xác suất": tier["min_prob"]}))
    kg.add_edge(Edge("patient", "STRATIFIED_AS", tier["id"]))

    # ---- Lớp 3: danh mục can thiệp ----
    # Mỗi mặt hàng mang theo mã SKU, khoảng giá và đường dẫn tra cứu lấy từ danh
    # mục bán lẻ. Nhờ vậy một nút "Device" trong đồ thị không chỉ là một cái tên
    # mà là một món hàng mua được — đúng phần việc mà tầng Application ở
    # Hình 1.1 phải làm.
    goi = pharm.goi_cham_soc(tier["id"])

    for i, device in enumerate(tier["devices"]):
        node_id = f"device_{i}"
        kg.add_node(Node(node_id, device, "Device", _thuoc_tinh_mat_hang(device)))
        kg.add_edge(Edge("icd10", "REQUIRES_DEVICE", node_id))

    for i, item in enumerate(tier["nutrition"]):
        node_id = f"nutrition_{i}"
        kg.add_node(Node(node_id, item, "Nutrition", _thuoc_tinh_mat_hang(item)))
        kg.add_edge(Edge("icd10", "RECOMMENDS_NUTRITION", node_id))

    for i, action in enumerate(tier["actions"]):
        node_id = f"action_{i}"
        kg.add_node(Node(node_id, action, "Action", {"Nhóm": "Hướng dẫn lâm sàng"}))
        kg.add_edge(Edge(tier["id"], "ADVISES", node_id))

    # ---- Lớp 4: gói chăm sóc sức khoẻ ----
    # Gói là nút gom: nó biến một danh sách rời rạc thành một đơn vị có chi phí
    # tính được, nhờ đó câu hỏi "theo lời khuyên này thì tốn bao nhiêu" có câu
    # trả lời ngay trên đồ thị chứ không phải cộng tay ở tầng giao diện.
    chi_phi = goi["chi_phi"]
    kg.add_node(Node("package", goi["title"], "CarePackage", {
        "Mục tiêu": goi["muc_tieu"],
        "Số mặt hàng": len(goi["san_pham"]),
        "Chi phí ban đầu": pharm.dinh_dang_khoang(
            chi_phi["ban_dau_min"], chi_phi["ban_dau_max"]),
        "Chi phí duy trì hằng tháng": pharm.dinh_dang_khoang(
            chi_phi["hang_thang_min"], chi_phi["hang_thang_max"]),
        "Tần suất theo dõi": goi["tan_suat_theo_doi"],
        "Nguồn giá": pharm.GHI_CHU_NGUON,
    }))
    kg.add_edge(Edge(tier["id"], "RECOMMENDS_PACKAGE", "package"))
    for node in [n for n in kg.nodes if n.type in {"Device", "Nutrition"}]:
        kg.add_edge(Edge("package", "INCLUDES_PRODUCT", node.id))

    # ---- Lớp 5: hạ tầng bán lẻ và dịch vụ ----
    kg.add_node(Node("retailer", pharm.RETAILER["name"], "Retailer", {
        "Hotline": pharm.RETAILER["hotline"],
        "Website": pharm.RETAILER["website"],
        "Điểm bán": pharm.RETAILER["so_diem_ban"],
    }))
    kg.add_edge(Edge("package", "FULFILLED_BY", "retailer"))
    for node in [n for n in kg.nodes if n.type in {"Device", "Nutrition"}]:
        kg.add_edge(Edge(node.id, "SUPPLIED_BY", "retailer"))

    for i, dich_vu in enumerate(goi["dich_vu"]):
        node_id = f"service_{i}"
        kg.add_node(Node(node_id, dich_vu, "Service", {"Nhóm": "Dịch vụ nhà thuốc"}))
        kg.add_edge(Edge("retailer", "PROVIDES_SERVICE", node_id))

    if tier["id"] == "tier_high":
        kg.add_node(Node("specialist", "Khám Bác sĩ chuyên khoa Nội tiết", "Service",
                         {"Nhóm": "Chuyển tuyến chuyên khoa"}))
        kg.add_edge(Edge("icd10", "REFERS_TO", "specialist"))

    return kg


def _thuoc_tinh_mat_hang(ten: str) -> dict[str, Any]:
    """Thuộc tính hiển thị của một nút hàng hoá, tra từ danh mục bán lẻ.

    Trả về thuộc tính tối thiểu khi tên không khớp danh mục, thay vì ném lỗi:
    nhãn có thể đến từ một tầng nguy cơ được mở rộng sau này mà danh mục chưa
    kịp cập nhật, và khi đó thiếu thông tin giá vẫn tốt hơn là hỏng cả đồ thị.
    """
    item = pharm.san_pham_theo_ten(ten)
    if item is None:
        return {"Nhóm": "Chưa có trong danh mục bán lẻ"}
    return {
        "Mã SKU": item["sku"],
        "Nhóm hàng": item["nhom_hang"],
        "Đơn vị": item["unit"],
        "Khoảng giá": pharm.dinh_dang_khoang(item["gia_min"], item["gia_max"]),
        "Công dụng": item["cong_dung"],
        "Mua định kỳ": "Có" if item["dinh_ky"] else "Không",
        "Tra cứu giá": item["tra_cuu"],
    }


# ==========================================================================
#  MIỀN 2 — ĐỒ THỊ TRI THỨC ĐÔ THỊ – TÀI CHÍNH
# ==========================================================================

# Phân khúc bất động sản theo mức giá dự báo, kèm gói tín dụng tương ứng.
PROPERTY_SEGMENTS = [
    {
        "id": "seg_luxury",
        "label": "Phân khúc Cao cấp",
        "min_price": 8.0,
        "loan": "Gói vay Bất động sản Cao cấp",
        "ltv": 0.70,
        "uu_dai": "ưu đãi 12 tháng đầu",
    },
    {
        "id": "seg_mid",
        "label": "Phân khúc Trung cấp",
        "min_price": 4.5,
        "loan": "Gói vay Mua nhà Linh hoạt",
        "ltv": 0.75,
        "uu_dai": "ưu đãi 24 tháng đầu",
    },
    {
        "id": "seg_afford",
        "label": "Phân khúc Bình dân",
        "min_price": 0.0,
        "loan": "Gói vay Nhà ở Xã hội",
        "ltv": 0.80,
        "uu_dai": "theo chương trình hỗ trợ",
    },
]


# Cùng lý do như RISK_TIERS ở trên: lãi suất, kỳ hạn và danh sách tiện ích chỉ
# được khai một lần, trong :mod:`src.kg.catalog_urban`. Ở đây chỉ dựng lại chuỗi
# mô tả cho người đọc, nên con số trên giao diện và con số dùng để tính trả góp
# không thể lệch nhau.
for _seg in PROPERTY_SEGMENTS:
    _terms = urban.FINANCE_TERMS[_seg["id"]]
    _cum = urban.UTILITY_CLUSTERS[_seg["id"]]
    _seg["rate"] = f"{_terms['lai_suat_nam']:.1%}/năm, {_seg['uu_dai']}".replace(".", ",")
    _seg["tenor"] = f"Tối đa {_terms['ky_han_nam']} năm"
    _seg["utilities"] = [u["ten"] for u in _cum["tien_ich"]]
    _seg["cluster_id"] = _cum["id"]
del _seg, _terms, _cum


def classify_segment(price: float) -> dict:
    """Ánh xạ giá dự báo (tỷ VNĐ) sang phân khúc thị trường."""
    for segment in PROPERTY_SEGMENTS:
        if price >= segment["min_price"]:
            return segment
    return PROPERTY_SEGMENTS[-1]


def build_property_kg(prop: dict, price: float,
                      model_name: str = "XGBoost") -> KnowledgeGraph:
    """Dựng đồ thị tri thức đô thị – tài chính cho một bất động sản.

    Chuỗi suy luận: thuộc tính vật lý → định giá AI → phân khúc thị trường →
    gói tín dụng ngân hàng và tiện ích đô thị xung quanh.
    """
    kg = KnowledgeGraph()
    segment = classify_segment(price)

    # ---- Lớp 1: thuộc tính vật lý ----
    kg.add_node(Node("property", "Bất động sản", "Property", {
        "Diện tích": prop.get("Area"),
        "Số tầng": prop.get("Floors"),
        "Số phòng ngủ": prop.get("Bedrooms"),
    }))

    for key, label, unit in [("Area", "Diện tích", "m²"), ("Floors", "Số tầng", "tầng"),
                             ("Bedrooms", "Phòng ngủ", "phòng"),
                             ("Bathrooms", "Phòng tắm", "phòng")]:
        if prop.get(key) is None:
            continue
        node_id = f"attr_{key.lower()}"
        kg.add_node(Node(node_id, f"{label} = {prop[key]:g} {unit}", "Attribute"))
        kg.add_edge(Edge("property", "HAS_ATTRIBUTE", node_id))

    # ---- Lớp 2: vị trí hành chính ----
    # Dùng ``or`` chứ không dùng giá trị mặc định của ``get``: khoá vẫn tồn tại
    # nhưng mang giá trị ``None`` khi ô dữ liệu khuyết (build_graph._row_to_dict
    # luôn tạo đủ khoá, và form nhập liệu ở Phase 6 cũng có thể gửi lên None).
    # Giá trị mặc định của ``get`` chỉ cứu được trường hợp *thiếu khoá*, nên nếu
    # dùng nó thì nhãn hiện ra là "Tỉnh/Thành None" mà không lỗi nào được ném.
    province = prop.get("Province") or "Không rõ"
    district = prop.get("District") or "Không rõ"
    kg.add_node(Node("district", f"Quận/Huyện {district}", "Location"))
    kg.add_node(Node("province", f"Tỉnh/Thành {province}", "Location"))
    kg.add_edge(Edge("property", "LOCATED_IN", "district"))
    kg.add_edge(Edge("district", "PART_OF", "province"))

    # ---- Lớp 3: định giá AI và phân khúc ----
    kg.add_node(Node("model", f"Mô hình {model_name}", "Model",
                     {"Loại": "Học máy truyền thống"}))
    kg.add_edge(Edge("property", "VALUATED_BY", "model"))

    kg.add_node(Node("valuation", f"Giá định giá {price:.2f} tỷ VNĐ", "Prediction",
                     {"Giá trị": round(price, 3), "Đơn vị": "tỷ VNĐ"}))
    kg.add_edge(Edge("model", "PRODUCES", "valuation"))

    kg.add_node(Node(segment["id"], segment["label"], "Segment",
                     {"Ngưỡng giá": segment["min_price"]}))
    kg.add_edge(Edge("valuation", "BELONGS_TO_SEGMENT", segment["id"]))

    # ---- Lớp 4: gói tài chính mua nhà ----
    # Hạn mức vay là phép nhân đơn giản, nhưng ba con số đi kèm — trả góp hằng
    # tháng, tổng tiền lãi, thu nhập tối thiểu — mới là thứ trả lời được câu hỏi
    # "tôi có mua nổi không". Cả ba tính bằng công thức niên kim ở
    # :mod:`src.kg.catalog_urban`, không phải con số áng chừng.
    tai_chinh = urban.ho_so_tai_chinh(price, segment)
    loan_amount = tai_chinh["han_muc_vay"]

    kg.add_node(Node("loan", segment["loan"], "LoanPackage", {
        "Tỷ lệ cho vay (LTV)": f"{segment['ltv']:.0%}",
        "Hạn mức": f"{loan_amount:.2f} tỷ VNĐ",
        "Vốn tự có tối thiểu": urban.ty_dong(tai_chinh["von_tu_co"]),
        "Lãi suất": segment["rate"],
        "Thời hạn": segment["tenor"],
        "Phí trả trước hạn": tai_chinh["phi_trong_ly"],
        "Ghi chú": tai_chinh["ghi_chu"],
    }))
    kg.add_edge(Edge(segment["id"], "ELIGIBLE_FOR", "loan"))

    kg.add_node(Node("repayment",
                     f"Trả góp {tai_chinh['tra_gop_thang'] * 1000:,.1f} triệu đ/tháng"
                     .replace(",", "."),
                     "Repayment", {
                         "Kỳ hạn": f"{tai_chinh['ky_han_nam']} năm",
                         "Số kỳ trả": tai_chinh["ky_han_nam"] * 12,
                         "Tổng tiền lãi": urban.ty_dong(tai_chinh["tong_tien_lai"]),
                         "Công thức": "Niên kim: M = P·r / (1 − (1+r)^(−n))",
                     }))
    kg.add_edge(Edge("loan", "AMORTISED_AS", "repayment"))

    kg.add_node(Node("income",
                     f"Thu nhập tối thiểu {tai_chinh['thu_nhap_toi_thieu'] * 1000:,.1f} triệu đ/tháng"
                     .replace(",", "."),
                     "Requirement", {
                         "Tỷ lệ nợ trên thu nhập (DTI)": f"{tai_chinh['dti_toi_da']:.0%}",
                         "Cách tính": "Trả góp hằng tháng chia cho ngưỡng DTI",
                     }))
    kg.add_edge(Edge("repayment", "REQUIRES_INCOME", "income"))

    kg.add_node(Node("bank", "Ngân hàng đối tác cho vay mua nhà", "Service",
                     {"Hotline": "1800 588822"}))
    kg.add_edge(Edge("loan", "PROVIDED_BY", "bank"))

    # ---- Lớp 5: hạ tầng tiện ích đô thị ----
    # Cụm tiện ích là nút gom, song song với nút gói chăm sóc ở đồ thị y sinh:
    # nó cho phép gắn bán kính tiếp cận và ghi chú nguồn vào một chỗ, thay vì
    # lặp lại trên từng tiện ích lẻ.
    cum = urban.cum_tien_ich(segment["id"])
    kg.add_node(Node(cum["id"], cum["label"], "Infrastructure", {
        "Mô tả": cum["mo_ta"],
        "Bán kính tiếp cận": f"{cum['ban_kinh_km']:g} km",
        "Số tiện ích": len(cum["tien_ich"]),
        "Ghi chú": cum["ghi_chu"],
    }))
    kg.add_edge(Edge("district", "HAS_INFRASTRUCTURE", cum["id"]))

    for i, utility in enumerate(cum["tien_ich"]):
        node_id = f"utility_{i}"
        kg.add_node(Node(node_id, utility["ten"], "Utility", {
            "Phân loại": utility["loai"],
            "Bán kính tham khảo": f"{cum['ban_kinh_km']:g} km",
        }))
        kg.add_edge(Edge(cum["id"], "INCLUDES_UTILITY", node_id))

    legal = prop.get("Legal status") or "Không rõ"
    kg.add_node(Node("legal", f"Pháp lý: {legal}", "Legal"))
    kg.add_edge(Edge("property", "HAS_LEGAL_STATUS", "legal"))
    if legal == "Have certificate":
        kg.add_edge(Edge("legal", "QUALIFIES_FOR", "loan",
                         {"Ghi chú": "Đủ điều kiện thế chấp"}))

    return kg


# ==========================================================================
#  BẢNG MÀU CHO TRỰC QUAN HOÁ
# ==========================================================================

NODE_COLOURS = {
    "Patient": "#E74C3C",
    "Biomarker": "#3498DB",
    "Model": "#9B59B6",
    "Prediction": "#E67E22",
    "Disease": "#C0392B",
    "RiskTier": "#16A085",
    "Device": "#2980B9",
    "Nutrition": "#27AE60",
    "Action": "#7F8C8D",
    "Service": "#F39C12",
    "CarePackage": "#1F6F8B",
    "Retailer": "#B8860B",
    "Property": "#E74C3C",
    "Attribute": "#3498DB",
    "Location": "#16A085",
    "Segment": "#8E44AD",
    "LoanPackage": "#D35400",
    "Repayment": "#A0522D",
    "Requirement": "#5D6D7E",
    "Infrastructure": "#1F6F8B",
    "Utility": "#27AE60",
    "Legal": "#7F8C8D",
}


# ==========================================================================
#  KIẾN TRÚC ĐA TẦNG — MÔ TẢ ĐỂ VẼ SƠ ĐỒ VÀ ĐỂ VIẾT BÁO CÁO
# ==========================================================================
#
# Khai báo tường minh việc mỗi loại thực thể thuộc tầng nào. Trước đây thông tin
# này chỉ nằm trong các dòng chú thích "---- Lớp N ----" bên trong hàm dựng đồ
# thị, nên không có cách nào vẽ lại sơ đồ kiến trúc mà không đọc mã nguồn.

KG_LAYERS: dict[str, list[dict[str, Any]]] = {
    "medical": [
        {"stt": 1, "ten": "Hồ sơ Sinh học Cá thể",
         "mo_ta": "Chỉ số sinh tồn đo được và đặc trưng tương tác thiết kế",
         "types": ["Patient", "Biomarker"]},
        {"stt": 2, "ten": "Suy luận AI và Chuẩn hoá Quốc tế",
         "mo_ta": "Mô hình Học máy sinh xác suất, ánh xạ sang mã bệnh ICD-10",
         "types": ["Model", "Prediction", "Disease", "RiskTier"]},
        {"stt": 3, "ten": "Danh mục Can thiệp",
         "mo_ta": "Thiết bị y tế, dinh dưỡng y học và hướng dẫn lâm sàng",
         "types": ["Device", "Nutrition", "Action"]},
        {"stt": 4, "ten": "Gói Chăm sóc Sức khoẻ",
         "mo_ta": "Gom danh mục thành một đơn vị có chi phí ban đầu và chi phí duy trì",
         "types": ["CarePackage"]},
        {"stt": 5, "ten": "Hạ tầng Bán lẻ Dược phẩm",
         "mo_ta": "Nhà thuốc cung ứng, dịch vụ tại quầy và chuyển tuyến chuyên khoa",
         "types": ["Retailer", "Service"]},
    ],
    "property": [
        {"stt": 1, "ten": "Thuộc tính Vật lý và Pháp lý",
         "mo_ta": "Diện tích, kết cấu phòng, số tầng và tình trạng giấy tờ",
         "types": ["Property", "Attribute", "Legal"]},
        {"stt": 2, "ten": "Vị trí Hành chính",
         "mo_ta": "Quận/Huyện và Tỉnh/Thành — nhóm đặc trưng quyết định giá",
         "types": ["Location"]},
        {"stt": 3, "ten": "Định giá AI và Phân khúc",
         "mo_ta": "Mô hình hồi quy sinh giá dự báo, ánh xạ sang phân khúc thị trường",
         "types": ["Model", "Prediction", "Segment"]},
        {"stt": 4, "ten": "Gói Tài chính Mua nhà",
         "mo_ta": "Hạn mức vay, trả góp niên kim, tổng lãi và thu nhập tối thiểu",
         "types": ["LoanPackage", "Repayment", "Requirement", "Service"]},
        {"stt": 5, "ten": "Hạ tầng Tiện ích Đô thị",
         "mo_ta": "Cụm tiện ích theo phân khúc, phân loại theo bốn nhóm chức năng",
         "types": ["Infrastructure", "Utility"]},
    ],
}


def layer_of(node_type: str, domain: str) -> int:
    """Số thứ tự tầng chứa một loại thực thể. Trả về 0 khi loại không thuộc miền."""
    for layer in KG_LAYERS[domain]:
        if node_type in layer["types"]:
            return layer["stt"]
    return 0
