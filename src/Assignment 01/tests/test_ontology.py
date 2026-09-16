"""Kiểm thử bản thể học — phân tầng, dựng đồ thị, kết xuất bộ ba.

Trọng tâm của tệp này là **các giá trị biên**. Phân tầng nguy cơ và phân khúc
giá đều so sánh bằng ``>=``, nên đúng tại ngưỡng là chỗ dễ sai nhất và cũng là
chỗ hậu quả nặng nhất: lệch một tầng ở đây kéo theo sai cả gói can thiệp lẫn
hạn mức vay.
"""

from __future__ import annotations

import pytest

from src.kg.ontology import (
    Edge,
    KnowledgeGraph,
    NODE_COLOURS,
    Node,
    PROPERTY_SEGMENTS,
    RISK_TIERS,
    build_medical_kg,
    build_property_kg,
    classify_risk,
    classify_segment,
)


# ==========================================================================
#  CẤU TRÚC DỮ LIỆU
# ==========================================================================

def test_them_nut_trung_dinh_danh_thi_bo_qua():
    """Định danh nút phải duy nhất — nạp lên Neo4j dựa vào tính chất này."""
    kg = KnowledgeGraph()
    kg.add_node(Node("a", "Lần đầu", "Patient"))
    kg.add_node(Node("a", "Lần hai", "Disease"))

    assert len(kg.nodes) == 1
    assert kg.nodes[0].label == "Lần đầu"


def test_bo_ba_dung_nhan_hien_thi_chu_khong_dung_dinh_danh():
    kg = KnowledgeGraph()
    kg.add_node(Node("p", "Bệnh nhân", "Patient"))
    kg.add_node(Node("d", "Đái tháo đường", "Disease"))
    kg.add_edge(Edge("p", "MAPS_TO_DISEASE", "d"))

    assert kg.triplets() == [("Bệnh nhân", "MAPS_TO_DISEASE", "Đái tháo đường")]


def test_bo_ba_giu_dinh_danh_khi_nut_khong_ton_tai():
    """Cạnh trỏ tới nút chưa khai báo thì giữ nguyên định danh, không nuốt mất."""
    kg = KnowledgeGraph()
    kg.add_node(Node("p", "Bệnh nhân", "Patient"))
    kg.add_edge(Edge("p", "REFERS_TO", "khong_ton_tai"))

    assert kg.triplets() == [("Bệnh nhân", "REFERS_TO", "khong_ton_tai")]


def test_thong_ke_dem_dung_bon_dai_luong():
    kg = KnowledgeGraph()
    kg.add_node(Node("p", "Bệnh nhân", "Patient"))
    kg.add_node(Node("b1", "Glucose", "Biomarker"))
    kg.add_node(Node("b2", "BMI", "Biomarker"))
    kg.add_edge(Edge("p", "HAS_BIOMARKER", "b1"))
    kg.add_edge(Edge("p", "HAS_BIOMARKER", "b2"))

    assert kg.stats() == {
        "Số thực thể (|V|)": 3,
        "Số cạnh (|E|)": 2,
        "Số loại quan hệ (|R|)": 1,
        "Số loại thực thể": 2,
    }


# ==========================================================================
#  PHÂN TẦNG — GIÁ TRỊ BIÊN
# ==========================================================================

@pytest.mark.parametrize("probability, expected", [
    (1.00, "tier_high"),
    (0.26, "tier_high"),
    (0.25, "tier_high"),        # đúng ngưỡng thuộc về tầng trên
    (0.2499, "tier_moderate"),
    (0.15, "tier_moderate"),    # đúng ngưỡng thuộc về tầng trên
    (0.1499, "tier_low"),
    (0.00, "tier_low"),
])
def test_phan_tang_nguy_co_tai_bien(probability, expected):
    assert classify_risk(probability)["id"] == expected


def test_xac_suat_am_van_tra_ve_tang_thap_nhat():
    """Không bao giờ trả về None — hàm gọi không có nhánh xử lý thiếu tầng."""
    assert classify_risk(-0.5)["id"] == "tier_low"


@pytest.mark.parametrize("price, expected", [
    (99.0, "seg_luxury"),
    (8.01, "seg_luxury"),
    (8.00, "seg_luxury"),
    (7.99, "seg_mid"),
    (4.50, "seg_mid"),
    (4.49, "seg_afford"),
    (0.00, "seg_afford"),
])
def test_phan_khuc_gia_tai_bien(price, expected):
    assert classify_segment(price)["id"] == expected


def test_gia_am_van_tra_ve_phan_khuc_thap_nhat():
    assert classify_segment(-1.0)["id"] == "seg_afford"


def test_bang_phan_tang_sap_xep_giam_dan():
    """Cả hai hàm phân loại duyệt tuần tự nên thứ tự giảm dần là điều kiện đúng.

    Nếu ai đó chèn một tầng mới vào giữa mà quên thứ tự, hàm phân loại sẽ trả
    về tầng sai một cách âm thầm — không lỗi, chỉ sai. Kiểm ở đây để việc chèn
    sai bị chặn ngay.
    """
    probs = [tier["min_prob"] for tier in RISK_TIERS]
    prices = [seg["min_price"] for seg in PROPERTY_SEGMENTS]

    assert probs == sorted(probs, reverse=True)
    assert prices == sorted(prices, reverse=True)
    assert probs[-1] == 0.0, "Tầng cuối phải nhận mọi giá trị còn lại"
    assert prices[-1] == 0.0, "Phân khúc cuối phải nhận mọi giá trị còn lại"


# ==========================================================================
#  ĐỒ THỊ Y SINH
# ==========================================================================

BENH_NHAN_DAY_DU = {
    "Age": 45, "Glucose": 160.0, "BMI": 31.0,
    "BloodPressure": 80.0, "Insulin": 120.0,
}


def test_do_thi_y_sinh_du_chuoi_suy_luan():
    kg = build_medical_kg(BENH_NHAN_DAY_DU, 0.83, model_name="XGBoost")
    triplets = {(s, r, o) for s, r, o in kg.triplets()}
    relations = {r for _s, r, _o in triplets}

    assert {"HAS_BIOMARKER", "EVALUATED_BY", "PRODUCES", "MAPS_TO_DISEASE",
            "STRATIFIED_AS", "REQUIRES_DEVICE", "RECOMMENDS_NUTRITION",
            "ADVISES", "SUPPLIED_BY"} <= relations


def test_chi_so_khuyet_khong_sinh_ra_nut_nan():
    """``None`` phải bị bỏ qua, không được thành nút "Insulin = None".

    Đây là lý do ``build_graph._native`` tồn tại: NaN của pandas không phải
    None của Python, nên nếu bỏ bước đổi kiểu thì nhánh này không chạy.
    """
    patient = dict(BENH_NHAN_DAY_DU, Insulin=None, BloodPressure=None)
    kg = build_medical_kg(patient, 0.83)
    biomarkers = [n.id for n in kg.nodes if n.type == "Biomarker"]

    assert "bio_insulin" not in biomarkers
    assert "bio_bloodpressure" not in biomarkers
    assert "bio_glucose" in biomarkers
    assert not any("None" in n.label for n in kg.nodes)


def test_dac_trung_tuong_tac_can_ca_hai_chi_so():
    co_du = build_medical_kg(BENH_NHAN_DAY_DU, 0.83)
    thieu_bmi = build_medical_kg(dict(BENH_NHAN_DAY_DU, BMI=None), 0.83)

    assert any(n.id == "bio_interaction" for n in co_du.nodes)
    assert not any(n.id == "bio_interaction" for n in thieu_bmi.nodes)


def test_chuyen_kham_chuyen_khoa_chi_danh_cho_nhom_nguy_co_cao():
    cao = build_medical_kg(BENH_NHAN_DAY_DU, 0.90)
    thap = build_medical_kg(BENH_NHAN_DAY_DU, 0.05)

    assert any(e.relation == "REFERS_TO" for e in cao.edges)
    assert not any(e.relation == "REFERS_TO" for e in thap.edges)


def test_moi_thiet_bi_va_dinh_duong_deu_co_noi_cung_ung():
    """Không sản phẩm nào được treo lơ lửng — người dùng phải mua được ở đâu đó."""
    kg = build_medical_kg(BENH_NHAN_DAY_DU, 0.83)
    can_mua = {n.id for n in kg.nodes if n.type in {"Device", "Nutrition"}}
    co_nguon = {e.source for e in kg.edges if e.relation == "SUPPLIED_BY"}

    assert can_mua and can_mua <= co_nguon


# ==========================================================================
#  ĐỒ THỊ BẤT ĐỘNG SẢN
# ==========================================================================

NHA_DAY_DU = {
    "Area": 100.0, "Floors": 4.0, "Bedrooms": 4.0, "Bathrooms": 5.0,
    "Province": "Hồ Chí Minh", "District": "Nhà Bè",
    "Legal status": "Have certificate",
}


def test_han_muc_vay_bang_gia_nhan_ty_le_cho_vay():
    """Con số hạn mức là thứ người dùng mang tới ngân hàng — phải đúng."""
    kg = build_property_kg(NHA_DAY_DU, 8.42)
    loan = next(n for n in kg.nodes if n.type == "LoanPackage")
    segment = classify_segment(8.42)

    assert loan.properties["Hạn mức"] == f"{8.42 * segment['ltv']:.2f} tỷ VNĐ"
    assert loan.properties["Tỷ lệ cho vay (LTV)"] == f"{segment['ltv']:.0%}"


def test_so_do_hop_le_moi_duoc_the_chap():
    co_so_do = build_property_kg(NHA_DAY_DU, 8.42)
    khong_so_do = build_property_kg(
        dict(NHA_DAY_DU, **{"Legal status": "Sale contract"}), 8.42)

    assert any(e.relation == "QUALIFIES_FOR" for e in co_so_do.edges)
    assert not any(e.relation == "QUALIFIES_FOR" for e in khong_so_do.edges)


@pytest.mark.parametrize("khoa", ["Province", "District", "Legal status"])
def test_gia_tri_dinh_danh_bang_None_khong_lot_ra_nhan(khoa):
    """``dict.get(khoa, mặc_định)`` KHÔNG cứu được khi khoá tồn tại với giá trị None.

    Trường hợp này có thật: ``build_graph._row_to_dict`` luôn tạo đủ khoá và
    đặt ``None`` cho ô khuyết, còn form nhập liệu ở Phase 6 cũng có thể gửi lên
    ``None``. Nếu không chặn, nhãn hiện ra là "Tỉnh/Thành None" — sai mà không
    có lỗi nào được ném.
    """
    kg = build_property_kg(dict(NHA_DAY_DU, **{khoa: None}), 6.0)

    assert not any("None" in n.label for n in kg.nodes), \
        f"Nhãn chứa None khi {khoa} khuyết"


def test_thuoc_tinh_khuyet_khong_sinh_ra_nut():
    kg = build_property_kg(dict(NHA_DAY_DU, Bathrooms=None), 6.0)

    assert not any(n.id == "attr_bathrooms" for n in kg.nodes)
    assert any(n.id == "attr_area" for n in kg.nodes)


# ==========================================================================
#  BẤT BIẾN CHUNG
# ==========================================================================

def test_moi_loai_thuc_the_deu_co_mau():
    """Thiếu màu thì pyvis vẽ ra màu xám mặc định — sai âm thầm, không lỗi."""
    kgs = [build_medical_kg(BENH_NHAN_DAY_DU, p) for p in (0.9, 0.2, 0.05)]
    kgs += [build_property_kg(NHA_DAY_DU, p) for p in (9.0, 6.0, 3.0)]

    types = {n.type for kg in kgs for n in kg.nodes}
    assert types <= set(NODE_COLOURS), f"Thiếu màu cho: {types - set(NODE_COLOURS)}"


def test_moi_canh_deu_noi_hai_nut_co_that():
    """Cạnh trỏ vào hư không sẽ làm câu MATCH khi nạp lên Neo4j im lặng bỏ qua."""
    for kg in [build_medical_kg(BENH_NHAN_DAY_DU, p) for p in (0.9, 0.2, 0.05)] + \
              [build_property_kg(NHA_DAY_DU, p) for p in (9.0, 6.0, 3.0)]:
        ids = {n.id for n in kg.nodes}
        for edge in kg.edges:
            assert edge.source in ids, f"Cạnh xuất phát từ nút lạ: {edge.source}"
            assert edge.target in ids, f"Cạnh trỏ tới nút lạ: {edge.target}"


def test_chi_so_bang_khong_van_sinh_dac_trung_tuong_tac():
    """0 là số đo thật, không phải thiếu dữ liệu.

    Form ở Phase 6 trả về 0 khi người dùng chưa động vào ô, và
    ``prepare_diabetes(clean=False)`` giữ nguyên số 0 quy ước của tập Pima.
    Kiểm bằng tính đúng-sai thì nút "Glucose = 0" vẫn hiện mà đặc trưng
    Glucose × BMI biến mất — đồ thị tự mâu thuẫn, không lỗi nào được ném.
    """
    kg = build_medical_kg(dict(BENH_NHAN_DAY_DU, Glucose=0.0), 0.5)

    assert any(n.id == "bio_glucose" for n in kg.nodes)
    assert any(n.id == "bio_interaction" for n in kg.nodes)


def test_chi_so_bang_khong_van_sinh_dac_trung_tuong_tac():
    """0 là số đo thật, không phải thiếu dữ liệu.

    Kiểm bằng tính đúng-sai (``if patient.get("Glucose")``) sẽ để lọt: nút
    "Glucose = 0 mg/dL" vẫn hiện mà đặc trưng tương tác biến mất, đồ thị tự mâu
    thuẫn với chính nó. Form nhập liệu ở Phase 6 trả về 0 khi người dùng chưa
    động vào ô, nên đây là ca sẽ gặp ngay.
    """
    kg = build_medical_kg(dict(BENH_NHAN_DAY_DU, Glucose=0.0), 0.5)

    assert any(n.id == "bio_glucose" for n in kg.nodes), "Vẫn phải ghi nhận số đo 0"
    assert any(n.id == "bio_interaction" for n in kg.nodes), \
        "Có đủ cả Glucose lẫn BMI thì phải có đặc trưng tương tác"
