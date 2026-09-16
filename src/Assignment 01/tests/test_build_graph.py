"""Kiểm thử bước dựng đồ thị — đổi kiểu, chọn ca, bố cục hình vẽ.

Hai bất biến được canh gắt nhất ở đây:

1. **Ranh giới kiểu dữ liệu.** ``NaN`` của pandas không phải ``None`` của
   Python, và trình điều khiển Neo4j không tuần tự hoá được ``numpy.int64``.
   Cả hai gặp nhau ở đúng một hàm, nên hàm đó phải đúng.
2. **Dải chọn ca khớp với hàm phân loại.** Nếu dải lệch khỏi ngưỡng phân tầng
   dù chỉ ở một điểm, ca "Nhóm Nguy cơ Cao" có thể là một bệnh nhân mà chính
   ``classify_risk`` xếp vào nhóm khác — sai mà không có lỗi nào được ném.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.kg import build_graph as bg
from src.kg.ontology import (
    NODE_COLOURS,
    PROPERTY_SEGMENTS,
    RISK_TIERS,
    classify_risk,
    classify_segment,
)


# ==========================================================================
#  ĐỔI KIỂU TẠI RANH GIỚI
# ==========================================================================

@pytest.mark.parametrize("dau_vao, mong_doi", [
    (None, None),
    (np.nan, None),
    (float("inf"), None),
    (float("-inf"), None),
    (pd.NA, None),
    (pd.NaT, None),
    ("", None),
    ("   ", None),
    ("  còn chữ  ", "còn chữ"),
    (np.int64(7), 7),
    (np.float64(1.5), 1.5),
    (np.bool_(True), True),
    (0, 0),
    (0.0, 0.0),
    (False, False),
])
def test_doi_kieu_ve_python_thuan(dau_vao, mong_doi):
    ket_qua = bg._native(dau_vao)

    assert ket_qua == mong_doi or (ket_qua is None and mong_doi is None)
    assert not isinstance(ket_qua, np.generic), "Trình điều khiển Neo4j không nuốt được kiểu NumPy"


def test_khong_bien_so_khong_thanh_thieu_du_lieu():
    """0 và False là giá trị thật; nhầm chúng thành khuyết là mất dữ liệu."""
    assert bg._native(0) is not None
    assert bg._native(0.0) is not None
    assert bg._native(np.int64(0)) == 0
    assert bg._native(False) is False


def test_ket_qua_doi_kieu_tuan_tu_hoa_duoc_bang_json():
    import json
    row = pd.Series({"a": np.int64(3), "b": np.nan, "c": np.float64(2.5), "d": "x"})

    ban_ghi = bg._row_to_dict(row, ["a", "b", "c", "d", "khong_co_cot"])

    assert json.loads(json.dumps(ban_ghi)) == {
        "a": 3, "b": None, "c": 2.5, "d": "x", "khong_co_cot": None}


# ==========================================================================
#  TÊN MÔ HÌNH
# ==========================================================================

class _PipelineGia:
    def __init__(self, estimator):
        self.steps = [("tien_xu_ly", object()), ("mo_hinh", estimator)]


class KNeighborsClassifier:  # noqa: D101 — chỉ cần đúng tên lớp
    pass


class MoHinhLaLam:  # noqa: D101
    pass


def test_ten_mo_hinh_doi_sang_ten_thuat_toan():
    """Báo cáo gọi "K-Nearest Neighbors"; đồ thị phải gọi giống để đối chiếu được."""
    assert bg._estimator_name(_PipelineGia(KNeighborsClassifier())) == "K-Nearest Neighbors"


def test_mo_hinh_la_thi_giu_nguyen_ten_lop():
    """Không nuốt mất tên — thà hiện tên lớp còn hơn hiện chuỗi rỗng."""
    assert bg._estimator_name(_PipelineGia(MoHinhLaLam())) == "MoHinhLaLam"


# ==========================================================================
#  DẢI CHỌN CA
# ==========================================================================

def test_dai_nguy_co_phu_kin_va_khong_chong_lan():
    dai = bg._bands(RISK_TIERS, "min_prob")

    assert [d[1] for d in dai] == [0.25, 0.15, 0.0]
    assert [d[2] for d in dai] == [np.inf, 0.25, 0.15]


@pytest.mark.parametrize("xac_suat", [round(x * 0.01, 2) for x in range(0, 101)])
def test_dai_nguy_co_khop_voi_ham_phan_tang(xac_suat):
    """Mỗi xác suất rơi vào đúng một dải, và dải ấy chính là tầng classify_risk trả về."""
    trung = [tier for tier, thap, cao in bg._bands(RISK_TIERS, "min_prob")
             if thap <= xac_suat < cao]

    assert len(trung) == 1, f"{xac_suat} rơi vào {len(trung)} dải"
    assert trung[0]["id"] == classify_risk(xac_suat)["id"]


@pytest.mark.parametrize("gia", [round(x * 0.25, 2) for x in range(0, 61)])
def test_dai_phan_khuc_khop_voi_ham_phan_loai(gia):
    trung = [seg for seg, thap, cao in bg._bands(PROPERTY_SEGMENTS, "min_price")
             if thap <= gia < cao]

    assert len(trung) == 1, f"{gia} rơi vào {len(trung)} dải"
    assert trung[0]["id"] == classify_segment(gia)["id"]


# ==========================================================================
#  CHỌN QUAN SÁT TRUNG VỊ
# ==========================================================================

def test_dai_rong_thi_tra_ve_none():
    values = pd.Series([0.1, 0.2], index=[10, 11])

    assert bg._median_of_band(values, 0.5, 1.0) is None


def test_lay_dung_quan_sat_giua_dai():
    values = pd.Series([0.10, 0.30, 0.50, 0.70, 0.90], index=[4, 3, 2, 1, 0])

    assert bg._median_of_band(values, 0.0, 1.0) == 2, "Giá trị 0.50 nằm giữa"


def test_bien_duoi_tinh_vao_dai_bien_tren_thi_khong():
    values = pd.Series([0.15, 0.25], index=[0, 1])

    assert bg._median_of_band(values, 0.15, 0.25) == 0


def test_ket_qua_khong_doi_khi_nhieu_quan_sat_trung_gia_tri():
    """K-Nearest Neighbors chỉ trả về vài mức xác suất rời rạc nên đây là ca thường gặp.

    Thứ tự sắp xếp phải là thứ tự toàn phần, nếu không việc chọn ca sẽ đổi theo
    thứ tự đầu vào và yêu cầu tái lập R14 sụp đổ.
    """
    values = pd.Series([0.2] * 7, index=[9, 3, 7, 1, 5, 2, 8])
    xao_tron = values.sample(frac=1.0, random_state=1)

    assert bg._median_of_band(values, 0.0, 1.0) == bg._median_of_band(xao_tron, 0.0, 1.0)


def test_goi_nhieu_lan_cho_cung_ket_qua():
    rng = np.random.default_rng(7)
    values = pd.Series(rng.choice([0.0, 0.2, 0.4, 0.6], size=200))

    ket_qua = {bg._median_of_band(values, 0.0, 1.0) for _ in range(5)}
    assert len(ket_qua) == 1


# ==========================================================================
#  BỐ CỤC HÌNH VẼ
# ==========================================================================

def test_moi_loai_thuc_the_deu_co_tang_hien_thi():
    """Thiếu tầng thì nút bị dồn về cột cuối, hình vẽ sai cấu trúc mà không báo lỗi."""
    from src.kg.ontology import build_medical_kg, build_property_kg

    dung = {
        "medical": [build_medical_kg({"Age": 40, "Glucose": 150.0, "BMI": 30.0,
                                      "BloodPressure": 80.0, "Insulin": 100.0}, p)
                    for p in (0.9, 0.2, 0.05)],
        "property": [build_property_kg({"Area": 100.0, "Floors": 3.0, "Bedrooms": 3.0,
                                        "Bathrooms": 2.0, "Province": "Hà Nội",
                                        "District": "Cầu Giấy",
                                        "Legal status": "Have certificate"}, p)
                     for p in (9.0, 6.0, 3.0)],
    }
    for domain, graphs in dung.items():
        types = {n.type for kg in graphs for n in kg.nodes}
        thieu = types - set(bg.NODE_LAYERS[domain])
        assert not thieu, f"Miền {domain} thiếu tầng cho: {thieu}"


def test_bang_tang_khong_khai_bao_loai_khong_ton_tai():
    for domain, layers in bg.NODE_LAYERS.items():
        la = set(layers) - set(NODE_COLOURS)
        assert not la, f"Miền {domain} khai báo tầng cho loại không có trong bản thể học: {la}"


def test_bo_cuc_xac_dinh_va_gom_theo_loai():
    import networkx as nx

    graph = nx.DiGraph()
    for node_id, kind, layer in [("a", "Device", 5), ("b", "Action", 5),
                                 ("c", "Device", 5), ("d", "Patient", 0)]:
        graph.add_node(node_id, kind=kind, layer=layer, label=node_id)

    lan_mot = bg._layered_positions(graph)
    lan_hai = bg._layered_positions(graph)

    assert lan_mot == lan_hai, "Bố cục phải xác định để hình vẽ tái lập được"
    assert lan_mot["d"][0] == 0.0 and lan_mot["a"][0] == 5.0, "Hoành độ là số hiệu tầng"

    # Cùng loại phải nằm thành một khối liền nhau trong cột — đó là điều làm
    # giảm số cạnh cắt nhau. Đọc thứ tự theo tung độ rồi kiểm loại không lặp lại
    # sau khi đã rời khỏi khối của nó.
    cot = sorted(["a", "b", "c"], key=lambda n: lan_mot[n][1])
    thu_tu_loai = [graph.nodes[n]["kind"] for n in cot]
    assert thu_tu_loai == ["Action", "Device", "Device"]


def test_cot_mot_nut_nam_giua():
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node("chi_mot", kind="Patient", layer=0, label="x")
    graph.add_node("kia", kind="Model", layer=1, label="y")
    graph.add_node("kia2", kind="Model", layer=1, label="z")

    assert bg._layered_positions(graph)["chi_mot"][1] == 0.0


# ==========================================================================
#  CHỌN CA TỪ DỮ LIỆU THẬT
# ==========================================================================

@pytest.mark.slow
def test_ca_y_sinh_dung_tang_va_lap_lai_duoc():
    lan_mot = bg.select_medical_cases()
    lan_hai = bg.select_medical_cases()

    assert [c.id for c in lan_mot] == [c.id for c in lan_hai], "Chạy lại phải ra đúng các ca ấy"
    assert [c.label for c in lan_mot] == [c.label for c in lan_hai]

    for case in lan_mot:
        xac_suat = case.prediction["Xác suất mắc bệnh"]
        assert case.prediction["Tầng nguy cơ"] == classify_risk(xac_suat)["label"], \
            "Tầng ghi trong ca phải khớp với hàm phân tầng"


@pytest.mark.slow
def test_ca_bat_dong_san_dung_phan_khuc_va_lap_lai_duoc():
    lan_mot = bg.select_property_cases()
    lan_hai = bg.select_property_cases()

    assert [c.id for c in lan_mot] == [c.id for c in lan_hai]

    for case in lan_mot:
        gia = case.prediction["Giá định giá (tỷ VNĐ)"]
        assert case.prediction["Phân khúc"] == classify_segment(gia)["label"]
        assert case.prediction["Gói vay"] == classify_segment(gia)["loan"]


@pytest.mark.slow
def test_du_sau_ca_phu_het_moi_tang_va_moi_phan_khuc():
    cases = bg.select_medical_cases() + bg.select_property_cases()

    assert len(cases) == 6
    assert {c.id for c in cases} == {"med_high", "med_moderate", "med_low",
                                     "prop_luxury", "prop_mid", "prop_afford"}


@pytest.mark.slow
def test_du_lieu_vao_cua_ca_khong_con_kieu_numpy():
    """Đây là thứ đi thẳng vào graph.json và vào trình điều khiển Neo4j."""
    import json

    for case in bg.select_medical_cases() + bg.select_property_cases():
        json.dumps(case.inputs)      # ném TypeError nếu còn kiểu NumPy
        json.dumps(case.prediction)
        assert not any(isinstance(v, np.generic) for v in case.inputs.values())
