"""Kiểm thử hai danh mục tri thức miền: bán lẻ dược phẩm và hạ tầng – tài chính.

Trọng tâm không phải là "hàm có chạy không" mà là ba nhóm bất biến dễ vỡ trong
im lặng:

* **Tính toàn vẹn tham chiếu** — mọi mã SKU mà gói chăm sóc trỏ tới đều tồn tại,
  và mọi phân khúc đều có đủ điều kiện vay lẫn cụm tiện ích. Thiếu một mắt xích
  ở đây thì đồ thị tri thức dựng ra vẫn hợp lệ về hình thức nhưng mất món hàng.
* **Tính đúng của phép tính tài chính** — công thức niên kim được đối chiếu với
  một phép tính độc lập, chứ không chỉ so với chính nó.
* **Tính nhất quán giữa hai nguồn khai báo** — chuỗi lãi suất hiển thị cho người
  đọc phải khớp con số dùng để tính trả góp.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kg import catalog_pharmacy as pharm
from src.kg import catalog_urban as urban
from src.kg import ontology as ont


# ==========================================================================
#  DANH MỤC BÁN LẺ DƯỢC PHẨM
# ==========================================================================

def test_ma_sku_khong_trung_nhau():
    ma = [p["sku"] for p in pharm.PRODUCTS]
    assert len(ma) == len(set(ma))


def test_ten_mat_hang_khong_trung_nhau():
    """Tra theo tên là đường đi mà ontology dùng để làm giàu nhãn nút."""
    ten = [p["name"] for p in pharm.PRODUCTS]
    assert len(ten) == len(set(ten))


@pytest.mark.parametrize("sp", pharm.PRODUCTS, ids=lambda p: p["sku"])
def test_moi_mat_hang_du_truong_va_khoang_gia_hop_le(sp):
    bat_buoc = {"sku", "name", "group", "nhom_hang", "unit",
                "gia_min", "gia_max", "cong_dung", "dinh_ky"}
    assert bat_buoc <= set(sp), f"{sp['sku']} thiếu {bat_buoc - set(sp)}"
    assert 0 < sp["gia_min"] <= sp["gia_max"], "Khoảng giá phải dương và không đảo chiều"
    assert sp["group"] in {"Device", "Nutrition"}, \
        "Loại hàng phải trùng loại thực thể mà ontology dùng, nếu không nút mất cạnh cung ứng"


@pytest.mark.parametrize("tier_id", list(pharm.CARE_PACKAGES))
def test_moi_goi_tro_toi_ma_san_pham_co_that(tier_id):
    """Gói trỏ tới mã lạ phải nổ ngay, không được lặng lẽ tạo giỏ hàng thiếu món."""
    for sku in pharm.CARE_PACKAGES[tier_id]["skus"]:
        assert pharm.san_pham(sku)["sku"] == sku


def test_ma_san_pham_khong_ton_tai_thi_nem_loi():
    with pytest.raises(KeyError):
        pharm.san_pham("LC-KHONG-CO")


def test_tra_ten_khong_khop_thi_tra_ve_none_chu_khong_nem():
    """Đường đi làm giàu nhãn được phép thiếu thông tin, không được làm hỏng đồ thị."""
    assert pharm.san_pham_theo_ten("Một mặt hàng không có thật") is None


def test_moi_tang_nguy_co_deu_co_goi_cham_soc():
    for tier in ont.RISK_TIERS:
        assert tier["id"] in pharm.CARE_PACKAGES


def test_chi_phi_hang_thang_khong_vuot_chi_phi_ban_dau():
    """Khoản mua một lần chỉ vào tổng ban đầu, nên ban đầu luôn ≥ hằng tháng."""
    for tier_id in pharm.CARE_PACKAGES:
        cp = pharm.uoc_tinh_chi_phi(tier_id)
        assert cp["ban_dau_min"] >= cp["hang_thang_min"]
        assert cp["ban_dau_max"] >= cp["hang_thang_max"]


def test_chi_phi_bang_tong_gia_cac_mat_hang_trong_goi():
    """Đối chiếu với một phép cộng độc lập thay vì gọi lại chính hàm cần kiểm."""
    for tier_id, goi in pharm.CARE_PACKAGES.items():
        cong_tay = sum(pharm.san_pham(sku)["gia_min"] for sku in goi["skus"])
        assert pharm.uoc_tinh_chi_phi(tier_id)["ban_dau_min"] == cong_tay


def test_duong_dan_tra_cuu_tro_ve_dung_ten_mien_nha_thuoc():
    sp = pharm.san_pham("LC-DEV-001")
    assert sp["tra_cuu"].startswith(pharm.RETAILER["website"])
    assert "Accu-Chek" in sp["tra_cuu"].replace("+", " ").replace("%2B", "+")


def test_ghi_chu_nguon_luon_mang_ngay_chot_gia():
    """Bỏ ngày chốt đi thì khoảng giá đọc như một bản ghi giá sống — sai sự thật."""
    assert pharm.GIA_CHOT_NGAY in pharm.GHI_CHU_NGUON


def test_dinh_dang_tien_dung_dau_cham_ngan_hang_nghin():
    assert pharm.dinh_dang_tien(1_090_000) == "1.090.000 đ"
    assert pharm.dinh_dang_khoang(500, 500) == "500 đ", "Hai đầu bằng nhau thì rút gọn"


# ==========================================================================
#  DANH MỤC HẠ TẦNG ĐÔ THỊ VÀ TÀI CHÍNH
# ==========================================================================

@pytest.mark.parametrize("seg", ont.PROPERTY_SEGMENTS, ids=lambda s: s["id"])
def test_moi_phan_khuc_deu_co_dieu_kien_vay_va_cum_tien_ich(seg):
    assert seg["id"] in urban.FINANCE_TERMS
    assert seg["id"] in urban.UTILITY_CLUSTERS


@pytest.mark.parametrize("cum", urban.UTILITY_CLUSTERS.values(),
                         ids=list(urban.UTILITY_CLUSTERS))
def test_moi_tien_ich_deu_co_phan_loai_chuc_nang(cum):
    hop_le = {"Thương mại", "Giáo dục", "Y tế", "Giao thông"}
    for ti in cum["tien_ich"]:
        assert ti["loai"] in hop_le, f"Phân loại lạ: {ti['loai']}"
    assert cum["ban_kinh_km"] > 0


def test_chuoi_lai_suat_hien_thi_khop_con_so_dung_de_tinh():
    """Hai nguồn khai báo lệch nhau thì người đọc thấy 7,9% mà máy tính theo 8,5%."""
    for seg in ont.PROPERTY_SEGMENTS:
        thuc = urban.FINANCE_TERMS[seg["id"]]["lai_suat_nam"]
        so = re.match(r"([\d,]+)%", seg["rate"])
        assert so, f"Chuỗi lãi suất không đọc được: {seg['rate']}"
        assert float(so.group(1).replace(",", ".")) == pytest.approx(thuc * 100, abs=0.01)
        assert str(urban.FINANCE_TERMS[seg["id"]]["ky_han_nam"]) in seg["tenor"]


def test_tra_gop_khop_phep_tinh_doc_lap():
    """Đối chiếu với công thức niên kim viết lại tay, không gọi hàm cần kiểm."""
    goc, lai_nam, nam = 5.0, 0.084, 20
    r = lai_nam / 12
    n = nam * 12
    mong_doi = goc * r * (1 + r) ** n / ((1 + r) ** n - 1)
    assert urban.tra_gop_hang_thang(goc, lai_nam, nam) == pytest.approx(mong_doi, rel=1e-12)


def test_lai_suat_bang_khong_thi_chia_deu_chu_khong_nem_loi():
    """Mẫu số của công thức niên kim tiến về 0 — nhánh này phải được tách riêng."""
    assert urban.tra_gop_hang_thang(12.0, 0.0, 1) == pytest.approx(1.0)


def test_tong_tien_lai_duong_va_tang_theo_ky_han():
    ngan = urban.tong_tien_lai(5.0, 0.08, 10)
    dai = urban.tong_tien_lai(5.0, 0.08, 25)
    assert 0 < ngan < dai, "Kéo dài kỳ hạn thì tổng lãi phải tăng"


@pytest.mark.parametrize("goc,nam", [(0, 20), (-1, 20), (5, 0), (5, -3)])
def test_tham_so_vo_ly_thi_nem_loi_chu_khong_tra_ve_so(goc, nam):
    with pytest.raises(ValueError):
        urban.tra_gop_hang_thang(goc, 0.08, nam)


@pytest.mark.parametrize("dti", [0, -0.1, 1.5])
def test_nguong_dti_ngoai_khoang_thi_nem_loi(dti):
    with pytest.raises(ValueError):
        urban.thu_nhap_toi_thieu(1.0, dti)


def test_ho_so_tai_chinh_nhat_quan_voi_han_muc_va_von_tu_co():
    gia = 8.42
    seg = ont.classify_segment(gia)
    hs = urban.ho_so_tai_chinh(gia, seg)
    assert hs["han_muc_vay"] == pytest.approx(gia * seg["ltv"])
    assert hs["han_muc_vay"] + hs["von_tu_co"] == pytest.approx(gia)
    assert hs["thu_nhap_toi_thieu"] == pytest.approx(
        hs["tra_gop_thang"] / hs["dti_toi_da"])


def test_ho_so_tai_chinh_dung_phan_khuc_duoc_truyen_vao():
    """Hàm không được tự phân khúc lại — hai nơi cùng quyết định là hai nơi có thể lệch."""
    gia = 8.42
    hs_cao = urban.ho_so_tai_chinh(gia, ont.classify_segment(gia))
    hs_ep = urban.ho_so_tai_chinh(gia, ont.PROPERTY_SEGMENTS[-1])
    assert hs_cao["ltv"] != hs_ep["ltv"]


def test_ty_dong_tu_doi_sang_trieu_khi_so_qua_nho():
    assert "triệu" in urban.ty_dong(0.0475)
    assert "tỷ" in urban.ty_dong(5.894)


# ==========================================================================
#  ĐỒ THỊ TRI THỨC SAU KHI GẮN HAI DANH MỤC
# ==========================================================================

def _kg_y_sinh(prob: float = 0.545):
    return ont.build_medical_kg(
        {"Age": 50, "Glucose": 159.0, "BMI": 30.4, "BloodPressure": 66.0}, prob)


def _kg_bat_dong_san(gia: float = 8.42):
    return ont.build_property_kg(
        {"Area": 100.0, "Floors": 4.0, "Bedrooms": 4.0, "Bathrooms": 5.0,
         "Province": "Hồ Chí Minh", "District": "Nhà Bè",
         "Legal status": "Have certificate"}, gia)


def test_moi_mat_hang_trong_do_thi_deu_mang_ma_sku():
    kg = _kg_y_sinh()
    hang = [n for n in kg.nodes if n.type in {"Device", "Nutrition"}]
    assert hang, "Tầng Nguy cơ Cao phải có mặt hàng"
    for nut in hang:
        assert nut.properties.get("Mã SKU", "").startswith("LC-"), \
            f"Nút {nut.label} không tra được về danh mục bán lẻ"


def test_goi_cham_soc_bao_tron_moi_mat_hang_trong_do_thi():
    kg = _kg_y_sinh()
    hang = {n.id for n in kg.nodes if n.type in {"Device", "Nutrition"}}
    trong_goi = {e.target for e in kg.edges if e.relation == "INCLUDES_PRODUCT"}
    assert hang == trong_goi, "Có mặt hàng nằm ngoài gói — chi phí gói sẽ khai thiếu"


def test_moi_mat_hang_deu_co_noi_cung_ung_sau_khi_doi_sang_nha_ban_le():
    kg = _kg_y_sinh()
    hang = {n.id for n in kg.nodes if n.type in {"Device", "Nutrition"}}
    co_nguon = {e.source for e in kg.edges if e.relation == "SUPPLIED_BY"}
    assert hang <= co_nguon


def test_do_thi_bat_dong_san_du_chuoi_tai_chinh():
    kg = _kg_bat_dong_san()
    quan_he = {e.relation for e in kg.edges}
    assert {"ELIGIBLE_FOR", "AMORTISED_AS", "REQUIRES_INCOME",
            "HAS_INFRASTRUCTURE", "INCLUDES_UTILITY"} <= quan_he


def test_nhan_tra_gop_trong_do_thi_khop_phep_tinh():
    gia = 8.42
    hs = urban.ho_so_tai_chinh(gia, ont.classify_segment(gia))
    kg = _kg_bat_dong_san(gia)
    nut = next(n for n in kg.nodes if n.type == "Repayment")
    assert f"{hs['tra_gop_thang'] * 1000:,.1f}".replace(",", ".") in nut.label


@pytest.mark.parametrize("mien", ["medical", "property"])
def test_moi_loai_thuc_the_deu_thuoc_dung_mot_tang_kien_truc(mien):
    """Một loại nằm ở hai tầng thì sơ đồ kiến trúc vẽ ra tự mâu thuẫn."""
    da_gap: set[str] = set()
    for lop in ont.KG_LAYERS[mien]:
        trung = da_gap & set(lop["types"])
        assert not trung, f"Loại {trung} xuất hiện ở hơn một tầng"
        da_gap |= set(lop["types"])


@pytest.mark.parametrize("mien,dung", [("medical", _kg_y_sinh),
                                       ("property", _kg_bat_dong_san)])
def test_kien_truc_da_tang_phu_het_loai_thuc_the_that(mien, dung):
    """Sơ đồ kiến trúc phải mô tả đúng đồ thị mà mã nguồn thực sự dựng ra."""
    thieu = {n.type for n in dung().nodes} - {
        t for lop in ont.KG_LAYERS[mien] for t in lop["types"]}
    assert not thieu, f"Miền {mien} thiếu tầng cho: {thieu}"


def test_so_thu_tu_tang_lien_tuc_tu_mot():
    for mien, cac_lop in ont.KG_LAYERS.items():
        assert [lop["stt"] for lop in cac_lop] == list(range(1, len(cac_lop) + 1)), \
            f"Số tầng của miền {mien} bị đứt quãng"


def test_layer_of_tra_ve_khong_khi_loai_khong_thuoc_mien():
    assert ont.layer_of("Patient", "medical") == 1
    assert ont.layer_of("Patient", "property") == 0
