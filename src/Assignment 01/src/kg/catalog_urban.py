"""Danh mục hạ tầng đô thị và gói tài chính — lớp ứng dụng của Đồ thị Bất động sản.

Mô hình hồi quy trả về một con số: 8,42 tỷ VNĐ. Con số ấy chưa trả lời được câu
hỏi mà người mua nhà thật sự hỏi — *vay được bao nhiêu, trả góp mỗi tháng bao
nhiêu, thu nhập bao nhiêu thì gánh nổi, và quanh đó có gì*. Tệp này khai báo hai
nhóm tri thức miền để trả lời đúng những câu ấy:

* **Hạ tầng đô thị** — cụm tiện ích đặc trưng theo phân khúc, có phân loại
  (giáo dục / y tế / thương mại / giao thông) và bán kính tiếp cận tham khảo.
* **Gói tài chính** — điều kiện vay theo phân khúc, kèm phép tính trả góp thật
  bằng công thức niên kim, không phải con số ước lượng bằng tay.

Ranh giới trung thực về số liệu
-------------------------------
Danh sách tiện ích là **mô hình hoá theo phân khúc**, không phải dữ liệu khảo
sát thực địa từng quận: tập dữ liệu gốc chỉ có địa chỉ hành chính, không có toạ
độ, nên không thể suy ra khoảng cách thật tới một siêu thị cụ thể. Điều kiện vay
là **minh hoạ theo mặt bằng lãi suất công bố**, không phải chào giá của một ngân
hàng cụ thể. Cả hai điều này được nói rõ trên giao diện web và trong báo cáo.

Phép tính trả góp thì ngược lại — nó là toán học thuần tuý và kiểm chứng được:
xem :func:`tra_gop_hang_thang`.
"""

from __future__ import annotations

from typing import Any


# ==========================================================================
#  HẠ TẦNG ĐÔ THỊ
# ==========================================================================
#
# Khoá trùng với ``id`` của ``PROPERTY_SEGMENTS`` trong src/kg/ontology.py.
# ``ban_kinh_km`` là bán kính tiếp cận tham khảo của cụm tiện ích, dùng để trả
# lời câu "quanh đó có gì trong vòng mấy cây số" — con số mô hình hoá, không
# phải kết quả đo đạc.

UTILITY_CLUSTERS: dict[str, dict[str, Any]] = {
    "seg_luxury": {
        "id": "infra_luxury",
        "label": "Cụm Hạ tầng Đô thị Cao cấp",
        "mo_ta": "Khu vực đã đô thị hoá hoàn chỉnh, hạ tầng dịch vụ đầy đủ.",
        "ban_kinh_km": 2.0,
        "tien_ich": [
            {"ten": "Trung tâm thương mại", "loai": "Thương mại"},
            {"ten": "Trường quốc tế", "loai": "Giáo dục"},
            {"ten": "Bệnh viện đa khoa", "loai": "Y tế"},
            {"ten": "Tuyến đường vành đai", "loai": "Giao thông"},
        ],
    },
    "seg_mid": {
        "id": "infra_mid",
        "label": "Cụm Hạ tầng Đô thị Trung cấp",
        "mo_ta": "Khu dân cư ổn định, đủ tiện ích thiết yếu trong bán kính đi bộ.",
        "ban_kinh_km": 1.5,
        "tien_ich": [
            {"ten": "Siêu thị", "loai": "Thương mại"},
            {"ten": "Trường công lập", "loai": "Giáo dục"},
            {"ten": "Trạm y tế phường", "loai": "Y tế"},
            {"ten": "Tuyến xe buýt nội đô", "loai": "Giao thông"},
        ],
    },
    "seg_afford": {
        "id": "infra_afford",
        "label": "Cụm Hạ tầng Đô thị Cơ bản",
        "mo_ta": "Khu vực đang đô thị hoá, tiện ích ở mức thiết yếu.",
        "ban_kinh_km": 3.0,
        "tien_ich": [
            {"ten": "Chợ dân sinh", "loai": "Thương mại"},
            {"ten": "Trường tiểu học", "loai": "Giáo dục"},
            {"ten": "Phòng khám đa khoa khu vực", "loai": "Y tế"},
            {"ten": "Tuyến xe buýt liên huyện", "loai": "Giao thông"},
        ],
    },
}

# Cảnh báo hiển thị bắt buộc, song song với GHI_CHU_NGUON của danh mục dược phẩm.
GHI_CHU_TIEN_ICH = (
    "Cụm tiện ích được mô hình hoá theo phân khúc giá, không phải kết quả khảo "
    "sát thực địa từng quận — tập dữ liệu gốc chỉ có địa chỉ hành chính, không "
    "có toạ độ."
)

GHI_CHU_TAI_CHINH = (
    "Điều kiện vay là minh hoạ theo mặt bằng lãi suất công bố, không phải chào "
    "giá của một ngân hàng cụ thể. Số tiền trả góp được tính bằng công thức niên "
    "kim từ chính các tham số hiển thị."
)


# ==========================================================================
#  GÓI TÀI CHÍNH
# ==========================================================================
#
# ``lai_suat_nam`` là lãi suất danh nghĩa dùng cho phép tính trả góp; nó phải
# khớp với con số trong trường ``rate`` mô tả bằng lời của PROPERTY_SEGMENTS.
# ``dti_toi_da`` là tỷ lệ nợ trên thu nhập mà ngân hàng thường chấp nhận — dùng
# để suy ngược ra mức thu nhập tối thiểu cần có.

FINANCE_TERMS: dict[str, dict[str, Any]] = {
    "seg_luxury": {
        "lai_suat_nam": 0.085,
        "ky_han_nam": 25,
        "dti_toi_da": 0.40,
        "phi_trong_ly": "0,5% dư nợ nếu trả trước hạn trong 3 năm đầu",
    },
    "seg_mid": {
        "lai_suat_nam": 0.079,
        "ky_han_nam": 20,
        "dti_toi_da": 0.40,
        "phi_trong_ly": "0,5% dư nợ nếu trả trước hạn trong 2 năm đầu",
    },
    "seg_afford": {
        "lai_suat_nam": 0.048,
        "ky_han_nam": 25,
        "dti_toi_da": 0.45,
        "phi_trong_ly": "Miễn phí trả trước hạn theo chương trình hỗ trợ",
    },
}


# ==========================================================================
#  PHÉP TÍNH TÀI CHÍNH
# ==========================================================================

def tra_gop_hang_thang(goc: float, lai_suat_nam: float, ky_han_nam: int) -> float:
    """Số tiền trả góp cố định mỗi tháng theo công thức niên kim.

    .. math::
        M = P \\cdot \\frac{r}{1 - (1+r)^{-n}}

    với :math:`P` là dư nợ gốc, :math:`r` là lãi suất **tháng**
    (``lai_suat_nam / 12``) và :math:`n` là tổng số kỳ trả.

    Đơn vị của kết quả trùng với đơn vị của ``goc`` (ở đây là tỷ VNĐ).

    Trường hợp ``lai_suat_nam == 0`` được tách riêng: công thức trên có mẫu số
    tiến về 0 nên chia thẳng sẽ ném ``ZeroDivisionError``, trong khi câu trả lời
    đúng về mặt tài chính đơn giản là chia đều gốc cho số kỳ.
    """
    if goc <= 0 or ky_han_nam <= 0:
        raise ValueError("Dư nợ gốc và kỳ hạn phải là số dương")

    so_ky = ky_han_nam * 12
    if lai_suat_nam == 0:
        return goc / so_ky

    r = lai_suat_nam / 12
    return goc * r / (1 - (1 + r) ** (-so_ky))


def tong_tien_lai(goc: float, lai_suat_nam: float, ky_han_nam: int) -> float:
    """Tổng tiền lãi phải trả trong toàn bộ kỳ hạn.

    Con số này thường lớn hơn nhiều so với trực giác của người vay — với kỳ hạn
    25 năm ở lãi suất 8,5%/năm, tiền lãi vượt cả dư nợ gốc. Đó chính là lý do nó
    được hiển thị tường minh thay vì chỉ khoe mỗi con số trả góp hằng tháng.
    """
    thang = tra_gop_hang_thang(goc, lai_suat_nam, ky_han_nam)
    return thang * ky_han_nam * 12 - goc


def thu_nhap_toi_thieu(tra_gop: float, dti_toi_da: float) -> float:
    """Thu nhập hằng tháng tối thiểu để khoản trả góp nằm trong ngưỡng DTI."""
    if not 0 < dti_toi_da <= 1:
        raise ValueError("Tỷ lệ DTI phải nằm trong khoảng (0, 1]")
    return tra_gop / dti_toi_da


def ho_so_tai_chinh(gia: float, segment: dict[str, Any]) -> dict[str, Any]:
    """Hồ sơ tài chính đầy đủ cho một bất động sản đã được định giá.

    Tham số ``segment`` là bản ghi phân khúc lấy từ
    :func:`src.kg.ontology.classify_segment`, nên hàm này không tự phân khúc lại
    — tránh việc hai nơi cùng quyết định một điều và có cơ hội lệch nhau.
    """
    terms = FINANCE_TERMS[segment["id"]]
    han_muc = gia * segment["ltv"]
    von_tu_co = gia - han_muc
    thang = tra_gop_hang_thang(han_muc, terms["lai_suat_nam"], terms["ky_han_nam"])

    return {
        "gia_dinh_gia": gia,
        "ltv": segment["ltv"],
        "han_muc_vay": han_muc,
        "von_tu_co": von_tu_co,
        "lai_suat_nam": terms["lai_suat_nam"],
        "ky_han_nam": terms["ky_han_nam"],
        "tra_gop_thang": thang,
        "tong_tien_lai": tong_tien_lai(han_muc, terms["lai_suat_nam"], terms["ky_han_nam"]),
        "dti_toi_da": terms["dti_toi_da"],
        "thu_nhap_toi_thieu": thu_nhap_toi_thieu(thang, terms["dti_toi_da"]),
        "phi_trong_ly": terms["phi_trong_ly"],
        "ghi_chu": GHI_CHU_TAI_CHINH,
    }


def cum_tien_ich(segment_id: str) -> dict[str, Any]:
    """Cụm hạ tầng đô thị tương ứng một phân khúc."""
    if segment_id not in UTILITY_CLUSTERS:
        raise KeyError(f"Không có cụm tiện ích cho phân khúc: {segment_id}")
    cum = dict(UTILITY_CLUSTERS[segment_id])
    cum["ghi_chu"] = GHI_CHU_TIEN_ICH
    return cum


def ty_dong(gia: float) -> str:
    """Định dạng một số tiền tính bằng tỷ VNĐ, tự đổi sang triệu khi quá nhỏ."""
    if abs(gia) < 0.1:
        return f"{gia * 1000:,.1f} triệu đ".replace(",", ".")
    return f"{gia:,.2f} tỷ đ".replace(",", ".")
