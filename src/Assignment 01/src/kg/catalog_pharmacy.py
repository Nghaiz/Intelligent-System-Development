"""Danh mục bán lẻ dược phẩm — lớp hạ tầng dịch vụ của Đồ thị Tri thức Y sinh.

Mô hình học máy trả về một xác suất. Bản thể học ở :mod:`src.kg.ontology` biến
xác suất ấy thành một **tầng nguy cơ** kèm mã bệnh ICD-10. Tệp này đi nốt chặng
cuối: nối tầng nguy cơ với **hàng hoá và dịch vụ có thật, mua được ở đâu, giá
khoảng bao nhiêu** — tức là biến khuyến nghị lâm sàng thành một giỏ hàng cụ thể.

Vì sao là một danh mục biên soạn tay chứ không phải trình thu thập dữ liệu
-------------------------------------------------------------------------
Phương án cào dữ liệu trực tiếp từ website nhà thuốc đã được cân nhắc và bị
loại, vì ba lý do đo được:

1. **Vi phạm yêu cầu R14 về tính tái lập.** Giá bán lẻ thay đổi theo ngày và
   theo chương trình khuyến mãi. Hai lần chạy ``run_pipeline.py`` cách nhau một
   tuần sẽ cho hai bộ số khác nhau, trong khi cả báo cáo được xây trên cam kết
   "chạy lại cho kết quả trùng khít".
2. **Đưa mạng vào đường đi bắt buộc.** Toàn bộ pipeline hiện chỉ chạm mạng ở
   đúng một bước tuỳ chọn (nạp đồ thị lên Neo4j) và bước ấy có đường lui offline.
   Một trình cào đặt trên đường đi chính sẽ làm hỏng cam kết đó.
3. **Phụ thuộc vào cấu trúc HTML của bên thứ ba**, thứ có thể đổi bất kỳ lúc nào
   mà không báo trước.

Cách làm ở đây trung thực hơn về mặt nguồn gốc dữ liệu: **tên sản phẩm là tên
thật đang được bán**, còn **giá là khoảng tham khảo có ghi ngày chốt**, và mỗi
mặt hàng mang theo một đường dẫn tra cứu để người dùng tự đối chiếu giá sống.
Báo cáo và giao diện web đều nói rõ điều này thay vì trình bày con số như một
bản ghi giá trực tiếp.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus


# ==========================================================================
#  NHÀ BÁN LẺ
# ==========================================================================

RETAILER: dict[str, Any] = {
    "id": "retailer_longchau",
    "name": "Nhà thuốc FPT Long Châu",
    "short": "FPT Long Châu",
    "website": "https://nhathuoclongchau.com.vn",
    "hotline": "1800 6928",
    "so_diem_ban": "hơn 1.600 nhà thuốc trên toàn quốc",
    "dich_vu": [
        "Tư vấn Dược sĩ miễn phí tại quầy và qua tổng đài",
        "Đo đường huyết mao mạch miễn phí tại nhà thuốc",
        "Giao hàng tận nơi trong 2 giờ tại nội thành",
    ],
}

# Ngày chốt khoảng giá tham khảo. Mọi con số giá trong tệp này gắn với mốc thời
# gian này; giao diện web và báo cáo đều hiển thị nó kèm theo giá.
GIA_CHOT_NGAY = "26/08/2026"

# Cảnh báo hiển thị bắt buộc — không được lược bỏ ở bất kỳ nơi tiêu thụ nào.
GHI_CHU_NGUON = (
    f"Khoảng giá tham khảo, biên soạn tay từ danh mục công khai, chốt ngày "
    f"{GIA_CHOT_NGAY}. Đây không phải bản ghi giá trực tiếp — bấm vào tên sản "
    f"phẩm để tra giá hiện hành."
)


def _tra_cuu(ten: str) -> str:
    """Đường dẫn tra cứu giá hiện hành cho một mặt hàng.

    Dùng trang tìm kiếm thay vì đường dẫn sản phẩm cố định: mã sản phẩm trên
    website bán lẻ đổi khi hàng được đóng gói lại, còn truy vấn tìm kiếm theo
    tên thì bền hơn nhiều.
    """
    return f"{RETAILER['website']}/tim-kiem?s={quote_plus(ten)}"


# ==========================================================================
#  DANH MỤC MẶT HÀNG
# ==========================================================================
#
# ``group`` phải trùng với loại thực thể dùng trong :mod:`src.kg.ontology`
# ("Device" hoặc "Nutrition"), vì bộ kiểm thử bắt mọi nút thuộc hai loại này
# đều phải có cạnh SUPPLIED_BY trỏ về nhà bán lẻ.
#
# ``gia_min``/``gia_max`` tính bằng **VNĐ**. Khoảng chứ không phải một con số:
# giá bán lẻ dao động theo quy cách đóng gói và theo chương trình khuyến mãi,
# nên một con số đơn lẻ sẽ tạo ra vẻ chính xác giả tạo.

PRODUCTS: list[dict[str, Any]] = [
    # ---- Thiết bị theo dõi ----
    {
        "sku": "LC-DEV-001",
        "name": "Máy đo đường huyết Accu-Chek Instant",
        "group": "Device",
        "nhom_hang": "Thiết bị theo dõi",
        "unit": "máy",
        "gia_min": 790_000,
        "gia_max": 1_090_000,
        "cong_dung": "Đo đường huyết mao mạch tại nhà, cho kết quả sau 4 giây.",
        "dinh_ky": False,
    },
    {
        "sku": "LC-DEV-002",
        "name": "Que thử đường huyết Accu-Chek Instant (hộp 50 que)",
        "group": "Device",
        "nhom_hang": "Vật tư tiêu hao",
        "unit": "hộp",
        "gia_min": 300_000,
        "gia_max": 420_000,
        "cong_dung": "Vật tư đi kèm máy đo; đo hai lần mỗi ngày thì dùng hết trong khoảng 25 ngày.",
        "dinh_ky": True,
    },
    {
        "sku": "LC-DEV-003",
        "name": "Kim chích máu Accu-Chek Softclix (hộp 25 kim)",
        "group": "Device",
        "nhom_hang": "Vật tư tiêu hao",
        "unit": "hộp",
        "gia_min": 90_000,
        "gia_max": 150_000,
        "cong_dung": "Kim lấy máu đầu ngón tay, dùng một lần cho mỗi lần đo.",
        "dinh_ky": True,
    },
    {
        "sku": "LC-DEV-004",
        "name": "Máy đo huyết áp bắp tay Omron HEM-7121",
        "group": "Device",
        "nhom_hang": "Thiết bị theo dõi",
        "unit": "máy",
        "gia_min": 850_000,
        "gia_max": 1_250_000,
        "cong_dung": "Theo dõi huyết áp — biến chứng tim mạch là rủi ro đi kèm của đái tháo đường.",
        "dinh_ky": False,
    },

    # ---- Dinh dưỡng y học ----
    {
        "sku": "LC-NUT-001",
        "name": "Sữa Abbott Glucerna 850g",
        "group": "Nutrition",
        "nhom_hang": "Dinh dưỡng y học",
        "unit": "lon",
        "gia_min": 690_000,
        "gia_max": 890_000,
        "cong_dung": "Sữa chỉ số đường huyết thấp, thay thế bữa phụ cho người đái tháo đường.",
        "dinh_ky": True,
    },
    {
        "sku": "LC-NUT-002",
        "name": "Sữa Nutricare Cerna 850g",
        "group": "Nutrition",
        "nhom_hang": "Dinh dưỡng y học",
        "unit": "lon",
        "gia_min": 480_000,
        "gia_max": 650_000,
        "cong_dung": "Lựa chọn nội địa cùng nhóm công dụng với Glucerna, giá thấp hơn.",
        "dinh_ky": True,
    },
    {
        "sku": "LC-NUT-003",
        "name": "Viên uống Dây thìa canh Diabetna (hộp 40 viên)",
        "group": "Nutrition",
        "nhom_hang": "Thực phẩm bảo vệ sức khoẻ",
        "unit": "hộp",
        "gia_min": 120_000,
        "gia_max": 195_000,
        "cong_dung": "Thực phẩm bảo vệ sức khoẻ hỗ trợ chuyển hoá đường. Không phải thuốc.",
        "dinh_ky": True,
    },
    {
        "sku": "LC-NUT-004",
        "name": "Đường ăn kiêng cỏ ngọt Equal Stevia (hộp 50 gói)",
        "group": "Nutrition",
        "nhom_hang": "Thực phẩm thay thế",
        "unit": "hộp",
        "gia_min": 55_000,
        "gia_max": 95_000,
        "cong_dung": "Chất tạo ngọt không calo, thay đường tinh luyện trong đồ uống.",
        "dinh_ky": True,
    },
    {
        "sku": "LC-NUT-005",
        "name": "Trà khổ qua rừng (hộp 20 túi lọc)",
        "group": "Nutrition",
        "nhom_hang": "Thực phẩm thay thế",
        "unit": "hộp",
        "gia_min": 45_000,
        "gia_max": 85_000,
        "cong_dung": "Trà thảo mộc dùng thay nước ngọt có đường.",
        "dinh_ky": True,
    },
    {
        "sku": "LC-NUT-006",
        "name": "Viên uống Vitamin tổng hợp & Khoáng chất",
        "group": "Nutrition",
        "nhom_hang": "Thực phẩm bảo vệ sức khoẻ",
        "unit": "hộp",
        "gia_min": 150_000,
        "gia_max": 320_000,
        "cong_dung": "Bổ sung vi chất cho người trưởng thành có chế độ ăn cân bằng.",
        "dinh_ky": True,
    },
]

_BY_SKU = {p["sku"]: p for p in PRODUCTS}
_BY_NAME = {p["name"]: p for p in PRODUCTS}


# ==========================================================================
#  GÓI CHĂM SÓC THEO TẦNG NGUY CƠ
# ==========================================================================
#
# Khoá của ánh xạ trùng với ``id`` của ``RISK_TIERS`` trong src/kg/ontology.py.
# Đây là chỗ duy nhất định nghĩa giỏ hàng theo tầng — ontology, ứng dụng web và
# báo cáo đều đọc lại từ đây thay vì mỗi nơi tự khai một danh sách.

CARE_PACKAGES: dict[str, dict[str, Any]] = {
    "tier_high": {
        "id": "pkg_high",
        "title": "Gói Theo dõi Chuyên sâu",
        "muc_tieu": "Đo đường huyết hằng ngày và chuẩn bị cho lần khám chuyên khoa.",
        "skus": ["LC-DEV-001", "LC-DEV-002", "LC-DEV-003",
                 "LC-NUT-001", "LC-NUT-003"],
        "dich_vu": [
            "Tư vấn Dược sĩ miễn phí về cách sử dụng máy đo",
            "Đo đường huyết mao mạch miễn phí tại nhà thuốc",
        ],
        "tan_suat_theo_doi": "Đo đường huyết đói và sau ăn 2 giờ, mỗi ngày",
    },
    "tier_moderate": {
        "id": "pkg_moderate",
        "title": "Gói Dự phòng Tiền Đái tháo đường",
        "muc_tieu": "Kiểm soát chế độ ăn và theo dõi huyết áp trước khi bệnh tiến triển.",
        "skus": ["LC-DEV-004", "LC-NUT-002", "LC-NUT-004", "LC-NUT-005"],
        "dich_vu": [
            "Tư vấn Dược sĩ về chế độ ăn giảm chỉ số đường huyết",
            "Đo đường huyết mao mạch miễn phí tại nhà thuốc",
        ],
        "tan_suat_theo_doi": "Kiểm tra lại đường huyết sau 3 tháng",
    },
    "tier_low": {
        "id": "pkg_low",
        "title": "Gói Duy trì Sức khoẻ",
        "muc_tieu": "Giữ nguyên hiện trạng và tầm soát định kỳ.",
        "skus": ["LC-NUT-006"],
        "dich_vu": [
            "Đo đường huyết mao mạch miễn phí tại nhà thuốc",
        ],
        "tan_suat_theo_doi": "Khám sức khoẻ định kỳ hằng năm",
    },
}


# ==========================================================================
#  TRUY VẤN
# ==========================================================================

def san_pham(sku: str) -> dict[str, Any]:
    """Trả về bản ghi mặt hàng theo mã SKU, kèm đường dẫn tra cứu giá.

    Ném ``KeyError`` khi mã không tồn tại thay vì trả về ``None``: một gói chăm
    sóc trỏ tới mặt hàng không có trong danh mục là lỗi cấu hình, và lỗi ấy phải
    nổ ngay lúc dựng đồ thị chứ không được lặng lẽ tạo ra một giỏ hàng thiếu món.
    """
    if sku not in _BY_SKU:
        raise KeyError(f"Mã sản phẩm không có trong danh mục: {sku}")
    item = dict(_BY_SKU[sku])
    item["tra_cuu"] = _tra_cuu(item["name"])
    return item


def san_pham_theo_ten(ten: str) -> dict[str, Any] | None:
    """Tra mặt hàng theo tên hiển thị. Trả về ``None`` khi không khớp.

    Khác với :func:`san_pham`, hàm này được phép trả về ``None`` vì nó phục vụ
    việc *làm giàu* một nhãn đã có sẵn trong đồ thị; nhãn không khớp danh mục
    thì chỉ đơn giản là không có thông tin giá đi kèm.
    """
    item = _BY_NAME.get(ten)
    if item is None:
        return None
    item = dict(item)
    item["tra_cuu"] = _tra_cuu(item["name"])
    return item


def goi_cham_soc(tier_id: str) -> dict[str, Any]:
    """Gói chăm sóc đầy đủ của một tầng nguy cơ, kèm danh sách mặt hàng và chi phí."""
    if tier_id not in CARE_PACKAGES:
        raise KeyError(f"Không có gói chăm sóc cho tầng: {tier_id}")

    goi = dict(CARE_PACKAGES[tier_id])
    goi["san_pham"] = [san_pham(sku) for sku in goi["skus"]]
    goi["chi_phi"] = uoc_tinh_chi_phi(tier_id)
    goi["ghi_chu_nguon"] = GHI_CHU_NGUON
    return goi


def uoc_tinh_chi_phi(tier_id: str) -> dict[str, int]:
    """Ước tính chi phí gói: khoảng tổng ban đầu và khoảng chi phí duy trì hằng tháng.

    ``dinh_ky=False`` là khoản mua một lần (máy đo), chỉ vào tổng ban đầu.
    ``dinh_ky=True`` là khoản lặp lại, vào cả tổng ban đầu lẫn chi phí hằng tháng.
    Giữ hai con số tách nhau là có chủ đích: gộp chung sẽ khiến gói Theo dõi
    Chuyên sâu trông đắt gấp ba lần thực tế của tháng thứ hai trở đi.
    """
    items = [_BY_SKU[sku] for sku in CARE_PACKAGES[tier_id]["skus"]]
    return {
        "ban_dau_min": sum(i["gia_min"] for i in items),
        "ban_dau_max": sum(i["gia_max"] for i in items),
        "hang_thang_min": sum(i["gia_min"] for i in items if i["dinh_ky"]),
        "hang_thang_max": sum(i["gia_max"] for i in items if i["dinh_ky"]),
    }


def dinh_dang_tien(gia: float) -> str:
    """Định dạng số tiền VNĐ theo quy ước Việt Nam (dấu chấm ngăn hàng nghìn)."""
    return f"{gia:,.0f}".replace(",", ".") + " đ"


def dinh_dang_khoang(gia_min: float, gia_max: float) -> str:
    """Định dạng một khoảng giá. Hai đầu bằng nhau thì rút gọn thành một con số."""
    if gia_min == gia_max:
        return dinh_dang_tien(gia_min)
    return f"{dinh_dang_tien(gia_min)} – {dinh_dang_tien(gia_max)}"
