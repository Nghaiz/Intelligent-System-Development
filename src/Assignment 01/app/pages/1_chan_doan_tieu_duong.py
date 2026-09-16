"""Trang 2 — Chẩn đoán nguy cơ Tiểu đường.

Form tám chỉ số → pipeline đã lưu → xác suất → tầng nguy cơ → gói can thiệp →
đồ thị tri thức dựng tại chỗ. Toàn bộ chuỗi ấy chạy trên **dữ liệu người dùng
vừa nhập**, không phải trên ca mẫu đã tính sẵn.

Bố cục theo nguyên tắc *nhập bên trái, kết quả bên phải, lý giải bên dưới*:
kết quả nằm ngang tầm mắt với form nên người dùng thấy ngay xác suất đổi theo
từng con số mình gõ, thay vì phải cuộn xuống mới biết. Ba tab dưới cùng chứa
phần cần đọc kỹ — khuyến nghị, so sánh mô hình, và lý giải phương pháp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Cho phép chạy thẳng trang này (``streamlit run app/pages/1_...py``) chứ không
# chỉ qua điểm vào ``app/app.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib
from src.kg import catalog_pharmacy as pharm
from src.kg.ontology import classify_risk, build_medical_kg

lib.ap_dung_css()
lib.thanh_dieu_huong()

TOM_TAT = lib.doc_tom_tat()["diabetes"]
COT = TOM_TAT["feature_columns"]

# Ô để trống nghĩa là "chưa đo", đúng quy ước của tập dữ liệu gốc. Ba cột không
# nằm trong nhóm 0-là-khuyết (Số lần mang thai, Chỉ số di truyền, Tuổi) luôn có
# giá trị, nên chúng mang sẵn số mặc định còn năm cột sinh học thì để trống.
MAC_DINH = {
    "Pregnancies": 1.0, "Glucose": None, "BloodPressure": None,
    "SkinThickness": None, "Insulin": None, "BMI": None,
    "DiabetesPedigreeFunction": 0.47, "Age": 33.0,
}

NHAN = {
    "Pregnancies": ("Số lần mang thai", "lần", 0.0, 20.0, 1.0),
    "Glucose": ("Đường huyết (Glucose)", "mg/dL", 0.0, 300.0, 1.0),
    "BloodPressure": ("Huyết áp tâm trương", "mmHg", 0.0, 200.0, 1.0),
    "SkinThickness": ("Độ dày nếp gấp da", "mm", 0.0, 110.0, 1.0),
    "Insulin": ("Insulin huyết thanh", "µU/mL", 0.0, 900.0, 1.0),
    "BMI": ("Chỉ số khối cơ thể (BMI)", "kg/m²", 0.0, 70.0, 0.1),
    "DiabetesPedigreeFunction": ("Chỉ số di truyền", "", 0.0, 3.0, 0.01),
    "Age": ("Tuổi", "năm", 1.0, 120.0, 1.0),
}

# Nhóm tám ô nhập theo *việc người dùng phải làm để có con số đó*, không theo
# thứ tự cột trong tập dữ liệu: ba ô đầu ai cũng tự khai được, bốn ô giữa phải
# đi xét nghiệm, ô cuối hỏi tiền sử gia đình. Thứ tự cột đưa vào mô hình vẫn do
# ``lib.khung_tieu_duong`` quyết định, nên đổi thứ tự hiển thị ở đây là an toàn.
NHOM = [
    ("👤 Thông tin cá nhân", ["Age", "Pregnancies", "BMI"]),
    ("🧪 Kết quả xét nghiệm", ["Glucose", "BloodPressure", "Insulin", "SkinThickness"]),
    ("🧬 Tiền sử gia đình", ["DiabetesPedigreeFunction"]),
]

# Màu của từng tầng nguy cơ — dùng cho vòng tròn, nhãn và danh sách khuyến nghị.
MAU_TANG = {
    "tier_high": lib.MAU["do"],
    "tier_moderate": lib.MAU["cam"],
    "tier_low": lib.MAU["xanh_la"],
}
LOP_CHIP = {"tier_high": "bad", "tier_moderate": "warn", "tier_low": "good"}

for _ten, _gia_tri in MAC_DINH.items():
    st.session_state.setdefault(f"dia_{_ten}", _gia_tri)


def _nap_ca(dau_vao: dict) -> None:
    """Đổ dữ liệu một ca mẫu vào form."""
    for ten in COT:
        st.session_state[f"dia_{ten}"] = (
            None if dau_vao.get(ten) is None else float(dau_vao[ten])
        )


def _xoa_form() -> None:
    """Trả form về mặc định — năm ô sinh học rỗng lại như lúc mới mở trang."""
    for ten, gia_tri in MAC_DINH.items():
        st.session_state[f"dia_{ten}"] = gia_tri


# ==========================================================================
#  ĐẦU TRANG
# ==========================================================================

lib.hero(
    "Chẩn đoán nguy cơ Tiểu đường",
    "Nhập tám chỉ số sinh học, nhận xác suất mắc bệnh cùng gói can thiệp cụ "
    "thể. <b>Ô nào chưa đo thì cứ để trống</b> — pipeline sẽ điền bằng trung vị "
    "học được từ tập huấn luyện, đúng như lúc huấn luyện.",
    nhan="Phân lớp nhị phân · Pima Indians Diabetes",
)

# ---- Ca mẫu ----
ca_mau = lib.ca_theo_mien("medical")
st.markdown("**Chưa có số liệu?** Nạp thử một bệnh nhân có thật trong tập dữ liệu:")
cot_ca = st.columns(len(ca_mau) + 1)
for o, ca in zip(cot_ca, ca_mau):
    o.button(lib.nhan_ca_ngan(ca["label"]), use_container_width=True,
             key=f"nut_{ca['id']}", help=ca["label"],
             on_click=_nap_ca, args=(ca["inputs"],))
cot_ca[-1].button("↺ Xoá form", use_container_width=True, on_click=_xoa_form)

st.markdown("")

# ==========================================================================
#  FORM VÀ KẾT QUẢ, NGANG TẦM MẮT VỚI NHAU
# ==========================================================================

trai, phai = st.columns([3, 2], gap="large")

with trai:
    for tieu_de, ds_cot in NHOM:
        with st.container(border=True):
            st.markdown(f"**{tieu_de}**")
            # Dựng lưới theo **hàng** chứ không theo cột: trên điện thoại các
            # cột xếp chồng lên nhau, nên dựng theo cột sẽ hiện hết cột trái
            # rồi mới tới cột phải — thứ tự trường đảo lộn. Mỗi hàng là một
            # khối riêng thì thứ tự giữ nguyên ở cả hai khổ màn hình.
            for vi_tri in range(0, len(ds_cot), 2):
                hang = st.columns(2)
                for o, ten in zip(hang, ds_cot[vi_tri:vi_tri + 2]):
                    nhan, don_vi, thap, cao, buoc = NHAN[ten]
                    o.number_input(
                        nhan + (f" — {don_vi}" if don_vi else ""),
                        key=f"dia_{ten}", min_value=thap, max_value=cao,
                        step=buoc, placeholder="để trống nếu chưa đo",
                    )

gia_tri = {ten: st.session_state[f"dia_{ten}"] for ten in COT}
khung = lib.khung_tieu_duong(gia_tri)
so_o_trong = sum(1 for v in gia_tri.values() if v is None)

with phai:
    with st.container(border=True):
        # Hộp chọn nằm ngay trên vòng xác suất mà nó chi phối: đổi thuật toán
        # thì con số ngay bên dưới đổi theo, không phải đi tìm.
        danh_muc, chon = lib.chon_mo_hinh(
            "diabetes", lib.DIA_COMPARISON, TOM_TAT["best_model"],
            "Năm thuật toán học theo năm nguyên lý khác nhau; đổi mô hình thì "
            "xác suất đổi theo.",
        )
        mo_hinh = lib.nap_mo_hinh(danh_muc[chon])
        xac_suat = float(mo_hinh.predict_proba(khung)[0, 1])
        tang = classify_risk(xac_suat)
        mau_tang = MAU_TANG[tang["id"]]

        lib.vong_nguy_co(xac_suat, "Xác suất mắc bệnh", mau_tang)
        st.markdown(
            f'<div style="text-align:center;margin:.2rem 0 .8rem">'
            f'<span class="t1-chip {LOP_CHIP[tang["id"]]}"><span class="dot">'
            f'</span>{tang["label"]}</span></div>',
            unsafe_allow_html=True,
        )
        lib.luoi_so([
            {"k": "Mã bệnh ICD-10", "v": tang["icd10"], "s": tang["icd10_name"],
             "mau": mau_tang},
        ])

        if tang["id"] == "tier_high":
            st.error("Cần khám chuyên khoa Nội tiết để xét nghiệm khẳng định.",
                     icon="🚨")
        elif tang["id"] == "tier_moderate":
            st.warning("Tiền đái tháo đường — can thiệp lối sống ngay là kịp.",
                       icon="⚠️")
        else:
            st.success("Chưa thấy dấu hiệu nguy cơ — khám định kỳ hàng năm.",
                       icon="✅")

        if so_o_trong:
            lib.ghi_chu(
                f"Đang có <b>{so_o_trong}</b> chỉ số bỏ trống, được điền bằng "
                "trung vị của tập huấn luyện. Đo đủ thì kết quả sát hơn."
            )

# ==========================================================================
#  BA TAB — KHUYẾN NGHỊ, SO SÁNH, LÝ GIẢI
# ==========================================================================

# Nhãn tab giữ ngắn: trên điện thoại một hàng ba nhãn dài tràn khỏi màn hình
# và người dùng phải kéo ngang mới thấy tab thứ ba tồn tại.
tab_kn, tab_gc, tab_ss, tab_ly = st.tabs(
    ["🧭 Khuyến nghị", "💊 Gói chăm sóc", "📊 So sánh", "🔬 Vì sao"])

# ---- Tab 1: khuyến nghị + đồ thị tri thức ----
with tab_kn:
    st.markdown(
        "Một xác suất tự nó không nói cho người dùng biết **phải làm gì tiếp "
        "theo**. Đồ thị tri thức bên dưới nối kết quả dự báo với tri thức miền "
        "đã chuẩn hoá — mã bệnh ICD-10, thiết bị y tế, dinh dưỡng, hướng dẫn "
        "lâm sàng — và nó được dựng tại chỗ từ đúng những con số bạn vừa nhập."
    )

    kn_trai, kn_phai = st.columns([2, 3], gap="large")

    with kn_trai:
        with st.container(border=True):
            st.markdown(f"### Gói can thiệp — {tang['label']}")
            if tang["devices"]:
                lib.tieu_muc("Thiết bị y tế")
                lib.danh_sach(tang["devices"], mau_tang)
            lib.tieu_muc("Dinh dưỡng")
            lib.danh_sach(tang["nutrition"], mau_tang)
            lib.tieu_muc("Hướng dẫn lâm sàng")
            lib.danh_sach(tang["actions"], mau_tang)

    with kn_phai:
        kg = build_medical_kg(gia_tri, xac_suat,
                              model_name=chon.split(" — ")[-1])
        lib.khoi_do_thi_tri_thuc(kg, chieu_cao=470)

    lib.xem_toan_mien("medical", "Y tế")

    st.info(
        "Đây là công cụ **sàng lọc**, không phải chẩn đoán y khoa. Kết quả chỉ "
        "có giá trị tham khảo và không thay thế kết luận của bác sĩ.",
        icon="⚠️",
    )

# ---- Tab 2: so sánh mô hình ----
with tab_ss:
    bang = lib.du_bao_toan_bo(danh_muc, khung, phan_lop=True)
    bang = bang.sort_values("Kết quả", ascending=False)
    chenh = float(bang["Kết quả"].max() - bang["Kết quả"].min())

    st.markdown("### Cùng một bệnh nhân, năm ranh giới quyết định")
    lib.bang_xep_hang(bang["Mô hình"].tolist(), bang["Kết quả"].tolist(),
                      lambda v: f"{v:.1%}", lib.MAU["do"],
                      danh_dau=TOM_TAT["best_model"])

    lib.luoi_so([
        {"k": "Chênh lệch cao – thấp", "v": f"{chenh:.1%}",
         "s": "trên cùng một vector đặc trưng", "mau": lib.MAU["do"]},
        {"k": "Đồng thuận về tầng nguy cơ",
         "v": f"{bang['Kết quả'].map(lambda p: classify_risk(p)['label']).nunique()}"
              " / 3 tầng",
         "s": "số tầng khác nhau mà năm mô hình rơi vào",
         "mau": lib.MAU["brand"]},
    ])

    st.markdown(
        "Khác biệt ấy **không phải nhiễu**. Mỗi thuật toán vẽ một ranh giới "
        "quyết định khác nhau trên cùng một không gian đặc trưng, nên cùng "
        "một bệnh nhân nằm ở hai phía khác nhau của hai ranh giới là chuyện "
        "bình thường — và cũng là lý do phải chọn mô hình bằng thực nghiệm "
        "chứ không bằng cảm tính."
    )

    with st.expander("Bảng chỉ số của năm mô hình trên tập kiểm tra"):
        st.dataframe(lib.doc_bang(lib.DIA_COMPARISON).round(4),
                     hide_index=True, use_container_width=True)

# ---- Tab 3: lý giải phương pháp ----
with tab_ly:
    st.markdown("### Vector đặc trưng thực sự đưa vào mô hình")
    st.markdown(
        "Đây là **tám cột thô** sau khi áp quy ước *0 là khuyết*. Bốn đặc trưng "
        "phái sinh (`Glucose × BMI`, nhóm đường huyết, nhóm BMI, nhóm tuổi), "
        "phần điền khuyết và phần chuẩn hoá nằm bên trong pipeline nên không "
        "hiện ở đây — đó chính là điều bảo đảm biểu diễn lúc dự đoán khớp biểu "
        "diễn lúc huấn luyện."
    )
    st.dataframe(khung, hide_index=True, use_container_width=True)

    st.markdown("### Ba điều quyết định con số bạn vừa thấy")

    with st.expander("① Vì sao ô để trống lại tốt hơn gõ số 0"):
        st.markdown(
            "Tập Pima quy ước số 0 ở năm cột sinh học nghĩa là *không đo được*, "
            "và mô hình đã học theo quy ước ấy. Gõ 0 vào ô Huyết áp mà không "
            "đổi thành khuyết thì lúc dự đoán mô hình nhận một bệnh nhân có "
            "huyết áp bằng 0 — một biểu diễn nó chưa từng thấy lúc huấn luyện, "
            "và nó vẫn trả về một con số, chỉ là con số sai."
        )
        st.json({"Số quan sát bị mã hoá bằng 0 trong tập gốc":
                 TOM_TAT["zero_as_missing"]}, expanded=True)

    with st.expander("② Vì sao ngưỡng phân tầng thấp hơn 0.50"):
        st.markdown(
            f"Ngưỡng quyết định tốt nhất đo được ở thực nghiệm quét θ là "
            f"**{TOM_TAT['best_threshold']}**, thấp hơn hẳn mức mặc định 0.50. "
            "Lý do nằm ở mục tiêu: đây là bài toán **sàng lọc**, nên bỏ sót một "
            "người bệnh tốn kém hơn nhiều so với gọi nhầm một người khoẻ đi xét "
            "nghiệm lại. Hạ ngưỡng làm Recall tăng từ 0.458 lên 0.854."
        )
        tep = lib.cfg.OUTPUTS_DIR / "dia_threshold_sweep.csv"
        if tep.exists():
            st.dataframe(lib.doc_bang(tep.name).round(4), hide_index=True,
                         use_container_width=True, height=260)

    with st.expander("③ Vì sao Accuracy không phải độ đo đáng tin ở đây"):
        st.markdown(
            f"Mô hình cơ sở chỉ đoán lớp đa số đạt Accuracy "
            f"**{TOM_TAT['baseline']['Accuracy']:.4f}** — nghe cao, nhưng Recall "
            f"bằng **{TOM_TAT['baseline']['Recall']:.0f}**: nó bỏ sót *toàn bộ* "
            "người bệnh. Với tập dữ liệu lệch lớp "
            f"({TOM_TAT['class_balance']['0']} âm tính so với "
            f"{TOM_TAT['class_balance']['1']} dương tính), Accuracy thưởng cho "
            "việc đoán bừa theo lớp đông, nên F1-Score và Recall mới là độ đo "
            "phản ánh đúng mục tiêu lâm sàng."
        )


# ---- Tab 4: gói chăm sóc sức khoẻ và giải pháp dược phẩm ----
with tab_gc:
    goi = pharm.goi_cham_soc(tang["id"])
    chi_phi = goi["chi_phi"]

    st.markdown(
        f"Tầng nguy cơ ở trên mới là một **kết luận**. Phần này là **việc phải "
        f"mua và phải làm**: {len(goi['san_pham'])} mặt hàng đang bán tại "
        f"{pharm.RETAILER['name']}, kèm chi phí ước tính để người dùng biết "
        f"lời khuyên vừa nhận tốn bao nhiêu trước khi quyết định làm theo."
    )

    st.markdown(f"### {goi['title']}")
    lib.ghi_chu(f"<b>Mục tiêu:</b> {lib._e(goi['muc_tieu'])}")

    lib.luoi_so([
        {"k": "Chi phí ban đầu",
         "v": pharm.dinh_dang_khoang(chi_phi["ban_dau_min"], chi_phi["ban_dau_max"]),
         "s": "gồm cả khoản mua một lần", "mau": mau_tang},
        {"k": "Duy trì hằng tháng",
         "v": (pharm.dinh_dang_khoang(chi_phi["hang_thang_min"],
                                      chi_phi["hang_thang_max"])
               if chi_phi["hang_thang_max"] else "Không phát sinh"),
         "s": "từ tháng thứ hai trở đi", "mau": mau_tang},
        {"k": "Tần suất theo dõi", "v": goi["tan_suat_theo_doi"],
         "s": "theo hướng dẫn của tầng nguy cơ", "mau": mau_tang},
    ])

    lib.tieu_muc("Giải pháp dược phẩm khuyến nghị")
    lib.the_san_pham(goi["san_pham"], mau_tang)

    gc_trai, gc_phai = st.columns([3, 2], gap="large")

    with gc_trai:
        lib.tieu_muc("Dịch vụ đi kèm tại nhà thuốc")
        lib.danh_sach(goi["dich_vu"], mau_tang)
        lib.tieu_muc("Hướng dẫn lâm sàng của tầng nguy cơ")
        lib.danh_sach(tang["actions"], mau_tang)

    with gc_phai:
        with st.container(border=True):
            st.markdown(f"### {pharm.RETAILER['short']}")
            lib.bang_dieu_khoan([
                ("Hệ thống điểm bán", pharm.RETAILER["so_diem_ban"]),
                ("Tổng đài", pharm.RETAILER["hotline"]),
                ("Chi phí ban đầu của gói",
                 pharm.dinh_dang_khoang(chi_phi["ban_dau_min"],
                                        chi_phi["ban_dau_max"])),
                ("Chi phí 12 tháng đầu",
                 pharm.dinh_dang_khoang(
                     chi_phi["ban_dau_min"] + chi_phi["hang_thang_min"] * 11,
                     chi_phi["ban_dau_max"] + chi_phi["hang_thang_max"] * 11)),
            ], mau_tang, nhan_manh={"Chi phí 12 tháng đầu"})
            st.link_button("Mở website nhà thuốc ↗", pharm.RETAILER["website"],
                           use_container_width=True)

    st.warning(
        f"{pharm.GHI_CHU_NGUON} Đây là gợi ý mua sắm theo tầng nguy cơ, "
        "**không phải đơn thuốc**. Thuốc kê đơn — Metformin và nhóm hạ đường "
        "huyết khác — không nằm trong danh mục này và chỉ được dùng khi có chỉ "
        "định của bác sĩ.",
        icon="💊",
    )
