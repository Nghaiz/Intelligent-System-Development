"""Trang 3 — Định giá Bất động sản.

Form thuộc tính nhà → pipeline đã lưu → giá dự báo → phân khúc thị trường →
gói vay ngân hàng → đồ thị tri thức dựng tại chỗ.

Khác trang chẩn đoán ở một chỗ đáng nhớ: ba đặc trưng phái sinh của nhà đất
được tính **ngoài** pipeline lúc huấn luyện, nên ``lib.khung_bat_dong_san``
phải gọi lại ``prep.add_housing_features`` trước khi đưa vào mô hình.

Bố cục giữ đúng khung của trang chẩn đoán — nhập bên trái, kết quả bên phải, ba
tab lý giải bên dưới — để người đã dùng một công cụ không phải học lại công cụ
kia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib
from src.kg import catalog_urban as urban
from src.kg.ontology import PROPERTY_SEGMENTS, classify_segment, build_property_kg

lib.ap_dung_css()
lib.thanh_dieu_huong()

TOM_TAT = lib.doc_tom_tat()["housing"]
DANH_MUC_HANG_MUC = TOM_TAT["categories"]
GOI_Y = lib.mac_dinh_nha_dat()
QUAN = lib.quan_theo_tinh()

NHAN_SO = {
    "Area": ("Diện tích", "m²", 1.0, 2000.0, 1.0),
    "Frontage": ("Mặt tiền", "m", 0.0, 200.0, 0.5),
    "Access Road": ("Đường vào", "m", 0.0, 200.0, 0.5),
    "Floors": ("Số tầng", "tầng", 1.0, 30.0, 1.0),
    "Bedrooms": ("Số phòng ngủ", "phòng", 0.0, 30.0, 1.0),
    "Bathrooms": ("Số phòng tắm", "phòng", 0.0, 30.0, 1.0),
}

NHAN_HANG_MUC = {
    "House direction": "Hướng nhà",
    "Legal status": "Tình trạng pháp lý",
    "Furniture state": "Nội thất",
}

# Ưu tiên hai thành phố lớn lên đầu hộp chọn Tỉnh/Thành: chúng chiếm phần lớn
# tin đăng trong tập dữ liệu nên cũng là lựa chọn thường gặp nhất.
TINH = sorted(QUAN, key=lambda t: (t not in {"Hà Nội", "Hồ Chí Minh"}, t))

# Màu của từng phân khúc — dùng cho thước ngang, nhãn và danh sách tiện ích.
MAU_PHAN_KHUC = {
    "seg_luxury": lib.MAU["brand_dam"],
    "seg_mid": lib.MAU["brand"],
    "seg_afford": lib.MAU["ngoc"],
}

# Ba vùng của thước giá, xếp tăng dần theo ngưỡng dưới — đọc ngược từ chính
# ``PROPERTY_SEGMENTS`` để thước không lệch khỏi ngưỡng phân khúc thật.
MOC_THUOC = [
    (s["min_price"], s["label"].replace("Phân khúc ", ""), MAU_PHAN_KHUC[s["id"]])
    for s in sorted(PROPERTY_SEGMENTS, key=lambda s: s["min_price"])
]
TRAN_THUOC = 16.0        # mép phải của thước, tính bằng tỷ VNĐ

for _ten in lib.HOU_NUMERIC:
    st.session_state.setdefault(f"hou_{_ten}", float(GOI_Y[_ten]))
st.session_state.setdefault("hou_Province", "Hà Nội" if "Hà Nội" in TINH else TINH[0])
st.session_state.setdefault("hou_District",
                            QUAN[st.session_state["hou_Province"]][0])
for _ten in NHAN_HANG_MUC:
    st.session_state.setdefault(f"hou_{_ten}", "Không rõ")


def _nap_ca(dau_vao: dict) -> None:
    """Đổ dữ liệu một tin đăng mẫu vào form.

    Đặt Tỉnh/Thành **trước** Quận/Huyện: danh sách quận được lọc theo tỉnh, nên
    ghi quận trước thì nó rơi ra ngoài danh sách của tỉnh cũ và Streamlit ném lỗi.
    """
    tinh = dau_vao.get("Province") or "Không rõ"
    st.session_state["hou_Province"] = tinh if tinh in QUAN else TINH[0]

    quan = dau_vao.get("District") or "Không rõ"
    hop_le = QUAN[st.session_state["hou_Province"]]
    st.session_state["hou_District"] = quan if quan in hop_le else hop_le[0]

    for ten in lib.HOU_NUMERIC:
        gia_tri = dau_vao.get(ten)
        st.session_state[f"hou_{ten}"] = (float(GOI_Y[ten]) if gia_tri is None
                                          else float(gia_tri))
    for ten in NHAN_HANG_MUC:
        gia_tri = dau_vao.get(ten) or "Không rõ"
        st.session_state[f"hou_{ten}"] = (gia_tri if gia_tri in DANH_MUC_HANG_MUC[ten]
                                          else "Không rõ")


def _xoa_form() -> None:
    """Trả form về trung vị của tập dữ liệu và Hà Nội."""
    _nap_ca({})


# ==========================================================================
#  ĐẦU TRANG
# ==========================================================================

lib.hero(
    "Định giá Bất động sản",
    "Khai diện tích, vị trí và pháp lý, nhận giá dự báo theo <b>tỷ VNĐ</b> cùng "
    "gói vay tương ứng. Ô định lượng để sẵn <b>trung vị của tập dữ liệu</b> — "
    "cũng đúng là con số mô hình đã thấy ở những tin đăng không khai báo "
    "trường đó.",
    nhan="Hồi quy · 30.229 tin đăng bất động sản Việt Nam",
)

# ---- Ca mẫu ----
ca_mau = lib.ca_theo_mien("property")
st.markdown("**Chưa có số liệu?** Nạp thử một tin đăng có thật trong tập dữ liệu:")
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
    with st.container(border=True):
        st.markdown("**📐 Quy mô và kết cấu**")
        # Dựng theo hàng, cùng lý do như form chẩn đoán: trên điện thoại các
        # cột xếp chồng, dựng theo cột thì thứ tự trường đảo lộn.
        for vi_tri in range(0, len(lib.HOU_NUMERIC), 2):
            hang = st.columns(2)
            for o, ten in zip(hang, lib.HOU_NUMERIC[vi_tri:vi_tri + 2]):
                nhan, don_vi, thap, cao, buoc = NHAN_SO[ten]
                o.number_input(f"{nhan} — {don_vi}", key=f"hou_{ten}",
                               min_value=thap, max_value=cao, step=buoc)

    with st.container(border=True):
        st.markdown("**📍 Vị trí hành chính**")
        v1, v2 = st.columns(2)
        v1.selectbox("Tỉnh/Thành", TINH, key="hou_Province")

        # Danh sách quận đọc lại sau khi tỉnh đã chọn; nếu quận đang giữ không
        # thuộc tỉnh mới thì lùi về phần tử đầu thay vì để Streamlit ném lỗi.
        quan_hop_le = QUAN[st.session_state["hou_Province"]]
        if st.session_state["hou_District"] not in quan_hop_le:
            st.session_state["hou_District"] = quan_hop_le[0]
        v2.selectbox("Quận/Huyện", quan_hop_le, key="hou_District")

    with st.container(border=True):
        st.markdown("**📄 Pháp lý và bàn giao**")
        p1, p2, p3 = st.columns(3)
        for o, ten in zip([p1, p2, p3], NHAN_HANG_MUC):
            o.selectbox(NHAN_HANG_MUC[ten], DANH_MUC_HANG_MUC[ten],
                        key=f"hou_{ten}")

gia_tri = {ten: st.session_state[f"hou_{ten}"]
           for ten in lib.HOU_NUMERIC + lib.HOU_CATEGORICAL}
khung = lib.khung_bat_dong_san(gia_tri)

with phai:
    with st.container(border=True):
        # Hộp chọn nằm ngay trên con số nó sinh ra: đổi thuật toán thì giá bên
        # dưới đổi theo, không phải đi tìm xem cái gì vừa thay đổi.
        danh_muc, chon = lib.chon_mo_hinh(
            "housing", lib.HOU_COMPARISON, TOM_TAT["best_model"],
            "Mô hình tuyến tính và mô hình cây cho ra hai mức giá khác nhau "
            "trên cùng một căn nhà.",
        )
        mo_hinh = lib.nap_mo_hinh(danh_muc[chon])
        gia = float(mo_hinh.predict(khung)[0])
        phan_khuc = classify_segment(gia)
        han_muc = gia * phan_khuc["ltv"]
        mau_pk = MAU_PHAN_KHUC[phan_khuc["id"]]
        don_gia = gia * 1000 / max(gia_tri["Area"], 1)

        lib.so_lon(f"{gia:.2f}", "tỷ VNĐ — giá dự báo")
        lib.thuoc_phan_khuc(gia, MOC_THUOC, TRAN_THUOC)
        st.markdown(
            f'<div style="text-align:center;margin:.7rem 0 .3rem">'
            f'<span class="t1-chip"><span class="dot"></span>'
            f'{phan_khuc["label"]}</span></div>',
            unsafe_allow_html=True,
        )
        lib.luoi_so([
            {"k": "Đơn giá", "v": f"{don_gia:.1f}", "s": "triệu đồng mỗi m²",
             "mau": mau_pk},
            {"k": "Hạn mức vay", "v": f"{han_muc:.2f} tỷ",
             "s": f"LTV {phan_khuc['ltv']:.0%}", "mau": lib.MAU["cam"]},
        ])

        if gia_tri["Legal status"] == "Have certificate":
            st.success("Đã có sổ — đủ điều kiện thế chấp.", icon="✅")
        else:
            st.warning("Chưa có sổ đỏ/sổ hồng — phần lớn ngân hàng sẽ không "
                       "nhận thế chấp tài sản này.", icon="⚠️")

# ==========================================================================
#  BA TAB — GÓI VAY, SO SÁNH, LÝ GIẢI
# ==========================================================================

# Nhãn tab giữ ngắn, cùng lý do như trang chẩn đoán.
tab_kn, tab_ht, tab_ss, tab_ly = st.tabs(
    ["🧭 Gói vay", "🏙️ Hạ tầng & Tài chính", "📊 So sánh", "🔬 Vì sao"])

# ---- Tab 1: gói vay + đồ thị tri thức ----
with tab_kn:
    st.markdown(
        "Một mức giá tự nó chưa trả lời được câu hỏi thật của người mua: **vay "
        "được bao nhiêu, và quanh đó có gì**. Đồ thị tri thức bên dưới nối giá "
        "dự báo với tri thức miền đã chuẩn hoá — phân khúc thị trường, gói tín "
        "dụng ngân hàng, tiện ích đô thị — dựng tại chỗ từ đúng những con số "
        "bạn vừa khai."
    )

    kn_trai, kn_phai = st.columns([2, 3], gap="large")

    with kn_trai:
        with st.container(border=True):
            st.markdown(f"### {phan_khuc['loan']}")
            lib.luoi_so([
                {"k": "Hạn mức ước tính", "v": f"{han_muc:.2f} tỷ VNĐ",
                 "s": f"LTV {phan_khuc['ltv']:.0%} trên giá dự báo",
                 "mau": mau_pk},
            ])
            lib.tieu_muc("Điều kiện vay")
            lib.danh_sach([f"Lãi suất {phan_khuc['rate']}",
                           f"Thời hạn {phan_khuc['tenor']}",
                           f"Vốn tự có tối thiểu "
                           f"{gia - han_muc:.2f} tỷ VNĐ "
                           f"({1 - phan_khuc['ltv']:.0%} giá trị)"], mau_pk)
            lib.tieu_muc("Tiện ích đô thị của phân khúc")
            lib.danh_sach(phan_khuc["utilities"], mau_pk)

    with kn_phai:
        kg = build_property_kg(gia_tri, gia, model_name=chon.split(" — ")[-1])
        lib.khoi_do_thi_tri_thuc(kg, chieu_cao=470)

    lib.xem_toan_mien("property", "Bất động sản")

    st.info(
        f"Giá dự báo mang sai số trung bình khoảng "
        f"{lib.doc_bang(lib.HOU_COMPARISON).set_index('Model').loc[TOM_TAT['best_model'], 'MAE']:.2f}"
        " tỷ VNĐ trên tập kiểm tra. Con số ở đây là tham khảo, không thay thế "
        "thẩm định giá chính thức.",
        icon="⚠️",
    )

# ---- Tab 2: so sánh mô hình ----
with tab_ss:
    bang = lib.du_bao_toan_bo(danh_muc, khung, phan_lop=False)
    bang = bang.sort_values("Kết quả", ascending=False)
    chenh = float(bang["Kết quả"].max() - bang["Kết quả"].min())

    st.markdown("### Cùng một căn nhà, năm cách vẽ mặt định giá")
    lib.bang_xep_hang(bang["Mô hình"].tolist(), bang["Kết quả"].tolist(),
                      lambda v: f"{v:.2f} tỷ", lib.MAU["cam"],
                      danh_dau=TOM_TAT["best_model"])

    lib.luoi_so([
        {"k": "Khoảng cách cao – thấp", "v": f"{chenh:.2f} tỷ VNĐ",
         "s": "trên cùng một vector đặc trưng", "mau": lib.MAU["cam"]},
        {"k": "Số phân khúc bị rơi vào",
         "v": f"{bang['Kết quả'].map(lambda g: classify_segment(g)['label']).nunique()}"
              " / 3",
         "s": "năm mô hình có đồng ý về phân khúc không",
         "mau": lib.MAU["brand"]},
    ])

    st.markdown(
        "Hồi quy tuyến tính chỉ dựng được **một mặt phẳng** nên thường kéo "
        "giá về gần trung bình, trong khi mô hình cây bắt được các ngưỡng "
        "phi tuyến của diện tích và vị trí — đó là lý do R² của hai nhóm "
        "chênh nhau rõ rệt trên tập kiểm tra."
    )

    with st.expander("Bảng chỉ số của năm mô hình trên tập kiểm tra"):
        st.dataframe(lib.doc_bang(lib.HOU_COMPARISON).round(4),
                     hide_index=True, use_container_width=True)

# ---- Tab 3: lý giải phương pháp ----
with tab_ly:
    st.markdown("### Vector đặc trưng thực sự đưa vào mô hình")
    st.markdown(
        "Mười một cột bạn vừa khai, cộng **ba cột phái sinh** do "
        "`prep.add_housing_features` tính ra: `Total_Area` (diện tích × số "
        "tầng), `Room_Density` (số phòng trên mỗi m²) và `Frontage_Ratio`. "
        "Phần mã hoá one-hot và chuẩn hoá nằm bên trong pipeline — gọi lại đúng "
        "hàm đã dùng lúc huấn luyện, chứ không chép công thức sang."
    )
    st.dataframe(khung, hide_index=True, use_container_width=True)

    st.markdown("### Ba điều quyết định con số bạn vừa thấy")

    with st.expander("① Vị trí hành chính là nhóm thông tin quyết định"):
        st.markdown(
            "Thực nghiệm ở Chương 3 bỏ hẳn nhóm đặc trưng định danh (tỉnh, "
            "quận, pháp lý, hướng nhà, nội thất) rồi huấn luyện lại: R² tụt từ "
            "**0.6141** xuống **0.3340** — mất hơn nửa khả năng giải thích. "
            "Cùng một căn nhà 80 m² đặt ở hai quận khác nhau cho hai mức giá "
            "khác hẳn, và đó là thông tin mô hình không suy ra được từ diện "
            "tích hay số phòng."
        )
        tep = lib.cfg.OUTPUTS_DIR / "hou_representation.csv"
        if tep.exists():
            st.dataframe(lib.doc_bang(tep.name).round(4), hide_index=True,
                         use_container_width=True)

    with st.expander("② Vì sao ô định lượng để sẵn số chứ không để trống"):
        st.markdown(
            "Khác bài toán tiểu đường: ở đây `clean_housing` đã điền ô khuyết "
            "bằng **trung vị của tập huấn luyện** ngay từ lúc huấn luyện. Nên "
            "giá trị gợi ý sẵn trên form đúng là con số mô hình từng thấy ở "
            "những tin đăng không khai báo trường đó — không biết Mặt tiền thì "
            "cứ để nguyên là khớp nhất."
        )
        st.json({"Tỷ lệ thiếu trong tập gốc (%)": TOM_TAT["missing_ratio"]},
                expanded=True)
        st.markdown(
            f"Cột `{', '.join(TOM_TAT['dropped_columns'])}` bị loại hẳn vì "
            f"thiếu quá {lib.cfg.MISSING_DROP_THRESHOLD:.0%} số quan sát — "
            "điền trung vị cho một cột thiếu tới mức ấy là bịa ra dữ liệu chứ "
            "không phải khôi phục nó."
        )

    with st.expander("③ Ngưỡng phân khúc và hạn mức vay lấy từ đâu"):
        st.markdown(
            "Ba phân khúc và ba gói vay là **tri thức miền được khai báo trong "
            "bản thể học** (`src/kg/ontology.py`), không phải thứ mô hình học "
            "ra. Mô hình chỉ trả về một con số; đồ thị tri thức là lớp biến con "
            "số ấy thành khuyến nghị hành động được."
        )
        st.dataframe(
            pd.DataFrame([
                {"Phân khúc": s["label"],
                 "Ngưỡng dưới (tỷ VNĐ)": s["min_price"],
                 "Gói vay": s["loan"],
                 "LTV": f"{s['ltv']:.0%}",
                 "Lãi suất": s["rate"],
                 "Thời hạn": s["tenor"]}
                for s in PROPERTY_SEGMENTS
            ]),
            hide_index=True, use_container_width=True,
        )


# ---- Tab 4: tiện ích hạ tầng đô thị và gói tài chính mua nhà ----
with tab_ht:
    tai_chinh = urban.ho_so_tai_chinh(gia, phan_khuc)
    cum = urban.cum_tien_ich(phan_khuc["id"])

    st.markdown(
        "Hạn mức vay mới trả lời được một nửa câu hỏi. Nửa còn lại — **mỗi "
        "tháng phải trả bao nhiêu, thu nhập bao nhiêu thì gánh nổi, và quanh "
        "đó có gì** — nằm ở đây. Ba con số tài chính bên dưới được tính bằng "
        "công thức niên kim từ đúng các tham số hiển thị, không phải ước lượng."
    )

    st.markdown("### Gói tài chính mua nhà")

    lib.luoi_so([
        {"k": "Trả góp hằng tháng",
         "v": f"{tai_chinh['tra_gop_thang'] * 1000:,.1f} triệu đ".replace(",", "."),
         "s": f"đều trong {tai_chinh['ky_han_nam']} năm", "mau": mau_pk},
        {"k": "Thu nhập tối thiểu",
         "v": f"{tai_chinh['thu_nhap_toi_thieu'] * 1000:,.1f} triệu đ".replace(",", "."),
         "s": f"để trả góp ≤ {tai_chinh['dti_toi_da']:.0%} thu nhập", "mau": mau_pk},
        {"k": "Vốn tự có cần chuẩn bị",
         "v": urban.ty_dong(tai_chinh["von_tu_co"]),
         "s": f"{1 - tai_chinh['ltv']:.0%} giá trị căn nhà", "mau": mau_pk},
        {"k": "Tổng tiền lãi cả kỳ hạn",
         "v": urban.ty_dong(tai_chinh["tong_tien_lai"]),
         "s": f"{tai_chinh['tong_tien_lai'] / tai_chinh['han_muc_vay']:.0%} dư nợ gốc",
         "mau": lib.MAU["do"]},
    ])

    ht_trai, ht_phai = st.columns([2, 3], gap="large")

    with ht_trai:
        with st.container(border=True):
            st.markdown(f"### {phan_khuc['loan']}")
            lib.bang_dieu_khoan([
                ("Giá định giá", urban.ty_dong(tai_chinh["gia_dinh_gia"])),
                ("Tỷ lệ cho vay (LTV)", f"{tai_chinh['ltv']:.0%}"),
                ("Hạn mức vay", urban.ty_dong(tai_chinh["han_muc_vay"])),
                ("Lãi suất", phan_khuc["rate"]),
                ("Kỳ hạn", f"{tai_chinh['ky_han_nam']} năm "
                           f"({tai_chinh['ky_han_nam'] * 12} kỳ trả)"),
                ("Phí trả trước hạn", tai_chinh["phi_trong_ly"]),
                ("Trả góp hằng tháng",
                 f"{tai_chinh['tra_gop_thang'] * 1000:,.1f} triệu đ".replace(",", ".")),
            ], mau_pk, nhan_manh={"Hạn mức vay", "Trả góp hằng tháng"})

    with ht_phai:
        st.markdown("##### Lịch trả nợ theo năm")
        lich = lib.lich_tra_no(tai_chinh["han_muc_vay"],
                               tai_chinh["lai_suat_nam"],
                               tai_chinh["ky_han_nam"])
        st.dataframe(lich, hide_index=True, use_container_width=True, height=252)
        lib.ghi_chu(
            f"Năm đầu tiên trả <b>{lich.iloc[0]['Trả lãi (tỷ)'] * 1000:,.0f} triệu đ</b> "
            f"tiền lãi nhưng chỉ giảm được "
            f"<b>{lich.iloc[0]['Trả gốc (tỷ)'] * 1000:,.0f} triệu đ</b> dư nợ. "
            "Đây là đặc điểm cố hữu của khoản vay niên kim: phần lãi dồn về đầu "
            "kỳ, nên trả trước hạn trong vài năm đầu tiết kiệm được nhiều nhất."
        )

    st.markdown(f"### {cum['label']}")
    lib.ghi_chu(f"{lib._e(cum['mo_ta'])}")
    lib.luoi_tien_ich(cum["tien_ich"], cum["ban_kinh_km"], mau_pk)

    st.info(
        f"{cum['ghi_chu']} {tai_chinh['ghi_chu']}",
        icon="🏙️",
    )
