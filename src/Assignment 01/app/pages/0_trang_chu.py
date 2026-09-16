"""Trang 1 — Trang chủ.

Trả lời đúng ba câu, theo thứ tự người lạ cần: *hệ thống này làm được gì cho
tôi* (hai thẻ công cụ), *nó đúng tới đâu* (bảng xếp hạng mô hình), *nó được
dựng thế nào* (ba khối mở rộng ở cuối). Mọi lý giải học thuật nằm trong khối
mở rộng chứ không chắn đường người muốn dùng thử ngay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib

lib.ap_dung_css()
lib.thanh_dieu_huong()

tom_tat = lib.doc_tom_tat()
dia, hou = tom_tat["diabetes"], tom_tat["housing"]
kg = tom_tat.get("knowledge_graph", {})

# Đọc hai bảng so sánh ngay từ đầu: chỉ số của mô hình tốt nhất trên thẻ
# công cụ phải lấy từ chính bảng ấy, không được viết cứng vào mã.
bang_dia = lib.doc_bang(lib.DIA_COMPARISON).set_index("Model")
bang_hou = lib.doc_bang(lib.HOU_COMPARISON).set_index("Model")
tot_dia = bang_dia.loc[dia["best_model"]]
tot_hou = bang_hou.loc[hou["best_model"]]

lib.hero(
    "Hai hệ thống thông minh, một chuỗi xử lý",
    "Nhập số liệu thật, nhận ngay dự đoán và <b>việc phải làm tiếp theo</b>. "
    "Mỗi kết quả đều đi trọn chuỗi <i>dữ liệu → vector đặc trưng → mô hình "
    "học máy → dự đoán → khuyến nghị</i>, chạy trên đúng pipeline đã huấn "
    "luyện chứ không phải một bản dựng lại.",
    nhan="Phát triển các Hệ thống Thông minh · Assignment 01",
)

# ---- Hai công cụ ----
trai, phai = st.columns(2, gap="medium")

with trai:
    lib.the_cong_cu(
        "🩺", "Chẩn đoán nguy cơ Tiểu đường",
        "Tám chỉ số sinh học → xác suất mắc bệnh → phân tầng nguy cơ → "
        "mã bệnh ICD-10, thiết bị đo, chế độ dinh dưỡng và hướng xử trí.",
        [f"Tốt nhất: {dia['best_model']}",
         f"F1 {tot_dia['F1-Score']:.3f}",
         f"{dia['n_raw']:,} bệnh nhân".replace(",", ".")],
        lib.MAU["do"], tre=1,
    )
    if st.button("🩺  Mở công cụ chẩn đoán", type="primary",
                 use_container_width=True):
        st.switch_page("pages/1_chan_doan_tieu_duong.py")

with phai:
    lib.the_cong_cu(
        "🏠", "Định giá Bất động sản",
        "Diện tích, vị trí hành chính, pháp lý → giá dự báo theo tỷ VNĐ → "
        "phân khúc thị trường → hạn mức vay, lãi suất và tiện ích quanh nhà.",
        [f"Tốt nhất: {hou['best_model']}",
         f"R² {tot_hou['R2']:.3f}",
         f"{hou['n_raw']:,} tin đăng".replace(",", ".")],
        lib.MAU["cam"], tre=2,
    )
    if st.button("🏠  Mở công cụ định giá", type="primary",
                 use_container_width=True):
        st.switch_page("pages/2_dinh_gia_bat_dong_san.py")

st.markdown("")

# ---- Số liệu tổng quan ----
lib.luoi_so([
    {"k": "Dữ liệu y tế", "v": f"{dia['n_raw']:,}".replace(",", "."),
     "s": "bệnh nhân, tập Pima Indians Diabetes", "mau": lib.MAU["do"]},
    {"k": "Dữ liệu nhà đất", "v": f"{hou['n_raw']:,}".replace(",", "."),
     "s": "tin đăng bất động sản Việt Nam", "mau": lib.MAU["cam"]},
    {"k": "Mô hình đã huấn luyện", "v": "5 + 5",
     "s": "cùng một giao thức đánh giá", "mau": lib.MAU["brand"]},
    {"k": "Đồ thị tri thức", "v": f"{kg.get('so_bo_ba', '—')}",
     "s": f"bộ ba trên {kg.get('so_ca', '—')} ca mẫu", "mau": lib.MAU["ngoc"]},
])

# ---- Bảng xếp hạng mô hình ----
st.header("Mô hình đúng tới đâu")
st.markdown(
    "Cùng một giao thức tách dữ liệu và cùng một hạt giống ngẫu nhiên cho "
    "cả hai bài toán, nên hai bảng dưới đây so sánh được với nhau."
)

m_trai, m_phai = st.columns(2, gap="large")

with m_trai:
    bang = lib.doc_bang(lib.DIA_COMPARISON)
    bang = bang[~bang["Model"].str.lower().str.startswith("baseline")]
    bang = bang.sort_values("F1-Score", ascending=False)
    st.subheader("🩺 Tiểu đường — F1-Score")
    lib.bang_xep_hang(bang["Model"].tolist(), bang["F1-Score"].tolist(),
                      lambda v: f"{v:.3f}", lib.MAU["do"],
                      danh_dau=dia["best_model"])
    lib.ghi_chu(
        f"Mô hình cơ sở đoán lớp đa số đạt Accuracy "
        f"<b>{dia['baseline']['Accuracy']:.3f}</b> nhưng Recall bằng "
        f"<b>0</b> — bỏ sót toàn bộ người bệnh. Đó là lý do bài toán sàng "
        "lọc không chấm điểm bằng Accuracy."
    )

with m_phai:
    bang = lib.doc_bang(lib.HOU_COMPARISON)
    bang = bang[~bang["Model"].str.lower().str.startswith("baseline")]
    bang = bang.sort_values("R2", ascending=False)
    st.subheader("🏠 Bất động sản — R²")
    lib.bang_xep_hang(bang["Model"].tolist(), bang["R2"].tolist(),
                      lambda v: f"{v:.3f}", lib.MAU["cam"],
                      danh_dau=hou["best_model"])
    lib.ghi_chu(
        f"Mô hình cơ sở đoán giá trung vị đạt R² "
        f"<b>{hou['baseline']['R2']:.4f}</b> — xấp xỉ không giải thích "
        "được gì. Sai số trung bình của mô hình tốt nhất là "
        f"<b>{bang['MAE'].min():.2f} tỷ VNĐ</b>."
    )

# ---- Phần dựng hệ thống, gói trong khối mở rộng ----
st.header("Hệ thống được dựng thế nào")

with st.expander("🧩 Kiến trúc — sáu tầng từ thế giới thực tới ứng dụng"):
    so_do = lib.cfg.FIGURES_DIR / "fig_system_diagram.png"
    if so_do.exists():
        st.image(str(so_do), use_container_width=True)
    else:
        st.info("Chưa có sơ đồ — chạy `python run_pipeline.py` để sinh.")

with st.expander("🔢 Biểu diễn dữ liệu — vì sao đây là phần quyết định"):
    st.markdown(
        "Điểm mấu chốt của cả bài: **biểu diễn lúc dự đoán phải khớp biểu "
        "diễn lúc huấn luyện**. Ứng dụng nạp lại đúng các pipeline đã lưu "
        "trong `models/` — phép điền khuyết, thiết kế đặc trưng, chuẩn hoá "
        "và mã hoá đều nằm sẵn bên trong, không bước nào được dựng lại bằng tay."
    )
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            f"**Tiểu đường** — {dia['n_features_raw']} đặc trưng thô → "
            f"{dia['n_features_engineered']} sau khi thêm bốn đặc trưng do "
            "con người thiết kế (`Glucose × BMI`, nhóm đường huyết, nhóm "
            "BMI, nhóm tuổi). Năm cột sinh học dùng số 0 để mã hoá *không "
            "đo được*; giữ nguyên số 0 sẽ dạy mô hình rằng có bệnh nhân "
            "huyết áp bằng 0."
        )
        st.json({"Số quan sát khuyết theo cột": dia["zero_as_missing"]},
                expanded=False)
    with b:
        st.markdown(
            f"**Bất động sản** — {len(hou['numeric_features'])} đặc trưng "
            f"định lượng và {len(hou['categorical_features'])} đặc trưng "
            f"định danh. Cột `{', '.join(hou['dropped_columns'])}` bị loại "
            f"vì thiếu quá {lib.cfg.MISSING_DROP_THRESHOLD:.0%} số quan "
            "sát; địa chỉ văn bản tự do được tách thành Tỉnh/Thành và "
            "Quận/Huyện."
        )
        st.json({"Tỷ lệ thiếu (%)": hou["missing_ratio"]}, expanded=False)

with st.expander("🕸️ Máy chủ đồ thị Neo4j — kiểm tra kết nối"):
    st.markdown(
        "Đồ thị tri thức của cả hai công cụ chạy được **không cần máy chủ**: "
        "chúng dựng tại chỗ từ dữ liệu bạn vừa nhập. Máy chủ Neo4j Aura chỉ "
        "là nơi lưu bản đã nạp sẵn của sáu ca mẫu, dùng để truy vấn Cypher."
    )
    if st.button("Kiểm tra kết nối", type="primary"):
        with st.spinner("Đang kết nối máy chủ đồ thị…"):
            trang_thai = lib.trang_thai_neo4j()

        if trang_thai["ket_noi"]:
            # Bọc trong dấu nháy ngược để Streamlit khỏi tự biến chuỗi có
            # dạng ``a@b`` thành một liên kết mailto.
            st.success(f"Đã kết nối — cơ sở dữ liệu "
                       f"`{trang_thai['co_so_du_lieu']}` · máy chủ "
                       f"{trang_thai['phien_ban']}")
            lib.luoi_so([
                {"k": "Nút trên máy chủ", "v": str(trang_thai["so_nut"])},
                {"k": "Cạnh trên máy chủ", "v": str(trang_thai["so_canh"])},
            ])
            st.dataframe(pd.DataFrame(trang_thai["theo_loai"]),
                         use_container_width=True, hide_index=True)
        else:
            st.warning(f"Không kết nối được: {trang_thai['ly_do']}")
            st.markdown(
                "Đây **không phải lỗi của ứng dụng**. Muốn bật Neo4j thì "
                "xem `setup-guides/01-neo4j-aura.md`; trên bản deploy thì "
                "khai thông tin đăng nhập trong mục Secrets của Streamlit, "
                "đừng commit vào kho."
            )

    tep_cypher = lib.cfg.OUTPUTS_DIR / "graph.cypher"
    if tep_cypher.exists():
        st.download_button("⬇️ Tải kịch bản Cypher dựng lại đồ thị",
                           tep_cypher.read_bytes(), file_name="graph.cypher",
                           mime="text/plain")

st.divider()
lib.ghi_chu(
    "Nguyễn Duy Nghĩa · D23CTPM01 · B23DCCN600 · Học viện Công nghệ Bưu "
    "chính Viễn thông — môn Phát triển các Hệ thống Thông minh. Mọi con số "
    "sinh ra từ <code>python run_pipeline.py</code> với hạt giống ngẫu nhiên "
    f"cố định <code>{tom_tat['config']['random_state']}</code>, nên chạy lại "
    "cho kết quả trùng khít."
)
