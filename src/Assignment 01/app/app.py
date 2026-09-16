"""Ứng dụng web Streamlit — điểm vào và khai báo điều hướng.

Ba trang được khai báo bằng ``st.navigation`` thay vì để Streamlit tự dò thư
mục ``pages/``: cách tự dò lấy **tên tệp** làm nhãn, mà tên tệp thì không mang
được dấu tiếng Việt.

Điều hướng đặt ``position="hidden"`` và ứng dụng **tự dựng lấy thanh điều
hướng** (``lib.thanh_dieu_huong``). Lý do là hai lỗi đo được ở bản dùng thanh
điều hướng dựng sẵn:

* ``position="sidebar"`` chiếm gần một phần tư bề ngang chỉ để hiện ba dòng
  chữ, và trên điện thoại nó bung ra phủ kín màn hình lúc mới mở.
* ``position="top"`` đẹp trên màn rộng, nhưng dưới một ngưỡng bề ngang Streamlit
  dồn điều hướng trở vào thanh bên — và ở đó nó hiện **tên tệp không dấu**
  (``chan doan tieu duong``) chứ không phải nhãn đã khai. Đo được ở khổ 390px.

Thanh tự dựng nằm trong luồng nội dung nên nó căn đúng cột nội dung ở mọi bề
ngang, tự xuống dòng trên màn hẹp, và giữ nguyên nhãn tiếng Việt.
"""

from __future__ import annotations

import streamlit as st

import lib

lib.dat_cau_hinh_trang()          # phải là lệnh Streamlit đầu tiên

dieu_huong = st.navigation(
    [st.Page(duong_dan, title=nhan, icon=icon, default=(i == 0))
     for i, (duong_dan, nhan, icon) in enumerate(lib.TRANG)],
    position="hidden",
)
dieu_huong.run()
