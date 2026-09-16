"""Phần dùng chung của ứng dụng web — hệ thống thị giác, nạp mô hình, dựng đồ thị.

Ba trang đều cần đúng những việc ấy. Gom về một chỗ để **biểu diễn lúc dự đoán
chỉ được định nghĩa một lần**: nếu mỗi trang tự dựng lấy DataFrame thì sớm muộn
hai trang sẽ lệch nhau về thứ tự cột hoặc về quy ước ô khuyết, mà không lỗi nào
được ném ra — mô hình vẫn trả về một con số, chỉ là con số sai.

Nguyên tắc xuyên suốt tệp này: **không dựng lại phép biến đổi bằng tay**. Mọi
bước tiền xử lý đều gọi lại đúng hàm trong ``src/preprocess.py`` mà lúc huấn
luyện đã dùng, còn phần chuẩn hoá và mã hoá thì nằm sẵn bên trong pipeline đã
lưu ở ``models/``. Đây chính là điều đề bài yêu cầu chứng minh (R11).
"""

from __future__ import annotations

import html
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ``streamlit run app/app.py`` đặt thư mục ``app/`` vào đầu sys.path chứ không
# đặt thư mục gốc dự án, nên ``from src import ...`` sẽ hỏng nếu không chèn.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config as cfg
from src import preprocess as prep
from src import train as tr
from src.kg import catalog_pharmacy as pharm
from src.kg import catalog_urban as urban
from src.kg import neo4j_client as nc
from src.kg.build_graph import SCHEMA_VERSION, strip_cdn_links
from src.kg.ontology import NODE_COLOURS, KnowledgeGraph

# ==========================================================================
#  HẰNG SỐ
# ==========================================================================

GRAPH_JSON = cfg.OUTPUTS_DIR / "graph.json"
TRIPLETS_CSV = cfg.OUTPUTS_DIR / "kg_triplets.csv"
SUMMARY_JSON = cfg.OUTPUTS_DIR / "pipeline_summary.json"

DIA_COMPARISON = "dia_model_comparison.csv"
HOU_COMPARISON = "hou_model_comparison.csv"

LENH_DUNG_LAI = "python run_pipeline.py"

# Nhãn của lựa chọn mặc định trong hộp chọn mô hình.
NHAN_TOT_NHAT = "★ Mô hình tốt nhất"

# Sáu cột định lượng người dùng nhập trên form định giá. Ba cột phái sinh
# (Total_Area, Room_Density, Frontage_Ratio) KHÔNG có ở đây: chúng do
# ``prep.add_housing_features`` tính ra, đúng như lúc huấn luyện.
HOU_NUMERIC = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
HOU_CATEGORICAL = ["House direction", "Legal status", "Furniture state",
                   "Province", "District"]

# Loại thực thể đứng ở gốc chuỗi suy luận, vẽ to hơn phần còn lại.
LOAI_GOC = {"Patient", "Property"}

# Ba trang của ứng dụng: (đường dẫn tệp, nhãn hiển thị, biểu tượng).
# Khai một lần ở đây vì cả ``app.py`` (dựng ``st.navigation``) lẫn
# ``thanh_dieu_huong`` (vẽ thanh điều hướng) đều đọc nó — tách đôi thì sớm muộn
# nhãn trên thanh lệch khỏi nhãn đã đăng ký, mà không có lỗi nào được ném ra.
TRANG = [
    ("pages/0_trang_chu.py", "Trang chủ", "🏠"),
    ("pages/1_chan_doan_tieu_duong.py", "Chẩn đoán Tiểu đường", "🩺"),
    ("pages/2_dinh_gia_bat_dong_san.py", "Định giá Bất động sản", "🏠"),
]

# Bảng màu của giao diện. Hai màu đầu trùng ``src/config.PALETTE`` để ảnh chụp
# màn hình và biểu đồ trong báo cáo LaTeX nhìn ra cùng một hệ thống thị giác.
MAU = {
    "brand": "#2E86AB",
    "brand_dam": "#1B5670",
    "ngoc": "#35A7A0",
    "cam": "#F18F01",
    "do": "#C73E1D",
    "xanh_la": "#2F8F5B",
    "muc": "#122130",
    "mo": "#5B6B7A",
}


# ==========================================================================
#  HỆ THỐNG THỊ GIÁC
# ==========================================================================

# Một bảng kiểu duy nhất cho cả ứng dụng. Ba nguyên tắc chi phối nó:
#
# 1. **Chuyển động phải có nghĩa.** Mọi hiệu ứng ở đây đều thuần CSS và chỉ
#    chạy một lần lúc khối xuất hiện — không có vòng lặp gây phân tán. Người
#    bật ``prefers-reduced-motion`` được tắt sạch ở cuối bảng.
# 2. **Lưới CSS thay cho ``st.columns`` ở phần thẻ số.** ``st.columns`` không
#    tự xuống dòng trên màn hẹp nên trước đây phải ép bằng media query; lưới
#    ``auto-fit`` tự giãn, không cần hack nào.
# 3. **Dấu tiếng Việt cần chỗ.** Dấu ngã và dấu hỏi nằm cao hơn chữ Latin, nên
#    mọi tiêu đề đều đặt ``line-height`` tối thiểu 1.35; thu nhỏ cỡ chữ mà giữ
#    line-height mặc định sẽ cắt cụt dấu ở dòng đầu.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

:root{
  --t1-brand:#2E86AB; --t1-brand-d:#1B5670; --t1-brand-l:#EAF4F9;
  --t1-teal:#35A7A0; --t1-accent:#F18F01; --t1-danger:#C73E1D;
  --t1-ok:#2F8F5B; --t1-warn:#D98324;
  --t1-ink:#122130; --t1-muted:#5B6B7A; --t1-line:#E4EBF1;
  --t1-bg:#F5F8FB; --t1-card:#FFFFFF;
  --t1-sh:0 1px 2px rgba(18,33,48,.04), 0 8px 24px -14px rgba(18,33,48,.22);
  --t1-sh-h:0 2px 5px rgba(18,33,48,.06), 0 20px 42px -20px rgba(18,33,48,.34);
  --t1-r:14px;
}

/* ---------- Nền và chữ ---------- */
html, body, .stApp, [class*="st-"], .stMarkdown, button, input, select, textarea{
  font-family:'Be Vietnam Pro','Segoe UI',system-ui,-apple-system,sans-serif;
}
/* Streamlit vẽ mọi biểu tượng bằng **ligature** của Material Symbols, và quy
   tắc chữ ở trên quét theo `[class*="st-"]` nên nó đè luôn font biểu tượng.
   Ligature mất font thì hiện ra nguyên văn tên nó — chuỗi "keyboard_arrow_right"
   nằm đè lên nhãn của mọi expander và mũi tên của mọi hộp chọn. Trả lại đúng
   font cho riêng nhóm ấy, và trả font đẳng chiều cho khối mã. */
[data-testid="stIconMaterial"], .material-symbols-rounded,
span[class*="material-symbols"]{
  font-family:'Material Symbols Rounded' !important;
}
code, kbd, pre, samp, [data-testid="stCode"] *{
  font-family:'Source Code Pro','Consolas','Courier New',monospace;
}
.stApp{ background:var(--t1-bg); }
/* Thanh công cụ của Streamlit (nút Deploy, menu ⋮) cao 60px và
   ``position:absolute`` — nó **đè** lên nội dung chứ không đẩy nội dung xuống.
   Đặt padding-top nhỏ hơn 60px là đầu trang bị che khuất. */
.block-container{ max-width:1240px; padding-top:calc(60px + .5rem); padding-bottom:4rem; }
h1,h2,h3,h4{ color:var(--t1-ink); letter-spacing:-.012em; }
h1{ font-size:2.0rem;  font-weight:800; line-height:1.36; }
h2{ font-size:1.38rem; font-weight:700; line-height:1.4; margin-top:.5rem; }
h3{ font-size:1.08rem; font-weight:700; line-height:1.45; }
p, li{ color:#2B3B4A; line-height:1.7; }
a{ color:var(--t1-brand); }
hr{ border-color:var(--t1-line); }

/* ---------- Điều hướng ---------- */
/* Ứng dụng tự dựng thanh điều hướng (xem ``app.py`` để biết vì sao), nên thanh
   bên và mọi tàn dư của điều hướng dựng sẵn đều bị chặn. An toàn ở đây, khác
   hẳn bản dùng ``position="top"``: ở đó thanh bên **là** đường chuyển trang
   duy nhất trên điện thoại, ẩn nó đi là khoá luôn người dùng. */
[data-testid="stSidebar"], [data-testid="stSidebarNav"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"]{
  display:none !important;
}
header[data-testid="stHeader"]{ background:transparent; }

/* Thanh điều hướng tự dựng: dính lên đỉnh khi cuốn, nằm trong luồng nội dung
   nên tự căn đúng cột nội dung ở mọi bề ngang. */
.st-key-t1-nav{
  position:sticky; top:0; z-index:60; margin:0 0 1.1rem;
  padding:.45rem; border-radius:16px;
  background:rgba(255,255,255,.88); backdrop-filter:blur(12px);
  border:1px solid var(--t1-line); box-shadow:var(--t1-sh);
  animation:t1-in .4s ease both;
}
/* Ép hàng ngang bằng bảng kiểu của **chính khối này**, không giao cho
   Streamlit. Đo được trên hai bản:
   
     1.61 → khối là ``stHorizontalBlock``, mặc định đã nằm ngang.
     1.62 → khối là ``stVerticalBlock`` (``flex-direction: column``) và mỗi
            khối con được bọc thêm một lớp ``stLayoutWrapper`` mới.
   
   Cùng một mã cho ra hai bố cục khác nhau, không lỗi nào được ném ra — thanh
   điều hướng nằm ngang ở máy cục bộ nhưng xếp dọc trên bản deploy. Ép cả
   ``flex-direction`` của chính khối chứa lẫn cách co giãn của lớp bọc thì bố
   cục không còn phụ thuộc bản Streamlit nữa. */
.st-key-t1-nav{
  display:flex !important; flex-direction:row !important; flex-wrap:wrap !important;
  align-items:center !important; gap:.35rem !important;
}
/* Lớp bọc mỗi liên kết: rộng bằng đúng nhãn, không chia đều cứng nhắc.
   Chọn theo ``> div`` để trúng cả ``stLayoutWrapper`` của 1.62 lẫn khối con
   trực tiếp của các bản trước. */
.st-key-t1-nav > div{
  flex:0 0 auto !important; width:auto !important; min-width:0 !important;
}
[data-testid="stPageLink"] a{
  display:flex; align-items:center; justify-content:center; gap:.45rem;
  padding:.58rem .8rem; border-radius:11px; text-decoration:none;
  border:1px solid transparent; transition:all .18s ease; line-height:1.4;
  /* Chốt chiều cao: trên màn hẹp nhãn dài xuống hai dòng còn "Trang chủ" một
     dòng, ba viên cao so le hẳn. */
  min-height:2.85rem;
}
[data-testid="stPageLink"] a:hover{
  background:var(--t1-brand-l);
  border-color:color-mix(in srgb,var(--t1-brand) 24%,#fff);
}
[data-testid="stPageLink"] a p{
  margin:0; font-weight:600; font-size:.93rem; color:#3B4B59; white-space:nowrap;
}
/* Chỉ báo trang hiện tại — xem ``app.py`` để biết vì sao phải tự xác định
   thay vì dựa vào dấu hiệu sẵn có của Streamlit. */
.st-key-t1-nav-on [data-testid="stPageLink"] a,
.st-key-t1-nav-on [data-testid="stPageLink"] a:hover{
  background:linear-gradient(120deg,var(--t1-brand-d),var(--t1-brand));
  border-color:transparent; box-shadow:0 10px 20px -14px rgba(27,86,112,.95);
}
.st-key-t1-nav-on [data-testid="stPageLink"] a p{ color:#fff !important; }

/* ---------- Chuyển động ---------- */
@keyframes t1-up{ from{opacity:0; transform:translateY(16px)} to{opacity:1; transform:none} }
@keyframes t1-in{ from{opacity:0} to{opacity:1} }
@keyframes t1-ring{ from{stroke-dashoffset:var(--t1-dash)} to{stroke-dashoffset:var(--t1-off)} }
@keyframes t1-bar{ from{width:0} to{width:var(--t1-w)} }
@keyframes t1-slide{ from{left:0; opacity:0} to{left:var(--t1-x); opacity:1} }
@keyframes t1-float{ 0%,100%{transform:translate(0,0)} 50%{transform:translate(-18px,16px)} }
.t1-d1{ animation-delay:.06s } .t1-d2{ animation-delay:.12s }
.t1-d3{ animation-delay:.18s } .t1-d4{ animation-delay:.24s }
.t1-d5{ animation-delay:.30s } .t1-d6{ animation-delay:.36s }

/* ---------- Dải tiêu đề ---------- */
.t1-hero{
  position:relative; overflow:hidden; border-radius:20px; color:#fff;
  padding:2.1rem 2.3rem; margin:.2rem 0 1.5rem;
  background:linear-gradient(125deg,#12455C 0%,#2E86AB 54%,#35A7A0 100%);
  box-shadow:0 22px 46px -26px rgba(18,69,92,.85);
  animation:t1-up .55s cubic-bezier(.22,1,.36,1) both;
}
.t1-hero::after{
  content:""; position:absolute; top:-55%; right:-8%; width:430px; height:430px;
  border-radius:50%; pointer-events:none;
  background:radial-gradient(circle at 32% 32%, rgba(255,255,255,.20), transparent 63%);
  animation:t1-float 11s ease-in-out infinite;
}
.t1-hero > *{ position:relative; z-index:1 }
.t1-hero h1{ color:#fff; font-size:2.05rem; line-height:1.34; margin:0 0 .55rem; }
.t1-hero p{ color:rgba(255,255,255,.92); font-size:1.02rem; line-height:1.7;
  max-width:70ch; margin:0; }
.t1-hero code{ background:rgba(255,255,255,.16); color:#fff; padding:.08rem .38rem;
  border-radius:5px; font-size:.92em; }
.t1-eyebrow{
  display:inline-block; font-size:.71rem; font-weight:700; letter-spacing:.15em;
  text-transform:uppercase; color:rgba(255,255,255,.95);
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.30);
  padding:.32rem .78rem; border-radius:999px; margin-bottom:.95rem;
}

/* ---------- Lưới thẻ số ---------- */
.t1-grid{ display:grid; gap:.85rem; margin:.1rem 0 1.2rem;
  grid-template-columns:repeat(auto-fit,minmax(186px,1fr)); }
.t1-grid-2{ grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }
/* Biến thể hẹp cho nhóm thẻ phụ (thống kê đồ thị): ô 186px làm bốn thẻ vỡ
   thành 3 + 1, để lại một thẻ lẻ loi ở dòng dưới. */
.t1-grid-gon{ grid-template-columns:repeat(auto-fit,minmax(132px,1fr)); gap:.55rem; }
.t1-grid-gon .t1-stat{ padding:.7rem .75rem .75rem .9rem; }
.t1-grid-gon .t1-stat .k{ font-size:.66rem; letter-spacing:.05em; }
.t1-grid-gon .t1-stat .v{ font-size:1.25rem; }
.t1-stat{
  position:relative; overflow:hidden; background:var(--t1-card);
  border:1px solid var(--t1-line); border-radius:var(--t1-r);
  padding:1rem 1.05rem 1.05rem 1.25rem; box-shadow:var(--t1-sh);
  animation:t1-up .5s cubic-bezier(.22,1,.36,1) both;
  transition:transform .22s ease, box-shadow .22s ease;
}
.t1-stat:hover{ transform:translateY(-3px); box-shadow:var(--t1-sh-h); }
.t1-stat::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:linear-gradient(180deg,var(--t1-accent-1,var(--t1-brand)),var(--t1-accent-2,var(--t1-teal))); }
.t1-stat .k{ font-size:.74rem; font-weight:700; letter-spacing:.07em;
  text-transform:uppercase; color:var(--t1-muted); }
.t1-stat .v{ font-size:1.62rem; font-weight:800; color:var(--t1-ink);
  line-height:1.3; margin:.3rem 0 0; }
.t1-stat .s{ font-size:.8rem; color:var(--t1-muted); margin:.2rem 0 0; line-height:1.5; }

/* ---------- Khối nội dung ---------- */
.t1-panel{
  background:var(--t1-card); border:1px solid var(--t1-line); border-radius:18px;
  padding:1.35rem 1.45rem; box-shadow:var(--t1-sh);
  animation:t1-up .5s cubic-bezier(.22,1,.36,1) both;
}
.t1-panel h3{ margin:0 0 .2rem; }
.t1-note{ color:var(--t1-muted); font-size:.85rem; line-height:1.65; }

/* ---------- Thẻ công cụ ở trang chủ ---------- */
.t1-tool{
  position:relative; overflow:hidden; height:100%;
  background:var(--t1-card); border:1px solid var(--t1-line); border-radius:18px;
  padding:1.5rem 1.5rem 1.35rem; box-shadow:var(--t1-sh);
  animation:t1-up .5s cubic-bezier(.22,1,.36,1) both;
  transition:transform .24s ease, box-shadow .24s ease, border-color .24s ease;
}
.t1-tool:hover{ transform:translateY(-4px); box-shadow:var(--t1-sh-h);
  border-color:color-mix(in srgb, var(--t1-tone) 45%, var(--t1-line)); }
.t1-tool::before{ content:""; position:absolute; inset:0 0 auto 0; height:4px;
  background:linear-gradient(90deg,var(--t1-tone),transparent 92%); }
.t1-tool .ic{ width:46px; height:46px; border-radius:13px; display:grid;
  place-items:center; font-size:1.45rem; margin-bottom:.85rem;
  background:color-mix(in srgb, var(--t1-tone) 13%, #fff);
  border:1px solid color-mix(in srgb, var(--t1-tone) 26%, #fff); }
.t1-tool h3{ margin:0 0 .35rem; font-size:1.16rem; }
.t1-tool .desc{ color:#3C4C5B; font-size:.93rem; line-height:1.65; margin:0 0 .9rem; }
/* Hai thẻ công cụ nằm cạnh nhau, mô tả dài ngắn khác nhau nên đáy thẻ — và
   nút bấm ngay dưới nó — lệch nhau một dòng. Chốt chỗ cho ba dòng mô tả để
   hai thẻ bằng nhau. Chỉ áp ở khổ rộng: trên điện thoại thẻ xếp chồng nên
   min-height chỉ tạo ra khoảng trống thừa. */
@media (min-width:769px){
  .t1-tool .desc{ min-height:4.8rem; }
}
.t1-tool .kv{ display:flex; flex-wrap:wrap; gap:.4rem .5rem; }

/* ---------- Nhãn tròn ---------- */
.t1-chip{ display:inline-flex; align-items:center; gap:.35rem; font-size:.78rem;
  font-weight:600; padding:.28rem .62rem; border-radius:999px;
  background:var(--t1-brand-l); color:var(--t1-brand-d);
  border:1px solid color-mix(in srgb, var(--t1-brand) 22%, #fff); }
.t1-chip.warn{ background:#FEF6EC; color:#94540A; border-color:#F6DCB6; }
.t1-chip.bad { background:#FCEEEA; color:#8E2C14; border-color:#F4CDC2; }
.t1-chip.good{ background:#ECF7F1; color:#1F6B45; border-color:#C6E7D6; }
.t1-chip .dot{ width:.5rem; height:.5rem; border-radius:50%; background:currentColor; }

/* ---------- Vòng nguy cơ ---------- */
.t1-gauge{ position:relative; width:186px; height:186px; margin:.2rem auto .6rem;
  animation:t1-in .4s ease both; }
.t1-gauge svg{ width:100%; height:100%; display:block; }
.t1-gauge .track{ fill:none; stroke:#EDF2F6; stroke-width:13; }
.t1-gauge .fill{ fill:none; stroke-width:13; stroke-linecap:round;
  transform:rotate(-90deg); transform-origin:50% 50%;
  animation:t1-ring 1.15s cubic-bezier(.22,1,.36,1) both .12s; }
.t1-gauge .mid{ position:absolute; inset:0; display:grid; place-content:center;
  text-align:center; }
.t1-gauge .num{ font-size:2.5rem; font-weight:800; line-height:1.1; }
/* Chú thích nằm **dưới** vòng tròn chứ không nằm trong lòng nó: chuỗi "Xác
   suất mắc bệnh" dài hơn đường kính phần rỗng, đặt vào trong là nó chạy đè lên
   nét vẽ của vòng. */
.t1-gauge-cap{ text-align:center; font-size:.75rem; font-weight:700;
  letter-spacing:.09em; text-transform:uppercase; color:var(--t1-muted);
  margin:0 0 .5rem; animation:t1-in .5s ease both .2s; }

/* ---------- Số lớn (giá) ---------- */
.t1-big{ text-align:center; padding:.4rem 0 .1rem; animation:t1-up .5s cubic-bezier(.22,1,.36,1) both; }
.t1-big .num{ font-size:3.1rem; font-weight:800; line-height:1.12;
  background:linear-gradient(96deg,var(--t1-brand-d),var(--t1-brand) 52%,var(--t1-teal));
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.t1-big .unit{ font-size:.95rem; font-weight:700; color:var(--t1-muted);
  letter-spacing:.06em; text-transform:uppercase; }

/* ---------- Thước phân khúc ---------- */
.t1-meter{ margin:1.1rem 0 .3rem; }
.t1-meter .track{ position:relative; height:11px; border-radius:999px; overflow:hidden;
  display:flex; background:#EDF2F6; }
.t1-meter .zone{ height:100%; }
.t1-meter .pin{ position:absolute; top:-6px; width:4px; height:23px; border-radius:3px;
  background:var(--t1-ink); box-shadow:0 0 0 3px #fff, 0 2px 8px rgba(18,33,48,.35);
  animation:t1-slide 1s cubic-bezier(.22,1,.36,1) both .18s; }
.t1-meter .lab{ display:flex; justify-content:space-between; margin-top:.55rem;
  font-size:.74rem; font-weight:600; color:var(--t1-muted); }

/* ---------- Thanh so sánh mô hình ---------- */
.t1-rank{ display:flex; flex-direction:column; gap:.6rem; margin:.3rem 0 .2rem; }
.t1-row{ animation:t1-up .45s cubic-bezier(.22,1,.36,1) both; }
.t1-row .head{ display:flex; justify-content:space-between; align-items:baseline;
  gap:.6rem; margin-bottom:.3rem; }
.t1-row .name{ font-size:.9rem; font-weight:600; color:var(--t1-ink); }
.t1-row .val{ font-size:.9rem; font-weight:800; color:var(--t1-ink);
  font-variant-numeric:tabular-nums; }
.t1-row .track{ height:9px; border-radius:999px; background:#EDF2F6; overflow:hidden; }
.t1-row .bar{ height:100%; border-radius:999px; width:var(--t1-w);
  animation:t1-bar .95s cubic-bezier(.22,1,.36,1) both .15s; }
.t1-row.best .name::after{ content:"★"; margin-left:.35rem; color:var(--t1-accent); }

/* ---------- Danh sách khuyến nghị ---------- */
.t1-list{ list-style:none; padding:0; margin:.35rem 0 .2rem; }
.t1-list li{ position:relative; padding:.42rem 0 .42rem 1.55rem; font-size:.92rem;
  line-height:1.62; color:#2B3B4A; border-bottom:1px dashed var(--t1-line);
  animation:t1-up .42s cubic-bezier(.22,1,.36,1) both; }
.t1-list li:last-child{ border-bottom:0; }
.t1-list li::before{ content:""; position:absolute; left:.28rem; top:.95rem;
  width:.42rem; height:.42rem; border-radius:50%; background:var(--t1-tone,var(--t1-brand)); }
.t1-sub{ font-size:.73rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
  color:var(--t1-muted); margin:.9rem 0 .1rem; }

/* ---------- Chú giải màu đồ thị ---------- */
.t1-legend{ display:flex; flex-wrap:wrap; gap:.35rem .55rem; margin:.2rem 0 .7rem; }
.t1-legend span.i{ display:inline-flex; align-items:center; gap:.35rem; font-size:.76rem;
  color:#42525F; background:#fff; border:1px solid var(--t1-line);
  padding:.2rem .52rem; border-radius:999px; }
.t1-legend span.d{ width:.55rem; height:.55rem; border-radius:50%; }

/* ---------- Tinh chỉnh widget của Streamlit ---------- */
.stButton>button{
  border-radius:11px; border:1px solid var(--t1-line); background:#fff;
  font-weight:600; color:#33434F; transition:all .18s ease;
  /* Chốt chiều cao tối thiểu: một hàng nút mà nhãn dài ngắn khác nhau sẽ có
     nút một dòng thấp hơn nút hai dòng, nhìn so le hẳn. */
  min-height:3.05rem;
}
.stButton>button:hover{
  border-color:var(--t1-brand); color:var(--t1-brand); transform:translateY(-1px);
  box-shadow:0 8px 18px -10px rgba(46,134,171,.75);
}
.stButton>button[kind="primary"]{
  background:linear-gradient(120deg,var(--t1-brand-d),var(--t1-brand)); color:#fff;
  border:0; box-shadow:0 10px 22px -12px rgba(27,86,112,.9);
}
.stButton>button[kind="primary"]:hover{ color:#fff; filter:brightness(1.07); }
[data-testid="stNumberInput"] input,
[data-baseweb="select"]>div,
[data-testid="stTextInput"] input{ border-radius:10px !important; }
[data-testid="stExpander"]{
  border:1px solid var(--t1-line) !important; border-radius:13px !important;
  background:var(--t1-card); box-shadow:var(--t1-sh); overflow:hidden;
}
[data-testid="stExpander"] summary{ font-weight:600; }
[data-testid="stTabs"] [role="tablist"]{ gap:.35rem; border-bottom:1px solid var(--t1-line); }
[data-testid="stTabs"] button[role="tab"]{ font-weight:600; padding:.55rem .9rem; }
[data-testid="stDataFrame"]{ border-radius:12px; overflow:hidden; border:1px solid var(--t1-line); }
[data-testid="stMetricValue"]{ font-weight:800; }
[data-testid="stAlert"]{ border-radius:12px; }
iframe{ border-radius:13px; }

/* ---------- Màn hẹp ---------- */
@media (max-width:768px){
  .block-container{ padding:calc(60px + .35rem) .75rem 3rem; }
  /* Ba nhãn điều hướng phải ở nguyên một hàng trên điện thoại: quy tắc xếp
     chồng chung sẽ biến thanh điều hướng thành ba dòng cao chiếm nửa màn hình. */
  /* Thanh điều hướng phải giữ một hàng ngang: quy tắc xếp chồng chung ở trên
     sẽ biến nó thành ba dòng cao chiếm gần nửa màn hình điện thoại. */
  /* Ba viên chia đều một hàng, không xuống dòng: xếp chồng thì thanh điều
     hướng cao gần nửa màn hình điện thoại. */
  .st-key-t1-nav{ padding:.35rem; flex-wrap:nowrap !important; gap:.25rem !important; }
  .st-key-t1-nav > div{ flex:1 1 0 !important; width:auto !important; }
  .st-key-t1-nav [data-testid="stElementContainer"],
  .st-key-t1-nav [data-testid="stPageLink"]{ width:100% !important; }
  [data-testid="stPageLink"]{ width:100%; }
  [data-testid="stPageLink"] a{ padding:.48rem .3rem; gap:.25rem; }
  [data-testid="stPageLink"] a p{
    font-size:.74rem; white-space:normal; text-align:center; line-height:1.3;
  }
  .t1-hero{ padding:1.25rem 1.1rem; border-radius:16px; margin-bottom:1.1rem; }
  .t1-hero h1{ font-size:1.42rem; line-height:1.42; }
  .t1-hero p{ font-size:.93rem; line-height:1.62; }
  .t1-eyebrow{ font-size:.64rem; letter-spacing:.1em; margin-bottom:.7rem; }
  h1{ font-size:1.48rem; line-height:1.42; }
  h2{ font-size:1.18rem; line-height:1.42; }
  [data-testid="stTabs"] [role="tablist"]{
    overflow-x:auto; scrollbar-width:none; gap:.2rem;
  }
  [data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar{ display:none; }
  [data-testid="stTabs"] button[role="tab"]{
    padding:.5rem .6rem; font-size:.87rem; white-space:nowrap;
  }
  .t1-gauge{ width:158px; height:158px; }
  .t1-big .num{ font-size:2.45rem; }
  .t1-panel{ padding:1.1rem 1.05rem; border-radius:15px; }
  /* st.columns không tự xếp chồng trên màn hẹp — ép nó xuống một cột. */
  [data-testid="stHorizontalBlock"]{ flex-direction:column; gap:.55rem; }
  [data-testid="stColumn"], [data-testid="column"]{
    width:100% !important; flex:1 1 100% !important; min-width:100% !important;
  }
}

/* ---------- Tôn trọng lựa chọn tắt chuyển động của người dùng ---------- */
@media (prefers-reduced-motion: reduce){
  *, *::before, *::after{ animation:none !important; transition:none !important; }
}
</style>
"""


def dat_cau_hinh_trang() -> None:
    """Đặt cấu hình trang. Phải là lệnh Streamlit **đầu tiên** của cả ứng dụng."""
    st.set_page_config(
        page_title="Hệ thống Thông minh — Tiểu đường & Bất động sản",
        page_icon="🧠",
        layout="wide",
        # Điều hướng nằm trên đỉnh (``st.navigation(position="top")``) nên ứng
        # dụng không dùng thanh bên nữa. Khai ``collapsed`` để Streamlit không
        # chừa chỗ cho nó ngay từ khung dựng đầu tiên — bỏ dòng này thì trên
        # điện thoại vẫn thấy nút bung menu nhấp nháy một nhịp lúc tải trang.
        initial_sidebar_state="collapsed",
    )


def ap_dung_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(CSS_MO_RONG, unsafe_allow_html=True)


def _e(gia_tri: Any) -> str:
    """Thoát ký tự HTML. Mọi chuỗi đi vào ``unsafe_allow_html`` đều qua đây."""
    return html.escape(str(gia_tri))


# ---- Khối dựng sẵn -------------------------------------------------------

def _trang_dang_mo() -> str:
    """Đường dẫn tệp của trang đang mở, suy từ URL trên trình duyệt.

    Ba cách khác đều không dùng được, đã thử từng cái:

    * ``href`` rỗng — Streamlit chỉ để rỗng ở trang mặc định; ở hai trang kia
      cả ba liên kết đều mang đường dẫn thật.
    * Lớp ``st-emotion-cache-*`` — đổi theo từng bản Streamlit.
    * Ghi nhãn vào ``st.session_state`` từ ``app.py`` — không tới được hàm này.

    So khớp bằng **đoạn cuối** của URL chứ không bằng cả đường dẫn, để bản
    deploy đặt dưới một đường dẫn gốc nào đó vẫn nhận đúng trang. Đoạn cuối
    không khớp mẩu nào thì đó là trang mặc định — đúng cho cả ``/`` lẫn
    ``/trang_chu``.
    """
    try:
        duoi = urlparse(st.context.url).path.rstrip("/").rsplit("/", 1)[-1]
    except Exception:            # ngữ cảnh không có URL (kiểm thử, kịch bản rời)
        return TRANG[0][0]
    for duong_dan, _nhan, _icon in TRANG:
        # "pages/1_chan_doan_tieu_duong.py" → "chan_doan_tieu_duong"
        if Path(duong_dan).stem.split("_", 1)[1] == duoi:
            return duong_dan
    return TRANG[0][0]


def thanh_dieu_huong() -> None:
    """Thanh điều hướng của ứng dụng, gọi ở đầu **mọi** trang.

    Dựng bằng ``st.page_link`` chứ không bằng thanh điều hướng dựng sẵn của
    Streamlit — lý do đầy đủ nằm ở đầu ``app.py``. Điểm cần nhớ: nó là **nội
    dung**, nên nó thừa hưởng đúng cột nội dung và không bao giờ lệch khỏi tiêu
    đề bên dưới, ở bất kỳ bề ngang nào.
    """
    # ``key`` gắn lớp ``st-key-t1-nav`` vào đúng khối này, nên bảng kiểu nhắm
    # được nó mà không phải bọc thẻ ``div`` thủ công — cách bọc tay không dùng
    # được ở Streamlit vì mỗi lệnh ``st.markdown`` tự đóng thẻ của nó lại.
    #
    # KHÔNG dùng ``horizontal=True``: bản đầu giao việc dựng hàng cho tham số
    # ấy, chạy đúng ở Streamlit 1.61 trên máy cục bộ nhưng bản Streamlit Cloud
    # cài (mới hơn, do sàn ``streamlit>=1.61``) lại xếp ba liên kết thành cột
    # dọc. Hàng ngang giờ do bảng kiểu của chính ứng dụng ép — xem khối
    # ``.st-key-t1-nav`` trong CSS — nên nó không đổi theo bản Streamlit nữa.
    hien_tai = _trang_dang_mo()
    with st.container(key="t1-nav"):
        for i, (duong_dan, _nhan, _icon) in enumerate(TRANG):
            # Mục đang mở nhận khoá riêng, và bảng kiểu bắt vào khoá ấy để tô
            # đậm — cách duy nhất gắn được lớp CSS lên một widget của Streamlit.
            khoa = "t1-nav-on" if duong_dan == hien_tai else f"t1-nav-{i}"
            with st.container(key=khoa):
                st.page_link(duong_dan, label=_nhan, icon=_icon)


def hero(tieu_de: str, mo_ta: str, nhan: str | None = None) -> None:
    """Dải tiêu đề gradient đứng đầu mỗi trang."""
    o_nhan = f'<span class="t1-eyebrow">{_e(nhan)}</span>' if nhan else ""
    st.markdown(
        f'<div class="t1-hero">{o_nhan}<h1>{_e(tieu_de)}</h1><p>{mo_ta}</p></div>',
        unsafe_allow_html=True,
    )


def luoi_so(muc: list[dict], rong: bool = False, gon: bool = False) -> None:
    """Lưới thẻ số liệu, xuất hiện lần lượt từ trái sang phải.

    Dùng lưới CSS ``auto-fit`` chứ không dùng ``st.columns``: lưới tự xuống
    dòng trên màn hẹp, còn ``st.columns`` thì phải ép bằng media query.

    Mỗi phần tử: ``{"k": nhãn, "v": giá trị, "s": ghi chú, "mau": mã màu}``.
    """
    o = []
    for i, m in enumerate(muc):
        mau = m.get("mau", MAU["brand"])
        phu = f'<p class="s">{_e(m["s"])}</p>' if m.get("s") else ""
        o.append(
            f'<div class="t1-stat t1-d{min(i + 1, 6)}" '
            f'style="--t1-accent-1:{mau};--t1-accent-2:{mau}88">'
            f'<p class="k">{_e(m["k"])}</p>'
            f'<p class="v">{_e(m["v"])}</p>{phu}</div>'
        )
    lop = "t1-grid t1-grid-2" if rong else ("t1-grid t1-grid-gon" if gon
                                            else "t1-grid")
    st.markdown(f'<div class="{lop}">{"".join(o)}</div>', unsafe_allow_html=True)


def the_cong_cu(icon: str, tieu_de: str, mo_ta: str, nhan: list[str],
                mau: str, tre: int = 1) -> None:
    """Thẻ giới thiệu một công cụ ở trang chủ."""
    chip = "".join(f'<span class="t1-chip">{_e(n)}</span>' for n in nhan)
    st.markdown(
        f'<div class="t1-tool t1-d{tre}" style="--t1-tone:{mau}">'
        f'<div class="ic">{icon}</div>'
        f'<h3>{_e(tieu_de)}</h3><p class="desc">{mo_ta}</p>'
        f'<div class="kv">{chip}</div></div>',
        unsafe_allow_html=True,
    )


def vong_nguy_co(ty_le: float, nhan: str, mau: str) -> None:
    """Vòng tròn tiến trình vẽ bằng SVG, chạy từ 0 tới giá trị thật.

    Chuyển động nằm ở ``stroke-dashoffset``: bắt đầu ở chu vi đầy đủ (vòng
    rỗng) và dừng ở phần còn lại tương ứng tỷ lệ. Cả hai mốc truyền vào bằng
    biến CSS nên không cần JavaScript — Streamlit lọc sạch thẻ ``script``.
    """
    ban_kinh = 62
    chu_vi = 2 * math.pi * ban_kinh
    con_lai = chu_vi * (1 - min(max(ty_le, 0.0), 1.0))
    st.markdown(
        f'<div class="t1-gauge">'
        f'<svg viewBox="0 0 150 150">'
        f'<circle class="track" cx="75" cy="75" r="{ban_kinh}"/>'
        f'<circle class="fill" cx="75" cy="75" r="{ban_kinh}" '
        f'style="stroke:{mau};stroke-dasharray:{chu_vi:.1f};'
        f'--t1-dash:{chu_vi:.1f};--t1-off:{con_lai:.1f}"/>'
        f'</svg>'
        f'<div class="mid"><span class="num" style="color:{mau}">{ty_le:.0%}</span>'
        f'</div></div><p class="t1-gauge-cap">{_e(nhan)}</p>',
        unsafe_allow_html=True,
    )


def so_lon(gia_tri: str, don_vi: str) -> None:
    """Con số kết quả cỡ lớn, tô gradient."""
    st.markdown(
        f'<div class="t1-big"><div class="num">{_e(gia_tri)}</div>'
        f'<div class="unit">{_e(don_vi)}</div></div>',
        unsafe_allow_html=True,
    )


def thuoc_phan_khuc(gia: float, moc: list[tuple[float, str, str]],
                    tran: float) -> None:
    """Thước ngang chia vùng, có kim trượt tới vị trí giá dự báo.

    ``moc`` là danh sách ``(ngưỡng dưới, nhãn, màu)`` xếp tăng dần. ``tran`` là
    giá trị ứng với mép phải của thước — giá vượt trần thì kim dừng ở mép.
    """
    vung, nhan = [], []
    for i, (duoi, ten, mau) in enumerate(moc):
        tren = moc[i + 1][0] if i + 1 < len(moc) else tran
        rong = max(0.0, min(tren, tran) - duoi) / tran * 100
        vung.append(f'<div class="zone" style="width:{rong:.2f}%;background:{mau}4D"></div>')
        nhan.append(f'<span>{_e(ten)}</span>')

    vi_tri = min(max(gia, 0.0) / tran, 1.0) * 100
    st.markdown(
        f'<div class="t1-meter"><div class="track">{"".join(vung)}'
        f'<div class="pin" style="--t1-x:calc({vi_tri:.2f}% - 2px)"></div></div>'
        f'<div class="lab">{"".join(nhan)}</div></div>',
        unsafe_allow_html=True,
    )


def bang_xep_hang(nhan: list[str], gia_tri: list[float], dinh_dang,
                  mau: str, danh_dau: str | None = None) -> None:
    """Danh sách thanh ngang so sánh các mô hình, thanh chạy từ 0 tới giá trị.

    Thay cho ``st.dataframe`` + ``ProgressColumn`` ở bản trước: cùng thông tin
    nhưng đọc nhanh hơn hẳn vì thứ hạng hiện ra ngay ở chiều dài thanh, và
    thanh có chuyển động nên mắt bắt được ngay mô hình nào vượt trội.
    """
    lon_nhat = max(gia_tri) if gia_tri else 1.0
    lon_nhat = lon_nhat if lon_nhat > 0 else 1.0
    o = []
    for i, (ten, v) in enumerate(zip(nhan, gia_tri)):
        rong = max(v, 0.0) / lon_nhat * 100
        lop = "t1-row best" if ten == danh_dau else "t1-row"
        o.append(
            f'<div class="{lop}" style="animation-delay:{0.05 * i:.2f}s">'
            f'<div class="head"><span class="name">{_e(ten)}</span>'
            f'<span class="val">{_e(dinh_dang(v))}</span></div>'
            f'<div class="track"><div class="bar" style="--t1-w:{rong:.2f}%;'
            f'background:linear-gradient(90deg,{mau}CC,{mau})"></div></div></div>'
        )
    st.markdown(f'<div class="t1-rank">{"".join(o)}</div>', unsafe_allow_html=True)


def danh_sach(muc: list[str], mau: str) -> None:
    """Danh sách khuyến nghị, từng dòng trượt lên lần lượt."""
    o = "".join(
        f'<li style="animation-delay:{0.04 * i:.2f}s">{_e(m)}</li>'
        for i, m in enumerate(muc)
    )
    st.markdown(f'<ul class="t1-list" style="--t1-tone:{mau}">{o}</ul>',
                unsafe_allow_html=True)


def tieu_muc(chu: str) -> None:
    st.markdown(f'<p class="t1-sub">{_e(chu)}</p>', unsafe_allow_html=True)


def ghi_chu(chu: str) -> None:
    """Dòng chú thích nhỏ. Nhận HTML để nhúng được ``<code>`` và ``<b>``."""
    st.markdown(f'<p class="t1-note">{chu}</p>', unsafe_allow_html=True)


def nhan_ca_ngan(nhan: str) -> str:
    """Rút gọn nhãn ca mẫu cho vừa **một dòng** trên nút bấm.

    Nhãn trong ``graph.json`` viết đủ để đọc trong báo cáo ("Nhóm Tiền Đái tháo
    đường — bệnh nhân #444"), nhưng trên một hàng bốn nút thì nhãn dài nhất
    xuống hai dòng còn ba nhãn kia một dòng, làm cả hàng cao so le. Rút gọn ở
    chỗ hiển thị chứ **không** sửa ``graph.json`` — tệp ấy là hợp đồng dữ liệu
    dùng chung với báo cáo và bộ kiểm thử.
    """
    for bo in ("Nhóm ", "Phân khúc "):
        if nhan.startswith(bo):
            nhan = nhan[len(bo):]
    return nhan.replace(" — bệnh nhân ", " · ").replace(" — tin đăng ", " · ")


def yeu_cau_tep(duong_dan: Path, lenh: str = LENH_DUNG_LAI) -> Path:
    """Dừng trang kèm hướng dẫn cụ thể khi tệp kết quả chưa được sinh ra.

    Thà dừng hẳn còn hơn để trang chạy tiếp với dữ liệu rỗng: một bảng trống
    trông y hệt một bảng thật mà mọi giá trị đều bằng 0.
    """
    if not duong_dan.exists():
        st.error(
            f"Thiếu tệp `{duong_dan.relative_to(ROOT).as_posix()}`. "
            f"Chạy `{lenh}` ở thư mục gốc dự án để sinh lại rồi tải lại trang."
        )
        st.stop()
    return duong_dan


# ==========================================================================
#  ĐỌC KẾT QUẢ ĐÃ SINH
# ==========================================================================

@st.cache_data(show_spinner=False)
def doc_tom_tat() -> dict:
    """Đọc ``outputs/pipeline_summary.json`` — nguồn của mọi con số trên trang chủ."""
    yeu_cau_tep(SUMMARY_JSON)
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def doc_bang(ten_tep: str) -> pd.DataFrame:
    """Đọc một bảng CSV trong ``outputs/``.

    Dùng ``utf-8-sig`` vì các bảng được ghi kèm BOM cho Excel đọc đúng tiếng
    Việt; đọc bằng ``utf-8`` thường sẽ khiến tên cột đầu tiên dính ký tự vô hình.
    """
    duong_dan = yeu_cau_tep(cfg.OUTPUTS_DIR / ten_tep)
    return pd.read_csv(duong_dan, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def doc_do_thi() -> dict:
    """Đọc ``outputs/graph.json`` và **kiểm số hiệu lược đồ trước khi dùng**.

    Phase 5 ghi tệp này, Phase 6 đọc nó. Số hiệu lược đồ tồn tại chính là để
    bên đọc phát hiện được khi bên ghi đổi cấu trúc; bỏ qua bước kiểm thì lỗi
    chỉ lộ ra dưới dạng ``KeyError`` ở một trang ngẫu nhiên nào đó.
    """
    yeu_cau_tep(GRAPH_JSON, "python -m src.kg.build_graph --offline")
    tai_lieu = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))

    phien_ban = tai_lieu.get("meta", {}).get("schema_version")
    if phien_ban != SCHEMA_VERSION:
        st.error(
            f"`outputs/graph.json` dùng lược đồ phiên bản {phien_ban!r}, còn ứng "
            f"dụng đọc theo phiên bản {SCHEMA_VERSION}. Chạy lại "
            "`python -m src.kg.build_graph --offline` để sinh tệp khớp phiên bản."
        )
        st.stop()
    return tai_lieu


def ca_theo_mien(mien: str) -> list[dict]:
    """Danh sách ca mẫu của một miền. Không có khoá ``cases`` ở mức trên cùng."""
    return doc_do_thi()["domains"][mien]["cases"]


# ==========================================================================
#  DANH MỤC MÔ HÌNH
# ==========================================================================

def _slug(ten_mo_hinh: str) -> str:
    """Đổi tên hiển thị thành phần đuôi tên tệp mà ``run_pipeline.py`` đã lưu.

    "K-Nearest Neighbors" → "k_nearest_neighbors", khớp
    ``models/diabetes_k_nearest_neighbors.joblib``.
    """
    return str(ten_mo_hinh).strip().lower().replace("-", "_").replace(" ", "_")


@st.cache_data(show_spinner=False)
def danh_muc_mo_hinh(tien_to: str, bang_so_sanh: str, ten_tot_nhat: str) -> dict[str, str]:
    """Ánh xạ *tên hiển thị* → *tên tệp* cho hộp chọn mô hình.

    Danh mục dựng từ chính bảng so sánh ở ``outputs/`` chứ không viết cứng: thêm
    hay bớt một mô hình trong ``src/train.py`` là hộp chọn tự đổi theo, không
    phải sửa ứng dụng. Mô hình nào chưa có tệp trong ``models/`` thì bị bỏ qua
    thay vì để người dùng chọn rồi nhận lỗi thiếu tệp.
    """
    danh_muc: dict[str, str] = {}

    tep_tot_nhat = cfg.MODELS_DIR / f"{tien_to}_best.joblib"
    if tep_tot_nhat.exists():
        danh_muc[f"{NHAN_TOT_NHAT} — {ten_tot_nhat}"] = tep_tot_nhat.name

    for ten in doc_bang(bang_so_sanh)["Model"]:
        if str(ten).lower().startswith("baseline"):
            continue
        tep = cfg.MODELS_DIR / f"{tien_to}_{_slug(ten)}.joblib"
        if tep.exists():
            danh_muc[str(ten)] = tep.name

    if not danh_muc:
        st.error(f"Không tìm thấy mô hình nào có tiền tố `{tien_to}_` trong "
                 f"`models/`. Chạy `{LENH_DUNG_LAI}` để huấn luyện lại.")
        st.stop()
    return danh_muc


@st.cache_resource(show_spinner="Đang nạp pipeline đã huấn luyện…")
def nap_mo_hinh(ten_tep: str):
    """Nạp pipeline đã lưu. ``cache_resource`` giữ đúng một bản trong bộ nhớ."""
    return tr.load_model(ten_tep)


def chon_mo_hinh(tien_to: str, bang_so_sanh: str, ten_tot_nhat: str,
                 tro_giup: str) -> tuple[dict[str, str], str]:
    """Hộp chọn mô hình, đặt ngay trong khối kết quả mà nó chi phối.

    Bản trước để nó ở một cột riêng giữa trang, rồi ở thanh bên. Cả hai đều
    tách điều khiển ra khỏi con số nó sinh ra, nên người dùng đổi mô hình xong
    phải đưa mắt đi tìm xem cái gì vừa đổi. Đặt ngay trên vòng xác suất thì
    quan hệ nguyên nhân – kết quả nằm gọn trong một tầm nhìn.
    """
    danh_muc = danh_muc_mo_hinh(tien_to, bang_so_sanh, ten_tot_nhat)
    return danh_muc, st.selectbox("Mô hình dự đoán", list(danh_muc),
                                  help=tro_giup)


# ==========================================================================
#  DỰNG ĐẦU VÀO — BIỂU DIỄN PHẢI KHỚP LÚC HUẤN LUYỆN
# ==========================================================================

def khung_tieu_duong(gia_tri: dict[str, Any]) -> pd.DataFrame:
    """Dựng ma trận đặc trưng một dòng cho bài toán tiểu đường.

    Chỉ tám cột thô đi vào; bốn đặc trưng phái sinh, phần điền khuyết và phần
    chuẩn hoá đều nằm **bên trong** pipeline đã lưu (xem
    ``train.diabetes_preprocessor``), nên ở đây không được động tới.

    ``clean_diabetes`` vẫn phải chạy: tập gốc quy ước số 0 ở năm cột sinh học là
    "không đo được", và mô hình đã học theo quy ước ấy. Người dùng gõ 0 vào ô
    Huyết áp mà không đổi thành khuyết thì lúc dự đoán mô hình nhận một bệnh
    nhân có huyết áp bằng 0 — một biểu diễn nó chưa từng thấy lúc huấn luyện.
    """
    cot = doc_tom_tat()["diabetes"]["feature_columns"]
    khung = pd.DataFrame([{ten: gia_tri.get(ten) for ten in cot}], columns=cot)
    # Ô để trống trả về ``None``; ép sang float biến nó thành NaN để
    # ``SimpleImputer`` trong pipeline điền bằng trung vị của tập huấn luyện.
    khung = khung.astype("float64")
    return prep.clean_diabetes(khung)


def khung_bat_dong_san(gia_tri: dict[str, Any]) -> pd.DataFrame:
    """Dựng ma trận đặc trưng một dòng cho bài toán định giá.

    Khác bài toán tiểu đường ở một điểm: ba đặc trưng phái sinh của nhà đất được
    tính **ngoài** pipeline lúc huấn luyện (``prepare_housing`` gọi
    ``add_housing_features`` rồi mới đưa vào ``ColumnTransformer``). Nên ở đây
    phải gọi lại đúng hàm ấy — gọi lại, chứ không chép công thức sang.
    """
    cot = doc_tom_tat()["housing"]["feature_columns"]
    tho = pd.DataFrame([gia_tri])
    for ten in HOU_NUMERIC:
        tho[ten] = pd.to_numeric(tho[ten], errors="coerce")
    for ten in HOU_CATEGORICAL:
        # Biến định danh để trống là một hạng mục thật ("người đăng tin không
        # khai báo"), đúng như ``clean_housing`` đã làm lúc huấn luyện.
        tho[ten] = tho[ten].fillna("Không rõ")
    return prep.add_housing_features(tho)[cot]


@st.cache_data(show_spinner="Đang đọc dữ liệu gốc để lấy giá trị mặc định…")
def _bang_nha_dat_da_lam_sach() -> pd.DataFrame:
    frame, _bo_cot = prep.clean_housing(prep.load_housing_raw())
    return frame


@st.cache_data(show_spinner=False)
def mac_dinh_nha_dat() -> dict[str, float]:
    """Trung vị của từng cột định lượng — dùng làm giá trị gợi ý sẵn trên form.

    Đây đúng là con số mà ``clean_housing`` đã điền vào các ô khuyết lúc huấn
    luyện, nên người dùng không biết Mặt tiền của căn nhà thì để nguyên giá trị
    gợi ý là khớp với điều mô hình từng thấy.
    """
    bang = _bang_nha_dat_da_lam_sach()
    return {ten: float(bang[ten].median()) for ten in HOU_NUMERIC}


@st.cache_data(show_spinner=False)
def quan_theo_tinh() -> dict[str, list[str]]:
    """Danh sách Quận/Huyện có thật của từng Tỉnh/Thành, rút từ dữ liệu gốc.

    Lọc theo tỉnh thay vì đổ cả mấy trăm quận vào một hộp chọn: tập dữ liệu có
    những cặp tỉnh–quận không tồn tại trên thực tế, và để người dùng ghép bừa
    thì mô hình nhận một tổ hợp nó chưa từng gặp.
    """
    bang = _bang_nha_dat_da_lam_sach()
    nhom = bang.groupby("Province")["District"].unique()
    return {str(tinh): sorted(str(quan) for quan in ds) for tinh, ds in nhom.items()}


# ==========================================================================
#  DỰ BÁO
# ==========================================================================

def du_bao_toan_bo(danh_muc: dict[str, str], khung: pd.DataFrame,
                   phan_lop: bool) -> pd.DataFrame:
    """Chạy **mọi** mô hình trong danh mục trên cùng một đầu vào.

    Đây là phần trả lời trực tiếp cho câu hỏi "chọn mô hình khác thì kết quả
    khác thế nào": cùng một bệnh nhân, cùng một căn nhà, năm thuật toán học theo
    năm nguyên lý khác nhau cho ra năm con số khác nhau — bảng này đặt chúng
    cạnh nhau thay vì bắt người dùng bấm đi bấm lại từng mô hình rồi tự nhớ.

    Bỏ qua lựa chọn "mô hình tốt nhất" vì nó chỉ là bản sao của một mô hình đã
    có tên riêng trong danh mục; để lại thì bảng có hai dòng trùng giá trị.
    """
    dong = []
    for nhan, ten_tep in danh_muc.items():
        if nhan.startswith(NHAN_TOT_NHAT):
            continue
        mo_hinh = nap_mo_hinh(ten_tep)
        ket_qua = (float(mo_hinh.predict_proba(khung)[0, 1]) if phan_lop
                   else float(mo_hinh.predict(khung)[0]))
        dong.append({"Mô hình": nhan, "Kết quả": ket_qua})
    return pd.DataFrame(dong)


# ==========================================================================
#  VẼ ĐỒ THỊ TRI THỨC
# ==========================================================================

def ve_do_thi(nut: list[dict], canh: list[dict], chieu_cao: int = 560) -> str:
    """Vẽ một đồ thị tương tác và trả về HTML **tự chứa**, nhúng thẳng được.

    Nhận danh sách từ điển chứ không nhận ``KnowledgeGraph``, để một hàm phục vụ
    được cả hai nguồn: đồ thị dựng tại chỗ từ dữ liệu người dùng nhập
    (``KnowledgeGraph.to_dict()``) và đồ thị đọc ra từ ``outputs/graph.json``.
    Hai nguồn ấy có cùng tên trường nên không cần lớp chuyển đổi ở giữa.

    Chỉ vẽ **một ca mỗi lần**: định danh nút chỉ duy nhất trong phạm vi một ca,
    nên trộn nhiều ca vào cùng một hình sẽ khiến các nút cùng tên chập lại.
    """
    from pyvis.network import Network

    mang = Network(height=f"{chieu_cao}px", width="100%", directed=True,
                   bgcolor="#FFFFFF", font_color="#22313F",
                   cdn_resources="in_line", notebook=False)
    # Cụm chặt hơn bản đầu (spring_length 180, gravity -9000). Lý do đo được:
    # pyvis tự thu khung vẽ cho vừa toàn mạng, nên mạng càng trải rộng thì hệ số
    # thu càng lớn và **nhãn càng nhỏ** — ở bản cũ nhãn nút xuống cỡ không đọc
    # nổi. Kéo các nút lại gần nhau thì hệ số thu nhỏ đi, chữ to lên tương ứng.
    mang.barnes_hut(gravity=-4200, spring_length=105, spring_strength=0.035,
                    damping=0.5)

    for mot_nut in nut:
        chu_thich = [mot_nut["type"]]
        chu_thich += [f"{khoa}: {gia_tri}"
                      for khoa, gia_tri in (mot_nut.get("properties") or {}).items()
                      if gia_tri is not None]
        mang.add_node(
            mot_nut["id"],
            label=textwrap.shorten(mot_nut["label"], width=28, placeholder="…"),
            title="\n".join(chu_thich),
            color=NODE_COLOURS.get(mot_nut["type"], "#95A5A6"),
            shape="dot",
            size=22 if mot_nut["type"] in LOAI_GOC else 13,
            font={"size": 18, "face": "Be Vietnam Pro, Segoe UI, sans-serif",
                  "color": "#22313F", "strokeWidth": 4, "strokeColor": "#FFFFFF"},
        )

    for mot_canh in canh:
        mang.add_edge(mot_canh["source"], mot_canh["target"],
                      label=mot_canh["relation"], color="#C2CAD2",
                      font={"size": 12, "color": "#6B7A88", "strokeWidth": 4,
                            "strokeColor": "#FFFFFF"})

    return strip_cdn_links(mang.generate_html(notebook=False))


def ve_do_thi_song(kg: KnowledgeGraph, chieu_cao: int = 560) -> str:
    """Vẽ đồ thị vừa dựng từ dữ liệu người dùng nhập."""
    payload = kg.to_dict()
    return ve_do_thi(payload["nodes"], payload["edges"], chieu_cao)


def bang_bo_ba(kg: KnowledgeGraph) -> pd.DataFrame:
    """Bảng bộ ba (Chủ thể, Quan hệ, Đối tượng) — bằng chứng KG = (V, E, R)."""
    return pd.DataFrame(kg.triplets(), columns=["Chủ thể", "Quan hệ", "Đối tượng"])


def chu_thich_mau(loai: list[str]) -> None:
    """Chú giải màu theo loại thực thể, dựng từ đúng bảng màu của bản thể học."""
    o = "".join(
        f'<span class="i"><span class="d" style="background:'
        f'{NODE_COLOURS.get(ten, "#95A5A6")}"></span>{_e(ten)}</span>'
        for ten in loai
    )
    st.markdown(f'<div class="t1-legend">{o}</div>', unsafe_allow_html=True)


def khoi_do_thi_tri_thuc(kg: KnowledgeGraph, chieu_cao: int = 500) -> None:
    """Đồ thị tri thức của một ca: chú giải màu, hình vẽ, thống kê, bảng bộ ba.

    Gom lại thành một khối vì hai trang dự đoán dùng y hệt nhau; tách rời thì
    sớm muộn hai trang lệch nhau về chiều cao khung hoặc về số bộ ba hiển thị.
    """
    chu_thich_mau(sorted({n.type for n in kg.nodes}))
    st.components.v1.html(ve_do_thi_song(kg, chieu_cao=chieu_cao),
                          height=chieu_cao + 20)

    thong_ke = kg.stats()
    luoi_so([{"k": k, "v": str(v)} for k, v in thong_ke.items()], gon=True)

    with st.expander(f"Bảng bộ ba tri thức — {len(kg.edges)} bộ ba "
                     "(Chủ thể, Quan hệ, Đối tượng)"):
        st.dataframe(bang_bo_ba(kg), hide_index=True, use_container_width=True)


def xem_toan_mien(mien: str, nhan: str) -> None:
    """Đồ thị gộp cả ba ca mẫu của một miền, nhúng từ tệp đã dựng sẵn.

    Tệp ``outputs/kg_*.html`` gộp nhiều ca với định danh có tiền tố theo ca nên
    các ca không chập vào nhau — không được tự gộp ở đây rồi bỏ tiền tố.
    """
    with st.expander(f"Xem đồ thị gộp cả ba ca mẫu của miền {nhan}"):
        tep = cfg.OUTPUTS_DIR / f"kg_{mien}.html"
        if not tep.exists():
            st.info("Chưa có tệp — chạy `python -m src.kg.build_graph --offline`.")
            return
        st.components.v1.html(tep.read_text(encoding="utf-8"), height=640,
                              scrolling=False)
        if TRIPLETS_CSV.exists():
            st.download_button("⬇️ Tải toàn bộ bảng bộ ba của sáu ca (CSV)",
                               TRIPLETS_CSV.read_bytes(),
                               file_name=TRIPLETS_CSV.name, mime="text/csv")


# ==========================================================================
#  TRẠNG THÁI NEO4J
# ==========================================================================

def _cai_dat_neo4j():
    """Ưu tiên Secrets của Streamlit, rồi mới tới tệp ``.env`` ở máy cục bộ.

    Trên Streamlit Cloud không có tệp ``.env``; thông tin đăng nhập khai trong
    mục Secrets và tới được đây qua ``st.secrets``.
    """
    try:
        bi_mat = dict(st.secrets)
    except Exception:            # chưa có secrets.toml — chuyện bình thường
        bi_mat = {}
    if bi_mat:
        cai_dat = nc.settings_from_mapping(bi_mat)
        if cai_dat is not None:
            return cai_dat
    return nc.settings_from_env()


def trang_thai_neo4j() -> dict[str, Any]:
    """Thử kết nối máy chủ đồ thị và báo lại trạng thái.

    Không kết nối được **không phải là lỗi của ứng dụng**: cả ba trang đều chạy
    trọn vẹn bằng đồ thị offline. Nhưng lý do thất bại cũng không bị nuốt im
    lặng — nó được hiện thẳng ra để người dùng biết đường sửa.
    """
    cai_dat = _cai_dat_neo4j()
    if cai_dat is None:
        return {"ket_noi": False,
                "ly_do": "Chưa khai báo NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD "
                         "trong `.env` hoặc trong Secrets của Streamlit."}
    try:
        with nc.Neo4jClient(cai_dat) as client:
            return {
                "ket_noi": True,
                # KHÔNG trả về ``cai_dat.describe()``: chuỗi ấy chứa tên đăng
                # nhập và tên máy chủ Aura. In ra console lúc dựng đồ thị thì
                # không sao, nhưng trang này chạy trên bản deploy công khai —
                # ở đó nó công bố một nửa cặp thông tin đăng nhập cho bất kỳ
                # ai mở link. Tên cơ sở dữ liệu là đủ để biết đang nối vào đâu.
                "co_so_du_lieu": cai_dat.database or "mặc định",
                "phien_ban": client.server_version(),
                "so_nut": client.count_nodes(),
                "so_canh": client.count_edges(),
                "theo_loai": client.summary_by_kind(),
            }
    except nc.Neo4jUnavailable as loi:
        return {"ket_noi": False, "ly_do": str(loi)}
    except Exception as loi:     # trình điều khiển ném kiểu ngoại lệ riêng của nó
        return {"ket_noi": False, "ly_do": f"{type(loi).__name__}: {loi}"}


# ==========================================================================
#  KHỐI GÓI CHĂM SÓC VÀ GÓI TÀI CHÍNH
# ==========================================================================
#
# Hai khối này là phần "Ứng dụng" ở tầng cuối của sơ đồ hệ thống: chúng biến
# một con số dự báo thành thứ người dùng cầm đi mua được, hoặc mang tới ngân
# hàng được. Chúng đọc thẳng từ hai danh mục trong ``src/kg/`` chứ không tự khai
# lại giá hay lãi suất, nên số trên giao diện và số trong đồ thị tri thức không
# thể lệch nhau.

CSS_MO_RONG = """
<style>
/* ---------- Thẻ mặt hàng ---------- */
.t1-prods{ display:grid; gap:.85rem;
  grid-template-columns:repeat(auto-fit, minmax(258px, 1fr)); margin:.5rem 0 .2rem; }
.t1-prod{ position:relative; display:flex; flex-direction:column; gap:.42rem;
  background:var(--t1-card); border:1px solid var(--t1-line); border-radius:15px;
  padding:.95rem 1.05rem 1rem; box-shadow:var(--t1-sh);
  animation:t1-up .45s cubic-bezier(.22,1,.36,1) both;
  transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.t1-prod:hover{ transform:translateY(-3px); box-shadow:var(--t1-sh-h);
  border-color:color-mix(in srgb, var(--t1-tone) 42%, var(--t1-line)); }
.t1-prod::before{ content:""; position:absolute; inset:0 0 auto 0; height:3px;
  border-radius:15px 15px 0 0;
  background:linear-gradient(90deg,var(--t1-tone),transparent 90%); }
.t1-prod .sku{ font-size:.68rem; font-weight:700; letter-spacing:.08em;
  color:var(--t1-muted); font-family:ui-monospace,Consolas,monospace; }
.t1-prod .ten{ font-size:.95rem; font-weight:700; line-height:1.45; margin:0;
  color:var(--t1-ink); }
.t1-prod .gia{ font-size:1.02rem; font-weight:800; color:var(--t1-tone);
  line-height:1.35; }
.t1-prod .cd{ font-size:.82rem; line-height:1.6; color:#3C4C5B; margin:0; }
.t1-prod .ft{ display:flex; flex-wrap:wrap; align-items:center; gap:.4rem;
  margin-top:auto; padding-top:.5rem; }
.t1-prod a{ font-size:.79rem; font-weight:700; text-decoration:none;
  color:var(--t1-tone); border-bottom:1px dashed currentColor; }
.t1-prod a:hover{ border-bottom-style:solid; }

/* ---------- Bảng điều khoản ---------- */
.t1-terms{ width:100%; border-collapse:separate; border-spacing:0;
  border:1px solid var(--t1-line); border-radius:14px; overflow:hidden;
  box-shadow:var(--t1-sh); margin:.45rem 0 .2rem; }
.t1-terms td{ padding:.6rem .95rem; font-size:.9rem; border-top:1px solid var(--t1-line); }
.t1-terms tr:first-child td{ border-top:0; }
.t1-terms td:first-child{ color:var(--t1-muted); width:52%; }
.t1-terms td:last-child{ text-align:right; font-weight:700; color:var(--t1-ink);
  font-variant-numeric:tabular-nums; }
.t1-terms tr.nhan td:last-child{ color:var(--t1-tone); font-size:1.02rem; }

/* ---------- Nhóm tiện ích ---------- */
.t1-utils{ display:grid; gap:.6rem;
  grid-template-columns:repeat(auto-fit, minmax(178px, 1fr)); margin:.45rem 0 .2rem; }
.t1-util{ display:flex; align-items:center; gap:.6rem; padding:.68rem .8rem;
  border:1px solid var(--t1-line); border-radius:13px; background:var(--t1-card);
  animation:t1-up .45s cubic-bezier(.22,1,.36,1) both; }
.t1-util .ic{ width:34px; height:34px; border-radius:10px; display:grid;
  place-items:center; font-size:1.05rem; flex:0 0 auto;
  background:color-mix(in srgb, var(--t1-tone) 14%, #fff);
  border:1px solid color-mix(in srgb, var(--t1-tone) 28%, #fff); }
.t1-util .tx b{ display:block; font-size:.88rem; line-height:1.4; }
.t1-util .tx span{ font-size:.74rem; color:var(--t1-muted); }
@media (prefers-reduced-motion:reduce){
  .t1-prod, .t1-util{ animation:none !important; }
}
</style>
"""

# Biểu tượng cho từng nhóm chức năng của tiện ích đô thị.
_ICON_TIEN_ICH = {
    "Thương mại": "🛒",
    "Giáo dục": "🎓",
    "Y tế": "🏥",
    "Giao thông": "🚌",
}


def the_san_pham(san_pham: list[dict], mau: str) -> None:
    """Lưới thẻ mặt hàng của gói chăm sóc, mỗi thẻ dẫn thẳng tới trang tra giá.

    Đường dẫn mở tab mới và mang ``rel="noopener"``: thiếu thuộc tính ấy thì
    trang đích truy được vào ``window.opener`` của ứng dụng.
    """
    o = []
    for i, sp in enumerate(san_pham):
        chip = ('<span class="t1-chip good"><span class="dot"></span>Mua định kỳ</span>'
                if sp["dinh_ky"] else
                '<span class="t1-chip"><span class="dot"></span>Mua một lần</span>')
        o.append(
            f'<div class="t1-prod" style="--t1-tone:{mau};animation-delay:{0.05 * i:.2f}s">'
            f'<span class="sku">{_e(sp["sku"])} · {_e(sp["nhom_hang"])}</span>'
            f'<p class="ten">{_e(sp["name"])}</p>'
            f'<span class="gia">{_e(pharm.dinh_dang_khoang(sp["gia_min"], sp["gia_max"]))}'
            f' <span style="font-weight:600;font-size:.78rem;color:var(--t1-muted)">'
            f'/ {_e(sp["unit"])}</span></span>'
            f'<p class="cd">{_e(sp["cong_dung"])}</p>'
            f'<div class="ft">{chip}'
            f'<a href="{_e(sp["tra_cuu"])}" target="_blank" rel="noopener">Tra giá hôm nay ↗</a>'
            f'</div></div>'
        )
    st.markdown(f'<div class="t1-prods">{"".join(o)}</div>', unsafe_allow_html=True)


def bang_dieu_khoan(dong: list[tuple[str, str]], mau: str,
                    nhan_manh: set[str] | None = None) -> None:
    """Bảng hai cột nhãn – giá trị cho điều khoản vay hoặc chi phí gói.

    ``nhan_manh`` là tập nhãn được tô đậm màu chủ đề — dành cho một hoặc hai con
    số mà người đọc thật sự đi tìm, phần còn lại giữ nguyên để không cạnh tranh.
    """
    nhan_manh = nhan_manh or set()
    hang = "".join(
        f'<tr class="{"nhan" if k in nhan_manh else ""}">'
        f'<td>{_e(k)}</td><td>{_e(v)}</td></tr>'
        for k, v in dong
    )
    st.markdown(
        f'<table class="t1-terms" style="--t1-tone:{mau}">{hang}</table>',
        unsafe_allow_html=True,
    )


def luoi_tien_ich(tien_ich: list[dict], ban_kinh: float, mau: str) -> None:
    """Lưới tiện ích hạ tầng đô thị, nhóm theo bốn loại chức năng."""
    o = []
    for i, ti in enumerate(tien_ich):
        icon = _ICON_TIEN_ICH.get(ti["loai"], "📍")
        o.append(
            f'<div class="t1-util" style="--t1-tone:{mau};animation-delay:{0.05 * i:.2f}s">'
            f'<div class="ic">{icon}</div>'
            f'<div class="tx"><b>{_e(ti["ten"])}</b>'
            f'<span>{_e(ti["loai"])} · trong bán kính {ban_kinh:g} km</span></div>'
            f'</div>'
        )
    st.markdown(f'<div class="t1-utils">{"".join(o)}</div>', unsafe_allow_html=True)


def lich_tra_no(han_muc: float, lai_suat_nam: float, ky_han_nam: int) -> pd.DataFrame:
    """Lịch trả nợ rút gọn theo năm — gốc, lãi và dư nợ còn lại.

    Tính lặp theo từng **tháng** rồi mới gộp theo năm, chứ không áp lãi suất năm
    lên dư nợ đầu năm: cách sau bỏ qua việc dư nợ giảm dần trong 12 tháng và
    khai vống tiền lãi lên vài phần trăm.
    """
    r = lai_suat_nam / 12
    thang = urban.tra_gop_hang_thang(han_muc, lai_suat_nam, ky_han_nam)
    du_no = han_muc
    hang = []

    for nam in range(1, ky_han_nam + 1):
        goc_nam = lai_nam = 0.0
        for _ in range(12):
            lai = du_no * r
            goc = thang - lai
            du_no = max(0.0, du_no - goc)
            goc_nam += goc
            lai_nam += lai
        hang.append({
            "Năm": nam,
            "Trả gốc (tỷ)": round(goc_nam, 4),
            "Trả lãi (tỷ)": round(lai_nam, 4),
            "Dư nợ cuối năm (tỷ)": round(du_no, 4),
        })

    return pd.DataFrame(hang)
