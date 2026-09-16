# -*- coding: utf-8 -*-
"""Sinh 4 hình minh hoạ Đồ thị Tri thức Neo4j cho báo cáo Assignment 02.

Chạy lại bằng:  python report/make_neo4j_figures.py

Bốn hình được xuất ra ``report/figures/``:

* ``fig_kg_schema.png``    — sơ đồ bản thể học (loại thực thể + loại quan hệ, kèm
  số lượng đếm thật từ Aura).
* ``fig_kg_full.png``      — toàn bộ đồ thị thật, tách theo 6 thành phần liên
  thông (3 ca y tế + 3 ca bất động sản).
* ``fig_kg_case_high.png`` — đồ thị của một ca nguy cơ cao (``case = med_high``).
* ``fig_kg_pipeline.png``  — sơ đồ luồng xử lý, nêu rõ Neo4j không tham gia dự đoán.

Hai hình sơ đồ khối (schema, pipeline) được dựng bằng HTML/CSS/SVG rồi chụp bằng
Chrome headless ở tỉ lệ 2x — cùng cách với chín hình lý thuyết sẵn có. Hai hình
đồ thị thật được vẽ bằng networkx + matplotlib từ dữ liệu đọc trực tiếp trên
Neo4j Aura; script không bao giờ tự bịa dữ liệu, mất kết nối là dừng và báo lỗi.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
#  Đường dẫn
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] / "src" / "Assignment 02"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
#  Nạp biến môi trường từ .env (tự phân tích, không phụ thuộc python-dotenv)
# --------------------------------------------------------------------------
def load_env(path: Path) -> None:
    """Đọc tệp .env theo cú pháp KHOÁ=GIÁ_TRỊ, bỏ qua dòng trống và chú thích."""
    if not path.exists():
        raise SystemExit(f"! Không tìm thấy tệp cấu hình {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------
#  Bảng màu — giữ nguyên bảng màu của Assignment 01 để hai báo cáo nhất quán
#  (nguồn: HTTM/src/kg/ontology.py, biến NODE_COLOURS)
# --------------------------------------------------------------------------
NODE_COLOURS = {
    "Patient": "#E74C3C",
    "Biomarker": "#3498DB",
    "Model": "#9B59B6",
    "Prediction": "#E67E22",
    "Disease": "#C0392B",
    "RiskTier": "#16A085",
    "Device": "#2980B9",
    "Nutrition": "#27AE60",
    "Action": "#7F8C8D",
    "Service": "#F39C12",
    "CarePackage": "#1F6F8B",
    "Retailer": "#B8860B",
    "Property": "#E74C3C",
    "Attribute": "#3498DB",
    "Location": "#16A085",
    "Segment": "#8E44AD",
    "LoanPackage": "#D35400",
    "Repayment": "#A0522D",
    "Requirement": "#5D6D7E",
    "Infrastructure": "#1F6F8B",
    "Utility": "#27AE60",
    "Legal": "#7F8C8D",
}

# Tên tiếng Việt của từng loại thực thể — dùng cho chú giải và sơ đồ bản thể học.
KIND_VI = {
    "Patient": "Bệnh nhân",
    "Biomarker": "Chỉ số sinh học",
    "Model": "Mô hình học máy",
    "Prediction": "Kết quả dự đoán",
    "Disease": "Mã bệnh ICD-10",
    "RiskTier": "Tầng nguy cơ",
    "Device": "Thiết bị y tế",
    "Nutrition": "Dinh dưỡng",
    "Action": "Hành động khuyến nghị",
    "Service": "Dịch vụ",
    "CarePackage": "Gói chăm sóc",
    "Retailer": "Nhà bán lẻ",
    "Property": "Bất động sản",
    "Attribute": "Thuộc tính nhà",
    "Location": "Địa điểm",
    "Segment": "Phân khúc giá",
    "LoanPackage": "Gói vay",
    "Repayment": "Kế hoạch trả góp",
    "Requirement": "Điều kiện thu nhập",
    "Infrastructure": "Cụm hạ tầng",
    "Utility": "Tiện ích",
    "Legal": "Tình trạng pháp lý",
}

# Tên tiếng Việt của từng loại quan hệ.
REL_VI = {
    "HAS_BIOMARKER": "có chỉ số",
    "CALCULATED_RISK": "đặc trưng dẫn xuất",
    "EVALUATED_BY": "được đánh giá bởi",
    "PRODUCES": "sinh ra",
    "MAPS_TO_DISEASE": "ánh xạ sang mã bệnh",
    "STRATIFIED_AS": "phân tầng thành",
    "ADVISES": "khuyên nên",
    "RECOMMENDS_PACKAGE": "gợi ý gói",
    "REQUIRES_DEVICE": "cần thiết bị",
    "RECOMMENDS_NUTRITION": "gợi ý dinh dưỡng",
    "REFERS_TO": "chuyển tuyến tới",
    "INCLUDES_PRODUCT": "gồm sản phẩm",
    "SUPPLIED_BY": "được cung cấp bởi",
    "FULFILLED_BY": "được đáp ứng bởi",
    "PROVIDES_SERVICE": "cung cấp dịch vụ",
    "HAS_ATTRIBUTE": "có thuộc tính",
    "LOCATED_IN": "toạ lạc tại",
    "PART_OF": "thuộc về",
    "VALUATED_BY": "được định giá bởi",
    "BELONGS_TO_SEGMENT": "thuộc phân khúc",
    "ELIGIBLE_FOR": "đủ điều kiện vay",
    "PROVIDED_BY": "được cấp bởi",
    "HAS_LEGAL_STATUS": "có pháp lý",
    "QUALIFIES_FOR": "đủ điều kiện cho",
    "AMORTISED_AS": "trả góp theo",
    "REQUIRES_INCOME": "yêu cầu thu nhập",
    "HAS_INFRASTRUCTURE": "có hạ tầng",
    "INCLUDES_UTILITY": "gồm tiện ích",
}

CASE_VI = {
    "med_high": "Ca y tế — nguy cơ cao",
    "med_moderate": "Ca y tế — tiền đái tháo đường",
    "med_low": "Ca y tế — khoẻ mạnh",
    "prop_luxury": "Ca nhà đất — cao cấp",
    "prop_mid": "Ca nhà đất — trung cấp",
    "prop_afford": "Ca nhà đất — bình dân",
}
CASE_ORDER = ["med_high", "med_moderate", "med_low",
              "prop_luxury", "prop_mid", "prop_afford"]

MEDICAL_KINDS = {"Patient", "Biomarker", "Model", "Prediction", "Disease",
                 "RiskTier", "Device", "Nutrition", "Action", "Service",
                 "CarePackage", "Retailer"}


def tint(hex_colour: str, ratio: float = 0.86) -> str:
    """Pha một màu với nền trắng để lấy nền nhạt cùng tông."""
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * ratio)
    g = int(g + (255 - g) * ratio)
    b = int(b + (255 - b) * ratio)
    return f"#{r:02X}{g:02X}{b:02X}"


# ==========================================================================
#  1. ĐỌC DỮ LIỆU THẬT TỪ NEO4J AURA
# ==========================================================================
def fetch_graph() -> dict:
    """Truy vấn toàn bộ đồ thị. Mất kết nối thì dừng hẳn, không sinh dữ liệu giả."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise SystemExit("! Chưa cài gói neo4j. Chạy: pip install neo4j")

    load_env(ROOT / ".env")
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    if not (uri and user and password):
        raise SystemExit("! Thiếu NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD trong .env")

    print(f"-> Ket noi Neo4j: {uri} (database={database})")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            labels = session.run("CALL db.labels()").value()
            rel_types = session.run("CALL db.relationshipTypes()").value()
            node_total = session.run("MATCH (n) RETURN count(n)").single()[0]
            rel_total = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]

            nodes = [dict(r["p"]) for r in session.run(
                "MATCH (n) RETURN properties(n) AS p")]
            edges = [(r["a"], r["t"], r["b"]) for r in session.run(
                "MATCH (a)-[r]->(b) RETURN a.uid AS a, type(r) AS t, b.uid AS b")]

            kind_counts = {r["k"]: r["c"] for r in session.run(
                "MATCH (n) RETURN n.kind AS k, count(*) AS c ORDER BY c DESC")}
            rel_counts = {r["t"]: r["c"] for r in session.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC")}
            # Dem tach theo mien: mot loai nhu :Model hay PRODUCES ton tai o ca hai
            # mien, nen con so gop lai se sai khi in len tung khung so do.
            kind_by_domain = {(r["d"], r["k"]): r["c"] for r in session.run(
                "MATCH (n) RETURN n.domain AS d, n.kind AS k, count(*) AS c")}
            rel_by_domain = {(r["d"], r["t"]): r["c"] for r in session.run(
                "MATCH (a)-[r]->(b) RETURN a.domain AS d, type(r) AS t, count(*) AS c")}
        driver.close()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "! KHONG KET NOI DUOC NEO4J AURA - dung lai, khong ve hinh bia.\n"
            f"  Chi tiet: {exc!r}")

    print(f"   OK: {node_total} thuc the, {rel_total} quan he, "
          f"{len(labels)} nhan, {len(rel_types)} loai quan he")
    return {
        "labels": labels, "rel_types": rel_types,
        "node_total": node_total, "rel_total": rel_total,
        "nodes": nodes, "edges": edges,
        "kind_counts": kind_counts, "rel_counts": rel_counts,
        "kind_by_domain": kind_by_domain, "rel_by_domain": rel_by_domain,
    }


# ==========================================================================
#  2. CHỤP HTML BẰNG CHROME HEADLESS (dùng cho hình 1 và hình 4)
# ==========================================================================
def find_chrome() -> str:
    for candidate in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                      r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                      r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                      r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]:
        if Path(candidate).exists():
            return candidate
    found = shutil.which("chrome") or shutil.which("msedge") or shutil.which("chromium")
    if found:
        return found
    raise SystemExit("! Khong tim thay Chrome/Edge de chup so do khoi.")


def shoot(html: str, out_png: Path, width: int, height: int) -> None:
    """Ghi HTML ra tệp tạm rồi chụp màn hình ở tỉ lệ 2x, nền trắng."""
    chrome = find_chrome()
    if out_png.exists():
        out_png.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "figure.html"
        html_path.write_text(html, encoding="utf-8")
        profile = Path(tmp) / "profile"
        cmd = [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
               "--default-background-color=FFFFFFFF",
               f"--user-data-dir={profile}",
               f"--screenshot={out_png}",
               f"--window-size={width},{height}",
               "--force-device-scale-factor=2",
               "--virtual-time-budget=3000",
               html_path.as_uri()]
        subprocess.run(cmd, capture_output=True, timeout=180)
    if not out_png.exists():
        raise SystemExit(f"! Chrome khong tao duoc {out_png}")
    print(f"   OK {out_png.name} ({width * 2}x{height * 2} px)")


CSS_BASE = """
  html,body{margin:0;padding:0;background:#fff;}
  *{box-sizing:border-box;font-family:"Segoe UI","Be Vietnam Pro","Noto Sans",Arial,sans-serif;}
"""


# ==========================================================================
#  3. HÌNH 1 — SƠ ĐỒ BẢN THỂ HỌC
# ==========================================================================
# Toạ độ khai báo tường minh (cột, hàng) để sơ đồ ổn định giữa các lần chạy.
MED_LAYOUT = {
    "Patient": (0, 0), "Model": (1, 0), "Prediction": (2, 0), "Disease": (3, 0),
    "Device": (4, 0), "Retailer": (5, 0),
    "Biomarker": (0, 2), "RiskTier": (1, 2), "CarePackage": (2, 2),
    "Nutrition": (4, 2), "Service": (5, 2),
    "Action": (1, 3),
}
# Moi canh: (nguon, quan he, dich, vi tri nhan tren doan 0..1, co in nhan hay khong).
# Vi tri nhan duoc chinh tay o vai canh dai de nhan khong de len hop khac.
MED_EDGES = [
    ("Patient", "HAS_BIOMARKER", "Biomarker", 0.50, True),
    ("Patient", "EVALUATED_BY", "Model", 0.50, True),
    ("Patient", "STRATIFIED_AS", "RiskTier", 0.50, True),
    ("Model", "PRODUCES", "Prediction", 0.50, True),
    ("Prediction", "MAPS_TO_DISEASE", "Disease", 0.50, True),
    ("Disease", "REQUIRES_DEVICE", "Device", 0.50, True),
    ("Disease", "RECOMMENDS_NUTRITION", "Nutrition", 0.62, True),
    ("Disease", "REFERS_TO", "Service", 0.74, True),
    ("Device", "SUPPLIED_BY", "Retailer", 0.50, True),
    ("Nutrition", "SUPPLIED_BY", "Retailer", 0.50, False),
    ("Retailer", "PROVIDES_SERVICE", "Service", 0.50, True),
    ("RiskTier", "ADVISES", "Action", 0.50, True),
    ("RiskTier", "RECOMMENDS_PACKAGE", "CarePackage", 0.50, True),
    ("CarePackage", "INCLUDES_PRODUCT", "Device", 0.45, True),
    ("CarePackage", "INCLUDES_PRODUCT", "Nutrition", 0.50, False),
    ("CarePackage", "FULFILLED_BY", "Retailer", 0.18, True),
]
PROP_LAYOUT = {
    "Property": (0, 0), "Model": (1, 0), "Prediction": (2, 0), "Segment": (3, 0),
    "LoanPackage": (4, 0), "Repayment": (5, 0), "Requirement": (5, 1),
    "Attribute": (0, 2), "Location": (1, 2), "Infrastructure": (2, 2),
    "Utility": (3, 2), "Service": (4, 3), "Legal": (0, 3),
}
PROP_EDGES = [
    ("Property", "HAS_ATTRIBUTE", "Attribute", 0.50, True),
    # Property va Legal cung o cot 0 (canh doc) nen trung duong voi canh
    # Property->Attribute; dat nhan gan dau canh (gan Property) de khong
    # de len tieu de hop :Attribute nam giua duong di.
    ("Property", "HAS_LEGAL_STATUS", "Legal", 0.12, True),
    ("Property", "VALUATED_BY", "Model", 0.50, True),
    ("Property", "LOCATED_IN", "Location", 0.50, True),
    ("Model", "PRODUCES", "Prediction", 0.50, True),
    ("Prediction", "BELONGS_TO_SEGMENT", "Segment", 0.50, True),
    ("Segment", "ELIGIBLE_FOR", "LoanPackage", 0.50, True),
    ("Legal", "QUALIFIES_FOR", "LoanPackage", 0.30, True),
    ("LoanPackage", "AMORTISED_AS", "Repayment", 0.50, True),
    ("LoanPackage", "PROVIDED_BY", "Service", 0.50, True),
    ("Repayment", "REQUIRES_INCOME", "Requirement", 0.50, True),
    ("Location", "HAS_INFRASTRUCTURE", "Infrastructure", 0.50, True),
    ("Infrastructure", "INCLUDES_UTILITY", "Utility", 0.50, True),
]

COL_X = [30, 250, 470, 690, 910, 1130]
ROW_Y = [26, 118, 210, 302]
BOX_W, BOX_H = 200, 62


def _panel(layout, edges, data, top, domain):
    """Sinh HTML + SVG cho khung so do cua mot mien ung dung.

    So luong in tren moi hop va moi canh la so dem RIENG cua mien do — mot loai
    nhu :Model hay PRODUCES ton tai o ca hai mien, dung so gop se sai.
    """
    kind_n = data["kind_by_domain"]
    rel_n = data["rel_by_domain"]
    boxes, arrows, labels = [], [], []
    centre = {}
    for kind, (col, row) in layout.items():
        x, y = COL_X[col], top + ROW_Y[row]
        centre[kind] = (x + BOX_W / 2, y + BOX_H / 2)
        colour = NODE_COLOURS.get(kind, "#64748B")
        boxes.append(
            f'<div class="node" style="left:{x}px;top:{y}px;'
            f'border-color:{colour};background:{tint(colour)};">'
            f'<div class="ntitle" style="color:{colour};">{KIND_VI.get(kind, kind)}</div>'
            f'<div class="nsub">:{kind} <b>({kind_n.get((domain, kind), 0)})</b></div></div>')

    for src, rel, dst, pos_t, show in edges:
        x1, y1 = centre[src]
        x2, y2 = centre[dst]
        dx, dy = x2 - x1, y2 - y1
        dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / dist, dy / dist

        def trim(cx, cy, sign):
            tx = (BOX_W / 2 + 4) / abs(ux) if abs(ux) > 1e-6 else 9e9
            ty = (BOX_H / 2 + 4) / abs(uy) if abs(uy) > 1e-6 else 9e9
            t = min(tx, ty)
            return cx + sign * ux * t, cy + sign * uy * t

        sx1, sy1 = trim(x1, y1, 1)
        sx2, sy2 = trim(x2, y2, -1)
        arrows.append(f'<path d="M {sx1:.1f} {sy1:.1f} L {sx2:.1f} {sy2:.1f}" '
                      f'stroke="#94a3b8" stroke-width="1.6" fill="none" '
                      f'marker-end="url(#ah)"/>')
        if not show:
            continue
        mx = sx1 + (sx2 - sx1) * pos_t
        my = sy1 + (sy2 - sy1) * pos_t
        labels.append(f'<div class="rel" style="left:{mx:.0f}px;top:{my:.0f}px;">'
                      f'{rel} <span class="relc">({rel_n.get((domain, rel), 0)})</span></div>')
    return "".join(boxes), "".join(arrows), "".join(labels)


def figure_schema(data: dict) -> None:
    counts = data["kind_counts"]
    med_c = sum(v for (d, _), v in data["kind_by_domain"].items() if d == "medical")
    prop_c = data["node_total"] - med_c
    m_boxes, m_arrows, m_labels = _panel(MED_LAYOUT, MED_EDGES, data, 78, "medical")
    p_boxes, p_arrows, p_labels = _panel(PROP_LAYOUT, PROP_EDGES, data, 512, "property")

    html = f"""<!DOCTYPE html><html lang="vi"><head><meta charset="UTF-8"><style>{CSS_BASE}
  .canvas{{width:1400px;height:960px;position:relative;background:#fff;}}
  .title{{position:absolute;top:14px;left:0;width:1400px;text-align:center;
        font-size:19px;font-weight:700;color:#0f172a;}}
  .sub{{position:absolute;top:41px;left:0;width:1400px;text-align:center;
        font-size:12.5px;color:#64748b;}}
  svg{{position:absolute;top:0;left:0;width:1400px;height:960px;}}
  .node{{position:absolute;width:{BOX_W}px;height:{BOX_H}px;border:2px solid;
        border-radius:10px;padding:9px 6px 0 6px;text-align:center;
        box-shadow:0 1px 3px rgba(0,0,0,.07);}}
  .ntitle{{font-size:13px;font-weight:700;line-height:1.2;}}
  .nsub{{font-size:10.5px;color:#475569;margin-top:4px;font-family:Consolas,monospace;}}
  .rel{{position:absolute;transform:translate(-50%,-50%);background:#fff;
        border:1px solid #e2e8f0;border-radius:5px;padding:1px 5px;
        font-size:8.6px;font-weight:600;color:#475569;
        font-family:Consolas,monospace;white-space:nowrap;}}
  .relc{{color:#0d9488;font-weight:700;}}
  .band{{position:absolute;left:18px;width:1364px;border-radius:12px;
        border:1.5px dashed #cbd5e1;}}
  .bandlab{{position:absolute;font-size:13px;font-weight:700;color:#1e293b;
        background:#fff;padding:0 8px;}}
  .foot{{position:absolute;bottom:12px;left:0;width:1400px;text-align:center;
        font-size:11.5px;color:#64748b;line-height:1.6;}}
</style></head><body><div class="canvas">
  <div class="title">Bản thể học Đồ thị Tri thức Neo4j — các loại thực thể và quan hệ</div>
  <div class="sub">Số trong ngoặc là số lượng thật đếm trên Neo4j Aura:
    <b>{data['node_total']}</b> thực thể, <b>{data['rel_total']}</b> quan hệ,
    <b>{len(counts)}</b> loại thực thể, <b>{len(data['rel_types'])}</b> loại quan hệ</div>
  <div class="band" style="top:62px;height:404px;"></div>
  <div class="bandlab" style="top:54px;left:44px;">MIỀN Y TẾ — dự đoán nguy cơ đái tháo đường ({med_c} thực thể)</div>
  <div class="band" style="top:496px;height:404px;"></div>
  <div class="bandlab" style="top:488px;left:44px;">MIỀN BẤT ĐỘNG SẢN — định giá nhà và tư vấn gói vay ({prop_c} thực thể)</div>
  <svg><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>
    {m_arrows}{p_arrows}</svg>
  {m_boxes}{p_boxes}{m_labels}{p_labels}
  <div class="foot">Mọi thực thể mang đồng thời nhãn chung <b>:Entity</b> và một nhãn riêng theo loại.
    Thuộc tính <b>case</b> tách sáu tình huống mẫu, thuộc tính <b>domain</b> tách hai miền ứng dụng.<br>
    Truy vấn tư vấn của API đi theo đường <b>RiskTier −[:ADVISES]→ Action</b> và
    <b>RiskTier −[:RECOMMENDS_PACKAGE]→ CarePackage −[:INCLUDES_PRODUCT]→ Device / Nutrition</b>.</div>
</div></body></html>"""
    shoot(html, FIG_DIR / "fig_kg_schema.png", 1400, 960)


# ==========================================================================
#  4. HÌNH 4 — SƠ ĐỒ LUỒNG XỬ LÝ
# ==========================================================================
PIPELINE_STEPS = [
    ("1", "Người dùng nhập", "5 chỉ số lâm sàng trên web hoặc ứng dụng", "c1"),
    ("2", "REST API", "Flask <code>POST /predict</code>, kiểm tra dữ liệu vào", "c2"),
    ("3", "Mô hình học máy", "Pipeline <code>.joblib</code> đã huấn luyện sẵn", "c3"),
    ("4", "Xác suất dương tính", "<code>p</code> trong đoạn [0; 1] — kết quả đã chốt", "c4"),
    ("5", "Ánh xạ tầng nguy cơ", "p ≥ 0,25 cao · p ≥ 0,15 vừa · còn lại thấp", "c5"),
    ("6", "Truy vấn Cypher", "<code>MATCH (t:RiskTier {case})</code> trên Neo4j", "c6"),
    ("7", "Tri thức trả về", "Hành động · Thiết bị · Dinh dưỡng · Dịch vụ", "c7"),
    ("8", "Hiển thị kết quả", "Nhãn, xác suất và phần tư vấn kèm theo", "c8"),
]


def figure_pipeline() -> None:
    steps = []
    for i, (num, label, cap, cls) in enumerate(PIPELINE_STEPS):
        steps.append(f'<div class="step {cls}"><div class="badge">{num}</div>'
                     f'<div class="label">{label}</div><div class="cap">{cap}</div></div>')
        if i < len(PIPELINE_STEPS) - 1:
            steps.append('<div class="arrow">&#8594;</div>')
    html = f"""<!DOCTYPE html><html lang="vi"><head><meta charset="UTF-8"><style>{CSS_BASE}
  .canvas{{width:1400px;height:440px;position:relative;background:#fff;}}
  .title{{position:absolute;top:16px;left:0;width:1400px;text-align:center;
        font-size:18px;font-weight:700;color:#0f172a;}}
  .sub{{position:absolute;top:42px;left:0;width:1400px;text-align:center;
        font-size:12px;color:#64748b;}}
  .row{{position:absolute;top:122px;left:24px;width:1352px;height:168px;
        display:flex;align-items:center;justify-content:space-between;}}
  .step{{width:148px;height:158px;border-radius:12px;border:1.5px solid #cbd5e1;
        box-shadow:0 2px 6px rgba(0,0,0,.06);position:relative;
        padding:24px 9px 8px 9px;text-align:center;}}
  .badge{{position:absolute;top:-14px;left:50%;transform:translateX(-50%);
        width:28px;height:28px;border-radius:50%;color:#fff;font-weight:700;
        font-size:13px;display:flex;align-items:center;justify-content:center;
        box-shadow:0 2px 4px rgba(0,0,0,.15);}}
  .label{{font-size:12.5px;font-weight:700;color:#1e293b;line-height:1.3;}}
  .cap{{font-size:10.3px;color:#64748b;margin-top:7px;line-height:1.45;}}
  .cap code{{font-family:Consolas,monospace;font-size:9.6px;color:#0f172a;}}
  .arrow{{width:20px;text-align:center;color:#94a3b8;font-size:20px;font-weight:700;}}
  .c1{{background:#eff6ff;}} .c1 .badge{{background:#2563eb;}}
  .c2{{background:#eff6ff;}} .c2 .badge{{background:#3b82f6;}}
  .c3{{background:#f5f3ff;}} .c3 .badge{{background:#7c3aed;}}
  .c4{{background:#fef2f2;border-color:#dc2626;}} .c4 .badge{{background:#dc2626;}}
  .c5{{background:#fffbeb;}} .c5 .badge{{background:#d97706;}}
  .c6{{background:#f0fdfa;}} .c6 .badge{{background:#0d9488;}}
  .c7{{background:#f0fdfa;}} .c7 .badge{{background:#14b8a6;}}
  .c8{{background:#ecfdf5;border-color:#059669;}} .c8 .badge{{background:#059669;}}
  .lane{{position:absolute;height:24px;border-radius:6px;font-size:11.5px;
        font-weight:700;display:flex;align-items:center;justify-content:center;}}
  .lane1{{left:24px;width:854px;top:90px;background:#e0e7ff;color:#3730a3;}}
  .lane2{{left:894px;width:482px;top:90px;background:#ccfbf1;color:#115e59;}}
  .note{{position:absolute;left:24px;top:318px;width:1352px;border-radius:10px;
        border:1.5px solid #dc2626;background:#fef2f2;padding:12px 16px;
        font-size:12.5px;color:#7f1d1d;line-height:1.6;}}
  .note b{{color:#b91c1c;}}
</style></head><body><div class="canvas">
  <div class="title">Luồng xử lý một yêu cầu dự đoán và vai trò của Đồ thị Tri thức</div>
  <div class="sub">Hai đường tách bạch: dự đoán chạy trước và độc lập, tri thức chỉ được ghép vào sau khi đã có xác suất</div>
  <div class="lane lane1">ĐƯỜNG DỰ ĐOÁN — bắt buộc, quyết định nhãn và xác suất</div>
  <div class="lane lane2">ĐƯỜNG BỔ SUNG TRI THỨC — tuỳ chọn</div>
  <div class="row">{''.join(steps)}</div>
  <div class="note"><b>Neo4j KHÔNG tham gia dự đoán, chỉ bổ sung tri thức.</b>
    Nhãn và xác suất được chốt hoàn toàn ở bước 3–4 bởi mô hình học máy; đồ thị chỉ được
    truy vấn ở bước 6, sau khi xác suất đã có, và chỉ để lấy nội dung tư vấn đi kèm.
    Vì vậy khi Neo4j mất kết nối hoặc trả lỗi, API vẫn trả về đúng nhãn và đúng xác suất —
    phần tư vấn được thay bằng một thông báo, còn kết quả dự đoán không hề thay đổi.</div>
</div></body></html>"""
    shoot(html, FIG_DIR / "fig_kg_pipeline.png", 1400, 440)


# ==========================================================================
#  5. CHUẨN BỊ MATPLOTLIB — bảo đảm hiển thị được tiếng Việt có dấu
# ==========================================================================
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ["Segoe UI", "Arial", "Tahoma", "Verdana", "DejaVu Sans"]:
        if candidate in available:
            rcParams["font.family"] = candidate
            print(f"-> Phong chu cho hinh do thi: {candidate}")
            break
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    return matplotlib


def check_glyphs(records) -> None:
    """Cảnh báo nếu matplotlib báo thiếu ký tự — dấu hiệu chữ bị vẽ thành ô vuông."""
    missing = [str(w.message) for w in records if "missing from font" in str(w.message)]
    if missing:
        print("   !! CANH BAO: phong chu thieu ky tu tieng Viet - "
              f"{len(missing)} luot. Vi du: {missing[0][:140]}")
    else:
        print("   OK: khong co ky tu nao thieu phong (tieng Viet du dau).")


def wrap(text: str, width: int, lines: int = 3) -> str:
    return "\n".join(textwrap.wrap(str(text), width)[:lines])


def build_digraph(nodes, edges):
    import networkx as nx
    by_uid = {n["uid"]: n for n in nodes}
    G = nx.DiGraph()
    for uid, node in by_uid.items():
        G.add_node(uid, **node)
    for a, t, b in edges:
        if a in by_uid and b in by_uid:
            G.add_edge(a, b, rel=t)
    return G, by_uid


def layered_positions(G, kind_layer: dict, kind_rank_order: list) -> dict:
    """Bố cục phân tầng: mỗi loại thực thể nằm cố định một tầng theo mạch suy
    luận của hệ thống, xếp trái sang phải; các đỉnh cùng tầng dàn đều theo
    chiều dọc. Không dùng lực đẩy ngẫu nhiên như spring_layout nên nhãn không
    bao giờ chồng lên nhau dù panel nhỏ hay nhiều đỉnh. Dùng chung cho hình 2
    (toàn bộ đồ thị) và hình 3 (ca nguy cơ cao)."""
    max_layer = max(kind_layer.values())
    depth = {u: kind_layer.get(d.get("kind"), max_layer) for u, d in G.nodes(data=True)}
    layers = defaultdict(list)
    for u, d in depth.items():
        layers[d].append(u)
    kind_rank = {k: i for i, k in enumerate(kind_rank_order)}
    pos = {}
    max_d = max(layers) or 1
    for d, members in layers.items():
        members.sort(key=lambda u: (kind_rank.get(G.nodes[u].get("kind"), 99),
                                    G.nodes[u].get("name", "")))
        span = max(len(members) - 1, 1)
        for i, u in enumerate(members):
            y = 0.5 if len(members) == 1 else 1.0 - i / span
            pos[u] = (d / max_d, y)
    return pos


# ==========================================================================
#  6. HÌNH 2 — TOÀN BỘ ĐỒ THỊ THẬT
# ==========================================================================
# Bo cuc phan tang rieng cho mien bat dong san (dung chung ham layered_positions
# voi mien y te qua KIND_LAYER/KIND_RANK khai bao o hinh 3). Moi loai thuc the
# co cot rieng (khong gop chung nhu BFS) de khong don qua nhieu dinh vao mot
# tang — tang chat nhat (Segment+Utility) toi da 5 dinh, ngang muc do cua
# mien y te, dam bao nhan khong the chong nhau du panel nho.
PROP_KIND_LAYER = {
    "Property": 0,
    "Attribute": 1,
    "Legal": 2, "Location": 2, "Model": 2,
    "Prediction": 3, "Infrastructure": 3,
    "Segment": 4, "Utility": 4,
    "LoanPackage": 5,
    "Repayment": 6, "Service": 6,
    "Requirement": 7,
}
PROP_KIND_RANK = ["Attribute", "Model", "Legal", "Location", "Prediction",
                  "Infrastructure", "Segment", "Utility", "LoanPackage",
                  "Repayment", "Service", "Requirement"]


def figure_full(data: dict) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    G, _ = build_digraph(data["nodes"], data["edges"])

    # Sáu tình huống mẫu là sáu thành phần liên thông tách biệt. Mỗi panel
    # dùng bố cục PHÂN TẦNG (như hình 3) thay vì spring_layout: với 15-22 đỉnh
    # nhãn tiếng Việt trong một khung nhỏ, lực đẩy ngẫu nhiên của spring_layout
    # không hề tính đến kích thước nhãn nên không bảo đảm tách rời — xếp mỗi
    # loại thực thể cố định một cột thì luôn tách nhau theo chiều dọc, bất kể
    # số đỉnh hay panel nhỏ tới đâu.
    # Chuong XI cua bao cao chi noi ve do thi tri thuc cua ung dung tieu duong;
    # ba tinh huong bat dong san la du lieu ton tu Assignment 01 va khong he
    # thong nao trong Assignment 02 truy van toi. Bo chung khoi hinh nay giup
    # moi panel rong gap doi -> het chong nhan. Mien bat dong san van xuat hien
    # day du o fig_kg_schema.png nen khong mat thong tin.
    med_cases = [c for c in CASE_ORDER if c.startswith("med_")]
    fig, axes = plt.subplots(1, len(med_cases), figsize=(26, 9.5), dpi=150)
    for ax, case in zip(axes.ravel(), med_cases):
        sub = G.subgraph([u for u, d in G.nodes(data=True) if d.get("case") == case])
        is_medical = case.startswith("med_")
        kind_layer = KIND_LAYER if is_medical else PROP_KIND_LAYER
        kind_rank = KIND_RANK if is_medical else PROP_KIND_RANK
        pos = layered_positions(sub, kind_layer, kind_rank)
        colours = [NODE_COLOURS.get(sub.nodes[u].get("kind"), "#64748B") for u in sub]
        nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#AEB6C2", width=1.0,
                               arrowsize=9, alpha=0.9, node_size=430,
                               connectionstyle="arc3,rad=0.08")
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colours, node_size=430,
                               edgecolors="white", linewidths=1.2)
        # Nhan bi cat NGAN hon hinh 3 rat nhieu (20 ky tu thay vi 30): o day bay
        # cot phan tang bi ep vao 1/3 be ngang hinh, nen nhan dai hon do se tran
        # sang cot ben canh va de chong nhau. Hinh nay chi can cho thay ba tang
        # nguy co dan toi ba bo tu van khac nhau; chi tiet tung nut doc o hinh 3.
        nx.draw_networkx_labels(
            sub, pos, {u: textwrap.shorten(str(sub.nodes[u].get("name", u)), 20,
                                           placeholder="…") for u in sub},
            ax=ax, font_size=7.2, font_color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))
        ax.set_title(f"{CASE_VI[case]}  ·  case = {case}\n"
                     f"{sub.number_of_nodes()} thực thể, {sub.number_of_edges()} quan hệ",
                     fontsize=11.5, fontweight="bold", color="#0F172A", pad=8)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        ax.set_axis_off()

    kinds = sorted({n.get("kind") for n in data["nodes"]
                    if n.get("kind") in MEDICAL_KINDS},
                   key=lambda k: KIND_VI.get(k, k))
    handles = [Line2D([], [], marker="o", linestyle="", markersize=9,
                      markerfacecolor=NODE_COLOURS.get(k, "#64748B"),
                      markeredgecolor="white",
                      label=f"{KIND_VI.get(k, k)} ({data['kind_counts'].get(k, 0)})")
               for k in kinds]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.006),
               title="Chú giải loại thực thể (kèm số lượng thật trên Neo4j)",
               title_fontsize=11)
    fig.suptitle("Ba nhánh tri thức y tế trên Neo4j Aura — mỗi tầng nguy cơ "
                 "dẫn tới một bộ tư vấn riêng" + chr(10) +
                 f"Đồ thị đầy đủ có {data['node_total']} thực thể và "
                 f"{data['rel_total']} quan hệ trên {len(data['kind_counts'])} loại "
                 f"thực thể; hình này vẽ phần miền y tế mà Ứng dụng 1 truy vấn tới",
                 fontsize=15, fontweight="bold", color="#0F172A", y=0.985)
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    out = FIG_DIR / "fig_kg_full.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"   OK {out.name}")


# ==========================================================================
#  7. HÌNH 3 — ĐỒ THỊ CỦA MỘT CA NGUY CƠ CAO
# ==========================================================================
# Bo cuc phan tang cua hinh 3: moi loai thuc the thuoc dung mot tang, xep theo
# mach suy luan that cua he thong. Dat tang theo loai (thay vi theo khoang cach
# BFS) giu cho moi canh deu chay tu trai sang phai va ten tang mo ta dung noi
# dung cot — BFS tung don ICD-10 chung cot voi nha thuoc, sai ve ngu nghia.
KIND_LAYER = {
    "Patient": 0,
    "Biomarker": 1, "Model": 1,
    "Prediction": 2, "RiskTier": 2,
    "Disease": 3, "Action": 3, "CarePackage": 3,
    "Device": 4, "Nutrition": 4,
    "Retailer": 5,
    "Service": 6,
}
STAGE_NAMES = [
    "Hồ sơ bệnh nhân",
    "Chỉ số và mô hình",
    "Dự đoán, tầng nguy cơ",
    "Chuẩn hoá, gói chăm sóc",
    "Sản phẩm cụ thể",
    "Nhà cung cấp",
    "Dịch vụ đi kèm",
]
# Thu tu tu tren xuong trong cung mot tang.
KIND_RANK = ["Prediction", "Disease", "Biomarker", "Model", "CarePackage",
             "Device", "Nutrition", "Action", "RiskTier", "Retailer", "Service"]


# Nut :Patient cua case med_high tren Neo4j chi luu 3/5 dac trung dau vao cua
# mo hinh (Glucose, BMI, Tuoi). Hai dac trung con lai duoc lay tu dong du lieu
# GOC trong diabetes/data/diabetes.csv trung khop DUY NHAT voi bo ba (Glucose
# =159, BMI=30.4, Age=36) va ca BloodPressure=66 (dung bang nut :Biomarker
# "BloodPressure = 66 mmHg" cua chinh case nay) — day la ban ghi that ma ca
# minh hoa med_high duoc dung tu do, khong phai so tu bia ra.
MED_HIGH_PREGNANCIES = 7.0
MED_HIGH_DPF = 0.383


def predict_case_high_probability(patient: dict) -> float:
    """Tinh lai xac suat duong tinh cua ca nguy co cao bang DUNG mo hinh dang
    trien khai (best_model = random_forest trong diabetes/model/metadata.json),
    dung preprocessor.joblib da fit san — chi .transform(), KHONG fit lai —
    y het quy trinh suy luan that trong diabetes/api/REST_API.py."""
    import __main__
    import json as _json
    import joblib
    import numpy as np
    import pandas as pd

    model_dir = ROOT / "diabetes" / "model"
    for fname in ("metadata.json", "preprocessor.joblib", "random_forest.joblib"):
        if not (model_dir / fname).exists():
            raise SystemExit(f"! Khong tim thay {model_dir / fname}")

    with (model_dir / "metadata.json").open("r", encoding="utf-8") as fh:
        metadata = _json.load(fh)
    feature_columns = metadata["feature_columns"]
    invalid_zero_columns = metadata.get("invalid_zero_columns", ["Glucose", "BMI"])

    # preprocessor.joblib duoc luu voi FunctionTransformer tro toi ham nay o
    # module __main__ (xem diabetes/api/REST_API.py) — phai dang ky lai dung
    # ten ham nay truoc khi joblib.load thi moi unpickle duoc, neu khong se
    # nem AttributeError: Can't get attribute 'convert_invalid_zero_to_nan'.
    def convert_invalid_zero_to_nan(frame):
        cleaned = frame.copy()
        cleaned[invalid_zero_columns] = cleaned[invalid_zero_columns].replace(0, np.nan)
        return cleaned

    setattr(__main__, "convert_invalid_zero_to_nan", convert_invalid_zero_to_nan)

    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = joblib.load(model_dir / "random_forest.joblib")

    values = {
        "Glucose": patient.get("Glucose", 0.0),
        "BMI": patient.get("BMI", 0.0),
        "Age": patient.get("Tuổi", 0.0),
        "Pregnancies": MED_HIGH_PREGNANCIES,
        "DiabetesPedigreeFunction": MED_HIGH_DPF,
    }
    raw_input = pd.DataFrame([values], columns=feature_columns)
    processed_input = preprocessor.transform(raw_input)  # chi transform, khong fit lai
    return float(model.predict_proba(processed_input)[0][1])


def patch_case_high_model(data: dict) -> None:
    """Nhãn trên Neo4j còn lưu mô hình KNN cũ cho case med_high; hệ thống đang
    triển khai dùng random_forest (metadata.json -> best_model). Sửa nhãn và
    tính lại xác suất bằng đúng mô hình thật NGAY SAU KHI đọc dữ liệu — trước
    khi bất kỳ hình nào dùng tới — để fig_kg_full.png và fig_kg_case_high.png
    luôn nhất quán với nhau bất kể thứ tự vẽ hình nào chạy trước.

    Sửa tại chỗ (in-place) trên chính các dict trong data["nodes"]: cả
    figure_full() lẫn figure_case_high() đều đọc lại từ data["nodes"] nên tự
    động thấy giá trị đã sửa, không cần truyền riêng."""
    nodes = [n for n in data["nodes"] if n.get("case") == "med_high"]
    if not nodes:
        return
    patient = next(n for n in nodes if n.get("kind") == "Patient")
    model_node = next(n for n in nodes if n.get("kind") == "Model")
    pred_node = next(n for n in nodes if n.get("kind") == "Prediction")
    proba_positive = predict_case_high_probability(patient)
    model_node["name"] = "Mô hình Random Forest"
    pred_node["name"] = f"Xác suất rủi ro {proba_positive * 100:.1f}%"
    pred_node["Xác suất"] = round(proba_positive, 4)


def figure_case_high(data: dict) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    case = "med_high"
    nodes = [n for n in data["nodes"] if n.get("case") == case]
    if not nodes:
        raise SystemExit("! Neo4j khong co du lieu cho case = med_high.")

    G, by_uid = build_digraph(nodes, data["edges"])

    root = next(u for u, d in G.nodes(data=True) if d.get("kind") == "Patient")
    pos = layered_positions(G, KIND_LAYER, KIND_RANK)
    max_d = max(KIND_LAYER.values())

    fig, ax = plt.subplots(figsize=(21, 12), dpi=150)
    colours = [NODE_COLOURS.get(G.nodes[u].get("kind"), "#64748B") for u in G]
    sizes = [1700 if G.nodes[u].get("kind") in ("Patient", "RiskTier", "Prediction")
             else 950 for u in G]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#98A2B3", width=1.5,
                           arrowsize=16, node_size=1500, alpha=0.92,
                           connectionstyle="arc3,rad=0.07")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colours, node_size=sizes,
                           edgecolors="white", linewidths=2.0)
    nx.draw_networkx_labels(
        G, pos, {u: wrap(G.nodes[u].get("name", u), 24, 4) for u in G}, ax=ax,
        font_size=7.4, font_color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E2E8F0", alpha=0.92))
    nx.draw_networkx_edge_labels(
        G, pos, {(a, b): REL_VI.get(d["rel"], d["rel"])
                 for a, b, d in G.edges(data=True)},
        ax=ax, font_size=6.6, font_color="#0D9488", rotate=False,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85))

    for d in sorted(set(KIND_LAYER.values())):
        if d < len(STAGE_NAMES):
            ax.text(d / max_d, 1.09, "Tầng {}".format(d) + chr(10) + STAGE_NAMES[d], ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold", color="#475569",
                    linespacing=1.4)

    patient = by_uid[root]
    pred = next((n for n in nodes if n.get("kind") == "Prediction"), {})
    tier = next((n for n in nodes if n.get("kind") == "RiskTier"), {})
    ax.set_title(
        "Đồ thị tri thức của một ca NGUY CƠ CAO  (case = med_high)\n"
        f"Bệnh nhân {patient.get('Tuổi', 0):.0f} tuổi · Glucose {patient.get('Glucose', 0):.0f} mg/dL"
        f" · BMI {patient.get('BMI', 0):.1f}  →  {pred.get('name', '')}"
        f"  →  {tier.get('name', '')}\n"
        f"{G.number_of_nodes()} thực thể, {G.number_of_edges()} quan hệ",
        fontsize=14.5, fontweight="bold", color="#0F172A", pad=24)

    kinds = sorted({d.get("kind") for _, d in G.nodes(data=True)},
                   key=lambda k: KIND_VI.get(k, k))
    handles = [Line2D([], [], marker="o", linestyle="", markersize=10,
                      markerfacecolor=NODE_COLOURS.get(k, "#64748B"),
                      markeredgecolor="white", label=KIND_VI.get(k, k))
               for k in kinds]
    ax.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
              fontsize=10, bbox_to_anchor=(0.5, -0.07))
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.14, 1.22)
    ax.set_axis_off()
    fig.tight_layout()
    out = FIG_DIR / "fig_kg_case_high.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"   OK {out.name}")


# ==========================================================================
#  ĐIỂM VÀO
# ==========================================================================
def main() -> None:
    data = fetch_graph()
    patch_case_high_model(data)

    print("-> Hinh 1/4: so do ban the hoc")
    figure_schema(data)
    print("-> Hinh 4/4: so do luong xu ly")
    figure_pipeline()

    setup_matplotlib()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        print("-> Hinh 2/4: toan bo do thi that")
        figure_full(data)
        print("-> Hinh 3/4: do thi ca nguy co cao")
        figure_case_high(data)
        check_glyphs(records)

    print("\nHoan tat. Bon hinh nam trong:", FIG_DIR)
    for name in ["fig_kg_schema.png", "fig_kg_full.png",
                 "fig_kg_case_high.png", "fig_kg_pipeline.png"]:
        path = FIG_DIR / name
        if path.exists():
            print(f"  {name:24s} {path.stat().st_size // 1024:>5d} KB")
        else:
            print(f"  {name:24s}  THIEU")


if __name__ == "__main__":
    sys.exit(main())
