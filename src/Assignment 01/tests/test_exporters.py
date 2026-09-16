"""Kiểm thử các hàm XUẤT TỆP bằng cách chạy thật chúng, không chấm tệp đã commit.

``test_artifacts.py`` đọc `outputs/graph.json` và bạn bè — những tệp đang nằm
trong git. Nó canh được hợp đồng dữ liệu, nhưng **không** canh được đoạn mã sinh
ra dữ liệu ấy: đập vỡ ``export_json`` ngay bây giờ thì cả tệp kia vẫn xanh, vì
nó đang chấm bài làm của hôm qua. Nó chỉ đỏ khi ai đó sinh lại tệp hỏng **rồi
commit**, tức là nó phát hiện tệp cũ chứ không phát hiện hàm hỏng.

Tệp này bịt chỗ đó: dựng vài ca giả, chạy đúng những hàm xuất tệp ấy vào thư
mục tạm, rồi kiểm hợp đồng trên sản phẩm vừa sinh ra.
"""

from __future__ import annotations

import json

import pytest

from src import config as cfg
from src.kg import build_graph as bg
from src.kg.ontology import build_medical_kg, build_property_kg

BENH_NHAN = {"Age": 45, "Glucose": 160.0, "BMI": 31.0,
             "BloodPressure": 80.0, "Insulin": 120.0}
NHA = {"Area": 100.0, "Floors": 4.0, "Bedrooms": 4.0, "Bathrooms": 5.0,
       "Province": "Hồ Chí Minh", "District": "Nhà Bè",
       "Legal status": "Have certificate"}


@pytest.fixture
def thu_muc_tam(tmp_path, monkeypatch):
    """Chuyển mọi đường ghi sang thư mục tạm để không đụng tệp thật."""
    outputs = tmp_path / "outputs"
    figures = tmp_path / "figures"
    outputs.mkdir()
    figures.mkdir()
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(cfg, "FIGURES_DIR", figures)
    return outputs, figures


@pytest.fixture
def cac_ca() -> list[bg.Case]:
    return [
        bg.Case(id="med_high", domain="medical", label="Ca y sinh thử",
                model="XGBoost", inputs=BENH_NHAN,
                prediction={"Xác suất mắc bệnh": 0.83},
                kg=build_medical_kg(BENH_NHAN, 0.83)),
        bg.Case(id="prop_luxury", domain="property", label="Ca bất động sản thử",
                model="XGBoost", inputs=NHA,
                prediction={"Giá định giá (tỷ VNĐ)": 8.42},
                kg=build_property_kg(NHA, 8.42)),
    ]


def test_xuat_json_dung_luoc_do_da_chot(thu_muc_tam, cac_ca):
    outputs, _figures = thu_muc_tam

    path = bg.export_json(cac_ca)

    assert path.parent == outputs, "Phải ghi vào đường dẫn lấy từ config"
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["meta"]["schema_version"] == 1, \
        "Đổi lược đồ thì phải tăng số hiệu — Phase 6 dựa vào nó"
    assert set(document) == {"meta", "domains"}
    assert set(document["domains"]) == {"medical", "property"}
    assert [c["id"] for c in document["domains"]["medical"]["cases"]] == ["med_high"]

    case = document["domains"]["property"]["cases"][0]
    assert {"id", "domain", "label", "model", "inputs",
            "prediction", "stats", "nodes", "edges"} <= set(case)
    assert case["stats"]["Số thực thể (|V|)"] == len(case["nodes"])


def test_xuat_json_khong_co_khoa_cases_o_muc_tren_cung(thu_muc_tam, cac_ca):
    """Chỗ Phase 6 dễ đoán nhầm nhất — mỗi ca nằm trong đúng một miền."""
    document = json.loads(bg.export_json(cac_ca).read_text(encoding="utf-8"))

    assert "cases" not in document


def test_xuat_bang_bo_ba_dung_so_dong(thu_muc_tam, cac_ca):
    _path, frame = bg.export_triplets(cac_ca)

    assert list(frame.columns) == ["Miền", "Ca", "Chủ thể", "Quan hệ", "Đối tượng"]
    assert len(frame) == sum(len(case.kg.edges) for case in cac_ca)
    assert not frame.isna().any().any()


def test_xuat_cypher_du_cau_lenh_va_khong_co_dau_nhay_le(thu_muc_tam, cac_ca):
    script = bg.export_cypher(cac_ca).read_text(encoding="utf-8")

    assert script.count("MERGE (n:Entity") == sum(len(c.kg.nodes) for c in cac_ca)
    assert script.count("MERGE (a)-[r:") == sum(len(c.kg.edges) for c in cac_ca)
    for line in script.splitlines():
        if line.startswith("//") or not line.strip():
            continue
        assert (line.count("'") - line.count("\\'")) % 2 == 0


def test_dinh_danh_trong_cypher_duoc_gan_tien_to_theo_ca(thu_muc_tam, cac_ca):
    """Không có tiền tố thì hai ca gộp thành một khi nạp lên cùng cơ sở dữ liệu."""
    script = bg.export_cypher(cac_ca).read_text(encoding="utf-8")

    assert "'med_high::patient'" in script
    assert "'prop_luxury::property'" in script


def test_xuat_html_tu_chua_khong_goi_ra_internet(thu_muc_tam, cac_ca):
    """Nộp bài kèm tệp rời thì máy chấm có thể không có mạng."""
    html = bg.export_html(cac_ca, "medical").read_text(encoding="utf-8", errors="ignore")

    assert "cdn.jsdelivr.net" not in html
    assert "vis-network" in html, "Gỡ nhầm cả thư viện vẽ thì đồ thị thành trang trắng"


def test_xuat_png_ra_anh_that(thu_muc_tam, cac_ca):
    _outputs, figures = thu_muc_tam

    path = bg.export_png(cac_ca[0], "kg_thu", "Tiêu đề thử")

    assert path.parent == figures
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 20_000


def test_xuat_png_chay_duoc_cho_ca_hai_mien(thu_muc_tam, cac_ca):
    """Hai miền dùng hai bảng tầng khác nhau; thiếu một loại là ném KeyError."""
    for case in cac_ca:
        assert bg.export_png(case, f"kg_{case.domain}_thu", case.label).exists()
