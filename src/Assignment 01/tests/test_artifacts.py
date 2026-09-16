"""Kiểm thử hợp đồng dữ liệu của các tệp đã xuất.

``outputs/graph.json`` là ranh giới giữa Phase 5 và Phase 6: ứng dụng web đọc
tệp này, báo cáo LaTeX ở Phase 8 trích số từ nó. Một thay đổi phá vỡ tương
thích ở đây sẽ làm hỏng cả hai phase sau mà không ai biết cho tới lúc chạy.

Nên các test dưới đây kiểm **hình dạng dữ liệu**, không kiểm nội dung cụ thể —
đổi ca mẫu vẫn phải xanh, còn đổi lược đồ thì phải đỏ.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config as cfg
from src.kg.build_graph import SCHEMA_VERSION
from src.kg.ontology import NODE_COLOURS

GRAPH_JSON = cfg.OUTPUTS_DIR / "graph.json"
TRIPLETS_CSV = cfg.OUTPUTS_DIR / "kg_triplets.csv"
CYPHER_SCRIPT = cfg.OUTPUTS_DIR / "graph.cypher"

THIEU_TEP = "Chưa có tệp — chạy `python -m src.kg.build_graph --offline` trước"


@pytest.fixture(scope="module")
def document() -> dict:
    if not GRAPH_JSON.exists():
        pytest.skip(THIEU_TEP)
    return json.loads(GRAPH_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases(document) -> list[dict]:
    return [case for domain in document["domains"].values()
            for case in domain["cases"]]


# ==========================================================================
#  LƯỢC ĐỒ
# ==========================================================================

def test_co_so_hieu_phien_ban_de_ben_doc_kiem_tra_duoc(document):
    """Không có số hiệu thì Phase 6 không phân biệt được lược đồ cũ với mới."""
    assert document["meta"]["schema_version"] == SCHEMA_VERSION


def test_du_hai_mien_va_moi_mien_co_nhan_doc_duoc(document):
    assert set(document["domains"]) == {"medical", "property"}
    for domain in document["domains"].values():
        assert domain["label"].strip()
        assert isinstance(domain["cases"], list) and domain["cases"]


def test_moi_ca_du_truong_bat_buoc(cases):
    bat_buoc = {"id", "domain", "label", "model", "inputs",
                "prediction", "stats", "nodes", "edges"}
    for case in cases:
        assert bat_buoc <= set(case), f"Ca {case.get('id')} thiếu {bat_buoc - set(case)}"


def test_dinh_danh_ca_khong_trung_nhau(cases):
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))


def test_bang_mau_di_kem_de_ve_lai_duoc_o_phase_6(document):
    assert document["meta"]["bang_mau"] == NODE_COLOURS


# ==========================================================================
#  TÍNH TOÀN VẸN CỦA ĐỒ THỊ
# ==========================================================================

def test_moi_nut_du_bon_truong_va_loai_co_trong_ban_the_hoc(cases):
    for case in cases:
        for node in case["nodes"]:
            assert {"id", "label", "type", "properties"} <= set(node)
            assert node["type"] in NODE_COLOURS
            assert node["label"].strip()


def test_moi_canh_noi_hai_nut_co_that_trong_cung_ca(cases):
    """Cạnh treo lơ lửng sẽ khiến câu MATCH lúc nạp lên Neo4j im lặng bỏ qua."""
    for case in cases:
        ids = {node["id"] for node in case["nodes"]}
        for edge in case["edges"]:
            assert edge["source"] in ids, f"{case['id']}: nguồn lạ {edge['source']}"
            assert edge["target"] in ids, f"{case['id']}: đích lạ {edge['target']}"


def test_dinh_danh_nut_khong_trung_trong_mot_ca(cases):
    for case in cases:
        ids = [node["id"] for node in case["nodes"]]
        assert len(ids) == len(set(ids)), f"Ca {case['id']} có định danh trùng"


def test_thong_ke_khop_voi_so_nut_va_so_canh_dem_duoc(cases):
    """Số liệu này đi thẳng vào báo cáo, không được lệch với dữ liệu sinh ra nó."""
    for case in cases:
        assert case["stats"]["Số thực thể (|V|)"] == len(case["nodes"])
        assert case["stats"]["Số cạnh (|E|)"] == len(case["edges"])
        assert case["stats"]["Số loại quan hệ (|R|)"] == \
            len({edge["relation"] for edge in case["edges"]})


def test_khong_nhan_nao_lot_chu_none_hay_nan(cases):
    """Dấu vết của việc quên đổi kiểu ở ranh giới pandas — xem build_graph._native."""
    for case in cases:
        for node in case["nodes"]:
            assert "None" not in node["label"]
            assert "nan" not in node["label"].lower().split()


def test_moi_ca_deu_co_du_chuoi_suy_luan_tu_du_lieu_toi_khuyen_nghi(cases):
    """Đây là lý do tồn tại của cả phase: con số phải dẫn tới việc làm được."""
    for case in cases:
        types = {node["type"] for node in case["nodes"]}
        assert {"Model", "Prediction"} <= types, f"Ca {case['id']} thiếu bước suy luận"
        khuyen_nghi = {"Action", "Device", "Nutrition", "LoanPackage", "Utility"}
        assert types & khuyen_nghi, f"Ca {case['id']} không dẫn tới khuyến nghị nào"


# ==========================================================================
#  BẢNG BỘ BA VÀ KỊCH BẢN CYPHER
# ==========================================================================

def test_bang_bo_ba_du_cot_va_khop_so_canh(cases):
    if not TRIPLETS_CSV.exists():
        pytest.skip(THIEU_TEP)
    frame = pd.read_csv(TRIPLETS_CSV, encoding="utf-8-sig")

    assert list(frame.columns) == ["Miền", "Ca", "Chủ thể", "Quan hệ", "Đối tượng"]
    assert len(frame) == sum(len(case["edges"]) for case in cases)
    assert not frame.isna().any().any(), "Ô trống trong bảng bộ ba"


def test_kich_ban_cypher_co_du_cau_lenh_cho_moi_nut_va_moi_canh(cases):
    if not CYPHER_SCRIPT.exists():
        pytest.skip(THIEU_TEP)
    script = CYPHER_SCRIPT.read_text(encoding="utf-8")

    assert "CREATE CONSTRAINT entity_uid IF NOT EXISTS" in script
    assert script.count("MERGE (n:Entity") == sum(len(c["nodes"]) for c in cases)
    assert script.count("MERGE (a)-[r:") == sum(len(c["edges"]) for c in cases)


def test_kich_ban_cypher_khong_chua_dau_nhay_hoac_backtick_lac(cases):
    """Một dấu nháy sổng ra là hỏng cả kịch bản, mà lỗi chỉ hiện lúc dán vào chạy."""
    if not CYPHER_SCRIPT.exists():
        pytest.skip(THIEU_TEP)

    for line in CYPHER_SCRIPT.read_text(encoding="utf-8").splitlines():
        if line.startswith("//") or not line.strip():
            continue
        assert (line.count("'") - line.count("\\'")) % 2 == 0, f"Dấu nháy lẻ: {line[:70]}"
        assert line.count("`") % 2 == 0, f"Backtick lẻ: {line[:70]}"


# ==========================================================================
#  TỆP HÌNH
# ==========================================================================

@pytest.mark.parametrize("ten", ["kg_medical.html", "kg_property.html"])
def test_tep_html_tu_chua_khong_goi_ra_internet(ten):
    """Nộp bài kèm tệp rời thì máy chấm có thể không có mạng."""
    path = cfg.OUTPUTS_DIR / ten
    if not path.exists():
        pytest.skip(THIEU_TEP)
    html = path.read_text(encoding="utf-8", errors="ignore")

    assert "cdn.jsdelivr.net" not in html
    assert "vis-network" in html, "Thư viện vẽ phải được nhúng tại chỗ"


@pytest.mark.parametrize("ten", ["kg_medical.png", "kg_property.png"])
def test_anh_tinh_ton_tai_va_khong_rong(ten):
    path = cfg.FIGURES_DIR / ten
    if not path.exists():
        pytest.skip(THIEU_TEP)

    assert path.stat().st_size > 20_000, "Ảnh quá nhỏ, nhiều khả năng vẽ hỏng"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
