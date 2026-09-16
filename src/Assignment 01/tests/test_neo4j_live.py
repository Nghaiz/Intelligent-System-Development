"""Kiểm thử cần một máy chủ Neo4j thật.

Tự bỏ qua khi ``.env`` chưa cấu hình, nên bộ kiểm thử vẫn xanh trên máy không
có tài khoản Aura — đúng tinh thần hai đường của Phase 5.

**Không đụng tới dữ liệu đang có.** Mọi thứ test này ghi đều mang định danh ca
``pytest_tmp`` và bị xoá trong bước dọn dẹp, kể cả khi test thất bại. Không có
lệnh xoá toàn bộ nào ở đây — ``wipe()`` được kiểm gián tiếp qua việc chạy
``build_graph`` chứ không gọi trực tiếp trong kiểm thử.
"""

from __future__ import annotations

import pytest

from src.kg import neo4j_client as nc
from src.kg.ontology import Edge, KnowledgeGraph, Node

pytestmark = pytest.mark.neo4j

CA_TAM = "pytest_tmp"


@pytest.fixture(scope="module")
def client():
    if not nc.is_configured():
        pytest.skip("Chưa cấu hình Neo4j trong .env — bỏ qua phần cần máy chủ")
    try:
        # __enter__ tự gọi connect(); gọi thêm connect() ở ngoài sẽ tạo trình
        # điều khiển thứ hai rồi bỏ rơi cái thứ nhất.
        with nc.Neo4jClient.from_env() as connected:
            yield connected
    except nc.Neo4jUnavailable as loi:
        pytest.skip(f"Không kết nối được máy chủ: {loi}")


@pytest.fixture
def don_dep(client):
    """Xoá sạch dấu vết của ca tạm, chạy cả trước lẫn sau để không phụ thuộc lần trước."""
    def _xoa():
        client.run(f"MATCH (n:{nc.BASE_LABEL} {{case: $case}}) DETACH DELETE n",
                   case=CA_TAM)
    _xoa()
    yield
    _xoa()


def _do_thi_tam() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(Node("patient", "Bệnh nhân kiểm thử", "Patient", {"Tuổi": 45}))
    kg.add_node(Node("model", "Mô hình kiểm thử", "Model"))
    kg.add_node(Node("prediction", "Xác suất 50.0%", "Prediction",
                     {"Xác suất": 0.5, "Ghi chú": None}))
    kg.add_edge(Edge("patient", "EVALUATED_BY", "model"))
    kg.add_edge(Edge("model", "PRODUCES", "prediction"))
    return kg


# ==========================================================================
#  KẾT NỐI
# ==========================================================================

def test_ket_noi_va_doc_duoc_phien_ban_may_chu(client):
    assert client.server_version().strip()
    assert client.run("RETURN 1 AS ok")[0]["ok"] == 1


def test_dem_duoc_nut_va_canh(client):
    assert client.count_nodes() >= 0
    assert client.count_edges() >= 0


# ==========================================================================
#  NẠP VÀ ĐỌC NGƯỢC
# ==========================================================================

def test_nap_roi_doc_nguoc_ra_dung_du_lieu(client, don_dep):
    client.ensure_constraint()
    counts = client.load(_do_thi_tam(), case_id=CA_TAM, domain="test")

    assert counts == {"nodes": 3, "edges": 2}

    rows = client.run(
        f"MATCH (n:{nc.BASE_LABEL} {{case: $case}}) "
        f"RETURN n.uid AS uid, n.name AS name, labels(n) AS labels ORDER BY uid",
        case=CA_TAM)

    assert [r["uid"] for r in rows] == [
        f"{CA_TAM}::model", f"{CA_TAM}::patient", f"{CA_TAM}::prediction"]
    assert set(rows[1]["labels"]) == {nc.BASE_LABEL, "Patient"}, \
        "Nút phải mang cả nhãn chung lẫn nhãn loại của nó"


def test_thuoc_tinh_rong_khong_duoc_ghi_len_may_chu(client, don_dep):
    client.load(_do_thi_tam(), case_id=CA_TAM, domain="test")

    row = client.run(
        f"MATCH (n:{nc.BASE_LABEL} {{uid: $uid}}) RETURN keys(n) AS keys",
        uid=f"{CA_TAM}::prediction")[0]

    assert "Xác suất" in row["keys"]
    assert "Ghi chú" not in row["keys"], "Gán null trong Cypher nghĩa là xoá thuộc tính"


def test_nap_hai_lan_khong_nhan_doi_du_lieu(client, don_dep):
    """MERGE theo uid — chạy lại phải cho đúng một trạng thái (yêu cầu R14)."""
    client.load(_do_thi_tam(), case_id=CA_TAM, domain="test")
    sau_lan_mot = client.run(
        f"MATCH (n:{nc.BASE_LABEL} {{case: $case}}) RETURN count(n) AS n",
        case=CA_TAM)[0]["n"]

    client.load(_do_thi_tam(), case_id=CA_TAM, domain="test")
    sau_lan_hai = client.run(
        f"MATCH (n:{nc.BASE_LABEL} {{case: $case}}) RETURN count(n) AS n",
        case=CA_TAM)[0]["n"]

    assert sau_lan_mot == sau_lan_hai == 3


def test_duong_di_nghiep_vu_truy_van_duoc(client, don_dep):
    """Không chỉ có nút — chuỗi quan hệ phải đi được từ đầu tới cuối."""
    client.load(_do_thi_tam(), case_id=CA_TAM, domain="test")

    rows = client.run(
        "MATCH (p:Patient {case: $case})-[:EVALUATED_BY]->(:Model)"
        "-[:PRODUCES]->(v:Prediction) RETURN v.name AS ket_qua", case=CA_TAM)

    assert [r["ket_qua"] for r in rows] == ["Xác suất 50.0%"]


def test_rang_buoc_duy_nhat_ton_tai(client):
    client.ensure_constraint()
    names = [r["name"] for r in client.run("SHOW CONSTRAINTS YIELD name RETURN name")]

    assert "entity_uid" in names
