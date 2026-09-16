"""Kiểm thử lớp kết nối Neo4j — đọc cấu hình, sinh Cypher, an toàn thông tin.

Không có test nào ở đây chạm tới mạng. Phần cần máy chủ thật nằm riêng trong
``test_neo4j_live.py`` và tự bỏ qua khi chưa cấu hình.

Hai nhóm quan trọng nhất:

- **An toàn thông tin đăng nhập** — mật khẩu không được lọt ra bất kỳ chuỗi nào
  có thể in ra màn hình hay ghi vào tệp nhật ký.
- **Sinh Cypher** — nhãn và tên quan hệ buộc phải nội suy thẳng vào câu lệnh vì
  Cypher không cho tham số hoá chúng. Đó là bề mặt tiêm mã, phải có rào chắn.
"""

from __future__ import annotations

import pytest

from src.kg import neo4j_client as nc
from src.kg.ontology import Edge, KnowledgeGraph, Node


# ==========================================================================
#  ĐỌC TỆP .env
# ==========================================================================

def test_doc_duoc_tep_co_bom_chu_thich_va_dau_nhay(tmp_path):
    """Tệp Aura tải về trên Windows có BOM; đọc sai mã hoá là mất khoá đầu tiên."""
    env = tmp_path / ".env"
    env.write_text(
        "# một dòng chú thích\n"
        "\n"
        "NEO4J_URI=neo4j+s://a1b2c3d4.databases.neo4j.io\n"
        'NEO4J_USERNAME="a1b2c3d4"\n'
        "NEO4J_PASSWORD='mat khau co dau cach'\n"
        "dòng rác không có dấu bằng\n",
        encoding="utf-8-sig",
    )

    found = nc.load_env(env)

    assert found["NEO4J_URI"] == "neo4j+s://a1b2c3d4.databases.neo4j.io"
    assert found["NEO4J_USERNAME"] == "a1b2c3d4", "Dấu nháy phải bị bóc"
    assert found["NEO4J_PASSWORD"] == "mat khau co dau cach"
    assert "dòng rác không có dấu bằng" not in found


def test_bien_moi_truong_that_thang_gia_tri_trong_tep(tmp_path, monkeypatch):
    """Điều kiện để Streamlit Cloud ghi đè được bằng st.secrets ở Phase 7."""
    env = tmp_path / ".env"
    env.write_text("NEO4J_URI=tu-trong-tep\nNEO4J_USER=u\nNEO4J_PASSWORD=p\n",
                   encoding="utf-8")
    monkeypatch.setenv("NEO4J_URI", "tu-moi-truong")

    assert nc.settings_from_env(env).uri == "tu-moi-truong"


def test_doc_tep_khong_lam_ban_moi_truong(tmp_path, monkeypatch):
    """Tác dụng phụ toàn cục làm bộ kiểm thử phụ thuộc thứ tự chạy.

    Bản đầu gọi ``os.environ.setdefault``: một bài kiểm thử nạp tệp .env giả là
    mọi bài sau nó thừa hưởng giá trị giả và tự bỏ qua, im lặng.
    """
    env = tmp_path / ".env"
    env.write_text("KHOA_KIEM_THU_KHONG_CO_THAT=gia-tri-gia\n", encoding="utf-8")
    monkeypatch.delenv("KHOA_KIEM_THU_KHONG_CO_THAT", raising=False)

    nc.load_env(env)

    import os
    assert "KHOA_KIEM_THU_KHONG_CO_THAT" not in os.environ


def test_khong_co_tep_thi_tra_ve_rong_chu_khong_nem_loi():
    assert nc.load_env("tep-khong-bao-gio-ton-tai.env") == {}


# ==========================================================================
#  DỰNG CẤU HÌNH
# ==========================================================================

def test_chap_nhan_ca_hai_ten_khoa_tai_khoan():
    """Aura ghi NEO4J_USERNAME, còn .env.example cũ của dự án ghi NEO4J_USER."""
    kieu_aura = nc.settings_from_mapping({
        "NEO4J_URI": "neo4j+s://x", "NEO4J_USERNAME": "u", "NEO4J_PASSWORD": "p"})
    kieu_cu = nc.settings_from_mapping({
        "NEO4J_URI": "neo4j+s://x", "NEO4J_USER": "u", "NEO4J_PASSWORD": "p"})

    assert kieu_aura.user == "u"
    assert kieu_cu.user == "u"


def test_neo4j_user_uu_tien_hon_neo4j_username():
    settings = nc.settings_from_mapping({
        "NEO4J_URI": "neo4j+s://x", "NEO4J_USER": "chon-toi",
        "NEO4J_USERNAME": "bo-qua", "NEO4J_PASSWORD": "p"})

    assert settings.user == "chon-toi"


@pytest.mark.parametrize("thieu", ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"])
def test_thieu_bat_ky_gia_tri_nao_thi_coi_nhu_chua_cau_hinh(thieu):
    """Cấu hình nửa vời phải rơi về đường offline, không được thử kết nối rồi treo."""
    day_du = {"NEO4J_URI": "neo4j+s://x", "NEO4J_USER": "u", "NEO4J_PASSWORD": "p"}
    assert nc.settings_from_mapping({**day_du, thieu: ""}) is None
    assert nc.settings_from_mapping({k: v for k, v in day_du.items() if k != thieu}) is None


def test_ten_co_so_du_lieu_la_tuy_chon():
    khong_khai_bao = nc.settings_from_mapping({
        "NEO4J_URI": "neo4j+s://x", "NEO4J_USER": "u", "NEO4J_PASSWORD": "p"})
    co_khai_bao = nc.settings_from_mapping({
        "NEO4J_URI": "neo4j+s://x", "NEO4J_USER": "u", "NEO4J_PASSWORD": "p",
        "NEO4J_DATABASE": "a1b2c3d4"})

    assert khong_khai_bao.database is None, "Để trống thì máy chủ tự chọn mặc định"
    assert co_khai_bao.database == "a1b2c3d4"


# ==========================================================================
#  AN TOÀN THÔNG TIN ĐĂNG NHẬP
# ==========================================================================

MAT_KHAU = "mAtKhAuSieuBiMat123"


def test_mo_ta_ket_noi_khong_bao_gio_chua_mat_khau():
    """describe() được in thẳng ra màn hình trong push_to_neo4j."""
    settings = nc.Neo4jSettings(
        uri="neo4j+s://a1b2c3d4.databases.neo4j.io", user="a1b2c3d4",
        password=MAT_KHAU, database="a1b2c3d4")

    assert MAT_KHAU not in settings.describe()
    assert "a1b2c3d4" in settings.describe()


def test_thong_bao_chua_cau_hinh_khong_chua_mat_khau(tmp_path, monkeypatch):
    # settings_from_env đọc os.environ (biến môi trường thắng giá trị trong tệp),
    # nên trên máy đã export sẵn NEO4J_* thì nhánh "chưa cấu hình" không bao giờ
    # chạy tới. Dọn sạch để kiểm đúng thứ định kiểm.
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(nc.Neo4jUnavailable) as loi:
        nc.Neo4jClient.from_env(tmp_path / "khong-co.env")

    assert MAT_KHAU not in str(loi.value)
    assert "setup-guides" in str(loi.value), "Phải chỉ người dùng tới hướng dẫn"


def test_truy_van_khi_chua_ket_noi_thi_nem_loi_ro_rang():
    """Im lặng trả về rỗng sẽ khiến bên gọi tưởng cơ sở dữ liệu trống."""
    client = nc.Neo4jClient(nc.Neo4jSettings("neo4j+s://x", "u", MAT_KHAU))

    with pytest.raises(nc.Neo4jUnavailable):
        client.run("RETURN 1")


# ==========================================================================
#  RÀO CHẮN TÊN NHÃN VÀ TÊN QUAN HỆ
# ==========================================================================

def test_loai_thuc_the_ngoai_ban_the_hoc_bi_chan():
    with pytest.raises(ValueError, match="bản thể học"):
        nc._check_label("Entity`) DETACH DELETE (n")


@pytest.mark.parametrize("ten_xau", [
    "khong_viet_hoa",
    "CO KHOANG TRANG",
    "1BAT_DAU_BANG_SO",
    "CO-GACH-NGANG",
    "XOA]->() DETACH DELETE n //",
    "",
])
def test_ten_quan_he_sai_quy_uoc_bi_chan(ten_xau):
    with pytest.raises(ValueError, match="quy ước"):
        nc._check_relation(ten_xau)


def test_ten_quan_he_dung_quy_uoc_duoc_chap_nhan():
    assert nc._check_relation("HAS_BIOMARKER") == "HAS_BIOMARKER"
    assert nc._check_relation("PART_OF") == "PART_OF"


# ==========================================================================
#  KẾT XUẤT BẢN GHI CHO CÂU UNWIND
# ==========================================================================

def _do_thi_mau() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(Node("patient", "Bệnh nhân", "Patient", {"Tuổi": 45, "Ghi chú": None}))
    kg.add_node(Node("model", "Mô hình XGBoost", "Model", {"Loại": ""}))
    kg.add_edge(Edge("patient", "EVALUATED_BY", "model", {"Trọng số": 1.0}))
    return kg


def test_dinh_danh_duoc_gan_tien_to_theo_ca():
    """Không có tiền tố thì sáu ca nạp chung sẽ chồng lên nhau thành một."""
    nodes, edges = nc.graph_rows(_do_thi_mau(), case_id="med_high", domain="medical")

    assert [n["uid"] for n in nodes] == ["med_high::patient", "med_high::model"]
    assert edges[0]["source"] == "med_high::patient"
    assert edges[0]["target"] == "med_high::model"
    assert all(n["case"] == "med_high" and n["domain"] == "medical" for n in nodes)


def test_thuoc_tinh_rong_bi_loai_bo():
    """Gán null trong Cypher nghĩa là XOÁ thuộc tính — gửi lên là vô nghĩa."""
    nodes, _edges = nc.graph_rows(_do_thi_mau(), "ca", "medical")

    assert nodes[0]["props"] == {"Tuổi": 45}, "None phải bị loại"
    assert nodes[1]["props"] == {}, "Chuỗi rỗng phải bị loại"


def test_hai_ca_khac_nhau_khong_dung_chung_dinh_danh():
    a, _ = nc.graph_rows(_do_thi_mau(), "ca_mot", "medical")
    b, _ = nc.graph_rows(_do_thi_mau(), "ca_hai", "medical")

    assert not ({n["uid"] for n in a} & {n["uid"] for n in b})


# ==========================================================================
#  SINH CYPHER RỜI
# ==========================================================================

def test_kich_ban_cypher_co_rang_buoc_va_du_cau_lenh():
    nodes, edges = nc.graph_rows(_do_thi_mau(), "ca", "medical")
    script = nc.to_cypher(nodes, edges)

    assert "CREATE CONSTRAINT entity_uid IF NOT EXISTS" in script
    assert script.count("MERGE (n:Entity") == 2
    assert "MERGE (a)-[r:EVALUATED_BY]->(b)" in script
    assert script.count(";") >= 3


def test_khoa_thuoc_tinh_tieng_viet_duoc_boc_backtick():
    """Cypher không nhận khoá đặt trong nháy; khoá có dấu cách phải bọc backtick."""
    assert nc._cypher_map({"Tỷ lệ cho vay (LTV)": "70%"}) == "{`Tỷ lệ cho vay (LTV)`: '70%'}"


def test_backtick_trong_khoa_bi_nhan_doi():
    """Backtick lọt vào khoá sẽ đóng sớm dấu bọc và làm hỏng cả câu lệnh."""
    assert nc._cypher_map({"a`b": 1}) == "{`a``b`: 1}"


@pytest.mark.parametrize("gia_tri, mong_doi", [
    (None, "null"),
    (True, "true"),
    (False, "false"),
    (42, "42"),
    ("chuỗi thường", "'chuỗi thường'"),
    ("có 'dấu nháy'", r"'có \'dấu nháy\''"),
    ("có \\ gạch chéo", r"'có \\ gạch chéo'"),
])
def test_gia_tri_duoc_thoat_dung(gia_tri, mong_doi):
    assert nc._cypher_value(gia_tri) == mong_doi


def test_gia_tri_doc_hai_khong_thoat_ra_khoi_dau_nhay():
    """Chuỗi từ dữ liệu không được đóng sớm dấu nháy rồi chèn câu lệnh mới."""
    doc_hai = "x' DETACH DELETE n //"
    ket_qua = nc._cypher_value(doc_hai)

    assert ket_qua.startswith("'") and ket_qua.endswith("'")
    assert ket_qua.count("'") - ket_qua.count("\\'") == 2, "Chỉ hai dấu nháy không bị thoát"


def test_mo_ta_ket_noi_cat_bo_tai_khoan_nhet_trong_uri():
    """Người dùng có thể nhét tài khoản thẳng vào URI theo thói quen của các CSDL khác.

    Trình điều khiển Neo4j từ chối URI kiểu đó, nên họ rơi đúng vào nhánh báo
    lỗi — nhánh in chuỗi mô tả ra màn hình. Không cắt userinfo là mật khẩu hiện
    nguyên văn trên stdout và trong nhật ký CI.
    """
    settings = nc.Neo4jSettings(
        uri=f"neo4j+s://neo4j:{MAT_KHAU}@a1b2c3d4.databases.neo4j.io",
        user="neo4j", password=MAT_KHAU)

    assert MAT_KHAU not in settings.describe()
    assert "a1b2c3d4.databases.neo4j.io" in settings.describe()


def test_bien_moi_truong_rong_khong_ghi_de_tep(tmp_path, monkeypatch):
    """Biến export ra rỗng nghĩa là "không khai báo", không phải "khai báo rỗng".

    Một dòng ``NEO4J_PASSWORD=`` sót trong shell sẽ vô hiệu hoá cả tệp .env hợp
    lệ, rồi hệ thống báo "chưa cấu hình Neo4j" — một lời giải thích sai khiến
    người dùng đi sửa nhầm chỗ.
    """
    env = tmp_path / ".env"
    env.write_text("NEO4J_URI=neo4j+s://x\nNEO4J_USER=u\nNEO4J_PASSWORD=that\n",
                   encoding="utf-8")
    monkeypatch.setenv("NEO4J_PASSWORD", "")

    settings = nc.settings_from_env(env)

    assert settings is not None, "Tệp .env hợp lệ không được biến rỗng vô hiệu hoá"
    assert settings.password == "that"


class _DriverHong:
    """Trình điều khiển giả luôn ném lỗi của tầng neo4j, không phải Neo4jUnavailable."""

    def execute_query(self, *args, **kwargs):
        raise ConnectionResetError("Aura ngủ giữa chừng")


def test_loi_giua_chung_doi_thanh_loi_ma_ben_goi_biet_bat():
    """Bên gọi chỉ bắt Neo4jUnavailable; lỗi lạ lọt ra sẽ kéo sập cả pipeline.

    Ba tình huống đã thử ở Phase 5 (không có .env, biến rỗng, sai mật khẩu) đều
    hỏng ngay lúc kết nối. Lớp hỏng GIỮA CHỪNG — Aura ngủ sau khi đã bắt tay —
    chưa từng được thử, mà đó mới là lúc 90 giây huấn luyện đã nằm trong bộ nhớ.
    """
    client = nc.Neo4jClient(nc.Neo4jSettings("neo4j+s://x", "u", MAT_KHAU))
    client._driver = _DriverHong()

    with pytest.raises(nc.Neo4jUnavailable) as loi:
        client.run("MATCH (n) RETURN n")

    assert "ConnectionResetError" in str(loi.value), "Phải giữ tên lỗi gốc"
    assert MAT_KHAU not in str(loi.value)
