"""Kết nối Neo4j và nạp Đồ thị Tri thức lên cơ sở dữ liệu đồ thị.

Hệ thống chạy theo **hai đường**, và đường thứ hai là mặc định:

1. **Có Neo4j** — nạp đồ thị lên AuraDB, truy vấn bằng Cypher, chụp ảnh
   Neo4j Browser để đưa vào báo cáo.
2. **Không có Neo4j** — xuất ``graph.json`` cùng một tệp Cypher rời, vẽ đồ thị
   bằng pyvis. Toàn bộ phần còn lại của hệ thống chạy y hệt.

Nhờ vậy **không phase nào bị chặn** vì thiếu tài khoản: thiếu biến môi trường
thì ``settings_from_env`` trả về ``None`` và bên gọi tự chuyển sang đường 2,
có in cảnh báo rõ ràng chứ không im lặng nuốt lỗi.

Vì sao tự đọc ``.env`` thay vì dùng ``python-dotenv``: chỉ cần khoảng mười lăm
dòng, đổi lại bớt được một phụ thuộc phải cài đúng phiên bản trên Streamlit
Cloud ở Phase 7. Biến môi trường thật luôn thắng giá trị trong tệp, nên khi
triển khai lên đám mây thì ``st.secrets`` vẫn ghi đè được.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import config as cfg
from .ontology import KnowledgeGraph, NODE_COLOURS

ENV_FILE = cfg.ROOT / ".env"

# Nhãn dùng chung cho mọi thực thể, để đặt được ràng buộc duy nhất tại một chỗ.
# Mỗi nút mang thêm nhãn thứ hai là loại của nó (Patient, Biomarker, ...).
BASE_LABEL = "Entity"

# Tên nhãn và tên quan hệ được nội suy thẳng vào câu Cypher (Cypher không cho
# tham số hoá nhãn), nên phải kiểm tra chúng thuộc đúng bộ từ vựng của mình.
# Dùng \Z chứ không dùng $: trong Python, $ còn khớp ngay TRƯỚC ký tự xuống dòng
# cuối chuỗi, nên "HAS_X\n" lọt qua rào chắn.
_RELATION_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*\Z")


# ==========================================================================
#  ĐỌC CẤU HÌNH
# ==========================================================================

def load_env(path: Path | str = ENV_FILE) -> dict[str, str]:
    """Đọc tệp ``.env`` và trả về các cặp khoá–giá trị. **Không** ghi vào ``os.environ``.

    Dùng ``utf-8-sig`` vì tệp thông tin đăng nhập tải từ Neo4j Aura trên
    Windows có BOM ở đầu; đọc bằng ``utf-8`` thường sẽ khiến khoá đầu tiên dính
    ký tự vô hình và ``NEO4J_URI`` biến thành một khoá không ai tìm thấy.

    Hàm này cố ý **không** đụng tới ``os.environ``. Bản đầu có gọi
    ``setdefault`` cho tiện, và cái tiện ấy trả giá ngay: một bài kiểm thử nạp
    tệp ``.env`` giả là mọi bài chạy sau nó thừa hưởng giá trị giả, âm thầm, cho
    tới hết phiên. Thứ tự ưu tiên giữa tệp và môi trường được quyết định ở
    ``settings_from_env`` — nơi nhìn thấy cả hai nguồn cùng lúc.
    """
    path = Path(path)
    if not path.exists():
        return {}

    found: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        found[key.strip()] = value.strip().strip('"').strip("'")
    return found


@dataclass(frozen=True)
class Neo4jSettings:
    """Thông tin kết nối một thể hiện Neo4j."""

    uri: str
    user: str
    password: str
    database: str | None = None

    def describe(self) -> str:
        """Chuỗi mô tả an toàn để in ra màn hình — không bao giờ chứa mật khẩu."""
        # rpartition("@") cắt bỏ phần userinfo: người dùng có thể nhét thẳng
        # tài khoản vào URI dạng neo4j+s://user:mật-khẩu@host. Trình điều khiển
        # từ chối URI kiểu đó, nên họ rơi đúng vào nhánh báo lỗi — nhánh in
        # chuỗi này ra màn hình. Không cắt là mật khẩu hiện nguyên văn.
        host = self.uri.split("//")[-1].rpartition("@")[2]
        return f"{self.user}@{host} (cơ sở dữ liệu: {self.database or 'mặc định'})"


def settings_from_mapping(source: Mapping[str, Any]) -> Neo4jSettings | None:
    """Dựng cấu hình từ một ánh xạ bất kỳ (``os.environ`` hoặc ``st.secrets``).

    Chấp nhận **cả hai** tên khoá cho tài khoản:

    - ``NEO4J_USER``     — tên dùng trong ``.env.example`` của dự án
    - ``NEO4J_USERNAME`` — tên trong tệp thông tin đăng nhập Aura tải về

    Aura bản hiện hành dùng **mã instance** làm tên đăng nhập và làm luôn tên
    cơ sở dữ liệu, chứ không phải chuỗi ``neo4j`` như tài liệu cũ mô tả. Vì thế
    ``NEO4J_DATABASE`` được tôn trọng khi có mặt; chỉ khi vắng mới để trình
    điều khiển tự chọn cơ sở dữ liệu mặc định của máy chủ.
    """
    uri = str(source.get("NEO4J_URI") or "").strip()
    user = str(source.get("NEO4J_USER") or source.get("NEO4J_USERNAME") or "").strip()
    password = str(source.get("NEO4J_PASSWORD") or "").strip()
    database = str(source.get("NEO4J_DATABASE") or "").strip() or None

    if not (uri and user and password):
        return None
    return Neo4jSettings(uri=uri, user=user, password=password, database=database)


def settings_from_env(env_file: Path | str = ENV_FILE) -> Neo4jSettings | None:
    """Dựng cấu hình từ ``.env`` và biến môi trường. ``None`` khi chưa khai báo đủ.

    Biến môi trường thật **ghi đè** giá trị trong tệp. Đó là điều kiện để
    Phase 7 chạy được trên Streamlit Cloud: ở đó không có tệp ``.env`` (đúng
    như phải thế, vì ``.gitignore`` chặn nó), thông tin đăng nhập nằm trong
    ``st.secrets`` và được đưa vào môi trường tiến trình.
    """
    # Chỉ những biến môi trường CÓ giá trị mới ghi đè. Biến export ra rỗng là
    # "không khai báo", không phải "khai báo bằng chuỗi rỗng" — nếu không, một
    # dòng NEO4J_PASSWORD= sót trong shell sẽ vô hiệu hoá cả tệp .env hợp lệ rồi
    # hệ thống báo "chưa cấu hình", một lời giải thích sai.
    thuc_su_co = {key: value for key, value in os.environ.items() if value}
    return settings_from_mapping({**load_env(env_file), **thuc_su_co})


def is_configured(env_file: Path | str = ENV_FILE) -> bool:
    """Có đủ ba biến môi trường để thử kết nối hay không."""
    return settings_from_env(env_file) is not None


# ==========================================================================
#  CHUẨN BỊ DỮ LIỆU CHO CÂU CYPHER
# ==========================================================================

def _check_label(label: str) -> str:
    """Chỉ cho phép loại thực thể đã khai báo trong bản thể học."""
    if label not in NODE_COLOURS:
        raise ValueError(
            f"Loại thực thể '{label}' không có trong bản thể học. "
            f"Khai báo nó trong ontology.NODE_COLOURS trước khi nạp lên Neo4j."
        )
    return label


def _check_relation(relation: str) -> str:
    """Tên quan hệ phải viết HOA_CÓ_GẠCH_DƯỚI theo quy ước RDF của dự án."""
    if not _RELATION_PATTERN.match(relation):
        raise ValueError(
            f"Tên quan hệ '{relation}' sai quy ước. Quan hệ phải viết hoa toàn bộ, "
            f"phân tách bằng dấu gạch dưới, ví dụ HAS_BIOMARKER."
        )
    return relation


def _drop_empty(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Bỏ thuộc tính rỗng — Neo4j coi việc gán ``null`` là xoá thuộc tính."""
    return {k: v for k, v in properties.items() if v is not None and v != ""}


def graph_rows(kg: KnowledgeGraph, case_id: str,
               domain: str) -> tuple[list[dict], list[dict]]:
    """Kết xuất đồ thị thành hai danh sách bản ghi sẵn sàng cho ``UNWIND``.

    Định danh nút được gắn tiền tố theo ca (``uid = "<ca>::<id>"``). Nếu không
    làm vậy, sáu ca mẫu cùng nạp lên một cơ sở dữ liệu sẽ chồng lên nhau, vì
    bản thể học đặt tên nút cố định (``patient``, ``model``, ``prediction``)
    cho mọi ca.
    """
    def uid(node_id: str) -> str:
        return f"{case_id}::{node_id}"

    nodes = [
        {
            "uid": uid(node.id),
            "name": node.label,
            "kind": _check_label(node.type),
            "case": case_id,
            "domain": domain,
            "props": _drop_empty(node.properties),
        }
        for node in kg.nodes
    ]
    edges = [
        {
            "source": uid(edge.source),
            "relation": _check_relation(edge.relation),
            "target": uid(edge.target),
            "props": _drop_empty(edge.properties),
        }
        for edge in kg.edges
    ]
    return nodes, edges


# ==========================================================================
#  SINH CYPHER RỜI (ĐƯỜNG LÙI KHI KHÔNG CÓ TÀI KHOẢN)
# ==========================================================================

def _cypher_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return "'" + text + "'"


def _cypher_map(properties: Mapping[str, Any]) -> str:
    """Cypher không nhận khoá đặt trong nháy; khoá có dấu cách phải bọc backtick."""
    if not properties:
        return "{}"
    items = []
    for key, value in properties.items():
        safe_key = str(key).replace("`", "``")
        items.append("`" + safe_key + "`: " + _cypher_value(value))
    return "{" + ", ".join(items) + "}"


def to_cypher(node_rows: Iterable[dict], edge_rows: Iterable[dict]) -> str:
    """Sinh kịch bản Cypher dán thẳng được vào Neo4j Browser.

    Đây là đường lùi cho người đọc báo cáo không có tài khoản Aura: mở Neo4j
    Desktop hoặc một thể hiện bất kỳ, dán tệp này vào là dựng lại nguyên đồ thị.
    """
    lines = [
        "// Đồ thị Tri thức — sinh tự động bởi src/kg/build_graph.py",
        "// Dán toàn bộ nội dung này vào Neo4j Browser rồi bấm Run.",
        "",
        "CREATE CONSTRAINT entity_uid IF NOT EXISTS",
        f"FOR (n:{BASE_LABEL}) REQUIRE n.uid IS UNIQUE;",
        "",
    ]
    for row in node_rows:
        props = {
            "name": row["name"],
            "kind": row["kind"],
            "case": row["case"],
            "domain": row["domain"],
            **row["props"],
        }
        lines.append(
            f"MERGE (n:{BASE_LABEL} {{uid: {_cypher_value(row['uid'])}}}) "
            f"SET n:{row['kind']}, n += {_cypher_map(props)};"
        )
    lines.append("")
    for row in edge_rows:
        tail = f" SET r += {_cypher_map(row['props'])};" if row["props"] else ";"
        lines.append(
            f"MATCH (a:{BASE_LABEL} {{uid: {_cypher_value(row['source'])}}}), "
            f"(b:{BASE_LABEL} {{uid: {_cypher_value(row['target'])}}}) "
            f"MERGE (a)-[r:{row['relation']}]->(b)" + tail
        )
    return "\n".join(lines) + "\n"


# ==========================================================================
#  LỚP BỌC TRÌNH ĐIỀU KHIỂN
# ==========================================================================

class Neo4jUnavailable(RuntimeError):
    """Không kết nối được Neo4j. Bên gọi bắt lỗi này để lùi sang đường offline."""


class Neo4jClient:
    """Lớp bọc mỏng quanh trình điều khiển chính thức, dùng như context manager.

        with Neo4jClient.from_env() as client:
            client.wipe()
            client.load(kg, case_id="med_high", domain="medical")
    """

    def __init__(self, settings: Neo4jSettings) -> None:
        self.settings = settings
        self._driver = None

    # --------------------------------------------------------------- khởi tạo
    @classmethod
    def from_env(cls, env_file: Path | str = ENV_FILE) -> "Neo4jClient":
        settings = settings_from_env(env_file)
        if settings is None:
            raise Neo4jUnavailable(
                "Chưa khai báo NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD trong .env. "
                "Xem setup-guides/01-neo4j-aura.md, hoặc bỏ qua để dùng đồ thị offline."
            )
        return cls(settings)

    def connect(self) -> "Neo4jClient":
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:                      # pragma: no cover
            raise Neo4jUnavailable(
                "Thiếu gói neo4j. Cài bằng: pip install 'neo4j>=5.20'"
            ) from exc

        try:
            self._driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.user, self.settings.password),
            )
            self._driver.verify_connectivity()
        except Exception as exc:
            self.close()
            raise Neo4jUnavailable(
                f"Không kết nối được tới {self.settings.describe()}: "
                f"{type(exc).__name__}. Kiểm tra instance đã ở trạng thái Running "
                f"và ba biến trong .env còn đúng."
            ) from exc
        return self

    def __enter__(self) -> "Neo4jClient":
        return self.connect()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # ---------------------------------------------------------------- truy vấn
    def run(self, query: str, **parameters) -> list[dict]:
        if self._driver is None:
            raise Neo4jUnavailable("Chưa gọi connect() trước khi truy vấn.")
        keyword = {"database_": self.settings.database} if self.settings.database else {}
        try:
            records, _summary, _keys = self._driver.execute_query(
                query, parameters_=parameters, **keyword
            )
        except Exception as exc:
            # Lỗi của trình điều khiển (ServiceUnavailable, SessionExpired,
            # ClientError...) không phải lớp con của Neo4jUnavailable, nên nếu
            # để nguyên thì một lần Aura ngủ giữa chừng sẽ kéo sập cả pipeline
            # sau khi đã huấn luyện xong — trong khi bên gọi đã có sẵn đường lùi
            # offline. Đổi thành lỗi mà bên gọi biết bắt, giữ nguyên tên lớp gốc
            # trong thông báo để không giấu mất nguyên nhân.
            raise Neo4jUnavailable(
                f"Truy vấn Neo4j thất bại ({type(exc).__name__}) trên "
                f"{self.settings.describe()}"
            ) from exc
        return [record.data() for record in records]

    def server_version(self) -> str:
        rows = self.run(
            "CALL dbms.components() YIELD name, versions, edition "
            "RETURN name AS name, versions[0] AS version, edition AS edition"
        )
        if not rows:
            return "không rõ"
        row = rows[0]
        return f"{row['name']} {row['version']} ({row['edition']})"

    # ------------------------------------------------------------- ghi dữ liệu
    def ensure_constraint(self) -> None:
        """Ràng buộc duy nhất trên ``uid`` — vừa bảo toàn dữ liệu vừa tạo chỉ mục."""
        self.run(
            f"CREATE CONSTRAINT entity_uid IF NOT EXISTS "
            f"FOR (n:{BASE_LABEL}) REQUIRE n.uid IS UNIQUE"
        )

    def wipe(self) -> int:
        """Xoá toàn bộ nút và cạnh, trả về số nút đã xoá.

        Nạp lại từ trạng thái rỗng khiến mỗi lần chạy cho ra đúng một kết quả,
        đáp ứng yêu cầu tái lập R14. Muốn giữ dữ liệu cũ thì đừng gọi hàm này.
        """
        before = self.count_nodes()
        self.run("MATCH (n) DETACH DELETE n")
        return before

    def count_nodes(self) -> int:
        return int(self.run("MATCH (n) RETURN count(n) AS c")[0]["c"])

    def count_edges(self) -> int:
        return int(self.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"])

    def load(self, kg: KnowledgeGraph, case_id: str, domain: str) -> dict[str, int]:
        """Nạp một đồ thị lên cơ sở dữ liệu, gộp theo ``uid`` nên chạy lại được.

        Nút được nhóm theo loại rồi nạp thành từng lô: Cypher không cho phép
        tham số hoá tên nhãn, nên nhãn buộc phải nội suy vào câu lệnh — mỗi lô
        một nhãn là cách làm điều đó mà không cần tới thư viện APOC.
        """
        node_rows, edge_rows = graph_rows(kg, case_id, domain)

        by_kind: dict[str, list[dict]] = {}
        for row in node_rows:
            by_kind.setdefault(row["kind"], []).append(row)

        for kind, rows in by_kind.items():
            self.run(
                f"UNWIND $rows AS row "
                f"MERGE (n:{BASE_LABEL} {{uid: row.uid}}) "
                f"SET n:{kind}, n.name = row.name, n.kind = row.kind, "
                f"    n.case = row.case, n.domain = row.domain, n += row.props",
                rows=rows,
            )

        by_relation: dict[str, list[dict]] = {}
        for row in edge_rows:
            by_relation.setdefault(row["relation"], []).append(row)

        for relation, rows in by_relation.items():
            self.run(
                f"UNWIND $rows AS row "
                f"MATCH (a:{BASE_LABEL} {{uid: row.source}}) "
                f"MATCH (b:{BASE_LABEL} {{uid: row.target}}) "
                f"MERGE (a)-[r:{relation}]->(b) "
                f"SET r += row.props",
                rows=rows,
            )

        return {"nodes": len(node_rows), "edges": len(edge_rows)}

    # ----------------------------------------------------------------- đọc lại
    def summary_by_kind(self) -> list[dict]:
        """Đếm thực thể theo loại — dùng để đối chiếu sau khi nạp."""
        return self.run(
            f"MATCH (n:{BASE_LABEL}) RETURN n.kind AS kind, count(*) AS n "
            f"ORDER BY n DESC, kind"
        )
