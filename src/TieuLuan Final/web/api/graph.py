"""Đọc đồ thị Neo4j cho Graph Universe: mẫu theo nhãn, mở rộng lân cận, tìm tên, Cypher chỉ đọc.

Chặn ghi có ba lớp: từ khóa đầu câu, danh sách từ cấm dùng chung với công cụ Cypher của Chương 5, và giao dịch
đọc `execute_read` để chính Neo4j từ chối nếu có câu lọt qua hai lớp đầu.
"""
import importlib.util
import re

from neo4j import unit_of_work
from neo4j.graph import Node, Path, Relationship

from .system import neo4j_port_open

FIRST_WORD = re.compile(r"^\s*(match|optional\s+match|with|unwind|return|call|show)\b", re.I)
EXTRA_PROCS = re.compile(r"\bapoc\.(do|trigger|custom|schema|nodes\.delete|systemdb|static|log)\b", re.I)
HIDDEN = {"embedding"}
MAX_ROWS, MAX_NODES, MAX_TEXT = 200, 600, 240
SKIP_KIND = {"Entity", "Chunk"}
SAMPLE = """MATCH (n:Entity) {where}
WITH n, COUNT {{ (n)--() }} AS degree ORDER BY degree DESC LIMIT $limit
WITH collect(n) AS ns
UNWIND ns AS a
OPTIONAL MATCH (a)-[r]->(b) WHERE b IN ns
RETURN collect(DISTINCT a) AS nodes, collect(DISTINCT r) AS rels"""
NODE_DETAIL = """MATCH (n) WHERE n.id = $id
WITH n LIMIT 1
OPTIONAL MATCH (n)-[r]-()
WITH n, type(r) AS t, count(r) AS c ORDER BY c DESC
RETURN n, collect({type: t, n: c}) AS degree"""
SEARCH = """MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($q)
RETURN n ORDER BY size(n.name) LIMIT $limit"""


class Unavailable(RuntimeError):
    """Neo4j hoặc tlf/ch5_neo.py chưa sẵn sàng; giao diện hiện trạng thái thay vì lỗi."""


def _neo():
    if importlib.util.find_spec("tlf.ch5_neo") is None:
        raise Unavailable("Chưa có tlf/ch5_neo.py.")
    if not neo4j_port_open():
        raise Unavailable("Neo4j không chạy. Khởi động: docker compose up -d")
    from tlf import ch5_neo
    return ch5_neo


def ensure_read_only(q):
    """Ném ValueError nếu câu Cypher không phải truy vấn đọc."""
    neo = _neo()
    if not FIRST_WORD.match(q):
        raise ValueError("Câu truy vấn phải bắt đầu bằng MATCH, OPTIONAL MATCH, WITH, UNWIND, RETURN, CALL hoặc SHOW.")
    if neo.WRITE_WORDS.search(q) or neo.WRITE_PROCS.search(q) or EXTRA_PROCS.search(q):
        raise ValueError("Chỉ cho phép truy vấn đọc: CREATE, DELETE, MERGE, SET và các lệnh ghi bị chặn.")


def _kind(node):
    labels = sorted(set(node.labels) - SKIP_KIND)
    if "Chunk" in node.labels:
        return "Chunk"
    return labels[0] if labels else "Entity"


def _node_id(node):
    return node.get("id") or node.element_id


def _node(node):
    return {"id": _node_id(node), "label": _kind(node),
            "name": node.get("name") or node.get("title") or node.get("ticker") or node.get("id") or "(không tên)"}


def _props(entity):
    out = {}
    for k, v in dict(entity).items():
        if k not in HIDDEN:
            out[k] = v[:MAX_TEXT] if isinstance(v, str) else (v if isinstance(v, (int, float, bool)) else str(v))
    return out


def _collect(v, nodes, rels):
    """Gom Node và Relationship ở mọi độ sâu của một ô kết quả."""
    if isinstance(v, Node):
        nodes[v.element_id] = v
    elif isinstance(v, Relationship):
        rels[v.element_id] = v
    elif isinstance(v, Path):
        _collect(list(v.nodes), nodes, rels)
        _collect(list(v.relationships), nodes, rels)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _collect(x, nodes, rels)
    elif isinstance(v, dict):
        _collect(list(v.values()), nodes, rels)


def _cell(v):
    if isinstance(v, Node):
        return _node(v) | {"properties": _props(v)}
    if isinstance(v, Relationship):
        return {"type": v.type, **_props(v)}
    if isinstance(v, Path):
        return " → ".join(str(n.get("name") or n.get("id")) for n in v.nodes)
    if isinstance(v, (list, tuple)):
        return [_cell(x) for x in v[:30]]
    if isinstance(v, dict):
        return {k: _cell(x) for k, x in v.items()}
    if isinstance(v, str):
        return v[:MAX_TEXT]
    return v if v is None or isinstance(v, (int, float, bool)) else str(v)


def _as_graph(nodes, rels):
    kept = dict(list(nodes.items())[:MAX_NODES])
    ids = {k: _node_id(n) for k, n in kept.items()}
    links = [{"source": ids[r.start_node.element_id], "target": ids[r.end_node.element_id], "type": r.type}
             for r in rels.values()
             if r.start_node.element_id in ids and r.end_node.element_id in ids]
    return {"nodes": [_node(n) for n in kept.values()], "links": links}


def overview():
    return _neo().stats()


def sample(label=None, limit=160):
    """Các nút bậc cao nhất, kèm cạnh giữa chúng; `label` phải nằm trong danh sách nhãn của stats()."""
    neo, limit = _neo(), max(10, min(int(limit), 400))
    if label and label not in {x["label"] for x in neo.stats()["by_label"]}:
        raise ValueError(f"Nhãn không có trong đồ thị: {label}")
    where = f"WHERE n:`{label}`" if label else ""
    with neo.driver().session() as s:
        rec = s.run(SAMPLE.format(where=where), limit=limit).single()
        nodes, rels = {}, {}
        _collect([rec["nodes"], rec["rels"]], nodes, rels)
        return _as_graph(nodes, rels)


def expand(node_id, limit=60):
    return _neo().subgraph(node_id, depth=1, limit=max(5, min(int(limit), 200)))


def node_detail(node_id):
    with _neo().driver().session() as s:
        rec = s.run(NODE_DETAIL, id=node_id).single()
        if rec is None:
            return None
        return _node(rec["n"]) | {"properties": _props(rec["n"]),
                                  "degree": [d for d in rec["degree"] if d["type"]]}


def search(q, limit=12):
    with _neo().driver().session() as s:
        return [_node(r["n"]) for r in s.run(SEARCH, q=q, limit=limit)]


def run_cypher(q, timeout=10.0):
    """Chạy Cypher chỉ đọc: trả bảng tối đa MAX_ROWS dòng và đồ thị gom từ mọi nút, cạnh, đường đi trong kết quả."""
    ensure_read_only(q)

    @unit_of_work(timeout=timeout)
    def work(tx):
        result = tx.run(q)
        keys, rows, nodes, rels = list(result.keys()), [], {}, {}
        for rec in result:
            if len(rows) >= MAX_ROWS:
                break
            rows.append([_cell(rec[k]) for k in keys])
            _collect(list(rec.values()), nodes, rels)
        return {"columns": keys, "rows": rows, "graph": _as_graph(nodes, rels),
                "truncated": len(rows) >= MAX_ROWS}
    with _neo().driver().session() as s:
        return s.execute_read(work)
