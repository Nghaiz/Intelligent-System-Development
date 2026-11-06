"""Đọc Neo4j cho trang Đồ thị tri thức. Mọi câu Cypher chạy trong giao dịch đọc, giá trị người dùng luôn là tham số;
câu đã chạy được trả về nguyên văn để giao diện hiển thị."""
import os
import socket
import threading
import time

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from neo4j.graph import Node, Path, Relationship

from .lazy import Unavailable, heavy_turn, need, present

GAZETTEER_GB = 1
URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", os.environ.get("DNX_NEO4J_PASSWORD", "dianexus2026"))  # mật khẩu dev trong docker-compose, chỉ 127.0.0.1
ADDR = ("127.0.0.1", 7687)
MAX_NODES = 400
SKIP_LABELS = ("Chunk", "Article")
SAMPLE = """MATCH (n) WHERE ($label IS NULL AND NOT any(l IN labels(n) WHERE l IN $skip)) OR $label IN labels(n)
WITH n ORDER BY coalesce(n.degree, 0) DESC LIMIT $limit
WITH collect(n) AS ns
UNWIND ns AS a
OPTIONAL MATCH (a)-[r]->(b) WHERE b IN ns
RETURN ns, collect(DISTINCT r) AS rels"""
FIND = """MATCH (n) WHERE toLower(coalesce(n.name, n.title, '')) CONTAINS toLower($q)
RETURN n ORDER BY size(coalesce(n.name, n.title)) LIMIT 8"""
EXPAND = """MATCH (s) WHERE coalesce(s.id, elementId(s)) IN $ids
MATCH p = (s)-[*1..{hops}]-(m)
RETURN p LIMIT $limit"""
LABELS = "CALL db.labels() YIELD label RETURN label"
TYPES = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
COUNT_TTL, SAMPLE_TTL = 8, 60
_driver, _lock = None, threading.Lock()
_counts = {"ts": 0.0, "value": {"ok": False}}
_sample_cache = {}


def _port_open():
    try:
        with socket.create_connection(ADDR, 0.3):
            return True
    except OSError:
        return False


def driver():
    global _driver
    if not _port_open():
        raise Unavailable("Neo4j chưa chạy. Khởi động: docker compose up -d")
    with _lock:
        if _driver is None:
            _driver = GraphDatabase.driver(URI, auth=AUTH, connection_timeout=2)
    return _driver


def _read(cypher, **params):
    def work(tx):
        return list(tx.run(cypher, **params))
    try:
        with driver().session() as s:
            return s.execute_read(work)
    except (ServiceUnavailable, OSError) as e:
        raise Unavailable(f"Neo4j không phản hồi: {e}") from e


def _count_of(cypher):
    return _read(cypher)[0]["c"]


def counts():
    """Đếm từ count store nên gần như tức thì; dùng cho thanh sinh hiệu."""
    if time.time() - _counts["ts"] < COUNT_TTL:
        return _counts["value"]
    try:
        value = {"ok": True, "nodes": _count_of("MATCH (n) RETURN count(n) AS c"),
                 "edges": _count_of("MATCH ()-[r]->() RETURN count(r) AS c")}
    except (Unavailable, Neo4jError) as e:
        value = {"ok": False, "error": str(e)[:160]}
    _counts.update(ts=time.time(), value=value)
    return value


def _node_id(node):
    return node.get("id") or node.element_id


def _kind(node):
    labels = sorted(set(node.labels) - {"Entity"})
    return labels[0] if labels else "Entity"


def _node(node):
    return {"id": _node_id(node), "label": _kind(node),
            "name": str(node.get("name") or node.get("title") or node.get("id") or "(không tên)")[:120]}


def _collect(v, nodes, rels):
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


def _as_graph(nodes, rels):
    kept = dict(list(nodes.items())[:MAX_NODES])
    ids = {k: _node_id(n) for k, n in kept.items()}
    links = [{"source": ids[r.start_node.element_id], "target": ids[r.end_node.element_id], "type": r.type}
             for r in rels.values() if r.start_node.element_id in ids and r.end_node.element_id in ids]
    return {"nodes": [_node(n) for n in kept.values()], "links": links}


def _catalog():
    labels = [r["label"] for r in _read(LABELS)]
    types = [r["relationshipType"] for r in _read(TYPES)]
    # Tên nhãn lấy từ chính db.labels() nên ghép vào câu đếm là an toàn; Cypher không nhận tham số cho nhãn.
    lab = [{"label": x, "count": _count_of(f"MATCH (n:`{x}`) RETURN count(n) AS c")} for x in labels]
    rel = [{"type": x, "count": _count_of(f"MATCH ()-[r:`{x}`]->() RETURN count(r) AS c")} for x in types]
    return sorted(lab, key=lambda d: -d["count"]), sorted(rel, key=lambda d: -d["count"])


def overview(label=None, limit=160):
    limit = max(10, min(int(limit), 300))
    key = (label, limit)
    hit = _sample_cache.get(key)
    if hit and time.time() - hit[0] < SAMPLE_TTL:
        return hit[1]
    labels, rels = _catalog()
    if label and label not in {x["label"] for x in labels}:
        raise ValueError(f"Nhãn không có trong đồ thị: {label[:40]}")
    nodes, found = {}, {}
    for r in _read(SAMPLE, label=label, skip=list(SKIP_LABELS), limit=limit):
        _collect([r["ns"], r["rels"]], nodes, found)
    value = {"labels": labels, "relations": rels, **_as_graph(nodes, found), "cypher": [SAMPLE]}
    _sample_cache[key] = (time.time(), value)
    return value


def _seeds(q):
    """Thực thể khớp câu hỏi: ưu tiên bộ nhận thực thể của kho tri thức, chưa có thì khớp tên trực tiếp."""
    if present("dnx.kb.search"):
        try:
            with heavy_turn("gazetteer", GAZETTEER_GB, "từ điển thực thể"):
                found = need("dnx.kb.search").entities(q)
        except FileNotFoundError:  # chỉ mục KB chưa dựng xong: khớp tên, Cypher hiện trên trang nói rõ đã dùng cách này
            found = []
        if found:
            return [x["id"] for x in found[:8]], "// thực thể từ dnx.kb.search.entities"
    return [_node_id(r["n"]) for r in _read(FIND, q=q)], FIND


def query(q=None, node=None, hops=1, limit=120):
    """Mở rộng lân cận từ câu tìm `q` hoặc từ một nút đã bấm."""
    hops, limit = max(1, min(int(hops), 3)), max(10, min(int(limit), 300))
    ids, how = ([node], "// nút đã bấm") if node else _seeds(q or "")
    if not ids:
        return {"nodes": [], "links": [], "seeds": [], "cypher": [how]}
    cypher = EXPAND.replace("{hops}", str(hops))  # hops đã kẹp 1..3; Cypher không nhận tham số cho độ dài đường
    nodes, rels = {}, {}
    for r in _read(cypher, ids=ids, limit=limit):
        _collect(r["p"], nodes, rels)
    return {**_as_graph(nodes, rels), "seeds": ids, "cypher": [how, cypher]}
