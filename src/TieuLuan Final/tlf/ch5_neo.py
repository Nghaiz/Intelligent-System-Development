"""Chương 5: lớp truy cập Neo4j dùng chung cho nạp dữ liệu, truy xuất, tác tử và Web App."""
import re

from neo4j import GraphDatabase, unit_of_work
from neo4j.graph import Node, Relationship

from . import ch5_schema as S

URI, AUTH = "bolt://127.0.0.1:7687", ("neo4j", "tieuluanfinal")
WRITE_WORDS = re.compile(r"\b(create|merge|delete|detach|set|remove|drop|load\s+csv|foreach)\b", re.I)
WRITE_PROCS = re.compile(r"\bcall\s+(apoc\.(create|merge|refactor|periodic|load|export|cypher)|db\.create|dbms)", re.I)
_driver = None


def driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(URI, auth=AUTH)
        _driver.verify_connectivity()
    return _driver


def _plain(v):
    if isinstance(v, Node):
        return {k: x for k, x in dict(v).items() if k != "embedding"} | {"_labels": sorted(v.labels)}
    if isinstance(v, Relationship):
        return dict(v) | {"_type": v.type}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def cypher(q, params=None):
    """Chạy một câu Cypher, trả danh sách dict. Node, cạnh được đổi về dict thuộc tính bỏ vector nhúng."""
    with driver().session() as s:
        return [{k: _plain(v) for k, v in rec.items()} for rec in s.run(q, params or {})]


def read_only(q, params=None, timeout=10.0, limit=50):
    """Cypher do LLM sinh: chỉ cho phép đọc, giới hạn thời gian và số dòng trả về."""
    if WRITE_WORDS.search(q) or WRITE_PROCS.search(q):
        raise ValueError("Công cụ Cypher chỉ cho phép truy vấn đọc.")

    @unit_of_work(timeout=timeout)
    def work(tx):
        return [{k: _plain(v) for k, v in r.items()} for r in list(tx.run(q, params or {}))[:limit]]
    with driver().session() as s:
        return s.execute_read(work)


def batched(query, rows, size=2000, **extra):
    """Nạp dữ liệu theo lô UNWIND để giao dịch nhỏ, vừa heap 768 MB."""
    for i in range(0, len(rows), size):
        cypher(query, {"rows": rows[i:i + size], **extra})
    return len(rows)


def _kind(node):
    labels = [x for x in node.labels if x != "Entity"]
    return labels[0] if labels else "Entity"


def subgraph(seed, depth=1, limit=200):
    """Lân cận của một nút (id hoặc tên) cho Web App: {"nodes":[{id,label,name}],"links":[{source,target,type}]}."""
    with driver().session() as s:
        rec = s.run(S.SUBGRAPH, seed=seed, depth=int(depth), limit=int(limit)).single()
        if rec is None:
            return {"nodes": [], "links": []}
        ids = {n.element_id: n.get("id") or n.element_id for n in rec["nodes"]}
        nodes = [{"id": ids[n.element_id], "label": _kind(n),
                  "name": n.get("name") or n.get("title") or n.get("id")} for n in rec["nodes"]]
        links = [{"source": ids[r.start_node.element_id], "target": ids[r.end_node.element_id], "type": r.type}
                 for r in rec["relationships"]
                 if r.start_node.element_id in ids and r.end_node.element_id in ids]
    return {"nodes": nodes, "links": links}


def stats():
    """Số node, số cạnh thật đọc từ Neo4j, theo nhãn và theo kiểu quan hệ."""
    labels = cypher(S.COUNT_LABELS)
    types = cypher(S.COUNT_TYPES)
    total_nodes = cypher("MATCH (n) RETURN count(n) AS n")[0]["n"]
    return {"nodes": total_nodes, "relationships": sum(t["n"] for t in types), "by_label": labels, "by_type": types}
