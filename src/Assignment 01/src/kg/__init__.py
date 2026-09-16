"""Đồ thị Tri thức — bản thể học, dựng đồ thị và kết nối Neo4j.

Ba tệp chia theo trách nhiệm:

- ``ontology``     — định nghĩa thực thể, quan hệ và hai bản thể học miền
- ``build_graph``  — dựng đồ thị từ dữ liệu thật, xuất tệp và nạp lên Neo4j
- ``neo4j_client`` — lớp bọc trình điều khiển Neo4j, tự lùi về đồ thị offline
"""

from .ontology import (
    Edge,
    KnowledgeGraph,
    NODE_COLOURS,
    Node,
    PROPERTY_SEGMENTS,
    RISK_TIERS,
    build_medical_kg,
    build_property_kg,
    classify_risk,
    classify_segment,
)

__all__ = [
    "Edge",
    "KnowledgeGraph",
    "NODE_COLOURS",
    "Node",
    "PROPERTY_SEGMENTS",
    "RISK_TIERS",
    "build_medical_kg",
    "build_property_kg",
    "classify_risk",
    "classify_segment",
]
