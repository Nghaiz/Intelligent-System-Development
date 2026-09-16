"""Tầng truy vấn đồ thị tri thức dùng chung cho cả ba REST API.

Nguyên tắc thiết kế quan trọng nhất: **đồ thị chỉ BỔ SUNG, không bao giờ QUYẾT
ĐỊNH**. Mô hình nơ-ron tính ra con số trước; module này chỉ dịch con số đó thành
một tầng tư vấn rồi lấy nội dung tương ứng về.

Hệ quả: mọi lỗi ở đây đều bị nuốt lại thành một chuỗi thông báo. Nếu Neo4j sập,
mất mạng, hay chưa cấu hình mật khẩu, API vẫn phải trả về dự đoán bình thường —
chỉ thiếu phần tư vấn kèm theo.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

_driver = None
if GraphDatabase is not None and PASSWORD:
    try:
        _driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    except Exception:
        _driver = None

QUERY = """
MATCH (t:A03Tier {case: $case})
OPTIONAL MATCH (t)-[:ADVISES]->(a:A03Advice)
RETURN t.name AS tier_name, t.range AS tier_range,
       collect({group: a.group, title: a.title, detail: a.detail}) AS advice
"""


def is_available() -> bool:
    return _driver is not None


def resolve_case(tiers, value):
    """Chọn tầng đầu tiên có ngưỡng dưới ≤ value.

    `tiers` là danh sách (ngưỡng_dưới, case) đã sắp giảm dần theo ngưỡng.
    """
    for threshold, case in tiers:
        if value >= threshold:
            return case
    return tiers[-1][1]


def fetch(case: str):
    """Lấy nội dung tư vấn của một tầng.

    Trả về (groups, tier_name, error). `groups` là danh sách
    {"title": tên_nhóm, "items": [{"title": ..., "content": ...}]}.
    """
    if _driver is None:
        return [], None, ("Chưa cấu hình kết nối Neo4j "
                          "(thiếu NEO4J_PASSWORD hoặc chưa cài package neo4j).")
    try:
        with _driver.session(database=DATABASE) as session:
            record = session.run(QUERY, case=case).single()
        if record is None:
            return [], None, f"Đồ thị tri thức chưa có dữ liệu cho tầng '{case}'."

        grouped = defaultdict(list)
        for entry in record["advice"]:
            if not entry or not entry.get("title"):
                continue
            grouped[entry.get("group") or "Khác"].append({
                "title": entry["title"],
                "content": entry.get("detail"),
            })
        groups = [{"title": g, "items": items} for g, items in grouped.items()]
        return groups, record["tier_name"], None
    except Exception as exc:                       # pragma: no cover
        return [], None, f"Không thể truy vấn đồ thị tri thức lúc này ({type(exc).__name__})."
