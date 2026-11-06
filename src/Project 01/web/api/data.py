"""Đọc SQLite `data/dianexus.db` qua kết nối chỉ đọc. Tên bảng phải nằm trong danh sách trắng; giá trị luôn tham số hóa."""
import json
import sqlite3
from contextlib import contextmanager

from dnx.db import DATA_TABLES, DB_PATH as DB, OPS_TABLES

from .lazy import Unavailable

ALLOWED = DATA_TABLES + OPS_TABLES
MAX_LIMIT = 200
BINS = 24
TOP_VALUES = 12
NUMERIC = ("INTEGER", "INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE", "BIGINT")
_stats = {}


@contextmanager
def ro():
    """mode=ro nên mọi câu ghi đều bị SQLite từ chối, kể cả khi lọt qua kiểm tra phía trên."""
    if not DB.exists():
        raise Unavailable("Đang chuẩn bị cơ sở dữ liệu.")
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=2)
    con.row_factory = sqlite3.Row
    try:
        yield con
    except sqlite3.OperationalError as e:
        raise Unavailable(f"Cơ sở dữ liệu chưa sẵn sàng: {e}") from e
    finally:
        con.close()


def _table(name):
    if name not in ALLOWED:
        raise ValueError(f"Bảng không được phép đọc: {name[:40]}")
    return name


def _quote(col):
    return '"' + col.replace('"', '""') + '"'


def _columns(con, table):
    return [{"name": r["name"], "type": r["type"]} for r in con.execute(f"PRAGMA table_info({table})")]


def _exists(con, table):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _count(con, table):
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def summary():
    """Số dòng từng bảng cho thanh sinh hiệu; chưa có DB thì báo chưa sẵn sàng, không báo 0."""
    if not DB.exists():
        return {"ok": False, "tables": {}}
    try:
        with ro() as con:
            tables = {t: _count(con, t) for t in ALLOWED if _exists(con, t)}
        return {"ok": True, "tables": tables, "rows": sum(tables.get(t, 0) for t in DATA_TABLES),
                "size_mb": round(DB.stat().st_size / 1e6, 1)}
    except Unavailable as e:
        return {"ok": False, "tables": {}, "error": str(e)}


def tables():
    with ro() as con:
        return [{"name": t, "group": "data" if t in DATA_TABLES else "ops", "rows": _count(con, t),
                 "columns": _columns(con, t)} for t in ALLOWED if _exists(con, t)]


def _existing(con, table):
    if not _exists(con, table):
        raise Unavailable(f"Bảng {table} chưa có dữ liệu.")
    return table


def page(table, limit=50, offset=0):
    table, limit, offset = _table(table), max(1, min(int(limit), MAX_LIMIT)), max(0, int(offset))
    with ro() as con:
        _existing(con, table)
        rows = con.execute(f"SELECT * FROM {table} LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return {"table": table, "columns": _columns(con, table), "rows": [dict(r) for r in rows],
                "total": _count(con, table), "limit": limit, "offset": offset}


def _numeric_stats(con, table, q):
    n, lo, hi, mean = con.execute(f"SELECT COUNT({q}), MIN({q}), MAX({q}), AVG({q}) FROM {table}").fetchone()
    out = {"kind": "num", "count": n, "min": lo, "max": hi, "mean": mean}
    if n and hi is not None and hi > lo:
        w = (hi - lo) / BINS
        rows = con.execute(f"SELECT MIN(CAST(({q} - ?) / ? AS INT), ?) b, COUNT(*) FROM {table} "
                           f"WHERE {q} IS NOT NULL GROUP BY b ORDER BY b", (lo, w, BINS - 1)).fetchall()
        out["hist"] = [{"x0": lo + b * w, "x1": lo + (b + 1) * w, "n": c} for b, c in rows]
    return out


def _top_stats(con, table, q):
    rows = con.execute(f"SELECT {q} v, COUNT(*) c FROM {table} WHERE {q} IS NOT NULL GROUP BY {q} "
                       f"ORDER BY c DESC LIMIT ?", (TOP_VALUES,)).fetchall()
    return {"kind": "top", "top": [{"value": str(v), "n": c} for v, c in rows]}


def _column_stats(con, table, col, total):
    q = _quote(col["name"])
    nulls = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {q} IS NULL").fetchone()[0]
    distinct = con.execute(f"SELECT COUNT(DISTINCT {q}) FROM {table}").fetchone()[0]
    numeric = col["type"].upper() in NUMERIC
    body = _numeric_stats(con, table, q) if numeric and distinct > TOP_VALUES else _top_stats(con, table, q)
    return {"name": col["name"], "type": col["type"], "nulls": nulls, "distinct": distinct,
            "null_pct": nulls / total if total else None, **body}


def _default_column(cols):
    return next((c["name"] for c in cols[1:] if c["type"].upper() in NUMERIC), cols[0]["name"])


def stats(table, col=None):
    """Thống kê một cột, tính khi cần rồi nhớ tạm: quét mọi cột của bảng vài trăm nghìn dòng tốn hàng chục giây."""
    table = _table(table)
    with ro() as con:
        _existing(con, table)
        cols, total = _columns(con, table), _count(con, table)
        name = col or _default_column(cols)
        chosen = next((c for c in cols if c["name"] == name), None)  # tên cột phải có trong PRAGMA
        if chosen is None:
            raise ValueError(f"Bảng {table} không có cột {name[:40]}")
        key = (table, name, total)
        if key not in _stats:
            _stats[key] = _column_stats(con, table, chosen, total)
    return {"table": table, "rows": total, "columns": cols, "column": _stats[key]}


def sessions(limit=60):
    with ro() as con:
        _existing(con, "sessions")
        return [dict(r) for r in con.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM agent_events e WHERE e.session_id = s.id) AS n_events, "
            "(SELECT content FROM messages m WHERE m.session_id = s.id AND m.role = 'user' ORDER BY m.id LIMIT 1) "
            "AS first_question FROM sessions s WHERE EXISTS (SELECT 1 FROM agent_events e WHERE e.session_id = s.id) "
            "ORDER BY s.created_at DESC LIMIT ?", (limit,))]  # phiên sàng lọc không có vết tác tử nên không phát lại được


def _event(row):
    d = dict(row)
    d["data"] = json.loads(d["data"]) if d.get("data") else {}
    return d


def session_detail(session_id):
    with ro() as con:
        events = [_event(r) for r in con.execute("SELECT * FROM agent_events WHERE session_id=? ORDER BY id",
                                                 (session_id,))]
        messages = [dict(r) for r in con.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id",
                                                 (session_id,))]
    return {"session_id": session_id, "events": events, "messages": messages}
