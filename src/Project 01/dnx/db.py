"""SQLite dùng chung `data/dianexus.db` (WAL): bảng dữ liệu do scripts/data_load.py nạp, bảng vận hành do DDL dưới đây.

Mỗi lời gọi ghi mở rồi đóng kết nối riêng nên an toàn giữa nhiều luồng/tiến trình của web và tác tử.
"""
import json
import sqlite3
import time
from contextlib import closing

from .common import DATA, _to_py

DB_PATH = DATA / "dianexus.db"
DATA_TABLES = ("patients_lab", "brfss", "hospital", "pima", "frankfurt")
OPS_TABLES = ("sessions", "messages", "predictions", "agent_events", "screenings", "feedback")

_DDL = """
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, title TEXT);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
  role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT REFERENCES sessions(id),
  payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS agent_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT REFERENCES sessions(id),
  t REAL, agent TEXT, kind TEXT, detail TEXT, data TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS screenings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT REFERENCES sessions(id),
  payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT REFERENCES sessions(id),
  rating INTEGER, comment TEXT, payload TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS ix_events_session ON agent_events(session_id, id);
CREATE INDEX IF NOT EXISTS ix_predictions_session ON predictions(session_id, id);
"""


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, default=_to_py)


def connect():
    """Kết nối mới, bật WAL và khóa ngoại, bảo đảm đủ bảng vận hành; hàng trả về dạng sqlite3.Row."""
    con = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(_DDL)
    return con


def _ensure_session(con, session_id):
    if session_id is not None:
        con.execute("INSERT OR IGNORE INTO sessions(id, created_at) VALUES(?, ?)", (session_id, _now()))


def _write(session_id, sql, args):
    with closing(connect()) as con, con:
        _ensure_session(con, session_id)
        return con.execute(sql, args).lastrowid


def log_event(session_id, event):
    """Ghi một Event của tác tử: {"t", "agent", "kind", "detail", "data"}."""
    return _write(session_id,
                  "INSERT INTO agent_events(session_id, t, agent, kind, detail, data, created_at) VALUES(?,?,?,?,?,?,?)",
                  (session_id, event.get("t"), event.get("agent"), event.get("kind"), event.get("detail"),
                   _dumps(event.get("data") or {}), _now()))


def save_message(session_id, role, content):
    return _write(session_id, "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
                  (session_id, role, content, _now()))


def save_prediction(session_id, payload):
    return _write(session_id, "INSERT INTO predictions(session_id, payload, created_at) VALUES(?,?,?)",
                  (session_id, _dumps(payload), _now()))


def save_screening(session_id, payload):
    return _write(session_id, "INSERT INTO screenings(session_id, payload, created_at) VALUES(?,?,?)",
                  (session_id, _dumps(payload), _now()))


def save_feedback(session_id, rating, comment="", payload=None):
    return _write(session_id, "INSERT INTO feedback(session_id, rating, comment, payload, created_at) VALUES(?,?,?,?,?)",
                  (session_id, rating, comment, _dumps(payload or {}), _now()))
