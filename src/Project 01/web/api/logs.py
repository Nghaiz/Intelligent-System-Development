"""Đuôi các tệp outputs/logs/*.log truyền qua SSE. Tên tệp chỉ khớp theo danh sách đã quét, không ghép đường dẫn."""
import asyncio
import json
import re

from dnx.common import LOGS

TAIL_BYTES = 16_000
READ_CHUNK = 1 << 20
MAX_LINE = 1600
POLL_S = 1.0
ERROR_RE = re.compile(r"\b(error|exception|traceback|failed|lỗi|critical|fatal)\b|0xC000", re.I)
WARN_RE = re.compile(r"\b(warn(ing)?|cảnh báo|deprecat\w*)\b", re.I)
DEBUG_RE = re.compile(r"\bdebug\b", re.I)
SPLIT_RE = re.compile(r"[\r\n]+")


def level_of(line):
    if ERROR_RE.search(line):
        return "error"
    if WARN_RE.search(line):
        return "warn"
    return "debug" if DEBUG_RE.search(line) else "info"


def list_files():
    files = sorted(LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in files]


def _targets(name):
    if name in (None, "", "all"):
        return list(LOGS.glob("*.log"))
    return [p for p in LOGS.glob("*.log") if p.name == name]


def _read_new(path, offset):
    """Đọc phần mới từ offset; tệp bị ghi lại nhỏ đi thì đọc từ đầu. Trả (các dòng, offset mới)."""
    size = path.stat().st_size
    if size < offset:
        offset = 0
    if size == offset:
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(READ_CHUNK)
    end = max(raw.rfind(b"\n"), raw.rfind(b"\r"))
    if end < 0 and len(raw) < READ_CHUNK:
        return [], offset  # dòng đang ghi dở, chờ lượt sau
    body = raw[:end + 1] if end >= 0 else raw
    return [s for s in SPLIT_RE.split(body.decode("utf-8", "replace")) if s.strip()], offset + len(body)


def _events(path, offsets):
    try:
        lines, offsets[path] = _read_new(path, offsets[path])
    except OSError:
        return []
    return [{"event": "line", "data": json.dumps(
        {"file": path.name, "level": level_of(s), "text": s[:MAX_LINE]}, ensure_ascii=False)} for s in lines]


async def tail(name, request):
    """Bắt đầu từ đoạn cuối mỗi tệp, sau đó chờ dòng mới; tệp log mới sinh được nhận ở lượt quét sau."""
    offsets = {p: max(0, p.stat().st_size - TAIL_BYTES) for p in _targets(name)}
    while not await request.is_disconnected():
        for p in _targets(name):
            offsets.setdefault(p, 0)
        for path in list(offsets):
            for ev in _events(path, offsets):
                yield ev
        await asyncio.sleep(POLL_S)
