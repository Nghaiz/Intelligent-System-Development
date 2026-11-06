"""Phiên hội chẩn: POST tạo một lượt chạy trong luồng riêng, GET stream phát lại Event từ đầu rồi theo tiếp qua SSE."""
import asyncio
import base64
import binascii
import json
import re
import threading
import time
import uuid
from collections import OrderedDict

from .lazy import Unavailable, heavy_turn, jsonable, need

AGENTS_GB = 4  # nạp cả XGBoost, kho tri thức và torch trong tiến trình web; LLM nằm ở Ollama
CONFIGS = ("full", "no_critic", "vector_only", "single_agent", "no_rules")
SID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")
KEEP = 24
MAX_QUERY = 2000
MAX_IMAGE = 8 << 20
POLL_S = 0.08
_runs = OrderedDict()
_busy = threading.Lock()  # GPU 8 GB: mỗi lúc một lượt hội chẩn


class Run:
    def __init__(self, sid):
        self.sid, self.events, self.done, self.t0 = sid, [], False, time.time()

    def push(self, ev):
        self.events.append(jsonable(ev))


def _worker(run, graph, kwargs):
    try:
        db = need("dnx.db")
        db.save_message(run.sid, "user", kwargs["query"])
        with heavy_turn("agents", AGENTS_GB, "hệ đa tác tử"):
            result = graph.run(on_event=run.push, session_id=run.sid, **kwargs)
        db.save_message(run.sid, "assistant", result["answer"])
        run.push({"kind": "result", "agent": "DiaNexus", "detail": "", "t": time.time() - run.t0, "data": result})
    except Exception as e:  # lỗi Ollama, thiếu commit, thiếu mô hình phải tới được người dùng nguyên văn
        run.push({"kind": "fatal", "agent": "DiaNexus", "detail": f"{type(e).__name__}: {e}",
                  "t": time.time() - run.t0, "data": {}})
    finally:
        run.done = True
        _busy.release()


def _image(b64):
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64.split(",")[-1], validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("Ảnh đính kèm không hợp lệ.") from e
    if len(raw) > MAX_IMAGE:
        raise ValueError("Ảnh đính kèm vượt 8 MB.")
    return raw


def start(query, profile=None, config="full", image_b64=None, session_id=None):
    query = (query or "").strip()
    if not query or len(query) > MAX_QUERY:
        raise ValueError(f"Câu hỏi phải có từ 1 đến {MAX_QUERY} ký tự.")
    if config not in CONFIGS:
        raise ValueError(f"Cấu hình không hợp lệ: {config[:40]}")
    if session_id and not SID_RE.match(session_id):
        raise ValueError("Mã phiên không hợp lệ.")
    image, graph = _image(image_b64), need("dnx.agents.graph")
    if not _busy.acquire(blocking=False):
        raise Unavailable("Đang có một lượt hội chẩn khác, hãy chờ nó xong.")
    run = Run(session_id or uuid.uuid4().hex)
    _runs[run.sid] = run  # cùng phiên hỏi tiếp thì lượt mới thay lượt cũ trong bộ đệm SSE
    _runs.move_to_end(run.sid)
    while len(_runs) > KEEP:
        _runs.popitem(last=False)
    kwargs = {"query": query, "profile": profile, "image": image, "config": config}
    threading.Thread(target=_worker, args=(run, graph, kwargs), daemon=True).start()
    return run.sid


def get(sid):
    return _runs.get(sid)


async def stream(run, request):
    i = 0
    while not await request.is_disconnected():
        while i < len(run.events):
            yield {"event": "ev", "data": json.dumps(run.events[i], ensure_ascii=False)}
            i += 1
        if run.done:
            yield {"event": "done", "data": "{}"}
            return
        await asyncio.sleep(POLL_S)
