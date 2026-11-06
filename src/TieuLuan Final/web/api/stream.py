"""Cầu nối generator chạy LLM (luồng riêng) sang SSE cho trang Trợ lý và Agent Arena."""
import asyncio
import inspect
import json
import threading
from collections.abc import Iterable

from .lab import jsonable, module_state

# Không cho hai lần chạy LLM cùng lúc: GPU chỉ 8 GB, serve_chN cũng tự giữ khóa của nó.
_busy = threading.Lock()


def call_filtered(fn, **kwargs):
    """Gọi hàm serve chỉ với các tham số nó khai, để lệch tên tùy chọn không làm hỏng trang."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


def _events(result):
    """Chuẩn hóa kết quả serve: generator sự kiện, chuỗi token rời, hoặc dict kết quả một lần."""
    if isinstance(result, dict):
        yield {"type": "final", **result}
    elif isinstance(result, str):
        yield {"type": "final", "answer": result}
    elif isinstance(result, Iterable):
        for ev in result:
            yield ev if isinstance(ev, dict) else {"type": "token", "text": str(ev)}


def _worker(start, emit, stop):
    """Mọi lỗi của serve thành sự kiện error để tới được người dùng, không chết im lặng trong luồng."""
    try:
        gen = _events(start())
        for ev in gen:
            emit(ev)
            if stop.is_set():
                gen.close()
                break
    except Exception as e:  # lỗi tải mô hình, thiếu commit, CUDA đều phải hiện nguyên văn
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


def _run(start, chapter, emit, stop):
    mod = module_state(chapter)[0]
    try:
        _worker(start, emit, stop)
    finally:
        # Trả GPU ngay sau mỗi lần chạy để các làn huấn luyện không bị giữ khóa khi trang đứng yên.
        if hasattr(mod, "unload"):
            mod.unload()
        _busy.release()
        emit(None)


async def sse(start, chapter, request):
    """Bọc start() thành luồng SSE; đóng trang thì đặt cờ dừng để generator ngừng ở sự kiện kế tiếp."""
    if not _busy.acquire(blocking=False):
        data = {"type": "error", "message": "Đang có một lần chạy LLM khác, hãy chờ nó xong."}
        yield {"event": "ev", "data": json.dumps(data, ensure_ascii=False)}
        return
    loop, queue, stop = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

    def emit(ev):
        loop.call_soon_threadsafe(queue.put_nowait, ev)
    threading.Thread(target=_run, args=(start, chapter, emit, stop), daemon=True).start()
    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield {"event": "ev", "data": json.dumps(jsonable(ev), ensure_ascii=False)}
            if await request.is_disconnected():
                break
    finally:
        stop.set()
