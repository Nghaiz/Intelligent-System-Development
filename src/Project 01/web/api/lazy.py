"""Nhập lười mô-đun của các làn khác; thiếu mô-đun hay thiếu mô hình trả 503 để giao diện báo "đang chuẩn bị"."""
import importlib
import importlib.util
import math
import threading
from contextlib import contextmanager
from functools import wraps

from dnx.mem import _reserved_gb, free_commit_gb, slot

TURN_WAIT_S = 90
COMMIT_MARGIN_GB = 1.0  # chừa cho VS Code và các làn khác, cao hơn lề mặc định của slot()
_heavy = threading.Lock()
_loaded = set()

LANES = {"dnx.ml.serve": "mô hình nguy cơ", "dnx.ml.serve_retina": "mô hình võng mạc", "dnx.kb.search": "kho tri thức",
         "dnx.rules": "bộ luật", "dnx.rl.policy": "chính sách sàng lọc", "dnx.agents.graph": "hệ đa tác tử",
         "dnx.db": "cơ sở dữ liệu"}


class Unavailable(RuntimeError):
    """Thành phần chưa sẵn sàng: HTTP 503 với {"error": ...}."""


def present(name):
    return importlib.util.find_spec(name) is not None


def need(name):
    what = LANES.get(name, name)
    if not present(name):
        raise Unavailable(f"Đang chuẩn bị {what}.")
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as e:
        raise Unavailable(f"Chưa nạp được {what}: {e}") from e


def serving(fn):
    """Mô-đun đã có nhưng chưa có mô hình trên đĩa, hoặc Ollama/Neo4j chưa bật, vẫn là chưa sẵn sàng."""
    @wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (FileNotFoundError, ConnectionError) as e:
            raise Unavailable(f"Chưa sẵn sàng: {e}") from e
    return wrapper


@contextmanager
def heavy_turn(key, gb, what):
    """Mỗi lúc một yêu cầu nặng. Lần nạp mô hình đầu đặt chỗ commit qua dnx.mem.slot; thiếu commit thì trả 503 ngay
    chứ không treo yêu cầu, vì slot() chờ vô hạn còn nạp liều sẽ làm cả máy sập."""
    if not _heavy.acquire(timeout=TURN_WAIT_S):
        raise Unavailable(f"Đang bận tác vụ nặng khác, chưa chạy được {what}.")
    try:
        if key in _loaded:
            yield
            return
        free = free_commit_gb() - _reserved_gb()  # tính cả phần các làn đã đặt chỗ, nếu không slot() sẽ chờ vô hạn
        if free < gb + COMMIT_MARGIN_GB:
            raise Unavailable(f"Chờ bộ nhớ để nạp {what}: cần {gb} GB commit, đang trống {free:.1f} GB.")
        with slot(gb, f"web {what}"):
            yield
        _loaded.add(key)
    finally:
        _heavy.release()


def jsonable(o):
    """Số numpy, NaN và kiểu lạ từ các làn thành JSON chuẩn; NaN không được đi qua JSON của trình duyệt."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if hasattr(o, "item"):
        return jsonable(o.item())
    if isinstance(o, bytes):
        return f"<{len(o)} byte>"
    return str(o)
