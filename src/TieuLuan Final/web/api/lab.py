"""Phát hiện `tlf/serve_chN.py` lúc chạy và gọi predict; thiếu tệp hay lỗi nạp đều trả trạng thái, không sập.

Các làn còn đang sửa serve nên mô-đun được nạp lại khi mtime đổi, server không cần khởi động lại.
"""
import importlib
import sys

import numpy as np

from tlf.common import ROOT

LAB_CHAPTERS = (2, 3, 4, 7, 8)
_loaded = {}


def jsonable(v):
    """Đổi kiểu NumPy và Path trong kết quả predict về kiểu JSON; NaN thành None."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v if v is None or isinstance(v, (str, int, float, bool)) else str(v)


def module_state(n):
    """Trả (mô-đun hoặc None, trạng thái, thông báo). Mô-đun chỉ nạp ở yêu cầu đầu tiên, không nạp lúc khởi động."""
    path = ROOT / "tlf" / f"serve_ch{n}.py"
    if not path.exists():
        return None, "missing", f"Chưa có tlf/serve_ch{n}.py"
    name, mtime = f"tlf.serve_ch{n}", path.stat().st_mtime
    try:
        held = _loaded.get(n)
        if held is None:
            mod = sys.modules.get(name) or importlib.import_module(name)
        elif held[1] != mtime:
            mod = importlib.reload(held[0])
        else:
            mod = held[0]
    except Exception as e:  # tệp của làn khác có thể đang viết dở; lỗi nào cũng phải thành trạng thái hiển thị
        return None, "error", f"{type(e).__name__}: {e}"
    _loaded[n] = (mod, mtime)
    return mod, "ok", ""


def specs():
    out = []
    for n in LAB_CHAPTERS:
        mod, state, msg = module_state(n)
        out.append({"chapter": n, "status": state, "message": msg,
                    "specs": jsonable(getattr(mod, "SPECS", [])) if mod else []})
    return out


def predict(n, spec_id, payload):
    mod, _, msg = module_state(n)
    if mod is None:
        raise LookupError(msg)
    return jsonable(mod.predict(spec_id, payload))


def example_path(n, name):
    """Đường dẫn ảnh mẫu; tên phải nằm trong `options` của SPECS nên không thể trỏ ra ngoài thư mục mẫu."""
    mod, _, msg = module_state(n)
    root = getattr(mod, "EXAMPLES", None)
    if root is None:
        raise LookupError(msg or "Mô-đun không có ảnh mẫu")
    allowed = {o for s in mod.SPECS for i in s["inputs"] for o in (i.get("options") or []) if isinstance(o, str)}
    if name not in allowed:
        raise LookupError(f"Ảnh mẫu không hợp lệ: {name}")
    return root / name
