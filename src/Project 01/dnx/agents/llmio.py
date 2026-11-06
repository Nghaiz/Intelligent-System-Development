"""Lời gọi LLM của tác tử. Giữ khóa GPU quanh mỗi lần gọi, hoặc quanh cả lượt đánh giá qua held()."""
import threading
from contextlib import contextmanager

from ..common import LLM_NAME
from ..llm import LLM, gpu_session

_tls = threading.local()


@contextmanager
def held():
    """Giữ một phiên LLM suốt khối lệnh để nhiều lần ask() không phải chờ khóa lại; gọi lồng nhau thì không khóa thêm."""
    if getattr(_tls, "held", False):
        yield
        return
    with gpu_session(LLM_NAME):
        _tls.held = True
        try:
            yield
        finally:
            _tls.held = False


def ask(user, system=None, max_new_tokens=256):
    """Trả (văn bản, usage). Là điểm duy nhất gọi LLM nên kiểm thử chỉ cần thay hàm này."""
    llm = LLM(LLM_NAME)
    with held():
        text = llm.chat(user, system=system, max_new_tokens=max_new_tokens)
    return text, llm.last_usage
