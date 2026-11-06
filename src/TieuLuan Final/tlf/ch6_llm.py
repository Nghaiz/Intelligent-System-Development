"""Cầu nối tới LLM cục bộ của Chương 6: mọi lời gọi đi qua LLM chung (Ollama), ở đây chỉ thêm chuỗi dừng và
sổ token."""
import time

from .common import LLM_NAME
from .llm import LLM, MODELS, gpu_session  # noqa: F401  (gpu_session được các kịch bản nhập từ đây)

MODEL = LLM_NAME if LLM_NAME in MODELS else MODELS[0]
STOP = "Observation:"


class AgentLLM(LLM):
    """LLM dừng ngay trước Observation, để tác tử không tự viết kết quả công cụ rồi tốn lượt sinh vô ích."""

    def _body(self, user, system, max_new_tokens, temperature, stream):
        body = super()._body(user, system, max_new_tokens, temperature, stream)
        body["options"]["stop"] = [STOP]
        return body


def open_llm(name=MODEL):
    """Tạo kết nối và làm nóng mô hình một lượt, để giây nạp mô hình không rơi vào độ trễ của câu đầu."""
    llm = AgentLLM(name)
    llm.chat("ok", max_new_tokens=1)
    return llm


def complete(llm, call):
    """Gọi LLM cho một Call; trả (văn bản, token vào, token ra, số giây).

    Token ra lấy từ Ollama. Token vào đếm lại bằng tokenizer của mô hình trên system + user, vì Ollama
    không tính
    phần prompt còn nằm trong bộ đệm; số này chưa gồm vài token khuôn hội thoại.
    """
    t0 = time.perf_counter()
    text = llm.chat(call.user, system=call.system, max_new_tokens=call.max_new)
    seconds = time.perf_counter() - t0
    n_in = llm.count_tokens(call.system) + llm.count_tokens(call.user)
    return text.strip(), n_in, llm.last_usage["completion_tokens"], seconds
