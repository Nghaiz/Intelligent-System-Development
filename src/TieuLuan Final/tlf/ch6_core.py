"""Lõi điều phối Chương 6: tác tử là generator, cùng một mã chạy được cả đánh giá lẫn truyền trực tiếp.

Kiến trúc `yield Call(...)` để xin LLM trả lời và `yield Note(...)` để ghi thông điệp không tốn token (công
cụ,
quyết định theo luật). Bộ điều phối giữ toàn bộ sổ sách, nên không tác tử nào tự khai số token của mình.
"""
import time
from collections import Counter
from dataclasses import dataclass

from .ch6_llm import complete


@dataclass
class Call:
    """Một lượt gọi LLM của tác tử `sender`, thông điệp trả lời đi tới `to`."""
    sender: str
    to: str
    system: str
    user: str
    max_new: int = 256


@dataclass
class Note:
    """Thông điệp không qua LLM: kết quả công cụ hoặc bước quyết định bằng luật."""
    sender: str
    to: str
    content: str


class Run:
    """Một lần chạy kiến trúc trên một câu hỏi: vết thông điệp, sổ token, đáp án."""

    def __init__(self, gen):
        self.gen = gen
        self.messages, self.meter = [], Counter()
        self.pending, self.result, self.done = None, None, False
        self.t0, self.wall_s, self.llm_s = time.perf_counter(), 0.0, 0.0
        self._next(None)

    def _next(self, reply):
        """Đẩy generator tới lời gọi LLM kế tiếp, ghi lại mọi Note gặp trên đường."""
        try:
            step = self.gen.send(reply)
            while isinstance(step, Note):
                self._log(step.sender, step.to, step.content, 0, 0, 0.0)
                step = self.gen.send(None)
            self.pending = step
        except StopIteration as stop:
            self.result, self.done, self.pending = stop.value, True, None
            self.wall_s = time.perf_counter() - self.t0

    def _log(self, sender, to, content, n_in, n_out, seconds):
        msg = {"type": "message", "from": sender, "to": to, "content": content, "tokens": n_in + n_out,
               "tokens_in": n_in, "tokens_out": n_out, "seconds": round(seconds, 3)}
        self.messages.append(msg)
        self.meter.update(tokens_in=n_in, tokens_out=n_out, llm_calls=int(n_in > 0), messages=1)

    def step(self, llm):
        """Thực hiện lời gọi LLM đang chờ rồi chạy tiếp tới lời gọi sau."""
        call = self.pending
        text, n_in, n_out, seconds = complete(llm, call)
        self.llm_s += seconds
        self._log(call.sender, call.to, text, n_in, n_out, seconds)
        self._next(text)


def run_to_end(gen, llm):
    """Chạy trọn một câu hỏi, dùng khi đánh giá."""
    run = Run(gen)
    while not run.done:
        run.step(llm)
    return run


def stream(gen, llm):
    """Chạy một câu hỏi và phát từng thông điệp ngay khi có; cuối cùng phát sự kiện final."""
    run, sent = Run(gen), 0
    while True:
        yield from run.messages[sent:]
        sent = len(run.messages)
        if run.done:
            break
        run.step(llm)
    yield {"type": "final", "answer": run.result["answer"], "rounds": run.result["rounds"],
           "seconds": round(run.wall_s, 2), **dict(run.meter)}
