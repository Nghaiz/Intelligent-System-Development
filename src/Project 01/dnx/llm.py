"""LLM cục bộ qua Ollama, dùng chung cho KG, tác tử và Web App.

Ollama chạy tiến trình riêng và ánh xạ trọng số GGUF từ đĩa nên gần như không tốn commit của Windows, trong khi
Python chỉ giữ một kết nối HTTP. Máy chủ: `scripts/ollama_serve.ps1`, chỉ nghe ở 127.0.0.1.

    with gpu_session("qwen2.5-7b"):
        llm = LLM("qwen2.5-7b")
        text = llm.chat("Câu hỏi", system="...", max_new_tokens=256)
"""
import json
import re
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

from .common import HF, SEED
from .gpu import heavy
from .mem import slot

HOST = "http://127.0.0.1:11434"
TAGS = {"qwen2.5-7b": "qwen2.5:7b", "gemma3-4b": "gemma3:4b", "qwen2.5-1.5b": "qwen2.5:1.5b"}
MODELS = tuple(TAGS)
THINK = re.compile(r"<think>.*?</think>\s*", re.S)
NUM_CTX = 4096


def _post(path, body, timeout=600):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Không gọi được Ollama ở {HOST}: {e}. Chạy scripts/ollama_serve.ps1") from e


@contextmanager
def gpu_session(name):
    """Mỗi lúc một LLM trên GPU. Bộ nhớ GPU trên Windows cũng cần commit, nên đặt chỗ gần bằng trọng số."""
    with slot(5, f"LLM {name}"), heavy(f"LLM {name}"):
        yield


class LLM:
    def __init__(self, name):
        if name not in TAGS:
            raise ValueError(f"LLM không hợp lệ: {name}; chọn trong {MODELS}")
        self.name = name
        self.tag = TAGS[name]
        self._tok = None
        self.last_usage = {}

    def _body(self, user, system, max_new_tokens, temperature, stream):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        opts = {"temperature": temperature, "num_predict": max_new_tokens, "num_ctx": NUM_CTX, "seed": SEED}
        return {"model": self.tag, "messages": msgs, "stream": stream, "options": opts, "keep_alive": "60s"}

    def chat(self, user, system=None, max_new_tokens=256, temperature=0.0):
        with _post("/api/chat", self._body(user, system, max_new_tokens, temperature, False)) as r:
            d = json.loads(r.read())
        self.last_usage = {"prompt_tokens": d.get("prompt_eval_count", 0), "completion_tokens": d.get("eval_count", 0),
                           "seconds": d.get("total_duration", 0) / 1e9}
        return THINK.sub("", d["message"]["content"]).strip()

    def chat_batch(self, users, system=None, max_new_tokens=256, temperature=0.0):
        return [self.chat(u, system, max_new_tokens, temperature) for u in users]

    def stream(self, user, system=None, max_new_tokens=256, temperature=0.0):
        """Phát từng mảnh chữ thật từ mô hình, dùng cho SSE của Web App."""
        with _post("/api/chat", self._body(user, system, max_new_tokens, temperature, True)) as r:
            for line in r:
                piece = json.loads(line).get("message", {}).get("content")
                if piece:
                    yield piece

    def count_tokens(self, text):
        """Đếm bằng tokenizer gốc của mô hình; chỉ nạp tokenizer.json, không cần transformers."""
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(HF / self.name / "tokenizer.json"))
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def timed(self, user, system=None, max_new_tokens=256):
        t0 = time.perf_counter()
        text = self.chat(user, system, max_new_tokens)
        return text, time.perf_counter() - t0, self.last_usage.get("completion_tokens", 0)
