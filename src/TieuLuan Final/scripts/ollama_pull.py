"""Tải các LLM vào Ollama (máy chủ đã chạy ở 127.0.0.1:11434) qua API HTTP, ghi tiến độ vào stdout."""
import json
import sys
import urllib.request

MODELS = sys.argv[1:] or ["qwen2.5:7b", "gemma3:4b", "qwen2.5:1.5b"]

for name in MODELS:
    body = json.dumps({"model": name, "stream": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/pull", data=body)
    last = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            ev = json.loads(line)
            if ev.get("status") != last:
                last = ev.get("status")
                print(name, last, ev.get("total", ""), flush=True)
            if "error" in ev:
                print("LOI", name, ev["error"], flush=True)
print("xong", flush=True)
