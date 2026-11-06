"""Đo nhanh ba LLM trên Ollama: thời gian nạp, tốc độ sinh, chất lượng thô, để chọn backend."""
import sys

from tlf.common import save_json
from tlf.llm import LLM, MODELS, gpu_session

sys.stdout.reconfigure(encoding="utf-8")
QS = ["Thủ đô của Việt Nam là gì? Trả lời một câu.",
      "Tom có 3 hộp, mỗi hộp 12 quả. Tom cho đi 7 quả. Còn bao nhiêu quả? Chỉ ghi số.",
      "Metformin được dùng để điều trị bệnh gì? Trả lời ngắn."]
rows = []
for name in MODELS:
    with gpu_session(name):
        llm = LLM(name)
        for q in QS:
            text, secs, n = llm.timed(q, "Bạn là trợ lý ngắn gọn.", 64)
            print(name, round(secs, 2), n, repr(text), flush=True)
            rows.append({"model": name, "q": q, "a": text, "s": secs, "tok": n})
save_json("ch5_llm_bench", {"rows": rows})
print("xong", flush=True)
