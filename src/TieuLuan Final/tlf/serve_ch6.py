"""Phục vụ Web App Chương 6. run() là generator để web truyền SSE; predict() gom kết quả cho biểu mẫu thường.

Ví dụ: for ev in run("debate", "Natalia sold 48 clips...", "qwen2.5-7b", dataset="gsm8k"): gửi json.dumps(ev).
Sự kiện: {"type":"message","from","to","content","tokens",...} rồi {"type":"final","answer","rounds",...}.
Mô hình nạp lười và giữ lại cùng khóa GPU; gọi unload() khi xong để trả GPU cho tiến trình khác.
"""
import threading

from .ch6_arch import ARCHS
from .ch6_core import stream
from .ch6_domain import DOMAINS
from .ch6_llm import MODEL, gpu_session, open_llm

SAMPLE = ("Bệnh nhân nam 58 tuổi, đái tháo đường típ 2. eGFR 42 mL/phút/1,73m2, kali 4,6 mmol/L, huyết áp 138/82 "
          "mmHg, nhịp tim 76 lần/phút, uống 4 ly rượu/ngày. Đơn đề xuất: Metformin 1000 mg, 2 lần/ngày.")
SPECS = [{"id": "ch6_agents", "title": "Hệ đa tác tử trên LLM cục bộ", "dataset": "gsm8k, pubmed, clinical",
          "inputs": [{"name": "arch", "type": "select", "label": "Kiến trúc", "default": "supervisor",
                      "options": ["single", "supervisor", "debate"]},
                     {"name": "dataset", "type": "select", "label": "Miền", "default": "clinical",
                      "options": list(DOMAINS)},
                     {"name": "question", "type": "textarea", "label": "Câu hỏi hoặc ca bệnh giả lập",
                      "default": SAMPLE},
                     {"name": "context", "type": "textarea", "label": "Tóm tắt bài báo (chỉ PubMedQA)",
                      "default": ""}],
          "output": "text"}]
_lock = threading.Lock()
_held = {"name": None, "llm": None, "session": None}


def unload():
    """Giải phóng mô hình đang giữ và nhả khóa GPU."""
    if _held["session"] is not None:
        _held["llm"] = None
        _held["session"].__exit__(None, None, None)
        _held.update(name=None, session=None)


def _get(model):
    if _held["name"] != model:
        unload()
        session = gpu_session(model)
        session.__enter__()
        _held.update(name=model, llm=open_llm(model), session=session)
    return _held["llm"]


def run(arch, question, model=MODEL, dataset="clinical", context=""):
    """Generator phát từng thông điệp của lần chạy, rồi một sự kiện final. Mỗi lúc phục vụ một lần chạy."""
    if arch not in ARCHS:
        raise ValueError(f"Kiến trúc không hợp lệ: {arch}. Chọn một trong {list(ARCHS)}")
    if dataset not in DOMAINS:
        raise ValueError(f"Miền không hợp lệ: {dataset}. Chọn một trong {list(DOMAINS)}")
    item = {"id": "web", "question": question, "context": context}
    with _lock:
        yield from stream(ARCHS[arch](item, DOMAINS[dataset]), _get(model))


def predict(spec_id, payload):
    """Giao diện chung của web: chạy hết rồi trả văn bản kết luận, vết thông điệp nằm trong `messages`."""
    if spec_id != "ch6_agents":
        raise KeyError(f"Không có mô hình {spec_id}")
    events = list(run(payload.get("arch", "supervisor"), payload["question"], payload.get("model", MODEL),
                      payload.get("dataset", "clinical"), payload.get("context", "")))
    final = events[-1]
    return {"text": str(final["answer"]), "messages": events[:-1], "final": final}
