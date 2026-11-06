"""Chương 5: API cho Web App. ask() là generator phát sự kiện để web truyền SSE; LLM chạy trên Ollama, khóa GPU chỉ giữ trong lúc sinh.

Sự kiện: {"type":"step","kind","text"}, {"type":"retrieved","chunks":[...]}, {"type":"token","text"},
{"type":"final","answer","sources","mode","model","dataset","seconds","note"}.
"""
import time

from . import ch5_agent as A
from . import ch5_neo as N
from . import ch5_rag as G
from .ch5_retrieval import Dense, Graph, encode, release_embedder, retrievers
from .llm import LLM, MODELS, gpu_session

MODES = ("naive", "graph", "hybrid", "agent")
DEFAULT_MODEL = "qwen2.5-7b"
ASK_MODELS = list(MODELS)
SPECS = [{"id": "ask", "title": "Hỏi đáp trên đồ thị tri thức", "dataset": "Hetionet, VN30, HotpotQA", "output": "text",
          "inputs": [{"name": "question", "type": "text", "label": "Câu hỏi", "default": "Mã FPT thuộc ngành nào?"},
                     {"name": "mode", "type": "select", "label": "Chế độ", "default": "graph", "options": list(MODES)},
                     {"name": "model", "type": "select", "label": "LLM", "default": DEFAULT_MODEL, "options": ASK_MODELS}]}]
_retr = {}


def get_retriever(ds, mode):
    if mode == "naive":
        return Dense(ds)
    if mode == "graph":
        return Graph(ds)
    if ds not in _retr:
        _retr[ds] = retrievers(ds)["Hybrid"]
    return _retr[ds]


def _note(ds):
    return {"hetionet": G.MEDICAL_NOTE, "vn30": G.INVEST_NOTE}.get(ds)


def _final(answer, sources, mode, model, ds, t0, **extra):
    return {"type": "final", "answer": answer, "sources": sources, "mode": mode, "model": model, "dataset": ds,
            "seconds": round(time.perf_counter() - t0, 2), "note": _note(ds), **extra}


def _ask_agent(question, model, ds, t0):
    with gpu_session(model):
        for ev in A.run_agent(LLM(model), question, ds):
            if ev["kind"] == "final":
                yield _final(ev["text"], ev["sources"], "agent", model, ds, t0, steps=ev["steps"], forced=ev["forced"])
            else:
                yield {"type": "step", **ev}


def _ask_rag(question, mode, model, ds, t0):
    yield {"type": "step", "kind": "thought", "text": f"Miền: {ds}. Nạp bộ nhúng và chỉ mục, lần đầu mất chừng một phút."}
    retriever = get_retriever(ds, mode)
    vec = encode([question], device="cpu")[0]
    yield {"type": "step", "kind": "action", "text": f"Truy xuất bằng {mode}."}
    hits = retriever.search(question, vec, 4)
    table = G.chunk_table(ds)
    yield {"type": "retrieved", "chunks": [{"id": c, "score": round(s, 4), "title": table[c]["title"], "source": table[c]["source"],
                                            "url": table[c].get("url"), "text": G.clip(table[c]["text"], 60)} for c, s in hits]}
    ids = [c for c, _ in hits]
    text = ""
    yield {"type": "step", "kind": "thought", "text": "Chờ tới lượt GPU rồi sinh câu trả lời."}
    with gpu_session(model):
        for piece in LLM(model).stream(G.user_prompt(question, G.context_of(ds, ids)), G.SYSTEM, 96):
            text += piece
            yield {"type": "token", "text": piece}
    yield _final(text.strip(), ids, mode, model, ds, t0)


def ask(question, mode="graph", model=DEFAULT_MODEL, dataset=None):
    """Generator sự kiện; mọi lỗi đổi thành sự kiện final có trường error để web hiển thị, không làm đứt SSE."""
    t0 = time.perf_counter()
    if mode not in MODES or model not in ASK_MODELS:
        raise ValueError(f"mode phải thuộc {MODES}, model phải thuộc {ASK_MODELS}")
    ds = dataset or G.route(question)
    if G.is_clinical(question):
        yield _final(G.REFUSAL, [], mode, model, ds, t0, blocked=True)
        return
    try:
        yield from (_ask_agent(question, model, ds, t0) if mode == "agent" else _ask_rag(question, mode, model, ds, t0))
    except Exception as e:
        yield _final("", [], mode, model, ds, t0, error=f"{type(e).__name__}: {e}")


def predict(spec_id, payload):
    """Giao diện chung của Web App: gom sự kiện của ask() thành một kết quả."""
    if spec_id != "ask":
        raise KeyError(spec_id)
    events = list(ask(payload["question"], payload.get("mode", "graph"), payload.get("model", DEFAULT_MODEL)))
    return {"text": events[-1]["answer"], "events": events}


def unload():
    """Giải phóng bộ nhúng e5 và chỉ mục BM25 đã nạp lười; lần hỏi sau nạp lại. Web gọi khi rảnh."""
    _retr.clear()
    return release_embedder()


def graph_stats():
    return N.stats()


def subgraph(seed, depth=1, limit=200):
    return N.subgraph(seed, depth, limit)

