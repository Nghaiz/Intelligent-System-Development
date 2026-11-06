"""Các điểm cuối REST và SSE. Lỗi có chủ đích đổi sang mã HTTP kèm `detail` tiếng Việt để giao diện hiện nguyên văn."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from neo4j.exceptions import ClientError, ServiceUnavailable
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from . import graph, lab, logs, replay, store, stream, system

router = APIRouter()
LLM_MODELS = ("qwen2.5-7b", "gemma3-4b", "qwen2.5-1.5b")
RAG_MODES = ("graphrag", "dense", "bm25")
ARENA_ARCHS = ("single", "supervisor", "debate")


class CypherIn(BaseModel):
    q: str = Field(min_length=1, max_length=4000)


def _neo_call(fn, *args):
    """Neo4j tắt thì 503, câu sai cú pháp hoặc bị chặn thì 400."""
    try:
        return fn(*args)
    except (graph.Unavailable, ServiceUnavailable) as e:
        raise HTTPException(503, str(e)) from e
    except (ValueError, ClientError) as e:
        raise HTTPException(400, getattr(e, "message", None) or str(e)) from e


@router.get("/overview")
def overview():
    data = store.overview()
    try:
        s = graph.overview()
        data["graph"] = {"nodes": s["nodes"], "relationships": s["relationships"]}
    except (graph.Unavailable, ServiceUnavailable):
        data["graph"] = None
    return lab.jsonable(data)


@router.get("/runs")
def runs():
    return lab.jsonable({"rows": store.runs()})


@router.get("/compare")
def compare():
    return lab.jsonable({"tables": store.compare()})


@router.get("/datasets")
def datasets():
    return lab.jsonable({"datasets": store.datasets()})


@router.get("/status")
def status():
    return system.status()


@router.get("/logs")
def log_files():
    return {"files": logs.list_files()}


@router.get("/logs/stream")
async def log_stream(request: Request, file: str = "all"):
    return EventSourceResponse(logs.tail(file, request))


@router.get("/graph/overview")
def graph_overview():
    return _neo_call(graph.overview)


@router.get("/graph/sample")
def graph_sample(label: str | None = None, limit: int = Query(160, ge=10, le=400)):
    return _neo_call(graph.sample, label, limit)


@router.get("/graph/expand")
def graph_expand(id: str, limit: int = Query(60, ge=5, le=200)):
    return _neo_call(graph.expand, id, limit)


@router.get("/graph/node")
def graph_node(id: str):
    found = _neo_call(graph.node_detail, id)
    if found is None:
        raise HTTPException(404, f"Không có nút {id}")
    return found


@router.get("/graph/search")
def graph_search(q: str = Query(min_length=2, max_length=80)):
    return {"nodes": _neo_call(graph.search, q)}


@router.post("/graph/cypher")
def graph_cypher(body: CypherIn):
    return _neo_call(graph.run_cypher, body.q)


@router.get("/lab/specs")
def lab_specs():
    return {"chapters": lab.specs()}


@router.post("/lab/predict/{chapter}/{spec_id}")
def lab_predict(chapter: int, spec_id: str, payload: dict):
    if chapter not in lab.LAB_CHAPTERS:
        raise HTTPException(404, f"Chương {chapter} không có trong Model Lab")
    try:
        return lab.predict(chapter, spec_id, payload)
    except (LookupError, FileNotFoundError) as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # lỗi trong serve của làn khác phải hiện nguyên văn trên trang
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


@router.get("/lab/example/{chapter}/{name}")
def lab_example(chapter: int, name: str):
    try:
        path = Path(lab.example_path(chapter, name))
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    return FileResponse(path)


@router.get("/chat/options")
def chat_options():
    mod, state, message = lab.module_state(5)
    modes = getattr(mod, "MODES", RAG_MODES) if mod else RAG_MODES
    return {"status": state, "message": message, "modes": list(modes), "models": list(LLM_MODELS),
            "has_ask": hasattr(mod, "ask")}


@router.get("/chat/stream")
async def chat_stream(request: Request, q: str = Query(min_length=2, max_length=800),
                      mode: str = "graphrag", model: str = LLM_MODELS[0], dataset: str = "hetionet"):
    mod, _, message = lab.module_state(5)
    if not hasattr(mod, "ask"):
        raise HTTPException(409, message or "tlf/serve_ch5.py chưa có hàm ask")

    def start():
        return stream.call_filtered(mod.ask, question=q, mode=mode, model=model, llm=model, dataset=dataset)
    return EventSourceResponse(stream.sse(start, 5, request))


@router.get("/arena/options")
def arena_options():
    mod, state, message = lab.module_state(6)
    spec = next(iter(getattr(mod, "SPECS", [])), None)
    return {"status": state, "message": message, "archs": list(ARENA_ARCHS), "models": list(LLM_MODELS),
            "inputs": lab.jsonable(spec["inputs"]) if spec else [], "replays": replay.listing()}


@router.get("/arena/replay/{index}")
def arena_replay(index: int):
    item = replay.load(index)
    if item is None:
        raise HTTPException(404, "Không có vết đã ghi này")
    return item


@router.get("/arena/stream")
async def arena_stream(request: Request, question: str = Query(min_length=2, max_length=2000),
                       arch: str = "supervisor", model: str = LLM_MODELS[0], dataset: str = "clinical",
                       context: str = ""):
    mod, _, message = lab.module_state(6)
    if not hasattr(mod, "run"):
        raise HTTPException(409, message or "tlf/serve_ch6.py chưa có hàm run")

    def start():
        return mod.run(arch, question, model, dataset=dataset, context=context)
    return EventSourceResponse(stream.sse(start, 6, request))
