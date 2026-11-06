"""Toàn bộ API của DiaNexus, đúng mục HTTP trong hợp đồng. Lỗi đi qua trình xử lý chung ở server.py: {"error": ...}."""
import base64
import io
import time

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from dnx.common import ROOT

from . import chat, data, graph, logs, store, system
from .lazy import heavy_turn, jsonable, need, present, serving

router = APIRouter()
GB = {"xgboost": 1, "torch_rl": 2, "retina": 2, "kb": 3, "kb_rerank": 2}  # commit đặt chỗ cho lần nạp đầu của từng mô hình
MAX_IMAGE = 8 << 20
MAX_PIXELS = 40_000_000
MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")
MODES = ("hybrid", "bm25", "dense", "graph", "hybrid_rerank")
RETINA_DIR = ROOT / "web" / "static" / "img" / "retina"


class ChatIn(BaseModel):
    query: str
    profile: dict | None = None
    config: str = "full"
    image_b64: str | None = None
    session_id: str | None = None


class PredictIn(BaseModel):
    profile: dict
    mode: str = "auto"
    session_id: str | None = None


class WhatIfIn(BaseModel):
    profile: dict
    changes: dict


class ScreenIn(BaseModel):
    answers: dict = {}
    session_id: str | None = None


class FeedbackIn(BaseModel):
    session_id: str
    rating: int
    comment: str = ""


@router.get("/status")
def status():
    return system.status()


@router.post("/chat")
def chat_start(body: ChatIn):
    return {"session_id": chat.start(**body.model_dump())}


@router.get("/chat/{sid}/stream")
async def chat_stream(sid: str, request: Request):
    run = chat.get(sid)
    if run is None:
        raise HTTPException(404, "Không có lượt hội chẩn này.")
    return EventSourceResponse(chat.stream(run, request), ping=10)


@router.post("/predict")
@serving
def predict(body: PredictIn):
    if body.mode not in ("auto", "full", "screening"):
        raise ValueError("Chế độ dự đoán không hợp lệ.")
    with heavy_turn("xgboost", GB["xgboost"], "mô hình nguy cơ"):
        out = need("dnx.ml.serve").predict(body.profile, mode=body.mode)
    if present("dnx.rules"):
        out = {**out, "rules": need("dnx.rules").evaluate(body.profile)}
    if body.session_id:
        need("dnx.db").save_prediction(body.session_id, jsonable(out))
    return jsonable(out)


@router.post("/whatif")
@serving
def whatif(body: WhatIfIn):
    with heavy_turn("xgboost", GB["xgboost"], "mô hình nguy cơ"):
        return jsonable(need("dnx.ml.serve").what_if(body.profile, body.changes))


@router.get("/model_card")
@serving
def model_card():
    return jsonable(need("dnx.ml.serve").model_card())


@router.post("/screen/next")
@serving
def screen_next(body: ScreenIn):
    with heavy_turn("torch_rl", GB["torch_rl"], "chính sách sàng lọc"):
        out = jsonable(need("dnx.rl.policy").next_question(body.answers))
    names = {k.replace("_", ""): v for k, v in need("dnx.ml.serve").LABEL_VI.items()}
    out["labels"] = {k: names.get(k.replace("_", ""), k) for k in out["q_values"]} | {"stop": "Dừng hỏi"}
    if out.get("stop") and body.session_id:
        need("dnx.db").save_screening(body.session_id, {"answers": body.answers, "result": out})
    return out


def _check_image(raw):
    if not raw.startswith(MAGIC):
        raise HTTPException(415, "Chỉ nhận ảnh PNG hoặc JPEG.")
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise HTTPException(413, "Ảnh quá nhiều điểm ảnh.")
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise HTTPException(415, "Tệp ảnh hỏng.") from e


@serving
def _grade(raw):
    with heavy_turn("retina", GB["retina"], "mô hình võng mạc"):
        out = need("dnx.ml.serve_retina").grade(raw)
    return {**out, "input_png_b64": _model_view(raw)}


def _model_view(raw):
    """Ảnh đúng khung mô hình nhìn thấy (đã cắt viền, 224 px) để thanh trước-sau trùng khít với Grad-CAM."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(need("dnx.ml.retina").preprocess(Image.open(io.BytesIO(raw)))).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@router.post("/retina")
async def retina(file: UploadFile = File(...)):
    raw = await file.read(MAX_IMAGE + 1)
    if len(raw) > MAX_IMAGE:
        raise HTTPException(413, "Ảnh vượt 8 MB.")
    await run_in_threadpool(_check_image, raw)
    return jsonable(await run_in_threadpool(_grade, raw))


@router.get("/retina/samples")
def retina_samples():
    files = sorted(p.name for p in RETINA_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return {"samples": [f"/static/img/retina/{n}" for n in files]}


@router.get("/search")
@serving
def search(q: str = Query(min_length=1, max_length=400), mode: str = "hybrid", k: int = Query(8, ge=1, le=20)):
    if mode not in MODES:
        raise ValueError(f"Chế độ tra cứu không hợp lệ: {mode[:30]}")
    kb = need("dnx.kb.search")
    key = "kb_rerank" if mode == "hybrid_rerank" else "kb"
    with heavy_turn(key, GB[key], "kho tri thức"):
        t0 = time.perf_counter()
        rows = kb.search(q, k=k, mode=mode)
    return jsonable({"q": q, "mode": mode, "seconds": time.perf_counter() - t0, "results": rows})


@router.get("/graph/overview")
def graph_overview(label: str | None = None, limit: int = Query(160, ge=10, le=300)):
    return jsonable(graph.overview(label, limit))


@router.get("/graph")
def graph_query(q: str | None = Query(None, max_length=200), id: str | None = Query(None, max_length=200),
                hops: int = Query(1, ge=1, le=3), limit: int = Query(120, ge=10, le=300)):
    if not q and not id:
        raise ValueError("Cần q hoặc id.")
    return jsonable(graph.query(q, id, hops, limit))


@router.get("/db")
def db_tables():
    return data.tables()


@router.get("/db/{table}")
def db_page(table: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return jsonable(data.page(table, limit, offset))


@router.get("/db/{table}/stats")
def db_stats(table: str, col: str | None = Query(None, max_length=80)):
    return jsonable(data.stats(table, col))


@router.get("/runs")
def runs():
    return {"rows": jsonable(store.runs())}


@router.get("/metrics")
def metric_list():
    return {"names": store.metric_names()}


@router.get("/metrics/{name}")
def metric(name: str):
    return jsonable(store.metric(name))


@router.get("/logs")
def log_files():
    return {"files": logs.list_files()}


@router.get("/logs/stream")
async def log_stream(request: Request, file: str = "all"):
    return EventSourceResponse(logs.tail(file, request), ping=10)


@router.get("/sessions")
def sessions():
    return {"sessions": jsonable(data.sessions())}


@router.get("/sessions/{sid}")
def session(sid: str):
    return jsonable(data.session_detail(sid))


@router.post("/feedback")
def feedback(body: FeedbackIn):
    if body.rating not in (-1, 1):
        raise ValueError("Đánh giá phải là -1 hoặc 1.")
    need("dnx.db").save_feedback(body.session_id, body.rating, body.comment[:1000])
    return {"ok": True}
