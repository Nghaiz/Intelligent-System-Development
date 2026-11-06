"""DiaNexus: Web localhost của Project 01.

Chạy từ src/Project 01:  python -m uvicorn web.server:app --host 127.0.0.1 --port 8097
Khởi động không nạp torch hay LLM; mô-đun của các làn khác chỉ được nhập ở yêu cầu đầu tiên cần nó.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTP  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

from web.api import routes  # noqa: E402
from web.api.lazy import Unavailable  # noqa: E402

STATIC = Path(__file__).parent / "static"
PAGES = {"", "consult", "screen", "risk", "retina", "graph", "search", "data", "lab", "logs"}
MAX_BODY = 12 << 20  # chặn sớm thân yêu cầu quá lớn; ảnh có giới hạn riêng trong routes
CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self'; connect-src 'self'; worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")

app = FastAPI(title="DiaNexus", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.include_router(routes.router, prefix="/api")


@app.middleware("http")
async def guard(request: Request, call_next):
    size = request.headers.get("content-length")
    if size and size.isdigit() and int(size) > MAX_BODY:
        return JSONResponse({"error": "Dữ liệu gửi lên quá lớn."}, status_code=413)
    response = await call_next(request)
    if not request.url.path.startswith("/api/docs"):  # trang /api/docs dùng tài nguyên ngoài của FastAPI
        response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])  # chặn DNS rebinding


@app.exception_handler(Unavailable)
async def _unavailable(_, e):
    return JSONResponse({"error": str(e)}, status_code=503)


@app.exception_handler(StarletteHTTP)
async def _http(_, e):
    return JSONResponse({"error": str(e.detail)}, status_code=e.status_code)


@app.exception_handler(RequestValidationError)
async def _invalid(_, e):
    first = e.errors()[0] if e.errors() else {}
    where = ".".join(str(x) for x in first.get("loc", []))
    return JSONResponse({"error": f"Dữ liệu không hợp lệ ở {where}: {first.get('msg', '')}"}, status_code=422)


@app.exception_handler(ValueError)
async def _value(_, e):
    return JSONResponse({"error": str(e)}, status_code=400)


@app.exception_handler(Exception)
async def _crash(_, e):
    return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


class FreshStatic(StaticFiles):
    """Trình duyệt phải hỏi lại mỗi lần, nếu không mô-đun JS cũ vẫn chạy sau khi sửa mã."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", FreshStatic(directory=STATIC), name="static")


@app.get("/{page:path}", include_in_schema=False)
def spa(page: str, request: Request):
    """Mọi trang dùng chung một index.html; đường dẫn lạ trả 404 thật thay vì một trang trống."""
    if page.strip("/") not in PAGES:
        raise HTTPException(404, f"Không có trang {request.url.path}")
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
