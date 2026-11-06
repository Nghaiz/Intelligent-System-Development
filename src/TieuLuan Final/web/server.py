"""Đồng Tâm AI Lab: Web App localhost của Tiểu luận Final.

Chạy từ src/TieuLuan Final:  python -m uvicorn web.server:app --host 127.0.0.1 --port 8090
Khởi động không nạp LLM, torch hay TensorFlow; serve_chN chỉ được nhập ở yêu cầu đầu tiên của trang cần nó.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("tensorflow", None)  # nhập TensorFlow trong Web App sẽ báo lỗi ngay, đúng hợp đồng

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from web.api import routes  # noqa: E402

STATIC = Path(__file__).parent / "static"
PAGES = {"", "graph", "chat", "arena", "lab", "atlas", "tracker", "logs"}

app = FastAPI(title="Đồng Tâm AI Lab", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.include_router(routes.router, prefix="/api")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090, log_level="info")
