"""Chụp ảnh màn hình ba giao diện Web và Mobile để đưa vào báo cáo.

Kịch bản cho mỗi ứng dụng:
    1. Khởi động REST API dưới dạng tiến trình nền, chờ /health trả 200.
    2. Mở trang Web (khung máy tính) → bấm "Điền dữ liệu mẫu" → chụp trạng thái
       đã nhập → bấm "Dự đoán" → chờ kết quả → chụp trạng thái kết quả.
    3. Lặp lại y hệt cho trang Mobile (khung điện thoại).
    4. Tắt tiến trình API.

Chạy:  python report/capture_screenshots.py

Kết quả: report/screenshots/<app>_<web|mobile>_<form|result>.png
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1] / "src" / "Assignment 03"
SHOTS = REPORT / "screenshots"
PYTHON = sys.executable

APPS = [
    {"key": "s1_diabetes", "dir": "diabetes_large", "port": 5001,
     "prefix": "/diabetes/v1", "label": "Sàng lọc tiểu đường"},
    {"key": "s2_house", "dir": "house_price_large", "port": 5002,
     "prefix": "/house-price/v1", "label": "Định giá bất động sản"},
    {"key": "s3_comments", "dir": "customer_comments", "port": 5003,
     "prefix": "/comments/v1", "label": "Phân loại nhận xét"},
]

WEB_VIEWPORT = {"width": 1440, "height": 960}
MOBILE_VIEWPORT = {"width": 900, "height": 940}


def wait_health(port: int, prefix: str, timeout: float = 75.0) -> bool:
    """Chờ tới khi <prefix>/health trả về HTTP 200 hoặc hết thời gian chờ."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}{prefix}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    return False


def shoot(page, url: str, out_prefix: str, viewport: dict) -> list[str]:
    """Mở trang, điền mẫu, chụp; bấm dự đoán, chờ kết quả, chụp lần hai."""
    saved = []
    page.set_viewport_size(viewport)
    page.goto(url, wait_until="networkidle", timeout=45_000)
    page.wait_for_timeout(1200)

    # Bước 1 — điền dữ liệu mẫu rồi chụp trạng thái đã nhập liệu
    try:
        page.click("#btnSample", timeout=8_000)
        page.wait_for_timeout(700)
    except Exception as exc:
        print(f"      ! không bấm được #btnSample: {exc}")

    p1 = SHOTS / f"{out_prefix}_form.png"
    page.screenshot(path=str(p1), full_page=True)
    saved.append(p1.name)

    # Bước 2 — gọi dự đoán, chờ khối kết quả hiện ra rồi chụp
    try:
        page.click("#btnPredict", timeout=8_000)
        try:
            page.wait_for_selector("#resultBody:visible", timeout=25_000)
        except Exception:
            page.wait_for_timeout(6_000)
        page.wait_for_timeout(2_200)      # chờ hiệu ứng thanh xác suất chạy xong

        # Khung điện thoại có vùng cuộn riêng bên trong, nên full_page không với
        # tới khối kết quả. Phải cuộn chính vùng đó xuống thì ảnh mới thấy kết quả.
        try:
            page.locator("#resultBody").scroll_into_view_if_needed(timeout=5_000)
            page.wait_for_timeout(900)
        except Exception:
            pass
    except Exception as exc:
        print(f"      ! không bấm được #btnPredict: {exc}")

    p2 = SHOTS / f"{out_prefix}_result.png"
    page.screenshot(path=str(p2), full_page=True)
    saved.append(p2.name)
    return saved


def main() -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    total = 0

    try:
        # --- khởi động cả ba API cùng lúc, mỗi API một cổng riêng ---
        for app in APPS:
            script = ROOT / app["dir"] / "api" / "rest_api.py"
            if not script.exists():
                print(f"  ! thiếu {script}")
                continue
            proc = subprocess.Popen(
                [PYTHON, str(script)], cwd=str(ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(proc)
            print(f"  khởi động {app['dir']} trên cổng {app['port']} (pid {proc.pid})")

        ready = []
        for app in APPS:
            ok = wait_health(app["port"], app["prefix"])
            print(f"  {'✓' if ok else '✗'} {app['prefix']}/health cổng {app['port']}")
            if ok:
                ready.append(app)

        if not ready:
            print("! Không API nào sẵn sàng — bỏ qua bước chụp ảnh.")
            return 1

        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=True)
            for app in ready:
                base = f"http://127.0.0.1:{app['port']}"
                print(f"  chụp {app['label']} …")

                ctx = browser.new_context(viewport=WEB_VIEWPORT,
                                          device_scale_factor=2, locale="vi-VN")
                page = ctx.new_page()
                for n in shoot(page, f"{base}/", f"{app['key']}_web", WEB_VIEWPORT):
                    print(f"      ✓ {n}")
                    total += 1
                ctx.close()

                ctx = browser.new_context(viewport=MOBILE_VIEWPORT,
                                          device_scale_factor=2, locale="vi-VN")
                page = ctx.new_page()
                for n in shoot(page, f"{base}/mobile", f"{app['key']}_mobile",
                               MOBILE_VIEWPORT):
                    print(f"      ✓ {n}")
                    total += 1
                ctx.close()
            browser.close()
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("  đã tắt toàn bộ tiến trình API")

    print(f"✓ chụp được {total} ảnh vào {SHOTS}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
