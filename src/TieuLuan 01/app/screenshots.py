"""Chụp giao diện sáu phân hệ Web App cho báo cáo bằng Playwright (Chromium headless).

Chạy app trước (python app/app.py), rồi:  python app/screenshots.py
Ảnh ghi vào Report/TieuLuan 01/assets/ui/; kịch bản dừng nếu trang có lỗi JavaScript.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8090/"
OUT = Path(__file__).resolve().parents[3] / "Report" / "TieuLuan 01" / "assets" / "ui"
WAIT = 1200


def shot(page, tab, button, name, before=None):
    page.click(f'.tab[data-tab="{tab}"]')
    page.wait_for_timeout(300)
    if before:
        before(page)
    page.click(button)
    page.wait_for_timeout(WAIT)
    page.locator(f"#{tab}").screenshot(path=OUT / f"{name}.png")
    print("->", name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:                      # bản trình duyệt khớp phiên bản playwright chưa tải: dùng bản có sẵn
            exe = sorted(Path.home().glob("AppData/Local/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell.exe"))
            if not exe:
                raise
            browser = p.chromium.launch(executable_path=str(exe[-1]))
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(URL)
        page.wait_for_timeout(500)
        shot(page, "diabetes", "#d-run", "ui_diabetes")
        shot(page, "house", "#h-run", "ui_house")
        shot(page, "sentiment", "#s-run", "ui_sentiment")
        shot(page, "vision", "#v-run", "ui_vision")

        def svhn(pg):
            pg.select_option("#v-ds", "svhn")
            pg.select_option("#v-arch", "mobilenetv2")
            pg.fill("#v-idx", "42")
        shot(page, "vision", "#v-run", "ui_vision_svhn", svhn)

        def days(pg):
            pg.fill("#st-days", "180")
            pg.dispatch_event("#st-days", "input")
        shot(page, "stock", "#st-run", "ui_stock", days)
        shot(page, "churn", "#c-run", "ui_churn", lambda pg: pg.click('#c-group button[data-g="churn"]'))
        browser.close()
    if errors:
        sys.exit(f"trang có lỗi JavaScript: {errors}")


if __name__ == "__main__":
    main()
