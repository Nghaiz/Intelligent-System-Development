"""Chụp giao diện Web App cho báo cáo (Hình 5.x) bằng Playwright.

Chạy app trước (python app/app.py), rồi:  python app/screenshots.py
Cần gói playwright và trình duyệt Chromium (pip install playwright && playwright install chromium).
Ảnh ghi vào Report/Assignment 06/assets/ui/; kịch bản dừng ngay nếu trang có lỗi JavaScript.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8080/"
OUT = Path(__file__).resolve().parents[3] / "Report" / "Assignment 06" / "assets" / "ui"
WAIT_MS = 1500          # đủ cho hiệu ứng số chạy (0,7 s) và thanh xác suất (0,8 s) dừng hẳn


def open_page(browser, theme, width=1440, height=900):
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(URL)
    page.evaluate(f"localStorage.setItem('a06-theme', '{theme}')")
    page.reload()
    page.wait_for_selector(".model")                 # /api/meta đã về và danh sách mô hình đã dựng
    page.wait_for_timeout(800)
    return page, errors


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page, errors = open_page(browser, "dark")
        page.screenshot(path=OUT / "hero.png", clip={"x": 0, "y": 0, "width": 1440, "height": 470})
        page.click("#s-run"); page.wait_for_timeout(WAIT_MS)
        page.click("#s-compare"); page.wait_for_timeout(WAIT_MS)
        page.locator("#stock").screenshot(path=OUT / "stock.png")

        page.click('.tab-btn[data-tab="churn"]'); page.wait_for_timeout(600)
        page.click("#c-run"); page.wait_for_timeout(WAIT_MS)
        page.click("#c-compare"); page.wait_for_timeout(WAIT_MS)
        page.locator("#churn").screenshot(path=OUT / "churn.png")

        page.click('.chip[data-src="sim"]'); page.wait_for_timeout(400)
        page.fill("#r-stop", "12"); page.dispatch_event("#r-stop", "input")
        page.wait_for_timeout(WAIT_MS)                  # dự báo tự chạy lại sau 0,3 s khi ngừng kéo
        page.click("#c-compare"); page.wait_for_timeout(WAIT_MS)
        page.locator("#churn .grid").screenshot(path=OUT / "churn_sim.png")

        light, errors_light = open_page(browser, "light")
        light.click("#s-run"); light.wait_for_timeout(WAIT_MS)
        light.locator("#stock .grid").screenshot(path=OUT / "stock_light.png")

        browser.close()
    all_errors = errors + errors_light
    if all_errors:
        sys.exit(f"trang có lỗi JavaScript: {all_errors}")
    print("đã chụp:", " ".join(sorted(f.name for f in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
