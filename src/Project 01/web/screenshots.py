"""Kiểm mọi trang bằng Playwright và chụp ảnh báo cáo. Máy chủ phải đang chạy ở 127.0.0.1:8097.

Từ src/Project 01:  python web/screenshots.py          kiểm 1440×900, 390×844, chụp vào Report/Project 01/figures/web_*.png
                    python web/screenshots.py --only risk,retina    chỉ chạy một số trang
Không chụp đè ảnh của trang còn đang chuẩn bị: trang chưa có dữ liệu thật thì báo PENDING và không ghi tệp web_*.png.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "D:/DevCache/playwright")

from playwright.sync_api import sync_playwright  # noqa: E402

BASE = "http://127.0.0.1:8097"
FIGS = ROOT.parents[1] / "Report" / "Project 01" / "figures"
OUT = ROOT / "outputs" / "metrics" / "web_check.json"
PAGES = ["overview", "consult", "screen", "risk", "retina", "graph", "search", "data", "lab", "logs"]
PATHS = {"overview": "/"}
SETTLE = 900
QUESTION = "HbA1c 7,2% và đường huyết đói 140 mg/dL có đáng lo không?"
LLM_TIMEOUT_MS = 240_000
PROFILE = {"age": "54", "bmi": "29.5", "hba1c": "7.2", "glucose": "140"}
MOBILE_DIR = None
SEARCH_MODES = ("bm25", "dense", "graph", "hybrid", "hybrid_rerank")


def path_of(name):
    return PATHS.get(name, f"/{name}")


HEAVY = ("consult", "screen", "risk", "retina", "search")  # các trang nạp mô hình: chỉ kiểm khi có --heavy


def expected_noise(text):
    """503 của phần đang chuẩn bị là trạng thái hợp đồng cho phép, không phải lỗi trang."""
    return "503" in text


def has_pending(page):
    return page.locator(".pending").count() > 0


# Ca thao tác thật: mỗi hàm trả False nếu trang chưa có dữ liệu thật để thao tác
def act_consult(page):
    if not page.request.get(f"{BASE}/api/status").json()["modules"]["dnx.agents.graph"]["ready"]:
        return False
    page.locator("summary").click()  # hồ sơ có xét nghiệm để thẻ nguy cơ và bộ luật cùng chạy
    for key, value in PROFILE.items():
        page.fill(f'[data-p="{key}"]', value)
    page.fill("#q", QUESTION)
    page.click("#send")
    page.wait_for_selector(".feedback, .msg.bot .pending", timeout=LLM_TIMEOUT_MS)
    page.wait_for_timeout(1500)
    return page.locator(".feedback").count() > 0


def act_screen(page):
    for _ in range(21):
        if page.locator("#again").count():
            return True
        if page.locator(".opts button").count() == 0:
            return False
        page.locator(".opts button").first.click()
        page.wait_for_timeout(350)
    return page.locator("#again").count() > 0


def act_risk(page):
    for _ in range(120):  # lần nạp đầu có thể chờ slot bộ nhớ; CSP chặn wait_for_function dạng chuỗi nên hỏi vòng
        if page.locator("#now").inner_text() != "—" or has_pending(page):
            break
        page.wait_for_timeout(1000)
    if has_pending(page):
        return False
    page.locator('input[data-g="hba1c"]').fill("5.6")
    page.locator('input[data-g="bmi"]').fill("24")
    page.wait_for_timeout(900)
    return not has_pending(page)


def act_retina(page):
    btn = page.locator(".samples button")
    if btn.count() == 0:
        return False
    btn.nth(min(2, btn.count() - 1)).click()
    page.wait_for_selector("#cc, #out .pending", timeout=60_000)
    return page.locator("#cc").count() > 0


def act_graph(page):
    page.wait_for_selector("#kg canvas, #kg .pending", timeout=30_000)
    if page.locator("#kg canvas").count() == 0:
        return False
    page.wait_for_timeout(8500)  # lực đồ thị dừng sau 6 giây, rồi khung hình tự vừa toàn đồ thị
    return True


def act_search(page):
    for _ in range(240):  # năm cột chạy lần lượt ở máy chủ; thiếu commit thì cột báo chờ rồi trang tự thử lại
        done = sum(page.locator(f'[data-m="{m}"] .hit').count() > 0 for m in SEARCH_MODES)
        if done == len(SEARCH_MODES):
            return True
        page.wait_for_timeout(1000)
    return False


def act_data(page):
    page.wait_for_selector("#hist svg, .pending", timeout=30_000)
    return page.locator("#hist svg").count() > 0


def act_lab(page):
    return page.locator("#bars svg").count() > 0


def act_logs(page):
    page.wait_for_selector("[data-s], #sess .pending", timeout=15_000)
    if page.locator("[data-s]").count() == 0:
        return False
    page.locator("[data-s]").first.click()
    page.wait_for_timeout(6000)  # đủ để phát lại hết vết
    page.locator("#strip").scroll_into_view_if_needed()
    return page.locator("#strip .tick").count() > 0


ACTS = {"logs": act_logs, "consult": act_consult, "screen": act_screen, "risk": act_risk, "retina": act_retina, "graph": act_graph,
        "search": act_search, "data": act_data, "lab": act_lab}


def overflow(page):
    return page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")


def visit(ctx, name, shot):
    page = ctx.new_page()
    problems = []
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" and not expected_noise(m.text) else None)
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.goto(BASE + path_of(name), wait_until="domcontentloaded")
    page.wait_for_selector(".page-head h1")
    page.wait_for_timeout(SETTLE)
    live = ACTS[name](page) if shot and name in ACTS else not has_pending(page)
    if shot:
        page.wait_for_timeout(SETTLE)
        if live:
            FIGS.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(FIGS / f"web_{name}.png"), timeout=90_000)
    elif MOBILE_DIR:
        page.screenshot(path=str(Path(MOBILE_DIR) / f"m_{name}.png"), timeout=90_000)
    row = {"page": name, "live": bool(live), "console_errors": problems[:5], "overflow_px": overflow(page)}
    page.close()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--heavy", action="store_true", help="kiểm cả các trang nạp mô hình, tốn thêm vài GB commit")
    ap.add_argument("--out", default=None, help="thư mục ảnh thay cho figures, dùng khi thử giao diện")
    args = ap.parse_args()
    global FIGS, MOBILE_DIR
    FIGS = Path(args.out) if args.out else FIGS
    MOBILE_DIR = args.out
    names = [n for n in PAGES if n in args.only.split(",")] if args.only else [n for n in PAGES if args.heavy or n not in HEAVY]
    report = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "desktop": [], "mobile": []}
    with sync_playwright() as p:
        # Dùng Chrome đã cài trên máy, không tải thêm trình duyệt vào D:/DevCache/playwright
        browser = p.chromium.launch(channel="chrome", args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        desktop = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        mobile = browser.new_context(viewport={"width": 390, "height": 844}, color_scheme="light", device_scale_factor=2)
        for n in names:
            report["desktop"].append(visit(desktop, n, True))
            report["mobile"].append(visit(mobile, n, False))
        browser.close()
    if not args.only:
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for key in ("desktop", "mobile"):
        for r in report[key]:
            state = "LIVE" if r["live"] else "PENDING"
            print(f"{key:8} {r['page']:9} {state:8} overflow={r['overflow_px']} errors={r['console_errors']}")
    return 1 if any(r["console_errors"] or r["overflow_px"] > 0 for k in ("desktop", "mobile") for r in report[k]) else 0


if __name__ == "__main__":
    sys.exit(main())
