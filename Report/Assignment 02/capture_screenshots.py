"""Capture report screenshots for the three ML demo apps (diabetes, house_price,
customer_behavior) — web + mobile, input-filled and result states.

Starts all three Flask REST APIs as background subprocesses, waits for each
`/health` endpoint to answer 200, then drives headless Chrome (via Playwright,
using the system Chrome install) to fill each app's form, submit it, and
screenshot both the "filled in" and "result" states for web and mobile layouts.

Run:
    d:\\Python\\HTTM_Assignment02\\.venv\\Scripts\\python.exe report/capture_screenshots.py
"""

import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2] / "src" / "Assignment 02"
PYTHON = sys.executable
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

APPS = [
    {
        "name": "diabetes",
        "script": ROOT / "diabetes" / "api" / "REST_API.py",
        "port": 5001,
    },
    {
        "name": "house_price",
        "script": ROOT / "house_price" / "api" / "REST_API.py",
        "port": 5002,
    },
    {
        "name": "customer_behavior",
        "script": ROOT / "customer_behavior" / "api" / "REST_API.py",
        "port": 5003,
    },
]

WEB_VIEWPORT = {"width": 1280, "height": 1000}
MOBILE_VIEWPORT = {"width": 700, "height": 1000}
DEVICE_SCALE = 2


def wait_for_health(port, timeout=60):
    """Poll /health until it returns HTTP 200 or timeout elapses."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


def start_servers():
    """Start all three Flask APIs as background subprocesses."""
    processes = []
    for app in APPS:
        log_path = Path(__file__).resolve().parent / f"_server_{app['name']}.log"
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [PYTHON, str(app["script"])],
            cwd=str(app["script"].parent),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append({"name": app["name"], "proc": proc, "port": app["port"], "log": log_file})
        print(f"[start] {app['name']} pid={proc.pid} port={app['port']}")
    return processes


def stop_servers(processes):
    for entry in processes:
        proc = entry["proc"]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=8)
        entry["log"].close()
        print(f"[stop] {entry['name']} pid={proc.pid}")


# ---------------------------------------------------------------------------
# Per-app fill helpers
# ---------------------------------------------------------------------------

def fill_diabetes(page):
    page.fill("#Glucose", "180")
    page.fill("#BMI", "40")
    page.fill("#Age", "55")
    page.fill("#Pregnancies", "7")
    page.fill("#DiabetesPedigreeFunction", "1.1")


def fill_house_price(page, mobile=False):
    if mobile:
        page.fill("#Area", "85")
        page.fill("#Frontage", "5")
        page.fill("#AccessRoad", "8")
        page.fill("#Floors", "4")
        page.fill("#Bedrooms", "4")
        page.fill("#Bathrooms", "3")
        page.select_option("#LegalStatus", "Have certificate")
        page.select_option("#FurnitureState", "Full")
        page.select_option("#Province", "Hà Nội")
    else:
        page.fill('[name="Area"]', "85")
        page.fill('[name="Frontage"]', "5")
        page.fill('[name="Access Road"]', "8")
        page.fill('[name="Floors"]', "4")
        page.fill('[name="Bedrooms"]', "4")
        page.fill('[name="Bathrooms"]', "3")
        page.select_option('[name="Legal status"]', "Have certificate")
        page.select_option('[name="Furniture state"]', "Full")
        page.select_option('[name="Province"]', "Hà Nội")


NEGATIVE_REVIEW = (
    "Very disappointed. The material feels cheap and thin, it runs two sizes "
    "too small and looks nothing like the picture. Returned it."
)


def fill_customer_behavior(page, mobile=False):
    if mobile:
        page.fill("#ReviewText", NEGATIVE_REVIEW)
        page.fill("#Age", "45")
        page.fill("#PositiveFeedbackCount", "0")
    else:
        page.fill('[name="Review Text"]', NEGATIVE_REVIEW)
        page.fill('[name="Age"]', "45")
        page.fill('[name="Positive Feedback Count"]', "0")


# ---------------------------------------------------------------------------
# Capture routine
# ---------------------------------------------------------------------------

def capture_model_table(page, short):
    """Capture the #compare-table element and save it as a separate image."""
    try:
        # Scroll the table into view
        page.locator("#compare-table").scroll_into_view_if_needed()
        page.wait_for_timeout(200)

        # Take screenshot of the element
        models_path = FIGURES_DIR / f"shot_{short}_web_models.png"
        page.locator("#compare-table").screenshot(path=str(models_path))
        return models_path
    except Exception as e:
        print(f"[warn] could not capture model table: {e}")
        return None


def capture_app(browser, base_url, app_name, fill_fn, submit_selector, mobile, port_label):
    viewport = MOBILE_VIEWPORT if mobile else WEB_VIEWPORT
    context = browser.new_context(viewport=viewport, device_scale_factor=DEVICE_SCALE)
    page = context.new_page()

    path = "/mobile" if mobile else "/web"
    page.goto(f"{base_url}{path}", wait_until="networkidle")
    # Let async metadata fetch populate <select> options before we touch them.
    page.wait_for_timeout(1200)

    fill_fn(page)

    tag = "mobile" if mobile else "web"
    short = {"diabetes": "dia", "house_price": "hou", "customer_behavior": "ecom"}[app_name]

    input_path = FIGURES_DIR / f"shot_{short}_{tag}_input.png"
    page.screenshot(path=str(input_path), full_page=False)
    print(f"[shot] {input_path.name}")

    page.click(submit_selector)
    page.wait_for_timeout(1800)

    if mobile:
        # The phone screen scrolls internally (#app-content); scroll the
        # result panel into view so the prediction + knowledge panel show.
        page.eval_on_selector(
            "#result-body",
            "el => el.scrollIntoView({block: 'center'})",
        )
        page.wait_for_timeout(300)

    result_path = FIGURES_DIR / f"shot_{short}_{tag}_result.png"
    page.screenshot(path=str(result_path), full_page=False)
    print(f"[shot] {result_path.name}")

    # For web only: capture the model comparison table separately
    if not mobile:
        models_path = capture_model_table(page, short)
        if models_path:
            print(f"[shot] {models_path.name}")
        page.wait_for_timeout(300)

    context.close()
    return (input_path, result_path) if mobile else (input_path, result_path, models_path)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    processes = start_servers()
    try:
        for app in APPS:
            ok = wait_for_health(app["port"])
            if not ok:
                raise RuntimeError(f"{app['name']} did not become healthy on port {app['port']} in time")
            print(f"[health] {app['name']} OK on port {app['port']}")

        produced = []
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)

            # diabetes — web + mobile
            result = capture_app(
                browser, "http://127.0.0.1:5001", "diabetes", fill_diabetes,
                "#submit-btn", mobile=False, port_label=5001,
            )
            produced.extend([r for r in result if r])

            produced += capture_app(
                browser, "http://127.0.0.1:5001", "diabetes", fill_diabetes,
                "#submit-btn", mobile=True, port_label=5001,
            )

            # house_price — web + mobile
            result = capture_app(
                browser, "http://127.0.0.1:5002", "house_price",
                lambda pg: fill_house_price(pg, mobile=False),
                "#submit-btn", mobile=False, port_label=5002,
            )
            produced.extend([r for r in result if r])

            produced += capture_app(
                browser, "http://127.0.0.1:5002", "house_price",
                lambda pg: fill_house_price(pg, mobile=True),
                "#submit-btn", mobile=True, port_label=5002,
            )

            # customer_behavior — web + mobile
            result = capture_app(
                browser, "http://127.0.0.1:5003", "customer_behavior",
                lambda pg: fill_customer_behavior(pg, mobile=False),
                "#submit-btn", mobile=False, port_label=5003,
            )
            produced.extend([r for r in result if r])

            produced += capture_app(
                browser, "http://127.0.0.1:5003", "customer_behavior",
                lambda pg: fill_customer_behavior(pg, mobile=True),
                "#submit-btn", mobile=True, port_label=5003,
            )

            browser.close()

        print(f"[done] captured {len(produced)} screenshots")
    finally:
        stop_servers(processes)


if __name__ == "__main__":
    main()
