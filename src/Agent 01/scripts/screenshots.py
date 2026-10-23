"""Chụp màn hình notebook đang mở trong JupyterLab (copy screen code / copy screen output).

Cần JupyterLab chạy ở cổng 8899 trong thư mục notebooks/, không token, windowingMode "none":
    python -m jupyterlab --no-browser --port 8899 --IdentityProvider.token='' --LabApp.user_settings_dir=<thư mục>
    python scripts/screenshots.py
Ảnh ghi vào Report/Agent 01/assets/nb/.
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "Report" / "Agent 01" / "assets" / "nb"
URL = "http://localhost:8899/lab/tree/agent01_rl_qlearning.ipynb"

# chỉ số ô trong notebook (đếm cả ô markdown) -> (tên ảnh, phần chụp)
SHOTS = {
    2: ("code_imports", "input"),
    4: ("code_maze", "input"),
    6: ("code_qlearning", "input"),
    12: ("code_viz", "input"),
    8: ("out_train", "cell"),
    10: ("out_path", "cell"),
    13: ("out_plot", "cell"),
    15: ("out_qtable", "cell"),
    17: ("out_save", "cell"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        # bản Chromium Playwright đã tải sẵn trên máy (bản mặc định của gói hiện tại chưa tải)
        exe = next(Path.home().glob("AppData/Local/ms-playwright/chromium-*/chrome-win64/chrome.exe"), None)
        browser = p.chromium.launch(executable_path=str(exe) if exe else None)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page.goto(URL)
        page.wait_for_selector(".jp-NotebookPanel .jp-Cell", timeout=60000)
        page.wait_for_function("document.querySelectorAll('.jp-NotebookPanel .jp-Cell').length >= 18")
        page.wait_for_selector("text=Python 3 (ipykernel) | Idle", timeout=60000)
        time.sleep(2)  # chờ phông chữ ổn định
        page.mouse.click(1590, 990)
        page.screenshot(path=str(OUT / "nb_window.png"))
        print("-> nb_window.png")
        # Ảnh từng ô: ẩn thanh bên trái và thu hẹp cửa sổ vừa đủ nội dung rộng nhất của ô, để khi đặt vào báo cáo
        # (rộng 16 cm) chữ to nhất có thể. Khung nhìn cao để CodeMirror vẽ đủ mọi dòng của ô dài.
        page.keyboard.press("Control+b")
        cells = page.locator(".jp-NotebookPanel .jp-Cell")
        for idx, (name, part) in SHOTS.items():
            cell = cells.nth(idx)
            for _ in range(2):  # đo lại sau khi đổi bề rộng vì bố cục chảy lại
                page.set_viewport_size({"width": fit_width(page, idx, part), "height": 3200})
                time.sleep(0.8)
            target = cell.locator(".jp-Cell-inputWrapper") if part == "input" else cell
            target.scroll_into_view_if_needed()
            time.sleep(0.8)
            target.screenshot(path=str(OUT / f"{name}.png"))
            print(f"-> {name}.png")
        browser.close()


MEASURE = """([idx, part]) => {
  const cell = document.querySelectorAll('.jp-NotebookPanel .jp-Cell')[idx];
  // bề rộng chữ thật (Range), không phải bề rộng khung: .cm-content và <pre> giãn theo khung chứa
  const textRight = el => { const r = document.createRange(); r.selectNodeContents(el); return r.getBoundingClientRect().right; };
  let right = 0;
  for (const line of cell.querySelectorAll('.cm-line')) right = Math.max(right, textRight(line));
  if (part === 'cell') {
    for (const el of cell.querySelectorAll('.jp-OutputArea-output pre')) right = Math.max(right, textRight(el));
    for (const el of cell.querySelectorAll('.jp-OutputArea-output table, .jp-OutputArea-output img'))
      right = Math.max(right, el.getBoundingClientRect().right);
  }
  return {right, margin: window.innerWidth - cell.getBoundingClientRect().right};
}"""


def split_tall(path, ratio=1.15):
    """Ảnh cao hơn ratio × rộng bị thu nhỏ theo chiều cao trang; cắt thành <tên>_1.png, <tên>_2.png... tại hàng ít
    điểm tối nhất (khe giữa hai dòng mã) gần vị trí cắt dự kiến. Trả về số phần."""
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    n = int(np.ceil(h / (w * ratio)))
    for old in path.parent.glob(f"{path.stem}_*.png"):
        old.unlink()
    if n <= 1:
        return 1
    dark = (np.asarray(im)[:, int(0.08 * w):, :].min(axis=2) < 160).sum(axis=1)
    cuts = [0]
    for k in range(1, n):
        target = k * h // n
        window = range(max(cuts[-1] + 50, target - 60), min(h - 50, target + 60))
        cuts.append(min(window, key=lambda y: (dark[y], abs(y - target))))
    cuts.append(h)
    for k in range(n):
        im.crop((0, cuts[k], w, cuts[k + 1])).save(path.parent / f"{path.stem}_{k + 1}.png")
    return n


def fit_width(page, idx, part):
    m = page.evaluate(MEASURE, [idx, part])
    # trần 1500 px: dòng in ra dài hơn (lộ trình 25 tọa độ) tự xuống dòng trong ô kết quả
    return int(min(1500, max(900, m["right"] + m["margin"] + 40)))


if __name__ == "__main__":
    import sys
    if "--split-only" not in sys.argv:
        main()
    for name, _ in SHOTS.values():
        parts = split_tall(OUT / f"{name}.png")
        if parts > 1:
            print(f"{name}: tách {parts} phần")
