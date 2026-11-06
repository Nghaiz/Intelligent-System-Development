"""Chương 3 — suy luận cho Web App: phân loại ảnh tải lên, trả xác suất top-k và Grad-CAM dạng PNG base64.

Mô hình nạp lười từ models/ch3/, nạp xong giữ lại, luôn chạy CPU để không tranh GPU với LLM.
"""
import base64
import binascii
import io
import time
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .ch3_common import ARCHS, ARCH_NAME, DS_IDS, EXAMPLES, MODELS
from .ch3_data import DATASETS, labels, net_input

MAX_BYTES = 8 * 2**20
MAX_PIXELS = 40_000_000
TOP_K = 5
SHOW = 224


def model_options(ds):
    return [{"value": a, "label": ARCH_NAME[a]} for a in ARCHS if (MODELS / f"{ds}__{a}.pt").exists()]


def examples(ds):
    return sorted(p.name for p in EXAMPLES.glob(f"{ds}_*.png"))


def _spec(ds):
    d = DATASETS[ds]
    return {"id": f"ch3_{ds}", "title": f"Phân loại ảnh {d['name']}", "dataset": ds, "output": "probs",
            "inputs": [{"name": "image", "type": "image", "label": "Ảnh tải lên (PNG hoặc JPEG)"},
                       {"name": "example", "type": "select", "label": "Hoặc chọn ảnh mẫu thật",
                        "default": next(iter(examples(ds)), None), "options": examples(ds)},
                       {"name": "model", "type": "select", "label": "Kiến trúc", "default": "resnet18",
                        "options": model_options(ds)}],
            "classes": labels(ds), "note": d["task"]}


SPECS = [_spec(ds) for ds in DS_IDS]


@lru_cache(maxsize=None)
def _load(ds, arch):
    """Nạp trọng số một lần cho mỗi cặp (bộ dữ liệu, kiến trúc); weights_only chặn pickle tùy ý."""
    from .ch3_models import build
    from .ch3_torch import GradCAM, checkpoint_model
    path = MODELS / f"{ds}__{arch}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có mô hình {arch} cho {ds}: {path.name}")
    model, mean, std = checkpoint_model(path, lambda kw: build(arch, len(DATASETS[ds]["classes"]), **kw))
    return model, mean, std, GradCAM(model)


def _decode(raw):
    """Nhận bytes, base64 hoặc data URL từ Web App; ranh giới hệ thống nên kiểm tra cỡ tệp và cỡ ảnh."""
    if isinstance(raw, str):
        try:
            raw = base64.b64decode(raw.split(",", 1)[-1], validate=True)
        except binascii.Error as e:
            raise ValueError("Ảnh không phải base64 hợp lệ") from e
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError("Thiếu dữ liệu ảnh")
    if len(raw) > MAX_BYTES:
        raise ValueError(f"Ảnh vượt {MAX_BYTES // 2**20} MB")
    try:
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > MAX_PIXELS:
            raise ValueError("Ảnh quá lớn")
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ValueError(f"Không đọc được ảnh: {e}") from e


def _pick_image(ds, payload):
    """Ảnh tải lên được ưu tiên; ảnh mẫu chỉ chọn theo tên trong danh sách, không nhận đường dẫn tự do."""
    if payload.get("image"):
        return _decode(payload["image"])
    name = payload.get("example") or next(iter(examples(ds)), None)
    if name not in examples(ds):
        raise ValueError(f"Ảnh mẫu không hợp lệ: {name}")
    return Image.open(EXAMPLES / name).convert("RGB")


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def heat_overlay(rgb, cam, alpha=0.5):
    """Trộn ảnh uint8 với bản đồ nhiệt [0, 1] tô theo thang xanh-vàng-đỏ; phóng to bằng SHOW×SHOW điểm ảnh."""
    big = np.asarray(Image.fromarray(rgb).resize((SHOW, SHOW), Image.BICUBIC), np.float32)
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((SHOW, SHOW), Image.BICUBIC),
                   np.float32) / 255
    heat = np.stack([np.clip(1.5 - abs(4 * c - k), 0, 1) for k in (3, 2, 1)], 2) * 255
    return ((1 - alpha) * big + alpha * heat).clip(0, 255).astype(np.uint8)


def predict(id, payload):
    """payload: {image: bytes|base64|data URL} hoặc {example: tên}, thêm model. Trả top-k và Grad-CAM."""
    import torch
    ds = str(id).removeprefix("ch3_")
    if ds not in DATASETS:
        raise KeyError(f"id không hợp lệ: {id}")
    arch = payload.get("model") or "resnet18"
    if arch not in ARCHS:
        raise ValueError(f"Kiến trúc không hợp lệ: {arch}")
    t0 = time.perf_counter()
    model, mean, std, cam = _load(ds, arch)
    size = DATASETS[ds]["size"]
    rgb = np.asarray(ImageOps.fit(_pick_image(ds, payload), (size, size), Image.LANCZOS), np.uint8)
    x = (torch.from_numpy(net_input(rgb[None]).copy()).permute(0, 3, 1, 2).float() / 255 - mean) / std
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].numpy()
    heat = cam(x)[0][0]
    if size == 28:                                  # bỏ phần đệm phản chiếu, chỉ phủ lên ảnh gốc
        heat = heat[2:-2, 2:-2]
    names, order = labels(ds), np.argsort(p)[::-1][:TOP_K]
    top = [{"label": names[int(i)], "prob": float(p[i])} for i in order]
    return {"label": top[0]["label"], "probs": {t["label"]: t["prob"] for t in top}, "top_k": top,
            "gradcam_png": _png(heat_overlay(rgb, heat)),
            "input_png": _png(np.asarray(Image.fromarray(rgb).resize((SHOW, SHOW), Image.NEAREST))),
            "model": ARCH_NAME[arch], "dataset": DATASETS[ds]["name"], "input_size": f"{size}×{size}",
            "latency_ms": 1000 * (time.perf_counter() - t0)}
