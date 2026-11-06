"""Phục vụ phân cấp võng mạc: nạp lười CNN đã chọn, suy luận CPU kèm Grad-CAM dạng PNG base64."""
import base64
import io
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError

from .retina import LABELS_VI, MDIR, SIZE, build, cam_layer, preprocess, to_input


@lru_cache(maxsize=1)
def _model():
    import torch
    sel = MDIR / "selected.txt"
    if not sel.exists():
        raise FileNotFoundError("Chưa có mô hình võng mạc: chạy scripts/ml_retina.py")
    name = sel.read_text(encoding="utf-8").strip()
    net = build(name, pretrained=False)
    net.load_state_dict(torch.load(MDIR / f"{name}.pt", map_location="cpu", weights_only=True))
    return net.eval(), name


def _overlay(rgb, cam):
    """Phủ bản đồ nhiệt lên ảnh đã tiền xử lý."""
    import matplotlib
    cmap = matplotlib.colormaps["jet"]
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)) / 255
    heat = cmap(c)[..., :3]
    alpha = 0.45 * c[..., None]
    out = ((1 - alpha) * (rgb / 255) + alpha * heat) * 255
    buf = io.BytesIO()
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grade(image_bytes):
    """Trả cấp 0–4, nhãn tiếng Việt, xác suất 5 lớp và Grad-CAM của lớp dự đoán."""
    import torch
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Tệp không phải ảnh hợp lệ: {e}") from e
    rgb = preprocess(img)
    net, name = _model()
    store = {}
    layer = cam_layer(net, name)
    h1 = layer.register_forward_hook(lambda _m, _i, o: store.update(a=o))
    h2 = layer.register_full_backward_hook(lambda _m, _i, o: store.update(g=o[0]))
    try:
        x = to_input(torch.from_numpy(rgb.copy())[None], "cpu").requires_grad_(True)
        net.zero_grad(set_to_none=True)
        logits = net(x)
        probs = torch.softmax(logits, 1)[0].detach().numpy()
        k = int(probs.argmax())
        logits[0, k].backward()
    finally:
        h1.remove()
        h2.remove()
    w = store["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(1))[0].detach().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return {"grade": k, "label_vi": LABELS_VI[k], "probs": [float(v) for v in probs],
            "gradcam_png_b64": _overlay(rgb, cam)}
