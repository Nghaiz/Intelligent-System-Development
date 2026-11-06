"""Phân cấp bệnh võng mạc 5 mức: tiền xử lý ảnh đáy mắt, mô hình ImageNet, tăng cường trên GPU, chỉ số và Grad-CAM."""
import copy

import numpy as np
from PIL import Image

from ..common import DATA, MODELS

SIZE = 224
RDIR = DATA / "retina"
MDIR = MODELS / "retina"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LABELS_VI = ["Không có dấu hiệu bệnh võng mạc", "Bệnh võng mạc không tăng sinh mức nhẹ",
             "Bệnh võng mạc không tăng sinh mức trung bình", "Bệnh võng mạc không tăng sinh mức nặng",
             "Bệnh võng mạc tăng sinh"]
SHORT_VI = ["Không bệnh", "Nhẹ", "Trung bình", "Nặng", "Tăng sinh"]
CFG = {"epochs": 15, "patience": 5, "lr": 2e-4, "wd": 1e-4, "bs": 48, "samples_per_epoch": 12000}


def preprocess(img):
    """Cắt viền đen, co cạnh dài về SIZE, đệm đen thành vuông; trả uint8 HWC. Huấn luyện và phục vụ dùng chung."""
    img = img.convert("RGB")
    g = np.asarray(img.convert("L").resize((128, 128))) > 12
    if g.any():
        ys, xs = np.where(g.any(1))[0], np.where(g.any(0))[0]
        sx, sy = img.width / 128, img.height / 128
        img = img.crop((int(xs[0] * sx), int(ys[0] * sy), int((xs[-1] + 1) * sx), int((ys[-1] + 1) * sy)))
    s = SIZE / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
    out = Image.new("RGB", (SIZE, SIZE))
    out.paste(img, ((SIZE - img.width) // 2, (SIZE - img.height) // 2))
    return np.asarray(out, dtype=np.uint8)


def build(name, pretrained=True):
    from torch import nn
    from torchvision import models as M
    if name == "resnet34":
        net = M.resnet34(weights=M.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, 5)
    elif name == "efficientnet_b0":
        net = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 5)
    else:
        raise ValueError(f"Mô hình lạ: {name}")
    return net


def cam_layer(net, name):
    return net.layer4[-1] if name == "resnet34" else net.features[-1]


def to_input(x_uint8, dev):
    """Lô NHWC uint8 thành NCHW đã chuẩn hóa ImageNet."""
    import torch
    x = x_uint8.to(dev).permute(0, 3, 1, 2).float() / 255
    m = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - m) / s


def augment(x):
    """Lật, xoay ±25°, phóng ±10 %, sáng/tương phản theo từng ảnh, chạy trên GPU nên không cần worker nạp dữ liệu."""
    import torch
    import torch.nn.functional as F
    n, dev = len(x), x.device
    x = x.permute(0, 3, 1, 2).float() / 255
    a = (torch.rand(n, device=dev) - 0.5) * 2 * np.deg2rad(25)
    z = 1 + (torch.rand(n, device=dev) - 0.5) * 0.2
    sx = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    sy = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(a) * z, torch.sin(a) * z
    zero = torch.zeros(n, device=dev)
    th = torch.stack([torch.stack([cos * sx, -sin, zero], 1), torch.stack([sin, cos * sy, zero], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False, padding_mode="zeros")
    x = x * (0.85 + 0.3 * torch.rand(n, 1, 1, 1, device=dev))
    mean = x.mean((2, 3), keepdim=True)
    x = (x - mean) * (0.85 + 0.3 * torch.rand(n, 1, 1, 1, device=dev)) + mean
    m = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x.clamp(0, 1) - m) / s


def metrics(y, pred):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "per_class_f1": [float(v) for v in f1_score(y, pred, average=None, labels=range(5))],
            "confusion": confusion_matrix(y, pred, labels=range(5)).tolist(), "n": len(y)}


def infer(net, X, dev, bs=128):
    """Trả xác suất softmax cho cả mảng ảnh uint8."""
    import torch
    net.eval()
    out = np.zeros((len(X), 5), dtype=np.float32)
    buf = np.empty((bs, SIZE, SIZE, 3), dtype=np.uint8)  # đệm dùng lại: commit cạn thì cấp phát mỗi lô sẽ sập
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev == "cuda"):
        for i in range(0, len(X), bs):
            n = min(bs, len(X) - i)
            buf[:n] = X[i:i + n]
            b = torch.from_numpy(buf[:n])
            out[i:i + n] = torch.softmax(net(to_input(b, dev)).float(), 1).cpu().numpy()
    return out


def fit(name, X, y, Xv, yv, dev, seed=42):
    """AdamW + cosine; mẫu theo trọng số 1/căn(tần suất), lưu trọng số tốt nhất theo QWK trên val."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    net = build(name).to(dev)
    c = CFG
    steps = c["samples_per_epoch"] // c["bs"]
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps * c["epochs"])
    crit, scaler = nn.CrossEntropyLoss(), torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    w = 1 / np.sqrt(np.bincount(y, minlength=5))[y]
    w /= w.sum()
    best, keep, wait, hist = -2.0, None, 0, []
    buf = np.empty((c["bs"], SIZE, SIZE, 3), dtype=np.uint8)
    ck, start = MDIR / f"{name}_ckpt.pt", 1
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sch.load_state_dict(st["sch"])
        best, keep, wait, hist, start = st["best"], st["keep"], st["wait"], st["hist"], st["epoch"] + 1
        print("tiếp tục từ epoch", start, flush=True)
    for ep in range(start, c["epochs"] + 1):
        rng = np.random.default_rng(seed + ep)  # theo epoch để tiếp tục sau sập vẫn tái lập được
        net.train()
        # Sắp chỉ số để đọc memmap theo thứ tự đĩa, sau đó xáo trong lô ở mức toàn epoch
        idx = rng.choice(len(y), steps * c["bs"], p=w)
        total = 0.0
        for s in range(steps):
            b = np.sort(idx[s * c["bs"]:(s + 1) * c["bs"]])
            for k, j in enumerate(b):
                buf[k] = X[j]  # gán từng ảnh: chỉ mục mảng trên memmap sẽ cấp phát mới
            xb = torch.from_numpy(buf).to(dev)
            yb = torch.as_tensor(y[b], device=dev, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                loss = crit(net(augment(xb)), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            total += loss.item()
        pv = infer(net, Xv, dev)
        vm = metrics(yv, pv.argmax(1))
        vl = float(-np.log(pv[np.arange(len(yv)), yv] + 1e-9).mean())
        hist.append({"epoch": ep, "train_loss": total / steps, "val_loss": vl, "val_qwk": vm["qwk"],
                     "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"]})
        print(name, hist[-1], flush=True)
        if vm["qwk"] > best + 1e-4:
            best, keep, wait = vm["qwk"], copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "best": best,
                    "keep": keep, "wait": wait, "hist": hist, "epoch": ep}, ck)
        if wait >= c["patience"]:
            break
    net.load_state_dict(keep)
    ck.unlink(missing_ok=True)
    return net, hist
