"""Chương 3 — huấn luyện PyTorch trên GPU có điểm lưu mỗi epoch, suy luận theo lô và Grad-CAM."""
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ch3_models import cam_layer


def to_tensor(x):
    """Lô uint8 NHWC (có thể là memmap) -> tensor uint8 NCHW; giữ uint8 để dữ liệu chiếm ít VRAM."""
    return torch.from_numpy(np.array(x)).permute(0, 3, 1, 2)


def augment(x, shift=4):
    """Tăng cường trên GPU cho PathMNIST: dịch ±4 điểm có phản chiếu ở biên, lật hai chiều, chuyển vị."""
    n, dev = x.shape[0], x.device
    t = (torch.randint(0, 2 * shift + 1, (n, 2), device=dev).float() - shift) * 2 / x.shape[2]
    eye = torch.eye(2, device=dev).expand(n, 2, 2)
    grid = F.affine_grid(torch.cat([eye, t[:, :, None]], 2), x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    for op in (lambda a: a.flip(3), lambda a: a.flip(2), lambda a: a.transpose(2, 3)):
        x = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.5, op(x), x)
    return x


@torch.no_grad()
def predict_proba(model, x, mean, std, batch=512, dev="cuda"):
    """x uint8 NHWC; đọc từng lô từ memmap để không nạp cả tập vào RAM."""
    model.eval()
    m, s = mean.to(dev), std.to(dev)
    out = []
    for i in range(0, len(x), batch):
        xb = (to_tensor(x[i:i + batch]).to(dev).float() / 255 - m) / s
        out.append(F.softmax(model(xb), 1).cpu())
    return torch.cat(out).numpy()


def channel_stats(xtr):
    """Trung bình và độ lệch chuẩn theo kênh, chỉ tính từ tập train."""
    a = np.asarray(xtr[:: max(1, len(xtr) // 20000)], np.float32) / 255
    return (torch.tensor(a.mean((0, 1, 2)), dtype=torch.float32).view(1, 3, 1, 1),
            torch.tensor(a.std((0, 1, 2)) + 1e-6, dtype=torch.float32).view(1, 3, 1, 1))


def make_optim(model, cfg, total_steps):
    """full: SGD Nesterov + OneCycle của Tiểu luận 01. plain: Adam 1e-3 để so ba nền tảng."""
    if cfg["recipe"] == "plain":
        return torch.optim.Adam(model.parameters(), lr=1e-3), None
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], nesterov=True,
                          weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total_steps,
                                                pct_start=cfg["pct_start"])
    return opt, sched


def fit_epoch(model, xtr, ytr, mean, std, cfg, opt, sched):
    """Một epoch trên tensor uint8 nằm sẵn trong VRAM; trả mất mát và độ đúng trung bình của tập train."""
    model.train()
    perm = torch.randperm(len(xtr), device=xtr.device)
    full = cfg["recipe"] == "full"
    total_loss = total_ok = 0.0
    for s in range(0, len(perm), cfg["batch"]):
        b = perm[s:s + cfg["batch"]]
        xb = (xtr[b].float() / 255 - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=full):
            out = model(augment(xb) if full else xb)
            loss = F.cross_entropy(out, ytr[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        total_loss += loss.item() * len(b)
        total_ok += (out.argmax(1) == ytr[b]).sum().item()
    return total_loss / len(perm), total_ok / len(perm)


def _save_ckpt(path, state):
    """Chỉ ghi đè một tệp điểm lưu cho mỗi lượt để không làm đầy ổ D."""
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _resume(path, model, opt, sched):
    """Khôi phục sau sự cố: trọng số, optimizer, vị trí trên lịch học và lịch sử các epoch đã xong."""
    if not path.exists():
        return {k: [] for k in ("loss", "acc", "val_loss", "val_acc", "time")}
    st = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    if sched:
        sched.load_state_dict(st["sched"])
    return st["hist"]


def train(model, data, cfg, ckpt, dev="cuda", log=print):
    """data = (xtr, ytr, xva, yva). Lưu điểm kiểm tra mỗi epoch; dùng trọng số epoch cuối như Tiểu luận 01."""
    xtr_np, ytr_np, xva, yva = data
    mean, std = channel_stats(xtr_np)
    if cfg["recipe"] == "plain":                         # plain chỉ chia 255, giống bản NumPy và Keras
        mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    model.to(dev)
    xtr, ytr = to_tensor(xtr_np).to(dev), torch.from_numpy(np.array(ytr_np)).to(dev)
    opt, sched = make_optim(model, cfg, cfg["epochs"] * math.ceil(len(xtr) / cfg["batch"]))
    hist = _resume(ckpt, model, opt, sched)
    for ep in range(len(hist["loss"]), cfg["epochs"]):
        t0 = time.perf_counter()
        tl, ta = fit_epoch(model, xtr, ytr, mean.to(dev), std.to(dev), cfg, opt, sched)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        pv = predict_proba(model, xva, mean, std, dev=dev)
        vl = float(-np.log(pv[np.arange(len(yva)), yva] + 1e-12).mean())
        va = float((pv.argmax(1) == yva).mean())
        for k, v in zip(hist, (tl, ta, vl, va, dt)):
            hist[k].append(v)
        _save_ckpt(ckpt, {"hist": hist, "model": model.state_dict(), "opt": opt.state_dict(),
                          "sched": sched.state_dict() if sched else None})
        log(f"epoch {ep + 1}/{cfg['epochs']} loss {tl:.4f} acc {ta:.4f} val_loss {vl:.4f} val_acc {va:.4f} "
            f"{dt:.1f}s")
    return hist, mean, std


def cpu_latency(model, shape, runs=30):
    """Mili giây cho một ảnh trên CPU, cùng cách đo với Tiểu luận 01 (bỏ 10 lượt đầu để làm nóng)."""
    model.cpu().eval()
    x = torch.randn(1, *shape)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
    return 1000 * (time.perf_counter() - t0) / runs


class GradCAM:
    """Grad-CAM: trọng số kênh là trung bình gradient của điểm lớp theo bản đồ đặc trưng tầng cuối,
    bản đồ nhiệt = ReLU(Σ_k α_k A_k), nội suy về kích thước ảnh và chia cho giá trị lớn nhất."""

    def __init__(self, model):
        self.model, self.acts = model.eval(), None
        cam_layer(model).register_forward_hook(lambda m, i, o: setattr(self, "acts", o))

    def __call__(self, x, cls=None):
        out = self.model(x)
        cls = out.argmax(1) if cls is None else cls
        grads = torch.autograd.grad(out.gather(1, cls.view(-1, 1)).sum(), self.acts)[0]
        alpha = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return cam[:, 0].detach().cpu().numpy(), cls.detach().cpu().numpy()


def checkpoint_model(path, build):
    """Nạp tệp .pt dạng Tiểu luận 01; build(kwargs) dựng mô hình đúng tham số đã lưu."""
    st = torch.load(path, map_location="cpu", weights_only=True)
    model = build(st.get("kwargs") or {})
    model.load_state_dict(st["state"])
    return model.eval(), st["mean"], st["std"]
