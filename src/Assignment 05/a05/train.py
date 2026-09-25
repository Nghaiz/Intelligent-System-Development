"""Mục 4 — vòng lặp huấn luyện dùng chung cho mọi mô hình và mọi tập.

Một epoch là: với mỗi lô, tính L(f_θ(x), y), lan truyền ngược, θ <- θ - η ∇L.
Checkpoint giữ trạng thái có val loss nhỏ nhất và được nạp lại trước khi đánh giá test.
"""
import copy
import json
import random
import time

import numpy as np
import torch
from torch import nn

from .export import MODELS
from .models import build_model

# Cấu hình chung: bốn mô hình trong cùng một tập dùng đúng một cấu hình.
IMAGE_CFG = {"epochs": 30, "batch_size": 256, "max_lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
             "grad_clip": 2.0}
TAB_CFG = {"epochs": 30, "batch_size": 512, "lr": 1e-3, "weight_decay": 1e-4, "patience": 5, "threads": 8}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _n_correct(out, y):
    if out.ndim == 2 and out.shape[1] > 1:
        return (out.argmax(1) == y).sum()
    return ((out.reshape(-1) > 0).float() == y).sum()


def train_one_epoch(model, batches, loss_fn, opt=None, sched=None, scaler=None, amp=False, clip=None):
    """Chạy một lượt qua dữ liệu. opt=None nghĩa là chỉ đánh giá, không cập nhật.
    clip: giới hạn chuẩn L2 của gradient (None = không cắt). Trả về (loss trung bình, accuracy)."""
    training = opt is not None
    model.train(training)
    tot_loss, tot_ok, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in batches:
            with torch.autocast(device_type=xb.device.type, dtype=torch.float16, enabled=amp):
                out = model(xb)
                loss = loss_fn(out, yb)
            if training:
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                else:
                    loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                if sched is not None:
                    sched.step()
            tot_loss = tot_loss + loss.detach().float() * len(yb)
            tot_ok = tot_ok + _n_correct(out.detach().float(), yb)
            n += len(yb)
    return float(tot_loss) / n, float(tot_ok) / n


def fit(model, batches, loss_fn, opt, epochs, device, sched=None, amp=False, patience=None, clip=None,
        log=print):
    """Huấn luyện `epochs` epoch, giữ trạng thái có val loss nhỏ nhất.

    batches(split) phải trả về một iterator lô mới cho 'train' hoặc 'val'.
    patience: dừng sớm nếu val loss không giảm sau chừng ấy epoch (None = không dừng).
    Trả về dict history, best_epoch, best_state (trên CPU), epoch_seconds.
    """
    scaler = torch.amp.GradScaler("cuda") if amp else None
    history, best, best_state, best_epoch, wait = [], float("inf"), None, 0, 0
    for ep in range(1, epochs + 1):
        _sync(device)
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, batches("train"), loss_fn, opt, sched, scaler, amp, clip)
        _sync(device)
        secs = time.perf_counter() - t0
        va_loss, va_acc = train_one_epoch(model, batches("val"), loss_fn, amp=amp)
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": secs})
        log(f"  epoch {ep:2d}  train {tr_loss:.4f}/{tr_acc:.4f}  val {va_loss:.4f}/{va_acc:.4f}  {secs:.1f}s")
        if va_loss < best:
            best, best_epoch, wait = va_loss, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if patience is not None and wait >= patience:
                log(f"  dừng sớm ở epoch {ep}, tốt nhất là epoch {best_epoch}")
                break
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_state": best_state,
            "epoch_seconds": float(np.mean([h["seconds"] for h in history]))}


def save_checkpoint(name, state, meta):
    """Ghi models/<name>.pt và cập nhật mục <name> trong models/metadata.json."""
    torch.save(state, MODELS / f"{name}.pt")
    path = MODELS / "metadata.json"
    all_meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    all_meta[name] = copy.deepcopy(meta)
    path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def predict(model, batches, amp=False):
    """Đầu ra thô (logits) và nhãn của cả một nhánh dữ liệu, trả về NumPy trên CPU."""
    model.eval()
    outs, ys = [], []
    for xb, yb in batches:
        with torch.autocast(device_type=xb.device.type, dtype=torch.float16, enabled=amp):
            outs.append(model(xb).float().cpu())
        ys.append(yb.cpu())
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def train_image_model(name, data, cfg=IMAGE_CFG, seed=42, log=print):
    """Huấn luyện một mô hình 2D trên CIFAR: SGD nesterov + OneCycle, AMP fp16 trên GPU.

    Cắt chuẩn gradient ở cả bốn mô hình: BasicCNN không có BatchNorm nên phân kỳ
    khi lr lên đỉnh 0,1 nếu không cắt (outputs/figdata/cifar10_basic_noclip_history.dat)."""
    set_seed(seed)
    dev = data.device
    model = build_model(name, 2, (3, 32, 32), data.n_classes).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["max_lr"], momentum=cfg["momentum"],
                          nesterov=True, weight_decay=cfg["weight_decay"])
    steps = -(-len(data.y["train"]) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["max_lr"], epochs=cfg["epochs"], steps_per_epoch=steps)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    batches = lambda split: data.batches(split, cfg["batch_size"], augment=split == "train",
                                         shuffle=split == "train", gen=gen)
    res = fit(model, batches, nn.CrossEntropyLoss(), opt, cfg["epochs"], dev, sched, amp=True,
              clip=cfg["grad_clip"], log=log)
    return model, res


def tabular_batches(X, y, batch_size, shuffle=False, gen=None):
    """Lô cho dữ liệu bảng: X (N, d) thành tensor (N, 1, d) để Conv1d trượt dọc các đặc trưng."""
    X, y = torch.as_tensor(X)[:, None, :], torch.as_tensor(y)
    order = torch.randperm(len(y), generator=gen) if shuffle else torch.arange(len(y))
    for s in range(0, len(y), batch_size):
        idx = order[s:s + batch_size]
        yield X[idx], y[idx]


def train_tabular_model(name, D, cfg=TAB_CFG, seed=42, log=print):
    """Huấn luyện một mô hình 1D trên Diabetes bằng CPU: AdamW, BCE có pos_weight, dừng sớm."""
    set_seed(seed)
    torch.set_num_threads(cfg["threads"])
    model = build_model(name, 1, (1, D["X"]["train"].shape[1]), 1)
    y_tr = D["y"]["train"]
    pos_weight = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum())
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn = lambda out, y: bce(out.reshape(-1), y)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    batches = lambda split: tabular_batches(D["X"][split], D["y"][split], cfg["batch_size"],
                                            shuffle=split == "train", gen=gen)
    res = fit(model, batches, loss_fn, opt, cfg["epochs"], "cpu", patience=cfg["patience"], log=log)
    return model, res
