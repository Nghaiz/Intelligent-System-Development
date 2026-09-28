"""Hạng mục 3 (Ch.3): vòng lặp huấn luyện PyTorch chung cho hai tập.

AdamW + clip chuẩn gradient 1,0 + dừng sớm theo val loss; giữ trọng số của epoch có val loss nhỏ nhất.
Dữ liệu nằm sẵn trên thiết bị dưới dạng tensor, tự chia lô, không dùng DataLoader.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from .export import MODELS
from .models_torch import CELLS, build_torch, count_params

SEED = 42
CFG = {
    "amzn": {"task": "reg", "epochs": 60, "batch": 64, "patience": 8},
    "kkbox": {"task": "clf", "epochs": 20, "batch": 512, "patience": 4},
}
OPT = {"lr": 1e-3, "weight_decay": 1e-4, "clip": 1.0}


def make_loss(task, y_train=None, device="cpu"):
    """Hồi quy: MSE. Phân loại: BCE trên logit, pos_weight = số âm / số dương của train."""
    if task == "reg":
        return nn.MSELoss()
    pos = float(y_train.sum())
    pw = torch.tensor((len(y_train) - pos) / pos, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def fit(model, data, cfg, device, max_epochs=None):
    """Huấn luyện một mô hình. data: dict X/y theo train/val (NumPy). Trả về model tốt nhất + history."""
    torch.manual_seed(SEED)
    Xtr = torch.as_tensor(data["X"]["train"], device=device)
    ytr = torch.as_tensor(data["y"]["train"], device=device)
    Xva = torch.as_tensor(data["X"]["val"], device=device)
    yva = torch.as_tensor(data["y"]["val"], device=device)
    model = model.to(device)
    loss_fn = make_loss(cfg["task"], data["y"]["train"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "epoch_s": []}
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(1, (max_epochs or cfg["epochs"]) + 1):
        sync(device)
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(Xtr), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), OPT["clip"])
            opt.step()
            total += loss.item() * len(idx)
        sync(device)
        elapsed = time.perf_counter() - t0
        val = evaluate_loss(model, Xva, yva, loss_fn, min(cfg["batch"] * 8, 2048))
        hist["epoch"].append(epoch)
        hist["train_loss"].append(total / len(Xtr))
        hist["val_loss"].append(val)
        hist["epoch_s"].append(elapsed)
        if val < best - 1e-6:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    hist["best_epoch"] = hist["epoch"][int(np.argmin(hist["val_loss"]))]
    return model, hist


@torch.no_grad()
def evaluate_loss(model, X, y, loss_fn, batch):
    model.eval()
    total = sum(loss_fn(model(X[i:i + batch]), y[i:i + batch]).item() * len(y[i:i + batch])
                for i in range(0, len(X), batch))
    return total / len(X)


@torch.no_grad()
def predict(model, X, device="cpu", batch=2048):
    model.eval()
    X = torch.as_tensor(X, device=device)
    return np.concatenate([model(X[i:i + batch]).float().cpu().numpy() for i in range(0, len(X), batch)])


def pick_device(data, cfg):
    """Đo 1 epoch của LSTM trên CPU và GPU, trả về thiết bị nhanh hơn (tập nhỏ có thể nhanh hơn trên CPU)."""
    times = {}
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        fit(build_torch("lstm"), data, cfg, dev, max_epochs=1)          # lượt làm nóng (cuDNN, bộ cấp phát)
        _, h = fit(build_torch("lstm"), data, cfg, dev, max_epochs=1)
        times[dev] = h["epoch_s"][0]
    return min(times, key=times.get), times


def run_torch(dataset, data, device):
    """Huấn luyện 4 mô hình của một tập, lưu .pth, trả về {cell: {history, pred_val, pred_test, ...}}."""
    cfg = CFG[dataset]
    out = {}
    for cell in CELLS:
        model, hist = fit(build_torch(cell, data["X"]["train"].shape[-1]), data, cfg, device)
        path = MODELS / f"{dataset}_torch_{cell}.pth"
        torch.save(model.state_dict(), path)
        out[cell] = {"history": hist, "params": count_params(model), "device": str(device),
                     "pred_val": predict(model, data["X"]["val"], device),
                     "pred_test": predict(model, data["X"]["test"], device), "file": path.name}
        print(f"{dataset} torch {cell:6s} epoch tốt {hist['best_epoch']:2d}/{len(hist['epoch'])} "
              f"val {min(hist['val_loss']):.4f} · {np.mean(hist['epoch_s']):.2f} s/epoch trên {device}")
    return out
