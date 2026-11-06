"""Chương 4: bốn kiến trúc hồi quy PyTorch và vòng huấn luyện dự báo trực tiếp H bước, chạy GPU."""
import copy
import time

import numpy as np
import torch
from torch import nn

from .common import model_dir

ARCHS = ("rnn", "lstm", "gru", "bilstm")
ARCH_NAMES = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}
CFG = {"stock": {"hidden": 32, "dropout": 0.3, "batch": 64, "epochs": 40, "patience": 8, "lr": 5e-4, "wd": 1e-4},
       "hourly": {"hidden": 64, "dropout": 0.1, "batch": 256, "epochs": 12, "patience": 3, "lr": 1e-3, "wd": 0.0}}
CLIP_NORM = 1.0
PRED_BATCH = 2048


def config_of(spec):
    return CFG["stock" if spec.log_target else "hourly"]


class SeqRegressor(nn.Module):
    """Một lớp hồi quy, Dropout rồi Linear ra H giá trị: dự báo trực tiếp nhiều bước từ trạng thái ẩn cuối."""

    def __init__(self, n_features, horizon, arch="lstm", hidden=64, dropout=0.2):
        super().__init__()
        cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU, "bilstm": nn.LSTM}[arch]
        self.rnn = cell(n_features, hidden, batch_first=True, bidirectional=arch == "bilstm")
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if arch == "bilstm" else 1), horizon)

    def forward(self, x):                       # x: (lô, T, D)
        out, _ = self.rnn(x)                    # out: (lô, T, hidden × số chiều)
        return self.head(self.drop(out[:, -1])) # (lô, H)


class Windows:
    """Giữ chuỗi nguyên trên GPU và cắt cửa sổ theo lô, nên 90 nghìn cửa sổ không tốn 90 nghìn bản sao."""

    def __init__(self, prep, device):
        self.x = torch.as_tensor(prep.x, device=device)
        self.offs = torch.arange(-prep.spec.window + 1, 1, device=device)
        self.device = device

    def batch(self, ts):
        return self.x[torch.as_tensor(ts, device=self.device)[:, None] + self.offs]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, win, ts, y, opt, batch, gen):
    """Một epoch: xáo trộn thứ tự cửa sổ, MSE, BPTT, cắt chuẩn gradient, Adam. Trả MSE trung bình."""
    model.train()
    order = gen.permutation(len(ts))
    total = 0.0
    for i in range(0, len(ts), batch):
        idx = order[i:i + batch]
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(win.batch(ts[idx])), y[idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        total += loss.item() * len(idx)
    return total / len(ts)


@torch.no_grad()
def predict_scaled(model, win, ts):
    model.eval()
    out = [model(win.batch(ts[i:i + PRED_BATCH])).cpu().numpy() for i in range(0, len(ts), PRED_BATCH)]
    return np.concatenate(out)


def _resume(path, model, opt, sched):
    """Tiếp tục từ điểm kiểm tra cuối nếu lượt trước đứt giữa chừng."""
    st = {"epoch": 0, "hist": {"loss": [], "val_loss": []}, "best": float("inf"), "best_state": None, "bad": 0}
    if path.exists():
        ck = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        st = ck["state"]
    return st


def fit(prep, arch, device, seed=42):
    """Dừng sớm theo MSE validation, khôi phục trọng số tốt nhất. Trả (model, lịch sử, giây, MSE val tốt nhất)."""
    cfg, s = config_of(prep.spec), prep.spec
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    ts, tv = prep.last_index("train"), prep.last_index("val")
    y, yv = (torch.as_tensor(prep.targets(t), device=device) for t in (ts, tv))
    win = Windows(prep, device)
    model = SeqRegressor(prep.x.shape[1], s.horizon, arch, cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    ckpt = model_dir("ch4") / f"last_{s.id}__{arch}__s{seed}.pt"
    st = _resume(ckpt, model, opt, sched)
    t0 = time.perf_counter()
    for ep in range(st["epoch"], cfg["epochs"]):
        tr = train_epoch(model, win, ts, y, opt, cfg["batch"], gen)
        va = float(((torch.as_tensor(predict_scaled(model, win, tv), device=device) - yv) ** 2).mean())
        sched.step(va)
        st["hist"]["loss"].append(tr)
        st["hist"]["val_loss"].append(va)
        st["epoch"] = ep + 1
        if va < st["best"] - 1e-5:
            st.update(best=va, best_state=copy.deepcopy(model.state_dict()), bad=0)
        else:
            st["bad"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "state": st}, ckpt)
        if st["bad"] >= cfg["patience"]:
            break
    model.load_state_dict(st["best_state"])
    ckpt.unlink(missing_ok=True)
    return model, st["hist"], time.perf_counter() - t0, st["best"]
