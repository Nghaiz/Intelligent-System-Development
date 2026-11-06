"""Chương 2: cấu hình chung và MLP PyTorch; cùng kiến trúc với bản NumPy và Keras."""
import numpy as np

CFG = {"cdc": {"hidden": (128, 64), "drop": 0.1, "lr": 1e-3, "bs": 512, "epochs": 20, "patience": 4},
       "housing": {"hidden": (128, 64), "drop": 0.1, "lr": 1e-3, "bs": 128, "epochs": 40, "patience": 6},
       "vsfc": {"hidden": (128, 64), "drop": 0.5, "lr": 5e-4, "bs": 64, "epochs": 25, "patience": 4}}
SEEDS = (11, 22, 33)
N_SEEDS = {"cdc": 1, "housing": 3, "vsfc": 3}  # CDC một seed để kịp thời hạn và trần 4 luồng


def class_weights(y):
    """Trọng số nghịch đảo tần suất, trung bình bằng 1 trên tập huấn luyện."""
    c = np.bincount(y)
    return len(y) / (len(c) * c)


def scale_target(y_tr):
    """Hồi quy: chuẩn hóa y theo train để mất mát và tốc độ học cùng thang với bài phân loại."""
    return float(y_tr.mean()), float(y_tr.std())


def build_torch(n_in, n_out, hidden, drop):
    from torch import nn
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, n_out))


def fit_torch(X, y, Xv, yv, n_out, w, cfg, seed, dev):
    """Adam mini-batch trên GPU, dữ liệu nằm sẵn trong VRAM; dừng sớm theo mất mát validation."""
    import copy

    import torch
    from torch import nn
    torch.manual_seed(seed)
    net = build_torch(X.shape[1], n_out, cfg["hidden"], cfg["drop"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    reg = n_out == 1
    Xt, Xv = torch.as_tensor(X, device=dev), torch.as_tensor(Xv, device=dev)
    yt = torch.as_tensor(y, device=dev, dtype=torch.float32 if reg else torch.long)
    yv = torch.as_tensor(yv, device=dev, dtype=torch.float32 if reg else torch.long)
    wt = None if w is None else torch.as_tensor(w, device=dev, dtype=torch.float32)
    tr_loss = nn.MSELoss() if reg else nn.CrossEntropyLoss(weight=wt)
    va_loss = nn.MSELoss() if reg else nn.CrossEntropyLoss()

    def out(a):
        return a[:, 0] if reg else a

    best, keep, wait, hist = float("inf"), None, 0, []
    for ep in range(1, cfg["epochs"] + 1):
        net.train()
        order, total = torch.randperm(len(Xt), device=dev), 0.0
        for i in range(0, len(Xt), cfg["bs"]):
            b = order[i:i + cfg["bs"]]
            opt.zero_grad()
            loss = tr_loss(out(net(Xt[b])), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            va = va_loss(out(net(Xv)), yv).item()
        hist.append({"epoch": ep, "train_loss": total / len(Xt), "val_loss": va})
        if va < best - 1e-5:
            best, keep, wait = va, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    net.load_state_dict(keep)
    return net.eval(), hist
