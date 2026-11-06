"""Ba mô hình sâu cho dữ liệu bảng: MLP, ResNet bảng, FT-Transformer; chung lớp nhúng đặc trưng và vòng huấn luyện."""
import copy

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn

from .tasks import TASKS

CFG = {"mlp": {"lr": 1e-3, "wd": 1e-4, "epochs": 30}, "resnet": {"lr": 1e-3, "wd": 1e-4, "epochs": 30},
       "ft_transformer": {"lr": 3e-4, "wd": 1e-5, "epochs": 20}}
BATCH, PATIENCE = 1024, 4


def prepare(d, masks, task):
    """Số: điền trung vị + chuẩn hóa; loại: mã nguyên, 0 dành cho giá trị lạ. Fit trên train."""
    t = TASKS[task]
    y = d[t["target"]].values.astype(np.float32)
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(d.loc[masks["train"], t["num"]])
    out, cards, enc, cats = {}, [], None, None
    if t["cat"]:
        cats = d[t["cat"]].astype(object).where(d[t["cat"]].notna(), "missing").astype(str)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(cats[masks["train"]])
        cards = [len(c) + 1 for c in enc.categories_]
    for s, m in masks.items():
        c = (enc.transform(cats[m]) + 1).astype(np.int64) if enc else np.zeros((m.sum(), 0), dtype=np.int64)
        out[s] = (num.transform(d.loc[m, t["num"]]).astype(np.float32), c, y[m])
    return out, len(t["num"]), cards


class Flat(nn.Module):
    """Ghép số với nhúng loại thành một vectơ."""

    def __init__(self, n_num, cards, dim=8):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(c, dim) for c in cards)
        self.out = n_num + dim * len(cards)

    def forward(self, xn, xc):
        return torch.cat([xn, *[e(xc[:, i]) for i, e in enumerate(self.emb)]], 1)


class MLP(nn.Module):
    def __init__(self, n_num, cards, hidden=(256, 128), drop=0.2):
        super().__init__()
        self.flat = Flat(n_num, cards)
        layers, d = [], self.flat.out
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, xn, xc):
        return self.net(self.flat(xn, xc))[:, 0]


class ResBlock(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm1d(d), nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(drop),
                               nn.Linear(2 * d, d), nn.Dropout(drop))

    def forward(self, x):
        return x + self.f(x)


class ResNetTab(nn.Module):
    """ResNet bảng theo Gorishniy và cộng sự (2021)."""

    def __init__(self, n_num, cards, d=128, blocks=3, drop=0.2):
        super().__init__()
        self.flat = Flat(n_num, cards)
        self.net = nn.Sequential(nn.Linear(self.flat.out, d), *[ResBlock(d, drop) for _ in range(blocks)],
                                 nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, xn, xc):
        return self.net(self.flat(xn, xc))[:, 0]


class FTTransformer(nn.Module):
    """Mỗi đặc trưng thành một token, thêm CLS, qua Transformer Encoder rồi phân loại trên CLS."""

    def __init__(self, n_num, cards, d=64, layers=3, heads=8, drop=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.b = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.emb = nn.ModuleList(nn.Embedding(c, d) for c in cards)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, "gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, xn, xc):
        toks = [xn[:, :, None] * self.w + self.b]
        if len(self.emb):
            toks.append(torch.stack([e(xc[:, i]) for i, e in enumerate(self.emb)], 1))
        x = torch.cat([self.cls.expand(len(xn), -1, -1), *toks], 1)
        return self.head(self.enc(x)[:, 0])[:, 0]


BUILD = {"mlp": MLP, "resnet": ResNetTab, "ft_transformer": FTTransformer}


@torch.no_grad()
def predict(net, xn, xc, bs=8192):
    net.eval()
    return torch.cat([net(xn[i:i + bs], xc[i:i + bs]).float() for i in range(0, len(xn), bs)])


def fit(name, data, n_num, cards, dev, seed=42):
    """AdamW, dừng sớm theo mất mát val; dữ liệu nằm sẵn trong VRAM nên không cần tiến trình nạp dữ liệu."""
    torch.manual_seed(seed)
    cfg = CFG[name]
    T = {s: tuple(torch.as_tensor(a, device=dev) for a in v) for s, v in data.items()}
    xn, xc, y = T["train"]
    net = BUILD[name](n_num, cards).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    crit = nn.BCEWithLogitsLoss()
    best, keep, wait, hist = float("inf"), None, 0, []
    for ep in range(1, cfg["epochs"] + 1):
        net.train()
        order, total = torch.randperm(len(y), device=dev), 0.0
        for i in range(0, len(y), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue  # BatchNorm không chạy được với lô một mẫu
            opt.zero_grad(set_to_none=True)
            loss = crit(net(xn[b], xc[b]), y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        va = crit(predict(net, *T["val"][:2]), T["val"][2]).item()
        hist.append({"epoch": ep, "train_loss": total / len(y), "val_loss": va})
        if va < best - 1e-5:
            best, keep, wait = va, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(keep)
    return net, hist, T
