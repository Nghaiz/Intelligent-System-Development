"""Mô hình dự đoán chịu thiếu đặc trưng: MLP nhận giá trị đã chuẩn hóa nhân mặt nạ, nối với mặt nạ."""
import json

import torch
from torch import nn

from ..common import MODELS, device
from .features import N

DIR = MODELS / "rl"
HIDDEN = (256, 256, 128)
DROPOUT = 0.1


class MaskedMLP(nn.Module):
    def __init__(self, n=N, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers, last = [], 2 * n
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.net = nn.Sequential(*layers, nn.Linear(last, 1))

    def forward(self, x, mask):
        return self.net(torch.cat([x * mask, mask], dim=-1)).squeeze(-1)

    def prob(self, x, mask):
        return torch.sigmoid(self(x, mask))


def save(model, mean, std, metrics):
    DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DIR / "predictor.pt")
    (DIR / "predictor.json").write_text(json.dumps({"mean": [float(v) for v in mean], "std": [float(v) for v in std],
                                                    "metrics": metrics}, ensure_ascii=False), encoding="utf-8")


def load(dev=None):
    """Trả model ở chế độ suy luận cùng mean, std dạng tensor trên thiết bị dev."""
    dev = dev or device()
    meta = json.loads((DIR / "predictor.json").read_text(encoding="utf-8"))
    model = MaskedMLP().to(dev)
    model.load_state_dict(torch.load(DIR / "predictor.pt", map_location=dev, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, torch.tensor(meta["mean"], device=dev), torch.tensor(meta["std"], device=dev)
