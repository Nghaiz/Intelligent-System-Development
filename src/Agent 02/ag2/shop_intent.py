"""Bộ phân loại ý định bằng PyTorch (Chương 8): túi đặc trưng băm + EmbeddingBag, huấn luyện trên GPU CUDA."""
import re
import time
import unicodedata
import zlib

import torch
from torch import nn

from .shop_data import INTENTS, strip_accents

N_BUCKETS = 1 << 16


def features(text):
    """Đặc trưng của câu: từ, cặp từ liền nhau và 3-gram ký tự, mỗi thứ ở cả dạng có dấu lẫn không dấu.
    Băm CRC32 vào N_BUCKETS ô nên không cần từ điển, câu có từ mới vẫn có đặc trưng."""
    text = unicodedata.normalize("NFC", text.lower())
    feats = []
    for form in {text, strip_accents(text)}:
        words = re.findall(r"\w+", form)
        feats += [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        feats += [f"c:{w[i:i + 3]}" for w in (f"<{w}>" for w in words) for i in range(len(w) - 2)]
    return [zlib.crc32(f.encode("utf-8")) % N_BUCKETS for f in feats] or [0]


class IntentNet(nn.Module):
    """EmbeddingBag(trung bình) -> Dropout -> Linear -> ReLU -> Linear(6 ý định)."""

    def __init__(self, dim=96, hidden=64, n_classes=len(INTENTS), dropout=0.3):
        super().__init__()
        self.emb = nn.EmbeddingBag(N_BUCKETS, dim, mode="mean")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, ids, offsets):
        return self.head(self.emb(ids, offsets))


def batchify(texts, device):
    """Ghép đặc trưng của nhiều câu thành (ids, offsets) đúng định dạng EmbeddingBag."""
    ids, offsets = [], []
    for t in texts:
        offsets.append(len(ids))
        ids += features(t)
    return torch.tensor(ids, device=device), torch.tensor(offsets, device=device)


def run_epoch(model, data, device, opt=None, batch=128, gen=None):
    """Một lượt qua dữ liệu; có opt thì huấn luyện, không thì chỉ đánh giá. Trả (loss trung bình, accuracy)."""
    model.train(opt is not None)
    order = torch.randperm(len(data), generator=gen).tolist() if opt else range(len(data))
    order = list(order)
    tot_loss, correct = 0.0, 0
    for i in range(0, len(order), batch):
        chunk = [data[j] for j in order[i:i + batch]]
        ids, offs = batchify([t for t, _ in chunk], device)
        y = torch.tensor([lab for _, lab in chunk], device=device)
        with torch.set_grad_enabled(opt is not None):
            logits = model(ids, offs)
            loss = nn.functional.cross_entropy(logits, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * len(chunk)
        correct += (logits.argmax(1) == y).sum().item()
    return tot_loss / len(data), correct / len(data)


def train_intent(data, epochs=30, lr=3e-3, seed=707):
    """Huấn luyện trên CUDA nếu có; giữ trọng số của epoch có val accuracy cao nhất (val loss thấp nhất khi hòa)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    hist, best, t0 = [], None, time.perf_counter()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, data["train"], device, opt, gen=gen)
        va_loss, va_acc = run_epoch(model, data["val"], device)
        hist.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if best is None or (va_acc, -va_loss) > (best[0], -best[1]):
            best = (va_acc, va_loss, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.load_state_dict(best[3])
    info = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "cuda": device.type == "cuda", "torch": torch.__version__, "epochs": epochs, "best_epoch": best[2],
            "train_seconds": time.perf_counter() - t0, "params": sum(p.numel() for p in model.parameters())}
    return model, hist, info


@torch.no_grad()
def predict_proba(model, texts):
    """Xác suất softmax của từng ý định cho mỗi câu (tensor CPU)."""
    model.eval()
    device = next(model.parameters()).device
    ids, offs = batchify(texts, device)
    return torch.softmax(model(ids, offs), dim=1).cpu()
