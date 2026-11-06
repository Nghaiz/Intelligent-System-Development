"""Huấn luyện MLP có mặt nạ trên CDC: mỗi lô che ngẫu nhiên với tỉ lệ khác nhau cho từng dòng."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from dnx.common import device, limit_threads, save_json, seed_all
from dnx.gpu import cap
from dnx.mem import slot
from dnx.rl import data, predictor
from dnx.runlog import log_run

EPOCHS, BATCH, PATIENCE = 60, 4096, 8


def random_mask(x):
    keep = torch.rand(len(x), 1, device=x.device)
    return (torch.rand(x.shape, device=x.device) < keep).float()


def main():
    limit_threads(4)
    seed_all(42)
    cap(0.12)
    t0, dev = time.time(), device()
    splits, mean, std = data.load()
    tens = {k: (torch.tensor((x - mean) / std, device=dev), torch.tensor(y, device=dev)) for k, (x, y) in splits.items()}
    Xtr, ytr = tens["train"]
    Xva, yva = tens["val"]
    model = predictor.MaskedMLP().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    fix_mask = torch.cat([random_mask(Xva) for _ in range(3)])
    fix_x, fix_y = Xva.repeat(3, 1), yva.repeat(3)
    best, best_l, wait, hist = None, 1e9, 0, []
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        total = 0.0
        for idx in perm.split(BATCH):
            x, y = Xtr[idx], ytr[idx]
            loss = F.binary_cross_entropy_with_logits(model(x, random_mask(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(model(fix_x, fix_mask), fix_y).item()
        hist.append({"epoch": ep + 1, "train": total / len(Xtr), "val": vl})
        print(f"ep {ep + 1} train {total / len(Xtr):.4f} val {vl:.4f}", flush=True)
        if vl < best_l - 1e-5:
            best_l, wait, best = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best)
    model.eval()
    Xte, yte = tens["test"]
    with torch.no_grad():
        p = model.prob(Xte, torch.ones_like(Xte)).cpu().numpy()
        p_none = model.prob(Xte, torch.zeros_like(Xte))[0].item()
    y = yte.cpu().numpy()
    m = {"auc_full": float(roc_auc_score(y, p)), "logloss_full": float(log_loss(y, p)),
         "prior_model": p_none, "prior_train": float(ytr.mean()), "val_logloss_masked": best_l,
         "n": {k: len(v[0]) for k, v in tens.items()}, "epochs": len(hist), "seconds": time.time() - t0}
    predictor.save(model, mean, std, m)
    save_json("rl_predictor", {**m, "history": hist})
    log_run("rl", "cdc_brfss", "MaskedMLP", "pytorch", {"auc": m["auc_full"], "logloss": m["logloss_full"]},
            m["seconds"], dev, {"epochs": len(hist), "batch": BATCH})
    print(m, flush=True)


if __name__ == "__main__":
    with slot(3, "rl predictor"):
        main()
