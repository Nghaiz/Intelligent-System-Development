"""Huấn luyện một lượt Double DQN: python scripts/rl_train_dqn.py --seed 0 --lam 0.002 [--steps N]."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dnx.common import device, limit_threads, save_json
from dnx.gpu import cap
from dnx.mem import slot
from dnx.rl import data, dqn, predictor
from dnx.runlog import log_run


def main(a):
    limit_threads(3)
    cap(a.vram)
    t0, dev = time.time(), device()
    model, mean, std = predictor.load(dev)
    splits, _, _ = data.load()
    mu, sd = mean.cpu().numpy(), std.cpu().numpy()
    Xtr = torch.tensor((splits["train"][0] - mu) / sd, device=dev)
    Xva = torch.tensor((splits["val"][0] - mu) / sd, device=dev)
    yva = torch.tensor(splits["val"][1], device=dev)
    cfg = {**dqn.CFG, "steps": a.steps}
    tag = f"s{a.seed}_l{a.lam:g}"

    def show(row):
        print(tag, {k: round(v, 4) for k, v in row.items()}, flush=True)

    best, hist = dqn.train(model, Xtr, Xva, yva, a.lam, a.seed, cfg, show)
    (predictor.DIR / "dqn").mkdir(parents=True, exist_ok=True)
    torch.save(best, predictor.DIR / "dqn" / f"{tag}.pt")
    save_json(f"rl_dqn_{tag}", {"seed": a.seed, "lam": a.lam, "cfg": cfg, "history": hist})
    top = max(hist, key=lambda r: r["objective"])
    log_run("rl", "cdc_brfss", f"DoubleDQN_{tag}", "pytorch",
            {"objective": top["objective"], "log_score": top["log_score"], "questions": top["questions"]},
            time.time() - t0, dev, {"seed": a.seed, "lam": a.lam, "steps": a.steps})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.002)
    ap.add_argument("--steps", type=int, default=dqn.CFG["steps"])
    ap.add_argument("--vram", type=float, default=0.1)
    args = ap.parse_args()
    with slot(3, f"rl dqn s{args.seed}"):
        main(args)
