"""Huấn luyện MLP, ResNet bảng, FT-Transformer trên GPU cho cả ba bài toán; ghi dl_<task>.json và xác suất val/test."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import limit_threads, save_json, seed_all, timer  # noqa: E402

limit_threads(8)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dnx import gpu  # noqa: E402
from dnx.mem import slot  # noqa: E402
from dnx.ml import deep  # noqa: E402
from dnx.ml.metrics import full_eval  # noqa: E402
from dnx.ml.tasks import MODEL_DIR, PRED_DIR, TASKS, frame  # noqa: E402
from dnx.runlog import log_run  # noqa: E402


def run(task, dev):
    d = frame(task)
    masks = {s: (d["split"] == s).values for s in ("train", "val", "test")}
    data, n_num, cards = deep.prepare(d, masks, task)
    out = {}
    for name in deep.BUILD:
        seed_all()
        times = {}
        with timer(times, "fit"):
            net, hist, T = deep.fit(name, data, n_num, cards, dev)
        pv = torch.sigmoid(deep.predict(net, *T["val"][:2])).cpu().numpy()
        pt = torch.sigmoid(deep.predict(net, *T["test"][:2])).cpu().numpy()
        yv, yt = data["val"][2].astype(int), data["test"][2].astype(int)
        res, _ = full_eval(yv, pv.astype(float), yt, pt.astype(float))
        res.update(seconds=times["fit"], framework="pytorch", history=hist, epochs_run=len(hist),
                   params=sum(p.numel() for p in net.parameters()),
                   peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20)
        out[name] = res
        np.savez_compressed(PRED_DIR / f"{task}_{name}.npz", yv=yv, pv=pv, yt=yt, pt=pt)
        torch.save(net.state_dict(), MODEL_DIR / f"{task}_{name}.pt")
        log_run("dl", task, name, "pytorch", {k: res["test"][k] for k in ("roc_auc", "pr_auc", "f1", "brier", "ece")},
                times["fit"], dev, {"epochs": len(hist), "val_pr_auc": res["val"]["pr_auc"]})
        print(task, name, f"ep {len(hist)} val PR {res['val']['pr_auc']:.4f} test AUC {res['test']['roc_auc']:.4f} "
                          f"({times['fit']:.0f}s)", flush=True)
        del net, T
    save_json(f"dl_{task}", {"task": task, "ts": time.strftime("%FT%T"), "n_num": n_num, "cards": cards, "models": out})


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy GPU cho ml_deep")
    gpu.cap(0.2)
    with slot(4, "ml_deep"):
        for task in (sys.argv[1:] or TASKS):
            run(task, "cuda")


if __name__ == "__main__":
    main()
