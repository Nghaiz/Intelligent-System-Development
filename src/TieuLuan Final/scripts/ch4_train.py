"""Chương 4: huấn luyện bốn kiến trúc PyTorch trên sáu chuỗi; lượt đã xong được bỏ qua khi chạy lại.

    PYTHONPATH=. python scripts/ch4_train.py [--series FPT weather] [--archs lstm gru]

Mỗi lượt ghi data/ch4/runs/<chuỗi>__<kiến trúc>__s<hạt>.npz; hạt giống có MSE validation thấp nhất được lưu thành
models/ch4/<chuỗi>__<kiến trúc>.pt cùng thống kê chuẩn hóa và phân vị sai số dùng cho khoảng dao động.
"""
import argparse
import json
from contextlib import nullcontext

import numpy as np
import torch

from tlf.ch4_data import OUT_DIR, SPECS, Prepared
from tlf.ch4_metrics import forecast_metrics
from tlf.ch4_torch import ARCHS, Windows, config_of, fit, n_params, predict_scaled
from tlf.common import limit_threads, model_dir, seed_all
from tlf.gpu import heavy
from tlf.mem import require_disk, slot
from tlf.runlog import log_run

SEED = 42
INTERVAL = (0.05, 0.95)
RUNS = OUT_DIR / "runs"


def interval_quantiles(prep, val_scaled):
    """Phân vị sai số validation theo từng bước, ở thang của y, nên khoảng không dùng dữ liệu test."""
    tv = prep.last_index("val")
    resid = (prep.targets(tv) - val_scaled) * prep.sd_h
    return np.quantile(resid, INTERVAL, axis=0)


def save_best(prep, arch, seed, model, best, q):
    path = model_dir("ch4") / f"{prep.spec.id}__{arch}.pt"
    if path.exists() and torch.load(path, map_location="cpu", weights_only=True)["val_mse"] <= best:
        return
    cfg = config_of(prep.spec)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "val_mse": best, "seed": seed,
                "hidden": cfg["hidden"], "dropout": cfg["dropout"], "columns": prep.columns,
                "mean": torch.as_tensor(prep.mean), "std": torch.as_tensor(prep.std),
                "sd_h": torch.as_tensor(prep.sd_h), "q": torch.as_tensor(q)}, path)


def run_one(prep, arch, seed, device):
    tag = f"{prep.spec.id}__{arch}__s{seed}"
    if (RUNS / f"{tag}.npz").exists():
        return
    seed_all(seed)
    model, hist, secs, best = fit(prep, arch, device, seed)
    win = Windows(prep, device)
    tt, tv = prep.last_index("test"), prep.last_index("val")
    test, val = predict_scaled(model, win, tt), predict_scaled(model, win, tv)
    save_best(prep, arch, seed, model, best, interval_quantiles(prep, val))
    m = forecast_metrics(prep.truth(tt)[:, 0], prep.to_level(test, tt)[:, 0], prep.level[tt])
    np.savez_compressed(RUNS / f"{tag}.npz", test=test.astype(np.float32), val=val.astype(np.float32))
    meta = {"hist": hist, "seconds": secs, "val_mse": best, "n_params": n_params(model), "epochs": len(hist["loss"])}
    (RUNS / f"{tag}.json").write_text(json.dumps(meta), encoding="utf-8")
    log_run("ch4", prep.spec.id, arch, "pytorch", m, secs, device, {"seed": seed, **config_of(prep.spec)})
    print(f"{tag}: {meta['epochs']} epoch, {secs:.0f}s, val {best:.4f}, rmse {m['rmse']:.4g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=list(SPECS))
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    ap.add_argument("--cpu", action="store_true", help="chuỗi ngắn chạy CPU, không cần khóa GPU")
    args = ap.parse_args()
    limit_threads(4)
    require_disk(0.1, "kết quả ch4")
    RUNS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    for series in args.series:
        prep = Prepared(series)
        with slot(2, f"ch4 {series}"), (nullcontext() if args.cpu else heavy(f"ch4 {series}")):
            for arch in args.archs:
                run_one(prep, arch, SEED, device)


if __name__ == "__main__":
    main()
