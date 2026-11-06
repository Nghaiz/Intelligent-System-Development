"""Chương 4: hai khung phụ chạy CPU. NumPy (BPTT tự viết) và Keras LSTM trên toàn bộ chuỗi FPT.

    PYTHONPATH=. python scripts/ch4_side_run.py numpy
    PYTHONPATH=. python scripts/ch4_side_run.py keras
"""
import json
import sys
import time

import numpy as np

from tlf.ch4_data import OUT_DIR, Prepared
from tlf.ch4_metrics import forecast_metrics
from tlf.common import limit_threads, save_json, seed_all
from tlf.mem import slot
from tlf.runlog import log_run

RUNS = OUT_DIR / "runs"
HIDDEN = 64
GRAD_SAMPLE = 8
GRAD_T = 12


def _record(prep, tag, framework, arch, test, hist, secs, extra=None):
    tt = prep.last_index("test")
    np.savez_compressed(RUNS / f"{tag}.npz", test=test.astype(np.float32))
    m = forecast_metrics(prep.truth(tt)[:, 0], prep.to_level(test, tt)[:, 0], prep.level[tt])
    meta = {"hist": hist, "seconds": secs, "epochs": len(hist["loss"]), **(extra or {})}
    (RUNS / f"{tag}.json").write_text(json.dumps(meta), encoding="utf-8")
    log_run("ch4", prep.spec.id, arch, framework, m, secs, "cpu", extra)
    print(f"{tag}: {meta['epochs']} epoch {secs:.0f}s rmse {m['rmse']:.4g}", flush=True)


def run_numpy(series):
    from tlf.ch4_scratch import RNNScratch, grad_check, train
    prep = Prepared(series)
    ts, tv, tt = (prep.last_index(k) for k in ("train", "val", "test"))
    X, Y = prep.windows(ts).astype(np.float64), prep.targets(ts).astype(np.float64)
    model = RNNScratch(X.shape[-1], HIDDEN, prep.spec.horizon, seed=42)
    check = grad_check(model, X[:GRAD_SAMPLE, -GRAD_T:], Y[:GRAD_SAMPLE])
    t0 = time.perf_counter()
    hist = train(model, X, Y, prep.windows(tv).astype(np.float64), prep.targets(tv).astype(np.float64))
    secs = time.perf_counter() - t0
    test = np.concatenate([model.forward(prep.windows(tt[i:i + 2048]).astype(np.float64)) for i in range(0, len(tt), 2048)])
    yhat = model.forward(X[:256])
    model.backward(yhat, Y[:256])
    n = sum(v.size for v in model.p.values())
    _record(prep, f"{series}__rnn__numpy", "numpy", "rnn", test, hist, secs,
            {"grad_check": check, "delta_norms": model.delta_norms, "n_params": n, "n_train": len(ts)})


def run_keras(series, arch, rec_drop=0.0):
    from tlf.ch4_keras import fit_keras, predict_keras
    tag = f"{series}__{arch}__keras" + ("_rd" if rec_drop else "")
    if (RUNS / f"{tag}.npz").exists():
        return
    prep = Prepared(series)
    seed_all(42)
    model, hist, secs = fit_keras(prep, arch, rec_drop=rec_drop)
    tt = prep.last_index("test")
    _record(prep, tag, "keras", arch, predict_keras(model, prep.windows(tt)), hist, secs,
            {"rec_drop": rec_drop, "n_params": model.count_params()})


def main(mode):
    limit_threads(4)
    RUNS.mkdir(exist_ok=True)
    if mode == "numpy":
        with slot(1, "ch4 numpy"):
            run_numpy("FPT")
        return
    with slot(4, "ch4 keras"):
        run_keras("FPT", "lstm")


if __name__ == "__main__":
    main(sys.argv[1])
