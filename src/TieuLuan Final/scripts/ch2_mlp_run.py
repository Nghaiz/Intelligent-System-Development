"""MLP cùng cấu hình trên ba nền tảng, ba seed: python scripts/ch2_mlp_run.py numpy|torch|keras <dataset>"""
import sys
import time

import numpy as np

from tlf.ch2_data import TITLES, get_split
from tlf.ch2_metrics import best_threshold, bootstrap_ci, clf_metrics, reg_metrics
from tlf.ch2_mlp import CFG, N_SEEDS, SEEDS, class_weights, fit_torch, scale_target
from tlf.common import device, limit_threads, model_dir, save_json
from tlf.gpu import cap, heavy
from tlf.mem import require_disk, slot
from tlf.runlog import log_run

GB = {"numpy": 1.5, "torch": 2.5, "keras": 3.5}


def run_numpy(s, y, yv, n_out, w, cfg, seed):
    from tlf.ch2_mlp_numpy import NumpyMLP
    net = NumpyMLP(s.X_tr.shape[1], n_out, cfg["hidden"], cfg["drop"], seed)
    sw = None if w is None else w[y]
    hist = net.fit(s.X_tr, y, s.X_va, yv, sw, cfg["lr"], cfg["epochs"], cfg["bs"], cfg["patience"])
    return net.predict_raw, hist


def run_torch(s, y, yv, n_out, w, cfg, seed):
    import torch
    dev = device()
    net, hist = fit_torch(s.X_tr, y, s.X_va, yv, n_out, w, cfg, seed, dev)

    def predict(X):
        with torch.no_grad():
            return np.concatenate([net(torch.as_tensor(X[i:i + 8192], device=dev)).cpu().numpy()
                                   for i in range(0, len(X), 8192)])
    return predict, hist


def run_keras(s, y, yv, n_out, w, cfg, seed):
    import keras
    keras.utils.set_random_seed(seed)
    layers = [keras.Input((s.X_tr.shape[1],))]
    for h in cfg["hidden"]:
        layers += [keras.layers.Dense(h, activation="relu"), keras.layers.Dropout(cfg["drop"])]
    net = keras.Sequential([*layers, keras.layers.Dense(n_out)])
    loss = "mse" if n_out == 1 else keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    net.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss=loss)
    stop = keras.callbacks.EarlyStopping(patience=cfg["patience"], min_delta=1e-5, restore_best_weights=True)
    cw = None if w is None else {i: float(v) for i, v in enumerate(w)}
    h = net.fit(s.X_tr, y, validation_data=(s.X_va, yv), epochs=cfg["epochs"], batch_size=cfg["bs"],
                class_weight=cw, callbacks=[stop], verbose=0)
    hist = [{"epoch": i + 1, "train_loss": a, "val_loss": b}
            for i, (a, b) in enumerate(zip(h.history["loss"], h.history["val_loss"]))]
    return (lambda X: net.predict(X, batch_size=8192, verbose=0)), hist


RUN = {"numpy": run_numpy, "torch": run_torch, "keras": run_keras}


def softmax(z):
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def one_seed(fw, ds, s, seed):
    """Một lượt huấn luyện; trả chỉ số test. Hồi quy học trên y chuẩn hóa rồi quy đổi về tỷ đồng."""
    cfg, reg = CFG[ds], s.task == "reg"
    y, yv = s.y_tr, s.y_va
    mu, sd = scale_target(y) if reg else (0.0, 1.0)
    if reg:
        y, yv = (y - mu) / sd, (yv - mu) / sd
    w = class_weights(y) if ds == "vsfc" else None
    n_out = 1 if reg else len(s.labels)
    t0 = time.perf_counter()
    predict, hist = RUN[fw](s, y, yv, n_out, w, cfg, seed)
    sec = time.perf_counter() - t0
    if reg:
        pred = predict(s.X_te)[:, 0] * sd + mu
        return reg_metrics(s.y_te, pred), pred, sec, hist, None
    proba, thr = softmax(predict(s.X_te)), None
    if ds == "cdc":
        thr = best_threshold(s.y_va, softmax(predict(s.X_va))[:, 1])
        pred = (proba[:, 1] >= thr).astype(int)
    else:
        pred = proba.argmax(1)
    return clf_metrics(s.y_te, pred, proba), pred, sec, hist, thr


def main(fw, ds):
    if fw == "keras":
        import tensorflow  # noqa: F401  nạp trước để limit_threads cấu hình được TF
    limit_threads(4)
    if fw == "torch":
        cap(0.12)
    s = get_split(ds, dense=True)
    rows, last = [], None
    for seed in SEEDS[:N_SEEDS[ds]]:
        m, pred, sec, hist, thr = one_seed(fw, ds, s, seed)
        m["ci_lo"], m["ci_hi"] = bootstrap_ci(np.asarray(s.y_te), np.asarray(pred), s.task)
        rows.append({**m, "seed": seed, "seconds": sec, "epochs_run": len(hist), "threshold": thr})
        last = (pred, hist)
        dev = device() if fw == "torch" else "cpu"
        log_run("ch2", TITLES[ds], f"MLP {fw}", fw, {k: v for k, v in m.items() if not k.startswith("ci")}, sec,
                dev, {"seed": seed, **{k: str(v) for k, v in CFG[ds].items()}})
        print(ds, fw, seed, {k: round(float(v), 4) for k, v in m.items()}, f"{sec:.1f}s {len(hist)}ep", flush=True)
    require_disk(0.1, f"ch2 lưu dự đoán {ds}")
    np.save(model_dir("ch2") / f"pred_{ds}_mlp_{fw}.npy", last[0])
    sizes = [s.X_tr.shape[1], *CFG[ds]["hidden"], 1 if s.task == "reg" else len(s.labels)]
    n = sum(a * b + b for a, b in zip(sizes, sizes[1:]))
    save_json(f"ch2_res_{ds}_mlp_{fw}", {"rows": rows, "history": last[1], "n_params": n})


if __name__ == "__main__":
    fw, ds = sys.argv[1], sys.argv[2]
    with slot(GB[fw], f"ch2 mlp {fw} {ds}"):
        if fw == "torch":
            with heavy(f"ch2 mlp torch {ds}"):
                main(fw, ds)
        else:
            main(fw, ds)
