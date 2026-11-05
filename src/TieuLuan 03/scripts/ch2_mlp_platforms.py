"""Đối chuẩn cùng một MLP trên ba nền tảng (NumPy, Keras, PyTorch) và ba tập dữ liệu của Chương 2.
Diabetes và House dùng lại kết quả PyTorch đã có; Comments huấn luyện mới trên TF-IDF + SVD 300 chiều.
Ghi outputs/metrics/ch2_platforms.json và generated/ch2_platforms.tex."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import limit_threads, seed_all  # noqa: E402

limit_threads(8)
seed_all()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from tl3 import ch2_data as cd  # noqa: E402
from tl3.ch2_metrics import best_threshold, clf_metrics, reg_metrics  # noqa: E402
from tl3.ch2_mlp import fit_mlp, proba  # noqa: E402
from tl3.ch2_mlp_numpy import NumpyMLP  # noqa: E402
from tl3.ch2_text import tfidf  # noqa: E402
from tl3.common import load_json, save_json  # noqa: E402
from tl3.gpu import cap  # noqa: E402
from tl3.texfmt import num, table  # noqa: E402

CLF_CFG = dict(hidden=(128, 64, 32), dropout=0.2, lr=1e-3, epochs=60, bs=512, patience=8)
REG_CFG = dict(hidden=(64, 32), dropout=0.1, lr=2e-3, epochs=300, bs=64, patience=25)


def keras_mlp(n_in, hidden, dropout, lr, task):
    """Cùng kiến trúc với TabularMLP của PyTorch: Dense, BatchNorm, ReLU, Dropout."""
    net = tf.keras.Sequential([tf.keras.layers.Input((n_in,))])
    for h in hidden:
        net.add(tf.keras.layers.Dense(h))
        net.add(tf.keras.layers.BatchNormalization())
        net.add(tf.keras.layers.ReLU())
        net.add(tf.keras.layers.Dropout(dropout))
    net.add(tf.keras.layers.Dense(1))
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) if task == "clf" else "mse"
    net.compile(tf.keras.optimizers.AdamW(lr, weight_decay=1e-4), loss)
    return net


def fit_keras(X, y, Xv, yv, task, cfg, pos_weight=None):
    net = keras_mlp(X.shape[1], cfg["hidden"], cfg["dropout"], cfg["lr"], task)
    cbs = [tf.keras.callbacks.EarlyStopping(patience=cfg["patience"], restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)]
    cw = {0: 1.0, 1: pos_weight} if pos_weight else None
    t0 = time.perf_counter()
    h = net.fit(X, y, validation_data=(Xv, yv), epochs=cfg["epochs"], batch_size=cfg["bs"], callbacks=cbs,
                class_weight=cw, verbose=0)
    return net, time.perf_counter() - t0, len(h.history["loss"])


def fit_numpy(X, y, Xv, yv, task, cfg, pos_weight=None):
    net = NumpyMLP(X.shape[1], cfg["hidden"], task, pos_weight or 1.0)
    t0 = time.perf_counter()
    ep = net.fit(X, y, Xv, yv, cfg["lr"], cfg["epochs"], cfg["bs"], cfg["patience"])
    return net, time.perf_counter() - t0, ep


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def clf_report(y_va, p_va, y_te, p_te, train_s, epochs):
    thr = best_threshold(y_va, p_va)[0]
    m = clf_metrics(y_te, p_te, thr)
    return {"acc": m["acc"], "f1": m["f1"], "auc": m["auc"], "macro_f1": f1_score(y_te, p_te >= thr, average="macro"),
            "train_s": train_s, "epochs": epochs}


def run_clf(Xtr, ytr, Xva, yva, Xte, yte, pos_weight):
    out = {}
    net, s, ep = fit_numpy(Xtr, ytr, Xva, yva, "clf", CLF_CFG, pos_weight)
    out["numpy"] = clf_report(yva, sigmoid(net.predict_raw(Xva)), yte, sigmoid(net.predict_raw(Xte)), s, ep)
    print("numpy", out["numpy"], flush=True)
    net, s, ep = fit_keras(Xtr, ytr, Xva, yva, "clf", CLF_CFG, pos_weight)
    pv, pt = (sigmoid(net.predict(x, batch_size=8192, verbose=0)[:, 0]) for x in (Xva, Xte))
    out["keras"] = clf_report(yva, pv, yte, pt, s, ep)
    print("keras", out["keras"], flush=True)
    return out


def diabetes():
    _, (X_tr, X_va, X_te, y_tr, y_va, y_te) = cd.diabetes_splits()
    pre = cd.diabetes_preprocessor().fit(X_tr)
    Xs = [pre.transform(x).astype(np.float32) for x in (X_tr, X_va, X_te)]
    d = load_json("ch2_diabetes")
    ys = [np.asarray(y, np.float32) for y in (y_tr, y_va, y_te)]
    out = run_clf(Xs[0], ys[0], Xs[1], ys[1], Xs[2], ys[2], None)  # PyTorch cũng không dùng pos_weight ở tập này
    t, m = d["test"]["mlp"], d["models"]["mlp"]
    out["torch"] = {"acc": t["acc"], "f1": t["f1"], "auc": t["auc"], "train_s": m["train_s"], "epochs": m["epochs"]}
    return out


def comments():
    """TF-IDF thưa được nén bằng SVD xuống 300 chiều để cả ba nền tảng dùng chung một ma trận đặc."""
    _, (X_tr, X_va, X_te, y_tr, y_va, y_te) = cd.reviews_splits()
    vec = tfidf(20000).fit(X_tr)
    svd = TruncatedSVD(300, random_state=42).fit(vec.transform(X_tr))
    Xs = [svd.transform(vec.transform(x)).astype(np.float32) for x in (X_tr, X_va, X_te)]
    Xs = [(x - Xs[0].mean(0)) / Xs[0].std(0) for x in Xs]
    ys = [np.asarray(y, np.float32) for y in (y_tr, y_va, y_te)]
    # lớp cần bắt là "không giới thiệu": đảo nhãn để pos_weight bù đúng lớp thiểu số
    pw = float((ys[0] == 1).sum() / (ys[0] == 0).sum())
    out = run_clf(Xs[0], 1 - ys[0], Xs[1], 1 - ys[1], Xs[2], 1 - ys[2], pw)
    cap(0.15)
    t0 = time.perf_counter()
    net, info = fit_mlp(Xs[0], 1 - ys[0], Xs[1], 1 - ys[1], task="clf", pos_weight=pw, hidden=CLF_CFG["hidden"],
                        dropout=CLF_CFG["dropout"], lr=CLF_CFG["lr"], epochs=CLF_CFG["epochs"], bs=CLF_CFG["bs"],
                        patience=CLF_CFG["patience"])
    out["torch"] = clf_report(1 - ys[1], proba(net, Xs[1]), 1 - ys[2], proba(net, Xs[2]),
                              time.perf_counter() - t0, info["epochs_run"])
    print("torch", out["torch"], flush=True)
    return out


def house():
    _, (X_tr, X_va, X_te, y_tr, y_va, y_te) = cd.house_splits()
    pre = cd.house_preprocessor().fit(X_tr)
    Xs = [pre.transform(x).astype(np.float32) for x in (X_tr, X_va, X_te)]
    mu, sd = y_tr.mean(), y_tr.std()
    z_tr, z_va = ((y_tr - mu) / sd).astype(np.float32), ((y_va - mu) / sd).astype(np.float32)
    out = {}
    net, s, ep = fit_numpy(Xs[0], z_tr, Xs[1], z_va, "reg", REG_CFG)
    out["numpy"] = {**reg_metrics(y_te, net.predict_raw(Xs[2]) * sd + mu), "train_s": s, "epochs": ep}
    print("numpy", out["numpy"], flush=True)
    net, s, ep = fit_keras(Xs[0], z_tr, Xs[1], z_va, "reg", REG_CFG)
    out["keras"] = {**reg_metrics(y_te, net.predict(Xs[2], verbose=0)[:, 0] * sd + mu), "train_s": s, "epochs": ep}
    print("keras", out["keras"], flush=True)
    d = load_json("ch2_house")
    out["torch"] = {**d["test"]["mlp"], "train_s": d["models"]["mlp"]["train_s"], "epochs": d["models"]["mlp"]["epochs"]}
    return out


def tables(res):
    names = {"numpy": "NumPy (từ đầu)", "keras": "Keras", "torch": "PyTorch"}
    rows = []
    for k, n in names.items():
        a, b, c = res["diabetes"][k], res["house"][k], res["comments"][k]
        rows.append([n, num(a["f1"], 4), num(a["auc"], 4), num(a["train_s"], 1), num(b["r2"], 4), num(b["mae"] / 1000, 1),
                     num(b["train_s"], 1), num(c["f1"], 4), num(c["auc"], 4), num(c["train_s"], 1)])
    table("ch2_platforms", "lrrrrrrrrr",
          ["Nền tảng", "F1", "AUC", "Giây", "$R^2$", "MAE (nghìn)", "Giây", "F1", "AUC", "Giây"], rows,
          note="Ba cột đầu là tiểu đường, ba cột tiếp theo là giá nhà, ba cột cuối là nhận xét (F1 của lớp không giới thiệu). "
               "Bản NumPy không có BatchNorm và Dropout; PyTorch chạy GPU, hai nền tảng kia chạy CPU.")


def main():
    res = {"diabetes": diabetes(), "house": house(), "comments": comments()}
    save_json("ch2_platforms", res)
    tables(res)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
