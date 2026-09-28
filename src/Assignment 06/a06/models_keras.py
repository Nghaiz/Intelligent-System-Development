"""Hạng mục 4 (Ch.4): bốn mô hình giống hệt bản PyTorch, viết bằng Keras (backend TensorFlow, CPU).

TensorFlow chỉ được import trong tệp này, để các module khác chạy được khi máy không có TF.
Cùng tối ưu, cùng lô, cùng dừng sớm, cùng dữ liệu đã chuẩn hoá như PyTorch.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")    # TF ≥ 2.11 trên Windows vốn chỉ chạy CPU

import numpy as np
import keras
import tensorflow as tf
from keras import layers

from .export import MODELS
from .models_torch import CELLS, HEAD, HIDDEN
from .train_torch import CFG, OPT, SEED


def configure_tf(threads):
    """Phải gọi trước khi TF tạo phép tính đầu tiên."""
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, threads // 8))


def build_keras(cell, task, T, n_features=8):
    """Input(T, D) -> tầng hồi quy 64 -> Dense(32, relu) -> Dense(1) (logit nếu phân loại)."""
    rnn = {"rnn": lambda: layers.SimpleRNN(HIDDEN),
           "lstm": lambda: layers.LSTM(HIDDEN),
           "gru": lambda: layers.GRU(HIDDEN),                        # reset_after=True: 2 vector bias như PyTorch
           "bilstm": lambda: layers.Bidirectional(layers.LSTM(HIDDEN))}[cell]
    model = keras.Sequential([layers.Input((T, n_features)), rnn(),
                              layers.Dense(HEAD, activation="relu"), layers.Dense(1)],
                             name=f"keras_{cell}")
    loss = (keras.losses.MeanSquaredError() if task == "reg"
            else keras.losses.BinaryCrossentropy(from_logits=True))
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=OPT["lr"], weight_decay=OPT["weight_decay"],
                                                   clipnorm=OPT["clip"]), loss=loss)
    return model


def sample_weights(task, y, y_train):
    """Trọng số mẫu tương đương pos_weight của BCEWithLogitsLoss: dương nặng (âm/dương) lần."""
    if task == "reg":
        return None
    pw = (len(y_train) - y_train.sum()) / y_train.sum()
    return np.where(y > 0.5, pw, 1.0).astype(np.float32)


class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times = getattr(self, "times", []) + [time.perf_counter() - self.t0]


def fit_keras(cell, data, cfg):
    keras.utils.set_random_seed(SEED)
    Xtr, ytr, Xva, yva = data["X"]["train"], data["y"]["train"], data["X"]["val"], data["y"]["val"]
    model = build_keras(cell, cfg["task"], Xtr.shape[1], Xtr.shape[2])
    sw_tr = sample_weights(cfg["task"], ytr, ytr)
    sw_va = sample_weights(cfg["task"], yva, ytr)
    val = (Xva, yva) if sw_va is None else (Xva, yva, sw_va)
    timer = EpochTimer()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                         restore_best_weights=True)
    h = model.fit(Xtr, ytr, sample_weight=sw_tr, validation_data=val, epochs=cfg["epochs"],
                  batch_size=cfg["batch"], shuffle=True, callbacks=[stop, timer], verbose=0)
    n = len(h.history["loss"])
    hist = {"epoch": list(range(1, n + 1)), "train_loss": h.history["loss"],
            "val_loss": h.history["val_loss"], "epoch_s": timer.times,
            "best_epoch": int(np.argmin(h.history["val_loss"])) + 1}
    return model, hist


def predict_keras(model, X, batch=4096):
    return np.concatenate([model(X[i:i + batch], training=False).numpy().ravel()
                           for i in range(0, len(X), batch)])


def run_keras(dataset, data):
    """Huấn luyện 4 mô hình Keras của một tập, lưu .keras, trả về cùng cấu trúc như run_torch."""
    cfg = CFG[dataset]
    out = {}
    for cell in CELLS:
        model, hist = fit_keras(cell, data, cfg)
        path = MODELS / f"{dataset}_keras_{cell}.keras"
        model.save(path)
        out[cell] = {"history": hist, "params": int(model.count_params()), "device": "cpu",
                     "pred_val": predict_keras(model, data["X"]["val"]),
                     "pred_test": predict_keras(model, data["X"]["test"]), "file": path.name}
        print(f"{dataset} keras {cell:6s} epoch tốt {hist['best_epoch']:2d}/{len(hist['epoch'])} "
              f"val {min(hist['val_loss']):.4f} · {np.mean(hist['epoch_s']):.2f} s/epoch trên cpu")
    return out
