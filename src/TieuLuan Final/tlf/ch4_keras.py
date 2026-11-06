"""Chương 4: mô hình Keras tương đương bản PyTorch (TensorFlow trên Windows chỉ chạy CPU)."""
import time

import numpy as np

from .ch4_torch import config_of

MAX_EPOCHS = 15
PATIENCE = 3


def build_keras_model(window, n_features, horizon, arch="lstm", hidden=64, dropout=0.2, lr=1e-3, rec_drop=0.0):
    """RNN, LSTM, GRU hoặc BiLSTM → Dropout → Dense(H); rec_drop bật recurrent dropout trên cổng hồi quy."""
    import keras
    from keras import layers
    kw = {"recurrent_dropout": rec_drop}
    rec = {"rnn": lambda: layers.SimpleRNN(hidden, **kw), "lstm": lambda: layers.LSTM(hidden, **kw),
           "gru": lambda: layers.GRU(hidden, **kw),
           "bilstm": lambda: layers.Bidirectional(layers.LSTM(hidden, **kw))}[arch]
    model = keras.Sequential([keras.Input((window, n_features)), rec(), layers.Dropout(dropout), layers.Dense(horizon)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss="mse")
    return model


def fit_keras(prep, arch, seed=42, rec_drop=0.0):
    """Dừng sớm và khôi phục trọng số tốt nhất. Trả (mô hình, lịch sử, giây)."""
    import keras
    keras.utils.set_random_seed(seed)
    cfg, s = config_of(prep.spec), prep.spec
    ts, tv = prep.last_index("train"), prep.last_index("val")
    Xtr, ytr, Xva, yva = prep.windows(ts), prep.targets(ts), prep.windows(tv), prep.targets(tv)
    model = build_keras_model(s.window, prep.x.shape[1], s.horizon, arch, cfg["hidden"], cfg["dropout"], cfg["lr"], rec_drop)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    t0 = time.perf_counter()
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=MAX_EPOCHS, batch_size=cfg["batch"], shuffle=True,
                  callbacks=[stop], verbose=0)
    return model, {k: [float(v) for v in h.history[k]] for k in ("loss", "val_loss")}, time.perf_counter() - t0


def predict_keras(model, X):
    return np.asarray(model.predict(X, batch_size=1024, verbose=0))
