"""Chương 3 — CNN cơ bản bằng Keras chạy CPU (TensorFlow trên Windows không còn hỗ trợ GPU).

TensorFlow tốn 1–2 GB bộ nhớ commit khi nạp nên chỉ import trong hàm, không bao giờ ở cấp mô-đun.
"""
import time


def build_basic(size, n_classes):
    """Conv32 → Pool → Conv64 → Pool → Dense128 → Dense(K), đúng kiến trúc bản NumPy và PyTorch."""
    import keras
    from keras import layers
    return keras.Sequential([
        layers.Input((size, size, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dense(n_classes),
    ], name="basic_cnn")


def train_keras(model, xtr, ytr, xva, yva, epochs, batch, log=print):
    """Adam 1e-3, cross-entropy trên logits; Callback ghi lịch sử và thời gian từng epoch."""
    import keras

    class Hist(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.h = {k: [] for k in ("loss", "acc", "val_loss", "val_acc", "time")}

        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.h["time"].append(time.perf_counter() - self.t0)
            for k, src in (("loss", "loss"), ("acc", "accuracy"), ("val_loss", "val_loss"),
                           ("val_acc", "val_accuracy")):
                self.h[k].append(float(logs[src]))
            log(f"epoch {epoch + 1}/{epochs} loss {logs['loss']:.4f} val_acc {logs['val_accuracy']:.4f} "
                f"{self.h['time'][-1]:.1f}s")

    model.compile(keras.optimizers.Adam(1e-3), keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    cb = Hist()
    model.fit(xtr, ytr, validation_data=(xva, yva), epochs=epochs, batch_size=batch, callbacks=[cb],
              verbose=0, shuffle=True)
    return cb.h
