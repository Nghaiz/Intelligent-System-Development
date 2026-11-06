"""Huấn luyện phần còn thiếu của Chương 3; mỗi lượt có kết quả rồi thì bỏ qua nên chạy lại an toàn sau sự cố.

    python scripts/ch3_train.py --fw torch                       bốn kiến trúc trên PathMNIST, GPU
    python scripts/ch3_train.py --fw platform-torch|numpy|keras  so ba nền tảng trên tập con SVHN
    python scripts/ch3_train.py --fw baseline                    Dummy và hồi quy softmax trên ba bộ
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.ch3_common import (ARCH_NAME, CKPT, DS_IDS, FULL, MODELS, PLATFORM, evaluate,  # noqa: E402
                            load_result, save_result)
from tlf.ch3_data import DATASETS, load, net_input  # noqa: E402
from tlf.common import LOGS, limit_threads, seed_all  # noqa: E402
from tlf.gpu import heavy  # noqa: E402
from tlf.mem import require_disk, slot  # noqa: E402
from tlf.runlog import log_run  # noqa: E402


def logger(name):
    f = open(LOGS / f"ch3_train_{name}.log", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()
    return log


def finish(run, ds, model, framework, device, hist, params, proba, yte, ms, log, extra=None):
    """Tính chỉ số trên tập test, ghi kết quả lượt chạy và nhật ký thực nghiệm chung."""
    seconds = float(sum(hist["time"]))
    res = {"run": run, "dataset": ds, "model": model, "framework": framework, "params": params,
           "seconds": seconds, "epochs": len(hist["loss"]), "infer_ms": ms, "history": hist,
           "reused_from": None, **evaluate(yte, proba), **(extra or {})}
    require_disk(0.1, f"ch3 ghi {run}")
    save_result(run, res, proba)
    (CKPT / f"{run}.pt").unlink(missing_ok=True)        # đã có mô hình cuối, điểm kiểm tra hết tác dụng
    log_run("ch3", ds, model, framework, {k: res[k] for k in ("accuracy", "f1_macro")}, seconds, device,
            {"params": params, "epochs": res["epochs"]})
    log(f"== {run}: acc {res['accuracy']:.4f} f1 {res['f1_macro']:.4f} {seconds:.0f}s params {params}")


def run_torch(arch, log):
    """Một lượt PathMNIST theo công thức Tiểu luận 01; đủ tập train, đệm 28×28 lên 32×32."""
    import torch
    from tlf.ch3_models import build, n_params
    from tlf.ch3_torch import cpu_latency, predict_proba, train
    ds, run = "pathmnist", f"pathmnist__{arch}"
    if load_result(run):
        return
    seed_all()
    xtr, ytr = load(ds, "train")
    xva, yva = load(ds, "val")
    xte, yte = load(ds, "test")
    model = build(arch, len(DATASETS[ds]["classes"]))
    data = (net_input(xtr), ytr, net_input(xva), yva)   # chỉ hai mảng đệm này nằm trong RAM
    with slot(4, f"ch3 {run}"), heavy(f"ch3 {run}"):
        hist, mean, std = train(model, data, FULL, CKPT / f"{run}.pt", log=log)
        proba = predict_proba(model, net_input(xte), mean, std)
        ms = cpu_latency(model, (3, 32, 32))
        del data
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "kwargs": {}}, MODELS / f"{run}.pt")
    finish(run, ds, ARCH_NAME[arch], "PyTorch", "cuda", hist, n_params(model), proba, yte, ms, log,
           {"arch": arch})


def platform_data():
    """Đủ 73.257 ảnh train và 26.032 ảnh test của SVHN; tập test chỉ để theo dõi, không chọn mô hình."""
    xtr, ytr = load("svhn", "train")
    xte, yte = load("svhn", "test")
    return np.array(xtr), ytr, np.array(xte), yte, np.array(xte), yte


def run_platform_torch(log):
    from tlf.ch3_models import SimpleCNN, n_params
    from tlf.ch3_torch import cpu_latency, predict_proba, train
    run = "svhn__platform_pytorch"
    if load_result(run):
        return
    seed_all()
    xtr, ytr, xva, yva, xte, yte = platform_data()
    model = SimpleCNN(10, 32)
    with slot(3, f"ch3 {run}"), heavy(f"ch3 {run}"):
        hist, mean, std = train(model, (xtr, ytr, xva, yva), PLATFORM, CKPT / f"{run}.pt", log=log)
        proba = predict_proba(model, xte, mean, std)
        ms = cpu_latency(model, (3, 32, 32))
    finish(run, "svhn", "CNN hai tầng", "PyTorch", "cuda", hist, n_params(model), proba, yte, ms, log,
           {"n_train": len(xtr)})


class BatchNCHW:
    """Giữ ảnh ở uint8, chỉ đổi sang float32 NCHW cho từng lô; đổi cả tập một lần tốn hơn 8 GB commit."""

    def __init__(self, images):
        self.images = images
        self.shape = (len(images), images.shape[3], images.shape[1], images.shape[2])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return np.ascontiguousarray(self.images[index].transpose(0, 3, 1, 2), np.float32) / 255


def run_numpy(log):
    """Bản NumPy thuần trên CPU; huấn luyện bằng lan truyền ngược viết tay, kiểm đạo hàm bằng sai phân."""
    from tlf.ch3_scratch import ScratchCNN, gradient_check, train_scratch
    run = "svhn__platform_numpy"
    if load_result(run):
        return
    limit_threads(4)
    xtr, ytr, xva, yva, xte, yte = platform_data()
    with slot(4, f"ch3 {run}"):
        xtr_f, xva_f, xte_f = BatchNCHW(xtr), BatchNCHW(xva), BatchNCHW(xte)
        m = ScratchCNN(xtr_f.shape[1:], 10)
        hist = train_scratch(m, xtr_f, ytr, xva_f, yva, PLATFORM["epochs"], batch=PLATFORM["batch"], log=log)
        proba = m.predict_proba(xte_f)
        t0 = time.perf_counter()
        m.predict_proba(xte_f[:256], batch=1)
        ms = 1000 * (time.perf_counter() - t0) / 256
    np.savez(MODELS / f"{run}.npz", **m.p)
    finish(run, "svhn", "CNN hai tầng", "NumPy", "cpu", hist, m.n_params(), proba, yte, ms, log,
           {"n_train": len(xtr), "grad_check": gradient_check()})


def run_keras(log):
    from tlf.ch3_keras import build_basic, train_keras
    run = "svhn__platform_keras"
    if load_result(run):
        return
    xtr, ytr, xva, yva, xte, yte = platform_data()
    with slot(5, f"ch3 {run}"):
        import keras
        limit_threads(4)
        keras.utils.set_random_seed(42)
        m = build_basic(32, 10)
        hist = train_keras(m, xtr.astype(np.float32) / 255, ytr, xva.astype(np.float32) / 255, yva,
                           PLATFORM["epochs"], PLATFORM["batch"], log)
        xf = xte.astype(np.float32) / 255
        logits = np.asarray(m.predict(xf, batch_size=512, verbose=0))
        t0 = time.perf_counter()
        for i in range(30):
            m(xf[i:i + 1], training=False)
        ms = 1000 * (time.perf_counter() - t0) / 30
    m.save(MODELS / f"{run}.keras")
    z = np.exp(logits - logits.max(1, keepdims=True))
    proba = z / z.sum(1, keepdims=True)
    finish(run, "svhn", "CNN hai tầng", "Keras", "cpu", hist, int(m.count_params()), proba, yte, ms, log,
           {"n_train": len(xtr)})


def pooled(x, k):
    """Thu ảnh bằng trung bình khối k×k rồi duỗi thẳng, cho mô hình tuyến tính làm mốc so sánh."""
    n, h, w, c = x.shape
    a = np.asarray(x, np.float32).reshape(n, h // k, k, w // k, k, c).mean((2, 4))
    return a.reshape(n, -1) / 255


def run_baseline(ds, log):
    """Hai mốc: đoán lớp đông nhất, và hồi quy softmax trên ảnh thu nhỏ còn cạnh 14–16 điểm ảnh."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    limit_threads(4)
    xtr, ytr = load(ds, "train")
    xte, yte = load(ds, "test")
    k = DATASETS[ds]["size"] // 16 if DATASETS[ds]["size"] > 28 else 2
    xa, xb = pooled(xtr, k), pooled(xte, k)
    for mid, clf in (("dummy", DummyClassifier(strategy="most_frequent")),
                     ("softmax", LogisticRegression(max_iter=100, C=1.0))):
        run = f"{ds}__baseline_{mid}"
        if load_result(run):
            continue
        t0 = time.perf_counter()
        clf.fit(xa, ytr)
        hist = {"loss": [0.0], "time": [time.perf_counter() - t0]}
        n = sum(c.size for c in (clf.coef_, clf.intercept_)) if mid == "softmax" else 0
        name = {"dummy": "Lớp đông nhất", "softmax": "Hồi quy softmax"}[mid]
        finish(run, ds, name, "scikit-learn", "cpu", hist, int(n), clf.predict_proba(xb), yte, 0.0, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", required=True, choices=["torch", "platform-torch", "numpy", "keras", "baseline"])
    ap.add_argument("--arch", nargs="*", default=["custom", "resnet18", "mobilenetv2", "vgg11"])
    args = ap.parse_args()
    log = logger(args.fw)
    if args.fw in ("torch", "platform-torch"):
        limit_threads(4)
    if args.fw == "torch":
        for arch in args.arch:
            run_torch(arch, log)
            gc.collect()                                # trả mảng đệm và tensor của lượt trước trước lượt kế
    elif args.fw == "baseline":
        for ds in DS_IDS:
            run_baseline(ds, log)
    else:
        {"platform-torch": run_platform_torch, "numpy": run_numpy, "keras": run_keras}[args.fw](log)
    print(json.dumps({"xong": args.fw}), flush=True)


if __name__ == "__main__":
    main()
