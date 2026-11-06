"""Gom kết quả Chương 3 thành ch3_datasets.json, ch3_compare.json, ch3_summary.json và ảnh mẫu.

    python scripts/ch3_collect.py [--data-only]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.ch3_common import ARCHS, ARCH_NAME, DS_IDS, EXAMPLES, load_proba, load_result  # noqa: E402
from tlf.ch3_data import DATASETS, labels, load, raw_sizes, splits  # noqa: E402
from tlf.ch3_models import MobileNetV2  # noqa: E402
from tlf.ch3_scratch import gradient_check  # noqa: E402
from tlf.common import limit_threads, save_json  # noqa: E402

KEYS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "params", "seconds", "infer_ms", "epochs")
N_EXAMPLES = 4
FORWARD_KEYS = ("acc_numpy", "acc_torch", "agree", "n_img", "numpy_net_ms_per_img", "torch_cpu_ms_per_img",
                "loop_ms_per_img", "im2col_ms_per_img")


def write_examples(ds):
    """Ảnh mẫu cho Web App: ảnh test thật, giữ nguyên kích thước, các lớp khác nhau; ResNet-18 đoán đúng."""
    x, y = load(ds, "test")
    ok = load_proba(f"{ds}__resnet18").argmax(1) == y
    rows = []
    for c in np.linspace(0, len(labels(ds)) - 1, N_EXAMPLES).astype(int):
        i = int(np.flatnonzero((y == c) & ok)[0])
        name = f"{ds}_{i}_lop{c}.png"
        Image.fromarray(np.array(x[i])).save(EXAMPLES / name)
        rows.append({"file": name, "label": labels(ds)[c], "split": "test", "index": i})
    return rows


def describe(ds):
    """Mô tả một bộ theo khuôn chN_datasets.json và các số phụ cho báo cáo."""
    d, names = DATASETS[ds], labels(ds)
    ys = {s: load(ds, s)[1] for s in splits(ds)}
    counts = np.bincount(np.concatenate(list(ys.values())), minlength=len(names))
    x, _ = load(ds, "train")
    a = np.asarray(x[:: max(1, len(x) // 5000)], np.float32) / 255
    return {"id": ds, "name": d["name"], "source": d["source"], "url": d["url"], "task": d["task"],
            "n_samples": int(counts.sum()), "n_features": d["size"] ** 2 * 3,
            "size_mb": round(raw_sizes()[ds], 1), "license": d["license"], "note": d["note"],
            "reused_from": d["reused_from"], "image_size": d["size"], "n_classes": len(names),
            "splits": {s: len(ys[s]) for s in ys}, "sample_rows": write_examples(ds),
            "distribution": {"labels": names, "values": counts.tolist()},
            "imbalance": float(counts.max() / counts.min()), "largest": names[int(counts.argmax())],
            "smallest": names[int(counts.argmin())], "largest_pct": float(counts.max() / counts.sum()),
            "smallest_pct": float(counts.min() / counts.sum()),
            "mean": a.mean((0, 1, 2)).tolist(), "std": a.std((0, 1, 2)).tolist()}


def row(res, model=None):
    return {"dataset": res["dataset"], "model": model or res["model"], "framework": res["framework"],
            "reused_from": res.get("reused_from"), **{k: res.get(k) for k in KEYS}}


def top_confusion(res, names):
    """Cặp lớp bị nhầm nhiều nhất theo tỉ lệ trong lớp thật."""
    cm = np.array(res["cm"], float)
    rate = cm / cm.sum(1, keepdims=True)
    np.fill_diagonal(rate, 0)
    i, j = np.unravel_index(rate.argmax(), rate.shape)
    return {"true": names[i], "pred": names[j], "rate": float(rate[i, j]), "count": int(cm[i, j])}


def overfit_gap(res):
    """Chênh lệch val_loss cuối so với val_loss nhỏ nhất: lớn nghĩa là mô hình quá khớp về cuối."""
    v = res["history"]["val_loss"]
    return float(v[-1] - min(v))


def summarize_ds(ds):
    """Số liệu chữ của một bộ cho macro \\SL{ch3_summary/<ds>/...}."""
    names = labels(ds)
    runs = {a: r for a in ARCHS if (r := load_result(f"{ds}__{a}"))}
    best = max(runs, key=lambda a: runs[a]["f1_macro"])
    out = {"best_arch": ARCH_NAME[best], "best_f1": runs[best]["f1_macro"],
           "best_acc": runs[best]["accuracy"], "confusion": top_confusion(runs[best], names), "n_models": len(runs)}
    for a, r in runs.items():
        out[a] = {"acc": r["accuracy"], "f1": r["f1_macro"], "params": r["params"], "seconds": r["seconds"],
                  "sec_epoch": r["seconds"] / r["epochs"], "ms": r["infer_ms"], "epochs": r["epochs"],
                  "overfit": overfit_gap(r), "train_acc": r["history"]["acc"][-1],
                  "last_val": r["history"]["val_acc"][-1]}
    out["gain_pct"] = 100 * (runs["resnet18"]["accuracy"] - runs["custom"]["accuracy"])
    for mid in ("dummy", "softmax"):
        r = load_result(f"{ds}__baseline_{mid}")
        out[mid] = {"acc": r["accuracy"], "f1": r["f1_macro"]}
    return out


def rf(layers):
    """Vùng cảm nhận qua dãy (kernel, stride): r tăng (k - 1)·j, j nhân với stride."""
    r, j = 1, 1
    for k, s in layers:
        r, j = r + (k - 1) * j, j * s
    return r


def receptive_fields():
    """Dãy (kernel, stride) của tầng tích chập cuối, đúng cấu hình cho ảnh 32×32."""
    custom = ([(3, 1), (3, 1), (2, 2)] * 3)[:-1]
    res = [(3, 1)] + [(3, 1)] * 4
    for _ in range(3):
        res += [(3, 2), (3, 1)] + [(3, 1)] * 2
    mob = [(3, 1)]
    for _, _, n, s in MobileNetV2.CFG:
        mob += [(3, s)] + [(3, 1)] * (n - 1)
    return {"custom": rf(custom), "resnet18": rf(res), "mobilenetv2": rf(mob)}


def platform_rows():
    """Cùng một CNN cơ bản qua các nền tảng: PyTorch, Keras của Tiểu luận 01 và lượt NumPy ngắn nếu có."""
    out = {}
    for ds in ("svhn", "eurosat"):
        a, b = load_result(f"{ds}__custom"), load_result(f"{ds}__keras_custom")
        out[ds] = {"torch_acc": a["accuracy"], "keras_acc": b["accuracy"], "torch_s": a["seconds"],
                   "keras_s": b["seconds"], "ratio": b["seconds"] / a["seconds"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    args = ap.parse_args()
    limit_threads(2)
    meta = [describe(ds) for ds in DS_IDS]
    save_json("ch3_datasets", {"datasets": meta})
    if args.data_only:
        return
    rows = []
    for ds in DS_IDS:
        rows += [row(load_result(f"{ds}__baseline_{m}")) for m in ("dummy", "softmax")]
        rows += [row(r) for a in ARCHS if (r := load_result(f"{ds}__{a}"))]
        rows += [row(r) for r in (load_result(f"{ds}__keras_custom"),) if r]
    rows += [row(r) for f in ("numpy", "pytorch") if (r := load_result(f"svhn__platform_{f}"))]
    save_json("ch3_compare", {"rows": rows, "primary": "f1_macro"})
    f = load_result("svhn__numpy_forward")
    nums = {f"platform_{k}": load_result(f"svhn__platform_{k}") for k in ("numpy", "pytorch")}
    short = {k: {"acc": r["accuracy"], "f1": r["f1_macro"], "sec_epoch": r["seconds"] / r["epochs"],
                 "n": r["n_train"], "epochs": r["epochs"]} for k, r in nums.items() if r}
    grad = gradient_check()
    save_json("ch3_summary", {**{ds: summarize_ds(ds) for ds in DS_IDS}, "platform": platform_rows(),
                              "short": short, "grad_check_max": max(grad.values(), default=None),
                              "numpy_forward": {k: f[k] for k in FORWARD_KEYS},
                              "rf": receptive_fields()})


if __name__ == "__main__":
    main()
