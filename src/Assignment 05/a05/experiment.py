"""Mục 4 + 5 — một lượt thí nghiệm hoàn chỉnh cho một tập: huấn luyện đủ bốn mô hình,
lưu checkpoint, ghi đường học và số liệu. Notebook 03/04/05 chỉ gọi hai hàm ở đây,
nên ba tập được xử lý theo đúng cùng một quy trình."""
import numpy as np
import torch

from . import metrics as M
from .export import save_metrics, write_dat
from .models import MODEL_NAMES
from .train import (IMAGE_CFG, TAB_CFG, predict, save_checkpoint, tabular_batches,
                    train_image_model, train_tabular_model)


def _history_dat(tag, history):
    write_dat(f"{tag}_history", {
        "epoch": [h["epoch"] for h in history],
        "train_loss": [h["train_loss"] for h in history],
        "val_loss": [h["val_loss"] for h in history],
        "train_acc": [100 * h["train_acc"] for h in history],
        "val_acc": [100 * h["val_acc"] for h in history]})


def run_image_experiment(ds_name, data, log=print):
    """Huấn luyện bốn mô hình 2D trên một tập CIFAR đã nạp lên GPU; trả về dict số liệu."""
    dev = data.device
    results = {}
    for name in MODEL_NAMES:
        log(f"== {ds_name} / {name}")
        model, res = train_image_model(name, data, log=log)
        tag = f"{ds_name}_{name}"
        _history_dat(tag, res["history"])
        logits, y = predict(model, data.batches("test", 1000), amp=True)
        x_probe = data.normalize(data.x["test"][:256])
        with torch.autocast("cuda", dtype=torch.float16):
            infer_ms = M.time_inference(model, x_probe)
        r = M.classification_report(logits, y)
        r.update(params=M.count_params(model), macs=M.count_macs(model, x_probe),
                 epoch_s=res["epoch_seconds"], infer_ms=infer_ms, best_epoch=res["best_epoch"],
                 epochs_run=len(res["history"]),
                 best_val_acc=max(h["val_acc"] for h in res["history"]))
        results[name] = r
        save_checkpoint(tag, res["best_state"], {"dataset": ds_name, "model": name, "config": IMAGE_CFG,
                                                 "device": torch.cuda.get_device_name(dev), **r})
        log(f"   test acc {r['acc']:.4f}  top5 {r['top5']:.4f}  macro-F1 {r['macro_f1']:.4f}")
    save_metrics(ds_name, {"models": results, "config": IMAGE_CFG,
                           "device": torch.cuda.get_device_name(dev),
                           "n_train": len(data.y["train"]), "n_val": len(data.y["val"]),
                           "n_test": len(data.y["test"])})
    return results


def run_tabular_experiment(D, log=print):
    """Huấn luyện bốn mô hình 1D trên Diabetes bằng CPU; ngưỡng chọn trên val theo F1."""
    results = {}
    for name in MODEL_NAMES:
        log(f"== diabetes / {name}")
        model, res = train_tabular_model(name, D, log=log)
        tag = f"diabetes_{name}"
        _history_dat(tag, res["history"])
        prob = {}
        for split in ("val", "test"):
            out, _ = predict(model, tabular_batches(D["X"][split], D["y"][split], 4096))
            prob[split] = 1 / (1 + np.exp(-out.reshape(-1)))
        y_val, y_test = D["y"]["val"].astype(int), D["y"]["test"].astype(int)
        thr = M.best_threshold(prob["val"], y_val)
        r = M.binary_report(prob["test"], y_test, thr)
        x_probe = torch.as_tensor(D["X"]["test"][:4096])[:, None, :]
        r.update(params=M.count_params(model), macs=M.count_macs(model, x_probe),
                 epoch_s=res["epoch_seconds"], infer_ms=M.time_inference(model, x_probe),
                 best_epoch=res["best_epoch"], epochs_run=len(res["history"]))
        results[name] = r
        c = M.binary_curves(prob["test"], y_test)
        write_dat(f"{tag}_roc", {"fpr": c["fpr"], "tpr": c["tpr"]})
        write_dat(f"{tag}_pr", {"recall": c["recall"], "precision": c["precision"]})
        save_checkpoint(tag, res["best_state"], {"dataset": "diabetes", "model": name, "config": TAB_CFG,
                                                 "device": "cpu", **r})
        log(f"   test F1 {r['f1']:.4f}  ROC-AUC {r['roc_auc']:.4f}  ngưỡng {thr:.3f}")
    y_test = D["y"]["test"]
    save_metrics("diabetes", {"models": results, "config": TAB_CFG, "device": "cpu",
                              "majority_acc": float(max(y_test.mean(), 1 - y_test.mean())),
                              "n_train": len(D["y"]["train"]), "n_val": len(D["y"]["val"]),
                              "n_test": len(D["y"]["test"])})
    return results
