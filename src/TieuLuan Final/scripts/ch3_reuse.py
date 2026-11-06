"""Nhập kết quả Tiểu luận 01 cho SVHN và EuroSAT, không huấn luyện lại, không sửa bài cũ.

Chép trọng số sang models/ch3/, chuyển pred npz thành proba của Chương 3, tính lại chỉ số từ đúng dự đoán
đã lưu và ghi reused_from vào từng kết quả. Lịch sử của Tiểu luận 01 đo val_acc trên tập test và không chọn
mô hình theo nó: trọng số dùng là epoch cuối.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlf.ch3_common import ARCHS, ARCH_NAME, MODELS, evaluate, save_result  # noqa: E402
from tlf.ch3_data import load  # noqa: E402
from tlf.common import ROOT, limit_threads  # noqa: E402
from tlf.mem import require_disk  # noqa: E402
from tlf.runlog import log_run  # noqa: E402

TL01 = ROOT.parent / "TieuLuan 01"


def reuse(ds, arch, src):
    r = src[arch]
    z = np.load(TL01 / "outputs" / "metrics" / f"ch3_pred_{ds}_{arch}.npz", allow_pickle=False)
    proba = z["prob"].astype(np.float32)
    y = load(ds, "test")[1]
    ev = evaluate(y, proba)
    if abs(ev["accuracy"] - r["acc"]) > 1e-3:   # pred và nhãn phải cùng thứ tự ảnh, không thì dừng thẳng
        raise ValueError(f"{ds}/{arch}: {ev['accuracy']:.4f} lệch số gốc {r['acc']:.4f}")
    shutil.copy2(TL01 / "models" / "ch3" / f"{ds}_{arch}.pt", MODELS / f"{ds}__{arch}.pt")
    hist = {k: r["hist"][k] for k in ("loss", "acc", "val_loss", "val_acc")}
    save_result(f"{ds}__{arch}", {
        "run": f"{ds}__{arch}", "dataset": ds, "model": ARCH_NAME[arch], "framework": "PyTorch", "arch": arch,
        "params": r["params"], "seconds": r["train_s"], "epochs": r["epochs"], "infer_ms": r["lat_cpu_ms"],
        "history": hist, "gflops": r["gflops"], **ev,
        "reused_from": f"src/TieuLuan 01/outputs/metrics/ch3_{ds}.json"}, proba)
    log_run("ch3", ds, ARCH_NAME[arch], "PyTorch", {k: ev[k] for k in ("accuracy", "f1_macro")}, r["train_s"],
            "cuda", {"params": r["params"], "epochs": r["epochs"], "reused_from": "TieuLuan 01"})
    print(ds, arch, round(ev["accuracy"], 4), round(r["acc"], 4), flush=True)


def reuse_keras(ds, src):
    """Bản Keras chỉ có chỉ số tổng hợp, không có tệp pred, nên giữ nguyên số gốc."""
    r = src["custom_keras"]
    save_result(f"{ds}__keras_custom", {
        "run": f"{ds}__keras_custom", "dataset": ds, "model": "CNN cơ bản", "framework": "Keras",
        "params": r["params"], "seconds": r["train_s"], "epochs": r["epochs"], "accuracy": r["acc"],
        "precision_macro": r["precision"], "recall_macro": r["recall"], "f1_macro": r["f1"],
        "infer_ms": r.get("lat_cpu_ms"), "history": {k: r["hist"][k] for k in r["hist"]},
        "reused_from": f"src/TieuLuan 01/outputs/metrics/ch3_{ds}.json"})


def main():
    limit_threads(2)
    require_disk(0.3, "ch3 chép mô hình Tiểu luận 01")
    for ds in ("svhn", "eurosat"):
        src = json.loads((TL01 / "outputs" / "metrics" / f"ch3_{ds}.json").read_text(encoding="utf-8"))
        for arch in ARCHS:
            reuse(ds, arch, src)
        reuse_keras(ds, src)
    scratch = json.loads((TL01 / "outputs" / "metrics" / "ch3_scratch.json").read_text(encoding="utf-8"))
    save_result("svhn__numpy_forward",
                {**scratch, "reused_from": "src/TieuLuan 01/outputs/metrics/ch3_scratch.json"})


if __name__ == "__main__":
    main()
