"""Huấn luyện hai CNN ImageNet phân cấp võng mạc, chọn theo QWK trên val, chấm test; giữ khóa gpu.heavy từng mô hình."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import limit_threads, save_json, seed_all  # noqa: E402

limit_threads(8)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dnx import gpu  # noqa: E402
from dnx.mem import slot  # noqa: E402
from dnx.ml.retina import CFG, MDIR, RDIR, fit, infer, metrics  # noqa: E402
from dnx.runlog import log_run  # noqa: E402

NAMES = ("resnet34", "efficientnet_b0")


def load(split):
    return np.load(RDIR / f"{split}_x.npy", mmap_mode="r"), np.load(RDIR / f"{split}_y.npy")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy GPU cho ml_retina")
    MDIR.mkdir(parents=True, exist_ok=True)
    (X, y), (Xv, yv), (Xt, yt) = load("train"), load("val"), load("test")
    out = {"config": CFG, "models": {}}
    for name in NAMES:
        seed_all()
        t0 = time.perf_counter()
        with slot(5, f"retina {name}"), gpu.heavy(f"retina {name}"):
            torch.cuda.reset_peak_memory_stats()
            print("bắt đầu", name, flush=True)
            net, hist = fit(name, X, y, Xv, yv, "cuda")
            pv, pt = infer(net, Xv, "cuda"), infer(net, Xt, "cuda")
            peak = torch.cuda.max_memory_allocated() / 2**20
            torch.save(net.cpu().state_dict(), MDIR / f"{name}.pt")
            del net
        secs = time.perf_counter() - t0
        np.savez_compressed(MDIR / f"preds_{name}.npz", yv=yv, pv=pv, yt=yt, pt=pt)
        vm, tm = metrics(yv, pv.argmax(1)), metrics(yt, pt.argmax(1))
        out["models"][name] = {"val": vm, "test": tm, "history": hist, "seconds": secs, "peak_vram_mb": peak}
        log_run("retina", "eyepacs", name, "pytorch", {k: tm[k] for k in ("accuracy", "macro_f1", "qwk")}, secs, "cuda",
                {"val_qwk": vm["qwk"], "epochs": len(hist)})
        print(name, "val QWK", vm["qwk"], "test QWK", tm["qwk"], f"{secs:.0f}s", flush=True)
        save_json("retina_results", out)
    best = max(out["models"], key=lambda k: out["models"][k]["val"]["qwk"])
    out["selected"] = best
    out["selected_by"] = "QWK trên val"
    save_json("retina_results", out)
    (MDIR / "selected.txt").write_text(best, encoding="utf-8")
    print("chọn", best, flush=True)


if __name__ == "__main__":
    main()
