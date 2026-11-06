"""Giải nén parquet võng mạc thành mảng uint8 224×224 (memmap) theo split sẵn có; lưu ảnh mẫu thật cho web và hình."""
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import DATA, ROOT, limit_threads, save_json  # noqa: E402

limit_threads(8)
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from dnx.mem import require_disk, slot  # noqa: E402
from dnx.ml.retina import RDIR, SIZE, preprocess  # noqa: E402

RAW = DATA / "raw" / "retina" / "data"
SPLITS = {"train": "train-*.parquet", "val": "validation-*.parquet", "test": "test-*.parquet"}
SHOW = ROOT / "web" / "static" / "img" / "retina"


def decode(b):
    return preprocess(Image.open(io.BytesIO(b)))


def run_split(name, pat, pool):
    files = sorted(RAW.glob(pat))
    n = sum(pq.read_metadata(f).num_rows for f in files)
    X = np.lib.format.open_memmap(RDIR / f"{name}_x.npy", mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
    y, i, samples = np.zeros(n, dtype=np.int64), 0, {}
    for f in files:
        t = pq.read_table(f)
        imgs = [r["bytes"] for r in t["image"].to_pylist()]
        labs = t["label"].to_pylist()
        for arr, lab, b in zip(pool.map(decode, imgs), labs, imgs):
            X[i], y[i] = arr, lab
            if name == "test" and len(samples.setdefault(lab, [])) < 3:
                samples[lab].append(b)  # ảnh thật từ tập test cho web và lưới minh họa
            i += 1
        print(name, i, "/", n, flush=True)
    X.flush()
    np.save(RDIR / f"{name}_y.npy", y)
    return n, np.bincount(y, minlength=5).tolist(), samples


def main():
    require_disk(6, "memmap võng mạc")
    RDIR.mkdir(parents=True, exist_ok=True)
    SHOW.mkdir(parents=True, exist_ok=True)
    out = {"source": "youssefedweqd/Diabetic_Retinopathy_Detection (Hugging Face)", "image_size": SIZE, "splits": {}}
    with slot(3, "data_retina"), ThreadPoolExecutor(8) as pool:
        for name, pat in SPLITS.items():
            n, counts, samples = run_split(name, pat, pool)
            out["splits"][name] = {"n": n, "class_counts": counts}
            for lab, bs in samples.items():
                for k, b in enumerate(bs[:2]):
                    Image.open(io.BytesIO(b)).convert("RGB").save(SHOW / f"grade{lab}_{k + 1}.jpg", quality=90)
    save_json("data_retina", out)
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
