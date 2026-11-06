"""Tải ba bộ dữ liệu Chương 2 về data/ch2/: chỉ CSV/Parquet thuần, không có mã nào được thực thi."""
import shutil
import urllib.request
from pathlib import Path

from tlf.common import data_dir
from tlf.mem import require_disk

D = data_dir("ch2")
CDC_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
VSFC_URL = "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/{}/0.parquet"
HOUSING_SLUG = "nguyentiennhan/vietnam-housing-dataset-2024"


def fetch(url, dst):
    if dst.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": "tieuluan-final/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    print(dst.name, dst.stat().st_size, flush=True)


def fetch_housing():
    dst = D / "vietnam_housing.csv"
    if dst.exists():
        return
    import kagglehub
    src = Path(kagglehub.dataset_download(HOUSING_SLUG))
    shutil.copy(next(src.rglob("*.csv")), dst)
    print(dst.name, dst.stat().st_size, flush=True)


if __name__ == "__main__":
    require_disk(0.1, "ch2 tải dữ liệu")
    fetch(CDC_URL, D / "cdc_diabetes.csv")
    for split in ("train", "validation", "test"):
        fetch(VSFC_URL.format(split), D / f"vsfc_{split}.parquet")
    fetch_housing()
