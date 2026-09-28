"""Hạng mục 2 (Ch.2): KKBox Churn, log nghe nhạc theo ngày của người dùng thật.

Mỗi người dùng thành một chuỗi 31 ngày (tháng 3/2017) × 8 đặc trưng. Tệp log ~1,4 GB được đọc
theo luồng từng khối bằng pyarrow nên không bao giờ nằm trọn trong RAM.
"""
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .export import DATA

KKBOX = DATA / "kkbox"
CACHE = KKBOX / "kkbox_seq.npz"
CACHE_INFO = KKBOX / "kkbox_seq_info.json"
LOG_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]
FEATURES = LOG_COLS + ["active"]
T_DAYS = 31
FIRST_DAY = 20170301
MAX_SECS = 86_400.0   # một ngày có 86.400 giây: giá trị lớn hơn (hay âm) là lỗi ghi log của KKBox
N_USERS = 100_000
SEED = 42


def _stream_logs(block_mb=64):
    """Duyệt user_logs_v2.csv theo khối RecordBatch (msno là chuỗi, còn lại là số)."""
    types = {c: pa.float64() if c == "total_secs" else pa.int32() for c in LOG_COLS}
    types.update(msno=pa.string(), date=pa.int32())
    reader = pacsv.open_csv(KKBOX / "user_logs_v2.csv",
                            read_options=pacsv.ReadOptions(block_size=block_mb << 20),
                            convert_options=pacsv.ConvertOptions(column_types=types))
    yield from reader


def sample_users(n=N_USERS, seed=SEED):
    """Người có nhãn VÀ có ít nhất một dòng log; lấy mẫu phân tầng theo is_churn, giữ nguyên tỉ lệ nhãn."""
    labels = pacsv.read_csv(KKBOX / "train_v2.csv").to_pandas()
    seen = set()
    for batch in _stream_logs():
        seen.update(pc.unique(batch.column("msno")).to_pylist())
    has_log = labels[labels.msno.isin(seen)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    frac = n / len(has_log)
    idx = []
    for _, g in has_log.groupby("is_churn"):
        k = int(round(len(g) * frac))
        idx.extend(rng.choice(g.index.to_numpy(), size=k, replace=False))
    sample = has_log.loc[np.sort(idx)].reset_index(drop=True)
    info = {"n_labeled": len(labels), "n_with_log": len(has_log), "n_sample": len(sample),
            "churn_rate_labeled": float(labels.is_churn.mean()),
            "churn_rate_with_log": float(has_log.is_churn.mean()),
            "churn_rate_sample": float(sample.is_churn.mean())}
    return sample, info


def build_sequences(sample):
    """Gom log của các người dùng trong mẫu thành tensor (N, 31, 8).

    Một ngày có thể có nhiều dòng (nhiều thiết bị) nên cộng dồn theo (người, ngày). Đặc trưng đếm
    và số giây lấy log1p vì phân phối đuôi rất dài; ngày không có log để 0 và cờ active = 0.
    """
    pos = {m: i for i, m in enumerate(sample.msno)}
    wanted = pa.array(list(pos))
    X = np.zeros((len(sample), T_DAYS, len(FEATURES)), dtype=np.float64)
    n_rows = n_kept = n_clipped = 0
    for batch in _stream_logs():
        n_rows += batch.num_rows
        batch = batch.filter(pc.is_in(batch.column("msno"), value_set=wanted))
        if batch.num_rows == 0:
            continue
        df = batch.to_pandas()
        n_kept += len(df)
        bad = (df.total_secs < 0) | (df.total_secs > MAX_SECS)
        n_clipped += int(bad.sum())
        df["total_secs"] = df.total_secs.clip(0, MAX_SECS)
        u = df.msno.map(pos).to_numpy()
        d = (df.date - FIRST_DAY).to_numpy()
        np.add.at(X, (u, d, slice(0, len(LOG_COLS))), df[LOG_COLS].to_numpy(np.float64))
    X[:, :, -1] = (X[:, :, :len(LOG_COLS)].sum(axis=2) > 0)
    X[:, :, :len(LOG_COLS)] = np.log1p(X[:, :, :len(LOG_COLS)])
    info = {"log_rows": n_rows, "log_rows_kept": n_kept, "secs_clipped": n_clipped}
    return X.astype(np.float32), info


def stratified_split(y, seed=SEED, fractions=(0.70, 0.15, 0.15)):
    """Chỉ số train/val/test phân tầng theo nhãn (mỗi phần giữ đúng tỉ lệ churn)."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        a = int(round(fractions[0] * len(idx)))
        b = a + int(round(fractions[1] * len(idx)))
        parts["train"].append(idx[:a])
        parts["val"].append(idx[a:b])
        parts["test"].append(idx[b:])
    return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}


def build_cache():
    """Một lần: lấy mẫu, gom chuỗi, chia tập, lưu .npz. Trả về dict thông tin để ghi metrics."""
    sample, info = sample_users()
    X, info2 = build_sequences(sample)
    y = sample.is_churn.to_numpy(np.int64)
    split = stratified_split(y)
    np.savez_compressed(CACHE, X=X, y=y, msno=sample.msno.to_numpy(str),
                        **{f"idx_{k}": v for k, v in split.items()})
    info = {**info, **info2, **{f"n_{k}": len(v) for k, v in split.items()}}
    CACHE_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def cache_info():
    """Thông tin của lần gom gần nhất; gom lại nếu chưa có cache."""
    if not (CACHE.exists() and CACHE_INFO.exists()):
        return build_cache()
    return json.loads(CACHE_INFO.read_text(encoding="utf-8"))


class Standardizer:
    """z-score theo từng đặc trưng, fit CHỈ trên train (gộp mọi ngày của mọi chuỗi train)."""

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def state(self):
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}


def load_churn():
    """Đọc .npz, chuẩn hoá bằng thống kê của train. Trả về dict X/y theo từng phần + dữ liệu thô."""
    z = np.load(CACHE)
    X, y = z["X"], z["y"]
    idx = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}
    scaler = Standardizer().fit(X[idx["train"]])
    return {"X": {k: scaler.transform(X[i]) for k, i in idx.items()},
            "y": {k: y[i].astype(np.float32) for k, i in idx.items()},
            "raw": X, "y_all": y, "msno": z["msno"], "idx": idx, "scaler": scaler,
            "features": FEATURES}
