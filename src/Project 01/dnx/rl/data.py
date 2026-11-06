"""Dữ liệu CDC cho RL: bảng brfss của L1 đã khử trùng, dùng nguyên cột split để so được với các mô hình khác."""
import numpy as np
import pandas as pd

from ..db import connect
from .features import KEYS

LABEL = "diabetes"


def load():
    """Trả dict split -> (X thô, y) cùng mean, std của tập huấn luyện để chuẩn hóa."""
    con = connect()
    try:
        df = pd.read_sql_query(f"SELECT {','.join(KEYS)}, {LABEL}, split FROM brfss", con)
    finally:
        con.close()
    if df.empty:
        raise RuntimeError("Bảng brfss rỗng: chạy scripts/data_load.py trước")
    X, y = df[KEYS].to_numpy(np.float32), df[LABEL].to_numpy(np.float32)
    out = {s: (X[(df.split == s).values], y[(df.split == s).values]) for s in ("train", "val", "test")}
    mean, std = out["train"][0].mean(0), out["train"][0].std(0) + 1e-6
    return out, mean, std
