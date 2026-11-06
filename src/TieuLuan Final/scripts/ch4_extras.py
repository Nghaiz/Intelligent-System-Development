"""Chương 4: đo độ nhạy đầu ra theo độ trễ trên mô hình đã huấn luyện; bằng chứng thực nghiệm của tiêu biến gradient.

Chạy CPU vì cuDNN không cho lan truyền ngược qua RNN ở chế độ eval.
"""
import numpy as np
import torch

from tlf.ch4_data import Prepared
from tlf.ch4_torch import ARCHS, SeqRegressor
from tlf.common import limit_threads, model_dir, save_json
from tlf.mem import slot

SAMPLES = 512
SERIES = ("weather", "power", "FPT")


def lag_sensitivity(prep, arch):
    """‖∂ŷ₁/∂x_{T−k}‖ trung bình trên mẫu test, chuẩn hóa để độ nhạy ở bước gần nhất bằng 1."""
    st = torch.load(model_dir("ch4") / f"{prep.spec.id}__{arch}.pt", map_location="cpu", weights_only=True)
    model = SeqRegressor(prep.x.shape[1], prep.spec.horizon, arch, st["hidden"], st["dropout"])
    model.load_state_dict(st["state"])
    model.eval()
    ts = prep.last_index("test")
    ts = ts[np.linspace(0, len(ts) - 1, SAMPLES).astype(int)]
    x = torch.as_tensor(prep.windows(ts)).requires_grad_(True)
    model(x)[:, 0].sum().backward()
    sens = x.grad.norm(dim=2).mean(0).numpy()[::-1]
    return (sens / sens[0]).tolist()


def main():
    limit_threads(4)
    out = {}
    with slot(1.5, "ch4 extras"):
        for s in SERIES:
            prep = Prepared(s)
            out[s] = {a: lag_sensitivity(prep, a) for a in ARCHS}
    save_json("ch4_sensitivity", {"series": out})


if __name__ == "__main__":
    main()
