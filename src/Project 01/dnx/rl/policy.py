"""Chính sách sàng lọc đã huấn luyện, dùng lúc chạy: nhận đáp án đã có, trả câu hỏi kế tiếp hoặc tín hiệu dừng.

Chạy trên CPU và nạp lười nên Web App khởi động không cần torch. Trọng số được chọn ghi ở models/rl/policy.json.
"""
import json
from functools import lru_cache

from .features import FEATURES, KEYS, N, STOP, answer_value

MAX_QUESTIONS = 12  # trần an toàn của phiên hỏi, chính sách thường dừng sớm hơn


@lru_cache(maxsize=1)
def _load():
    import torch

    from . import dqn, predictor
    meta = json.loads((predictor.DIR / "policy.json").read_text(encoding="utf-8"))
    model, mean, std = predictor.load("cpu")
    qnet = dqn.QNet()
    qnet.load_state_dict(torch.load(predictor.DIR / "dqn" / meta["weights"], map_location="cpu", weights_only=True))
    qnet.eval()
    return model, qnet, mean, std, meta


def encode(answers):
    """Đáp án dạng hồ sơ sang vector giá trị thô và mặt nạ; khóa ngoài 21 đặc trưng bị bỏ qua."""
    raw, mask = [0.0] * N, [0.0] * N
    for key, value in (answers or {}).items():
        hit = answer_value(key, value)
        if hit is not None:
            raw[hit[0]], mask[hit[0]] = hit[1], 1.0
    return raw, mask


def next_question(answers):
    """Đúng hợp đồng: {"stop", "feature", "question_vi", "options", "prob", "q_values"}."""
    import torch

    from .dqn import masked_q
    from .env import make_state
    model, qnet, mean, std, _ = _load()
    raw, mask = encode(answers)
    with torch.no_grad():
        m = torch.tensor([mask])
        x = (torch.tensor([raw]) - mean) / std
        p = model.prob(x, m)
        s = make_state(x, m, p)
        q = masked_q(qnet(s), s)[0]
    prob = round(float(p[0]), 4)
    q_values = {(KEYS[i] if i != STOP else "stop"): round(float(q[i]), 3) for i in range(N + 1) if q[i] > -1e8}
    a = int(q.argmax())
    if a == STOP or sum(mask) >= MAX_QUESTIONS:
        return {"stop": True, "feature": None, "question_vi": "", "options": [], "prob": prob, "q_values": q_values}
    f = FEATURES[a]
    return {"stop": False, "feature": f.key, "question_vi": f.question_vi,
            "options": [{"value": v, "label": t} for v, t in f.options], "prob": prob, "q_values": q_values}
