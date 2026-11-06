"""Chương 5: phân mảnh văn bản. fixed_chunks là mốc so sánh, recursive_chunks là chiến lược dùng thật."""
import re

SEPARATORS = ("\n== ", "\n=== ", "\n\n", "\n", ". ", " ")


def words(text):
    return len(text.split())


def fixed_chunks(text, size=180, overlap=30):
    """Cắt cơ học theo số từ, cửa sổ trượt có chồng lấn."""
    toks = text.split()
    step = size - overlap
    return [" ".join(toks[i:i + size]) for i in range(0, max(len(toks) - overlap, 1), step)]


def _split(text, seps, size):
    """Tách đệ quy theo dấu phân cách thô trước, mảnh còn dài mới xuống dấu phân cách mịn hơn."""
    if words(text) <= size or not seps:
        return [text]
    parts = [p for p in text.split(seps[0]) if p.strip()]
    if len(parts) == 1:
        return _split(text, seps[1:], size)
    out = []
    for i, p in enumerate(parts):
        piece = (seps[0].lstrip("\n") if i and seps[0].startswith("\n=") else "") + p
        out += _split(piece, seps[1:], size)
    return out


def recursive_chunks(text, size=180, overlap=30):
    """Gộp các mảnh liền kề tới gần size từ; mảnh sau mang theo overlap từ cuối mảnh trước."""
    merged, cur = [], []
    for part in _split(text, SEPARATORS, size):
        n = words(part)
        if cur and sum(map(words, cur)) + n > size:
            merged.append(" ".join(cur))
            tail = merged[-1].split()[-overlap:]
            cur = [" ".join(tail)] if overlap else []
        cur.append(part.strip())
    if cur:
        merged.append(" ".join(cur))
    return [re.sub(r"[=\s]{2,}", " ", m).strip() for m in merged]
