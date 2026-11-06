"""Bộ luật an toàn suy diễn tiến: dữ liệu ở rules.json, ở đây chỉ có bộ máy so khớp và chuẩn hóa dữ kiện.

Mỗi vòng chạy các luật theo ưu tiên giảm dần; luật cháy có thể thêm dữ kiện (then) làm luật khác cháy ở vòng sau,
lặp tới khi không còn dữ kiện mới. Ngưỡng chẩn đoán lấy từ KB qua thresholds(); khi KB chưa có thì dùng giá trị trong
rules.json, ghi rõ threshold_source để người đọc biết nguồn, không dùng dự phòng âm thầm.
"""
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

RULES_FILE = Path(__file__).with_name("rules.json")
SEVERITY_RANK = {"urgent": 2, "warn": 1, "info": 0}
MAX_ROUNDS = 8
MMOL_TO_MGDL = 18.0
_MISSING = object()
KIND_VI = {"diabetes": "đái tháo đường", "prediabetes": "tiền đái tháo đường"}


def norm_text(s):
    """Chữ thường dạng NFC, đồng nhất đ và bỏ khoảng trắng thừa để so từ khóa tiếng Việt."""
    return " ".join(unicodedata.normalize("NFC", str(s or "")).lower().split())


def strip_accents(s):
    t = unicodedata.normalize("NFD", s)
    return "".join(c for c in t if unicodedata.category(c) != "Mn").replace("đ", "d")


@lru_cache(maxsize=1)
def load_rules():
    rows = json.loads(RULES_FILE.read_text(encoding="utf-8"))["rules"]
    return tuple(sorted(rows, key=lambda r: -r["priority"]))


def egfr_ckd_epi(creatinine, age, sex):
    """eGFR theo CKD-EPI 2021 không dùng chủng tộc; creatinine mg/dL."""
    female = sex == "female"
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    r = creatinine / k
    return 142 * min(r, 1) ** a * max(r, 1) ** -1.2 * 0.9938 ** age * (1.012 if female else 1.0)


def prepare(facts):
    """Chuẩn hóa dữ kiện đầu vào trước khi suy diễn: văn bản, đơn vị glucose, thuốc, eGFR suy từ creatinine."""
    f = {k: v for k, v in (facts or {}).items() if v is not None}
    f["text"] = norm_text(f.get("text", ""))
    if f.get("glucose") is not None and f.get("glucose_unit") == "mmol/L":
        f["glucose"] = round(f["glucose"] * MMOL_TO_MGDL, 1)
    f["meds"] = {norm_text(m) for m in f.get("meds", [])}
    if "egfr" not in f and {"creatinine", "age", "sex"} <= f.keys():
        f["egfr"] = round(egfr_ckd_epi(f["creatinine"], f["age"], f["sex"]), 1)
    f.setdefault("glucose_kind", "random")
    return f


def _contains_any(text, words):
    bare = strip_accents(text)
    return any(w in text or strip_accents(w) in bare for w in words)


def _test(actual, op, value):
    if actual is _MISSING:
        return False
    if op == "contains_any":
        return _contains_any(actual, value)
    if op == "has":
        return value in actual
    if op == "is":
        return actual == value
    try:
        return {">=": actual >= value, ">": actual > value, "<": actual < value, "<=": actual <= value}[op]
    except TypeError:
        return False


def kb_value(spec, op, default):
    """Giá trị ngưỡng từ KB: dòng cùng đơn vị và cùng nghĩa; trả (giá trị, nguồn, url). KB chưa dựng thì dùng rules.json."""
    try:
        from .kb.search import thresholds
        rows = thresholds(spec["test"])
    except (ImportError, OSError, RuntimeError) as e:
        print(f"[rules] KB chưa sẵn sàng, dùng ngưỡng trong rules.json: {e}", flush=True)
        return default, "rules.json", None
    word = KIND_VI[spec["kind"]]
    for row in rows:
        if row.get("unit") == spec["unit"] and str(row.get("meaning", "")).startswith(word):
            value = row["value"]
            return float(value[0] if isinstance(value, list) else value), row.get("evidence") or "KB", row.get("url")
    return default, "rules.json", None


def _check(cond, facts, seen):
    value = cond.get("value")
    if "threshold" in cond:
        value, src, url = kb_value(cond["threshold"], cond["op"], value)
        seen.setdefault(f'{cond["fact"]}_threshold', {"value": value, "threshold_source": src, "url": url})
    actual = facts.get(cond["fact"], _MISSING)
    ok = _test(actual, cond["op"], value)
    if ok and cond["op"] != "contains_any":
        seen[cond["fact"]] = sorted(actual) if isinstance(actual, set) else actual
    elif ok:
        seen[cond["fact"]] = next(w for w in value if _contains_any(actual, [w]))
    return ok


def _fires(rule, facts):
    when, seen = rule["when"], {}
    if "all" in when:
        return all(_check(c, facts, seen) for c in when["all"]), seen
    return any([_check(c, facts, seen) for c in when["any"]]), seen


def evaluate(facts):
    """Trả các luật đã cháy, nặng trước rồi tới ưu tiên cao trước, đúng dạng trong hợp đồng."""
    f, fired = prepare(facts), {}
    for _ in range(MAX_ROUNDS):
        changed = False
        for rule in load_rules():
            if rule["id"] in fired:
                continue
            ok, seen = _fires(rule, f)
            if not ok:
                continue
            fired[rule["id"]] = (rule, seen)
            for k, v in rule.get("then", {}).items():
                changed = changed or f.get(k) != v
                f[k] = v
            changed = True
        if not changed:
            break
    rows = sorted(fired.values(), key=lambda x: (-SEVERITY_RANK[x[0]["severity"]], -x[0]["priority"]))
    return [{"rule_id": r["id"], "severity": r["severity"], "message": r["message"], "source": r["source"],
             "fired_on": seen, "group": r["group"]} for r, seen in rows]


def has_urgent(fired):
    """Khẩn khi có luật mức urgent; luật cảnh báo thuốc mức urgent cũng tính vì cần liên hệ bác sĩ ngay."""
    return any(r["severity"] == "urgent" for r in fired)


def flags(fired):
    """Các dữ kiện suy ra dùng cho điều phối: no_dose, out_of_scope, high_risk..."""
    out = {}
    for row in fired:
        rule = next(r for r in load_rules() if r["id"] == row["rule_id"])
        out.update(rule.get("then", {}))
    return out
