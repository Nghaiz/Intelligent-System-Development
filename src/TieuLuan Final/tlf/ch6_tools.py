"""Công cụ của tác tử Chương 6: máy tính an toàn, tra câu trong tóm tắt, tra luật thuốc."""
import ast
import operator as op
import re

from .ch6_rules import GUIDELINES

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and abs(_eval(node.right)) <= 12:
        return _eval(node.left) ** _eval(node.right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError("chỉ nhận số và + - * / ** %")


def calc(expr, _item=None):
    """Tính biểu thức số học bằng cây cú pháp, không dùng eval nên mã do LLM sinh không chạy được."""
    clean = expr.replace("$", "").replace(",", "").replace("×", "*").replace("^", "**").strip()
    clean = re.sub(r"(\d+\.?\d*)\s*%", r"(\1/100)", clean)
    try:
        value = _eval(ast.parse(clean, mode="eval").body)
        return f"{round(value, 6):g}" if abs(value) < 1e15 else str(value)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as e:
        return f"error: {e}"


def find(keyword, item):
    """Trả các câu trong tóm tắt chứa từ khóa."""
    sents = re.split(r"(?<=[.!?])\s+", item.get("context", ""))
    words = [w.lower() for w in re.findall(r"\w{3,}", keyword)]
    hits = [s for s in sents if any(w in s.lower() for w in words)]
    return " ".join(hits[:4]) or "no sentence matches"


def guideline(drug, _item=None):
    """Trả luật của thuốc trong bộ luật tổng hợp."""
    key = drug.strip().lower().split()[0] if drug.strip() else ""
    return GUIDELINES.get(key, "Không có luật cho thuốc này. Danh mục: " + ", ".join(GUIDELINES))
