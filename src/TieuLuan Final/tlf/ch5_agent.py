"""Chương 5: tác tử ReAct. LLM viết Thought và Action, chương trình chạy công cụ và trả Observation, lặp tới Final."""
import ast
import json
import operator
import re
import time

from . import ch5_neo as N
from . import ch5_rag as G
from .ch5_retrieval import Dense, encode

MAX_STEPS = 4
OBS_CHARS = 700
SYSTEM_REACT = """Bạn là tác tử tra cứu trên đồ thị tri thức Neo4j. Công cụ:
- cypher: chạy một truy vấn Cypher chỉ đọc.
- vector_search: tìm đoạn văn liên quan theo nghĩa.
- calculator: tính biểu thức số học, ví dụ (120-100)/100*100.
Mỗi lượt viết đúng một khối, rồi dừng để đợi Observation:
Thought: suy nghĩ ngắn
Action: tên công cụ
Input: đầu vào trên một dòng
Khi đủ dữ liệu, viết:
Thought: suy nghĩ ngắn
Final: đáp án ngắn nhất, chỉ một tên, một số hoặc yes/no
Chỉ dùng dữ liệu từ Observation, không tự nghĩ ra."""
SCHEMA_HINT = {
    "hetionet": """Lược đồ: (:Compound {name})-[:TREATS|PALLIATES]->(:Disease {name}), (:Compound)-[:BINDS]->(:Gene {name}),
(:Compound)-[:CAUSES]->(:SideEffect {name}), (:Disease)-[:ASSOCIATES]->(:Gene), (:Disease)-[:PRESENTS]->(:Symptom),
(:Gene)-[:PARTICIPATES]->(:Pathway {name}), (:PharmacologicClass {name})-[:INCLUDES]->(:Compound).
Ví dụ: MATCH (c:Compound {name:'Metformin'})-[:BINDS]->(g:Gene)<-[:ASSOCIATES]-(:Disease {name:'type 2 diabetes mellitus'})
RETURN g.name""",
    "vn30": """Lược đồ: (:Company {ticker,name})-[:IN_SECTOR]->(:Sector {name}),
(:Company)-[:HAS_YEAR]->(:YearStat {year,ret_pct,vol_pct,high,low,close}),
(:Company)-[:HAS_MONTH]->(:MonthStat {month:'2023-05',ret_pct}), (:Company)-[:CORRELATED {rho}]-(:Company).
Ví dụ: MATCH (c:Company)-[:IN_SECTOR]->(:Sector {name:'Ngân hàng'}), (c)-[:HAS_YEAR]->(y:YearStat {year:2023})
RETURN c.ticker, y.ret_pct ORDER BY y.ret_pct DESC LIMIT 3""",
    "hotpot": """Lược đồ: (:Chunk {title,text})-[:ABOUT|MENTIONS]->(:TextEntity {name}), (:TextEntity)-[:LINKS_TO]->(:TextEntity).
Câu hỏi tiếng Anh: ưu tiên vector_search, tra từng thực thể một, trả lời bằng tiếng Anh.""",
}
STEP = re.compile(r"Action:\s*(?P<tool>\w+)\s*\n\s*Input:\s*(?P<input>.+)", re.I)
FINAL = re.compile(r"Final:\s*(?P<answer>.+)", re.I)
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expr):
    """Tính biểu thức bằng cây cú pháp, không dùng eval: chỉ số và các phép toán trong OPS."""
    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](walk(n.operand))
        raise ValueError("Biểu thức chỉ được chứa số và + - * / **")
    return round(walk(ast.parse(expr.replace(",", "."), mode="eval")), 4)


class Tools:
    """Ba công cụ của tác tử; mọi nguồn đã dùng được gom lại để trả kèm câu trả lời."""

    def __init__(self, ds):
        self.ds, self.dense, self.sources = ds, Dense(ds), []

    def run(self, tool, arg):
        arg = arg.strip().strip("`")
        try:
            return getattr(self, f"_{tool}")(arg)
        except AttributeError:
            return f"Lỗi: không có công cụ {tool}."
        except Exception as e:
            return f"Lỗi: {type(e).__name__}: {str(e)[:160]}"

    def _cypher(self, q):
        rows = N.read_only(q, limit=20)
        self.sources.append(f"cypher: {q}")
        return json.dumps(rows, ensure_ascii=False, default=str) if rows else "Không có dòng nào."

    def _vector_search(self, q):
        hits = self.dense.search(q, encode([q], device="cpu")[0], 3)
        self.sources += [c for c, _ in hits]
        return G.context_of(self.ds, [c for c, _ in hits])

    def _calculator(self, expr):
        return str(calculator(expr))


def parse_step(text):
    """Lấy khối đầu tiên của LLM: Action thì chạy công cụ, Final thì kết thúc. Action đứng trước Final được ưu tiên."""
    step, final = STEP.search(text), FINAL.search(text)
    if step and (not final or step.start() < final.start()):
        return "action", step["tool"].lower(), step["input"].split("\nObservation")[0].split("\n")[0]
    if final:
        return "final", None, final["answer"].split("\n")[0].strip()
    return "none", None, text.strip()


def force_final(llm, question, log, ds):
    """Hết lượt mà chưa có Final: nhờ LLM tổng hợp từ Observation đã có, đánh dấu forced trong vết."""
    ctx = "\n".join(s["text"] for s in log if s["kind"] == "observation") or "(không có)"
    return llm.chat(G.user_prompt(question, ctx[-3000:]), G.SYSTEM, 48)


def run_agent(llm, question, ds, max_new_tokens=120):
    """Generator phát từng bước {"kind","text"}, cuối cùng phát {"kind":"final",...}."""
    if G.is_clinical(question):
        yield {"kind": "final", "text": G.REFUSAL, "sources": [], "blocked": True, "steps": 0, "forced": False}
        return
    tools, log, seen = Tools(ds), [], set()
    history = f"{SCHEMA_HINT[ds]}\n\nCâu hỏi: {question}\n"
    for step in range(1, MAX_STEPS + 1):
        text = llm.chat(history, SYSTEM_REACT, max_new_tokens)
        kind, tool, arg = parse_step(text)
        thought = re.search(r"Thought:\s*(.+)", text)
        if thought:
            log.append({"kind": "thought", "text": thought[1].strip()})
            yield log[-1]
        if kind == "final":
            yield {"kind": "final", "text": arg, "sources": tools.sources, "blocked": False, "steps": step, "forced": False}
            return
        if kind == "none" or (tool, arg) in seen:
            break
        seen.add((tool, arg))
        log.append({"kind": "action", "text": f"{tool}({arg})"})
        yield log[-1]
        obs = tools.run(tool, arg)[:OBS_CHARS]
        log.append({"kind": "observation", "text": obs})
        yield log[-1]
        history += f"Thought: {thought[1].strip() if thought else ''}\nAction: {tool}\nInput: {arg}\nObservation: {obs}\n"
    yield {"kind": "final", "text": force_final(llm, question, log, ds), "sources": tools.sources, "blocked": False,
           "steps": MAX_STEPS, "forced": True}


def answer(llm, question, ds):
    """Chạy trọn tác tử, trả câu trả lời cùng vết để chấm điểm và lưu."""
    t0 = time.perf_counter()
    trace = list(run_agent(llm, question, ds))
    return {"answer": trace[-1]["text"], "sources": trace[-1]["sources"], "trace": trace, "steps": trace[-1]["steps"],
            "forced": trace[-1]["forced"], "blocked": trace[-1]["blocked"], "seconds": time.perf_counter() - t0}
