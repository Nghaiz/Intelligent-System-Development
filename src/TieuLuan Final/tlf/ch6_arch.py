"""Ba kiến trúc tác tử của Chương 6 và baseline một lượt gọi, viết bằng generator, không dùng framework."""
import re

from .ch6_core import Call, Note

ACTION = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.S)
FINAL = re.compile(r"Final Answer:\s*(.+)", re.S)
USER = "Người dùng"
CLOSE_TOKENS = 80      # đủ để kết luận kèm một câu lý do
LABEL_TOKENS = 12      # chỉ đủ cho nhãn
TEXT = {
    "en": {"only": "Reply only with {hint}.",
           "format": "Tool: {tool}\nUse this format, one step per reply:\nThought: short reasoning\n"
                     "Action: tool[input]\nAfter a tool result arrives, continue. When done write:\n"
                     "Final Answer: {hint}",
           "stop": "Stop using tools. Write exactly one line: Final Answer: {hint}",
           "again": "You already have that result. Write the Final Answer now.",
           "label": "Reply with only {hint}.",
           "role": "Your role: {role}.", "sup": "You are the supervisor. {task} Team: {team}.",
           "assign": "Write one line 'ASSIGN: names' then one line with the subtask for them.",
           "brief": "Supervisor brief: {plan}", "reports": "Reports:", "give": "Give {hint}.",
           "verifier": "You are a strict verifier. {task}", "draft": "Proposed conclusion:",
           "check": "Check it independently against the task. Reply PASS, or FAIL followed by one "
                    "sentence naming the exact mistake.",
           "redo": "Your earlier answer {draft} was rejected: {check}",
           "pro": "Solve it directly and defend your answer.",
           "con": "You are a skeptic: look for the trap, solve it by a different route and do not trust "
                  "the obvious answer.",
           "mine": "Your answer: {mine}", "other": "Opponent says:\n{other}",
           "rebut": "Find the flaw or concede. End with 'Final Answer: {hint}'.",
           "judge": "You are an impartial judge. {task}",
           "decide": "Agent A:\n{a}\n\nAgent B:\n{b}\n\nDecide. End with 'Final Answer: {hint}'."},
    "vi": {"only": "Chỉ trả lời {hint}.",
           "format": "Công cụ: {tool}\nTrả lời bằng tiếng Việt, không dùng tiếng Trung hay tiếng Anh. Giữ "
                     "nguyên các từ khóa sau, mỗi lần một bước:\nThought: lý luận ngắn\n"
                     "Action: tool[đầu vào]\nCó kết quả công cụ thì làm tiếp. Xong thì viết:\n"
                     "Final Answer: {hint}",
           "stop": "Dừng dùng công cụ. Viết đúng một dòng: Final Answer: {hint}",
           "again": "Bạn đã có kết quả đó. Hãy viết Final Answer ngay.",
           "label": "Chỉ trả lời {hint}.",
           "role": "Vai trò của bạn: {role}.", "sup": "Bạn là điều phối viên. {task} Đội: {team}. "
           "Trả lời bằng tiếng Việt.",
           "assign": "Viết một dòng 'ASSIGN: tên' rồi một dòng nêu việc cần làm.",
           "brief": "Điều phối giao việc: {plan}", "reports": "Các báo cáo:", "give": "Kết luận: {hint}.",
           "verifier": "Bạn là người kiểm chứng nghiêm ngặt. {task} Trả lời bằng tiếng Việt.",
           "draft": "Kết luận đề xuất:",
           "check": "Tự kiểm tra lại theo đề. Trả lời PASS, hoặc FAIL kèm một câu nêu đúng lỗi.",
           "redo": "Kết luận trước của bạn ({draft}) bị bác: {check}",
           "pro": "Giải trực tiếp và bảo vệ đáp án của mình.",
           "con": "Bạn là người hoài nghi: tìm bẫy, giải theo đường khác và đừng tin đáp án hiển nhiên.",
           "mine": "Đáp án của bạn: {mine}", "other": "Đối phương nói:\n{other}",
           "rebut": "Chỉ ra chỗ sai hoặc nhận sai. Kết thúc bằng 'Final Answer: {hint}'. "
                    "Trả lời bằng tiếng Việt.",
           "judge": "Bạn là trọng tài công tâm. {task} Trả lời bằng tiếng Việt.",
           "decide": "Tác tử A:\n{a}\n\nTác tử B:\n{b}\n\nHãy quyết định. Kết thúc bằng "
                     "'Final Answer: {hint}'."},
}


def say(dom, key, **kw):
    """Câu lệnh theo ngôn ngữ của miền: đề tiếng Việt mà lệnh tiếng Anh làm mô hình trôi sang tiếng Trung."""
    return TEXT[dom.lang][key].format(hint=dom.answer_hint, tool=dom.tool_doc, task=dom.task, **kw)


def direct(item, dom):
    """Baseline: một lượt gọi, không công cụ, không suy luận từng bước."""
    text = yield Call("LLM", USER, f"{dom.task} {say(dom, 'only')}", dom.prompt(item), 24)
    return {"answer": dom.parse(text), "rounds": 1, "text": text}


def settle(item, dom, agent, to, text):
    """Bảo đảm kết luận có nhãn đọc được: chưa có thì hỏi lại một câu ngắn, token vẫn được ghi."""
    if dom.parse(text) is not None:
        return text
    ask = f"{dom.prompt(item)}\n\n{text[-500:]}\n\n{say(dom, 'label')}"
    return (yield Call(agent, to, dom.task, ask, LABEL_TOKENS))


def react(item, dom, agent="ReAct", to=USER, persona="", brief="", max_steps=4):
    """Vòng Thought-Action-Observation; hết lượt thì buộc kết luận và hỏi lại nhãn nếu còn thiếu."""
    system = f"{dom.task} {persona}".strip() + "\n" + say(dom, "format")
    log, asked = dom.prompt(item) + (f"\n\n{brief}" if brief else ""), set()
    for _ in range(max_steps):
        text = yield Call(agent, to, system, log, dom.max_new)
        done, act = FINAL.search(text), ACTION.search(text)
        fresh = act and act.group(1) in dom.tools and (act.group(1), act.group(2).lower()) not in asked
        if fresh:     # tác tử viết cả Action lẫn Final Answer: kết luận chưa dựa trên kết quả công cụ nên bỏ
            name, arg = act.group(1), act.group(2)
            asked.add((name, arg.lower()))
            obs = dom.tools[name](arg, item)
            yield Note(f"Công cụ {name}", agent, f"{arg} -> {obs}")
            log += f"\n{text[:act.end()].strip()}\nObservation: {obs}"
        elif done and dom.parse(done.group(1)) is not None:
            return done.group(1).strip()
        else:
            log += f"\n{say(dom, 'again') if act else text}\n{say(dom, 'stop')}"
    text = yield Call(agent, to, system, f"{log}\n{say(dom, 'stop')}", CLOSE_TOKENS)
    text = text.split("Final Answer:")[-1].strip()
    return (yield from settle(item, dom, agent, to, text))


def single(item, dom):
    """Single ReAct: một tác tử làm hết."""
    text = yield from react(item, dom)
    return {"answer": dom.parse(text), "rounds": 1, "text": text}


def _assign(text, roster):
    """Đọc dòng ASSIGN của Supervisor; không nêu ai hợp lệ thì giao cho chuyên gia đầu danh sách."""
    line = next((s for s in text.splitlines() if "ASSIGN" in s.upper()), "")
    names = [n for n in roster if n.lower() in line.lower()]
    return names[:2] or list(roster)[:1]


def supervisor(item, dom, max_revise=1):
    """Điều phối giao việc, chuyên gia làm, kiểm chứng duyệt; bị bác thì làm lại tối đa max_revise lần."""
    team = "; ".join(f"{k}: {v}" for k, v in dom.roster.items())
    system = say(dom, "sup", team=team)
    plan = yield Call("Supervisor", "Đội chuyên gia", system,
                      f"{dom.prompt(item)}\n\n{say(dom, 'assign')}", 80)
    names, brief, rounds = _assign(plan, dom.roster), say(dom, "brief", plan=plan), 1
    while True:
        reports = {}
        for name in names:
            role = say(dom, "role", role=dom.roster[name])
            reports[name] = yield from react(item, dom, name, "Supervisor", role, brief, 3)
        draft = next(iter(reports.values()))
        if len(reports) > 1:
            lines = "\n".join(f"{k}: {v}" for k, v in reports.items())
            ask = f"{dom.prompt(item)}\n\n{say(dom, 'reports')}\n{lines}\n{say(dom, 'give')}"
            draft = yield Call("Supervisor", "Verifier", system, ask, dom.max_new // 2)
            draft = yield from settle(item, dom, "Supervisor", "Verifier", draft)
        ask = f"{dom.prompt(item)}\n\n{say(dom, 'draft')}\n{draft}\n\n{say(dom, 'check')}"
        check = yield Call("Verifier", "Supervisor", say(dom, "verifier"), ask, 90)
        if check.strip().upper().startswith("PASS") or rounds > max_revise:
            break
        rounds, brief = rounds + 1, say(dom, "redo", draft=draft, check=check)
    yield Note("Supervisor", USER, f"Kết luận: {draft}")
    return {"answer": dom.parse(draft), "rounds": rounds, "text": draft, "verifier": check[:200]}


def _rebut(item, dom, agent, to, stance, mine, other):
    """Một lượt phản biện: đọc đáp án đối phương, bác bỏ hoặc nhận sai, luôn kết thúc bằng nhãn."""
    ask = (f"{dom.prompt(item)}\n\n{say(dom, 'mine', mine=mine)}\n{say(dom, 'other', other=other)}\n"
           f"{say(dom, 'rebut')}")
    text = yield Call(agent, to, f"{dom.task} {stance}", ask, dom.max_new)
    return (yield from settle(item, dom, agent, to, text))


def debate(item, dom, max_rounds=2):
    """Hai tác tử giải độc lập, đọc chéo nhau. Chỉ hai nhãn đọc được và trùng nhau mới là đồng thuận;
    còn lại, kể cả khi một bên không ra nhãn, trọng tài quyết."""
    pro, con = say(dom, "pro"), say(dom, "con")
    a = yield from react(item, dom, "Tác tử A", "Tác tử B", pro, max_steps=3)
    b = yield from react(item, dom, "Tác tử B", "Tác tử A", con, max_steps=3)
    rounds = 1

    def agreed():
        return dom.parse(a) is not None and dom.parse(a) == dom.parse(b)

    while not agreed() and rounds < max_rounds:
        rounds += 1
        a = yield from _rebut(item, dom, "Tác tử A", "Tác tử B", pro, a, b)
        b = yield from _rebut(item, dom, "Tác tử B", "Tác tử A", con, b, a)
    if agreed():
        yield Note("Trọng tài", USER, f"Hai bên đồng thuận: {dom.parse(a)}")
        return {"answer": dom.parse(a), "rounds": rounds, "text": a, "consensus": True}
    ask = f"{dom.prompt(item)}\n\n{say(dom, 'decide', a=a, b=b)}"
    verdict = yield Call("Trọng tài", USER, say(dom, "judge"), ask, dom.max_new)
    verdict = yield from settle(item, dom, "Trọng tài", USER, verdict)
    return {"answer": dom.parse(verdict), "rounds": rounds, "text": verdict, "consensus": False}


ARCHS = {"direct": direct, "single": single, "supervisor": supervisor, "debate": debate}
