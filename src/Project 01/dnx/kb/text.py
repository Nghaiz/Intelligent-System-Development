"""Làm sạch, tách câu, chunk theo câu, tách token cho BM25. Dùng chung cho chỉ mục và truy xuất."""
import re
import unicodedata

CHUNK_WORDS = 180
OVERLAP_WORDS = 30
MIN_WORDS = 15
TOKEN = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.U)
SENT = re.compile(r"(?<=[.!?;])[\"”)\]]*\s+(?=[\"“(\[]*[A-ZÀ-ỸĐ0-9])")
HEAD = re.compile(r"^=+\s*(.+?)\s*=+$")
DROP_SECTIONS = re.compile(r"^(tham khảo|liên kết ngoài|xem thêm|chú thích|đọc thêm|nguồn|references|related|"
                           r"see also|clinical trials|external links|acknowledgments?)\b", re.I)
STOP = set("""the of a an in on at to for and or is was were are be been by with from that which who what when where how
did does do has had have as it its this these those than also can may not no but if into such their there will
là của và có nào gì bao nhiêu trong với các những một được cho theo hay vừa ngoài mà ở thì khi đã sẽ này đó như
về do bị để ra rất nhiều người""".split())


def nfc(s):
    return unicodedata.normalize("NFC", s)


def clean_lines(text, source):
    """Bỏ dòng điều hướng ngắn của trang web, nối dòng PDF bị ngắt giữa câu, giữ hàng bảng và tiêu đề."""
    text = nfc(text).replace("\xa0", " ").replace("\r", "")
    pdf = source in ("bo_y_te", "medlineplus_vi") or text.count("\n") > len(text) / 70
    out = []
    for ln in text.split("\n"):
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if not ln:
            continue
        if HEAD.match(ln) or " | " in ln:
            out.append(ln)
        elif pdf:
            if re.fullmatch(r"(trang\s*)?\d{1,3}", ln, re.I):
                continue
            if out and not HEAD.match(out[-1]) and not re.search(r"[.!?:;]$", out[-1]):
                out[-1] += " " + ln
            else:
                out.append(ln)
        elif len(ln.split()) >= 6 or re.search(r"[.!?:]$", ln) or re.search(r"\d", ln):
            out.append(ln)
    return out


def sections(lines):
    """Chia theo tiêu đề; bỏ mục tham khảo, liên kết ngoài."""
    cur, body, skip = "", [], False
    for ln in lines:
        m = HEAD.match(ln)
        if m:
            if body and not skip:
                yield cur, body
            cur, body = m.group(1), []
            skip = bool(DROP_SECTIONS.match(cur))
        else:
            body.append(ln)
    if body and not skip:
        yield cur, body


def split_sentences(line):
    if " | " in line:
        return [line]
    return [s.strip() for s in SENT.split(line) if s.strip()]


def _words(s):
    return len(s.split())


def chunk_doc(text, source):
    """Gom câu tới gần CHUNK_WORDS từ, chunk sau mang lại các câu cuối (tối đa OVERLAP_WORDS từ) của chunk trước."""
    out = []
    for sec, body in sections(clean_lines(text, source)):
        sents = []
        for ln in body:
            for s in split_sentences(ln):
                toks = s.split()
                sents += [" ".join(toks[i:i + CHUNK_WORDS]) for i in range(0, len(toks), CHUNK_WORDS)]
        cur, n, fresh = [], 0, 0
        for s in sents:
            w = _words(s)
            if cur and n + w > CHUNK_WORDS:
                out.append((sec, " ".join(cur)))
                keep, k = [], 0
                for t in reversed(cur):
                    if k + _words(t) > OVERLAP_WORDS:
                        break
                    keep.insert(0, t)
                    k += _words(t)
                cur, n, fresh = keep, k, 0
            cur.append(s)
            n += w
            fresh += w
        if cur and fresh and (n >= MIN_WORDS):
            out.append((sec, " ".join(cur)))
    return out


def tokenize(text):
    """Token Unicode giữ số thập phân và cụm gạch nối (6.5, glp-1); tiếng Việt tách theo âm tiết."""
    toks = TOKEN.findall(nfc(text).lower())
    return [t for t in toks if t not in STOP and (len(t) > 1 or t.isdigit())]
