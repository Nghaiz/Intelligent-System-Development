"""Dữ liệu của hệ đa tác tử mua sắm (Chương 8): catalog tổng hợp, tồn kho và tập câu truy vấn có nhãn ý định.

Mọi thứ sinh tất định từ seed nên kết quả chạy lại giống hệt. Giá tính bằng triệu đồng.
"""
import random
import unicodedata

SHOP_SEED = 707
INTENTS = ["tim_san_pham", "so_sanh", "hoi_ton_kho", "hoi_gia", "chao_hoi", "khac"]

# loại hàng -> (từ khoá nhận diện, tên hiển thị trong câu, thương hiệu, khoảng giá triệu đồng)
CATEGORIES = {
    "laptop": (("laptop", "máy tính xách tay", "notebook"), ("laptop", "máy tính xách tay"),
               ("Dell", "Asus", "Lenovo", "HP", "Acer"), (11.0, 36.0)),
    "dien_thoai": (("điện thoại", "phone", "smartphone"), ("điện thoại", "smartphone", "phone"),
                   ("Samsung", "Xiaomi", "Oppo", "Apple", "Vivo"), (3.0, 28.0)),
    "tai_nghe": (("tai nghe", "headphone", "earbuds"), ("tai nghe", "headphone", "earbuds"),
                 ("Sony", "JBL", "Soundcore", "Apple", "Samsung"), (0.3, 6.5)),
    "chuot": (("chuột", "mouse", "bàn phím", "keyboard"), ("chuột", "bàn phím", "mouse"),
              ("Logitech", "Razer", "Asus", "Dell", "Akko"), (0.2, 2.5)),
    "giay": (("giày", "sneaker", "shoes"), ("giày", "sneaker", "giày chạy bộ"),
             ("Nike", "Adidas", "Puma", "Biti's", "Asics"), (0.8, 4.2)),
    "balo": (("balo", "ba lô", "backpack", "túi"), ("balo", "ba lô", "backpack"),
             ("Mikkor", "Samsonite", "Xiaomi", "Targus", "Herschel"), (0.4, 3.0)),
}


def strip_accents(s):
    """Bỏ dấu tiếng Việt (người dùng hay gõ không dấu): 'điện thoại' -> 'dien thoai'."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def build_catalog(seed=SHOP_SEED, per_category=10):
    """60 mặt hàng (6 loại × 10): id, tên, loại, thương hiệu, giá, điểm đánh giá, khuyến mãi."""
    rng = random.Random(seed)
    items = []
    for cat, (_, names, brands, (lo, hi)) in CATEGORIES.items():
        for k in range(per_category):
            brand = brands[k % len(brands)]
            items.append({"id": f"p{len(items) + 1:02d}", "category": cat, "brand": brand,
                          "name": f"{brand} {names[0].title()} {rng.choice('XSAMPZ')}{rng.randint(1, 9)}0",
                          "price": round(rng.uniform(lo, hi), 1), "rating": round(rng.uniform(3.4, 5.0), 1),
                          "promo": rng.choice([0, 0, 0, 5, 10, 15])})
    return items


def build_stock(catalog, seed=SHOP_SEED):
    """Tồn kho hiện tại và một bản chụp cũ (stale) lệch ở ~1/3 mặt hàng: bản cũ báo còn hàng ở chỗ nay đã hết."""
    rng = random.Random(seed + 1)
    now = {p["id"]: rng.choice([0, 0, 1, 3, 5, 8, 12]) for p in catalog}
    old = {i: (q if rng.random() > 0.35 else (6 if q == 0 else 0)) for i, q in now.items()}
    return now, old


SLOT_TEMPLATES = {
    "tim_san_pham": ["tìm giúp tôi {cat} dưới {budget}", "mình cần mua {cat} tầm {budget}",
                     "gợi ý {cat} {brand} giá rẻ", "có {cat} nào khoảng {budget} không",
                     "tư vấn cho em một chiếc {cat}", "show me {cat} under {budget}", "i want to buy a {cat}",
                     "recommend a good {cat} for me", "looking for {brand} {cat}", "cần {cat} xịn giá {budget}",
                     "chọn giúp {cat} cho dân văn phòng", "find a cheap {cat}", "đề xuất vài mẫu {cat} {brand}",
                     "muốn sắm {cat} mới khoảng {budget}"],
    "so_sanh": ["so sánh {cat} {brand} với {brand2}", "{brand} hay {brand2} thì {cat} nào tốt hơn",
                "nên chọn {cat} {brand} hay {brand2}", "khác nhau giữa {cat} {brand} và {brand2} là gì",
                "compare {brand} and {brand2} {cat}", "which {cat} is better {brand} or {brand2}",
                "{brand} vs {brand2} {cat}", "đánh giá {cat} {brand} so với {brand2}",
                "giữa {brand} và {brand2} mua {cat} nào", "difference between {brand} {cat} and {brand2}",
                "so {cat} {brand} và {brand2} về giá", "{cat} {brand} có hơn {brand2} không",
                "help me compare {brand} {cat} with {brand2}", "bảng so sánh {cat} {brand} {brand2}"],
    "hoi_ton_kho": ["{cat} {brand} còn hàng không", "shop còn {cat} {brand} không ạ", "khi nào có hàng {cat} {brand}",
                    "kiểm tra tồn kho {cat} giúp mình", "is the {brand} {cat} in stock", "do you have {cat} available",
                    "{cat} {brand} hết hàng chưa", "còn bao nhiêu chiếc {cat} {brand}", "check stock for {brand} {cat}",
                    "kho còn {cat} {brand} màu đen không", "{brand} {cat} sold out", "có sẵn {cat} để lấy ngay không",
                    "bao giờ nhập thêm {cat} {brand}", "any {cat} left in the warehouse"],
    "hoi_gia": ["{cat} {brand} giá bao nhiêu", "giá {cat} {brand} hiện tại", "có khuyến mãi {cat} không",
                "{cat} {brand} đang giảm giá à", "how much is the {brand} {cat}", "price of {brand} {cat}",
                "any discount on {cat}", "mã giảm giá cho {cat} {brand}", "{cat} có trả góp không",
                "bao nhiêu tiền một chiếc {cat} {brand}", "is there a sale on {brand} {cat}",
                "{cat} {brand} bán bao nhiêu vậy", "ưu đãi tháng này cho {cat}", "what does the {cat} cost"],
}
FREE_TEMPLATES = {
    "chao_hoi": ["xin chào", "chào shop", "hello", "hi there", "chào bạn mình mới vào", "good morning", "alo shop ơi",
                 "chào buổi tối", "hey", "xin chào bạn khỏe không", "cảm ơn shop nhé", "thanks a lot", "tạm biệt",
                 "ok cảm ơn bạn"],
    "khac": ["thời tiết hôm nay thế nào", "giao hàng mất mấy ngày", "đổi trả như thế nào", "tôi muốn khiếu nại đơn hàng",
             "kể chuyện cười đi", "what is the capital of france", "how do i reset my password", "mấy giờ rồi",
             "cửa hàng ở đâu vậy", "can i talk to a human", "đơn hàng của tôi đang ở đâu", "phí ship bao nhiêu",
             "write me a poem", "tôi quên mật khẩu tài khoản"],
}
# Khuôn câu chỉ dùng cho tập train: mở rộng vốn từ của từng ý định (từ đồng nghĩa, cách hỏi khác) để mô hình
# không học thuộc 9 khuôn; tập val/test vẫn chỉ gồm khuôn chưa từng thấy khi huấn luyện.
EXTRA_TRAIN = {
    "tim_san_pham": ["cần tìm {cat}", "mua {cat} ở đâu tốt", "suggest some {cat}", "có mẫu {cat} nào đẹp không",
                     "need a new {cat} around {budget}", "kiếm cho mình {cat} {brand}"],
    "so_sanh": ["{cat} {brand} và {brand2} cái nào đáng mua", "{brand} or {brand2} for {cat}",
                "so sánh giúp mình hai {cat}", "{brand} với {brand2} khác gì nhau", "{cat} nào hơn {brand} hay {brand2}",
                "{brand} versus {brand2}"],
    "hoi_ton_kho": ["{cat} còn không", "còn hàng {cat} {brand} chứ", "{brand} {cat} out of stock",
                    "hàng {cat} về chưa", "is {cat} available now", "số lượng {cat} {brand} trong kho"],
    "hoi_gia": ["giá {cat} thế nào", "{cat} bao nhiêu tiền", "cost of {brand} {cat}", "có voucher {cat} không",
                "{brand} {cat} giá tốt nhất", "{cat} {brand} có sale không", "how much does {cat} cost",
                "giá bán {cat} {brand} là bao nhiêu"],
    "chao_hoi": ["chào", "hi", "hello shop", "chào anh", "cảm ơn", "bye", "chúc ngày tốt lành", "rất vui được gặp"],
    "khac": ["phí vận chuyển là bao nhiêu", "ship về hà nội mất mấy ngày", "làm sao đổi mật khẩu", "hôm nay trời mưa không",
             "how long is delivery", "tôi muốn hủy đơn", "where is my order", "cho tôi gặp nhân viên",
             "hát một bài đi", "chính sách bảo hành thế nào"],
}
PREFIX = ["", "", "dạ ", "ừm ", "cho hỏi ", "ad ơi ", "please "]
SUFFIX = ["", "", " ạ", " nhé", " với", " shop ơi", "?", " nha", " thanks"]
BUDGETS = ["5 triệu", "500k", "2 triệu", "15 triệu", "20tr", "1 triệu rưỡi", "3 million", "800 nghìn", "25 triệu",
           "10 triệu", "1.5 triệu", "300k"]


def _fill(tpl, rng):
    cat = rng.choice(list(CATEGORIES))
    _, names, brands, _ = CATEGORIES[cat]
    b1, b2 = rng.sample(brands, 2)
    return tpl.format(cat=rng.choice(names), brand=b1, brand2=b2, budget=rng.choice(BUDGETS))


def _augment(text, rng):
    """Biến thể người dùng thật: thêm đầu/cuối câu, bỏ dấu (30 %), viết hoa chữ đầu, hoán đổi một ký tự (8 %)."""
    text = rng.choice(PREFIX) + text + rng.choice(SUFFIX)
    if rng.random() < 0.30:
        text = strip_accents(text)
    if rng.random() < 0.08 and len(text) > 6:
        i = rng.randrange(1, len(text) - 2)
        text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text[0].upper() + text[1:] if rng.random() < 0.4 else text


def build_intent_dataset(seed=SHOP_SEED, n_per_template=(40, 30, 30)):
    """Chia KHUÔN câu (không chia câu) thành train 9 / val 2 / test 3 mỗi ý định, rồi mới sinh câu: tập test đo khả
    năng tổng quát sang cách nói chưa gặp chứ không phải nhớ câu. Trả {split: [(câu, nhãn), ...]}."""
    rng = random.Random(seed)
    data = {"train": [], "val": [], "test": []}
    for intent in INTENTS:
        tpls = list(SLOT_TEMPLATES.get(intent) or FREE_TEMPLATES[intent])
        rng.shuffle(tpls)
        parts = {"train": tpls[:9] + EXTRA_TRAIN[intent], "val": tpls[9:11], "test": tpls[11:14]}
        for (split, group), n in zip(parts.items(), n_per_template):
            seen = set()
            for tpl in group:
                for _ in range(n):
                    s = _augment(_fill(tpl, rng), rng)
                    if s not in seen:
                        seen.add(s)
                        data[split].append((s, INTENTS.index(intent)))
    for v in data.values():
        rng.shuffle(v)
    return data
