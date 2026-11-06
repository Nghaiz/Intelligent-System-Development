"""Chương 5: danh mục VN30 và đường dẫn dữ liệu thô dùng chung cho tải, nạp Neo4j và sinh câu hỏi."""
import json

from .common import data_dir

DATA5 = data_dir("ch5")
RAW = DATA5 / "raw"
RAW.mkdir(exist_ok=True)
START, END = "2018-01-01", "2026-10-31"

# Rổ tham chiếu tự khai báo để minh họa, không phải ảnh chụp chính thức của HOSE tại một ngày cụ thể.
# (mã, tên ngắn, ngành, truy vấn tìm bài Wikipedia tiếng Việt)
VN30 = [
    ("ACB", "Ngân hàng Á Châu", "Ngân hàng", "Ngân hàng Á Châu"),
    ("BCM", "Becamex IDC", "Bất động sản khu công nghiệp", "Becamex IDC"),
    ("BID", "BIDV", "Ngân hàng", "Ngân hàng Đầu tư và Phát triển Việt Nam"),
    ("BVH", "Tập đoàn Bảo Việt", "Bảo hiểm", "Tập đoàn Bảo Việt"),
    ("CTG", "VietinBank", "Ngân hàng", "VietinBank"),
    ("FPT", "Tập đoàn FPT", "Công nghệ", "FPT (công ty)"),
    ("GAS", "PV Gas", "Năng lượng", "PV Gas"),
    ("GVR", "Tập đoàn Cao su Việt Nam", "Nguyên vật liệu", "Tập đoàn Công nghiệp Cao su Việt Nam"),
    ("HDB", "HDBank", "Ngân hàng", "HDBank"),
    ("HPG", "Tập đoàn Hòa Phát", "Nguyên vật liệu", "Tập đoàn Hòa Phát"),
    ("LPB", "LPBank", "Ngân hàng", "Ngân hàng Bưu điện Liên Việt"),
    ("MBB", "MB Bank", "Ngân hàng", "Ngân hàng Quân đội"),
    ("MSN", "Tập đoàn Masan", "Hàng tiêu dùng", "Masan"),
    ("MWG", "Thế Giới Di Động", "Bán lẻ", "Thế Giới Di Động"),
    ("PLX", "Petrolimex", "Năng lượng", "Petrolimex"),
    ("SAB", "Sabeco", "Hàng tiêu dùng", "Sabeco"),
    ("SHB", "SHB", "Ngân hàng", "Ngân hàng Sài Gòn – Hà Nội"),
    ("SSB", "SeABank", "Ngân hàng", "SeABank"),
    ("SSI", "Chứng khoán SSI", "Chứng khoán", "Công ty Cổ phần Chứng khoán SSI"),
    ("STB", "Sacombank", "Ngân hàng", "Sacombank"),
    ("TCB", "Techcombank", "Ngân hàng", "Techcombank"),
    ("TPB", "TPBank", "Ngân hàng", "TPBank"),
    ("VCB", "Vietcombank", "Ngân hàng", "Vietcombank"),
    ("VHM", "Vinhomes", "Bất động sản", "Vinhomes"),
    ("VIB", "VIB", "Ngân hàng", "Ngân hàng Quốc tế Việt Nam"),
    ("VIC", "Tập đoàn Vingroup", "Bất động sản", "Vingroup"),
    ("VJC", "Vietjet Air", "Hàng không", "Vietjet Air"),
    ("VNM", "Vinamilk", "Hàng tiêu dùng", "Vinamilk"),
    ("VPB", "VPBank", "Ngân hàng", "VPBank"),
    ("VRE", "Vincom Retail", "Bất động sản", "Vincom Retail"),
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
