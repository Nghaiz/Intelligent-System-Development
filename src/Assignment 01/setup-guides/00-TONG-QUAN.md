# Tổng quan — Những tài khoản bạn cần chuẩn bị

Toàn bộ dự án chỉ cần **3 tài khoản**, tất cả đều **miễn phí**. Bảng dưới cho biết
cái nào đã xong, cái nào cần bạn làm, và làm vào lúc nào.

| # | Dịch vụ | Dùng để làm gì | Trạng thái | Cần làm khi nào |
|---|---------|----------------|-----------|-----------------|
| 1 | **GitHub** | Chứa mã nguồn, dán link vào báo cáo | ✅ **Đã xong** — máy bạn đã đăng nhập sẵn tài khoản `Nghaiz` | Không cần làm gì |
| 2 | **Neo4j Aura Free** | Chứa Đồ thị Tri thức (Knowledge Graph) | ⏳ Cần bạn tạo | **Phase 5** — tôi sẽ nhắc bạn |
| 3 | **Streamlit Community Cloud** | Deploy website lên internet | ⏳ Cần bạn tạo | **Phase 7** — tôi sẽ nhắc bạn |

---

## Quan trọng: bạn KHÔNG bị chặn nếu chưa làm

Dự án được thiết kế để **chạy đủ 100% mà không cần tài khoản nào ngoài GitHub**:

- **Chưa có Neo4j?** Hệ thống tự động chuyển sang chế độ đồ thị offline
  (`outputs/graph.json` + thư viện pyvis). Đồ thị vẫn vẽ ra được, vẫn có hình
  cho báo cáo, vẫn tương tác được trên web. Kèm theo `outputs/graph.cypher` —
  dán vào một Neo4j bất kỳ là dựng lại nguyên đồ thị. Neo4j chỉ làm báo cáo
  "xịn" hơn vì có ảnh chụp giao diện Neo4j Browser thật.
- **Chưa có Streamlit Cloud?** Website vẫn chạy được ở máy bạn bằng một lệnh
  (`streamlit run app/app.py`), vẫn chụp được ảnh đưa vào báo cáo. Streamlit
  Cloud chỉ thêm một đường link public để nộp kèm.

Nên bạn cứ để tôi làm tiếp, khi nào tới bước cần thì tôi dừng lại và hướng dẫn.

---

## Thứ tự đọc

1. [`01-neo4j-aura.md`](01-neo4j-aura.md) — tạo cơ sở dữ liệu đồ thị (≈ 5 phút)
2. [`02-streamlit-cloud.md`](02-streamlit-cloud.md) — deploy website (≈ 3 phút)

Mỗi file đều viết theo kiểu **bấm gì, thấy gì, gõ gì** — không cần biết trước gì cả.

---

## Nguyên tắc bảo mật

Mật khẩu và khóa API **không bao giờ** được viết thẳng vào mã nguồn hay đẩy lên
GitHub. Dự án đọc chúng từ file `.env` ở thư mục gốc, và file này đã được
`.gitignore` chặn sẵn.

Mẫu file có sẵn tại [`.env.example`](../.env.example) — bạn chỉ cần chép thành
`.env` rồi điền giá trị thật vào.
