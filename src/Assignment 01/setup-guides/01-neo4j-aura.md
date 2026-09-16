# Hướng dẫn 1 — Tạo Neo4j Aura Free (cơ sở dữ liệu Đồ thị Tri thức)

**Thời gian:** khoảng 5 phút · **Chi phí:** miễn phí vĩnh viễn (gói AuraDB Free)
**Cần cho:** Phase 5 — Knowledge Graph

> Nếu bạn bỏ qua bước này, dự án vẫn chạy đủ bằng đồ thị offline. Xem
> [`00-TONG-QUAN.md`](00-TONG-QUAN.md). Làm bước này chỉ để báo cáo có thêm ảnh
> chụp Neo4j Browser thật, giống mẫu `A1_CT_tupv.879.pdf`.

---

## Bước 1 — Mở trang đăng ký

Vào: **https://console.neo4j.io**

## Bước 2 — Đăng ký tài khoản

Trang sẽ hiện nút đăng nhập. Chọn **"Sign up"**, rồi chọn cách nhanh nhất:

- Bấm **"Continue with Google"** và chọn tài khoản Gmail của bạn — xong luôn, không cần xác thực email.
- Hoặc điền email + mật khẩu, sau đó vào hộp thư bấm link xác nhận.

Nếu nó hỏi vài câu khảo sát (bạn làm nghề gì, dùng Neo4j để làm gì), chọn đại
"Student" / "Learning" rồi bấm tiếp. Không ảnh hưởng gì.

## Bước 3 — Tạo instance

Sau khi vào được Console, bạn sẽ thấy nút **"New Instance"** hoặc
**"Create instance"**. Bấm vào.

Chọn gói **AuraDB Free** (thẻ có chữ *Free*, ghi 0 USD). **Đừng chọn** Professional
— nó tính tiền.

Điền:

| Ô | Điền gì |
|---|---------|
| Instance Name | `httm-knowledge-graph` |
| Region | Chọn cái gần nhất, ví dụ *Singapore* hoặc *Asia* |

Bấm **"Create Instance"**.

## Bước 4 — ⚠️ LƯU MẬT KHẨU NGAY

Đây là bước quan trọng nhất. Ngay sau khi bấm Create, Neo4j sẽ hiện **một lần
duy nhất** một hộp thoại chứa thông tin đăng nhập.

Có nút **"Download and continue"** — **bấm nút đó** để tải file `.txt` về máy.
Nếu bạn đóng hộp thoại mà chưa lưu, mật khẩu **mất vĩnh viễn**, phải xóa instance
tạo lại từ đầu.

File tải về có dạng như sau — đây chính là nội dung bạn sẽ dùng ở Bước 6:

```dotenv
NEO4J_URI=neo4j+s://a1b2c3d4.databases.neo4j.io
NEO4J_USERNAME=a1b2c3d4
NEO4J_PASSWORD=aB3xK9mQ7wR2tY5uP0nZ
NEO4J_DATABASE=a1b2c3d4
AURA_INSTANCEID=a1b2c3d4
AURA_INSTANCENAME=Free instance
```

> **Đừng sửa `NEO4J_USERNAME` thành `neo4j`.** Aura bản hiện hành dùng **mã
> instance** (chuỗi 8 ký tự, ở ví dụ trên là `a1b2c3d4`) làm cả tên đăng nhập
> lẫn tên cơ sở dữ liệu. Nhiều tài liệu cũ trên mạng vẫn ghi `neo4j` — làm theo
> sẽ nhận lỗi `AuthError`, còn sửa `NEO4J_DATABASE` thành `neo4j` sẽ nhận lỗi
> `DatabaseNotFound`. Cứ chép nguyên giá trị Aura đưa.

## Bước 5 — Lấy URI kết nối

Đợi khoảng 1–3 phút để instance chuyển từ trạng thái *Creating* sang **Running**
(chấm màu xanh).

Trên thẻ instance, bạn sẽ thấy dòng địa chỉ dạng:

```
neo4j+s://a1b2c3d4.databases.neo4j.io
```

Đó là **URI**. Chép lại (có nút copy hình 2 tờ giấy chồng nhau).

## Bước 6 — Đưa 3 giá trị vào dự án

Ở thư mục gốc dự án (`d:\Python\HTTM`), tạo file tên **`.env`** (đúng tên này,
có dấu chấm đầu, không có đuôi `.txt`).

Cách dễ nhất, và cũng ít sai nhất: mở file `.txt` vừa tải ở Bước 4, **chép toàn
bộ nội dung** rồi dán vào `.env`. Không phải sửa dòng nào cả.

**Không** thêm dấu nháy, **không** thêm dấu cách quanh dấu `=`, và **không** đổi
`NEO4J_USERNAME` hay `NEO4J_DATABASE` thành `neo4j` (xem cảnh báo ở Bước 4).

## Bước 7 — Kiểm tra và nạp đồ thị

Chạy một lệnh:

```bash
python -m src.kg.build_graph
```

Lệnh này dựng sáu đồ thị tri thức từ dữ liệu thật, xuất `outputs/graph.json`,
bảng bộ ba, hai ảnh tĩnh, hai tệp HTML tương tác — rồi tự nạp lên Aura nếu đọc
được `.env`. Cuối màn hình sẽ in tên máy chủ và tổng số nút đã nạp.

Sau đó mở Neo4j Browser ở **https://console.neo4j.io** → nút **Query**, chạy:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

rồi chụp màn hình để đưa vào báo cáo.

**Bạn không cần đưa mật khẩu cho ai** — chương trình đọc thẳng từ file `.env`,
và file này đã bị `.gitignore` chặn nên không lên GitHub.

Không có tài khoản Aura thì bỏ qua bước này: lệnh trên vẫn chạy đủ, chỉ in thêm
một dòng báo đang dùng đồ thị offline. Muốn dựng lại đồ thị trên một Neo4j bất
kỳ thì dán tệp `outputs/graph.cypher` vào Neo4j Browser.

---

## Xử lý sự cố

**"Instance kẹt ở trạng thái Creating quá 5 phút"**
Tải lại trang. Nếu vẫn kẹt, xóa và tạo lại ở region khác.

**"Tôi lỡ đóng hộp thoại mật khẩu rồi"**
Không khôi phục được. Vào thẻ instance → menu `...` → **Delete**, rồi làm lại từ
Bước 3. Instance mới sẽ sinh mật khẩu mới.

**"Tài khoản Free chỉ được 1 instance"**
Đúng vậy, 1 là đủ cho dự án này.

**"Instance bị tạm dừng sau vài ngày không dùng"**
Gói Free tự ngủ sau 3 ngày không hoạt động. Vào Console bấm **"Resume"** là chạy
lại, dữ liệu không mất.
