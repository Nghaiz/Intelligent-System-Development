# Hướng dẫn 2 — Deploy website lên Streamlit Community Cloud

**Thời gian:** khoảng 3 phút · **Chi phí:** miễn phí
**Cần cho:** Phase 7 — Deploy

> Bỏ qua bước này thì website vẫn chạy ở máy bạn bằng lệnh
> `streamlit run app/app.py`. Bước này chỉ để có một đường link public dán vào
> báo cáo và gửi giảng viên bấm thử.

---

## Điều kiện tiên quyết

Repo GitHub phải **public** và đã push xong. Tôi sẽ lo phần này ở Phase 0 và
Phase 7 — khi tôi bảo "đã push xong" thì bạn mới bắt đầu hướng dẫn này.

---

## Bước 1 — Đăng nhập

Vào: **https://share.streamlit.io**

Bấm **"Continue with GitHub"**. Đăng nhập bằng đúng tài khoản GitHub `Nghaiz`
đang dùng cho dự án.

## Bước 2 — Cấp quyền

GitHub hiện trang hỏi quyền truy cập (*Authorize Streamlit*). Bấm
**"Authorize streamlit"**.

Đây là bước bắt buộc để Streamlit đọc được mã nguồn repo của bạn. Nó chỉ đọc,
không sửa gì.

## Bước 3 — Tạo app

Trên trang chính, bấm nút **"Create app"** (góc trên bên phải).

Nó hỏi *"Do you already have an app?"* → chọn **"Yes, I have an app"** →
**"Deploy a public app from GitHub"**.

Điền form:

| Ô | Điền gì |
|---|---------|
| Repository | `Nghaiz/Assignment-01-Intelligent-System` |
| Branch | `main` |
| Main file path | `app/app.py` |
| App URL | để mặc định, hoặc đổi thành `httm-nghia` cho gọn |

## Bước 4 — (Tùy chọn) Khai báo mật khẩu Neo4j

Nếu bạn **đã làm** [Hướng dẫn 1](01-neo4j-aura.md) và muốn website public cũng
kết nối Neo4j thật:

Trước khi bấm Deploy, bấm **"Advanced settings..."** → ô **Secrets**, dán vào:

```toml
NEO4J_URI = "neo4j+s://a1b2c3d4.databases.neo4j.io"
NEO4J_USERNAME = "a1b2c3d4"
NEO4J_PASSWORD = "mật-khẩu-của-bạn"
NEO4J_DATABASE = "a1b2c3d4"
```

Chép **đúng bốn giá trị** từ file `.env` của bạn sang, chỉ đổi cách viết. Đừng
sửa `NEO4J_USERNAME` thành `neo4j`: Aura bản hiện hành dùng mã instance (chuỗi
8 ký tự) làm cả tên đăng nhập lẫn tên cơ sở dữ liệu, viết `neo4j` sẽ nhận lỗi
`AuthError`. Lý do đầy đủ ở [Hướng dẫn 1, Bước 4](01-neo4j-aura.md).

Chú ý: định dạng ở đây **có dấu nháy kép** và **có dấu cách** quanh dấu `=` —
khác với file `.env`. Đây là quy ước riêng của Streamlit, chép đúng như trên.

Bỏ qua ô này cũng không sao — website sẽ tự dùng đồ thị offline.

## Bước 5 — Deploy

Bấm **"Deploy"**.

Màn hình hiện log cài đặt chạy dần. Lần đầu mất khoảng **3–5 phút** vì phải cài
xgboost và scikit-learn. Cứ để yên, đừng tải lại trang.

Xong sẽ ra đường link dạng:

```
https://httm-nguyen-duy-nghia-assignment-01-intelligent-system.streamlit.app
```

## Bước 6 — Báo cho tôi

Gửi tôi đường link đó. Tôi sẽ:
1. Mở bằng trình duyệt tự động, kiểm tra cả 3 trang chạy đúng
2. Chụp ảnh giao diện ở cả chế độ máy tính và điện thoại
3. Chèn ảnh + link vào báo cáo LaTeX, giống Figure 2.7 và 2.8 của mẫu

---

## Xử lý sự cố

**"Error installing requirements"**
Mở tab *Manage app* ở góc dưới bên phải để xem log. Gửi tôi dòng lỗi màu đỏ,
tôi sẽ sửa `requirements.txt`.

**"App chạy nhưng báo thiếu file model"**
Nghĩa là thư mục `models/` chưa được push lên. Báo tôi, tôi push bổ sung.

**"App ngủ, hiện nút Yes, get this app back up"**
Bình thường — app free tự ngủ sau 7 ngày không ai vào. Bấm nút đó, 30 giây sau
dậy. Trước khi nộp bài nhớ vào đánh thức trước.

**"Muốn đổi tên đường link"**
*Manage app* → **Settings** → **General** → sửa ô *Custom subdomain*.
