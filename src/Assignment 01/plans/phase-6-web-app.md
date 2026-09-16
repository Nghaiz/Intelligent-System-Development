# Phase 6 — Ứng dụng web Streamlit

**Mục tiêu:** biến mô hình đã huấn luyện thành ứng dụng dùng được.
Đáp ứng yêu cầu R11 và R12.

**Tệp sở hữu:** `app/app.py`, `app/lib.py`, `app/pages/*.py`, `.streamlit/config.toml`

## Ba trang

1. **Trang chủ** — hai thẻ công cụ, bảng xếp hạng mô hình, ba khối mở rộng
   (kiến trúc, biểu diễn dữ liệu, trạng thái Neo4j).
2. **Chẩn đoán Tiểu đường** — form 8 chỉ số, chọn mô hình, xác suất, phân tầng
   nguy cơ, gói can thiệp, đồ thị tri thức.
3. **Định giá Bất động sản** — form thuộc tính nhà, giá dự báo, phân khúc, gói
   vay ngân hàng, đồ thị tri thức.

**Không có trang Đồ thị Tri thức riêng.** Bản đầu tách nó ra thành trang thứ tư,
nhưng nội dung trùng gần hết với phần đồ thị đã có sẵn trong hai trang dự đoán —
cùng ca mẫu, cùng bảng bộ ba, cùng cách vẽ. Gộp vào tab *Khuyến nghị* của từng
trang thì bằng chứng KG = (V, E, R) vẫn đủ cho R11/R12 mà người dùng không phải
đi qua một trang chỉ để xem lại thứ vừa xem. Trạng thái Neo4j chuyển thành một
khối mở rộng ở trang chủ.

## Hợp đồng dữ liệu với Phase 5

Hai trang dự đoán đọc `outputs/graph.json` do `python -m src.kg.build_graph`
sinh ra, để lấy ca mẫu. Chốt lược đồ ở đây để bên đọc và bên ghi không tự suy
diễn mỗi bên một kiểu:

```jsonc
{
  "meta": {
    "schema_version": 1,        // kiểm giá trị này trước khi đọc tiếp
    "sinh_boi": "src/kg/build_graph.py",
    "so_ca": 6,
    "bang_mau": { "Patient": "#E74C3C", ... }   // loại thực thể → mã màu
  },
  "domains": {
    "medical":  { "label": "...", "cases": [ ... ] },
    "property": { "label": "...", "cases": [ ... ] }
  }
}
```

**Không có khoá `cases` ở mức trên cùng** — mỗi ca nằm trong đúng một miền. Một
ca gồm: `id`, `domain`, `label`, `model`, `inputs` (dữ liệu vào, ô khuyết là
`null`), `prediction`, `stats`, `nodes`, `edges`. Mỗi nút có `id`, `label`,
`type`, `properties`; mỗi cạnh có `source`, `relation`, `target`, `properties`.
Định danh nút chỉ duy nhất **trong phạm vi một ca** — vẽ nhiều ca cùng lúc thì
phải gắn tiền tố, giống cách `neo4j_client.graph_rows` làm.

Tệp `tests/test_artifacts.py` canh đúng hợp đồng này, nên nó đỏ ngay nếu Phase 5
đổi lược đồ mà quên báo. Đổi lược đồ thì tăng `schema_version`.

Hai tệp `outputs/kg_*.html` tự chứa hoàn toàn (không gọi ra Internet), nhúng
thẳng bằng `st.components.v1.html` được — dùng cho khối *Xem đồ thị gộp cả ba ca
mẫu*. Đồ thị dựng **trực tiếp từ dữ liệu người dùng nhập** thì gọi
`build_medical_kg` / `build_property_kg` thay vì đọc tệp.

## Ràng buộc bắt buộc

Ứng dụng **phải dùng lại đúng pipeline đã lưu** trong `models/`, không được
dựng lại phép biến đổi bằng tay. Đây chính là điều đề bài yêu cầu chứng minh:
biểu diễn lúc dự đoán phải khớp biểu diễn lúc huấn luyện.

## Điều hướng — vì sao tự dựng

`st.navigation` đặt `position="hidden"`, thanh điều hướng do `lib.thanh_dieu_huong`
vẽ bằng `st.page_link`. Hai phương án dựng sẵn đều đã thử và đều hỏng:

| Phương án | Hỏng ở đâu |
|---|---|
| `position="sidebar"` | Chiếm gần ¼ bề ngang chỉ để hiện ba dòng chữ; trên điện thoại bung ra phủ kín màn hình lúc mới mở |
| `position="top"` | Đẹp trên màn rộng, nhưng dưới một ngưỡng bề ngang Streamlit dồn điều hướng trở vào thanh bên — và ở đó nó hiện **tên tệp không dấu** (`chan doan tieu duong`). Đo được ở khổ 390px |

Thanh tự dựng nằm trong luồng nội dung nên nó căn đúng cột nội dung ở mọi bề
ngang, và giữ nguyên nhãn tiếng Việt. Trang đang mở được suy từ **đoạn cuối của
URL**, không phải từ `href` rỗng (Streamlit chỉ để rỗng ở trang mặc định) cũng
không phải từ lớp `st-emotion-cache-*` (đổi theo từng bản Streamlit).

## Responsive

Đã đo bằng trình duyệt thật ở **390 / 768 / 1024 / 1440 px**: không trang nào
tràn ngang (`document.scrollWidth == innerWidth`), thanh điều hướng luôn thẳng
hàng với dải tiêu đề bên dưới, và ba liên kết luôn hiện đủ. Thẻ số dùng lưới CSS
`auto-fit` nên tự xuống dòng; `st.columns` thì phải ép bằng media query vì nó
không tự xếp chồng.

## Tiêu chí hoàn thành

- [x] `streamlit run app/app.py` khởi động không lỗi
- [x] Ba ca thử nghiệm mẫu cho mỗi bài toán, chạy đúng
- [x] Đo responsive ở bốn khổ màn hình, không tràn ngang
- [x] Ảnh chụp giao diện ở cả chế độ máy tính và điện thoại (Phase 7)
