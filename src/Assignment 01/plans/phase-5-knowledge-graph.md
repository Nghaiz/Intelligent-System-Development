# Phase 5 — Đồ thị Tri thức Neo4j

**Mục tiêu:** biến dự báo thống kê thành khuyến nghị hành động được, bằng cách
nối kết quả mô hình với tri thức miền đã chuẩn hoá.

**Tệp sở hữu:** `src/kg/ontology.py`, `src/kg/build_graph.py`,
`src/kg/neo4j_client.py`

## Vì sao cần

Mô hình chỉ trả về một con số — 83.1 phần trăm, hay 8.13 tỷ. Con số đó không tự
nó nói cho người dùng biết phải làm gì tiếp theo. Đồ thị lấp khoảng trống ấy.

## Hai bản thể học

**Y sinh, 4 lớp:** Bệnh nhân, Chỉ số sinh học, Dự báo AI, Mã bệnh ICD-10,
Thiết bị / Dinh dưỡng / Hướng dẫn, Nhà thuốc cung ứng.

**Đô thị và Tài chính, 5 lớp:** Bất động sản, Thuộc tính, Vị trí hành chính,
Định giá AI, Phân khúc thị trường, Gói vay ngân hàng và Tiện ích.

Biểu diễn hình thức: KG = (V, E, R) với V là tập thực thể, E tập cạnh, R tập
quan hệ ngữ nghĩa theo quy ước RDF.

## Chiến lược hai đường

1. **Có Neo4j:** sinh Cypher, nạp lên Aura Free, chụp ảnh Neo4j Browser.
2. **Không có Neo4j:** xuất `graph.json`, vẽ bằng pyvis. Web deploy vẫn chạy.

Đường 2 là mặc định, nên **không phase nào bị chặn** vì thiếu tài khoản.
Hướng dẫn tạo tài khoản: `setup-guides/01-neo4j-aura.md`.

## Tiêu chí hoàn thành

- [x] `graph.json` sinh ra cho cả hai miền — 6 ca, `outputs/graph.json`
- [x] Ảnh đồ thị tương tác đưa được vào báo cáo — `figures/kg_*.png` (tĩnh, cho
      LaTeX) và `outputs/kg_*.html` (tương tác, tự chứa, cho web app)
- [x] Bảng bộ ba tri thức xuất ra được — `outputs/kg_triplets.csv`, 96 bộ ba
- [x] Hệ thống chạy đúng khi biến môi trường Neo4j để trống — đã thử ba tình
      huống: không có tệp `.env`, biến để trống, và mật khẩu sai. Cả ba đều in
      cảnh báo rồi chạy tiếp, thoát mã 0.

## Đã làm xong — 26/08/2026

Chạy bằng `python -m src.kg.build_graph` (thêm `--offline` để không đụng mạng,
`--no-wipe` để nạp thêm thay vì xoá đi nạp lại).

**Sáu ca mẫu, không phải số bịa.** Mỗi tầng nguy cơ và mỗi phân khúc thị trường
lấy một quan sát thật, xác suất và giá đều do mô hình trong `models/` dự báo.
Ca được chọn là quan sát nằm ở **trung vị** của dải — mô tả dải trung thực hơn
giá trị cực trị, và vì thứ tự sắp xếp cố định nên chạy lại luôn ra đúng sáu ca.

**Đã nạp lên Aura thật:** 95 thực thể, 96 cạnh, 21 loại quan hệ, Neo4j Kernel
5.27-aura. Truy vấn kiểm chứng chạy được cả hai chiều nghiệp vụ — từ bệnh nhân
tới hướng dẫn lâm sàng, và từ thuộc tính nhà tới hạn mức vay cụ thể.

**Bộ kiểm thử:** 289 bài trong `tests/`, chạy bằng `python -m pytest`, khoảng
15 giây. Phần cần máy chủ Neo4j tự bỏ qua khi `.env` chưa cấu hình. Viết bộ
kiểm thử này phơi ra ba lỗi mà việc chạy tay không thấy — xem mục dưới.

**Đã gộp vào `run_pipeline.py`:** một lệnh dựng lại tất cả. Bước đồ thị chạy
sau cùng, chỉ **đọc** `models/` chứ không ghi, và tự lùi về đồ thị offline khi
không có mạng — nên lệnh gộp vẫn chạy trọn ở máy không có tài khoản nào.

**Năm điều phát hiện khi làm, đáng ghi vào báo cáo và phần bẫy:**

1. **Aura bản hiện hành dùng mã instance làm tên đăng nhập lẫn tên cơ sở dữ
   liệu**, không phải chuỗi `neo4j`. Đặt `NEO4J_USER=neo4j` nhận `AuthError`;
   đặt `NEO4J_DATABASE=neo4j` nhận `DatabaseNotFound`. `.env.example` và
   `setup-guides/01-neo4j-aura.md` đã sửa lại cho khớp; mã chấp nhận cả
   `NEO4J_USER` lẫn `NEO4J_USERNAME`.
2. **`NaN` của pandas không phải `None` của Python.** Bản thể học kiểm tra thiếu
   dữ liệu bằng `is None`, nên nếu không đổi kiểu ở ranh giới thì đồ thị mọc ra
   những nút "Insulin = nan". Trình điều khiển Neo4j cũng không tuần tự hoá
   được `numpy.int64`. Cả hai gộp vào một chỗ: `build_graph._native`.
3. **Bản thể học đặt tên nút cố định** (`patient`, `model`, `prediction`) cho
   mọi ca, nên sáu ca nạp chung một cơ sở dữ liệu sẽ chồng lên nhau. Định danh
   phải gắn tiền tố theo ca: `uid = "<ca>::<id>"`.
4. **`dict.get(khoá, mặc_định)` không cứu được khi khoá tồn tại với giá trị
   `None`.** `build_property_kg` dùng dạng ấy cho Tỉnh, Quận và Pháp lý, mà
   `_row_to_dict` thì luôn tạo đủ khoá và đặt `None` cho ô khuyết — nên nhãn
   hiện ra là "Tỉnh/Thành None", sai mà không lỗi nào được ném. Bài kiểm thử
   phơi ra; đã đổi sang `prop.get("Province") or "Không rõ"`. Phase 6 gọi thẳng
   hàm này với dữ liệu từ form nên lỗi sẽ hiện ngay trên giao diện.
5. **Hàm đọc `.env` ghi vào `os.environ` là tác dụng phụ nguy hiểm.** Nó khiến
   bộ kiểm thử phụ thuộc thứ tự chạy: một bài nạp tệp giả là bảy bài cần máy
   chủ thật sau đó **tự bỏ qua trong im lặng**, bộ kiểm thử vẫn xanh. Đã bỏ
   hẳn tác dụng phụ; thứ tự ưu tiên tệp ↔ môi trường quyết định tại
   `settings_from_env`, nơi nhìn thấy cả hai nguồn cùng lúc.
