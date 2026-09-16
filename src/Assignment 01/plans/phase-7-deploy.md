# Phase 7 — Deploy lên internet

**Mục tiêu:** một đường link công khai để dán vào báo cáo và gửi giảng viên.

**Đường link:**
https://httm-nguyen-duy-nghia-assignment-01-intelligent-system.streamlit.app

## Việc phải làm

1. Đẩy toàn bộ mã và thư mục `models/` lên GitHub, kho công khai.
2. Người dùng làm theo `setup-guides/02-streamlit-cloud.md`.
3. Kiểm tra ba trang chạy đúng **trên bản deploy**, không chỉ trên máy.
4. Chụp ảnh giao diện thật để đưa vào báo cáo.

## Điểm cần chú ý

- Thư mục `models/` **phải** được đẩy lên, nếu không ứng dụng sẽ báo thiếu tệp.
  Kiểm tra `.gitignore` không chặn đuôi `.joblib`. Hiện có 13 tệp mô hình và 15
  tệp `outputs/` được theo dõi.
- Mật khẩu Neo4j khai báo trong mục Secrets của Streamlit, **không** commit.

## Ghim phiên bản — bài học phải trả giá mới có

Bản đầu khai `streamlit>=1.36`, sau nâng lên `>=1.61`. Cả hai đều là **dải**, và
Streamlit Cloud luôn cài bản mới nhất thoả dải ấy. Kết quả: máy cục bộ chạy
1.61.1 còn bản deploy chạy 1.62.0, và giữa hai bản ấy DOM khác nhau —
`st.container` xếp khối con thành cột thay vì hàng, thêm một lớp bọc
`stLayoutWrapper` mới. Thanh điều hướng nằm ngang ở máy nhưng **xếp dọc trên bản
deploy**, cùng một mã, không lỗi nào được ném ra.

Lỗi kiểu này không bắt được bằng cách nhìn kỹ hơn ở máy cục bộ, vì thứ chạy ở
hai nơi không phải là một. Hai việc bắt buộc:

1. **Ghim đúng một phiên bản** cho mọi gói ảnh hưởng tới giao diện hoặc tới
   việc nạp mô hình — `streamlit==1.62.0`, và trần phiên bản sẵn có cho
   `scikit-learn` / `xgboost`.
2. **Nghiệm thu trên bản deploy**, không nghiệm thu trên `localhost`. Bản
   `localhost` chỉ chứng minh mã chạy được, không chứng minh mã chạy được ở nơi
   người chấm sẽ mở.

Đổi phiên bản ghim thì phải cài đúng bản ấy ở máy rồi đo lại giao diện trước
khi đẩy.

## Tiêu chí hoàn thành

- [x] Đường link streamlit.app mở được từ máy khác
- [x] Ba trang không lỗi trên bản deploy
- [x] Thanh điều hướng nằm ngang, thẳng cột nội dung, ở cả khổ điện thoại lẫn
      máy tính — đo trên chính bản deploy
- [x] Ảnh chụp đã lưu vào `report/figures/`
