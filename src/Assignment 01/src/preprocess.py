"""Tiền xử lý và biểu diễn dữ liệu cho hai bài toán.

Đây là lớp chuyển hoá "thế giới thực → biểu diễn tính toán": mỗi quan sát thô
được đưa về một vector đặc trưng x ∈ R^d mà thuật toán học máy nhận vào.

Cùng một hàm được dùng ở ba nơi — notebook, script huấn luyện và ứng dụng web —
nên biểu diễn lúc dự đoán luôn khớp biểu diễn lúc huấn luyện (yêu cầu R11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# ==========================================================================
#  BÀI TOÁN 1 — TIỂU ĐƯỜNG (PHÂN LỚP)
# ==========================================================================

def load_diabetes_raw() -> pd.DataFrame:
    """Đọc tập Pima Indians Diabetes ở dạng thô, chưa xử lý."""
    return pd.read_csv(cfg.DIABETES_CSV)


def clean_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """Đánh dấu giá trị 0 phi lý thành khuyết, KHÔNG điền giá trị tại đây.

    Năm cột sinh học trong tập gốc dùng số 0 để mã hoá "không đo được".
    Giữ nguyên số 0 sẽ dạy mô hình rằng tồn tại bệnh nhân có huyết áp bằng 0 —
    một quan hệ sai hoàn toàn về mặt y sinh.

    Việc điền khuyết được cố tình **để lại cho pipeline** (xem
    ``train.diabetes_preprocessor``) chứ không làm ở đây, vì hai lý do:

    1. Điền bằng trung vị tính trên toàn bộ dữ liệu sẽ để thông tin của tập
       kiểm tra rò rỉ vào tập huấn luyện.
    2. Điền bằng trung vị **nhóm theo ``Outcome``** còn nghiêm trọng hơn: nó mã
       hoá thẳng nhãn cần dự đoán vào đặc trưng. Với tập này, Insulin khuyết
       374/768 và SkinThickness khuyết 227/768 quan sát, nên gần một nửa dữ
       liệu sẽ mang dấu vết của nhãn và mô hình đạt độ chính xác giả tạo ~87%,
       vượt xa mức 77–79% mà các nghiên cứu công bố trên cùng tập dữ liệu.
    """
    df = df.copy()
    df[cfg.DIABETES_ZERO_AS_MISSING] = df[cfg.DIABETES_ZERO_AS_MISSING].replace(0, np.nan)
    return df


def add_diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm đặc trưng phái sinh mã hoá tri thức y khoa vào biểu diễn.

    Bốn đặc trưng dưới đây không có trong dữ liệu gốc. Chúng là ví dụ trực tiếp
    của "human-designed features" — điểm phân biệt Học máy truyền thống với Học
    sâu: con người đưa tri thức miền vào, thay vì để mạng tự học biểu diễn.
    """
    df = df.copy()

    # Rối loạn đường huyết và béo phì cộng hưởng phi tuyến với nhau; tích số
    # nắm bắt được tương tác đó, còn mô hình tuyến tính thì không tự suy ra.
    df["Glucose_BMI_Risk"] = df["Glucose"] * df["BMI"]

    # Ngưỡng chẩn đoán tiền đái tháo đường theo Hiệp hội Đái tháo đường Hoa Kỳ.
    df["Glucose_Level"] = pd.cut(
        df["Glucose"], bins=[0, 99, 125, 500], labels=[0, 1, 2]
    ).astype(int)

    # Phân loại thể trạng theo chuẩn WHO.
    df["BMI_Class"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    # Nguy cơ tăng theo tuổi nhưng không tuyến tính; nhóm tuổi giúp cây quyết
    # định tách ngưỡng gọn hơn.
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]
    ).astype(int)

    return df


def prepare_diabetes(clean: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Trả về ma trận đặc trưng X và vector mục tiêu y cho bài toán phân lớp.

    X trả về vẫn **còn giá trị khuyết** khi ``clean=True``; phần điền khuyết và
    thiết kế đặc trưng do pipeline đảm nhiệm để tránh rò rỉ dữ liệu.
    """
    df = load_diabetes_raw()
    if clean:
        df = clean_diabetes(df)

    y = df[cfg.DIABETES_TARGET]
    X = df.drop(columns=[cfg.DIABETES_TARGET])
    return X, y


def diabetes_eda_frame() -> pd.DataFrame:
    """Bản dữ liệu đã điền khuyết, chỉ dùng để vẽ biểu đồ khám phá.

    Phân tích khám phá không huấn luyện mô hình nào nên không có khái niệm rò
    rỉ ở đây; điền bằng trung vị toàn tập là đủ để biểu đồ phản ánh đúng phân
    bố sau khi đã loại các giá trị 0 phi lý.
    """
    df = clean_diabetes(load_diabetes_raw())
    numeric = df.columns.drop(cfg.DIABETES_TARGET)
    df[numeric] = df[numeric].fillna(df[numeric].median())
    return add_diabetes_features(df)


# ==========================================================================
#  BÀI TOÁN 2 — ĐỊNH GIÁ BẤT ĐỘNG SẢN (HỒI QUY)
# ==========================================================================

def load_housing_raw() -> pd.DataFrame:
    """Đọc tập Vietnam Housing ở dạng thô.

    Tệp gốc có BOM ở đầu nên phải dùng ``utf-8-sig``; đọc bằng ``utf-8`` thường
    sẽ khiến tên cột đầu tiên dính ký tự vô hình và mọi phép chọn cột đều lỗi.
    """
    return pd.read_csv(cfg.HOUSING_CSV, encoding="utf-8-sig")


def parse_address(df: pd.DataFrame) -> pd.DataFrame:
    """Tách trường địa chỉ văn bản tự do thành Tỉnh/Thành và Quận/Huyện.

    Địa chỉ tuân theo quy ước hành chính Việt Nam, đọc từ nhỏ đến lớn:
    ``... , Phường/Xã , Quận/Huyện , Tỉnh/Thành``. Do đó phần tử cuối là cấp
    tỉnh và phần tử áp chót là cấp huyện, bất kể chuỗi dài bao nhiêu đoạn.
    """
    df = df.copy()
    parts = df["Address"].fillna("").str.split(",")

    def _clean(value: object) -> str:
        text = str(value).strip().rstrip(".").strip()
        return text if text else "Không rõ"

    # Địa chỉ dưới 3 đoạn là tiêu đề tin rao chứ không phải địa chỉ hành chính
    # (ví dụ "Bán nhà chính chủ Phó Đức Chính"), nên không tách được cấp nào.
    enough = parts.str.len() >= 3

    df["Province"] = np.where(enough, parts.str[-1].map(_clean), "Không rõ")
    df["District"] = np.where(enough, parts.str[-2].map(_clean), "Không rõ")

    return df


def clean_housing(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Loại cột thiếu quá nhiều, tách địa chỉ và điền khuyết phần còn lại.

    Trả về **bộ đôi**: bảng đã làm sạch, và danh sách tên những cột bị loại vì
    thiếu quá ngưỡng. Danh sách ấy đi thẳng vào báo cáo nên không được nuốt mất
    — cả ba nơi gọi đều giải nén hai giá trị.
    """
    df = parse_address(df)

    # Bỏ cột mà phần lớn quan sát không có giá trị — điền khuyết ở mức thiếu
    # trên 80% chỉ tạo ra một hằng số giả, không mang thông tin nào.
    missing_ratio = df.isna().mean()
    too_sparse = missing_ratio[missing_ratio > cfg.MISSING_DROP_THRESHOLD].index.tolist()
    df = df.drop(columns=too_sparse)

    # Địa chỉ thô đã được rút thành hai cột hành chính nên không còn cần thiết.
    df = df.drop(columns=["Address"])

    numeric_cols = df.select_dtypes(include="number").columns.drop(cfg.HOUSING_TARGET)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Với biến định danh, phần thiếu tự nó là một thông tin ("người đăng tin
    # không khai báo"), nên được giữ thành một hạng mục riêng thay vì gán bừa.
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].fillna("Không rõ")

    return df, too_sparse


def add_housing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm đặc trưng phái sinh phản ánh nghiệp vụ định giá bất động sản."""
    df = df.copy()

    # Người môi giới định giá theo tổng diện tích sử dụng, không theo diện tích
    # đất; nhân với số tầng là cách xấp xỉ trực tiếp đại lượng đó.
    df["Total_Area"] = df["Area"] * df["Floors"].clip(lower=1)

    # Số phòng trên mỗi đơn vị diện tích thể hiện mật độ khai thác căn nhà.
    df["Room_Density"] = (df["Bedrooms"] + df["Bathrooms"]) / df["Area"].clip(lower=1)

    # Tỷ lệ mặt tiền trên diện tích phân biệt nhà phố mặt đường với nhà trong hẻm.
    if "Frontage" in df.columns:
        df["Frontage_Ratio"] = df["Frontage"] / df["Area"].clip(lower=1)

    return df


def prepare_housing(engineered: bool = True) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Trả về X, y và danh sách cột đã bị loại vì thiếu dữ liệu quá nhiều."""
    df, dropped = clean_housing(load_housing_raw())
    if engineered:
        df = add_housing_features(df)

    y = df[cfg.HOUSING_TARGET]
    X = df.drop(columns=[cfg.HOUSING_TARGET])
    return X, y, dropped


def housing_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Phân loại cột thành nhóm định lượng và nhóm định danh."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical
