"""Xây dựng và huấn luyện các mô hình Học máy truyền thống.

Mỗi mô hình được gói trong một ``Pipeline`` gồm bước tiền xử lý và bước học.
Nhờ vậy phép chuẩn hoá và mã hoá được học **chỉ trên tập huấn luyện**, rồi áp
lại y nguyên cho tập kiểm tra và cho dữ liệu người dùng nhập trên web — loại bỏ
rò rỉ dữ liệu (data leakage) và bảo đảm biểu diễn nhất quán.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from . import config as cfg
from . import preprocess as prep


# ==========================================================================
#  TÁCH DỮ LIỆU
# ==========================================================================

def three_way_split(X: pd.DataFrame, y: pd.Series, stratify: bool = False):
    """Chia dữ liệu thành ba tập: huấn luyện, kiểm tra và ngoại kiểm.

    Tập ngoại kiểm (OOD) được cắt ra **trước tiên** và không tham gia bất kỳ
    bước nào của quá trình phát triển — kể cả tinh chỉnh siêu tham số. Nó đóng
    vai trò tập dữ liệu "thế giới thực" để đo khả năng tổng quát hoá thật sự,
    trong khi tập kiểm tra thông thường đã gián tiếp ảnh hưởng tới lựa chọn
    mô hình qua nhiều vòng thử nghiệm.
    """
    strat = y if stratify else None
    X_rest, X_ood, y_rest, y_ood = train_test_split(
        X, y, test_size=cfg.OOD_SIZE, random_state=cfg.RANDOM_STATE, stratify=strat
    )

    strat_rest = y_rest if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=strat_rest,
    )
    return X_train, X_test, X_ood, y_train, y_test, y_ood


# ==========================================================================
#  BỘ TIỀN XỬ LÝ
# ==========================================================================

def numeric_preprocessor() -> StandardScaler:
    """Chuẩn hoá về trung bình 0, độ lệch chuẩn 1.

    Bắt buộc với các mô hình dựa trên khoảng cách hoặc gradient (KNN, SVM,
    Logistic Regression): nếu không chuẩn hoá, đặc trưng có thang đo lớn như
    Insulin sẽ lấn át hoàn toàn đặc trưng thang nhỏ như DiabetesPedigreeFunction.
    """
    return StandardScaler()


def diabetes_preprocessor(engineered: bool = True, scale: bool = True) -> Pipeline:
    """Bộ tiền xử lý cho bài toán tiểu đường, đặt hoàn toàn bên trong pipeline.

    Thứ tự ba bước là có chủ đích:

    1. **Điền khuyết** — trung vị được học từ *chỉ tập huấn luyện*. Đặt bước này
       trong pipeline thay vì xử lý trước là điều kiện đủ để loại bỏ rò rỉ dữ
       liệu: ``fit`` chỉ nhìn thấy tập huấn luyện, còn tập kiểm tra chỉ đi qua
       ``transform``.
    2. **Thiết kế đặc trưng** — chạy *sau* khi điền khuyết, để các đặc trưng
       phái sinh như ``Glucose × BMI`` được tính từ giá trị hợp lệ chứ không
       lan truyền giá trị khuyết.
    3. **Chuẩn hoá** — đưa mọi đặc trưng, kể cả đặc trưng phái sinh, về cùng
       thang đo.

    ``set_output(transform="pandas")`` giữ nguyên tên cột giữa các bước; thiếu
    nó thì bước thiết kế đặc trưng nhận mảng NumPy và không truy cập được cột
    theo tên.
    """
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if engineered:
        steps.append(
            ("features", FunctionTransformer(prep.add_diabetes_features, validate=False))
        )
    if scale:
        steps.append(("scale", StandardScaler()))

    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


def mixed_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Bộ tiền xử lý cho dữ liệu vừa có biến số vừa có biến danh mục.

    Giới hạn 30 hạng mục phổ biến nhất cho mỗi biến định danh: cột District có
    hàng trăm giá trị, mã hoá hết sẽ tạo ma trận thưa khổng lồ và khiến mô hình
    học thuộc từng quận hiếm thay vì học quy luật giá.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=30,
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


# ==========================================================================
#  DANH MỤC MÔ HÌNH — BÀI TOÁN PHÂN LỚP
# ==========================================================================

def build_classifiers() -> dict[str, object]:
    """Năm mô hình phân lớp truyền thống, mỗi mô hình một nguyên lý học khác nhau."""
    return {
        # Học ranh giới tuyến tính trong không gian log-odds.
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=cfg.RANDOM_STATE
        ),
        # Không học tham số; quyết định dựa trên k láng giềng gần nhất.
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11),
        # Chia đệ quy không gian đặc trưng bằng các ngưỡng trên từng trục.
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=cfg.RANDOM_STATE
        ),
        # Tổng hợp nhiều cây trên các mẫu bootstrap để giảm phương sai.
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=cfg.RANDOM_STATE, n_jobs=-1
        ),
        # Cộng dồn tuần tự các cây yếu, mỗi cây sửa sai số của tổ hợp trước đó.
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=cfg.RANDOM_STATE,
        ),
    }


def build_classifier_baseline() -> DummyClassifier:
    """Mô hình cơ sở: luôn đoán lớp chiếm đa số.

    Mọi mô hình học được đều phải vượt mốc này, nếu không thì việc học không
    mang lại giá trị nào so với một quy tắc đoán mù (yêu cầu R6).
    """
    return DummyClassifier(strategy="most_frequent")


# ==========================================================================
#  DANH MỤC MÔ HÌNH — BÀI TOÁN HỒI QUY
# ==========================================================================

def build_regressors() -> dict[str, object]:
    """Năm mô hình hồi quy truyền thống."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12, random_state=cfg.RANDOM_STATE
        ),
        # ``min_samples_leaf=5`` chặn cây mọc tới lá chỉ chứa một quan sát.
        # Ngoài tác dụng chống học thuộc, nó còn quyết định kích thước mô hình:
        # để mặc định, rừng 200 cây trên 21.764 dòng chiếm 156 MB trên đĩa và
        # vượt giới hạn 100 MB mỗi tệp của GitHub. Đặt bằng 5 thì còn 44 MB mà
        # R² chỉ giảm từ 0.5976 xuống 0.5925.
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=18, min_samples_leaf=5,
            random_state=cfg.RANDOM_STATE, n_jobs=-1
        ),
        # SVR có độ phức tạp bậc hai theo số mẫu nên rất chậm trên 30 nghìn dòng;
        # cache lớn hơn và dung sai nới rộng giữ thời gian huấn luyện chấp nhận được.
        "Support Vector Regression": SVR(C=10.0, epsilon=0.2, cache_size=1000, tol=1e-2),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=cfg.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_regressor_baseline() -> DummyRegressor:
    """Mô hình cơ sở: luôn dự đoán giá trung vị của tập huấn luyện."""
    return DummyRegressor(strategy="median")


# ==========================================================================
#  HUẤN LUYỆN
# ==========================================================================

def make_pipeline(preprocessor, estimator) -> Pipeline:
    """Ghép bước tiền xử lý và bước học thành một khối triển khai duy nhất."""
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def fit_all(
    models: dict[str, object],
    preprocessor_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Huấn luyện toàn bộ danh mục mô hình trên cùng một tập dữ liệu.

    ``preprocessor_factory`` phải tạo ra một đối tượng **mới** cho mỗi mô hình:
    dùng chung một instance sẽ khiến các pipeline chia sẻ trạng thái đã fit và
    kết quả phụ thuộc vào thứ tự huấn luyện.
    """
    fitted: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        pipe = make_pipeline(preprocessor_factory(), estimator)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted


def save_model(pipeline: Pipeline, filename: str) -> None:
    """Lưu pipeline đã huấn luyện để ứng dụng web nạp lại.

    Nén mức 3 thu nhỏ tệp khoảng ba tới bốn lần: cấu trúc cây gồm nhiều mảng
    lặp lại nên nén rất hiệu quả. Đây là điều kiện để kho mã đẩy được lên
    GitHub và để Streamlit Cloud tải mô hình đủ nhanh.
    """
    joblib.dump(pipeline, cfg.MODELS_DIR / filename, compress=3)


def load_model(filename: str) -> Pipeline:
    """Nạp pipeline đã lưu."""
    return joblib.load(cfg.MODELS_DIR / filename)
