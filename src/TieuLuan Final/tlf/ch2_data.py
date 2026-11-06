"""Chương 2: nạp, làm sạch, chia và biểu diễn ba bộ dữ liệu.

Mọi phép biến đổi học tham số từ dữ liệu (chuẩn hóa, mã hóa theo mục tiêu, TF-IDF, chọn đặc trưng) chỉ được
fit trên tập huấn luyện. Đó là ràng buộc chính của mô-đun này.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from .common import SEED, data_dir

D = data_dir("ch2")
DATASETS = ("cdc", "housing", "vsfc")
TITLES = {"cdc": "CDC Diabetes", "housing": "Vietnam Housing", "vsfc": "UIT-VSFC"}

CDC_TARGET = "Diabetes_binary"
CDC_SCALE = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
CDC_BINARY = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
              "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"]

HOUSE_NUM = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
HOUSE_CAT = ["House direction", "Balcony direction", "Legal status", "Furniture state", "Province", "Kind"]
HOUSE_TE = ["District", "Ward", "Street"]
HOUSE_TARGET = "Price"

VSFC_LABELS = ["Tiêu cực", "Trung tính", "Tích cực"]


@dataclass
class Split:
    """Ba phần dữ liệu đã biến đổi; pre là bộ biến đổi đã fit (chỉ trên train)."""
    X_tr: object
    X_va: object
    X_te: object
    y_tr: np.ndarray
    y_va: np.ndarray
    y_te: np.ndarray
    pre: object
    task: str
    labels: list = field(default_factory=list)
    names: list = field(default_factory=list)


def split_70_15_15(X, y):
    """Chia phân tầng 70/15/15: tách test trước, rồi tách validation từ phần còn lại."""
    X_a, X_te, y_a, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_a, y_a, test_size=0.15 / 0.85, stratify=y_a, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ------------------------------------------------------------------ CDC Diabetes
def load_cdc():
    """Bỏ cột ID và các hồ sơ trùng khít để một hồ sơ không nằm ở cả train lẫn test."""
    df = pd.read_csv(D / "cdc_diabetes.csv").drop(columns="ID")
    return df.drop_duplicates().reset_index(drop=True)


def cdc_split():
    df = load_cdc()
    X, y = df[CDC_SCALE + CDC_BINARY], df[CDC_TARGET].to_numpy(dtype=int)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_70_15_15(X, y)
    pre = ColumnTransformer([("num", StandardScaler(), CDC_SCALE)], remainder="passthrough",
                            verbose_feature_names_out=False)
    Xs = [pre.fit_transform(X_tr), pre.transform(X_va), pre.transform(X_te)]
    Xs = [x.astype(np.float32) for x in Xs]
    return Split(*Xs, y_tr, y_va, y_te, pre, "clf", ["Không", "Có"], list(pre.get_feature_names_out()))


# ------------------------------------------------------------------ Vietnam Housing
def parse_address(addr):
    """'Đường X, Phường Y, Quận Z, Tỉnh T' -> loại địa chỉ và bốn cấp vị trí; cấp vắng để rỗng."""
    parts = [p.strip() for p in str(addr).split(",") if p.strip()]
    parts = [""] * (4 - len(parts)) + parts[-4:]
    street, ward, district, province = parts
    if street.startswith("Dự án"):
        kind = "Dự án"
    elif street.startswith(("Đường", "Phố")):
        kind = "Đường phố"
    else:
        kind = "Khác"
    return {"Province": province, "Kind": kind,
            "District": f"{district}|{province}",
            "Ward": f"{ward}|{district}|{province}",
            "Street": f"{street}|{district}|{province}"}


def housing_features(df):
    """Thêm năm cột vị trí tách từ Address; không dùng Price nên an toàn trước khi chia tập."""
    loc = pd.DataFrame([parse_address(a) for a in df["Address"]], index=df.index)
    return pd.concat([df, loc], axis=1)


def load_housing():
    return housing_features(pd.read_csv(D / "vietnam_housing.csv"))


def housing_preprocessor():
    """Ba nhánh: số (log diện tích, trung vị, cờ thiếu), hạng mục thưa (one-hot), vị trí (mã hóa theo mục tiêu)."""
    num = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                        SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="Không rõ"),
                        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False))
    te = TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=SEED)
    return ColumnTransformer([("num", num, HOUSE_NUM), ("cat", cat, HOUSE_CAT), ("te", te, HOUSE_TE)],
                             verbose_feature_names_out=False)


def group_split_idx(groups, seed=SEED):
    """Chia 70/15/15 theo Address: mọi tin đăng cùng địa chỉ nằm trọn trong một phần."""
    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    a, te = next(gss.split(idx, groups=groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    i_tr, i_va = next(gss.split(a, groups=groups[a]))
    return a[i_tr], a[i_va], te


def transform_split(df, tr, va, te):
    """Fit tiền xử lý chỉ trên tr rồi áp cho cả ba phần; dùng chung cho chia theo nhóm và chia ngẫu nhiên."""
    cols = HOUSE_NUM + HOUSE_CAT + HOUSE_TE
    pre = housing_preprocessor()
    y = df[HOUSE_TARGET].to_numpy(dtype=np.float32)
    X_tr = pre.fit_transform(df.iloc[tr][cols], y[tr]).astype(np.float32)
    X_va = pre.transform(df.iloc[va][cols]).astype(np.float32)
    X_te = pre.transform(df.iloc[te][cols]).astype(np.float32)
    return Split(X_tr, X_va, X_te, y[tr], y[va], y[te], pre, "reg", [], list(pre.get_feature_names_out()))


def housing_split():
    df = load_housing()
    return transform_split(df, *group_split_idx(df["Address"].to_numpy()))


def housing_random_split():
    """Cách chia ngây thơ theo từng tin đăng, chỉ dùng để đo độ lạc quan so với chia theo địa chỉ."""
    df = load_housing()
    a, te = train_test_split(np.arange(len(df)), test_size=0.15, random_state=SEED)
    tr, va = train_test_split(a, test_size=0.15 / 0.85, random_state=SEED)
    return transform_split(df, tr, va, te)


# ------------------------------------------------------------------ UIT-VSFC
def normalize_vi(text):
    """Chữ thường, gom token che tên riêng (wzjwz...) thành một token, bỏ dấu câu và số, giữ dấu tiếng Việt."""
    text = re.sub(r"wzjwz\d+", " tenrieng ", text.lower())
    text = re.sub(r"[^\w\s]|\d|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def vsfc_vectorizer():
    """Từ đơn và cặp liền kề: tiếng Việt viết tách âm tiết, cặp giữ được 'không hay', 'giáo viên'."""
    return TfidfVectorizer(preprocessor=normalize_vi, token_pattern=r"\S+", ngram_range=(1, 2),
                           min_df=2, max_df=0.9, sublinear_tf=True)


def load_vsfc(part="train"):
    """Phần 'train', 'validation', 'test' theo đúng phân chia gốc của tác giả."""
    return pd.read_parquet(D / f"vsfc_{part}.parquet")


def vsfc_split(top_k=None):
    """top_k=None: TF-IDF đầy đủ, thưa. top_k=k: giữ k đặc trưng chi2 cao nhất, ma trận đặc cho cây và MLP."""
    tr, va, te = load_vsfc("train"), load_vsfc("validation"), load_vsfc("test")
    steps = [("tfidf", vsfc_vectorizer())]
    if top_k:
        steps.append(("chi2", SelectKBest(chi2, k=top_k)))
    pre = Pipeline(steps)
    y_tr = tr["sentiment"].to_numpy(dtype=int)
    Xs = [pre.fit_transform(tr["sentence"], y_tr), pre.transform(va["sentence"]), pre.transform(te["sentence"])]
    if top_k:
        Xs = [x.toarray().astype(np.float32) for x in Xs]
    return Split(*Xs, y_tr, va["sentiment"].to_numpy(dtype=int), te["sentiment"].to_numpy(dtype=int),
                 pre, "clf", VSFC_LABELS, list(pre.get_feature_names_out()))


def get_split(name, dense=False):
    """Điểm vào chung; dense=True cho mô hình cần ma trận đặc (rừng, boosting, MLP)."""
    if name == "cdc":
        return cdc_split()
    if name == "housing":
        return housing_split()
    return vsfc_split(top_k=5000 if dense else None)
