"""Chương 2: họ mô hình cổ điển và lưới siêu tham số; chọn trên validation, test chỉ dùng một lần ở cuối."""
from itertools import product

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from .common import SEED

JOBS = 4
NAMES = {"dummy": "Dummy", "linear": "LogReg/Ridge", "svm": "Linear SVM", "tree": "Cây quyết định",
         "rf": "Random Forest", "xgb": "XGBoost", "mlp_numpy": "MLP NumPy", "mlp_keras": "MLP Keras",
         "mlp_torch": "MLP PyTorch"}
GRID = {
    "linear": {"C": [0.1, 1, 10]},
    "ridge": {"alpha": [0.1, 1, 10, 100]},
    "svm": {"C": [0.01, 0.1, 1]},
    "tree": {"max_depth": [6, 10, 14], "min_samples_leaf": [5, 20]},
    "rf": {"max_depth": [12, 20, None], "min_samples_leaf": [2, 10]},
    "xgb": {"max_depth": [3, 5, 7], "reg_lambda": [1, 5]},
}


def grid(family, task):
    """Mọi tổ hợp tham số của một họ; họ tuyến tính dùng Ridge khi hồi quy."""
    g = GRID["ridge" if family == "linear" and task == "reg" else family]
    return [dict(zip(g, vals)) for vals in product(*g.values())]


def _xgb(p, task):
    common = dict(n_estimators=500, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  early_stopping_rounds=30, n_jobs=JOBS, random_state=SEED, **p)
    return XGBRegressor(**common) if task == "reg" else XGBClassifier(**common)


def make(family, p, task, balanced=False):
    """Tạo mô hình; balanced=True nhân trọng số lớp nghịch đảo tần suất (dùng cho UIT-VSFC ba lớp lệch)."""
    cw = "balanced" if balanced else None
    if family == "dummy":
        return DummyRegressor() if task == "reg" else DummyClassifier(strategy="most_frequent")
    if family == "xgb":
        return _xgb(p, task)
    if task == "reg":
        return {"linear": lambda: Ridge(**p),
                "svm": lambda: LinearSVR(random_state=SEED, max_iter=5000, dual="auto", **p),
                "tree": lambda: DecisionTreeRegressor(random_state=SEED, **p),
                "rf": lambda: RandomForestRegressor(n_estimators=200, max_features=0.5, n_jobs=JOBS,
                                                    random_state=SEED, **p)}[family]()
    return {"linear": lambda: LogisticRegression(max_iter=3000, class_weight=cw, **p),
            "svm": lambda: CalibratedClassifierCV(LinearSVC(class_weight=cw, dual="auto", **p), cv=3),
            "tree": lambda: DecisionTreeClassifier(class_weight=cw, random_state=SEED, **p),
            "rf": lambda: RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=JOBS,
                                                 random_state=SEED, **p)}[family]()
