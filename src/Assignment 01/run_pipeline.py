"""Chạy toàn bộ quy trình thực nghiệm từ dữ liệu thô đến mô hình và biểu đồ.

Một lệnh duy nhất tái tạo lại mọi con số và mọi hình xuất hiện trong báo cáo:

    python run_pipeline.py

Kết quả ghi ra ba nơi: ``models/`` (pipeline đã huấn luyện), ``outputs/``
(bảng kết quả CSV) và ``figures/`` (biểu đồ PNG).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

from src import config as cfg
from src import evaluate as ev
from src import preprocess as prep
from src import train as tr
from src import viz


def log(message: str) -> None:
    print(f"  {message}", flush=True)


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}", flush=True)


# ==========================================================================
#  BÀI TOÁN 1 — CHẨN ĐOÁN TIỂU ĐƯỜNG
# ==========================================================================

def run_diabetes() -> dict:
    section("BÀI TOÁN 1 — CHẨN ĐOÁN NGUY CƠ TIỂU ĐƯỜNG (PHÂN LỚP)")
    summary: dict = {}

    raw = prep.load_diabetes_raw()
    # Bản đã điền khuyết dùng riêng cho biểu đồ; mô hình dùng bản còn khuyết,
    # để việc điền khuyết diễn ra bên trong pipeline và không rò rỉ dữ liệu.
    eda = prep.diabetes_eda_frame()
    log(f"Dữ liệu thô: {raw.shape[0]} quan sát × {raw.shape[1] - 1} đặc trưng")

    zero_counts = {c: int((raw[c] == 0).sum()) for c in cfg.DIABETES_ZERO_AS_MISSING}
    log(f"Giá trị 0 phi lý đã đánh dấu khuyết: {zero_counts}")
    summary["zero_as_missing"] = zero_counts
    summary["n_raw"] = int(raw.shape[0])
    summary["n_features_raw"] = int(raw.shape[1] - 1)
    summary["n_features_engineered"] = int(eda.shape[1] - 1)
    summary["class_balance"] = raw[cfg.DIABETES_TARGET].value_counts().to_dict()

    # ---------------- Phân tích khám phá ----------------
    log("Đang sinh biểu đồ khám phá dữ liệu...")
    viz.plot_system_diagram()
    viz.plot_target_pie(raw[cfg.DIABETES_TARGET], "dia_pie_outcome",
                        labels=["Âm tính (0)", "Dương tính (1)"])
    viz.plot_missing_bar(raw[cfg.DIABETES_ZERO_AS_MISSING].replace(0, np.nan),
                         "dia_missing_ratio")
    for col in ["Glucose", "BMI", "Age", "Insulin"]:
        viz.plot_kde(eda, col, f"dia_kde_{col.lower()}")
    viz.plot_histogram(eda, "Glucose", "dia_hist_glucose")
    viz.plot_boxplot(eda, "Age", "dia_box_age")
    viz.plot_boxplot_grid(
        eda, ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "dia_box_grid")
    viz.plot_kde_by_class(eda, ["Glucose", "BMI", "Age"], cfg.DIABETES_TARGET,
                          "dia_kde_by_class", labels=["Khoẻ mạnh", "Mắc bệnh"])
    viz.plot_correlation_heatmap(
        eda, "dia_corr_heatmap",
        "Ma trận tương quan Pearson (8 đặc trưng gốc + 4 đặc trưng thiết kế)")

    # ---------------- Chia dữ liệu ----------------
    X, y = prep.prepare_diabetes(clean=True)
    X_train, X_test, X_ood, y_train, y_test, y_ood = tr.three_way_split(
        X, y, stratify=True)
    log(f"Chia dữ liệu — huấn luyện {len(X_train)} · kiểm tra {len(X_test)} "
        f"· ngoại kiểm {len(X_ood)}")
    summary["split"] = {"train": len(X_train), "test": len(X_test), "ood": len(X_ood)}

    # ---------------- Mô hình cơ sở ----------------
    baseline = tr.make_pipeline(tr.diabetes_preprocessor(), tr.build_classifier_baseline())
    baseline.fit(X_train, y_train)
    base_metrics = ev.classification_metrics(y_test, baseline.predict(X_test))
    log(f"Mô hình cơ sở (đoán lớp đa số) — Accuracy {base_metrics['Accuracy']:.4f}, "
        f"Recall {base_metrics['Recall']:.4f}")
    summary["baseline"] = base_metrics

    # ---------------- Thực nghiệm 1: so sánh mô hình ----------------
    log("Thực nghiệm 1 — huấn luyện 5 mô hình...")
    started = time.time()
    fitted = tr.fit_all(tr.build_classifiers(), tr.diabetes_preprocessor, X_train, y_train)
    results = ev.compare_classifiers(fitted, X_test, y_test)
    results_with_base = pd.concat(
        [pd.DataFrame([{"Model": "Baseline (Dummy)", **base_metrics}]), results],
        ignore_index=True)
    ev.save_table(results_with_base, "dia_model_comparison")
    log(f"Hoàn tất trong {time.time() - started:.1f}s")
    print(results_with_base.to_string(index=False, float_format="%.4f"))

    viz.plot_model_comparison(results, ["Accuracy", "Precision", "Recall", "F1-Score"],
                              "dia_model_comparison",
                              "So sánh 5 mô hình phân lớp trên cùng tập kiểm tra")

    # Hai hình bổ sung nhìn cùng một kết quả từ hai góc mà biểu đồ cột không
    # thấy: ROC đo khả năng xếp hạng độc lập với ngưỡng, còn lưới ma trận nhầm
    # lẫn cho biết mỗi mô hình sai *theo kiểu gì* chứ không chỉ sai bao nhiêu.
    viz.plot_roc_curves(fitted, X_test, y_test, "dia_roc_curves",
                        "Đường cong ROC của 5 mô hình phân lớp (tập kiểm tra)")
    viz.plot_confusion_grid(fitted, X_test, y_test, "dia_confusion_grid",
                            labels=["Khoẻ mạnh", "Mắc bệnh"],
                            title="Ma trận nhầm lẫn của 5 mô hình — cùng thang màu")

    best_name = results.iloc[0]["Model"]
    best_model = fitted[best_name]
    log(f"Mô hình tốt nhất theo F1-Score: {best_name}")
    summary["best_model"] = best_name

    # ---------------- Thực nghiệm 2: siêu tham số ----------------
    log("Thực nghiệm 2 — khảo sát k trong K-Nearest Neighbors...")
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    k_sweep = ev.sweep_hyperparameter(
        lambda k: tr.make_pipeline(tr.diabetes_preprocessor(), KNeighborsClassifier(n_neighbors=k)),
        k_values, X_train, y_train, X_test, y_test,
        ev.classification_metrics, "F1-Score", "k")
    ev.save_table(k_sweep, "dia_knn_k_sweep")
    print(k_sweep.to_string(index=False, float_format="%.4f"))
    viz.plot_hyperparameter_sweep(
        k_sweep, "k", ["F1-Score", "Accuracy", "Recall", "Precision"],
        "dia_knn_k_sweep",
        "Ảnh hưởng của số láng giềng k tới hiệu năng K-Nearest Neighbors")
    best_k = int(k_sweep.loc[k_sweep["F1-Score"].idxmax(), "k"])
    log(f"Giá trị k tối ưu: {best_k}")
    summary["best_k"] = best_k

    # Quét ngưỡng phải chạy trên mô hình cho xác suất liên tục. Cây quyết định
    # nông chỉ trả về vài giá trị xác suất rời rạc theo độ thuần của lá, nên mọi
    # ngưỡng trong khoảng giữa hai giá trị đó cho ra đúng một kết quả — đường
    # cong sẽ phẳng và thực nghiệm mất hoàn toàn ý nghĩa.
    log("Thực nghiệm 2b — khảo sát ngưỡng quyết định θ trên XGBoost...")
    threshold_model = fitted["XGBoost"]
    thresholds = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30,
                  0.25, 0.20, 0.15, 0.10, 0.07, 0.05]
    th_sweep = ev.sweep_decision_threshold(threshold_model, X_test, y_test, thresholds)
    ev.save_table(th_sweep, "dia_threshold_sweep")
    print(th_sweep.to_string(index=False, float_format="%.4f"))
    viz.plot_threshold_curve(th_sweep, "dia_threshold_curve")

    best_theta = float(th_sweep.loc[th_sweep["F1-Score"].idxmax(), "Threshold"])
    n_distinct = th_sweep["TP"].nunique()
    log(f"Ngưỡng tối ưu theo F1-Score: θ = {best_theta} "
        f"({n_distinct} mức TP khác nhau trên {len(thresholds)} ngưỡng)")
    summary["best_threshold"] = best_theta
    summary["threshold_distinct_outcomes"] = int(n_distinct)

    viz.plot_confusion_matrix(threshold_model, X_test, y_test, "dia_confusion_default",
                              labels=["Khoẻ mạnh", "Mắc bệnh"])
    viz.plot_confusion_matrix(threshold_model, X_test, y_test, "dia_confusion_tuned",
                              labels=["Khoẻ mạnh", "Mắc bệnh"], threshold=best_theta)

    # ---------------- Thực nghiệm 3: biểu diễn dữ liệu ----------------
    log("Thực nghiệm 3 — so sánh các phương án biểu diễn dữ liệu...")
    X_raw, y_raw = prep.prepare_diabetes(clean=False)
    Xr_tr, Xr_te, _, yr_tr, yr_te, _ = tr.three_way_split(X_raw, y_raw, stratify=True)

    selected = ev.correlation_feature_selection(X_train, y_train, threshold=0.20)
    log(f"Đặc trưng được chọn theo tương quan (|ρ| ≥ 0.20): {selected}")
    summary["selected_features"] = selected

    # Thuật toán và siêu tham số giữ nguyên tuyệt đối ở cả bốn phương án; chỉ
    # ma trận đặc trưng X thay đổi, nên mọi chênh lệch quy hoàn toàn về biểu diễn.
    def logistic():
        return LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE)

    rep_rows = []
    for label, (Xtr_, Xte_, ytr_, yte_, prep_step) in {
        "Thô — giữ nguyên giá trị 0 phi lý":
            (Xr_tr, Xr_te, yr_tr, yr_te, tr.diabetes_preprocessor(engineered=False)),
        "Làm sạch, chưa chuẩn hoá":
            (X_train, X_test, y_train, y_test,
             tr.diabetes_preprocessor(engineered=False, scale=False)),
        "Làm sạch + chuẩn hoá":
            (X_train, X_test, y_train, y_test,
             tr.diabetes_preprocessor(engineered=False)),
        "Làm sạch + chuẩn hoá + đặc trưng thiết kế":
            (X_train, X_test, y_train, y_test, tr.diabetes_preprocessor(engineered=True)),
    }.items():
        pipe = tr.make_pipeline(prep_step, logistic())
        pipe.fit(Xtr_, ytr_)
        rep_rows.append({"Representation": label,
                         **ev.classification_metrics(yte_, pipe.predict(Xte_))})

    rep_results = (pd.DataFrame(rep_rows)
                     .sort_values("F1-Score", ascending=False).reset_index(drop=True))
    ev.save_table(rep_results, "dia_representation")
    print(rep_results.to_string(index=False, float_format="%.4f"))
    viz.plot_representation_comparison(
        rep_results, "F1-Score", "dia_representation",
        "Ảnh hưởng của cách biểu diễn dữ liệu tới F1-Score (Logistic Regression)")

    # ---------------- Kiểm chuẩn ngoại kiểm ----------------
    log("Kiểm chuẩn trên tập ngoại kiểm (dữ liệu mô hình chưa từng thấy)...")
    ood_results = ev.compare_classifiers(fitted, X_ood, y_ood)
    ev.save_table(ood_results, "dia_ood_benchmark")
    viz.plot_ood_comparison(results, ood_results, "F1-Score", "dia_ood_comparison",
                            "Hiệu năng trên tập kiểm tra so với tập ngoại kiểm")

    # ---------------- Lưu mô hình ----------------
    tr.save_model(best_model, "diabetes_best.joblib")
    tr.save_model(threshold_model, "diabetes_xgboost_tuned.joblib")
    for name, pipe in fitted.items():
        tr.save_model(pipe, f"diabetes_{name.lower().replace(' ', '_').replace('-', '_')}.joblib")
    viz.plot_feature_importance(
        fitted["XGBoost"], "dia_feature_importance",
        title="Mức độ quan trọng của đặc trưng — XGBoost (Tiểu đường)")

    summary["feature_columns"] = X.columns.tolist()
    log(f"Đã lưu {len(fitted) + 1} mô hình vào models/")
    return summary


# ==========================================================================
#  BÀI TOÁN 2 — ĐỊNH GIÁ BẤT ĐỘNG SẢN
# ==========================================================================

def run_housing() -> dict:
    section("BÀI TOÁN 2 — ĐỊNH GIÁ BẤT ĐỘNG SẢN VIỆT NAM (HỒI QUY)")
    summary: dict = {}

    raw = prep.load_housing_raw()
    log(f"Dữ liệu thô: {raw.shape[0]} tin đăng × {raw.shape[1] - 1} thuộc tính")
    summary["n_raw"] = int(raw.shape[0])
    summary["missing_ratio"] = (raw.isna().mean() * 100).round(2).to_dict()

    viz.plot_missing_bar(raw, "hou_missing_ratio")

    cleaned, dropped = prep.clean_housing(raw)
    engineered = prep.add_housing_features(cleaned)
    log(f"Cột bị loại vì thiếu trên {cfg.MISSING_DROP_THRESHOLD:.0%}: {dropped}")
    summary["dropped_columns"] = dropped

    # ---------------- Phân tích khám phá ----------------
    log("Đang sinh biểu đồ khám phá dữ liệu...")
    viz.plot_kde(cleaned, cfg.HOUSING_TARGET, "hou_kde_price", "(tỷ VNĐ)")
    viz.plot_kde(cleaned, "Area", "hou_kde_area", "(m²)")
    viz.plot_histogram(cleaned, cfg.HOUSING_TARGET, "hou_hist_price")
    viz.plot_boxplot(cleaned, cfg.HOUSING_TARGET, "hou_box_price", "(tỷ VNĐ)")
    viz.plot_boxplot_grid(cleaned, ["Price", "Area", "Floors", "Bedrooms",
                                    "Bathrooms", "Frontage", "Access Road"],
                          "hou_box_grid", ncols=4)
    viz.plot_scatter(cleaned, "Area", cfg.HOUSING_TARGET, "hou_scatter_area_price")
    viz.plot_price_area_bedrooms(
        cleaned, "hou_price_area_bedrooms",
        "Khám phá liên hợp: Giá — Diện tích — Số phòng ngủ")
    viz.plot_category_bar(cleaned, "Province", cfg.HOUSING_TARGET, "hou_bar_province")
    viz.plot_category_bar(cleaned, "Legal status", cfg.HOUSING_TARGET, "hou_bar_legal", top=5)
    viz.plot_correlation_heatmap(engineered, "hou_corr_heatmap",
                                 "Ma trận tương quan Pearson — đặc trưng bất động sản")

    # ---------------- Chia dữ liệu ----------------
    X, y, _ = prep.prepare_housing(engineered=True)
    numeric, categorical = prep.housing_column_types(X)
    log(f"Biểu diễn: {len(numeric)} đặc trưng định lượng, "
        f"{len(categorical)} đặc trưng định danh")
    summary["numeric_features"] = numeric
    summary["categorical_features"] = categorical

    X_train, X_test, X_ood, y_train, y_test, y_ood = tr.three_way_split(X, y)
    log(f"Chia dữ liệu — huấn luyện {len(X_train)} · kiểm tra {len(X_test)} "
        f"· ngoại kiểm {len(X_ood)}")
    summary["split"] = {"train": len(X_train), "test": len(X_test), "ood": len(X_ood)}

    def make_prep():
        return tr.mixed_preprocessor(numeric, categorical)

    # ---------------- Mô hình cơ sở ----------------
    baseline = tr.make_pipeline(make_prep(), tr.build_regressor_baseline())
    baseline.fit(X_train, y_train)
    base_metrics = ev.regression_metrics(y_test, baseline.predict(X_test))
    log(f"Mô hình cơ sở (đoán trung vị) — MAE {base_metrics['MAE']:.4f}, "
        f"R² {base_metrics['R2']:.4f}")
    summary["baseline"] = base_metrics

    # ---------------- Thực nghiệm 1: so sánh mô hình ----------------
    log("Thực nghiệm 1 — huấn luyện 5 mô hình (SVR chậm, vui lòng đợi)...")
    started = time.time()
    fitted = tr.fit_all(tr.build_regressors(), make_prep, X_train, y_train)
    results = ev.compare_regressors(fitted, X_test, y_test)
    results_with_base = pd.concat(
        [pd.DataFrame([{"Model": "Baseline (Median)", **base_metrics}]), results],
        ignore_index=True)
    ev.save_table(results_with_base, "hou_model_comparison")
    log(f"Hoàn tất trong {time.time() - started:.1f}s")
    print(results_with_base.to_string(index=False, float_format="%.4f"))

    viz.plot_model_comparison(results, ["MAE", "RMSE", "R2"], "hou_model_comparison",
                              "So sánh 5 mô hình hồi quy trên cùng tập kiểm tra")

    best_name = results.iloc[0]["Model"]
    best_model = fitted[best_name]
    log(f"Mô hình tốt nhất theo R²: {best_name}")
    summary["best_model"] = best_name

    y_pred_best = best_model.predict(X_test)
    viz.plot_prediction_scatter(y_test, y_pred_best, "hou_pred_vs_true")

    # Bài hồi quy không có ROC, nhưng câu hỏi "sai theo kiểu gì" thì vẫn nguyên
    # giá trị — và bảng thứ tư của hình này chính là chỗ phát hiện độ chệch đơn
    # điệu theo phân khúc giá được ghi lại ở Mục 4.2 của báo cáo.
    viz.plot_residual_analysis(y_test, y_pred_best, "hou_residual_analysis",
                               f"Phân tích phần dư của mô hình {best_name}")

    # ---------------- Thực nghiệm 2: siêu tham số ----------------
    log("Thực nghiệm 2 — khảo sát độ sâu tối đa của Random Forest...")
    depths = [4, 6, 8, 12, 16, 20, 24, 28, 32, 40]
    depth_sweep = ev.sweep_hyperparameter(
        lambda d: tr.make_pipeline(
            make_prep(),
            RandomForestRegressor(n_estimators=120, max_depth=d, min_samples_leaf=5,
                                  random_state=cfg.RANDOM_STATE, n_jobs=-1)),
        depths, X_train, y_train, X_test, y_test,
        ev.regression_metrics, "R2", "max_depth")
    ev.save_table(depth_sweep, "hou_depth_sweep")
    print(depth_sweep.to_string(index=False, float_format="%.4f"))
    viz.plot_hyperparameter_sweep(
        depth_sweep, "max_depth", ["R2"], "hou_depth_sweep",
        "Ảnh hưởng của độ sâu tối đa tới R² của Random Forest")
    best_depth = int(depth_sweep.loc[depth_sweep["R2"].idxmax(), "max_depth"])
    log(f"Độ sâu tối ưu: {best_depth}")
    summary["best_depth"] = best_depth

    # ---------------- Thực nghiệm 3: biểu diễn dữ liệu ----------------
    log("Thực nghiệm 3 — so sánh các phương án biểu diễn dữ liệu...")
    X_plain, y_plain, _ = prep.prepare_housing(engineered=False)
    num_p, cat_p = prep.housing_column_types(X_plain)
    Xp_tr, Xp_te, _, yp_tr, yp_te, _ = tr.three_way_split(X_plain, y_plain)

    X_num_only = X[numeric]
    Xn_tr, Xn_te, _, yn_tr, yn_te, _ = tr.three_way_split(X_num_only, y)

    rep_rows = []
    for label, (Xtr_, Xte_, ytr_, yte_, nums, cats) in {
        "Chỉ đặc trưng định lượng": (Xn_tr, Xn_te, yn_tr, yn_te, numeric, []),
        "Định lượng + định danh, chưa thiết kế": (Xp_tr, Xp_te, yp_tr, yp_te, num_p, cat_p),
        "Đầy đủ + đặc trưng thiết kế": (X_train, X_test, y_train, y_test, numeric, categorical),
    }.items():
        pipe = tr.make_pipeline(tr.mixed_preprocessor(nums, cats),
                                RandomForestRegressor(n_estimators=120, max_depth=16, min_samples_leaf=5,
                                                      random_state=cfg.RANDOM_STATE,
                                                      n_jobs=-1))
        pipe.fit(Xtr_, ytr_)
        rep_rows.append({"Representation": label,
                         **ev.regression_metrics(yte_, pipe.predict(Xte_))})

    rep_results = pd.DataFrame(rep_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    ev.save_table(rep_results, "hou_representation")
    print(rep_results.to_string(index=False, float_format="%.4f"))
    viz.plot_representation_comparison(
        rep_results, "R2", "hou_representation",
        "Ảnh hưởng của cách biểu diễn dữ liệu tới R² (Random Forest)")

    # ---------------- Kiểm chuẩn ngoại kiểm ----------------
    log("Kiểm chuẩn trên tập ngoại kiểm...")
    ood_results = ev.compare_regressors(fitted, X_ood, y_ood)
    ev.save_table(ood_results, "hou_ood_benchmark")
    viz.plot_ood_comparison(results, ood_results, "R2", "hou_ood_comparison",
                            "Hiệu năng R² trên tập kiểm tra so với tập ngoại kiểm")

    # ---------------- Lưu mô hình ----------------
    tr.save_model(best_model, "housing_best.joblib")
    for name, pipe in fitted.items():
        tr.save_model(pipe, f"housing_{name.lower().replace(' ', '_').replace('-', '_')}.joblib")
    viz.plot_feature_importance(
        fitted["XGBoost"], "hou_feature_importance",
        title="Mức độ quan trọng của đặc trưng — XGBoost (Bất động sản)")

    summary["feature_columns"] = X.columns.tolist()
    summary["categories"] = {c: sorted(X[c].astype(str).unique().tolist())[:60]
                             for c in categorical}
    log(f"Đã lưu {len(fitted) + 1} mô hình vào models/")
    return summary


# ==========================================================================
#  ĐIỂM VÀO
# ==========================================================================

def run_knowledge_graph(offline: bool) -> dict:
    """Dựng Đồ thị Tri thức từ những mô hình vừa lưu ở hai bước trên.

    Bước này **đọc** ``models/`` chứ không ghi, nên nó không đụng tới thứ mà hai
    bước trước vừa sinh ra. Nó cũng là bước duy nhất trong toàn pipeline có thể
    chạm tới mạng — và cũng vì thế mà nó không bao giờ được phép làm hỏng cả
    lệnh: thiếu tài khoản Neo4j thì ``build_all`` in cảnh báo rồi đi tiếp bằng
    đồ thị offline, đúng như thiết kế hai đường của Phase 5.
    """
    # Nhập tại chỗ để hai bước huấn luyện ở trên không phải chờ nạp networkx,
    # pyvis và trình điều khiển Neo4j khi người dùng chạy với --skip-kg.
    from src.kg.build_graph import build_all
    from src.kg import ontology as ont

    # Sơ đồ kiến trúc đa tầng vẽ từ chính KG_LAYERS của bản thể học, nên nó luôn
    # mô tả đúng lược đồ mà mã nguồn đang dùng.
    viz.plot_kg_architecture(
        ont.KG_LAYERS["medical"], ont.NODE_COLOURS, "fig_kg_architecture_medical",
        "Kiến trúc Đồ thị Tri thức Y sinh — 5 tầng",
        "Từ chỉ số sinh tồn tới gói chăm sóc mua được tại nhà thuốc")
    viz.plot_kg_architecture(
        ont.KG_LAYERS["property"], ont.NODE_COLOURS, "fig_kg_architecture_property",
        "Kiến trúc Đồ thị Tri thức Đô thị – Tài chính — 5 tầng",
        "Từ thuộc tính căn nhà tới hạn mức vay và tiện ích hạ tầng quanh đó")

    result = build_all(offline=offline)
    return {
        "so_ca": len(result["cases"]),
        "so_bo_ba": len(result["triplets"]),
        "neo4j": result["neo4j"]["nodes"] if result["neo4j"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy lại toàn bộ pipeline: dữ liệu, mô hình, thực nghiệm, đồ thị.")
    parser.add_argument("--skip-kg", action="store_true",
                        help="Bỏ qua bước dựng Đồ thị Tri thức")
    parser.add_argument("--kg-offline", action="store_true",
                        help="Dựng đồ thị nhưng không kết nối Neo4j")
    args = parser.parse_args()

    started = time.time()
    report = {
        "diabetes": run_diabetes(),
        "housing": run_housing(),
        "config": {
            "random_state": cfg.RANDOM_STATE,
            "test_size": cfg.TEST_SIZE,
            "ood_size": cfg.OOD_SIZE,
        },
    }
    # Bước đồ thị chạy sau cùng và mất chưa tới 10 giây, nhưng nếu nó ném lỗi
    # thì 90 giây huấn luyện phía trên mất trắng vì tóm tắt chưa kịp ghi. Giữ
    # lỗi lại, ghi tóm tắt, rồi mới ném tiếp — không nuốt, chỉ hoãn.
    kg_error: Exception | None = None
    if not args.skip_kg:
        try:
            report["knowledge_graph"] = run_knowledge_graph(offline=args.kg_offline)
        except Exception as exc:                       # noqa: BLE001 — ném lại ở cuối
            kg_error = exc
            report["knowledge_graph"] = {"loi": f"{type(exc).__name__}: {exc}"}

    path = cfg.OUTPUTS_DIR / "pipeline_summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")

    section("HOÀN TẤT")
    log(f"Tổng thời gian: {time.time() - started:.1f}s")
    log(f"Biểu đồ:     {len(list(cfg.FIGURES_DIR.glob('*.png')))} tệp trong figures/")
    log(f"Bảng kết quả: {len(list(cfg.OUTPUTS_DIR.glob('*.csv')))} tệp trong outputs/")
    log(f"Mô hình:     {len(list(cfg.MODELS_DIR.glob('*.joblib')))} tệp trong models/")
    if args.skip_kg:
        log("Đồ thị:      bỏ qua theo yêu cầu (--skip-kg)")
    elif kg_error is not None:
        log(f"Đồ thị:      HỎNG — {type(kg_error).__name__}")
    else:
        kg = report["knowledge_graph"]
        tren_may_chu = f"{kg['neo4j']} nút trên Neo4j" if kg["neo4j"] else "offline"
        log(f"Đồ thị:      {kg['so_ca']} ca, {kg['so_bo_ba']} bộ ba, {tren_may_chu}")
    log(f"Tóm tắt:     {path.name}")

    if kg_error is not None:
        raise kg_error


if __name__ == "__main__":
    main()
