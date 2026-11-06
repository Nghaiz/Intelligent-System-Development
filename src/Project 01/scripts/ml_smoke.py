"""Chạy thử API phục vụ theo hợp đồng và ghi ml_api_smoke.json; lỗi hợp đồng thì dừng bằng AssertionError."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnx.common import ROOT, limit_threads, save_json  # noqa: E402

limit_threads(4)
from dnx.ml import serve, serve_retina  # noqa: E402

LAB = {"age": 55, "sex": "male", "bmi": 31.0, "hba1c": 6.9, "glucose": 170, "hypertension": 1, "heart_disease": 0,
       "smoking": "former"}
LOW = {"age": 30, "sex": "female", "bmi": 22.0, "hba1c": 5.0, "glucose": 90, "hypertension": 0, "heart_disease": 0,
       "smoking": "never"}
SCREEN = {"age": 60, "sex": "female", "bmi": 33.0, "high_bp": 1, "high_chol": 1, "gen_hlth": 4, "phys_activity": 0,
          "smoking": "never"}
HOSP = {"age": "[70-80)", "gender": "Female", "race": "Caucasian", "time_in_hospital": 6, "num_medications": 18,
        "number_inpatient": 3, "number_emergency": 1, "number_diagnoses": 9, "diag_1": "428", "insulin": "Up",
        "diabetesmed": "Yes", "change": "Ch", "discharge_disposition_id": 1, "admission_type_id": 1}
HOSP_LOW = {"age": "[40-50)", "gender": "Male", "time_in_hospital": 1, "number_inpatient": 0, "number_diagnoses": 3,
            "discharge_disposition_id": 1}


def check_predict(r, mode):
    assert r["mode"] == mode and 0 <= r["prob"] <= 1 and r["label"] in (0, 1) and r["calibrated"] is True
    assert r["label"] == int(r["prob"] >= r["threshold"])
    mags = [abs(s["contribution"]) for s in r["shap"]]
    assert mags == sorted(mags, reverse=True), "shap phải giảm dần"
    assert all({"feature", "label_vi", "value", "contribution"} <= set(s) for s in r["shap"])
    # Cộng SHAP + base phải tái tạo log-odds của xác suất thô
    import math
    margin = sum(s["contribution"] for s in r["shap"]) + r["base_value"]
    assert abs(margin - math.log(r["raw_prob"] / (1 - r["raw_prob"]))) < 1e-2, "SHAP không cộng về margin"


def expect_error(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"Mong lỗi với {a}")


def main():
    out, t0 = {}, time.perf_counter()
    r = serve.predict(LAB)
    out["load_and_first_predict_s"] = time.perf_counter() - t0
    check_predict(r, "full")
    out["predict_full"] = {k: v for k, v in r.items() if k != "shap"} | {"top3": r["shap"][:3]}
    t0 = time.perf_counter()
    low = serve.predict(LOW)
    out["predict_ms"] = (time.perf_counter() - t0) * 1000
    check_predict(low, "full")
    assert low["prob"] < r["prob"]
    out["predict_full_low"] = {"prob": low["prob"], "label": low["label"]}
    s = serve.predict(SCREEN)
    check_predict(s, "screening")
    out["predict_screening"] = {k: v for k, v in s.items() if k != "shap"} | {"top3": s["shap"][:3]}
    out["what_if_hba1c"] = serve.what_if(LAB, {"hba1c": 5.5})
    out["what_if_bmi_screening"] = serve.what_if(SCREEN, {"bmi": 25.0, "phys_activity": 1})
    a = out["what_if_hba1c"]
    assert abs(a["delta"] - (a["after"] - a["before"])) < 1e-12
    out["readmission_high"] = serve.readmission(HOSP)
    out["readmission_low"] = serve.readmission(HOSP_LOW)
    assert out["readmission_high"]["prob"] > out["readmission_low"]["prob"]
    out["errors"] = {"sex": expect_error(serve.predict, {**LAB, "sex": "x"}),
                     "full_missing": expect_error(serve.predict, {"age": 40}, "full")}
    card = serve.model_card()
    assert set(card["tasks"]) == {"full", "screening", "readmission"}
    out["model_card_tasks"] = list(card["tasks"])
    img = ROOT / "web/static/img/retina"
    if (serve_retina.MDIR / "selected.txt").exists():
        out["retina"] = {}
        for f in sorted(img.glob("*.jpg")):
            t0 = time.perf_counter()
            g = serve_retina.grade(f.read_bytes())
            assert 0 <= g["grade"] <= 4 and len(g["probs"]) == 5 and abs(sum(g["probs"]) - 1) < 1e-4
            assert g["gradcam_png_b64"].startswith("iVBOR")
            out["retina"][f.name] = {"grade": g["grade"], "label_vi": g["label_vi"], "p": round(max(g["probs"]), 3),
                                     "ms": round((time.perf_counter() - t0) * 1000)}
        out["retina_bad_input"] = expect_error(serve_retina.grade, b"not an image")
    save_json("ml_api_smoke", out)
    print("OK", out)


if __name__ == "__main__":
    main()
