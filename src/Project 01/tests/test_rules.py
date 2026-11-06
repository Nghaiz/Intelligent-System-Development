"""Bộ luật: đủ số lượng và nhóm, ngưỡng biên, suy diễn tiến nhiều bước, ngưỡng lấy từ KB."""
import sys
import types

import pytest

from dnx import rules as R


def ids(**facts):
    return {r["rule_id"] for r in R.evaluate(facts)}


def test_rule_table_shape():
    rows = R.load_rules()
    assert len(rows) >= 25
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["group"] for r in rows} == {"urgent", "diagnostic", "medication", "policy", "risk"}
    for r in rows:
        assert r["severity"] in R.SEVERITY_RANK and r["source"] and r["message"]


def test_output_contract():
    row = R.evaluate({"glucose": 700})[0]
    assert set(row) >= {"rule_id", "severity", "message", "source", "fired_on"}
    assert row["rule_id"] == "URG-01" and row["fired_on"]["glucose"] == 700


@pytest.mark.parametrize("g,rule,fires", [(299, "URG-03", False), (300, "URG-03", True), (54, "URG-04", False),
                                          (53.9, "URG-04", True), (70, "URG-12", False), (69, "URG-12", True)])
def test_glucose_boundaries(g, rule, fires):
    assert (rule in ids(glucose=g)) is fires


def test_mmol_is_converted():
    assert "URG-03" in ids(glucose=17.0, glucose_unit="mmol/L")


def test_hba1c_boundaries_use_fallback_without_kb(monkeypatch):
    monkeypatch.setitem(sys.modules, "dnx.kb.search", None)
    assert "DX-01" in ids(hba1c=6.5)
    assert "DX-02" in ids(hba1c=6.4) and "DX-01" not in ids(hba1c=6.4)
    assert not ids(hba1c=5.6)
    row = next(r for r in R.evaluate({"hba1c": 6.5}) if r["rule_id"] == "DX-01")
    assert row["fired_on"]["hba1c_threshold"]["threshold_source"] == "rules.json"


def test_threshold_read_from_kb(monkeypatch):
    kb = types.ModuleType("dnx.kb.search")
    kb.thresholds = lambda t: [{"test": t, "op": ">=", "value": 7.0, "unit": "%", "meaning": "đái tháo đường theo nguồn",
                                "evidence": "bằng chứng", "url": "u"}]
    monkeypatch.setitem(sys.modules, "dnx.kb.search", kb)
    assert "DX-01" not in ids(hba1c=6.8)
    row = next(r for r in R.evaluate({"hba1c": 7.1}) if r["rule_id"] == "DX-01")
    assert row["fired_on"]["hba1c_threshold"]["threshold_source"] == "bằng chứng"


def test_metformin_egfr():
    assert "MED-01" in ids(meds=["metformin"], egfr=25)
    assert "MED-02" in ids(meds=["metformin"], egfr=40)
    assert not {"MED-01", "MED-02"} & ids(meds=["metformin"], egfr=60)


def test_egfr_computed_from_creatinine():
    assert R.egfr_ckd_epi(1.0, 50, "male") == pytest.approx(91.7, abs=0.05)   # tính tay: 142 × 0,8812 × 0,7328
    assert "MED-01" in ids(meds=["metformin"], creatinine=4.0, age=60, sex="male")


def test_forward_chaining_two_steps():
    assert {"URG-04", "MED-07"} <= ids(glucose=50, meds=["insulin"])          # hạ đường huyết -> luật insulin
    assert {"RSK-03", "RSK-04", "RSK-05"} <= ids(risk_prob=0.7, hypertension=1)   # hai luật trung gian -> RSK-05
    assert "RSK-05" not in ids(risk_prob=0.2, hypertension=1)


def test_text_rules_accept_no_diacritics():
    assert "URG-07" in ids(text="toi bi dau nguc du doi")
    assert "POL-01" in ids(text="Kê đơn metformin cho tôi")
    assert R.flags(R.evaluate({"text": "liều metformin"}))["no_dose"] is True


def test_empty_and_none_facts_are_quiet():
    assert R.evaluate({}) == []
    assert R.evaluate({"glucose": None, "hba1c": None, "text": None}) == []


def test_sorted_by_severity():
    rows = R.evaluate({"glucose": 320, "bmi": 31})
    ranks = [R.SEVERITY_RANK[r["severity"]] for r in rows]
    assert ranks == sorted(ranks, reverse=True)
