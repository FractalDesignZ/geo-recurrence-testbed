"""FO/cert regression gate: locked thresholds; artifact hygiene; no train."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from reachability_gen.run_stalk_fo_cert_regression import (
    CERT_VERDICT_LOCKED,
    CITE30_FO_EXACT,
    CITE30_HN_MAX,
    CITE30_K16_MIN,
    CITE30_VERDICT_NEEDLE,
    CYCLE,
    FO_CATCH_EXACT,
    FO_TOTAL_LOCKED,
    FOCUS_T,
    HARD_UNANIMOUS_MIN_COUNT,
    HARD_UNANIMOUS_MIN_RATE,
    MATCHED_COLLATERAL_ABS_MAX,
    ORACLE_FO_KILLED_MAX,
    P_CORRECT_CLEAN_MIN,
    REM22_CATCH_EXACT,
    VERDICTS,
    check_gate_a_hard_unanimous_oracle,
    check_gate_b_cert_refuse,
    check_gate_c_smoke_cite30,
    decide_verdict,
    run_regression,
)

ROOT = Path(__file__).resolve().parents[1]
CITE30 = ROOT / "artifacts" / "stalk_hop_ood_hn.json"
CITE31 = ROOT / "artifacts" / "stalk_hn_fail_open_autopsy.json"
CITE35 = ROOT / "artifacts" / "stalk_reach_certificates.json"
CITE38 = ROOT / "artifacts" / "stalk_energy_selector.json"


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_FO_CERT_REGRESSION"
    assert FOCUS_T == 16
    assert FO_TOTAL_LOCKED == 45
    assert HARD_UNANIMOUS_MIN_COUNT == 30
    assert HARD_UNANIMOUS_MIN_RATE == 0.70
    assert ORACLE_FO_KILLED_MAX == 0
    assert FO_CATCH_EXACT == 45
    assert REM22_CATCH_EXACT == 22
    assert P_CORRECT_CLEAN_MIN == 0.99
    assert MATCHED_COLLATERAL_ABS_MAX == 0.01
    assert CERT_VERDICT_LOCKED == "CERT_FO_CATCH"
    assert CITE30_FO_EXACT == 45
    assert CITE30_HN_MAX == 0.10
    assert CITE30_K16_MIN == 0.95
    assert CITE30_VERDICT_NEEDLE == "HN_SHATTER_CONFIRMED"
    assert set(VERDICTS) == {"PASS_REGRESSION", "FAIL"}


def test_no_train_generation_flags():
    import reachability_gen.run_stalk_fo_cert_regression as m

    doc = m.__doc__ or ""
    assert "no train" in doc.lower()
    assert "science_open=false" in doc.lower()
    assert m.FOCUS_T == 16


def test_decide_verdict_pass_and_fail():
    ok = {"pass": True, "reasons": []}
    bad = {"pass": False, "reasons": ["x"]}
    d = decide_verdict(ok, ok, ok, require_smoke=True)
    assert d["verdict"] == "PASS_REGRESSION"
    assert d["science_open"] is False
    d2 = decide_verdict(ok, bad, ok, require_smoke=True)
    assert d2["verdict"] == "FAIL"
    d3 = decide_verdict(ok, ok, None, require_smoke=False)
    assert d3["verdict"] == "PASS_REGRESSION"
    d4 = decide_verdict(ok, ok, None, require_smoke=True)
    assert d4["verdict"] == "FAIL"


@pytest.mark.skipif(not CITE31.is_file() or not CITE38.is_file(), reason="sealed cites missing")
def test_gate_a_on_sealed_artifacts():
    cite31 = json.loads(CITE31.read_text())
    cite38 = json.loads(CITE38.read_text())
    out = check_gate_a_hard_unanimous_oracle(cite31, cite38)
    assert out["pass"] is True
    assert out["summary"]["HARD_UNANIMOUS"] >= HARD_UNANIMOUS_MIN_COUNT
    assert out["summary"]["oracle_fo_killed"] == 0


@pytest.mark.skipif(not CITE35.is_file(), reason="sealed #35 missing")
def test_gate_b_on_sealed_artifacts():
    cite35 = json.loads(CITE35.read_text())
    out = check_gate_b_cert_refuse(cite35)
    assert out["pass"] is True
    assert out["summary"]["fo_killed"] == 45
    assert out["summary"]["p_correct_given_clean"] >= 0.99


@pytest.mark.skipif(not CITE30.is_file(), reason="sealed #30 missing")
def test_gate_c_on_sealed_artifacts():
    cite30 = json.loads(CITE30.read_text())
    out = check_gate_c_smoke_cite30(cite30)
    assert out["pass"] is True
    assert out["summary"]["FAIL_OPEN"] == 45


@pytest.mark.skipif(
    not all(p.is_file() for p in (CITE30, CITE31, CITE35, CITE38)),
    reason="sealed cites missing",
)
def test_full_regression_pass(tmp_path: Path):
    out = tmp_path / "stalk_fo_cert_regression.json"
    payload = run_regression(
        cite30_path=CITE30,
        cite31_path=CITE31,
        cite35_path=CITE35,
        cite38_path=CITE38,
        out_path=out,
        require_smoke=True,
        write=True,
    )
    assert payload["verdict"] == "PASS_REGRESSION"
    assert payload["science_open"] is False
    assert out.is_file()
    disk = json.loads(out.read_text())
    assert disk["verdict"] == "PASS_REGRESSION"


@pytest.mark.skipif(not CITE35.is_file(), reason="sealed #35 missing")
def test_gate_b_fails_on_regressed_fo_catch():
    cite35 = json.loads(CITE35.read_text())
    bad = copy.deepcopy(cite35)
    bad["decision"]["fo_killed"] = 20
    bad["decision"]["verdict"] = "CERT_PARTIAL"
    out = check_gate_b_cert_refuse(bad)
    assert out["pass"] is False
    assert any("fo_killed" in r for r in out["reasons"])


@pytest.mark.skipif(not CITE38.is_file() or not CITE31.is_file(), reason="sealed cites missing")
def test_gate_a_fails_if_oracle_kills_fo():
    cite31 = json.loads(CITE31.read_text())
    cite38 = json.loads(CITE38.read_text())
    bad = copy.deepcopy(cite38)
    bad["selector_oracle_gap"]["oracle_fo_killed"] = 5
    out = check_gate_a_hard_unanimous_oracle(cite31, bad)
    assert out["pass"] is False
    assert any("oracle_fo_killed" in r for r in out["reasons"])
