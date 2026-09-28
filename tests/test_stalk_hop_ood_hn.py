"""Stalk hop-OOD HN: science_open fail-closed; FAIL_OPEN/CLOSED; gate; no train."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_hop_ood_hn import (  # noqa: E402
    CONF_THRESH,
    CITE_28_HOPS,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    D_CLOSED_MIN,
    EPI_CLOSED_MIN,
    FOCUS_T,
    GATE_CONF_MIN,
    GATE_COVERAGE_MIN,
    GATE_D_MIN,
    HN_RECOVER_MIN,
    HN_SHATTER_MAX,
    PRIMARY_AGG,
    _hn_verdict,
)
from reachability_gen.run_stalk_seed_ensemble import PRIMARY_AGG as SE_PRIMARY  # noqa: E402


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_HOP_OOD_HN"
    assert FOCUS_T == 16
    assert CONF_THRESH == 0.80
    assert D_CLOSED_MIN == 0.10
    assert EPI_CLOSED_MIN == 0.15
    assert GATE_D_MIN == 0.10
    assert GATE_CONF_MIN == 0.80
    assert HN_SHATTER_MAX == 0.50
    assert HN_RECOVER_MIN == 0.90
    assert GATE_COVERAGE_MIN == 0.25
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert PRIMARY_AGG == "prob_mean" or SE_PRIMARY == "prob_mean"
    assert abs(CITE_28_HOPS["ens_hard_neg_acc"] - 0.06666666666666667) < 1e-9
    assert abs(CITE_28_HOPS["CD"] - 0.553125) < 1e-9


def test_no_train_flags():
    import reachability_gen.run_stalk_hop_ood_hn as m

    assert m.DEFAULT_ENSEMBLE_SEEDS == tuple(range(10))
    doc = m.__doc__ or ""
    assert "No train" in doc or "no train" in doc.lower()
    assert "science_open=false" in doc.lower() or "science_open=false" in doc
    # T locked
    assert m.FOCUS_T == 16


def test_hn_verdict_shatter():
    baseline = {
        "ensemble": {"hard_neg_acc": 0.067},
        "fail_mode": {"verdict": "FAIL_CLOSED_DOMINANT"},
        "gate_overlay": {
            "coverage": 0.1,
            "ens_prob_mean_accepted": {"hard_neg_acc": 0.2},
            "fail_mode_accepted": {"verdict": "FAIL_CLOSED_DOMINANT"},
            "hn_slice": {"hn_coverage": 0.05},
        },
    }
    vote = {
        "ensemble": {"hard_neg_acc": 0.10},
        "fail_mode": {"verdict": "FAIL_CLOSED_DOMINANT"},
    }
    out = _hn_verdict(baseline, vote)
    assert out["label"] == "HN_SHATTER_CONFIRMED"


def test_hn_verdict_partial_via_gate():
    baseline = {
        "ensemble": {"hard_neg_acc": 0.067},
        "fail_mode": {"verdict": "FAIL_CLOSED_DOMINANT"},
        "gate_overlay": {
            "coverage": 0.3,
            "ens_prob_mean_accepted": {"hard_neg_acc": 0.70},
            "fail_mode_accepted": {"verdict": "FAIL_CLOSED_DOMINANT"},
            "hn_slice": {"hn_coverage": 0.2},
        },
    }
    vote = {
        "ensemble": {"hard_neg_acc": 0.08},
        "fail_mode": {"verdict": "FAIL_CLOSED_DOMINANT"},
    }
    out = _hn_verdict(baseline, vote)
    assert out["label"] == "HN_PARTIAL_RECOVER"


def test_hn_verdict_recovered_does_not_imply_open():
    baseline = {
        "ensemble": {"hard_neg_acc": 0.067},
        "fail_mode": {"verdict": "FAIL_CLOSED_DOMINANT"},
        "gate_overlay": {
            "coverage": 0.30,
            "ens_prob_mean_accepted": {"hard_neg_acc": 0.95},
            "fail_mode_accepted": {"verdict": "FAIL_CLOSED_DOMINANT"},
            "hn_slice": {"hn_coverage": 0.30},
        },
    }
    vote = {
        "ensemble": {"hard_neg_acc": 0.10},
        "fail_mode": {"verdict": "FAIL_CLOSED_DOMINANT"},
    }
    out = _hn_verdict(baseline, vote)
    assert out["label"] == "HN_RECOVERED"
    assert "does not widen" in out["note"].lower() or "§22" in out["note"]
