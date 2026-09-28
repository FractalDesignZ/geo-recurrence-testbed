"""Stalk competent dissonance: science_open fail-closed; CD formula; no train."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_competent_dissonance import (  # noqa: E402
    ACC_FLOOR,
    CYCLE,
    DEFAULT_BAG_SEEDS,
    DEFAULT_ENSEMBLE_SEEDS,
    DIS_SET_ACC_MIN,
    D_HARD_MIN,
    D_SAT,
    EASY_HOP,
    SHATTER_DROP,
    _competent_dissonance,
    _verdict_arm,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_COMPETENT_DISSONANCE"
    assert D_SAT == 0.25
    assert ACC_FLOOR == 0.85
    assert D_HARD_MIN == 0.05
    assert DIS_SET_ACC_MIN == 0.65
    assert EASY_HOP == 8
    assert SHATTER_DROP == 0.10
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert tuple(DEFAULT_BAG_SEEDS) == (0, 1, 2, 3, 4)


def test_cd_formula_caps_chaos_disagree():
    # high μ, moderate D_hard
    r = _competent_dissonance(0.90, 0.10)
    assert abs(r["CD"] - 0.90 * (0.10 / 0.25)) < 1e-9
    assert abs(r["CD_raw"] - 0.90 * 0.10) < 1e-9
    # excess D beyond D_SAT does not inflate CD
    r2 = _competent_dissonance(0.90, 0.50)
    r3 = _competent_dissonance(0.90, 0.25)
    assert abs(r2["CD"] - r3["CD"]) < 1e-9
    assert r2["CD_raw"] > r3["CD_raw"]
    # echo → CD → 0
    r0 = _competent_dissonance(0.95, 0.0)
    assert r0["CD"] == 0.0


def test_verdict_competent_vs_echo_vs_chaos():
    # COMPETENT: high μ, enough D_hard, strong disagree-set acc
    assert (
        _verdict_arm(
            mu_acc=0.90,
            d_hard=0.08,
            disagree_set_member_acc=0.80,
            rides_on_disagreement=True,
        )
        == "COMPETENT"
    )
    # COMPETENT via rides even if D_hard low (localized lift; not echo)
    assert (
        _verdict_arm(
            mu_acc=0.90,
            d_hard=0.02,
            disagree_set_member_acc=0.80,
            rides_on_disagreement=True,
        )
        == "COMPETENT"
    )
    # ECHO_RISK: competent, no hard disagree, no rides
    assert (
        _verdict_arm(
            mu_acc=0.90,
            d_hard=0.01,
            disagree_set_member_acc=0.90,
            rides_on_disagreement=False,
        )
        == "ECHO_RISK"
    )
    # CHAOS: weak members
    assert (
        _verdict_arm(
            mu_acc=0.66,
            d_hard=0.40,
            disagree_set_member_acc=0.40,
            rides_on_disagreement=True,
        )
        == "CHAOS"
    )


def test_no_train_constants_documented():
    # Import module docstring / architecture flags via run_cycle defaults
    import reachability_gen.run_stalk_competent_dissonance as m

    assert m.FOCUS_T == 16
    assert "eval-only" in m.__doc__.lower() or "No training" in m.__doc__
