"""Stalk seed stability: science_open fail-closed; #14 freeze; seeds 0..9."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_seed_stability import (  # noqa: E402
    CYCLE,
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    HN_SELECT_WEIGHT,
    OVERALL_SELECT_WEIGHT,
    PREREG_HARD_NEG,
    PREREG_K16,
    RECONFIRM_SEEDS,
    SEED_PASS_GOAL,
    Z95,
    _ci95,
    _new_ckpt,
    _reconfirm_ckpt,
)
from reachability_gen.run_stalk_stabilize_multi_seed import (  # noqa: E402
    LR_MAX,
    LR_MIN,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_SEED_STABILITY"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert SEED_PASS_GOAL == 8  # of 10 (=80% same as 4/5)
    assert DEFAULT_EPOCHS == 60
    assert tuple(DEFAULT_SEEDS) == tuple(range(10))
    assert tuple(RECONFIRM_SEEDS) == (0, 1, 2, 3, 4)
    assert LR_MAX == 1.5e-3
    assert LR_MIN == 1.5e-4
    # Select locked to #14 — no HN overweight, no K16 weight in select
    assert HN_SELECT_WEIGHT == 0.5
    assert OVERALL_SELECT_WEIGHT == 0.5
    assert HN_SELECT_WEIGHT <= 0.5
    assert abs(HN_SELECT_WEIGHT + OVERALL_SELECT_WEIGHT - 1.0) < 1e-12


def test_ckpt_path_helpers():
    assert _reconfirm_ckpt(0).name == "fractal_core_stalk_stabilize_seed0_best.pt"
    assert _new_ckpt(5).name == "fractal_core_stalk_seed_stability_seed5_best.pt"


def test_ci95_normal_approx():
    assert Z95 == 1.96
    ci = _ci95(0.9, 0.1, 10)
    assert ci["halfwidth"] == pytest.approx(1.96 * 0.1 / (10**0.5))
    assert ci["lo"] == pytest.approx(0.9 - ci["halfwidth"])
    assert ci["hi"] == pytest.approx(0.9 + ci["halfwidth"])
