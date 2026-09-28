"""Stalk objective V1: science_open fail-closed; #14 select; HN/longhop train curriculum."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_objective_v1 import (  # noqa: E402
    CYCLE,
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    HN_SAMPLE_WEIGHT,
    HN_SELECT_WEIGHT,
    LONGHOP_MIN_HOP,
    LONGHOP_SAMPLE_WEIGHT,
    LR_MAX,
    LR_MIN,
    MID_SAMPLE_WEIGHT,
    OVERALL_SELECT_WEIGHT,
    PREREG_HARD_NEG,
    PREREG_K16,
    SEED_PASS_GOAL,
    _cosine_lr,
    _example_sample_weight,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_OBJECTIVE_V1"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert SEED_PASS_GOAL == 4
    assert DEFAULT_EPOCHS == 60
    assert tuple(DEFAULT_SEEDS) == (0, 1, 2, 3, 4)
    assert LR_MAX == 1.5e-3
    assert LR_MIN == 1.5e-4
    # Select locked to #14 — no HN overweight, no K16 weight in select
    assert HN_SELECT_WEIGHT == 0.5
    assert OVERALL_SELECT_WEIGHT == 0.5
    assert HN_SELECT_WEIGHT <= 0.5
    assert abs(HN_SELECT_WEIGHT + OVERALL_SELECT_WEIGHT - 1.0) < 1e-12
    # Train curriculum
    assert HN_SAMPLE_WEIGHT == 2.0
    assert LONGHOP_SAMPLE_WEIGHT == 2.0
    assert MID_SAMPLE_WEIGHT == 1.0
    assert LONGHOP_MIN_HOP == 5


def test_example_sample_weights():
    assert _example_sample_weight({"hop_distance": -1}) == HN_SAMPLE_WEIGHT
    assert _example_sample_weight({"hop_distance": 5}) == LONGHOP_SAMPLE_WEIGHT
    assert _example_sample_weight({"hop_distance": 6}) == LONGHOP_SAMPLE_WEIGHT
    assert _example_sample_weight({"hop_distance": 2}) == MID_SAMPLE_WEIGHT
    assert _example_sample_weight({"hop_distance": 4}) == MID_SAMPLE_WEIGHT


def test_cosine_lr_endpoints_objective_v1():
    assert abs(_cosine_lr(1, 60, LR_MAX, LR_MIN) - LR_MAX) < 1e-12
    assert abs(_cosine_lr(60, 60, LR_MAX, LR_MIN) - LR_MIN) < 1e-12
    mid = _cosine_lr(30, 60, LR_MAX, LR_MIN)
    assert LR_MIN < mid < LR_MAX
