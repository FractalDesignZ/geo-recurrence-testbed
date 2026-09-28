"""Stalk stabilize V3: science_open fail-closed; equal-weight HN+K16+overall select."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_stabilize_v3 import (  # noqa: E402
    CYCLE,
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    DEFAULT_SELECT_LONGHOP,
    HN_WEIGHT,
    K16_WEIGHT,
    LR_MAX,
    LR_MIN,
    OVERALL_WEIGHT,
    PREREG_HARD_NEG,
    PREREG_K16,
    SEED_PASS_GOAL,
    _cosine_lr,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_STABILIZE_V3"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert SEED_PASS_GOAL == 4
    assert DEFAULT_EPOCHS == 60
    assert tuple(DEFAULT_SEEDS) == (0, 1, 2, 3, 4)
    assert LR_MAX == 1.5e-3
    assert LR_MIN == 1.5e-4
    # Equal-weight joint; HN must not overweight (>0.5 forbidden after V2)
    assert abs(HN_WEIGHT - 1.0 / 3.0) < 1e-12
    assert abs(K16_WEIGHT - 1.0 / 3.0) < 1e-12
    assert abs(OVERALL_WEIGHT - 1.0 / 3.0) < 1e-12
    assert abs(HN_WEIGHT + K16_WEIGHT + OVERALL_WEIGHT - 1.0) < 1e-12
    assert HN_WEIGHT <= 0.5
    assert DEFAULT_SELECT_LONGHOP.name == "id_select_longhop.jsonl"


def test_cosine_lr_endpoints_v3():
    assert abs(_cosine_lr(1, 60, LR_MAX, LR_MIN) - LR_MAX) < 1e-12
    assert abs(_cosine_lr(60, 60, LR_MAX, LR_MIN) - LR_MIN) < 1e-12
    mid = _cosine_lr(30, 60, LR_MAX, LR_MIN)
    assert LR_MIN < mid < LR_MAX
