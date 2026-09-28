"""Stalk stabilize V2: science_open fail-closed; gated HN-weight selection knobs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_stabilize_v2 import (  # noqa: E402
    CYCLE,
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    HN_WEIGHT,
    LR_MAX,
    LR_MIN,
    OVERALL_GATE,
    OVERALL_WEIGHT,
    PREREG_HARD_NEG,
    PREREG_K16,
    SEED_PASS_GOAL,
    _cosine_lr,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_STABILIZE_V2"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert SEED_PASS_GOAL == 4
    assert DEFAULT_EPOCHS == 90
    assert tuple(DEFAULT_SEEDS) == (0, 1, 2, 3, 4)
    assert LR_MAX == 1.5e-3
    assert LR_MIN == 1.0e-4
    assert HN_WEIGHT == 0.7
    assert OVERALL_WEIGHT == 0.3
    assert abs(HN_WEIGHT + OVERALL_WEIGHT - 1.0) < 1e-12
    assert OVERALL_GATE == 0.85


def test_cosine_lr_endpoints_v2():
    assert abs(_cosine_lr(1, 90, LR_MAX, LR_MIN) - LR_MAX) < 1e-12
    assert abs(_cosine_lr(90, 90, LR_MAX, LR_MIN) - LR_MIN) < 1e-12
    mid = _cosine_lr(45, 90, LR_MAX, LR_MIN)
    assert LR_MIN < mid < LR_MAX
