"""Stalk stabilize multi-seed: science_open fail-closed; prereg + harden helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_stabilize_multi_seed import (  # noqa: E402
    CYCLE,
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    LR_MAX,
    LR_MIN,
    PREREG_HARD_NEG,
    PREREG_K16,
    SEED_PASS_GOAL,
    _cosine_lr,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_STABILIZE_MULTI_SEED"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert SEED_PASS_GOAL == 4
    assert DEFAULT_EPOCHS == 60
    assert tuple(DEFAULT_SEEDS) == (0, 1, 2, 3, 4)
    assert LR_MAX == 1.5e-3
    assert LR_MIN == 1.5e-4


def test_cosine_lr_endpoints():
    assert abs(_cosine_lr(1, 60, LR_MAX, LR_MIN) - LR_MAX) < 1e-12
    assert abs(_cosine_lr(60, 60, LR_MAX, LR_MIN) - LR_MIN) < 1e-12
    mid = _cosine_lr(30, 60, LR_MAX, LR_MIN)
    assert LR_MIN < mid < LR_MAX
