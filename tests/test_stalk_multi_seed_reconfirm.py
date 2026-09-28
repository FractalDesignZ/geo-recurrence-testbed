"""Stalk multi-seed reconfirm: science_open fail-closed; prereg helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_multi_seed_reconfirm import (  # noqa: E402
    CYCLE,
    PREREG_HARD_NEG,
    PREREG_K16,
    _mean,
    _std,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_MULTI_SEED_RECONFIRM"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75


def test_mean_std_helpers():
    assert _mean([1.0, 2.0, 3.0]) == 2.0
    assert abs(_std([1.0, 2.0, 3.0]) - 1.0) < 1e-9
    assert _std([5.0]) == 0.0
