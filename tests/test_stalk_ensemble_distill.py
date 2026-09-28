"""Stalk ensemble distill: science_open fail-closed; α/τ; #14 select freeze."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_ensemble_distill import (  # noqa: E402
    CYCLE,
    DEFAULT_STUDENT_SEEDS,
    DEFAULT_TEACHER_SEEDS,
    DISTILL_ALPHA,
    DISTILL_TAU,
    PREREG_HARD_NEG,
    PREREG_K16,
    _student_ckpt,
)
from reachability_gen.run_stalk_seed_ensemble import _ckpt_for_seed  # noqa: E402


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_ENSEMBLE_DISTILL"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert DISTILL_ALPHA == 0.5
    assert DISTILL_TAU == 2.0
    assert tuple(DEFAULT_TEACHER_SEEDS) == tuple(range(10))
    assert tuple(DEFAULT_STUDENT_SEEDS) == (0, 1, 2)


def test_ckpt_path_helpers():
    assert _student_ckpt(0).name == "fractal_core_stalk_ensemble_distill_seed0_best.pt"
    assert _ckpt_for_seed(0).name == "fractal_core_stalk_stabilize_seed0_best.pt"
    assert _ckpt_for_seed(5).name == "fractal_core_stalk_seed_stability_seed5_best.pt"
