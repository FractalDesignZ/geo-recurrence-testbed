"""Stalk seed ensemble: science_open fail-closed; primary=prob_mean; #14/#18 freeze."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_seed_ensemble import (  # noqa: E402
    CYCLE,
    DEFAULT_SEEDS,
    LIFT_EPS,
    PRIMARY_AGG,
    PREREG_HARD_NEG,
    PREREG_K16,
    SECONDARY_AGGS,
    _aggregate_preds,
    _ckpt_for_seed,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_SEED_ENSEMBLE"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert LIFT_EPS == 0.01
    assert PRIMARY_AGG == "prob_mean"
    assert "logit_mean" in SECONDARY_AGGS
    assert "majority_vote" in SECONDARY_AGGS
    assert tuple(DEFAULT_SEEDS) == tuple(range(10))


def test_ckpt_path_helpers():
    assert _ckpt_for_seed(0).name == "fractal_core_stalk_stabilize_seed0_best.pt"
    assert _ckpt_for_seed(4).name == "fractal_core_stalk_stabilize_seed4_best.pt"
    assert _ckpt_for_seed(5).name == "fractal_core_stalk_seed_stability_seed5_best.pt"
    assert _ckpt_for_seed(9).name == "fractal_core_stalk_seed_stability_seed9_best.pt"


def test_aggregate_preds_shapes():
    # M=3, N=4, C=2
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.0], [0.0, 1.5], [0.0, 1.0], [1.0, 0.0]],
            [[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]],
        ]
    )
    pm = _aggregate_preds(logits, method="prob_mean")
    lm = _aggregate_preds(logits, method="logit_mean")
    mv = _aggregate_preds(logits, method="majority_vote")
    assert pm.shape == (4,)
    assert lm.shape == (4,)
    assert mv.shape == (4,)
    # first example: all members prefer class 0
    assert int(pm[0].item()) == 0
    assert int(mv[0].item()) == 0
    # second: all prefer class 1
    assert int(pm[1].item()) == 1
    assert int(mv[1].item()) == 1
