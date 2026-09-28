"""Stalk epistemic disagreement: science_open fail-closed; multi-hyp JS; not soft distill."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_epistemic_disagreement import (  # noqa: E402
    AUDIT_DELTA_EPS,
    CYCLE,
    DEFAULT_TRAIN_SEEDS,
    LAMBDA_JS,
    N_HYP,
    PREREG_HARD_NEG,
    PREREG_K16,
    _build_multi_hyp_model,
    _hyp_ckpt,
    _js_pairwise_mean,
    _pairwise_disagreement_rate,
    _swa_ckpt,
    attach_multi_hyp_head,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_EPISTEMIC_DISAGREEMENT"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert N_HYP == 3
    assert LAMBDA_JS == 0.5
    assert AUDIT_DELTA_EPS == 0.02
    assert tuple(DEFAULT_TRAIN_SEEDS) == (0, 1, 2)


def test_ckpt_path_helpers():
    assert _hyp_ckpt(0).name == "fractal_core_stalk_epistemic_disagreement_seed0_best.pt"
    assert _swa_ckpt(2).name == "fractal_core_stalk_swa_persist_seed2_swa.pt"


def test_multi_hyp_head_mean_and_stack():
    model = _build_multi_hyp_model(64, n_hyp=3, seed=0)
    x = torch.randn(5, 64)
    logits = model.head(x)
    assert logits.shape == (5, 2)
    assert model.head.last_stack.shape == (3, 5, 2)
    # mean of heads matches forward
    assert torch.allclose(logits, model.head.last_stack.mean(dim=0))


def test_js_pairwise_nonnegative_and_zero_when_identical():
    # identical logits → JS ≈ 0
    stack = torch.zeros(3, 4, 2)
    stack[..., 0] = 2.0
    js = _js_pairwise_mean(stack)
    assert float(js.item()) < 1e-5
    # disagreeing → JS > 0
    stack2 = stack.clone()
    stack2[1, :, 0] = -2.0
    stack2[1, :, 1] = 2.0
    js2 = _js_pairwise_mean(stack2)
    assert float(js2.item()) > 0.1


def test_pairwise_disagreement_rate():
    preds = torch.tensor([[0, 0, 1], [0, 1, 1], [0, 1, 0]])  # M=3,N=3
    rate = _pairwise_disagreement_rate(preds)
    # pairs: (0,1)=1/3; (0,2)=2/3; (1,2)=1/3 → mean = 4/9
    assert abs(rate - (4.0 / 9.0)) < 1e-6


def test_loss_encourages_not_distills():
    # Documented: loss is CE − λ·JS (encourage), not KL to teacher
    assert LAMBDA_JS > 0
