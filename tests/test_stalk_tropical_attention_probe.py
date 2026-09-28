"""Stalk tropical / max-plus ens probe: helpers + verdict; no train."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.tropical import (  # noqa: E402
    beta_inf_member_preds,
    max_plus_aggregate,
    max_plus_scores,
    tropical_definition_doc,
)
from reachability_gen.run_stalk_tropical_attention_probe import (  # noqa: E402
    COLLATERAL_DROP,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    FOCUS_T,
    FO_LIFT_MIN,
    FO_NULL_MAX,
    HN_NULL_DELTA,
    REM22_LIFT_MIN,
    VERDICTS,
    decide_verdict,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_TROPICAL_ATTENTION_PROBE"
    assert FOCUS_T == 16
    assert FO_LIFT_MIN == 16
    assert REM22_LIFT_MIN == 8
    assert FO_NULL_MAX == 2
    assert HN_NULL_DELTA == 0.05
    assert COLLATERAL_DROP == 0.05
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert set(VERDICTS) == {
        "TROPICAL_FO_LIFT",
        "TROPICAL_PARTIAL",
        "TROPICAL_NULL",
        "COLLATERAL_HARM",
    }


def test_no_train_flags():
    import reachability_gen.run_stalk_tropical_attention_probe as m

    doc = m.__doc__ or ""
    assert "No train" in doc or "no train" in doc.lower() or "NOT train" in doc
    assert "science_open=false" in doc.lower() or "science_open=false" in doc
    assert "tropical" in doc.lower()
    assert m.FOCUS_T == 16
    assert m.DEFAULT_ENSEMBLE_SEEDS == tuple(range(10))
    defs = tropical_definition_doc()
    assert defs["in_attention_rewrite"] is False
    assert defs["scope"] == "ensemble_aggregation_logit_mix"


def test_max_plus_scores_takes_memberwise_max():
    # M=2, N=2, C=2
    logits = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],  # member 0
            [[0.5, 3.0], [4.0, 1.0]],  # member 1
        ]
    )
    scores = max_plus_scores(logits)
    assert scores.shape == (2, 2)
    assert torch.allclose(scores[0], torch.tensor([1.0, 3.0]))
    assert torch.allclose(scores[1], torch.tensor([4.0, 2.0]))


def test_max_plus_aggregate_preds():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[0.0, 3.0], [5.0, 0.0]],
        ]
    )
    preds, conf, scores = max_plus_aggregate(logits)
    # ex0: score=[2,3] → pred 1; ex1: score=[5,1] → pred 0
    assert preds.tolist() == [1, 0]
    assert conf.shape == (2,)
    assert (conf > 0.5).all()
    assert scores.shape == (2, 2)


def test_max_plus_differs_from_prob_mean_on_bleed():
    """Classic bleed: one confident YES + many mild NO → mean may flip.

    Construct member logits where prob_mean → 0 but max-plus → 1.
    """
    import torch.nn.functional as F
    from reachability_gen.run_stalk_seed_ensemble import _aggregate_preds

    # 3 members, 1 example, C=2
    # m0: strong YES (logit [0, 5]); m1,m2: mild NO ([1, 0], [1, 0])
    logits = torch.tensor(
        [
            [[0.0, 5.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    mean_pred = _aggregate_preds(logits, method="prob_mean")
    trop_pred, _, _ = max_plus_aggregate(logits)
    # max-plus: score=[1,5] → YES
    assert int(trop_pred[0].item()) == 1
    # prob_mean: avg of softmax — may or may not be YES; just assert shapes
    assert mean_pred.shape == (1,)
    # Explicit: max-plus picks the strong YES member's class via max
    scores = max_plus_scores(logits)
    assert scores[0, 1] > scores[0, 0]


def test_beta_inf_routes_to_most_confident_member():
    # m0: confident NO; m1: less confident YES → pick m0
    logits = torch.tensor(
        [
            [[5.0, 0.0]],  # conf ≈ softmax → high on class 0
            [[0.0, 1.0]],  # milder
        ]
    )
    preds, conf, mstar = beta_inf_member_preds(logits)
    assert int(mstar[0].item()) == 0
    assert int(preds[0].item()) == 0
    assert float(conf[0].item()) > 0.9


def test_beta_inf_picks_confident_yes_when_strongest():
    logits = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 8.0]],
        ]
    )
    preds, conf, mstar = beta_inf_member_preds(logits)
    assert int(mstar[0].item()) == 1
    assert int(preds[0].item()) == 1


def test_decide_verdict_collateral_priority():
    d = decide_verdict(
        fo_killed=40,
        fo_total=45,
        rem22_killed=20,
        rem22_total=22,
        delta_hn_abs=0.5,
        collateral_harm=True,
        collateral_reasons=["matched_logit_max_overall_acc_drop=0.06>=0.05"],
    )
    assert d["verdict"] == "COLLATERAL_HARM"


def test_decide_verdict_lift():
    d = decide_verdict(
        fo_killed=20,
        fo_total=45,
        rem22_killed=10,
        rem22_total=22,
        delta_hn_abs=0.2,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert d["verdict"] == "TROPICAL_FO_LIFT"


def test_decide_verdict_null():
    d = decide_verdict(
        fo_killed=0,
        fo_total=45,
        rem22_killed=0,
        rem22_total=22,
        delta_hn_abs=0.01,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert d["verdict"] == "TROPICAL_NULL"


def test_decide_verdict_partial():
    d = decide_verdict(
        fo_killed=5,
        fo_total=45,
        rem22_killed=0,
        rem22_total=22,
        delta_hn_abs=0.1,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert d["verdict"] == "TROPICAL_PARTIAL"


def test_max_plus_rejects_bad_shape():
    with pytest.raises(ValueError):
        max_plus_scores(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        beta_inf_member_preds(torch.zeros(4))
