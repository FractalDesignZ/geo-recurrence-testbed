"""Stalk energy selector: helpers + verdict; no train."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.energy_selector import (  # noqa: E402
    ALPHA_SOUND,
    BETA_CONE,
    GAMMA_CONF,
    compute_energy_stack,
    e_cone,
    e_conf,
    e_disagree,
    e_free_scalar,
    e_sound,
    energy_argmin_preds,
    energy_definition_doc,
    energy_weighted_preds,
    local_degree_features,
    member_preds_and_conf,
    oracle_member_preds,
    pearson_corr,
)
from reachability_gen.run_stalk_energy_selector import (  # noqa: E402
    COLLATERAL_DROP,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    FOCUS_T,
    FO_BEATS_MIN,
    FO_NULL_MAX,
    GAP_CLOSE_MIN,
    REM22_BEATS_MIN,
    VERDICTS,
    decide_verdict,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_ENERGY_SELECTOR"
    assert FOCUS_T == 16
    assert FO_BEATS_MIN == 8
    assert REM22_BEATS_MIN == 4
    assert FO_NULL_MAX == 2
    assert GAP_CLOSE_MIN == 0.02
    assert COLLATERAL_DROP == 0.05
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert set(VERDICTS) == {
        "ENERGY_BEATS_MEAN",
        "ENERGY_PARTIAL",
        "ENERGY_NULL",
        "ENERGY_IS_CERT",
        "COLLATERAL_HARM",
    }
    assert ALPHA_SOUND == 10.0
    assert BETA_CONE == 1.0
    assert GAMMA_CONF == 0.1


def test_no_train_flags():
    import reachability_gen.run_stalk_energy_selector as m

    doc = m.__doc__ or ""
    assert "eval-only" in doc.lower() or "NOT" in doc
    assert "science_open=false" in doc.lower()
    assert "orientation" in doc.lower()
    assert m.FOCUS_T == 16
    defs = energy_definition_doc()
    assert defs["primary"]["uses_bfs_checker"] is False
    assert defs["primary"]["uses_orientation"] is False
    assert defs["secondary"]["uses_bfs_checker"] is True
    assert defs["secondary"]["label"] == "cert-energy"


def test_e_sound_outdeg0_yes():
    assert e_sound(1, out_s=0, in_t=2, s=0, t=1) == 1.0
    assert e_sound(0, out_s=0, in_t=2, s=0, t=1) == 0.0
    assert e_sound(1, out_s=0, in_t=0, s=0, t=1) == 2.0
    assert e_sound(1, out_s=0, in_t=0, s=3, t=3) == 0.0  # s==t exempt


def test_e_cone_sparse_yes():
    assert e_cone(1, out_s=0, in_t=0) == 1.0
    assert e_cone(0, out_s=0, in_t=0) == 0.0
    assert abs(e_cone(1, out_s=1, in_t=1) - 1.0 / 3.0) < 1e-12


def test_e_disagree_cert():
    assert e_disagree(1, reach=False) == 1.0
    assert e_disagree(0, reach=True) == 1.0
    assert e_disagree(1, reach=True) == 0.0
    assert e_disagree(0, reach=False) == 0.0


def test_e_free_formula():
    # YES on empty local + low conf
    e = e_free_scalar(1, 0.5, out_s=0, in_t=0, s=0, t=1)
    expected = 10.0 * 2.0 + 1.0 * 1.0 + 0.1 * e_conf(0.5)
    assert abs(e - expected) < 1e-9


def test_local_degree_features_encoding():
    row = {
        "encoding": "N 3 EDGES 0,1 QUERY 0 2",
        "n": 3,
        "s": 0,
        "t": 2,
        "y": 0,
    }
    loc = local_degree_features(row)
    assert loc["out_s"] == 1
    assert loc["in_t"] == 0
    assert loc["s"] == 0
    assert loc["t"] == 2


def test_energy_argmin_picks_lowest_energy_member():
    # M=2, N=1, C=2
    # m0: YES high conf; m1: NO high conf — on out_s=0,in_t=0,s≠t → m0 high E_sound
    logits = torch.tensor(
        [
            [[0.0, 5.0]],  # YES
            [[5.0, 0.0]],  # NO
        ]
    )
    rows = [
        {
            "encoding": "N 2 EDGES QUERY 0 1",
            "n": 2,
            "s": 0,
            "t": 1,
            "y": 0,
        }
    ]
    E = compute_energy_stack(logits, rows, kind="free")
    assert E[0, 0] > E[1, 0]  # YES on empty out pays sound cost
    preds, conf, mstar = energy_argmin_preds(logits, E)
    assert int(mstar[0].item()) == 1
    assert int(preds[0].item()) == 0


def test_oracle_picks_correct_member():
    logits = torch.tensor(
        [
            [[5.0, 0.0]],  # NO wrong if y=1
            [[0.0, 5.0]],  # YES correct
        ]
    )
    labels = torch.tensor([1])
    preds, conf, mstar = oracle_member_preds(logits, labels)
    assert int(mstar[0].item()) == 1
    assert int(preds[0].item()) == 1


def test_oracle_falls_back_to_member0_when_all_wrong():
    logits = torch.tensor(
        [
            [[5.0, 0.0]],
            [[4.0, 0.0]],
        ]
    )
    labels = torch.tensor([1])
    preds, conf, mstar = oracle_member_preds(logits, labels)
    assert int(mstar[0].item()) == 0
    assert int(preds[0].item()) == 0


def test_energy_weighted_softmin():
    logits = torch.tensor(
        [
            [[0.0, 3.0]],
            [[3.0, 0.0]],
        ]
    )
    # Force E so m1 much lower
    E = torch.tensor([[10.0], [0.1]], dtype=torch.float64)
    preds, conf = energy_weighted_preds(logits, E, tau=1.0)
    assert preds.shape == (1,)
    assert conf.shape == (1,)
    # weight on m1 dominates → NO
    assert int(preds[0].item()) == 0


def test_pearson_corr_identity():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert abs(pearson_corr(xs, xs) - 1.0) < 1e-9
    assert abs(pearson_corr(xs, [-x for x in xs]) + 1.0) < 1e-9


def test_decide_verdict_collateral_priority():
    d = decide_verdict(
        fo_killed_free=40,
        fo_total=45,
        rem22_killed_free=20,
        rem22_total=22,
        gap_close_overall=0.5,
        delta_hn_abs=0.5,
        delta_hn=0.5,
        energy_is_just_conf=False,
        fo_killed_cert=45,
        rem22_killed_cert=22,
        collateral_harm=True,
        collateral_reasons=["matched_E_free_K16_drop=0.06>=0.05"],
    )
    assert d["verdict"] == "COLLATERAL_HARM"


def test_decide_verdict_beats_mean():
    d = decide_verdict(
        fo_killed_free=10,
        fo_total=45,
        rem22_killed_free=5,
        rem22_total=22,
        gap_close_overall=0.05,
        delta_hn_abs=0.2,
        delta_hn=0.2,
        energy_is_just_conf=False,
        fo_killed_cert=45,
        rem22_killed_cert=22,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert d["verdict"] == "ENERGY_BEATS_MEAN"


def test_decide_verdict_is_cert():
    d = decide_verdict(
        fo_killed_free=0,
        fo_total=45,
        rem22_killed_free=0,
        rem22_total=22,
        gap_close_overall=0.0,
        delta_hn_abs=0.01,
        delta_hn=0.01,
        energy_is_just_conf=False,
        fo_killed_cert=45,
        rem22_killed_cert=22,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert d["verdict"] == "ENERGY_IS_CERT"


def test_decide_verdict_null():
    d = decide_verdict(
        fo_killed_free=0,
        fo_total=45,
        rem22_killed_free=0,
        rem22_total=22,
        gap_close_overall=0.0,
        delta_hn_abs=0.01,
        delta_hn=0.01,
        energy_is_just_conf=False,
        fo_killed_cert=0,
        rem22_killed_cert=0,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert d["verdict"] == "ENERGY_NULL"


def test_decide_verdict_partial():
    d = decide_verdict(
        fo_killed_free=3,
        fo_total=45,
        rem22_killed_free=0,
        rem22_total=22,
        gap_close_overall=0.01,
        delta_hn_abs=0.08,
        delta_hn=0.08,
        energy_is_just_conf=False,
        fo_killed_cert=5,
        rem22_killed_cert=2,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert d["verdict"] == "ENERGY_PARTIAL"


def test_member_preds_rejects_bad_shape():
    with pytest.raises(ValueError):
        member_preds_and_conf(torch.zeros(2, 3))
