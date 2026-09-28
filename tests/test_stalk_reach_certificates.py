"""Stalk reach certificates: post-hoc path-witness / checker-BFS; no train."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.reach_certificates import (  # noqa: E402
    ReachCertificate,
    apply_cert_policy_preds,
    certify_prediction,
    find_path_witness,
    validate_path_witness,
)
from reachability_gen.run_stalk_reach_certificates import (  # noqa: E402
    COLLATERAL_DROP,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    FOCUS_T,
    FO_CATCH_MIN,
    REM22_CATCH_MIN,
    VERDICTS,
    decide_verdict,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_REACH_CERTIFICATES"
    assert FOCUS_T == 16
    assert FO_CATCH_MIN == 40
    assert REM22_CATCH_MIN == 16
    assert COLLATERAL_DROP == 0.05
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert set(VERDICTS) == {
        "CERT_FO_CATCH",
        "CERT_PARTIAL",
        "CERT_NOISE",
        "COLLATERAL_HARM",
    }


def test_no_train_flags():
    import reachability_gen.run_stalk_reach_certificates as m

    doc = m.__doc__ or ""
    assert "No train" in doc or "no train" in doc.lower()
    assert "science_open=false" in doc.lower() or "science_open=false" in doc
    assert "tropical" in doc.lower()
    assert m.FOCUS_T == 16
    assert m.DEFAULT_ENSEMBLE_SEEDS == tuple(range(10))


def test_validate_path_witness_accepts_valid():
    edges = [(0, 1), (1, 2), (2, 3)]
    assert validate_path_witness(edges, [0, 1, 2, 3], s=0, t=3) is True
    assert validate_path_witness(edges, [0], s=0, t=0) is True


def test_validate_path_witness_rejects_missing_edge_or_bad_ends():
    edges = [(0, 1), (1, 2)]
    assert validate_path_witness(edges, [0, 2], s=0, t=2) is False  # no 0→2
    assert validate_path_witness(edges, [0, 1, 2], s=0, t=1) is False  # bad end
    assert validate_path_witness(edges, [], s=0, t=1) is False


def test_find_path_witness_shortest():
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    w = find_path_witness(4, edges, 0, 3)
    assert w is not None
    assert w[0] == 0 and w[-1] == 3
    assert validate_path_witness(edges, w, s=0, t=3)
    # unreachable
    assert find_path_witness(4, edges, 3, 0) is None
    assert find_path_witness(3, [(0, 1)], 0, 0) == (0,)


def test_certify_yes_clean_when_path_exists():
    edges = [(0, 1), (1, 2)]
    c = certify_prediction(3, edges, 0, 2, pred=1)
    assert c.clean is True
    assert c.kind in ("path_witness", "self_reach")
    assert c.witness is not None
    assert validate_path_witness(edges, c.witness, s=0, t=2)


def test_certify_yes_dirty_when_unreachable():
    edges = [(1, 2)]
    c = certify_prediction(3, edges, 0, 2, pred=1)
    assert c.clean is False
    assert c.kind == "dirty_yes"
    assert c.reachable_by_checker is False


def test_certify_no_clean_when_unreachable():
    edges = [(1, 2)]
    c = certify_prediction(3, edges, 0, 2, pred=0)
    assert c.clean is True
    assert c.kind == "unreachable_checker"


def test_certify_no_dirty_when_reachable():
    edges = [(0, 1), (1, 2)]
    c = certify_prediction(3, edges, 0, 2, pred=0)
    assert c.clean is False
    assert c.kind == "dirty_no"
    assert c.reachable_by_checker is True


def test_certify_provided_witness():
    edges = [(0, 1), (1, 2), (0, 2)]
    c = certify_prediction(3, edges, 0, 2, pred=1, witness=[0, 2])
    assert c.clean is True
    assert c.witness == (0, 2)
    # invalid provided witness falls back to reconstruct
    c2 = certify_prediction(3, edges, 0, 2, pred=1, witness=[0, 9, 2])
    assert c2.clean is True
    assert c2.witness is not None


def test_apply_cert_policy_force_closed_dirty_yes():
    rows = [
        {
            "encoding": "N 3 EDGES 1,2 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 0,
        },
        {
            "encoding": "N 3 EDGES 0,1 1,2 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 1,
        },
    ]
    preds = torch.tensor([1, 1], dtype=torch.long)
    out, certs, dy, dn = apply_cert_policy_preds(preds, rows)
    assert dy == [0]
    assert dn == []
    assert int(out[0].item()) == 0  # force-closed
    assert int(out[1].item()) == 1  # clean YES kept
    assert certs[0].clean is False
    assert certs[1].clean is True


def test_apply_cert_policy_keeps_dirty_no():
    rows = [
        {
            "encoding": "N 3 EDGES 0,1 1,2 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 1,
        }
    ]
    preds = torch.tensor([0], dtype=torch.long)
    out, certs, dy, dn = apply_cert_policy_preds(preds, rows)
    assert dn == [0]
    assert dy == []
    assert int(out[0].item()) == 0  # no oracle open
    assert certs[0].kind == "dirty_no"


def test_decide_verdict_enum():
    assert (
        decide_verdict(
            fo_killed=42,
            fo_total=45,
            rem22_killed=18,
            rem22_total=22,
            dirty_conc_ok=True,
            collateral_harm=False,
            collateral_reasons=[],
        )["verdict"]
        == "CERT_FO_CATCH"
    )
    assert (
        decide_verdict(
            fo_killed=23,
            fo_total=45,
            rem22_killed=0,
            rem22_total=22,
            dirty_conc_ok=True,
            collateral_harm=False,
            collateral_reasons=[],
        )["verdict"]
        == "CERT_PARTIAL"
    )
    assert (
        decide_verdict(
            fo_killed=3,
            fo_total=45,
            rem22_killed=0,
            rem22_total=22,
            dirty_conc_ok=False,
            collateral_harm=False,
            collateral_reasons=[],
        )["verdict"]
        == "CERT_NOISE"
    )
    assert (
        decide_verdict(
            fo_killed=45,
            fo_total=45,
            rem22_killed=22,
            rem22_total=22,
            dirty_conc_ok=True,
            collateral_harm=True,
            collateral_reasons=["matched_drop"],
        )["verdict"]
        == "COLLATERAL_HARM"
    )


def test_reach_certificate_to_dict():
    c = ReachCertificate(
        clean=True,
        pred=1,
        kind="path_witness",
        reason="test",
        witness=(0, 1),
        reachable_by_checker=True,
    )
    d = c.to_dict()
    assert d["witness"] == [0, 1]
    assert d["clean"] is True
