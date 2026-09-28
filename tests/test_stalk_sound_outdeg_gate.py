"""Stalk sound outdeg gate: local incidence force-unreach; no BFS; no train."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_sound_outdeg_gate import (  # noqa: E402
    COLLATERAL_DROP,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    FOCUS_T,
    FO_CORE_KILL_MIN,
    HN_CORE_MIN,
    VERDICTS,
    apply_outdeg_gate_preds,
    decide_verdict,
    gate_trigger_outdeg0,
    out_degree,
    out_neighborhood_size,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_SOUND_OUTDEG_GATE"
    assert FOCUS_T == 16
    assert FO_CORE_KILL_MIN == 40
    assert HN_CORE_MIN == 0.80
    assert COLLATERAL_DROP == 0.05
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert set(VERDICTS) == {
        "FO_CORE_KILLED",
        "FO_PARTIAL",
        "FO_UNMOVED",
        "COLLATERAL_HARM",
    }


def test_no_train_flags():
    import reachability_gen.run_stalk_sound_outdeg_gate as m

    doc = m.__doc__ or ""
    assert "No train" in doc or "no train" in doc.lower()
    assert "science_open=false" in doc.lower() or "science_open=false" in doc
    assert m.FOCUS_T == 16
    assert m.DEFAULT_ENSEMBLE_SEEDS == tuple(range(10))


def test_out_degree_local_incidence():
    # s=0 has no out-edges; s=1 has one
    edges = [(1, 2), (2, 3), (3, 1)]
    assert out_degree(4, edges, 0) == 0
    assert out_degree(4, edges, 1) == 1
    assert out_degree(4, edges, 2) == 1
    assert out_neighborhood_size(4, edges, 0) == 0
    assert out_neighborhood_size(4, edges, 1) == 1


def test_gate_trigger_outdeg0():
    # isolated source, t != s
    row = {
        "encoding": "N 4 EDGES 1,2 2,3 3,1 QUERY 0 2",
        "n": 4,
        "s": 0,
        "t": 2,
        "y": 0,
    }
    assert gate_trigger_outdeg0(row) is True
    # source with out-edge — no trigger
    row2 = {
        "encoding": "N 4 EDGES 0,1 1,2 QUERY 0 3",
        "n": 4,
        "s": 0,
        "t": 3,
        "y": 0,
    }
    assert gate_trigger_outdeg0(row2) is False
    # s == t — no trigger even if outdeg 0
    row3 = {
        "encoding": "N 3 EDGES 1,2 QUERY 0 0",
        "n": 3,
        "s": 0,
        "t": 0,
        "y": 1,
    }
    assert gate_trigger_outdeg0(row3) is False


def test_apply_outdeg_gate_force_zero_when_outdeg0():
    # Two rows: first triggers (outdeg0), second does not
    rows = [
        {
            "encoding": "N 4 EDGES 1,2 2,3 QUERY 0 2",
            "n": 4,
            "s": 0,
            "t": 2,
            "y": 0,
        },
        {
            "encoding": "N 4 EDGES 0,1 1,2 QUERY 0 3",
            "n": 4,
            "s": 0,
            "t": 3,
            "y": 0,
        },
    ]
    preds = torch.tensor([1, 1], dtype=torch.long)
    gated, triggers = apply_outdeg_gate_preds(preds, rows)
    assert triggers == [0]
    assert int(gated[0].item()) == 0  # forced unreachable
    assert int(gated[1].item()) == 1  # unchanged


def test_apply_outdeg_gate_unchanged_when_outdeg_positive():
    rows = [
        {
            "encoding": "N 3 EDGES 0,1 1,2 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 1,
        }
    ]
    preds = torch.tensor([1], dtype=torch.long)
    gated, triggers = apply_outdeg_gate_preds(preds, rows)
    assert triggers == []
    assert int(gated[0].item()) == 1


def test_decide_verdict_enum():
    assert (
        decide_verdict(
            fo_killed=42,
            fo_total=45,
            gated_hn=0.85,
            collateral_harm=False,
            collateral_reasons=[],
        )["verdict"]
        == "FO_CORE_KILLED"
    )
    assert (
        decide_verdict(
            fo_killed=23,
            fo_total=45,
            gated_hn=0.30,
            collateral_harm=False,
            collateral_reasons=[],
        )["verdict"]
        == "FO_PARTIAL"
    )
    assert (
        decide_verdict(
            fo_killed=0,
            fo_total=45,
            gated_hn=0.067,
            collateral_harm=False,
            collateral_reasons=[],
        )["verdict"]
        == "FO_UNMOVED"
    )
    # collateral wins even if core killed
    assert (
        decide_verdict(
            fo_killed=42,
            fo_total=45,
            gated_hn=0.90,
            collateral_harm=True,
            collateral_reasons=["matched_ood_HN_drop"],
        )["verdict"]
        == "COLLATERAL_HARM"
    )
    # FO_CORE needs both thresholds
    assert (
        decide_verdict(
            fo_killed=42,
            fo_total=45,
            gated_hn=0.50,
            collateral_harm=False,
            collateral_reasons=[],
        )["verdict"]
        == "FO_PARTIAL"
    )
