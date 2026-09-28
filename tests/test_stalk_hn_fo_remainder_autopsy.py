"""Stalk HN FO remainder autopsy: local-sound cuts; wall below coverage min."""

from __future__ import annotations

import pytest

from reachability_gen.run_stalk_hn_fo_remainder_autopsy import (  # noqa: E402
    COVERAGE_MIN,
    CYCLE,
    FO_KILLED_N,
    FO_REMAINDER_N,
    GATE_VERDICTS,
    VERDICTS,
    decide_verdict,
    evaluate_sound_cuts,
    in_degree,
    local_incidence_features,
    sound_cut_predicates,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_HN_FO_REMAINDER_AUTOPSY"
    assert FO_REMAINDER_N == 22
    assert FO_KILLED_N == 23
    assert COVERAGE_MIN == 8
    assert set(VERDICTS) == {
        "STRUCTURAL_CLUSTER_REMAINDER",
        "DIFFUSE",
        "LOCAL_SOUND_CUT_FOUND",
        "LOCAL_SOUND_WALL",
        "INCONCLUSIVE",
    }
    assert set(GATE_VERDICTS) == {
        "FO_REMAINDER_KILLED",
        "FO_REMAINDER_PARTIAL",
        "COLLATERAL_HARM",
    }


def test_no_train_flags():
    import reachability_gen.run_stalk_hn_fo_remainder_autopsy as m

    doc = m.__doc__ or ""
    assert "No train" in doc or "no train" in doc.lower()
    assert "science_open=false" in doc.lower() or "science_open=false" in doc
    assert m.FOCUS_T == 16 or True  # FOCUS_T imported from hop_ood
    from reachability_gen.run_stalk_hn_fo_remainder_autopsy import FOCUS_T

    assert FOCUS_T == 16


def test_in_degree_local_incidence():
    edges = [(1, 0), (2, 0), (1, 2)]
    assert in_degree(4, edges, 0) == 2
    assert in_degree(4, edges, 1) == 0
    assert in_degree(4, edges, 2) == 1


def test_local_features_outdeg0_and_indeg_t0():
    # isolated source, hub-ish t with indeg
    row = {
        "encoding": "N 4 EDGES 1,2 2,3 3,1 QUERY 0 2",
        "n": 4,
        "s": 0,
        "t": 2,
        "y": 0,
    }
    lf = local_incidence_features(row)
    assert lf["C_outdeg0"] is True
    assert lf["outdeg_s"] == 0
    assert lf["C_deadend_nbrs"] is False  # no nbrs

    # indeg(t)==0
    row2 = {
        "encoding": "N 4 EDGES 0,1 1,2 QUERY 0 3",
        "n": 4,
        "s": 0,
        "t": 3,
        "y": 0,
    }
    lf2 = local_incidence_features(row2)
    assert lf2["C_indeg_t0"] is True
    assert lf2["indeg_t"] == 0
    assert lf2["C_outdeg0"] is False


def test_deadend_nbrs_sound_local_cut():
    # s→1, 1 has no out; t=2 not in star → unreachable by local star
    row = {
        "encoding": "N 4 EDGES 0,1 2,3 QUERY 0 2",
        "n": 4,
        "s": 0,
        "t": 2,
        "y": 0,
    }
    lf = local_incidence_features(row)
    assert lf["outdeg_s"] == 1
    assert lf["all_nbr_outdeg0"] is True
    assert lf["C_deadend_nbrs"] is True
    assert lf["C_outdeg1_deadend"] is True
    # if t is the neighbor, not a force-unreach (t reachable in 1 hop)
    row_reach = {
        "encoding": "N 4 EDGES 0,1 2,3 QUERY 0 1",
        "n": 4,
        "s": 0,
        "t": 1,
        "y": 1,
    }
    lf_r = local_incidence_features(row_reach)
    assert lf_r["C_deadend_nbrs"] is False


def test_evaluate_sound_cuts_coverage_and_soundness():
    rows = [
        {  # rem id 0: deadend star
            "encoding": "N 4 EDGES 0,1 2,3 QUERY 0 2",
            "n": 4,
            "s": 0,
            "t": 2,
            "y": 0,
        },
        {  # rem id 1: large path potential, not deadend
            "encoding": "N 4 EDGES 0,1 1,2 2,3 QUERY 0 3",
            "n": 4,
            "s": 0,
            "t": 3,
            "y": 1,  # reachable — must not trigger sound cuts
        },
        {  # rem id 2: indeg_t0
            "encoding": "N 3 EDGES 0,1 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 0,
        },
    ]
    ev = evaluate_sound_cuts(rows, remainder_ids=[0, 2])
    assert ev["C_deadend_nbrs"]["sound"] is True
    assert ev["C_indeg_t0"]["sound"] is True
    assert 0 in ev["C_deadend_nbrs"]["remainder_ids"]
    assert 2 in ev["C_indeg_t0"]["remainder_ids"]
    # reachable row must not be a C_deadend / C_indeg_t0 violation
    assert ev["C_deadend_nbrs"]["full_violations_y_neq_0"] == 0
    assert ev["C_indeg_t0"]["full_violations_y_neq_0"] == 0


def test_decide_verdict_wall_when_coverage_low():
    # Fabricate cut_eval with max coverage 5 < 8
    cut_eval = {
        "C_outdeg0": {
            "remainder_hits": 0,
            "actionable": False,
            "sound": True,
        },
        "C_indeg_t0": {
            "remainder_hits": 0,
            "actionable": False,
            "sound": True,
        },
        "C_deadend_nbrs": {
            "remainder_hits": 5,
            "actionable": False,
            "sound": True,
        },
        "C_outdeg1_deadend": {
            "remainder_hits": 5,
            "actionable": False,
            "sound": True,
        },
    }

    def _fake_table(n_reach_med, outdeg_med, deadend_true, n=22):
        return {
            "numeric": {
                "n_reach_from_s": {"median": n_reach_med},
                "outdeg_s": {"median": outdeg_med},
            },
            "bool_counts": {"C_deadend_nbrs": {"true": deadend_true, "false": n - deadend_true}},
        }

    d = decide_verdict(
        cut_eval=cut_eval,
        remainder_table=_fake_table(19.0, 1.5, 5),
        killed_table=_fake_table(1.0, 0.0, 0, n=23),
        ok_table=_fake_table(19.0, 2.0, 0, n=16),
    )
    assert d["verdict"] == "LOCAL_SOUND_WALL"
    assert d["gate_run"] is False
    assert d["best_sound_coverage"] == 5


def test_decide_verdict_cut_found_when_actionable():
    cut_eval = {
        "C_deadend_nbrs": {
            "remainder_hits": 10,
            "actionable": True,
            "sound": True,
        },
        "C_outdeg0": {
            "remainder_hits": 0,
            "actionable": False,
            "sound": True,
        },
    }
    d = decide_verdict(
        cut_eval=cut_eval,
        remainder_table={
            "numeric": {
                "n_reach_from_s": {"median": 2.0},
                "outdeg_s": {"median": 1.0},
            },
            "bool_counts": {"C_deadend_nbrs": {"true": 10, "false": 12}},
        },
        killed_table={
            "numeric": {
                "n_reach_from_s": {"median": 1.0},
                "outdeg_s": {"median": 0.0},
            },
            "bool_counts": {"C_deadend_nbrs": {"true": 0, "false": 23}},
        },
        ok_table={
            "numeric": {
                "n_reach_from_s": {"median": 19.0},
                "outdeg_s": {"median": 2.0},
            },
            "bool_counts": {"C_deadend_nbrs": {"true": 0, "false": 16}},
        },
    )
    assert d["verdict"] == "LOCAL_SOUND_CUT_FOUND"
    assert d["gate_run"] is True
    assert "C_deadend_nbrs" in d["actionable_cuts"]


def test_sound_cut_predicates_locked_names():
    names = set(sound_cut_predicates())
    assert names == {
        "C_outdeg0",
        "C_indeg_t0",
        "C_deadend_nbrs",
        "C_outdeg1_deadend",
    }
