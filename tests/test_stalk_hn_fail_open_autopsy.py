"""Stalk HN FAIL_OPEN autopsy: classification helpers / clustering summaries."""

from __future__ import annotations

import math

import pytest

from reachability_gen.run_stalk_hn_fail_open_autopsy import (  # noqa: E402
    CONF_THRESH,
    CYCLE,
    FOCUS_T,
    STRUCT_NUMERIC_KEYS,
    VERDICTS,
    _median_outside_iqr,
    _summarize_numeric,
    classify_member_agreement,
    decide_autopsy_verdict,
    structural_features,
    summarize_cohort_features,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_HN_FAIL_OPEN_AUTOPSY"
    assert FOCUS_T == 16
    assert CONF_THRESH == 0.80
    assert set(VERDICTS) == {
        "STRUCTURAL_CLUSTER",
        "DIFFUSE",
        "DATA_ARTIFACT",
        "INCONCLUSIVE",
    }


def test_no_train_flags():
    import reachability_gen.run_stalk_hn_fail_open_autopsy as m

    doc = m.__doc__ or ""
    assert "No train" in doc or "no train" in doc.lower()
    assert "science_open=false" in doc.lower() or "science_open=false" in doc
    assert m.FOCUS_T == 16
    assert m.DEFAULT_ENSEMBLE_SEEDS == tuple(range(10))


def test_classify_member_agreement_tags():
    assert (
        classify_member_agreement([1] * 10, [0.9] * 10) == "HARD_UNANIMOUS"
    )
    assert (
        classify_member_agreement([1] * 10, [0.9] * 9 + [0.5]) == "SOFT_AGREE"
    )
    assert classify_member_agreement([1] * 9 + [0], [0.9] * 10) == "MIXED_PRED"
    assert classify_member_agreement([], []) == "MIXED_PRED"


def test_summarize_numeric_and_cohort():
    s = _summarize_numeric([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["median"] == 3.0
    assert s["min"] == 1.0
    assert s["max"] == 5.0
    empty = _summarize_numeric([])
    assert empty["n"] == 0
    assert math.isnan(empty["mean"])

    feats = [
        {
            "n": 32,
            "n_edges": 10,
            "p": 0.06,
            "p_emp": 0.01,
            "token_len": 40,
            "deg_s": 2,
            "deg_t": 3,
            "n_reach_from_s": 4,
            "n_reach_to_t": 5,
            "frac_reach_from_s": 0.125,
            "frac_reach_to_t": 0.156,
            "near_miss_bridges": 0,
            "frontier_exits": 0,
            "max_dist_from_s": 2,
            "arm_id": None,
            "arm_meta": None,
        },
        {
            "n": 32,
            "n_edges": 20,
            "p": 0.06,
            "p_emp": 0.02,
            "token_len": 50,
            "deg_s": 4,
            "deg_t": 1,
            "n_reach_from_s": 8,
            "n_reach_to_t": 2,
            "frac_reach_from_s": 0.25,
            "frac_reach_to_t": 0.0625,
            "near_miss_bridges": 2,
            "frontier_exits": 1,
            "max_dist_from_s": 5,
            "arm_id": None,
            "arm_meta": None,
        },
    ]
    out = summarize_cohort_features(feats, STRUCT_NUMERIC_KEYS)
    assert out["n"] == 2
    assert out["bins"]["frontier"]["frontier>0"] == 1
    assert out["bins"]["frontier"]["frontier==0"] == 1
    assert out["construction_tags"]["none"] == 2


def test_structural_features_hard_neg_toy():
    # Two components, no bridge: s-side {0,1}, t-side {2,3}.
    encoding = "N 4 EDGES 0,1 1,0 2,3 3,2 QUERY 0 2"
    row = {
        "encoding": encoding,
        "n": 4,
        "p": 0.1,
        "s": 0,
        "t": 2,
        "y": 0,
        "hop_distance": -1,
        "arm_id": None,
        "arm_meta": None,
        "edge_hash": "x",
        "seed": 1,
    }
    f = structural_features(row)
    assert f["n"] == 4
    assert f["n_edges"] == 4
    assert f["n_reach_from_s"] == 2
    assert f["n_reach_to_t"] == 2
    assert f["frontier_exits"] == 0
    # Missing bridges: every u in R_out × v in R_in\R_out absent → 2*2=4
    assert f["near_miss_bridges"] == 4

    # Frontier exit without reachability to t: 0→1→2, t=3 isolated from 2.
    encoding2 = "N 4 EDGES 0,1 1,2 QUERY 0 3"
    row2 = {**row, "encoding": encoding2, "t": 3}
    f2 = structural_features(row2)
    assert f2["n_reach_from_s"] == 3  # {0,1,2}
    assert f2["n_reach_to_t"] == 1  # {3}
    assert f2["frontier_exits"] == 0  # no edge leaves R_out
    assert f2["near_miss_bridges"] == 3  # {0,1,2} × {3} all missing


def test_median_outside_iqr():
    assert _median_outside_iqr(10.0, 1.0, 5.0) is True
    assert _median_outside_iqr(3.0, 1.0, 5.0) is False
    assert _median_outside_iqr(float("nan"), 1.0, 5.0) is False


def _fake_summary(n: int, medians: dict, near_miss_bins: dict) -> dict:
    numeric = {
        k: {
            "n": n,
            "mean": v,
            "median": v,
            "std": 0.0,
            "min": v,
            "max": v,
            "q25": v * 0.8 if v else 0.0,
            "q75": v * 1.2 if v else 1.0,
        }
        for k, v in medians.items()
    }
    # fill required keys
    for k in (
        "n_edges",
        "p_emp",
        "token_len",
        "deg_s",
        "deg_t",
        "n_reach_from_s",
        "n_reach_to_t",
        "frac_reach_from_s",
        "frac_reach_to_t",
        "near_miss_bridges",
        "frontier_exits",
        "max_dist_from_s",
    ):
        if k not in numeric:
            numeric[k] = {
                "n": n,
                "mean": 1.0,
                "median": 1.0,
                "std": 0.0,
                "min": 1.0,
                "max": 1.0,
                "q25": 0.5,
                "q75": 1.5,
            }
    # near_miss_bins arg may use legacy keys; also populate frontier/frac_reach_s
    frontier_bins = {}
    for k, v in near_miss_bins.items():
        if "near_miss>" in k or k == "frontier>0":
            frontier_bins["frontier>0"] = frontier_bins.get("frontier>0", 0) + v
        elif "near_miss=" in k or k == "frontier==0":
            frontier_bins["frontier==0"] = frontier_bins.get("frontier==0", 0) + v
        else:
            frontier_bins[k] = v
    # Mirror frontier mass into reach bins only when FO-sized + strongly frontier>0
    fo_heavy = (
        n >= 40
        and frontier_bins.get("frontier>0", 0) >= 0.75 * n
    )
    ok_like_low = (
        n < 40
        and frontier_bins.get("frontier==0", 0) >= 0.75 * n
    )
    if fo_heavy:
        reach_bins = {"frac_reach_s>=0.25": n, "frac_reach_s<0.25": 0}
    elif ok_like_low:
        reach_bins = {"frac_reach_s>=0.25": 0, "frac_reach_s<0.25": n}
    else:
        # matched / diffuse: same proportional split, no dominant contrast
        hi = frontier_bins.get("frontier>0", n // 2)
        reach_bins = {
            "frac_reach_s>=0.25": hi,
            "frac_reach_s<0.25": n - hi,
        }
    return {
        "n": n,
        "numeric": numeric,
        "bins": {
            "n": {"32": n},
            "p": {"0.06": n},
            "near_miss": frontier_bins,
            "frontier": frontier_bins,
            "frac_reach_s": reach_bins,
        },
        "construction_tags": {"none": n},
    }


def test_decide_verdict_diffuse():
    # Same medians → DIFFUSE
    med = {
        "n_edges": 40.0,
        "p_emp": 0.04,
        "token_len": 100.0,
        "deg_s": 3.0,
        "deg_t": 3.0,
        "n_reach_from_s": 8.0,
        "n_reach_to_t": 8.0,
        "frac_reach_from_s": 0.25,
        "frac_reach_to_t": 0.25,
        "near_miss_bridges": 2.0,
        "frontier_exits": 1.0,
        "max_dist_from_s": 4.0,
    }
    fo = _fake_summary(45, med, {"near_miss>0": 30, "near_miss==0": 15})
    ok = _fake_summary(16, med, {"near_miss>0": 10, "near_miss==0": 6})
    fc = _fake_summary(179, med, {"near_miss>0": 100, "near_miss==0": 79})
    out = decide_autopsy_verdict(
        fo, ok, fc, artifact_flags={"label_inconsistency": False, "encoding_defect": False}
    )
    assert out["verdict"] == "DIFFUSE"


def test_decide_verdict_cluster():
    fo_med = {
        "n_edges": 40.0,
        "p_emp": 0.04,
        "token_len": 100.0,
        "deg_s": 3.0,
        "deg_t": 3.0,
        "n_reach_from_s": 20.0,  # high
        "n_reach_to_t": 20.0,
        "frac_reach_from_s": 0.6,
        "frac_reach_to_t": 0.6,
        "near_miss_bridges": 5.0,
        "frontier_exits": 3.0,
        "max_dist_from_s": 8.0,
    }
    ok_med = {
        "n_edges": 40.0,
        "p_emp": 0.04,
        "token_len": 100.0,
        "deg_s": 3.0,
        "deg_t": 3.0,
        "n_reach_from_s": 4.0,
        "n_reach_to_t": 4.0,
        "frac_reach_from_s": 0.1,
        "frac_reach_to_t": 0.1,
        "near_miss_bridges": 0.0,
        "frontier_exits": 0.0,
        "max_dist_from_s": 2.0,
    }
    fo = _fake_summary(45, fo_med, {"near_miss>0": 40, "near_miss==0": 5})
    # OK IQR for n_reach: q25=0.8*4=3.2, q75=1.2*4=4.8 — FO med 20 outside
    ok = _fake_summary(16, ok_med, {"near_miss>0": 2, "near_miss==0": 14})
    fc = _fake_summary(179, ok_med, {"near_miss>0": 20, "near_miss==0": 159})
    out = decide_autopsy_verdict(
        fo, ok, fc, artifact_flags={"label_inconsistency": False, "encoding_defect": False}
    )
    assert out["verdict"] == "STRUCTURAL_CLUSTER"
    assert out["cluster_hits"]


def test_decide_verdict_data_artifact():
    med = {"n_edges": 1.0}
    fo = _fake_summary(45, med, {"near_miss==0": 45})
    ok = _fake_summary(16, med, {"near_miss==0": 16})
    fc = _fake_summary(10, med, {"near_miss==0": 10})
    out = decide_autopsy_verdict(
        fo,
        ok,
        fc,
        artifact_flags={
            "label_inconsistency": True,
            "encoding_defect": False,
            "reasons": ["bad"],
        },
    )
    assert out["verdict"] == "DATA_ARTIFACT"
