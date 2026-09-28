"""Stalk orientation collapse probe: metric helpers + verdict; no train."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.orientation import (  # noqa: E402
    auroc_binary,
    build_in_out_index,
    decide_verdict,
    example_orientation_from_states,
    mean_metrics,
    median_outside_iqr,
    neighbor_mean,
    orientation_definition_doc,
    slot_orientation_metrics,
    summary_numeric,
)
from reachability_gen.run_stalk_orientation_collapse_probe import (  # noqa: E402
    AUROC_PARTIAL,
    AUROC_SEP,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    FOCUS_T,
    NAN_CAP,
    VERDICTS,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_ORIENTATION_COLLAPSE_PROBE"
    assert FOCUS_T == 16
    assert AUROC_SEP == 0.75
    assert AUROC_PARTIAL == 0.60
    assert NAN_CAP == 0.50
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert set(VERDICTS) == {
        "ORIENT_SEPARATES_FO",
        "ORIENT_PARTIAL",
        "ORIENT_NULL",
        "INCONCLUSIVE_ARCH",
    }


def test_no_train_flags():
    import reachability_gen.run_stalk_orientation_collapse_probe as m

    doc = m.__doc__ or ""
    assert "no train" in doc.lower() or "No train" in doc
    assert "science_open=false" in doc.lower() or "science_open=false" in doc
    assert "Phase 4" in doc or "phase4" in doc.lower() or "Phase4" in doc
    assert m.FOCUS_T == 16
    defs = orientation_definition_doc()
    assert defs["explicit_in_out_channels"] is False
    assert defs["attention_weights_used"] is False
    assert defs["scope"] == "directed_incidence_x_final_hidden"


def test_build_in_out_index():
    n_in, n_out = build_in_out_index(3, [(0, 1), (1, 2), (0, 2)])
    assert n_out[0] == [1, 2]
    assert n_in[2] == [1, 0]
    assert n_in[0] == []
    assert n_out[2] == []


def test_neighbor_mean_empty_and_nonempty():
    H = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    z = neighbor_mean(H, [])
    assert torch.allclose(z, torch.zeros(2))
    m = neighbor_mean(H, [0, 1])
    assert torch.allclose(m, torch.tensor([0.5, 1.0]))


def test_slot_orientation_separated_vs_collapsed():
    # Node 1 in-neigh=0 (vec x), out-neigh=2 (vec y) — orthogonal → cos≈0
    H = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    n_in, n_out = build_in_out_index(3, [(0, 1), (1, 2)])
    m = slot_orientation_metrics(H, n_in, n_out, 1)
    assert m["deg_in"] == 1.0 and m["deg_out"] == 1.0
    assert abs(m["cos_orient"]) < 1e-5
    assert m["l2_orient"] == pytest.approx(math.sqrt(2.0), rel=1e-5)

    # Collapsed: in and out neighbors share same direction
    H2 = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ]
    )
    m2 = slot_orientation_metrics(H2, n_in, n_out, 1)
    assert m2["cos_orient"] == pytest.approx(1.0, abs=1e-5)


def test_example_orientation_keys():
    H = torch.randn(4, 8)
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    d = example_orientation_from_states(H, 4, edges, s=0, t=3)
    for suffix in ("_t", "_s"):
        assert f"cos_orient{suffix}" in d
        assert f"l2_orient{suffix}" in d
        assert f"mass_ratio{suffix}" in d


def test_mean_metrics_nanmean():
    a = {"cos_orient_t": 1.0, "l2_orient_t": 2.0}
    b = {"cos_orient_t": float("nan"), "l2_orient_t": 4.0}
    m = mean_metrics([a, b])
    assert m["cos_orient_t"] == pytest.approx(1.0)
    assert m["l2_orient_t"] == pytest.approx(3.0)


def test_auroc_perfect_and_chance():
    # perfect: scores = labels
    assert auroc_binary([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    # inverted
    assert auroc_binary([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)
    # chance-ish with ties
    r = auroc_binary([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1])
    assert r == pytest.approx(0.5)


def test_summary_and_median_outside_iqr():
    s = summary_numeric([1.0, 2.0, 3.0, 4.0, float("nan")])
    assert s["n"] == 5
    assert s["n_finite"] == 4
    assert s["nan_rate"] == pytest.approx(0.2)
    assert s["median"] == pytest.approx(2.5)
    assert median_outside_iqr(10.0, 1.0, 3.0) is True
    assert median_outside_iqr(2.0, 1.0, 3.0) is False


def test_decide_verdict_inconclusive():
    d = decide_verdict(
        auroc_cos_fo=0.9,
        auroc_l2_fo=0.9,
        auroc_cos_rem=0.9,
        auroc_l2_rem=0.9,
        rem_outside_ok_cos=True,
        rem_outside_ok_l2=True,
        nan_rate_rem=0.8,
        nan_rate_ok=0.1,
    )
    assert d["verdict"] == "INCONCLUSIVE_ARCH"


def test_decide_verdict_separates():
    d = decide_verdict(
        auroc_cos_fo=0.80,
        auroc_l2_fo=0.55,
        auroc_cos_rem=0.70,
        auroc_l2_rem=0.55,
        rem_outside_ok_cos=True,
        rem_outside_ok_l2=False,
        nan_rate_rem=0.0,
        nan_rate_ok=0.0,
    )
    assert d["verdict"] == "ORIENT_SEPARATES_FO"


def test_decide_verdict_null():
    d = decide_verdict(
        auroc_cos_fo=0.50,
        auroc_l2_fo=0.52,
        auroc_cos_rem=0.48,
        auroc_l2_rem=0.51,
        rem_outside_ok_cos=False,
        rem_outside_ok_l2=False,
        nan_rate_rem=0.0,
        nan_rate_ok=0.0,
    )
    assert d["verdict"] == "ORIENT_NULL"


def test_decide_verdict_partial():
    d = decide_verdict(
        auroc_cos_fo=0.65,
        auroc_l2_fo=0.55,
        auroc_cos_rem=0.60,
        auroc_l2_rem=0.55,
        rem_outside_ok_cos=False,
        rem_outside_ok_l2=False,
        nan_rate_rem=0.0,
        nan_rate_ok=0.0,
    )
    assert d["verdict"] == "ORIENT_PARTIAL"
