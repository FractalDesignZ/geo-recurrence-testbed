"""Stalk bag diversity: science_open fail-closed; bootstrap+subspace; not multi-hyp/distill/SWA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.encode import parse_instance  # noqa: E402
from reachability_gen.run_stalk_bag_diversity import (  # noqa: E402
    CYCLE,
    DEFAULT_MEMBER_SEEDS,
    EDGE_KEEP_P,
    LIFT_EPS,
    PREREG_HARD_NEG,
    PREREG_K16,
    PRIMARY_AGG,
    _bag_ckpt,
    _bootstrap_indices,
    _stable_unit_hash,
    _subgraph_row,
    build_member_train,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_BAG_DIVERSITY"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert LIFT_EPS == 0.01
    assert EDGE_KEEP_P == 0.75
    assert tuple(DEFAULT_MEMBER_SEEDS) == (0, 1, 2, 3, 4)
    assert PRIMARY_AGG == "prob_mean"


def test_ckpt_path_helper():
    assert _bag_ckpt(3).name == "fractal_core_stalk_bag_diversity_seed3_best.pt"


def test_stable_hash_deterministic_and_unit():
    a = _stable_unit_hash(0, "eh", 1, 2, "bag_subspace")
    b = _stable_unit_hash(0, "eh", 1, 2, "bag_subspace")
    c = _stable_unit_hash(1, "eh", 1, 2, "bag_subspace")
    assert a == b
    assert a != c
    assert 0.0 <= a < 1.0


def test_bootstrap_indices_size_and_seed_diversity():
    i0 = _bootstrap_indices(100, 0)
    i1 = _bootstrap_indices(100, 1)
    assert len(i0) == 100
    assert len(i1) == 100
    assert i0 != i1
    assert min(i0) >= 0 and max(i0) < 100


def test_subgraph_row_keeps_self_and_reduces_or_keeps_edges():
    row = {
        "encoding": "N 4 EDGES 0,1 1,2 2,3 0,0 QUERY 0 3",
        "edge_hash": "testhash",
        "n": 4,
        "s": 0,
        "t": 3,
        "y": 1,
    }
    out = _subgraph_row(row, member_seed=0, keep_p=0.75)
    n, edges, s, t = parse_instance(out["encoding"])
    assert n == 4 and s == 0 and t == 3
    # self-loop 0,0 always kept if present
    assert (0, 0) in edges
    assert out["bag_n_edges_kept"] <= out["bag_n_edges_orig"]
    assert out["bag_n_edges_kept"] == len(edges)


def test_build_member_train_meta():
    train = [
        {
            "encoding": f"N 3 EDGES 0,1 1,2 QUERY 0 2",
            "edge_hash": f"h{i}",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 1,
            "split": "train",
        }
        for i in range(20)
    ]
    meta = build_member_train(train, member_seed=2, keep_p=0.75)
    assert meta["n_rows"] == 20
    assert 0.0 < meta["mean_edge_keep_frac"] <= 1.0
    assert meta["edge_keep_p"] == 0.75


def test_not_multi_hyp_not_distill_not_swa_in_module_doc():
    import reachability_gen.run_stalk_bag_diversity as m

    doc = (m.__doc__ or "").lower()
    assert "not multi-hyp" in doc or "not soft" in doc
    assert "bootstrap" in doc
    assert m.EDGE_KEEP_P == 0.75
