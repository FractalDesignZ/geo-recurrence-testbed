"""Tests for CYCLE_SHEAF_DENSITY_STRESS generator (hard-neg R_out asserts)."""

from __future__ import annotations

import random

import pytest

from reachability_gen.gen_sheaf_density_stress import (
    DENSITY_K,
    DENSITY_N,
    DENSITY_P,
    assert_genuinely_unreachable,
    generate_sheaf_density_stress_cell,
    _harvest_hard_negs_rejection,
)
from reachability_gen.graph import (
    er_digraph,
    hop_distance,
    reachable_bfs,
    reachable_dfs,
    reachable_out_set,
)
from reachability_gen.encode import parse_instance


def test_reachable_out_set_bfs_dfs_agree():
    rng = random.Random(0)
    for _ in range(20):
        edges = er_digraph(DENSITY_N, DENSITY_P, rng)
        for s in range(DENSITY_N):
            a = reachable_out_set(DENSITY_N, edges, s, method="bfs")
            b = reachable_out_set(DENSITY_N, edges, s, method="dfs")
            assert a == b
            for t in range(DENSITY_N):
                in_set = t in a
                assert reachable_bfs(DENSITY_N, edges, s, t) is in_set
                assert reachable_dfs(DENSITY_N, edges, s, t) is in_set


def test_assert_genuinely_unreachable_rejects_reachable():
    # Path 0→1→2; 0 can reach 2.
    edges = [(0, 1), (1, 2)]
    with pytest.raises(AssertionError):
        assert_genuinely_unreachable(3, edges, 0, 2)


def test_assert_genuinely_unreachable_accepts_hard_neg():
    # 0→1 and 2→3; 0 cannot reach 3; both ends deg>=1.
    edges = [(0, 1), (2, 3)]
    assert_genuinely_unreachable(4, edges, 0, 3)
    r = reachable_out_set(4, edges, 0, method="bfs")
    assert 3 not in r


def test_harvest_hard_negs_rejection_r_out():
    rng = random.Random(7)
    edges = er_digraph(DENSITY_N, DENSITY_P, rng)
    negs = _harvest_hard_negs_rejection(DENSITY_N, edges, rng)
    for u, v in negs:
        assert v not in reachable_out_set(DENSITY_N, edges, u, method="bfs")
        assert v not in reachable_out_set(DENSITY_N, edges, u, method="dfs")
        assert_genuinely_unreachable(DENSITY_N, edges, u, v)


def test_density_cell_true_er_no_backbone_and_hard_neg():
    examples, report = generate_sheaf_density_stress_cell(
        seed=189_001, n_pos=8, n_neg=8, max_graph_draws=20_000
    )
    assert len(examples) == 16
    assert report["construction"] == "true_er_digraph"
    assert report["path_backbone"] is False
    assert report["p_requested"] == DENSITY_P
    assert report["science_open"] is False
    # Empirical p should stay near requested (true ER, no backbone).
    assert abs(report["p_empirical"]["mean"] - DENSITY_P) < 0.05

    n_pos = sum(1 for ex in examples if ex.y == 1)
    n_neg = sum(1 for ex in examples if ex.y == 0)
    assert n_pos == 8 and n_neg == 8

    for ex in examples:
        n, edges, s, t = parse_instance(ex.encoding)
        assert n == DENSITY_N
        if ex.y == 1:
            assert hop_distance(n, edges, s, t) == DENSITY_K
            assert ex.hop_distance == DENSITY_K
        else:
            assert_genuinely_unreachable(n, edges, s, t)
            assert t not in reachable_out_set(n, edges, s, method="bfs")
            assert t not in reachable_out_set(n, edges, s, method="dfs")
