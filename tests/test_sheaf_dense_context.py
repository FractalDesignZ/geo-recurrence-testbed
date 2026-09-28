"""Tests for CYCLE_SHEAF_DENSE_CONTEXT generator (hard-neg + band)."""

from __future__ import annotations

import random

import pytest

from reachability_gen.gen_sheaf_dense_context import (
    CELL1_N,
    CELL1_P,
    CELL2_N,
    DENSE_K,
    SEQ_LEN_BAND,
    generate_er_balanced_cell,
    tune_cell2_p,
)
from reachability_gen.gen_sheaf_density_stress import assert_genuinely_unreachable as _assert
from reachability_gen.encode import parse_instance
from reachability_gen.graph import er_digraph, hop_distance


def test_tune_cell2_p_returns_reasonable():
    tune = tune_cell2_p(
        target_seq_mean=111.0,
        target_edge_mean=36.0,
        seed=7,
        p_init=0.036,
        n=CELL2_N,
    )
    assert 0.01 <= tune["p_tuned"] <= 0.08
    assert tune["probes"]


def test_cell1_mini_true_er_hard_neg_and_band_spirit():
    examples, report = generate_er_balanced_cell(
        seed=191_001,
        n=CELL1_N,
        p=CELL1_P,
        k=DENSE_K,
        n_pos=4,
        n_neg=4,
        split="test_dense_c1",
        cell_id="cell1_dense",
        max_graph_draws=30_000,
    )
    assert len(examples) == 8
    assert report["construction"] == "true_er_digraph"
    assert report["path_backbone"] is False
    assert abs(report["p_empirical"]["mean"] - CELL1_P) < 0.05
    assert report["science_open"] is False
    # Dense band spirit: mean should be near ~110 (not [45,70]).
    assert report["token_len"]["mean"] > 70

    for ex in examples:
        if ex.y == 0:
            n, edges, s, t = parse_instance(ex.encoding)
            _assert(n, edges, s, t)
        else:
            n, edges, s, t = parse_instance(ex.encoding)
            assert hop_distance(n, edges, s, t) == DENSE_K
