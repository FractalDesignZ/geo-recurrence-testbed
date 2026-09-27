"""Tests for FractalCore mask / parity / stalk-local Gate0 (MEASURE)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.models.fractal_core import (  # noqa: E402
    DISCRETE_T_VALUES,
    FractalCore,
    build_adjacency_attn_mask,
    build_node_slot_batch,
    _verify_param_parity,
)
from reachability_gen.overfit_fractal import run_overfit_fractal  # noqa: E402
from reachability_gen.overfit_ff import ensure_balanced_batch  # noqa: E402
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS  # noqa: E402
from reachability_gen.arms import FractalCoreArm  # noqa: E402


def test_adjacency_mask_allows_self_and_incoming_only():
    A = build_adjacency_attn_mask(3, [(0, 1), (1, 2)], max_n=4)
    # self
    assert A[0, 0] == 0.0 and A[1, 1] == 0.0 and A[2, 2] == 0.0
    # edge 0→1: query=1 key=0 allowed
    assert A[1, 0] == 0.0
    assert A[0, 1] < 0  # reverse denied
    assert A[2, 1] == 0.0
    # disconnected
    assert A[2, 0] < 0
    # pad row/col denied (except we leave pad as -inf; no self on pad)
    assert A[3, 3] < 0
    assert A[0, 3] < 0


def test_node_slot_batch_recovers_edges_from_encoding():
    ex = [
        {"encoding": "N 4 EDGES 0,1 1,2 QUERY 0 2", "y": 1},
        {"encoding": "N 3 EDGES 0,1 QUERY 0 2", "y": 0},
    ]
    batch = build_node_slot_batch(ex, max_n=8)
    assert tuple(batch["node_ids"].shape) == (2, 8)
    assert batch["node_mask"][0].sum().item() == 4
    assert batch["node_mask"][1].sum().item() == 3
    assert int(batch["s_idx"][0]) == 0 and int(batch["t_idx"][0]) == 2
    # edge 0→1 present for first graph
    assert batch["attn_mask"][0, 1, 0].item() == 0.0


def test_fractal_forward_shapes_discrete_and_parity():
    model = FractalCore(d=64, T=4, mlp_expansion=10, max_nodes=64, max_T=16)
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    assert parity["within_5pct"] is True
    assert parity["science_open"] is False
    # no halt_gate / no c_proj; has local stalk/probe
    assert not hasattr(model, "halt_gate") or model.halt_gate is None
    assert hasattr(model, "stalk_proj") and hasattr(model, "probe_proj")
    assert not hasattr(model, "c_proj")
    ex = [
        {"encoding": "N 5 EDGES 0,1 1,2 2,3 3,4 QUERY 0 4", "y": 1},
        {"encoding": "N 5 EDGES 0,1 2,3 QUERY 0 4", "y": 0},
    ]
    batch = build_node_slot_batch(ex, max_n=64)
    logits, traj, info = model(
        batch["node_ids"],
        batch["node_mask"],
        batch["attn_mask"],
        batch["s_idx"],
        batch["t_idx"],
        return_trajectory=True,
        return_halt=True,
    )
    assert tuple(logits.shape) == (2, 2)
    assert traj is not None and len(traj) == 5  # z0 + 4 cycles
    assert info is not None
    assert info["adaptive_halt"] is False
    assert info["broadcast_c"] is False
    assert info["local_potential"] is True
    assert info["mean_halt_step"] == 4.0


def test_disconnected_target_leak_near_zero():
    model = FractalCore(d=64, T=6, mlp_expansion=10, max_nodes=64, max_T=16)
    ex = [
        {"encoding": "N 5 EDGES 0,1 2,3 QUERY 0 4", "y": 0},
        {"encoding": "N 4 EDGES 0,1 QUERY 0 3", "y": 0},
        {"encoding": "N 6 EDGES 0,1 1,2 3,4 QUERY 0 5", "y": 0},
    ]
    leak = model.disconnected_target_leak(ex, T_values=DISCRETE_T_VALUES, atol=1e-3)
    assert leak["ok"] is True
    for T, row in leak["by_T"].items():
        assert row["max_l2"] <= 1e-3, (T, row)


def test_fractal_arm_attach_and_forward():
    arm = FractalCoreArm(T=4, d=64, mlp_expansion=10)
    arm.attach_default_model(seed=0, max_nodes=32)
    assert arm.has_real_model
    logits, traj = arm.forward(
        {"encoding": ["N 4 EDGES 0,1 1,2 QUERY 0 2"], "y": [1]},
        return_trajectory=True,
    )
    assert len(logits) == 1 and len(logits[0]) == 2
    assert traj is not None and len(traj) == 5


def test_verify_param_parity_fail_closed():
    with pytest.raises(AssertionError):
        _verify_param_parity(1, ff_baseline=FF_BASELINE_PARAMS)


@pytest.mark.slow
def test_fractal_overfit_gate0_balanced():
    batch, note, _meta = ensure_balanced_batch(
        Path("data/train_tiny.jsonl"), n_pos=16, n_neg=16, regenerate=True
    )
    assert len(batch) == 32, note
    result = run_overfit_fractal(
        batch,
        steps=100,
        d=64,
        T=6,
        lr=3e-3,
        seed=0,
        loss_threshold=1e-3,
        require_per_class=True,
        mlp_expansion=10,
        max_nodes=64,
    )
    assert result["ok"] is True
    assert result["final_acc"] >= 1.0 - 1e-9
    assert result["final_loss"] < 1e-3
    assert result["science_open"] is False
    assert result.get("disconnect_leak", {}).get("ok") is True
