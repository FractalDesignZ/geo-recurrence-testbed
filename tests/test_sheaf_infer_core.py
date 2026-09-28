"""Tests for SheafInferCore edge-token gate / parity / Gate0 (MEASURE)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.models.sheaf_infer_core import (  # noqa: E402
    DISCRETE_T_VALUES,
    SheafInferCore,
    build_sheaf_batch,
    ste_hard_gate,
    _verify_param_parity,
)
from reachability_gen.overfit_sheaf import run_overfit_sheaf  # noqa: E402
from reachability_gen.overfit_ff import ensure_balanced_batch  # noqa: E402
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS  # noqa: E402


def test_ste_hard_gate_eval_hard():
    logits = torch.tensor([-2.0, 0.0, 2.0])
    g = ste_hard_gate(logits, theta=0.5, training=False, mode="ste")
    assert g.tolist() == [0.0, 0.0, 1.0]


def test_sheaf_batch_no_oracle_attn_mask():
    ex = [
        {"encoding": "N 4 EDGES 0,1 1,2 QUERY 0 2", "y": 1},
        {"encoding": "N 3 EDGES 0,1 QUERY 0 2", "y": 0},
    ]
    batch = build_sheaf_batch(ex, max_n=8)
    assert "attn_mask" not in batch  # no hard A oracle in batch
    assert "edge_index" in batch and "gold_adj" in batch
    assert tuple(batch["node_ids"].shape) == (2, 8)
    assert batch["edge_mask"][0].sum().item() == 2
    # gold adj: edge 0→1 → gold[1,0]=1
    assert batch["gold_adj"][0, 1, 0].item() == 1.0


def test_sheaf_forward_shapes_parity_and_no_oracle():
    model = SheafInferCore(d=64, T=4, mlp_expansion=12, max_nodes=64, max_T=16, residual_alpha=1.0)
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    assert parity["within_5pct"] is True
    assert parity["science_open"] is False
    assert 115_157 <= model.param_count() <= 127_279
    ex = [
        {"encoding": "N 5 EDGES 0,1 1,2 2,3 3,4 QUERY 0 4", "y": 1},
        {"encoding": "N 5 EDGES 0,1 2,3 QUERY 0 4", "y": 0},
    ]
    batch = build_sheaf_batch(ex, max_n=64)
    logits, traj, info = model(
        batch["node_ids"],
        batch["node_mask"],
        batch["edge_index"],
        batch["edge_mask"],
        batch["s_idx"],
        batch["t_idx"],
        return_trajectory=True,
        return_halt=True,
        return_states=True,
        return_edge_logits=True,
    )
    assert tuple(logits.shape) == (2, 2)
    assert traj is not None and len(traj) == 5
    assert info is not None
    assert info["hard_A_oracle"] is False
    assert info["broadcast_c"] is False
    assert info["local_potential"] is True
    # Connected: ‖h_t‖>0; disconnect: ‖h_t‖≈0 under A_hat
    assert float(info["target_l2"][0]) > 1e-6
    assert float(info["target_l2"][1]) <= 1e-3


def test_disconnected_target_norm_near_zero():
    model = SheafInferCore(d=64, T=6, mlp_expansion=12, max_nodes=64, max_T=16, residual_alpha=1.0)
    ex = [
        {"encoding": "N 5 EDGES 0,1 2,3 QUERY 0 4", "y": 0},
        {"encoding": "N 4 EDGES 0,1 QUERY 0 3", "y": 0},
        {"encoding": "N 6 EDGES 0,1 1,2 3,4 QUERY 0 5", "y": 0},
    ]
    leak = model.disconnected_target_norm(ex, T_values=DISCRETE_T_VALUES, atol=1e-3)
    assert leak["ok"] is True
    for T, row in leak["by_T"].items():
        assert row["max_l2"] <= 1e-3, (T, row)


def test_verify_param_parity_fail_closed():
    with pytest.raises(AssertionError):
        _verify_param_parity(1, ff_baseline=FF_BASELINE_PARAMS)


@pytest.mark.slow
def test_sheaf_overfit_gate0_balanced():
    batch, note, _meta = ensure_balanced_batch(
        Path("data/train_tiny.jsonl"), n_pos=16, n_neg=16, regenerate=True
    )
    assert len(batch) == 32, note
    result = run_overfit_sheaf(
        batch,
        steps=150,
        d=64,
        T=6,
        lr=3e-3,
        seed=0,
        loss_threshold=1e-3,
        require_per_class=True,
        mlp_expansion=12,
        max_nodes=64,
    )
    assert result["ok"] is True
    assert result["final_acc"] >= 1.0 - 1e-9
    assert result["final_loss"] < 1e-3
    assert result["final_edge_recon_acc"] >= 0.99
    assert result["science_open"] is False
    assert result.get("disconnect_target_norm", {}).get("ok") is True
