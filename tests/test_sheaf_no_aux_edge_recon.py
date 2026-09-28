"""Tests for no-aux edge-recon cycle plumbing (MEASURE)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.models.sheaf_infer_core import (  # noqa: E402
    SheafInferCore,
    _verify_param_parity,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS  # noqa: E402
from reachability_gen.run_sheaf_no_aux_edge_recon import (  # noqa: E402
    CYCLE,
    EDGE_RECON_WEIGHT,
    PR8_MERGE_SHA,
)
from reachability_gen.train.sheaf_trainer import SheafTrainer  # noqa: E402


def test_edge_recon_weight_locked_off():
    assert EDGE_RECON_WEIGHT == 0.0
    assert CYCLE == "CYCLE_SHEAF_NO_AUX_EDGE_RECON"
    assert PR8_MERGE_SHA.startswith("e0877eb")


def test_trainer_respects_zero_recon_weight():
    torch.manual_seed(0)
    m = SheafInferCore(
        d=64, T=6, mlp_expansion=12, max_nodes=32, max_T=16, neutral_init=True
    )
    trainer = SheafTrainer(m, edge_recon_weight=0.0)
    assert trainer.edge_recon_weight == 0.0
    parity = _verify_param_parity(m.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    assert parity["within_5pct"] is True
