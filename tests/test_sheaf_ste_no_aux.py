"""Tests for STE no-aux cycle plumbing (MEASURE)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.models.sheaf_infer_core import (  # noqa: E402
    SheafInferCore,
    _verify_param_parity,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS  # noqa: E402
from reachability_gen.run_sheaf_ste_no_aux import (  # noqa: E402
    CYCLE,
    DEFAULT_EPOCHS_STE,
    EDGE_RECON_WEIGHT,
    GATE_DETACH_DIFFUSION,
    GATE_MODE,
    PR9_MERGE_SHA,
)
from reachability_gen.train.sheaf_trainer import SheafTrainer  # noqa: E402


def test_ste_no_aux_knobs_locked():
    assert CYCLE == "CYCLE_SHEAF_STE_NO_AUX"
    assert EDGE_RECON_WEIGHT == 0.0
    assert GATE_DETACH_DIFFUSION is False
    assert GATE_MODE in ("ste", "gumbel")
    assert DEFAULT_EPOCHS_STE >= 60
    assert PR9_MERGE_SHA.startswith("2834256")


def test_model_accepts_ste_no_detach():
    torch.manual_seed(0)
    m = SheafInferCore(
        d=64,
        T=6,
        mlp_expansion=12,
        max_nodes=32,
        max_T=16,
        neutral_init=True,
        gate_detach_diffusion=False,
        gate_mode="ste",
    )
    assert m.gate_detach_diffusion is False
    assert m.gate_mode == "ste"
    trainer = SheafTrainer(m, edge_recon_weight=0.0)
    assert trainer.edge_recon_weight == 0.0
    parity = _verify_param_parity(m.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    assert parity["within_5pct"] is True


def test_gumbel_mode_constructs():
    m = SheafInferCore(
        d=64,
        T=6,
        mlp_expansion=12,
        max_nodes=32,
        max_T=16,
        neutral_init=True,
        gate_detach_diffusion=False,
        gate_mode="gumbel",
        gumbel_temp=1.0,
    )
    assert m.gate_mode == "gumbel"
