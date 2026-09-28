"""Tests for neutral_init bake-in removal + Gate A/C helpers (MEASURE)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.models.sheaf_infer_core import SheafInferCore  # noqa: E402
from reachability_gen.reach_cue_audit import (  # noqa: E402
    audit_reach_cues,
    build_degree_balanced_eval,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS  # noqa: E402
from reachability_gen.models.sheaf_infer_core import _verify_param_parity  # noqa: E402


def test_neutral_init_removes_bake_in():
    torch.manual_seed(0)
    m = SheafInferCore(
        d=64, T=6, mlp_expansion=12, max_nodes=64, max_T=16, neutral_init=True
    )
    last = m.edge_encoder[-1]
    assert abs(float(last.bias.item())) < 1e-6
    assert abs(float(m.absent_bias.item())) < 1e-6
    assert abs(float(m.head.bias[0].item())) < 1e-6
    assert abs(float(m.head.bias[1].item())) < 1e-6
    # Not the energy ±1 hardcode
    energy = m.head.weight.detach()[:, -1]
    assert not (
        abs(float(energy[0].item()) + 1.0) < 1e-6
        and abs(float(energy[1].item()) - 1.0) < 1e-6
    )
    parity = _verify_param_parity(m.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    assert parity["within_5pct"] is True
    assert m.neutral_init is True


def test_legacy_init_keeps_bake_in_for_repro():
    m = SheafInferCore(
        d=64, T=6, mlp_expansion=12, max_nodes=64, max_T=16, neutral_init=False
    )
    last = m.edge_encoder[-1]
    assert abs(float(last.bias.item()) - 4.0) < 1e-6
    assert abs(float(m.absent_bias.item()) + 4.0) < 1e-6
    assert abs(float(m.head.bias[0].item()) - 5.0) < 1e-6


def test_reach_cue_audit_degree_balanced_near_chance():
    ood_path = Path("data/covariate_matched_ood.jsonl")
    if not ood_path.exists():
        pytest.skip("matched OOD missing")
    from reachability_gen.overfit_ff import load_jsonl

    rows = load_jsonl(ood_path)
    bal = build_degree_balanced_eval(rows, seed=0)
    audit = audit_reach_cues(bal["rows"], ceiling=0.52)
    assert audit["pass_ceiling"] is True
    assert audit["max_acc"] <= 0.52
