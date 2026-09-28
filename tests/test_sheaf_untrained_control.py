"""Standing untrained-control: bake-in detection + agreement vs sealed ckpt."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.models.sheaf_infer_core import SheafInferCore  # noqa: E402
from reachability_gen.run_sheaf_untrained_control import (  # noqa: E402
    document_init_bake_in,
    run_audit,
)

CKPT = Path("artifacts/sheaf_infer_gate1_best.pt")
OOD = Path("data/covariate_matched_ood.jsonl")


def test_init_bake_in_documents_reachability_oracle():
    m = SheafInferCore(d=64, T=6, mlp_expansion=12)
    doc = document_init_bake_in(m)
    assert doc["edge_encoder_last_weight_zeros"] is True
    assert doc["edge_encoder_last_bias"] == 4.0
    assert doc["absent_bias"] == -4.0
    assert doc["gate_theta"] == 0.5
    assert doc["residual_alpha"] == 1.0
    assert doc["head_energy_col"] == [-1.0, 1.0]
    assert doc["head_bias"] == [5.0, -5.0]


@pytest.mark.skipif(not CKPT.exists() or not OOD.exists(), reason="sealed artifacts missing")
def test_untrained_matches_sealed_matched_ood_t16(tmp_path):
    out = tmp_path / "audit.json"
    report = run_audit(ckpt_path=CKPT, out_path=out, seed=0)
    m16 = report["datasets"]["matched_ood"]["by_T"]["16"]
    assert m16["untrained"]["overall_acc"] == pytest.approx(1.0)
    assert m16["untrained"]["hard_neg_acc"] == pytest.approx(1.0)
    assert m16["K_strata"]["16"]["untrained_acc"] == pytest.approx(1.0)
    assert m16["prediction_agreement"] == pytest.approx(1.0)
    assert report["verdict"] == "SEALS_COMPROMISED_INIT_BAKE_IN"
    assert report["seals_invalidated"] is True
