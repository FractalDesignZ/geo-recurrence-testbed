"""Stalk untrained-control: mid/chance vs sealed; no silent seal revoke."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.models.fractal_core import FractalCore  # noqa: E402
from reachability_gen.run_stalk_untrained_control import (  # noqa: E402
    document_stalk_init,
    run_audit,
)

CKPT = Path("artifacts/fractal_core_stalk_gate1_best.pt")
OOD = Path("data/covariate_matched_ood.jsonl")


def test_stalk_init_is_not_sheaf_oracle():
    m = FractalCore(d=64, T=6, mlp_expansion=10)
    doc = document_stalk_init(m)
    assert doc["stalk_proj_eye"] is True
    assert doc["probe_proj_eye"] is True
    assert doc["architecture"]["broadcast_c"] is False
    assert doc["architecture"]["soft_ACT"] is False
    assert doc["science_open"] is False


@pytest.mark.skipif(not CKPT.exists() or not OOD.exists(), reason="sealed artifacts missing")
def test_untrained_not_bake_in_vs_sealed(tmp_path):
    out = tmp_path / "stalk_audit.json"
    report = run_audit(ckpt_path=CKPT, ood_path=OOD, out_path=out, seed=0)
    m16 = report["by_T"]["16"]
    # Mid-band untrained — NOT sealed 1.0 bake-in
    assert m16["untrained"]["overall_acc"] < 0.75
    assert m16["trained"]["overall_acc"] > 0.9
    assert m16["prediction_agreement"] < 0.95
    assert report["seals_invalidated"] is False
    assert report["verdict"] == "OPEN_STILL_CONTINGENT_NEEDS_MULTI_SEED"
    assert report["science_open"] is False
