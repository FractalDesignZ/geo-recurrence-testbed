"""Stalk sheaf energy FO probe: helpers + verdict; no train."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.sheaf_energy import (  # noqa: E402
    QUANTILE_TAU,
    coboundary_energy,
    decide_verdict,
    energy_refuse_preds,
    example_sheaf_energy_from_states,
    mass_field,
    pearson_corr,
    quantile,
    scalar_dirichlet,
    sheaf_energy_definition_doc,
    target_align_field,
)
from reachability_gen.run_stalk_sheaf_energy_fo_probe import (  # noqa: E402
    AUROC_PARTIAL,
    AUROC_SEP,
    COLLATERAL_DROP,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    FOCUS_T,
    VERDICTS,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_SHEAF_ENERGY_FO_PROBE"
    assert FOCUS_T == 16
    assert QUANTILE_TAU == 0.90
    assert AUROC_SEP == 0.75
    assert AUROC_PARTIAL == 0.60
    assert COLLATERAL_DROP == 0.05
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert set(VERDICTS) == {
        "ENERGY_TRACKS_CERT",
        "ENERGY_PARTIAL",
        "ENERGY_NULL",
        "COLLATERAL_HARM",
    }


def test_no_train_flags():
    import reachability_gen.run_stalk_sheaf_energy_fo_probe as m

    doc = m.__doc__ or ""
    assert "eval-only" in doc.lower() or "MEASURE" in doc
    assert "science_open=false" in doc.lower()
    assert "sheaf train" in doc.lower() or "NO sheaf" in doc
    assert m.FOCUS_T == 16
    defs = sheaf_energy_definition_doc()
    assert defs["primary"]["learned_restriction"] is False
    assert defs["primary"]["uses_sheaf_train"] is False
    assert defs["primary"]["restriction_maps"] == "identity"
    assert defs["primary"]["full_sheaf_laplacian"] is False


def test_coboundary_identity_on_simple_edge():
    # Two nodes, one edge 0→1; H[0]=[1,0], H[1]=[0,0] → ||diff||²=1
    H = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    out = coboundary_energy(H, n=2, edges=[(0, 1)])
    assert abs(out["E_cob_raw"] - 1.0) < 1e-9
    assert abs(out["E_cob"] - 1.0) < 1e-9
    assert out["n_edges"] == 1.0


def test_coboundary_zero_when_constant_section():
    H = torch.ones(3, 4)
    out = coboundary_energy(H, n=3, edges=[(0, 1), (1, 2), (0, 2)])
    assert abs(out["E_cob"]) < 1e-9
    assert abs(out["E_cob_raw"]) < 1e-9


def test_scalar_dirichlet_and_fields():
    H = torch.tensor([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    f = mass_field(H, 3)
    assert abs(f[0] - 3.0) < 1e-9
    assert abs(f[1] - 4.0) < 1e-9
    assert abs(f[2] - 0.0) < 1e-9
    d = scalar_dirichlet(f, [(0, 1)])
    assert abs(d["E_raw"] - 1.0) < 1e-9
    a = target_align_field(H, 3, t=0)
    assert abs(a[0] - 1.0) < 1e-6
    assert abs(a[2] - 0.0) < 1e-9  # zero mass → 0


def test_example_sheaf_energy_keys():
    H = torch.randn(4, 8)
    edges = [(0, 1), (1, 2), (2, 3)]
    out = example_sheaf_energy_from_states(H, 4, edges, s=0, t=3)
    for k in ("E_cob", "E_dir", "E_align", "n_edges", "mass_t"):
        assert k in out
        assert out[k] == out[k]  # finite


def test_quantile_and_refuse():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert abs(quantile(xs, 0.9) - 8.1) < 1e-9 or abs(quantile(xs, 0.9) - 8.0) < 1.0
    preds = torch.tensor([1, 1, 0, 1])
    energies = [0.1, 5.0, 9.0, 0.2]
    out, refused = energy_refuse_preds(preds, energies, tau=1.0)
    assert list(out.tolist()) == [1, 0, 0, 1]
    assert refused == [False, True, False, False]


def test_pearson_corr_basic():
    assert abs(pearson_corr([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(pearson_corr([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9


def test_decide_verdict_tracks_partial_null_harm():
    harm = decide_verdict(
        auroc_fo=0.99,
        fo_killed=45,
        fo_total=45,
        rem22_killed=22,
        rem22_total=22,
        collateral_harm=True,
        collateral_reasons=["matched_drop"],
    )
    assert harm["verdict"] == "COLLATERAL_HARM"

    tracks = decide_verdict(
        auroc_fo=0.90,
        fo_killed=42,
        fo_total=45,
        rem22_killed=18,
        rem22_total=22,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert tracks["verdict"] == "ENERGY_TRACKS_CERT"

    partial = decide_verdict(
        auroc_fo=0.65,
        fo_killed=5,
        fo_total=45,
        rem22_killed=2,
        rem22_total=22,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert partial["verdict"] == "ENERGY_PARTIAL"

    null = decide_verdict(
        auroc_fo=0.50,
        fo_killed=1,
        fo_total=45,
        rem22_killed=0,
        rem22_total=22,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert null["verdict"] == "ENERGY_NULL"
