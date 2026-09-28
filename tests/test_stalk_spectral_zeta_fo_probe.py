"""Stalk spectral zeta FO probe: helpers + verdict; no train."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.spectral_zeta import (  # noqa: E402
    EPS_ZERO,
    HEAT_TS,
    QUANTILE_TAU,
    S_PRIMARY,
    combinatorial_laplacian,
    decide_verdict,
    directed_outdeg,
    energy_refuse_preds,
    example_spectral_scores,
    finite_spectral_zeta,
    heat_trace,
    normalized_laplacian,
    pearson_corr,
    quantile,
    spectral_zeta_definition_doc,
)
from reachability_gen.run_stalk_spectral_zeta_fo_probe import (  # noqa: E402
    AUROC_PARTIAL,
    AUROC_SEP,
    COLLATERAL_DROP,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    FOCUS_T,
    PRIMARY_SCORE,
    VERDICTS,
)
from reachability_gen.run_stalk_seed_ensemble import PRIMARY_AGG  # noqa: E402


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_SPECTRAL_ZETA_FO_PROBE"
    assert FOCUS_T == 16
    assert QUANTILE_TAU == 0.90
    assert S_PRIMARY == 2.0
    assert AUROC_SEP == 0.75
    assert AUROC_PARTIAL == 0.60
    assert COLLATERAL_DROP == 0.05
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert PRIMARY_SCORE == "zeta_comb"
    assert PRIMARY_AGG == "prob_mean"
    assert set(VERDICTS) == {
        "ZETA_FO_CATCH",
        "ZETA_PARTIAL",
        "ZETA_NULL",
        "COLLATERAL_HARM",
        "INVALID",
    }
    assert HEAT_TS == (0.5, 1.0)


def test_no_train_flags():
    import reachability_gen.run_stalk_spectral_zeta_fo_probe as m

    doc = m.__doc__ or ""
    assert "eval-only" in doc.lower() or "MEASURE" in doc
    assert "science_open=false" in doc.lower()
    assert "sheaf" in doc.lower()
    assert m.FOCUS_T == 16
    defs = spectral_zeta_definition_doc()
    assert defs["primary"]["learned_rho"] is False
    assert defs["primary"]["uses_sheaf_train"] is False
    assert defs["primary"]["continuum_fem"] is False
    assert defs["primary"]["rho"] == "Id"
    assert defs["primary"]["s"] == 2.0
    assert defs["primary"]["uses_model_states"] is False


def test_path_graph_combinatorial_zeta():
    # Path 0—1—2: L = [[1,-1,0],[-1,2,-1],[0,-1,1]]; known evals 0,1,3
    edges = [(0, 1), (1, 2)]
    out = example_spectral_scores(3, edges)
    assert out["n_zero_comb"] == 1.0
    # ζ(2) = 1^{-2} + 3^{-2} = 1 + 1/9 = 10/9
    assert abs(out["zeta_comb"] - (1.0 + 1.0 / 9.0)) < 1e-6
    import math
    expected_h05 = sum(math.exp(-0.5 * lam) for lam in (0.0, 1.0, 3.0))
    assert abs(out["heat_t0_5"] - expected_h05) < 1e-5
    expected_h1 = sum(math.exp(-1.0 * lam) for lam in (0.0, 1.0, 3.0))
    assert abs(out["heat_t1_0"] - expected_h1) < 1e-5


def test_complete_graph_one_zero_mode():
    # K3 directed as cycle+chords → symmetrized complete
    edges = [(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)]
    out = example_spectral_scores(3, edges)
    assert out["n_zero_comb"] == 1.0
    assert out["zeta_comb"] == out["zeta_comb"]
    assert out["n_edges_sym"] == 3.0


def test_isolated_nodes_zero_modes():
    edges = [(0, 1)]  # node 2 isolated
    out = example_spectral_scores(3, edges)
    assert out["n_zero_comb"] >= 2.0  # one component + isolate


def test_finite_zeta_omits_zeros():
    evals = torch.tensor([0.0, 0.0, 1.0, 4.0], dtype=torch.float64)
    z = finite_spectral_zeta(evals, s=2.0, eps_zero=EPS_ZERO)
    assert abs(z["zeta"] - (1.0 + 1.0 / 16.0)) < 1e-9
    assert z["n_zero"] == 2.0


def test_normalized_laplacian_symmetric():
    A = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    L = normalized_laplacian(A)
    assert torch.allclose(L, L.T)
    evals = torch.linalg.eigh(L).eigenvalues
    assert float(evals[0]) > -1e-8


def test_outdeg_and_refuse():
    deg = directed_outdeg(3, [(0, 1), (0, 2), (1, 2)])
    assert deg == [2.0, 1.0, 0.0]
    preds = torch.tensor([1, 1, 0, 1])
    energies = [0.1, 5.0, 9.0, 0.2]
    out, refused = energy_refuse_preds(preds, energies, tau=1.0)
    assert list(out.tolist()) == [1, 0, 0, 1]
    assert refused == [False, True, False, False]
    assert abs(pearson_corr([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert quantile(xs, 0.9) == quantile(xs, 0.9)


def test_decide_verdict_all_labels():
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

    inv = decide_verdict(
        auroc_fo=0.99,
        fo_killed=45,
        fo_total=45,
        rem22_killed=22,
        rem22_total=22,
        collateral_harm=False,
        collateral_reasons=[],
        invalid=True,
        invalid_reasons=["cite_floors"],
    )
    assert inv["verdict"] == "INVALID"

    catch = decide_verdict(
        auroc_fo=0.90,
        fo_killed=42,
        fo_total=45,
        rem22_killed=18,
        rem22_total=22,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert catch["verdict"] == "ZETA_FO_CATCH"

    partial = decide_verdict(
        auroc_fo=0.65,
        fo_killed=5,
        fo_total=45,
        rem22_killed=2,
        rem22_total=22,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert partial["verdict"] == "ZETA_PARTIAL"

    null = decide_verdict(
        auroc_fo=0.50,
        fo_killed=1,
        fo_total=45,
        rem22_killed=0,
        rem22_total=22,
        collateral_harm=False,
        collateral_reasons=[],
    )
    assert null["verdict"] == "ZETA_NULL"


def test_combinatorial_path_evals():
    A = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    L = combinatorial_laplacian(A)
    evals = torch.linalg.eigh(L).eigenvalues
    expected = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    assert torch.allclose(evals, expected, atol=1e-6)
    assert abs(heat_trace(evals, 1.0) - sum(float(__import__("math").exp(-lam)) for lam in [0.0, 1.0, 3.0])) < 1e-6
