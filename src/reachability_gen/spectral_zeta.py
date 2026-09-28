"""Finite spectral zeta / heat of graph Laplacian on hard Â (MEASURE).

CYCLE_STALK_SPECTRAL_ZETA_FO_PROBE — discrete port of conformal spectral-zeta
FEM note (Dirichlet conformal-invariant; ζ sensitive to ρ) onto the hard
attention graph Â used by reachability certificates. No continuum FEM, no
mesh, no learned ρ, no sheaf train.

Primary::

    A_sym from directed Â (undirected skeleton)
    L_comb = D − A_sym
    ζ(s=2) = Σ_{λ > EPS} λ^{−2}   # omit zero modes

Secondary: normalized L, ρ=outdeg generalized eigenproblem, heat Tr(e^{−tL}).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch

from reachability_gen.sheaf_energy import (  # reuse refuse / stats helpers
    QUANTILE_TAU,
    energy_refuse_preds,
    pearson_corr,
    quantile,
)

EPS_ZERO = 1e-8
S_PRIMARY = 2.0
HEAT_TS = (0.5, 1.0)


def spectral_zeta_definition_doc() -> dict[str, Any]:
    """Machine-readable L/ζ/heat defs for artifact / prereg cite."""
    return {
        "port": (
            "discrete graph heat/ζ on hard Â — NOT continuum FEM; "
            "attachment conformal spectral-zeta note is motivation only"
        ),
        "primary": {
            "name": "zeta_comb_s2",
            "L": "combinatorial_symmetrized",
            "formula": (
                "A_sym[i,j]=1 if (i→j) or (j→i) in Â (i≠j); "
                "L=D−A_sym; ζ(2)=Σ_{λ>EPS} λ^{−2}; ρ=Id"
            ),
            "s": S_PRIMARY,
            "rho": "Id",
            "symmetrized": True,
            "zero_mode": f"omit λ ≤ {EPS_ZERO}",
            "learned_rho": False,
            "uses_bfs_checker": False,
            "uses_sheaf_train": False,
            "uses_model_states": False,
            "continuum_fem": False,
        },
        "secondary": [
            {
                "name": "zeta_norm_s2",
                "L": "normalized_symmetrized",
                "formula": "L=I−D^{+}½ A_sym D^{+}½; ζ(2) omit zeros",
            },
            {
                "name": "zeta_comb_rho_outdeg_s2",
                "rho": "max(outdeg_directed(i), 1)",
                "formula": "L_comb v = λ diag(ρ) v  (generalized / whitened)",
            },
            {
                "name": "heat_tr",
                "formula": "H(t)=Tr(e^{−t L_comb})=Σ exp(−t λ_i); t in {0.5,1.0}",
            },
        ],
        "refuse_threshold": {
            "name": "tau_matched_q90",
            "quantile": QUANTILE_TAU,
            "calibrate_on": "matched_ood",
            "policy": "if pred=YES and zeta_comb>tau → force pred=0",
            "score": "zeta_comb_s2",
        },
        "eps_zero": EPS_ZERO,
        "s_primary": S_PRIMARY,
        "heat_ts": list(HEAT_TS),
    }


def _symmetrize_adj(
    n: int,
    edges: Sequence[tuple[int, int]],
    *,
    edge_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Build undirected adjacency (max over directed; no self-loops)."""
    A = torch.zeros(n, n, dtype=torch.float64)
    if edge_weights is None:
        for u, v in edges:
            ui, vi = int(u), int(v)
            if ui == vi:
                continue
            if not (0 <= ui < n and 0 <= vi < n):
                raise ValueError(f"edge ({ui},{vi}) out of range for n={n}")
            A[ui, vi] = 1.0
            A[vi, ui] = 1.0
    else:
        if len(edge_weights) != len(edges):
            raise ValueError("edge_weights length must match edges")
        for (u, v), w in zip(edges, edge_weights):
            ui, vi = int(u), int(v)
            if ui == vi:
                continue
            if not (0 <= ui < n and 0 <= vi < n):
                raise ValueError(f"edge ({ui},{vi}) out of range for n={n}")
            ww = float(w)
            # keep max weight if both directions present
            if ww > float(A[ui, vi]):
                A[ui, vi] = ww
                A[vi, ui] = ww
    return A


def directed_outdeg(n: int, edges: Sequence[tuple[int, int]]) -> list[float]:
    """Directed out-degree on Â (self-loops ignored)."""
    deg = [0.0] * n
    for u, v in edges:
        ui, vi = int(u), int(v)
        if ui == vi:
            continue
        if not (0 <= ui < n and 0 <= vi < n):
            raise ValueError(f"edge ({ui},{vi}) out of range for n={n}")
        deg[ui] += 1.0
    return deg


def combinatorial_laplacian(A: torch.Tensor) -> torch.Tensor:
    """L = D − A for symmetric nonnegative A."""
    d = A.sum(dim=1)
    return torch.diag(d) - A


def normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
    """L_sym = I − D^{+}½ A D^{+}½ (isolates → row/col zero off-diag; diag 0)."""
    n = A.shape[0]
    d = A.sum(dim=1)
    d_inv_sqrt = torch.zeros_like(d)
    pos = d > EPS_ZERO
    d_inv_sqrt[pos] = d[pos].rsqrt()
    Dmh = torch.diag(d_inv_sqrt)
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    # isolates: leave L_ii=0 (already via Dmh row zero → (I−…)ii=1? wait)
    # Standard: isolated node has L_ii=0 for combinatorial; for normalized
    # convention often L_ii=0 as well (no degree). Use:
    # L = I − Dmh A Dmh, then zero out isolate diagonals back to 0.
    L = I - Dmh @ A @ Dmh
    # For isolates d_i=0: Dmh_i=0 ⇒ (Dmh A Dmh)_i*=0 ⇒ L_ii=1. Fix to 0
    # so they count as zero-modes of the disconnected component.
    iso = ~pos
    if bool(iso.any()):
        L[iso, :] = 0.0
        L[:, iso] = 0.0
    # Symmetrize numerically
    L = 0.5 * (L + L.T)
    return L


def _eigh_psd(L: torch.Tensor) -> torch.Tensor:
    """Ascending eigenvalues of symmetric L (float64)."""
    Lh = 0.5 * (L + L.T)
    evals, _ = torch.linalg.eigh(Lh)
    # numerical clamp tiny negatives from float noise
    return torch.clamp(evals, min=0.0)


def finite_spectral_zeta(
    evals: torch.Tensor,
    *,
    s: float = S_PRIMARY,
    eps_zero: float = EPS_ZERO,
) -> dict[str, float]:
    """ζ(s) = Σ_{λ>eps} λ^{−s}; omit zero modes."""
    pos = evals > eps_zero
    n_zero = int((~pos).sum().item())
    if not bool(pos.any()):
        return {
            "zeta": float("nan"),
            "n_zero": float(n_zero),
            "n_pos": 0.0,
            "lambda_min_pos": float("nan"),
            "lambda_max": float(evals.max().item()) if evals.numel() else float("nan"),
        }
    lam = evals[pos]
    z = torch.sum(lam.pow(-float(s)))
    return {
        "zeta": float(z.item()),
        "n_zero": float(n_zero),
        "n_pos": float(pos.sum().item()),
        "lambda_min_pos": float(lam.min().item()),
        "lambda_max": float(evals.max().item()),
    }


def heat_trace(
    evals: torch.Tensor,
    t: float,
) -> float:
    """H(t) = Σ exp(−t λ_i) including zeros."""
    return float(torch.sum(torch.exp(-float(t) * evals)).item())


def example_spectral_scores(
    n: int,
    edges: Sequence[tuple[int, int]],
    *,
    s: float = S_PRIMARY,
    eps_zero: float = EPS_ZERO,
    heat_ts: Sequence[float] = HEAT_TS,
) -> dict[str, float]:
    """Full per-example spectral score dict on hard Â."""
    A = _symmetrize_adj(n, edges)
    n_edges_sym = float((A > 0).sum().item() / 2.0)
    L_comb = combinatorial_laplacian(A)
    evals_c = _eigh_psd(L_comb)
    zc = finite_spectral_zeta(evals_c, s=s, eps_zero=eps_zero)

    L_norm = normalized_laplacian(A)
    evals_n = _eigh_psd(L_norm)
    zn = finite_spectral_zeta(evals_n, s=s, eps_zero=eps_zero)

    # ρ = max(outdeg, 1) generalized: whitened L̃ = ρ^{-1/2} L ρ^{-1/2}
    outdeg = directed_outdeg(n, edges)
    rho = torch.tensor([max(d, 1.0) for d in outdeg], dtype=torch.float64)
    rho_inv_sqrt = rho.rsqrt()
    R = torch.diag(rho_inv_sqrt)
    L_rho = R @ L_comb @ R
    evals_r = _eigh_psd(L_rho)
    zr = finite_spectral_zeta(evals_r, s=s, eps_zero=eps_zero)

    # edge-mass secondary: w_ij = (outdeg(i)+outdeg(j))/2 on directed endpoints
    weights = [
        (outdeg[int(u)] + outdeg[int(v)]) / 2.0
        if (0 <= int(u) < n and 0 <= int(v) < n)
        else 1.0
        for u, v in edges
    ]
    A_w = _symmetrize_adj(n, edges, edge_weights=weights)
    L_w = combinatorial_laplacian(A_w)
    evals_w = _eigh_psd(L_w)
    zw = finite_spectral_zeta(evals_w, s=s, eps_zero=eps_zero)

    out: dict[str, float] = {
        "zeta_comb": zc["zeta"],
        "zeta_norm": zn["zeta"],
        "zeta_rho_outdeg": zr["zeta"],
        "zeta_edge_mass": zw["zeta"],
        "n_zero_comb": zc["n_zero"],
        "n_zero_norm": zn["n_zero"],
        "n_pos_comb": zc["n_pos"],
        "lambda_min_pos_comb": zc["lambda_min_pos"],
        "lambda_max_comb": zc["lambda_max"],
        "n_nodes": float(n),
        "n_edges_sym": n_edges_sym,
        "n_edges_dir": float(len(edges)),
        "outdeg_mean": float(sum(outdeg) / max(n, 1)),
    }
    for t in heat_ts:
        key = f"heat_t{str(t).replace('.', '_')}"
        out[key] = heat_trace(evals_c, float(t))
    return out


def decide_verdict(
    *,
    auroc_fo: float,
    fo_killed: int,
    fo_total: int,
    rem22_killed: int,
    rem22_total: int,
    collateral_harm: bool,
    collateral_reasons: list[str],
    invalid: bool = False,
    invalid_reasons: Optional[list[str]] = None,
    auroc_sep: float = 0.75,
    auroc_partial: float = 0.60,
    fo_track_min: int = 40,
    rem_track_min: int = 16,
    fo_partial_min: int = 8,
    rem_partial_min: int = 4,
    fo_null_max: int = 2,
    rem_null_max: int = 2,
) -> dict[str, Any]:
    """LOCKED verdict enum; INVALID / COLLATERAL priority."""
    base = {
        "auroc_fo": auroc_fo,
        "fo_killed": fo_killed,
        "fo_total": fo_total,
        "rem22_killed": rem22_killed,
        "rem22_total": rem22_total,
    }
    if invalid:
        return {
            "verdict": "INVALID",
            "reasons": list(invalid_reasons or ["invalid"]),
            **base,
        }
    if collateral_harm:
        return {
            "verdict": "COLLATERAL_HARM",
            "reasons": list(collateral_reasons),
            **base,
        }
    tracks = (
        (auroc_fo == auroc_fo and auroc_fo >= auroc_sep)
        and fo_killed >= fo_track_min
        and rem22_killed >= rem_track_min
    )
    if tracks:
        return {
            "verdict": "ZETA_FO_CATCH",
            "reasons": [
                f"auroc>={auroc_sep}",
                f"fo_killed={fo_killed}>={fo_track_min}",
                f"rem22_killed={rem22_killed}>={rem_track_min}",
            ],
            **base,
        }
    partial = (
        (auroc_fo == auroc_fo and auroc_fo >= auroc_partial)
        or fo_killed >= fo_partial_min
        or rem22_killed >= rem_partial_min
    )
    if partial:
        return {
            "verdict": "ZETA_PARTIAL",
            "reasons": [
                f"auroc={auroc_fo}",
                f"fo_killed={fo_killed}",
                f"rem22_killed={rem22_killed}",
            ],
            **base,
        }
    nullish = (
        fo_killed <= fo_null_max
        and rem22_killed <= rem_null_max
        and (
            auroc_fo != auroc_fo
            or (0.40 <= auroc_fo <= 0.60)
            or auroc_fo < auroc_partial
        )
    )
    if nullish:
        return {
            "verdict": "ZETA_NULL",
            "reasons": [
                f"auroc={auroc_fo}",
                f"fo_killed={fo_killed}<={fo_null_max}",
                f"rem22_killed={rem22_killed}<={rem_null_max}",
            ],
            **base,
        }
    return {
        "verdict": "ZETA_PARTIAL",
        "reasons": ["fallback_partial", f"auroc={auroc_fo}"],
        **base,
    }


# re-export helpers used by runner
__all__ = [
    "EPS_ZERO",
    "HEAT_TS",
    "QUANTILE_TAU",
    "S_PRIMARY",
    "combinatorial_laplacian",
    "decide_verdict",
    "directed_outdeg",
    "energy_refuse_preds",
    "example_spectral_scores",
    "finite_spectral_zeta",
    "heat_trace",
    "normalized_laplacian",
    "pearson_corr",
    "quantile",
    "spectral_zeta_definition_doc",
]
