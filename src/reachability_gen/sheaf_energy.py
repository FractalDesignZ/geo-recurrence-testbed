"""Identity-restriction sheaf / coboundary Dirichlet energy (MEASURE).

CYCLE_STALK_SHEAF_ENERGY_FO_PROBE — instrument Kant/sheaf A2 (consistency)
and A4 (Dirichlet regularity) on sealed stalk final_states under hard Â
without learned restriction maps and without sheaf training.

Primary::

    E_cob(H; Â) = Σ_{(u→v)∈Â} ||H[u] − H[v]||₂²
    Ẽ_cob       = E_cob / max(|E|, 1)

Secondary scalars::

    f(i)=||H[i]||₂ → E_dir;  a(i)=cos(H[i],H[t]) → E_align

Full sheaf Laplacian with learned ρ is unavailable this cycle
(no restriction-map learning; sheaf unsupervised NOT opened).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch

EPS = 1e-8
QUANTILE_TAU = 0.90  # matched-OOD threshold for energy_refuse


def sheaf_energy_definition_doc() -> dict[str, Any]:
    """Machine-readable energy defs for artifact / prereg cite."""
    return {
        "primary": {
            "name": "E_cob",
            "axiom": "A2_consistency_identity_restriction",
            "formula": (
                "E_cob = Σ_{(u→v)∈Â} ||H[u]−H[v]||₂² / max(|E|,1); "
                "H = FractalCore final_states[:n]; ρ=Id"
            ),
            "restriction_maps": "identity",
            "learned_restriction": False,
            "uses_bfs_checker": False,
            "uses_sheaf_train": False,
            "full_sheaf_laplacian": False,
            "note": (
                "Honest proxy: sheaf-Laplacian quadratic form under Id "
                "restrictions; full sheaf Laplacian unavailable without "
                "restriction-map learning (STOP / INVALID sheaf cells stay closed)"
            ),
        },
        "secondary": [
            {
                "name": "E_dir",
                "axiom": "A4_dirichlet_mass_field",
                "formula": "E_dir = Σ_{(u→v)} (||H[u]||−||H[v]||)² / max(|E|,1)",
            },
            {
                "name": "E_align",
                "axiom": "A4_dirichlet_target_alignment",
                "formula": (
                    "E_align = Σ_{(u→v)} (cos(H[u],H[t])−cos(H[v],H[t]))² "
                    "/ max(|E|,1); cos→0 if near-zero"
                ),
            },
        ],
        "ensemble_agg": "mean_over_members",
        "refuse_threshold": {
            "name": "tau_matched_q90",
            "quantile": QUANTILE_TAU,
            "calibrate_on": "matched_ood",
            "policy": "if pred=YES and E_cob>tau → force pred=0",
        },
        "eps": EPS,
    }


def coboundary_energy(
    H: torch.Tensor,
    n: int,
    edges: Sequence[tuple[int, int]],
) -> dict[str, float]:
    """Vector coboundary energy under identity restriction.

    ``H`` may be ``[M, d]`` padded; only ``H[:n]`` is used.
    """
    if H.dim() != 2:
        raise ValueError(f"H must be [M,d], got {tuple(H.shape)}")
    if n > H.shape[0]:
        raise ValueError(f"n={n} > H.shape[0]={H.shape[0]}")
    Hn = H[:n]
    e_count = 0
    total = Hn.new_zeros(())
    for u, v in edges:
        ui, vi = int(u), int(v)
        if not (0 <= ui < n and 0 <= vi < n):
            raise ValueError(f"edge ({ui},{vi}) out of range for n={n}")
        diff = Hn[ui] - Hn[vi]
        total = total + torch.dot(diff, diff)
        e_count += 1
    e_cob = float(total.item())
    denom = float(max(e_count, 1))
    return {
        "E_cob_raw": e_cob,
        "E_cob": e_cob / denom,
        "n_edges": float(e_count),
        "n_nodes": float(n),
    }


def scalar_dirichlet(
    values: Sequence[float],
    edges: Sequence[tuple[int, int]],
) -> dict[str, float]:
    """Dirichlet energy of a scalar field on directed edges."""
    n = len(values)
    e_count = 0
    total = 0.0
    for u, v in edges:
        ui, vi = int(u), int(v)
        if not (0 <= ui < n and 0 <= vi < n):
            raise ValueError(f"edge ({ui},{vi}) out of range for n={n}")
        d = float(values[ui]) - float(values[vi])
        total += d * d
        e_count += 1
    denom = float(max(e_count, 1))
    return {
        "E_raw": total,
        "E_mean": total / denom,
        "n_edges": float(e_count),
    }


def mass_field(H: torch.Tensor, n: int) -> list[float]:
    """f(i) = ||H[i]||₂ for i in 0..n-1."""
    Hn = H[:n]
    norms = torch.linalg.vector_norm(Hn, dim=-1)
    return [float(x) for x in norms.tolist()]


def target_align_field(
    H: torch.Tensor,
    n: int,
    t: int,
    *,
    eps: float = EPS,
) -> list[float]:
    """a(i) = cos(H[i], H[t]); 0 if either near-zero."""
    if not (0 <= t < n):
        raise ValueError(f"t={t} out of range for n={n}")
    Hn = H[:n]
    ht = Hn[t]
    nt = float(torch.linalg.vector_norm(ht).item())
    out: list[float] = []
    for i in range(n):
        hi = Hn[i]
        ni = float(torch.linalg.vector_norm(hi).item())
        if ni < eps or nt < eps:
            out.append(0.0)
        else:
            c = float(torch.dot(hi, ht).item()) / (ni * nt)
            out.append(max(-1.0, min(1.0, c)))
    return out


def example_sheaf_energy_from_states(
    H: torch.Tensor,
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
    *,
    eps: float = EPS,
) -> dict[str, float]:
    """Full per-example sheaf energy dict from final_states."""
    del s  # reserved for future source-conditioned fields
    cob = coboundary_energy(H, n, edges)
    f = mass_field(H, n)
    a = target_align_field(H, n, int(t), eps=eps)
    d_mass = scalar_dirichlet(f, edges)
    d_align = scalar_dirichlet(a, edges)
    return {
        "E_cob": cob["E_cob"],
        "E_cob_raw": cob["E_cob_raw"],
        "E_dir": d_mass["E_mean"],
        "E_dir_raw": d_mass["E_raw"],
        "E_align": d_align["E_mean"],
        "E_align_raw": d_align["E_raw"],
        "n_edges": cob["n_edges"],
        "n_nodes": cob["n_nodes"],
        "mass_mean": sum(f) / max(len(f), 1),
        "mass_t": f[int(t)] if 0 <= int(t) < len(f) else float("nan"),
        "align_mean": sum(a) / max(len(a), 1),
    }


def mean_metrics(dicts: Sequence[dict[str, float]]) -> dict[str, float]:
    """Elementwise nanmean over member dicts."""
    if not dicts:
        return {}
    keys = list(dicts[0].keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts]
        finite = [v for v in vals if v == v]
        if not finite:
            out[k] = float("nan")
        else:
            out[k] = sum(finite) / len(finite)
    return out


def quantile(xs: Sequence[float], q: float) -> float:
    """Inclusive linear quantile; ``q`` in [0, 1]."""
    vals = sorted(v for v in xs if v == v)
    n = len(vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(vals[0])
    q = max(0.0, min(1.0, float(q)))
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def energy_refuse_preds(
    ens_preds: torch.Tensor,
    energies: Sequence[float],
    *,
    tau: float,
) -> tuple[torch.Tensor, list[bool]]:
    """Fail-closed refuse: YES + E>tau → force NO.

    Returns (preds_refused, refused_mask).
    """
    out = ens_preds.clone()
    refused: list[bool] = []
    for i, e in enumerate(energies):
        yes = int(out[i].item()) == 1
        hit = yes and (e == e) and float(e) > float(tau)
        refused.append(bool(hit))
        if hit:
            out[i] = 0
    return out, refused


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r; nan if degenerate."""
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x == x and y == y]
    n = len(pairs)
    if n < 2:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    dx = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    dy = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if dx < EPS or dy < EPS:
        return float("nan")
    return float(num / (dx * dy))


def decide_verdict(
    *,
    auroc_fo: float,
    fo_killed: int,
    fo_total: int,
    rem22_killed: int,
    rem22_total: int,
    collateral_harm: bool,
    collateral_reasons: list[str],
    auroc_sep: float = 0.75,
    auroc_partial: float = 0.60,
    fo_track_min: int = 40,
    rem_track_min: int = 16,
    fo_partial_min: int = 8,
    rem_partial_min: int = 4,
    fo_null_max: int = 2,
    rem_null_max: int = 2,
) -> dict[str, Any]:
    """LOCKED verdict enum with collateral priority."""
    base = {
        "auroc_fo": auroc_fo,
        "fo_killed": fo_killed,
        "fo_total": fo_total,
        "rem22_killed": rem22_killed,
        "rem22_total": rem22_total,
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
            "verdict": "ENERGY_TRACKS_CERT",
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
            "verdict": "ENERGY_PARTIAL",
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
            "verdict": "ENERGY_NULL",
            "reasons": [
                f"auroc={auroc_fo}",
                f"fo_killed={fo_killed}<={fo_null_max}",
                f"rem22_killed={rem22_killed}<={rem_null_max}",
            ],
            **base,
        }
    return {
        "verdict": "ENERGY_PARTIAL",
        "reasons": ["fallback_partial", f"auroc={auroc_fo}"],
        **base,
    }
