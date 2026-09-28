"""Orientation / in–out collapse probe helpers (MEASURE).

CYCLE_STALK_ORIENTATION_COLLAPSE_PROBE — Dir-GNN-*style* probe, not architecture.

FractalCore has no explicit in/out channels: directed adjacency-masked MHA
attends in-neighbors only. Best-effort proxy = directed incidence × final
hidden states from a frozen forward pass (`return_states=True`).

Definitions (LOCKED in prereg)::

    h_in[i]  = mean_{j: j→i} H[j]   (0 if empty)
    h_out[i] = mean_{j: i→j} H[j]   (0 if empty)

    cos_orient = cos(h_in, h_out)     # nan if either near-zero
    l2_orient  = ||h_in - h_out||_2
    mass_ratio = ||h_in|| / (||h_in|| + ||h_out|| + eps)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


EPS = 1e-8


def orientation_definition_doc() -> dict[str, Any]:
    """Machine-readable proxy defs for artifact / prereg cite."""
    return {
        "scope": "directed_incidence_x_final_hidden",
        "explicit_in_out_channels": False,
        "attention_weights_used": False,
        "architecture_note": (
            "FractalCore directed A-mask attends in-neighbors only; "
            "no Dir-GNN dual aggregators. h_out is post-hoc incidence×H, "
            "not attended mass."
        ),
        "primary_slot": "target",
        "secondary_slot": "source",
        "ensemble_agg": "mean_over_members",
        "eps": EPS,
    }


def build_in_out_index(
    n: int,
    edges: Sequence[tuple[int, int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Return (N_in, N_out) adjacency lists for nodes ``0..n-1``."""
    n_in: list[list[int]] = [[] for _ in range(n)]
    n_out: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        ui, vi = int(u), int(v)
        if not (0 <= ui < n and 0 <= vi < n):
            raise ValueError(f"edge ({ui},{vi}) out of range for n={n}")
        n_out[ui].append(vi)
        n_in[vi].append(ui)
    return n_in, n_out


def neighbor_mean(
    H: torch.Tensor,
    neighbors: Sequence[int],
) -> torch.Tensor:
    """Mean of ``H[j]`` over neighbors; zeros if empty. ``H`` is ``[n, d]``."""
    if H.dim() != 2:
        raise ValueError(f"H must be [n,d], got {tuple(H.shape)}")
    if not neighbors:
        return torch.zeros(H.shape[1], dtype=H.dtype, device=H.device)
    idx = torch.tensor(list(neighbors), dtype=torch.long, device=H.device)
    return H.index_select(0, idx).mean(dim=0)


def slot_orientation_metrics(
    H: torch.Tensor,
    n_in: Sequence[Sequence[int]],
    n_out: Sequence[Sequence[int]],
    slot: int,
    *,
    eps: float = EPS,
) -> dict[str, float]:
    """Orientation scalars at one node slot from incidence×H."""
    if not (0 <= slot < H.shape[0]):
        raise ValueError(f"slot={slot} out of range for n={H.shape[0]}")
    h_in = neighbor_mean(H, n_in[slot])
    h_out = neighbor_mean(H, n_out[slot])
    mass_in = float(torch.linalg.vector_norm(h_in).item())
    mass_out = float(torch.linalg.vector_norm(h_out).item())
    l2 = float(torch.linalg.vector_norm(h_in - h_out).item())
    deg_in = int(len(n_in[slot]))
    deg_out = int(len(n_out[slot]))
    if mass_in < eps or mass_out < eps:
        cos = float("nan")
    else:
        cos = float(
            torch.dot(h_in, h_out).item() / (mass_in * mass_out)
        )
        # numerical clamp
        cos = max(-1.0, min(1.0, cos))
    return {
        "cos_orient": cos,
        "l2_orient": l2,
        "mass_in": mass_in,
        "mass_out": mass_out,
        "mass_ratio": mass_in / (mass_in + mass_out + eps),
        "deg_in": float(deg_in),
        "deg_out": float(deg_out),
        "empty_in": float(deg_in == 0),
        "empty_out": float(deg_out == 0),
    }


def example_orientation_from_states(
    H: torch.Tensor,
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
    *,
    eps: float = EPS,
) -> dict[str, float]:
    """Full per-example orientation dict (target primary + source secondary).

    ``H`` may be ``[M, d]`` padded; only ``H[:n]`` is used.
    """
    if H.dim() != 2:
        raise ValueError(f"H must be [M,d], got {tuple(H.shape)}")
    if n > H.shape[0]:
        raise ValueError(f"n={n} > H.shape[0]={H.shape[0]}")
    Hn = H[:n]
    n_in, n_out = build_in_out_index(n, edges)
    mt = slot_orientation_metrics(Hn, n_in, n_out, int(t), eps=eps)
    ms = slot_orientation_metrics(Hn, n_in, n_out, int(s), eps=eps)
    out: dict[str, float] = {}
    for k, v in mt.items():
        out[f"{k}_t"] = v
    for k, v in ms.items():
        out[f"{k}_s"] = v
    return out


def mean_metrics(dicts: Sequence[dict[str, float]]) -> dict[str, float]:
    """Elementwise nanmean over member dicts (same keys)."""
    if not dicts:
        return {}
    keys = list(dicts[0].keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts]
        finite = [v for v in vals if v == v]  # not nan
        if not finite:
            out[k] = float("nan")
        else:
            out[k] = sum(finite) / len(finite)
    return out


def median(xs: Sequence[float]) -> float:
    vals = sorted(v for v in xs if v == v)
    n = len(vals)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return 0.5 * (vals[mid - 1] + vals[mid])


def quartile(xs: Sequence[float], q: float) -> float:
    """Simple inclusive quantile; ``q`` in [0,1]."""
    vals = sorted(v for v in xs if v == v)
    n = len(vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(vals[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def summary_numeric(xs: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in xs if v == v]
    n_all = len(xs)
    n_nan = n_all - len(vals)
    if not vals:
        return {
            "n": n_all,
            "n_finite": 0,
            "nan_rate": 1.0 if n_all else float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": n_all,
        "n_finite": len(vals),
        "nan_rate": float(n_nan) / float(n_all) if n_all else float("nan"),
        "mean": sum(vals) / len(vals),
        "median": median(vals),
        "q25": quartile(vals, 0.25),
        "q75": quartile(vals, 0.75),
        "min": min(vals),
        "max": max(vals),
    }


def auroc_binary(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC for binary labels {0,1}; higher score → class 1.

    Mann–Whitney / rank form. Returns nan if a class is empty or all scores nan.
    Ties: average ranks.
    """
    pairs = [
        (float(s), int(y))
        for s, y in zip(scores, labels)
        if s == s  # finite score
    ]
    if not pairs:
        return float("nan")
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # sort by score ascending; assign average ranks for ties
    pairs.sort(key=lambda p: p[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        # ranks are 1-indexed
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(ranks[k] for k, (_, y) in enumerate(pairs) if y == 1)
    # U = sum_pos_ranks - n_pos*(n_pos+1)/2
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def median_outside_iqr(med: float, q25: float, q75: float) -> bool:
    if med != med or q25 != q25 or q75 != q75:
        return False
    return med < q25 or med > q75


def decide_verdict(
    *,
    auroc_cos_fo: float,
    auroc_l2_fo: float,
    auroc_cos_rem: float,
    auroc_l2_rem: float,
    rem_outside_ok_cos: bool,
    rem_outside_ok_l2: bool,
    nan_rate_rem: float,
    nan_rate_ok: float,
    auroc_sep: float = 0.75,
    auroc_partial: float = 0.60,
    nan_cap: float = 0.50,
) -> dict[str, Any]:
    """LOCKED verdict enum for orientation collapse probe."""
    reasons: list[str] = []
    if nan_rate_rem > nan_cap or nan_rate_ok > nan_cap:
        return {
            "verdict": "INCONCLUSIVE_ARCH",
            "reasons": [
                f"nan_rate_rem={nan_rate_rem:.3f}",
                f"nan_rate_ok={nan_rate_ok:.3f}",
                f"cap={nan_cap}",
                "proxy_unusable_on_hypothesis_set",
            ],
            "auroc_cos_fo": auroc_cos_fo,
            "auroc_l2_fo": auroc_l2_fo,
            "auroc_cos_rem": auroc_cos_rem,
            "auroc_l2_rem": auroc_l2_rem,
            "rem_outside_ok_cos": rem_outside_ok_cos,
            "rem_outside_ok_l2": rem_outside_ok_l2,
        }

    best_fo = max(
        auroc_cos_fo if auroc_cos_fo == auroc_cos_fo else 0.0,
        auroc_l2_fo if auroc_l2_fo == auroc_l2_fo else 0.0,
    )
    rem_out = rem_outside_ok_cos or rem_outside_ok_l2
    # which metric drives SEPARATES: need AUROC≥sep AND rem outside on THAT metric
    sep_cos = (
        auroc_cos_fo == auroc_cos_fo
        and auroc_cos_fo >= auroc_sep
        and rem_outside_ok_cos
    )
    sep_l2 = (
        auroc_l2_fo == auroc_l2_fo
        and auroc_l2_fo >= auroc_sep
        and rem_outside_ok_l2
    )
    if sep_cos or sep_l2:
        if sep_cos:
            reasons.append(f"auroc_cos_fo={auroc_cos_fo:.3f}>={auroc_sep}")
            reasons.append("rem_outside_ok_cos")
        if sep_l2:
            reasons.append(f"auroc_l2_fo={auroc_l2_fo:.3f}>={auroc_sep}")
            reasons.append("rem_outside_ok_l2")
        return {
            "verdict": "ORIENT_SEPARATES_FO",
            "reasons": reasons,
            "auroc_cos_fo": auroc_cos_fo,
            "auroc_l2_fo": auroc_l2_fo,
            "auroc_cos_rem": auroc_cos_rem,
            "auroc_l2_rem": auroc_l2_rem,
            "rem_outside_ok_cos": rem_outside_ok_cos,
            "rem_outside_ok_l2": rem_outside_ok_l2,
        }

    partial = best_fo >= auroc_partial or rem_out
    if partial:
        reasons.append(f"best_fo_auroc={best_fo:.3f}")
        reasons.append(f"rem_outside={rem_out}")
        reasons.append("not_ORIENT_SEPARATES_FO")
        return {
            "verdict": "ORIENT_PARTIAL",
            "reasons": reasons,
            "auroc_cos_fo": auroc_cos_fo,
            "auroc_l2_fo": auroc_l2_fo,
            "auroc_cos_rem": auroc_cos_rem,
            "auroc_l2_rem": auroc_l2_rem,
            "rem_outside_ok_cos": rem_outside_ok_cos,
            "rem_outside_ok_l2": rem_outside_ok_l2,
        }

    # NULL: weak AUROC and rem inside OK IQR
    in_null_band = (
        (auroc_cos_fo != auroc_cos_fo or 0.40 <= auroc_cos_fo <= 0.60)
        and (auroc_l2_fo != auroc_l2_fo or 0.40 <= auroc_l2_fo <= 0.60)
    )
    if in_null_band and not rem_out:
        return {
            "verdict": "ORIENT_NULL",
            "reasons": [
                f"auroc_cos_fo={auroc_cos_fo}",
                f"auroc_l2_fo={auroc_l2_fo}",
                "rem_inside_ok_iqr",
                "falsifies_orientation_collapse_explanatory_lever_under_proxy",
            ],
            "auroc_cos_fo": auroc_cos_fo,
            "auroc_l2_fo": auroc_l2_fo,
            "auroc_cos_rem": auroc_cos_rem,
            "auroc_l2_rem": auroc_l2_rem,
            "rem_outside_ok_cos": rem_outside_ok_cos,
            "rem_outside_ok_l2": rem_outside_ok_l2,
        }

    # fallthrough → PARTIAL (e.g. AUROC below partial but weird band)
    return {
        "verdict": "ORIENT_PARTIAL",
        "reasons": [
            f"best_fo_auroc={best_fo:.3f}",
            f"rem_outside={rem_out}",
            "fallthrough_not_NULL_not_SEPARATES",
        ],
        "auroc_cos_fo": auroc_cos_fo,
        "auroc_l2_fo": auroc_l2_fo,
        "auroc_cos_rem": auroc_cos_rem,
        "auroc_l2_rem": auroc_l2_rem,
        "rem_outside_ok_cos": rem_outside_ok_cos,
        "rem_outside_ok_l2": rem_outside_ok_l2,
    }
