"""Explicit energy scoring for stalk member selection (MEASURE).

CYCLE_STALK_ENERGY_SELECTOR — separate generation (#22 ens members) from
selection via E(Â, outputs). Not a trained head. Not multi-hyp JS. Not bag.

Cert-free (primary)::

    E_sound(m) = 1[out_s==0 ∧ s≠t ∧ pred_m==1] + 1[in_t==0 ∧ s≠t ∧ pred_m==1]
    E_cone(m)  = 1[pred_m==1] / (1 + out_s + in_t)
    E_conf(m)  = −log(p_m + ε)
    E_free(m)  = α·E_sound + β·E_cone + γ·E_conf

Cert-tied (secondary / cert-energy)::

    E_disagree(m) = 1[(pred==1 ∧ ¬reach) ∨ (pred==0 ∧ reach)]
    E_cert(m)     = α·E_disagree + γ·E_conf

Locked: α=10, β=1, γ=0.1, ε=1e-8, τ=1.0 (energy_weighted).
No orientation metrics. Checker BFS only inside E_cert (post-hoc).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from reachability_gen.encode import parse_instance
from reachability_gen.graph import reachable_bfs

# Locked prereg weights
ALPHA_SOUND = 10.0
BETA_CONE = 1.0
GAMMA_CONF = 0.1
EPS_CONF = 1e-8
TAU_WEIGHTED = 1.0


def energy_definition_doc() -> dict[str, Any]:
    """Machine-readable energy defs for artifact / prereg cite."""
    return {
        "primary": {
            "name": "E_free",
            "formula": (
                "E_free = α·E_sound + β·E_cone + γ·(−log(p+ε)); "
                "E_sound = 1[out_s==0∧s≠t∧pred=1] + 1[in_t==0∧s≠t∧pred=1]; "
                "E_cone = 1[pred=1]/(1+out_s+in_t)"
            ),
            "alpha": ALPHA_SOUND,
            "beta": BETA_CONE,
            "gamma": GAMMA_CONF,
            "eps": EPS_CONF,
            "uses_bfs_checker": False,
            "uses_orientation": False,
        },
        "secondary": {
            "name": "E_cert",
            "formula": (
                "E_cert = α·E_disagree + γ·(−log(p+ε)); "
                "E_disagree = 1[(pred=1∧¬reach)∨(pred=0∧reach)]"
            ),
            "alpha": ALPHA_SOUND,
            "gamma": GAMMA_CONF,
            "uses_bfs_checker": True,
            "label": "cert-energy",
            "note": (
                "If E_cert selection ≈ #35 certificate policy, report as "
                "cert-energy; primary scientific arm remains E_free"
            ),
        },
        "selection": {
            "primary_rule": "energy_argmin",
            "ties": "lowest_member_index",
            "weighted_tau": TAU_WEIGHTED,
        },
    }


def local_degree_features(row: Mapping[str, Any]) -> dict[str, int]:
    """Cert-free local incidence on Â (no BFS)."""
    enc = row.get("encoding")
    if enc:
        n, edges, s, t = parse_instance(str(enc))
    else:
        n = int(row["n"])
        edges = [(int(a), int(b)) for a, b in row["edges"]]
        s, t = int(row["s"]), int(row["t"])
    n = int(row.get("n", n))
    s = int(row.get("s", s))
    t = int(row.get("t", t))
    out_deg = [0] * n
    in_deg = [0] * n
    for u, v in edges:
        out_deg[int(u)] += 1
        in_deg[int(v)] += 1
    return {
        "n": n,
        "s": s,
        "t": t,
        "out_s": int(out_deg[s]),
        "in_t": int(in_deg[t]),
        "in_s": int(in_deg[s]),
        "out_t": int(out_deg[t]),
    }


def _row_edges(row: Mapping[str, Any]) -> tuple[int, list[tuple[int, int]], int, int]:
    enc = row.get("encoding")
    if enc:
        n, edges, s, t = parse_instance(str(enc))
    else:
        n = int(row["n"])
        edges = [(int(a), int(b)) for a, b in row["edges"]]
        s, t = int(row["s"]), int(row["t"])
    return (
        int(row.get("n", n)),
        [(int(a), int(b)) for a, b in edges],
        int(row.get("s", s)),
        int(row.get("t", t)),
    )


def e_sound(pred: int, out_s: int, in_t: int, s: int, t: int) -> float:
    """Local sound mismatch (cert-free)."""
    pred = int(pred)
    e = 0.0
    if out_s == 0 and s != t and pred == 1:
        e += 1.0
    if in_t == 0 and s != t and pred == 1:
        e += 1.0
    return e


def e_cone(pred: int, out_s: int, in_t: int) -> float:
    """Sparse-cone YES penalty (cert-free)."""
    if int(pred) != 1:
        return 0.0
    return 1.0 / (1.0 + float(out_s) + float(in_t))


def e_conf(p_max: float, *, eps: float = EPS_CONF) -> float:
    """Confidence energy −log(p+ε)."""
    return -math.log(float(p_max) + eps)


def e_disagree(pred: int, reach: bool) -> float:
    """Checker disagreement (cert-tied)."""
    pred = int(pred)
    if pred == 1 and not reach:
        return 1.0
    if pred == 0 and reach:
        return 1.0
    return 0.0


def e_free_scalar(
    pred: int,
    p_max: float,
    out_s: int,
    in_t: int,
    s: int,
    t: int,
    *,
    alpha: float = ALPHA_SOUND,
    beta: float = BETA_CONE,
    gamma: float = GAMMA_CONF,
) -> float:
    """Combined cert-free energy for one member on one example."""
    return (
        alpha * e_sound(pred, out_s, in_t, s, t)
        + beta * e_cone(pred, out_s, in_t)
        + gamma * e_conf(p_max)
    )


def e_cert_scalar(
    pred: int,
    p_max: float,
    reach: bool,
    *,
    alpha: float = ALPHA_SOUND,
    gamma: float = GAMMA_CONF,
) -> float:
    """Combined cert-tied energy for one member on one example."""
    return alpha * e_disagree(pred, reach) + gamma * e_conf(p_max)


def member_preds_and_conf(
    logits_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """logits (M,N,C) → hard preds (M,N), max-prob conf (M,N)."""
    if logits_stack.dim() != 3:
        raise ValueError(
            f"logits_stack must be (M,N,C), got shape {tuple(logits_stack.shape)}"
        )
    preds = logits_stack.argmax(dim=-1)
    conf = F.softmax(logits_stack, dim=-1).max(dim=-1).values
    return preds, conf


def compute_energy_stack(
    logits_stack: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str = "free",
) -> torch.Tensor:
    """Per-member energy E[m,n] for kind in {free, cert}.

    ``logits_stack`` shape ``(M, N, C)`` → ``(M, N)``.
    """
    preds, conf = member_preds_and_conf(logits_stack)
    m, n = preds.shape
    if n != len(rows):
        raise ValueError(f"rows len {len(rows)} != N={n}")
    # Precompute local / reach features per example
    locals_: list[dict[str, int]] = []
    reaches: list[Optional[bool]] = []
    for row in rows:
        loc = local_degree_features(row)
        locals_.append(loc)
        if kind == "cert":
            nn, edges, s, t = _row_edges(row)
            reaches.append(bool(reachable_bfs(nn, edges, s, t)))
        else:
            reaches.append(None)

    E = torch.empty((m, n), dtype=torch.float64)
    for j in range(n):
        loc = locals_[j]
        reach = reaches[j]
        for i in range(m):
            pred = int(preds[i, j].item())
            p = float(conf[i, j].item())
            if kind == "free":
                E[i, j] = e_free_scalar(
                    pred, p, loc["out_s"], loc["in_t"], loc["s"], loc["t"]
                )
            elif kind == "cert":
                assert reach is not None
                E[i, j] = e_cert_scalar(pred, p, reach)
            else:
                raise ValueError(f"unknown energy kind: {kind}")
    return E


def energy_argmin_preds(
    logits_stack: torch.Tensor,
    energy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select member with min energy; return (preds[N], conf[N], m_star[N]).

    Ties → lowest member index (torch.argmin).
    """
    if energy.dim() != 2:
        raise ValueError(f"energy must be (M,N), got {tuple(energy.shape)}")
    m_star = energy.argmin(dim=0)  # (N,)
    preds_m, conf_m = member_preds_and_conf(logits_stack)
    n = energy.shape[1]
    idx = torch.arange(n, device=logits_stack.device)
    # gather on CPU-friendly long indices
    m_star_dev = m_star.to(device=logits_stack.device)
    preds = preds_m[m_star_dev, idx]
    conf = conf_m[m_star_dev, idx]
    return preds, conf, m_star


def energy_weighted_preds(
    logits_stack: torch.Tensor,
    energy: torch.Tensor,
    *,
    tau: float = TAU_WEIGHTED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmin-weight member probs by energy; return (preds[N], conf[N])."""
    # w_m ∝ exp(−E/τ); stable softmin
    e = energy.to(dtype=torch.float64)
    scaled = -e / float(tau)
    scaled = scaled - scaled.max(dim=0, keepdim=True).values
    w = torch.exp(scaled)
    w = w / w.sum(dim=0, keepdim=True).clamp_min(1e-12)  # (M, N)
    probs = F.softmax(logits_stack, dim=-1).to(dtype=torch.float64)  # (M,N,C)
    # weighted mean over members
    mean_p = (w.unsqueeze(-1) * probs).sum(dim=0)  # (N, C)
    preds = mean_p.argmax(dim=-1)
    conf = mean_p.max(dim=-1).values
    return preds.long(), conf


def oracle_member_preds(
    logits_stack: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upper bound: pick lowest-index correct member when any is correct.

    Else member 0. Returns (preds[N], conf[N], m_star[N]).
    """
    preds_m, conf_m = member_preds_and_conf(logits_stack)
    m, n = preds_m.shape
    labels = labels.long().view(-1)
    if labels.numel() != n:
        raise ValueError(f"labels len {labels.numel()} != N={n}")
    correct = preds_m == labels.unsqueeze(0)  # (M, N)
    # For each example: first True index, else 0
    # add sentinel: if none correct, argmax of (correct int + tiny) → use any;
    # use: where any correct, take argmax of (correct * (M - index)) style
    # Simplest: for each n, scan
    m_star = torch.zeros(n, dtype=torch.long)
    for j in range(n):
        found = False
        for i in range(m):
            if bool(correct[i, j].item()):
                m_star[j] = i
                found = True
                break
        if not found:
            m_star[j] = 0
    idx = torch.arange(n)
    preds = preds_m[m_star, idx]
    conf = conf_m[m_star, idx]
    return preds, conf, m_star


def cert_refuse_preds(
    logits_stack: torch.Tensor,
    energy_cert: torch.Tensor,
    *,
    refuse_threshold: float = ALPHA_SOUND,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """energy_argmin under E_cert; if min E_cert ≥ refuse_threshold → force NO.

    Returns (preds[N], conf[N], refused_mask[N] bool-as-long).
    """
    preds, conf, _m = energy_argmin_preds(logits_stack, energy_cert)
    min_e = energy_cert.min(dim=0).values
    refused = min_e >= float(refuse_threshold)
    preds = preds.clone()
    preds[refused] = 0
    return preds, conf, refused.long()


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson r; nan if undefined."""
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx < 1e-15 or dy < 1e-15:
        return float("nan")
    return num / (dx * dy)
