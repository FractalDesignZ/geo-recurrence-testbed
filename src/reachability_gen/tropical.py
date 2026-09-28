"""Tropical / max-plus helpers for ens aggregation probes (MEASURE).

Scoped Phase-2 probe (CYCLE_STALK_TROPICAL_ATTENTION_PROBE):
replace sum-product ens aggregation with max-plus / β→∞ member routing.
Not an in-attention MultiheadAttention rewrite (impractical on frozen
softmax-trained weights; would invent untrained tropical layers).

Definitions (LOCKED in prereg)::

    logit_max (max-plus):
        score_c = max_m L[m, c]
        pred = argmax_c score_c
        conf = max_c softmax(score)_c

    beta_inf_member (β→∞ hard member routing):
        m* = argmax_m max_c softmax(L_m)_c
        pred = argmax_c L[m*, c]
        conf = max_c softmax(L[m*])_c
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def max_plus_scores(logits_stack: torch.Tensor) -> torch.Tensor:
    """Max-plus aggregate member logits: score[n,c] = max_m L[m,n,c].

    ``logits_stack`` shape ``(M, N, C)`` → ``(N, C)``.
    """
    if logits_stack.dim() != 3:
        raise ValueError(
            f"logits_stack must be (M,N,C), got shape {tuple(logits_stack.shape)}"
        )
    return logits_stack.max(dim=0).values


def max_plus_aggregate(
    logits_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(preds[N], conf[N], scores[N,C])`` under logit_max."""
    scores = max_plus_scores(logits_stack)
    preds = scores.argmax(dim=-1)
    conf = F.softmax(scores, dim=-1).max(dim=-1).values
    return preds, conf, scores


def beta_inf_member_preds(
    logits_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """β→∞ hard routing to highest-confidence member.

    Returns ``(preds[N], conf[N], member_index[N])``.
    """
    if logits_stack.dim() != 3:
        raise ValueError(
            f"logits_stack must be (M,N,C), got shape {tuple(logits_stack.shape)}"
        )
    probs = F.softmax(logits_stack, dim=-1)  # (M, N, C)
    member_conf = probs.max(dim=-1).values  # (M, N)
    m_star = member_conf.argmax(dim=0)  # (N,)
    n = logits_stack.shape[1]
    idx = torch.arange(n, device=logits_stack.device)
    chosen_logits = logits_stack[m_star, idx, :]  # (N, C)
    preds = chosen_logits.argmax(dim=-1)
    conf = F.softmax(chosen_logits, dim=-1).max(dim=-1).values
    return preds, conf, m_star


def tropical_definition_doc() -> dict[str, Any]:
    """Machine-readable tropical defs for artifact / prereg cite."""
    return {
        "scope": "ensemble_aggregation_logit_mix",
        "in_attention_rewrite": False,
        "in_attention_rewrite_reason": (
            "Frozen #14/#18/#22 members use softmax MultiheadAttention; "
            "replacing attn mix with max-plus on QK scores trained under "
            "sum-product is not a fair port test and would invent an "
            "untrained tropical operator. Scoped probe = ens aggregation."
        ),
        "primary": {
            "name": "logit_max",
            "formula": "score_c = max_m L[m,c]; pred = argmax_c score_c",
            "semiring": "max-plus (tropical ⊕ = max over members)",
        },
        "secondary": {
            "name": "beta_inf_member",
            "formula": (
                "m* = argmax_m max_c softmax(L_m)_c; "
                "pred = argmax_c L[m*,c]"
            ),
            "note": "β→∞ softmax over members by confidence; hard routing",
        },
        "baseline_contrast": "prob_mean: argmax_c mean_m softmax(L_m)_c",
    }
