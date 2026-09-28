"""CYCLE_STALK_EPISTEMIC_DISAGREEMENT — MEASURE audit + multi-hyp diversity.

(1) Eval-only disagreement / epistemic-vs-aleatoric-style audit on matched-OOD
    T16 for frozen #14/#18 ensemble members vs SWA vs singles — tests whether
    #22 lift rides on disagreement.
(2) One structured-diversity train: shared sealed stalk trunk + H=3 heads;
    CE per head minus λ·mean pairwise JS (encourage disagreement). NOT soft
    distill vs ensemble teacher.

Prereg floors (multi-hyp mean @ matched-OOD T16): hard-neg≥0.95 AND K16≥0.75.
  PASS_CANDIDATE → floors PASS; science_open=false (do not widen §22)
  MEASURE         → miss floor(s) but HN or K16 ≥ #14 seed0
  STOP            → both HN and K16 < #14 seed0

Usage::

    python -m reachability_gen.run_stalk_epistemic_disagreement
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import HOP_UNREACHABLE
from reachability_gen.models.fractal_core import (
    DISCRETE_T_VALUES,
    MANDELBROT_ANALOGY_NOTE,
    FractalCore,
    _verify_param_parity,
    build_node_slot_batch,
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.reach_cue_audit import build_degree_balanced_eval
from reachability_gen.run_fractal_core_gate1 import (
    DEFAULT_BATCH,
    DEFAULT_CLIP,
    DEFAULT_D,
    DEFAULT_MLP,
    DEFAULT_T,
    _eval_split,
    _mean,
    _n_heads,
    _split_train_val,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS
from reachability_gen.run_stalk_multi_seed_reconfirm import _eval_all_T, _std
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    SECONDARY_AGGS,
    _aggregate_preds,
    _ckpt_for_seed,
    _collect_member_logits,
    _eval_ensemble_methods,
    _load_model_from_ckpt,
    _metrics_from_preds,
)
from reachability_gen.run_stalk_stabilize_multi_seed import LR_MAX, LR_MIN, _cosine_lr
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_EPISTEMIC_DISAGREEMENT"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_epistemic_disagreement.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))
DEFAULT_SWA_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TRAIN_SEEDS = (0, 1, 2)
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16

DEFAULT_EPOCHS = 60
DEFAULT_CLIP_H = DEFAULT_CLIP
PREREG_HARD_NEG = 0.95
PREREG_K16 = 0.75
N_HYP = 3
LAMBDA_JS = 0.5  # encourage pairwise JS (NOT soft distill)
AUDIT_DELTA_EPS = 0.02  # Δ_high − Δ_low threshold for "rides on disagreement"
PR14_REF_SEED = 0


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _swa_ckpt(seed: int) -> Path:
    return Path(f"artifacts/fractal_core_stalk_swa_persist_seed{seed}_swa.pt")


def _hyp_ckpt(seed: int) -> Path:
    return Path(f"artifacts/fractal_core_stalk_epistemic_disagreement_seed{seed}_best.pt")


def _entropy_nats(probs: Any, eps: float = 1e-8) -> Any:
    """probs: (..., C) → entropy (...,)."""
    import torch

    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def _pairwise_disagreement_rate(preds: Any) -> float:
    """preds: (M, N) int — mean fraction of disagreeing pairs per example, then mean."""
    import torch

    m, n = preds.shape
    if m < 2 or n == 0:
        return float("nan")
    total = 0.0
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            total += float((preds[i] != preds[j]).float().mean().item())
            count += 1
    return total / count if count else float("nan")


def _pairwise_disagreement_per_example(preds: Any) -> Any:
    """preds (M, N) → (N,) mean pairwise disagreement fraction."""
    import torch

    m, n = preds.shape
    if m < 2:
        return torch.zeros(n, dtype=torch.float32)
    acc = torch.zeros(n, dtype=torch.float32)
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            acc += (preds[i] != preds[j]).float()
            count += 1
    return acc / float(count)


def _epistemic_aleatoric(logits_stack: Any) -> dict[str, float]:
    """logits (M, N, C) → mean total / aleatoric / epistemic entropy (nats)."""
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits_stack, dim=-1)  # (M, N, C)
    mean_p = probs.mean(dim=0)  # (N, C)
    total = _entropy_nats(mean_p)  # (N,)
    alea = _entropy_nats(probs).mean(dim=0)  # (N,)
    epi = total - alea
    return {
        "total_entropy_mean": float(total.mean().item()),
        "aleatoric_entropy_mean": float(alea.mean().item()),
        "epistemic_entropy_mean": float(epi.mean().item()),
        "n": int(total.numel()),
    }


def _js_pairwise_mean(logit_stack: Any, eps: float = 1e-8) -> Any:
    """logit_stack (H, B, C) → scalar mean pairwise JS (nats)."""
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logit_stack, dim=-1)
    h = probs.shape[0]
    if h < 2:
        return probs.new_zeros(())
    acc = probs.new_zeros(())
    n = 0
    for i in range(h):
        for j in range(i + 1, h):
            m = 0.5 * (probs[i] + probs[j])
            kl_im = (probs[i].clamp_min(eps) * (probs[i].clamp_min(eps).log() - m.clamp_min(eps).log())).sum(-1)
            kl_jm = (probs[j].clamp_min(eps) * (probs[j].clamp_min(eps).log() - m.clamp_min(eps).log())).sum(-1)
            js = 0.5 * (kl_im + kl_jm)
            acc = acc + js.mean()
            n += 1
    return acc / float(n)


class MultiHypHead:
    """Placeholder — real class defined in _ensure_multi_hyp_class()."""

    pass


def _multi_hyp_head_cls():
    import torch.nn as nn

    class _MultiHypHead(nn.Module):
        """Drop-in replacement for FractalCore.head: H linear heads; forward = mean logits."""

        def __init__(self, d: int, n_hyp: int = N_HYP):
            super().__init__()
            self.n_hyp = int(n_hyp)
            self.heads = nn.ModuleList([nn.Linear(d, 2) for _ in range(self.n_hyp)])
            self.last_stack: Any = None

        def forward(self, x: Any) -> Any:
            import torch

            stack = torch.stack([h(x) for h in self.heads], dim=0)  # (H, B, 2)
            self.last_stack = stack
            return stack.mean(dim=0)

    return _MultiHypHead


def attach_multi_hyp_head(model: Any, *, n_hyp: int = N_HYP) -> Any:
    """Replace model.head with MultiHypHead; return the head module."""
    cls = _multi_hyp_head_cls()
    d = int(model.d)
    hyp = cls(d, n_hyp=n_hyp)
    model.head = hyp
    return hyp


def _build_multi_hyp_model(max_nodes: int, *, n_hyp: int = N_HYP, seed: int = 0) -> Any:
    import torch

    torch.manual_seed(seed)
    model = FractalCore(
        d=DEFAULT_D,
        T=DEFAULT_T,
        n_heads=_n_heads(DEFAULT_D),
        mlp_expansion=DEFAULT_MLP,
        max_nodes=max_nodes,
        max_T=max(DEFAULT_T, max(T_VALUES)),
        use_tau=True,
        apply_cycle_rmsnorm=True,
    )
    attach_multi_hyp_head(model, n_hyp=n_hyp)
    return model


def _load_multi_hyp_from_ckpt(ckpt: Path, max_nodes: int) -> tuple[Any, dict[str, Any]]:
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_hyp = int(blob.get("n_hyp", N_HYP))
    model = _build_multi_hyp_model(max_nodes, n_hyp=n_hyp, seed=0)
    # state_dict may include multi_hyp_heads.* and possibly stale head.*
    sd = blob["state_dict"]
    # Filter keys that belong to current model
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, blob


def _clone_state(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_fractal_id2k_multi_hyp(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    d: int = DEFAULT_D,
    T: int = DEFAULT_T,
    mlp_expansion: int = DEFAULT_MLP,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    grad_clip: float = DEFAULT_CLIP_H,
    batch_size: int = DEFAULT_BATCH,
    seed: int = 0,
    max_nodes: int = DEFAULT_MAX_NODE_ID,
    n_hyp: int = N_HYP,
    lambda_js: float = LAMBDA_JS,
    ckpt_path: Path,
) -> dict[str, Any]:
    """#14 harden + multi-hyp heads; CE mean − λ·JS (encourage disagreement)."""
    import torch
    import torch.nn as nn
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW

    del d, mlp_expansion  # fixed via _build_multi_hyp_model
    torch.manual_seed(seed)
    model = _build_multi_hyp_model(max_nodes, n_hyp=n_hyp, seed=seed)
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    # Note: multi-hyp adds (H-1)*d*2 params vs single head — may exceed 5% slightly
    opt = AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    param_count = model.param_count()
    print(
        f"[stalk-epi] params={param_count} parity_ok={parity['within_5pct']} "
        f"n_hyp={n_hyp} λ_js={lambda_js} T={T} lr={lr_max}→{lr_min} cosine "
        f"clip={grad_clip} (NOT soft distill)",
        file=sys.stderr,
    )

    best_key: tuple[float, float, float, int] = (-1.0, -1.0, -1.0, 0)
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    best_sel: dict[str, float] = {}
    train_hist: list[dict[str, Any]] = []
    n_sat = 0
    n_steps = 0
    val_by_hop_final: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        lr = _cosine_lr(epoch, epochs, lr_max, lr_min)
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        order = torch.randperm(len(train)).tolist()
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        epoch_js: list[float] = []
        for start in range(0, len(train), batch_size):
            idx = order[start : start + batch_size]
            batch_rows = [train[i] for i in idx]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            opt.zero_grad(set_to_none=True)
            logits_mean, _, _ = model(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                return_halt=True,
                T=T,
                adaptive_halt=False,
            )
            stack = model.head.last_stack  # (H, B, 2)
            labels = batch["labels"]
            ce_terms = [ce(stack[h], labels) for h in range(stack.shape[0])]
            ce_loss = torch.stack(ce_terms).mean()
            js = _js_pairwise_mean(stack)
            # Encourage disagreement: minimize −λ·JS
            loss = ce_loss - float(lambda_js) * js
            loss.backward()
            pre_clip = clip_grad_norm_(model.parameters(), grad_clip)
            if float(pre_clip.item() if hasattr(pre_clip, "item") else pre_clip) >= grad_clip:
                n_sat += 1
            opt.step()
            n_steps += 1
            with torch.no_grad():
                preds = logits_mean.argmax(dim=-1)
                acc = (preds == labels).float().mean()
            epoch_losses.append(float(loss.item()))
            epoch_accs.append(float(acc.item()))
            epoch_js.append(float(js.item()))

        train_loss = _mean(epoch_losses)
        train_acc = _mean(epoch_accs)
        train_js = _mean(epoch_js)
        # Select on head-prob_mean (= model.head mean logits path)
        sel_stats = _eval_split(model, val, max_nodes=max_nodes, T=FOCUS_T)
        sel_overall = float(sel_stats["overall_acc"])
        sel_hard = float(sel_stats["hard_neg_acc"])
        val_ttrain = _eval_split(model, val, max_nodes=max_nodes, T=T)
        ov_ttrain = float(val_ttrain["overall_acc"])
        val_by_hop_final = sel_stats["by_hop"]

        joint = 0.5 * sel_hard + 0.5 * sel_overall
        key = (joint, sel_overall, sel_hard, -epoch)
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_sel = {
                "hard_neg_acc_T16_val": sel_hard,
                "overall_acc_T16_val": sel_overall,
                "joint_score_T16_val": joint,
            }
            best_state = _clone_state(model)

        train_hist.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_js_mean": train_js,
                "val_acc_T_train": ov_ttrain,
                "sel_hard_neg_T16": sel_hard,
                "sel_overall_T16": sel_overall,
                "sel_joint_T16": joint,
                "selected": improved,
            }
        )
        print(
            f"[stalk-epi] epoch {epoch}/{epochs} lr={lr:.5g} "
            f"train_acc={train_acc:.4f} js={train_js:.4f} val@T{T}={ov_ttrain:.4f} "
            f"sel@T16 HN={sel_hard:.4f} ov={sel_overall:.4f} "
            f"(best@ep{best_epoch} joint={best_sel.get('joint_score_T16_val', float('nan')):.4f})",
            file=sys.stderr,
        )

    hparams = {
        "d": DEFAULT_D,
        "T": T,
        "mlp_expansion": DEFAULT_MLP,
        "max_nodes": max_nodes,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "lr_schedule": "cosine",
        "epochs": epochs,
        "grad_clip": grad_clip,
        "n_hyp": n_hyp,
        "lambda_js": lambda_js,
        "loss": "mean_h CE − λ·mean_pairwise_JS (encourage disagreement; NOT soft distill)",
        "adaptive_halt": False,
        "local_potential": True,
        "broadcast_c": False,
    }

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(
            {
                "epoch": best_epoch,
                "val_acc": best_sel.get("overall_acc_T16_val"),
                "sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
                "sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
                "state_dict": best_state,
                "arm": f"fractal-stalk-epistemic-disagreement-T{T}-d{DEFAULT_D}-hyp{n_hyp}",
                "science_open": False,
                "cycle": CYCLE,
                "n_hyp": n_hyp,
                "lambda_js": lambda_js,
                "selection_rule": (
                    "lexicographic (0.5*HN+0.5*overall @ T16 ID-val on head-prob_mean, "
                    "overall, HN, -epoch); matched-OOD never used for selection"
                ),
                "hparams": hparams,
            },
            ckpt_path,
        )

    return {
        "arm": f"fractal-stalk-epistemic-disagreement-T{T}-hyp{n_hyp}",
        "param_count": param_count,
        "param_parity": parity,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
        "best_sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
        "best_sel_joint_T16_val": best_sel.get("joint_score_T16_val"),
        "n_hyp": n_hyp,
        "lambda_js": lambda_js,
        "selection_rule": (
            "lexicographic (0.5*HN+0.5*overall @ T16 ID-val on head-prob_mean, "
            "overall, HN, -epoch)"
        ),
        "train_history": train_hist,
        "val_by_hop": val_by_hop_final,
        "grad_clip": grad_clip,
        "grad_clip_sat_rate": n_sat / n_steps if n_steps else float("nan"),
        "lr_max": lr_max,
        "lr_min": lr_min,
        "lr_schedule": "cosine",
        "d": DEFAULT_D,
        "T": T,
        "mlp_expansion": DEFAULT_MLP,
        "max_nodes": max_nodes,
        "adaptive_halt": False,
        "local_potential": True,
        "broadcast_c": False,
        "checkpoint_path": str(ckpt_path),
        "science_open": False,
    }


def _mask_metrics(
    preds: Any,
    labels: Any,
    hops: list[int],
    mask: Any,
) -> dict[str, float]:
    """Metrics on a boolean mask over examples."""
    import torch

    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return {
            "n": 0,
            "overall_acc": float("nan"),
            "hard_neg_acc": float("nan"),
            "K16": float("nan"),
        }
    sub_preds = preds[idx]
    sub_labels = labels[idx]
    sub_hops = [hops[int(i)] for i in idx.tolist()]
    m = _metrics_from_preds(sub_preds, sub_labels, sub_hops)
    return {
        "n": int(m["n"]),
        "overall_acc": float(m["overall_acc"]),
        "hard_neg_acc": float(m["hard_neg_acc"]),
        "K16": float(m["K16"]),
    }


def _audit_bag(
    name: str,
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Full disagreement / epi-alea audit + lift×disagreement median split."""
    import torch
    import torch.nn.functional as F

    if not models:
        return {"name": name, "n_members": 0, "skipped": True}

    logits, labels, hops = _collect_member_logits(
        models, rows, T=T, max_nodes=max_nodes
    )  # (M, N, C)
    probs = F.softmax(logits, dim=-1)
    hard = logits.argmax(dim=-1)  # (M, N)
    ens_preds = _aggregate_preds(logits, method=PRIMARY_AGG)
    singles_preds = hard  # (M, N)

    # Per-example singles mean accuracy proxy: fraction of members correct
    correct_m = (hard == labels.unsqueeze(0)).float()  # (M, N)
    singles_frac_correct = correct_m.mean(dim=0)  # (N,)
    ens_correct = (ens_preds == labels).float()

    pair_rate = _pairwise_disagreement_rate(hard)
    pair_per = _pairwise_disagreement_per_example(hard)  # (N,)
    epi = _epistemic_aleatoric(logits)

    ens_metrics = _metrics_from_preds(ens_preds, labels, hops)
    singles_list = []
    for mi in range(logits.shape[0]):
        m = _metrics_from_preds(hard[mi], labels, hops)
        m["seed"] = seeds[mi] if mi < len(seeds) else mi
        singles_list.append(m)
    singles_mean = {
        "overall_acc": _mean([s["overall_acc"] for s in singles_list]),
        "hard_neg_acc": _mean([s["hard_neg_acc"] for s in singles_list]),
        "K16": _mean([s["K16"] for s in singles_list]),
        "std_overall": _std([s["overall_acc"] for s in singles_list]),
        "std_hard_neg": _std([s["hard_neg_acc"] for s in singles_list]),
        "std_K16": _std([s["K16"] for s in singles_list]),
    }
    deltas = {
        "overall": ens_metrics["overall_acc"] - singles_mean["overall_acc"],
        "hard_neg": ens_metrics["hard_neg_acc"] - singles_mean["hard_neg_acc"],
        "K16": ens_metrics["K16"] - singles_mean["K16"],
    }

    # Median split on pairwise disagreement
    med = float(pair_per.median().item()) if pair_per.numel() else float("nan")
    high = pair_per >= med
    low = pair_per < med
    # Also: any disagreement vs none
    any_dis = pair_per > 0
    no_dis = pair_per == 0

    def _bin_report(mask: Any, label: str) -> dict[str, Any]:
        ens_m = _mask_metrics(ens_preds, labels, hops, mask)
        # singles mean accuracy on bin ≈ mean over members of member acc on bin
        # Use mean fraction-correct as overall proxy, plus hop-stratified via averaging members
        member_ov = []
        member_hn = []
        member_k16 = []
        for mi in range(hard.shape[0]):
            mm = _mask_metrics(hard[mi], labels, hops, mask)
            member_ov.append(mm["overall_acc"])
            member_hn.append(mm["hard_neg_acc"])
            member_k16.append(mm["K16"])
        sm = {
            "overall_acc": _mean([x for x in member_ov if x == x]),
            "hard_neg_acc": _mean([x for x in member_hn if x == x]),
            "K16": _mean([x for x in member_k16 if x == x]),
        }
        return {
            "label": label,
            "n": ens_m["n"],
            "ens": ens_m,
            "singles_mean": sm,
            "delta": {
                "overall": ens_m["overall_acc"] - sm["overall_acc"]
                if ens_m["overall_acc"] == ens_m["overall_acc"] and sm["overall_acc"] == sm["overall_acc"]
                else float("nan"),
                "hard_neg": ens_m["hard_neg_acc"] - sm["hard_neg_acc"]
                if ens_m["hard_neg_acc"] == ens_m["hard_neg_acc"] and sm["hard_neg_acc"] == sm["hard_neg_acc"]
                else float("nan"),
                "K16": ens_m["K16"] - sm["K16"]
                if ens_m["K16"] == ens_m["K16"] and sm["K16"] == sm["K16"]
                else float("nan"),
            },
            "mean_pairwise_disagreement": float(pair_per[mask].mean().item())
            if mask.any()
            else float("nan"),
            "mean_singles_frac_correct": float(singles_frac_correct[mask].mean().item())
            if mask.any()
            else float("nan"),
            "ens_acc_raw": float(ens_correct[mask].mean().item()) if mask.any() else float("nan"),
        }

    bins = {
        "high_disagreement": _bin_report(high, "high_disagreement"),
        "low_disagreement": _bin_report(low, "low_disagreement"),
        "any_disagreement": _bin_report(any_dis, "any_disagreement"),
        "no_disagreement": _bin_report(no_dis, "no_disagreement"),
    }

    def _safe_sub(a: float, b: float) -> float:
        if a != a or b != b:
            return float("nan")
        return a - b

    lift_gap = {
        "overall": _safe_sub(
            bins["high_disagreement"]["delta"]["overall"],
            bins["low_disagreement"]["delta"]["overall"],
        ),
        "hard_neg": _safe_sub(
            bins["high_disagreement"]["delta"]["hard_neg"],
            bins["low_disagreement"]["delta"]["hard_neg"],
        ),
        "K16": _safe_sub(
            bins["high_disagreement"]["delta"]["K16"],
            bins["low_disagreement"]["delta"]["K16"],
        ),
    }

    rides = any(
        lift_gap[k] == lift_gap[k] and lift_gap[k] >= AUDIT_DELTA_EPS
        for k in ("overall", "hard_neg", "K16")
    )

    # Pairwise matrix (compact): mean disagreement per seed-pair
    m = hard.shape[0]
    pair_matrix: list[list[float]] = []
    for i in range(m):
        row = []
        for j in range(m):
            if i == j:
                row.append(0.0)
            else:
                row.append(float((hard[i] != hard[j]).float().mean().item()))
        pair_matrix.append(row)

    return {
        "name": name,
        "n_members": len(models),
        "seeds": seeds,
        "T": T,
        "n_examples": int(labels.numel()),
        "pairwise_disagreement_rate": pair_rate,
        "disagreement_median": med,
        "fraction_examples_any_disagreement": float((pair_per > 0).float().mean().item()),
        "uncertainty": epi,
        "ensemble_prob_mean": {
            "overall_acc": ens_metrics["overall_acc"],
            "hard_neg_acc": ens_metrics["hard_neg_acc"],
            "K8": ens_metrics["K8"],
            "K12": ens_metrics["K12"],
            "K16": ens_metrics["K16"],
        },
        "singles_mean": singles_mean,
        "singles": [
            {
                "seed": s["seed"],
                "overall_acc": s["overall_acc"],
                "hard_neg_acc": s["hard_neg_acc"],
                "K16": s["K16"],
            }
            for s in singles_list
        ],
        "deltas_ens_minus_singles_mean": deltas,
        "bins": bins,
        "lift_gap_high_minus_low": lift_gap,
        "rides_on_disagreement": rides,
        "audit_delta_eps": AUDIT_DELTA_EPS,
        "pairwise_disagreement_matrix": pair_matrix,
        "skipped": False,
    }


def _audit_single_models(
    name: str,
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Per-model metrics (epistemic=0 within model); bag disagreement if ≥2."""
    if not models:
        return {"name": name, "n_members": 0, "skipped": True}
    if len(models) >= 2:
        return _audit_bag(name, models, rows, T=T, max_nodes=max_nodes, seeds=seeds)
    # Single model: report metrics only
    cell = _eval_ensemble_methods(
        models, rows, T=T, max_nodes=max_nodes, methods=(PRIMARY_AGG,)
    )
    s0 = cell["singles"][0]
    return {
        "name": name,
        "n_members": 1,
        "seeds": seeds,
        "T": T,
        "pairwise_disagreement_rate": 0.0,
        "uncertainty": {
            "total_entropy_mean": float("nan"),
            "aleatoric_entropy_mean": float("nan"),
            "epistemic_entropy_mean": 0.0,
            "note": "single model — epistemic style ≈ 0 by construction",
        },
        "metrics": {
            "overall_acc": s0["overall_acc"],
            "hard_neg_acc": s0["hard_neg_acc"],
            "K16": s0["K16"],
        },
        "rides_on_disagreement": False,
        "skipped": False,
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS,
    swa_seeds: tuple[int, ...] = DEFAULT_SWA_SEEDS,
    train_seeds: tuple[int, ...] = DEFAULT_TRAIN_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    grad_clip: float = DEFAULT_CLIP_H,
    n_hyp: int = N_HYP,
    lambda_js: float = LAMBDA_JS,
    skip_train: bool = False,
) -> dict[str, Any]:
    import torch

    t0 = time.time()
    if not id_data.exists():
        raise FileNotFoundError(id_data)
    if not ood_data.exists():
        raise FileNotFoundError(ood_data)

    rows = load_jsonl(id_data)
    train, val = _split_train_val(rows)
    ood_rows = load_jsonl(ood_data)
    max_nodes = max(
        DEFAULT_MAX_NODE_ID,
        max(int(r["n"]) for r in rows),
        max(int(r["n"]) for r in ood_rows),
    )
    deg_bal = build_degree_balanced_eval(ood_rows, seed=0)
    deg_rows = deg_bal["rows"]

    # ---- Part A: load ensemble + SWA ----
    print(f"[stalk-epi] AUDIT loading #14/#18 ensemble n={len(ensemble_seeds)}", file=sys.stderr)
    ens_models: list[Any] = []
    ens_meta: list[dict[str, Any]] = []
    for seed in ensemble_seeds:
        ckpt = _ckpt_for_seed(seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing ensemble member ckpt: {ckpt}")
        model, blob = _load_model_from_ckpt(ckpt, max_nodes)
        ens_models.append(model)
        ens_meta.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "source": "pr14" if seed <= 4 else "pr18",
                "best_epoch": blob.get("epoch"),
            }
        )

    swa_models: list[Any] = []
    swa_meta: list[dict[str, Any]] = []
    for seed in swa_seeds:
        ckpt = _swa_ckpt(seed)
        if not ckpt.exists():
            print(f"[stalk-epi] SWA ckpt missing seed={seed}: {ckpt}", file=sys.stderr)
            continue
        model, blob = _load_model_from_ckpt(ckpt, max_nodes)
        swa_models.append(model)
        swa_meta.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "source": "pr25_swa",
                "n_swa": blob.get("n_swa"),
            }
        )

    print("[stalk-epi] AUDIT matched-OOD T16 ensemble bag", file=sys.stderr)
    audit_ens = _audit_bag(
        "ensemble_pr14_pr18",
        ens_models,
        ood_rows,
        T=FOCUS_T,
        max_nodes=max_nodes,
        seeds=list(ensemble_seeds),
    )
    print("[stalk-epi] AUDIT matched-OOD T16 SWA bag", file=sys.stderr)
    audit_swa = _audit_bag(
        "swa_pr25",
        swa_models,
        ood_rows,
        T=FOCUS_T,
        max_nodes=max_nodes,
        seeds=[m["seed"] for m in swa_meta],
    ) if swa_models else {"name": "swa_pr25", "skipped": True, "reason": "no_swa_ckpts"}

    # Horizon summary: disagreement rate @ each T for ensemble
    ens_disagree_by_T: dict[str, Any] = {}
    for T in T_VALUES:
        logits, labels, hops = _collect_member_logits(
            ens_models, ood_rows, T=int(T), max_nodes=max_nodes
        )
        hard = logits.argmax(dim=-1)
        ens_disagree_by_T[str(T)] = {
            "pairwise_disagreement_rate": _pairwise_disagreement_rate(hard),
            "uncertainty": _epistemic_aleatoric(logits),
            "n": int(labels.numel()),
        }

    audit_verdict = (
        "AUDIT_LIFTS_ON_DISAGREEMENT"
        if audit_ens.get("rides_on_disagreement")
        else "AUDIT_LIFT_NOT_DISAGREEMENT_DOMINATED"
    )

    # ---- Part B: multi-hyp train ----
    train_results: list[dict[str, Any]] = []
    hyp_models: list[Any] = []
    if not skip_train:
        for seed in train_seeds:
            ckpt = _hyp_ckpt(seed)
            print(f"[stalk-epi] TRAIN multi-hyp seed={seed} → {ckpt}", file=sys.stderr)
            tr = train_fractal_id2k_multi_hyp(
                train,
                val,
                epochs=epochs,
                seed=seed,
                max_nodes=max_nodes,
                lr_max=lr_max,
                lr_min=lr_min,
                grad_clip=grad_clip,
                n_hyp=n_hyp,
                lambda_js=lambda_js,
                ckpt_path=ckpt,
            )
            train_results.append({"seed": seed, **tr})
            model, blob = _load_multi_hyp_from_ckpt(ckpt, max_nodes)
            hyp_models.append(model)
    else:
        for seed in train_seeds:
            ckpt = _hyp_ckpt(seed)
            if ckpt.exists():
                model, _ = _load_multi_hyp_from_ckpt(ckpt, max_nodes)
                hyp_models.append(model)
                train_results.append({"seed": seed, "checkpoint_path": str(ckpt), "loaded_existing": True})

    # Eval multi-hyp
    hyp_per_seed: list[dict[str, Any]] = []
    hyp_matched_by_T: dict[str, list[dict[str, float]]] = {str(T): [] for T in T_VALUES}
    for i, model in enumerate(hyp_models):
        seed = train_seeds[i] if i < len(train_seeds) else i
        matched = _eval_all_T(model, ood_rows, max_nodes=max_nodes, T_values=T_VALUES)
        deg = _eval_all_T(model, deg_rows, max_nodes=max_nodes, T_values=T_VALUES)
        t16 = matched[str(FOCUS_T)]
        cell = {
            "seed": seed,
            "overall_acc": float(t16["overall_acc"]),
            "hard_neg_acc": float(t16["hard_neg_acc"]),
            "K8": float(t16["K8"]),
            "K12": float(t16["K12"]),
            "K16": float(t16["K16"]),
            "matched_ood_by_T": matched,
            "degree_balanced_by_T": deg,
            "prereg_pass": bool(
                float(t16["hard_neg_acc"]) >= PREREG_HARD_NEG
                and float(t16["K16"]) >= PREREG_K16
            ),
        }
        # Within-model head disagreement @ T16
        # Collect per-head logits by calling forward and reading last_stack
        head_logits_chunks = []
        label_chunks = []
        hops_h: list[int] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(ood_rows), 64):
                batch_rows = ood_rows[start : start + 64]
                batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
                _ = model(
                    batch["node_ids"],
                    batch["node_mask"],
                    batch["attn_mask"],
                    batch["s_idx"],
                    batch["t_idx"],
                    return_halt=True,
                    T=FOCUS_T,
                    adaptive_halt=False,
                )
                stack = model.head.last_stack  # (H, B, 2)
                head_logits_chunks.append(stack)
                label_chunks.append(batch["labels"])
                for ex in batch_rows:
                    hops_h.append(int(ex.get("hop_distance", HOP_UNREACHABLE)))
        head_logits = torch.cat(head_logits_chunks, dim=1)  # (H, N, 2)
        labels_h = torch.cat(label_chunks, dim=0)
        head_hard = head_logits.argmax(dim=-1)
        cell["within_model_head_disagreement_rate"] = _pairwise_disagreement_rate(head_hard)
        cell["within_model_uncertainty"] = _epistemic_aleatoric(head_logits)
        head_ens = _aggregate_preds(head_logits, method=PRIMARY_AGG)
        cell["head_prob_mean_check"] = _metrics_from_preds(head_ens, labels_h, hops_h)
        hyp_per_seed.append(cell)
        for T in T_VALUES:
            tt = matched[str(T)]
            hyp_matched_by_T[str(T)].append(
                {
                    "overall_acc": float(tt["overall_acc"]),
                    "hard_neg_acc": float(tt["hard_neg_acc"]),
                    "K16": float(tt["K16"]),
                }
            )

    def _mean_std_arm(cells: list[dict[str, float]], key: str) -> dict[str, float]:
        xs = [c[key] for c in cells]
        return {"mean": _mean(xs), "std": _std(xs), "n": len(xs)}

    hyp_mean_t16: dict[str, Any] = {}
    if hyp_per_seed:
        hyp_mean_t16 = {
            "overall_acc": _mean([c["overall_acc"] for c in hyp_per_seed]),
            "hard_neg_acc": _mean([c["hard_neg_acc"] for c in hyp_per_seed]),
            "K8": _mean([c["K8"] for c in hyp_per_seed]),
            "K12": _mean([c["K12"] for c in hyp_per_seed]),
            "K16": _mean([c["K16"] for c in hyp_per_seed]),
            "std_overall": _std([c["overall_acc"] for c in hyp_per_seed]),
            "std_hard_neg": _std([c["hard_neg_acc"] for c in hyp_per_seed]),
            "std_K16": _std([c["K16"] for c in hyp_per_seed]),
            "seed_pass": sum(1 for c in hyp_per_seed if c["prereg_pass"]),
            "n_seeds": len(hyp_per_seed),
            "mean_within_head_disagreement": _mean(
                [c["within_model_head_disagreement_rate"] for c in hyp_per_seed]
            ),
        }

    # References: #14 seed0 + ensemble (already audited)
    ref_pr14, _ = _load_model_from_ckpt(_ckpt_for_seed(PR14_REF_SEED), max_nodes)
    ref_matched = _eval_all_T(ref_pr14, ood_rows, max_nodes=max_nodes, T_values=T_VALUES)
    ref_t16 = ref_matched[str(FOCUS_T)]
    pr14_ref = {
        "seed": PR14_REF_SEED,
        "overall_acc": float(ref_t16["overall_acc"]),
        "hard_neg_acc": float(ref_t16["hard_neg_acc"]),
        "K16": float(ref_t16["K16"]),
    }

    floors_pass = bool(
        hyp_per_seed
        and hyp_mean_t16["hard_neg_acc"] == hyp_mean_t16["hard_neg_acc"]
        and hyp_mean_t16["hard_neg_acc"] >= PREREG_HARD_NEG
        and hyp_mean_t16["K16"] == hyp_mean_t16["K16"]
        and hyp_mean_t16["K16"] >= PREREG_K16
    )
    ge_hn = (
        hyp_per_seed
        and hyp_mean_t16["hard_neg_acc"] == hyp_mean_t16["hard_neg_acc"]
        and hyp_mean_t16["hard_neg_acc"] >= pr14_ref["hard_neg_acc"]
    )
    ge_k16 = (
        hyp_per_seed
        and hyp_mean_t16["K16"] == hyp_mean_t16["K16"]
        and hyp_mean_t16["K16"] >= pr14_ref["K16"]
    )

    if not hyp_per_seed:
        train_verdict = "STOP"
        residue = "No multi-hyp models trained/loaded; train arm STOP."
        open_status = "STOP"
    elif floors_pass:
        train_verdict = "PASS_CANDIDATE"
        residue = (
            f"Multi-hyp PASS_CANDIDATE: T16 mean HN={hyp_mean_t16['hard_neg_acc']:.4f}≥{PREREG_HARD_NEG}, "
            f"K16={hyp_mean_t16['K16']:.4f}≥{PREREG_K16}; seed_pass={hyp_mean_t16['seed_pass']}/{hyp_mean_t16['n_seeds']}. "
            "FLAG human — harness keeps science_open=false; do not widen §22."
        )
        open_status = "PASS_CANDIDATE_SCIENCE_OPEN_FALSE"
    elif ge_hn or ge_k16:
        train_verdict = "MEASURE"
        residue = (
            f"Multi-hyp MEASURE: T16 mean HN={hyp_mean_t16['hard_neg_acc']:.4f} "
            f"K16={hyp_mean_t16['K16']:.4f} (floors_pass={floors_pass}); "
            f"vs #14 seed0 HN={pr14_ref['hard_neg_acc']:.4f} K16={pr14_ref['K16']:.4f}; "
            f"ge_hn={ge_hn} ge_k16={ge_k16}. science_open=false."
        )
        open_status = "MEASURE"
    else:
        train_verdict = "STOP"
        residue = (
            f"Multi-hyp STOP: T16 mean HN={hyp_mean_t16['hard_neg_acc']:.4f} "
            f"K16={hyp_mean_t16['K16']:.4f} both < #14 seed0 "
            f"(HN={pr14_ref['hard_neg_acc']:.4f} K16={pr14_ref['K16']:.4f}). "
            "science_open=false."
        )
        open_status = "STOP"

    # Optional: audit multi-hyp models as a bag (across seeds) if ≥2
    audit_hyp_bag = (
        _audit_bag(
            "multi_hyp_seeds",
            hyp_models,
            ood_rows,
            T=FOCUS_T,
            max_nodes=max_nodes,
            seeds=list(train_seeds[: len(hyp_models)]),
        )
        if len(hyp_models) >= 2
        else {"name": "multi_hyp_seeds", "skipped": True}
    )

    elapsed = time.time() - t0
    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "base_sha": _git_sha(),
        "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
        "frozen_recipe": (
            "Audit #14/#18 ens disagreement + multi-hyp heads with JS encourage; "
            "NOT soft distill; #14 select freeze"
        ),
        "architecture": {
            "local_potential": True,
            "broadcast_c": False,
            "soft_ACT": False,
            "hard_A_ij_mask": True,
            "stalk_slot": "s",
            "probe_slot": "t",
            "intermediates": 0,
            "discrete_T_values": list(T_VALUES),
            "multi_hyp_heads": n_hyp,
        },
        "knobs": {
            "n_hyp": n_hyp,
            "lambda_js": lambda_js,
            "loss": "mean_h CE − λ·mean_pairwise_JS (encourage; NOT soft distill)",
            "train_seeds": list(train_seeds),
            "ensemble_seeds": list(ensemble_seeds),
            "swa_seeds": list(swa_seeds),
            "epochs": epochs,
            "audit_delta_eps": AUDIT_DELTA_EPS,
            "skip_train": skip_train,
        },
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "batch_size": DEFAULT_BATCH,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "grad_clip": grad_clip,
            "ff_baseline": FF_BASELINE_PARAMS,
            "matched_to": "PR#14 freeze + PR#12 floors; anti-distill JS",
        },
        "prereg": {
            "hard_neg_multihyp_mean_ge": PREREG_HARD_NEG,
            "K16_at_T16_multihyp_mean_ge": PREREG_K16,
            "selection_rule": (
                "lex (0.5*HN+0.5*overall @ T16 ID-val on head-prob_mean, overall, HN, -epoch); "
                "no OOD peek"
            ),
            "note": (
                "science_open never self-stamped; do not widen §22. "
                "Select/curriculum remain CLOSED. NOT soft distill."
            ),
        },
        "datasets": {"id": str(id_data), "matched_ood": str(ood_data)},
        "degree_balanced_construction": {
            k: v for k, v in deg_bal.items() if k != "rows"
        },
        "members_ensemble": ens_meta,
        "members_swa": swa_meta,
        "audit": {
            "verdict": audit_verdict,
            "ensemble_pr14_pr18_T16": audit_ens,
            "swa_pr25_T16": audit_swa,
            "ensemble_disagreement_by_T": ens_disagree_by_T,
            "multi_hyp_seed_bag_T16": audit_hyp_bag,
        },
        "train": {
            "per_seed_train": [
                {k: v for k, v in tr.items() if k != "train_history"}
                | {"train_history_len": len(tr.get("train_history", []))}
                for tr in train_results
            ],
            "train_history_by_seed": {
                str(tr["seed"]): tr.get("train_history", []) for tr in train_results
            },
            "per_seed_ood_T16": [
                {
                    "seed": c["seed"],
                    "overall_acc": c["overall_acc"],
                    "hard_neg_acc": c["hard_neg_acc"],
                    "K8": c["K8"],
                    "K12": c["K12"],
                    "K16": c["K16"],
                    "prereg_pass": c["prereg_pass"],
                    "within_model_head_disagreement_rate": c[
                        "within_model_head_disagreement_rate"
                    ],
                    "within_model_uncertainty": c["within_model_uncertainty"],
                }
                for c in hyp_per_seed
            ],
            "mean_T16": hyp_mean_t16,
            "matched_ood_mean_by_T": {
                str(T): {
                    "overall": _mean_std_arm(hyp_matched_by_T[str(T)], "overall_acc"),
                    "hard_neg": _mean_std_arm(hyp_matched_by_T[str(T)], "hard_neg_acc"),
                    "K16": _mean_std_arm(hyp_matched_by_T[str(T)], "K16"),
                }
                for T in T_VALUES
            }
            if hyp_per_seed
            else {},
        },
        "references": {
            "pr14_seed0_T16": pr14_ref,
            "ensemble_prob_mean_T16": audit_ens.get("ensemble_prob_mean"),
            "ensemble_singles_mean_T16": audit_ens.get("singles_mean"),
            "swa_bag_prob_mean_T16": audit_swa.get("ensemble_prob_mean")
            if not audit_swa.get("skipped")
            else None,
        },
        "prereg_floors_pass": floors_pass,
        "audit_verdict": audit_verdict,
        "verdict": train_verdict,
        "open_status": open_status,
        "residue": residue,
        "prefer_corridor": (
            "Prefer #14 MEASURE_STILL + #22 ens overlay if multi-hyp misses floors; "
            "audit diagnoses whether #22 lift rides on disagreement. "
            "Select/curriculum CLOSED. Distill STOP. SWA MEASURE. §22 not widened."
        ),
        "policy": (
            "PASS_CANDIDATE → floors; science_open=false. "
            "MEASURE → ge #14 seed0 on HN or K16. STOP → both <. "
            "Audit separate. Sheaf unsupervised ignored."
        ),
        "elapsed_sec": elapsed,
    }

    # Keep full train histories in artifact (useful) — already in train_history_by_seed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-epi] wrote {out_path} verdict={train_verdict} audit={audit_verdict} "
        f"science_open=false elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=CYCLE)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--n-hyp", type=int, default=N_HYP)
    p.add_argument("--lambda-js", type=float, default=LAMBDA_JS)
    p.add_argument("--skip-train", action="store_true", help="Audit only / load existing ckpts")
    p.add_argument(
        "--train-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_TRAIN_SEEDS),
        help="Comma-separated train seeds",
    )
    args = p.parse_args(argv)
    train_seeds = tuple(int(x) for x in args.train_seeds.split(",") if x.strip() != "")
    report = run_cycle(
        id_data=args.id_data,
        ood_data=args.ood_data,
        out_path=args.out,
        epochs=args.epochs,
        n_hyp=args.n_hyp,
        lambda_js=args.lambda_js,
        skip_train=args.skip_train,
        train_seeds=train_seeds,
    )
    print(json.dumps({"verdict": report["verdict"], "audit_verdict": report["audit_verdict"], "science_open": report["science_open"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
