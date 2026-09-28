"""CYCLE_STALK_BAG_DIVERSITY — MEASURE DGE-style independent stalk bag.

Independent hard-A stalk members (separate full params) with:
  (1) data bootstrap of ID train (with replacement, size=|train|);
  (2) graph-subspace edge keep-mask (EDGE_KEEP_P) fixed per member/example.
Select/eval use full hard A (no drop). Eval primary = bag prob_mean.
Report pairwise disagreement vs #22 (#14/#18 freeze). NOT multi-hyp, NOT soft
distill, NOT SWA-only.

Prereg floors (bag ens @ matched-OOD T16): hard-neg≥0.95 AND K16≥0.75.
  PASS_CANDIDATE → floors PASS + lift HN and K16 vs bag singles mean; science_open=false
  MEASURE_LIFT   → ≥0.01 abs lift on overall/HN/K16 vs bag singles mean
  STOP_NO_LIFT   → no such lift

Usage::

    python -m reachability_gen.run_stalk_bag_diversity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.encode import encode_instance, parse_instance
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
from reachability_gen.run_stalk_epistemic_disagreement import (
    _pairwise_disagreement_rate,
)
from reachability_gen.run_stalk_multi_seed_reconfirm import _eval_all_T, _std
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    SECONDARY_AGGS,
    _aggregate_preds,
    _ckpt_for_seed as _ref22_ckpt_for_seed,
    _collect_member_logits,
    _eval_ensemble_methods,
    _load_model_from_ckpt,
    _metrics_from_preds,
)
from reachability_gen.run_stalk_stabilize_multi_seed import LR_MAX, LR_MIN, _cosine_lr
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_BAG_DIVERSITY"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_bag_diversity.json")
DEFAULT_MEMBER_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_REF22_SEEDS = tuple(range(10))
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16

DEFAULT_EPOCHS = 60
DEFAULT_CLIP_H = DEFAULT_CLIP
PREREG_HARD_NEG = 0.95
PREREG_K16 = 0.75
LIFT_EPS = 0.01
EDGE_KEEP_P = 0.75  # graph-subspace: keep each off-diag edge w/ this prob
PR14_REF_SEED = 0


def _bag_ckpt(seed: int) -> Path:
    return Path(f"artifacts/fractal_core_stalk_bag_diversity_seed{seed}_best.pt")


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


def _stable_unit_hash(*parts: Any) -> float:
    """Deterministic U[0,1) from parts (for edge keep decisions)."""
    h = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    # 24 hex chars → int → [0,1)
    return int(h[:12], 16) / float(16**12)


def _bootstrap_indices(n: int, seed: int) -> list[int]:
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed) + 17_001)
    idx = torch.randint(0, n, (n,), generator=g)
    return [int(i) for i in idx.tolist()]


def _subgraph_row(row: dict[str, Any], *, member_seed: int, keep_p: float) -> dict[str, Any]:
    """Return a shallow copy with encoding rewritten to a fixed edge subspace."""
    enc = str(row["encoding"])
    n, edges, s, t = parse_instance(enc)
    edge_hash = row.get("edge_hash", "")
    kept: list[tuple[int, int]] = []
    for u, v in edges:
        u_i, v_i = int(u), int(v)
        if u_i == v_i:
            kept.append((u_i, v_i))
            continue
        u01 = _stable_unit_hash(member_seed, edge_hash, u_i, v_i, "bag_subspace")
        if u01 < keep_p:
            kept.append((u_i, v_i))
    out = dict(row)
    out["encoding"] = encode_instance(n, kept, s, t)
    out["bag_n_edges_orig"] = len(edges)
    out["bag_n_edges_kept"] = len(kept)
    out["bag_member_seed"] = member_seed
    return out


def build_member_train(
    train: list[dict[str, Any]],
    *,
    member_seed: int,
    keep_p: float = EDGE_KEEP_P,
) -> dict[str, Any]:
    """Bootstrap + graph-subspace rewrite for one bag member."""
    idx = _bootstrap_indices(len(train), member_seed)
    boot = [train[i] for i in idx]
    subspaced = [_subgraph_row(r, member_seed=member_seed, keep_p=keep_p) for r in boot]
    n_orig = sum(int(r["bag_n_edges_orig"]) for r in subspaced)
    n_kept = sum(int(r["bag_n_edges_kept"]) for r in subspaced)
    return {
        "rows": subspaced,
        "bootstrap_indices_head": idx[:16],
        "n_rows": len(subspaced),
        "edge_keep_p": keep_p,
        "mean_edge_keep_frac": (n_kept / n_orig) if n_orig else float("nan"),
        "n_edges_orig_sum": n_orig,
        "n_edges_kept_sum": n_kept,
    }


def train_bag_member(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
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
    ckpt_path: Path,
    keep_p: float = EDGE_KEEP_P,
) -> dict[str, Any]:
    """Train one independent stalk on bootstrap+subspace train; select on full val."""
    import torch

    from reachability_gen.train.fractal_trainer import FractalTrainer

    torch.manual_seed(seed)
    bag_meta = build_member_train(train_rows, member_seed=seed, keep_p=keep_p)
    train_bag = bag_meta["rows"]

    model = FractalCore(
        d=d,
        T=T,
        n_heads=_n_heads(d),
        mlp_expansion=mlp_expansion,
        max_nodes=max_nodes,
        max_T=max(T, max(DISCRETE_T_VALUES)),
        use_tau=True,
        apply_cycle_rmsnorm=True,
    )
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    trainer = FractalTrainer(
        model,
        lr=lr_max,
        weight_decay=0.01,
        grad_clip=grad_clip,
        adaptive_halt=False,
    )
    param_count = model.param_count()
    print(
        f"[stalk-bag] seed={seed} params={param_count} parity_ok={parity['within_5pct']} "
        f"boot_n={bag_meta['n_rows']} edge_keep≈{bag_meta['mean_edge_keep_frac']:.3f} "
        f"d={d} T={T} lr={lr_max}→{lr_min} cosine clip={grad_clip}",
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
        for pg in trainer.opt.param_groups:
            pg["lr"] = lr

        order = torch.randperm(len(train_bag)).tolist()
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        for start in range(0, len(train_bag), batch_size):
            idx = order[start : start + batch_size]
            batch_rows = [train_bag[i] for i in idx]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            loss, acc = trainer.train_step(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                batch["labels"],
                T=T,
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            n_steps += 1
            if trainer.last_pre_clip_grad_norm >= grad_clip:
                n_sat += 1

        train_loss = _mean(epoch_losses)
        train_acc = _mean(epoch_accs)
        # Selection: FULL hard-A ID val @ T16 (no edge drop; no OOD peek)
        sel_stats = _eval_split(model, val_rows, max_nodes=max_nodes, T=FOCUS_T)
        sel_overall = float(sel_stats["overall_acc"])
        sel_hard = float(sel_stats["hard_neg_acc"])
        val_ttrain = _eval_split(model, val_rows, max_nodes=max_nodes, T=T)
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
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        train_hist.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc_T_train": ov_ttrain,
                "sel_hard_neg_T16": sel_hard,
                "sel_overall_T16": sel_overall,
                "sel_joint_T16": joint,
                "selected": improved,
            }
        )
        print(
            f"[stalk-bag] seed={seed} epoch {epoch}/{epochs} lr={lr:.5g} "
            f"train_acc={train_acc:.4f} val@T{T}={ov_ttrain:.4f} "
            f"sel@T16 HN={sel_hard:.4f} ov={sel_overall:.4f} "
            f"(best@ep{best_epoch} joint={best_sel.get('joint_score_T16_val', float('nan')):.4f})",
            file=sys.stderr,
        )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    hparams = {
        "d": d,
        "T": T,
        "mlp_expansion": mlp_expansion,
        "max_nodes": max_nodes,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "lr_schedule": "cosine",
        "epochs": epochs,
        "grad_clip": grad_clip,
        "edge_keep_p": keep_p,
        "diversity": "bootstrap+graph_subspace (DGE-style separate params)",
        "adaptive_halt": False,
        "local_potential": True,
        "broadcast_c": False,
    }
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(
            {
                "epoch": best_epoch,
                "val_acc": best_sel.get("overall_acc_T16_val"),
                "sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
                "sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
                "state_dict": best_state,
                "arm": f"fractal-stalk-bag-diversity-T{T}-d{d}-mlp{mlp_expansion}",
                "science_open": False,
                "cycle": CYCLE,
                "member_seed": seed,
                "bag_meta": {
                    k: bag_meta[k]
                    for k in (
                        "n_rows",
                        "edge_keep_p",
                        "mean_edge_keep_frac",
                        "n_edges_orig_sum",
                        "n_edges_kept_sum",
                        "bootstrap_indices_head",
                    )
                },
                "selection_rule": (
                    "lexicographic (0.5*HN+0.5*overall @ T16 ID-val full hard-A, "
                    "overall, HN, -epoch); matched-OOD never used for selection; "
                    "train used bootstrap+edge-subspace only"
                ),
                "hparams": hparams,
            },
            ckpt_path,
        )

    return {
        "arm": f"fractal-stalk-bag-diversity-T{T}-d{d}",
        "param_count": param_count,
        "param_parity": parity,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
        "best_sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
        "best_sel_joint_T16_val": best_sel.get("joint_score_T16_val"),
        "selection_rule": (
            "lexicographic (0.5*HN+0.5*overall @ T16 ID-val full hard-A, "
            "overall, HN, -epoch)"
        ),
        "bag_meta": {
            k: bag_meta[k]
            for k in (
                "n_rows",
                "edge_keep_p",
                "mean_edge_keep_frac",
                "n_edges_orig_sum",
                "n_edges_kept_sum",
                "bootstrap_indices_head",
            )
        },
        "train_history": train_hist,
        "val_by_hop": val_by_hop_final,
        "grad_clip": grad_clip,
        "grad_clip_sat_rate": n_sat / n_steps if n_steps else float("nan"),
        "lr_max": lr_max,
        "lr_min": lr_min,
        "lr_schedule": "cosine",
        "d": d,
        "T": T,
        "mlp_expansion": mlp_expansion,
        "max_nodes": max_nodes,
        "adaptive_halt": False,
        "local_potential": True,
        "broadcast_c": False,
        "checkpoint_path": str(ckpt_path),
        "science_open": False,
        "member_seed": seed,
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    member_seeds: tuple[int, ...] = DEFAULT_MEMBER_SEEDS,
    ref22_seeds: tuple[int, ...] = DEFAULT_REF22_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    grad_clip: float = DEFAULT_CLIP_H,
    keep_p: float = EDGE_KEEP_P,
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

    per_seed_train: list[dict[str, Any]] = []
    for seed in member_seeds:
        ckpt = _bag_ckpt(seed)
        if skip_train and ckpt.exists():
            print(f"[stalk-bag] skip_train seed={seed} using {ckpt}", file=sys.stderr)
            per_seed_train.append(
                {
                    "seed": seed,
                    "checkpoint_path": str(ckpt),
                    "retrained": False,
                    "mode": "existing",
                }
            )
            continue
        print(f"[stalk-bag] === member seed {seed} train (bootstrap+subspace) ===", file=sys.stderr)
        tr = train_bag_member(
            train,
            val,
            epochs=epochs,
            seed=seed,
            max_nodes=max_nodes,
            ckpt_path=ckpt,
            lr_max=lr_max,
            lr_min=lr_min,
            grad_clip=grad_clip,
            keep_p=keep_p,
        )
        per_seed_train.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "retrained": True,
                "mode": "trained",
                "best_epoch": tr.get("best_epoch"),
                "best_sel_hard_neg_T16_val": tr.get("best_sel_hard_neg_T16_val"),
                "best_sel_overall_T16_val": tr.get("best_sel_overall_T16_val"),
                "bag_meta": tr.get("bag_meta"),
                "train_history_len": len(tr.get("train_history") or []),
            }
        )

    # Load bag models
    bag_models = []
    bag_seeds_loaded: list[int] = []
    for seed in member_seeds:
        ckpt = _bag_ckpt(seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing bag ckpt: {ckpt}")
        model, _blob = _load_model_from_ckpt(ckpt, max_nodes)
        bag_models.append(model)
        bag_seeds_loaded.append(seed)

    # Load #22 reference members
    ref_models = []
    ref_seeds_loaded: list[int] = []
    ref_missing: list[int] = []
    for seed in ref22_seeds:
        ckpt = _ref22_ckpt_for_seed(seed)
        if not ckpt.exists():
            ref_missing.append(seed)
            continue
        model, _blob = _load_model_from_ckpt(ckpt, max_nodes)
        ref_models.append(model)
        ref_seeds_loaded.append(seed)

    print(
        f"[stalk-bag] eval bag n={len(bag_models)} vs #22 ref n={len(ref_models)} "
        f"(missing ref seeds={ref_missing})",
        file=sys.stderr,
    )

    # Ensemble eval across T
    bag_by_T: dict[str, Any] = {}
    for T in T_VALUES:
        bag_by_T[f"T{T}"] = _eval_ensemble_methods(
            bag_models, ood_rows, T=T, max_nodes=max_nodes
        )

    ref_by_T: dict[str, Any] = {}
    if ref_models:
        for T in T_VALUES:
            ref_by_T[f"T{T}"] = _eval_ensemble_methods(
                ref_models, ood_rows, T=T, max_nodes=max_nodes
            )

    bag_t16 = bag_by_T[f"T{FOCUS_T}"]
    primary = bag_t16[PRIMARY_AGG]
    singles_mean = bag_t16["singles_mean"]
    deltas = {
        "overall": primary["overall_acc"] - singles_mean["overall_acc"],
        "hard_neg": primary["hard_neg_acc"] - singles_mean["hard_neg_acc"],
        "K16": primary["K16"] - singles_mean["K16"],
    }
    floors_hn = primary["hard_neg_acc"] >= PREREG_HARD_NEG
    floors_k16 = primary["K16"] >= PREREG_K16
    floors_pass = bool(floors_hn and floors_k16)
    lift_both = deltas["hard_neg"] > 0 and deltas["K16"] > 0
    lift_any = any(deltas[k] >= LIFT_EPS for k in ("overall", "hard_neg", "K16"))

    if floors_pass and lift_both:
        verdict = "PASS_CANDIDATE"
    elif lift_any:
        verdict = "MEASURE_LIFT"
    else:
        verdict = "STOP_NO_LIFT"

    # Pairwise disagreement bag vs #22 on T16
    bag_logits, bag_labels, bag_hops = _collect_member_logits(
        bag_models, ood_rows, T=FOCUS_T, max_nodes=max_nodes
    )
    bag_hard = bag_logits.argmax(dim=-1)
    bag_pair = _pairwise_disagreement_rate(bag_hard)

    ref_pair = float("nan")
    ref_primary = None
    if ref_models:
        ref_logits, _, _ = _collect_member_logits(
            ref_models, ood_rows, T=FOCUS_T, max_nodes=max_nodes
        )
        ref_hard = ref_logits.argmax(dim=-1)
        ref_pair = _pairwise_disagreement_rate(ref_hard)
        ref_primary = ref_by_T[f"T{FOCUS_T}"][PRIMARY_AGG]

    disagree_note = (
        "DISAGREE_GE_REF"
        if (bag_pair == bag_pair and ref_pair == ref_pair and bag_pair >= ref_pair)
        else "DISAGREE_LT_REF"
        if (bag_pair == bag_pair and ref_pair == ref_pair)
        else "DISAGREE_REF_MISSING"
    )

    # #14 seed0 reference
    pr14_ckpt = _ref22_ckpt_for_seed(PR14_REF_SEED)
    pr14_metrics = None
    if pr14_ckpt.exists():
        m0, _ = _load_model_from_ckpt(pr14_ckpt, max_nodes)
        ev = _eval_all_T(m0, ood_rows, max_nodes=max_nodes)
        pr14_metrics = ev.get(str(FOCUS_T))

    # Degree-balanced secondary
    deg_bal = build_degree_balanced_eval(ood_rows, seed=0)
    deg_rows = deg_bal["rows"]
    deg_bag = _eval_ensemble_methods(
        bag_models, deg_rows, T=FOCUS_T, max_nodes=max_nodes
    )

    # Per-member singles table at T16
    per_member = []
    for i, seed in enumerate(bag_seeds_loaded):
        s = bag_t16["singles"][i]
        per_member.append(
            {
                "seed": seed,
                "overall_acc": s["overall_acc"],
                "hard_neg_acc": s["hard_neg_acc"],
                "K8": s["K8"],
                "K12": s["K12"],
                "K16": s["K16"],
                "prereg_pass": bool(
                    s["hard_neg_acc"] >= PREREG_HARD_NEG and s["K16"] >= PREREG_K16
                ),
            }
        )
    n_member_pass = sum(1 for m in per_member if m["prereg_pass"])

    elapsed = time.time() - t0
    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "open_status": "science_open_false_not_widened",
        "verdict": verdict,
        "disagree_diagnostic": disagree_note,
        "prereg_floors_pass": floors_pass,
        "base_sha": _git_sha(),
        "elapsed_sec": elapsed,
        "architecture": {
            "name": "FractalCore hard-A stalk (independent bag members)",
            "local_potential": True,
            "broadcast_c": False,
            "adaptive_halt": False,
            "param_count": bag_models[0].param_count() if bag_models else None,
            "separate_params_per_member": True,
            "not_multi_hyp_heads": True,
            "not_soft_distill": True,
            "not_swa_only": True,
            "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
        },
        "datasets": {
            "id": str(id_data),
            "ood": str(ood_data),
            "n_id_train": len(train),
            "n_id_val": len(val),
            "n_ood": len(ood_rows),
        },
        "frozen_recipe": {
            "select": "lex (0.5*HN+0.5*overall, overall, HN, -epoch) @ ID-val T16 full hard-A",
            "epochs": epochs,
            "lr": f"{lr_max}→{lr_min} cosine",
            "grad_clip": grad_clip,
            "T_train": DEFAULT_T,
            "edge_keep_p_train_only": keep_p,
            "bootstrap": True,
        },
        "knobs": {
            "member_seeds": list(member_seeds),
            "n_members": len(member_seeds),
            "edge_keep_p": keep_p,
            "diversity": "bootstrap+graph_subspace",
            "primary_agg": PRIMARY_AGG,
            "secondary_aggs": list(SECONDARY_AGGS),
            "skip_train": skip_train,
        },
        "prereg": {
            "hard_neg_floor": PREREG_HARD_NEG,
            "K16_floor": PREREG_K16,
            "lift_eps": LIFT_EPS,
            "floors_apply_to": "bag_ensemble_prob_mean_T16",
        },
        "T16_primary": {
            "method": PRIMARY_AGG,
            "overall_acc": primary["overall_acc"],
            "hard_neg_acc": primary["hard_neg_acc"],
            "K8": primary["K8"],
            "K12": primary["K12"],
            "K16": primary["K16"],
            "floors_hard_neg_pass": floors_hn,
            "floors_K16_pass": floors_k16,
        },
        "T16_singles_mean": singles_mean,
        "T16_deltas_vs_singles_mean": deltas,
        "T16_secondary": {
            m: {
                "overall_acc": bag_t16[m]["overall_acc"],
                "hard_neg_acc": bag_t16[m]["hard_neg_acc"],
                "K16": bag_t16[m]["K16"],
            }
            for m in SECONDARY_AGGS
        },
        "T16_per_member": per_member,
        "T16_member_prereg_pass_count": n_member_pass,
        "pairwise_disagreement": {
            "bag_T16": bag_pair,
            "ref22_T16": ref_pair,
            "diagnostic": disagree_note,
            "bag_n_members": len(bag_models),
            "ref22_n_members": len(ref_models),
            "ref22_seeds": ref_seeds_loaded,
            "ref22_missing_seeds": ref_missing,
        },
        "ref22_T16_primary": ref_primary,
        "pr14_seed0_T16": pr14_metrics,
        "causal_horizon_bag": {
            f"T{T}": {
                "ens": {
                    "overall_acc": bag_by_T[f"T{T}"][PRIMARY_AGG]["overall_acc"],
                    "hard_neg_acc": bag_by_T[f"T{T}"][PRIMARY_AGG]["hard_neg_acc"],
                    "K16": bag_by_T[f"T{T}"][PRIMARY_AGG]["K16"],
                },
                "singles_mean": bag_by_T[f"T{T}"]["singles_mean"],
            }
            for T in T_VALUES
        },
        "degree_balanced_T16": (
            {
                "n": deg_bag["n"] if deg_bag else 0,
                PRIMARY_AGG: deg_bag[PRIMARY_AGG] if deg_bag else None,
                "singles_mean": deg_bag["singles_mean"] if deg_bag else None,
            }
            if deg_bag
            else {"skipped": True}
        ),
        "per_seed_train": per_seed_train,
        "policy": {
            "science_open_harness": False,
            "widen_section_22": False,
            "select_curriculum": "CLOSED",
            "not_multi_hyp": True,
            "not_soft_distill": True,
            "not_swa_only": True,
        },
        "prefer_corridor": (
            "Prefer #14 MEASURE_STILL + #22 ens overlay unless bag PASS_CANDIDATE "
            "and human seal; harness never widens §22."
        ),
        "residue": [
            "Bag diversity vs seed-only #22 pairwise disagree comparison",
            "Independent params + bootstrap/graph-subspace (DGE-style)",
            "Select/curriculum remain CLOSED",
        ],
        "references": {
            "pr14_stabilize": "artifacts/stalk_stabilize_multi_seed.json",
            "pr22_ensemble": "artifacts/stalk_seed_ensemble.json",
            "pr26_epistemic": "artifacts/stalk_epistemic_disagreement.json",
            "cycle_note": "docs/CYCLE_STALK_BAG_DIVERSITY.md",
        },
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "epochs": epochs,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "grad_clip": grad_clip,
            "edge_keep_p": keep_p,
            "member_seeds": list(member_seeds),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-bag] wrote {out_path} verdict={verdict} disagree={disagree_note} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=CYCLE)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_MEMBER_SEEDS),
        help="Bag member seeds (default 0..4)",
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr-max", type=float, default=LR_MAX)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CLIP_H)
    p.add_argument("--edge-keep-p", type=float, default=EDGE_KEEP_P)
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing bag ckpts (eval-only)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    report = run_cycle(
        id_data=args.id_data,
        ood_data=args.ood_data,
        out_path=args.out,
        member_seeds=tuple(args.seeds),
        epochs=args.epochs,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        grad_clip=args.grad_clip,
        keep_p=args.edge_keep_p,
        skip_train=args.skip_train,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "disagree_diagnostic": report["disagree_diagnostic"],
                "science_open": report["science_open"],
                "prereg_floors_pass": report["prereg_floors_pass"],
                "T16_primary": report["T16_primary"],
                "T16_singles_mean": report["T16_singles_mean"],
                "T16_deltas_vs_singles_mean": report["T16_deltas_vs_singles_mean"],
                "pairwise_disagreement": report["pairwise_disagreement"],
                "ref22_T16_primary": report["ref22_T16_primary"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
