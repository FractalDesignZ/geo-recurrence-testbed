"""CYCLE_STALK_STABILIZE_V3 — MEASURE continue (science_open=false always).

Middle-out from PR #15 STOP_FRAGILE / prefer #14 corridor: supervised/hard A_ij
+ stalk-local FractalCore. 5 seeds, ~60 ep (#14 budget), cosine 1.5e-3→1.5e-4,
clip 2.5. Selection change primarily: equal-weight joint HN + K16 + overall
with K16 from a selection-only long-hop aux (NOT matched-OOD floors).
HN weight = 1/3 ≤ 0.5 (no HN overweight; V2 0.7 rejected).

Prereg (mean): hard-neg≥0.95 AND K16@T16≥0.75; seed-wise goal ≥4/5.
  PASS_CANDIDATE_FOR_OPEN → still science_open=false (human only; no auto-widen)
  MEASURE_STILL / STOP_FRAGILE → document; no widen

Usage::

    python -m reachability_gen.run_stalk_stabilize_v3
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
from reachability_gen.run_stalk_multi_seed_reconfirm import (
    _eval_all_T,
    _hop_acc,
    _load_sealed,
    _std,
    _untrained_control,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_STABILIZE_V3"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_stabilize_v3.json")
DEFAULT_SEALED = Path("artifacts/fractal_core_stalk_gate1_matched_ood.json")
DEFAULT_SELECT_LONGHOP = Path("data/id_select_longhop.jsonl")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16

# Harden V3 (preregistered — see docs/CYCLE_STALK_STABILIZE_V3.md)
# Stay near #14 corridor; change select primarily (include true K16 via select-aux).
DEFAULT_EPOCHS = 60
LR_MAX = 1.5e-3
LR_MIN = 1.5e-4
DEFAULT_CLIP_H = DEFAULT_CLIP  # 2.5
# Equal-weight joint; HN weight MUST be ≤0.5 (V2 0.7 → STOP_FRAGILE)
HN_WEIGHT = 1.0 / 3.0
K16_WEIGHT = 1.0 / 3.0
OVERALL_WEIGHT = 1.0 / 3.0
PREREG_HARD_NEG = 0.95
PREREG_K16 = 0.75
SEED_PASS_GOAL = 4  # of 5
SEED_PASS_STRETCH = 5


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


def _cosine_lr(epoch: int, epochs: int, lr_max: float, lr_min: float) -> float:
    """Epoch-indexed cosine anneal; epoch in 1..epochs."""
    if epochs <= 1:
        return lr_max
    t = (epoch - 1) / (epochs - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))


def train_fractal_id2k_harden(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    select_longhop: list[dict[str, Any]],
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
) -> dict[str, Any]:
    """Train with cosine LR + equal-weight HN+K16+overall selection (K16 via select-aux)."""
    import torch

    from reachability_gen.train.fractal_trainer import FractalTrainer

    torch.manual_seed(seed)
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
        f"[stalk-stabilize-v3] params={param_count} parity_ok={parity['within_5pct']} "
        f"d={d} mlp={mlp_expansion} T={T} lr={lr_max}→{lr_min} cosine clip={grad_clip}",
        file=sys.stderr,
    )

    # Selection (V3 prereg): joint = (1/3)*HN_id + (1/3)*K16_sel + (1/3)*overall_id
    # K16 from selection-only longhop aux (matched seq_len band; NOT floors OOD).
    # Key = (joint, min(HN,K16), overall, -epoch). HN weight=1/3 ≤0.5 (no overweight).
    # Matched-OOD floors set never used here.
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

        order = torch.randperm(len(train)).tolist()
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        for start in range(0, len(train), batch_size):
            idx = order[start : start + batch_size]
            batch_rows = [train[i] for i in idx]
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
        # Selection eval: ID val @ T=16 for HN/overall + select-longhop for K16
        # (prereg; matched-OOD floors never peeked)
        sel_stats = _eval_split(model, val, max_nodes=max_nodes, T=FOCUS_T)
        sel_overall = float(sel_stats["overall_acc"])
        sel_hard = float(sel_stats["hard_neg_acc"])
        sel_lh = _eval_split(model, select_longhop, max_nodes=max_nodes, T=FOCUS_T)
        sel_k16 = _hop_acc(sel_lh, 16)
        if sel_k16 != sel_k16:  # NaN guard
            sel_k16 = 0.0
        # Also track train-T val for continuity with prior logs
        val_ttrain = _eval_split(model, val, max_nodes=max_nodes, T=T)
        ov_ttrain = float(val_ttrain["overall_acc"])
        val_by_hop_final = sel_stats["by_hop"]

        joint = (
            HN_WEIGHT * sel_hard
            + K16_WEIGHT * sel_k16
            + OVERALL_WEIGHT * sel_overall
        )
        gate_min = min(sel_hard, sel_k16)
        key = (joint, gate_min, sel_overall, -epoch)
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_sel = {
                "hard_neg_acc_T16_val": sel_hard,
                "overall_acc_T16_val": sel_overall,
                "k16_acc_T16_select_longhop": sel_k16,
                "joint_score_T16_val": joint,
                "min_HN_K16": gate_min,
                "hn_weight": HN_WEIGHT,
                "k16_weight": K16_WEIGHT,
                "overall_weight": OVERALL_WEIGHT,
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
                "sel_k16_T16_longhop": sel_k16,
                "sel_joint_T16": joint,
                "sel_min_HN_K16": gate_min,
                "selected": improved,
            }
        )
        print(
            f"[stalk-stabilize-v3] epoch {epoch}/{epochs} lr={lr:.5g} "
            f"train_acc={train_acc:.4f} val@T{T}={ov_ttrain:.4f} "
            f"sel@T16 HN={sel_hard:.4f} K16={sel_k16:.4f} ov={sel_overall:.4f} "
            f"(best@ep{best_epoch} joint={best_sel.get('joint_score_T16_val', float('nan')):.4f} "
            f"minHK={best_sel.get('min_HN_K16', float('nan')):.4f})",
            file=sys.stderr,
        )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(
            {
                "epoch": best_epoch,
                "val_acc": best_sel.get("overall_acc_T16_val"),
                "sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
                "sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
                "sel_k16_T16_select_longhop": best_sel.get("k16_acc_T16_select_longhop"),
                "sel_joint_T16_val": best_sel.get("joint_score_T16_val"),
                "sel_min_HN_K16": best_sel.get("min_HN_K16"),
                "state_dict": best_state,
                "arm": f"fractal-stalk-stabilize-v3-T{T}-d{d}-mlp{mlp_expansion}",
                "science_open": False,
                "cycle": CYCLE,
                "selection_rule": (
                    "lexicographic ((1/3)*HN_id+(1/3)*K16_select_longhop+(1/3)*overall_id @ T16, "
                    "min(HN,K16), overall, -epoch); HN_weight=1/3≤0.5; "
                    "matched-OOD floors never used for selection"
                ),
                "hparams": {
                    "d": d,
                    "T": T,
                    "mlp_expansion": mlp_expansion,
                    "max_nodes": max_nodes,
                    "lr_max": lr_max,
                    "lr_min": lr_min,
                    "lr_schedule": "cosine",
                    "epochs": epochs,
                    "grad_clip": grad_clip,
                    "adaptive_halt": False,
                    "local_potential": True,
                    "broadcast_c": False,
                    "hn_weight": HN_WEIGHT,
                    "k16_weight": K16_WEIGHT,
                    "overall_weight": OVERALL_WEIGHT,
                    "select_longhop": "data/id_select_longhop.jsonl",
                },
            },
            ckpt_path,
        )

    return {
        "arm": f"fractal-stalk-stabilize-v3-T{T}-d{d}-mlp{mlp_expansion}",
        "param_count": param_count,
        "param_parity": parity,
        "epochs": epochs,
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_sel.get("overall_acc_T16_val"),
        "best_sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
        "best_sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
        "selection_rule": (
            "lexicographic ((1/3)*HN_id+(1/3)*K16_select_longhop+(1/3)*overall_id @ T16, "
            "min(HN,K16), overall, -epoch); HN_weight=1/3"
        ),
        "best_sel_joint_T16_val": best_sel.get("joint_score_T16_val"),
        "best_sel_k16_T16_select_longhop": best_sel.get("k16_acc_T16_select_longhop"),
        "best_sel_min_HN_K16": best_sel.get("min_HN_K16"),
        "hn_weight": HN_WEIGHT,
        "k16_weight": K16_WEIGHT,
        "overall_weight": OVERALL_WEIGHT,
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
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    select_longhop_path: Path = DEFAULT_SELECT_LONGHOP,
    out_path: Path = DEFAULT_OUT,
    sealed_path: Path = DEFAULT_SEALED,
    epochs: int = DEFAULT_EPOCHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    grad_clip: float = DEFAULT_CLIP_H,
) -> dict[str, Any]:
    import torch

    t0 = time.time()
    if not id_data.exists():
        raise FileNotFoundError(id_data)
    if not ood_data.exists():
        raise FileNotFoundError(ood_data)
    if not select_longhop_path.exists():
        raise FileNotFoundError(select_longhop_path)

    rows = load_jsonl(id_data)
    train, val = _split_train_val(rows)
    ood_rows = load_jsonl(ood_data)
    select_longhop_raw = load_jsonl(select_longhop_path)
    # Fail-closed: drop any (edge_hash,s,t) that also appears in floors OOD
    # so selection never peeks at matched-OOD evaluation pairs.
    ood_keys = {
        f"{r['edge_hash']}:{r['s']}:{r['t']}" for r in ood_rows
    }
    select_longhop = [
        r
        for r in select_longhop_raw
        if f"{r['edge_hash']}:{r['s']}:{r['t']}" not in ood_keys
    ]
    n_sel_dropped = len(select_longhop_raw) - len(select_longhop)
    if not select_longhop:
        raise RuntimeError("select_longhop empty after floors-overlap filter")
    max_nodes = max(
        DEFAULT_MAX_NODE_ID,
        max(int(r["n"]) for r in rows),
        max(int(r["n"]) for r in ood_rows),
        max(int(r["n"]) for r in select_longhop),
    )

    sealed = _load_sealed(sealed_path)
    deg_bal = build_degree_balanced_eval(ood_rows, seed=0)
    deg_rows = deg_bal["rows"]

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "base_sha": _git_sha(),
        "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
        "architecture": {
            "local_potential": True,
            "broadcast_c": False,
            "soft_ACT": False,
            "hard_A_ij_mask": True,
            "stalk_slot": "s",
            "probe_slot": "t",
            "intermediates": 0,
            "discrete_T_values": list(T_VALUES),
        },
        "harden_knobs": {
            "epochs": epochs,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "lr_schedule": "cosine",
            "grad_clip": grad_clip,
            "n_seeds": len(seeds),
            "hn_weight": HN_WEIGHT,
            "k16_weight": K16_WEIGHT,
            "overall_weight": OVERALL_WEIGHT,
            "select_longhop": str(select_longhop_path),
            "selection": (
                "T16 lexicographic ((1/3)*HN_id+(1/3)*K16_select_longhop+(1/3)*overall_id, "
                "min(HN,K16), overall, -epoch); HN_weight=1/3≤0.5; "
                "K16 from selection-only longhop aux (matched seq_len band); "
                "matched-OOD floors never used for ckpt selection. "
                "V2 HN-heavy 0.7 rejected (STOP_FRAGILE)."
            ),
            "vs_pr14_pr15": {
                "epochs": "60 (same as #14; V2 was 90)",
                "lr": "cosine 1.5e-3→1.5e-4 (same as #14; V2→1e-4)",
                "selection_pr14": "joint 0.5*HN+0.5*overall @ T16 ID-val (no K16)",
                "selection_pr15_v2": "gated 0.7*HN+0.3*overall (ov≥0.85) → STOP_FRAGILE",
                "selection_v3": (
                    "equal-weight HN+K16+overall with K16 from id_select_longhop aux; "
                    "HN_weight=1/3"
                ),
                "seeds_was": [0, 1, 2, 3, 4],
                "prior_best_corridor": "PR#14 MEASURE_STILL 2/5",
                "prior_v2_verdict": "STOP_FRAGILE 1/5 K16 collapse",
            },
        },
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "lr_schedule": "cosine",
            "epochs": epochs,
            "grad_clip": grad_clip,
            "batch_size": DEFAULT_BATCH,
            "adaptive_halt": False,
            "ff_baseline": FF_BASELINE_PARAMS,
            "matched_to": "sealed CYCLE_STALK_LOCALIZATION + PR#12/#14 floors",
            "hn_weight": HN_WEIGHT,
            "k16_weight": K16_WEIGHT,
            "overall_weight": OVERALL_WEIGHT,
            "select_longhop": str(select_longhop_path),
        },
        "prereg": {
            "hard_neg_mean_ge": PREREG_HARD_NEG,
            "K16_at_T16_mean_ge": PREREG_K16,
            "n_seeds": len(seeds),
            "seed_pass_goal_ge": SEED_PASS_GOAL,
            "seed_pass_stretch": SEED_PASS_STRETCH,
            "selection_rule": (
                "Before any matched-OOD floors look: best ckpt = argmax "
                "((1/3)*HN_id+(1/3)*K16_select_longhop+(1/3)*overall_id @ T16, "
                "min(HN,K16), overall, -epoch). "
                "HN_weight=1/3 ≤0.5 (no HN overweight; V2 0.7 forbidden). "
                "K16 from data/id_select_longhop.jsonl (selection-only aux; "
                "matched seq_len band; disjoint from floors set after overlap filter). "
                "Floors evaluated only on matched-OOD after train. "
                "If ≥4/5 → PASS_CANDIDATE_FOR_OPEN only; no auto-widen."
            ),
            "note": (
                "Mean floors match PR #12/#14. science_open never self-stamped; "
                "PASS_CANDIDATE_FOR_OPEN still leaves science_open=false."
            ),
        },
        "sealed_single_seed_OPEN": sealed,
        "seeds": list(seeds),
        "degree_balanced_construction": {
            k: v for k, v in deg_bal.items() if k != "rows"
        },
        "datasets": {
            "id": str(id_data),
            "matched_ood": str(ood_data),
            "select_longhop": str(select_longhop_path),
            "select_longhop_n": len(select_longhop),
            "select_longhop_n_raw": len(select_longhop_raw),
            "select_longhop_floors_overlap_dropped": n_sel_dropped,
            "select_longhop_note": (
                "SELECTION ONLY — not used for final floors; "
                "provides K16 signal missing from ID-val hops≤6; "
                "runtime filter drops any pair also in matched-OOD"
            ),
        },
        "policy": (
            "PASS_CANDIDATE_FOR_OPEN → flag human; do NOT set science_open=true. "
            "MEASURE_STILL / STOP_FRAGILE → append ledger. "
            "Sheaf unsupervised path ignored this cycle. science_open=human only."
        ),
    }

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[stalk-stabilize-v3] === seed {seed} train (harden) ===", file=sys.stderr)
        ckpt = Path(f"artifacts/fractal_core_stalk_stabilize_v3_seed{seed}_best.pt")
        tr = train_fractal_id2k_harden(
            train,
            val,
            select_longhop,
            epochs=epochs,
            seed=seed,
            max_nodes=max_nodes,
            ckpt_path=ckpt,
            lr_max=lr_max,
            lr_min=lr_min,
            grad_clip=grad_clip,
        )
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
        ckpt_blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_blob["state_dict"])
        parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)

        print(f"[stalk-stabilize-v3] seed={seed} OOD eval T∈{T_VALUES}", file=sys.stderr)
        ood_by_T = _eval_all_T(model, ood_rows, max_nodes=max_nodes)
        deg_by_T = _eval_all_T(model, deg_rows, max_nodes=max_nodes)

        print(f"[stalk-stabilize-v3] seed={seed} untrained control", file=sys.stderr)
        untrained = _untrained_control(
            ood_rows, max_nodes=max_nodes, seed=seed, trained_model=model
        )

        t16 = ood_by_T[str(FOCUS_T)]
        hard = float(t16["hard_neg_acc"])
        k16 = float(t16["K16"])
        overall = float(t16["overall_acc"])
        seed_prereg = {
            "hard_neg_ok": hard == hard and hard >= PREREG_HARD_NEG,
            "K16_ok": k16 == k16 and k16 >= PREREG_K16,
            "pass": bool(
                (hard == hard and hard >= PREREG_HARD_NEG)
                and (k16 == k16 and k16 >= PREREG_K16)
            ),
            "observed_hard_neg_T16": hard,
            "observed_K16_T16": k16,
            "observed_overall_T16": overall,
        }
        print(
            f"[stalk-stabilize-v3] seed={seed} T16 overall={overall:.4f} "
            f"hard_neg={hard:.4f} K16={k16:.4f} prereg={seed_prereg['pass']}",
            file=sys.stderr,
        )

        train_slim = {
            k: v
            for k, v in tr.items()
            if k not in ("train_history", "val_by_hop")
        }
        train_slim["train_history_len"] = len(tr.get("train_history") or [])
        train_slim["final_epoch_sel"] = (
            (tr.get("train_history") or [{}])[-1]
            if tr.get("train_history")
            else None
        )

        per_seed.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "param_count": model.param_count(),
                "param_parity": parity,
                "train": train_slim,
                "matched_ood_by_T": ood_by_T,
                "degree_balanced_ood_by_T": deg_by_T,
                "untrained_control": untrained,
                "T16": {
                    "overall_acc": overall,
                    "hard_neg_acc": hard,
                    "K8": float(t16["K8"]),
                    "K12": float(t16["K12"]),
                    "K16": k16,
                },
                "degbal_T16": {
                    "overall_acc": float(deg_by_T[str(FOCUS_T)]["overall_acc"]),
                    "hard_neg_acc": float(deg_by_T[str(FOCUS_T)]["hard_neg_acc"]),
                    "K16": float(deg_by_T[str(FOCUS_T)]["K16"]),
                },
                "untrained_T16": untrained["focus_T16"],
                "prereg": seed_prereg,
                "science_open": False,
            }
        )

    def _agg(getter) -> dict[str, float]:
        xs = [float(getter(r)) for r in per_seed]
        return {"mean": _mean(xs), "std": _std(xs), "values": xs, "n": len(xs)}

    mean_std = {
        "T16_overall": _agg(lambda r: r["T16"]["overall_acc"]),
        "T16_hard_neg": _agg(lambda r: r["T16"]["hard_neg_acc"]),
        "T16_K8": _agg(lambda r: r["T16"]["K8"]),
        "T16_K12": _agg(lambda r: r["T16"]["K12"]),
        "T16_K16": _agg(lambda r: r["T16"]["K16"]),
        "best_sel_hard_neg_val": _agg(
            lambda r: r["train"]["best_sel_hard_neg_T16_val"]
        ),
        "best_sel_k16_select_longhop": _agg(
            lambda r: r["train"].get("best_sel_k16_T16_select_longhop", float("nan"))
        ),
        "degbal_T16_overall": _agg(lambda r: r["degbal_T16"]["overall_acc"]),
        "degbal_T16_hard_neg": _agg(lambda r: r["degbal_T16"]["hard_neg_acc"]),
        "degbal_T16_K16": _agg(lambda r: r["degbal_T16"]["K16"]),
        "untrained_T16_overall": _agg(
            lambda r: r["untrained_T16"]["overall_acc"]
        ),
        "untrained_T16_hard_neg": _agg(
            lambda r: r["untrained_T16"]["hard_neg_acc"]
        ),
        "untrained_vs_trained_agree_T16": _agg(
            lambda r: r["untrained_T16"].get("prediction_agreement", float("nan"))
        ),
    }

    by_T_agg: dict[str, Any] = {}
    for T in T_VALUES:
        by_T_agg[str(T)] = {
            "overall": _agg(
                lambda r, tt=T: r["matched_ood_by_T"][str(tt)]["overall_acc"]
            ),
            "hard_neg": _agg(
                lambda r, tt=T: r["matched_ood_by_T"][str(tt)]["hard_neg_acc"]
            ),
            "K8": _agg(lambda r, tt=T: r["matched_ood_by_T"][str(tt)]["K8"]),
            "K12": _agg(lambda r, tt=T: r["matched_ood_by_T"][str(tt)]["K12"]),
            "K16": _agg(lambda r, tt=T: r["matched_ood_by_T"][str(tt)]["K16"]),
        }

    n_pass = sum(1 for r in per_seed if r["prereg"]["pass"])
    hard_mean = mean_std["T16_hard_neg"]["mean"]
    k16_mean = mean_std["T16_K16"]["mean"]
    floors_mean_pass = bool(
        hard_mean == hard_mean
        and hard_mean >= PREREG_HARD_NEG
        and k16_mean == k16_mean
        and k16_mean >= PREREG_K16
        and len(per_seed) >= 5
    )
    seed_goal_pass = n_pass >= SEED_PASS_GOAL

    if floors_mean_pass and seed_goal_pass:
        verdict = "PASS_CANDIDATE_FOR_OPEN"
        residue = (
            f"Stabilize V3 PASS candidate: mean T16 hard-neg={hard_mean:.4f}≥"
            f"{PREREG_HARD_NEG}, K16={k16_mean:.4f}≥{PREREG_K16}; "
            f"{n_pass}/{len(per_seed)} seeds individual pass (≥{SEED_PASS_GOAL} goal). "
            "FLAG Fractal-1 / human for OPEN review — harness keeps science_open=false."
        )
        open_status = "PASS_CANDIDATE_FOR_OPEN_SCIENCE_OPEN_FALSE"
    elif n_pass <= 1 or (
        k16_mean == k16_mean and k16_mean < 0.5 and hard_mean < PREREG_HARD_NEG
    ):
        verdict = "STOP_FRAGILE"
        residue = (
            f"Stabilize V3 STOP_FRAGILE: hard-neg={hard_mean:.4f}, K16={k16_mean:.4f}; "
            f"{n_pass}/{len(per_seed)} seeds pass. Multi-seed corridor remains fragile "
            "under V3 harden. science_open=false; do not widen."
        )
        open_status = "MEASURE_STOP_FRAGILE"
    else:
        verdict = "MEASURE_STILL"
        means_note = (
            "mean floors PASS"
            if floors_mean_pass
            else (
                f"mean floors MISS (HN={hard_mean:.4f} need ≥{PREREG_HARD_NEG}, "
                f"K16={k16_mean:.4f} need ≥{PREREG_K16})"
            )
        )
        residue = (
            f"Stabilize V3 MEASURE_STILL: {means_note}; "
            f"{n_pass}/{len(per_seed)} seeds pass (goal ≥{SEED_PASS_GOAL}). "
            "V3 harden did not clear full prereg (means AND seed goal). "
            "science_open=false; claim not widened."
        )
        open_status = (
            "MEASURE_STILL_MEANS_PASS_SEED_GOAL_MISS"
            if floors_mean_pass
            else "MEASURE_STILL"
        )

    # Override: floors miss with moderate pass count stays MEASURE_STILL
    # (STOP only when ≤1/5 or deep collapse). Already handled above.

    sealed_cmp = {
        "sealed_T16_overall": sealed.get("T16_overall"),
        "sealed_T16_hard_neg": sealed.get("T16_hard_neg"),
        "sealed_T16_K16": sealed.get("T16_K16"),
        "stabilize_v3_mean_T16_overall": mean_std["T16_overall"]["mean"],
        "stabilize_v3_mean_T16_hard_neg": hard_mean,
        "stabilize_v3_mean_T16_K16": k16_mean,
        "stabilize_v3_std_T16_overall": mean_std["T16_overall"]["std"],
        "stabilize_v3_std_T16_hard_neg": mean_std["T16_hard_neg"]["std"],
        "stabilize_v3_std_T16_K16": mean_std["T16_K16"]["std"],
        "pr14_stabilize_mean_overall": 0.929,
        "pr14_stabilize_mean_hard_neg": 0.957,
        "pr14_stabilize_mean_K16": 0.863,
        "pr14_seed_pass": "2/5",
        "pr15_v2_mean_overall": 0.813,
        "pr15_v2_mean_hard_neg": 0.955,
        "pr15_v2_mean_K16": 0.423,
        "pr15_v2_seed_pass": "1/5",
        "pr12_reconfirm_mean_overall": 0.803,
        "pr12_reconfirm_mean_hard_neg": 0.918,
        "pr12_reconfirm_mean_K16": 0.654,
        "this_cycle_prereg_thresholds": {
            "hard_neg_mean": PREREG_HARD_NEG,
            "K16_mean": PREREG_K16,
            "seed_pass_goal": SEED_PASS_GOAL,
        },
    }

    report.update(
        {
            "per_seed": per_seed,
            "mean_std": mean_std,
            "by_T_mean_std": by_T_agg,
            "vs_sealed_OPEN": sealed_cmp,
            "n_seeds": len(per_seed),
            "n_prereg_pass_individual": n_pass,
            "prereg_mean_pass": floors_mean_pass,
            "prereg_seed_goal_pass": seed_goal_pass,
            "verdict": verdict,
            "open_status": open_status,
            "residue": residue,
            "elapsed_sec": time.time() - t0,
            "science_open": False,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[stalk-stabilize-v3] wrote {out_path}", file=sys.stderr)
    print(
        f"[stalk-stabilize-v3] verdict={verdict} science_open=False "
        f"mean hard_neg={hard_mean:.4f}±{mean_std['T16_hard_neg']['std']:.4f} "
        f"K16={k16_mean:.4f}±{mean_std['T16_K16']['std']:.4f} "
        f"seed_pass={n_pass}/{len(per_seed)}",
        file=sys.stderr,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sealed", type=Path, default=DEFAULT_SEALED)
    p.add_argument("--select-longhop", type=Path, default=DEFAULT_SELECT_LONGHOP)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--lr-max", type=float, default=LR_MAX)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CLIP_H)
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
        select_longhop_path=args.select_longhop,
        out_path=args.out,
        sealed_path=args.sealed,
        epochs=args.epochs,
        seeds=tuple(args.seeds),
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        grad_clip=args.grad_clip,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "open_status": report["open_status"],
                "science_open": report["science_open"],
                "prereg_mean_pass": report["prereg_mean_pass"],
                "n_prereg_pass_individual": report["n_prereg_pass_individual"],
                "mean_std": {
                    k: {"mean": v["mean"], "std": v["std"]}
                    for k, v in report["mean_std"].items()
                    if k.startswith("T16_") or k.startswith("untrained_")
                },
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
