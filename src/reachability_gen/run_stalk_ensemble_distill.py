"""CYCLE_STALK_ENSEMBLE_DISTILL — MEASURE distill frozen ensemble → one stalk student.

Teacher = frozen #14/#18 hard-A stalk ckpts; soft target = prob_mean (mean softmax).
Student = one sealed hard-A FractalCore stalk. Select = frozen #14 0.5 HN + 0.5 overall
@ ID-val T16 (CLOSED — no new select/upsample). Multi-seed students 0,1,2 if cheap.

Prereg floors (student mean @ matched-OOD T16): hard-neg≥0.95 AND K16≥0.75.
  PASS_CANDIDATE → floors PASS; science_open=false
  MEASURE         → miss floor(s) but mean HN or K16 ≥ #14 seed0 (same-run)
  STOP            → mean HN and K16 both < #14 seed0

Metaphor: map≠location (ensemble map → student location); negatives=mirror (HN floor).

Usage::

    python -m reachability_gen.run_stalk_ensemble_distill
"""

from __future__ import annotations

import argparse
import json
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
from reachability_gen.run_stalk_multi_seed_reconfirm import _eval_all_T, _std
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    _ckpt_for_seed,
    _eval_ensemble_methods,
    _load_model_from_ckpt,
)
from reachability_gen.run_stalk_stabilize_multi_seed import LR_MAX, LR_MIN, _cosine_lr
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_ENSEMBLE_DISTILL"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_ensemble_distill.json")
DEFAULT_TEACHER_SEEDS = tuple(range(10))  # 0..9 = #14/#18
DEFAULT_STUDENT_SEEDS = (0, 1, 2)  # multi-seed if cheap
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16

DEFAULT_EPOCHS = 60
DEFAULT_CLIP_H = DEFAULT_CLIP
PREREG_HARD_NEG = 0.95
PREREG_K16 = 0.75
DISTILL_ALPHA = 0.5  # hard CE weight
DISTILL_TAU = 2.0  # temperature
PR14_REF_SEED = 0  # #14 single reference


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


def _student_ckpt(seed: int) -> Path:
    return Path(f"artifacts/fractal_core_stalk_ensemble_distill_seed{seed}_best.pt")


def _teacher_soft_probs_batch(
    teachers: list[Any],
    batch: dict[str, Any],
    *,
    T: int,
    tau: float,
) -> Any:
    """Mean softmax of teacher logits / tau → (B, C) soft targets (no grad)."""
    import torch
    import torch.nn.functional as F

    member_probs = []
    with torch.no_grad():
        for model in teachers:
            model.eval()
            logits, _, _ = model(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                return_halt=True,
                T=T,
                adaptive_halt=False,
            )
            member_probs.append(F.softmax(logits / tau, dim=-1))
        stacked = torch.stack(member_probs, dim=0)  # (M, B, C)
        return stacked.mean(dim=0)


def _cache_teacher_soft_for_rows(
    teachers: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    tau: float,
    batch_size: int = DEFAULT_BATCH,
) -> Any:
    """Precompute teacher soft probs for all rows @ T. Returns (N, C) float tensor."""
    import torch

    chunks: list[Any] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
        soft = _teacher_soft_probs_batch(teachers, batch, T=T, tau=tau)
        chunks.append(soft.cpu())
    return torch.cat(chunks, dim=0)


def train_student_distill(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    teachers: list[Any],
    *,
    teacher_soft_train: Any,
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
    alpha: float = DISTILL_ALPHA,
    tau: float = DISTILL_TAU,
    ckpt_path: Path,
) -> dict[str, Any]:
    """Distill student from cached teacher soft labels; #14 select on ID-val T16."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW

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
    opt = AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    param_count = model.param_count()
    print(
        f"[stalk-ens-distill] seed={seed} params={param_count} "
        f"parity_ok={parity['within_5pct']} α={alpha} τ={tau} "
        f"lr={lr_max}→{lr_min} cosine",
        file=sys.stderr,
    )

    best_key: tuple[float, float, float, int] = (-1.0, -1.0, -1.0, 0)
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    best_sel: dict[str, float] = {}
    train_hist: list[dict[str, Any]] = []
    n_sat = 0
    n_steps = 0

    for epoch in range(1, epochs + 1):
        lr = _cosine_lr(epoch, epochs, lr_max, lr_min)
        for pg in opt.param_groups:
            pg["lr"] = lr

        order = torch.randperm(len(train)).tolist()
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        epoch_ce: list[float] = []
        epoch_kl: list[float] = []
        model.train()
        for start in range(0, len(train), batch_size):
            idx = order[start : start + batch_size]
            batch_rows = [train[i] for i in idx]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            soft = teacher_soft_train[idx].to(batch["labels"].device)

            opt.zero_grad(set_to_none=True)
            logits, _, _ = model(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                return_halt=True,
                T=T,
                adaptive_halt=False,
            )
            hard_loss = ce(logits, batch["labels"])
            log_p = F.log_softmax(logits / tau, dim=-1)
            kl = F.kl_div(log_p, soft, reduction="batchmean") * (tau * tau)
            loss = alpha * hard_loss + (1.0 - alpha) * kl
            loss.backward()
            pre_clip = clip_grad_norm_(model.parameters(), grad_clip)
            n_steps += 1
            if float(pre_clip.item() if hasattr(pre_clip, "item") else pre_clip) >= grad_clip:
                n_sat += 1
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == batch["labels"]).float().mean()
            epoch_losses.append(float(loss.item()))
            epoch_accs.append(float(acc.item()))
            epoch_ce.append(float(hard_loss.item()))
            epoch_kl.append(float(kl.item()))

        train_loss = _mean(epoch_losses)
        train_acc = _mean(epoch_accs)
        sel_stats = _eval_split(model, val, max_nodes=max_nodes, T=FOCUS_T)
        sel_overall = float(sel_stats["overall_acc"])
        sel_hard = float(sel_stats["hard_neg_acc"])
        val_ttrain = _eval_split(model, val, max_nodes=max_nodes, T=T)
        ov_ttrain = float(val_ttrain["overall_acc"])

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
                "train_ce": _mean(epoch_ce),
                "train_kl": _mean(epoch_kl),
                "train_acc": train_acc,
                "val_acc_T_train": ov_ttrain,
                "sel_hard_neg_T16": sel_hard,
                "sel_overall_T16": sel_overall,
                "sel_joint_T16": joint,
                "selected": improved,
            }
        )
        print(
            f"[stalk-ens-distill] seed={seed} epoch {epoch}/{epochs} lr={lr:.5g} "
            f"loss={train_loss:.4f} ce={_mean(epoch_ce):.4f} kl={_mean(epoch_kl):.4f} "
            f"acc={train_acc:.4f} val@T{T}={ov_ttrain:.4f} "
            f"sel@T16 HN={sel_hard:.4f} ov={sel_overall:.4f} "
            f"(best@ep{best_epoch} joint={best_sel.get('joint_score_T16_val', float('nan')):.4f})",
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
                "state_dict": best_state,
                "arm": f"fractal-stalk-ensemble-distill-T{T}-d{d}-mlp{mlp_expansion}",
                "science_open": False,
                "cycle": CYCLE,
                "selection_rule": (
                    "lexicographic (0.5*HN+0.5*overall @ T16 ID-val, overall, HN, -epoch); "
                    "matched-OOD never used for selection; frozen #14 select"
                ),
                "distill": {"alpha": alpha, "tau": tau, "teacher": "prob_mean_ensemble"},
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
                },
            },
            ckpt_path,
        )

    return {
        "arm": f"fractal-stalk-ensemble-distill-T{T}-d{d}-mlp{mlp_expansion}",
        "param_count": param_count,
        "param_parity": parity,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
        "best_sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
        "best_sel_joint_T16_val": best_sel.get("joint_score_T16_val"),
        "selection_rule": (
            "lexicographic (0.5*HN+0.5*overall @ T16 ID-val, overall, HN, -epoch)"
        ),
        "train_history": train_hist,
        "grad_clip_sat_rate": n_sat / n_steps if n_steps else float("nan"),
        "lr_max": lr_max,
        "lr_min": lr_min,
        "alpha": alpha,
        "tau": tau,
        "checkpoint_path": str(ckpt_path),
        "science_open": False,
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    teacher_seeds: tuple[int, ...] = DEFAULT_TEACHER_SEEDS,
    student_seeds: tuple[int, ...] = DEFAULT_STUDENT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    grad_clip: float = DEFAULT_CLIP_H,
    alpha: float = DISTILL_ALPHA,
    tau: float = DISTILL_TAU,
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

    # Load frozen teacher ensemble
    teacher_meta: list[dict[str, Any]] = []
    teachers: list[Any] = []
    for seed in teacher_seeds:
        ckpt = _ckpt_for_seed(seed)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Teacher ckpt missing for seed={seed}: {ckpt} "
                "(fill via #22 policy first; this cycle does not retrain teachers)"
            )
        model, blob = _load_model_from_ckpt(ckpt, max_nodes)
        teacher_meta.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "source": "pr14" if seed <= 4 else "pr18",
                "best_epoch": blob.get("epoch"),
                "param_count": model.param_count(),
            }
        )
        teachers.append(model)
    print(
        f"[stalk-ens-distill] loaded {len(teachers)} frozen teachers; "
        f"caching soft labels on ID train @ T={DEFAULT_T}",
        file=sys.stderr,
    )
    teacher_soft_train = _cache_teacher_soft_for_rows(
        teachers, train, T=DEFAULT_T, max_nodes=max_nodes, tau=tau
    )

    # Train students
    per_seed: list[dict[str, Any]] = []
    for seed in student_seeds:
        ckpt = _student_ckpt(seed)
        print(f"[stalk-ens-distill] train student seed={seed}", file=sys.stderr)
        tr = train_student_distill(
            train,
            val,
            teachers,
            teacher_soft_train=teacher_soft_train,
            epochs=epochs,
            seed=seed,
            max_nodes=max_nodes,
            ckpt_path=ckpt,
            lr_max=lr_max,
            lr_min=lr_min,
            grad_clip=grad_clip,
            alpha=alpha,
            tau=tau,
        )
        student, _ = _load_model_from_ckpt(ckpt, max_nodes)
        matched = _eval_all_T(student, ood_rows, max_nodes=max_nodes)
        deg = _eval_all_T(student, deg_rows, max_nodes=max_nodes)
        t16 = matched[str(FOCUS_T)]
        hn = float(t16["hard_neg_acc"])
        k16 = float(t16["K16"])
        cell = {
            "seed": seed,
            "checkpoint_path": str(ckpt),
            "train": {
                k: tr[k]
                for k in (
                    "best_epoch",
                    "best_sel_hard_neg_T16_val",
                    "best_sel_overall_T16_val",
                    "best_sel_joint_T16_val",
                    "param_count",
                    "param_parity",
                    "grad_clip_sat_rate",
                    "alpha",
                    "tau",
                )
            },
            "matched_ood_by_T": matched,
            "degree_balanced_ood_by_T": deg,
            "T16": {
                "overall_acc": float(t16["overall_acc"]),
                "hard_neg_acc": hn,
                "K8": float(t16["K8"]),
                "K12": float(t16["K12"]),
                "K16": k16,
            },
            "prereg_pass": bool(
                hn == hn
                and hn >= PREREG_HARD_NEG
                and k16 == k16
                and k16 >= PREREG_K16
            ),
        }
        per_seed.append(cell)
        print(
            f"[stalk-ens-distill] student seed={seed} T16 "
            f"ov={cell['T16']['overall_acc']:.4f} HN={hn:.4f} K16={k16:.4f} "
            f"pass={cell['prereg_pass']}",
            file=sys.stderr,
        )

    # Re-eval teacher ensemble + #14 seed0 on same OOD (fair compare)
    print("[stalk-ens-distill] re-eval teacher ensemble + #14 seed0", file=sys.stderr)
    ens_matched: dict[str, Any] = {}
    ens_deg: dict[str, Any] = {}
    for Tv in T_VALUES:
        ens_matched[str(Tv)] = _eval_ensemble_methods(
            teachers, ood_rows, T=int(Tv), max_nodes=max_nodes, methods=(PRIMARY_AGG,)
        )
        ens_deg[str(Tv)] = _eval_ensemble_methods(
            teachers, deg_rows, T=int(Tv), max_nodes=max_nodes, methods=(PRIMARY_AGG,)
        )
    ens_t16 = ens_matched[str(FOCUS_T)][PRIMARY_AGG]

    # #14 seed0 single + #14 seeds 0..4 mean from teacher singles at T16
    pr14_seeds = [s for s in teacher_seeds if 0 <= s <= 4]
    pr14_seed0 = teachers[teacher_seeds.index(PR14_REF_SEED)]
    pr14_seed0_matched = _eval_all_T(pr14_seed0, ood_rows, max_nodes=max_nodes)
    pr14_seed0_t16 = pr14_seed0_matched[str(FOCUS_T)]

    # Mean of #14 singles from ensemble singles table (same logits as ens eval)
    ens_singles = ens_matched[str(FOCUS_T)]["singles"]
    pr14_indices = [teacher_seeds.index(s) for s in pr14_seeds]
    pr14_singles = [ens_singles[i] for i in pr14_indices]
    pr14_mean = {
        "overall_acc": _mean([s["overall_acc"] for s in pr14_singles]),
        "hard_neg_acc": _mean([s["hard_neg_acc"] for s in pr14_singles]),
        "K16": _mean([s["K16"] for s in pr14_singles]),
        "std_overall": _std([s["overall_acc"] for s in pr14_singles]),
        "std_hard_neg": _std([s["hard_neg_acc"] for s in pr14_singles]),
        "std_K16": _std([s["K16"] for s in pr14_singles]),
        "n": len(pr14_singles),
    }

    student_overalls = [c["T16"]["overall_acc"] for c in per_seed]
    student_hns = [c["T16"]["hard_neg_acc"] for c in per_seed]
    student_k16s = [c["T16"]["K16"] for c in per_seed]
    student_mean = {
        "overall_acc": _mean(student_overalls),
        "hard_neg_acc": _mean(student_hns),
        "K16": _mean(student_k16s),
        "std_overall": _std(student_overalls),
        "std_hard_neg": _std(student_hns),
        "std_K16": _std(student_k16s),
        "n": len(per_seed),
        "n_prereg_pass": sum(1 for c in per_seed if c["prereg_pass"]),
    }

    floors_pass = bool(
        student_mean["hard_neg_acc"] == student_mean["hard_neg_acc"]
        and student_mean["hard_neg_acc"] >= PREREG_HARD_NEG
        and student_mean["K16"] == student_mean["K16"]
        and student_mean["K16"] >= PREREG_K16
    )

    ref_hn = float(pr14_seed0_t16["hard_neg_acc"])
    ref_k16 = float(pr14_seed0_t16["K16"])
    ge_ref_hn = student_mean["hard_neg_acc"] >= ref_hn
    ge_ref_k16 = student_mean["K16"] >= ref_k16

    if floors_pass:
        verdict = "PASS_CANDIDATE"
        residue = (
            f"Student mean PASS_CANDIDATE under ensemble distill (α={alpha}, τ={tau}): "
            f"T16 HN={student_mean['hard_neg_acc']:.4f}≥{PREREG_HARD_NEG}, "
            f"K16={student_mean['K16']:.4f}≥{PREREG_K16} "
            f"(n={student_mean['n']}; seed PASS {student_mean['n_prereg_pass']}/{student_mean['n']}). "
            f"vs ensemble HN={ens_t16['hard_neg_acc']:.4f} K16={ens_t16['K16']:.4f}; "
            f"vs #14 seed0 HN={ref_hn:.4f} K16={ref_k16:.4f}. "
            "FLAG human — harness keeps science_open=false. Select/curriculum stay CLOSED."
        )
        open_status = "PASS_CANDIDATE_SCIENCE_OPEN_FALSE"
    elif ge_ref_hn or ge_ref_k16:
        verdict = "MEASURE"
        residue = (
            f"Student mean MEASURE (partial transfer): "
            f"T16 HN={student_mean['hard_neg_acc']:.4f} K16={student_mean['K16']:.4f} "
            f"(floors_pass={floors_pass}); ≥ #14 seed0 on "
            f"{'HN' if ge_ref_hn else ''}"
            f"{'+' if ge_ref_hn and ge_ref_k16 else ''}"
            f"{'K16' if ge_ref_k16 else ''} "
            f"(seed0 HN={ref_hn:.4f} K16={ref_k16:.4f}). "
            f"ensemble HN={ens_t16['hard_neg_acc']:.4f} K16={ens_t16['K16']:.4f}. "
            "science_open=false."
        )
        open_status = "MEASURE_PARTIAL_TRANSFER"
    else:
        verdict = "STOP"
        residue = (
            f"Student mean STOP (no transfer vs #14 seed0): "
            f"T16 HN={student_mean['hard_neg_acc']:.4f}<{ref_hn:.4f} and "
            f"K16={student_mean['K16']:.4f}<{ref_k16:.4f}. "
            f"ensemble HN={ens_t16['hard_neg_acc']:.4f} K16={ens_t16['K16']:.4f}. "
            "science_open=false."
        )
        open_status = "STOP_NO_DISTILL"

    compare = {
        "student_mean": student_mean,
        "ensemble_prob_mean": {
            "overall_acc": ens_t16["overall_acc"],
            "hard_neg_acc": ens_t16["hard_neg_acc"],
            "K16": ens_t16["K16"],
            "K8": ens_t16["K8"],
            "K12": ens_t16["K12"],
        },
        "pr14_seed0": {
            "overall_acc": float(pr14_seed0_t16["overall_acc"]),
            "hard_neg_acc": ref_hn,
            "K16": ref_k16,
            "K8": float(pr14_seed0_t16["K8"]),
            "K12": float(pr14_seed0_t16["K12"]),
        },
        "pr14_seeds0to4_mean": pr14_mean,
        "deltas_student_minus_ensemble": {
            "overall": student_mean["overall_acc"] - ens_t16["overall_acc"],
            "hard_neg": student_mean["hard_neg_acc"] - ens_t16["hard_neg_acc"],
            "K16": student_mean["K16"] - ens_t16["K16"],
        },
        "deltas_student_minus_pr14_seed0": {
            "overall": student_mean["overall_acc"] - float(pr14_seed0_t16["overall_acc"]),
            "hard_neg": student_mean["hard_neg_acc"] - ref_hn,
            "K16": student_mean["K16"] - ref_k16,
        },
    }

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "base_sha": _git_sha(),
        "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
        "metaphor": {
            "map_ne_location": (
                "Ensemble is a multi-seed map; student is one location. "
                "Distill asks whether the map compresses."
            ),
            "negatives_eq_mirror": (
                "Hard-neg @ T16 mirrors whether the student kept the "
                "ensemble reject boundary."
            ),
        },
        "frozen_recipe": (
            "Distill frozen #14/#18 prob_mean ensemble into one hard-A stalk; "
            "NO new select/upsample; select=#14 0.5/0.5 ID-val T16"
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
        },
        "distill_knobs": {
            "alpha": alpha,
            "tau": tau,
            "teacher_aggregator": PRIMARY_AGG,
            "teacher_seeds": list(teacher_seeds),
            "student_seeds": list(student_seeds),
            "epochs": epochs,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "lr_schedule": "cosine",
            "grad_clip": grad_clip,
            "T_train": DEFAULT_T,
            "selection_rule": "frozen #14 0.5*HN+0.5*overall @ ID-val T16",
            "soft_label_cache": "ID train @ T_train once",
        },
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "batch_size": DEFAULT_BATCH,
            "ff_baseline": FF_BASELINE_PARAMS,
            "matched_to": "PR#14/#18/#22 freeze + PR#12 floors",
        },
        "prereg": {
            "hard_neg_student_mean_ge": PREREG_HARD_NEG,
            "K16_at_T16_student_mean_ge": PREREG_K16,
            "alpha": alpha,
            "tau": tau,
            "student_seeds": list(student_seeds),
            "selection_rule": (
                "Frozen #14 lex select on ID-val T16. Floors on matched-OOD "
                "student mean after distill. No OOD peek."
            ),
            "note": (
                "science_open never self-stamped; PASS_CANDIDATE still leaves "
                "science_open=false. Select/curriculum remain CLOSED."
            ),
        },
        "teachers": teacher_meta,
        "per_seed": per_seed,
        "compare_T16": compare,
        "ensemble_matched_ood_by_T": {
            str(T): {
                "prob_mean": ens_matched[str(T)][PRIMARY_AGG],
                "singles_mean": ens_matched[str(T)]["singles_mean"],
            }
            for T in T_VALUES
        },
        "ensemble_degbal_T16": {
            "prob_mean": {
                "overall_acc": ens_deg[str(FOCUS_T)][PRIMARY_AGG]["overall_acc"],
                "hard_neg_acc": ens_deg[str(FOCUS_T)][PRIMARY_AGG]["hard_neg_acc"],
                "K16": ens_deg[str(FOCUS_T)][PRIMARY_AGG]["K16"],
            }
        },
        "pr14_seed0_matched_ood_by_T": pr14_seed0_matched,
        "degree_balanced_construction": {
            k: v for k, v in deg_bal.items() if k != "rows"
        },
        "datasets": {"id": str(id_data), "matched_ood": str(ood_data)},
        "policy": (
            "PASS_CANDIDATE → student mean floors; science_open=false. "
            "MEASURE → miss floor but ≥ #14 seed0 on HN or K16. "
            "STOP → both HN and K16 < #14 seed0. Sheaf unsupervised ignored."
        ),
        "prereg_floors_pass": floors_pass,
        "verdict": verdict,
        "open_status": open_status,
        "residue": residue,
        "elapsed_sec": time.time() - t0,
        "prefer_corridor": (
            "PR#14 MEASURE_STILL singles; ensemble PASS_CANDIDATE is inference map; "
            "student is distilled location"
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[stalk-ens-distill] wrote {out_path}", file=sys.stderr)
    print(
        f"[stalk-ens-distill] verdict={verdict} science_open=False "
        f"student_mean T16 ov={student_mean['overall_acc']:.4f} "
        f"HN={student_mean['hard_neg_acc']:.4f} K16={student_mean['K16']:.4f} "
        f"ens HN={ens_t16['hard_neg_acc']:.4f} K16={ens_t16['K16']:.4f} "
        f"pr14s0 HN={ref_hn:.4f} K16={ref_k16:.4f}",
        file=sys.stderr,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--teacher-seeds", type=int, nargs="+", default=list(DEFAULT_TEACHER_SEEDS)
    )
    p.add_argument(
        "--student-seeds", type=int, nargs="+", default=list(DEFAULT_STUDENT_SEEDS)
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr-max", type=float, default=LR_MAX)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CLIP_H)
    p.add_argument("--alpha", type=float, default=DISTILL_ALPHA)
    p.add_argument("--tau", type=float, default=DISTILL_TAU)
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
        teacher_seeds=tuple(args.teacher_seeds),
        student_seeds=tuple(args.student_seeds),
        epochs=args.epochs,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        grad_clip=args.grad_clip,
        alpha=args.alpha,
        tau=args.tau,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "open_status": report["open_status"],
                "science_open": report["science_open"],
                "prereg_floors_pass": report["prereg_floors_pass"],
                "compare_T16": report["compare_T16"],
                "n_student_seeds": report["distill_knobs"]["student_seeds"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
