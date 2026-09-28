"""CYCLE_STALK_SWA_PERSIST — MEASURE SWA single-model persistence (science_open=false).

Train sealed hard-A stalk under frozen #14 recipe. Accumulate Polyak/SWA average
of weights from epoch SWA_START onward. Primary = SWA ckpt (one weight vector).
Paired baseline = #14 lex select on same trajectory. Not soft distill (#23 STOP).

Prereg floors (SWA mean @ matched-OOD T16): hard-neg≥0.95 AND K16≥0.75.
  PASS_CANDIDATE → floors PASS; science_open=false (do not widen §22)
  MEASURE         → miss floor(s) but SWA mean HN or K16 ≥ within-run select mean
  STOP            → SWA mean HN and K16 both < within-run select mean

Usage::

    python -m reachability_gen.run_stalk_swa_persist
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

CYCLE = "CYCLE_STALK_SWA_PERSIST"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_swa_persist.json")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TEACHER_SEEDS = tuple(range(10))  # #14/#18 for ensemble compare
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16

DEFAULT_EPOCHS = 60
DEFAULT_CLIP_H = DEFAULT_CLIP
SWA_START = 31  # second half of 60-ep #14 corridor
PREREG_HARD_NEG = 0.95
PREREG_K16 = 0.75
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


def _select_ckpt(seed: int) -> Path:
    return Path(f"artifacts/fractal_core_stalk_swa_persist_seed{seed}_select.pt")


def _clone_state(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _swa_update(swa_state: dict[str, Any], model: Any, n_averaged: int) -> None:
    """In-place Polyak update: swa ← swa + (θ − swa) / n_averaged."""
    with_model = model.state_dict()
    for k, v in with_model.items():
        # Keep buffers / non-float as latest copy (no average)
        if not v.is_floating_point():
            swa_state[k] = v.detach().cpu().clone()
            continue
        prev = swa_state[k]
        cur = v.detach().cpu()
        swa_state[k] = prev + (cur - prev) / float(n_averaged)


def train_fractal_id2k_swa(
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
    swa_start: int = SWA_START,
    swa_ckpt_path: Path,
    select_ckpt_path: Path,
) -> dict[str, Any]:
    """#14 harden + Polyak SWA from swa_start; also keep #14 lex select ckpt."""
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
        f"[stalk-swa] params={param_count} parity_ok={parity['within_5pct']} "
        f"d={d} mlp={mlp_expansion} T={T} lr={lr_max}→{lr_min} cosine "
        f"clip={grad_clip} swa_start={swa_start}",
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

    swa_state: Optional[dict[str, Any]] = None
    n_swa = 0

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

        # Polyak SWA after epoch end (post train+select eval)
        swa_updated = False
        if epoch >= swa_start:
            n_swa += 1
            if swa_state is None:
                swa_state = _clone_state(model)
            else:
                _swa_update(swa_state, model, n_swa)
            swa_updated = True

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
                "swa_updated": swa_updated,
                "n_swa": n_swa,
            }
        )
        print(
            f"[stalk-swa] epoch {epoch}/{epochs} lr={lr:.5g} "
            f"train_acc={train_acc:.4f} val@T{T}={ov_ttrain:.4f} "
            f"sel@T16 HN={sel_hard:.4f} ov={sel_overall:.4f} "
            f"(best@ep{best_epoch} joint={best_sel.get('joint_score_T16_val', float('nan')):.4f}"
            f"{'; swa_n=' + str(n_swa) if swa_updated else ''})",
            file=sys.stderr,
        )

    # ID-val metrics of final SWA (no OOD)
    swa_sel: dict[str, float] = {}
    if swa_state is not None:
        model.load_state_dict(swa_state)
        swa_stats = _eval_split(model, val, max_nodes=max_nodes, T=FOCUS_T)
        swa_sel = {
            "hard_neg_acc_T16_val": float(swa_stats["hard_neg_acc"]),
            "overall_acc_T16_val": float(swa_stats["overall_acc"]),
            "joint_score_T16_val": float(
                0.5 * float(swa_stats["hard_neg_acc"])
                + 0.5 * float(swa_stats["overall_acc"])
            ),
        }

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
        "adaptive_halt": False,
        "local_potential": True,
        "broadcast_c": False,
        "swa_start": swa_start,
        "n_swa": n_swa,
    }

    swa_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    select_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if swa_state is not None:
        torch.save(
            {
                "epoch": epochs,
                "n_swa": n_swa,
                "swa_start": swa_start,
                "val_acc": swa_sel.get("overall_acc_T16_val"),
                "sel_hard_neg_T16_val": swa_sel.get("hard_neg_acc_T16_val"),
                "sel_overall_T16_val": swa_sel.get("overall_acc_T16_val"),
                "state_dict": swa_state,
                "arm": f"fractal-stalk-swa-persist-T{T}-d{d}-mlp{mlp_expansion}",
                "science_open": False,
                "cycle": CYCLE,
                "role": "swa",
                "selection_rule": (
                    "SWA Polyak average of state_dict for epochs >= swa_start; "
                    "matched-OOD never used for selection"
                ),
                "hparams": hparams,
            },
            swa_ckpt_path,
        )

    if best_state is not None:
        torch.save(
            {
                "epoch": best_epoch,
                "val_acc": best_sel.get("overall_acc_T16_val"),
                "sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
                "sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
                "state_dict": best_state,
                "arm": f"fractal-stalk-swa-persist-select-T{T}-d{d}-mlp{mlp_expansion}",
                "science_open": False,
                "cycle": CYCLE,
                "role": "select_paired",
                "selection_rule": (
                    "lexicographic (0.5*HN+0.5*overall @ T16 ID-val, overall, HN, -epoch); "
                    "matched-OOD never used for selection"
                ),
                "hparams": hparams,
            },
            select_ckpt_path,
        )

    return {
        "arm": f"fractal-stalk-swa-persist-T{T}-d{d}-mlp{mlp_expansion}",
        "param_count": param_count,
        "param_parity": parity,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_sel_hard_neg_T16_val": best_sel.get("hard_neg_acc_T16_val"),
        "best_sel_overall_T16_val": best_sel.get("overall_acc_T16_val"),
        "best_sel_joint_T16_val": best_sel.get("joint_score_T16_val"),
        "swa_sel_hard_neg_T16_val": swa_sel.get("hard_neg_acc_T16_val"),
        "swa_sel_overall_T16_val": swa_sel.get("overall_acc_T16_val"),
        "swa_sel_joint_T16_val": swa_sel.get("joint_score_T16_val"),
        "swa_start": swa_start,
        "n_swa": n_swa,
        "selection_rule": (
            "lexicographic (0.5*HN+0.5*overall @ T16 ID-val, overall, HN, -epoch) "
            "+ Polyak SWA from swa_start"
        ),
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
        "swa_checkpoint_path": str(swa_ckpt_path),
        "select_checkpoint_path": str(select_ckpt_path),
        "science_open": False,
    }


def _t16_cell(matched: dict[str, Any]) -> dict[str, float]:
    t16 = matched[str(FOCUS_T)]
    return {
        "overall_acc": float(t16["overall_acc"]),
        "hard_neg_acc": float(t16["hard_neg_acc"]),
        "K8": float(t16["K8"]),
        "K12": float(t16["K12"]),
        "K16": float(t16["K16"]),
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    epochs: int = DEFAULT_EPOCHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    teacher_seeds: tuple[int, ...] = DEFAULT_TEACHER_SEEDS,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    grad_clip: float = DEFAULT_CLIP_H,
    swa_start: int = SWA_START,
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

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[stalk-swa] === seed {seed} train (SWA persist) ===", file=sys.stderr)
        swa_path = _swa_ckpt(seed)
        sel_path = _select_ckpt(seed)
        tr = train_fractal_id2k_swa(
            train,
            val,
            epochs=epochs,
            seed=seed,
            max_nodes=max_nodes,
            swa_start=swa_start,
            swa_ckpt_path=swa_path,
            select_ckpt_path=sel_path,
            lr_max=lr_max,
            lr_min=lr_min,
            grad_clip=grad_clip,
        )

        swa_model, _ = _load_model_from_ckpt(swa_path, max_nodes)
        sel_model, _ = _load_model_from_ckpt(sel_path, max_nodes)

        print(f"[stalk-swa] seed={seed} OOD eval SWA + select", file=sys.stderr)
        swa_matched = _eval_all_T(swa_model, ood_rows, max_nodes=max_nodes)
        sel_matched = _eval_all_T(sel_model, ood_rows, max_nodes=max_nodes)
        swa_deg = _eval_all_T(swa_model, deg_rows, max_nodes=max_nodes)
        sel_deg = _eval_all_T(sel_model, deg_rows, max_nodes=max_nodes)

        swa_t16 = _t16_cell(swa_matched)
        sel_t16 = _t16_cell(sel_matched)
        swa_pass = bool(
            swa_t16["hard_neg_acc"] >= PREREG_HARD_NEG
            and swa_t16["K16"] >= PREREG_K16
        )
        cell = {
            "seed": seed,
            "swa_checkpoint_path": str(swa_path),
            "select_checkpoint_path": str(sel_path),
            "train": {
                k: tr[k]
                for k in (
                    "best_epoch",
                    "best_sel_hard_neg_T16_val",
                    "best_sel_overall_T16_val",
                    "best_sel_joint_T16_val",
                    "swa_sel_hard_neg_T16_val",
                    "swa_sel_overall_T16_val",
                    "swa_sel_joint_T16_val",
                    "swa_start",
                    "n_swa",
                    "param_count",
                    "param_parity",
                    "grad_clip_sat_rate",
                )
            },
            "swa_matched_ood_by_T": swa_matched,
            "select_matched_ood_by_T": sel_matched,
            "swa_T16": swa_t16,
            "select_T16": sel_t16,
            "swa_degbal_T16": _t16_cell(swa_deg),
            "select_degbal_T16": _t16_cell(sel_deg),
            "prereg_pass": swa_pass,
            "delta_swa_minus_select_T16": {
                "overall": swa_t16["overall_acc"] - sel_t16["overall_acc"],
                "hard_neg": swa_t16["hard_neg_acc"] - sel_t16["hard_neg_acc"],
                "K16": swa_t16["K16"] - sel_t16["K16"],
            },
        }
        per_seed.append(cell)
        print(
            f"[stalk-swa] seed={seed} SWA T16 ov={swa_t16['overall_acc']:.4f} "
            f"HN={swa_t16['hard_neg_acc']:.4f} K16={swa_t16['K16']:.4f} "
            f"pass={swa_pass} | select HN={sel_t16['hard_neg_acc']:.4f} "
            f"K16={sel_t16['K16']:.4f}",
            file=sys.stderr,
        )

    def _mean_arm(key: str) -> dict[str, float]:
        overalls = [c[key]["overall_acc"] for c in per_seed]
        hns = [c[key]["hard_neg_acc"] for c in per_seed]
        k16s = [c[key]["K16"] for c in per_seed]
        return {
            "overall_acc": _mean(overalls),
            "hard_neg_acc": _mean(hns),
            "K16": _mean(k16s),
            "std_overall": _std(overalls),
            "std_hard_neg": _std(hns),
            "std_K16": _std(k16s),
            "n": len(per_seed),
            "n_prereg_pass": sum(
                1
                for c in per_seed
                if c[key]["hard_neg_acc"] >= PREREG_HARD_NEG
                and c[key]["K16"] >= PREREG_K16
            ),
        }

    swa_mean = _mean_arm("swa_T16")
    select_mean = _mean_arm("select_T16")

    # Re-eval frozen ensemble + #14 seed0 for compare (no train)
    print("[stalk-swa] re-eval frozen #14/#18 ensemble + #14 seed0", file=sys.stderr)
    teachers: list[Any] = []
    for ts in teacher_seeds:
        ckpt = _ckpt_for_seed(ts)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing ensemble member ckpt: {ckpt}")
        model, _ = _load_model_from_ckpt(ckpt, max_nodes)
        teachers.append(model)
    ens_matched = _eval_ensemble_methods(
        teachers, ood_rows, T=FOCUS_T, max_nodes=max_nodes, methods=(PRIMARY_AGG,)
    )
    ens_t16 = ens_matched[PRIMARY_AGG]
    pr14_seed0, _ = _load_model_from_ckpt(_ckpt_for_seed(PR14_REF_SEED), max_nodes)
    pr14_seed0_matched = _eval_all_T(pr14_seed0, ood_rows, max_nodes=max_nodes)
    pr14_t16 = _t16_cell(pr14_seed0_matched)

    floors_pass = bool(
        swa_mean["hard_neg_acc"] >= PREREG_HARD_NEG
        and swa_mean["K16"] >= PREREG_K16
    )
    ge_sel_hn = swa_mean["hard_neg_acc"] >= select_mean["hard_neg_acc"]
    ge_sel_k16 = swa_mean["K16"] >= select_mean["K16"]

    if floors_pass:
        verdict = "PASS_CANDIDATE"
        residue = (
            f"SWA mean PASS_CANDIDATE (swa_start={swa_start}): "
            f"T16 HN={swa_mean['hard_neg_acc']:.4f}≥{PREREG_HARD_NEG}, "
            f"K16={swa_mean['K16']:.4f}≥{PREREG_K16} "
            f"(n={swa_mean['n']}; seed PASS {swa_mean['n_prereg_pass']}/{swa_mean['n']}). "
            f"vs within-run select HN={select_mean['hard_neg_acc']:.4f} "
            f"K16={select_mean['K16']:.4f}; "
            f"vs ens HN={ens_t16['hard_neg_acc']:.4f} K16={ens_t16['K16']:.4f}. "
            "FLAG human — harness keeps science_open=false. "
            "Do not widen §22 ensemble scope. Select/curriculum stay CLOSED."
        )
        open_status = "PASS_CANDIDATE_SCIENCE_OPEN_FALSE"
    elif ge_sel_hn or ge_sel_k16:
        verdict = "MEASURE"
        residue = (
            f"SWA mean MEASURE (partial persist vs select): "
            f"T16 HN={swa_mean['hard_neg_acc']:.4f} K16={swa_mean['K16']:.4f} "
            f"(floors_pass={floors_pass}); ≥ within-run select on "
            f"{'HN' if ge_sel_hn else ''}"
            f"{'+' if ge_sel_hn and ge_sel_k16 else ''}"
            f"{'K16' if ge_sel_k16 else ''} "
            f"(select HN={select_mean['hard_neg_acc']:.4f} "
            f"K16={select_mean['K16']:.4f}). "
            f"ens HN={ens_t16['hard_neg_acc']:.4f} K16={ens_t16['K16']:.4f}. "
            "science_open=false."
        )
        open_status = "MEASURE_PARTIAL_PERSIST"
    else:
        verdict = "STOP"
        residue = (
            f"SWA mean STOP (no persist benefit vs within-run select): "
            f"T16 HN={swa_mean['hard_neg_acc']:.4f}<{select_mean['hard_neg_acc']:.4f} and "
            f"K16={swa_mean['K16']:.4f}<{select_mean['K16']:.4f}. "
            f"ens HN={ens_t16['hard_neg_acc']:.4f} K16={ens_t16['K16']:.4f}. "
            "science_open=false. Prefer #14 select + #22 ens overlay."
        )
        open_status = "STOP_NO_SWA_LIFT"

    compare = {
        "swa_mean": swa_mean,
        "select_mean": select_mean,
        "ensemble_prob_mean": {
            "overall_acc": ens_t16["overall_acc"],
            "hard_neg_acc": ens_t16["hard_neg_acc"],
            "K16": ens_t16["K16"],
            "K8": ens_t16["K8"],
            "K12": ens_t16["K12"],
        },
        "pr14_seed0": pr14_t16,
        "deltas_swa_minus_select": {
            "overall": swa_mean["overall_acc"] - select_mean["overall_acc"],
            "hard_neg": swa_mean["hard_neg_acc"] - select_mean["hard_neg_acc"],
            "K16": swa_mean["K16"] - select_mean["K16"],
        },
        "deltas_swa_minus_ensemble": {
            "overall": swa_mean["overall_acc"] - ens_t16["overall_acc"],
            "hard_neg": swa_mean["hard_neg_acc"] - ens_t16["hard_neg_acc"],
            "K16": swa_mean["K16"] - ens_t16["K16"],
        },
        "deltas_swa_minus_pr14_seed0": {
            "overall": swa_mean["overall_acc"] - pr14_t16["overall_acc"],
            "hard_neg": swa_mean["hard_neg_acc"] - pr14_t16["hard_neg_acc"],
            "K16": swa_mean["K16"] - pr14_t16["K16"],
        },
    }

    by_T_swa: dict[str, Any] = {}
    for Tv in T_VALUES:
        ovs = [c["swa_matched_ood_by_T"][str(Tv)]["overall_acc"] for c in per_seed]
        hns = [c["swa_matched_ood_by_T"][str(Tv)]["hard_neg_acc"] for c in per_seed]
        k16s = [c["swa_matched_ood_by_T"][str(Tv)]["K16"] for c in per_seed]
        by_T_swa[str(Tv)] = {
            "overall": {"mean": _mean(ovs), "std": _std(ovs)},
            "hard_neg": {"mean": _mean(hns), "std": _std(hns)},
            "K16": {"mean": _mean(k16s), "std": _std(k16s)},
        }

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "base_sha": _git_sha(),
        "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
        "metaphor": {
            "map_ne_location": (
                "Ensemble is a multi-seed map; SWA asks whether trajectory "
                "diversity inside one train collapses to a durable location."
            ),
            "negatives_eq_mirror": (
                "Hard-neg @ T16 mirrors whether the SWA location kept the "
                "reject boundary."
            ),
        },
        "frozen_recipe": (
            "SWA Polyak on frozen #14 harden (hard-A stalk); "
            "NO new select/upsample; select=#14 0.5/0.5 ID-val T16 as paired baseline; "
            "NOT soft distill"
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
        "swa_knobs": {
            "swa_start": swa_start,
            "swa_method": "polyak_state_dict_mean",
            "bn_update": False,
            "note": "RMSNorm only — no BN update pass",
            "seeds": list(seeds),
            "epochs": epochs,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "lr_schedule": "cosine",
            "grad_clip": grad_clip,
            "T_train": DEFAULT_T,
            "selection_rule_paired": "frozen #14 0.5*HN+0.5*overall @ ID-val T16",
            "primary": "swa",
        },
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "batch_size": DEFAULT_BATCH,
            "ff_baseline": FF_BASELINE_PARAMS,
            "matched_to": "PR#14 freeze + PR#12 floors; compare #22 ens",
        },
        "prereg": {
            "hard_neg_swa_mean_ge": PREREG_HARD_NEG,
            "K16_at_T16_swa_mean_ge": PREREG_K16,
            "swa_start": swa_start,
            "seeds": list(seeds),
            "selection_rule": (
                "Paired #14 lex select on ID-val T16. Floors on matched-OOD "
                "SWA mean after train. No OOD peek."
            ),
            "note": (
                "science_open never self-stamped; PASS_CANDIDATE still leaves "
                "science_open=false. Do not widen §22 ensemble scope. "
                "Select/curriculum remain CLOSED. Not soft distill."
            ),
        },
        "per_seed": per_seed,
        "compare_T16": compare,
        "by_T_swa_mean_std": by_T_swa,
        "pr14_seed0_matched_ood_by_T": pr14_seed0_matched,
        "degree_balanced_construction": {
            k: v for k, v in deg_bal.items() if k != "rows"
        },
        "datasets": {"id": str(id_data), "matched_ood": str(ood_data)},
        "policy": (
            "PASS_CANDIDATE → SWA mean floors; science_open=false; do not widen §22. "
            "MEASURE → miss floor but SWA ≥ within-run select on HN or K16. "
            "STOP → both HN and K16 < within-run select. Sheaf unsupervised ignored. "
            "Not soft distill (#23 STOP)."
        ),
        "prereg_floors_pass": floors_pass,
        "verdict": verdict,
        "open_status": open_status,
        "residue": residue,
        "elapsed_sec": time.time() - t0,
        "prefer_corridor": (
            "PR#14 MEASURE_STILL singles + #22/#24 ens overlay; "
            "SWA tests single-model persistence without soft distill"
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[stalk-swa] wrote {out_path}", file=sys.stderr)
    print(
        f"[stalk-swa] verdict={verdict} science_open=False "
        f"swa_mean T16 ov={swa_mean['overall_acc']:.4f} "
        f"HN={swa_mean['hard_neg_acc']:.4f} K16={swa_mean['K16']:.4f} "
        f"select HN={select_mean['hard_neg_acc']:.4f} "
        f"K16={select_mean['K16']:.4f} "
        f"ens HN={ens_t16['hard_neg_acc']:.4f} K16={ens_t16['K16']:.4f}",
        file=sys.stderr,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument(
        "--teacher-seeds", type=int, nargs="+", default=list(DEFAULT_TEACHER_SEEDS)
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--swa-start", type=int, default=SWA_START)
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
        out_path=args.out,
        epochs=args.epochs,
        seeds=tuple(args.seeds),
        teacher_seeds=tuple(args.teacher_seeds),
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        grad_clip=args.grad_clip,
        swa_start=args.swa_start,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "open_status": report["open_status"],
                "science_open": report["science_open"],
                "prereg_floors_pass": report["prereg_floors_pass"],
                "compare_T16": report["compare_T16"],
                "n_seeds": report["swa_knobs"]["seeds"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
