"""CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN — Gates A–D (MEASURE; science_open=false).

Protocol (fail-closed):
  Gate A: neutral untrained ≈ chance (reachability ~0.50; Â FPR/FNR ~0.5)
  Gate B: neutral_init removes edge+4 / absent−4 / energy±5 bake-in
  Gate C: degree-balanced hard-negs + reach-cue audit ≤0.52 ceiling
  Gate D: ≥3 seeds; report mean±std

If learning fails floors → STOP with residue, no OPEN.
If passes floors → MEASURE candidate only (never stamp science_open).

Usage::

    python -m reachability_gen.run_sheaf_neutral_init_retrain
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

from reachability_gen.models.sheaf_infer_core import (
    DEFAULT_EDGE_RECON_WEIGHT,
    DEFAULT_GATE_THETA,
    DISCRETE_T_VALUES,
    SheafInferCore,
    _verify_param_parity,
    build_sheaf_batch,
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.overfit_ff import ensure_balanced_batch
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID
from reachability_gen.reach_cue_audit import (
    audit_reach_cues,
    build_degree_balanced_eval,
    write_audit_artifact,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS
from reachability_gen.run_sheaf_infer_gate1 import (
    DEFAULT_BATCH,
    DEFAULT_CLIP,
    DEFAULT_D,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MLP,
    DEFAULT_T,
    DYNAMIC_T_VALUES,
    PREREG_HARD_NEG,
    PREREG_K16,
    _eval_split,
    _split_train_val,
)

DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/sheaf_neutral_init_retrain.json")
DEFAULT_CUE = Path("artifacts/sheaf_reach_cue_audit.json")
DEFAULT_SEEDS = (0, 1, 2)
CHANCE_LO = 0.40
CHANCE_HI = 0.60
A_HAT_CHANCE_LO = 0.35
A_HAT_CHANCE_HI = 0.65


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0 if xs else float("nan")
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


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


def document_neutral_init(model: SheafInferCore) -> dict[str, Any]:
    last = model.edge_encoder[-1]
    first = model.edge_encoder[0]
    return {
        "neutral_init": bool(getattr(model, "neutral_init", False)),
        "edge_encoder_first_bias_abs_max": float(first.bias.detach().abs().max()),
        "edge_encoder_last_bias": float(last.bias.detach().item()),
        "edge_encoder_last_weight_l2": float(last.weight.detach().norm()),
        "absent_bias": float(model.absent_bias.detach().item()),
        "head_bias": model.head.bias.detach().tolist(),
        "head_energy_col": model.head.weight.detach()[:, -1].tolist(),
        "self_logit_hardcoded": 8.0,
        "residual_alpha": float(model.residual_alpha),
        "gate_theta": float(model.gate_theta),
        "param_count": model.param_count(),
        "bake_in_removed": {
            "no_edge_bias_plus4": abs(float(last.bias.detach().item())) < 0.5,
            "no_absent_minus4": abs(float(model.absent_bias.detach().item())) < 0.5,
            "no_energy_bias_pm5": (
                abs(float(model.head.bias.detach()[0].item())) < 1.0
                and abs(float(model.head.bias.detach()[1].item())) < 1.0
            ),
        },
    }


def _ahat_fpr_fnr(
    model: SheafInferCore,
    rows: list[dict[str, Any]],
    *,
    max_nodes: int,
    batch_size: int = 64,
) -> dict[str, float]:
    """Off-diagonal Â gate vs gold_adj FPR/FNR (eval hard gate)."""
    import torch

    model.eval()
    tp = fp = tn = fn = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = build_sheaf_batch(batch_rows, max_n=max_nodes)
            E = model.encode_edge_logits(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
            )
            gate = model.logits_to_gate(E, batch["node_mask"])
            gold = batch["gold_adj"]
            bsz, mlen, _ = gold.shape
            eye = torch.eye(mlen, device=gold.device, dtype=torch.bool)
            real = (batch["node_mask"].unsqueeze(1) * batch["node_mask"].unsqueeze(2)).bool()
            off = real & ~eye.unsqueeze(0)
            g = gate[off]
            y = gold[off]
            tp += float(((g > 0.5) & (y > 0.5)).sum())
            fp += float(((g > 0.5) & (y <= 0.5)).sum())
            tn += float(((g <= 0.5) & (y <= 0.5)).sum())
            fn += float(((g <= 0.5) & (y > 0.5)).sum())
    pos = tp + fn
    neg = tn + fp
    return {
        "ahat_fpr": fp / neg if neg else float("nan"),
        "ahat_fnr": fn / pos if pos else float("nan"),
        "ahat_tp": tp,
        "ahat_fp": fp,
        "ahat_tn": tn,
        "ahat_fn": fn,
        "ahat_edge_acc": (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn)
        else float("nan"),
    }


def _reach_fpr_fnr(eval_stats: dict[str, Any], rows: list[dict[str, Any]], model: SheafInferCore, *, T: int, max_nodes: int) -> dict[str, float]:
    """Reachability FPR/FNR at given T."""
    import torch
    import torch.nn.functional as F

    model.eval()
    preds_all: list[Any] = []
    labels_all: list[Any] = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch_rows = rows[start : start + 64]
            batch = build_sheaf_batch(batch_rows, max_n=max_nodes)
            logits, _, _ = model(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
                batch["s_idx"],
                batch["t_idx"],
                T=T,
            )
            preds_all.append(logits.argmax(dim=-1).cpu())
            labels_all.append(batch["labels"].cpu())
    P = torch.cat(preds_all)
    L = torch.cat(labels_all)
    y0 = L == 0
    y1 = L == 1
    fpr = float(((P == 1) & y0).sum() / y0.sum()) if y0.any() else float("nan")
    fnr = float(((P == 0) & y1).sum() / y1.sum()) if y1.any() else float("nan")
    overall = float((P == L).float().mean())
    return {"reach_fpr": fpr, "reach_fnr": fnr, "reach_acc": overall, "T": T}


def gate_a_untrained(
    *,
    max_nodes: int,
    ood_rows: list[dict[str, Any]],
    seed: int = 0,
) -> dict[str, Any]:
    """Gate A: neutral untrained must be near chance (not baked)."""
    import torch

    torch.manual_seed(seed)
    model = SheafInferCore(
        d=DEFAULT_D,
        T=DEFAULT_T,
        mlp_expansion=DEFAULT_MLP,
        max_nodes=max_nodes,
        max_T=max(DEFAULT_T, max(DISCRETE_T_VALUES)),
        residual_alpha=1.0,
        gate_detach_diffusion=True,
        gate_theta=DEFAULT_GATE_THETA,
        gate_mode="ste",
        neutral_init=True,
    )
    init_doc = document_neutral_init(model)
    ahat = _ahat_fpr_fnr(model, ood_rows, max_nodes=max_nodes)
    # t=0 reachability: use T=16 focus (same as prereg) and T=6 train
    reach16 = _reach_fpr_fnr({}, ood_rows, model, T=16, max_nodes=max_nodes)
    reach6 = _reach_fpr_fnr({}, ood_rows, model, T=6, max_nodes=max_nodes)
    # Also evaluate on id val-like slice for t=0 report
    reach_acc = reach16["reach_acc"]
    ahat_fpr, ahat_fnr = ahat["ahat_fpr"], ahat["ahat_fnr"]

    def _near_chance(x: float, lo: float, hi: float) -> bool:
        return x == x and lo <= x <= hi

    reach_ok = _near_chance(reach_acc, CHANCE_LO, CHANCE_HI)
    # Â: legacy bake-in has FNR≈0 (all listed edges ON). Neutral must NOT
    # recover listed edges (FNR near chance). FPR≈0 is expected with
    # absent_bias=0 and hard θ=0.5 (σ(0)=0.5 ≯ θ → absent OFF) — document,
    # not treat as bake-in. Disqualify if Â FNR near 0 (still baked).
    ahat_fnr_ok = _near_chance(ahat_fnr, A_HAT_CHANCE_LO, A_HAT_CHANCE_HI)
    ahat_not_perfect = not (
        ahat_fnr == ahat_fnr and ahat_fnr < 0.05 and ahat_fpr == ahat_fpr and ahat_fpr < 0.05
    )
    ahat_ok = bool(ahat_fnr_ok and ahat_not_perfect)
    above_chance = reach_acc == reach_acc and reach_acc > CHANCE_HI
    bake_flags = init_doc["bake_in_removed"]
    bake_ok = all(bake_flags.values())
    passed = bool(reach_ok and ahat_ok and bake_ok and not above_chance)
    return {
        "gate": "A",
        "passed": passed,
        "disqualify_above_chance": bool(above_chance),
        "init": init_doc,
        "matched_ood_T16": reach16,
        "matched_ood_T6": reach6,
        "ahat": ahat,
        "ahat_note": (
            "absent_bias=0 ⇒ hard-gate FPR≈0 by construction (not bake-in). "
            "Gate A requires Â FNR≈chance (listed edges not all ON)."
        ),
        "thresholds": {
            "reach_acc_lo": CHANCE_LO,
            "reach_acc_hi": CHANCE_HI,
            "ahat_fnr_lo": A_HAT_CHANCE_LO,
            "ahat_fnr_hi": A_HAT_CHANCE_HI,
        },
        "checks": {
            "reach_ok": reach_ok,
            "ahat_fnr_ok": ahat_fnr_ok,
            "ahat_not_perfect": ahat_not_perfect,
            "bake_ok": bake_ok,
        },
        "science_open": False,
    }


def train_neutral_seed(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    seed: int,
    max_nodes: int,
    epochs: int,
    ckpt_path: Path,
) -> dict[str, Any]:
    import torch
    from reachability_gen.train.sheaf_trainer import SheafTrainer

    torch.manual_seed(seed)
    model = SheafInferCore(
        d=DEFAULT_D,
        T=DEFAULT_T,
        mlp_expansion=DEFAULT_MLP,
        max_nodes=max_nodes,
        max_T=max(DEFAULT_T, max(DISCRETE_T_VALUES)),
        use_tau=False,
        apply_cycle_rmsnorm=False,
        residual_alpha=1.0,
        gate_detach_diffusion=True,
        gate_theta=DEFAULT_GATE_THETA,
        gate_mode="ste",
        neutral_init=True,
    )
    # Untrained baseline on val before training
    untrained_val = _eval_split(model, val, max_nodes=max_nodes, T=DEFAULT_T)
    untrained_init = document_neutral_init(model)
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    trainer = SheafTrainer(
        model,
        lr=DEFAULT_LR,
        weight_decay=0.0,
        grad_clip=DEFAULT_CLIP,
        edge_recon_weight=DEFAULT_EDGE_RECON_WEIGHT,
    )
    best_val_acc = -1.0
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    hist: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train)).tolist()
        ep_loss: list[float] = []
        ep_acc: list[float] = []
        ep_recon: list[float] = []
        for start in range(0, len(train), DEFAULT_BATCH):
            idx = order[start : start + DEFAULT_BATCH]
            batch_rows = [train[i] for i in idx]
            batch = build_sheaf_batch(batch_rows, max_n=max_nodes)
            loss, acc = trainer.train_step(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
                batch["s_idx"],
                batch["t_idx"],
                batch["labels"],
                batch["gold_adj"],
            )
            ep_loss.append(loss)
            ep_acc.append(acc)
            if trainer.last_edge_recon_acc == trainer.last_edge_recon_acc:
                ep_recon.append(trainer.last_edge_recon_acc)
        val_stats = _eval_split(model, val, max_nodes=max_nodes, T=DEFAULT_T)
        ov = float(val_stats["overall_acc"])
        hist.append(
            {
                "epoch": epoch,
                "train_loss": _mean(ep_loss),
                "train_acc": _mean(ep_acc),
                "train_edge_recon_acc": _mean(ep_recon),
                "val_acc": ov,
                "val_loss": float(val_stats["overall_loss"]),
            }
        )
        if ov > best_val_acc:
            best_val_acc = ov
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"[neutral seed={seed}] ep {epoch}/{epochs} "
            f"train={_mean(ep_acc):.4f} recon={_mean(ep_recon):.4f} "
            f"val={ov:.4f} best={best_val_acc:.4f}@ep{best_epoch}",
            file=sys.stderr,
        )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(
            {
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "state_dict": best_state,
                "arm": f"sheaf-neutral-T{DEFAULT_T}-d{DEFAULT_D}-mlp{DEFAULT_MLP}",
                "science_open": False,
                "cycle": "CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN",
                "neutral_init": True,
                "seed": seed,
                "hparams": {
                    "d": DEFAULT_D,
                    "T": DEFAULT_T,
                    "mlp_expansion": DEFAULT_MLP,
                    "max_nodes": max_nodes,
                    "lr": DEFAULT_LR,
                    "grad_clip": DEFAULT_CLIP,
                    "gate_theta": DEFAULT_GATE_THETA,
                    "gate_mode": "ste",
                    "residual_alpha": 1.0,
                    "gate_detach_diffusion": True,
                    "edge_recon_weight": DEFAULT_EDGE_RECON_WEIGHT,
                    "neutral_init": True,
                    "hard_A_oracle_eval": False,
                },
            },
            ckpt_path,
        )
    return {
        "seed": seed,
        "param_count": model.param_count(),
        "param_parity": parity,
        "untrained_val_acc": float(untrained_val["overall_acc"]),
        "untrained_init": untrained_init,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "train_history": hist,
        "checkpoint_path": str(ckpt_path),
        "science_open": False,
        "model": model,
    }


def eval_ood_full(
    model: SheafInferCore,
    ood_rows: list[dict[str, Any]],
    *,
    max_nodes: int,
    T_values: tuple[int, ...] = DYNAMIC_T_VALUES,
) -> dict[str, Any]:
    by_T: dict[str, Any] = {}
    for T in T_values:
        stats = _eval_split(model, ood_rows, max_nodes=max_nodes, T=T)
        reach = _reach_fpr_fnr(stats, ood_rows, model, T=T, max_nodes=max_nodes)
        stats = dict(stats)
        stats["reach_fpr"] = reach["reach_fpr"]
        stats["reach_fnr"] = reach["reach_fnr"]
        # K strata
        k_acc = {}
        for k, v in stats.get("pos_by_hop", {}).items():
            k_acc[k] = float(v["acc_mean"])
        stats["K_acc"] = k_acc
        by_T[str(T)] = stats
        print(
            f"[neutral-ood] T={T} acc={stats['overall_acc']:.4f} "
            f"hard_neg={stats['hard_neg_acc']:.4f} "
            f"FPR={reach['reach_fpr']:.4f} FNR={reach['reach_fnr']:.4f} "
            f"K16={k_acc.get('16', float('nan'))}",
            file=sys.stderr,
        )
    ahat = _ahat_fpr_fnr(model, ood_rows, max_nodes=max_nodes)
    t16 = by_T.get("16", {})
    hard = float(t16.get("hard_neg_acc", float("nan")))
    k16 = float(t16.get("K_acc", {}).get("16", float("nan")))
    # Prefer K16; if missing, fall back to K8 for documented scope
    k8 = float(t16.get("K_acc", {}).get("8", float("nan")))
    prereg_hard = hard == hard and hard >= PREREG_HARD_NEG
    prereg_k16 = k16 == k16 and k16 >= PREREG_K16
    prereg_k8_alt = k8 == k8 and k8 >= PREREG_K16  # same numeric floor if K16 heavy
    return {
        "by_T": by_T,
        "ahat": ahat,
        "prereg": {
            "hard_neg_at_T16_ge": PREREG_HARD_NEG,
            "K16_at_T16_ge": PREREG_K16,
            "observed_hard_neg_T16": hard,
            "observed_K16_T16": k16,
            "observed_K8_T16": k8,
            "hard_neg_ok": prereg_hard,
            "K16_ok": prereg_k16,
            "K8_alt_ok": prereg_k8_alt,
            "pass": bool(prereg_hard and prereg_k16),
            "pass_relaxed_K8": bool(prereg_hard and (prereg_k16 or prereg_k8_alt)),
        },
        "science_open": False,
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    cue_path: Path = DEFAULT_CUE,
    epochs: int = DEFAULT_EPOCHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    skip_overfit: bool = False,
) -> dict[str, Any]:
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

    report: dict[str, Any] = {
        "cycle": "CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN",
        "mode": "MEASURE",
        "science_open": False,
        "base_sha": _git_sha(),
        "merge_pr7_sha_expected_prefix": "034a074",
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "lr": DEFAULT_LR,
            "epochs": epochs,
            "grad_clip": DEFAULT_CLIP,
            "batch_size": DEFAULT_BATCH,
            "neutral_init": True,
            "param_parity_window": [115157, 127279],
            "ff_baseline": FF_BASELINE_PARAMS,
        },
        "seeds": list(seeds),
        "gates": {},
        "scope": {
            "protocol": (
                "Gate0 overfit + ID val + matched-OOD; K16 preferred; "
                "document if K8-only scope."
            ),
            "datasets": {
                "id": str(id_data),
                "matched_ood": str(ood_data),
            },
        },
    }

    # --- Gate B documented via init flags; Gate A first ---
    print("[gate A] neutral untrained control…", file=sys.stderr)
    gate_a = gate_a_untrained(max_nodes=max_nodes, ood_rows=ood_rows, seed=0)
    report["gates"]["A"] = {k: v for k, v in gate_a.items() if k != "model"}
    if not gate_a["passed"]:
        report["verdict"] = "STOP_GATE_A_FAIL"
        report["residue"] = (
            "Neutral untrained still above chance or bake-in not removed. "
            "Iterate init; no OPEN."
        )
        report["elapsed_s"] = time.time() - t0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    # --- Gate B: confirm bake-in removed (from Gate A init doc) ---
    bake = gate_a["init"]["bake_in_removed"]
    gate_b = {
        "gate": "B",
        "passed": bool(all(bake.values())),
        "bake_in_removed": bake,
        "init": gate_a["init"],
        "note": (
            "neutral_init=True: Xavier/Gaussian zero-mean; biases 0; "
            "no edge+4 / absent−4 / energy±5. STE OK for learning Â."
        ),
        "science_open": False,
    }
    report["gates"]["B"] = gate_b
    if not gate_b["passed"]:
        report["verdict"] = "STOP_GATE_B_FAIL"
        report["residue"] = "Bake-in flags still set; no OPEN."
        report["elapsed_s"] = time.time() - t0
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    # --- Gate C: reach-cue audit + degree-balanced hard-negs ---
    print("[gate C] reach-cue audit + degree-balanced hard-negs…", file=sys.stderr)
    cue_ood = audit_reach_cues(ood_rows, ceiling=0.52)
    cue_id = audit_reach_cues(rows, ceiling=0.52)
    deg_bal = build_degree_balanced_eval(ood_rows, seed=0)
    cue_bal = audit_reach_cues(deg_bal["rows"], ceiling=0.52)
    # Pass Gate C if degree-balanced eval cue ceiling holds (primary trust set)
    # Document raw OOD cue even if higher — honest.
    gate_c = {
        "gate": "C",
        "passed": bool(cue_bal["pass_ceiling"]),
        "ceiling": 0.52,
        "matched_ood_raw": {
            "max_acc": cue_ood["max_acc"],
            "max_rule": cue_ood["max_rule"],
            "pass_ceiling": cue_ood["pass_ceiling"],
        },
        "id_2k_raw": {
            "max_acc": cue_id["max_acc"],
            "max_rule": cue_id["max_rule"],
            "pass_ceiling": cue_id["pass_ceiling"],
        },
        "degree_balanced_ood": {
            "n": deg_bal["n"],
            "n_pos": deg_bal["n_pos"],
            "n_neg": deg_bal["n_neg"],
            "shortfall_unmatched_pos": deg_bal["shortfall_unmatched_pos"],
            "cue_max_acc": cue_bal["max_acc"],
            "cue_max_rule": cue_bal["max_rule"],
            "pass_ceiling": cue_bal["pass_ceiling"],
        },
        "artifact": str(cue_path),
        "science_open": False,
    }
    write_audit_artifact(
        {
            "matched_ood": cue_ood,
            "id_2k": cue_id,
            "degree_balanced_ood_cue": cue_bal,
            "degree_balanced_construction": {
                k: v for k, v in deg_bal.items() if k != "rows"
            },
        },
        cue_path,
        extra={"cycle": "CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN", "science_open": False},
    )
    report["gates"]["C"] = gate_c
    # Do not hard-STOP on raw id_2k cue fail — document; require balanced ceiling
    if not gate_c["passed"]:
        report["verdict"] = "STOP_GATE_C_CUE_CEILING"
        report["residue"] = (
            f"Degree-balanced reach-cue max_acc={cue_bal['max_acc']:.4f} > 0.52; "
            "do not trust model metrics. No OPEN."
        )
        report["elapsed_s"] = time.time() - t0
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    # --- Gate 0 overfit (optional sanity) ---
    overfit_summary: dict[str, Any] = {"skipped": skip_overfit}
    if not skip_overfit:
        print("[gate0] overfit balanced…", file=sys.stderr)
        batch, note, _meta = ensure_balanced_batch(
            Path("data/train_tiny.jsonl"), n_pos=16, n_neg=16, regenerate=True
        )
        # Monkey-patch: run_overfit_sheaf needs neutral_init — call inline
        overfit_summary = _overfit_neutral(batch, max_nodes=max_nodes)
        overfit_summary["balanced_note"] = note

    # --- Gate D: multi-seed train + eval ---
    print(f"[gate D] train {len(seeds)} seeds…", file=sys.stderr)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        ckpt = Path(f"artifacts/sheaf_neutral_gate1_seed{seed}_best.pt")
        tr = train_neutral_seed(
            train, val, seed=seed, max_nodes=max_nodes, epochs=epochs, ckpt_path=ckpt
        )
        model = tr.pop("model")
        ood_eval = eval_ood_full(model, ood_rows, max_nodes=max_nodes)
        # Degree-balanced secondary eval at T16
        bal_stats = _eval_split(
            model, deg_bal["rows"], max_nodes=max_nodes, T=16
        )
        ahat_trained = _ahat_fpr_fnr(model, ood_rows, max_nodes=max_nodes)
        seed_row = {
            "seed": seed,
            "param_count": tr["param_count"],
            "param_parity_ok": tr["param_parity"]["within_5pct"],
            "untrained_val_acc": tr["untrained_val_acc"],
            "best_epoch": tr["best_epoch"],
            "best_val_acc": tr["best_val_acc"],
            "checkpoint_path": tr["checkpoint_path"],
            "matched_ood": {
                "T16_overall": ood_eval["by_T"]["16"]["overall_acc"],
                "T16_hard_neg": ood_eval["by_T"]["16"]["hard_neg_acc"],
                "T16_K16": ood_eval["by_T"]["16"]["K_acc"].get("16"),
                "T16_K8": ood_eval["by_T"]["16"]["K_acc"].get("8"),
                "T16_K12": ood_eval["by_T"]["16"]["K_acc"].get("12"),
                "T16_fpr": ood_eval["by_T"]["16"]["reach_fpr"],
                "T16_fnr": ood_eval["by_T"]["16"]["reach_fnr"],
                "by_T": {
                    t: {
                        "overall_acc": v["overall_acc"],
                        "hard_neg_acc": v["hard_neg_acc"],
                        "K_acc": v["K_acc"],
                        "reach_fpr": v["reach_fpr"],
                        "reach_fnr": v["reach_fnr"],
                    }
                    for t, v in ood_eval["by_T"].items()
                },
            },
            "degree_balanced_ood_T16": {
                "overall_acc": bal_stats["overall_acc"],
                "hard_neg_acc": bal_stats["hard_neg_acc"],
                "n": bal_stats["n"],
            },
            "ahat_trained": ahat_trained,
            "prereg": ood_eval["prereg"],
            "train_history_tail": tr["train_history"][-3:],
            "science_open": False,
        }
        per_seed.append(seed_row)

    def _agg(key_path: list[str]) -> dict[str, float]:
        xs: list[float] = []
        for row in per_seed:
            cur: Any = row
            for k in key_path:
                cur = cur[k]
            if cur is not None and cur == cur:
                xs.append(float(cur))
        return {"mean": _mean(xs), "std": _std(xs), "n": len(xs), "values": xs}

    summary = {
        "val_acc": _agg(["best_val_acc"]),
        "untrained_val_acc": _agg(["untrained_val_acc"]),
        "T16_overall": _agg(["matched_ood", "T16_overall"]),
        "T16_hard_neg": _agg(["matched_ood", "T16_hard_neg"]),
        "T16_K16": _agg(["matched_ood", "T16_K16"]),
        "T16_K8": _agg(["matched_ood", "T16_K8"]),
        "T16_fpr": _agg(["matched_ood", "T16_fpr"]),
        "T16_fnr": _agg(["matched_ood", "T16_fnr"]),
        "degbal_T16_overall": _agg(["degree_balanced_ood_T16", "overall_acc"]),
        "degbal_T16_hard_neg": _agg(["degree_balanced_ood_T16", "hard_neg_acc"]),
        "ahat_fpr": _agg(["ahat_trained", "ahat_fpr"]),
        "ahat_fnr": _agg(["ahat_trained", "ahat_fnr"]),
    }

    n_prereg = sum(1 for r in per_seed if r["prereg"]["pass"])
    n_relaxed = sum(1 for r in per_seed if r["prereg"]["pass_relaxed_K8"])
    floors_pass = n_prereg == len(per_seed)
    floors_relaxed = n_relaxed == len(per_seed)

    gate_d = {
        "gate": "D",
        "passed": len(per_seed) >= 3,
        "n_seeds": len(per_seed),
        "n_prereg_pass": n_prereg,
        "n_relaxed_K8_pass": n_relaxed,
        "per_seed": per_seed,
        "mean_std": summary,
        "science_open": False,
    }
    report["gates"]["D"] = gate_d
    report["gate0_overfit"] = overfit_summary
    report["reach_cue_artifact"] = str(cue_path)

    # Final verdict — fail-closed
    if floors_pass:
        report["verdict"] = "MEASURE_CANDIDATE_PASS_FLOORS"
        report["residue"] = (
            "Neutral-init trained meets prereg floors (hard-neg≥0.95 and K16≥0.75) "
            "across seeds. MEASURE candidate only — do NOT stamp science_open "
            "without human review. Untrained≪trained required (Gate A held)."
        )
    elif floors_relaxed:
        report["verdict"] = "MEASURE_PARTIAL_K8_SCOPE"
        report["residue"] = (
            "Prereg K16 floor missed or partial; K8-relaxed floors held. "
            "Documented scope — not full OPEN candidate. science_open=false."
        )
    else:
        report["verdict"] = "STOP_LEARNING_FAIL"
        report["residue"] = (
            "Neutral-init training failed prereg floors. STOP; do not revive "
            "learned OPEN from bake-in seals. science_open=false."
        )

    report["elapsed_s"] = time.time() - t0
    # Strip non-JSON from nested if any
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[neutral] verdict={report['verdict']} wrote {out_path} "
        f"elapsed={report['elapsed_s']:.1f}s",
        file=sys.stderr,
    )
    return report


def _overfit_neutral(examples: list[dict[str, Any]], *, max_nodes: int) -> dict[str, Any]:
    """Gate0 overfit with neutral_init (inline; overfit_sheaf defaults legacy)."""
    import torch
    from reachability_gen.train.sheaf_trainer import SheafTrainer
    from reachability_gen.overfit_ff import (
        BALANCED_LOSS_THRESHOLD,
        _per_class_accuracy,
    )

    torch.manual_seed(0)
    batch = build_sheaf_batch(examples, max_n=max_nodes)
    model = SheafInferCore(
        d=DEFAULT_D,
        T=DEFAULT_T,
        mlp_expansion=DEFAULT_MLP,
        max_nodes=max_nodes,
        max_T=max(DEFAULT_T, max(DISCRETE_T_VALUES)),
        residual_alpha=1.0,
        gate_detach_diffusion=True,
        neutral_init=True,
    )
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    trainer = SheafTrainer(
        model, lr=3e-3, weight_decay=0.01, grad_clip=DEFAULT_CLIP, edge_recon_weight=1.0
    )
    passed_at = None
    last = {}
    for step in range(1, 151):
        trainer.train_step(
            batch["node_ids"],
            batch["node_mask"],
            batch["edge_index"],
            batch["edge_mask"],
            batch["s_idx"],
            batch["t_idx"],
            batch["labels"],
            batch["gold_adj"],
        )
        eval_out = trainer.eval_step(
            batch["node_ids"],
            batch["node_mask"],
            batch["edge_index"],
            batch["edge_mask"],
            batch["s_idx"],
            batch["t_idx"],
            batch["labels"],
            batch["gold_adj"],
        )
        eval_loss, eval_acc = eval_out[0], eval_out[1]
        with torch.no_grad():
            logits, _, _ = model(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
                batch["s_idx"],
                batch["t_idx"],
            )
            pc = _per_class_accuracy(logits.argmax(-1), batch["labels"])
        recon = trainer.last_edge_recon_acc
        last = {
            "step": step,
            "eval_loss": float(eval_loss),
            "eval_acc": float(eval_acc),
            "recon_offdiag": float(recon) if recon == recon else None,
            "per_class": pc,
        }
        if (
            eval_loss < BALANCED_LOSS_THRESHOLD
            and eval_acc >= 1.0 - 1e-9
            and pc.get("y0", 0) >= 1.0 - 1e-9
            and pc.get("y1", 0) >= 1.0 - 1e-9
            and recon == recon
            and recon >= 0.99
            and passed_at is None
        ):
            passed_at = step
            break
    return {
        "passed": passed_at is not None,
        "passed_at_step": passed_at,
        "last": last,
        "param_count": model.param_count(),
        "param_parity_ok": parity["within_5pct"],
        "neutral_init": True,
        "science_open": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cue-out", type=Path, default=DEFAULT_CUE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--skip-overfit", action="store_true")
    args = p.parse_args(argv)
    report = run_cycle(
        id_data=args.id_data,
        ood_data=args.ood_data,
        out_path=args.out,
        cue_path=args.cue_out,
        epochs=args.epochs,
        seeds=tuple(args.seeds),
        skip_overfit=args.skip_overfit,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "science_open": report["science_open"],
                "gates_passed": {
                    g: report["gates"][g].get("passed") for g in report["gates"]
                },
                "mean_std": report.get("gates", {}).get("D", {}).get("mean_std"),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    # Exit 0 even on STOP — MEASURE honesty; nonzero only on crash
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
