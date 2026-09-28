"""CYCLE_SHEAF_STE_NO_AUX — Gates A–D (MEASURE; science_open=false).

After PR #9 STOP (no-aux + gate_detach_diffusion=True collapsed OOD), retrain
with the same Gates A–D / neutral_init / edge_recon_weight=0, but enable
end-to-end STE (or Gumbel-Sigmoid) into Φ: gate_detach_diffusion=False so
reachability CE can train Â without aux edge-recon.

Fail-closed: same prereg floors (hard-neg≥0.95 and K16≥0.75 at T16). Longer
budget (60 ep default). PASS→MEASURE candidate; FAIL→STOP residue.
Never stamp science_open.

Usage::

    python -m reachability_gen.run_sheaf_ste_no_aux
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.models.sheaf_infer_core import (
    DEFAULT_GATE_THETA,
    DISCRETE_T_VALUES,
    SheafInferCore,
    _verify_param_parity,
    build_sheaf_batch,
)
from reachability_gen.overfit_ff import ensure_balanced_batch, load_jsonl
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
    DEFAULT_LR,
    DEFAULT_MLP,
    DEFAULT_T,
    DYNAMIC_T_VALUES,
    PREREG_HARD_NEG,
    PREREG_K16,
    _eval_split,
    _split_train_val,
)
from reachability_gen.run_sheaf_neutral_init_retrain import (
    _ahat_fpr_fnr,
    _mean,
    _std,
    document_neutral_init,
    eval_ood_full,
    gate_a_untrained,
)

CYCLE = "CYCLE_SHEAF_STE_NO_AUX"
EDGE_RECON_WEIGHT = 0.0  # aux OFF — reachability CE only
GATE_DETACH_DIFFUSION = False  # STE/Gumbel end-to-end into Φ
GATE_MODE = "ste"  # or "gumbel"
GUMBEL_TEMP = 1.0
DEFAULT_EPOCHS_STE = 60  # longer budget vs locked 30-ep no-aux STOP
PR8_MERGE_SHA = "e0877ebf52a16efd81df1c80ca6fb3f1c3289a31"
PR9_MERGE_SHA = "2834256a12a7a1d9ff826738cd860051f71e19ac"
PR8_BASELINE_ARTIFACT = Path("artifacts/sheaf_neutral_init_retrain.json")
PR9_BASELINE_ARTIFACT = Path("artifacts/sheaf_no_aux_edge_recon.json")

DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/sheaf_ste_no_aux.json")
DEFAULT_CUE = Path("artifacts/sheaf_ste_no_aux_reach_cue_audit.json")
DEFAULT_SEEDS = (0, 1, 2)


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


def train_ste_no_aux_seed(
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
        gate_detach_diffusion=GATE_DETACH_DIFFUSION,
        gate_theta=DEFAULT_GATE_THETA,
        gate_mode=GATE_MODE,
        gumbel_temp=GUMBEL_TEMP,
        neutral_init=True,
    )
    untrained_val = _eval_split(model, val, max_nodes=max_nodes, T=DEFAULT_T)
    untrained_init = document_neutral_init(model)
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    trainer = SheafTrainer(
        model,
        lr=DEFAULT_LR,
        weight_decay=0.0,
        grad_clip=DEFAULT_CLIP,
        edge_recon_weight=EDGE_RECON_WEIGHT,
    )
    best_val_acc = -1.0
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    hist: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train)).tolist()
        ep_loss: list[float] = []
        ep_acc: list[float] = []
        ep_ce: list[float] = []
        ep_recon: list[float] = []
        for start in range(0, len(train), DEFAULT_BATCH):
            idx = order[start : start + DEFAULT_BATCH]
            batch_rows = [train[i] for i in idx]
            batch = build_sheaf_batch(batch_rows, max_n=max_nodes)
            # Pass gold_adj for telemetry only; weight=0 ⇒ recon term unused.
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
            if trainer.last_ce == trainer.last_ce:
                ep_ce.append(trainer.last_ce)
            if trainer.last_edge_recon_acc == trainer.last_edge_recon_acc:
                ep_recon.append(trainer.last_edge_recon_acc)
            elif trainer.last_recon == trainer.last_recon:
                ep_recon.append(float("nan"))
        val_stats = _eval_split(model, val, max_nodes=max_nodes, T=DEFAULT_T)
        ov = float(val_stats["overall_acc"])
        hist.append(
            {
                "epoch": epoch,
                "train_loss": _mean(ep_loss),
                "train_acc": _mean(ep_acc),
                "train_ce": _mean(ep_ce),
                "train_edge_recon_acc": _mean(
                    [x for x in ep_recon if x == x]
                ),
                "val_acc": ov,
                "val_loss": float(val_stats["overall_loss"]),
                "edge_recon_weight": EDGE_RECON_WEIGHT,
            }
        )
        if ov > best_val_acc:
            best_val_acc = ov
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"[ste-no-aux seed={seed}] ep {epoch}/{epochs} "
            f"train={_mean(ep_acc):.4f} ce={_mean(ep_ce):.4f} "
            f"recon_telemetry={_mean([x for x in ep_recon if x == x]):.4f} "
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
                "arm": f"sheaf-ste-no-aux-T{DEFAULT_T}-d{DEFAULT_D}-mlp{DEFAULT_MLP}",
                "science_open": False,
                "cycle": CYCLE,
                "neutral_init": True,
                "edge_recon_weight": EDGE_RECON_WEIGHT,
                "aux_edge_recon": False,
                "seed": seed,
                "hparams": {
                    "d": DEFAULT_D,
                    "T": DEFAULT_T,
                    "mlp_expansion": DEFAULT_MLP,
                    "max_nodes": max_nodes,
                    "lr": DEFAULT_LR,
                    "grad_clip": DEFAULT_CLIP,
                    "gate_theta": DEFAULT_GATE_THETA,
                    "gate_mode": GATE_MODE,
                    "residual_alpha": 1.0,
                    "gate_detach_diffusion": GATE_DETACH_DIFFUSION,
                    "edge_recon_weight": EDGE_RECON_WEIGHT,
                    "aux_edge_recon": False,
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
        "edge_recon_weight": EDGE_RECON_WEIGHT,
        "science_open": False,
        "model": model,
    }


def _overfit_ste_no_aux(examples: list[dict[str, Any]], *, max_nodes: int) -> dict[str, Any]:
    """Gate0 overfit with neutral_init and aux OFF (CE only)."""
    import torch
    from reachability_gen.overfit_ff import (
        BALANCED_LOSS_THRESHOLD,
        _per_class_accuracy,
    )
    from reachability_gen.train.sheaf_trainer import SheafTrainer

    torch.manual_seed(0)
    batch = build_sheaf_batch(examples, max_n=max_nodes)
    model = SheafInferCore(
        d=DEFAULT_D,
        T=DEFAULT_T,
        mlp_expansion=DEFAULT_MLP,
        max_nodes=max_nodes,
        max_T=max(DEFAULT_T, max(DISCRETE_T_VALUES)),
        residual_alpha=1.0,
        gate_detach_diffusion=GATE_DETACH_DIFFUSION,
        gate_mode=GATE_MODE,
        gumbel_temp=GUMBEL_TEMP,
        neutral_init=True,
    )
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    trainer = SheafTrainer(
        model,
        lr=3e-3,
        weight_decay=0.01,
        grad_clip=DEFAULT_CLIP,
        edge_recon_weight=EDGE_RECON_WEIGHT,
    )
    passed_at = None
    last: dict[str, Any] = {}
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
        # With aux OFF, Gate0 pass is CE+acc+per-class only (no recon floor).
        last = {
            "step": step,
            "eval_loss": float(eval_loss),
            "eval_acc": float(eval_acc),
            "recon_offdiag": None,
            "per_class": pc,
            "edge_recon_weight": EDGE_RECON_WEIGHT,
        }
        if (
            eval_loss < BALANCED_LOSS_THRESHOLD
            and eval_acc >= 1.0 - 1e-9
            and pc.get("y0", 0) >= 1.0 - 1e-9
            and pc.get("y1", 0) >= 1.0 - 1e-9
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
        "aux_edge_recon": False,
        "edge_recon_weight": EDGE_RECON_WEIGHT,
        "science_open": False,
        "note": "Gate0 without recon floor (aux OFF); CE+acc+per-class only.",
    }


def quick_stalk_untrained_control(
    ood_rows: list[dict[str, Any]], *, max_nodes: int
) -> dict[str, Any]:
    """Report-only: FractalCore stalk untrained on matched-OOD T16."""
    import torch
    from reachability_gen.models.fractal_core import FractalCore
    from reachability_gen.run_fractal_core_gate1 import _eval_split as stalk_eval

    torch.manual_seed(0)
    # Match stalk seal mlp×10 / d=64 for fair report.
    model = FractalCore(
        d=64,
        T=6,
        mlp_expansion=10,
        max_nodes=max_nodes,
        max_T=16,
        residual_alpha=0.5,
        use_tau=True,
        apply_cycle_rmsnorm=True,
        adaptive_halt=False,
    )
    n_params = (
        model.param_count()
        if hasattr(model, "param_count")
        else sum(p.numel() for p in model.parameters())
    )
    t16 = stalk_eval(model, ood_rows, max_nodes=max_nodes, T=16)
    t6 = stalk_eval(model, ood_rows, max_nodes=max_nodes, T=6)
    return {
        "report_only": True,
        "arm": "fractal-stalk-untrained",
        "param_count": int(n_params),
        "matched_ood_T16": {
            "overall_acc": float(t16["overall_acc"]),
            "hard_neg_acc": float(t16["hard_neg_acc"]),
        },
        "matched_ood_T6": {
            "overall_acc": float(t6["overall_acc"]),
            "hard_neg_acc": float(t6["hard_neg_acc"]),
        },
        "note": (
            "Cheap untrained FractalCore stalk control for context only — "
            "not a sheaf Gate. science_open=false."
        ),
        "science_open": False,
    }


def load_pr8_baseline(path: Path = PR8_BASELINE_ARTIFACT) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text())
    ms = data.get("gates", {}).get("D", {}).get("mean_std", {})
    return {
        "available": True,
        "path": str(path),
        "cycle": data.get("cycle"),
        "verdict": data.get("verdict"),
        "pr8_merge_sha": PR8_MERGE_SHA,
        "edge_recon_weight": data.get("hparams", {}).get(
            "edge_recon_weight", 1.0
        ),
        "mean_std": {
            k: {"mean": v.get("mean"), "std": v.get("std")}
            for k, v in ms.items()
        },
        "science_open": False,
    }


def compare_to_with_aux(
    no_aux_summary: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    if not baseline.get("available"):
        return {"available": False}
    base_ms = baseline.get("mean_std", {})
    deltas: dict[str, Any] = {}
    for key in (
        "T16_overall",
        "T16_hard_neg",
        "T16_K16",
        "T16_K8",
        "T16_fpr",
        "T16_fnr",
        "ahat_fpr",
        "ahat_fnr",
        "val_acc",
    ):
        na = no_aux_summary.get(key, {})
        wa = base_ms.get(key, {})
        if not na or not wa:
            continue
        na_m = na.get("mean")
        wa_m = wa.get("mean")
        if na_m is None or wa_m is None:
            continue
        if na_m != na_m or wa_m != wa_m:
            continue
        deltas[key] = {
            "no_aux_mean": na_m,
            "with_aux_mean": wa_m,
            "delta_no_aux_minus_with_aux": na_m - wa_m,
        }
    return {
        "available": True,
        "pr8_merge_sha": baseline.get("pr8_merge_sha"),
        "with_aux_verdict": baseline.get("verdict"),
        "deltas": deltas,
        "science_open": False,
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    cue_path: Path = DEFAULT_CUE,
    epochs: int = DEFAULT_EPOCHS_STE,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    skip_overfit: bool = False,
    skip_stalk: bool = False,
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

    pr8_baseline = load_pr8_baseline()

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "base_sha": _git_sha(),
        "pr8_merge_sha": PR8_MERGE_SHA,
        "aux_edge_recon": False,
        "edge_recon_weight": EDGE_RECON_WEIGHT,
        "gate_detach_diffusion": GATE_DETACH_DIFFUSION,
        "gate_mode": GATE_MODE,
        "gumbel_temp": GUMBEL_TEMP,
        "pr9_merge_sha": PR9_MERGE_SHA,
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "lr": DEFAULT_LR,
            "epochs": epochs,
            "grad_clip": DEFAULT_CLIP,
            "batch_size": DEFAULT_BATCH,
            "neutral_init": True,
            "edge_recon_weight": EDGE_RECON_WEIGHT,
            "aux_edge_recon": False,
            "gate_detach_diffusion": GATE_DETACH_DIFFUSION,
            "gate_mode": GATE_MODE,
            "gumbel_temp": GUMBEL_TEMP,
            "param_parity_window": [115157, 127279],
            "ff_baseline": FF_BASELINE_PARAMS,
            "same_as_pr9_except": (
                "gate_detach_diffusion=False (STE/Gumbel end-to-end into Φ); "
                f"epochs={epochs} (longer budget)"
            ),
        },
        "seeds": list(seeds),
        "gates": {},
        "pr8_with_aux_baseline": pr8_baseline,
        "scope": {
            "protocol": (
                "Gates A–D; neutral_init; edge_recon_weight=0; "
                "gate_detach_diffusion=False (STE). Longer budget. "
                "Prereg floors unchanged. Compare to PR#8/#9."
            ),
            "datasets": {
                "id": str(id_data),
                "matched_ood": str(ood_data),
            },
        },
    }

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

    bake = gate_a["init"]["bake_in_removed"]
    gate_b = {
        "gate": "B",
        "passed": bool(all(bake.values())),
        "bake_in_removed": bake,
        "init": gate_a["init"],
        "note": (
            "neutral_init=True: Xavier/Gaussian zero-mean; biases 0; "
            "no edge+4 / absent−4 / energy±5. STE OK for learning Â. "
            "This cycle: aux OFF + gate_detach_diffusion=False — CE trains Â "
            "via STE/Gumbel into Φ, or remain chance."
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

    print("[gate C] reach-cue audit + degree-balanced hard-negs…", file=sys.stderr)
    cue_ood = audit_reach_cues(ood_rows, ceiling=0.52)
    cue_id = audit_reach_cues(rows, ceiling=0.52)
    deg_bal = build_degree_balanced_eval(ood_rows, seed=0)
    cue_bal = audit_reach_cues(deg_bal["rows"], ceiling=0.52)
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
        extra={"cycle": CYCLE, "science_open": False, "aux_edge_recon": False, "gate_detach_diffusion": GATE_DETACH_DIFFUSION, "gate_mode": GATE_MODE},
    )
    report["gates"]["C"] = gate_c
    if not gate_c["passed"]:
        report["verdict"] = "STOP_GATE_C_CUE_CEILING"
        report["residue"] = (
            f"Degree-balanced reach-cue max_acc={cue_bal['max_acc']:.4f} > 0.52; "
            "do not trust model metrics. No OPEN."
        )
        report["elapsed_s"] = time.time() - t0
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    if not skip_stalk:
        print("[stalk] quick untrained FractalCore control (report only)…", file=sys.stderr)
        report["stalk_untrained_control"] = quick_stalk_untrained_control(
            ood_rows, max_nodes=max_nodes
        )
    else:
        report["stalk_untrained_control"] = {"skipped": True, "science_open": False}

    overfit_summary: dict[str, Any] = {"skipped": skip_overfit}
    if not skip_overfit:
        print("[gate0] overfit balanced (aux OFF)…", file=sys.stderr)
        batch, note, _meta = ensure_balanced_batch(
            Path("data/train_tiny.jsonl"), n_pos=16, n_neg=16, regenerate=True
        )
        overfit_summary = _overfit_ste_no_aux(batch, max_nodes=max_nodes)
        overfit_summary["balanced_note"] = note

    print(f"[gate D] train {len(seeds)} seeds STE no-aux (detach={GATE_DETACH_DIFFUSION}, mode={GATE_MODE}, ep={epochs})…", file=sys.stderr)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        ckpt = Path(f"artifacts/sheaf_ste_no_aux_gate1_seed{seed}_best.pt")
        tr = train_ste_no_aux_seed(
            train, val, seed=seed, max_nodes=max_nodes, epochs=epochs, ckpt_path=ckpt
        )
        model = tr.pop("model")
        ood_eval = eval_ood_full(model, ood_rows, max_nodes=max_nodes)
        bal_stats = _eval_split(model, deg_bal["rows"], max_nodes=max_nodes, T=16)
        ahat_trained = _ahat_fpr_fnr(model, ood_rows, max_nodes=max_nodes)
        # Untrained vs trained agreement on seed0 T16
        seed_row = {
            "seed": seed,
            "param_count": tr["param_count"],
            "param_parity_ok": tr["param_parity"]["within_5pct"],
            "untrained_val_acc": tr["untrained_val_acc"],
            "best_epoch": tr["best_epoch"],
            "best_val_acc": tr["best_val_acc"],
            "checkpoint_path": tr["checkpoint_path"],
            "edge_recon_weight": EDGE_RECON_WEIGHT,
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
        "T16_K12": _agg(["matched_ood", "T16_K12"]),
        "T16_K8": _agg(["matched_ood", "T16_K8"]),
        "T16_fpr": _agg(["matched_ood", "T16_fpr"]),
        "T16_fnr": _agg(["matched_ood", "T16_fnr"]),
        "degbal_T16_overall": _agg(["degree_balanced_ood_T16", "overall_acc"]),
        "degbal_T16_hard_neg": _agg(["degree_balanced_ood_T16", "hard_neg_acc"]),
        "ahat_fpr": _agg(["ahat_trained", "ahat_fpr"]),
        "ahat_fnr": _agg(["ahat_trained", "ahat_fnr"]),
        "ahat_edge_acc": _agg(["ahat_trained", "ahat_edge_acc"]),
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
        "edge_recon_weight": EDGE_RECON_WEIGHT,
        "science_open": False,
    }
    report["gates"]["D"] = gate_d
    report["gate0_overfit"] = overfit_summary
    report["reach_cue_artifact"] = str(cue_path)
    report["comparison_to_pr8_with_aux"] = compare_to_with_aux(summary, pr8_baseline)

    # Untrained vs trained agreement (seed0) — learning signal check
    import torch

    torch.manual_seed(0)
    untrained = SheafInferCore(
        d=DEFAULT_D,
        T=DEFAULT_T,
        mlp_expansion=DEFAULT_MLP,
        max_nodes=max_nodes,
        max_T=max(DEFAULT_T, max(DISCRETE_T_VALUES)),
        residual_alpha=1.0,
        gate_detach_diffusion=GATE_DETACH_DIFFUSION,
        gate_theta=DEFAULT_GATE_THETA,
        gate_mode=GATE_MODE,
        gumbel_temp=GUMBEL_TEMP,
        neutral_init=True,
    )
    # Load seed0 trained if present
    agree = float("nan")
    ckpt0 = Path("artifacts/sheaf_ste_no_aux_gate1_seed0_best.pt")
    if ckpt0.exists() and per_seed:
        trained = SheafInferCore(
            d=DEFAULT_D,
            T=DEFAULT_T,
            mlp_expansion=DEFAULT_MLP,
            max_nodes=max_nodes,
            max_T=max(DEFAULT_T, max(DISCRETE_T_VALUES)),
            residual_alpha=1.0,
            gate_detach_diffusion=GATE_DETACH_DIFFUSION,
            gate_theta=DEFAULT_GATE_THETA,
            gate_mode=GATE_MODE,
            gumbel_temp=GUMBEL_TEMP,
            neutral_init=True,
        )
        blob = torch.load(ckpt0, map_location="cpu", weights_only=False)
        trained.load_state_dict(blob["state_dict"])
        untrained.eval()
        trained.eval()
        agree_n = 0
        agree_hit = 0
        with torch.no_grad():
            for start in range(0, len(ood_rows), 64):
                br = ood_rows[start : start + 64]
                batch = build_sheaf_batch(br, max_n=max_nodes)
                lu, _, _ = untrained(
                    batch["node_ids"],
                    batch["node_mask"],
                    batch["edge_index"],
                    batch["edge_mask"],
                    batch["s_idx"],
                    batch["t_idx"],
                    T=16,
                )
                lt, _, _ = trained(
                    batch["node_ids"],
                    batch["node_mask"],
                    batch["edge_index"],
                    batch["edge_mask"],
                    batch["s_idx"],
                    batch["t_idx"],
                    T=16,
                )
                pu = lu.argmax(-1)
                pt = lt.argmax(-1)
                agree_hit += int((pu == pt).sum())
                agree_n += int(pu.numel())
        agree = agree_hit / agree_n if agree_n else float("nan")
    report["untrained_vs_trained_seed0_matched_ood_T16"] = {
        "prediction_agreement": agree,
        "note": "≈0.5 ⇒ untrained≪trained (learning signal). ≈1.0 ⇒ bake-in/residue.",
        "science_open": False,
    }

    if floors_pass:
        report["verdict"] = "MEASURE_CANDIDATE_PASS_FLOORS"
        report["residue"] = (
            "STE no-aux (edge_recon_weight=0, gate_detach_diffusion=False) meets "
            "prereg floors across seeds. MEASURE candidate only — "
            "do NOT stamp science_open."
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
            "STE no-aux failed prereg floors. STOP with residue: end-to-end STE "
            "without aux still insufficient under this budget, or needs "
            "Gumbel / different coupling. science_open=false."
        )

    report["elapsed_s"] = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[ste-no-aux] verdict={report['verdict']} wrote {out_path} "
        f"elapsed={report['elapsed_s']:.1f}s",
        file=sys.stderr,
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cue-out", type=Path, default=DEFAULT_CUE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS_STE)
    p.add_argument("--gate-mode", type=str, default=GATE_MODE, choices=["ste", "gumbel"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--skip-overfit", action="store_true")
    p.add_argument("--skip-stalk", action="store_true")
    args = p.parse_args(argv)
    import reachability_gen.run_sheaf_ste_no_aux as _mod
    _mod.GATE_MODE = args.gate_mode
    report = run_cycle(
        id_data=args.id_data,
        ood_data=args.ood_data,
        out_path=args.out,
        cue_path=args.cue_out,
        epochs=args.epochs,
        seeds=tuple(args.seeds),
        skip_overfit=args.skip_overfit,
        skip_stalk=args.skip_stalk,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "science_open": report["science_open"],
                "edge_recon_weight": report["edge_recon_weight"],
                "gate_detach_diffusion": report.get("gate_detach_diffusion"),
                "gate_mode": report.get("gate_mode"),
                "pr8_merge_sha": report["pr8_merge_sha"],
                "pr9_merge_sha": report.get("pr9_merge_sha"),
                "gates_passed": {
                    g: report["gates"][g].get("passed") for g in report["gates"]
                },
                "mean_std": report.get("gates", {}).get("D", {}).get("mean_std"),
                "comparison_to_pr8_with_aux": report.get(
                    "comparison_to_pr8_with_aux"
                ),
                "stalk_untrained_control": report.get("stalk_untrained_control"),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
