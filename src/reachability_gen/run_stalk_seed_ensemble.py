"""CYCLE_STALK_SEED_ENSEMBLE — MEASURE inference ensemble of frozen #14/#18 stalks.

Freeze hard-A stalk singles (PR #14 seeds 0..4 + PR #18 seeds 5..9). NO new
select/upsample train unless a ckpt is missing. Eval matched-OOD T∈{6,8,12,16}
with prereg primary aggregator **prob_mean** (secondary: logit_mean, majority_vote).
Report ensemble overall/HN/K16 vs mean of singles; optional leave-one-out.

Prereg floors (ensemble primary @ T16): hard-neg≥0.95 AND K16≥0.75.
  PASS_CANDIDATE → floors PASS + lift HN and K16 vs singles mean; science_open=false
  MEASURE_LIFT   → ≥0.01 abs lift on overall/HN/K16 vs singles mean
  STOP_NO_LIFT   → no such lift

Usage::

    python -m reachability_gen.run_stalk_seed_ensemble
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
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
    _n_heads,
    _split_train_val,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS
from reachability_gen.run_stalk_stabilize_multi_seed import (
    LR_MAX,
    LR_MIN,
    train_fractal_id2k_harden,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_SEED_ENSEMBLE"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_seed_ensemble.json")
DEFAULT_SEEDS = tuple(range(10))  # 0..9
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16

DEFAULT_EPOCHS = 60  # only if filling missing seeds
DEFAULT_CLIP_H = DEFAULT_CLIP
PREREG_HARD_NEG = 0.95
PREREG_K16 = 0.75
LIFT_EPS = 0.01  # absolute lift vs singles mean for MEASURE_LIFT
PRIMARY_AGG = "prob_mean"
SECONDARY_AGGS = ("logit_mean", "majority_vote")


def _ckpt_for_seed(seed: int) -> Path:
    if 0 <= seed <= 4:
        return Path(f"artifacts/fractal_core_stalk_stabilize_seed{seed}_best.pt")
    return Path(f"artifacts/fractal_core_stalk_seed_stability_seed{seed}_best.pt")


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


def _load_model_from_ckpt(ckpt: Path, max_nodes: int):
    import torch

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
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def _metrics_from_preds(
    preds: Any,
    labels: Any,
    hops: list[int],
) -> dict[str, Any]:
    """Compute overall / hard-neg / K strata from 1d pred/label tensors + hops."""
    import torch

    correct = (preds == labels).float()
    hop_accs: dict[int, list[float]] = defaultdict(list)
    for i, hop in enumerate(hops):
        hop_accs[int(hop)].append(float(correct[i].item()))
    by_hop: dict[str, Any] = {}
    for k in sorted(hop_accs):
        by_hop[str(k)] = {
            "n": len(hop_accs[k]),
            "acc_mean": _mean(hop_accs[k]),
            "correct": int(round(sum(hop_accs[k]))),
        }
    n_tot = sum(v["n"] for v in by_hop.values())
    overall = (
        sum(v["acc_mean"] * v["n"] for v in by_hop.values()) / n_tot
        if n_tot
        else float("nan")
    )
    hard = by_hop.get(str(HOP_UNREACHABLE), by_hop.get("-1", {}))
    return {
        "overall_acc": float(overall),
        "hard_neg_acc": float(hard.get("acc_mean", float("nan"))) if hard else float("nan"),
        "hard_neg_n": int(hard.get("n", 0)) if hard else 0,
        "K8": float(by_hop.get("8", {}).get("acc_mean", float("nan"))),
        "K12": float(by_hop.get("12", {}).get("acc_mean", float("nan"))),
        "K16": float(by_hop.get("16", {}).get("acc_mean", float("nan"))),
        "by_hop": by_hop,
        "n": n_tot,
    }


def _aggregate_preds(
    logits_stack: Any,
    *,
    method: str,
) -> Any:
    """logits_stack: (n_members, n, C) → preds (n,)."""
    import torch
    import torch.nn.functional as F

    if method == "prob_mean":
        probs = F.softmax(logits_stack, dim=-1)
        return probs.mean(dim=0).argmax(dim=-1)
    if method == "logit_mean":
        return logits_stack.mean(dim=0).argmax(dim=-1)
    if method == "majority_vote":
        hard = logits_stack.argmax(dim=-1)  # (M, n)
        # vote for class 1 if sum>= ceil(M/2) ... use mode; ties → 0
        # For binary: sum of hard preds; if sum > M/2 → 1; if sum < M/2 → 0; equal → 0
        m = hard.shape[0]
        votes = hard.sum(dim=0)
        # strict majority; ties → 0
        return (votes > (m / 2.0)).long()
    raise ValueError(f"unknown aggregate method: {method}")


def _collect_member_logits(
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    batch_size: int = 64,
) -> tuple[Any, Any, list[int]]:
    """Return (logits_stack[M,N,C], labels[N], hops[N])."""
    import torch

    for m in models:
        m.eval()
    logit_chunks: list[Any] = []
    label_chunks: list[Any] = []
    hops: list[int] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            member_logits = []
            for model in models:
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
                member_logits.append(logits)
            stacked = torch.stack(member_logits, dim=0)  # (M, B, C)
            logit_chunks.append(stacked)
            label_chunks.append(batch["labels"])
            for ex in batch_rows:
                hops.append(int(ex.get("hop_distance", HOP_UNREACHABLE)))
    logits_all = torch.cat(logit_chunks, dim=1)  # (M, N, C)
    labels_all = torch.cat(label_chunks, dim=0)
    return logits_all, labels_all, hops


def _eval_ensemble_methods(
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    methods: tuple[str, ...] = (PRIMARY_AGG,) + SECONDARY_AGGS,
) -> dict[str, Any]:
    logits, labels, hops = _collect_member_logits(
        models, rows, T=T, max_nodes=max_nodes
    )
    out: dict[str, Any] = {"T": T, "n_members": len(models), "n": len(hops)}
    for method in methods:
        preds = _aggregate_preds(logits, method=method)
        metrics = _metrics_from_preds(preds, labels, hops)
        metrics["method"] = method
        out[method] = metrics
    # also per-member singles from same logits
    singles: list[dict[str, Any]] = []
    for mi in range(logits.shape[0]):
        preds_i = logits[mi].argmax(dim=-1)
        m = _metrics_from_preds(preds_i, labels, hops)
        m["member_index"] = mi
        singles.append(m)
    out["singles"] = singles
    out["singles_mean"] = {
        "overall_acc": _mean([s["overall_acc"] for s in singles]),
        "hard_neg_acc": _mean([s["hard_neg_acc"] for s in singles]),
        "K8": _mean([s["K8"] for s in singles]),
        "K12": _mean([s["K12"] for s in singles]),
        "K16": _mean([s["K16"] for s in singles]),
        "std_overall": _std([s["overall_acc"] for s in singles]),
        "std_hard_neg": _std([s["hard_neg_acc"] for s in singles]),
        "std_K16": _std([s["K16"] for s in singles]),
    }
    return out


def _ensure_ckpts(
    seeds: tuple[int, ...],
    *,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    max_nodes: int,
    epochs: int,
    lr_max: float,
    lr_min: float,
    grad_clip: float,
) -> list[dict[str, Any]]:
    """Resolve ckpts; train missing under frozen #14 harden only."""
    meta: list[dict[str, Any]] = []
    for seed in seeds:
        ckpt = _ckpt_for_seed(seed)
        if ckpt.exists():
            meta.append(
                {
                    "seed": seed,
                    "checkpoint_path": str(ckpt),
                    "mode": "existing",
                    "source": "pr14" if seed <= 4 else "pr18",
                    "retrained": False,
                }
            )
            continue
        print(
            f"[stalk-seed-ens] MISSING ckpt seed={seed} → train frozen #14 harden",
            file=sys.stderr,
        )
        tr = train_fractal_id2k_harden(
            train_rows,
            val_rows,
            epochs=epochs,
            seed=seed,
            max_nodes=max_nodes,
            ckpt_path=ckpt,
            lr_max=lr_max,
            lr_min=lr_min,
            grad_clip=grad_clip,
        )
        meta.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "mode": "trained_fill_missing",
                "source": "pr14_freeze_fill",
                "retrained": True,
                "best_epoch": tr.get("best_epoch"),
            }
        )
    return meta


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    grad_clip: float = DEFAULT_CLIP_H,
    leave_one_out: bool = True,
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

    member_meta = _ensure_ckpts(
        seeds,
        train_rows=train,
        val_rows=val,
        max_nodes=max_nodes,
        epochs=epochs,
        lr_max=lr_max,
        lr_min=lr_min,
        grad_clip=grad_clip,
    )
    n_trained_fill = sum(1 for m in member_meta if m["retrained"])

    print(f"[stalk-seed-ens] loading {len(seeds)} members", file=sys.stderr)
    models: list[Any] = []
    for m in member_meta:
        model, blob = _load_model_from_ckpt(Path(m["checkpoint_path"]), max_nodes)
        m["param_count"] = model.param_count()
        m["param_parity"] = _verify_param_parity(
            model.param_count(), ff_baseline=FF_BASELINE_PARAMS
        )
        m["best_epoch"] = m.get("best_epoch", blob.get("epoch"))
        models.append(model)

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "base_sha": _git_sha(),
        "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
        "frozen_recipe": (
            "Inference ensemble of PR#14/#18 hard-A stalk singles; "
            "NO new select/upsample; primary=prob_mean"
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
        "ensemble_knobs": {
            "n_members": len(seeds),
            "seeds": list(seeds),
            "primary_aggregator": PRIMARY_AGG,
            "secondary_aggregators": list(SECONDARY_AGGS),
            "leave_one_out": leave_one_out,
            "lift_eps": LIFT_EPS,
            "n_trained_fill_missing": n_trained_fill,
            "train_policy": (
                "eval-only if all ckpts present; else fill missing under "
                "frozen #14 harden only (no select/upsample change)"
            ),
        },
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train_of_members": DEFAULT_T,
            "batch_size": DEFAULT_BATCH,
            "ff_baseline": FF_BASELINE_PARAMS,
            "matched_to": "PR#14/#18 freeze + PR#12 floors",
        },
        "prereg": {
            "hard_neg_ensemble_ge": PREREG_HARD_NEG,
            "K16_at_T16_ensemble_ge": PREREG_K16,
            "primary_aggregator": PRIMARY_AGG,
            "lift_eps": LIFT_EPS,
            "selection_rule": (
                "No new ckpt selection. Members = existing #14/#18 bests. "
                "Floors on matched-OOD after ensemble aggregate."
            ),
            "note": (
                "science_open never self-stamped; PASS_CANDIDATE still leaves "
                "science_open=false. Select/curriculum remain CLOSED."
            ),
        },
        "members": member_meta,
        "degree_balanced_construction": {
            k: v for k, v in deg_bal.items() if k != "rows"
        },
        "datasets": {"id": str(id_data), "matched_ood": str(ood_data)},
        "policy": (
            "PASS_CANDIDATE → floors+dual lift; science_open=false. "
            "MEASURE_LIFT → ≥0.01 lift on overall/HN/K16. "
            "STOP_NO_LIFT → no lift. Sheaf unsupervised ignored."
        ),
    }

    matched_by_T: dict[str, Any] = {}
    deg_by_T: dict[str, Any] = {}
    for T in T_VALUES:
        print(f"[stalk-seed-ens] matched-OOD ensemble T={T}", file=sys.stderr)
        matched_by_T[str(T)] = _eval_ensemble_methods(
            models, ood_rows, T=int(T), max_nodes=max_nodes
        )
        print(f"[stalk-seed-ens] deg-bal ensemble T={T}", file=sys.stderr)
        deg_by_T[str(T)] = _eval_ensemble_methods(
            models, deg_rows, T=int(T), max_nodes=max_nodes
        )

    t16 = matched_by_T[str(FOCUS_T)]
    primary = t16[PRIMARY_AGG]
    singles_mean = t16["singles_mean"]

    def _delta(ens: float, base: float) -> float:
        if ens != ens or base != base:
            return float("nan")
        return ens - base

    deltas = {
        "overall": _delta(primary["overall_acc"], singles_mean["overall_acc"]),
        "hard_neg": _delta(primary["hard_neg_acc"], singles_mean["hard_neg_acc"]),
        "K16": _delta(primary["K16"], singles_mean["K16"]),
        "K8": _delta(primary["K8"], singles_mean["K8"]),
        "K12": _delta(primary["K12"], singles_mean["K12"]),
    }

    floors_pass = bool(
        primary["hard_neg_acc"] == primary["hard_neg_acc"]
        and primary["hard_neg_acc"] >= PREREG_HARD_NEG
        and primary["K16"] == primary["K16"]
        and primary["K16"] >= PREREG_K16
    )
    lift_hn = deltas["hard_neg"] == deltas["hard_neg"] and deltas["hard_neg"] > 0
    lift_k16 = deltas["K16"] == deltas["K16"] and deltas["K16"] > 0
    any_lift_eps = any(
        deltas[k] == deltas[k] and deltas[k] >= LIFT_EPS
        for k in ("overall", "hard_neg", "K16")
    )

    if floors_pass and lift_hn and lift_k16:
        verdict = "PASS_CANDIDATE"
        residue = (
            f"Ensemble PASS_CANDIDATE under frozen #14/#18 members (primary={PRIMARY_AGG}): "
            f"T16 HN={primary['hard_neg_acc']:.4f}≥{PREREG_HARD_NEG}, "
            f"K16={primary['K16']:.4f}≥{PREREG_K16}; "
            f"lift vs singles mean ΔHN={deltas['hard_neg']:+.4f} ΔK16={deltas['K16']:+.4f}. "
            "FLAG human — harness keeps science_open=false. Select/curriculum stay CLOSED."
        )
        open_status = "PASS_CANDIDATE_SCIENCE_OPEN_FALSE"
    elif any_lift_eps:
        verdict = "MEASURE_LIFT"
        residue = (
            f"Ensemble MEASURE_LIFT (primary={PRIMARY_AGG}): "
            f"T16 overall={primary['overall_acc']:.4f} HN={primary['hard_neg_acc']:.4f} "
            f"K16={primary['K16']:.4f}; vs singles mean "
            f"Δov={deltas['overall']:+.4f} ΔHN={deltas['hard_neg']:+.4f} "
            f"ΔK16={deltas['K16']:+.4f} (lift_eps={LIFT_EPS}). "
            f"floors_pass={floors_pass}. science_open=false."
        )
        open_status = "MEASURE_LIFT"
    else:
        verdict = "STOP_NO_LIFT"
        residue = (
            f"Ensemble STOP_NO_LIFT (primary={PRIMARY_AGG}): "
            f"T16 overall={primary['overall_acc']:.4f} HN={primary['hard_neg_acc']:.4f} "
            f"K16={primary['K16']:.4f}; vs singles mean "
            f"Δov={deltas['overall']:+.4f} ΔHN={deltas['hard_neg']:+.4f} "
            f"ΔK16={deltas['K16']:+.4f} — no ≥{LIFT_EPS} lift. science_open=false."
        )
        open_status = "STOP_NO_LIFT"

    # Leave-one-out (primary only @ focus T + all T for completeness)
    loo: dict[str, Any] = {"enabled": leave_one_out}
    if leave_one_out:
        loo_rows: list[dict[str, Any]] = []
        for hold in range(len(models)):
            subset = [models[i] for i in range(len(models)) if i != hold]
            print(
                f"[stalk-seed-ens] LOO hold seed={seeds[hold]} n={len(subset)} T={FOCUS_T}",
                file=sys.stderr,
            )
            cell = _eval_ensemble_methods(
                subset,
                ood_rows,
                T=FOCUS_T,
                max_nodes=max_nodes,
                methods=(PRIMARY_AGG,),
            )
            p = cell[PRIMARY_AGG]
            loo_rows.append(
                {
                    "held_out_seed": seeds[hold],
                    "n_members": len(subset),
                    "overall_acc": p["overall_acc"],
                    "hard_neg_acc": p["hard_neg_acc"],
                    "K8": p["K8"],
                    "K12": p["K12"],
                    "K16": p["K16"],
                }
            )
        loo["per_held_out"] = loo_rows
        loo["mean_std"] = {
            "overall": {
                "mean": _mean([r["overall_acc"] for r in loo_rows]),
                "std": _std([r["overall_acc"] for r in loo_rows]),
            },
            "hard_neg": {
                "mean": _mean([r["hard_neg_acc"] for r in loo_rows]),
                "std": _std([r["hard_neg_acc"] for r in loo_rows]),
            },
            "K16": {
                "mean": _mean([r["K16"] for r in loo_rows]),
                "std": _std([r["K16"] for r in loo_rows]),
            },
        }

    # Secondary methods summary @ T16
    secondary_summary = {
        m: {
            "overall_acc": t16[m]["overall_acc"],
            "hard_neg_acc": t16[m]["hard_neg_acc"],
            "K16": t16[m]["K16"],
            "delta_vs_singles_mean": {
                "overall": _delta(t16[m]["overall_acc"], singles_mean["overall_acc"]),
                "hard_neg": _delta(t16[m]["hard_neg_acc"], singles_mean["hard_neg_acc"]),
                "K16": _delta(t16[m]["K16"], singles_mean["K16"]),
            },
        }
        for m in SECONDARY_AGGS
    }

    # Per-seed singles table @ T16 (attach seed ids)
    singles_table = []
    for i, s in enumerate(t16["singles"]):
        seed = seeds[i]
        hn = s["hard_neg_acc"]
        k16 = s["K16"]
        singles_table.append(
            {
                "seed": seed,
                "overall_acc": s["overall_acc"],
                "hard_neg_acc": hn,
                "K8": s["K8"],
                "K12": s["K12"],
                "K16": k16,
                "prereg_pass": bool(
                    hn == hn and hn >= PREREG_HARD_NEG and k16 == k16 and k16 >= PREREG_K16
                ),
            }
        )

    report.update(
        {
            "matched_ood_by_T": matched_by_T,
            "degree_balanced_ood_by_T": deg_by_T,
            "T16_primary": {
                "method": PRIMARY_AGG,
                "overall_acc": primary["overall_acc"],
                "hard_neg_acc": primary["hard_neg_acc"],
                "K8": primary["K8"],
                "K12": primary["K12"],
                "K16": primary["K16"],
            },
            "T16_singles_mean": singles_mean,
            "T16_singles_table": singles_table,
            "T16_deltas_vs_singles_mean": deltas,
            "T16_secondary": secondary_summary,
            "T16_degbal_primary": {
                "method": PRIMARY_AGG,
                "overall_acc": deg_by_T[str(FOCUS_T)][PRIMARY_AGG]["overall_acc"],
                "hard_neg_acc": deg_by_T[str(FOCUS_T)][PRIMARY_AGG]["hard_neg_acc"],
                "K16": deg_by_T[str(FOCUS_T)][PRIMARY_AGG]["K16"],
            },
            "leave_one_out": loo,
            "prereg_floors_pass": floors_pass,
            "verdict": verdict,
            "open_status": open_status,
            "residue": residue,
            "elapsed_sec": time.time() - t0,
            "science_open": False,
            "prefer_corridor": "PR#14 MEASURE_STILL singles; ensemble is inference overlay",
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[stalk-seed-ens] wrote {out_path}", file=sys.stderr)
    print(
        f"[stalk-seed-ens] verdict={verdict} science_open=False "
        f"primary={PRIMARY_AGG} T16 ov={primary['overall_acc']:.4f} "
        f"HN={primary['hard_neg_acc']:.4f} K16={primary['K16']:.4f} "
        f"ΔHN={deltas['hard_neg']:+.4f} ΔK16={deltas['K16']:+.4f}",
        file=sys.stderr,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr-max", type=float, default=LR_MAX)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CLIP_H)
    p.add_argument(
        "--no-loo",
        action="store_true",
        help="Skip leave-one-out (default: run LOO)",
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
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        grad_clip=args.grad_clip,
        leave_one_out=not args.no_loo,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "open_status": report["open_status"],
                "science_open": report["science_open"],
                "prereg_floors_pass": report["prereg_floors_pass"],
                "T16_primary": report["T16_primary"],
                "T16_singles_mean": {
                    k: report["T16_singles_mean"][k]
                    for k in (
                        "overall_acc",
                        "hard_neg_acc",
                        "K16",
                        "std_overall",
                        "std_hard_neg",
                        "std_K16",
                    )
                },
                "T16_deltas_vs_singles_mean": report["T16_deltas_vs_singles_mean"],
                "T16_secondary": report["T16_secondary"],
                "leave_one_out_mean_std": report["leave_one_out"].get("mean_std"),
                "n_trained_fill_missing": report["ensemble_knobs"][
                    "n_trained_fill_missing"
                ],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
