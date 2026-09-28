"""CYCLE_STALK_SEED_STABILITY — MEASURE seed envelope under frozen #14.

Freeze PR #14 recipe exactly (0.5/0.5 select, 60ep cosine, hard-A, uniform ID).
NO new select weights. NO train upsample. Seeds 0..9:
  - 0..4 reconfirm existing #14 ckpts (eval only)
  - 5..9 train under identical train_fractal_id2k_harden

Prereg (mean): hard-neg≥0.95 AND K16@T16≥0.75; seed-wise goal ≥8/10 (80%).
  PASS_CANDIDATE_FOR_OPEN → still science_open=false (human only)
  MEASURE_ENVELOPE / STOP_FRAGILE → document; prefer #14 unchanged

Usage::

    python -m reachability_gen.run_stalk_seed_stability
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
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.reach_cue_audit import build_degree_balanced_eval
from reachability_gen.run_fractal_core_gate1 import (
    DEFAULT_BATCH,
    DEFAULT_CLIP,
    DEFAULT_D,
    DEFAULT_MLP,
    DEFAULT_T,
    _mean,
    _n_heads,
    _split_train_val,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS
from reachability_gen.run_stalk_multi_seed_reconfirm import (
    _eval_all_T,
    _load_sealed,
    _std,
    _untrained_control,
)
from reachability_gen.run_stalk_stabilize_multi_seed import (
    LR_MAX,
    LR_MIN,
    train_fractal_id2k_harden,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_SEED_STABILITY"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_seed_stability.json")
DEFAULT_SEALED = Path("artifacts/fractal_core_stalk_gate1_matched_ood.json")
DEFAULT_SEEDS = tuple(range(10))  # 0..9
RECONFIRM_SEEDS = (0, 1, 2, 3, 4)
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16

# Frozen #14 recipe (see docs/CYCLE_STALK_SEED_STABILITY.md)
DEFAULT_EPOCHS = 60
DEFAULT_CLIP_H = DEFAULT_CLIP  # 2.5
PREREG_HARD_NEG = 0.95
PREREG_K16 = 0.75
SEED_PASS_GOAL = 8  # of 10 (= 80%, same rate as 4/5)
SEED_PASS_STRETCH = 10
HN_SELECT_WEIGHT = 0.5
OVERALL_SELECT_WEIGHT = 0.5
Z95 = 1.96  # normal-approx 95% CI

# Existing #14 ckpt paths for reconfirm
def _reconfirm_ckpt(seed: int) -> Path:
    return Path(f"artifacts/fractal_core_stalk_stabilize_seed{seed}_best.pt")


def _new_ckpt(seed: int) -> Path:
    return Path(f"artifacts/fractal_core_stalk_seed_stability_seed{seed}_best.pt")


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


def _ci95(mean: float, std: float, n: int) -> dict[str, float]:
    """Normal-approx 95% CI; cheap; does not gate verdict."""
    if n <= 0 or mean != mean:
        return {"lo": float("nan"), "hi": float("nan"), "halfwidth": float("nan")}
    se = (std / math.sqrt(n)) if n > 0 else float("nan")
    hw = Z95 * se if se == se else float("nan")
    return {"lo": mean - hw, "hi": mean + hw, "halfwidth": hw, "se": se}


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
    return model, blob


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    sealed_path: Path = DEFAULT_SEALED,
    epochs: int = DEFAULT_EPOCHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    grad_clip: float = DEFAULT_CLIP_H,
    include_untrained: bool = True,
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

    sealed = _load_sealed(sealed_path)
    deg_bal = build_degree_balanced_eval(ood_rows, seed=0)
    deg_rows = deg_bal["rows"]

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "base_sha": _git_sha(),
        "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
        "frozen_recipe": "PR#14 CYCLE_STALK_STABILIZE_MULTI_SEED (0.5/0.5 select, 60ep cosine, hard-A, uniform ID)",
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
            "selection": (
                "ID val @ T16 lexicographic (0.5*HN+0.5*overall, overall, HN, -epoch); "
                "matched-OOD never used for ckpt selection. FROZEN from #14."
            ),
            "train_objective": "uniform ID (NO upsample / curriculum — #14 frozen)",
            "hn_select_weight": HN_SELECT_WEIGHT,
            "overall_select_weight": OVERALL_SELECT_WEIGHT,
            "k16_in_select": False,
            "vs_prior": {
                "prefer": "PR#14 MEASURE_STILL corridor unchanged",
                "closed": ["#15 V2 select", "#16 V3 select-aux", "#17 ObjV1 train upsample"],
                "this_cycle": "seed panel 0..9 only; reconfirm 0..4 + train 5..9",
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
            "matched_to": "PR#14 freeze + PR#12 floors",
        },
        "prereg": {
            "hard_neg_mean_ge": PREREG_HARD_NEG,
            "K16_at_T16_mean_ge": PREREG_K16,
            "n_seeds": len(seeds),
            "seed_pass_goal_ge": SEED_PASS_GOAL,
            "seed_pass_stretch": SEED_PASS_STRETCH,
            "seed_pass_rate_goal": SEED_PASS_GOAL / max(len(seeds), 1),
            "selection_rule": (
                "Frozen #14: best ckpt = argmax "
                "(0.5*HN+0.5*overall @ T16 ID-val, overall, HN, -epoch). "
                "NO new select. NO upsample. Floors on matched-OOD after."
            ),
            "ci95": "normal-approx mean ± 1.96*std/sqrt(n); cheap; not a gate",
            "note": (
                "Mean floors match PR #12/#14. science_open never self-stamped; "
                "PASS_CANDIDATE_FOR_OPEN still leaves science_open=false. "
                "Prefer #14 unchanged."
            ),
        },
        "sealed_single_seed_OPEN": sealed,
        "seeds": list(seeds),
        "reconfirm_seeds": list(RECONFIRM_SEEDS),
        "degree_balanced_construction": {
            k: v for k, v in deg_bal.items() if k != "rows"
        },
        "datasets": {"id": str(id_data), "matched_ood": str(ood_data)},
        "policy": (
            "PASS_CANDIDATE_FOR_OPEN → flag human; do NOT set science_open=true. "
            "MEASURE_ENVELOPE → document seed variance under frozen #14; prefer #14 unchanged. "
            "STOP_FRAGILE → append ledger. Sheaf unsupervised ignored. science_open=human only."
        ),
        "include_untrained": include_untrained,
    }

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        is_reconfirm = seed in RECONFIRM_SEEDS
        if is_reconfirm:
            ckpt = _reconfirm_ckpt(seed)
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Reconfirm seed {seed}: missing #14 ckpt {ckpt}"
                )
            print(
                f"[stalk-seed-stab] === seed {seed} RECONFIRM (#14 ckpt) ===",
                file=sys.stderr,
            )
            model, blob = _load_model_from_ckpt(ckpt, max_nodes)
            train_slim = {
                "mode": "reconfirm_pr14_ckpt",
                "best_epoch": blob.get("epoch"),
                "best_sel_hard_neg_T16_val": blob.get("sel_hard_neg_T16_val"),
                "best_sel_overall_T16_val": blob.get("sel_overall_T16_val"),
                "best_sel_joint_T16_val": (
                    None
                    if blob.get("sel_hard_neg_T16_val") is None
                    else 0.5 * float(blob["sel_hard_neg_T16_val"])
                    + 0.5 * float(blob.get("sel_overall_T16_val") or 0.0)
                ),
                "checkpoint_source": str(ckpt),
                "retrained": False,
            }
        else:
            ckpt = _new_ckpt(seed)
            print(
                f"[stalk-seed-stab] === seed {seed} TRAIN (frozen #14 harden) ===",
                file=sys.stderr,
            )
            tr = train_fractal_id2k_harden(
                train,
                val,
                epochs=epochs,
                seed=seed,
                max_nodes=max_nodes,
                ckpt_path=ckpt,
                lr_max=lr_max,
                lr_min=lr_min,
                grad_clip=grad_clip,
            )
            model, blob = _load_model_from_ckpt(ckpt, max_nodes)
            train_slim = {
                k: v
                for k, v in tr.items()
                if k not in ("train_history", "val_by_hop")
            }
            train_slim["train_history_len"] = len(tr.get("train_history") or [])
            train_slim["mode"] = "train_frozen_pr14"
            train_slim["retrained"] = True
            train_slim["checkpoint_source"] = str(ckpt)

        parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)

        print(f"[stalk-seed-stab] seed={seed} OOD eval T∈{T_VALUES}", file=sys.stderr)
        ood_by_T = _eval_all_T(model, ood_rows, max_nodes=max_nodes)
        deg_by_T = _eval_all_T(model, deg_rows, max_nodes=max_nodes)

        if include_untrained:
            print(f"[stalk-seed-stab] seed={seed} untrained control", file=sys.stderr)
            untrained = _untrained_control(
                ood_rows, max_nodes=max_nodes, seed=seed, trained_model=model
            )
        else:
            untrained = {
                "focus_T16": {
                    "overall_acc": float("nan"),
                    "hard_neg_acc": float("nan"),
                    "K16": float("nan"),
                    "prediction_agreement": float("nan"),
                },
                "skipped": True,
            }

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
            f"[stalk-seed-stab] seed={seed} mode={'reconfirm' if is_reconfirm else 'train'} "
            f"T16 overall={overall:.4f} hard_neg={hard:.4f} K16={k16:.4f} "
            f"prereg={seed_prereg['pass']}",
            file=sys.stderr,
        )

        per_seed.append(
            {
                "seed": seed,
                "mode": "reconfirm_pr14" if is_reconfirm else "train_frozen_pr14",
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

    def _agg(getter) -> dict[str, Any]:
        xs = [float(getter(r)) for r in per_seed]
        m = _mean(xs)
        s = _std(xs)
        out = {"mean": m, "std": s, "values": xs, "n": len(xs)}
        out["ci95"] = _ci95(m, s, len(xs))
        return out

    mean_std = {
        "T16_overall": _agg(lambda r: r["T16"]["overall_acc"]),
        "T16_hard_neg": _agg(lambda r: r["T16"]["hard_neg_acc"]),
        "T16_K8": _agg(lambda r: r["T16"]["K8"]),
        "T16_K12": _agg(lambda r: r["T16"]["K12"]),
        "T16_K16": _agg(lambda r: r["T16"]["K16"]),
        "degbal_T16_overall": _agg(lambda r: r["degbal_T16"]["overall_acc"]),
        "degbal_T16_hard_neg": _agg(lambda r: r["degbal_T16"]["hard_neg_acc"]),
        "degbal_T16_K16": _agg(lambda r: r["degbal_T16"]["K16"]),
    }
    if include_untrained:
        mean_std["untrained_T16_overall"] = _agg(
            lambda r: r["untrained_T16"]["overall_acc"]
        )
        mean_std["untrained_T16_hard_neg"] = _agg(
            lambda r: r["untrained_T16"]["hard_neg_acc"]
        )
        mean_std["untrained_vs_trained_agree_T16"] = _agg(
            lambda r: r["untrained_T16"].get("prediction_agreement", float("nan"))
        )

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
    n_seeds = len(per_seed)
    seed_pass_rate = n_pass / n_seeds if n_seeds else 0.0
    hard_mean = mean_std["T16_hard_neg"]["mean"]
    k16_mean = mean_std["T16_K16"]["mean"]
    floors_mean_pass = bool(
        hard_mean == hard_mean
        and hard_mean >= PREREG_HARD_NEG
        and k16_mean == k16_mean
        and k16_mean >= PREREG_K16
        and n_seeds >= 5
    )
    seed_goal_pass = n_pass >= SEED_PASS_GOAL

    if floors_mean_pass and seed_goal_pass:
        verdict = "PASS_CANDIDATE_FOR_OPEN"
        residue = (
            f"Seed-stability PASS candidate under frozen #14: mean T16 hard-neg="
            f"{hard_mean:.4f}≥{PREREG_HARD_NEG}, K16={k16_mean:.4f}≥{PREREG_K16}; "
            f"{n_pass}/{n_seeds} seeds pass (rate={seed_pass_rate:.2f}, goal ≥"
            f"{SEED_PASS_GOAL}/{n_seeds}). FLAG Fractal-1 / human — harness keeps "
            "science_open=false. Prefer #14 recipe unchanged."
        )
        open_status = "PASS_CANDIDATE_FOR_OPEN_SCIENCE_OPEN_FALSE"
    elif n_pass <= 2 or (
        k16_mean == k16_mean and k16_mean < 0.5 and hard_mean < PREREG_HARD_NEG
    ):
        verdict = "STOP_FRAGILE"
        residue = (
            f"Seed-stability STOP_FRAGILE under frozen #14: hard-neg={hard_mean:.4f}, "
            f"K16={k16_mean:.4f}; {n_pass}/{n_seeds} seeds pass (rate={seed_pass_rate:.2f}). "
            "Corridor remains fragile. science_open=false; prefer #14 unchanged "
            "(do not invent new select/upsample)."
        )
        open_status = "MEASURE_STOP_FRAGILE"
    else:
        verdict = "MEASURE_ENVELOPE"
        means_note = (
            "mean floors PASS"
            if floors_mean_pass
            else (
                f"mean floors MISS (HN={hard_mean:.4f} need ≥{PREREG_HARD_NEG}, "
                f"K16={k16_mean:.4f} need ≥{PREREG_K16})"
            )
        )
        residue = (
            f"Seed-stability MEASURE_ENVELOPE under frozen #14: {means_note}; "
            f"{n_pass}/{n_seeds} seeds pass (rate={seed_pass_rate:.2f}, goal ≥"
            f"{SEED_PASS_GOAL}/{n_seeds}). Documents seed variance; does not clear "
            "PASS_CANDIDATE. Prefer #14 corridor unchanged. science_open=false."
        )
        open_status = (
            "MEASURE_ENVELOPE_MEANS_PASS_SEED_GOAL_MISS"
            if floors_mean_pass
            else "MEASURE_ENVELOPE"
        )

    pr14_cmp = {
        "pr14_n5_mean_overall": 0.929,
        "pr14_n5_mean_hard_neg": 0.957,
        "pr14_n5_mean_K16": 0.863,
        "pr14_n5_seed_pass": "2/5",
        "this_n_mean_overall": mean_std["T16_overall"]["mean"],
        "this_n_mean_hard_neg": hard_mean,
        "this_n_mean_K16": k16_mean,
        "this_n_std_overall": mean_std["T16_overall"]["std"],
        "this_n_std_hard_neg": mean_std["T16_hard_neg"]["std"],
        "this_n_std_K16": mean_std["T16_K16"]["std"],
        "this_ci95_hard_neg": mean_std["T16_hard_neg"]["ci95"],
        "this_ci95_K16": mean_std["T16_K16"]["ci95"],
        "this_ci95_overall": mean_std["T16_overall"]["ci95"],
        "this_cycle_prereg_thresholds": {
            "hard_neg_mean": PREREG_HARD_NEG,
            "K16_mean": PREREG_K16,
            "seed_pass_goal": SEED_PASS_GOAL,
            "n_seeds": n_seeds,
        },
        "prefer": "PR#14 recipe unchanged",
    }

    report.update(
        {
            "per_seed": per_seed,
            "mean_std": mean_std,
            "by_T_mean_std": by_T_agg,
            "vs_pr14": pr14_cmp,
            "n_seeds": n_seeds,
            "n_prereg_pass_individual": n_pass,
            "seed_pass_rate": seed_pass_rate,
            "prereg_mean_pass": floors_mean_pass,
            "prereg_seed_goal_pass": seed_goal_pass,
            "verdict": verdict,
            "open_status": open_status,
            "residue": residue,
            "elapsed_sec": time.time() - t0,
            "science_open": False,
            "prefer_corridor": "PR#14 CYCLE_STALK_STABILIZE_MULTI_SEED MEASURE_STILL",
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[stalk-seed-stab] wrote {out_path}", file=sys.stderr)
    print(
        f"[stalk-seed-stab] verdict={verdict} science_open=False "
        f"mean hard_neg={hard_mean:.4f}±{mean_std['T16_hard_neg']['std']:.4f} "
        f"K16={k16_mean:.4f}±{mean_std['T16_K16']['std']:.4f} "
        f"seed_pass={n_pass}/{n_seeds} rate={seed_pass_rate:.2f}",
        file=sys.stderr,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sealed", type=Path, default=DEFAULT_SEALED)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--lr-max", type=float, default=LR_MAX)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CLIP_H)
    p.add_argument(
        "--no-untrained",
        action="store_true",
        help="Skip untrained control (optional; default include)",
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
        sealed_path=args.sealed,
        epochs=args.epochs,
        seeds=tuple(args.seeds),
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        grad_clip=args.grad_clip,
        include_untrained=not args.no_untrained,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "open_status": report["open_status"],
                "science_open": report["science_open"],
                "prereg_mean_pass": report["prereg_mean_pass"],
                "n_prereg_pass_individual": report["n_prereg_pass_individual"],
                "seed_pass_rate": report["seed_pass_rate"],
                "mean_std": {
                    k: {
                        "mean": v["mean"],
                        "std": v["std"],
                        "ci95": v.get("ci95"),
                    }
                    for k, v in report["mean_std"].items()
                    if k.startswith("T16_") or k.startswith("untrained_")
                },
                "prefer_corridor": report["prefer_corridor"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
