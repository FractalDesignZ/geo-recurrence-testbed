"""CYCLE_STALK_MULTI_SEED_RECONFIRM — MEASURE (science_open=false always).

Train/eval stalk-local FractalCore (≥3 seeds) matching sealed OPEN architecture:
hard A_ij mask, stalk@s / probe@t local, no c broadcast, no soft ACT.
Matched-OOD T∈{6,8,12,16}; report mean±std overall / hard-neg / K16.
Untrained control per seed; degree-balanced secondary if available.
Compare to sealed single-seed OPEN Gate1.

Prereg (this cycle, mean across seeds): hard-neg≥0.95 AND K16@T16≥0.75.
  PASS → MEASURE reconfirm note (do NOT silently widen science_open)
  FAIL → contingent OPEN at risk; append note

Usage::

    python -m reachability_gen.run_stalk_multi_seed_reconfirm
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
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MLP,
    DEFAULT_T,
    _eval_split,
    _n_heads,
    _split_train_val,
    train_fractal_id2k,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS
from reachability_gen.run_stalk_untrained_control import (
    document_stalk_init,
    _eval_rows,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_MULTI_SEED_RECONFIRM"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_multi_seed_reconfirm.json")
DEFAULT_SEALED = Path("artifacts/fractal_core_stalk_gate1_matched_ood.json")
DEFAULT_SEEDS = (0, 1, 2)
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16
# This-cycle prereg (mean across seeds) — relaxed vs sealed single-seed 0.98/0.80
PREREG_HARD_NEG = 0.95
PREREG_K16 = 0.75
SEALED_OPEN_HARD_NEG = 0.98
SEALED_OPEN_K16 = 0.80


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


def _hop_acc(stats: dict[str, Any], hop: int) -> float:
    return float(stats.get("by_hop", {}).get(str(hop), {}).get("acc_mean", float("nan")))


def _load_sealed(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "path": str(path)}
    art = json.loads(path.read_text(encoding="utf-8"))
    dyn = art.get("ood_eval", {}).get("dynamic", {})
    t16 = dyn.get("dynamic_T16_unroll", {})
    by_t: dict[str, Any] = {}
    fixed = art.get("ood_eval", {}).get("fixed", {})
    if "fixed_T6_unroll" in fixed:
        by_t["6"] = {
            "overall_acc": fixed["fixed_T6_unroll"].get("overall_acc"),
            "hard_neg_acc": fixed["fixed_T6_unroll"].get("hard_neg_acc"),
            "K8": _hop_acc(fixed["fixed_T6_unroll"], 8),
            "K12": _hop_acc(fixed["fixed_T6_unroll"], 12),
            "K16": _hop_acc(fixed["fixed_T6_unroll"], 16),
        }
    for T in (8, 12, 16):
        key = f"dynamic_T{T}_unroll"
        if key in dyn:
            s = dyn[key]
            by_t[str(T)] = {
                "overall_acc": s.get("overall_acc"),
                "hard_neg_acc": s.get("hard_neg_acc"),
                "K8": _hop_acc(s, 8),
                "K12": _hop_acc(s, 12),
                "K16": _hop_acc(s, 16),
            }
    return {
        "present": True,
        "path": str(path),
        "cycle": art.get("cycle"),
        "science_open_harness": art.get("science_open", False),
        "prereg_sealed": art.get("prereg"),
        "train_best_val_acc": (art.get("train") or {}).get("best_val_acc"),
        "train_best_epoch": (art.get("train") or {}).get("best_epoch"),
        "by_T": by_t,
        "T16_overall": t16.get("overall_acc"),
        "T16_hard_neg": t16.get("hard_neg_acc"),
        "T16_K16": _hop_acc(t16, 16),
        "note": (
            "Single-seed sealed OPEN Gate1 (PR #2). Contigent until multi-seed "
            "reconfirm; science_open human-only."
        ),
    }


def _untrained_control(
    ood_rows: list[dict[str, Any]],
    *,
    max_nodes: int,
    seed: int,
    trained_model: Optional[FractalCore] = None,
) -> dict[str, Any]:
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
    init_doc = document_stalk_init(model)
    by_T: dict[str, Any] = {}
    for T in T_VALUES:
        u = _eval_rows(model, ood_rows, T=int(T), max_nodes=max_nodes)
        cell: dict[str, Any] = {
            "overall_acc": u["overall_acc"],
            "hard_neg_acc": u["hard_neg_acc"],
            "K8": _hop_acc(u, 8),
            "K12": _hop_acc(u, 12),
            "K16": _hop_acc(u, 16),
            "fpr": u["fpr"],
            "fnr": u["fnr"],
        }
        if trained_model is not None:
            t = _eval_rows(trained_model, ood_rows, T=int(T), max_nodes=max_nodes)
            agree = float((u["preds"] == t["preds"]).float().mean())
            cell["trained_overall_acc"] = t["overall_acc"]
            cell["trained_hard_neg_acc"] = t["hard_neg_acc"]
            cell["prediction_agreement"] = agree
            cell["delta_overall"] = t["overall_acc"] - u["overall_acc"]
            cell["delta_hard_neg"] = t["hard_neg_acc"] - u["hard_neg_acc"]
        by_T[str(T)] = cell
    return {
        "seed": seed,
        "init_doc": init_doc,
        "by_T": by_T,
        "focus_T16": by_T[str(FOCUS_T)],
        "science_open": False,
    }


def _eval_all_T(
    model: FractalCore,
    rows: list[dict[str, Any]],
    *,
    max_nodes: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for T in T_VALUES:
        s = _eval_split(model, rows, max_nodes=max_nodes, T=int(T))
        out[str(T)] = {
            "overall_acc": float(s["overall_acc"]),
            "hard_neg_acc": float(s["hard_neg_acc"]),
            "hard_neg_n": int(s["hard_neg_n"]),
            "K8": _hop_acc(s, 8),
            "K12": _hop_acc(s, 12),
            "K16": _hop_acc(s, 16),
            "by_hop": s["by_hop"],
            "n": s["n"],
            "T": T,
        }
    return out


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    sealed_path: Path = DEFAULT_SEALED,
    epochs: int = DEFAULT_EPOCHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
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
        "hparams": {
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "lr": DEFAULT_LR,
            "epochs": epochs,
            "grad_clip": DEFAULT_CLIP,
            "batch_size": DEFAULT_BATCH,
            "adaptive_halt": False,
            "ff_baseline": FF_BASELINE_PARAMS,
            "matched_to": "sealed CYCLE_STALK_LOCALIZATION Gate1",
        },
        "prereg": {
            "hard_neg_mean_ge": PREREG_HARD_NEG,
            "K16_at_T16_mean_ge": PREREG_K16,
            "n_seeds_min": 3,
            "note": (
                "Mean across seeds on matched-OOD T16. "
                "science_open never self-stamped; human seal only."
            ),
        },
        "sealed_single_seed_OPEN": sealed,
        "seeds": list(seeds),
        "degree_balanced_construction": {
            k: v for k, v in deg_bal.items() if k != "rows"
        },
        "datasets": {"id": str(id_data), "matched_ood": str(ood_data)},
        "policy": (
            "PASS → MEASURE reconfirm note only; do NOT silently widen "
            "science_open / §6 claim. FAIL → contingent OPEN at risk; append note. "
            "science_open=human only."
        ),
    }

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[stalk-reconfirm] === seed {seed} train ===", file=sys.stderr)
        ckpt = Path(f"artifacts/fractal_core_stalk_reconfirm_seed{seed}_best.pt")
        tr = train_fractal_id2k(
            train,
            val,
            epochs=epochs,
            seed=seed,
            max_nodes=max_nodes,
            ckpt_path=ckpt,
        )
        # Reload best
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

        print(f"[stalk-reconfirm] seed={seed} OOD eval T∈{T_VALUES}", file=sys.stderr)
        ood_by_T = _eval_all_T(model, ood_rows, max_nodes=max_nodes)
        deg_by_T = _eval_all_T(model, deg_rows, max_nodes=max_nodes)

        print(f"[stalk-reconfirm] seed={seed} untrained control", file=sys.stderr)
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
            f"[stalk-reconfirm] seed={seed} T16 overall={overall:.4f} "
            f"hard_neg={hard:.4f} K16={k16:.4f} prereg={seed_prereg['pass']}",
            file=sys.stderr,
        )

        # Drop heavy train history from per-seed blob (keep summary)
        train_slim = {
            k: v
            for k, v in tr.items()
            if k not in ("train_history", "val_by_hop")
        }
        train_slim["train_history_len"] = len(tr.get("train_history") or [])
        train_slim["final_epoch_val_acc"] = (
            (tr.get("train_history") or [{}])[-1].get("val_acc")
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
        "best_val_acc": _agg(lambda r: r["train"]["best_val_acc"]),
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

    # Causal horizon mean±std per T
    by_T_agg: dict[str, Any] = {}
    for T in T_VALUES:
        by_T_agg[str(T)] = {
            "overall": _agg(lambda r, tt=T: r["matched_ood_by_T"][str(tt)]["overall_acc"]),
            "hard_neg": _agg(lambda r, tt=T: r["matched_ood_by_T"][str(tt)]["hard_neg_acc"]),
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
        and len(per_seed) >= 3
    )
    # Prefer all-seeds pass; mean floors is the stated prereg
    prereg_pass = floors_mean_pass

    sealed_cmp = {
        "sealed_T16_overall": sealed.get("T16_overall"),
        "sealed_T16_hard_neg": sealed.get("T16_hard_neg"),
        "sealed_T16_K16": sealed.get("T16_K16"),
        "reconfirm_mean_T16_overall": mean_std["T16_overall"]["mean"],
        "reconfirm_mean_T16_hard_neg": hard_mean,
        "reconfirm_mean_T16_K16": k16_mean,
        "reconfirm_std_T16_overall": mean_std["T16_overall"]["std"],
        "reconfirm_std_T16_hard_neg": mean_std["T16_hard_neg"]["std"],
        "reconfirm_std_T16_K16": mean_std["T16_K16"]["std"],
        "delta_overall_vs_sealed": (
            mean_std["T16_overall"]["mean"] - float(sealed["T16_overall"])
            if sealed.get("present") and sealed.get("T16_overall") is not None
            else float("nan")
        ),
        "delta_hard_neg_vs_sealed": (
            hard_mean - float(sealed["T16_hard_neg"])
            if sealed.get("present") and sealed.get("T16_hard_neg") is not None
            else float("nan")
        ),
        "delta_K16_vs_sealed": (
            k16_mean - float(sealed["T16_K16"])
            if sealed.get("present") and sealed.get("T16_K16") is not None
            else float("nan")
        ),
        "sealed_prereg_thresholds": {
            "hard_neg": SEALED_OPEN_HARD_NEG,
            "K16": SEALED_OPEN_K16,
        },
        "this_cycle_prereg_thresholds": {
            "hard_neg_mean": PREREG_HARD_NEG,
            "K16_mean": PREREG_K16,
        },
    }

    if prereg_pass:
        verdict = "MEASURE_RECONFIRM_PASS"
        residue = (
            f"Multi-seed stalk reconfirm PASS: mean T16 hard-neg="
            f"{hard_mean:.4f}≥{PREREG_HARD_NEG}, K16={k16_mean:.4f}≥{PREREG_K16} "
            f"({n_pass}/{len(per_seed)} seeds individual pass). "
            "Append MEASURE/reconfirm note only — do NOT silently widen "
            "science_open / §6 claim. Human seal required for any claim update."
        )
        open_status = "OPEN_STANDING_RECONFIRMED_CONTINGENT_NOTE"
    else:
        verdict = "OPEN_CONTINGENT_AT_RISK"
        residue = (
            f"Multi-seed stalk reconfirm FAIL prereg means: hard-neg="
            f"{hard_mean:.4f} (need ≥{PREREG_HARD_NEG}), K16={k16_mean:.4f} "
            f"(need ≥{PREREG_K16}); {n_pass}/{len(per_seed)} seeds individual pass. "
            "Stalk §6 OPEN remains contingent AT RISK. Append note; "
            "do not widen science_open; prefer truth over prior OPEN."
        )
        open_status = "OPEN_CONTINGENT_AT_RISK"

    report.update(
        {
            "per_seed": per_seed,
            "mean_std": mean_std,
            "by_T_mean_std": by_T_agg,
            "vs_sealed_OPEN": sealed_cmp,
            "n_seeds": len(per_seed),
            "n_prereg_pass_individual": n_pass,
            "prereg_mean_pass": prereg_pass,
            "verdict": verdict,
            "open_status": open_status,
            "residue": residue,
            "elapsed_sec": time.time() - t0,
            "science_open": False,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[stalk-reconfirm] wrote {out_path}", file=sys.stderr)
    print(
        f"[stalk-reconfirm] verdict={verdict} science_open=False "
        f"mean hard_neg={hard_mean:.4f}±{mean_std['T16_hard_neg']['std']:.4f} "
        f"K16={k16_mean:.4f}±{mean_std['T16_K16']['std']:.4f}",
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
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "open_status": report["open_status"],
                "science_open": report["science_open"],
                "prereg_mean_pass": report["prereg_mean_pass"],
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
