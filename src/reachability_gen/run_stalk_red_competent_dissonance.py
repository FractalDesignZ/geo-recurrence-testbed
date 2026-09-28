"""CYCLE_STALK_RED_COMPETENT_DISSONANCE — MEASURE eval-only RED audit.

1) Re-report / refresh #22 boundary disagree (strip easy; cite #28).
2) Eval frozen #14/#18/#22 ens on new RED OOD (p/K outside train priors).
3) Report global D, D_hard, CD, agree vs disagree ens acc.
4) Classify FAIL_OPEN vs FAIL_CLOSED on ens-wrong examples.

No train. No #27 bag. science_open=false (not widened).

Usage::

    python -m reachability_gen.run_stalk_red_competent_dissonance
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
from reachability_gen.models.fractal_core import DISCRETE_T_VALUES
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_fractal_core_gate1 import DEFAULT_D, _mean
from reachability_gen.run_stalk_competent_dissonance import (
    ACC_FLOOR,
    DIS_SET_ACC_MIN,
    D_HARD_MIN,
    D_SAT,
    EASY_HOP,
    _audit_competent,
    _competent_dissonance,
    _member_acc_on_mask,
    _pairwise_rate_on_mask,
    _slice_mask,
)
from reachability_gen.run_stalk_epistemic_disagreement import (
    AUDIT_DELTA_EPS,
    _entropy_nats,
    _pairwise_disagreement_per_example,
    _pairwise_disagreement_rate,
)
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    _aggregate_preds,
    _ckpt_for_seed,
    _collect_member_logits,
    _load_model_from_ckpt,
    _metrics_from_preds,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_RED_COMPETENT_DISSONANCE"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_RED = Path("data/stalk_red_competent_dissonance.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_red_competent_dissonance.json")
DEFAULT_CITED_28 = Path("artifacts/stalk_competent_dissonance.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)
FOCUS_T = 16

# FAIL_OPEN / FAIL_CLOSED thresholds (LOCKED before run)
CONF_THRESH = 0.80
D_CLOSED_MIN = 0.10
EPI_CLOSED_MIN = 0.15

# #28 cite (locked numbers for boundary table)
CITE_28 = {
    "global_pairwise_disagreement": 0.1623611107468605,
    "D_HN": 0.1118518518904845,
    "D_K16": 0.3372222240600321,
    "D_easy_K8": 0.2250000022765663,
    "D_hard": 0.2245370379752583,
    "CD": 0.8111400496856206,
    "mu_acc": 0.903125,
    "verdict": "COMPETENT",
}
CITE_TOL = 0.02  # abs tolerance for refresh vs cite


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


def _slice_mask_hop(hops: list[int], hop: int) -> Any:
    import torch

    h = torch.tensor(hops, dtype=torch.long)
    return h == int(hop)


def _metrics_from_preds_ext(
    preds: Any, labels: Any, hops: list[int]
) -> dict[str, float]:
    """Extend seed-ensemble metrics with K20 when present."""
    base = _metrics_from_preds(preds, labels, hops)
    import torch

    h = torch.tensor(hops, dtype=torch.long)
    mask20 = h == 20
    if bool(mask20.any()):
        base["K20"] = float((preds[mask20] == labels[mask20]).float().mean().item())
    else:
        base["K20"] = float("nan")
    base["n"] = int(preds.numel())
    return base


def _audit_red(
    name: str,
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    seeds: list[int],
) -> dict[str, Any]:
    """RED audit: CD with K16+K20 long hops + FAIL_OPEN/CLOSED on ens-wrong."""
    import torch
    import torch.nn.functional as F

    if not models:
        return {"name": name, "n_members": 0, "skipped": True}

    logits, labels, hops = _collect_member_logits(
        models, rows, T=T, max_nodes=max_nodes
    )
    hard = logits.argmax(dim=-1)  # (M, N)
    ens_preds = _aggregate_preds(logits, method=PRIMARY_AGG)
    pair_per = _pairwise_disagreement_per_example(hard)  # (N,)

    probs = F.softmax(logits, dim=-1)  # (M, N, C)
    mean_p = probs.mean(dim=0)  # (N, C)
    ens_max_prob = mean_p.max(dim=-1).values  # (N,)
    total_H = _entropy_nats(mean_p)
    alea_H = _entropy_nats(probs).mean(dim=0)
    epi_H = total_H - alea_H

    global_disagree = _pairwise_disagreement_rate(hard)

    slice_disagree = {
        "hard_neg": _pairwise_rate_on_mask(hard, _slice_mask(hops, "hard_neg")),
        "K16": _pairwise_rate_on_mask(hard, _slice_mask_hop(hops, 16)),
        "K20": _pairwise_rate_on_mask(hard, _slice_mask_hop(hops, 20)),
        "K12": _pairwise_rate_on_mask(hard, _slice_mask_hop(hops, 12)),
        "easy_K8": _pairwise_rate_on_mask(hard, _slice_mask_hop(hops, 8)),
    }
    d_k16 = slice_disagree["K16"]
    d_k20 = slice_disagree["K20"]
    if d_k16 == d_k16 and d_k20 == d_k20:
        d_long = 0.5 * (d_k16 + d_k20)
    elif d_k20 == d_k20:
        d_long = d_k20
    elif d_k16 == d_k16:
        d_long = d_k16
    else:
        d_long = float("nan")
    d_hn = slice_disagree["hard_neg"]
    d_hard = (
        0.5 * (d_hn + d_long)
        if d_hn == d_hn and d_long == d_long
        else float("nan")
    )

    agree_mask = pair_per == 0
    disagree_mask = pair_per > 0
    member_agree = _member_acc_on_mask(hard, labels, hops, agree_mask)
    member_disagree = _member_acc_on_mask(hard, labels, hops, disagree_mask)

    def _ens_on(mask: Any) -> dict[str, float]:
        if not bool(mask.any()):
            return {
                "n": 0,
                "overall_acc": float("nan"),
                "hard_neg_acc": float("nan"),
                "K16": float("nan"),
                "K20": float("nan"),
            }
        idx = mask.nonzero(as_tuple=False).view(-1)
        sub_hops = [hops[int(i)] for i in idx.tolist()]
        return _metrics_from_preds_ext(ens_preds[idx], labels[idx], sub_hops)

    ens_agree = _ens_on(agree_mask)
    ens_disagree = _ens_on(disagree_mask)
    ens_all = _metrics_from_preds_ext(ens_preds, labels, hops)

    # singles mean overall
    singles = []
    for mi, seed in enumerate(seeds):
        m = _metrics_from_preds_ext(hard[mi], labels, hops)
        m["seed"] = seed
        singles.append(m)
    mu_acc = _mean([s["overall_acc"] for s in singles if s["overall_acc"] == s["overall_acc"]])
    cd = _competent_dissonance(mu_acc, d_hard if d_hard == d_hard else 0.0)

    # FAIL_OPEN / FAIL_CLOSED on ens-wrong
    wrong_mask = ens_preds != labels
    n_wrong = int(wrong_mask.sum().item())
    n_total = int(labels.numel())
    fail_open = 0
    fail_closed = 0
    fail_ambig = 0
    wrong_details: list[dict[str, Any]] = []
    if n_wrong > 0:
        w_idx = wrong_mask.nonzero(as_tuple=False).view(-1)
        for i in w_idx.tolist():
            d_ex = float(pair_per[i].item())
            conf = float(ens_max_prob[i].item())
            epi = float(epi_H[i].item())
            if d_ex == 0.0 and conf >= CONF_THRESH:
                tag = "FAIL_OPEN"
                fail_open += 1
            elif (
                d_ex >= D_CLOSED_MIN
                or epi >= EPI_CLOSED_MIN
                or conf < CONF_THRESH
            ):
                tag = "FAIL_CLOSED"
                fail_closed += 1
            else:
                tag = "FAIL_AMBIG"
                fail_ambig += 1
            wrong_details.append(
                {
                    "i": int(i),
                    "hop": int(hops[i]),
                    "label": int(labels[i].item()),
                    "ens_pred": int(ens_preds[i].item()),
                    "D_ex": d_ex,
                    "ens_max_prob": conf,
                    "epi": epi,
                    "tag": tag,
                }
            )

    if n_wrong == 0:
        red_verdict = "N/A_PERFECT"
    else:
        r_open = fail_open / n_wrong
        r_closed = fail_closed / n_wrong
        if r_closed >= r_open and r_closed >= 0.50:
            red_verdict = "FAIL_CLOSED_DOMINANT"
        elif r_open > r_closed:
            red_verdict = "FAIL_OPEN_DOMINANT"
        else:
            red_verdict = "FAIL_MIXED"

    singles_mean = {
        "overall_acc": mu_acc,
        "hard_neg_acc": _mean(
            [s["hard_neg_acc"] for s in singles if s["hard_neg_acc"] == s["hard_neg_acc"]]
        ),
        "K16": _mean([s["K16"] for s in singles if s.get("K16") == s.get("K16")]),
        "K20": _mean([s["K20"] for s in singles if s.get("K20") == s.get("K20")]),
    }

    return {
        "name": name,
        "n_members": len(models),
        "seeds": seeds,
        "T": T,
        "n_examples": n_total,
        "skipped": False,
        "global_pairwise_disagreement": global_disagree,
        "fraction_examples_any_disagreement": float((pair_per > 0).float().mean().item()),
        "slice_pairwise_disagreement": slice_disagree,
        "D_long": d_long,
        "D_hard": d_hard,
        "agree_set": {
            "n": ens_agree["n"],
            "member_mean": member_agree,
            "ens_prob_mean": ens_agree,
        },
        "disagree_set": {
            "n": ens_disagree["n"],
            "member_mean": member_disagree,
            "ens_prob_mean": ens_disagree,
        },
        "competent_dissonance": cd,
        "formula": {
            "CD": "mu_acc * min(D_hard, D_SAT) / D_SAT",
            "D_hard": "0.5 * (D_HN + D_long)",
            "D_long": "0.5 * (D_K16 + D_K20)",
            "D_SAT": D_SAT,
            "CONF_THRESH": CONF_THRESH,
            "D_CLOSED_MIN": D_CLOSED_MIN,
            "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
        },
        "ensemble_prob_mean": ens_all,
        "singles_mean": singles_mean,
        "singles": singles,
        "uncertainty": {
            "total_entropy_mean": float(total_H.mean().item()),
            "aleatoric_entropy_mean": float(alea_H.mean().item()),
            "epistemic_entropy_mean": float(epi_H.mean().item()),
            "n": n_total,
        },
        "fail_mode": {
            "n_wrong": n_wrong,
            "n_total": n_total,
            "ens_overall_acc": ens_all["overall_acc"],
            "FAIL_OPEN": fail_open,
            "FAIL_CLOSED": fail_closed,
            "FAIL_AMBIG": fail_ambig,
            "rate_FAIL_OPEN": (fail_open / n_wrong) if n_wrong else 0.0,
            "rate_FAIL_CLOSED": (fail_closed / n_wrong) if n_wrong else 0.0,
            "rate_FAIL_AMBIG": (fail_ambig / n_wrong) if n_wrong else 0.0,
            "verdict": red_verdict,
            # cap detail dump
            "wrong_examples_sample": wrong_details[:40],
            "n_wrong_detailed": len(wrong_details),
        },
        "verdict_cd_arm": (
            "COMPETENT"
            if (
                mu_acc >= ACC_FLOOR
                and (d_hard == d_hard and d_hard >= D_HARD_MIN)
                and (
                    member_disagree["overall_acc"] == member_disagree["overall_acc"]
                    and member_disagree["overall_acc"] >= DIS_SET_ACC_MIN
                )
            )
            else (
                "ECHO_RISK"
                if mu_acc >= ACC_FLOOR
                and (d_hard != d_hard or d_hard < D_HARD_MIN)
                else "CHAOS"
            )
        ),
    }


def _boundary_refresh_ok(audit: dict[str, Any]) -> dict[str, Any]:
    """Compare matched-OOD refresh to #28 cite."""
    if audit.get("skipped"):
        return {"ok": False, "note": "skipped"}
    sl = audit["slice_pairwise_disagreement"]
    cd = audit["competent_dissonance"]["CD"]
    checks = {
        "D_HN": abs(sl["hard_neg"] - CITE_28["D_HN"]),
        "D_K16": abs(sl["K16"] - CITE_28["D_K16"]),
        "D_easy_K8": abs(sl["easy_K8"] - CITE_28["D_easy_K8"]),
        "D_hard": abs(audit["D_hard"] - CITE_28["D_hard"]),
        "CD": abs(cd - CITE_28["CD"]),
        "global": abs(
            audit["global_pairwise_disagreement"]
            - CITE_28["global_pairwise_disagreement"]
        ),
    }
    ok = all(v <= CITE_TOL for v in checks.values())
    return {
        "ok": ok,
        "tolerance": CITE_TOL,
        "abs_deltas_vs_cite28": checks,
        "refresh": {
            "global_pairwise_disagreement": audit["global_pairwise_disagreement"],
            "D_HN": sl["hard_neg"],
            "D_K16": sl["K16"],
            "D_easy_K8": sl["easy_K8"],
            "D_hard": audit["D_hard"],
            "CD": cd,
            "mu_acc": audit["competent_dissonance"]["mu_acc"],
            "verdict": audit.get("verdict"),
        },
        "cite_28": CITE_28,
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    red_data: Path = DEFAULT_RED,
    out_path: Path = DEFAULT_OUT,
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS,
    skip_matched_refresh: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    if not red_data.exists():
        raise FileNotFoundError(
            f"RED data missing: {red_data} — run "
            "python -m reachability_gen.gen_stalk_red_competent_dissonance first"
        )
    red_rows = load_jsonl(red_data)
    max_nodes = max(
        DEFAULT_MAX_NODE_ID,
        max(int(r["n"]) for r in red_rows),
    )
    ood_rows: list[dict[str, Any]] = []
    if not skip_matched_refresh:
        if not ood_data.exists():
            raise FileNotFoundError(ood_data)
        ood_rows = load_jsonl(ood_data)
        max_nodes = max(max_nodes, max(int(r["n"]) for r in ood_rows))

    print(
        f"[stalk-red-cd] loading #14/#18/#22 ens n={len(ensemble_seeds)}",
        file=sys.stderr,
    )
    ens_models: list[Any] = []
    ens_meta: list[dict[str, Any]] = []
    for seed in ensemble_seeds:
        ckpt = _ckpt_for_seed(seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing #22 member ckpt: {ckpt}")
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

    boundary: dict[str, Any]
    if ood_rows:
        print("[stalk-red-cd] boundary refresh matched-OOD T16 #22", file=sys.stderr)
        matched = _audit_competent(
            "pr14_pr18_ens_pr22_matched_refresh",
            ens_models,
            ood_rows,
            T=FOCUS_T,
            max_nodes=max_nodes,
            seeds=list(ensemble_seeds),
        )
        boundary = {
            "cite_28": CITE_28,
            "refresh_check": _boundary_refresh_ok(matched),
            "matched_ood_T16": {
                "global_pairwise_disagreement": matched["global_pairwise_disagreement"],
                "slice_pairwise_disagreement": matched["slice_pairwise_disagreement"],
                "D_hard": matched["D_hard"],
                "competent_dissonance": matched["competent_dissonance"],
                "agree_set": matched["agree_set"],
                "disagree_set": matched["disagree_set"],
                "ensemble_prob_mean": matched["ensemble_prob_mean"],
                "verdict": matched["verdict"],
            },
            "strip_easy_note": (
                "D_hard = 0.5*(D_HN+D_K16); easy_K8 reported but excluded from D_hard"
            ),
        }
    else:
        boundary = {
            "cite_28": CITE_28,
            "refresh_check": {"ok": False, "note": "skipped"},
            "strip_easy_note": (
                "D_hard = 0.5*(D_HN+D_K16); easy_K8 reported but excluded from D_hard"
            ),
        }

    print("[stalk-red-cd] AUDIT RED T16 #22 ens", file=sys.stderr)
    red_audit = _audit_red(
        "pr14_pr18_ens_pr22_RED",
        ens_models,
        red_rows,
        T=FOCUS_T,
        max_nodes=max_nodes,
        seeds=list(ensemble_seeds),
    )

    red_verdict = red_audit.get("fail_mode", {}).get("verdict", "FAIL_MIXED")
    elapsed = time.time() - t0

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "open_status": "science_open_false_not_widened",
        "base_sha": _git_sha(),
        "architecture": {
            "kind": "eval_only_red_competent_dissonance",
            "train": False,
            "bag_noise_trained": False,
            "pr27_bag_included": False,
            "d": DEFAULT_D,
            "aggregator_cite": PRIMARY_AGG,
            "hard_A": True,
            "broadcast_c": False,
        },
        "datasets": {
            "matched_ood_cite": str(ood_data) if ood_rows else None,
            "red": str(red_data),
            "n_matched": len(ood_rows) if ood_rows else 0,
            "n_red": len(red_rows),
            "id_cited": str(id_data),
            "cited_28_artifact": str(DEFAULT_CITED_28),
        },
        "members_pr22": ens_meta,
        "prereg": {
            "metrics": [
                "boundary_strip_easy_cite28_plus_refresh",
                "red_global_pairwise_disagreement",
                "red_D_hard_CD",
                "red_agree_vs_disagree_ens_acc",
                "FAIL_OPEN_vs_FAIL_CLOSED",
            ],
            "formula": {
                "CD": "mu_acc * min(D_hard, D_SAT) / D_SAT",
                "D_hard_matched": "0.5 * (D_HN + D_K16)",
                "D_hard_RED": "0.5 * (D_HN + 0.5*(D_K16+D_K20))",
                "D_SAT": D_SAT,
                "CONF_THRESH": CONF_THRESH,
                "D_CLOSED_MIN": D_CLOSED_MIN,
                "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
            },
            "labels_fail": [
                "FAIL_OPEN",
                "FAIL_CLOSED",
                "FAIL_AMBIG",
                "FAIL_CLOSED_DOMINANT",
                "FAIL_OPEN_DOMINANT",
                "FAIL_MIXED",
                "N/A_PERFECT",
            ],
            "non_goals": [
                "no_train",
                "no_pr27_bag",
                "science_open_not_widened",
                "section22_unchanged",
            ],
        },
        "boundary_strip_easy": boundary,
        "audit": {
            "red_T16": red_audit,
        },
        "verdict": red_verdict,
        "verdict_cd_arm_on_red": red_audit.get("verdict_cd_arm"),
        "reading": (
            "RED probes whether #22 COMPETENT map fails open (unified confident "
            "wrong) or fails closed (high D / uncertainty) when p/K leave train "
            "priors. §22 not widened from this residue."
        ),
        "elapsed_sec": elapsed,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-red-cd] verdict={red_verdict} "
        f"ens_acc={red_audit.get('ensemble_prob_mean', {}).get('overall_acc')} "
        f"D_hard={red_audit.get('D_hard')} CD={red_audit.get('competent_dissonance', {}).get('CD')} "
        f"wrote {out_path} ({elapsed:.1f}s)",
        file=sys.stderr,
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--red-data", type=Path, default=DEFAULT_RED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENSEMBLE_SEEDS),
    )
    p.add_argument(
        "--skip-matched-refresh",
        action="store_true",
        help="Skip matched-OOD boundary refresh (cite #28 only)",
    )
    args = p.parse_args(argv)
    run_cycle(
        id_data=args.id_data,
        ood_data=args.ood_data,
        red_data=args.red_data,
        out_path=args.out,
        ensemble_seeds=tuple(args.seeds),
        skip_matched_refresh=args.skip_matched_refresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
