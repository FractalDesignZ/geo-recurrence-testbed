"""CYCLE_STALK_HOP_OOD_HN — MEASURE eval-only hop-OOD HN shatter stress.

Deep-cut frozen #14/#18/#22 ens on data/ood_hops.jsonl @ T=16:
  1) Baseline prob_mean — replicate #28 ood_hops shatter numbers
  2) Disagreement/confidence gate — abstain when D high OR conf low
  3) Secondary majority_vote aggregator (cheap codebase knob)

FAIL_OPEN / FAIL_CLOSED + CD lens (same as #29). No train.
science_open=false (not widened; §22 unchanged).

Usage::

    python -m reachability_gen.run_stalk_hop_ood_hn
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
    _competent_dissonance,
    _member_acc_on_mask,
    _pairwise_rate_on_mask,
    _slice_mask,
)
from reachability_gen.run_stalk_epistemic_disagreement import (
    _entropy_nats,
    _pairwise_disagreement_per_example,
    _pairwise_disagreement_rate,
)
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    SECONDARY_AGGS,
    _aggregate_preds,
    _ckpt_for_seed,
    _collect_member_logits,
    _load_model_from_ckpt,
    _metrics_from_preds,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_HOP_OOD_HN"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)
FOCUS_T = 16

# FAIL_OPEN / FAIL_CLOSED thresholds (LOCKED — same as #29)
CONF_THRESH = 0.80
D_CLOSED_MIN = 0.10
EPI_CLOSED_MIN = 0.15

# Gate overlay (LOCKED — abstain when D high OR conf low)
GATE_D_MIN = 0.10
GATE_CONF_MIN = 0.80

# HN verdict floors (LOCKED)
HN_SHATTER_MAX = 0.50
HN_RECOVER_MIN = 0.90
GATE_COVERAGE_MIN = 0.25
VOTE_LIFT_MIN = 0.10

# #28 ood_hops stress cite (replicate check)
CITE_28_HOPS = {
    "ens_hard_neg_acc": 0.06666666666666667,
    "ens_K16": 0.9875,
    "ens_overall_acc": 0.5104166666666666,
    "global_pairwise_disagreement": 0.2860648151901033,
    "D_HN": 0.3583333319466975,
    "D_K16": 0.21111110933125019,
    "D_hard": 0.28472222063897384,
    "CD": 0.553125,
    "mu_acc": 0.553125,
}
CITE_TOL = 0.02


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


def _metrics_ext(preds: Any, labels: Any, hops: list[int]) -> dict[str, float]:
    """Extend seed-ensemble metrics with hop buckets present in ood_hops."""
    base = _metrics_from_preds(preds, labels, hops)
    import torch

    h = torch.tensor(hops, dtype=torch.long)
    for k in (8, 12, 16):
        mask = h == int(k)
        key = f"K{k}"
        if bool(mask.any()):
            base[key] = float((preds[mask] == labels[mask]).float().mean().item())
        else:
            base[key] = float("nan")
    base["n"] = int(preds.numel())
    return base


def _fail_mode_counts(
    *,
    ens_preds: Any,
    labels: Any,
    pair_per: Any,
    ens_max_prob: Any,
    epi_H: Any,
    hops: list[int],
    mask: Optional[Any] = None,
) -> dict[str, Any]:
    """Classify ens-wrong as FAIL_OPEN / FAIL_CLOSED / FAIL_AMBIG (#29 defs)."""
    import torch

    wrong_mask = ens_preds != labels
    if mask is not None:
        wrong_mask = wrong_mask & mask
    n_wrong = int(wrong_mask.sum().item())
    n_total = int(labels.numel() if mask is None else mask.sum().item())
    fail_open = fail_closed = fail_ambig = 0
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
        verdict = "N/A_PERFECT"
    else:
        r_open = fail_open / n_wrong
        r_closed = fail_closed / n_wrong
        if r_closed >= r_open and r_closed >= 0.50:
            verdict = "FAIL_CLOSED_DOMINANT"
        elif r_open > r_closed:
            verdict = "FAIL_OPEN_DOMINANT"
        else:
            verdict = "MIXED"
    return {
        "n_wrong": n_wrong,
        "n_total": n_total,
        "FAIL_OPEN": fail_open,
        "FAIL_CLOSED": fail_closed,
        "FAIL_AMBIG": fail_ambig,
        "rate_FAIL_OPEN": (fail_open / n_wrong) if n_wrong else 0.0,
        "rate_FAIL_CLOSED": (fail_closed / n_wrong) if n_wrong else 0.0,
        "rate_FAIL_AMBIG": (fail_ambig / n_wrong) if n_wrong else 0.0,
        "verdict": verdict,
        "wrong_examples_sample": wrong_details[:40],
        "n_wrong_detailed": len(wrong_details),
    }


def _audit_arm(
    name: str,
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    seeds: list[int],
    agg: str = PRIMARY_AGG,
) -> dict[str, Any]:
    """Full hop-OOD audit for one aggregator: CD + FAIL_OPEN/CLOSED + slices."""
    import torch
    import torch.nn.functional as F

    if not models:
        return {"name": name, "n_members": 0, "skipped": True}

    logits, labels, hops = _collect_member_logits(
        models, rows, T=T, max_nodes=max_nodes
    )
    hard = logits.argmax(dim=-1)  # (M, N)
    ens_preds = _aggregate_preds(logits, method=agg)
    pair_per = _pairwise_disagreement_per_example(hard)

    probs = F.softmax(logits, dim=-1)
    mean_p = probs.mean(dim=0)
    ens_max_prob = mean_p.max(dim=-1).values
    total_H = _entropy_nats(mean_p)
    alea_H = _entropy_nats(probs).mean(dim=0)
    epi_H = total_H - alea_H

    global_disagree = _pairwise_disagreement_rate(hard)
    slice_disagree = {
        "hard_neg": _pairwise_rate_on_mask(hard, _slice_mask(hops, "hard_neg")),
        "K16": _pairwise_rate_on_mask(hard, _slice_mask(hops, "K16")),
        "K12": _pairwise_rate_on_mask(hard, _slice_mask(hops, "K12")),
        "easy_K8": _pairwise_rate_on_mask(hard, _slice_mask(hops, "easy")),
    }
    d_hn = slice_disagree["hard_neg"]
    d_k16 = slice_disagree["K16"]
    d_hard = (
        0.5 * (d_hn + d_k16)
        if d_hn == d_hn and d_k16 == d_k16
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
                "K12": float("nan"),
                "K8": float("nan"),
            }
        idx = mask.nonzero(as_tuple=False).view(-1)
        sub_hops = [hops[int(i)] for i in idx.tolist()]
        return _metrics_ext(ens_preds[idx], labels[idx], sub_hops)

    ens_agree = _ens_on(agree_mask)
    ens_disagree = _ens_on(disagree_mask)
    ens_all = _metrics_ext(ens_preds, labels, hops)

    singles = []
    for mi, seed in enumerate(seeds):
        m = _metrics_ext(hard[mi], labels, hops)
        m["seed"] = seed
        singles.append(m)
    mu_acc = _mean(
        [s["overall_acc"] for s in singles if s["overall_acc"] == s["overall_acc"]]
    )
    cd = _competent_dissonance(mu_acc, d_hard if d_hard == d_hard else 0.0)

    fail_mode = _fail_mode_counts(
        ens_preds=ens_preds,
        labels=labels,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hops=hops,
    )
    fail_mode["ens_overall_acc"] = ens_all["overall_acc"]

    # Gate overlay: abstain when D high OR conf low
    abstain_mask = (pair_per >= GATE_D_MIN) | (ens_max_prob < GATE_CONF_MIN)
    accept_mask = ~abstain_mask
    n_total = int(labels.numel())
    n_accept = int(accept_mask.sum().item())
    n_abstain = int(abstain_mask.sum().item())
    coverage = n_accept / n_total if n_total else 0.0
    ens_accepted = _ens_on(accept_mask)
    fail_mode_accepted = _fail_mode_counts(
        ens_preds=ens_preds,
        labels=labels,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hops=hops,
        mask=accept_mask,
    )
    # HN coverage within HN slice
    import torch as _torch

    hn_mask = _torch.tensor(hops, dtype=_torch.long) == int(HOP_UNREACHABLE)
    hn_accept = accept_mask & hn_mask
    hn_n = int(hn_mask.sum().item())
    hn_accept_n = int(hn_accept.sum().item())
    gate = {
        "GATE_D_MIN": GATE_D_MIN,
        "GATE_CONF_MIN": GATE_CONF_MIN,
        "rule": "abstain if D_ex >= GATE_D_MIN OR ens_max_prob < GATE_CONF_MIN",
        "n_total": n_total,
        "n_accept": n_accept,
        "n_abstain": n_abstain,
        "coverage": coverage,
        "ens_prob_mean_accepted": ens_accepted,
        "fail_mode_accepted": fail_mode_accepted,
        "hn_slice": {
            "n_hn": hn_n,
            "n_hn_accept": hn_accept_n,
            "hn_coverage": (hn_accept_n / hn_n) if hn_n else 0.0,
            "ens_hard_neg_acc_accepted": ens_accepted.get("hard_neg_acc", float("nan")),
        },
        "note": (
            "Abstain is not counted as correct — no invented accuracy OPEN. "
            "Accepted-set metrics are conditional on keep."
        ),
    }

    singles_mean = {
        "overall_acc": mu_acc,
        "hard_neg_acc": _mean(
            [
                s["hard_neg_acc"]
                for s in singles
                if s["hard_neg_acc"] == s["hard_neg_acc"]
            ]
        ),
        "K16": _mean([s["K16"] for s in singles if s.get("K16") == s.get("K16")]),
        "K12": _mean([s["K12"] for s in singles if s.get("K12") == s.get("K12")]),
        "K8": _mean([s["K8"] for s in singles if s.get("K8") == s.get("K8")]),
    }

    return {
        "name": name,
        "aggregator": agg,
        "n_members": len(models),
        "seeds": seeds,
        "T": T,
        "n_examples": n_total,
        "skipped": False,
        "global_pairwise_disagreement": global_disagree,
        "fraction_examples_any_disagreement": float(
            (pair_per > 0).float().mean().item()
        ),
        "slice_pairwise_disagreement": slice_disagree,
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
            "D_hard": "0.5 * (D_HN + D_K16)",
            "D_SAT": D_SAT,
            "CONF_THRESH": CONF_THRESH,
            "D_CLOSED_MIN": D_CLOSED_MIN,
            "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
            "GATE_D_MIN": GATE_D_MIN,
            "GATE_CONF_MIN": GATE_CONF_MIN,
        },
        "ensemble": ens_all,
        "singles_mean": singles_mean,
        "singles": singles,
        "uncertainty": {
            "total_entropy_mean": float(total_H.mean().item()),
            "aleatoric_entropy_mean": float(alea_H.mean().item()),
            "epistemic_entropy_mean": float(epi_H.mean().item()),
            "n": n_total,
        },
        "fail_mode": fail_mode,
        "gate_overlay": gate,
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


def _replicate_ok(audit: dict[str, Any]) -> dict[str, Any]:
    """Compare baseline to #28 ood_hops cite."""
    if audit.get("skipped"):
        return {"ok": False, "note": "skipped"}
    ens = audit["ensemble"]
    sl = audit["slice_pairwise_disagreement"]
    cd = audit["competent_dissonance"]
    checks = {
        "ens_hard_neg_acc": abs(ens["hard_neg_acc"] - CITE_28_HOPS["ens_hard_neg_acc"]),
        "ens_K16": abs(ens["K16"] - CITE_28_HOPS["ens_K16"]),
        "ens_overall_acc": abs(ens["overall_acc"] - CITE_28_HOPS["ens_overall_acc"]),
        "global": abs(
            audit["global_pairwise_disagreement"]
            - CITE_28_HOPS["global_pairwise_disagreement"]
        ),
        "D_HN": abs(sl["hard_neg"] - CITE_28_HOPS["D_HN"]),
        "D_K16": abs(sl["K16"] - CITE_28_HOPS["D_K16"]),
        "D_hard": abs(audit["D_hard"] - CITE_28_HOPS["D_hard"]),
        "CD": abs(cd["CD"] - CITE_28_HOPS["CD"]),
    }
    ok = all(v <= CITE_TOL for v in checks.values())
    return {
        "ok": ok,
        "tolerance": CITE_TOL,
        "abs_deltas_vs_cite28_hops": checks,
        "cite_28_hops": CITE_28_HOPS,
    }


def _hn_verdict(
    baseline: dict[str, Any],
    vote: dict[str, Any],
) -> dict[str, Any]:
    """HN_SHATTER_CONFIRMED | HN_PARTIAL_RECOVER | HN_RECOVERED (MEASURE only)."""
    b_hn = baseline["ensemble"]["hard_neg_acc"]
    v_hn = vote["ensemble"]["hard_neg_acc"]
    g = baseline["gate_overlay"]
    g_hn = g["ens_prob_mean_accepted"].get("hard_neg_acc", float("nan"))
    g_cov = float(g["coverage"])
    g_hn_cov = float(g["hn_slice"]["hn_coverage"])

    b_fail = baseline["fail_mode"]["verdict"]
    g_fail = g["fail_mode_accepted"]["verdict"]
    v_fail = vote["fail_mode"]["verdict"]

    def _closed_ok(v: str) -> bool:
        return v in ("FAIL_CLOSED_DOMINANT", "N/A_PERFECT")

    recovered_paths: list[str] = []
    if b_hn == b_hn and b_hn >= HN_RECOVER_MIN and _closed_ok(b_fail):
        recovered_paths.append("baseline_full")
    if v_hn == v_hn and v_hn >= HN_RECOVER_MIN and _closed_ok(v_fail):
        recovered_paths.append("majority_vote_full")
    if (
        g_hn == g_hn
        and g_hn >= HN_RECOVER_MIN
        and g_cov >= GATE_COVERAGE_MIN
        and _closed_ok(g_fail)
    ):
        recovered_paths.append("gate_accepted")

    if recovered_paths:
        label = "HN_RECOVERED"
    else:
        partial = False
        reasons: list[str] = []
        if g_hn == g_hn and g_hn >= HN_SHATTER_MAX and g_hn_cov > 0:
            partial = True
            reasons.append(
                f"gate_accepted_HN={g_hn:.3f} (hn_cov={g_hn_cov:.3f})"
            )
        if (
            b_hn == b_hn
            and v_hn == v_hn
            and (v_hn - b_hn) >= VOTE_LIFT_MIN
        ):
            partial = True
            reasons.append(f"vote_lift={v_hn - b_hn:+.3f}")
        if partial:
            label = "HN_PARTIAL_RECOVER"
        elif b_hn == b_hn and b_hn < HN_SHATTER_MAX:
            label = "HN_SHATTER_CONFIRMED"
            reasons.append(f"baseline_HN={b_hn:.3f}<{HN_SHATTER_MAX}")
        else:
            label = "HN_SHATTER_CONFIRMED"
            reasons.append("no_overlay_cleared_HN_RECOVER_MIN")

    return {
        "label": label,
        "baseline_HN": b_hn,
        "majority_vote_HN": v_hn,
        "gate_accepted_HN": g_hn,
        "gate_coverage": g_cov,
        "gate_hn_coverage": g_hn_cov,
        "recovered_paths": recovered_paths,
        "note": (
            "Even HN_RECOVERED does not widen §22; hop-OOD remains MEASURE "
            "residue until human seal. science_open=false."
        ),
        "thresholds": {
            "HN_SHATTER_MAX": HN_SHATTER_MAX,
            "HN_RECOVER_MIN": HN_RECOVER_MIN,
            "GATE_COVERAGE_MIN": GATE_COVERAGE_MIN,
            "VOTE_LIFT_MIN": VOTE_LIFT_MIN,
        },
    }


def run_cycle(
    *,
    hops_data: Path = DEFAULT_HOPS,
    out_path: Path = DEFAULT_OUT,
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS,
) -> dict[str, Any]:
    t0 = time.time()
    if not hops_data.exists():
        raise FileNotFoundError(f"ood_hops missing: {hops_data}")
    rows = load_jsonl(hops_data)
    max_nodes = max(
        DEFAULT_MAX_NODE_ID,
        max(int(r["n"]) for r in rows),
    )

    print(
        f"[stalk-hop-ood-hn] loading #14/#18/#22 ens n={len(ensemble_seeds)}",
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

    print(
        f"[stalk-hop-ood-hn] AUDIT baseline {PRIMARY_AGG} ood_hops T{FOCUS_T}",
        file=sys.stderr,
    )
    baseline = _audit_arm(
        "pr14_pr18_ens_pr22_ood_hops_prob_mean",
        ens_models,
        rows,
        T=FOCUS_T,
        max_nodes=max_nodes,
        seeds=list(ensemble_seeds),
        agg=PRIMARY_AGG,
    )
    replicate = _replicate_ok(baseline)

    print(
        "[stalk-hop-ood-hn] AUDIT secondary majority_vote",
        file=sys.stderr,
    )
    vote = _audit_arm(
        "pr14_pr18_ens_pr22_ood_hops_majority_vote",
        ens_models,
        rows,
        T=FOCUS_T,
        max_nodes=max_nodes,
        seeds=list(ensemble_seeds),
        agg="majority_vote",
    )

    hn = _hn_verdict(baseline, vote)
    fail_verdict = baseline["fail_mode"]["verdict"]
    # Prefer MIXED naming to match prereg enum (not FAIL_MIXED)
    if fail_verdict == "FAIL_MIXED":
        fail_verdict = "MIXED"
    cycle_verdict = f"{fail_verdict}+{hn['label']}"
    elapsed = time.time() - t0

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "open_status": "science_open_false_not_widened",
        "section22_unchanged": True,
        "hop_ood_remains_measure_residue": True,
        "base_sha": _git_sha(),
        "architecture": {
            "kind": "eval_only_hop_ood_hn_overlay_stress",
            "train": False,
            "bag_noise_trained": False,
            "pr27_bag_included": False,
            "select_reopened": False,
            "curriculum_reopened": False,
            "d": DEFAULT_D,
            "aggregator_primary": PRIMARY_AGG,
            "aggregator_secondary": "majority_vote",
            "hard_A": True,
            "broadcast_c": False,
            "T_fixed": FOCUS_T,
        },
        "datasets": {
            "ood_hops": str(hops_data),
            "n_ood_hops": len(rows),
            "hop_buckets": {
                "hard_neg": sum(
                    1 for r in rows if int(r.get("hop_distance", 0)) == int(HOP_UNREACHABLE)
                ),
                "K16": sum(1 for r in rows if int(r.get("hop_distance", 0)) == 16),
                "K12": sum(1 for r in rows if int(r.get("hop_distance", 0)) == 12),
                "K8": sum(1 for r in rows if int(r.get("hop_distance", 0)) == 8),
            },
        },
        "members_pr22": ens_meta,
        "prereg": {
            "metrics": [
                "ens_acc_overall_HN_K16_K12_K8",
                "global_D_D_HN_D_K16_D_hard_CD",
                "agree_vs_disagree_ens_acc",
                "FAIL_OPEN_vs_FAIL_CLOSED",
                "gate_overlay_abstain",
                "majority_vote_secondary",
            ],
            "formula": {
                "CD": "mu_acc * min(D_hard, D_SAT) / D_SAT",
                "D_hard": "0.5 * (D_HN + D_K16)",
                "D_SAT": D_SAT,
                "CONF_THRESH": CONF_THRESH,
                "D_CLOSED_MIN": D_CLOSED_MIN,
                "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
                "GATE_D_MIN": GATE_D_MIN,
                "GATE_CONF_MIN": GATE_CONF_MIN,
            },
            "labels_fail": [
                "FAIL_OPEN",
                "FAIL_CLOSED",
                "FAIL_AMBIG",
                "FAIL_CLOSED_DOMINANT",
                "FAIL_OPEN_DOMINANT",
                "MIXED",
                "N/A_PERFECT",
            ],
            "labels_hn": [
                "HN_SHATTER_CONFIRMED",
                "HN_PARTIAL_RECOVER",
                "HN_RECOVERED",
            ],
            "non_goals": [
                "no_train",
                "no_pr27_bag",
                "no_select_curriculum",
                "science_open_not_widened",
                "section22_unchanged",
                "T_fixed_16",
            ],
        },
        "replicate_cite28_hops": replicate,
        "audit": {
            "baseline_prob_mean_T16": baseline,
            "majority_vote_T16": vote,
        },
        "hn_verdict": hn,
        "fail_mode_verdict": fail_verdict,
        "verdict": cycle_verdict,
        "verdict_cd_arm": baseline.get("verdict_cd_arm"),
        "reading": (
            "Hop-OOD HN shatter residue from #28: characterize under FAIL_OPEN/"
            "FAIL_CLOSED + optional gate/vote overlays. Even HN_RECOVERED does "
            "not widen §22; science_open=false."
        ),
        "elapsed_sec": elapsed,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-hop-ood-hn] verdict={cycle_verdict} "
        f"ens_HN={baseline['ensemble'].get('hard_neg_acc')} "
        f"gate_HN={baseline['gate_overlay']['ens_prob_mean_accepted'].get('hard_neg_acc')} "
        f"FAIL_OPEN={baseline['fail_mode'].get('FAIL_OPEN')} "
        f"FAIL_CLOSED={baseline['fail_mode'].get('FAIL_CLOSED')} "
        f"CD={baseline['competent_dissonance'].get('CD')} "
        f"replicate_ok={replicate.get('ok')} "
        f"wrote {out_path} ({elapsed:.1f}s)",
        file=sys.stderr,
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hops-data", type=Path, default=DEFAULT_HOPS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENSEMBLE_SEEDS),
    )
    args = p.parse_args(argv)
    run_cycle(
        hops_data=args.hops_data,
        out_path=args.out,
        ensemble_seeds=tuple(args.seeds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
