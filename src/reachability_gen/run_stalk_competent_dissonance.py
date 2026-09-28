"""CYCLE_STALK_COMPETENT_DISSONANCE — MEASURE eval-only audit.

Compare frozen #14/#18/#22 ensemble map vs #27 DGE bag on matched-OOD T16:
  (a) global pairwise disagree
  (b) disagree on hard-neg / K16 / easy(K8) slices
  (c) member accuracy on agree-set vs disagree-set
  (d) competent-dissonance score CD = μ_acc * min(D_hard, D_SAT)/D_SAT
  (e) optional ood_hops T16 stress (#22 shatter probe)

No training. science_open=false (not widened).

Verdict per arm: COMPETENT / ECHO_RISK / CHAOS (see docs).

Usage::

    python -m reachability_gen.run_stalk_competent_dissonance
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
from reachability_gen.reach_cue_audit import build_degree_balanced_eval
from reachability_gen.run_fractal_core_gate1 import DEFAULT_D, _mean
from reachability_gen.run_stalk_bag_diversity import _bag_ckpt
from reachability_gen.run_stalk_epistemic_disagreement import (
    AUDIT_DELTA_EPS,
    _audit_bag,
    _pairwise_disagreement_per_example,
    _pairwise_disagreement_rate,
)
from reachability_gen.run_stalk_multi_seed_reconfirm import _std
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    _aggregate_preds,
    _ckpt_for_seed,
    _collect_member_logits,
    _load_model_from_ckpt,
    _metrics_from_preds,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_COMPETENT_DISSONANCE"
DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_STRESS_OOD = Path("data/ood_hops.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_competent_dissonance.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))
DEFAULT_BAG_SEEDS = (0, 1, 2, 3, 4)
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16

# ---- Competent-dissonance formula (LOCKED before run; see cycle doc) ----
D_SAT = 0.25
ACC_FLOOR = 0.85
D_HARD_MIN = 0.05
DIS_SET_ACC_MIN = 0.65
EASY_HOP = 8  # shortest positive K in matched-OOD
SHATTER_DROP = 0.10  # abs drop vs matched-OOD for shatter note


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


def _slice_mask(hops: list[int], kind: str) -> Any:
    import torch

    h = torch.tensor(hops, dtype=torch.long)
    if kind == "hard_neg":
        return h == int(HOP_UNREACHABLE)
    if kind == "K16":
        return h == 16
    if kind == "easy":
        return h == int(EASY_HOP)
    if kind == "K12":
        return h == 12
    if kind == "K8":
        return h == 8
    raise ValueError(kind)


def _pairwise_rate_on_mask(preds: Any, mask: Any) -> float:
    """preds (M, N); mask (N,) bool → pairwise disagree rate on masked examples."""
    import torch

    if not bool(mask.any()):
        return float("nan")
    sub = preds[:, mask]
    return _pairwise_disagreement_rate(sub)


def _member_acc_on_mask(
    hard: Any,
    labels: Any,
    hops: list[int],
    mask: Any,
) -> dict[str, float]:
    """Mean over members of overall/HN/K16 on masked examples."""
    import torch

    if not bool(mask.any()):
        return {
            "n": 0,
            "overall_acc": float("nan"),
            "hard_neg_acc": float("nan"),
            "K16": float("nan"),
        }
    idx = mask.nonzero(as_tuple=False).view(-1)
    sub_hops = [hops[int(i)] for i in idx.tolist()]
    ovs, hns, k16s = [], [], []
    for mi in range(hard.shape[0]):
        m = _metrics_from_preds(hard[mi, idx], labels[idx], sub_hops)
        ovs.append(m["overall_acc"])
        hns.append(m["hard_neg_acc"])
        k16s.append(m["K16"])
    return {
        "n": int(idx.numel()),
        "overall_acc": _mean([x for x in ovs if x == x]),
        "hard_neg_acc": _mean([x for x in hns if x == x]),
        "K16": _mean([x for x in k16s if x == x]),
    }


def _competent_dissonance(mu_acc: float, d_hard: float) -> dict[str, float]:
    """CD = μ_acc * min(D_hard, D_SAT) / D_SAT; also CD_raw = μ_acc * D_hard."""
    if mu_acc != mu_acc or d_hard != d_hard:
        return {
            "CD": float("nan"),
            "CD_raw": float("nan"),
            "D_hard": d_hard,
            "D_clip": float("nan"),
            "mu_acc": mu_acc,
            "D_SAT": D_SAT,
        }
    d_clip = min(d_hard, D_SAT) / D_SAT
    return {
        "CD": float(mu_acc * d_clip),
        "CD_raw": float(mu_acc * d_hard),
        "D_hard": float(d_hard),
        "D_clip": float(d_clip),
        "mu_acc": float(mu_acc),
        "D_SAT": D_SAT,
    }


def _verdict_arm(
    *,
    mu_acc: float,
    d_hard: float,
    disagree_set_member_acc: float,
    rides_on_disagreement: bool,
) -> str:
    """COMPETENT / ECHO_RISK / CHAOS — fail-closed."""
    if mu_acc != mu_acc:
        return "CHAOS"
    competent_gate = (
        mu_acc >= ACC_FLOOR
        and (d_hard >= D_HARD_MIN or rides_on_disagreement)
        and (
            disagree_set_member_acc == disagree_set_member_acc
            and disagree_set_member_acc >= DIS_SET_ACC_MIN
        )
    )
    if competent_gate:
        return "COMPETENT"
    echo_gate = (
        mu_acc >= ACC_FLOOR
        and (d_hard != d_hard or d_hard < D_HARD_MIN)
        and not rides_on_disagreement
    )
    if echo_gate:
        return "ECHO_RISK"
    return "CHAOS"


def _audit_competent(
    name: str,
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Full competent-dissonance audit for one bag @ fixed T."""
    import torch
    import torch.nn.functional as F

    if not models:
        return {"name": name, "n_members": 0, "skipped": True}

    # Reuse #26-style audit for lift×disagreement + epi/alea
    base = _audit_bag(name, models, rows, T=T, max_nodes=max_nodes, seeds=seeds)

    logits, labels, hops = _collect_member_logits(
        models, rows, T=T, max_nodes=max_nodes
    )
    hard = logits.argmax(dim=-1)  # (M, N)
    ens_preds = _aggregate_preds(logits, method=PRIMARY_AGG)
    pair_per = _pairwise_disagreement_per_example(hard)

    # (a) global
    global_disagree = float(base["pairwise_disagreement_rate"])

    # (b) slice disagree
    slice_disagree = {
        "hard_neg": _pairwise_rate_on_mask(hard, _slice_mask(hops, "hard_neg")),
        "K16": _pairwise_rate_on_mask(hard, _slice_mask(hops, "K16")),
        "easy_K8": _pairwise_rate_on_mask(hard, _slice_mask(hops, "easy")),
        "K12": _pairwise_rate_on_mask(hard, _slice_mask(hops, "K12")),
    }
    d_hard = 0.5 * (
        slice_disagree["hard_neg"] + slice_disagree["K16"]
        if slice_disagree["hard_neg"] == slice_disagree["hard_neg"]
        and slice_disagree["K16"] == slice_disagree["K16"]
        else float("nan")
    )

    # (c) agree-set vs disagree-set
    agree_mask = pair_per == 0
    disagree_mask = pair_per > 0
    member_agree = _member_acc_on_mask(hard, labels, hops, agree_mask)
    member_disagree = _member_acc_on_mask(hard, labels, hops, disagree_mask)
    ens_agree = _metrics_from_preds(
        ens_preds[agree_mask],
        labels[agree_mask],
        [hops[i] for i, m in enumerate(agree_mask.tolist()) if m],
    ) if bool(agree_mask.any()) else {
        "n": 0,
        "overall_acc": float("nan"),
        "hard_neg_acc": float("nan"),
        "K16": float("nan"),
    }
    ens_disagree = _metrics_from_preds(
        ens_preds[disagree_mask],
        labels[disagree_mask],
        [hops[i] for i, m in enumerate(disagree_mask.tolist()) if m],
    ) if bool(disagree_mask.any()) else {
        "n": 0,
        "overall_acc": float("nan"),
        "hard_neg_acc": float("nan"),
        "K16": float("nan"),
    }

    mu_acc = float(base["singles_mean"]["overall_acc"])
    cd = _competent_dissonance(mu_acc, d_hard)
    rides = bool(base.get("rides_on_disagreement", False))
    verdict = _verdict_arm(
        mu_acc=mu_acc,
        d_hard=d_hard if d_hard == d_hard else 0.0,
        disagree_set_member_acc=float(member_disagree["overall_acc"]),
        rides_on_disagreement=rides,
    )

    return {
        "name": name,
        "n_members": len(models),
        "seeds": seeds,
        "T": T,
        "n_examples": int(labels.numel()),
        "skipped": False,
        # (a)
        "global_pairwise_disagreement": global_disagree,
        "fraction_examples_any_disagreement": float(
            base["fraction_examples_any_disagreement"]
        ),
        # (b)
        "slice_pairwise_disagreement": slice_disagree,
        "D_hard": d_hard,
        # (c)
        "agree_set": {
            "n": member_agree["n"],
            "member_mean": member_agree,
            "ens_prob_mean": {
                "n": ens_agree["n"],
                "overall_acc": ens_agree["overall_acc"],
                "hard_neg_acc": ens_agree.get("hard_neg_acc", float("nan")),
                "K16": ens_agree.get("K16", float("nan")),
            },
        },
        "disagree_set": {
            "n": member_disagree["n"],
            "member_mean": member_disagree,
            "ens_prob_mean": {
                "n": ens_disagree["n"],
                "overall_acc": ens_disagree["overall_acc"],
                "hard_neg_acc": ens_disagree.get("hard_neg_acc", float("nan")),
                "K16": ens_disagree.get("K16", float("nan")),
            },
        },
        # (d)
        "competent_dissonance": cd,
        "formula": {
            "CD": "mu_acc * min(D_hard, D_SAT) / D_SAT",
            "CD_raw": "mu_acc * D_hard",
            "D_hard": "0.5 * (D_HN + D_K16)",
            "D_SAT": D_SAT,
            "easy_hop": EASY_HOP,
            "ACC_FLOOR": ACC_FLOOR,
            "D_HARD_MIN": D_HARD_MIN,
            "DIS_SET_ACC_MIN": DIS_SET_ACC_MIN,
            "AUDIT_DELTA_EPS": AUDIT_DELTA_EPS,
        },
        # cite ens / singles
        "ensemble_prob_mean": base["ensemble_prob_mean"],
        "singles_mean": base["singles_mean"],
        "singles": base["singles"],
        "deltas_ens_minus_singles_mean": base["deltas_ens_minus_singles_mean"],
        "uncertainty": base["uncertainty"],
        "lift_gap_high_minus_low": base["lift_gap_high_minus_low"],
        "lift_gap_any_minus_none": base["lift_gap_any_minus_none"],
        "rides_split": base["rides_split"],
        "rides_on_disagreement": rides,
        "verdict": verdict,
        "bins_summary": {
            "high_disagreement_n": base["bins"]["high_disagreement"]["n"],
            "low_disagreement_n": base["bins"]["low_disagreement"]["n"],
            "any_disagreement_n": base["bins"]["any_disagreement"]["n"],
            "no_disagreement_n": base["bins"]["no_disagreement"]["n"],
        },
    }


def _pair_disagree_by_T(
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    max_nodes: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for T in T_VALUES:
        logits, _labels, _hops = _collect_member_logits(
            models, rows, T=T, max_nodes=max_nodes
        )
        hard = logits.argmax(dim=-1)
        out[str(T)] = _pairwise_disagreement_rate(hard)
    return out


def _shatter_note(
    matched: dict[str, Any],
    stress: dict[str, Any],
) -> dict[str, Any]:
    """#22 shatter probe: large abs drop on ens HN/K16 or CD collapse."""
    if matched.get("skipped") or stress.get("skipped"):
        return {"shattered": False, "note": "skipped"}
    m_ens = matched["ensemble_prob_mean"]
    s_ens = stress["ensemble_prob_mean"]
    drops = {
        "overall": m_ens["overall_acc"] - s_ens["overall_acc"],
        "hard_neg": m_ens["hard_neg_acc"] - s_ens["hard_neg_acc"],
        "K16": m_ens["K16"] - s_ens["K16"],
    }
    cd_m = matched["competent_dissonance"]["CD"]
    cd_s = stress["competent_dissonance"]["CD"]
    cd_drop = (
        cd_m - cd_s if cd_m == cd_m and cd_s == cd_s else float("nan")
    )
    shattered = any(
        drops[k] == drops[k] and drops[k] >= SHATTER_DROP
        for k in ("hard_neg", "K16")
    )
    return {
        "shattered": shattered,
        "drops_matched_minus_stress": drops,
        "CD_matched": cd_m,
        "CD_stress": cd_s,
        "CD_drop": cd_drop,
        "threshold": SHATTER_DROP,
        "note": (
            "ens HN or K16 dropped ≥ threshold on ood_hops vs matched-OOD"
            if shattered
            else "no HN/K16 shatter drop ≥ threshold"
        ),
    }


def run_cycle(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    stress_ood: Path = DEFAULT_STRESS_OOD,
    out_path: Path = DEFAULT_OUT,
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS,
    bag_seeds: tuple[int, ...] = DEFAULT_BAG_SEEDS,
    skip_stress: bool = False,
    skip_by_T: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    if not ood_data.exists():
        raise FileNotFoundError(ood_data)

    ood_rows = load_jsonl(ood_data)
    max_nodes = max(
        DEFAULT_MAX_NODE_ID,
        max(int(r["n"]) for r in ood_rows),
    )
    stress_rows: list[dict[str, Any]] = []
    if not skip_stress and stress_ood.exists():
        stress_rows = load_jsonl(stress_ood)
        max_nodes = max(max_nodes, max(int(r["n"]) for r in stress_rows))
    elif not skip_stress:
        print(f"[stalk-cd] stress OOD missing: {stress_ood}", file=sys.stderr)

    # Load #22 map
    print(
        f"[stalk-cd] loading #14/#18/#22 ens n={len(ensemble_seeds)}",
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

    # Load #27 bag
    print(
        f"[stalk-cd] loading #27 bag n={len(bag_seeds)}",
        file=sys.stderr,
    )
    bag_models: list[Any] = []
    bag_meta: list[dict[str, Any]] = []
    for seed in bag_seeds:
        ckpt = _bag_ckpt(seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing #27 bag ckpt: {ckpt}")
        model, blob = _load_model_from_ckpt(ckpt, max_nodes)
        bag_models.append(model)
        bag_meta.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "source": "pr27_bag",
                "best_epoch": blob.get("epoch"),
            }
        )

    print("[stalk-cd] AUDIT matched-OOD T16 #22 ens", file=sys.stderr)
    audit_22 = _audit_competent(
        "pr14_pr18_ens_pr22",
        ens_models,
        ood_rows,
        T=FOCUS_T,
        max_nodes=max_nodes,
        seeds=list(ensemble_seeds),
    )
    print("[stalk-cd] AUDIT matched-OOD T16 #27 bag", file=sys.stderr)
    audit_27 = _audit_competent(
        "pr27_bag",
        bag_models,
        ood_rows,
        T=FOCUS_T,
        max_nodes=max_nodes,
        seeds=list(bag_seeds),
    )

    by_T_22: dict[str, float] = {}
    by_T_27: dict[str, float] = {}
    if not skip_by_T:
        print("[stalk-cd] pair-disagree by T (matched-OOD)", file=sys.stderr)
        by_T_22 = _pair_disagree_by_T(ens_models, ood_rows, max_nodes=max_nodes)
        by_T_27 = _pair_disagree_by_T(bag_models, ood_rows, max_nodes=max_nodes)

    stress_22: dict[str, Any] = {"skipped": True}
    stress_27: dict[str, Any] = {"skipped": True}
    shatter: dict[str, Any] = {"shattered": False, "note": "stress skipped"}
    if stress_rows:
        print("[stalk-cd] AUDIT ood_hops T16 #22 ens (shatter probe)", file=sys.stderr)
        stress_22 = _audit_competent(
            "pr14_pr18_ens_pr22_ood_hops",
            ens_models,
            stress_rows,
            T=FOCUS_T,
            max_nodes=max_nodes,
            seeds=list(ensemble_seeds),
        )
        print("[stalk-cd] AUDIT ood_hops T16 #27 bag", file=sys.stderr)
        stress_27 = _audit_competent(
            "pr27_bag_ood_hops",
            bag_models,
            stress_rows,
            T=FOCUS_T,
            max_nodes=max_nodes,
            seeds=list(bag_seeds),
        )
        shatter = _shatter_note(audit_22, stress_22)

    deg_bal = build_degree_balanced_eval(ood_rows, seed=0)

    v22 = audit_22.get("verdict", "CHAOS")
    v27 = audit_27.get("verdict", "CHAOS")
    cd22 = audit_22.get("competent_dissonance", {}).get("CD", float("nan"))
    cd27 = audit_27.get("competent_dissonance", {}).get("CD", float("nan"))
    prefer = (
        "pr22_ens"
        if (cd22 == cd22 and (cd27 != cd27 or cd22 >= cd27) and v22 == "COMPETENT")
        or (v22 == "COMPETENT" and v27 != "COMPETENT")
        else (
            "pr27_bag"
            if v27 == "COMPETENT" and v22 != "COMPETENT"
            else (
                "pr22_ens"
                if cd22 == cd22 and (cd27 != cd27 or cd22 >= cd27)
                else "pr27_bag"
            )
        )
    )
    cycle_verdict = f"{v22}_vs_{v27}"

    narrative = (
        "#22 low global disagree ≠ echo chamber when rides_on_disagreement "
        "(§24 / #26); COMPETENT requires competence + hard-slice/localized "
        "disagree geometry. #27 high disagree with weak members → CHAOS risk."
    )

    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "open_status": "science_open_false_not_widened",
        "base_sha": _git_sha(),
        "architecture": {
            "kind": "eval_only_competent_dissonance_audit",
            "train": False,
            "bag_noise_trained": False,
            "d": DEFAULT_D,
            "aggregator_cite": PRIMARY_AGG,
            "hard_A": True,
            "broadcast_c": False,
        },
        "datasets": {
            "ood_primary": str(ood_data),
            "ood_stress": str(stress_ood) if stress_rows else None,
            "n_ood": len(ood_rows),
            "n_stress": len(stress_rows) if stress_rows else 0,
            "id_cited": str(id_data),
        },
        "degree_balanced_construction": {
            "n": deg_bal.get("n"),
            "note": "secondary cite only; primary=full matched-OOD",
        },
        "members_pr22": ens_meta,
        "members_pr27": bag_meta,
        "prereg": {
            "metrics": [
                "global_pairwise_disagreement",
                "slice_disagree_hard_neg_K16_easyK8",
                "member_acc_agree_vs_disagree_set",
                "competent_dissonance_CD",
                "optional_ood_hops_shatter",
            ],
            "formula": {
                "CD": "mu_acc * min(D_hard, D_SAT) / D_SAT",
                "CD_raw": "mu_acc * D_hard",
                "D_hard": "0.5 * (D_HN + D_K16)",
                "D_SAT": D_SAT,
                "easy_hop": EASY_HOP,
            },
            "verdict_thresholds": {
                "ACC_FLOOR": ACC_FLOOR,
                "D_HARD_MIN": D_HARD_MIN,
                "DIS_SET_ACC_MIN": DIS_SET_ACC_MIN,
                "AUDIT_DELTA_EPS": AUDIT_DELTA_EPS,
                "SHATTER_DROP": SHATTER_DROP,
            },
            "labels": ["COMPETENT", "ECHO_RISK", "CHAOS"],
            "train": False,
            "science_open_widened": False,
        },
        "policy": {
            "no_train": True,
            "no_bag_noise": True,
            "science_open_not_widened": True,
            "select_curriculum_closed": True,
            "narrative_lock": narrative,
        },
        "audit": {
            "matched_ood_T16": {
                "pr22_ens": audit_22,
                "pr27_bag": audit_27,
            },
            "pairwise_disagreement_by_T": {
                "pr22_ens": by_T_22,
                "pr27_bag": by_T_27,
            },
            "ood_hops_T16_stress": {
                "pr22_ens": stress_22,
                "pr27_bag": stress_27,
                "shatter_pr22": shatter,
            },
        },
        "verdict_pr22": v22,
        "verdict_pr27": v27,
        "cycle_verdict": cycle_verdict,
        "prefer_arm": prefer,
        "CD_pr22": cd22,
        "CD_pr27": cd27,
        "elapsed_sec": round(time.time() - t0, 3),
        "mandelbrot_analogy": (
            "competent dissonance = accurate members that usefully disagree "
            "on hard slices; chaos = over-dispersed weak bag; echo ≠ low "
            "global rate when disagreement-localized lift holds"
        ),
        "prefer_corridor": (
            "Prefer #14 MEASURE_STILL + #22 ens overlay when COMPETENT; "
            "do not chase #27 bag noise. science_open=false; §22 unchanged."
        ),
        "residue": {
            "select_curriculum": "CLOSED",
            "distill": "STOP",
            "swa": "MEASURE",
            "multi_hyp": "STOP",
            "bag_diversity": "MEASURE_LIFT",
            "section_22_widened": False,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-cd] wrote {out_path} cycle_verdict={cycle_verdict} "
        f"pr22={v22} pr27={v27} CD22={cd22} CD27={cd27} "
        f"science_open=false elapsed={report['elapsed_sec']}s",
        file=sys.stderr,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CYCLE_STALK_COMPETENT_DISSONANCE — eval-only audit"
    )
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--stress-ood", type=Path, default=DEFAULT_STRESS_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--ensemble-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_ENSEMBLE_SEEDS),
    )
    p.add_argument(
        "--bag-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_BAG_SEEDS),
    )
    p.add_argument("--skip-stress", action="store_true")
    p.add_argument("--skip-by-T", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ens = tuple(int(x) for x in args.ensemble_seeds.split(",") if x.strip() != "")
    bag = tuple(int(x) for x in args.bag_seeds.split(",") if x.strip() != "")
    report = run_cycle(
        id_data=args.id_data,
        ood_data=args.ood_data,
        stress_ood=args.stress_ood,
        out_path=args.out,
        ensemble_seeds=ens,
        bag_seeds=bag,
        skip_stress=args.skip_stress,
        skip_by_T=args.skip_by_T,
    )
    print(
        json.dumps(
            {
                "cycle": report["cycle"],
                "cycle_verdict": report["cycle_verdict"],
                "verdict_pr22": report["verdict_pr22"],
                "verdict_pr27": report["verdict_pr27"],
                "CD_pr22": report["CD_pr22"],
                "CD_pr27": report["CD_pr27"],
                "prefer_arm": report["prefer_arm"],
                "science_open": report["science_open"],
                "elapsed_sec": report["elapsed_sec"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
