"""CYCLE_STALK_TROPICAL_ATTENTION_PROBE — MEASURE eval-only tropical ens probe.

Eval-only @ T=16 on data/ood_hops.jsonl (+ matched-OOD collateral) with
frozen #14/#18/#22 ens. Tropical = max-plus / β→∞ ens aggregation over
member logits (NOT in-attention rewrite; NOT train; NOT β-anneal; NOT DEAR).
science_open=false (not widened; §22 unchanged).
Cert #35 reported as reference only — do not claim tropical ≡ certificate.

Usage::

    python -m reachability_gen.run_stalk_tropical_attention_probe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_fractal_core_gate1 import DEFAULT_D, _mean
from reachability_gen.run_stalk_competent_dissonance import (
    _competent_dissonance,
    _pairwise_rate_on_mask,
    _slice_mask,
)
from reachability_gen.run_stalk_epistemic_disagreement import (
    _entropy_nats,
    _pairwise_disagreement_per_example,
    _pairwise_disagreement_rate,
)
from reachability_gen.run_stalk_hop_ood_hn import (
    CONF_THRESH,
    D_CLOSED_MIN,
    EPI_CLOSED_MIN,
    FOCUS_T,
    _fail_mode_counts,
    _metrics_ext,
)
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    _aggregate_preds,
    _ckpt_for_seed,
    _collect_member_logits,
    _load_model_from_ckpt,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID
from reachability_gen.tropical import (
    beta_inf_member_preds,
    max_plus_aggregate,
    tropical_definition_doc,
)

CYCLE = "CYCLE_STALK_TROPICAL_ATTENTION_PROBE"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_MATCHED = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_tropical_attention_probe.json")
DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE32 = Path("artifacts/stalk_sound_outdeg_gate.json")
DEFAULT_CITE35 = Path("artifacts/stalk_reach_certificates.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))

VERDICTS = (
    "TROPICAL_FO_LIFT",
    "TROPICAL_PARTIAL",
    "TROPICAL_NULL",
    "COLLATERAL_HARM",
)

# Thresholds (LOCKED in prereg)
FO_LIFT_MIN = 16  # of 45
REM22_LIFT_MIN = 8  # of 22
FO_NULL_MAX = 2  # FO killed ≤ this for NULL
HN_NULL_DELTA = 0.05
COLLATERAL_DROP = 0.05
CITE_30_FO = 45
CITE_30_HN = 0.06666666666666667
CITE_30_K16 = 0.9875
CITE_30_OV = 0.5104166666666666
CITE_TOL = 0.02
CITE_32_REM = 22


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


def decide_verdict(
    *,
    fo_killed: int,
    fo_total: int,
    rem22_killed: int,
    rem22_total: int,
    delta_hn_abs: float,
    collateral_harm: bool,
    collateral_reasons: list[str],
) -> dict[str, Any]:
    """LOCKED verdict enum with collateral priority."""
    if collateral_harm:
        return {
            "verdict": "COLLATERAL_HARM",
            "reasons": list(collateral_reasons),
            "fo_killed": fo_killed,
            "fo_total": fo_total,
            "rem22_killed": rem22_killed,
            "rem22_total": rem22_total,
            "delta_hn_abs": delta_hn_abs,
        }
    if (
        fo_killed >= FO_LIFT_MIN
        and rem22_killed >= REM22_LIFT_MIN
    ):
        return {
            "verdict": "TROPICAL_FO_LIFT",
            "reasons": [
                f"fo_killed={fo_killed}>={FO_LIFT_MIN}",
                f"rem22_killed={rem22_killed}>={REM22_LIFT_MIN}",
            ],
            "fo_killed": fo_killed,
            "fo_total": fo_total,
            "rem22_killed": rem22_killed,
            "rem22_total": rem22_total,
            "delta_hn_abs": delta_hn_abs,
        }
    if (
        rem22_killed == 0
        and fo_killed <= FO_NULL_MAX
        and delta_hn_abs <= HN_NULL_DELTA
    ):
        return {
            "verdict": "TROPICAL_NULL",
            "reasons": [
                f"fo_killed={fo_killed}<={FO_NULL_MAX}",
                "rem22_killed=0",
                f"|ΔHN|={delta_hn_abs:.4f}<={HN_NULL_DELTA}",
                "falsifies_sum_product_bleed_hypothesis_at_ens_layer",
            ],
            "fo_killed": fo_killed,
            "fo_total": fo_total,
            "rem22_killed": rem22_killed,
            "rem22_total": rem22_total,
            "delta_hn_abs": delta_hn_abs,
        }
    return {
        "verdict": "TROPICAL_PARTIAL",
        "reasons": [
            f"fo_killed={fo_killed}/{fo_total}",
            f"rem22_killed={rem22_killed}/{rem22_total}",
            f"|ΔHN|={delta_hn_abs:.4f}",
            "not_TROPICAL_FO_LIFT",
        ],
        "fo_killed": fo_killed,
        "fo_total": fo_total,
        "rem22_killed": rem22_killed,
        "rem22_total": rem22_total,
        "delta_hn_abs": delta_hn_abs,
    }


def _load_ens(
    ensemble_seeds: tuple[int, ...], max_nodes: int
) -> tuple[list[Any], list[dict[str, Any]]]:
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
    return ens_models, ens_meta


def _eval_slice_metrics(
    ens_preds: Any,
    labels: Any,
    hops: list[int],
    *,
    pair_per: Any,
    ens_max_prob: Any,
    epi_H: Any,
    hard: Any,
    seeds: list[int],
) -> dict[str, Any]:
    ens_all = _metrics_ext(ens_preds, labels, hops)
    fail_mode = _fail_mode_counts(
        ens_preds=ens_preds,
        labels=labels,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hops=hops,
    )
    fail_mode["ens_overall_acc"] = ens_all["overall_acc"]
    global_disagree = _pairwise_disagreement_rate(hard)
    d_hn = _pairwise_rate_on_mask(hard, _slice_mask(hops, "hard_neg"))
    d_k16 = _pairwise_rate_on_mask(hard, _slice_mask(hops, "K16"))
    d_hard = (
        0.5 * (d_hn + d_k16)
        if d_hn == d_hn and d_k16 == d_k16
        else float("nan")
    )
    singles = []
    for mi, seed in enumerate(seeds):
        m = _metrics_ext(hard[mi], labels, hops)
        m["seed"] = seed
        singles.append(m)
    mu_acc = _mean(
        [s["overall_acc"] for s in singles if s["overall_acc"] == s["overall_acc"]]
    )
    cd = _competent_dissonance(mu_acc, d_hard if d_hard == d_hard else 0.0)
    return {
        "ensemble": ens_all,
        "fail_mode": fail_mode,
        "competent_dissonance": cd,
        "D_hard": d_hard,
        "global_pairwise_disagreement": global_disagree,
        "singles_mean_overall": mu_acc,
    }


def _fo_ids_from_fail(
    ens_preds: Any,
    labels: Any,
    pair_per: Any,
    ens_max_prob: Any,
) -> list[int]:
    wrong = ens_preds != labels
    ids: list[int] = []
    for i in wrong.nonzero(as_tuple=False).view(-1).tolist():
        d_ex = float(pair_per[i].item())
        conf = float(ens_max_prob[i].item())
        if d_ex == 0.0 and conf >= CONF_THRESH:
            ids.append(int(i))
    return ids


def _arm_metrics_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "ensemble": block["ensemble"],
        "fail_mode": {
            k: block["fail_mode"][k]
            for k in (
                "FAIL_OPEN",
                "FAIL_CLOSED",
                "FAIL_AMBIG",
                "n_wrong",
                "verdict",
                "ens_overall_acc",
            )
            if k in block["fail_mode"]
        },
        "CD": block["competent_dissonance"],
        "D_hard": block["D_hard"],
    }


def run_cycle(
    *,
    hops_data: Path = DEFAULT_HOPS,
    matched_data: Path = DEFAULT_MATCHED,
    out_path: Path = DEFAULT_OUT,
    cite30_path: Path = DEFAULT_CITE30,
    cite31_path: Path = DEFAULT_CITE31,
    cite32_path: Path = DEFAULT_CITE32,
    cite35_path: Path = DEFAULT_CITE35,
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS,
    skip_matched: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    if not hops_data.exists():
        raise FileNotFoundError(f"ood_hops missing: {hops_data}")
    rows = load_jsonl(hops_data)
    max_nodes = max(DEFAULT_MAX_NODE_ID, max(int(r["n"]) for r in rows))

    cite30: Optional[dict[str, Any]] = None
    cite31: Optional[dict[str, Any]] = None
    cite32: Optional[dict[str, Any]] = None
    cite35: Optional[dict[str, Any]] = None
    if cite30_path.exists():
        cite30 = json.loads(cite30_path.read_text())
    if cite31_path.exists():
        cite31 = json.loads(cite31_path.read_text())
    if cite32_path.exists():
        cite32 = json.loads(cite32_path.read_text())
    if cite35_path.exists():
        cite35 = json.loads(cite35_path.read_text())

    cite31_fo: Optional[list[int]] = None
    if cite31 is not None:
        cite31_fo = list(
            cite31.get("example_id_list_FO_HN")
            or cite31.get("cohorts", {}).get("FO_HN", {}).get("ids")
            or []
        )

    rem22_ids: Optional[list[int]] = None
    if cite32 is not None:
        rem22_ids = list(
            cite32.get("fo_analysis", {}).get("fo_still_wrong_ids") or []
        )

    print(
        f"[stalk-tropical] loading #14/#18/#22 ens n={len(ensemble_seeds)}",
        file=sys.stderr,
    )
    ens_models, ens_meta = _load_ens(ensemble_seeds, max_nodes)

    print(
        f"[stalk-tropical] infer ood_hops T{FOCUS_T} (baseline + tropical)",
        file=sys.stderr,
    )
    import torch
    import torch.nn.functional as F

    logits, labels, hops = _collect_member_logits(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )
    hard = logits.argmax(dim=-1)
    # Member disagreement / epi still from hard member preds (structural)
    pair_per = _pairwise_disagreement_per_example(hard)
    probs = F.softmax(logits, dim=-1)
    mean_p = probs.mean(dim=0)
    baseline_conf = mean_p.max(dim=-1).values
    total_H = _entropy_nats(mean_p)
    alea_H = _entropy_nats(probs).mean(dim=0)
    epi_H = total_H - alea_H

    # --- Arm 0: baseline prob_mean ---
    ens_preds = _aggregate_preds(logits, method=PRIMARY_AGG)
    baseline = _eval_slice_metrics(
        ens_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=baseline_conf,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    fo_ids = _fo_ids_from_fail(ens_preds, labels, pair_per, baseline_conf)
    if cite31_fo is not None and len(cite31_fo) == CITE_30_FO:
        if set(fo_ids) == set(cite31_fo):
            fo_ids = list(cite31_fo)

    if rem22_ids is None or len(rem22_ids) != CITE_32_REM:
        got_n = 0 if rem22_ids is None else len(rem22_ids)
        print(
            "[stalk-tropical] WARN: rem22 ids missing/mismatch; "
            f"got {got_n} expected {CITE_32_REM}",
            file=sys.stderr,
        )
        rem22_ids = list(
            (cite32 or {}).get("fo_analysis", {}).get("fo_still_wrong_ids") or []
        )

    # --- Arm 1: logit_max (primary tropical) ---
    trop_preds, trop_conf, _trop_scores = max_plus_aggregate(logits)
    tropical = _eval_slice_metrics(
        trop_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=trop_conf,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    # --- Arm 2: beta_inf_member (secondary) ---
    beta_preds, beta_conf, beta_mstar = beta_inf_member_preds(logits)
    beta_arm = _eval_slice_metrics(
        beta_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=beta_conf,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    def _killed(ids: list[int], preds: Any) -> list[int]:
        return [
            i for i in ids if int(preds[i].item()) == int(labels[i].item())
        ]

    fo_killed_trop = _killed(fo_ids, trop_preds)
    fo_killed_beta = _killed(fo_ids, beta_preds)
    rem_killed_trop = _killed(rem22_ids, trop_preds)
    rem_killed_beta = _killed(rem22_ids, beta_preds)

    # FO under tropical arm's own labeling (secondary diagnostic)
    trop_fo_ids = _fo_ids_from_fail(trop_preds, labels, pair_per, trop_conf)
    beta_fo_ids = _fo_ids_from_fail(beta_preds, labels, pair_per, beta_conf)

    b_hn = float(baseline["ensemble"]["hard_neg_acc"])
    t_hn = float(tropical["ensemble"]["hard_neg_acc"])
    delta_hn = t_hn - b_hn
    delta_hn_abs = abs(delta_hn)

    # Matched-OOD collateral
    matched_block: dict[str, Any] = {"skipped": True}
    collateral_reasons: list[str] = []
    if not skip_matched and matched_data.exists():
        print(
            f"[stalk-tropical] infer matched-OOD T{FOCUS_T} collateral",
            file=sys.stderr,
        )
        mrows = load_jsonl(matched_data)
        m_max = max(max_nodes, max(int(r["n"]) for r in mrows))
        if m_max > max_nodes:
            ens_models, ens_meta = _load_ens(ensemble_seeds, m_max)
            max_nodes = m_max
        m_logits, m_labels, m_hops = _collect_member_logits(
            ens_models, mrows, T=FOCUS_T, max_nodes=max_nodes
        )
        m_ens = _aggregate_preds(m_logits, method=PRIMARY_AGG)
        m_trop, _, _ = max_plus_aggregate(m_logits)
        m_beta, _, _ = beta_inf_member_preds(m_logits)
        m_base_m = _metrics_ext(m_ens, m_labels, m_hops)
        m_trop_m = _metrics_ext(m_trop, m_labels, m_hops)
        m_beta_m = _metrics_ext(m_beta, m_labels, m_hops)

        def _deltas(gated: dict[str, Any]) -> dict[str, float]:
            return {
                "overall_acc": float(
                    gated["overall_acc"] - m_base_m["overall_acc"]
                ),
                "hard_neg_acc": float(
                    gated["hard_neg_acc"] - m_base_m["hard_neg_acc"]
                ),
                "K16": float(gated["K16"] - m_base_m["K16"]),
            }

        d_trop = _deltas(m_trop_m)
        d_beta = _deltas(m_beta_m)
        for name, d in (("logit_max", d_trop), ("beta_inf_member", d_beta)):
            for k in ("overall_acc", "hard_neg_acc", "K16"):
                if d[k] <= -COLLATERAL_DROP:
                    collateral_reasons.append(
                        f"matched_{name}_{k}_drop={-d[k]:.4f}>={COLLATERAL_DROP}"
                    )
        matched_block = {
            "skipped": False,
            "T": FOCUS_T,
            "n": len(mrows),
            "baseline_prob_mean": m_base_m,
            "logit_max": m_trop_m,
            "beta_inf_member": m_beta_m,
            "deltas_vs_baseline": {
                "logit_max": d_trop,
                "beta_inf_member": d_beta,
            },
        }

    collateral_harm = len(collateral_reasons) > 0

    decision = decide_verdict(
        fo_killed=len(fo_killed_trop),
        fo_total=len(fo_ids),
        rem22_killed=len(rem_killed_trop),
        rem22_total=len(rem22_ids),
        delta_hn_abs=delta_hn_abs,
        collateral_harm=collateral_harm,
        collateral_reasons=collateral_reasons,
    )
    # Also record beta decision for transparency (primary = logit_max)
    decision_beta = decide_verdict(
        fo_killed=len(fo_killed_beta),
        fo_total=len(fo_ids),
        rem22_killed=len(rem_killed_beta),
        rem22_total=len(rem22_ids),
        delta_hn_abs=abs(
            float(beta_arm["ensemble"]["hard_neg_acc"]) - b_hn
        ),
        collateral_harm=collateral_harm,
        collateral_reasons=collateral_reasons,
    )

    # Cite #30 replicate check
    b_fo = int(baseline["fail_mode"]["FAIL_OPEN"])
    b_k16 = float(baseline["ensemble"]["K16"])
    b_ov = float(baseline["ensemble"]["overall_acc"])
    cite30_ok = (
        b_fo == CITE_30_FO
        and abs(b_hn - CITE_30_HN) <= CITE_TOL
        and abs(b_k16 - CITE_30_K16) <= CITE_TOL
        and abs(b_ov - CITE_30_OV) <= CITE_TOL
    )

    # Cert reference from #35 (no re-infer)
    cert_ref: dict[str, Any] = {"available": False}
    if cite35 is not None:
        ct = cite35.get("comparison_table", {}).get("cert", {})
        cert_ref = {
            "available": True,
            "source": str(cite35_path),
            "verdict": (cite35.get("decision") or {}).get("verdict"),
            "HN": ct.get("HN"),
            "K16": ct.get("K16"),
            "overall": ct.get("overall"),
            "FO": ct.get("FO"),
            "FO_killed_of_45": ct.get("FO_killed_of_45"),
            "rem22_killed": ct.get("rem22_killed"),
            "note": (
                "Sealed #35 CERT_FO_CATCH reference only; "
                "tropical ≢ certificate"
            ),
        }

    elapsed = time.time() - t0
    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "section22_unchanged": True,
        "hop_ood_remains_measure_residue": True,
        "architecture": {
            "probe_scope": "ensemble_aggregation_logit_mix",
            "in_attention_rewrite": False,
            "tropical_defs": tropical_definition_doc(),
            "frozen_ckpts": "#14/#18/#22 ens",
            "no_train": True,
            "no_beta_anneal": True,
            "no_dear": True,
            "d": DEFAULT_D,
        },
        "datasets": {
            "ood_hops": str(hops_data),
            "matched_ood": str(matched_data),
            "T": FOCUS_T,
            "n_ood_hops": len(rows),
        },
        "members_pr22": ens_meta,
        "prereg": {
            "arms": [
                "baseline_prob_mean",
                "logit_max",
                "beta_inf_member",
                "cert_ref_#35",
            ],
            "verdicts": list(VERDICTS),
            "FO_LIFT_MIN": FO_LIFT_MIN,
            "REM22_LIFT_MIN": REM22_LIFT_MIN,
            "FO_NULL_MAX": FO_NULL_MAX,
            "HN_NULL_DELTA": HN_NULL_DELTA,
            "COLLATERAL_DROP": COLLATERAL_DROP,
            "CONF_THRESH": CONF_THRESH,
            "D_CLOSED_MIN": D_CLOSED_MIN,
            "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
            "primary_tropical_arm": "logit_max",
        },
        "replicate_cite30": {
            "ok": cite30_ok,
            "FO": b_fo,
            "HN": b_hn,
            "K16": b_k16,
            "overall": b_ov,
            "expected_FO": CITE_30_FO,
            "expected_HN": CITE_30_HN,
            "cite30_path": str(cite30_path) if cite30 else None,
        },
        "fo_ids_baseline": fo_ids,
        "rem22_ids": rem22_ids,
        "arms": {
            "baseline_prob_mean": _arm_metrics_block(baseline),
            "logit_max": {
                **_arm_metrics_block(tropical),
                "fo_killed_of_baseline45": len(fo_killed_trop),
                "fo_killed_ids": fo_killed_trop,
                "rem22_killed": len(rem_killed_trop),
                "rem22_killed_ids": rem_killed_trop,
                "arm_own_FO_count": len(trop_fo_ids),
            },
            "beta_inf_member": {
                **_arm_metrics_block(beta_arm),
                "fo_killed_of_baseline45": len(fo_killed_beta),
                "fo_killed_ids": fo_killed_beta,
                "rem22_killed": len(rem_killed_beta),
                "rem22_killed_ids": rem_killed_beta,
                "arm_own_FO_count": len(beta_fo_ids),
                "member_routing_unique": int(
                    torch.unique(beta_mstar).numel()
                ),
            },
            "cert_ref_#35": cert_ref,
        },
        "comparison_table": {
            "baseline": {
                "HN": b_hn,
                "K16": b_k16,
                "overall": b_ov,
                "FO": b_fo,
                "FC": int(baseline["fail_mode"]["FAIL_CLOSED"]),
                "rem22_FO": len(rem22_ids),
            },
            "logit_max": {
                "HN": float(tropical["ensemble"]["hard_neg_acc"]),
                "K16": float(tropical["ensemble"]["K16"]),
                "overall": float(tropical["ensemble"]["overall_acc"]),
                "FO": int(tropical["fail_mode"]["FAIL_OPEN"]),
                "FC": int(tropical["fail_mode"]["FAIL_CLOSED"]),
                "FO_killed_of_45": len(fo_killed_trop),
                "rem22_killed": len(rem_killed_trop),
                "delta_HN": delta_hn,
            },
            "beta_inf_member": {
                "HN": float(beta_arm["ensemble"]["hard_neg_acc"]),
                "K16": float(beta_arm["ensemble"]["K16"]),
                "overall": float(beta_arm["ensemble"]["overall_acc"]),
                "FO": int(beta_arm["fail_mode"]["FAIL_OPEN"]),
                "FC": int(beta_arm["fail_mode"]["FAIL_CLOSED"]),
                "FO_killed_of_45": len(fo_killed_beta),
                "rem22_killed": len(rem_killed_beta),
                "delta_HN": float(beta_arm["ensemble"]["hard_neg_acc"]) - b_hn,
            },
            "cert_ref_#35": {
                "HN": cert_ref.get("HN"),
                "K16": cert_ref.get("K16"),
                "overall": cert_ref.get("overall"),
                "FO": cert_ref.get("FO"),
                "FO_killed_of_45": cert_ref.get("FO_killed_of_45"),
                "rem22_killed": cert_ref.get("rem22_killed"),
                "note": "reference only; not a tropical arm",
            },
        },
        "collateral": matched_block,
        "decision": {
            **decision,
            "primary_arm": "logit_max",
            "secondary_arm_decision": decision_beta,
        },
        "verdict": decision["verdict"],
        "residue": (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_WALL/CERT_FO_CATCH/"
            + decision["verdict"]
        ),
        "open_status": {
            "science_open": False,
            "section22_widened": False,
            "claim_hop_ood_open": False,
            "tropical_equals_certificate": False,
        },
        "elapsed_sec": elapsed,
        "git_sha": _git_sha(),
        "reading": [],
    }

    # Reading bullets
    report["reading"] = [
        (
            f"Primary tropical arm logit_max: FO killed "
            f"{len(fo_killed_trop)}/45; rem22 killed {len(rem_killed_trop)}/22; "
            f"HN {b_hn:.3f}→{t_hn:.3f} (Δ={delta_hn:+.3f})."
        ),
        (
            f"Secondary beta_inf_member: FO killed {len(fo_killed_beta)}/45; "
            f"rem22 killed {len(rem_killed_beta)}/22."
        ),
        (
            "Scoped probe = ens aggregation only (in-attn rewrite impractical "
            "on frozen softmax MHA). Tropical ≢ certificate (#35 reference)."
        ),
        (
            f"Verdict={decision['verdict']}. science_open=false; "
            "§22 not widened; no hop-OOD OPEN claim."
        ),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-tropical] wrote {out_path} verdict={decision['verdict']} "
        f"FO_killed={len(fo_killed_trop)}/45 rem22={len(rem_killed_trop)}/22 "
        f"HN={b_hn:.3f}→{t_hn:.3f} elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="CYCLE_STALK_TROPICAL_ATTENTION_PROBE (MEASURE, eval-only)"
    )
    p.add_argument("--hops", type=Path, default=DEFAULT_HOPS)
    p.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cite30", type=Path, default=DEFAULT_CITE30)
    p.add_argument("--cite31", type=Path, default=DEFAULT_CITE31)
    p.add_argument("--cite32", type=Path, default=DEFAULT_CITE32)
    p.add_argument("--cite35", type=Path, default=DEFAULT_CITE35)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENSEMBLE_SEEDS),
    )
    p.add_argument("--skip-matched", action="store_true")
    args = p.parse_args(argv)
    run_cycle(
        hops_data=args.hops,
        matched_data=args.matched,
        out_path=args.out,
        cite30_path=args.cite30,
        cite31_path=args.cite31,
        cite32_path=args.cite32,
        cite35_path=args.cite35,
        ensemble_seeds=tuple(args.seeds),
        skip_matched=args.skip_matched,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
