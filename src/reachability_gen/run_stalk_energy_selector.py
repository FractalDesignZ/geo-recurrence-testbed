"""CYCLE_STALK_ENERGY_SELECTOR — MEASURE eval-only energy member selection.

Eval-only @ T=16 on data/ood_hops.jsonl (+ matched-OOD collateral) with
frozen #14/#18/#22 ens. Separate generation from selection via explicit
E_free (cert-free degree/cone/local) and secondary E_cert (cert-energy).
NOT a trained head. NOT multi-hyp JS. NOT bag. NOT orientation. NOT tropical
anneal. science_open=false (not widened; §22 unchanged).

Usage::

    python -m reachability_gen.run_stalk_energy_selector
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.energy_selector import (
    ALPHA_SOUND,
    BETA_CONE,
    EPS_CONF,
    GAMMA_CONF,
    TAU_WEIGHTED,
    cert_refuse_preds,
    compute_energy_stack,
    e_conf,
    energy_argmin_preds,
    energy_definition_doc,
    energy_weighted_preds,
    member_preds_and_conf,
    oracle_member_preds,
    pearson_corr,
)
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

CYCLE = "CYCLE_STALK_ENERGY_SELECTOR"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_MATCHED = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_energy_selector.json")
DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE32 = Path("artifacts/stalk_sound_outdeg_gate.json")
DEFAULT_CITE35 = Path("artifacts/stalk_reach_certificates.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))

VERDICTS = (
    "ENERGY_BEATS_MEAN",
    "ENERGY_PARTIAL",
    "ENERGY_NULL",
    "ENERGY_IS_CERT",
    "COLLATERAL_HARM",
)

# Thresholds (LOCKED in prereg)
GAP_CLOSE_MIN = 0.02
GAP_PARTIAL_MIN = 0.005
FO_BEATS_MIN = 8  # of 45
REM22_BEATS_MIN = 4  # of 22
FO_NULL_MAX = 2
HN_NULL_DELTA = 0.05
HN_PARTIAL_DELTA = 0.01
COLLATERAL_DROP = 0.05
CONF_CORR_MAX = 0.95  # |r(E, −log p)| ≥ this + no sound/cone diff → falsify
CERT_FO_TRACK = 40  # of 45 for ENERGY_IS_CERT secondary track
CERT_REM_TRACK = 16  # of 22
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
    fo_killed_free: int,
    fo_total: int,
    rem22_killed_free: int,
    rem22_total: int,
    gap_close_overall: float,
    delta_hn_abs: float,
    delta_hn: float,
    energy_is_just_conf: bool,
    fo_killed_cert: int,
    rem22_killed_cert: int,
    collateral_harm: bool,
    collateral_reasons: list[str],
) -> dict[str, Any]:
    """LOCKED verdict enum with collateral priority."""
    base = {
        "fo_killed_free": fo_killed_free,
        "fo_total": fo_total,
        "rem22_killed_free": rem22_killed_free,
        "rem22_total": rem22_total,
        "gap_close_overall": gap_close_overall,
        "delta_hn": delta_hn,
        "fo_killed_cert": fo_killed_cert,
        "rem22_killed_cert": rem22_killed_cert,
        "energy_is_just_conf": energy_is_just_conf,
    }
    if collateral_harm:
        return {
            "verdict": "COLLATERAL_HARM",
            "reasons": list(collateral_reasons),
            **base,
        }

    beats = (
        (
            gap_close_overall >= GAP_CLOSE_MIN
            or fo_killed_free >= FO_BEATS_MIN
            or rem22_killed_free >= REM22_BEATS_MIN
        )
        and not energy_is_just_conf
    )
    if beats:
        return {
            "verdict": "ENERGY_BEATS_MEAN",
            "reasons": [
                f"gap_close={gap_close_overall:.4f}",
                f"fo_killed_free={fo_killed_free}",
                f"rem22_killed_free={rem22_killed_free}",
                "energy_not_just_confidence",
            ],
            **base,
        }

    cert_tracks = (
        fo_killed_cert >= CERT_FO_TRACK
        and rem22_killed_cert >= CERT_REM_TRACK
    )
    free_weak = (
        fo_killed_free <= FO_NULL_MAX
        and rem22_killed_free == 0
        and gap_close_overall < GAP_PARTIAL_MIN
        and delta_hn_abs <= HN_NULL_DELTA
    ) or (
        fo_killed_free < FO_BEATS_MIN
        and rem22_killed_free < REM22_BEATS_MIN
        and gap_close_overall < GAP_CLOSE_MIN
    )
    if cert_tracks and free_weak:
        return {
            "verdict": "ENERGY_IS_CERT",
            "reasons": [
                f"cert_fo_killed={fo_killed_cert}>={CERT_FO_TRACK}",
                f"cert_rem22_killed={rem22_killed_cert}>={CERT_REM_TRACK}",
                "cert_free_did_not_beat_mean_toward_oracle",
                "only_working_energy_is_certificate_tied",
            ],
            **base,
        }

    if (
        rem22_killed_free == 0
        and fo_killed_free <= FO_NULL_MAX
        and delta_hn_abs <= HN_NULL_DELTA
        and gap_close_overall < GAP_PARTIAL_MIN
    ):
        return {
            "verdict": "ENERGY_NULL",
            "reasons": [
                f"fo_killed_free={fo_killed_free}<={FO_NULL_MAX}",
                "rem22_killed_free=0",
                f"|ΔHN|={delta_hn_abs:.4f}<={HN_NULL_DELTA}",
                f"gap_close={gap_close_overall:.4f}<{GAP_PARTIAL_MIN}",
            ],
            **base,
        }

    # PARTIAL: some movement
    partial_ok = (
        fo_killed_free >= 1
        or gap_close_overall >= GAP_PARTIAL_MIN
        or delta_hn >= HN_PARTIAL_DELTA
    )
    if partial_ok:
        return {
            "verdict": "ENERGY_PARTIAL",
            "reasons": [
                f"fo_killed_free={fo_killed_free}/{fo_total}",
                f"rem22_killed_free={rem22_killed_free}/{rem22_total}",
                f"gap_close={gap_close_overall:.4f}",
                f"ΔHN={delta_hn:+.4f}",
                "not_ENERGY_BEATS_MEAN",
            ],
            **base,
        }

    return {
        "verdict": "ENERGY_NULL",
        "reasons": [
            f"fo_killed_free={fo_killed_free}",
            f"rem22_killed_free={rem22_killed_free}",
            f"gap_close={gap_close_overall:.4f}",
            "no_meaningful_beat_of_mean_toward_oracle",
        ],
        **base,
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
    import torch
    import torch.nn.functional as F

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
        f"[stalk-energy] loading #14/#18/#22 ens n={len(ensemble_seeds)}",
        file=sys.stderr,
    )
    ens_models, ens_meta = _load_ens(ensemble_seeds, max_nodes)

    print(
        f"[stalk-energy] infer ood_hops T{FOCUS_T} (baseline + energy arms)",
        file=sys.stderr,
    )
    logits, labels, hops = _collect_member_logits(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )
    hard = logits.argmax(dim=-1)
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
            "[stalk-energy] WARN: rem22 ids missing/mismatch; "
            f"got {got_n} expected {CITE_32_REM}",
            file=sys.stderr,
        )
        rem22_ids = list(
            (cite32 or {}).get("fo_analysis", {}).get("fo_still_wrong_ids") or []
        )

    # --- Energies ---
    print("[stalk-energy] scoring E_free + E_cert", file=sys.stderr)
    E_free = compute_energy_stack(logits, rows, kind="free")
    E_cert = compute_energy_stack(logits, rows, kind="cert")

    # Arm 1: energy_argmin free
    free_preds, free_conf, free_mstar = energy_argmin_preds(logits, E_free)
    free_arm = _eval_slice_metrics(
        free_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=free_conf,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    # Arm 1b: energy_weighted free
    w_preds, w_conf = energy_weighted_preds(logits, E_free, tau=TAU_WEIGHTED)
    weighted_arm = _eval_slice_metrics(
        w_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=w_conf,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    # Arm 2: oracle member
    ora_preds, ora_conf, ora_mstar = oracle_member_preds(logits, labels)
    oracle_arm = _eval_slice_metrics(
        ora_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=ora_conf,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    # Arm 3: energy_argmin cert
    cert_preds, cert_conf, cert_mstar = energy_argmin_preds(logits, E_cert)
    cert_arm = _eval_slice_metrics(
        cert_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=cert_conf,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    # Arm 3b: cert refuse
    refuse_preds, refuse_conf, refused = cert_refuse_preds(
        logits, E_cert, refuse_threshold=ALPHA_SOUND
    )
    refuse_arm = _eval_slice_metrics(
        refuse_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=refuse_conf,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    def _killed(ids: list[int], preds: Any) -> list[int]:
        return [
            i for i in ids if int(preds[i].item()) == int(labels[i].item())
        ]

    fo_killed_free = _killed(fo_ids, free_preds)
    fo_killed_cert = _killed(fo_ids, cert_preds)
    fo_killed_refuse = _killed(fo_ids, refuse_preds)
    fo_killed_weighted = _killed(fo_ids, w_preds)
    rem_killed_free = _killed(rem22_ids, free_preds)
    rem_killed_cert = _killed(rem22_ids, cert_preds)
    rem_killed_refuse = _killed(rem22_ids, refuse_preds)
    rem_killed_weighted = _killed(rem22_ids, w_preds)
    rem_killed_oracle = _killed(rem22_ids, ora_preds)
    fo_killed_oracle = _killed(fo_ids, ora_preds)

    b_hn = float(baseline["ensemble"]["hard_neg_acc"])
    b_ov = float(baseline["ensemble"]["overall_acc"])
    b_k16 = float(baseline["ensemble"]["K16"])
    f_hn = float(free_arm["ensemble"]["hard_neg_acc"])
    f_ov = float(free_arm["ensemble"]["overall_acc"])
    o_ov = float(oracle_arm["ensemble"]["overall_acc"])
    o_hn = float(oracle_arm["ensemble"]["hard_neg_acc"])
    delta_hn = f_hn - b_hn
    delta_hn_abs = abs(delta_hn)

    gap_mean = o_ov - b_ov
    gap_free = o_ov - f_ov
    gap_close = gap_mean - gap_free  # positive ⇒ free closer to oracle than mean

    # Energy vs confidence correlation (falsify if ≈)
    preds_m, conf_m = member_preds_and_conf(logits)
    e_flat = E_free.reshape(-1).tolist()
    nlogp_flat = [e_conf(float(c)) for c in conf_m.reshape(-1).tolist()]
    r_e_conf = pearson_corr(e_flat, nlogp_flat)
    # Do sound/cone terms ever differentiate members on any example?
    # Compare E_free with γ=0 (sound+cone only) variance across members
    sound_cone_differs = False
    for j in range(E_free.shape[1]):
        col = E_free[:, j]
        # subtract conf contribution approximately: if variance of E remains
        # after removing γ*(-log p), sound/cone differ
        conf_part = GAMMA_CONF * torch.tensor(
            [e_conf(float(conf_m[i, j].item())) for i in range(E_free.shape[0])],
            dtype=torch.float64,
        )
        residual = col - conf_part
        if float(residual.max() - residual.min()) > 1e-9:
            sound_cone_differs = True
            break
    energy_is_just_conf = (
        (r_e_conf == r_e_conf)
        and abs(float(r_e_conf)) >= CONF_CORR_MAX
        and not sound_cone_differs
    )

    # Matched-OOD collateral
    matched_block: dict[str, Any] = {"skipped": True}
    collateral_reasons: list[str] = []
    if not skip_matched and matched_data.exists():
        print(
            f"[stalk-energy] infer matched-OOD T{FOCUS_T} collateral",
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
        m_E_free = compute_energy_stack(m_logits, mrows, kind="free")
        m_free, _, _ = energy_argmin_preds(m_logits, m_E_free)
        m_E_cert = compute_energy_stack(m_logits, mrows, kind="cert")
        m_cert, _, _ = energy_argmin_preds(m_logits, m_E_cert)
        m_base_m = _metrics_ext(m_ens, m_labels, m_hops)
        m_free_m = _metrics_ext(m_free, m_labels, m_hops)
        m_cert_m = _metrics_ext(m_cert, m_labels, m_hops)

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

        d_free = _deltas(m_free_m)
        d_cert = _deltas(m_cert_m)
        # Primary collateral = E_free arm
        for k in ("overall_acc", "hard_neg_acc", "K16"):
            if d_free[k] <= -COLLATERAL_DROP:
                collateral_reasons.append(
                    f"matched_E_free_{k}_drop={-d_free[k]:.4f}>={COLLATERAL_DROP}"
                )
        matched_block = {
            "skipped": False,
            "T": FOCUS_T,
            "n": len(mrows),
            "baseline_prob_mean": m_base_m,
            "energy_argmin_free": m_free_m,
            "energy_argmin_cert": m_cert_m,
            "deltas_vs_baseline": {
                "energy_argmin_free": d_free,
                "energy_argmin_cert": d_cert,
            },
        }

    collateral_harm = len(collateral_reasons) > 0

    decision = decide_verdict(
        fo_killed_free=len(fo_killed_free),
        fo_total=len(fo_ids),
        rem22_killed_free=len(rem_killed_free),
        rem22_total=len(rem22_ids),
        gap_close_overall=float(gap_close),
        delta_hn_abs=delta_hn_abs,
        delta_hn=delta_hn,
        energy_is_just_conf=energy_is_just_conf,
        fo_killed_cert=len(fo_killed_cert),
        rem22_killed_cert=len(rem_killed_cert),
        collateral_harm=collateral_harm,
        collateral_reasons=collateral_reasons,
    )

    b_fo = int(baseline["fail_mode"]["FAIL_OPEN"])
    cite30_ok = (
        b_fo == CITE_30_FO
        and abs(b_hn - CITE_30_HN) <= CITE_TOL
        and abs(b_k16 - CITE_30_K16) <= CITE_TOL
        and abs(b_ov - CITE_30_OV) <= CITE_TOL
    )

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
                "E_cert expected to track; E_free is primary scientific arm"
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
            "probe_scope": "member_selection_via_explicit_energy",
            "generation": "#22 ens members frozen",
            "selection": "energy_argmin / energy_weighted / oracle",
            "trained_head": False,
            "multi_hyp_js": False,
            "bag_chaos": False,
            "orientation_metrics": False,
            "tropical_anneal": False,
            "energy_defs": energy_definition_doc(),
            "exact_E_free": (
                f"{ALPHA_SOUND}·E_sound + {BETA_CONE}·E_cone + "
                f"{GAMMA_CONF}·(−log(p+{EPS_CONF}))"
            ),
            "exact_E_cert": (
                f"{ALPHA_SOUND}·E_disagree + {GAMMA_CONF}·(−log(p+{EPS_CONF}))"
            ),
            "frozen_ckpts": "#14/#18/#22 ens",
            "no_train": True,
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
                "energy_argmin_free",
                "energy_weighted_free",
                "oracle_member",
                "energy_argmin_cert",
                "cert_energy_refuse",
                "cert_ref_#35",
            ],
            "verdicts": list(VERDICTS),
            "GAP_CLOSE_MIN": GAP_CLOSE_MIN,
            "FO_BEATS_MIN": FO_BEATS_MIN,
            "REM22_BEATS_MIN": REM22_BEATS_MIN,
            "FO_NULL_MAX": FO_NULL_MAX,
            "HN_NULL_DELTA": HN_NULL_DELTA,
            "COLLATERAL_DROP": COLLATERAL_DROP,
            "CONF_CORR_MAX": CONF_CORR_MAX,
            "CONF_THRESH": CONF_THRESH,
            "D_CLOSED_MIN": D_CLOSED_MIN,
            "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
            "primary_scientific_arm": "energy_argmin_free",
            "alpha": ALPHA_SOUND,
            "beta": BETA_CONE,
            "gamma": GAMMA_CONF,
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
        "selector_oracle_gap": {
            "oracle_overall": o_ov,
            "oracle_HN": o_hn,
            "baseline_overall": b_ov,
            "free_overall": f_ov,
            "gap_mean": gap_mean,
            "gap_free": gap_free,
            "gap_close_overall": gap_close,
            "oracle_fo_killed": len(fo_killed_oracle),
            "oracle_rem22_killed": len(rem_killed_oracle),
            "note": (
                "gap_close > 0 means E_free closer to oracle than prob_mean"
            ),
        },
        "energy_vs_confidence": {
            "pearson_r_E_free_vs_neglog_p": r_e_conf,
            "sound_cone_differs_across_members": sound_cone_differs,
            "energy_is_just_confidence_falsified": energy_is_just_conf,
            "CONF_CORR_MAX": CONF_CORR_MAX,
        },
        "arms": {
            "baseline_prob_mean": _arm_metrics_block(baseline),
            "energy_argmin_free": {
                **_arm_metrics_block(free_arm),
                "fo_killed_of_baseline45": len(fo_killed_free),
                "fo_killed_ids": fo_killed_free,
                "rem22_killed": len(rem_killed_free),
                "rem22_killed_ids": rem_killed_free,
                "member_routing_unique": int(torch.unique(free_mstar).numel()),
            },
            "energy_weighted_free": {
                **_arm_metrics_block(weighted_arm),
                "fo_killed_of_baseline45": len(fo_killed_weighted),
                "rem22_killed": len(rem_killed_weighted),
            },
            "oracle_member": {
                **_arm_metrics_block(oracle_arm),
                "fo_killed_of_baseline45": len(fo_killed_oracle),
                "rem22_killed": len(rem_killed_oracle),
                "member_routing_unique": int(torch.unique(ora_mstar).numel()),
            },
            "energy_argmin_cert": {
                **_arm_metrics_block(cert_arm),
                "fo_killed_of_baseline45": len(fo_killed_cert),
                "fo_killed_ids": fo_killed_cert,
                "rem22_killed": len(rem_killed_cert),
                "rem22_killed_ids": rem_killed_cert,
                "member_routing_unique": int(torch.unique(cert_mstar).numel()),
                "label": "cert-energy",
            },
            "cert_energy_refuse": {
                **_arm_metrics_block(refuse_arm),
                "fo_killed_of_baseline45": len(fo_killed_refuse),
                "rem22_killed": len(rem_killed_refuse),
                "n_refused": int(refused.sum().item()),
                "label": "cert-energy+refuse",
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
            "energy_argmin_free": {
                "HN": f_hn,
                "K16": float(free_arm["ensemble"]["K16"]),
                "overall": f_ov,
                "FO": int(free_arm["fail_mode"]["FAIL_OPEN"]),
                "FC": int(free_arm["fail_mode"]["FAIL_CLOSED"]),
                "FO_killed_of_45": len(fo_killed_free),
                "rem22_killed": len(rem_killed_free),
                "delta_HN": delta_hn,
                "gap_to_oracle_overall": gap_free,
            },
            "energy_weighted_free": {
                "HN": float(weighted_arm["ensemble"]["hard_neg_acc"]),
                "K16": float(weighted_arm["ensemble"]["K16"]),
                "overall": float(weighted_arm["ensemble"]["overall_acc"]),
                "FO": int(weighted_arm["fail_mode"]["FAIL_OPEN"]),
                "FO_killed_of_45": len(fo_killed_weighted),
                "rem22_killed": len(rem_killed_weighted),
            },
            "oracle_member": {
                "HN": o_hn,
                "K16": float(oracle_arm["ensemble"]["K16"]),
                "overall": o_ov,
                "FO": int(oracle_arm["fail_mode"]["FAIL_OPEN"]),
                "FO_killed_of_45": len(fo_killed_oracle),
                "rem22_killed": len(rem_killed_oracle),
            },
            "energy_argmin_cert": {
                "HN": float(cert_arm["ensemble"]["hard_neg_acc"]),
                "K16": float(cert_arm["ensemble"]["K16"]),
                "overall": float(cert_arm["ensemble"]["overall_acc"]),
                "FO": int(cert_arm["fail_mode"]["FAIL_OPEN"]),
                "FO_killed_of_45": len(fo_killed_cert),
                "rem22_killed": len(rem_killed_cert),
                "label": "cert-energy",
            },
            "cert_energy_refuse": {
                "HN": float(refuse_arm["ensemble"]["hard_neg_acc"]),
                "K16": float(refuse_arm["ensemble"]["K16"]),
                "overall": float(refuse_arm["ensemble"]["overall_acc"]),
                "FO": int(refuse_arm["fail_mode"]["FAIL_OPEN"]),
                "FO_killed_of_45": len(fo_killed_refuse),
                "rem22_killed": len(rem_killed_refuse),
                "n_refused": int(refused.sum().item()),
            },
            "cert_ref_#35": {
                "HN": cert_ref.get("HN"),
                "K16": cert_ref.get("K16"),
                "overall": cert_ref.get("overall"),
                "FO": cert_ref.get("FO"),
                "FO_killed_of_45": cert_ref.get("FO_killed_of_45"),
                "rem22_killed": cert_ref.get("rem22_killed"),
                "note": "reference only",
            },
        },
        "collateral": matched_block,
        "decision": {
            **decision,
            "primary_arm": "energy_argmin_free",
        },
        "verdict": decision["verdict"],
        "residue": (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_WALL/CERT_FO_CATCH/COLLATERAL_HARM/"
            "INCONCLUSIVE_ARCH/"
            + decision["verdict"]
        ),
        "open_status": {
            "science_open": False,
            "section22_widened": False,
            "claim_hop_ood_open": False,
            "orientation_used": False,
            "tropical_anneal": False,
        },
        "elapsed_sec": elapsed,
        "git_sha": _git_sha(),
        "reading": [],
    }

    report["reading"] = [
        (
            f"E_free energy_argmin: FO killed {len(fo_killed_free)}/45; "
            f"rem22 killed {len(rem_killed_free)}/22; "
            f"HN {b_hn:.3f}→{f_hn:.3f} (Δ={delta_hn:+.3f}); "
            f"overall {b_ov:.3f}→{f_ov:.3f}; "
            f"gap_close vs oracle={gap_close:+.4f}."
        ),
        (
            f"Oracle upper bound: overall {o_ov:.3f}; HN {o_hn:.3f}; "
            f"FO killed {len(fo_killed_oracle)}/45; "
            f"rem22 killed {len(rem_killed_oracle)}/22."
        ),
        (
            f"E_cert (cert-energy): FO killed {len(fo_killed_cert)}/45; "
            f"rem22 killed {len(rem_killed_cert)}/22; "
            f"refuse overlay FO {len(fo_killed_refuse)}/45 "
            f"(n_refused={int(refused.sum().item())})."
        ),
        (
            f"Energy↔confidence: r={r_e_conf}; "
            f"sound/cone differs={sound_cone_differs}; "
            f"just_conf_falsified={energy_is_just_conf}."
        ),
        (
            f"Verdict={decision['verdict']}. science_open=false; "
            "§22 not widened; no hop-OOD OPEN claim; no orientation; "
            "no tropical anneal."
        ),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-energy] wrote {out_path} verdict={decision['verdict']} "
        f"FO_free={len(fo_killed_free)}/45 rem22={len(rem_killed_free)}/22 "
        f"gap_close={gap_close:+.4f} HN={b_hn:.3f}→{f_hn:.3f} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="CYCLE_STALK_ENERGY_SELECTOR (MEASURE, eval-only)"
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
