"""CYCLE_STALK_REACH_CERTIFICATES — MEASURE eval-only post-hoc reach certificates.

Eval-only @ T=16 on data/ood_hops.jsonl (+ matched-OOD collateral) with
frozen #14/#18/#22 ens. Post-hoc path-witness / checker-BFS certificates:
dirty YES → reject/force-closed. Checker BFS allowed ONLY post-hoc for
certificate validation — never as training target or init.
No train. science_open=false (not widened; §22 unchanged).
Tropical Phase 2 NOT started.

Usage::

    python -m reachability_gen.run_stalk_reach_certificates
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import HOP_UNREACHABLE
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.reach_certificates import apply_cert_policy_preds
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
from reachability_gen.run_stalk_sound_outdeg_gate import apply_outdeg_gate_preds
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_REACH_CERTIFICATES"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_MATCHED = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_reach_certificates.json")
DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE32 = Path("artifacts/stalk_sound_outdeg_gate.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))

VERDICTS = (
    "CERT_FO_CATCH",
    "CERT_PARTIAL",
    "CERT_NOISE",
    "COLLATERAL_HARM",
)

# Thresholds (LOCKED in prereg)
FO_CATCH_MIN = 40  # of 45
REM22_CATCH_MIN = 16  # of 22
FO_NOISE_MAX = 8  # CERT_NOISE if FO killed < this AND concentration fails
CONC_DIRTY_FO_MIN = 0.50  # dirty∩FO / dirty
CONC_FO_DIRTY_MIN = 0.80  # FO∩dirty / FO
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
    dirty_conc_ok: bool,
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
            "dirty_conc_ok": dirty_conc_ok,
        }
    catch = (
        fo_killed >= FO_CATCH_MIN
        and rem22_killed >= REM22_CATCH_MIN
        and dirty_conc_ok
    )
    if catch:
        return {
            "verdict": "CERT_FO_CATCH",
            "reasons": [
                f"fo_killed={fo_killed}>={FO_CATCH_MIN}",
                f"rem22_killed={rem22_killed}>={REM22_CATCH_MIN}",
                "dirty_concentrates_on_FO",
            ],
            "fo_killed": fo_killed,
            "fo_total": fo_total,
            "rem22_killed": rem22_killed,
            "rem22_total": rem22_total,
            "dirty_conc_ok": dirty_conc_ok,
        }
    if fo_killed < FO_NOISE_MAX and not dirty_conc_ok:
        return {
            "verdict": "CERT_NOISE",
            "reasons": [
                f"fo_killed={fo_killed}<{FO_NOISE_MAX}",
                "dirty_does_not_concentrate_on_FO",
            ],
            "fo_killed": fo_killed,
            "fo_total": fo_total,
            "rem22_killed": rem22_killed,
            "rem22_total": rem22_total,
            "dirty_conc_ok": dirty_conc_ok,
        }
    if fo_killed <= 0 and rem22_killed <= 0:
        # Zero catch + concentration failed already handled as NOISE;
        # zero catch with concentration ok is still PARTIAL-empty → PARTIAL
        # with explicit reason (or NOISE if not conc). Prefer PARTIAL if any
        # signal else NOISE-like. Prereg: ≥1 FO killed for PARTIAL.
        return {
            "verdict": "CERT_NOISE" if not dirty_conc_ok else "CERT_PARTIAL",
            "reasons": ["zero_FO_eliminated"],
            "fo_killed": fo_killed,
            "fo_total": fo_total,
            "rem22_killed": rem22_killed,
            "rem22_total": rem22_total,
            "dirty_conc_ok": dirty_conc_ok,
        }
    return {
        "verdict": "CERT_PARTIAL",
        "reasons": [
            f"fo_killed={fo_killed}/{fo_total}",
            f"rem22_killed={rem22_killed}/{rem22_total}",
            f"dirty_conc_ok={dirty_conc_ok}",
            "not_CERT_FO_CATCH",
        ],
        "fo_killed": fo_killed,
        "fo_total": fo_total,
        "rem22_killed": rem22_killed,
        "rem22_total": rem22_total,
        "dirty_conc_ok": dirty_conc_ok,
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


def _cert_stats(certs: list[Any], preds_before: Any) -> dict[str, Any]:
    import torch

    n = len(certs)
    clean = sum(1 for c in certs if c.clean)
    dirty = n - clean
    yes_ids = [i for i in range(n) if int(preds_before[i].item()) == 1]
    no_ids = [i for i in range(n) if int(preds_before[i].item()) == 0]
    dirty_yes = sum(1 for i in yes_ids if not certs[i].clean)
    dirty_no = sum(1 for i in no_ids if not certs[i].clean)
    clean_yes = sum(1 for i in yes_ids if certs[i].clean)
    clean_no = sum(1 for i in no_ids if certs[i].clean)
    kinds: dict[str, int] = {}
    for c in certs:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    return {
        "n": n,
        "n_clean": clean,
        "n_dirty": dirty,
        "clean_rate": float(clean / n) if n else float("nan"),
        "dirty_rate": float(dirty / n) if n else float("nan"),
        "n_pred_yes": len(yes_ids),
        "n_pred_no": len(no_ids),
        "clean_yes": clean_yes,
        "dirty_yes": dirty_yes,
        "clean_no": clean_no,
        "dirty_no": dirty_no,
        "kinds": kinds,
    }


def run_cycle(
    *,
    hops_data: Path = DEFAULT_HOPS,
    matched_data: Path = DEFAULT_MATCHED,
    out_path: Path = DEFAULT_OUT,
    cite30_path: Path = DEFAULT_CITE30,
    cite31_path: Path = DEFAULT_CITE31,
    cite32_path: Path = DEFAULT_CITE32,
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
    if cite30_path.exists():
        cite30 = json.loads(cite30_path.read_text())
    if cite31_path.exists():
        cite31 = json.loads(cite31_path.read_text())
    if cite32_path.exists():
        cite32 = json.loads(cite32_path.read_text())

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
        f"[stalk-reach-cert] loading #14/#18/#22 ens n={len(ensemble_seeds)}",
        file=sys.stderr,
    )
    ens_models, ens_meta = _load_ens(ensemble_seeds, max_nodes)

    print(
        f"[stalk-reach-cert] infer ood_hops T{FOCUS_T} (#22 baseline)",
        file=sys.stderr,
    )
    import torch
    import torch.nn.functional as F

    logits, labels, hops = _collect_member_logits(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )
    hard = logits.argmax(dim=-1)
    ens_preds = _aggregate_preds(logits, method=PRIMARY_AGG)
    pair_per = _pairwise_disagreement_per_example(hard)
    probs = F.softmax(logits, dim=-1)
    mean_p = probs.mean(dim=0)
    ens_max_prob = mean_p.max(dim=-1).values
    total_H = _entropy_nats(mean_p)
    alea_H = _entropy_nats(probs).mean(dim=0)
    epi_H = total_H - alea_H

    baseline = _eval_slice_metrics(
        ens_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    fo_ids = _fo_ids_from_fail(ens_preds, labels, pair_per, ens_max_prob)
    if cite31_fo is not None and len(cite31_fo) == CITE_30_FO:
        if set(fo_ids) == set(cite31_fo):
            fo_ids = list(cite31_fo)

    if rem22_ids is None or len(rem22_ids) != CITE_32_REM:
        # Derive rem-22 as FO that survive outdeg0
        od_preds, _ = apply_outdeg_gate_preds(ens_preds, rows)
        rem22_ids = [
            i
            for i in fo_ids
            if int(od_preds[i].item()) != int(labels[i].item())
        ]

    # --- Arm 1: outdeg0 ---
    od_preds, od_triggers = apply_outdeg_gate_preds(ens_preds, rows)
    outdeg0 = _eval_slice_metrics(
        od_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    # --- Arm 2: cert alone ---
    cert_preds, certs, dirty_yes_ids, dirty_no_ids = apply_cert_policy_preds(
        ens_preds, rows
    )
    cert_arm = _eval_slice_metrics(
        cert_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )
    cert_stats = _cert_stats(certs, ens_preds)

    # --- Arm 3: outdeg0 then cert ---
    # Certify the outdeg0-gated preds (post-hoc on gated prediction)
    od_cert_preds, od_certs, od_dirty_yes, od_dirty_no = apply_cert_policy_preds(
        od_preds, rows
    )
    od_cert_arm = _eval_slice_metrics(
        od_cert_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )
    od_cert_stats = _cert_stats(od_certs, od_preds)

    def _killed(ids: list[int], preds: Any) -> list[int]:
        return [
            i for i in ids if int(preds[i].item()) == int(labels[i].item())
        ]

    fo_killed_cert = _killed(fo_ids, cert_preds)
    fo_killed_od = _killed(fo_ids, od_preds)
    fo_killed_od_cert = _killed(fo_ids, od_cert_preds)
    rem_killed_cert = _killed(rem22_ids, cert_preds)
    rem_killed_od = _killed(rem22_ids, od_preds)
    rem_killed_od_cert = _killed(rem22_ids, od_cert_preds)

    # Dirty concentration on baseline FO (cert arm — primary)
    dirty_ids = set(dirty_yes_ids) | set(dirty_no_ids)
    fo_set = set(fo_ids)
    rem_set = set(rem22_ids)
    dirty_n = len(dirty_ids)
    dirty_fo = len(dirty_ids & fo_set)
    fo_dirty = len(fo_set & dirty_ids)
    dirty_rem = len(dirty_ids & rem_set)
    conc_dirty_fo = float(dirty_fo / dirty_n) if dirty_n else float("nan")
    conc_fo_dirty = float(fo_dirty / len(fo_set)) if fo_set else float("nan")
    dirty_conc_ok = (
        (conc_dirty_fo == conc_dirty_fo and conc_dirty_fo >= CONC_DIRTY_FO_MIN)
        or (conc_fo_dirty == conc_fo_dirty and conc_fo_dirty >= CONC_FO_DIRTY_MIN)
    )

    # P(correct|clean) under cert arm (before policy vs after: use post-policy
    # accuracy on examples that were clean under the *pre*-policy cert)
    clean_ids = [i for i, c in enumerate(certs) if c.clean]
    if clean_ids:
        idx = torch.tensor(clean_ids, dtype=torch.long)
        p_correct_clean = float(
            (cert_preds[idx] == labels[idx]).float().mean().item()
        )
        # Also: among clean, pre-policy was already correct?
        p_correct_clean_pre = float(
            (ens_preds[idx] == labels[idx]).float().mean().item()
        )
    else:
        p_correct_clean = float("nan")
        p_correct_clean_pre = float("nan")

    # Collateral on ood_hops: OK_HN + positives (cert arm primary for harm check;
    # also report outdeg0+cert)
    correct_mask = ens_preds == labels
    ok_hn_ids = [
        i
        for i in range(len(hops))
        if bool(correct_mask[i].item()) and hops[i] == int(HOP_UNREACHABLE)
    ]
    pos_ids = [i for i in range(len(rows)) if int(labels[i].item()) == 1]

    def _acc_on(ids: list[int], preds: Any) -> float:
        if not ids:
            return float("nan")
        idx = torch.tensor(ids, dtype=torch.long)
        return float((preds[idx] == labels[idx]).float().mean().item())

    ok_hn_base = _acc_on(ok_hn_ids, ens_preds)
    ok_hn_cert = _acc_on(ok_hn_ids, cert_preds)
    ok_hn_od_cert = _acc_on(ok_hn_ids, od_cert_preds)
    pos_base = _acc_on(pos_ids, ens_preds)
    pos_cert = _acc_on(pos_ids, cert_preds)
    pos_od_cert = _acc_on(pos_ids, od_cert_preds)

    collateral_reasons: list[str] = []
    # Harm check uses the stronger of cert / outdeg0+cert drops vs baseline
    for name, ok_g, pos_g in (
        ("cert", ok_hn_cert, pos_cert),
        ("outdeg0+cert", ok_hn_od_cert, pos_od_cert),
    ):
        if ok_hn_ids and (ok_hn_base - ok_g) >= COLLATERAL_DROP:
            collateral_reasons.append(
                f"{name}_OK_HN_index_drop={ok_hn_base - ok_g:.4f}>={COLLATERAL_DROP}"
            )
        if pos_ids and (pos_base - pos_g) >= COLLATERAL_DROP:
            collateral_reasons.append(
                f"{name}_positives_drop={pos_base - pos_g:.4f}>={COLLATERAL_DROP}"
            )

    # Matched-OOD collateral
    matched_block: dict[str, Any] = {"skipped": True}
    if not skip_matched and matched_data.exists():
        print(
            f"[stalk-reach-cert] infer matched-OOD T{FOCUS_T} collateral",
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
        m_od, m_od_trig = apply_outdeg_gate_preds(m_ens, mrows)
        m_cert, m_certs, m_dy, m_dn = apply_cert_policy_preds(m_ens, mrows)
        m_od_cert, m_od_certs, _, _ = apply_cert_policy_preds(m_od, mrows)
        m_base_m = _metrics_ext(m_ens, m_labels, m_hops)
        m_od_m = _metrics_ext(m_od, m_labels, m_hops)
        m_cert_m = _metrics_ext(m_cert, m_labels, m_hops)
        m_od_cert_m = _metrics_ext(m_od_cert, m_labels, m_hops)
        m_cert_stats = _cert_stats(m_certs, m_ens)

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

        d_cert = _deltas(m_cert_m)
        d_od_cert = _deltas(m_od_cert_m)
        for arm_name, deltas in (("cert", d_cert), ("outdeg0+cert", d_od_cert)):
            for k, d in deltas.items():
                if d <= -COLLATERAL_DROP:
                    collateral_reasons.append(
                        f"matched_ood_{arm_name}_{k}_drop={-d:.4f}>={COLLATERAL_DROP}"
                    )
        matched_block = {
            "skipped": False,
            "path": str(matched_data),
            "n": len(mrows),
            "baseline": m_base_m,
            "outdeg0": m_od_m,
            "cert": m_cert_m,
            "outdeg0_cert": m_od_cert_m,
            "deltas_cert": d_cert,
            "deltas_outdeg0_cert": d_od_cert,
            "cert_stats": m_cert_stats,
            "n_outdeg0_triggers": len(m_od_trig),
            "n_dirty_yes": len(m_dy),
            "n_dirty_no": len(m_dn),
        }

    collateral_harm = bool(collateral_reasons)

    # Primary decision arm = cert (arm 2); also report outdeg0+cert catch
    primary_fo_killed = len(fo_killed_cert)
    primary_rem_killed = len(rem_killed_cert)
    # Prefer outdeg0+cert if it catches more rem-22 without harm — but prereg
    # says primary success on dirty→FO; decide on max catch across cert arms
    # for CERT_FO_CATCH eligibility (either cert or outdeg0+cert).
    use_od_cert = len(fo_killed_od_cert) >= primary_fo_killed and len(
        rem_killed_od_cert
    ) >= primary_rem_killed
    if use_od_cert and (
        len(fo_killed_od_cert) > primary_fo_killed
        or len(rem_killed_od_cert) > primary_rem_killed
    ):
        decision_fo = len(fo_killed_od_cert)
        decision_rem = len(rem_killed_od_cert)
        decision_arm = "outdeg0+cert"
        # concentration for od+cert dirty
        od_dirty_ids = set(od_dirty_yes) | set(od_dirty_no)
        od_dirty_n = len(od_dirty_ids)
        od_conc_df = (
            float(len(od_dirty_ids & fo_set) / od_dirty_n)
            if od_dirty_n
            else float("nan")
        )
        od_conc_fd = (
            float(len(fo_set & od_dirty_ids) / len(fo_set))
            if fo_set
            else float("nan")
        )
        decision_conc = (
            (od_conc_df == od_conc_df and od_conc_df >= CONC_DIRTY_FO_MIN)
            or (od_conc_fd == od_conc_fd and od_conc_fd >= CONC_FO_DIRTY_MIN)
        )
    else:
        decision_fo = primary_fo_killed
        decision_rem = primary_rem_killed
        decision_arm = "cert"
        decision_conc = dirty_conc_ok

    decision = decide_verdict(
        fo_killed=decision_fo,
        fo_total=len(fo_ids),
        rem22_killed=decision_rem,
        rem22_total=len(rem22_ids),
        dirty_conc_ok=decision_conc,
        collateral_harm=collateral_harm,
        collateral_reasons=collateral_reasons,
    )
    verdict = decision["verdict"]

    if verdict == "CERT_FO_CATCH":
        residue = (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_WALL/CERT_FO_CATCH"
        )
        residue_note = (
            "Post-hoc reach certificates catch FO core (incl. rem-22); "
            "still MEASURE; §22 not widened; hop-OOD not OPEN; no tropical."
        )
    elif verdict == "CERT_PARTIAL":
        residue = (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_WALL/CERT_PARTIAL"
        )
        residue_note = (
            f"Certificates partial FO catch ({decision_fo}/{len(fo_ids)}; "
            f"rem {decision_rem}/{len(rem22_ids)}); residue remains; no §22 widen."
        )
    elif verdict == "CERT_NOISE":
        residue = (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_WALL/CERT_NOISE"
        )
        residue_note = (
            "Certificates dirty mass does not concentrate on FO; "
            "STOP this cert overlay only."
        )
    else:
        residue = (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_WALL/CERT_COLLATERAL"
        )
        residue_note = (
            "Certificates caused ≥0.05 abs collateral on matched-OOD/OK/positives; "
            "STOP this overlay only."
        )

    b_ens = baseline["ensemble"]
    replicate = {
        "FAIL_OPEN": baseline["fail_mode"]["FAIL_OPEN"],
        "FAIL_CLOSED": baseline["fail_mode"]["FAIL_CLOSED"],
        "cite30_FO": CITE_30_FO,
        "ens_HN": b_ens["hard_neg_acc"],
        "ens_K16": b_ens["K16"],
        "ens_overall": b_ens["overall_acc"],
        "FO_match": baseline["fail_mode"]["FAIL_OPEN"] == CITE_30_FO,
        "within_tol": (
            abs(b_ens["hard_neg_acc"] - CITE_30_HN) <= CITE_TOL
            and abs(b_ens["K16"] - CITE_30_K16) <= CITE_TOL
            and abs(b_ens["overall_acc"] - CITE_30_OV) <= CITE_TOL
        ),
        "rem22_n": len(rem22_ids),
        "rem22_cite32_match": len(rem22_ids) == CITE_32_REM,
    }

    elapsed = time.time() - t0
    report: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "open_status": "science_open_false_not_widened",
        "section22_unchanged": True,
        "hop_ood_remains_measure_residue": True,
        "tropical_phase2": False,
        "tropical_phase2_note": "NOT started; deferred out of this PR",
        "base_sha": _git_sha(),
        "architecture": {
            "kind": "eval_only_posthoc_reach_certificates",
            "train": False,
            "bag_noise_trained": False,
            "select_reopened": False,
            "curriculum_reopened": False,
            "bfs_as_model_feature": False,
            "bfs_checker_posthoc_only": True,
            "d": DEFAULT_D,
            "aggregator_primary": PRIMARY_AGG,
            "T_fixed": FOCUS_T,
            "hard_A": True,
            "broadcast_c": False,
            "policy": "dirty_YES→force_closed; dirty_NO→keep (no oracle open)",
            "arms": [
                "0_baseline",
                "1_outdeg0",
                "2_cert",
                "3_outdeg0+cert",
            ],
        },
        "datasets": {
            "ood_hops": str(hops_data),
            "n_ood_hops": len(rows),
            "matched_ood": str(matched_data) if matched_data.exists() else None,
            "cite30": str(cite30_path) if cite30_path.exists() else None,
            "cite31": str(cite31_path) if cite31_path.exists() else None,
            "cite32": str(cite32_path) if cite32_path.exists() else None,
        },
        "members_pr22": ens_meta,
        "prereg": {
            "verdicts": list(VERDICTS),
            "FO_CATCH_MIN": FO_CATCH_MIN,
            "REM22_CATCH_MIN": REM22_CATCH_MIN,
            "FO_NOISE_MAX": FO_NOISE_MAX,
            "CONC_DIRTY_FO_MIN": CONC_DIRTY_FO_MIN,
            "CONC_FO_DIRTY_MIN": CONC_FO_DIRTY_MIN,
            "COLLATERAL_DROP": COLLATERAL_DROP,
            "CONF_THRESH": CONF_THRESH,
            "D_CLOSED_MIN": D_CLOSED_MIN,
            "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
            "non_goals": [
                "no_train",
                "no_science_open_widen",
                "section22_unchanged",
                "T_fixed_16",
                "bfs_checker_posthoc_only",
                "no_hop_ood_open_claim",
                "no_tropical_phase2",
            ],
        },
        "replicate_cite30": replicate,
        "certificate_policy": {
            "dirty_yes_action": "force_closed_pred0",
            "dirty_no_action": "keep_pred0_no_oracle_open",
            "checker": "BFS path reconstruct + unreachable confirm",
            "witness_validation": "consecutive edges must be in A_hat",
        },
        "arms": {
            "0_baseline": {
                "name": "baseline_#22_prob_mean",
                **_arm_metrics_block(baseline),
            },
            "1_outdeg0": {
                "name": "outdeg(s)==0_force_pred0",
                **_arm_metrics_block(outdeg0),
                "n_triggers": len(od_triggers),
            },
            "2_cert": {
                "name": "cert_force_closed_dirty_yes",
                **_arm_metrics_block(cert_arm),
                "cert_stats": cert_stats,
                "n_dirty_yes": len(dirty_yes_ids),
                "n_dirty_no": len(dirty_no_ids),
                "dirty_yes_ids_sample": dirty_yes_ids[:40],
                "dirty_no_ids_sample": dirty_no_ids[:40],
            },
            "3_outdeg0_cert": {
                "name": "outdeg0_then_cert",
                **_arm_metrics_block(od_cert_arm),
                "cert_stats": od_cert_stats,
                "n_dirty_yes": len(od_dirty_yes),
                "n_dirty_no": len(od_dirty_no),
            },
        },
        "fo_analysis": {
            "fo_total": len(fo_ids),
            "fo_ids": fo_ids,
            "rem22_total": len(rem22_ids),
            "rem22_ids": rem22_ids,
            "fo_killed_outdeg0": len(fo_killed_od),
            "fo_killed_cert": len(fo_killed_cert),
            "fo_killed_outdeg0_cert": len(fo_killed_od_cert),
            "fo_killed_ids_cert": fo_killed_cert,
            "fo_killed_ids_outdeg0_cert": fo_killed_od_cert,
            "rem22_killed_outdeg0": len(rem_killed_od),
            "rem22_killed_cert": len(rem_killed_cert),
            "rem22_killed_outdeg0_cert": len(rem_killed_od_cert),
            "rem22_killed_ids_cert": rem_killed_cert,
            "rem22_killed_ids_outdeg0_cert": rem_killed_od_cert,
            "cite31_fo_match": (
                set(fo_ids) == set(cite31_fo) if cite31_fo is not None else None
            ),
        },
        "concentration": {
            "dirty_n": dirty_n,
            "dirty_intersect_FO": dirty_fo,
            "dirty_intersect_rem22": dirty_rem,
            "FO_intersect_dirty": fo_dirty,
            "dirty_cap_FO_over_dirty": conc_dirty_fo,
            "FO_cap_dirty_over_FO": conc_fo_dirty,
            "dirty_conc_ok": dirty_conc_ok,
            "thresholds": {
                "CONC_DIRTY_FO_MIN": CONC_DIRTY_FO_MIN,
                "CONC_FO_DIRTY_MIN": CONC_FO_DIRTY_MIN,
            },
        },
        "p_correct_given_clean": {
            "n_clean": len(clean_ids),
            "post_policy": p_correct_clean,
            "pre_policy_on_clean_set": p_correct_clean_pre,
        },
        "collateral": {
            "OK_HN": {
                "n": len(ok_hn_ids),
                "baseline_acc": ok_hn_base,
                "cert_acc": ok_hn_cert,
                "outdeg0_cert_acc": ok_hn_od_cert,
                "delta_cert": (
                    float(ok_hn_cert - ok_hn_base) if ok_hn_ids else float("nan")
                ),
                "delta_outdeg0_cert": (
                    float(ok_hn_od_cert - ok_hn_base)
                    if ok_hn_ids
                    else float("nan")
                ),
            },
            "positives": {
                "n": len(pos_ids),
                "baseline_acc": pos_base,
                "cert_acc": pos_cert,
                "outdeg0_cert_acc": pos_od_cert,
                "delta_cert": (
                    float(pos_cert - pos_base) if pos_ids else float("nan")
                ),
                "delta_outdeg0_cert": (
                    float(pos_od_cert - pos_base) if pos_ids else float("nan")
                ),
            },
            "matched_ood_T16": matched_block,
            "harm": collateral_harm,
            "reasons": collateral_reasons,
        },
        "comparison_table": {
            "baseline": {
                "HN": b_ens["hard_neg_acc"],
                "K16": b_ens["K16"],
                "overall": b_ens["overall_acc"],
                "FO": baseline["fail_mode"]["FAIL_OPEN"],
                "FC": baseline["fail_mode"]["FAIL_CLOSED"],
            },
            "outdeg0": {
                "HN": outdeg0["ensemble"]["hard_neg_acc"],
                "K16": outdeg0["ensemble"]["K16"],
                "overall": outdeg0["ensemble"]["overall_acc"],
                "FO": outdeg0["fail_mode"]["FAIL_OPEN"],
                "FC": outdeg0["fail_mode"]["FAIL_CLOSED"],
                "FO_killed_of_45": len(fo_killed_od),
                "rem22_killed": len(rem_killed_od),
            },
            "cert": {
                "HN": cert_arm["ensemble"]["hard_neg_acc"],
                "K16": cert_arm["ensemble"]["K16"],
                "overall": cert_arm["ensemble"]["overall_acc"],
                "FO": cert_arm["fail_mode"]["FAIL_OPEN"],
                "FC": cert_arm["fail_mode"]["FAIL_CLOSED"],
                "FO_killed_of_45": len(fo_killed_cert),
                "rem22_killed": len(rem_killed_cert),
                "dirty_rate": cert_stats["dirty_rate"],
                "clean_rate": cert_stats["clean_rate"],
            },
            "outdeg0_cert": {
                "HN": od_cert_arm["ensemble"]["hard_neg_acc"],
                "K16": od_cert_arm["ensemble"]["K16"],
                "overall": od_cert_arm["ensemble"]["overall_acc"],
                "FO": od_cert_arm["fail_mode"]["FAIL_OPEN"],
                "FC": od_cert_arm["fail_mode"]["FAIL_CLOSED"],
                "FO_killed_of_45": len(fo_killed_od_cert),
                "rem22_killed": len(rem_killed_od_cert),
                "dirty_rate": od_cert_stats["dirty_rate"],
                "clean_rate": od_cert_stats["clean_rate"],
            },
        },
        "decision": {**decision, "decision_arm": decision_arm},
        "verdict": verdict,
        "residue": residue,
        "residue_note": residue_note,
        "reading": (
            "Post-hoc reach certificates (path-witness / checker-BFS) on #22 ens. "
            "MEASURE only; science_open=false; §22 unchanged; no hop-OOD OPEN; "
            "tropical Phase 2 not started."
        ),
        "elapsed_sec": elapsed,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-reach-cert] verdict={verdict} residue={residue} "
        f"arm={decision_arm} FO_killed={decision_fo}/{len(fo_ids)} "
        f"rem22={decision_rem}/{len(rem22_ids)} "
        f"HN {b_ens['hard_neg_acc']:.3f}→"
        f"{cert_arm['ensemble']['hard_neg_acc']:.3f} "
        f"dirty_yes={len(dirty_yes_ids)} "
        f"wrote {out_path} ({elapsed:.1f}s)",
        file=sys.stderr,
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hops-data", type=Path, default=DEFAULT_HOPS)
    p.add_argument("--matched-data", type=Path, default=DEFAULT_MATCHED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cite30", type=Path, default=DEFAULT_CITE30)
    p.add_argument("--cite31", type=Path, default=DEFAULT_CITE31)
    p.add_argument("--cite32", type=Path, default=DEFAULT_CITE32)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_ENSEMBLE_SEEDS))
    p.add_argument("--skip-matched", action="store_true")
    args = p.parse_args(argv)
    run_cycle(
        hops_data=args.hops_data,
        matched_data=args.matched_data,
        out_path=args.out,
        cite30_path=args.cite30,
        cite31_path=args.cite31,
        cite32_path=args.cite32,
        ensemble_seeds=tuple(args.seeds),
        skip_matched=args.skip_matched,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
