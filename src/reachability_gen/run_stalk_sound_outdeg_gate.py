"""CYCLE_STALK_SOUND_OUTDEG_GATE — MEASURE eval-only sound outdeg gate.

Eval-only @ T=16 on data/ood_hops.jsonl (+ matched-OOD collateral) with
frozen #14/#18/#22 ens. Primary overlay: if outdeg(s)==0 and t!=s, force
ens pred = unreachable (sound hard-Â local incidence; NOT a BFS oracle).
No train. science_open=false (not widened; §22 unchanged).

Usage::

    python -m reachability_gen.run_stalk_sound_outdeg_gate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import HOP_UNREACHABLE
from reachability_gen.encode import parse_instance
from reachability_gen.graph import adjacency_list
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_fractal_core_gate1 import DEFAULT_D, _mean
from reachability_gen.run_stalk_competent_dissonance import (
    D_SAT,
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

CYCLE = "CYCLE_STALK_SOUND_OUTDEG_GATE"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_MATCHED = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_sound_outdeg_gate.json")
DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))

# Verdicts (LOCKED)
VERDICTS = (
    "FO_CORE_KILLED",
    "FO_PARTIAL",
    "FO_UNMOVED",
    "COLLATERAL_HARM",
)

# Thresholds (LOCKED)
FO_CORE_KILL_MIN = 40  # of 45
HN_CORE_MIN = 0.80
COLLATERAL_DROP = 0.05
CITE_30_FO = 45
CITE_30_HN = 0.06666666666666667
CITE_30_K16 = 0.9875
CITE_30_OV = 0.5104166666666666
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


def out_degree(n: int, edges: Sequence[tuple[int, int]], s: int) -> int:
    """Directed out-degree of s (local incidence; no BFS)."""
    if not (0 <= s < n):
        raise ValueError(f"s={s} out of range for n={n}")
    return sum(1 for u, _v in edges if int(u) == s)


def out_neighborhood_size(n: int, edges: Sequence[tuple[int, int]], s: int) -> int:
    """|out-neighborhood(s)| — equals outdeg for simple digraphs (no multi-edges)."""
    adj = adjacency_list(n, edges)
    return len(set(adj[s]))


def gate_trigger_outdeg0(row: dict[str, Any]) -> bool:
    """Primary sound gate: outdeg(s)==0 and t!=s."""
    n, edges, s, t = parse_instance(str(row["encoding"]))
    n = int(row.get("n", n))
    s = int(row.get("s", s))
    t = int(row.get("t", t))
    return out_degree(n, edges, s) == 0 and t != s


def apply_outdeg_gate_preds(
    ens_preds: Any,
    rows: Sequence[dict[str, Any]],
) -> tuple[Any, list[int]]:
    """Force ens pred → 0 where outdeg(s)==0 and t≠s. Returns (gated, trigger_ids)."""
    import torch

    gated = ens_preds.clone()
    triggers: list[int] = []
    for i, row in enumerate(rows):
        if gate_trigger_outdeg0(row):
            gated[i] = 0
            triggers.append(i)
    return gated, triggers


def decide_verdict(
    *,
    fo_killed: int,
    fo_total: int,
    gated_hn: float,
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
            "gated_hn": gated_hn,
        }
    if fo_killed <= 0:
        return {
            "verdict": "FO_UNMOVED",
            "reasons": ["zero_FO_eliminated"],
            "fo_killed": fo_killed,
            "fo_total": fo_total,
            "gated_hn": gated_hn,
        }
    if fo_killed >= FO_CORE_KILL_MIN and gated_hn >= HN_CORE_MIN:
        return {
            "verdict": "FO_CORE_KILLED",
            "reasons": [
                f"fo_killed={fo_killed}>={FO_CORE_KILL_MIN}",
                f"gated_hn={gated_hn:.4f}>={HN_CORE_MIN}",
            ],
            "fo_killed": fo_killed,
            "fo_total": fo_total,
            "gated_hn": gated_hn,
        }
    return {
        "verdict": "FO_PARTIAL",
        "reasons": [
            f"fo_killed={fo_killed}/{fo_total}",
            f"gated_hn={gated_hn:.4f}",
            "not_FO_CORE_KILLED",
        ],
        "fo_killed": fo_killed,
        "fo_total": fo_total,
        "gated_hn": gated_hn,
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
    """Acc + fail-mode + CD for one pred vector (baseline or gated)."""
    import torch

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
    # CD uses member hard preds (unchanged by gate) + ens metrics context
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
    hops: list[int],
    pair_per: Any,
    ens_max_prob: Any,
) -> list[int]:
    """Baseline FAIL_OPEN example indices (HN expected)."""
    wrong = ens_preds != labels
    ids: list[int] = []
    import torch

    for i in wrong.nonzero(as_tuple=False).view(-1).tolist():
        d_ex = float(pair_per[i].item())
        conf = float(ens_max_prob[i].item())
        if d_ex == 0.0 and conf >= CONF_THRESH:
            ids.append(int(i))
    return ids


def run_cycle(
    *,
    hops_data: Path = DEFAULT_HOPS,
    matched_data: Path = DEFAULT_MATCHED,
    out_path: Path = DEFAULT_OUT,
    cite30_path: Path = DEFAULT_CITE30,
    cite31_path: Path = DEFAULT_CITE31,
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
    if cite30_path.exists():
        cite30 = json.loads(cite30_path.read_text())
    if cite31_path.exists():
        cite31 = json.loads(cite31_path.read_text())

    # Prefer FO ids from #31 autopsy when present
    cite31_fo: Optional[list[int]] = None
    if cite31 is not None:
        cite31_fo = list(
            cite31.get("example_id_list_FO_HN")
            or cite31.get("cohorts", {}).get("FO_HN", {}).get("ids")
            or []
        )

    print(
        f"[stalk-sound-outdeg] loading #14/#18/#22 ens n={len(ensemble_seeds)}",
        file=sys.stderr,
    )
    ens_models, ens_meta = _load_ens(ensemble_seeds, max_nodes)

    print(
        f"[stalk-sound-outdeg] infer ood_hops T{FOCUS_T} (#22 baseline)",
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

    fo_ids = _fo_ids_from_fail(ens_preds, labels, hops, pair_per, ens_max_prob)
    if cite31_fo is not None and len(cite31_fo) == CITE_30_FO:
        # Prefer sealed autopsy ids when they match count
        if set(fo_ids) == set(cite31_fo):
            fo_ids = list(cite31_fo)
        else:
            # still use live fo_ids but record mismatch
            pass

    # Local incidence features (cheap; no BFS in gate)
    outdeg_s = []
    out_nei = []
    triggers_preview = []
    for i, row in enumerate(rows):
        n, edges, s, t = parse_instance(str(row["encoding"]))
        n = int(row.get("n", n))
        s = int(row.get("s", s))
        t = int(row.get("t", t))
        od = out_degree(n, edges, s)
        on = out_neighborhood_size(n, edges, s)
        outdeg_s.append(od)
        out_nei.append(on)
        if od == 0 and t != s:
            triggers_preview.append(i)
    # Equivalence check arm(2)==arm(1)
    nei_eq = all(outdeg_s[i] == out_nei[i] for i in range(len(rows)))

    gated_preds, trigger_ids = apply_outdeg_gate_preds(ens_preds, rows)
    assert trigger_ids == triggers_preview

    gated = _eval_slice_metrics(
        gated_preds,
        labels,
        hops,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hard=hard,
        seeds=list(ensemble_seeds),
    )

    # FO killed: baseline FO that become correct under gate
    fo_killed_ids = [
        i for i in fo_ids if int(gated_preds[i].item()) == int(labels[i].item())
    ]
    fo_killed = len(fo_killed_ids)
    fo_still_open = [
        i
        for i in fo_ids
        if int(gated_preds[i].item()) != int(labels[i].item())
    ]

    # Collateral on ood_hops: OK_HN index-set + positives
    correct_mask = ens_preds == labels
    ok_hn_ids = [
        i
        for i in range(len(hops))
        if bool(correct_mask[i].item()) and hops[i] == int(HOP_UNREACHABLE)
    ]
    pos_ids = [i for i, h in enumerate(hops) if h != int(HOP_UNREACHABLE) and int(labels[i].item()) == 1]
    # also all y=1
    pos_ids = [i for i in range(len(rows)) if int(labels[i].item()) == 1]

    def _acc_on(ids: list[int], preds: Any) -> float:
        if not ids:
            return float("nan")
        idx = torch.tensor(ids, dtype=torch.long)
        return float((preds[idx] == labels[idx]).float().mean().item())

    ok_hn_base = _acc_on(ok_hn_ids, ens_preds)
    ok_hn_gate = _acc_on(ok_hn_ids, gated_preds)
    pos_base = _acc_on(pos_ids, ens_preds)
    pos_gate = _acc_on(pos_ids, gated_preds)

    collateral_reasons: list[str] = []
    if ok_hn_ids and (ok_hn_base - ok_hn_gate) >= COLLATERAL_DROP:
        collateral_reasons.append(
            f"OK_HN_index_drop={ok_hn_base - ok_hn_gate:.4f}>={COLLATERAL_DROP}"
        )
    if pos_ids and (pos_base - pos_gate) >= COLLATERAL_DROP:
        collateral_reasons.append(
            f"positives_drop={pos_base - pos_gate:.4f}>={COLLATERAL_DROP}"
        )

    # Matched-OOD collateral (cheap — same frozen ens)
    matched_block: dict[str, Any] = {"skipped": True}
    if not skip_matched and matched_data.exists():
        print(
            f"[stalk-sound-outdeg] infer matched-OOD T{FOCUS_T} collateral",
            file=sys.stderr,
        )
        mrows = load_jsonl(matched_data)
        m_max = max(max_nodes, max(int(r["n"]) for r in mrows))
        if m_max > max_nodes:
            # reload ens with larger max_nodes if needed
            ens_models, ens_meta = _load_ens(ensemble_seeds, m_max)
            max_nodes = m_max
        m_logits, m_labels, m_hops = _collect_member_logits(
            ens_models, mrows, T=FOCUS_T, max_nodes=max_nodes
        )
        m_ens = _aggregate_preds(m_logits, method=PRIMARY_AGG)
        m_gated, m_triggers = apply_outdeg_gate_preds(m_ens, mrows)
        m_base_m = _metrics_ext(m_ens, m_labels, m_hops)
        m_gate_m = _metrics_ext(m_gated, m_labels, m_hops)
        deltas = {
            "overall_acc": float(m_gate_m["overall_acc"] - m_base_m["overall_acc"]),
            "hard_neg_acc": float(
                m_gate_m["hard_neg_acc"] - m_base_m["hard_neg_acc"]
            ),
            "K16": float(m_gate_m["K16"] - m_base_m["K16"]),
        }
        for k, d in deltas.items():
            if d <= -COLLATERAL_DROP:
                collateral_reasons.append(
                    f"matched_ood_{k}_drop={-d:.4f}>={COLLATERAL_DROP}"
                )
        matched_block = {
            "skipped": False,
            "path": str(matched_data),
            "n": len(mrows),
            "n_gate_triggers": len(m_triggers),
            "baseline": m_base_m,
            "gated": m_gate_m,
            "deltas": deltas,
        }

    collateral_harm = bool(collateral_reasons)
    gated_hn = float(gated["ensemble"]["hard_neg_acc"])
    decision = decide_verdict(
        fo_killed=fo_killed,
        fo_total=len(fo_ids),
        gated_hn=gated_hn,
        collateral_harm=collateral_harm,
        collateral_reasons=collateral_reasons,
    )
    verdict = decision["verdict"]

    # Residue update
    if verdict == "FO_CORE_KILLED":
        residue = "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_KILLED"
        residue_note = (
            "Sound outdeg(s)==0 gate eliminated ≥40/45 FO and HN≥0.80; "
            "still MEASURE; §22 not widened; hop-OOD not OPEN."
        )
    elif verdict == "FO_PARTIAL":
        residue = "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL"
        residue_note = (
            f"Sound outdeg gate killed {fo_killed}/{len(fo_ids)} FO; "
            "residue remains; no §22 widen."
        )
    elif verdict == "FO_UNMOVED":
        residue = "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_UNMOVED"
        residue_note = "outdeg(s)==0 gate did not eliminate FO core; unpaid."
    else:
        residue = "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_COLLATERAL"
        residue_note = (
            "outdeg gate caused ≥0.05 abs collateral on matched-OOD/OK/positives; "
            "STOP this overlay only; do not park whole corridor unless forced."
        )

    # Replicate baseline vs #30
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
    }

    # Soundness audit: every trigger must have y=0
    soundness_violations = []
    for i in trigger_ids:
        if int(labels[i].item()) != 0:
            soundness_violations.append(i)

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
            "kind": "eval_only_sound_outdeg_gate_overlay",
            "train": False,
            "bag_noise_trained": False,
            "select_reopened": False,
            "curriculum_reopened": False,
            "bfs_oracle_gate": False,
            "d": DEFAULT_D,
            "aggregator_primary": PRIMARY_AGG,
            "T_fixed": FOCUS_T,
            "hard_A": True,
            "broadcast_c": False,
            "gate_primary": "outdeg(s)==0 and t!=s → force pred=0",
            "gate_optional_arm2": "|out-neighborhood(s)|==0 (identical on simple digraphs)",
        },
        "datasets": {
            "ood_hops": str(hops_data),
            "n_ood_hops": len(rows),
            "matched_ood": str(matched_data) if matched_data.exists() else None,
            "cite30": str(cite30_path) if cite30_path.exists() else None,
            "cite31": str(cite31_path) if cite31_path.exists() else None,
        },
        "members_pr22": ens_meta,
        "prereg": {
            "verdicts": list(VERDICTS),
            "FO_CORE_KILL_MIN": FO_CORE_KILL_MIN,
            "HN_CORE_MIN": HN_CORE_MIN,
            "COLLATERAL_DROP": COLLATERAL_DROP,
            "CONF_THRESH": CONF_THRESH,
            "D_CLOSED_MIN": D_CLOSED_MIN,
            "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
            "non_goals": [
                "no_train",
                "no_science_open_widen",
                "section22_unchanged",
                "T_fixed_16",
                "no_bfs_oracle_gate",
                "no_hop_ood_open_claim",
            ],
        },
        "outdeg_def": {
            "formula": "outdeg(s)=|{v:(s,v) in A_hat}|",
            "matches_autopsy_directed_out_edges": True,
            "not_total_degree": True,
            "not_bfs_out_closure": True,
            "out_neighborhood_equiv_outdeg_on_simple": nei_eq,
        },
        "replicate_cite30": replicate,
        "arms": {
            "0_baseline": {
                "name": "baseline_#22_prob_mean",
                "ensemble": baseline["ensemble"],
                "fail_mode": {
                    k: baseline["fail_mode"][k]
                    for k in (
                        "FAIL_OPEN",
                        "FAIL_CLOSED",
                        "FAIL_AMBIG",
                        "n_wrong",
                        "verdict",
                        "ens_overall_acc",
                    )
                    if k in baseline["fail_mode"]
                },
                "CD": baseline["competent_dissonance"],
                "D_hard": baseline["D_hard"],
            },
            "1_outdeg0_force_unreach": {
                "name": "outdeg(s)==0_force_pred0",
                "ensemble": gated["ensemble"],
                "fail_mode": {
                    k: gated["fail_mode"][k]
                    for k in (
                        "FAIL_OPEN",
                        "FAIL_CLOSED",
                        "FAIL_AMBIG",
                        "n_wrong",
                        "verdict",
                        "ens_overall_acc",
                    )
                    if k in gated["fail_mode"]
                },
                "CD": gated["competent_dissonance"],
                "D_hard": gated["D_hard"],
                "n_triggers": len(trigger_ids),
                "trigger_ids_sample": trigger_ids[:40],
            },
        },
        "fo_analysis": {
            "fo_total": len(fo_ids),
            "fo_ids": fo_ids,
            "fo_killed": fo_killed,
            "fo_killed_ids": fo_killed_ids,
            "fo_still_wrong": len(fo_still_open),
            "fo_still_wrong_ids": fo_still_open,
            "cite31_fo_ids": cite31_fo,
            "cite31_fo_match": (
                set(fo_ids) == set(cite31_fo) if cite31_fo is not None else None
            ),
            "fo_with_outdeg0": sum(1 for i in fo_ids if i in set(trigger_ids)),
        },
        "collateral": {
            "OK_HN": {
                "n": len(ok_hn_ids),
                "baseline_acc": ok_hn_base,
                "gated_acc": ok_hn_gate,
                "delta": (
                    float(ok_hn_gate - ok_hn_base) if ok_hn_ids else float("nan")
                ),
            },
            "positives": {
                "n": len(pos_ids),
                "baseline_acc": pos_base,
                "gated_acc": pos_gate,
                "delta": float(pos_gate - pos_base) if pos_ids else float("nan"),
            },
            "matched_ood_T16": matched_block,
            "harm": collateral_harm,
            "reasons": collateral_reasons,
        },
        "soundness": {
            "n_triggers": len(trigger_ids),
            "n_violations_y_neq_0": len(soundness_violations),
            "violation_ids": soundness_violations,
            "note": "outdeg(s)==0 ∧ t≠s ⇒ gold y=0 on digraphs without self-loops",
        },
        "comparison_table": {
            "baseline": {
                "HN": b_ens["hard_neg_acc"],
                "K16": b_ens["K16"],
                "overall": b_ens["overall_acc"],
                "FO": baseline["fail_mode"]["FAIL_OPEN"],
                "FC": baseline["fail_mode"]["FAIL_CLOSED"],
            },
            "gated": {
                "HN": gated["ensemble"]["hard_neg_acc"],
                "K16": gated["ensemble"]["K16"],
                "overall": gated["ensemble"]["overall_acc"],
                "FO": gated["fail_mode"]["FAIL_OPEN"],
                "FC": gated["fail_mode"]["FAIL_CLOSED"],
                "FO_killed_of_45": fo_killed,
            },
        },
        "decision": decision,
        "verdict": verdict,
        "residue": residue,
        "residue_note": residue_note,
        "reading": (
            "Sound local outdeg(s)==0 force-unreach overlay on #22 ens. "
            "MEASURE only; science_open=false; §22 unchanged; no hop-OOD OPEN claim."
        ),
        "elapsed_sec": elapsed,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-sound-outdeg] verdict={verdict} residue={residue} "
        f"FO_killed={fo_killed}/{len(fo_ids)} "
        f"HN {b_ens['hard_neg_acc']:.3f}→{gated_hn:.3f} "
        f"triggers={len(trigger_ids)} "
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
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_ENSEMBLE_SEEDS))
    p.add_argument("--skip-matched", action="store_true")
    args = p.parse_args(argv)
    run_cycle(
        hops_data=args.hops_data,
        matched_data=args.matched_data,
        out_path=args.out,
        cite30_path=args.cite30,
        cite31_path=args.cite31,
        ensemble_seeds=tuple(args.seeds),
        skip_matched=args.skip_matched,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
