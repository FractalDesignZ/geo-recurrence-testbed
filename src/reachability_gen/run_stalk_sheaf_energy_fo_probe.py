"""CYCLE_STALK_SHEAF_ENERGY_FO_PROBE — MEASURE Dirichlet/coboundary FO probe.

Eval-only @ T=16 on data/ood_hops.jsonl (+ matched-OOD τ + collateral) with
frozen #14/#18/#22 ens. Identity-restriction coboundary energy on sealed
stalk final_states (Kant/sheaf A2/A4 instrumented; NO sheaf train; NO
restriction-map learning). science_open=false (not widened; §22 unchanged).

Usage::

    python -m reachability_gen.run_stalk_sheaf_energy_fo_probe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.encode import parse_instance
from reachability_gen.models.fractal_core import build_node_slot_batch
from reachability_gen.orientation import auroc_binary, summary_numeric
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_stalk_hop_ood_hn import FOCUS_T, _metrics_ext
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    _aggregate_preds,
    _ckpt_for_seed,
    _collect_member_logits,
    _load_model_from_ckpt,
)
from reachability_gen.sheaf_energy import (
    QUANTILE_TAU,
    decide_verdict,
    energy_refuse_preds,
    example_sheaf_energy_from_states,
    mean_metrics,
    pearson_corr,
    quantile,
    sheaf_energy_definition_doc,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_SHEAF_ENERGY_FO_PROBE"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_MATCHED = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_sheaf_energy_fo_probe.json")
DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE32 = Path("artifacts/stalk_sound_outdeg_gate.json")
DEFAULT_CITE33 = Path("artifacts/stalk_hn_fo_remainder_autopsy.json")
DEFAULT_CITE35 = Path("artifacts/stalk_reach_certificates.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))

VERDICTS = (
    "ENERGY_TRACKS_CERT",
    "ENERGY_PARTIAL",
    "ENERGY_NULL",
    "COLLATERAL_HARM",
)

COLLATERAL_DROP = 0.05
AUROC_SEP = 0.75
AUROC_PARTIAL = 0.60
CITE_30_FO = 45
CITE_32_REM = 22
CITE_TOL = 0.02
CITE_30_HN = 0.06666666666666667
CITE_30_K16 = 0.9875


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


def _edges_of(ex: dict[str, Any]) -> tuple[int, list[tuple[int, int]], int, int]:
    enc = ex.get("encoding")
    if enc:
        n, edges, s, t = parse_instance(str(enc))
    else:
        n = int(ex["n"])
        edges = [(int(a), int(b)) for a, b in ex["edges"]]
        s, t = int(ex["s"]), int(ex["t"])
    n = int(ex.get("n", n))
    s = int(ex.get("s", s))
    t = int(ex.get("t", t))
    return n, [(int(a), int(b)) for a, b in edges], s, t


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


def _collect_sheaf_energy(
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    batch_size: int = 32,
) -> list[dict[str, float]]:
    """Per-example ens-mean sheaf energy metrics (Id coboundary + scalars)."""
    import torch

    for m in models:
        m.eval()
    out: list[Optional[dict[str, float]]] = [None] * len(rows)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            member_lists: list[list[dict[str, float]]] = [
                [] for _ in range(len(batch_rows))
            ]
            for model in models:
                logits, _, info = model(
                    batch["node_ids"],
                    batch["node_mask"],
                    batch["attn_mask"],
                    batch["s_idx"],
                    batch["t_idx"],
                    return_halt=True,
                    return_states=True,
                    T=T,
                    adaptive_halt=False,
                )
                del logits
                assert info is not None and "final_states" in info
                states = info["final_states"]  # [B, M, d]
                for bi, ex in enumerate(batch_rows):
                    n, edges, s, t = _edges_of(ex)
                    H = states[bi]
                    member_lists[bi].append(
                        example_sheaf_energy_from_states(H, n, edges, s, t)
                    )
            for bi in range(len(batch_rows)):
                out[start + bi] = mean_metrics(member_lists[bi])
    assert all(x is not None for x in out)
    return [x for x in out if x is not None]  # type: ignore[misc]


def _stratum_table(
    metrics: list[dict[str, float]],
    ids: list[int],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    block: dict[str, Any] = {"n": len(ids), "ids_head": ids[:8], "numeric": {}}
    for k in keys:
        xs = [float(metrics[i][k]) for i in ids if i < len(metrics)]
        block["numeric"][k] = summary_numeric(xs)
    return block


def run_cycle(
    *,
    hops_data: Path = DEFAULT_HOPS,
    matched_data: Path = DEFAULT_MATCHED,
    out_path: Path = DEFAULT_OUT,
    cite30_path: Path = DEFAULT_CITE30,
    cite31_path: Path = DEFAULT_CITE31,
    cite32_path: Path = DEFAULT_CITE32,
    cite33_path: Path = DEFAULT_CITE33,
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

    cite30 = json.loads(cite30_path.read_text()) if cite30_path.exists() else {}
    cite31 = json.loads(cite31_path.read_text()) if cite31_path.exists() else {}
    cite32 = json.loads(cite32_path.read_text()) if cite32_path.exists() else {}
    cite33 = json.loads(cite33_path.read_text()) if cite33_path.exists() else {}
    cite35 = json.loads(cite35_path.read_text()) if cite35_path.exists() else {}

    fo_ids = list(
        cite31.get("example_id_list_FO_HN")
        or cite32.get("fo_analysis", {}).get("fo_ids")
        or []
    )
    rem_ids = list(
        cite33.get("cohorts", {}).get("FO_REMAINDER", {}).get("ids")
        or cite32.get("fo_analysis", {}).get("fo_still_wrong_ids")
        or []
    )
    killed_ids = list(
        cite33.get("cohorts", {}).get("FO_KILLED", {}).get("ids")
        or cite32.get("fo_analysis", {}).get("fo_killed_ids")
        or []
    )
    ok_ids = list(cite33.get("cohorts", {}).get("OK_HN", {}).get("ids") or [])
    fc_ids = list(cite33.get("cohorts", {}).get("FC_HN", {}).get("ids") or [])

    print(f"[{CYCLE}] loading ens seeds={list(ensemble_seeds)} …", flush=True)
    ens_models, ens_meta = _load_ens(ensemble_seeds, max_nodes)

    print(f"[{CYCLE}] collecting member logits @ T={FOCUS_T} …", flush=True)
    logits, labels, hops = _collect_member_logits(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )
    ens_preds = _aggregate_preds(logits, method=PRIMARY_AGG)
    baseline_m = _metrics_ext(ens_preds, labels, hops)

    print(f"[{CYCLE}] collecting sheaf energies …", flush=True)
    metrics = _collect_sheaf_energy(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )
    e_cob = [float(m["E_cob"]) for m in metrics]
    e_dir = [float(m["E_dir"]) for m in metrics]
    e_align = [float(m["E_align"]) for m in metrics]

    keys = (
        "E_cob",
        "E_dir",
        "E_align",
        "E_cob_raw",
        "mass_mean",
        "mass_t",
        "align_mean",
        "n_edges",
    )
    strata = {
        "OK_HN": ok_ids,
        "FO_HN": fo_ids,
        "FO_REMAINDER": rem_ids,
        "FO_KILLED": killed_ids,
        "FC_HN": fc_ids,
    }
    tables = {name: _stratum_table(metrics, ids, keys) for name, ids in strata.items()}

    # AUROC: higher E_cob → FO
    neg_ids = sorted(set(ok_ids) | set(fc_ids))
    scores_fo = [e_cob[i] for i in fo_ids if i < len(e_cob)]
    scores_neg = [e_cob[i] for i in neg_ids if i < len(e_cob)]
    scores_ok = [e_cob[i] for i in ok_ids if i < len(e_cob)]
    scores_rem = [e_cob[i] for i in rem_ids if i < len(e_cob)]
    auroc_fo_neg = auroc_binary(
        scores_fo + scores_neg, [1] * len(scores_fo) + [0] * len(scores_neg)
    )
    auroc_fo_ok = auroc_binary(
        scores_fo + scores_ok, [1] * len(scores_fo) + [0] * len(scores_ok)
    )
    auroc_rem_neg = auroc_binary(
        scores_rem + scores_neg, [1] * len(scores_rem) + [0] * len(scores_neg)
    )

    # Matched-OOD: calibrate τ then collateral under energy_refuse
    matched_block: dict[str, Any] = {"skipped": True}
    tau = float("nan")
    collateral_reasons: list[str] = []
    refuse_preds = ens_preds.clone()
    refused_mask: list[bool] = [False] * len(rows)

    if not skip_matched and matched_data.exists():
        print(
            f"[{CYCLE}] matched-OOD τ calibrate + collateral @ T={FOCUS_T} …",
            flush=True,
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
        m_base_m = _metrics_ext(m_ens, m_labels, m_hops)
        m_metrics = _collect_sheaf_energy(
            ens_models, mrows, T=FOCUS_T, max_nodes=max_nodes
        )
        m_ecob = [float(m["E_cob"]) for m in m_metrics]
        tau = quantile(m_ecob, QUANTILE_TAU)
        m_refuse, m_refused = energy_refuse_preds(m_ens, m_ecob, tau=tau)
        m_ref_m = _metrics_ext(m_refuse, m_labels, m_hops)
        d_matched = {
            "overall_acc": float(m_ref_m["overall_acc"] - m_base_m["overall_acc"]),
            "hard_neg_acc": float(
                m_ref_m["hard_neg_acc"] - m_base_m["hard_neg_acc"]
            ),
            "K16": float(m_ref_m["K16"] - m_base_m["K16"]),
        }
        for k, d in d_matched.items():
            if d <= -COLLATERAL_DROP:
                collateral_reasons.append(
                    f"matched_ood_energy_refuse_{k}_drop={-d:.4f}>={COLLATERAL_DROP}"
                )
        matched_block = {
            "skipped": False,
            "path": str(matched_data),
            "n": len(mrows),
            "tau_quantile": QUANTILE_TAU,
            "tau": tau,
            "baseline": m_base_m,
            "energy_refuse": m_ref_m,
            "deltas_energy_refuse": d_matched,
            "n_refused": int(sum(m_refused)),
            "E_cob_summary": summary_numeric(m_ecob),
        }
        # Apply same τ on hop-OOD
        refuse_preds, refused_mask = energy_refuse_preds(ens_preds, e_cob, tau=tau)
    else:
        # Fallback: calibrate τ on hop-OOD non-FO (document; weaker)
        non_fo = [i for i in range(len(e_cob)) if i not in set(fo_ids)]
        tau = quantile([e_cob[i] for i in non_fo], QUANTILE_TAU)
        refuse_preds, refused_mask = energy_refuse_preds(ens_preds, e_cob, tau=tau)
        collateral_reasons.append("matched_ood_skipped_tau_from_hops_nonFO")

    refuse_m = _metrics_ext(refuse_preds, labels, hops)

    def _killed(ids: list[int], preds: Any) -> list[int]:
        return [
            i
            for i in ids
            if i < len(preds) and int(preds[i].item()) == int(labels[i].item())
        ]

    fo_killed = _killed(fo_ids, refuse_preds)
    rem_killed = _killed(rem_ids, refuse_preds)

    # Concentration of high energy / refuse on FO
    fo_set = set(fo_ids)
    rem_set = set(rem_ids)
    high_ids = [i for i, flag in enumerate(refused_mask) if flag]
    fo_high = [i for i in high_ids if i in fo_set]
    rem_high = [i for i in high_ids if i in rem_set]
    concentration = {
        "tau": tau,
        "n_refused": len(high_ids),
        "FO_cap_refused_over_FO": (
            len(fo_high) / len(fo_ids) if fo_ids else float("nan")
        ),
        "refused_cap_FO_over_refused": (
            len(fo_high) / len(high_ids) if high_ids else float("nan")
        ),
        "FO_intersect_refused": len(fo_high),
        "rem22_intersect_refused": len(rem_high),
        "refused_intersect_FO": len(fo_high),
        "refused_intersect_rem22": len(rem_high),
    }

    # Compare to #35 dirty concentration (cite)
    cert_conc = cite35.get("concentration") or {}
    dirty_n = int(cert_conc.get("dirty_n") or 0)
    dirty_fo = int(cert_conc.get("dirty_intersect_FO") or 0)
    # Proxy: score dirty flag as 1 on FO (all FO dirty under #35) vs E_cob
    # Use FO label as dirty proxy for correlation on strata FO∪OK∪FC
    corr_ids = sorted(set(fo_ids) | set(ok_ids) | set(fc_ids))
    dirty_proxy = [1.0 if i in fo_set else 0.0 for i in corr_ids]
    e_corr = [e_cob[i] for i in corr_ids if i < len(e_cob)]
    # align lengths
    dirty_proxy = dirty_proxy[: len(e_corr)]
    r_e_dirty = pearson_corr(e_corr, dirty_proxy)

    collateral_harm = bool(
        any("matched_ood_energy_refuse" in r for r in collateral_reasons)
    )
    # Also flag large hop-OOD K16 drop as harm signal on refuse
    d_hops = {
        "overall_acc": float(refuse_m["overall_acc"] - baseline_m["overall_acc"]),
        "hard_neg_acc": float(
            refuse_m["hard_neg_acc"] - baseline_m["hard_neg_acc"]
        ),
        "K16": float(refuse_m["K16"] - baseline_m["K16"]),
    }

    decision = decide_verdict(
        auroc_fo=auroc_fo_neg,
        fo_killed=len(fo_killed),
        fo_total=len(fo_ids),
        rem22_killed=len(rem_killed),
        rem22_total=len(rem_ids),
        collateral_harm=collateral_harm,
        collateral_reasons=collateral_reasons,
        auroc_sep=AUROC_SEP,
        auroc_partial=AUROC_PARTIAL,
    )

    cite_ok = (
        len(fo_ids) == CITE_30_FO
        and len(rem_ids) == CITE_32_REM
        and abs(float(baseline_m["hard_neg_acc"]) - CITE_30_HN) <= CITE_TOL
        and abs(float(baseline_m["K16"]) - CITE_30_K16) <= CITE_TOL
    )

    contrast = {}
    for k in ("E_cob", "E_dir", "E_align"):
        contrast[k] = {
            name: tables[name]["numeric"][k]["median"] for name in strata
        }

    auroc_table = {
        "FO_HN_vs_OK_FC": {"E_cob": auroc_fo_neg},
        "FO_HN_vs_OK_HN": {"E_cob": auroc_fo_ok},
        "rem22_vs_OK_FC": {"E_cob": auroc_rem_neg},
    }

    elapsed = time.time() - t0
    verdict = str(decision["verdict"])
    reading_parts = [
        f"verdict={verdict}",
        f"AUROC(E_cob→FO vs OK∪FC)={auroc_fo_neg}",
        f"energy_refuse FO_killed={len(fo_killed)}/{len(fo_ids)}",
        f"rem22={len(rem_killed)}/{len(rem_ids)}",
        f"tau={tau}",
        f"matched_deltas={matched_block.get('deltas_energy_refuse')}",
        f"r(E_cob, dirty_proxy)={r_e_dirty}",
        "science_open=false; §22 unchanged; no sheaf train.",
    ]

    artifact: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "section22_unchanged": True,
        "sheaf_unsupervised_opened": False,
        "restriction_map_learning": False,
        "train": False,
        "T_fixed": FOCUS_T,
        "base_sha_note": "main after PR #39",
        "git_sha": _git_sha(),
        "elapsed_sec": elapsed,
        "prereg": {
            "energy": sheaf_energy_definition_doc(),
            "verdicts": list(VERDICTS),
            "arms": ["0_baseline", "1_energy_refuse"],
            "cert_reference": "#35 CERT_FO_CATCH",
            "tau_quantile": QUANTILE_TAU,
            "collateral_drop": COLLATERAL_DROP,
            "auroc_sep": AUROC_SEP,
            "auroc_partial": AUROC_PARTIAL,
        },
        "datasets": {
            "ood_hops": str(hops_data),
            "matched_ood": str(matched_data),
            "n_hops": len(rows),
        },
        "members_pr22": ens_meta,
        "architecture": {
            "model": "FractalCore",
            "agg": PRIMARY_AGG,
            "states": "final_states identity-restriction coboundary",
        },
        "energy_formula": sheaf_energy_definition_doc()["primary"]["formula"],
        "replicate_cite30": {
            "FO_n": len(fo_ids),
            "FO_match": len(fo_ids) == CITE_30_FO,
            "rem22_n": len(rem_ids),
            "rem22_match": len(rem_ids) == CITE_32_REM,
            "baseline_HN": baseline_m["hard_neg_acc"],
            "baseline_K16": baseline_m["K16"],
            "baseline_overall": baseline_m["overall_acc"],
            "cite_ok": cite_ok,
        },
        "arms": {
            "0_baseline": {
                "metrics": baseline_m,
                "FAIL_OPEN_cite": len(fo_ids),
            },
            "1_energy_refuse": {
                "metrics": refuse_m,
                "tau": tau,
                "n_refused": int(sum(refused_mask)),
                "fo_killed": len(fo_killed),
                "fo_killed_ids_head": fo_killed[:16],
                "rem22_killed": len(rem_killed),
                "rem22_killed_ids_head": rem_killed[:16],
                "deltas_vs_baseline": d_hops,
            },
        },
        "strata_tables": tables,
        "contrast_medians": contrast,
        "auroc_table": auroc_table,
        "concentration": concentration,
        "cert_dirty_compare": {
            "cite35_verdict": cite35.get("verdict"),
            "cite35_dirty_n": dirty_n,
            "cite35_dirty_intersect_FO": dirty_fo,
            "cite35_FO_cap_dirty": cert_conc.get("FO_cap_dirty_over_FO"),
            "pearson_E_cob_vs_FO_proxy": r_e_dirty,
            "energy_FO_cap_refused": concentration["FO_cap_refused_over_FO"],
        },
        "matched_ood": matched_block,
        "collateral": {
            "harm": collateral_harm,
            "reasons": collateral_reasons,
            "drop_threshold": COLLATERAL_DROP,
        },
        "decision": decision,
        "verdict": verdict,
        "open_status": {
            "science_open": False,
            "section22_widened": False,
            "sheaf_unsupervised": False,
            "hop_ood_open_claimed": False,
        },
        "residue": (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_WALL/CERT_FO_CATCH/" + verdict
        ),
        "hop_ood_remains_measure_residue": True,
        "reading": " ".join(reading_parts),
        "E_cob_global_summary": summary_numeric(e_cob),
        "E_dir_global_summary": summary_numeric(e_dir),
        "E_align_global_summary": summary_numeric(e_align),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"[{CYCLE}] wrote {out_path} verdict={verdict} "
        f"AUROC={auroc_fo_neg} FO_killed={len(fo_killed)}/{len(fo_ids)} "
        f"rem22={len(rem_killed)}/{len(rem_ids)} tau={tau} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return artifact


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=CYCLE)
    p.add_argument("--hops", type=Path, default=DEFAULT_HOPS)
    p.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-matched", action="store_true")
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_ENSEMBLE_SEEDS),
    )
    args = p.parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    art = run_cycle(
        hops_data=args.hops,
        matched_data=args.matched,
        out_path=args.out,
        ensemble_seeds=seeds,
        skip_matched=bool(args.skip_matched),
    )
    return 0 if art.get("science_open") is False else 1


if __name__ == "__main__":
    raise SystemExit(main())
