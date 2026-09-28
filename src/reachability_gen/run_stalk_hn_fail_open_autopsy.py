"""CYCLE_STALK_HN_FAIL_OPEN_AUTOPSY — MEASURE structural autopsy of HN FAIL_OPEN core.

Eval-only @ T=16 on data/ood_hops.jsonl with frozen #14/#18/#22 ens.
Autopsy the 45 FAIL_OPEN hard-neg examples named HN_FAIL_OPEN_CORE in #30.
No train. science_open=false (not widened; §22 unchanged).

Usage::

    python -m reachability_gen.run_stalk_hn_fail_open_autopsy
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import HOP_UNREACHABLE
from reachability_gen.encode import parse_instance
from reachability_gen.graph import adjacency_list, hop_distances_from, reachable_out_set
from reachability_gen.hard_negatives import total_degrees
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_fractal_core_gate1 import DEFAULT_D
from reachability_gen.run_stalk_epistemic_disagreement import (
    _entropy_nats,
    _pairwise_disagreement_per_example,
)
from reachability_gen.run_stalk_hop_ood_hn import (
    CONF_THRESH,
    D_CLOSED_MIN,
    EPI_CLOSED_MIN,
    FOCUS_T,
    _fail_mode_counts,
)
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    _aggregate_preds,
    _ckpt_for_seed,
    _collect_member_logits,
    _load_model_from_ckpt,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID, split_encoding_tokens

CYCLE = "CYCLE_STALK_HN_FAIL_OPEN_AUTOPSY"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))

# Autopsy verdicts (LOCKED)
VERDICTS = (
    "STRUCTURAL_CLUSTER",
    "DIFFUSE",
    "DATA_ARTIFACT",
    "INCONCLUSIVE",
)

# Member-logit tags (LOCKED)
MEMBER_TAGS = ("HARD_UNANIMOUS", "SOFT_AGREE", "MIXED_PRED")

# Cite #30 scalars (replicate FO count)
CITE_30_FO = 45
CITE_30_FC = 190
CITE_30_HN = 0.06666666666666667


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


def _r_in_set(n: int, edges: Sequence[tuple[int, int]], t: int) -> set[int]:
    """Nodes that can reach t (includes t): BFS on reversed digraph."""
    radj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        radj[v].append(u)
    from collections import deque

    seen: set[int] = {t}
    q: deque[int] = deque([t])
    while q:
        u = q.popleft()
        for v in radj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def _frontier_exits(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
) -> int:
    """Count nodes u in R_out(s) with ≥1 out-edge leaving R_out (distractor leak)."""
    r_out = reachable_out_set(n, edges, s)
    adj = adjacency_list(n, edges)
    count = 0
    for u in r_out:
        for v in adj[u]:
            if v not in r_out:
                count += 1
                break
    return count


def _near_miss_bridges(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
) -> int:
    """Count missing one-edge bridges: pairs (u in R_out, v in R_in\\R_out) with no u→v.

    For true hard-negs, existing u→v with v in R_in would imply reachability;
    so this counts *absent* bridges that would connect s-component to t-component.
    """
    r_out = reachable_out_set(n, edges, s)
    r_in = _r_in_set(n, edges, t)
    edge_set = {(int(u), int(v)) for u, v in edges}
    targets = r_in - r_out
    if not targets:
        return 0
    count = 0
    for u in r_out:
        for v in targets:
            if (u, v) not in edge_set:
                count += 1
    return count


def structural_features(row: dict[str, Any]) -> dict[str, Any]:
    """Locked structural feature vector for one ood_hops row."""
    n, edges, s, t = parse_instance(str(row["encoding"]))
    n = int(row.get("n", n))
    s = int(row.get("s", s))
    t = int(row.get("t", t))
    n_edges = len(edges)
    denom = n * (n - 1) if n > 1 else 1
    p_emp = n_edges / denom
    deg = total_degrees(n, edges)
    r_out = reachable_out_set(n, edges, s)
    r_in = _r_in_set(n, edges, t)
    dists = hop_distances_from(n, edges, s)
    reachable_dists = [d for d in dists if d >= 0]
    max_dist = max(reachable_dists) if reachable_dists else 0
    tok_len = len(split_encoding_tokens(str(row["encoding"])))
    return {
        "n": n,
        "n_edges": n_edges,
        "p": float(row.get("p", p_emp)),
        "p_emp": float(p_emp),
        "token_len": int(tok_len),
        "hop_distance": int(row.get("hop_distance", -999)),
        "y": int(row.get("y", -1)),
        "s": s,
        "t": t,
        "deg_s": int(deg[s]),
        "deg_t": int(deg[t]),
        "n_reach_from_s": int(len(r_out)),
        "n_reach_to_t": int(len(r_in)),
        "frac_reach_from_s": float(len(r_out) / n) if n else 0.0,
        "frac_reach_to_t": float(len(r_in) / n) if n else 0.0,
        "near_miss_bridges": int(_near_miss_bridges(n, edges, s, t)),
        "frontier_exits": int(_frontier_exits(n, edges, s)),
        "max_dist_from_s": int(max_dist),
        "arm_id": row.get("arm_id"),
        "arm_meta": row.get("arm_meta"),
        "edge_hash": row.get("edge_hash"),
        "seed": int(row.get("seed", -1)),
    }


def _summarize_numeric(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    xs = sorted(float(v) for v in vals)
    n = len(xs)
    mean = sum(xs) / n
    mid = n // 2
    median = xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    q25 = xs[max(0, (n - 1) // 4)]
    q75 = xs[min(n - 1, (3 * (n - 1)) // 4)]
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "min": xs[0],
        "max": xs[-1],
        "q25": q25,
        "q75": q75,
    }


def summarize_cohort_features(
    feats: list[dict[str, Any]],
    keys: Sequence[str],
) -> dict[str, Any]:
    """Mean/median/IQR tables + categorical counters for a cohort."""
    out: dict[str, Any] = {"n": len(feats)}
    numeric: dict[str, Any] = {}
    for k in keys:
        vals = [float(f[k]) for f in feats if f.get(k) is not None]
        numeric[k] = _summarize_numeric(vals)
    out["numeric"] = numeric
    # bins
    n_bin = Counter(int(f["n"]) for f in feats)
    p_bin = Counter(round(float(f["p"]), 4) for f in feats)
    # near_miss_bridges (missing u→v) is almost always >0 on true HN; use
    # frontier_exits + high-reach as discriminative bins instead.
    fr_bin = Counter(
        "frontier>0" if int(f.get("frontier_exits", 0)) > 0 else "frontier==0"
        for f in feats
    )
    reach_bin = Counter(
        "frac_reach_s>=0.25"
        if float(f.get("frac_reach_from_s", 0)) >= 0.25
        else "frac_reach_s<0.25"
        for f in feats
    )
    out["bins"] = {
        "n": {str(k): int(v) for k, v in sorted(n_bin.items())},
        "p": {str(k): int(v) for k, v in sorted(p_bin.items())},
        "frontier": dict(fr_bin),
        "frac_reach_s": dict(reach_bin),
        # keep near_miss key for back-compat with tests/decision (maps frontier)
        "near_miss": dict(fr_bin),
    }
    tags = Counter(
        "none"
        if f.get("arm_id") is None and f.get("arm_meta") is None
        else f"arm_id={f.get('arm_id')}"
        for f in feats
    )
    out["construction_tags"] = dict(tags)
    return out


STRUCT_NUMERIC_KEYS: tuple[str, ...] = (
    "n",
    "n_edges",
    "p",
    "p_emp",
    "token_len",
    "deg_s",
    "deg_t",
    "n_reach_from_s",
    "n_reach_to_t",
    "frac_reach_from_s",
    "frac_reach_to_t",
    "near_miss_bridges",
    "frontier_exits",
    "max_dist_from_s",
)


def classify_member_agreement(
    member_preds: Sequence[int],
    member_confs: Sequence[float],
    *,
    conf_thresh: float = CONF_THRESH,
) -> str:
    """HARD_UNANIMOUS | SOFT_AGREE | MIXED_PRED (LOCKED)."""
    preds = [int(p) for p in member_preds]
    confs = [float(c) for c in member_confs]
    if not preds:
        return "MIXED_PRED"
    if any(p != 1 for p in preds):
        return "MIXED_PRED"
    n_hi = sum(1 for c in confs if c >= conf_thresh)
    if n_hi == len(preds):
        return "HARD_UNANIMOUS"
    return "SOFT_AGREE"


def _median_outside_iqr(fo_med: float, ctrl_q25: float, ctrl_q75: float) -> bool:
    if any(math.isnan(x) for x in (fo_med, ctrl_q25, ctrl_q75)):
        return False
    return fo_med < ctrl_q25 or fo_med > ctrl_q75


def _dominant_bin(bin_counts: dict[str, int], *, min_frac: float = 0.50) -> Optional[str]:
    n = sum(bin_counts.values())
    if n <= 0:
        return None
    best_k, best_v = max(bin_counts.items(), key=lambda kv: kv[1])
    if best_v / n >= min_frac:
        return best_k
    return None


def decide_autopsy_verdict(
    fo_summary: dict[str, Any],
    ok_hn_summary: dict[str, Any],
    fc_hn_summary: dict[str, Any],
    *,
    artifact_flags: dict[str, Any],
) -> dict[str, Any]:
    """Apply locked STRUCTURAL_CLUSTER | DIFFUSE | DATA_ARTIFACT | INCONCLUSIVE."""
    if artifact_flags.get("label_inconsistency") or artifact_flags.get(
        "encoding_defect"
    ):
        return {
            "verdict": "DATA_ARTIFACT",
            "reasons": artifact_flags.get("reasons", ["artifact_flag"]),
        }

    # Construction tags: FO all none is expected, not an artifact by itself.
    reasons: list[str] = []
    cluster_hits: list[str] = []

    fo_num = fo_summary.get("numeric", {})
    ok_num = ok_hn_summary.get("numeric", {})
    # Features that can vary within HN (n/p fixed at 32/0.06 on this slice)
    vary_keys = [
        k
        for k in (
            "n_edges",
            "p_emp",
            "token_len",
            "deg_s",
            "deg_t",
            "n_reach_from_s",
            "n_reach_to_t",
            "frac_reach_from_s",
            "frac_reach_to_t",
            "near_miss_bridges",
            "frontier_exits",
            "max_dist_from_s",
        )
        if k in fo_num and k in ok_num
    ]
    for k in vary_keys:
        fo_med = float(fo_num[k]["median"])
        if _median_outside_iqr(fo_med, float(ok_num[k]["q25"]), float(ok_num[k]["q75"])):
            cluster_hits.append(
                f"{k}: FO_med={fo_med:.4g} outside OK_HN IQR "
                f"[{ok_num[k]['q25']:.4g},{ok_num[k]['q75']:.4g}]"
            )

    # Dominant frontier / reach-bin contrast (discriminative on fixed-n HN)
    fo_fr = fo_summary.get("bins", {}).get("frontier", {})
    ok_fr = ok_hn_summary.get("bins", {}).get("frontier", {})
    fo_dom = _dominant_bin(fo_fr)
    ok_dom = _dominant_bin(ok_fr)
    if fo_dom is not None and fo_dom != ok_dom:
        cluster_hits.append(
            f"frontier_bin: FO dominant={fo_dom} vs OK_HN dominant={ok_dom}"
        )
        reasons.append("frontier_bin_contrast")

    fo_rs = fo_summary.get("bins", {}).get("frac_reach_s", {})
    ok_rs = ok_hn_summary.get("bins", {}).get("frac_reach_s", {})
    fo_rs_dom = _dominant_bin(fo_rs)
    ok_rs_dom = _dominant_bin(ok_rs)
    if fo_rs_dom is not None and fo_rs_dom != ok_rs_dom:
        cluster_hits.append(
            f"frac_reach_s_bin: FO dominant={fo_rs_dom} vs OK_HN dominant={ok_rs_dom}"
        )
        reasons.append("frac_reach_s_bin_contrast")

    # n/p bins: on this dataset HN are fixed — note but do not count as cluster
    fo_n_bins = fo_summary.get("bins", {}).get("n", {})
    ok_n_bins = ok_hn_summary.get("bins", {}).get("n", {})
    if fo_n_bins == ok_n_bins or (
        set(fo_n_bins.keys()) == set(ok_n_bins.keys()) and len(fo_n_bins) == 1
    ):
        reasons.append("n_p_fixed_across_all_HN_not_discriminative")

    if len(cluster_hits) >= 1 and (
        fo_dom is not None and fo_dom != ok_dom
        or len(cluster_hits) >= 2
    ):
        return {
            "verdict": "STRUCTURAL_CLUSTER",
            "reasons": reasons + cluster_hits,
            "cluster_hits": cluster_hits,
        }

    if not cluster_hits:
        return {
            "verdict": "DIFFUSE",
            "reasons": reasons
            + ["no_FO_median_outside_OK_HN_IQR_on_varying_features"],
            "cluster_hits": [],
        }

    # Exactly one soft hit without bin contrast → inconclusive
    return {
        "verdict": "INCONCLUSIVE",
        "reasons": reasons + cluster_hits + ["single_soft_hit_no_bin_contrast"],
        "cluster_hits": cluster_hits,
    }


def _cohort_ids(mask_list: list[bool]) -> list[int]:
    return [i for i, m in enumerate(mask_list) if m]


def run_cycle(
    *,
    hops_data: Path = DEFAULT_HOPS,
    out_path: Path = DEFAULT_OUT,
    cite30_path: Path = DEFAULT_CITE30,
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS,
) -> dict[str, Any]:
    t0 = time.time()
    if not hops_data.exists():
        raise FileNotFoundError(f"ood_hops missing: {hops_data}")
    rows = load_jsonl(hops_data)
    max_nodes = max(DEFAULT_MAX_NODE_ID, max(int(r["n"]) for r in rows))

    cite30: Optional[dict[str, Any]] = None
    if cite30_path.exists():
        cite30 = json.loads(cite30_path.read_text())

    print(
        f"[stalk-hn-fo-autopsy] loading #14/#18/#22 ens n={len(ensemble_seeds)}",
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
        f"[stalk-hn-fo-autopsy] infer ood_hops T{FOCUS_T} for full FAIL_OPEN + members",
        file=sys.stderr,
    )
    import torch
    import torch.nn.functional as F

    logits, labels, hops = _collect_member_logits(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )
    hard = logits.argmax(dim=-1)  # (M, N)
    ens_preds = _aggregate_preds(logits, method=PRIMARY_AGG)
    pair_per = _pairwise_disagreement_per_example(hard)
    probs = F.softmax(logits, dim=-1)
    mean_p = probs.mean(dim=0)
    ens_max_prob = mean_p.max(dim=-1).values
    total_H = _entropy_nats(mean_p)
    alea_H = _entropy_nats(probs).mean(dim=0)
    epi_H = total_H - alea_H
    member_max_prob = probs.max(dim=-1).values  # (M, N)

    fail_mode = _fail_mode_counts(
        ens_preds=ens_preds,
        labels=labels,
        pair_per=pair_per,
        ens_max_prob=ens_max_prob,
        epi_H=epi_H,
        hops=hops,
    )
    # Prefer full wrong list (not truncated sample)
    wrong_details = []
    wrong_mask = ens_preds != labels
    w_idx = wrong_mask.nonzero(as_tuple=False).view(-1)
    for i in w_idx.tolist():
        d_ex = float(pair_per[i].item())
        conf = float(ens_max_prob[i].item())
        epi = float(epi_H[i].item())
        if d_ex == 0.0 and conf >= CONF_THRESH:
            tag = "FAIL_OPEN"
        elif d_ex >= D_CLOSED_MIN or epi >= EPI_CLOSED_MIN or conf < CONF_THRESH:
            tag = "FAIL_CLOSED"
        else:
            tag = "FAIL_AMBIG"
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

    fo_ids = [d["i"] for d in wrong_details if d["tag"] == "FAIL_OPEN"]
    fc_ids = [d["i"] for d in wrong_details if d["tag"] == "FAIL_CLOSED"]
    fo_hn_ids = [d["i"] for d in wrong_details if d["tag"] == "FAIL_OPEN" and d["hop"] == int(HOP_UNREACHABLE)]
    fc_hn_ids = [d["i"] for d in wrong_details if d["tag"] == "FAIL_CLOSED" and d["hop"] == int(HOP_UNREACHABLE)]

    correct_mask = ens_preds == labels
    ok_hn_ids = [
        i
        for i in range(len(hops))
        if bool(correct_mask[i].item()) and hops[i] == int(HOP_UNREACHABLE)
    ]

    # Structural features for all rows (cheap)
    all_feats = [structural_features(r) for r in rows]
    fo_feats = [all_feats[i] for i in fo_hn_ids]
    fc_wrong_feats = [all_feats[i] for i in fc_ids]
    fc_hn_feats = [all_feats[i] for i in fc_hn_ids]
    ok_hn_feats = [all_feats[i] for i in ok_hn_ids]

    fo_summary = summarize_cohort_features(fo_feats, STRUCT_NUMERIC_KEYS)
    fc_wrong_summary = summarize_cohort_features(fc_wrong_feats, STRUCT_NUMERIC_KEYS)
    fc_hn_summary = summarize_cohort_features(fc_hn_feats, STRUCT_NUMERIC_KEYS)
    ok_hn_summary = summarize_cohort_features(ok_hn_feats, STRUCT_NUMERIC_KEYS)

    # Member-logit autopsy on FO_HN
    member_rows: list[dict[str, Any]] = []
    member_tag_counts: Counter = Counter()
    for i in fo_hn_ids:
        m_preds = [int(hard[m, i].item()) for m in range(hard.shape[0])]
        m_confs = [float(member_max_prob[m, i].item()) for m in range(hard.shape[0])]
        # conf on predicted class (class-1 for FO)
        m_p1 = [float(probs[m, i, 1].item()) for m in range(probs.shape[0])]
        tag = classify_member_agreement(m_preds, m_confs)
        member_tag_counts[tag] += 1
        member_rows.append(
            {
                "i": int(i),
                "member_preds": m_preds,
                "member_max_prob": m_confs,
                "member_p1": m_p1,
                "n_members_pred1": sum(1 for p in m_preds if p == 1),
                "n_members_conf_ge_080": sum(1 for c in m_confs if c >= CONF_THRESH),
                "mean_member_p1": sum(m_p1) / len(m_p1) if m_p1 else float("nan"),
                "tag": tag,
                "ens_max_prob": float(ens_max_prob[i].item()),
                "D_ex": float(pair_per[i].item()),
            }
        )

    # Artifact checks
    artifact_flags: dict[str, Any] = {
        "label_inconsistency": False,
        "encoding_defect": False,
        "reasons": [],
    }
    for i in fo_hn_ids:
        f = all_feats[i]
        if int(f["y"]) != 0 or int(f["hop_distance"]) != int(HOP_UNREACHABLE):
            artifact_flags["label_inconsistency"] = True
            artifact_flags["reasons"].append(f"i={i} y/hop mismatch")
        if int(rows[i].get("y", -1)) != 0:
            artifact_flags["label_inconsistency"] = True
            artifact_flags["reasons"].append(f"i={i} row.y!=0")
        # FO must be pred=1 label=0
        if int(ens_preds[i].item()) != 1 or int(labels[i].item()) != 0:
            artifact_flags["label_inconsistency"] = True
            artifact_flags["reasons"].append(f"i={i} ens_pred/label not 1/0")

    decision = decide_autopsy_verdict(
        fo_summary, ok_hn_summary, fc_hn_summary, artifact_flags=artifact_flags
    )
    verdict = decision["verdict"]

    # Residue update
    if verdict == "STRUCTURAL_CLUSTER":
        residue = "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER"
        residue_note = (
            "FO core concentrates on structural contrast vs OK_HN "
            "(see cluster_hits); still unpaid; no repair claim."
        )
    elif verdict == "DIFFUSE":
        residue = "HN_FAIL_OPEN_CORE/DIFFUSE"
        residue_note = (
            "FO core scattered across HN structural space vs OK_HN; "
            "not a single near-miss/density bin; still unpaid."
        )
    elif verdict == "DATA_ARTIFACT":
        residue = "HN_FAIL_OPEN_CORE/DATA_ARTIFACT"
        residue_note = "FO core tied to construction/label artifact; investigate data."
    else:
        residue = "HN_FAIL_OPEN_CORE/INCONCLUSIVE"
        residue_note = (
            "FO core autopsy inconclusive; HN_FAIL_OPEN_CORE remains unpaid."
        )

    # Contrast tables (FO vs controls) — key medians
    def _med(summary: dict[str, Any], k: str) -> float:
        return float(summary.get("numeric", {}).get(k, {}).get("median", float("nan")))

    contrast_keys = [
        "n_edges",
        "p_emp",
        "token_len",
        "deg_s",
        "deg_t",
        "n_reach_from_s",
        "n_reach_to_t",
        "frac_reach_from_s",
        "frac_reach_to_t",
        "near_miss_bridges",
        "frontier_exits",
        "max_dist_from_s",
    ]
    contrast_table = []
    for k in contrast_keys:
        contrast_table.append(
            {
                "feature": k,
                "FO_HN_median": _med(fo_summary, k),
                "FO_HN_mean": float(fo_summary["numeric"][k]["mean"]),
                "OK_HN_median": _med(ok_hn_summary, k),
                "OK_HN_mean": float(ok_hn_summary["numeric"][k]["mean"])
                if ok_hn_summary["n"]
                else float("nan"),
                "FC_HN_median": _med(fc_hn_summary, k),
                "FC_HN_mean": float(fc_hn_summary["numeric"][k]["mean"])
                if fc_hn_summary["n"]
                else float("nan"),
                "FC_wrong_median": _med(fc_wrong_summary, k),
            }
        )

    # Replicate check vs #30
    replicate = {
        "FAIL_OPEN": len(fo_ids),
        "FAIL_CLOSED": len(fc_ids),
        "FO_HN": len(fo_hn_ids),
        "cite30_FO": CITE_30_FO,
        "cite30_FC": CITE_30_FC,
        "FO_match": len(fo_ids) == CITE_30_FO and len(fo_hn_ids) == CITE_30_FO,
        "note": "All #30 FAIL_OPEN are HN; FO_HN must equal FAIL_OPEN=45.",
    }
    if cite30 is not None:
        c_fm = (
            cite30.get("audit", {})
            .get("baseline_prob_mean_T16", {})
            .get("fail_mode", {})
        )
        replicate["cite30_artifact_FO"] = c_fm.get("FAIL_OPEN")
        replicate["cite30_artifact_FC"] = c_fm.get("FAIL_CLOSED")

    ens_hn_acc = float(
        (
            ens_preds[
                torch.tensor([h == int(HOP_UNREACHABLE) for h in hops], dtype=torch.bool)
            ]
            == labels[
                torch.tensor([h == int(HOP_UNREACHABLE) for h in hops], dtype=torch.bool)
            ]
        )
        .float()
        .mean()
        .item()
    )

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
            "kind": "eval_only_hn_fail_open_structural_autopsy",
            "train": False,
            "bag_noise_trained": False,
            "select_reopened": False,
            "curriculum_reopened": False,
            "d": DEFAULT_D,
            "aggregator_primary": PRIMARY_AGG,
            "T_fixed": FOCUS_T,
            "hard_A": True,
            "broadcast_c": False,
        },
        "datasets": {
            "ood_hops": str(hops_data),
            "n_ood_hops": len(rows),
            "cite30": str(cite30_path) if cite30_path.exists() else None,
        },
        "members_pr22": ens_meta,
        "prereg": {
            "questions": [
                "graph_structure_of_45",
                "compare_vs_FC_and_correct_HN",
                "cluster_vs_scattered",
                "member_logits_unanimous_vs_soft",
                "verdict_enum_plus_residue",
            ],
            "verdicts": list(VERDICTS),
            "member_tags": list(MEMBER_TAGS),
            "CONF_THRESH": CONF_THRESH,
            "D_CLOSED_MIN": D_CLOSED_MIN,
            "EPI_CLOSED_MIN": EPI_CLOSED_MIN,
            "non_goals": [
                "no_train",
                "no_science_open_widen",
                "section22_unchanged",
                "T_fixed_16",
                "no_hn_accuracy_repair_claim",
                "no_park_whole_stalk_corridor",
            ],
        },
        "replicate_cite30": replicate,
        "fail_mode": {
            "FAIL_OPEN": len(fo_ids),
            "FAIL_CLOSED": len(fc_ids),
            "FAIL_AMBIG": sum(1 for d in wrong_details if d["tag"] == "FAIL_AMBIG"),
            "n_wrong": len(wrong_details),
            "ens_HN": ens_hn_acc,
            "ens_overall": float((ens_preds == labels).float().mean().item()),
        },
        "cohorts": {
            "FO_HN": {"n": len(fo_hn_ids), "ids": fo_hn_ids},
            "FC_wrong": {"n": len(fc_ids), "ids": fc_ids},
            "FC_HN": {"n": len(fc_hn_ids), "ids": fc_hn_ids},
            "OK_HN": {"n": len(ok_hn_ids), "ids": ok_hn_ids},
        },
        "structure_tables": {
            "FO_HN": fo_summary,
            "FC_wrong": fc_wrong_summary,
            "FC_HN": fc_hn_summary,
            "OK_HN": ok_hn_summary,
            "contrast_medians": contrast_table,
        },
        "member_logits": {
            "tag_counts": dict(member_tag_counts),
            "examples": member_rows,
            "reading": (
                "HARD_UNANIMOUS = all 10 pred=1 and conf≥0.80; "
                "SOFT_AGREE = all pred=1 but some conf<0.80; "
                "MIXED_PRED should not occur when D_ex==0."
            ),
        },
        "artifact_flags": artifact_flags,
        "autopsy_decision": decision,
        "verdict": verdict,
        "residue": residue,
        "residue_note": residue_note,
        "example_id_list_FO_HN": fo_hn_ids,
        "reading": (
            "Structural autopsy of #30 HN_FAIL_OPEN_CORE (45). "
            "MEASURE only; science_open=false; §22 unchanged; no HN repair claim."
        ),
        "elapsed_sec": elapsed,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[stalk-hn-fo-autopsy] verdict={verdict} residue={residue} "
        f"FO_HN={len(fo_hn_ids)} FC_HN={len(fc_hn_ids)} OK_HN={len(ok_hn_ids)} "
        f"member_tags={dict(member_tag_counts)} "
        f"wrote {out_path} ({elapsed:.1f}s)",
        file=sys.stderr,
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hops-data", type=Path, default=DEFAULT_HOPS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cite30", type=Path, default=DEFAULT_CITE30)
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
        cite30_path=args.cite30,
        ensemble_seeds=tuple(args.seeds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
