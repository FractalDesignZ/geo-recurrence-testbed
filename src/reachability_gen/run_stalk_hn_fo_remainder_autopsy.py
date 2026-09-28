"""CYCLE_STALK_HN_FO_REMAINDER_AUTOPSY — MEASURE autopsy of outdeg>0 FO remainder.

Eval-only @ T=16 on data/ood_hops.jsonl. Structural contrast of the 22 FO
remaining after #32 outdeg0 gate vs killed FO / OK_HN / FC_HN. Hunt a
local-sound cut (no full BFS as gate feature); if coverage <8/22 →
LOCAL_SOUND_WALL and stop overlay chase. No train. science_open=false.

Usage::

    python -m reachability_gen.run_stalk_hn_fo_remainder_autopsy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.graph import adjacency_list
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_stalk_hn_fail_open_autopsy import (
    _summarize_numeric,
    structural_features,
)
from reachability_gen.run_stalk_hop_ood_hn import FOCUS_T
from reachability_gen.run_stalk_sound_outdeg_gate import out_degree

CYCLE = "CYCLE_STALK_HN_FO_REMAINDER_AUTOPSY"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_hn_fo_remainder_autopsy.json")
DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE32 = Path("artifacts/stalk_sound_outdeg_gate.json")

# Autopsy / cycle verdicts (LOCKED)
VERDICTS = (
    "STRUCTURAL_CLUSTER_REMAINDER",
    "DIFFUSE",
    "LOCAL_SOUND_CUT_FOUND",
    "LOCAL_SOUND_WALL",
    "INCONCLUSIVE",
)

# Gate overlay verdicts (LOCKED — only if gate run)
GATE_VERDICTS = (
    "FO_REMAINDER_KILLED",
    "FO_REMAINDER_PARTIAL",
    "COLLATERAL_HARM",
)

COVERAGE_MIN = 8  # of 22
FO_REMAINDER_N = 22
FO_KILLED_N = 23
CITE_30_FO = 45


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


def in_degree(n: int, edges: Sequence[tuple[int, int]], node: int) -> int:
    """Directed in-degree of node (local incidence; no BFS)."""
    if not (0 <= node < n):
        raise ValueError(f"node={node} out of range for n={n}")
    return sum(1 for _u, v in edges if int(v) == node)


def local_incidence_features(row: dict[str, Any]) -> dict[str, Any]:
    """Local-only features allowed as gate candidates (no BFS)."""
    n, edges, s, t = parse_instance(str(row["encoding"]))
    n = int(row.get("n", n))
    s = int(row.get("s", s))
    t = int(row.get("t", t))
    adj = adjacency_list(n, edges)
    radj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        radj[int(v)].append(int(u))
    nout = set(adj[s])
    nin_t = set(radj[t])
    od_s = out_degree(n, edges, s)
    id_s = in_degree(n, edges, s)
    od_t = out_degree(n, edges, t)
    id_t = in_degree(n, edges, t)
    nbr_outdegs = [out_degree(n, edges, v) for v in nout]
    all_nbr_outdeg0 = bool(nout) and all(d == 0 for d in nbr_outdegs)
    t_in_star = (t == s) or (t in nout)
    return {
        "n": n,
        "s": s,
        "t": t,
        "y": int(row.get("y", -1)),
        "outdeg_s": int(od_s),
        "indeg_s": int(id_s),
        "outdeg_t": int(od_t),
        "indeg_t": int(id_t),
        "nout_s": int(len(nout)),
        "nin_t": int(len(nin_t)),
        "t_in_Nout_s": bool(t in nout),
        "s_in_Nin_t": bool(s in nin_t),
        "all_nbr_outdeg0": bool(all_nbr_outdeg0),
        "min_nbr_outdeg": int(min(nbr_outdegs)) if nbr_outdegs else None,
        "max_nbr_outdeg": int(max(nbr_outdegs)) if nbr_outdegs else None,
        "sum_nbr_outdeg": int(sum(nbr_outdegs)),
        "t_in_star": bool(t_in_star),
        # Sound local-cut candidates (when true ⇒ y must be 0 for t≠s)
        "C_outdeg0": bool(od_s == 0 and t != s),
        "C_indeg_t0": bool(id_t == 0 and t != s),
        # Degrees of 1-hop neighbors only — not iterative BFS closure
        "C_deadend_nbrs": bool(all_nbr_outdeg0 and (not t_in_star) and t != s),
        "C_outdeg1_deadend": bool(
            od_s == 1 and all_nbr_outdeg0 and (not t_in_star) and t != s
        ),
    }


def sound_cut_predicates() -> dict[str, str]:
    """Name → description of sound local cuts under evaluation."""
    return {
        "C_outdeg0": "outdeg(s)==0 ∧ t≠s (already used in #32)",
        "C_indeg_t0": "indeg(t)==0 ∧ t≠s",
        "C_deadend_nbrs": (
            "all v∈N+(s) have outdeg(v)==0 ∧ t∉{s}∪N+(s) "
            "(1-hop star dead-end; not full BFS)"
        ),
        "C_outdeg1_deadend": (
            "outdeg(s)==1 ∧ unique nbr has outdeg0 ∧ t∉star"
        ),
    }


def evaluate_sound_cuts(
    rows: Sequence[dict[str, Any]],
    remainder_ids: Sequence[int],
) -> dict[str, Any]:
    """Coverage on remainder + full-slice soundness (trigger ⇒ y==0)."""
    rem_set = set(int(i) for i in remainder_ids)
    out: dict[str, Any] = {}
    for cname in sound_cut_predicates():
        rem_hits: list[int] = []
        triggers = 0
        violations = 0
        violation_ids: list[int] = []
        for i, row in enumerate(rows):
            lf = local_incidence_features(row)
            if not lf[cname]:
                continue
            triggers += 1
            if int(row["y"]) != 0:
                violations += 1
                violation_ids.append(i)
            if i in rem_set:
                rem_hits.append(i)
        out[cname] = {
            "description": sound_cut_predicates()[cname],
            "remainder_hits": len(rem_hits),
            "remainder_total": len(rem_set),
            "remainder_ids": rem_hits,
            "meets_coverage_min": len(rem_hits) >= COVERAGE_MIN,
            "full_triggers": triggers,
            "full_violations_y_neq_0": violations,
            "violation_ids": violation_ids,
            "sound": violations == 0,
            "actionable": (
                violations == 0 and len(rem_hits) >= COVERAGE_MIN
            ),
        }
    return out


def _cohort_feature_table(
    rows: Sequence[dict[str, Any]],
    ids: Sequence[int],
) -> dict[str, Any]:
    """Local + structural (BFS ok for autopsy only) summaries."""
    local_keys = [
        "outdeg_s",
        "indeg_s",
        "outdeg_t",
        "indeg_t",
        "nout_s",
        "nin_t",
        "sum_nbr_outdeg",
    ]
    struct_keys = [
        "n_reach_from_s",
        "n_reach_to_t",
        "max_dist_from_s",
        "frac_reach_from_s",
        "frac_reach_to_t",
        "deg_s",
        "deg_t",
        "n_edges",
    ]
    bool_keys = [
        "t_in_Nout_s",
        "s_in_Nin_t",
        "all_nbr_outdeg0",
        "C_outdeg0",
        "C_indeg_t0",
        "C_deadend_nbrs",
        "C_outdeg1_deadend",
    ]
    locals_list: list[dict[str, Any]] = []
    structs_list: list[dict[str, Any]] = []
    for i in ids:
        row = rows[int(i)]
        lf = local_incidence_features(row)
        sf = structural_features(row)
        lf["id"] = int(i)
        sf["id"] = int(i)
        locals_list.append(lf)
        structs_list.append(sf)

    numeric: dict[str, Any] = {}
    for k in local_keys + struct_keys:
        vals: list[float] = []
        src = locals_list if k in local_keys else structs_list
        for d in src:
            v = d.get(k)
            if v is not None:
                vals.append(float(v))
        numeric[k] = _summarize_numeric(vals)

    bool_counts: dict[str, Any] = {}
    for k in bool_keys:
        c = Counter(bool(d[k]) for d in locals_list)
        bool_counts[k] = {"true": int(c.get(True, 0)), "false": int(c.get(False, 0))}

    outdeg_hist = Counter(int(d["outdeg_s"]) for d in locals_list)
    n_reach_hist = Counter(int(d["n_reach_from_s"]) for d in structs_list)
    maxd_hist = Counter(int(d["max_dist_from_s"]) for d in structs_list)

    return {
        "n": len(ids),
        "ids": [int(i) for i in ids],
        "numeric": numeric,
        "bool_counts": bool_counts,
        "hist_outdeg_s": {str(k): v for k, v in sorted(outdeg_hist.items())},
        "hist_n_reach_from_s": {str(k): v for k, v in sorted(n_reach_hist.items())},
        "hist_max_dist_from_s": {str(k): v for k, v in sorted(maxd_hist.items())},
        "per_example_local": [
            {
                "id": d["id"],
                "outdeg_s": d["outdeg_s"],
                "indeg_t": d["indeg_t"],
                "nout_s": d["nout_s"],
                "all_nbr_outdeg0": d["all_nbr_outdeg0"],
                "C_deadend_nbrs": d["C_deadend_nbrs"],
                "C_indeg_t0": d["C_indeg_t0"],
            }
            for d in locals_list
        ],
        "per_example_struct": [
            {
                "id": d["id"],
                "n_reach_from_s": d["n_reach_from_s"],
                "n_reach_to_t": d["n_reach_to_t"],
                "max_dist_from_s": d["max_dist_from_s"],
            }
            for d in structs_list
        ],
    }


def contrast_medians(tables: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Median contrast table across cohorts for key features."""
    keys = [
        "outdeg_s",
        "indeg_s",
        "outdeg_t",
        "indeg_t",
        "nout_s",
        "nin_t",
        "n_reach_from_s",
        "n_reach_to_t",
        "max_dist_from_s",
        "frac_reach_from_s",
        "frac_reach_to_t",
        "deg_s",
        "deg_t",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = {
            name: tables[name]["numeric"][k]["median"]
            for name in tables
            if k in tables[name]["numeric"]
        }
    return out


def decide_verdict(
    *,
    cut_eval: dict[str, Any],
    remainder_table: dict[str, Any],
    killed_table: dict[str, Any],
    ok_table: dict[str, Any],
) -> dict[str, Any]:
    """LOCKED verdict enum. LOCAL_SOUND_WALL if no actionable cut."""
    actionable = [
        name for name, info in cut_eval.items() if info.get("actionable")
    ]
    reasons: list[str] = []

    if actionable:
        return {
            "verdict": "LOCAL_SOUND_CUT_FOUND",
            "reasons": [f"actionable_cuts={actionable}"],
            "actionable_cuts": actionable,
            "gate_run": True,
        }

    # No actionable cut — wall vs diffuse vs cluster characterization
    rem_n_reach_med = remainder_table["numeric"]["n_reach_from_s"]["median"]
    kil_n_reach_med = killed_table["numeric"]["n_reach_from_s"]["median"]
    ok_n_reach_med = ok_table["numeric"]["n_reach_from_s"]["median"]
    rem_outdeg_med = remainder_table["numeric"]["outdeg_s"]["median"]
    deadend_true = remainder_table["bool_counts"]["C_deadend_nbrs"]["true"]
    best_cov = max(info["remainder_hits"] for info in cut_eval.values())

    reasons.append(f"best_sound_coverage={best_cov}/{FO_REMAINDER_N}<{COVERAGE_MIN}")
    reasons.append(
        f"rem_n_reach_med={rem_n_reach_med} vs killed={kil_n_reach_med} "
        f"vs OK={ok_n_reach_med}"
    )
    reasons.append(f"rem_outdeg_med={rem_outdeg_med} (all >0 by construction)")
    reasons.append(f"C_deadend_nbrs_hits={deadend_true}/{FO_REMAINDER_N}")

    # Characterization side-tag (not the primary cycle verdict when wall)
    # Remainder looks like OK/FC on out-closure (large R_out), unlike killed
    # isolated-source cluster — unpaid FO need multi-hop reasoning.
    if rem_n_reach_med >= 10 and abs(rem_n_reach_med - ok_n_reach_med) <= 5:
        char = "DIFFUSE_MULTI_HOP_LIKE_OK_FC"
        reasons.append(
            "remainder out-closure overlaps OK_HN/FC_HN "
            "(not isolated-source; multi-hop unpaid)"
        )
    elif deadend_true >= FO_REMAINDER_N // 2:
        char = "STRUCTURAL_CLUSTER_REMAINDER"
        reasons.append("deadend-star dominates remainder")
    else:
        char = "MIXED_REMAINDER"
        reasons.append(
            f"mixed: deadend-star {deadend_true}/22 + large-R_out majority"
        )

    return {
        "verdict": "LOCAL_SOUND_WALL",
        "reasons": reasons,
        "actionable_cuts": [],
        "gate_run": False,
        "characterization": char,
        "best_sound_coverage": best_cov,
        "coverage_min": COVERAGE_MIN,
    }


def run_cycle(
    *,
    hops_data: Path = DEFAULT_HOPS,
    out_path: Path = DEFAULT_OUT,
    cite30_path: Path = DEFAULT_CITE30,
    cite31_path: Path = DEFAULT_CITE31,
    cite32_path: Path = DEFAULT_CITE32,
) -> dict[str, Any]:
    t0 = time.time()
    if not hops_data.exists():
        raise FileNotFoundError(f"ood_hops missing: {hops_data}")
    for p, label in [
        (cite32_path, "cite32 outdeg gate"),
        (cite31_path, "cite31 autopsy"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} missing: {p}")

    rows = load_jsonl(hops_data)
    cite30 = json.loads(cite30_path.read_text()) if cite30_path.exists() else None
    cite31 = json.loads(cite31_path.read_text())
    cite32 = json.loads(cite32_path.read_text())

    fo_analysis = cite32.get("fo_analysis") or {}
    rem_ids = list(fo_analysis.get("fo_still_wrong_ids") or [])
    kil_ids = list(fo_analysis.get("fo_killed_ids") or [])
    if len(rem_ids) != FO_REMAINDER_N or len(kil_ids) != FO_KILLED_N:
        raise RuntimeError(
            f"cite32 FO split unexpected: rem={len(rem_ids)} kil={len(kil_ids)} "
            f"(expected {FO_REMAINDER_N}/{FO_KILLED_N})"
        )

    ok_ids = list(cite31.get("cohorts", {}).get("OK_HN", {}).get("ids") or [])
    fc_ids = list(cite31.get("cohorts", {}).get("FC_HN", {}).get("ids") or [])
    fo_all = list(
        cite31.get("example_id_list_FO_HN")
        or cite31.get("cohorts", {}).get("FO_HN", {}).get("ids")
        or []
    )

    print(
        f"[fo-remainder-autopsy] offline on {len(rows)} ood_hops; "
        f"rem={len(rem_ids)} kil={len(kil_ids)} OK={len(ok_ids)} FC={len(fc_ids)}",
        file=sys.stderr,
    )

    tables = {
        "FO_REMAINDER": _cohort_feature_table(rows, rem_ids),
        "FO_KILLED": _cohort_feature_table(rows, kil_ids),
        "OK_HN": _cohort_feature_table(rows, ok_ids),
        "FC_HN": _cohort_feature_table(rows, fc_ids),
    }
    medians = contrast_medians(tables)
    cut_eval = evaluate_sound_cuts(rows, rem_ids)
    decision = decide_verdict(
        cut_eval=cut_eval,
        remainder_table=tables["FO_REMAINDER"],
        killed_table=tables["FO_KILLED"],
        ok_table=tables["OK_HN"],
    )

    # Cite #32 replicate
    cite32_fo_still = int(fo_analysis.get("fo_still_wrong", len(rem_ids)))
    cite32_fo_killed = int(fo_analysis.get("fo_killed", len(kil_ids)))
    cite32_verdict = cite32.get("verdict")
    replicate = {
        "cite32_verdict": cite32_verdict,
        "cite32_fo_killed": cite32_fo_killed,
        "cite32_fo_still": cite32_fo_still,
        "rem_ids_match_n": len(rem_ids) == FO_REMAINDER_N,
        "kil_ids_match_n": len(kil_ids) == FO_KILLED_N,
        "fo_union_is_45": len(set(rem_ids) | set(kil_ids)) == CITE_30_FO
        and len(set(rem_ids) & set(kil_ids)) == 0,
        "cite31_FO_n": len(fo_all),
        "cite30_FO": (cite30 or {})
        .get("arms", {})
        .get("0_baseline", {})
        .get("fail_mode", {})
        .get("FAIL_OPEN")
        if cite30
        else CITE_30_FO,
    }

    residue = (
        "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL"
        if decision["verdict"] == "LOCAL_SOUND_WALL"
        else "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL"
    )
    if decision["verdict"] == "LOCAL_SOUND_CUT_FOUND":
        residue = (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_CUT_FOUND"
        )

    artifact: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "section22_unchanged": True,
        "hop_ood_remains_measure_residue": True,
        "open_status": "science_open_false_not_widened",
        "base_sha": _git_sha(),
        "architecture": {
            "kind": "fo_remainder_autopsy_offline",
            "T_fixed": FOCUS_T,
            "train": False,
            "bfs_oracle_gate": False,
            "gate_run": bool(decision.get("gate_run")),
            "hard_A": True,
            "select_reopened": False,
            "curriculum_reopened": False,
            "bag_noise_trained": False,
        },
        "datasets": {
            "ood_hops": str(hops_data),
            "n_ood_hops": len(rows),
            "cite30": str(cite30_path) if cite30_path.exists() else None,
            "cite31": str(cite31_path),
            "cite32": str(cite32_path),
        },
        "cohorts": {
            "FO_REMAINDER": {"n": len(rem_ids), "ids": rem_ids},
            "FO_KILLED": {"n": len(kil_ids), "ids": kil_ids},
            "OK_HN": {"n": len(ok_ids), "ids": ok_ids},
            "FC_HN": {"n": len(fc_ids), "ids": fc_ids},
        },
        "contrast_medians": medians,
        "structure_tables": {
            name: {
                "n": tables[name]["n"],
                "numeric": tables[name]["numeric"],
                "bool_counts": tables[name]["bool_counts"],
                "hist_outdeg_s": tables[name]["hist_outdeg_s"],
                "hist_n_reach_from_s": tables[name]["hist_n_reach_from_s"],
                "hist_max_dist_from_s": tables[name]["hist_max_dist_from_s"],
            }
            for name in tables
        },
        "per_example": {
            "FO_REMAINDER_local": tables["FO_REMAINDER"]["per_example_local"],
            "FO_REMAINDER_struct": tables["FO_REMAINDER"]["per_example_struct"],
        },
        "local_sound_cuts": {
            "allowed": list(sound_cut_predicates().keys()),
            "descriptions": sound_cut_predicates(),
            "coverage_min": COVERAGE_MIN,
            "evaluation": cut_eval,
            "any_actionable": any(v["actionable"] for v in cut_eval.values()),
        },
        "gate_overlay": {
            "run": False,
            "reason": (
                "no actionable local-sound cut "
                f"(best coverage < {COVERAGE_MIN}/22)"
            ),
            "gate_verdict": None,
            "note": "MEASURE only; never OPEN",
        },
        "prereg": {
            "verdicts": list(VERDICTS),
            "gate_verdicts": list(GATE_VERDICTS),
            "COVERAGE_MIN": COVERAGE_MIN,
            "FOCUS_T": FOCUS_T,
            "non_goals": [
                "no_train",
                "no_bfs_gate",
                "no_science_open_widen",
                "no_section22_widen",
                "no_hop_ood_open_claim",
                "stop_overlay_chase_on_LOCAL_SOUND_WALL",
            ],
        },
        "replicate_cite32": replicate,
        "decision": decision,
        "verdict": decision["verdict"],
        "residue": residue,
        "residue_note": (
            "No sound local cut covers ≥8/22 remainder FO; "
            "remaining FO require multi-hop reasoning; "
            "stop overlay chase for this core. science_open=false."
            if decision["verdict"] == "LOCAL_SOUND_WALL"
            else "see decision"
        ),
        "reading": (
            "FO remainder autopsy after #32 outdeg0. MEASURE only; "
            "science_open=false; §22 unchanged; no hop-OOD OPEN claim; "
            "no BFS gate."
        ),
        "elapsed_sec": time.time() - t0,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"[fo-remainder-autopsy] verdict={artifact['verdict']} "
        f"residue={residue} elapsed={artifact['elapsed_sec']:.2f}s "
        f"-> {out_path}",
        file=sys.stderr,
    )
    return artifact


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=CYCLE)
    p.add_argument("--hops-data", type=Path, default=DEFAULT_HOPS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cite30", type=Path, default=DEFAULT_CITE30)
    p.add_argument("--cite31", type=Path, default=DEFAULT_CITE31)
    p.add_argument("--cite32", type=Path, default=DEFAULT_CITE32)
    args = p.parse_args(argv)
    run_cycle(
        hops_data=args.hops_data,
        out_path=args.out,
        cite30_path=args.cite30,
        cite31_path=args.cite31,
        cite32_path=args.cite32,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
