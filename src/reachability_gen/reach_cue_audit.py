"""Reach-cue audit: single-feature degree/reach ceilings (MEASURE / Gate C).

Before trusting model metrics, document how well trivial endpoint-degree or
local-reach heuristics classify y on the eval set. Prereg ceiling: any single
feature rule should be ≤0.52 accuracy (near chance on balanced labels) —
otherwise model scores may reflect cues, not learned Â.

Also builds a degree-balanced hard-neg view that matches endpoint degree
bins of positives (for secondary eval).

science_open=false always. No invented metrics.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.hard_negatives import total_degrees


def endpoint_degree_features(row: Mapping[str, Any]) -> dict[str, float]:
    """Extract endpoint degree / graph cues from a JSONL row."""
    enc = row.get("encoding")
    if enc:
        n, edges, s, t = parse_instance(str(enc))
    else:
        n = int(row["n"])
        edges = [(int(a), int(b)) for a, b in row["edges"]]
        s, t = int(row["s"]), int(row["t"])
    n = int(row.get("n", n))
    s = int(row.get("s", s))
    t = int(row.get("t", t))
    out_deg = [0] * n
    in_deg = [0] * n
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v] += 1
    tot = total_degrees(n, edges)
    return {
        "n": float(n),
        "n_edges": float(len(edges)),
        "out_s": float(out_deg[s]),
        "in_t": float(in_deg[t]),
        "in_s": float(in_deg[s]),
        "out_t": float(out_deg[t]),
        "tot_s": float(tot[s]),
        "tot_t": float(tot[t]),
        "out_s_plus_in_t": float(out_deg[s] + in_deg[t]),
        "y": float(int(row["y"])),
        "hop_distance": float(int(row.get("hop_distance", -1))),
    }


def _threshold_rule_acc(
    feats: Sequence[Mapping[str, float]],
    key: str,
    *,
    greater_means_pos: bool = True,
) -> dict[str, Any]:
    """Best single-threshold accuracy of ``feat[key]`` vs y (balanced report)."""
    xs = [f[key] for f in feats]
    ys = [int(f["y"]) for f in feats]
    if not xs:
        return {"feature": key, "acc": float("nan"), "threshold": None, "n": 0}
    candidates = sorted(set(xs))
    # Also try midpoints between unique values
    thr_list: list[float] = []
    for i, v in enumerate(candidates):
        thr_list.append(v)
        if i + 1 < len(candidates):
            thr_list.append(0.5 * (v + candidates[i + 1]))
    best_acc = -1.0
    best_thr: Optional[float] = None
    best_side = "gt"
    n = len(ys)
    for thr in thr_list:
        for side in ("gt", "ge"):
            preds = []
            for x in xs:
                if side == "gt":
                    pos = x > thr
                else:
                    pos = x >= thr
                if not greater_means_pos:
                    pos = not pos
                preds.append(1 if pos else 0)
            acc = sum(int(p == y) for p, y in zip(preds, ys)) / n
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
                best_side = side
    # Constant predictors
    base0 = sum(1 for y in ys if y == 0) / n
    base1 = 1.0 - base0
    majority = max(base0, base1)
    return {
        "feature": key,
        "acc": float(best_acc),
        "threshold": best_thr,
        "side": best_side,
        "greater_means_pos": greater_means_pos,
        "n": n,
        "majority_baseline": float(majority),
        "frac_y1": float(base1),
    }


def audit_reach_cues(
    rows: Sequence[Mapping[str, Any]],
    *,
    ceiling: float = 0.52,
) -> dict[str, Any]:
    """Run single-feature degree/reach rules; flag if any > ceiling."""
    feats = [endpoint_degree_features(r) for r in rows]
    # Gate C ceiling applies to single-feature *degree/reach* rules only.
    degree_keys = [
        "out_s",
        "in_t",
        "tot_s",
        "tot_t",
        "out_s_plus_in_t",
    ]
    covariate_keys = ["n_edges", "n"]
    rules = [_threshold_rule_acc(feats, k) for k in degree_keys]
    for r in rules:
        r["rule_class"] = "degree_reach"
    cov_rules = [_threshold_rule_acc(feats, k) for k in covariate_keys]
    for r in cov_rules:
        r["rule_class"] = "covariate_document_only"
    rules.extend(cov_rules)
    n = len(feats)
    ys = [int(f["y"]) for f in feats]
    for name, pred in (("constant_0", 0), ("constant_1", 1)):
        rules.append(
            {
                "feature": name,
                "acc": sum(1 for y in ys if y == pred) / n if n else float("nan"),
                "threshold": None,
                "n": n,
                "rule_class": "baseline_document_only",
            }
        )
    # Degree-bin majority (2-feature) — document; not single-feature ceiling
    bin_correct = 0
    bin_total = 0
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for f in feats:
        buckets[(int(f["out_s"]), int(f["in_t"]))].append(int(f["y"]))
    for labs in buckets.values():
        if not labs:
            continue
        maj = 1 if sum(labs) >= (len(labs) - sum(labs)) else 0
        bin_correct += sum(1 for y in labs if y == maj)
        bin_total += len(labs)
    bin_acc = bin_correct / bin_total if bin_total else float("nan")
    rules.append(
        {
            "feature": "out_s_x_in_t_bin_majority",
            "acc": float(bin_acc),
            "n_bins": len(buckets),
            "n": bin_total,
            "rule_class": "two_feature_document_only",
        }
    )

    deg_rules = [r for r in rules if r.get("rule_class") == "degree_reach"]
    max_rule = max(deg_rules, key=lambda r: (r["acc"] if r["acc"] == r["acc"] else -1))
    max_acc = float(max_rule["acc"])
    pass_ceiling = bool(max_acc == max_acc and max_acc <= ceiling)
    return {
        "n": n,
        "ceiling": ceiling,
        "rules": rules,
        "max_degree_reach_rule": max_rule["feature"],
        "max_degree_reach_acc": max_acc,
        # aliases used by harness
        "max_rule": max_rule["feature"],
        "max_acc": max_acc,
        "pass_ceiling": pass_ceiling,
        "note": (
            "Gate C ceiling: single-feature degree/reach rules ≤ "
            f"{ceiling} on this set. Covariates (n, |E|) and 2-feature bin "
            "majority are documented only. science_open=false."
        ),
        "science_open": False,
    }


def _deg_bin(out_s: float, in_t: float) -> tuple[int, int]:
    # Cap bins so rare high degrees collapse
    return (min(int(out_s), 5), min(int(in_t), 5))


def build_degree_balanced_eval(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Match hard-neg (y=0) endpoint degree bins to positive (y=1) counts.

    For each (out_s, in_t) bin present in positives, sample the same number of
    hard-negs (with replacement if short). Returns balanced rows + report.
    """
    import random

    rng = random.Random(seed)
    pos = [r for r in rows if int(r["y"]) == 1]
    neg = [r for r in rows if int(r["y"]) == 0]
    pos_bins: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    neg_bins: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in pos:
        f = endpoint_degree_features(r)
        pos_bins[_deg_bin(f["out_s"], f["in_t"])].append(dict(r))
    for r in neg:
        f = endpoint_degree_features(r)
        neg_bins[_deg_bin(f["out_s"], f["in_t"])].append(dict(r))

    balanced: list[dict[str, Any]] = []
    bin_report: list[dict[str, Any]] = []
    shortfall = 0
    for b, pl in sorted(pos_bins.items()):
        need = len(pl)
        nl = neg_bins.get(b, [])
        balanced.extend(pl)
        if not nl:
            shortfall += need
            bin_report.append(
                {"bin": list(b), "n_pos": need, "n_neg_available": 0, "n_neg_taken": 0}
            )
            continue
        taken = [dict(rng.choice(nl)) for _ in range(need)]
        for r in taken:
            r["degree_balanced"] = True
            r["degree_bin"] = list(b)
        balanced.extend(taken)
        bin_report.append(
            {
                "bin": list(b),
                "n_pos": need,
                "n_neg_available": len(nl),
                "n_neg_taken": need,
                "with_replacement": need > len(nl),
            }
        )

    n_pos = sum(1 for r in balanced if int(r["y"]) == 1)
    n_neg = sum(1 for r in balanced if int(r["y"]) == 0)
    return {
        "rows": balanced,
        "n": len(balanced),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "shortfall_unmatched_pos": shortfall,
        "bins": bin_report,
        "seed": seed,
        "science_open": False,
    }


def write_audit_artifact(
    audit: Mapping[str, Any],
    path: Path,
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = dict(audit)
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "audit_reach_cues",
    "build_degree_balanced_eval",
    "endpoint_degree_features",
    "write_audit_artifact",
]
