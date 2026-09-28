"""Generate single-cell RED_TEST substrate: K=20, requested p=0.15 (MEASURE).

CYCLE_SHEAF_RED_TEST — thin gen helper
-------------------------------------
Cell: hop K=20, ER request p=0.15, seq_len constrained to sealed ID band
[45, 70] (fail INVALID upstream if mean drifts toward ~200).

Feasibility (documented in report):
  Pure ER at p=0.15 under the band yields ~0 K=20 hits (expected |E| ≫ 21).
  Construction therefore uses **path-backbone** of length 20 + ≤1 distractor
  so |E|∈{20,21} → token_len ∈{66,69}. Empirical p ≪ 0.15; requested p is
  recorded honestly. Do not invent ER density that the band forbids.

Quotas: ≥32 pos (hop=20) + ≥32 hard-neg (50/50); default 64+64.
Fixed seed. science_open=false always.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    K_TRAIN_MAX,
    is_ood_hop,
)
from reachability_gen.encode import edge_hash as compute_edge_hash
from reachability_gen.encode import encode_instance
from reachability_gen.gen_covariate_matched_ood import (
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    _empirical_p,
    _encoding_token_len,
    _harvest_hard_negs_in_band,
    plant_path_graph,
)
from reachability_gen.graph import er_digraph, hop_distance, hop_distances_from
from reachability_gen.schema import ReachabilityExample
from reachability_gen.tokenize import split_encoding_tokens

RED_TEST_SEED: int = 168_000
RED_K: int = 20
RED_P_REQUESTED: float = 0.15
SPLIT_NAME: str = "sheaf_red_test_k20"
N_POS_DEFAULT: int = 64
N_NEG_DEFAULT: int = 64
# n support: path needs ≥21 nodes; keep small to maximize empirical p under band.
N_SUPPORT: tuple[int, ...] = (21, 22, 23)
EXTRA_EDGES: tuple[int, ...] = (0, 1)  # |E| ≤ 21 for band
ER_SEARCH_N: tuple[int, ...] = (21, 22, 24, 26, 28)
ER_SEARCH_TRIALS: int = 200


def _in_band(token_len: int) -> bool:
    return SEQ_LEN_MIN <= token_len <= SEQ_LEN_MAX


def _make_example(
    *,
    seed: int,
    n: int,
    p: float,
    edges: list[tuple[int, int]],
    s: int,
    t: int,
    y: int,
    hop: int,
    n_attempts: int = 1,
) -> ReachabilityExample:
    return ReachabilityExample(
        split=SPLIT_NAME,
        seed=seed,
        n=n,
        p=p,
        edge_hash=compute_edge_hash(edges),
        s=s,
        t=t,
        y=y,
        hop_distance=hop,
        is_ood=is_ood_hop(hop, k_train_max=K_TRAIN_MAX),
        encoding=encode_instance(n, edges, s, t),
        arm_id=None,
        arm_meta={
            "construction": "path_backbone",
            "p_requested": RED_P_REQUESTED,
            "p_empirical": p,
            "K": RED_K,
            "cycle": "CYCLE_SHEAF_RED_TEST",
        },
        n_attempts=n_attempts,
    )


def run_er_p015_band_search(
    *,
    seed: int = RED_TEST_SEED,
    trials_per_n: int = ER_SEARCH_TRIALS,
    p: float = RED_P_REQUESTED,
    k: int = RED_K,
) -> dict[str, Any]:
    """Document ER(p) × band feasibility for hop K (search only)."""
    rng = random.Random(seed ^ 0x4ED1)
    cells: list[dict[str, Any]] = []
    total_in_band = 0
    total_with_k = 0
    for n in ER_SEARCH_N:
        if n < k + 1:
            continue
        in_band = 0
        with_k = 0
        token_lens: list[int] = []
        for _ in range(trials_per_n):
            edges = er_digraph(n, p, rng)
            probe = _encoding_token_len(n, edges, 0, min(1, n - 1))
            if not _in_band(probe):
                continue
            in_band += 1
            token_lens.append(probe)
            for s in range(n):
                dists = hop_distances_from(n, edges, s)
                if any(d == k for d in dists):
                    with_k += 1
                    break
        total_in_band += in_band
        total_with_k += with_k
        cells.append(
            {
                "n": n,
                "p": p,
                "trials": trials_per_n,
                "in_band": in_band,
                "with_hop_k": with_k,
                "probe_token_len_mean": (
                    sum(token_lens) / len(token_lens) if token_lens else None
                ),
            }
        )
    return {
        "p_requested": p,
        "K": k,
        "seq_len_band": [SEQ_LEN_MIN, SEQ_LEN_MAX],
        "cells": cells,
        "total_in_band": total_in_band,
        "total_with_hop_k_in_band": total_with_k,
        "note": (
            "ER under sealed band is scarce/impossible for K=20 at p=0.15 "
            "(expected |E| ≫ band cap 21). Path-backbone used for locked cell."
            if total_with_k == 0
            else "ER found some in-band K hits; still prefer path-backbone for lock."
        ),
    }


def generate_sheaf_red_test_cell(
    *,
    seed: int = RED_TEST_SEED,
    n_pos: int = N_POS_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    max_graph_draws: int = 40_000,
    run_er_search: bool = True,
    er_search_trials: int = ER_SEARCH_TRIALS,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    """Generate balanced K=20 / hard-neg cell under seq_len band."""
    if n_pos < 1 or n_neg < 1:
        raise ValueError("n_pos and n_neg must be >= 1")
    if n_pos != n_neg:
        raise ValueError(
            f"50/50 balance requires n_pos == n_neg; got {n_pos} vs {n_neg}"
        )

    er_report: Optional[dict[str, Any]] = None
    if run_er_search:
        print("[red-test-gen] ER p=0.15 band search…", file=sys.stderr)
        er_report = run_er_p015_band_search(
            seed=seed, trials_per_n=er_search_trials
        )
        print(
            f"[red-test-gen] ER in_band={er_report['total_in_band']} "
            f"with_K20={er_report['total_with_hop_k_in_band']}",
            file=sys.stderr,
        )

    master = random.Random(seed)
    pos_pool: list[tuple[int, float, list[tuple[int, int]], int, int]] = []
    neg_pool: list[tuple[int, float, list[tuple[int, int]], int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    draws = 0
    reject_reasons: Counter = Counter()

    def _key(edges, s, t) -> tuple[str, int, int]:
        return (compute_edge_hash(edges), s, t)

    while len(pos_pool) < n_pos or len(neg_pool) < n_neg:
        if draws >= max_graph_draws:
            break
        draws += 1
        prefer_pos = len(pos_pool) < n_pos and (
            len(neg_pool) >= n_neg or master.random() < 0.75
        )
        n = master.choice(N_SUPPORT)
        if n < RED_K + 1:
            n = RED_K + 1
        extra = master.choice(EXTRA_EDGES)
        max_extra = max(0, 21 - RED_K)
        extra = min(extra, max_extra)
        edges, s_path, t_path = plant_path_graph(n, RED_K, extra, master)
        p_emp = _empirical_p(n, edges)

        if prefer_pos:
            tl = _encoding_token_len(n, edges, s_path, t_path)
            if not _in_band(tl):
                reject_reasons["pos_out_of_band"] += 1
                continue
            d = hop_distance(n, edges, s_path, t_path)
            if d != RED_K:
                reject_reasons["hop_mismatch"] += 1
                continue
            key = _key(edges, s_path, t_path)
            if key in seen:
                reject_reasons["dup_pos"] += 1
                continue
            seen.add(key)
            pos_pool.append((n, p_emp, list(edges), s_path, t_path))
        else:
            negs = _harvest_hard_negs_in_band(n, edges)
            master.shuffle(negs)
            added = 0
            for s, t in negs:
                if len(neg_pool) >= n_neg:
                    break
                key = _key(edges, s, t)
                if key in seen:
                    reject_reasons["dup_neg"] += 1
                    continue
                seen.add(key)
                neg_pool.append((n, p_emp, list(edges), s, t))
                added += 1
            if added == 0:
                reject_reasons["no_hard_neg_in_band"] += 1

        if draws % 2000 == 0:
            print(
                f"[red-test-gen] draws={draws} pos={len(pos_pool)}/{n_pos} "
                f"neg={len(neg_pool)}/{n_neg}",
                file=sys.stderr,
            )

    if len(pos_pool) < n_pos or len(neg_pool) < n_neg:
        raise RuntimeError(
            f"RED_TEST cell shortfall: pos={len(pos_pool)}/{n_pos} "
            f"neg={len(neg_pool)}/{n_neg} after {draws} draws; "
            f"rejects={dict(reject_reasons)}"
        )

    examples: list[ReachabilityExample] = []
    for i, (n, p_emp, edges, s, t) in enumerate(pos_pool[:n_pos]):
        examples.append(
            _make_example(
                seed=seed + i,
                n=n,
                p=p_emp,
                edges=edges,
                s=s,
                t=t,
                y=1,
                hop=RED_K,
            )
        )
    for i, (n, p_emp, edges, s, t) in enumerate(neg_pool[:n_neg]):
        examples.append(
            _make_example(
                seed=seed + 10_000 + i,
                n=n,
                p=p_emp,
                edges=edges,
                s=s,
                t=t,
                y=0,
                hop=HOP_UNREACHABLE,
            )
        )

    token_lens = [
        len(split_encoding_tokens(ex.encoding)) for ex in examples
    ]
    emp_ps = [float(ex.p) for ex in examples]
    report: dict[str, Any] = {
        "cycle": "CYCLE_SHEAF_RED_TEST",
        "science_open": False,
        "split": SPLIT_NAME,
        "seed": seed,
        "K": RED_K,
        "p_requested": RED_P_REQUESTED,
        "construction": "path_backbone",
        "n_support": list(N_SUPPORT),
        "extra_edges": list(EXTRA_EDGES),
        "seq_len_band": [SEQ_LEN_MIN, SEQ_LEN_MAX],
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_examples": len(examples),
        "draws": draws,
        "reject_reasons": dict(reject_reasons),
        "token_len": {
            "min": min(token_lens),
            "max": max(token_lens),
            "mean": sum(token_lens) / len(token_lens),
            "n": len(token_lens),
            "note": (
                "K=20 forces |E|≥20 → token_len≥66 under locked encoding; "
                "mean sits at the high end of sealed band [45,70], far from "
                "the ~200 Gate2 confound. INVALID if mean drifts toward ~200."
            ),
        },
        "p_empirical": {
            "min": min(emp_ps),
            "max": max(emp_ps),
            "mean": sum(emp_ps) / len(emp_ps),
            "note": (
                "Empirical p ≪ p_requested=0.15 under band+|E| cap; "
                "joint (K=20, p=0.15, band) is infeasible for pure ER."
            ),
        },
        "er_band_search": er_report,
        "hard_neg_discipline": True,
        "science_open": False,
    }
    return examples, report


def write_jsonl(path: Path, examples: Sequence[ReachabilityExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), sort_keys=True) + "\n")
    return len(examples)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate CYCLE_SHEAF_RED_TEST K=20 cell (MEASURE)"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/sheaf_red_test_k20.jsonl"),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/sheaf_red_test_generation_report.json"),
    )
    p.add_argument("--seed", type=int, default=RED_TEST_SEED)
    p.add_argument("--n-pos", type=int, default=N_POS_DEFAULT)
    p.add_argument("--n-neg", type=int, default=N_NEG_DEFAULT)
    p.add_argument("--max-graph-draws", type=int, default=40_000)
    p.add_argument("--skip-er-search", action="store_true")
    p.add_argument("--er-search-trials", type=int, default=ER_SEARCH_TRIALS)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    examples, report = generate_sheaf_red_test_cell(
        seed=args.seed,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        max_graph_draws=args.max_graph_draws,
        run_er_search=not args.skip_er_search,
        er_search_trials=args.er_search_trials,
    )
    n = write_jsonl(args.out, examples)
    write_report(args.report, report)
    print(
        f"wrote {n} → {args.out}; report → {args.report}; "
        f"token_len mean={report['token_len']['mean']:.2f}; "
        f"p_emp mean={report['p_empirical']['mean']:.4f}",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "n": n,
                "token_len_mean": report["token_len"]["mean"],
                "p_empirical_mean": report["p_empirical"]["mean"],
                "science_open": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
