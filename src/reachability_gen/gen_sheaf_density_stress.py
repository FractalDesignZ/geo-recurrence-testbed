"""Generate CYCLE_SHEAF_DENSITY_STRESS cell: true ER digraph p=0.15 (MEASURE).

Cell: n=16, K=8, Bernoulli ER digraph p=0.15. **No path-backbone** — do not
drop density to force the sealed seq_len band. If seq_len mean cannot land in
[45, 70] under true p=0.15, the runner marks INVALID and stops (fail-closed).

Hard-negatives: rejection-sampled with full BFS/DFS out-closure. Assert
``v not in R_out(u)`` (genuinely unreachable). science_open=false always.
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
)
from reachability_gen.graph import (
    er_digraph,
    hop_distance,
    hop_distances_from,
    reachable_bfs,
    reachable_dfs,
    reachable_out_set,
)
from reachability_gen.hard_negatives import is_hard_negative, total_degrees
from reachability_gen.schema import ReachabilityExample
from reachability_gen.tokenize import split_encoding_tokens

DENSITY_SEED: int = 189_000
DENSITY_K: int = 8
DENSITY_N: int = 16
DENSITY_P: float = 0.15
SPLIT_NAME: str = "sheaf_density_stress_p015"
N_POS_DEFAULT: int = 64
N_NEG_DEFAULT: int = 64
SEQ_LEN_BAND = (SEQ_LEN_MIN, SEQ_LEN_MAX)


def assert_genuinely_unreachable(
    n: int,
    edges: Sequence[tuple[int, int]],
    u: int,
    v: int,
) -> None:
    """Hard-neg invariant: v ∉ R_out(u) under both BFS and DFS closures."""
    r_bfs = reachable_out_set(n, edges, u, method="bfs")
    r_dfs = reachable_out_set(n, edges, u, method="dfs")
    assert r_bfs == r_dfs, (
        f"R_out BFS/DFS mismatch for u={u}: bfs={sorted(r_bfs)} dfs={sorted(r_dfs)}"
    )
    assert v not in r_bfs, (
        f"hard-neg violation: v={v} ∈ R_out({u}) (BFS); not genuinely unreachable"
    )
    assert v not in r_dfs, (
        f"hard-neg violation: v={v} ∈ R_out({u}) (DFS); not genuinely unreachable"
    )
    assert not reachable_bfs(n, edges, u, v), "reachable_bfs True for hard-neg"
    assert not reachable_dfs(n, edges, u, v), "reachable_dfs True for hard-neg"
    ok, reason = is_hard_negative(n, edges, u, v, y=0)
    assert ok, f"is_hard_negative rejected: {reason}"


def _harvest_hop_k_pairs(
    n: int,
    edges: Sequence[tuple[int, int]],
    k: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for s in range(n):
        dists = hop_distances_from(n, edges, s)
        for t, d in enumerate(dists):
            if d == k:
                pairs.append((s, t))
    return pairs


def _harvest_hard_negs_rejection(
    n: int,
    edges: Sequence[tuple[int, int]],
    rng: random.Random,
    *,
    max_tries: int = 256,
) -> list[tuple[int, int]]:
    """Rejection-sample hard-negs; each candidate checked via full R_out BFS/DFS."""
    deg = total_degrees(n, edges)
    out_sets = {
        u: reachable_out_set(n, edges, u, method="bfs") for u in range(n)
    }
    # Cross-check DFS agreement on a sample of sources (all for n=16).
    for u in range(n):
        r_dfs = reachable_out_set(n, edges, u, method="dfs")
        assert out_sets[u] == r_dfs, f"R_out mismatch at u={u}"

    candidates: list[tuple[int, int]] = []
    for u in range(n):
        if deg[u] < 1:
            continue
        r_out = out_sets[u]
        for v in range(n):
            if u == v:
                continue
            if deg[v] < 1:
                continue
            if v in r_out:
                continue
            candidates.append((u, v))

    rng.shuffle(candidates)
    accepted: list[tuple[int, int]] = []
    for u, v in candidates[:max_tries]:
        try:
            assert_genuinely_unreachable(n, edges, u, v)
        except AssertionError:
            continue
        accepted.append((u, v))
    return accepted


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
            "construction": "true_er_digraph",
            "p_requested": DENSITY_P,
            "p_empirical": p,
            "K": DENSITY_K,
            "n_fixed": DENSITY_N,
            "cycle": "CYCLE_SHEAF_DENSITY_STRESS",
            "path_backbone": False,
        },
        n_attempts=1,
    )


def generate_sheaf_density_stress_cell(
    *,
    seed: int = DENSITY_SEED,
    n_pos: int = N_POS_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    n: int = DENSITY_N,
    p: float = DENSITY_P,
    k: int = DENSITY_K,
    max_graph_draws: int = 80_000,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    """Generate balanced K-hop / hard-neg cell under **true** ER digraph p.

    No path-backbone. No density drop to force seq_len band. Caller / runner
    must mark INVALID if token_len mean ∉ [45, 70].
    """
    if n_pos < 1 or n_neg < 1:
        raise ValueError("n_pos and n_neg must be >= 1")
    if n_pos != n_neg:
        raise ValueError(
            f"50/50 balance requires n_pos == n_neg; got {n_pos} vs {n_neg}"
        )
    if n < k + 1:
        raise ValueError(f"n={n} too small for hop K={k}")

    master = random.Random(seed)
    pos_pool: list[tuple[float, list[tuple[int, int]], int, int]] = []
    neg_pool: list[tuple[float, list[tuple[int, int]], int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    draws = 0
    reject_reasons: Counter = Counter()

    def _key(edges, s, t) -> tuple[str, int, int]:
        return (compute_edge_hash(edges), s, t)

    while len(pos_pool) < n_pos or len(neg_pool) < n_neg:
        if draws >= max_graph_draws:
            break
        draws += 1
        edges = er_digraph(n, p, master)
        p_emp = _empirical_p(n, edges)

        prefer_pos = len(pos_pool) < n_pos and (
            len(neg_pool) >= n_neg or master.random() < 0.55
        )

        if prefer_pos:
            pairs = _harvest_hop_k_pairs(n, edges, k)
            if not pairs:
                reject_reasons["no_hop_k"] += 1
            else:
                master.shuffle(pairs)
                added = False
                for s, t in pairs:
                    d = hop_distance(n, edges, s, t)
                    if d != k:
                        reject_reasons["hop_mismatch"] += 1
                        continue
                    key = _key(edges, s, t)
                    if key in seen:
                        reject_reasons["dup_pos"] += 1
                        continue
                    seen.add(key)
                    pos_pool.append((p_emp, list(edges), s, t))
                    added = True
                    break
                if not added:
                    reject_reasons["pos_dup_exhausted"] += 1
        else:
            negs = _harvest_hard_negs_rejection(n, edges, master)
            if not negs:
                reject_reasons["no_hard_neg"] += 1
            else:
                added = False
                for s, t in negs:
                    if len(neg_pool) >= n_neg:
                        break
                    key = _key(edges, s, t)
                    if key in seen:
                        reject_reasons["dup_neg"] += 1
                        continue
                    assert_genuinely_unreachable(n, edges, s, t)
                    seen.add(key)
                    neg_pool.append((p_emp, list(edges), s, t))
                    added = True
                    break
                if not added:
                    reject_reasons["neg_dup_exhausted"] += 1

        if draws % 2000 == 0:
            print(
                f"[density-gen] draws={draws} pos={len(pos_pool)}/{n_pos} "
                f"neg={len(neg_pool)}/{n_neg}",
                file=sys.stderr,
            )

    if len(pos_pool) < n_pos or len(neg_pool) < n_neg:
        raise RuntimeError(
            f"DENSITY_STRESS cell shortfall: pos={len(pos_pool)}/{n_pos} "
            f"neg={len(neg_pool)}/{n_neg} after {draws} draws; "
            f"rejects={dict(reject_reasons)}"
        )

    examples: list[ReachabilityExample] = []
    for i, (p_emp, edges, s, t) in enumerate(pos_pool[:n_pos]):
        examples.append(
            _make_example(
                seed=seed + i,
                n=n,
                p=p_emp,
                edges=edges,
                s=s,
                t=t,
                y=1,
                hop=k,
            )
        )
    for i, (p_emp, edges, s, t) in enumerate(neg_pool[:n_neg]):
        assert_genuinely_unreachable(n, edges, s, t)
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

    # Final hard-neg assert pass over written examples.
    for ex in examples:
        if int(ex.y) != 0:
            continue
        from reachability_gen.encode import parse_instance

        n_e, edges_e, s_e, t_e = parse_instance(ex.encoding)
        assert_genuinely_unreachable(n_e, edges_e, s_e, t_e)

    token_lens = [
        len(split_encoding_tokens(ex.encoding)) for ex in examples
    ]
    emp_ps = [float(ex.p) for ex in examples]
    mean_tl = sum(token_lens) / len(token_lens)
    mean_in_band = SEQ_LEN_BAND[0] <= mean_tl <= SEQ_LEN_BAND[1]
    all_in_band = all(SEQ_LEN_BAND[0] <= L <= SEQ_LEN_BAND[1] for L in token_lens)
    mean_p = sum(emp_ps) / len(emp_ps)

    report: dict[str, Any] = {
        "cycle": "CYCLE_SHEAF_DENSITY_STRESS",
        "science_open": False,
        "split": SPLIT_NAME,
        "seed": seed,
        "K": k,
        "n": n,
        "p_requested": p,
        "construction": "true_er_digraph",
        "path_backbone": False,
        "path_backbone_forbidden": True,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_examples": len(examples),
        "draws": draws,
        "reject_reasons": dict(reject_reasons),
        "seq_len_band": list(SEQ_LEN_BAND),
        "token_len": {
            "min": min(token_lens),
            "max": max(token_lens),
            "mean": mean_tl,
            "n": len(token_lens),
            "mean_in_band": mean_in_band,
            "all_in_band": all_in_band,
            "note": (
                "Under true ER p=0.15 @ n=16, expected |E|≈36 → token_len≈114; "
                "sealed band [45,70] requires |E|∈[13,21]. If mean ∉ band, "
                "runner marks INVALID — do not fake density with path-backbone."
            ),
        },
        "p_empirical": {
            "min": min(emp_ps),
            "max": max(emp_ps),
            "mean": mean_p,
            "near_requested": abs(mean_p - p) <= 0.03,
            "note": (
                "Empirical p from |E|/(n(n-1)) on accepted ER draws; "
                "must stay near p_requested (no backbone density drop)."
            ),
        },
        "hard_neg_discipline": {
            "rejection_sampled": True,
            "R_out_bfs_dfs_assert": True,
            "degree_filter": "deg(s)>=1 and deg(t)>=1",
        },
        "feasibility": {
            "seq_len_mean_in_band": mean_in_band,
            "would_be_invalid_if_mean_out_of_band": not mean_in_band,
        },
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
        description="Generate CYCLE_SHEAF_DENSITY_STRESS true-ER p=0.15 cell"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/sheaf_density_stress_p015.jsonl"),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/sheaf_density_stress_generation_report.json"),
    )
    p.add_argument("--seed", type=int, default=DENSITY_SEED)
    p.add_argument("--n-pos", type=int, default=N_POS_DEFAULT)
    p.add_argument("--n-neg", type=int, default=N_NEG_DEFAULT)
    p.add_argument("--max-graph-draws", type=int, default=80_000)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    examples, report = generate_sheaf_density_stress_cell(
        seed=args.seed,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        max_graph_draws=args.max_graph_draws,
    )
    n = write_jsonl(args.out, examples)
    write_report(args.report, report)
    print(
        f"wrote {n} → {args.out}; report → {args.report}; "
        f"token_len mean={report['token_len']['mean']:.2f}; "
        f"p_emp mean={report['p_empirical']['mean']:.4f}; "
        f"mean_in_band={report['token_len']['mean_in_band']}",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "n": n,
                "token_len_mean": report["token_len"]["mean"],
                "p_empirical_mean": report["p_empirical"]["mean"],
                "mean_in_band": report["token_len"]["mean_in_band"],
                "science_open": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
