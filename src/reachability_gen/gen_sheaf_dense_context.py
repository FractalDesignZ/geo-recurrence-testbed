"""Generate CYCLE_SHEAF_DENSE_CONTEXT cells (MEASURE, fail-closed).

Cell 1 Dense: n=16, true ER p=0.15, K=8, seq_len band [100,140].
Cell 2 Matched sparse: n=32, ER p≈0.036 tuned so |E| and seq_len match Cell1
empirically in-band; K=8.

Hard-negatives: rejection-sampled; assert v ∉ R_out(u) via full BFS and DFS.
No path-backbone. science_open=false always. No HF.
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
from reachability_gen.gen_covariate_matched_ood import _empirical_p
from reachability_gen.gen_sheaf_density_stress import (
    assert_genuinely_unreachable,
    _harvest_hard_negs_rejection,
    _harvest_hop_k_pairs,
)
from reachability_gen.graph import er_digraph, hop_distance
from reachability_gen.schema import ReachabilityExample
from reachability_gen.tokenize import split_encoding_tokens

DENSE_SEED: int = 191_000
DENSE_K: int = 8
CELL1_N: int = 16
CELL1_P: float = 0.15
CELL2_N: int = 32
CELL2_P_SEED: float = 0.036  # starting guess; tuned to match Cell1
N_POS_DEFAULT: int = 64
N_NEG_DEFAULT: int = 64
SEQ_LEN_BAND: tuple[int, int] = (100, 140)
SEQ_LEN_TARGET_MEAN: tuple[float, float] = (110.0, 120.0)
SPLIT_CELL1: str = "sheaf_dense_context_cell1"
SPLIT_CELL2: str = "sheaf_dense_context_cell2"


def _token_len(encoding: str) -> int:
    return len(split_encoding_tokens(encoding))


def _make_example(
    *,
    split: str,
    seed: int,
    n: int,
    p: float,
    edges: list[tuple[int, int]],
    s: int,
    t: int,
    y: int,
    hop: int,
    cell_id: str,
    p_requested: float,
    cycle: str = "CYCLE_SHEAF_DENSE_CONTEXT",
) -> ReachabilityExample:
    return ReachabilityExample(
        split=split,
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
            "p_requested": p_requested,
            "p_empirical": p,
            "K": DENSE_K,
            "n_fixed": n,
            "cell": cell_id,
            "cycle": cycle,
            "path_backbone": False,
            "seq_len_band": list(SEQ_LEN_BAND),
        },
        n_attempts=1,
    )


def generate_er_balanced_cell(
    *,
    seed: int,
    n: int,
    p: float,
    k: int = DENSE_K,
    n_pos: int = N_POS_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    split: str,
    cell_id: str,
    max_graph_draws: int = 100_000,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    """Balanced K-hop / hard-neg under true ER digraph (no backbone)."""
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

        if draws % 5000 == 0:
            print(
                f"[dense-ctx-gen:{cell_id}] draws={draws} "
                f"pos={len(pos_pool)}/{n_pos} neg={len(neg_pool)}/{n_neg}",
                file=sys.stderr,
            )

    if len(pos_pool) < n_pos or len(neg_pool) < n_neg:
        raise RuntimeError(
            f"DENSE_CONTEXT {cell_id} shortfall: pos={len(pos_pool)}/{n_pos} "
            f"neg={len(neg_pool)}/{n_neg} after {draws} draws n={n} p={p}; "
            f"rejects={dict(reject_reasons)}"
        )

    examples: list[ReachabilityExample] = []
    for i, (p_emp, edges, s, t) in enumerate(pos_pool[:n_pos]):
        examples.append(
            _make_example(
                split=split,
                seed=seed + i,
                n=n,
                p=p_emp,
                edges=edges,
                s=s,
                t=t,
                y=1,
                hop=k,
                cell_id=cell_id,
                p_requested=p,
            )
        )
    for i, (p_emp, edges, s, t) in enumerate(neg_pool[:n_neg]):
        assert_genuinely_unreachable(n, edges, s, t)
        examples.append(
            _make_example(
                split=split,
                seed=seed + 10_000 + i,
                n=n,
                p=p_emp,
                edges=edges,
                s=s,
                t=t,
                y=0,
                hop=HOP_UNREACHABLE,
                cell_id=cell_id,
                p_requested=p,
            )
        )

    from reachability_gen.encode import parse_instance

    for ex in examples:
        if int(ex.y) != 0:
            continue
        n_e, edges_e, s_e, t_e = parse_instance(ex.encoding)
        assert_genuinely_unreachable(n_e, edges_e, s_e, t_e)

    report = _cell_report(
        examples=examples,
        cell_id=cell_id,
        split=split,
        seed=seed,
        n=n,
        p_requested=p,
        k=k,
        n_pos=n_pos,
        n_neg=n_neg,
        draws=draws,
        reject_reasons=dict(reject_reasons),
    )
    return examples, report


def _cell_report(
    *,
    examples: Sequence[ReachabilityExample],
    cell_id: str,
    split: str,
    seed: int,
    n: int,
    p_requested: float,
    k: int,
    n_pos: int,
    n_neg: int,
    draws: int,
    reject_reasons: dict[str, int],
) -> dict[str, Any]:
    token_lens = [_token_len(ex.encoding) for ex in examples]
    emp_ps = [float(ex.p) for ex in examples]
    edge_counts = []
    from reachability_gen.encode import parse_instance

    for ex in examples:
        _, edges, _, _ = parse_instance(ex.encoding)
        edge_counts.append(len(edges))
    mean_tl = sum(token_lens) / len(token_lens)
    mean_e = sum(edge_counts) / len(edge_counts)
    mean_p = sum(emp_ps) / len(emp_ps)
    mean_in_band = SEQ_LEN_BAND[0] <= mean_tl <= SEQ_LEN_BAND[1]
    all_in_band = all(SEQ_LEN_BAND[0] <= L <= SEQ_LEN_BAND[1] for L in token_lens)
    return {
        "cycle": "CYCLE_SHEAF_DENSE_CONTEXT",
        "science_open": False,
        "cell": cell_id,
        "split": split,
        "seed": seed,
        "K": k,
        "n": n,
        "p_requested": p_requested,
        "construction": "true_er_digraph",
        "path_backbone": False,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_examples": len(examples),
        "draws": draws,
        "reject_reasons": reject_reasons,
        "seq_len_band": list(SEQ_LEN_BAND),
        "seq_len_target_mean": list(SEQ_LEN_TARGET_MEAN),
        "token_len": {
            "min": min(token_lens),
            "max": max(token_lens),
            "mean": mean_tl,
            "n": len(token_lens),
            "mean_in_band": mean_in_band,
            "all_in_band": all_in_band,
        },
        "edge_count": {
            "min": min(edge_counts),
            "max": max(edge_counts),
            "mean": mean_e,
        },
        "p_empirical": {
            "min": min(emp_ps),
            "max": max(emp_ps),
            "mean": mean_p,
            "near_requested": abs(mean_p - p_requested) <= 0.03,
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


def _probe_seq_len_mean(n: int, p: float, seed: int, n_graphs: int = 40) -> float:
    """Quick Monte Carlo of encoding token_len mean under ER(n,p)."""
    rng = random.Random(seed)
    lens: list[int] = []
    for _ in range(n_graphs):
        edges = er_digraph(n, p, rng)
        enc = encode_instance(n, edges, 0, min(1, n - 1))
        lens.append(_token_len(enc))
    return sum(lens) / len(lens)


def tune_cell2_p(
    *,
    target_seq_mean: float,
    target_edge_mean: float,
    seed: int = DENSE_SEED + 50,
    p_init: float = CELL2_P_SEED,
    n: int = CELL2_N,
) -> dict[str, Any]:
    """Tune Cell2 ER p so seq_len / |E| approximate Cell1 targets."""
    # Expected |E| = p * n * (n-1). Match edge count first, then refine on seq_len.
    n_pairs = n * (n - 1)
    p_from_e = max(0.005, min(0.2, target_edge_mean / n_pairs))
    candidates = sorted(
        {
            round(p_init, 4),
            round(p_from_e, 4),
            round(p_from_e - 0.002, 4),
            round(p_from_e + 0.002, 4),
            round(p_from_e - 0.004, 4),
            round(p_from_e + 0.004, 4),
            round(p_from_e - 0.006, 4),
            round(p_from_e + 0.006, 4),
            0.030,
            0.032,
            0.034,
            0.036,
            0.038,
            0.040,
        }
    )
    candidates = [c for c in candidates if 0.005 <= c <= 0.2]
    best_p = p_init
    best_err = float("inf")
    probes: list[dict[str, Any]] = []
    for i, p in enumerate(candidates):
        mean_tl = _probe_seq_len_mean(n, p, seed + i * 17)
        # Approximate |E| from p.
        e_mean = p * n_pairs
        err = abs(mean_tl - target_seq_mean) + 0.15 * abs(e_mean - target_edge_mean)
        probes.append(
            {
                "p": p,
                "seq_len_probe_mean": mean_tl,
                "E_expected": e_mean,
                "err": err,
            }
        )
        if err < best_err:
            best_err = err
            best_p = p
    return {
        "p_tuned": best_p,
        "p_init": p_init,
        "p_from_edge_match": p_from_e,
        "target_seq_mean": target_seq_mean,
        "target_edge_mean": target_edge_mean,
        "best_err": best_err,
        "probes": probes,
    }


def generate_dense_context_pair(
    *,
    seed: int = DENSE_SEED,
    n_pos: int = N_POS_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    max_graph_draws: int = 100_000,
) -> dict[str, Any]:
    """Generate Cell1 dense + Cell2 matched sparse; return examples + reports."""
    cell1_ex, cell1_rep = generate_er_balanced_cell(
        seed=seed,
        n=CELL1_N,
        p=CELL1_P,
        k=DENSE_K,
        n_pos=n_pos,
        n_neg=n_neg,
        split=SPLIT_CELL1,
        cell_id="cell1_dense",
        max_graph_draws=max_graph_draws,
    )
    tune = tune_cell2_p(
        target_seq_mean=float(cell1_rep["token_len"]["mean"]),
        target_edge_mean=float(cell1_rep["edge_count"]["mean"]),
        seed=seed + 50,
        p_init=CELL2_P_SEED,
        n=CELL2_N,
    )
    cell2_p = float(tune["p_tuned"])
    cell2_ex, cell2_rep = generate_er_balanced_cell(
        seed=seed + 1000,
        n=CELL2_N,
        p=cell2_p,
        k=DENSE_K,
        n_pos=n_pos,
        n_neg=n_neg,
        split=SPLIT_CELL2,
        cell_id="cell2_matched_sparse",
        max_graph_draws=max_graph_draws,
    )
    cell2_rep["p_tune"] = tune
    both_in_band = bool(
        cell1_rep["token_len"]["mean_in_band"] and cell2_rep["token_len"]["mean_in_band"]
    )
    return {
        "cell1": {"examples": cell1_ex, "report": cell1_rep},
        "cell2": {"examples": cell2_ex, "report": cell2_rep},
        "p_tune": tune,
        "both_seq_len_in_band": both_in_band,
        "seq_len_band": list(SEQ_LEN_BAND),
        "science_open": False,
    }


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
        description="Generate CYCLE_SHEAF_DENSE_CONTEXT Cell1+Cell2"
    )
    p.add_argument(
        "--out-cell1",
        type=Path,
        default=Path("data/sheaf_dense_context_cell1.jsonl"),
    )
    p.add_argument(
        "--out-cell2",
        type=Path,
        default=Path("data/sheaf_dense_context_cell2.jsonl"),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/sheaf_dense_context_generation_report.json"),
    )
    p.add_argument("--seed", type=int, default=DENSE_SEED)
    p.add_argument("--n-pos", type=int, default=N_POS_DEFAULT)
    p.add_argument("--n-neg", type=int, default=N_NEG_DEFAULT)
    p.add_argument("--max-graph-draws", type=int, default=100_000)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    pair = generate_dense_context_pair(
        seed=args.seed,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        max_graph_draws=args.max_graph_draws,
    )
    n1 = write_jsonl(args.out_cell1, pair["cell1"]["examples"])
    n2 = write_jsonl(args.out_cell2, pair["cell2"]["examples"])
    report = {
        "cycle": "CYCLE_SHEAF_DENSE_CONTEXT",
        "science_open": False,
        "seed": args.seed,
        "seq_len_band": list(SEQ_LEN_BAND),
        "both_seq_len_in_band": pair["both_seq_len_in_band"],
        "p_tune": pair["p_tune"],
        "cell1": pair["cell1"]["report"],
        "cell2": pair["cell2"]["report"],
    }
    write_report(args.report, report)
    print(
        f"wrote cell1={n1} → {args.out_cell1}; cell2={n2} → {args.out_cell2}; "
        f"report → {args.report}; "
        f"seq1={report['cell1']['token_len']['mean']:.2f} "
        f"seq2={report['cell2']['token_len']['mean']:.2f} "
        f"p2={report['cell2']['p_requested']:.4f} "
        f"both_in_band={pair['both_seq_len_in_band']}",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "n_cell1": n1,
                "n_cell2": n2,
                "seq_len_cell1": report["cell1"]["token_len"]["mean"],
                "seq_len_cell2": report["cell2"]["token_len"]["mean"],
                "p_cell2": report["cell2"]["p_requested"],
                "both_in_band": pair["both_seq_len_in_band"],
                "science_open": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
