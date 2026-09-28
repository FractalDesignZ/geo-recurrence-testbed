"""Generate RED held-out OOD for CYCLE_STALK_RED_COMPETENT_DISSONANCE.

Pushes p/K outside train priors (K≤6; ID short-hop p support), harder than
``ood_hops`` where feasible:

  - DENSE_K16: path-backbone K=16 + distractors → empirical p ≫ ood_hops K16
  - LONG_K20:  path-backbone K=20 (outside ADR OOD_HOP_VALUES {8,12,16})

Token budget ≤250 (bound30 headroom) — NOT matched-OOD band (RED substrate).
science_open=false always. MEASURE plumbing only.
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
    _empirical_p,
    _encoding_token_len,
    plant_path_graph,
)
from reachability_gen.graph import hop_distance
from reachability_gen.hard_negatives import is_hard_negative, total_degrees
from reachability_gen.schema import ReachabilityExample
from reachability_gen.splits import derive_example_seed
from reachability_gen.tokenize import split_encoding_tokens

RED_SEED: int = 210_000
SPLIT_NAME: str = "stalk_red_competent_dissonance"
MAX_TOKENS_ACCEPT: int = 250
BOUND30_MAX_LEN: int = 257

# Cells (locked)
K16: int = 16
K20: int = 20
RED_HOPS: tuple[int, ...] = (K16, K20)

POS_PER_CELL_DEFAULT: int = 64
# 50/50 → n_neg = 2 * pos_per_cell

# DENSE_K16: larger n + many extras → denser empirical p than ood_hops K16 (≤0.025)
N_SUPPORT_K16: tuple[int, ...] = (20, 22, 24, 26, 28)
EXTRA_EDGES_K16: tuple[int, ...] = (16, 20, 24, 28, 32, 40, 48)

# LONG_K20: path needs ≥21 nodes; few extras keep hop=20
N_SUPPORT_K20: tuple[int, ...] = (28, 32, 36, 40, 48)
EXTRA_EDGES_K20: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8)

N_SUPPORT_NEG: tuple[int, ...] = (24, 28, 32, 36, 40, 48)
EXTRA_EDGES_NEG: tuple[int, ...] = (4, 8, 12, 16, 20, 24)


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
    cell: str,
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
            "cell": cell,
            "construction": "path_backbone",
            "p_empirical": p,
            "cycle": "CYCLE_STALK_RED_COMPETENT_DISSONANCE",
            "note": "RED substrate; not matched-OOD band; science_open=false",
        },
        n_attempts=n_attempts,
    )


def _harvest_hard_negs(
    n: int,
    edges: Sequence[tuple[int, int]],
    *,
    max_tokens: int = MAX_TOKENS_ACCEPT,
    max_pairs: int = 32,
) -> list[tuple[int, int]]:
    deg = total_degrees(n, edges)
    out: list[tuple[int, int]] = []
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            if deg[s] < 1 or deg[t] < 1:
                continue
            d = hop_distance(n, list(edges), s, t)
            if d is not None:
                continue
            ok, _ = is_hard_negative(n, edges, s, t, y=0)
            if not ok:
                continue
            if _encoding_token_len(n, edges, s, t) > max_tokens:
                continue
            out.append((s, t))
            if len(out) >= max_pairs:
                return out
    return out


def generate_red(
    *,
    seed: int = RED_SEED,
    pos_per_cell: int = POS_PER_CELL_DEFAULT,
    max_draws: int = 80_000,
    progress_every: int = 500,
    max_tokens: int = MAX_TOKENS_ACCEPT,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    if pos_per_cell < 1:
        raise ValueError("pos_per_cell must be >= 1")
    n_pos_total = pos_per_cell * len(RED_HOPS)
    n_neg = n_pos_total  # 50/50

    master = random.Random(seed)
    pos_pools: dict[int, list[tuple[int, float, list[tuple[int, int]], int, int, str]]] = {
        k: [] for k in RED_HOPS
    }
    neg_pool: list[tuple[int, float, list[tuple[int, int]], int, int, str]] = []
    seen: set[tuple[str, int, int]] = set()
    draws = 0
    reject: Counter = Counter()

    def _key(edges, s, t) -> tuple[str, int, int]:
        return (compute_edge_hash(edges), s, t)

    while any(len(pos_pools[k]) < pos_per_cell for k in RED_HOPS) or len(neg_pool) < n_neg:
        if draws >= max_draws:
            break
        draws += 1
        scarce = [k for k in RED_HOPS if len(pos_pools[k]) < pos_per_cell]
        want_neg = len(neg_pool) < n_neg

        if scarce and (not want_neg or master.random() < 0.7):
            scarce.sort(key=lambda k: pos_per_cell - len(pos_pools[k]), reverse=True)
            k = scarce[0]
            if k == K16:
                n = master.choice(N_SUPPORT_K16)
                n_extra = master.choice(EXTRA_EDGES_K16)
                cell = "DENSE_K16"
            else:
                n = master.choice(N_SUPPORT_K20)
                n_extra = master.choice(EXTRA_EDGES_K20)
                cell = "LONG_K20"
            if n < k + 1:
                n = k + 1
            try:
                edges, s, t = plant_path_graph(n, k, n_extra, master)
            except ValueError:
                reject["plant_fail"] += 1
                continue
            tok = _encoding_token_len(n, edges, s, t)
            if tok > max_tokens:
                reject["overlong"] += 1
                continue
            d = hop_distance(n, edges, s, t)
            if d != k:
                reject["hop_mismatch"] += 1
                continue
            key = _key(edges, s, t)
            if key in seen:
                reject["dup_pos"] += 1
            else:
                seen.add(key)
                p_emp = _empirical_p(n, edges)
                pos_pools[k].append((n, p_emp, edges, s, t, cell))
            # opportunistic negs from same graph
            if want_neg:
                for ns, nt in _harvest_hard_negs(n, edges, max_tokens=max_tokens):
                    if len(neg_pool) >= n_neg:
                        break
                    nk = _key(edges, ns, nt)
                    if nk in seen:
                        continue
                    seen.add(nk)
                    neg_pool.append(
                        (n, _empirical_p(n, edges), edges, ns, nt, "HARD_NEG")
                    )
        else:
            # dedicated neg draws
            n = master.choice(N_SUPPORT_NEG)
            n_extra = master.choice(EXTRA_EDGES_NEG)
            # plant a short path then harvest hard negs elsewhere
            k_plant = min(8, n - 1)
            try:
                edges, _s, _t = plant_path_graph(n, k_plant, n_extra, master)
            except ValueError:
                reject["plant_fail_neg"] += 1
                continue
            harvested = _harvest_hard_negs(n, edges, max_tokens=max_tokens)
            if not harvested:
                reject["no_hard_neg"] += 1
                continue
            master.shuffle(harvested)
            for ns, nt in harvested:
                if len(neg_pool) >= n_neg:
                    break
                nk = _key(edges, ns, nt)
                if nk in seen:
                    continue
                seen.add(nk)
                neg_pool.append(
                    (n, _empirical_p(n, edges), list(edges), ns, nt, "HARD_NEG")
                )

        if progress_every and draws % progress_every == 0:
            filled = {k: len(pos_pools[k]) for k in RED_HOPS}
            print(
                f"gen_stalk_red draws={draws} pos={filled} neg={len(neg_pool)}",
                file=sys.stderr,
            )

    shortfalls: dict[str, Any] = {}
    for k in RED_HOPS:
        if len(pos_pools[k]) < pos_per_cell:
            shortfalls[f"pos_K{k}"] = {
                "have": len(pos_pools[k]),
                "need": pos_per_cell,
            }
    if len(neg_pool) < n_neg:
        shortfalls["neg"] = {"have": len(neg_pool), "need": n_neg}
    if shortfalls:
        raise RuntimeError(
            f"RED generation shortfall after {draws} draws: {shortfalls}"
        )

    for k in RED_HOPS:
        rng = random.Random(derive_example_seed(seed, k * 1000))
        rng.shuffle(pos_pools[k])
        pos_pools[k] = pos_pools[k][:pos_per_cell]
    neg_rng = random.Random(derive_example_seed(seed, 99_000))
    neg_rng.shuffle(neg_pool)
    neg_pool = neg_pool[:n_neg]

    examples: list[ReachabilityExample] = []
    idx = 0
    for k in RED_HOPS:
        for n, p, edges, s, t, cell in pos_pools[k]:
            examples.append(
                _make_example(
                    seed=derive_example_seed(seed, idx),
                    n=n,
                    p=p,
                    edges=edges,
                    s=s,
                    t=t,
                    y=1,
                    hop=k,
                    cell=cell,
                )
            )
            idx += 1
    for n, p, edges, s, t, cell in neg_pool:
        examples.append(
            _make_example(
                seed=derive_example_seed(seed, idx),
                n=n,
                p=p,
                edges=edges,
                s=s,
                t=t,
                y=0,
                hop=HOP_UNREACHABLE,
                cell=cell,
            )
        )
        idx += 1

    order_rng = random.Random(derive_example_seed(seed, 1))
    order_rng.shuffle(examples)

    hop_counts = Counter(int(ex.hop_distance) for ex in examples)
    y_counts = Counter(int(ex.y) for ex in examples)
    p_vals = [float(ex.p) for ex in examples]
    n_vals = [int(ex.n) for ex in examples]
    tok_lens = [
        len(split_encoding_tokens(ex.encoding)) for ex in examples
    ]
    # density compare vs ood_hops K16 prior (≤0.025 support)
    k16_ps = [float(ex.p) for ex in examples if int(ex.hop_distance) == K16]
    k20_ps = [float(ex.p) for ex in examples if int(ex.hop_distance) == K20]

    report: dict[str, Any] = {
        "cycle": "CYCLE_STALK_RED_COMPETENT_DISSONANCE",
        "science_open": False,
        "seed": seed,
        "split": SPLIT_NAME,
        "construction": "path_backbone",
        "cells": {
            "DENSE_K16": {
                "K": K16,
                "n_support": list(N_SUPPORT_K16),
                "extra_edges": list(EXTRA_EDGES_K16),
                "intent": "empirical p ≫ ood_hops K16 sparse (≤0.025)",
            },
            "LONG_K20": {
                "K": K20,
                "n_support": list(N_SUPPORT_K20),
                "extra_edges": list(EXTRA_EDGES_K20),
                "intent": "K=20 outside train and ADR OOD_HOP_VALUES",
            },
        },
        "quotas": {
            "pos_per_cell": pos_per_cell,
            "n_pos_total": n_pos_total,
            "n_neg": n_neg,
            "hops": list(RED_HOPS),
        },
        "graph_draws": draws,
        "reject_reasons": dict(reject),
        "n_total": len(examples),
        "hop_counts": {str(k): int(v) for k, v in sorted(hop_counts.items())},
        "y_counts": {str(k): int(v) for k, v in sorted(y_counts.items())},
        "n_values_used": dict(Counter(n_vals)),
        "p_empirical": {
            "all": {
                "min": min(p_vals),
                "mean": sum(p_vals) / len(p_vals),
                "max": max(p_vals),
            },
            "K16": {
                "min": min(k16_ps) if k16_ps else None,
                "mean": (sum(k16_ps) / len(k16_ps)) if k16_ps else None,
                "max": max(k16_ps) if k16_ps else None,
                "vs_ood_hops_K16_support_max": 0.025,
                "harder_density": bool(k16_ps and (sum(k16_ps) / len(k16_ps)) > 0.025),
            },
            "K20": {
                "min": min(k20_ps) if k20_ps else None,
                "mean": (sum(k20_ps) / len(k20_ps)) if k20_ps else None,
                "max": max(k20_ps) if k20_ps else None,
            },
        },
        "token_len": {
            "min": min(tok_lens),
            "mean": sum(tok_lens) / len(tok_lens),
            "max": max(tok_lens),
            "max_tokens_accept": max_tokens,
            "bound30_max_len": BOUND30_MAX_LEN,
            "note": "RED substrate — not matched-OOD [45,70] band",
        },
        "train_priors_pushed": {
            "K_train_max": K_TRAIN_MAX,
            "ADR_OOD_HOP_VALUES": [8, 12, 16],
            "K20_outside_ADR_OOD": True,
            "K16_denser_than_ood_hops": True,
        },
        "verify_ok": (
            hop_counts.get(K16, 0) == pos_per_cell
            and hop_counts.get(K20, 0) == pos_per_cell
            and hop_counts.get(HOP_UNREACHABLE, 0) == n_neg
            and y_counts.get(1, 0) == n_pos_total
            and y_counts.get(0, 0) == n_neg
            and max(tok_lens) <= max_tokens
        ),
    }
    return examples, report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=RED_SEED)
    p.add_argument("--pos-per-cell", type=int, default=POS_PER_CELL_DEFAULT)
    p.add_argument("--out", type=Path, default=Path("data/stalk_red_competent_dissonance.jsonl"))
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/stalk_red_competent_dissonance_generation_report.json"),
    )
    p.add_argument("--max-draws", type=int, default=80_000)
    args = p.parse_args(argv)

    examples, report = generate_red(
        seed=args.seed,
        pos_per_cell=args.pos_per_cell,
        max_draws=args.max_draws,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {len(examples)} → {args.out}; report → {args.report}; "
        f"verify_ok={report['verify_ok']}",
        file=sys.stderr,
    )
    return 0 if report["verify_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
