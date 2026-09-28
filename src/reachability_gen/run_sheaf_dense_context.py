"""CYCLE_SHEAF_DENSE_CONTEXT — two-cell MEASURE RED_TEST (fail-closed).

Frozen Gate1 ckpt (no retrain). Band [100,140], mean target ~110–120.
T∈{8,12}. 128/cell (64 pos K=8 + 64 BFS hard-neg). science_open=false.

Cell 1 Dense: n=16, true ER p=0.15, K=8.
Cell 2 Matched sparse: n=32, ER p≈0.036 tuned to match |E|/seq_len of Cell1.

Prereg floors (before interpret):
  FPR/FNR(Â)≤0.02; hard-neg≥0.950; K8 pos≥0.950 at T=8 and T=12 (each cell).

Attribution:
  Cell1 Â fail, Cell2 Â pass → Mode A (dense hallucination)
  Both Â fail → Encoder Context Ceiling (zero-shot 110+ tokens; do NOT claim
    architecture cannot learn — note retrain as next cycle)
  Â both clean, Cell1 reachability drops → Mode C
  Both pass all floors → MEASURE PASS (no science_open widen)

INVALID if either cell seq_len mean ∉ [100,140].

Usage::

    python -m reachability_gen.run_sheaf_dense_context
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import HOP_UNREACHABLE
from reachability_gen.encode import parse_instance
from reachability_gen.gen_sheaf_dense_context import (
    CELL1_N,
    CELL1_P,
    CELL2_N,
    DENSE_K,
    DENSE_SEED,
    SEQ_LEN_BAND,
    SEQ_LEN_TARGET_MEAN,
    generate_dense_context_pair,
    write_jsonl,
    write_report,
)
from reachability_gen.gen_sheaf_density_stress import assert_genuinely_unreachable
from reachability_gen.models.sheaf_infer_core import (
    DEFAULT_GATE_THETA,
    SheafInferCore,
    _verify_param_parity,
    build_sheaf_batch,
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID, split_encoding_tokens

DEFAULT_CKPT = Path("artifacts/sheaf_infer_gate1_best.pt")
DEFAULT_DATA_C1 = Path("data/sheaf_dense_context_cell1.jsonl")
DEFAULT_DATA_C2 = Path("data/sheaf_dense_context_cell2.jsonl")
DEFAULT_GEN_REPORT = Path("artifacts/sheaf_dense_context_generation_report.json")
DEFAULT_OUT = Path("artifacts/sheaf_infer_dense_context.json")

EVAL_T_VALUES: tuple[int, ...] = (8, 12)
EXPECTED_PARAM_COUNT = 123206

PREREG_FPR_MAX = 0.02
PREREG_FNR_MAX = 0.02
PREREG_HARD_NEG = 0.950
PREREG_POS_REACH = 0.950


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _prereg_floors_block() -> dict[str, Any]:
    return {
        "written_before_interpret": True,
        "compiler_clean": {
            "FPR_A_hat_le": PREREG_FPR_MAX,
            "FNR_A_hat_le": PREREG_FNR_MAX,
        },
        "hard_neg_ge": PREREG_HARD_NEG,
        "positive_reachability_ge": {
            "T8": PREREG_POS_REACH,
            "T12": PREREG_POS_REACH,
        },
        "modes": {
            "A": "Cell1 Â fail + Cell2 Â pass → dense hallucination",
            "Encoder_Context_Ceiling": (
                "Both Â fail — zero-shot 110+ tokens; do NOT claim architecture "
                "cannot learn; note retrain as next cycle"
            ),
            "C": "Â both clean, Cell1 reachability drops below floors",
            "PASS": "both cells pass all floors (MEASURE PASS; no science_open widen)",
            "INVALID": "seq_len mean ∉ [100,140] for either cell",
        },
        "note": (
            "Prereg floors locked before metric interpretation. "
            "Report PASS/FAIL/INVALID honestly. science_open=false regardless."
        ),
    }


def _load_frozen_sheaf(
    ckpt_path: Path,
    *,
    max_nodes: int,
    max_T: int,
) -> tuple[SheafInferCore, dict[str, Any], int]:
    import torch

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Gate1 ckpt missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    model = SheafInferCore(
        d=int(hp.get("d", 64)),
        T=int(hp.get("T", 6)),
        mlp_expansion=int(hp.get("mlp_expansion", 12)),
        max_nodes=int(hp.get("max_nodes", max_nodes)),
        max_T=max(int(hp.get("T", 6)), max_T, max(EVAL_T_VALUES)),
        use_tau=False,
        apply_cycle_rmsnorm=False,
        residual_alpha=float(hp.get("residual_alpha", 1.0)),
        gate_detach_diffusion=bool(hp.get("gate_detach_diffusion", True)),
        gate_theta=float(hp.get("gate_theta", DEFAULT_GATE_THETA)),
        gate_mode=str(hp.get("gate_mode", "ste")),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = model.param_count()
    return model, ckpt, n_params


def _a_hat_fpr_fnr(
    model: SheafInferCore,
    rows: list[dict[str, Any]],
    *,
    max_nodes: int,
    batch_size: int = 16,
) -> dict[str, Any]:
    import torch

    model.eval()
    tp = fp = tn = fn = 0
    n_examples = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = build_sheaf_batch(batch_rows, max_n=max_nodes)
            E = model.encode_edge_logits(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
            )
            gate = model.logits_to_gate(E, batch["node_mask"])
            gold = batch["gold_adj"]
            node_mask = batch["node_mask"]
            bsz, mlen, _ = gold.shape
            eye = torch.eye(mlen, device=gold.device, dtype=torch.bool)
            real = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).bool()
            off = real & (~eye.unsqueeze(0))
            pred = gate > 0.5
            g = gold > 0.5
            tp += int(((pred & g) & off).sum().item())
            fp += int(((pred & ~g) & off).sum().item())
            tn += int(((~pred & ~g) & off).sum().item())
            fn += int(((~pred & g) & off).sum().item())
            n_examples += len(batch_rows)

    neg = fp + tn
    pos = tp + fn
    fpr = float(fp / neg) if neg > 0 else float("nan")
    fnr = float(fn / pos) if pos > 0 else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_offdiag_cells": tp + fp + tn + fn,
        "n_examples": n_examples,
        "FPR": fpr,
        "FNR": fnr,
        "comparable": "offdiag_real_nodes_vs_gold_A",
        "compiler_clean": bool(
            fpr == fpr
            and fnr == fnr
            and fpr <= PREREG_FPR_MAX
            and fnr <= PREREG_FNR_MAX
        ),
        "note": (
            "Â = hard gate(σ(E_hat)>θ) vs gold adjacency from edge tokens; "
            "diagonal excluded; pad excluded. Not an eval oracle for reachability."
        ),
    }


def _eval_reachability(
    model: SheafInferCore,
    rows: list[dict[str, Any]],
    *,
    max_nodes: int,
    T: int,
    batch_size: int = 16,
) -> dict[str, Any]:
    import torch

    model.eval()
    hop_accs: dict[int, list[float]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = build_sheaf_batch(batch_rows, max_n=max_nodes)
            logits, _, _ = model(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
                batch["s_idx"],
                batch["t_idx"],
                return_halt=True,
                T=T,
            )
            labels = batch["labels"]
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).float()
            for i, ex in enumerate(batch_rows):
                hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
                hop_accs[hop].append(float(correct[i].item()))

    by_hop: dict[str, Any] = {}
    for hk in sorted(hop_accs):
        xs = hop_accs[hk]
        by_hop[str(hk)] = {
            "n": len(xs),
            "acc_mean": _mean(xs),
            "correct": int(round(sum(xs))),
        }
    n_tot = sum(int(v["n"]) for v in by_hop.values())
    overall = (
        sum(float(v["acc_mean"]) * int(v["n"]) for v in by_hop.values()) / n_tot
        if n_tot
        else float("nan")
    )
    hard = by_hop.get(str(HOP_UNREACHABLE), by_hop.get("-1", {}))
    kpos = by_hop.get(str(DENSE_K), {})
    return {
        "T": T,
        "overall_acc": overall,
        "n": n_tot,
        "by_hop": by_hop,
        "hard_neg_acc": float(hard.get("acc_mean", float("nan"))) if hard else float("nan"),
        "hard_neg_n": int(hard.get("n", 0)) if hard else 0,
        "positive_acc": float(kpos.get("acc_mean", float("nan"))) if kpos else float("nan"),
        "positive_n": int(kpos.get("n", 0)) if kpos else 0,
        "K8_acc": float(kpos.get("acc_mean", float("nan"))) if kpos else float("nan"),
        "K8_n": int(kpos.get("n", 0)) if kpos else 0,
        "hard_A_oracle": False,
        "adaptive_halt": False,
        "discrete_T": True,
    }


def _seq_len_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lens = [
        len(split_encoding_tokens(str(r.get("encoding", "")))) for r in rows
    ]
    edge_counts = []
    for r in rows:
        _, edges, _, _ = parse_instance(str(r["encoding"]))
        edge_counts.append(len(edges))
    mean = _mean(lens)
    mean_in_band = bool(
        mean == mean and SEQ_LEN_BAND[0] <= mean <= SEQ_LEN_BAND[1]
    )
    return {
        "min": min(lens) if lens else None,
        "max": max(lens) if lens else None,
        "mean": mean,
        "n": len(lens),
        "band": list(SEQ_LEN_BAND),
        "target_mean": list(SEQ_LEN_TARGET_MEAN),
        "mean_in_band": mean_in_band,
        "edge_count_mean": _mean([float(x) for x in edge_counts]),
        "edge_count_min": min(edge_counts) if edge_counts else None,
        "edge_count_max": max(edge_counts) if edge_counts else None,
    }


def _verify_hard_negs_in_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_checked = 0
    for r in rows:
        if int(r.get("y", -1)) != 0:
            continue
        n, edges, s, t = parse_instance(str(r["encoding"]))
        assert_genuinely_unreachable(n, edges, s, t)
        n_checked += 1
    return {
        "n_hard_neg_checked": n_checked,
        "R_out_bfs_dfs_assert": True,
        "ok": True,
    }


def _cell_floors(
    *,
    fpr: float,
    fnr: float,
    hard_neg_t8: float,
    hard_neg_t12: float,
    pos_t8: float,
    pos_t12: float,
) -> dict[str, Any]:
    compiler_ok = (
        fpr == fpr
        and fnr == fnr
        and fpr <= PREREG_FPR_MAX
        and fnr <= PREREG_FNR_MAX
    )
    a_fail = (
        (fpr == fpr and fpr > PREREG_FPR_MAX)
        or (fnr == fnr and fnr > PREREG_FNR_MAX)
    )
    hard_ok = (
        hard_neg_t8 == hard_neg_t8
        and hard_neg_t12 == hard_neg_t12
        and hard_neg_t8 >= PREREG_HARD_NEG
        and hard_neg_t12 >= PREREG_HARD_NEG
    )
    pos_ok = (
        pos_t8 == pos_t8
        and pos_t12 == pos_t12
        and pos_t8 >= PREREG_POS_REACH
        and pos_t12 >= PREREG_POS_REACH
    )
    reach_ok = hard_ok and pos_ok
    pass_all = compiler_ok and reach_ok
    return {
        "compiler_ok": compiler_ok,
        "A_hat_fail": a_fail,
        "hard_neg_ok": hard_ok,
        "pos_ok": pos_ok,
        "reach_ok": reach_ok,
        "pass_all": pass_all,
        "FPR": fpr,
        "FNR": fnr,
        "hard_neg_T8": hard_neg_t8,
        "hard_neg_T12": hard_neg_t12,
        "positive_K8_T8": pos_t8,
        "positive_K8_T12": pos_t12,
    }


def _attribute_dense_context(
    *,
    c1: dict[str, Any],
    c2: dict[str, Any],
    run_valid: bool,
) -> dict[str, Any]:
    if not run_valid:
        return {
            "mode": "INVALID",
            "note": "Attribution suspended: seq_len mean out of band [100,140].",
        }

    c1_a_fail = bool(c1["A_hat_fail"])
    c2_a_fail = bool(c2["A_hat_fail"])
    c1_clean = bool(c1["compiler_ok"])
    c2_clean = bool(c2["compiler_ok"])

    # Mode A: Cell1 Â fail, Cell2 Â pass → dense hallucination
    if c1_a_fail and (not c2_a_fail) and c2_clean:
        return {
            "mode": "A",
            "label": "dense_hallucination",
            "note": "Cell1 Â fail + Cell2 Â pass → Mode A (dense hallucination).",
            "cell1_A_hat_fail": True,
            "cell2_A_hat_fail": False,
        }

    # Both Â fail → Encoder Context Ceiling
    if c1_a_fail and c2_a_fail:
        return {
            "mode": "Encoder_Context_Ceiling",
            "label": "encoder_context_ceiling",
            "note": (
                "Both cells Â fail at zero-shot 110+ tokens. "
                "Do NOT claim architecture cannot learn — note retrain as next cycle."
            ),
            "cell1_A_hat_fail": True,
            "cell2_A_hat_fail": True,
            "next_cycle_hint": "retrain_at_dense_band",
        }

    # Â both clean, Cell1 reachability drops → Mode C
    if c1_clean and c2_clean and (not c1["reach_ok"]):
        return {
            "mode": "C",
            "label": "clean_A_hat_cell1_reach_drop",
            "note": "Â both clean; Cell1 reachability below floors → Mode C.",
            "cell1_reach_ok": False,
            "cell2_reach_ok": bool(c2["reach_ok"]),
        }

    # Both pass all floors → MEASURE PASS
    if c1["pass_all"] and c2["pass_all"]:
        return {
            "mode": "PASS",
            "label": "measure_pass",
            "note": (
                "Both cells pass all floors — MEASURE PASS. "
                "Do not widen science_open / K≤16 claim."
            ),
            "science_open_widen": False,
        }

    # Residual failure patterns
    modes: list[str] = []
    if c1_a_fail:
        modes.append("cell1_A")
    if c2_a_fail:
        modes.append("cell2_A")
    if c1_clean and not c1["reach_ok"]:
        modes.append("cell1_reach")
    if c2_clean and not c2["reach_ok"]:
        modes.append("cell2_reach")
    return {
        "mode": "FAIL" if not modes else "multi_mode:" + ",".join(modes),
        "modes_fired": modes,
        "note": "Unclassified residual under locked floors; fail-closed.",
        "cell1_pass_all": bool(c1["pass_all"]),
        "cell2_pass_all": bool(c2["pass_all"]),
    }


def _eval_cell(
    model: SheafInferCore,
    rows: list[dict[str, Any]],
    *,
    max_nodes: int,
    cell_id: str,
) -> dict[str, Any]:
    seq = _seq_len_stats(rows)
    hard_neg_verify = _verify_hard_negs_in_rows(rows)
    a_hat = _a_hat_fpr_fnr(model, rows, max_nodes=max_nodes)
    by_T: dict[str, Any] = {}
    for T in EVAL_T_VALUES:
        stats = _eval_reachability(model, rows, max_nodes=max_nodes, T=T)
        by_T[f"T{T}"] = stats
        print(
            f"[dense-ctx:{cell_id}] T={T} overall={stats['overall_acc']:.4f} "
            f"hard_neg={stats['hard_neg_acc']:.4f} "
            f"pos_K8={stats['positive_acc']:.4f}",
            file=sys.stderr,
        )
    t8 = by_T["T8"]
    t12 = by_T["T12"]
    floors = _cell_floors(
        fpr=float(a_hat["FPR"]),
        fnr=float(a_hat["FNR"]),
        hard_neg_t8=float(t8["hard_neg_acc"]),
        hard_neg_t12=float(t12["hard_neg_acc"]),
        pos_t8=float(t8["positive_acc"]),
        pos_t12=float(t12["positive_acc"]),
    )
    emp_ps = [float(r.get("p", float("nan"))) for r in rows]
    return {
        "cell": cell_id,
        "n_examples": len(rows),
        "n_pos": sum(1 for r in rows if int(r.get("y", -1)) == 1),
        "n_neg": sum(1 for r in rows if int(r.get("y", -1)) == 0),
        "p_empirical_mean": _mean(emp_ps),
        "seq_len_stats": seq,
        "hard_neg_discipline": hard_neg_verify,
        "A_hat_vs_gold": a_hat,
        "reachability": by_T,
        "floors": floors,
        "table": {
            "T8": {
                "overall": t8["overall_acc"],
                "hard_neg": t8["hard_neg_acc"],
                "K8": t8["positive_acc"],
            },
            "T12": {
                "overall": t12["overall_acc"],
                "hard_neg": t12["hard_neg_acc"],
                "K8": t12["positive_acc"],
            },
        },
    }


def ensure_dense_context_data(
    data_c1: Path,
    data_c2: Path,
    report_path: Path,
    *,
    n_pos: int,
    n_neg: int,
    seed: int,
    force_regen: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if (
        data_c1.exists()
        and data_c2.exists()
        and report_path.exists()
        and not force_regen
    ):
        rows1 = load_jsonl(data_c1)
        rows2 = load_jsonl(data_c2)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if len(rows1) >= (n_pos + n_neg) and len(rows2) >= (n_pos + n_neg):
            return rows1, rows2, report

    pair = generate_dense_context_pair(
        seed=seed, n_pos=n_pos, n_neg=n_neg
    )
    write_jsonl(data_c1, pair["cell1"]["examples"])
    write_jsonl(data_c2, pair["cell2"]["examples"])
    report = {
        "cycle": "CYCLE_SHEAF_DENSE_CONTEXT",
        "science_open": False,
        "seed": seed,
        "seq_len_band": list(SEQ_LEN_BAND),
        "both_seq_len_in_band": pair["both_seq_len_in_band"],
        "p_tune": pair["p_tune"],
        "cell1": pair["cell1"]["report"],
        "cell2": pair["cell2"]["report"],
    }
    write_report(report_path, report)
    return load_jsonl(data_c1), load_jsonl(data_c2), report


def run_dense_context(
    *,
    ckpt_path: Path = DEFAULT_CKPT,
    data_c1: Path = DEFAULT_DATA_C1,
    data_c2: Path = DEFAULT_DATA_C2,
    gen_report_path: Path = DEFAULT_GEN_REPORT,
    out_path: Path = DEFAULT_OUT,
    n_pos: int = 64,
    n_neg: int = 64,
    seed: int = DENSE_SEED,
    force_regen: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    prereg_floors = _prereg_floors_block()

    rows1, rows2, gen_report = ensure_dense_context_data(
        data_c1,
        data_c2,
        gen_report_path,
        n_pos=n_pos,
        n_neg=n_neg,
        seed=seed,
        force_regen=force_regen,
    )

    seq1 = _seq_len_stats(rows1)
    seq2 = _seq_len_stats(rows2)
    run_valid = bool(seq1["mean_in_band"] and seq2["mean_in_band"])
    invalid_reason = None
    if not run_valid:
        invalid_reason = (
            f"seq_len mean out of band {list(SEQ_LEN_BAND)}: "
            f"cell1={seq1['mean']:.4f} (in_band={seq1['mean_in_band']}), "
            f"cell2={seq2['mean']:.4f} (in_band={seq2['mean_in_band']}). "
            "DENSE_CONTEXT INVALID — stop."
        )
        print(f"[dense-ctx] INVALID: {invalid_reason}", file=sys.stderr)

    max_n_data = max(
        max(int(r["n"]) for r in rows1),
        max(int(r["n"]) for r in rows2),
    )
    max_nodes = max(DEFAULT_MAX_NODE_ID, max_n_data)

    model, ckpt, n_params = _load_frozen_sheaf(
        ckpt_path, max_nodes=max_nodes, max_T=max(EVAL_T_VALUES)
    )
    parity = _verify_param_parity(n_params, ff_baseline=FF_BASELINE_PARAMS)
    if n_params != EXPECTED_PARAM_COUNT:
        print(
            f"WARN: param_count={n_params} expected≈{EXPECTED_PARAM_COUNT}",
            file=sys.stderr,
        )

    cell1 = _eval_cell(model, rows1, max_nodes=max_nodes, cell_id="cell1_dense")
    cell1["spec"] = {
        "n": CELL1_N,
        "p_requested": CELL1_P,
        "K": DENSE_K,
        "construction": "true_er_digraph",
        "label": "dense",
    }
    cell1["dataset_path"] = str(data_c1)

    cell2 = _eval_cell(
        model, rows2, max_nodes=max_nodes, cell_id="cell2_matched_sparse"
    )
    cell2_p = float(
        (gen_report.get("cell2") or {}).get("p_requested", CELL2_N and 0.036)
    )
    cell2["spec"] = {
        "n": CELL2_N,
        "p_requested": cell2_p,
        "K": DENSE_K,
        "construction": "true_er_digraph",
        "label": "matched_sparse_control",
        "p_tune": gen_report.get("p_tune"),
    }
    cell2["dataset_path"] = str(data_c2)

    attribution = _attribute_dense_context(
        c1=cell1["floors"],
        c2=cell2["floors"],
        run_valid=run_valid,
    )

    prereg_pass = bool(
        run_valid
        and cell1["floors"]["pass_all"]
        and cell2["floors"]["pass_all"]
    )

    artifact: dict[str, Any] = {
        "cycle": "CYCLE_SHEAF_DENSE_CONTEXT",
        "mode": "MEASURE",
        "science_open": False,
        "run_id": f"sheaf-dense-context-{uuid.uuid4().hex[:10]}",
        "run_valid": run_valid,
        "invalid_reason": invalid_reason,
        "cells": {
            "cell1_dense": {
                "n": CELL1_N,
                "p_requested": CELL1_P,
                "K": DENSE_K,
                "T_values": list(EVAL_T_VALUES),
                "n_per_cell": n_pos + n_neg,
                "n_pos": n_pos,
                "n_neg": n_neg,
            },
            "cell2_matched_sparse": {
                "n": CELL2_N,
                "p_requested": cell2_p,
                "K": DENSE_K,
                "T_values": list(EVAL_T_VALUES),
                "n_per_cell": n_pos + n_neg,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "matched_to": "cell1 |E| and seq_len",
            },
        },
        "seq_len_band": list(SEQ_LEN_BAND),
        "seq_len_target_mean": list(SEQ_LEN_TARGET_MEAN),
        "prereg_floors": prereg_floors,
        "checkpoint": {
            "path": str(ckpt_path),
            "verified_exists": True,
            "param_count": n_params,
            "expected_param_count": EXPECTED_PARAM_COUNT,
            "param_parity": parity,
            "frozen": True,
            "retrain": False,
            "ckpt_epoch": ckpt.get("epoch"),
            "ckpt_val_acc": ckpt.get("val_acc"),
            "ckpt_arm": ckpt.get("arm"),
            "hard_A_oracle_eval": False,
            "gate_detach_diffusion": True,
        },
        "cell1": cell1,
        "cell2": cell2,
        "attribution": attribution,
        "prereg": {
            "floors": prereg_floors,
            "observed": {
                "cell1_FPR": cell1["floors"]["FPR"],
                "cell1_FNR": cell1["floors"]["FNR"],
                "cell2_FPR": cell2["floors"]["FPR"],
                "cell2_FNR": cell2["floors"]["FNR"],
                "cell1_hard_neg_T8": cell1["floors"]["hard_neg_T8"],
                "cell1_hard_neg_T12": cell1["floors"]["hard_neg_T12"],
                "cell1_pos_T8": cell1["floors"]["positive_K8_T8"],
                "cell1_pos_T12": cell1["floors"]["positive_K8_T12"],
                "cell2_hard_neg_T8": cell2["floors"]["hard_neg_T8"],
                "cell2_hard_neg_T12": cell2["floors"]["hard_neg_T12"],
                "cell2_pos_T8": cell2["floors"]["positive_K8_T8"],
                "cell2_pos_T12": cell2["floors"]["positive_K8_T12"],
                "seq_len_cell1": seq1["mean"],
                "seq_len_cell2": seq2["mean"],
            },
            "pass": prereg_pass,
            "fail_closed": not prereg_pass,
            "run_valid": run_valid,
            "invalid": not run_valid,
        },
        "generation_report": str(gen_report_path),
        "generation_report_summary": {
            "both_seq_len_in_band": gen_report.get("both_seq_len_in_band"),
            "cell1_token_len_mean": (gen_report.get("cell1") or {})
            .get("token_len", {})
            .get("mean"),
            "cell2_token_len_mean": (gen_report.get("cell2") or {})
            .get("token_len", {})
            .get("mean"),
            "cell2_p_requested": (gen_report.get("cell2") or {}).get("p_requested"),
            "p_tune": gen_report.get("p_tune"),
        },
        "baseline_sheaf_seal": "af8e49f",
        "prior_density_stress_merge": "a6665bc",
        "elapsed_sec": time.time() - t0,
        "science_open": False,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"[dense-ctx] wrote {out_path}", file=sys.stderr)
    print(
        f"[dense-ctx] valid={run_valid} prereg_pass={prereg_pass} "
        f"mode={artifact['attribution']['mode']} "
        f"seq1={seq1['mean']:.2f} seq2={seq2['mean']:.2f} "
        f"c1_FPR={cell1['floors']['FPR']:.4f} c1_FNR={cell1['floors']['FNR']:.4f} "
        f"c2_FPR={cell2['floors']['FPR']:.4f} c2_FNR={cell2['floors']['FNR']:.4f}",
        file=sys.stderr,
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CYCLE_SHEAF_DENSE_CONTEXT MEASURE runner"
    )
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--data-cell1", type=Path, default=DEFAULT_DATA_C1)
    p.add_argument("--data-cell2", type=Path, default=DEFAULT_DATA_C2)
    p.add_argument("--gen-report", type=Path, default=DEFAULT_GEN_REPORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-pos", type=int, default=64)
    p.add_argument("--n-neg", type=int, default=64)
    p.add_argument("--seed", type=int, default=DENSE_SEED)
    p.add_argument("--force-regen", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    try:
        art = run_dense_context(
            ckpt_path=args.ckpt,
            data_c1=args.data_cell1,
            data_c2=args.data_cell2,
            gen_report_path=args.gen_report,
            out_path=args.out,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            seed=args.seed,
            force_regen=args.force_regen,
        )
    except AssertionError as e:
        print(f"FAIL-CLOSED: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise
    if not art.get("run_valid", True):
        print(f"INVALID: {art.get('invalid_reason')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
