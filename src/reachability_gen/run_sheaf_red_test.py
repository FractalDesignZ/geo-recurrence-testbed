"""CYCLE_SHEAF_RED_TEST — single cell MEASURE (fail-closed).

Cell: K=20, p_requested=0.15, T∈{20,24}, frozen SheafInferCore Gate1 ckpt.
No retrain. No HF Jobs. science_open=false always (do not widen).

Prereg floors (written into artifact **before** interpreting metrics):
  - Compiler clean: FPR(Â)≤0.02 and FNR(Â)≤0.02
  - Hard-neg ≥ 0.950
  - K20 reachability ≥ 0.800 at T=24

Attribution modes:
  A — FPR(Â)>0.02
  B — FNR(Â)>0.02
  C — Â exact (compiler clean) but K20 @ T=24 drops below floor
  else PASS or multi-mode (comma-joined)

Seq-len: mean must stay in sealed band spirit (K=20 floor ≈66); FAIL INVALID
if mean drifts toward ~200 (threshold: mean ≥ 150).

Usage::

    python -m reachability_gen.run_sheaf_red_test
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
from reachability_gen.gen_sheaf_red_test import (
    RED_K,
    RED_P_REQUESTED,
    RED_TEST_SEED,
    generate_sheaf_red_test_cell,
    write_jsonl,
    write_report,
)
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
DEFAULT_DATA = Path("data/sheaf_red_test_k20.jsonl")
DEFAULT_GEN_REPORT = Path("artifacts/sheaf_red_test_generation_report.json")
DEFAULT_OUT = Path("artifacts/sheaf_infer_red_test.json")

EVAL_T_VALUES: tuple[int, ...] = (20, 24)
EXPECTED_PARAM_COUNT = 123206
SEQ_LEN_INVALID_MEAN = 150.0  # drift toward ~200 Gate2 confound
SEQ_LEN_BAND = (45, 70)
SEQ_LEN_ID_TARGET = (55.0, 65.0)

# Prereg floors — locked before metric interpretation.
PREREG_FPR_MAX = 0.02
PREREG_FNR_MAX = 0.02
PREREG_HARD_NEG = 0.950
PREREG_K20_T24 = 0.800


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _prereg_floors_block() -> dict[str, Any]:
    """Write floors first; callers must stamp this before reading metrics."""
    return {
        "written_before_interpret": True,
        "compiler_clean": {
            "FPR_A_hat_le": PREREG_FPR_MAX,
            "FNR_A_hat_le": PREREG_FNR_MAX,
        },
        "hard_neg_ge": PREREG_HARD_NEG,
        "K20_reachability_at_T24_ge": PREREG_K20_T24,
        "modes": {
            "A": "FPR(A_hat) > 0.02",
            "B": "FNR(A_hat) > 0.02",
            "C": "A_hat compiler-clean but K20@T24 < 0.800",
            "PASS": "all floors met",
            "multi_mode": "comma-joined subset of A,B,C when multiple fire",
        },
        "note": (
            "Prereg floors locked before metric interpretation. "
            "Report PASS/FAIL honestly. science_open=false regardless."
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
    batch_size: int = 32,
) -> dict[str, Any]:
    """Off-diagonal Â vs gold A: FPR / FNR (comparable cells only)."""
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
            # Comparable off-diag cells only.
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
    batch_size: int = 32,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

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
    for k in sorted(hop_accs):
        xs = hop_accs[k]
        by_hop[str(k)] = {
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
    k20 = by_hop.get(str(RED_K), {})
    return {
        "T": T,
        "overall_acc": overall,
        "n": n_tot,
        "by_hop": by_hop,
        "hard_neg_acc": float(hard.get("acc_mean", float("nan"))) if hard else float("nan"),
        "hard_neg_n": int(hard.get("n", 0)) if hard else 0,
        "K20_acc": float(k20.get("acc_mean", float("nan"))) if k20 else float("nan"),
        "K20_n": int(k20.get("n", 0)) if k20 else 0,
        "hard_A_oracle": False,
        "adaptive_halt": False,
        "discrete_T": True,
    }


def _seq_len_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lens = [
        len(split_encoding_tokens(str(r.get("encoding", "")))) for r in rows
    ]
    mean = _mean(lens)
    invalid = bool(mean == mean and mean >= SEQ_LEN_INVALID_MEAN)
    in_sealed_band = all(SEQ_LEN_BAND[0] <= L <= SEQ_LEN_BAND[1] for L in lens)
    return {
        "min": min(lens) if lens else None,
        "max": max(lens) if lens else None,
        "mean": mean,
        "n": len(lens),
        "sealed_band": list(SEQ_LEN_BAND),
        "id_target_mean_band": list(SEQ_LEN_ID_TARGET),
        "all_in_sealed_band": in_sealed_band,
        "invalid_drift_toward_200": invalid,
        "invalid_threshold_mean_ge": SEQ_LEN_INVALID_MEAN,
        "note": (
            "K=20 forces token_len≥66; mean sits high in sealed [45,70] vs "
            "ID target ≈55–65, but must NOT approach ~200 (Gate2 confound). "
            "Run marked INVALID if mean ≥ "
            f"{SEQ_LEN_INVALID_MEAN}."
        ),
    }


def _attribute_mode(
    *,
    fpr: float,
    fnr: float,
    k20_t24: float,
    hard_neg_t24: float,
) -> dict[str, Any]:
    modes: list[str] = []
    fpr_bad = fpr == fpr and fpr > PREREG_FPR_MAX
    fnr_bad = fnr == fnr and fnr > PREREG_FNR_MAX
    compiler_clean = (not fpr_bad) and (not fnr_bad) and (fpr == fpr) and (fnr == fnr)
    k20_drop = k20_t24 == k20_t24 and k20_t24 < PREREG_K20_T24
    hard_ok = hard_neg_t24 == hard_neg_t24 and hard_neg_t24 >= PREREG_HARD_NEG

    if fpr_bad:
        modes.append("A")
    if fnr_bad:
        modes.append("B")
    if compiler_clean and k20_drop:
        modes.append("C")

    if not modes and compiler_clean and (not k20_drop) and hard_ok:
        label = "PASS"
    elif not modes:
        # Floors missed without A/B/C (e.g. hard-neg only) — report as FAIL.
        label = "FAIL"
    elif len(modes) == 1:
        label = modes[0]
    else:
        label = "multi_mode:" + ",".join(modes)

    return {
        "mode": label,
        "modes_fired": modes,
        "compiler_clean": compiler_clean,
        "fpr_exceeds": fpr_bad,
        "fnr_exceeds": fnr_bad,
        "k20_drop_under_clean_A_hat": bool(compiler_clean and k20_drop),
        "hard_neg_ok_T24": hard_ok,
    }


def ensure_red_test_data(
    data_path: Path,
    report_path: Path,
    *,
    n_pos: int,
    n_neg: int,
    seed: int,
    force_regen: bool = False,
) -> list[dict[str, Any]]:
    if data_path.exists() and not force_regen:
        rows = load_jsonl(data_path)
        if len(rows) >= (n_pos + n_neg):
            return rows
    examples, report = generate_sheaf_red_test_cell(
        seed=seed, n_pos=n_pos, n_neg=n_neg
    )
    write_jsonl(data_path, examples)
    write_report(report_path, report)
    return load_jsonl(data_path)


def run_red_test(
    *,
    ckpt_path: Path = DEFAULT_CKPT,
    data_path: Path = DEFAULT_DATA,
    gen_report_path: Path = DEFAULT_GEN_REPORT,
    out_path: Path = DEFAULT_OUT,
    n_pos: int = 64,
    n_neg: int = 64,
    seed: int = RED_TEST_SEED,
    force_regen: bool = False,
) -> dict[str, Any]:
    t0 = time.time()

    # 1) Lock prereg floors BEFORE any metric interpretation.
    prereg_floors = _prereg_floors_block()

    rows = ensure_red_test_data(
        data_path,
        gen_report_path,
        n_pos=n_pos,
        n_neg=n_neg,
        seed=seed,
        force_regen=force_regen,
    )
    seq_stats = _seq_len_stats(rows)
    run_valid = not bool(seq_stats["invalid_drift_toward_200"])
    invalid_reason = None
    if not run_valid:
        invalid_reason = (
            f"seq_len mean={seq_stats['mean']} ≥ {SEQ_LEN_INVALID_MEAN} "
            "(drift toward ~200 Gate2 confound) — run INVALID"
        )

    max_n_data = max(int(r["n"]) for r in rows)
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

    # Â telemetry (independent of T unroll).
    a_hat = _a_hat_fpr_fnr(model, rows, max_nodes=max_nodes)

    by_T: dict[str, Any] = {}
    for T in EVAL_T_VALUES:
        stats = _eval_reachability(model, rows, max_nodes=max_nodes, T=T)
        by_T[f"T{T}"] = stats
        print(
            f"[red-test] T={T} overall={stats['overall_acc']:.4f} "
            f"hard_neg={stats['hard_neg_acc']:.4f} "
            f"K20={stats['K20_acc']:.4f}",
            file=sys.stderr,
        )

    t24 = by_T["T24"]
    hard_neg_t24 = float(t24["hard_neg_acc"])
    k20_t24 = float(t24["K20_acc"])
    fpr = float(a_hat["FPR"])
    fnr = float(a_hat["FNR"])

    attribution = _attribute_mode(
        fpr=fpr, fnr=fnr, k20_t24=k20_t24, hard_neg_t24=hard_neg_t24
    )

    compiler_ok = (
        fpr == fpr
        and fnr == fnr
        and fpr <= PREREG_FPR_MAX
        and fnr <= PREREG_FNR_MAX
    )
    hard_ok = hard_neg_t24 == hard_neg_t24 and hard_neg_t24 >= PREREG_HARD_NEG
    k20_ok = k20_t24 == k20_t24 and k20_t24 >= PREREG_K20_T24
    prereg_pass = bool(run_valid and compiler_ok and hard_ok and k20_ok)

    gen_report = None
    if gen_report_path.exists():
        gen_report = json.loads(gen_report_path.read_text(encoding="utf-8"))

    artifact: dict[str, Any] = {
        "cycle": "CYCLE_SHEAF_RED_TEST",
        "mode": "MEASURE",
        "science_open": False,
        "run_id": f"sheaf-red-test-{uuid.uuid4().hex[:10]}",
        "run_valid": run_valid,
        "invalid_reason": invalid_reason,
        "cell": {
            "K": RED_K,
            "p_requested": RED_P_REQUESTED,
            "T_values": list(EVAL_T_VALUES),
            "construction": "path_backbone",
            "note": (
                "Joint (K=20, ER p=0.15, seq_len band) infeasible; "
                "path-backbone keeps band; empirical p recorded in gen report."
            ),
        },
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
        "dataset": {
            "path": str(data_path),
            "n_examples": len(rows),
            "n_pos": sum(1 for r in rows if int(r.get("y", -1)) == 1),
            "n_neg": sum(1 for r in rows if int(r.get("y", -1)) == 0),
            "seed": seed,
            "generation_report": str(gen_report_path),
            "hard_neg_discipline": True,
        },
        "seq_len_stats": seq_stats,
        "A_hat_vs_gold": a_hat,
        "reachability": by_T,
        "telemetry": {
            "reachability_overall_T20": by_T["T20"]["overall_acc"],
            "reachability_overall_T24": by_T["T24"]["overall_acc"],
            "hard_neg_T20": by_T["T20"]["hard_neg_acc"],
            "hard_neg_T24": by_T["T24"]["hard_neg_acc"],
            "K20_T20": by_T["T20"]["K20_acc"],
            "K20_T24": by_T["T24"]["K20_acc"],
            "FPR_A_hat": fpr,
            "FNR_A_hat": fnr,
        },
        "attribution": attribution,
        "prereg": {
            "floors": prereg_floors,
            "observed": {
                "FPR_A_hat": fpr,
                "FNR_A_hat": fnr,
                "hard_neg_T24": hard_neg_t24,
                "K20_T24": k20_t24,
            },
            "compiler_clean_ok": compiler_ok,
            "hard_neg_ok": hard_ok,
            "K20_T24_ok": k20_ok,
            "pass": prereg_pass,
            "fail_closed": not prereg_pass,
            "run_valid": run_valid,
        },
        "generation_report_summary": {
            "p_empirical_mean": (gen_report or {}).get("p_empirical", {}).get("mean"),
            "er_total_with_K20_in_band": (
                (gen_report or {}).get("er_band_search") or {}
            ).get("total_with_hop_k_in_band"),
            "token_len_mean": (gen_report or {}).get("token_len", {}).get("mean"),
        },
        "baseline_sheaf_seal": "af8e49f",
        "prior_measure_sha": "f374e7c",
        "elapsed_sec": time.time() - t0,
        "science_open": False,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp floors-first copy already embedded; write full artifact.
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"[red-test] wrote {out_path}", file=sys.stderr)
    print(
        f"[red-test] valid={run_valid} prereg_pass={prereg_pass} "
        f"mode={attribution['mode']} "
        f"FPR={fpr:.4f} FNR={fnr:.4f} "
        f"hard_neg_T24={hard_neg_t24:.4f} K20_T24={k20_t24:.4f} "
        f"seq_mean={seq_stats['mean']:.2f}",
        file=sys.stderr,
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CYCLE_SHEAF_RED_TEST MEASURE runner")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--gen-report", type=Path, default=DEFAULT_GEN_REPORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-pos", type=int, default=64)
    p.add_argument("--n-neg", type=int, default=64)
    p.add_argument("--seed", type=int, default=RED_TEST_SEED)
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
        art = run_red_test(
            ckpt_path=args.ckpt,
            data_path=args.data,
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
