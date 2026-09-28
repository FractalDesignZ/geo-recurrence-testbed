"""CYCLE_SHEAF_DENSITY_STRESS — single cell MEASURE (fail-closed).

Cell: K=8, n=16, true ER digraph p=0.15, T∈{8,12}, frozen SheafInferCore
Gate1 ckpt (123206 params). No retrain. No path-backbone. No HF Jobs.
science_open=false always (do not widen §6 sheaf claim).

Prereg floors (written into artifact **before** interpreting metrics):
  - FPR(Â)≤0.02 and FNR(Â)≤0.02
  - Hard-neg ≥ 0.950
  - Positive reachability ≥ 0.950 at T=8 and T=12

Attribution:
  A — FPR(Â)>0.02
  B — FNR(Â)>0.02
  C — Â compiler-clean but accuracy drops below floors
  else PASS (or FAIL / multi_mode)

Seq-len: mean **must** be in [45,70]. If impossible under true p=0.15,
mark INVALID and stop — do **not** fake density with path-backbone.

Usage::

    python -m reachability_gen.run_sheaf_density_stress
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
from reachability_gen.gen_sheaf_density_stress import (
    DENSITY_K,
    DENSITY_N,
    DENSITY_P,
    DENSITY_SEED,
    SEQ_LEN_BAND,
    assert_genuinely_unreachable,
    generate_sheaf_density_stress_cell,
    write_jsonl,
    write_report,
)
from reachability_gen.encode import parse_instance
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
DEFAULT_DATA = Path("data/sheaf_density_stress_p015.jsonl")
DEFAULT_GEN_REPORT = Path("artifacts/sheaf_density_stress_generation_report.json")
DEFAULT_OUT = Path("artifacts/sheaf_infer_density_stress.json")

EVAL_T_VALUES: tuple[int, ...] = (8, 12)
EXPECTED_PARAM_COUNT = 123206

# Prereg floors — locked before metric interpretation.
PREREG_FPR_MAX = 0.02
PREREG_FNR_MAX = 0.02
PREREG_HARD_NEG = 0.950
PREREG_POS_REACH = 0.950


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
        "positive_reachability_ge": {
            "T8": PREREG_POS_REACH,
            "T12": PREREG_POS_REACH,
        },
        "modes": {
            "A": "FPR(A_hat) > 0.02",
            "B": "FNR(A_hat) > 0.02",
            "C": "A_hat compiler-clean but accuracy drops below floors",
            "PASS": "all floors met",
            "multi_mode": "comma-joined subset of A,B,C when multiple fire",
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
    kpos = by_hop.get(str(DENSITY_K), {})
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
    mean = _mean(lens)
    mean_in_band = bool(
        mean == mean and SEQ_LEN_BAND[0] <= mean <= SEQ_LEN_BAND[1]
    )
    all_in_band = all(SEQ_LEN_BAND[0] <= L <= SEQ_LEN_BAND[1] for L in lens)
    return {
        "min": min(lens) if lens else None,
        "max": max(lens) if lens else None,
        "mean": mean,
        "n": len(lens),
        "sealed_band": list(SEQ_LEN_BAND),
        "mean_in_band": mean_in_band,
        "all_in_band": all_in_band,
        "invalid_if_mean_out_of_band": not mean_in_band,
        "note": (
            "Density stress requires seq_len mean ∈ [45,70] under true ER "
            "p=0.15 with no path-backbone. Out-of-band mean → INVALID."
        ),
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


def _attribute_mode(
    *,
    fpr: float,
    fnr: float,
    hard_neg_ok: bool,
    pos_ok_t8: bool,
    pos_ok_t12: bool,
) -> dict[str, Any]:
    modes: list[str] = []
    fpr_bad = fpr == fpr and fpr > PREREG_FPR_MAX
    fnr_bad = fnr == fnr and fnr > PREREG_FNR_MAX
    compiler_clean = (
        (not fpr_bad) and (not fnr_bad) and (fpr == fpr) and (fnr == fnr)
    )
    acc_drop = not (hard_neg_ok and pos_ok_t8 and pos_ok_t12)

    if fpr_bad:
        modes.append("A")
    if fnr_bad:
        modes.append("B")
    if compiler_clean and acc_drop:
        modes.append("C")

    if not modes and compiler_clean and not acc_drop:
        label = "PASS"
    elif not modes:
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
        "acc_drop_under_clean_A_hat": bool(compiler_clean and acc_drop),
        "hard_neg_ok": hard_neg_ok,
        "pos_ok_T8": pos_ok_t8,
        "pos_ok_T12": pos_ok_t12,
    }


def ensure_density_data(
    data_path: Path,
    report_path: Path,
    *,
    n_pos: int,
    n_neg: int,
    seed: int,
    force_regen: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if data_path.exists() and report_path.exists() and not force_regen:
        rows = load_jsonl(data_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if len(rows) >= (n_pos + n_neg):
            return rows, report
    examples, report = generate_sheaf_density_stress_cell(
        seed=seed, n_pos=n_pos, n_neg=n_neg
    )
    write_jsonl(data_path, examples)
    write_report(report_path, report)
    return load_jsonl(data_path), report


def run_density_stress(
    *,
    ckpt_path: Path = DEFAULT_CKPT,
    data_path: Path = DEFAULT_DATA,
    gen_report_path: Path = DEFAULT_GEN_REPORT,
    out_path: Path = DEFAULT_OUT,
    n_pos: int = 64,
    n_neg: int = 64,
    seed: int = DENSITY_SEED,
    force_regen: bool = False,
) -> dict[str, Any]:
    t0 = time.time()

    # 1) Lock prereg floors BEFORE any metric interpretation.
    prereg_floors = _prereg_floors_block()

    rows, gen_report = ensure_density_data(
        data_path,
        gen_report_path,
        n_pos=n_pos,
        n_neg=n_neg,
        seed=seed,
        force_regen=force_regen,
    )
    seq_stats = _seq_len_stats(rows)
    hard_neg_verify = _verify_hard_negs_in_rows(rows)

    run_valid = bool(seq_stats["mean_in_band"])
    invalid_reason = None
    if not run_valid:
        invalid_reason = (
            f"seq_len mean={seq_stats['mean']:.4f} not in sealed band "
            f"{list(SEQ_LEN_BAND)} under true ER digraph p={DENSITY_P} "
            f"n={DENSITY_N} (no path-backbone). Density stress INVALID — stop."
        )
        print(f"[density-stress] INVALID: {invalid_reason}", file=sys.stderr)

    emp_ps = [float(r.get("p", float("nan"))) for r in rows]
    p_emp_mean = _mean(emp_ps)

    # Even on INVALID we may still compute telemetry; prereg.pass stays false.
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

    a_hat = _a_hat_fpr_fnr(model, rows, max_nodes=max_nodes)

    by_T: dict[str, Any] = {}
    for T in EVAL_T_VALUES:
        stats = _eval_reachability(model, rows, max_nodes=max_nodes, T=T)
        by_T[f"T{T}"] = stats
        print(
            f"[density-stress] T={T} overall={stats['overall_acc']:.4f} "
            f"hard_neg={stats['hard_neg_acc']:.4f} "
            f"pos_K8={stats['positive_acc']:.4f}",
            file=sys.stderr,
        )

    t8 = by_T["T8"]
    t12 = by_T["T12"]
    fpr = float(a_hat["FPR"])
    fnr = float(a_hat["FNR"])
    hard_neg_t8 = float(t8["hard_neg_acc"])
    hard_neg_t12 = float(t12["hard_neg_acc"])
    pos_t8 = float(t8["positive_acc"])
    pos_t12 = float(t12["positive_acc"])

    hard_ok = (
        hard_neg_t8 == hard_neg_t8
        and hard_neg_t12 == hard_neg_t12
        and hard_neg_t8 >= PREREG_HARD_NEG
        and hard_neg_t12 >= PREREG_HARD_NEG
    )
    pos_ok_t8 = pos_t8 == pos_t8 and pos_t8 >= PREREG_POS_REACH
    pos_ok_t12 = pos_t12 == pos_t12 and pos_t12 >= PREREG_POS_REACH
    compiler_ok = (
        fpr == fpr
        and fnr == fnr
        and fpr <= PREREG_FPR_MAX
        and fnr <= PREREG_FNR_MAX
    )

    attribution = _attribute_mode(
        fpr=fpr,
        fnr=fnr,
        hard_neg_ok=hard_ok,
        pos_ok_t8=pos_ok_t8,
        pos_ok_t12=pos_ok_t12,
    )

    prereg_pass = bool(
        run_valid and compiler_ok and hard_ok and pos_ok_t8 and pos_ok_t12
    )

    artifact: dict[str, Any] = {
        "cycle": "CYCLE_SHEAF_DENSITY_STRESS",
        "mode": "MEASURE",
        "science_open": False,
        "run_id": f"sheaf-density-stress-{uuid.uuid4().hex[:10]}",
        "run_valid": run_valid,
        "invalid_reason": invalid_reason,
        "cell": {
            "K": DENSITY_K,
            "n": DENSITY_N,
            "p_requested": DENSITY_P,
            "T_values": list(EVAL_T_VALUES),
            "construction": "true_er_digraph",
            "path_backbone": False,
            "note": (
                "True ER digraph Bernoulli(p=0.15); no path-backbone density drop. "
                "If seq_len mean ∉ [45,70], INVALID."
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
            "hard_neg_discipline": hard_neg_verify,
            "p_empirical_mean": p_emp_mean,
            "construction": "true_er_digraph",
            "path_backbone": False,
        },
        "seq_len_stats": seq_stats,
        "A_hat_vs_gold": a_hat,
        "reachability": by_T,
        "telemetry": {
            "reachability_overall_T8": by_T["T8"]["overall_acc"],
            "reachability_overall_T12": by_T["T12"]["overall_acc"],
            "hard_neg_T8": hard_neg_t8,
            "hard_neg_T12": hard_neg_t12,
            "positive_K8_T8": pos_t8,
            "positive_K8_T12": pos_t12,
            "FPR_A_hat": fpr,
            "FNR_A_hat": fnr,
            "p_empirical_mean": p_emp_mean,
            "seq_len_mean": seq_stats["mean"],
        },
        "attribution": attribution if run_valid else {
            **attribution,
            "mode": "INVALID",
            "note": "Attribution suspended: run INVALID (seq_len mean out of band).",
        },
        "prereg": {
            "floors": prereg_floors,
            "observed": {
                "FPR_A_hat": fpr,
                "FNR_A_hat": fnr,
                "hard_neg_T8": hard_neg_t8,
                "hard_neg_T12": hard_neg_t12,
                "positive_K8_T8": pos_t8,
                "positive_K8_T12": pos_t12,
                "seq_len_mean": seq_stats["mean"],
            },
            "compiler_clean_ok": compiler_ok,
            "hard_neg_ok": hard_ok,
            "positive_T8_ok": pos_ok_t8,
            "positive_T12_ok": pos_ok_t12,
            "pass": prereg_pass,
            "fail_closed": not prereg_pass,
            "run_valid": run_valid,
            "invalid": not run_valid,
        },
        "generation_report_summary": {
            "p_empirical_mean": (gen_report or {}).get("p_empirical", {}).get("mean"),
            "token_len_mean": (gen_report or {}).get("token_len", {}).get("mean"),
            "mean_in_band": (gen_report or {}).get("token_len", {}).get("mean_in_band"),
            "construction": (gen_report or {}).get("construction"),
            "path_backbone": (gen_report or {}).get("path_backbone"),
        },
        "baseline_sheaf_seal": "af8e49f",
        "prior_red_test_merge": "7a02dea",
        "elapsed_sec": time.time() - t0,
        "science_open": False,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"[density-stress] wrote {out_path}", file=sys.stderr)
    print(
        f"[density-stress] valid={run_valid} prereg_pass={prereg_pass} "
        f"mode={artifact['attribution']['mode']} "
        f"FPR={fpr:.4f} FNR={fnr:.4f} "
        f"hard_neg_T8={hard_neg_t8:.4f} hard_neg_T12={hard_neg_t12:.4f} "
        f"pos_T8={pos_t8:.4f} pos_T12={pos_t12:.4f} "
        f"seq_mean={seq_stats['mean']:.2f} p_emp={p_emp_mean:.4f}",
        file=sys.stderr,
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CYCLE_SHEAF_DENSITY_STRESS MEASURE runner"
    )
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--gen-report", type=Path, default=DEFAULT_GEN_REPORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-pos", type=int, default=64)
    p.add_argument("--n-neg", type=int, default=64)
    p.add_argument("--seed", type=int, default=DENSITY_SEED)
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
        art = run_density_stress(
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
