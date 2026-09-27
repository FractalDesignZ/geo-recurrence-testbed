"""Overfit sanity CLI for FractalCore stalk-local (Gate 0 — MEASURE).

Balanced gate (shared helpers with FF/Geo):
  - exactly 16 y=1 with hop_distance K in [2, 6]
  - exactly 16 y=0 hard negatives (deg(s)>=1, deg(t)>=1, unreachable)
  - Pass: acc=1.0 AND CE loss < 1e-3 within <=100 steps
  - Assert disconnected stalk-ablation leak at t ~0 across T
    (atol=DEFAULT_DISCONNECT_LEAK_ATOL); fail-closed if miss

No science OPEN claims.

Usage::

    python -m reachability_gen.overfit_fractal --balanced
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import ID_HOP_MAX
from reachability_gen.models.fractal_core import (
    DEFAULT_DISCONNECT_LEAK_ATOL,
    DISCRETE_T_VALUES,
    _verify_param_parity,
)
from reachability_gen.overfit_ff import (
    BALANCED_LOSS_THRESHOLD,
    BALANCED_N_NEG,
    BALANCED_N_POS,
    _diagnose_failure,
    _per_class_accuracy,
    ensure_balanced_batch,
    ensure_id_hop_batch,
)
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

DEFAULT_T = int(ID_HOP_MAX)  # 6
DEFAULT_LR = 3e-3  # Gate0 overfit (Geo-matched); Gate1 uses 1.5e-3
DEFAULT_CLIP = 2.5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Overfit Gate 0 for torch FractalCore "
            "(MEASURE plumbing; no science OPEN). Default: --balanced."
        )
    )
    p.add_argument(
        "--examples",
        type=Path,
        default=Path("data/train_tiny.jsonl"),
        help="Train JSONL path (default: data/train_tiny.jsonl).",
    )
    p.add_argument("--steps", type=int, default=100, help="Max train steps.")
    p.add_argument("--batch-size", type=int, default=32, help="Legacy batch size.")
    p.add_argument("--d", type=int, default=64, help="Model width d.")
    p.add_argument("--T", type=int, default=DEFAULT_T, help="Max cycles T.")
    p.add_argument("--lr", type=float, default=DEFAULT_LR, help="AdamW lr.")
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CLIP, help="Grad clip.")
    p.add_argument("--seed", type=int, default=0, help="Torch RNG seed.")
    p.add_argument(
        "--loss-threshold",
        type=float,
        default=None,
        help="Target CE loss (default: 1e-3 balanced).",
    )
    p.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Do not regenerate train JSONL when examples are scarce.",
    )
    p.add_argument(
        "--mlp-expansion",
        type=int,
        default=10,
        help="MLP expansion (default 10 for FF param parity).",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=DEFAULT_MAX_NODE_ID,
        help=f"Node-slot capacity (default {DEFAULT_MAX_NODE_ID}).",
    )
    bal = p.add_mutually_exclusive_group()
    bal.add_argument(
        "--balanced",
        dest="balanced",
        action="store_true",
        default=True,
        help="Balanced 16+16 hard-negative gate (default).",
    )
    bal.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
        help="Legacy ID-hop batch gate.",
    )
    p.add_argument("--n-pos", type=int, default=BALANCED_N_POS)
    p.add_argument("--n-neg", type=int, default=BALANCED_N_NEG)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON summary path (e.g. artifacts/fractal_core_stalk_overfit.json).",
    )
    return p


def run_overfit_fractal(
    examples: list[dict[str, Any]],
    *,
    steps: int = 100,
    d: int = 64,
    T: int = DEFAULT_T,
    lr: float = DEFAULT_LR,
    grad_clip: float = DEFAULT_CLIP,
    seed: int = 0,
    loss_threshold: float = BALANCED_LOSS_THRESHOLD,
    require_per_class: bool = True,
    mlp_expansion: int = 10,
    max_nodes: int = DEFAULT_MAX_NODE_ID,
    leak_atol: float = DEFAULT_DISCONNECT_LEAK_ATOL,
    adaptive_halt: bool = False,  # ignored; soft ACT stripped
) -> dict[str, Any]:
    """Train FractalCore on a fixed batch; return Gate-0 diagnostics."""
    import torch

    from reachability_gen.models.fractal_core import (
        FractalCore,
        build_node_slot_batch,
    )
    from reachability_gen.train.fractal_trainer import FractalTrainer

    torch.manual_seed(seed)
    batch = build_node_slot_batch(examples, max_n=max_nodes)
    model = FractalCore(
        d=d,
        T=T,
        n_heads=4 if d % 4 == 0 else 2,
        mlp_expansion=mlp_expansion,
        max_nodes=max_nodes,
        max_T=max(T, max(DISCRETE_T_VALUES)),
        use_tau=True,
        apply_cycle_rmsnorm=True,
    )
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    trainer = FractalTrainer(
        model,
        lr=lr,
        weight_decay=0.01,
        grad_clip=grad_clip,
        adaptive_halt=False,
    )

    losses: list[float] = []
    accs: list[float] = []
    per_class_hist: list[dict[str, float]] = []
    passed_at: Optional[int] = None
    last_halt: dict[str, Any] = {}

    node_ids = batch["node_ids"]
    node_mask = batch["node_mask"]
    attn_mask = batch["attn_mask"]
    s_idx = batch["s_idx"]
    t_idx = batch["t_idx"]
    labels = batch["labels"]

    for step in range(1, steps + 1):
        loss, acc = trainer.train_step(
            node_ids, node_mask, attn_mask, s_idx, t_idx, labels
        )
        losses.append(loss)
        accs.append(acc)
        eval_out = trainer.eval_step(
            node_ids,
            node_mask,
            attn_mask,
            s_idx,
            t_idx,
            labels,
            return_drift=True,
        )
        eval_loss, eval_acc, drifts, halt = eval_out  # type: ignore[misc]
        last_halt = {}
        for k, v in (halt or {}).items():
            if k not in (
                "mean_halt_step",
                "mean_ponder",
                "mean_u_final",
                "halt_eps",
                "adaptive_halt",
                "T",
            ):
                continue
            if isinstance(v, bool):
                last_halt[k] = v
            elif isinstance(v, (float, int)):
                last_halt[k] = float(v)
            else:
                last_halt[k] = v
        with torch.no_grad():
            logits, _, _ = trainer.model(
                node_ids,
                node_mask,
                attn_mask,
                s_idx,
                t_idx,
            )
            preds = logits.argmax(dim=-1)
            pc = _per_class_accuracy(preds, labels)
        per_class_hist.append(pc)
        class_ok = (not require_per_class) or (
            pc.get("y0", 0.0) >= 1.0 - 1e-9 and pc.get("y1", 0.0) >= 1.0 - 1e-9
        )
        if (
            eval_loss < loss_threshold
            and eval_acc >= 1.0 - 1e-9
            and class_ok
            and passed_at is None
        ):
            passed_at = step
            losses[-1] = eval_loss
            accs[-1] = eval_acc

    final_pc = per_class_hist[-1] if per_class_hist else {}
    final_loss, final_acc = losses[-1], accs[-1]
    below = final_loss < loss_threshold or (
        passed_at is not None and final_loss < loss_threshold
    )
    # Prefer passed_at for early success even if later noise.
    if passed_at is not None:
        # Re-read best-at-pass metrics from history
        final_loss = losses[passed_at - 1]
        final_acc = accs[passed_at - 1]
        final_pc = per_class_hist[passed_at - 1]
        below = True
    acc_ok = final_acc >= 1.0 - 1e-9
    class_ok = (not require_per_class) or (
        final_pc.get("y0", 0.0) >= 1.0 - 1e-9 and final_pc.get("y1", 0.0) >= 1.0 - 1e-9
    )
    ok = (passed_at is not None) or (below and acc_ok and class_ok)
    ok = bool(ok) and acc_ok and (class_ok if require_per_class else True)

    disconnect_leak = model.disconnected_target_leak(
        examples,
        T_values=DISCRETE_T_VALUES,
        max_n=max_nodes,
        atol=leak_atol,
    )
    ok = bool(ok) and bool(disconnect_leak.get("ok"))

    return {
        "ok": bool(ok),
        "gate": "gate0_overfit_fractal_stalk",
        "cycle": "CYCLE_STALK_LOCALIZATION",
        "steps_run": steps,
        "passed_at": passed_at,
        "final_loss": float(final_loss),
        "final_acc": float(final_acc),
        "losses": losses,
        "accs": accs,
        "per_class": final_pc,
        "n_examples": len(examples),
        "label_counts": {
            "y0": sum(1 for e in examples if int(e["y"]) == 0),
            "y1": sum(1 for e in examples if int(e["y"]) == 1),
        },
        "d": d,
        "T": T,
        "lr": lr,
        "grad_clip": grad_clip,
        "mlp_expansion": mlp_expansion,
        "max_nodes": max_nodes,
        "adaptive_halt": False,
        "local_potential": True,
        "broadcast_c": False,
        "loss_threshold": loss_threshold,
        "require_per_class": require_per_class,
        "param_count": model.param_count(),
        "param_parity": parity,
        "halt_diagnostics": last_halt,
        "disconnect_leak": disconnect_leak,
        "disconnect_leak_atol": leak_atol,
        "science_open": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "FAIL: torch is required for overfit_fractal (pip install torch)",
            file=sys.stderr,
        )
        return 2

    if args.loss_threshold is None:
        loss_threshold = (
            BALANCED_LOSS_THRESHOLD if args.balanced else 0.05
        )
    else:
        loss_threshold = float(args.loss_threshold)

    meta: dict[str, Any] = {}
    if args.balanced:
        batch, note, meta = ensure_balanced_batch(
            args.examples,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            regenerate=not args.no_regenerate,
        )
        print(f"data[balanced]: {note}", file=sys.stderr)
        need = args.n_pos + args.n_neg
        if len(batch) < need:
            print(
                f"FAIL: need exactly {args.n_pos}+{args.n_neg}={need} balanced "
                f"examples, got {len(batch)}. {note}",
                file=sys.stderr,
            )
            return 1
        n1 = sum(1 for e in batch if int(e["y"]) == 1)
        n0 = sum(1 for e in batch if int(e["y"]) == 0)
        if n1 != args.n_pos or n0 != args.n_neg:
            print(
                f"FAIL: balanced counts mismatch y1={n1}/{args.n_pos} "
                f"y0={n0}/{args.n_neg}",
                file=sys.stderr,
            )
            return 1
        require_per_class = True
    else:
        batch, note = ensure_id_hop_batch(
            args.examples,
            target_n=args.batch_size,
            regenerate=not args.no_regenerate,
        )
        print(f"data: {note}", file=sys.stderr)
        if len(batch) < 2:
            print(
                f"FAIL: need ≥2 ID-hop examples, got {len(batch)}. {note}",
                file=sys.stderr,
            )
            return 1
        require_per_class = False

    try:
        result = run_overfit_fractal(
            batch,
            steps=args.steps,
            d=args.d,
            T=args.T,
            lr=args.lr,
            grad_clip=args.grad_clip,
            seed=args.seed,
            loss_threshold=loss_threshold,
            require_per_class=require_per_class,
            mlp_expansion=args.mlp_expansion,
            max_nodes=args.max_nodes,
        )
    except AssertionError as e:
        print(f"FAIL: param parity — {e}", file=sys.stderr)
        return 1

    pc = result.get("per_class") or {}
    summary = {
        "ok": result["ok"],
        "arm": "fractal_core",
        "gate": "gate0",
        "balanced": bool(args.balanced),
        "final_loss": result["final_loss"],
        "final_acc": result["final_acc"],
        "per_class": {
            "y0_acc": pc.get("y0"),
            "y1_acc": pc.get("y1"),
            "y0_correct": pc.get("y0_correct"),
            "y1_correct": pc.get("y1_correct"),
            "y0_n": pc.get("y0_n"),
            "y1_n": pc.get("y1_n"),
        },
        "steps": result["steps_run"],
        "passed_at": result["passed_at"],
        "n_examples": result["n_examples"],
        "label_counts": result["label_counts"],
        "loss_threshold": loss_threshold,
        "T": result["T"],
        "param_count": result["param_count"],
        "param_parity": result["param_parity"],
        "halt_diagnostics": result.get("halt_diagnostics"),
        "disconnect_leak": result.get("disconnect_leak"),
        "disconnect_leak_atol": result.get("disconnect_leak_atol"),
        "local_potential": True,
        "broadcast_c": False,
        "cycle": "CYCLE_STALK_LOCALIZATION",
        "reject_reasons": meta.get("reject_reasons"),
        "science_open": False,
    }
    print(json.dumps(summary, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"overfit_fractal: balanced={args.balanced} n={result['n_examples']} "
        f"params={result['param_count']} T={result['T']} "
        f"steps={result['steps_run']} passed_at={result['passed_at']} "
        f"final_loss={result['final_loss']:.6f} final_acc={result['final_acc']:.4f} "
        f"acc_y0={pc.get('y0')} acc_y1={pc.get('y1')}",
        file=sys.stderr,
    )

    if result["ok"]:
        print(
            f"PASS: fractal Gate0 (loss < {loss_threshold}, acc=1.0)",
            file=sys.stderr,
        )
        return 0

    diag = _diagnose_failure(
        result["losses"],
        result["accs"],
        loss_threshold=loss_threshold,
        per_class={"y0": pc.get("y0", 0.0), "y1": pc.get("y1", 0.0)}
        if require_per_class
        else None,
    )
    print(f"FAIL: fractal Gate0 — {diag}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
