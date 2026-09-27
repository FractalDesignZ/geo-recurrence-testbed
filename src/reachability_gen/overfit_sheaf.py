"""Overfit sanity CLI for SheafInferCore (Gate 0 — MEASURE).

Balanced gate:
  - exactly 16 y=1 with hop_distance K in [2, 6]
  - exactly 16 y=0 hard negatives
  - Pass: reachability CE < 1e-3 AND edge recon offdiag acc ≥ 0.99
    within ≤150 steps; acc=1.0 per class
  - Assert ‖h_t‖≈0 on disconnect under A_hat across T (document STE)

No science OPEN claims. Gold edges = aux recon only (not eval oracle).

Usage::

    python -m reachability_gen.overfit_sheaf --balanced \\
      --out artifacts/sheaf_infer_overfit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import ID_HOP_MAX
from reachability_gen.models.sheaf_infer_core import (
    DEFAULT_DISCONNECT_LEAK_ATOL,
    DEFAULT_EDGE_RECON_WEIGHT,
    DEFAULT_GATE_THETA,
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
DEFAULT_LR = 3e-3
DEFAULT_CLIP = 2.5
DEFAULT_STEPS = 150
DEFAULT_MLP = 12
EDGE_RECON_ACC_THRESHOLD = 0.99


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Overfit Gate 0 for SheafInferCore "
            "(MEASURE; no science OPEN). Default: --balanced."
        )
    )
    p.add_argument(
        "--examples",
        type=Path,
        default=Path("data/train_tiny.jsonl"),
    )
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CLIP)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss-threshold", type=float, default=None)
    p.add_argument("--no-regenerate", action="store_true")
    p.add_argument("--mlp-expansion", type=int, default=DEFAULT_MLP)
    p.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODE_ID)
    p.add_argument("--gate-theta", type=float, default=DEFAULT_GATE_THETA)
    p.add_argument(
        "--gate-mode",
        choices=("ste", "gumbel"),
        default="ste",
    )
    p.add_argument(
        "--edge-recon-weight",
        type=float,
        default=DEFAULT_EDGE_RECON_WEIGHT,
        help="Aux BCE weight on E_hat vs gold (TRAIN only).",
    )
    p.add_argument(
        "--edge-recon-acc-threshold",
        type=float,
        default=EDGE_RECON_ACC_THRESHOLD,
    )
    bal = p.add_mutually_exclusive_group()
    bal.add_argument(
        "--balanced", dest="balanced", action="store_true", default=True
    )
    bal.add_argument(
        "--no-balanced", dest="balanced", action="store_false"
    )
    p.add_argument("--n-pos", type=int, default=BALANCED_N_POS)
    p.add_argument("--n-neg", type=int, default=BALANCED_N_NEG)
    p.add_argument("--out", type=Path, default=None)
    return p


def run_overfit_sheaf(
    examples: list[dict[str, Any]],
    *,
    steps: int = DEFAULT_STEPS,
    d: int = 64,
    T: int = DEFAULT_T,
    lr: float = DEFAULT_LR,
    grad_clip: float = DEFAULT_CLIP,
    seed: int = 0,
    loss_threshold: float = BALANCED_LOSS_THRESHOLD,
    require_per_class: bool = True,
    mlp_expansion: int = DEFAULT_MLP,
    max_nodes: int = DEFAULT_MAX_NODE_ID,
    leak_atol: float = DEFAULT_DISCONNECT_LEAK_ATOL,
    gate_theta: float = DEFAULT_GATE_THETA,
    gate_mode: str = "ste",
    edge_recon_weight: float = DEFAULT_EDGE_RECON_WEIGHT,
    edge_recon_acc_threshold: float = EDGE_RECON_ACC_THRESHOLD,
) -> dict[str, Any]:
    """Train SheafInferCore on a fixed batch; return Gate-0 diagnostics."""
    import torch

    from reachability_gen.models.sheaf_infer_core import (
        SheafInferCore,
        build_sheaf_batch,
    )
    from reachability_gen.train.sheaf_trainer import SheafTrainer

    torch.manual_seed(seed)
    batch = build_sheaf_batch(examples, max_n=max_nodes)
    model = SheafInferCore(
        d=d,
        T=T,
        mlp_expansion=mlp_expansion,
        max_nodes=max_nodes,
        max_T=max(T, max(DISCRETE_T_VALUES)),
        use_tau=False,
        apply_cycle_rmsnorm=False,
        residual_alpha=1.0,
        gate_theta=gate_theta,
        gate_mode=gate_mode,
        gate_detach_diffusion=True,
    )
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    trainer = SheafTrainer(
        model,
        lr=lr,
        weight_decay=0.01,
        grad_clip=grad_clip,
        edge_recon_weight=edge_recon_weight,
    )

    losses: list[float] = []
    accs: list[float] = []
    recon_accs: list[float] = []
    ce_losses: list[float] = []
    per_class_hist: list[dict[str, float]] = []
    passed_at: Optional[int] = None
    last_halt: dict[str, Any] = {}

    node_ids = batch["node_ids"]
    node_mask = batch["node_mask"]
    edge_index = batch["edge_index"]
    edge_mask = batch["edge_mask"]
    gold_adj = batch["gold_adj"]
    s_idx = batch["s_idx"]
    t_idx = batch["t_idx"]
    labels = batch["labels"]

    for step in range(1, steps + 1):
        loss, acc = trainer.train_step(
            node_ids,
            node_mask,
            edge_index,
            edge_mask,
            s_idx,
            t_idx,
            labels,
            gold_adj,
        )
        losses.append(loss)
        accs.append(acc)
        ce_losses.append(trainer.last_ce)
        recon_accs.append(trainer.last_edge_recon_acc)
        eval_out = trainer.eval_step(
            node_ids,
            node_mask,
            edge_index,
            edge_mask,
            s_idx,
            t_idx,
            labels,
            gold_adj,
            return_drift=True,
        )
        eval_loss, eval_acc, drifts, halt = eval_out  # type: ignore[misc]
        last_halt = {}
        for k, v in (halt or {}).items():
            if k not in (
                "mean_halt_step",
                "mean_ponder",
                "adaptive_halt",
                "T",
                "gate_theta",
                "gate_mode",
                "hard_A_oracle",
            ):
                continue
            if isinstance(v, bool):
                last_halt[k] = v
            elif isinstance(v, (float, int)):
                last_halt[k] = float(v) if not isinstance(v, bool) else v
            else:
                last_halt[k] = v
        eval_recon = trainer.last_edge_recon_acc
        with torch.no_grad():
            logits, _, _ = trainer.model(
                node_ids,
                node_mask,
                edge_index,
                edge_mask,
                s_idx,
                t_idx,
            )
            preds = logits.argmax(dim=-1)
            pc = _per_class_accuracy(preds, labels)
        per_class_hist.append(pc)
        class_ok = (not require_per_class) or (
            pc.get("y0", 0.0) >= 1.0 - 1e-9 and pc.get("y1", 0.0) >= 1.0 - 1e-9
        )
        recon_ok = (
            eval_recon == eval_recon and eval_recon >= edge_recon_acc_threshold
        )
        # Reachability CE (eval_loss) — not total train loss with aux.
        if (
            eval_loss < loss_threshold
            and eval_acc >= 1.0 - 1e-9
            and class_ok
            and recon_ok
            and passed_at is None
        ):
            passed_at = step
            losses[-1] = eval_loss
            accs[-1] = eval_acc
            recon_accs[-1] = eval_recon
            ce_losses[-1] = eval_loss

    final_pc = per_class_hist[-1] if per_class_hist else {}
    final_loss = ce_losses[-1] if ce_losses else losses[-1]
    final_acc = accs[-1]
    final_recon = recon_accs[-1] if recon_accs else float("nan")
    if passed_at is not None:
        final_loss = ce_losses[passed_at - 1]
        final_acc = accs[passed_at - 1]
        final_pc = per_class_hist[passed_at - 1]
        final_recon = recon_accs[passed_at - 1]
    acc_ok = final_acc >= 1.0 - 1e-9
    class_ok = (not require_per_class) or (
        final_pc.get("y0", 0.0) >= 1.0 - 1e-9 and final_pc.get("y1", 0.0) >= 1.0 - 1e-9
    )
    recon_ok = final_recon == final_recon and final_recon >= edge_recon_acc_threshold
    below = final_loss < loss_threshold
    ok = (passed_at is not None) or (below and acc_ok and class_ok and recon_ok)
    ok = bool(ok) and acc_ok and below and recon_ok
    if require_per_class:
        ok = bool(ok) and class_ok

    disconnect_norm = model.disconnected_target_norm(
        examples,
        T_values=DISCRETE_T_VALUES,
        max_n=max_nodes,
        atol=leak_atol,
    )
    disconnect_abl = model.disconnected_stalk_ablation_leak(
        examples,
        T_values=DISCRETE_T_VALUES,
        max_n=max_nodes,
        atol=leak_atol,
    )
    # Primary Gate0 disconnect: ‖h_t‖=0; document ablation too.
    ok = bool(ok) and bool(disconnect_norm.get("ok"))

    ste_note = None
    if not disconnect_norm.get("ok"):
        ste_note = (
            "Disconnect ‖h_t‖ exceeded atol under inferred A_hat. "
            f"gate_mode={gate_mode}. Possible STE/Gumbel softening or "
            "phantom edges — fail-closed, science_open=false."
        )

    return {
        "ok": bool(ok),
        "gate": "gate0_overfit_sheaf_infer",
        "cycle": "CYCLE_SHEAF_INFERENCE",
        "steps_run": steps,
        "passed_at": passed_at,
        "final_loss": float(final_loss),
        "final_acc": float(final_acc),
        "final_edge_recon_acc": float(final_recon),
        "edge_recon_acc_threshold": edge_recon_acc_threshold,
        "losses": losses,
        "accs": accs,
        "ce_losses": ce_losses,
        "recon_accs": recon_accs,
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
        "gate_theta": gate_theta,
        "gate_mode": gate_mode,
        "edge_recon_weight": edge_recon_weight,
        "aux_edge_recon_train_only": True,
        "hard_A_oracle_eval": False,
        "local_potential": True,
        "broadcast_c": False,
        "loss_threshold": loss_threshold,
        "require_per_class": require_per_class,
        "param_count": model.param_count(),
        "param_parity": parity,
        "halt_diagnostics": last_halt,
        "disconnect_target_norm": disconnect_norm,
        "disconnect_stalk_ablation": disconnect_abl,
        "disconnect_leak_atol": leak_atol,
        "ste_softening_note": ste_note,
        "baseline_stalk_seal": "b144dac",
        "science_open": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "FAIL: torch is required for overfit_sheaf (pip install torch)",
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
        result = run_overfit_sheaf(
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
            gate_theta=args.gate_theta,
            gate_mode=args.gate_mode,
            edge_recon_weight=args.edge_recon_weight,
            edge_recon_acc_threshold=args.edge_recon_acc_threshold,
        )
    except AssertionError as e:
        print(f"FAIL: param parity — {e}", file=sys.stderr)
        return 1

    pc = result.get("per_class") or {}
    summary = {
        "ok": result["ok"],
        "arm": "sheaf_infer_core",
        "gate": "gate0",
        "balanced": bool(args.balanced),
        "final_loss": result["final_loss"],
        "final_acc": result["final_acc"],
        "final_edge_recon_acc": result["final_edge_recon_acc"],
        "edge_recon_acc_threshold": result["edge_recon_acc_threshold"],
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
        "disconnect_target_norm": result.get("disconnect_target_norm"),
        "disconnect_stalk_ablation": result.get("disconnect_stalk_ablation"),
        "disconnect_leak_atol": result.get("disconnect_leak_atol"),
        "ste_softening_note": result.get("ste_softening_note"),
        "gate_theta": result["gate_theta"],
        "gate_mode": result["gate_mode"],
        "aux_edge_recon_train_only": True,
        "hard_A_oracle_eval": False,
        "local_potential": True,
        "broadcast_c": False,
        "cycle": "CYCLE_SHEAF_INFERENCE",
        "baseline_stalk_seal": "b144dac",
        "reject_reasons": meta.get("reject_reasons"),
        "science_open": False,
    }
    print(json.dumps(summary, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"overfit_sheaf: balanced={args.balanced} n={result['n_examples']} "
        f"params={result['param_count']} T={result['T']} "
        f"steps={result['steps_run']} passed_at={result['passed_at']} "
        f"final_loss={result['final_loss']:.6f} final_acc={result['final_acc']:.4f} "
        f"edge_recon={result['final_edge_recon_acc']:.4f} "
        f"acc_y0={pc.get('y0')} acc_y1={pc.get('y1')} "
        f"disconnect_ok={result['disconnect_target_norm'].get('ok')}",
        file=sys.stderr,
    )

    if result["ok"]:
        print(
            f"PASS: sheaf Gate0 (CE < {loss_threshold}, recon≥"
            f"{args.edge_recon_acc_threshold}, ‖h_t‖≈0 disconnect)",
            file=sys.stderr,
        )
        return 0

    diag = _diagnose_failure(
        result["ce_losses"],
        result["accs"],
        loss_threshold=loss_threshold,
        per_class={"y0": pc.get("y0", 0.0), "y1": pc.get("y1", 0.0)}
        if require_per_class
        else None,
    )
    print(f"FAIL: sheaf Gate0 — {diag}", file=sys.stderr)
    if result.get("ste_softening_note"):
        print(f"NOTE: {result['ste_softening_note']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
