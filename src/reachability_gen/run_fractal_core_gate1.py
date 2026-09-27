"""Gate 1: train FractalCore on id_2k (30 epochs) + matched-OOD eval (MEASURE).

Protocol (matched to bound30 recurrent where sensible)
------------------------------------------------------
- Dataset: exact ``data/id_2k.jsonl`` (never regenerated here).
- Epochs: 30 locked; no early stop (track best val-acc ckpt).
- LR / clip: ``lr=1.5e-3``, ``grad_clip=2.5`` (bound30 Phase-A recurrent).
- Architecture: d=64, mlp×10, T_train=6, RMSNorm +c, ACT soft halt.
- Param parity: ``_verify_param_parity`` ±5% of FF ~121218 (fail-closed).
- Eval: ``data/covariate_matched_ood.jsonl`` (regenerate if missing).
  Modes: fixed T=6 + dynamic T∈{8,12,16}; adaptive + fixed-unroll variants.

Writes ``artifacts/fractal_core_gate1_matched_ood.json`` with
``science_open: false`` always. Mandelbrot analogy is aspirational only.

Usage::

    python -m reachability_gen.run_fractal_core_gate1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import HOP_UNREACHABLE, OOD_HOP_VALUES
from reachability_gen.gen_covariate_matched_ood import (
    verify_covariate_matched_ood,
)
from reachability_gen.models.fractal_core import (
    DEFAULT_HALT_EPS,
    MANDELBROT_ANALOGY_NOTE,
    FractalCore,
    _verify_param_parity,
    build_node_slot_batch,
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_id_2k_rematch import FF_BASELINE_PARAMS
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

DEFAULT_EPOCHS = 30
DEFAULT_D = 64
DEFAULT_T = 6
DEFAULT_MLP = 10
DEFAULT_LR = 1.5e-3  # bound30 recurrent
DEFAULT_CLIP = 2.5
DEFAULT_BATCH = 32
DYNAMIC_T_VALUES: tuple[int, ...] = tuple(OOD_HOP_VALUES)  # 8,12,16

DEFAULT_ID = Path("data/id_2k.jsonl")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/fractal_core_gate1_matched_ood.json")
DEFAULT_CKPT = Path("artifacts/fractal_core_gate1_best.pt")
DEFAULT_OVERFIT = Path("artifacts/fractal_core_overfit.json")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _split_train_val(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    return train, val


def _n_heads(d: int) -> int:
    if d % 4 == 0:
        return 4
    if d % 2 == 0:
        return 2
    return 1


def _overall_from_hop(by_hop: dict[str, Any]) -> tuple[float, float, int]:
    n_tot = 0
    acc_sum = 0.0
    loss_sum = 0.0
    for v in by_hop.values():
        n = int(v.get("n", 0))
        if n <= 0:
            continue
        acc_sum += float(v["acc_mean"]) * n
        loss_sum += float(v["loss_mean"]) * n
        n_tot += n
    if n_tot == 0:
        return float("nan"), float("nan"), 0
    return acc_sum / n_tot, loss_sum / n_tot, n_tot


def _ensure_matched_ood(path: Path) -> None:
    if path.exists():
        rows = load_jsonl(path)
        ok, issues = verify_covariate_matched_ood(rows)
        if ok:
            return
        print(
            f"WARN: existing {path} failed verify ({issues}); regenerating",
            file=sys.stderr,
        )
    print(f"[gate1] regenerating {path} via gen_covariate_matched_ood", file=sys.stderr)
    from reachability_gen.gen_covariate_matched_ood import main as gen_main

    rc = gen_main(["--out", str(path)])
    if rc not in (0, None):
        raise RuntimeError(f"gen_covariate_matched_ood failed rc={rc}")


def _eval_split(
    model: FractalCore,
    rows: list[dict[str, Any]],
    *,
    max_nodes: int,
    T: int,
    adaptive_halt: bool,
    batch_size: int = 64,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model.eval()
    hop_losses: dict[int, list[float]] = defaultdict(list)
    hop_accs: dict[int, list[float]] = defaultdict(list)
    halt_steps: list[float] = []
    ponders: list[float] = []

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            logits, _, halt = model(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                return_halt=True,
                T=T,
                adaptive_halt=adaptive_halt,
            )
            labels = batch["labels"]
            losses = F.cross_entropy(logits, labels, reduction="none")
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).float()
            if halt and "halt_step" in halt:
                halt_steps.extend(halt["halt_step"].detach().cpu().tolist())
            if halt and "p_trajectory" in halt:
                p = halt["p_trajectory"]
                if p.numel():
                    idx = torch.arange(1, p.shape[1] + 1, device=p.device).float()
                    pond = (p * idx).sum(dim=1)
                    ponders.extend(pond.detach().cpu().tolist())
            for i, ex in enumerate(batch_rows):
                hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
                hop_losses[hop].append(float(losses[i].item()))
                hop_accs[hop].append(float(correct[i].item()))

    by_hop: dict[str, Any] = {}
    for k in sorted(set(hop_losses) | set(hop_accs)):
        by_hop[str(k)] = {
            "n": len(hop_losses[k]),
            "loss_mean": _mean(hop_losses[k]),
            "acc_mean": _mean(hop_accs[k]),
            "correct": int(round(sum(hop_accs[k]))),
        }
    overall_acc, overall_loss, n_tot = _overall_from_hop(by_hop)
    hard_neg = by_hop.get(str(HOP_UNREACHABLE), by_hop.get("-1", {}))
    pos_hops = {
        k: v
        for k, v in by_hop.items()
        if int(k) > 0
    }
    return {
        "overall_acc": overall_acc,
        "overall_loss": overall_loss,
        "n": n_tot,
        "by_hop": by_hop,
        "hard_neg_acc": float(hard_neg.get("acc_mean", float("nan"))) if hard_neg else float("nan"),
        "hard_neg_n": int(hard_neg.get("n", 0)) if hard_neg else 0,
        "pos_by_hop": pos_hops,
        "T": T,
        "adaptive_halt": adaptive_halt,
        "mean_halt_step": _mean(halt_steps),
        "mean_ponder": _mean(ponders),
    }


def _mask_leakage_note(hard_neg_acc: float) -> dict[str, Any]:
    """Document whether hard-neg is algebraically perfect under the mask."""
    algebraic_target = 1.0
    ok = hard_neg_acc == hard_neg_acc and abs(hard_neg_acc - algebraic_target) < 1e-9
    return {
        "prereg_hard_neg_target": algebraic_target,
        "observed_hard_neg_acc": hard_neg_acc,
        "algebraically_perfect": bool(ok),
        "explanation": (
            "With adjacency-only + self mask and target-node readout, "
            "unreachable (s,t) cannot receive source-seeded signal along edges. "
            + (
                "Hard-neg acc=1.0 is consistent with that algebra."
                if ok
                else (
                    "Hard-neg acc < 1.0 ⇒ residual leakage: broadcast boundary "
                    "prompt c(s,t) on all nodes, shared Φ weights, soft ACT "
                    "mixing, and/or finite-T under-propagation — not a fake mask."
                )
            )
        ),
        "science_open": False,
    }


def train_fractal_id2k(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    d: int = DEFAULT_D,
    T: int = DEFAULT_T,
    mlp_expansion: int = DEFAULT_MLP,
    lr: float = DEFAULT_LR,
    grad_clip: float = DEFAULT_CLIP,
    batch_size: int = DEFAULT_BATCH,
    seed: int = 0,
    max_nodes: int = DEFAULT_MAX_NODE_ID,
    ckpt_path: Path = DEFAULT_CKPT,
    adaptive_halt: bool = True,
) -> dict[str, Any]:
    import torch

    from reachability_gen.train.fractal_trainer import FractalTrainer

    torch.manual_seed(seed)
    model = FractalCore(
        d=d,
        T=T,
        n_heads=_n_heads(d),
        mlp_expansion=mlp_expansion,
        max_nodes=max_nodes,
        max_T=max(T, max(DYNAMIC_T_VALUES)),
        use_tau=True,
        apply_cycle_rmsnorm=True,
        halt_eps=DEFAULT_HALT_EPS,
    )
    parity = _verify_param_parity(model.param_count(), ff_baseline=FF_BASELINE_PARAMS)
    trainer = FractalTrainer(
        model,
        lr=lr,
        weight_decay=0.01,
        grad_clip=grad_clip,
        adaptive_halt=adaptive_halt,
    )
    param_count = model.param_count()
    print(
        f"[fractal] params={param_count} parity_ok={parity['within_5pct']} "
        f"d={d} mlp={mlp_expansion} T={T} lr={lr} clip={grad_clip}",
        file=sys.stderr,
    )

    best_val_acc = -1.0
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    train_hist: list[dict[str, Any]] = []
    n_sat = 0
    n_steps = 0
    val_by_hop_final: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train)).tolist()
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        for start in range(0, len(train), batch_size):
            idx = order[start : start + batch_size]
            batch_rows = [train[i] for i in idx]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            loss, acc = trainer.train_step(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                batch["labels"],
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            n_steps += 1
            if trainer.last_pre_clip_grad_norm >= grad_clip:
                n_sat += 1

        train_loss = _mean(epoch_losses)
        train_acc = _mean(epoch_accs)
        val_stats = _eval_split(
            model, val, max_nodes=max_nodes, T=T, adaptive_halt=adaptive_halt
        )
        ov_acc = float(val_stats["overall_acc"])
        train_hist.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": ov_acc,
                "val_loss": float(val_stats["overall_loss"]),
                "mean_halt_step": val_stats.get("mean_halt_step"),
            }
        )
        val_by_hop_final = val_stats["by_hop"]
        if ov_acc > best_val_acc:
            best_val_acc = ov_acc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"[fractal] epoch {epoch}/{epochs} train_acc={train_acc:.4f} "
            f"val_acc={ov_acc:.4f} (best={best_val_acc:.4f}@ep{best_epoch}) "
            f"halt={val_stats.get('mean_halt_step')}",
            file=sys.stderr,
        )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(
            {
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "state_dict": best_state,
                "arm": f"fractal-T{T}-d{d}-mlp{mlp_expansion}",
                "science_open": False,
                "hparams": {
                    "d": d,
                    "T": T,
                    "mlp_expansion": mlp_expansion,
                    "max_nodes": max_nodes,
                    "lr": lr,
                    "grad_clip": grad_clip,
                    "adaptive_halt": adaptive_halt,
                    "halt_eps": DEFAULT_HALT_EPS,
                },
            },
            ckpt_path,
        )

    return {
        "arm": f"fractal-T{T}-d{d}-mlp{mlp_expansion}",
        "param_count": param_count,
        "param_parity": parity,
        "epochs": epochs,
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "train_history": train_hist,
        "val_by_hop": val_by_hop_final,
        "grad_clip": grad_clip,
        "grad_clip_sat_rate": n_sat / n_steps if n_steps else float("nan"),
        "lr": lr,
        "d": d,
        "T": T,
        "mlp_expansion": mlp_expansion,
        "max_nodes": max_nodes,
        "adaptive_halt": adaptive_halt,
        "checkpoint_path": str(ckpt_path),
        "science_open": False,
    }


def run_gate1(
    *,
    id_data: Path = DEFAULT_ID,
    ood_data: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    ckpt_path: Path = DEFAULT_CKPT,
    overfit_path: Path = DEFAULT_OVERFIT,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = 0,
    skip_train: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    if not id_data.exists():
        raise FileNotFoundError(f"id_2k missing: {id_data}")
    _ensure_matched_ood(ood_data)

    rows = load_jsonl(id_data)
    train, val = _split_train_val(rows)
    if not train or not val:
        raise RuntimeError(f"id_2k split empty train={len(train)} val={len(val)}")

    max_n_data = max(int(r["n"]) for r in rows)
    ood_rows = load_jsonl(ood_data)
    max_n_ood = max(int(r["n"]) for r in ood_rows)
    max_nodes = max(DEFAULT_MAX_NODE_ID, max_n_data, max_n_ood)

    if skip_train and ckpt_path.exists():
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt.get("hparams", {})
        model = FractalCore(
            d=int(hp.get("d", DEFAULT_D)),
            T=int(hp.get("T", DEFAULT_T)),
            n_heads=_n_heads(int(hp.get("d", DEFAULT_D))),
            mlp_expansion=int(hp.get("mlp_expansion", DEFAULT_MLP)),
            max_nodes=int(hp.get("max_nodes", max_nodes)),
            max_T=max(int(hp.get("T", DEFAULT_T)), max(DYNAMIC_T_VALUES)),
            use_tau=True,
            apply_cycle_rmsnorm=True,
        )
        model.load_state_dict(ckpt["state_dict"])
        train_summary = {
            "skipped_train": True,
            "checkpoint_path": str(ckpt_path),
            "best_val_acc": ckpt.get("val_acc"),
            "best_epoch": ckpt.get("epoch"),
            "param_count": model.param_count(),
            "science_open": False,
        }
    else:
        train_summary = train_fractal_id2k(
            train,
            val,
            epochs=epochs,
            seed=seed,
            max_nodes=max_nodes,
            ckpt_path=ckpt_path,
        )
        import torch

        model = FractalCore(
            d=DEFAULT_D,
            T=DEFAULT_T,
            n_heads=_n_heads(DEFAULT_D),
            mlp_expansion=DEFAULT_MLP,
            max_nodes=max_nodes,
            max_T=max(DEFAULT_T, max(DYNAMIC_T_VALUES)),
            use_tau=True,
            apply_cycle_rmsnorm=True,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    # Matched OOD eval: fixed T + dynamic T; adaptive + fixed unroll.
    evals: dict[str, Any] = {}
    for label, T, ah in [
        ("fixed_T6_adaptive", DEFAULT_T, True),
        ("fixed_T6_unroll", DEFAULT_T, False),
    ]:
        evals[label] = _eval_split(
            model, ood_rows, max_nodes=max_nodes, T=T, adaptive_halt=ah
        )
        print(
            f"[gate1-ood] {label} acc={evals[label]['overall_acc']:.4f} "
            f"hard_neg={evals[label]['hard_neg_acc']:.4f}",
            file=sys.stderr,
        )

    dynamic: dict[str, Any] = {}
    for T in DYNAMIC_T_VALUES:
        key = f"dynamic_T{T}_adaptive"
        dynamic[key] = _eval_split(
            model, ood_rows, max_nodes=max_nodes, T=T, adaptive_halt=True
        )
        key_f = f"dynamic_T{T}_unroll"
        dynamic[key_f] = _eval_split(
            model, ood_rows, max_nodes=max_nodes, T=T, adaptive_halt=False
        )
        print(
            f"[gate1-ood] T={T} adaptive_acc={dynamic[key]['overall_acc']:.4f} "
            f"unroll_acc={dynamic[key_f]['overall_acc']:.4f} "
            f"hard_neg_ad={dynamic[key]['hard_neg_acc']:.4f}",
            file=sys.stderr,
        )

    # Primary table: fixed adaptive + dynamic unroll (prereg framing).
    primary = evals["fixed_T6_adaptive"]
    # Recover K≥8 positives via unroll: look at hop-8/12/16 acc under dynamic.
    recover: dict[str, Any] = {}
    for T in DYNAMIC_T_VALUES:
        stats = dynamic[f"dynamic_T{T}_unroll"]
        hop = stats["by_hop"].get(str(T), {})
        recover[str(T)] = {
            "T_unroll": T,
            "hop_acc": hop.get("acc_mean"),
            "hop_n": hop.get("n"),
            "overall_acc": stats["overall_acc"],
            "hard_neg_acc": stats["hard_neg_acc"],
        }

    leakage = _mask_leakage_note(float(primary["hard_neg_acc"]))

    overfit_summary = None
    if overfit_path.exists():
        overfit_summary = json.loads(overfit_path.read_text(encoding="utf-8"))

    artifact = {
        "cycle": "CYCLE_FRACTAL_CORE_GENESIS",
        "mode": "MEASURE",
        "science_open": False,
        "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
        "run_id": f"fractal-gate1-{uuid.uuid4().hex[:10]}",
        "dataset_id": str(id_data),
        "ood_dataset": str(ood_data),
        "train": train_summary,
        "gate0_overfit": overfit_summary,
        "ood_eval": {
            "fixed": evals,
            "dynamic": dynamic,
            "recover_K_via_unroll": recover,
            "mask_leakage": leakage,
        },
        "prereg_aspirational": {
            "hard_neg_ood_1_0_if_mask_forbids": (
                "Aspirational: hard-neg acc=1.0 on OOD if mask algebraically "
                "forbids disconnected pairs. Report observed honestly."
            ),
            "recover_K_ge_8_via_unroll": (
                "Aspirational: recover K≥8 positives via dynamic unroll. "
                "See recover_K_via_unroll table."
            ),
            "observed_hard_neg": primary["hard_neg_acc"],
            "observed_recover": recover,
        },
        "hparams_matched_to_bound30_recurrent": {
            "lr": DEFAULT_LR,
            "grad_clip": DEFAULT_CLIP,
            "epochs": epochs,
            "d": DEFAULT_D,
            "mlp_expansion": DEFAULT_MLP,
            "T_train": DEFAULT_T,
            "note": (
                "LR/clip/epochs/mlp match bound30 recurrent Geo/Loop settings; "
                "FractalCore adds adjacency mask, +c boundary, ACT halt."
            ),
        },
        "elapsed_sec": time.time() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"[gate1] wrote {out_path}", file=sys.stderr)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FractalCore Gate1 MEASURE runner")
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID)
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--overfit", type=Path, default=DEFAULT_OVERFIT)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Load existing ckpt and only run OOD eval.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    try:
        run_gate1(
            id_data=args.id_data,
            ood_data=args.ood_data,
            out_path=args.out,
            ckpt_path=args.ckpt,
            overfit_path=args.overfit,
            epochs=args.epochs,
            seed=args.seed,
            skip_train=args.skip_train,
        )
    except AssertionError as e:
        print(f"FAIL-CLOSED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
