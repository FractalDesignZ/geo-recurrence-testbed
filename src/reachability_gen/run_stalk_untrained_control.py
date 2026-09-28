"""Standing untrained-control for stalk-local FractalCore (fail-closed).

Matches sealed CYCLE_STALK_LOCALIZATION OPEN config: local stalk@s / probe@t,
no c broadcast, no soft ACT, hard A_ij mask, discrete T ∈ {6,8,12,16}.
Scores untrained vs sealed Gate1 ckpt on matched-OOD with hard-neg, K strata,
and overall. science_open never auto-widened; seals revoked only if bake-in
proven (untrained ≈ sealed ≈ 1.0 + agreement ≈ 1.0).

Usage::

    python -m reachability_gen.run_stalk_untrained_control
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import HOP_UNREACHABLE
from reachability_gen.models.fractal_core import (
    DISCRETE_T_VALUES,
    FractalCore,
    build_node_slot_batch,
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

DEFAULT_CKPT = Path("artifacts/fractal_core_stalk_gate1_best.pt")
DEFAULT_OOD = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_untrained_control_audit.json")
T_VALUES: tuple[int, ...] = tuple(DISCRETE_T_VALUES)  # 6,8,12,16
FOCUS_T = 16


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _n_heads(d: int) -> int:
    if d % 4 == 0:
        return 4
    if d % 2 == 0:
        return 2
    return 1


def document_stalk_init(model: FractalCore) -> dict[str, Any]:
    """Record stalk-local init (NOT an edge-gate bake-in oracle like sheaf)."""
    stalk_w = model.stalk_proj.weight.detach()
    probe_w = model.probe_proj.weight.detach()
    eye = stalk_w.new_zeros(stalk_w.shape)
    eye.fill_diagonal_(1.0)
    return {
        "architecture": {
            "broadcast_c": False,
            "soft_ACT": False,
            "local_potential": True,
            "stalk_slot": "s",
            "probe_slot": "t",
            "intermediates": 0,
            "discrete_T_values": list(T_VALUES),
            "hard_A_ij_mask": True,
        },
        "stalk_proj_eye": bool(
            float((stalk_w - eye).abs().max()) < 1e-6
            and float(model.stalk_proj.bias.detach().abs().sum()) == 0.0
        ),
        "probe_proj_eye": bool(
            float((probe_w - eye).abs().max()) < 1e-6
            and float(model.probe_proj.bias.detach().abs().sum()) == 0.0
        ),
        "residual_alpha": float(model.residual_alpha),
        "head_weight_norm": float(model.head.weight.detach().norm().item()),
        "head_bias": model.head.bias.detach().tolist(),
        "mechanism": (
            "Stalk-local FractalCore: hard adjacency mask + stalk@s / probe@t "
            "potentials (eye init), intermediates=0, no c broadcast, no soft ACT. "
            "Unlike SheafInferCore, there is no listed-edge gate bias / energy "
            "readout oracle at init — untrained performance is NOT expected to "
            "reproduce sealed Gate1 ceilings unless a hidden bake-in exists."
        ),
        "science_open": False,
    }


def _build_model(
    hp: dict[str, Any],
    *,
    load_state: Optional[dict[str, Any]] = None,
    seed: int = 0,
) -> FractalCore:
    import torch

    torch.manual_seed(seed)
    d = int(hp.get("d", 64))
    T = int(hp.get("T", 6))
    max_nodes = int(hp.get("max_nodes", DEFAULT_MAX_NODE_ID))
    mlp = int(hp.get("mlp_expansion", 10))
    model = FractalCore(
        d=d,
        T=T,
        n_heads=_n_heads(d),
        mlp_expansion=mlp,
        max_nodes=max_nodes,
        max_T=max(T, max(T_VALUES)),
        use_tau=True,
        apply_cycle_rmsnorm=True,
    )
    if load_state is not None:
        model.load_state_dict(load_state)
    model.eval()
    return model


def _eval_rows(
    model: FractalCore,
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    hop_accs: dict[int, list[float]] = defaultdict(list)
    hop_losses: dict[int, list[float]] = defaultdict(list)
    preds_all: list[Any] = []
    labels_all: list[Any] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            logits, _, _ = model(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                return_halt=True,
                T=T,
                adaptive_halt=False,
            )
            labels = batch["labels"]
            losses = F.cross_entropy(logits, labels, reduction="none")
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).float()
            for i, ex in enumerate(batch_rows):
                hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
                hop_accs[hop].append(float(correct[i].item()))
                hop_losses[hop].append(float(losses[i].item()))
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())

    by_hop: dict[str, Any] = {}
    for k in sorted(set(hop_accs) | set(hop_losses)):
        by_hop[str(k)] = {
            "n": len(hop_accs[k]),
            "acc_mean": _mean(hop_accs[k]),
            "loss_mean": _mean(hop_losses[k]),
            "correct": int(round(sum(hop_accs[k]))),
        }
    n_tot = sum(v["n"] for v in by_hop.values())
    overall = (
        sum(v["acc_mean"] * v["n"] for v in by_hop.values()) / n_tot
        if n_tot
        else float("nan")
    )
    hard = by_hop.get(str(HOP_UNREACHABLE), by_hop.get("-1", {}))
    P = torch.cat(preds_all)
    L = torch.cat(labels_all)
    y0 = L == 0
    y1 = L == 1
    fpr = float(((P == 1) & y0).sum() / y0.sum()) if y0.any() else float("nan")
    fnr = float(((P == 0) & y1).sum() / y1.sum()) if y1.any() else float("nan")
    return {
        "overall_acc": overall,
        "hard_neg_acc": float(hard.get("acc_mean", float("nan"))) if hard else float("nan"),
        "hard_neg_n": int(hard.get("n", 0)) if hard else 0,
        "by_hop": by_hop,
        "n": n_tot,
        "T": T,
        "fpr": fpr,
        "fnr": fnr,
        "preds": P,
        "labels": L,
    }


def _decide_verdict(m16: dict[str, Any]) -> tuple[str, str, bool]:
    """Fail-closed verdict. Only revoke seal if bake-in proven."""
    u_overall = float(m16["untrained"]["overall_acc"])
    u_hard = float(m16["untrained"]["hard_neg_acc"])
    t_overall = float(m16["trained"]["overall_acc"])
    t_hard = float(m16["trained"]["hard_neg_acc"])
    agree = float(m16["prediction_agreement"])
    k16_u = float(m16.get("K_strata", {}).get("16", {}).get("untrained_acc", float("nan")))
    k16_t = float(m16.get("K_strata", {}).get("16", {}).get("trained_acc", float("nan")))

    bake_in = (
        abs(u_overall - 1.0) < 1e-6
        and abs(u_hard - 1.0) < 1e-6
        and abs(agree - 1.0) < 1e-6
        and (k16_u == k16_u and abs(k16_u - 1.0) < 1e-6)
        and abs(t_overall - 1.0) < 1e-2  # sealed near ceiling
    )
    if bake_in:
        return (
            "SEALS_COMPROMISED_INIT_BAKE_IN",
            (
                "Untrained stalk-local FractalCore reproduces sealed Gate1 "
                "matched-OOD T16 overall/hard-neg/K16 ≈ 1.0 with ~100% prediction "
                "agreement vs sealed ckpt. OPEN stalk claim is at INVALIDATION risk "
                "(init+architecture bake-in)."
            ),
            True,
        )

    # Mid / chance untrained while trained is high → OPEN still contingent
    if u_overall < 0.75 and t_overall > 0.9 and agree < 0.95:
        if u_overall < 0.55:
            band = "near-chance"
        else:
            band = "mid (~0.6)"
        return (
            "OPEN_STILL_CONTINGENT_NEEDS_MULTI_SEED",
            (
                f"Untrained matched-OOD T16 overall={u_overall:.3f} ({band}); "
                f"sealed trained overall={t_overall:.3f}, hard-neg={t_hard:.3f}, "
                f"K16_trained={k16_t:.3f}, agreement={agree:.3f}. "
                "Bake-in NOT proven. Stalk OPEN remains standing but contingent — "
                "needs multi-seed trained reconfirm. Do not silently widen science_open."
            ),
            False,
        )

    return (
        "PARTIAL_OR_AMBIGUOUS",
        (
            f"Mixed untrained vs sealed (u_overall={u_overall:.3f}, "
            f"t_overall={t_overall:.3f}, agree={agree:.3f}). "
            "Prefer fail-closed: do not revoke stalk OPEN without bake-in proof; "
            "do not widen science_open."
        ),
        False,
    )


def run_audit(
    *,
    ckpt_path: Path = DEFAULT_CKPT,
    ood_path: Path = DEFAULT_OOD,
    out_path: Path = DEFAULT_OUT,
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    if not ood_path.exists():
        raise FileNotFoundError(f"missing matched-OOD: {ood_path}")

    rows = load_jsonl(ood_path)
    max_nodes = max(DEFAULT_MAX_NODE_ID, max(int(r["n"]) for r in rows))

    ckpt_meta: dict[str, Any]
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = dict(ckpt.get("hparams") or {})
        hp.setdefault("d", 64)
        hp.setdefault("T", 6)
        hp.setdefault("mlp_expansion", 10)
        hp.setdefault("max_nodes", max_nodes)
        hp["max_nodes"] = max(int(hp["max_nodes"]), max_nodes)
        trained = _build_model(hp, load_state=ckpt["state_dict"], seed=seed)
        ckpt_meta = {
            "path": str(ckpt_path),
            "present": True,
            "epoch": ckpt.get("epoch"),
            "val_acc": ckpt.get("val_acc"),
            "arm": ckpt.get("arm"),
            "cycle": ckpt.get("cycle"),
            "hparams": hp,
            "sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        }
    else:
        hp = {
            "d": 64,
            "T": 6,
            "mlp_expansion": 10,
            "max_nodes": max_nodes,
            "local_potential": True,
            "broadcast_c": False,
            "adaptive_halt": False,
        }
        trained = None
        ckpt_meta = {
            "path": str(ckpt_path),
            "present": False,
            "note": "sealed ckpt missing; untrained-only scores reported",
            "hparams": hp,
        }

    untrained = _build_model(hp, seed=seed)
    init_doc = document_stalk_init(untrained)

    report: dict[str, Any] = {
        "audit": "FAIL_CLOSED_STALK_UNTRAINED_CONTROL",
        "cycle": "CYCLE_STALK_LOCALIZATION",
        "base_sha": _git_sha(),
        "ckpt": ckpt_meta,
        "init_doc": init_doc,
        "ood": {"path": str(ood_path), "n": len(rows)},
        "T_values": list(T_VALUES),
        "focus_T": FOCUS_T,
        "by_T": {},
        "science_open": False,
        "policy": (
            "Do not silently widen science_open. Only revoke stalk OPEN if "
            "bake-in proven (untrained≈sealed≈1.0)."
        ),
    }

    for T in T_VALUES:
        u = _eval_rows(untrained, rows, T=int(T), max_nodes=max_nodes)
        if trained is not None:
            t = _eval_rows(trained, rows, T=int(T), max_nodes=max_nodes)
            agree = float((u["preds"] == t["preds"]).float().mean())
            t_clean = {k: v for k, v in t.items() if k not in ("preds", "labels")}
            delta_overall = t["overall_acc"] - u["overall_acc"]
            delta_hard = t["hard_neg_acc"] - u["hard_neg_acc"]
        else:
            t_clean = None
            agree = float("nan")
            delta_overall = float("nan")
            delta_hard = float("nan")
        u_clean = {k: v for k, v in u.items() if k not in ("preds", "labels")}
        cell: dict[str, Any] = {
            "untrained": u_clean,
            "trained": t_clean,
            "prediction_agreement": agree,
            "delta_overall": delta_overall,
            "delta_hard_neg": delta_hard,
            "K_strata": {},
        }
        hops = u["by_hop"]
        for hop_key in sorted(hops, key=lambda x: int(x)):
            if int(hop_key) > 0:
                cell["K_strata"][hop_key] = {
                    "untrained_acc": hops[hop_key]["acc_mean"],
                    "trained_acc": (
                        t_clean["by_hop"][hop_key]["acc_mean"]
                        if t_clean and hop_key in t_clean["by_hop"]
                        else float("nan")
                    ),
                    "n": hops[hop_key]["n"],
                }
        report["by_T"][str(T)] = cell

    foc = report["by_T"].get(str(FOCUS_T))
    if foc is None or foc.get("trained") is None:
        verdict = "MISSING_SEALED_CKPT_OR_T16"
        claim = "Cannot close bake-in verdict without sealed ckpt + T16."
        seals_bad = False
    else:
        verdict, claim, seals_bad = _decide_verdict(foc)

    report["verdict"] = verdict
    report["verdict_claim"] = claim
    report["seals_invalidated"] = bool(seals_bad)
    report["focus_summary"] = {
        "T": FOCUS_T,
        "untrained_overall": foc["untrained"]["overall_acc"] if foc else None,
        "trained_overall": (foc["trained"]["overall_acc"] if foc and foc["trained"] else None),
        "untrained_hard_neg": foc["untrained"]["hard_neg_acc"] if foc else None,
        "trained_hard_neg": (foc["trained"]["hard_neg_acc"] if foc and foc["trained"] else None),
        "agreement": foc["prediction_agreement"] if foc else None,
        "K8_untrained": foc["K_strata"].get("8", {}).get("untrained_acc") if foc else None,
        "K8_trained": foc["K_strata"].get("8", {}).get("trained_acc") if foc else None,
        "K12_untrained": foc["K_strata"].get("12", {}).get("untrained_acc") if foc else None,
        "K12_trained": foc["K_strata"].get("12", {}).get("trained_acc") if foc else None,
        "K16_untrained": foc["K_strata"].get("16", {}).get("untrained_acc") if foc else None,
        "K16_trained": foc["K_strata"].get("16", {}).get("trained_acc") if foc else None,
    }

    # Compact tables for docs
    tables: list[dict[str, Any]] = []
    for T in T_VALUES:
        cell = report["by_T"][str(T)]
        tables.append(
            {
                "T": T,
                "untrained_overall": cell["untrained"]["overall_acc"],
                "trained_overall": (
                    cell["trained"]["overall_acc"] if cell["trained"] else None
                ),
                "untrained_hard_neg": cell["untrained"]["hard_neg_acc"],
                "trained_hard_neg": (
                    cell["trained"]["hard_neg_acc"] if cell["trained"] else None
                ),
                "K8_u": cell["K_strata"].get("8", {}).get("untrained_acc"),
                "K8_t": cell["K_strata"].get("8", {}).get("trained_acc"),
                "K12_u": cell["K_strata"].get("12", {}).get("untrained_acc"),
                "K12_t": cell["K_strata"].get("12", {}).get("trained_acc"),
                "K16_u": cell["K_strata"].get("16", {}).get("untrained_acc"),
                "K16_t": cell["K_strata"].get("16", {}).get("trained_acc"),
                "agreement": cell["prediction_agreement"],
            }
        )
    report["tables"] = tables

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--ood", type=Path, default=DEFAULT_OOD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    report = run_audit(
        ckpt_path=args.ckpt,
        ood_path=args.ood,
        out_path=args.out,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "seals_invalidated": report["seals_invalidated"],
                "out": str(args.out),
                "focus_summary": report.get("focus_summary"),
                "tables": report.get("tables"),
            },
            indent=2,
        )
    )
    return 0 if report.get("verdict") else 1


if __name__ == "__main__":
    raise SystemExit(main())
