"""Standing untrained-control harness for SheafInferCore (fail-closed).

Compares a freshly constructed (untrained) SheafInferCore against the sealed
Gate1 checkpoint on the same eval sets. If untrained ≈ sealed on hard-neg /
K strata / overall with ~100% prediction agreement, seals claiming *learned*
restriction maps are INVALID (init+architecture bake-in).

Usage::

    python -m reachability_gen.run_sheaf_untrained_control
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import HOP_UNREACHABLE
from reachability_gen.models.sheaf_infer_core import SheafInferCore, build_sheaf_batch
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

DEFAULT_CKPT = Path("artifacts/sheaf_infer_gate1_best.pt")
DEFAULT_OUT = Path("artifacts/sheaf_untrained_control_audit.json")

DATASETS: dict[str, dict[str, Any]] = {
    "matched_ood": {
        "path": Path("data/covariate_matched_ood.jsonl"),
        "T_values": (6, 8, 12, 16),
        "focus_T": 16,
    },
    "red_test_k20": {
        "path": Path("data/sheaf_red_test_k20.jsonl"),
        "T_values": (20, 24),
        "focus_T": 24,
    },
    "dense_context_cell1": {
        "path": Path("data/sheaf_dense_context_cell1.jsonl"),
        "T_values": (8, 12),
        "focus_T": 12,
    },
    "dense_context_cell2": {
        "path": Path("data/sheaf_dense_context_cell2.jsonl"),
        "T_values": (8, 12),
        "focus_T": 12,
    },
}


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


def document_init_bake_in(model: SheafInferCore) -> dict[str, Any]:
    """Record exact init constants that bake directed reachability."""
    last = model.edge_encoder[-1]
    return {
        "edge_encoder_last_weight_zeros": bool(
            float(last.weight.detach().abs().sum()) == 0.0
        ),
        "edge_encoder_last_bias": float(last.bias.detach().item()),
        "absent_bias": float(model.absent_bias.detach().item()),
        "gate_theta": float(model.gate_theta),
        "self_logit_hardcoded": 8.0,
        "residual_alpha": float(model.residual_alpha),
        "W_msg_eye": True,
        "W_out_eye": True,
        "mlp_zero_at_init": True,
        "stalk_proj_eye": True,
        "head_energy_col": model.head.weight.detach()[:, -1].tolist(),
        "head_bias": model.head.bias.detach().tolist(),
        "mechanism": (
            "Listed edges: last Linear W=0, bias=+4 → logit=4 → σ≈0.982 > θ=0.5 → "
            "gate ON. Absent cells: absent_bias=-4 → gate OFF. Diagonal forced "
            "logit=+8 ON. residual_alpha=1 + W_msg=I + MLP=0 ⇒ pure diffusion on "
            "gated A_hat. Readout [z_t; ‖z_t‖] with energy col (−1,+1) and bias "
            "(+5,−5): ‖z_t‖=0 → class 0; ‖z_t‖>0 → class 1. Untrained model is a "
            "discrete reachability oracle on listed edge tokens."
        ),
    }


def _build_model(
    hp: dict[str, Any],
    *,
    load_state: Optional[dict[str, Any]] = None,
    seed: int = 0,
) -> SheafInferCore:
    import torch

    torch.manual_seed(seed)
    model = SheafInferCore(
        d=int(hp["d"]),
        T=int(hp["T"]),
        mlp_expansion=int(hp["mlp_expansion"]),
        max_nodes=int(hp.get("max_nodes", DEFAULT_MAX_NODE_ID)),
        residual_alpha=float(hp.get("residual_alpha", 1.0)),
        gate_theta=float(hp.get("gate_theta", 0.5)),
        gate_mode=str(hp.get("gate_mode", "ste")),
        gate_detach_diffusion=bool(hp.get("gate_detach_diffusion", True)),
    )
    if load_state is not None:
        model.load_state_dict(load_state)
    model.eval()
    return model


def _eval_rows(
    model: SheafInferCore,
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int = DEFAULT_MAX_NODE_ID,
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


def run_audit(
    *,
    ckpt_path: Path = DEFAULT_CKPT,
    out_path: Path = DEFAULT_OUT,
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing sealed ckpt: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hparams"])
    untrained = _build_model(hp, seed=seed)
    trained = _build_model(hp, load_state=ckpt["state_dict"], seed=seed)
    init_doc = document_init_bake_in(untrained)

    report: dict[str, Any] = {
        "audit": "FAIL_CLOSED_UNTRAINED_CONTROL",
        "cycle": "CYCLE_SHEAF_INFERENCE",
        "base_sha": _git_sha(),
        "ckpt": {
            "path": str(ckpt_path),
            "epoch": ckpt.get("epoch"),
            "val_acc": ckpt.get("val_acc"),
            "hparams": hp,
            "sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        },
        "init_bake_in": init_doc,
        "datasets": {},
        "science_open": False,
    }

    for name, cfg in DATASETS.items():
        path: Path = cfg["path"]
        if not path.exists():
            report["datasets"][name] = {"missing": True, "path": str(path)}
            continue
        rows = load_jsonl(path)
        ds: dict[str, Any] = {"path": str(path), "n": len(rows), "by_T": {}}
        for T in cfg["T_values"]:
            u = _eval_rows(untrained, rows, T=int(T))
            t = _eval_rows(trained, rows, T=int(T))
            agree = float((u["preds"] == t["preds"]).float().mean())
            u_clean = {k: v for k, v in u.items() if k not in ("preds", "labels")}
            t_clean = {k: v for k, v in t.items() if k not in ("preds", "labels")}
            cell: dict[str, Any] = {
                "untrained": u_clean,
                "trained": t_clean,
                "prediction_agreement": agree,
                "delta_overall": t["overall_acc"] - u["overall_acc"],
                "delta_hard_neg": t["hard_neg_acc"] - u["hard_neg_acc"],
            }
            for hop_key in sorted(u["by_hop"], key=lambda x: int(x)):
                if int(hop_key) > 0:
                    cell.setdefault("K_strata", {})[hop_key] = {
                        "untrained_acc": u["by_hop"][hop_key]["acc_mean"],
                        "trained_acc": t["by_hop"][hop_key]["acc_mean"],
                        "n": u["by_hop"][hop_key]["n"],
                    }
            ds["by_T"][str(T)] = cell
        focus = str(cfg["focus_T"])
        foc = ds["by_T"][focus]
        ds["focus_T"] = cfg["focus_T"]
        ds["focus_summary"] = {
            "untrained_overall": foc["untrained"]["overall_acc"],
            "trained_overall": foc["trained"]["overall_acc"],
            "untrained_hard_neg": foc["untrained"]["hard_neg_acc"],
            "trained_hard_neg": foc["trained"]["hard_neg_acc"],
            "agreement": foc["prediction_agreement"],
            "untrained_fpr": foc["untrained"]["fpr"],
            "untrained_fnr": foc["untrained"]["fnr"],
            "trained_fpr": foc["trained"]["fpr"],
            "trained_fnr": foc["trained"]["fnr"],
        }
        report["datasets"][name] = ds

    m16 = report["datasets"].get("matched_ood", {}).get("by_T", {}).get("16")
    if m16 is None:
        verdict = "MISSING_MATCHED_OOD"
        claim = "matched_ood T=16 unavailable; cannot close verdict."
        seals_bad = False
        k16_u = float("nan")
    else:
        k16_u = m16.get("K_strata", {}).get("16", {}).get("untrained_acc", float("nan"))
        u_ok = (
            abs(m16["untrained"]["overall_acc"] - 1.0) < 1e-9
            and abs(m16["untrained"]["hard_neg_acc"] - 1.0) < 1e-9
            and abs(m16["prediction_agreement"] - 1.0) < 1e-9
        )
        k16_ok = abs(k16_u - 1.0) < 1e-9 if k16_u == k16_u else False
        if u_ok and k16_ok:
            verdict = "SEALS_COMPROMISED_INIT_BAKE_IN"
            claim = (
                "Untrained SheafInferCore reproduces sealed Gate1 matched-OOD T16 "
                "overall/hard-neg/K16 = 1.000 with 100% prediction agreement vs sealed "
                "ckpt. science_open 'learned' claim is INVALID; performance is "
                "init+architecture (edges gated on; energy readout = reachability)."
            )
            seals_bad = True
        elif m16["untrained"]["overall_acc"] < 0.6 and m16["trained"]["overall_acc"] > 0.9:
            verdict = "OPUS_CLAIM_NOT_REPRODUCED"
            claim = (
                "Untrained near chance; trained >> untrained. Opus claim not "
                "reproduced here; keep standing untrained_control."
            )
            seals_bad = False
        else:
            verdict = "PARTIAL_OR_AMBIGUOUS"
            claim = "Mixed results; prefer fail-closed narrowing of science_open."
            seals_bad = False

    report["verdict"] = verdict
    report["verdict_claim"] = claim
    report["matched_ood_T16_untrained_K16"] = k16_u
    report["seals_invalidated"] = bool(seals_bad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    report = run_audit(ckpt_path=args.ckpt, out_path=args.out, seed=args.seed)
    print(json.dumps({
        "verdict": report["verdict"],
        "seals_invalidated": report["seals_invalidated"],
        "out": str(args.out),
        "matched_ood_T16": report["datasets"].get("matched_ood", {}).get("focus_summary"),
    }, indent=2))
    return 0 if report.get("verdict") else 1


if __name__ == "__main__":
    raise SystemExit(main())
