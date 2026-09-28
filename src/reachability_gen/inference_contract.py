"""Inference contract layer around frozen #22 ens + #35 reach certificates.

llama.cpp-abstraction frame (MEASURE only; science_open=false):
  frozen checkpoint(s) + fixed T=16 + fixed ens agg ``prob_mean``
  + explicit forward artifacts + post-hoc reachability certificates as
  execution witnesses + refuse rule (dirty YES => FAIL_CLOSED).

Not a ggml/kernel port. Not quant-as-FO-fix. Not sheaf train. Not ens remix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.reach_certificates import (
    ReachCertificate,
    apply_cert_policy_preds,
    certify_prediction,
)
from reachability_gen.run_stalk_hn_fail_open_autopsy import (
    CONF_THRESH as DEFAULT_CONF_THRESH,
    classify_member_agreement,
)
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    _aggregate_preds,
    _ckpt_for_seed,
)

# Locked contract constants (prereg CYCLE_STALK_INFERENCE_CONTRACT)
CONTRACT_T = 16
CONTRACT_AGG = PRIMARY_AGG  # "prob_mean"
CONTRACT_SEEDS: tuple[int, ...] = tuple(range(10))
RECEIPT_FORMAT = "geo-recurrence-run-receipt/v1"
SCIENCE_OPEN_LOCKED = False


@dataclass(frozen=True)
class InferenceContract:
    """Frozen forward interpreter + refuse policy (fail-closed).

    Harness must never stamp science_open=True. Defaults lock #22/#35.
    """

    T: int = CONTRACT_T
    ens_agg: str = CONTRACT_AGG
    ensemble_seeds: tuple[int, ...] = CONTRACT_SEEDS
    force_closed_dirty_yes: bool = True
    force_open_dirty_no: bool = False  # fail-closed: never oracle-open
    conf_thresh: float = DEFAULT_CONF_THRESH
    science_open: bool = SCIENCE_OPEN_LOCKED

    def __post_init__(self) -> None:
        if self.T != CONTRACT_T:
            raise ValueError(
                f"InferenceContract.T locked to {CONTRACT_T}, got {self.T}"
            )
        if self.ens_agg != CONTRACT_AGG:
            raise ValueError(
                f"InferenceContract.ens_agg locked to {CONTRACT_AGG!r}, "
                f"got {self.ens_agg!r} (no ens remix)"
            )
        if self.science_open:
            raise ValueError(
                "InferenceContract.science_open must be False "
                "(harness never self-stamps true)"
            )
        if self.force_open_dirty_no:
            raise ValueError(
                "force_open_dirty_no=True forbidden under fail-closed contract"
            )
        if not self.force_closed_dirty_yes:
            raise ValueError(
                "force_closed_dirty_yes=False forbidden under refuse lock "
                "(dirty YES must FAIL_CLOSED)"
            )


@dataclass
class ContractExampleResult:
    """Per-example contract emission (prediction, cert, refuse, HARD_UNANIMOUS)."""

    index: int
    pred_raw: int
    pred_contract: int
    refuse: bool
    cert_clean: bool
    cert_kind: str
    cert_reason: str
    hard_unanimous: bool
    member_agreement: str
    member_votes: list[int]
    member_confs: list[float]
    witness: Optional[list[int]] = None
    reachable_by_checker: Optional[bool] = None
    label: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContractBatchResult:
    """Batch emission from ``run_with_certificates``."""

    contract: InferenceContract
    examples: list[ContractExampleResult]
    n: int
    n_refuse: int
    n_dirty_yes: int
    n_dirty_no: int
    n_clean: int
    n_hard_unanimous: int
    preds_raw: list[int]
    preds_contract: list[int]
    matched_floors: Optional[dict[str, Any]] = None
    layout_hash: Optional[str] = None
    science_open: bool = SCIENCE_OPEN_LOCKED

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": {
                "T": self.contract.T,
                "ens_agg": self.contract.ens_agg,
                "ensemble_seeds": list(self.contract.ensemble_seeds),
                "force_closed_dirty_yes": self.contract.force_closed_dirty_yes,
                "force_open_dirty_no": self.contract.force_open_dirty_no,
                "conf_thresh": self.contract.conf_thresh,
                "science_open": self.contract.science_open,
            },
            "n": self.n,
            "n_refuse": self.n_refuse,
            "n_dirty_yes": self.n_dirty_yes,
            "n_dirty_no": self.n_dirty_no,
            "n_clean": self.n_clean,
            "n_hard_unanimous": self.n_hard_unanimous,
            "preds_raw": list(self.preds_raw),
            "preds_contract": list(self.preds_contract),
            "matched_floors": self.matched_floors,
            "layout_hash": self.layout_hash,
            "science_open": self.science_open,
            "examples": [e.to_dict() for e in self.examples],
        }


def default_contract() -> InferenceContract:
    """Canonical locked contract (#22/#35)."""
    return InferenceContract()


def ckpt_layout(
    seeds: Sequence[int] = CONTRACT_SEEDS,
) -> list[dict[str, Any]]:
    """List checkpoint paths + size/mtime for layout hash / receipt."""
    out: list[dict[str, Any]] = []
    for seed in seeds:
        path = _ckpt_for_seed(int(seed))
        entry: dict[str, Any] = {
            "seed": int(seed),
            "path": str(path),
            "source": "pr14" if int(seed) <= 4 else "pr18",
            "exists": path.exists(),
        }
        if path.exists():
            st = path.stat()
            entry["size_bytes"] = int(st.st_size)
            entry["mtime_ns"] = int(st.st_mtime_ns)
        out.append(entry)
    return out


def compute_layout_hash(
    *,
    seeds: Sequence[int] = CONTRACT_SEEDS,
    T: int = CONTRACT_T,
    ens_agg: str = CONTRACT_AGG,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """Deterministic layout hash (ckpt identity + T + agg + seeds)."""
    payload = {
        "T": int(T),
        "ens_agg": str(ens_agg),
        "seeds": [int(s) for s in seeds],
        "ckpts": ckpt_layout(seeds),
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _member_votes_and_confs(
    logits_stack: Any,
    example_index: int,
) -> tuple[list[int], list[float]]:
    """logits_stack: (M, N, C) → votes + max-prob confs for one example."""
    import torch.nn.functional as F

    member = logits_stack[:, example_index, :]  # (M, C)
    hard = member.argmax(dim=-1).tolist()
    probs = F.softmax(member, dim=-1)
    confs = probs.max(dim=-1).values.tolist()
    return [int(h) for h in hard], [float(c) for c in confs]


def run_with_certificates(
    rows: Sequence[dict[str, Any]],
    *,
    contract: Optional[InferenceContract] = None,
    logits_stack: Any = None,
    labels: Any = None,
    models: Optional[list[Any]] = None,
    max_nodes: Optional[int] = None,
    matched_rows: Optional[Sequence[dict[str, Any]]] = None,
    matched_logits_stack: Any = None,
    include_examples: bool = True,
) -> ContractBatchResult:
    """One path that always emits (prediction, cert, refuse, HARD_UNANIMOUS, floors).

    Prefer passing precomputed ``logits_stack`` (M,N,C) from frozen ens forward.
    If ``models`` given instead, collects logits at contract.T (eval-only).

    Refuse rule (locked): dirty YES => FAIL_CLOSED (pred_contract=0); dirty NO kept.
    """
    import torch

    from reachability_gen.run_stalk_seed_ensemble import _collect_member_logits
    from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

    c = contract or default_contract()
    # Re-validate locked fields even if constructed via object.__new__ tricks
    if c.T != CONTRACT_T or c.ens_agg != CONTRACT_AGG or c.science_open:
        raise ValueError("contract violated locked InferenceContract invariants")
    if not c.force_closed_dirty_yes or c.force_open_dirty_no:
        raise ValueError("contract violated fail-closed refuse lock")

    rows_list = list(rows)
    if not rows_list:
        raise ValueError("rows must be non-empty")

    if logits_stack is None:
        if models is None:
            raise ValueError("provide logits_stack or models")
        mn = max_nodes or max(
            DEFAULT_MAX_NODE_ID, max(int(r["n"]) for r in rows_list)
        )
        logits_stack, labels_t, _hops = _collect_member_logits(
            models, rows_list, T=c.T, max_nodes=mn
        )
        if labels is None:
            labels = labels_t
    else:
        if labels is None and rows_list and "y" in rows_list[0]:
            labels = torch.tensor(
                [int(r["y"]) for r in rows_list], dtype=torch.long
            )

    preds_raw_t = _aggregate_preds(logits_stack, method=c.ens_agg)
    preds_contract_t, certs, dirty_yes, dirty_no = apply_cert_policy_preds(
        preds_raw_t,
        rows_list,
        force_closed_dirty_yes=c.force_closed_dirty_yes,
        force_open_dirty_no=c.force_open_dirty_no,
    )

    examples: list[ContractExampleResult] = []
    n_hard = 0
    preds_raw: list[int] = []
    preds_contract: list[int] = []
    n_refuse = 0

    for i in range(len(rows_list)):
        raw = int(preds_raw_t[i].item())
        after = int(preds_contract_t[i].item())
        cert = certs[i]
        votes, confs = _member_votes_and_confs(logits_stack, i)
        tag = classify_member_agreement(
            votes, confs, conf_thresh=c.conf_thresh
        )
        hard_u = tag == "HARD_UNANIMOUS"
        if hard_u:
            n_hard += 1
        refuse = (not cert.clean) and raw == 1 and after == 0
        if refuse:
            n_refuse += 1
        preds_raw.append(raw)
        preds_contract.append(after)
        if include_examples:
            lab = None
            if labels is not None:
                lab = int(labels[i].item())
            examples.append(
                ContractExampleResult(
                    index=i,
                    pred_raw=raw,
                    pred_contract=after,
                    refuse=refuse,
                    cert_clean=bool(cert.clean),
                    cert_kind=str(cert.kind),
                    cert_reason=str(cert.reason),
                    hard_unanimous=hard_u,
                    member_agreement=tag,
                    member_votes=votes,
                    member_confs=confs,
                    witness=(
                        list(cert.witness) if cert.witness is not None else None
                    ),
                    reachable_by_checker=cert.reachable_by_checker,
                    label=lab,
                )
            )

    matched_floors = None
    if matched_rows is not None:
        matched_floors = _matched_floors_block(
            matched_rows=list(matched_rows),
            contract=c,
            models=models,
            matched_logits_stack=matched_logits_stack,
            max_nodes=max_nodes,
        )

    layout = compute_layout_hash(
        seeds=c.ensemble_seeds, T=c.T, ens_agg=c.ens_agg
    )
    n = len(rows_list)
    n_dirty_yes = len(dirty_yes)
    n_dirty_no = len(dirty_no)
    n_clean = n - n_dirty_yes - n_dirty_no

    return ContractBatchResult(
        contract=c,
        examples=examples,
        n=n,
        n_refuse=n_refuse,
        n_dirty_yes=n_dirty_yes,
        n_dirty_no=n_dirty_no,
        n_clean=n_clean,
        n_hard_unanimous=n_hard,
        preds_raw=preds_raw,
        preds_contract=preds_contract,
        matched_floors=matched_floors,
        layout_hash=layout,
        science_open=SCIENCE_OPEN_LOCKED,
    )


def _matched_floors_block(
    *,
    matched_rows: list[dict[str, Any]],
    contract: InferenceContract,
    models: Optional[list[Any]],
    matched_logits_stack: Any,
    max_nodes: Optional[int],
) -> dict[str, Any]:
    """Compute matched-OOD baseline vs cert-on Δ (collateral floors)."""
    import torch

    from reachability_gen.run_stalk_hop_ood_hn import _metrics_ext
    from reachability_gen.run_stalk_seed_ensemble import _collect_member_logits
    from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

    if matched_logits_stack is None:
        if models is None:
            return {"skipped": True, "reason": "no_matched_logits_or_models"}
        mn = max_nodes or max(
            DEFAULT_MAX_NODE_ID, max(int(r["n"]) for r in matched_rows)
        )
        matched_logits_stack, labels, hops = _collect_member_logits(
            models, matched_rows, T=contract.T, max_nodes=mn
        )
    else:
        labels = torch.tensor(
            [int(r["y"]) for r in matched_rows], dtype=torch.long
        )
        hops = [int(r["hop_distance"]) for r in matched_rows]

    preds_raw = _aggregate_preds(matched_logits_stack, method=contract.ens_agg)
    preds_cert, _certs, _dy, _dn = apply_cert_policy_preds(
        preds_raw,
        matched_rows,
        force_closed_dirty_yes=contract.force_closed_dirty_yes,
        force_open_dirty_no=contract.force_open_dirty_no,
    )
    base = _metrics_ext(preds_raw, labels, hops)
    cert = _metrics_ext(preds_cert, labels, hops)
    delta = {
        "overall_acc": float(cert["overall_acc"] - base["overall_acc"]),
        "hard_neg_acc": float(cert["hard_neg_acc"] - base["hard_neg_acc"]),
        "K16": float(cert.get("K16", float("nan")) - base.get("K16", float("nan"))),
    }
    return {
        "skipped": False,
        "baseline": {
            "overall_acc": base["overall_acc"],
            "hard_neg_acc": base["hard_neg_acc"],
            "K16": base.get("K16"),
        },
        "cert_on": {
            "overall_acc": cert["overall_acc"],
            "hard_neg_acc": cert["hard_neg_acc"],
            "K16": cert.get("K16"),
        },
        "delta_cert_minus_baseline": delta,
        "floor_abs_max": 0.01,
        "floor_ok": all(
            abs(delta[k]) <= 0.01
            for k in ("overall_acc", "hard_neg_acc", "K16")
            if delta[k] == delta[k]
        ),
    }


def emit_run_receipt(
    batch: ContractBatchResult,
    *,
    dataset: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Machine-readable GGUF-like run receipt (auditability; not new science)."""
    c = batch.contract
    receipt = {
        "format": RECEIPT_FORMAT,
        "science_open": SCIENCE_OPEN_LOCKED,
        "layout_hash": batch.layout_hash or compute_layout_hash(
            seeds=c.ensemble_seeds, T=c.T, ens_agg=c.ens_agg
        ),
        "T": c.T,
        "ens_agg": c.ens_agg,
        "ens_seeds": list(c.ensemble_seeds),
        "ckpt_layout": ckpt_layout(c.ensemble_seeds),
        "dataset": dataset,
        "n": batch.n,
        "cert_status": {
            "n_clean": batch.n_clean,
            "n_dirty_yes": batch.n_dirty_yes,
            "n_dirty_no": batch.n_dirty_no,
            "n_refuse": batch.n_refuse,
            "force_closed_dirty_yes": c.force_closed_dirty_yes,
            "force_open_dirty_no": c.force_open_dirty_no,
        },
        "hard_unanimous": {
            "n": batch.n_hard_unanimous,
            "conf_thresh": c.conf_thresh,
        },
        "matched_floors": batch.matched_floors,
        "note": (
            "Audit receipt for frozen-forward inference contract; "
            "not a science_open claim; certificates are post-hoc witnesses."
        ),
    }
    if extra:
        receipt["extra"] = extra
    return receipt


def certify_and_refuse(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
    pred: int,
    *,
    witness: Optional[Sequence[int]] = None,
    force_closed_dirty_yes: bool = True,
) -> tuple[int, ReachCertificate, bool]:
    """Single-example contract refuse helper (no ens).

    Returns (pred_contract, cert, refuse).
    """
    cert = certify_prediction(n, edges, s, t, pred, witness=witness)
    refuse = False
    out = int(pred)
    if force_closed_dirty_yes and (not cert.clean) and out == 1:
        out = 0
        refuse = True
    return out, cert, refuse


__all__ = [
    "CONTRACT_AGG",
    "CONTRACT_SEEDS",
    "CONTRACT_T",
    "ContractBatchResult",
    "ContractExampleResult",
    "InferenceContract",
    "RECEIPT_FORMAT",
    "SCIENCE_OPEN_LOCKED",
    "certify_and_refuse",
    "ckpt_layout",
    "compute_layout_hash",
    "default_contract",
    "emit_run_receipt",
    "run_with_certificates",
]
