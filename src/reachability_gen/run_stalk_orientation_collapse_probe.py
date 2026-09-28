"""CYCLE_STALK_ORIENTATION_COLLAPSE_PROBE — MEASURE eval-only orientation probe.

Eval-only @ T=16 on data/ood_hops.jsonl with frozen #14/#18/#22 ens.
Dir-GNN-*style* probe via directed incidence × final hidden (no explicit
in/out channels; not architecture). No train. No Phase 4. No tropical
retrain. No §22 widen. science_open=false.

Usage::

    python -m reachability_gen.run_stalk_orientation_collapse_probe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.encode import parse_instance
from reachability_gen.models.fractal_core import build_node_slot_batch
from reachability_gen.orientation import (
    auroc_binary,
    decide_verdict,
    example_orientation_from_states,
    mean_metrics,
    median_outside_iqr,
    orientation_definition_doc,
    summary_numeric,
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_stalk_seed_ensemble import (
    PRIMARY_AGG,
    _aggregate_preds,
    _ckpt_for_seed,
    _collect_member_logits,
    _load_model_from_ckpt,
)
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

CYCLE = "CYCLE_STALK_ORIENTATION_COLLAPSE_PROBE"
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_orientation_collapse_probe.json")
DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE32 = Path("artifacts/stalk_sound_outdeg_gate.json")
DEFAULT_CITE33 = Path("artifacts/stalk_hn_fo_remainder_autopsy.json")
DEFAULT_ENSEMBLE_SEEDS = tuple(range(10))
FOCUS_T = 16

VERDICTS = (
    "ORIENT_SEPARATES_FO",
    "ORIENT_PARTIAL",
    "ORIENT_NULL",
    "INCONCLUSIVE_ARCH",
)

AUROC_SEP = 0.75
AUROC_PARTIAL = 0.60
NAN_CAP = 0.50
CITE_30_FO = 45
CITE_32_REM = 22


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


def _edges_of(ex: dict[str, Any]) -> tuple[int, list[tuple[int, int]], int, int]:
    enc = ex.get("encoding")
    if enc:
        n, edges, s, t = parse_instance(str(enc))
    else:
        n = int(ex["n"])
        edges = [(int(a), int(b)) for a, b in ex["edges"]]
        s, t = int(ex["s"]), int(ex["t"])
    n = int(ex.get("n", n))
    s = int(ex.get("s", s))
    t = int(ex.get("t", t))
    return n, [(int(a), int(b)) for a, b in edges], s, t


def _load_ens(
    ensemble_seeds: tuple[int, ...], max_nodes: int
) -> tuple[list[Any], list[dict[str, Any]]]:
    ens_models: list[Any] = []
    ens_meta: list[dict[str, Any]] = []
    for seed in ensemble_seeds:
        ckpt = _ckpt_for_seed(seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing #22 member ckpt: {ckpt}")
        model, blob = _load_model_from_ckpt(ckpt, max_nodes)
        ens_models.append(model)
        ens_meta.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "source": "pr14" if seed <= 4 else "pr18",
                "best_epoch": blob.get("epoch"),
            }
        )
    return ens_models, ens_meta


def _collect_orientation(
    models: list[Any],
    rows: list[dict[str, Any]],
    *,
    T: int,
    max_nodes: int,
    batch_size: int = 32,
) -> list[dict[str, float]]:
    """Per-example ens-mean orientation metrics (incidence × final hidden)."""
    import torch

    for m in models:
        m.eval()
    out: list[Optional[dict[str, float]]] = [None] * len(rows)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = build_node_slot_batch(batch_rows, max_n=max_nodes)
            # member_metrics[b] = list of per-member dicts
            member_lists: list[list[dict[str, float]]] = [
                [] for _ in range(len(batch_rows))
            ]
            for model in models:
                logits, _, info = model(
                    batch["node_ids"],
                    batch["node_mask"],
                    batch["attn_mask"],
                    batch["s_idx"],
                    batch["t_idx"],
                    return_halt=True,
                    return_states=True,
                    T=T,
                    adaptive_halt=False,
                )
                del logits
                assert info is not None and "final_states" in info
                states = info["final_states"]  # [B, M, d]
                for bi, ex in enumerate(batch_rows):
                    n, edges, s, t = _edges_of(ex)
                    H = states[bi]
                    member_lists[bi].append(
                        example_orientation_from_states(H, n, edges, s, t)
                    )
            for bi in range(len(batch_rows)):
                out[start + bi] = mean_metrics(member_lists[bi])
    assert all(x is not None for x in out)
    return [x for x in out if x is not None]  # type: ignore[misc]


def _stratum_table(
    metrics: list[dict[str, float]],
    ids: list[int],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    block: dict[str, Any] = {"n": len(ids), "ids_head": ids[:8], "numeric": {}}
    for k in keys:
        xs = [float(metrics[i][k]) for i in ids if i < len(metrics)]
        block["numeric"][k] = summary_numeric(xs)
    return block


def run_cycle(
    *,
    hops_data: Path = DEFAULT_HOPS,
    out_path: Path = DEFAULT_OUT,
    cite30_path: Path = DEFAULT_CITE30,
    cite31_path: Path = DEFAULT_CITE31,
    cite32_path: Path = DEFAULT_CITE32,
    cite33_path: Path = DEFAULT_CITE33,
    ensemble_seeds: tuple[int, ...] = DEFAULT_ENSEMBLE_SEEDS,
) -> dict[str, Any]:
    t0 = time.time()
    if not hops_data.exists():
        raise FileNotFoundError(f"ood_hops missing: {hops_data}")
    rows = load_jsonl(hops_data)
    max_nodes = max(DEFAULT_MAX_NODE_ID, max(int(r["n"]) for r in rows))

    cite30 = json.loads(cite30_path.read_text()) if cite30_path.exists() else {}
    cite31 = json.loads(cite31_path.read_text()) if cite31_path.exists() else {}
    cite32 = json.loads(cite32_path.read_text()) if cite32_path.exists() else {}
    cite33 = json.loads(cite33_path.read_text()) if cite33_path.exists() else {}

    fo_ids = list(cite31.get("example_id_list_FO_HN") or cite32.get("fo_analysis", {}).get("fo_ids") or [])
    rem_ids = list(
        cite33.get("cohorts", {}).get("FO_REMAINDER", {}).get("ids")
        or cite32.get("fo_analysis", {}).get("fo_still_wrong_ids")
        or []
    )
    killed_ids = list(
        cite33.get("cohorts", {}).get("FO_KILLED", {}).get("ids")
        or cite32.get("fo_analysis", {}).get("fo_killed_ids")
        or []
    )
    ok_ids = list(cite33.get("cohorts", {}).get("OK_HN", {}).get("ids") or [])
    fc_ids = list(cite33.get("cohorts", {}).get("FC_HN", {}).get("ids") or [])

    print(f"[{CYCLE}] loading ens seeds={list(ensemble_seeds)} …", flush=True)
    ens_models, ens_meta = _load_ens(ensemble_seeds, max_nodes)

    # Hygiene: replicate #22 ens preds / FO count
    print(f"[{CYCLE}] collecting member logits @ T={FOCUS_T} …", flush=True)
    import torch
    import torch.nn.functional as F

    logits, labels, hops = _collect_member_logits(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )
    ens_preds = _aggregate_preds(logits, method=PRIMARY_AGG)
    probs = F.softmax(logits, dim=-1).mean(dim=0)
    ens_max_prob = probs.max(dim=-1).values
    hard = logits.argmax(dim=-1)
    # pairwise D for FO labeling (reuse hop_ood rule: D==0 & conf>=0.8)
    from reachability_gen.run_stalk_epistemic_disagreement import (
        _pairwise_disagreement_per_example,
    )
    from reachability_gen.run_stalk_hop_ood_hn import CONF_THRESH

    pair_per = _pairwise_disagreement_per_example(hard)
    fo_rep: list[int] = []
    wrong = ens_preds != labels
    for i in wrong.nonzero(as_tuple=False).view(-1).tolist():
        if float(pair_per[i].item()) == 0.0 and float(ens_max_prob[i].item()) >= CONF_THRESH:
            fo_rep.append(int(i))

    print(f"[{CYCLE}] collecting orientation proxies …", flush=True)
    metrics = _collect_orientation(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )

    keys = (
        "cos_orient_t",
        "l2_orient_t",
        "mass_ratio_t",
        "mass_in_t",
        "mass_out_t",
        "deg_in_t",
        "deg_out_t",
        "empty_in_t",
        "empty_out_t",
        "cos_orient_s",
        "l2_orient_s",
        "mass_ratio_s",
    )
    strata = {
        "OK_HN": ok_ids,
        "FO_HN": fo_ids,
        "FO_REMAINDER": rem_ids,
        "FO_KILLED": killed_ids,
        "FC_HN": fc_ids,
    }
    tables = {name: _stratum_table(metrics, ids, keys) for name, ids in strata.items()}

    def _scores(ids: list[int], key: str) -> list[float]:
        return [float(metrics[i][key]) for i in ids if i < len(metrics)]

    # AUROC: FO vs OK (label 1 = FO), rem vs OK
    cos_fo = _scores(fo_ids, "cos_orient_t")
    cos_ok = _scores(ok_ids, "cos_orient_t")
    l2_fo = _scores(fo_ids, "l2_orient_t")
    l2_ok = _scores(ok_ids, "l2_orient_t")
    cos_rem = _scores(rem_ids, "cos_orient_t")
    l2_rem = _scores(rem_ids, "l2_orient_t")

    auroc_cos_fo = auroc_binary(cos_fo + cos_ok, [1] * len(cos_fo) + [0] * len(cos_ok))
    # higher collapse → FO: use −l2 so higher score = more FO
    auroc_l2_fo = auroc_binary(
        [-x for x in l2_fo] + [-x for x in l2_ok],
        [1] * len(l2_fo) + [0] * len(l2_ok),
    )
    auroc_cos_rem = auroc_binary(
        cos_rem + cos_ok, [1] * len(cos_rem) + [0] * len(cos_ok)
    )
    auroc_l2_rem = auroc_binary(
        [-x for x in l2_rem] + [-x for x in l2_ok],
        [1] * len(l2_rem) + [0] * len(l2_ok),
    )

    ok_cos = tables["OK_HN"]["numeric"]["cos_orient_t"]
    ok_l2 = tables["OK_HN"]["numeric"]["l2_orient_t"]
    rem_cos = tables["FO_REMAINDER"]["numeric"]["cos_orient_t"]
    rem_l2 = tables["FO_REMAINDER"]["numeric"]["l2_orient_t"]
    rem_out_cos = median_outside_iqr(
        rem_cos["median"], ok_cos["q25"], ok_cos["q75"]
    )
    rem_out_l2 = median_outside_iqr(
        rem_l2["median"], ok_l2["q25"], ok_l2["q75"]
    )
    nan_rate_rem = float(rem_cos["nan_rate"])
    nan_rate_ok = float(ok_cos["nan_rate"])

    decision = decide_verdict(
        auroc_cos_fo=auroc_cos_fo,
        auroc_l2_fo=auroc_l2_fo,
        auroc_cos_rem=auroc_cos_rem,
        auroc_l2_rem=auroc_l2_rem,
        rem_outside_ok_cos=rem_out_cos,
        rem_outside_ok_l2=rem_out_l2,
        nan_rate_rem=nan_rate_rem,
        nan_rate_ok=nan_rate_ok,
        auroc_sep=AUROC_SEP,
        auroc_partial=AUROC_PARTIAL,
        nan_cap=NAN_CAP,
    )

    rem_separates = bool(rem_out_cos or rem_out_l2)

    # Median contrast table
    contrast = {}
    for k in ("cos_orient_t", "l2_orient_t", "mass_ratio_t", "cos_orient_s", "l2_orient_s"):
        contrast[k] = {
            name: tables[name]["numeric"][k]["median"] for name in strata
        }

    auroc_table = {
        "FO_HN_vs_OK_HN": {
            "cos_orient_t": auroc_cos_fo,
            "neg_l2_orient_t": auroc_l2_fo,
        },
        "rem22_vs_OK_HN": {
            "cos_orient_t": auroc_cos_rem,
            "neg_l2_orient_t": auroc_l2_rem,
        },
    }

    cite_ok = (
        len(fo_ids) == CITE_30_FO
        and len(rem_ids) == CITE_32_REM
        and set(fo_rep) == set(fo_ids)
    )

    elapsed = time.time() - t0
    artifact: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": False,
        "section22_unchanged": True,
        "hop_ood_remains_measure_residue": True,
        "phase4_energy_selector": "NOT_STARTED",
        "architecture": orientation_definition_doc(),
        "base_sha": _git_sha(),
        "datasets": {
            "ood_hops": str(hops_data),
            "n": len(rows),
            "T": FOCUS_T,
        },
        "members_pr22": ens_meta,
        "prereg": {
            "verdicts": list(VERDICTS),
            "auroc_sep": AUROC_SEP,
            "auroc_partial": AUROC_PARTIAL,
            "nan_cap": NAN_CAP,
            "primary_slot": "target",
            "proxy": "directed_incidence_x_final_hidden",
            "no_train": True,
            "no_phase4": True,
            "no_tropical_retrain": True,
        },
        "replicate_cite": {
            "fo_ids_n": len(fo_ids),
            "rem22_n": len(rem_ids),
            "ok_hn_n": len(ok_ids),
            "fc_hn_n": len(fc_ids),
            "fo_killed_n": len(killed_ids),
            "fo_replicate_n": len(fo_rep),
            "fo_replicate_match_cite31": set(fo_rep) == set(fo_ids),
            "cite_ok": cite_ok,
            "cite30_path": str(cite30_path),
            "cite31_path": str(cite31_path),
            "cite32_path": str(cite32_path),
            "cite33_path": str(cite33_path),
        },
        "strata_tables": tables,
        "contrast_medians": contrast,
        "auroc": auroc_table,
        "rem22_separates": rem_separates,
        "rem22_outside_ok_iqr": {
            "cos_orient_t": rem_out_cos,
            "l2_orient_t": rem_out_l2,
        },
        "decision": decision,
        "verdict": decision["verdict"],
        "residue": (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/"
            "LOCAL_SOUND_WALL/CERT_FO_CATCH/COLLATERAL_HARM/"
            + decision["verdict"]
        ),
        "elapsed_sec": elapsed,
        "open_status": "MEASURE_ONLY",
        "reading": (
            f"verdict={decision['verdict']}; rem22_separates={rem_separates}; "
            f"auroc_cos_fo={auroc_cos_fo}; auroc_neg_l2_fo={auroc_l2_fo}; "
            "science_open=false; §22 unchanged; Phase4 NOT_STARTED."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"[{CYCLE}] verdict={decision['verdict']} rem22_separates={rem_separates} "
        f"auroc_cos_fo={auroc_cos_fo:.4f} auroc_l2_fo={auroc_l2_fo:.4f} "
        f"elapsed={elapsed:.1f}s → {out_path}",
        flush=True,
    )
    return artifact


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=CYCLE)
    p.add_argument("--hops-data", type=Path, default=DEFAULT_HOPS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cite30", type=Path, default=DEFAULT_CITE30)
    p.add_argument("--cite31", type=Path, default=DEFAULT_CITE31)
    p.add_argument("--cite32", type=Path, default=DEFAULT_CITE32)
    p.add_argument("--cite33", type=Path, default=DEFAULT_CITE33)
    args = p.parse_args(argv)
    log_path = Path(str(args.out).replace(".json", "_run.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    with log_path.open("w") as logf:
        old = sys.stdout
        sys.stdout = _Tee(old, logf)  # type: ignore[assignment]
        try:
            art = run_cycle(
                hops_data=args.hops_data,
                out_path=args.out,
                cite30_path=args.cite30,
                cite31_path=args.cite31,
                cite32_path=args.cite32,
                cite33_path=args.cite33,
            )
        finally:
            sys.stdout = old
    print(f"log → {log_path}")
    print(f"verdict={art['verdict']} science_open={art['science_open']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
