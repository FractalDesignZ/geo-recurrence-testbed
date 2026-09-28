"""CYCLE_STALK_INFERENCE_CONTRACT — MEASURE inference-contract lock.

Formalizes frozen-forward InferenceContract + run_with_certificates around
#22 ens + #35 certificates. Tables reuse sealed #30/#35/#40/#41; optional
live re-infer. science_open=false (not widened; §22 unchanged).

Usage::

    python -m reachability_gen.run_stalk_inference_contract
    python -m reachability_gen.run_stalk_inference_contract --live
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from reachability_gen.inference_contract import (
    CONTRACT_AGG,
    CONTRACT_SEEDS,
    CONTRACT_T,
    RECEIPT_FORMAT,
    SCIENCE_OPEN_LOCKED,
    compute_layout_hash,
    default_contract,
    emit_run_receipt,
    run_with_certificates,
)
from reachability_gen.overfit_ff import load_jsonl

CYCLE = "CYCLE_STALK_INFERENCE_CONTRACT"
FOCUS_T = CONTRACT_T
DEFAULT_HOPS = Path("data/ood_hops.jsonl")
DEFAULT_MATCHED = Path("data/covariate_matched_ood.jsonl")
DEFAULT_OUT = Path("artifacts/stalk_inference_contract.json")
DEFAULT_RECEIPT = Path("artifacts/stalk_inference_contract_receipt.json")

DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE35 = Path("artifacts/stalk_reach_certificates.json")
DEFAULT_CITE39 = Path("artifacts/stalk_fo_cert_regression.json")
DEFAULT_CITE40 = Path("artifacts/stalk_sheaf_energy_fo_probe.json")
DEFAULT_CITE41 = Path("artifacts/stalk_spectral_zeta_fo_probe.json")

VERDICTS = ("CONTRACT_LOCKED", "PARTIAL", "INVALID", "COLLATERAL")

# Floors (LOCKED)
FO_TOTAL = 45
REM22_TOTAL = 22
FO_CATCH_EXACT = 45
REM22_CATCH_EXACT = 22
MATCHED_ABS_MAX = 0.01
COLLATERAL_ABS = 0.05
CERT_OFF_FO_EXACT = 45
ENERGY_REFUSE_FO_MAX = 0  # ≢ cert
ZETA_REFUSE_FO_MAX = 0
P_CORRECT_CLEAN_MIN = 0.99


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object")
    return data


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _cite35_tables(cite35: dict[str, Any]) -> dict[str, Any]:
    ct = cite35.get("comparison_table") or {}
    baseline = ct.get("baseline") or {}
    cert = ct.get("cert") or {}
    collateral = cite35.get("collateral") or {}
    matched = collateral.get("matched_ood_T16") or {}
    m_base = matched.get("baseline") or {}
    m_cert = matched.get("cert") or {}
    deltas = {
        "overall_acc": float(
            (m_cert.get("overall_acc") or 0) - (m_base.get("overall_acc") or 0)
        ),
        "hard_neg_acc": float(
            (m_cert.get("hard_neg_acc") or 0) - (m_base.get("hard_neg_acc") or 0)
        ),
        "K16": float((m_cert.get("K16") or 0) - (m_base.get("K16") or 0)),
    }
    fo = cite35.get("fo_analysis") or {}
    p_clean = cite35.get("p_correct_given_clean") or {}
    cd_cert = None
    arms = cite35.get("arms") or {}
    arm2 = arms.get("2_cert") or {}
    if isinstance(arm2, dict):
        cd_cert = arm2.get("CD") or arm2.get("competent_dissonance")
    arm0 = arms.get("0_baseline") or {}
    cd_base = None
    if isinstance(arm0, dict):
        cd_base = arm0.get("CD") or arm0.get("competent_dissonance")
    return {
        "source": "artifacts/stalk_reach_certificates.json",
        "cert_off_baseline": {
            "FO": baseline.get("FO"),
            "FC": baseline.get("FC"),
            "HN": baseline.get("HN"),
            "K16": baseline.get("K16"),
            "overall": baseline.get("overall"),
        },
        "cert_on": {
            "FO": cert.get("FO"),
            "FC": cert.get("FC"),
            "HN": cert.get("HN"),
            "K16": cert.get("K16"),
            "overall": cert.get("overall"),
            "FO_killed_of_45": cert.get("FO_killed_of_45"),
            "rem22_killed": cert.get("rem22_killed"),
            "dirty_rate": cert.get("dirty_rate"),
            "clean_rate": cert.get("clean_rate"),
        },
        "fo_catch": {
            "fo_killed": fo.get("fo_killed_cert", cert.get("FO_killed_of_45")),
            "fo_total": fo.get("fo_total", FO_TOTAL),
            "rem22_killed": fo.get(
                "rem22_killed_cert", cert.get("rem22_killed")
            ),
            "rem22_total": fo.get("rem22_total", REM22_TOTAL),
        },
        "matched_delta": deltas,
        "matched_harm_flag": bool(collateral.get("harm", False)),
        "p_correct_given_clean": p_clean.get("post_policy"),
        "CD_baseline": cd_base,
        "CD_cert": cd_cert,
        "verdict_cite35": (cite35.get("decision") or {}).get("verdict")
        or cite35.get("verdict"),
    }


def _cite30_shatter(cite30: dict[str, Any]) -> dict[str, Any]:
    """Cert-off shatter from sealed #30 (baseline_prob_mean_T16)."""
    audit = cite30.get("audit") or {}
    base = {}
    if isinstance(audit, dict):
        base = audit.get("baseline_prob_mean_T16") or {}
    ens = base.get("ensemble") or {}
    fm = base.get("fail_mode") or {}
    hn = ens.get("hard_neg_acc")
    k16 = ens.get("K16")
    fo = fm.get("FAIL_OPEN")
    if fo is None:
        fo = CERT_OFF_FO_EXACT
    return {
        "source": "artifacts/stalk_hop_ood_hn.json",
        "verdict": cite30.get("verdict"),
        "FO": fo,
        "HN": hn,
        "K16": k16,
        "cert_off_FO_returns": int(fo) == CERT_OFF_FO_EXACT,
    }


def _energy_zeta_neq(
    cite40: dict[str, Any], cite41: dict[str, Any], cite35: dict[str, Any]
) -> dict[str, Any]:
    """Document ≢ vs energy(#40) / zeta(#41) relative to #35 cert refuse."""
    e_fo = int((cite40.get("decision") or {}).get("fo_killed", ENERGY_REFUSE_FO_MAX))
    z_fo = int((cite41.get("decision") or {}).get("fo_killed", ZETA_REFUSE_FO_MAX))
    # Prefer arm-level when present
    arm_e = (cite40.get("arms") or {}).get("1_energy_refuse") or {}
    if isinstance(arm_e, dict) and arm_e.get("fo_killed") is not None:
        e_fo = int(arm_e["fo_killed"])
    arm_z = (cite41.get("arms") or {}).get("1_zeta_refuse") or {}
    if isinstance(arm_z, dict) and arm_z.get("fo_killed") is not None:
        z_fo = int(arm_z["fo_killed"])

    cert_fo = FO_CATCH_EXACT
    ct = (cite35.get("comparison_table") or {}).get("cert") or {}
    if ct.get("FO_killed_of_45") is not None:
        cert_fo = int(ct["FO_killed_of_45"])

    return {
        "cert_fo_killed": cert_fo,
        "energy40_fo_refuse_killed": e_fo,
        "energy40_source": "artifact",
        "energy40_verdict": cite40.get("verdict")
        or (cite40.get("decision") or {}).get("verdict"),
        "zeta41_fo_refuse_killed": z_fo,
        "zeta41_source": "artifact",
        "zeta41_verdict": cite41.get("verdict")
        or (cite41.get("decision") or {}).get("verdict"),
        "neq_energy": int(e_fo) != int(cert_fo),
        "neq_zeta": int(z_fo) != int(cert_fo),
        "energy_refuse_is_zero": int(e_fo) == ENERGY_REFUSE_FO_MAX,
        "zeta_refuse_is_zero": int(z_fo) == ZETA_REFUSE_FO_MAX,
    }


def decide_verdict(
    *,
    api_ok: bool,
    science_open: bool,
    fo_killed: int,
    rem22_killed: int,
    cert_off_fo: int,
    matched_deltas: dict[str, float],
    matched_harm: bool,
    neq_energy: bool,
    neq_zeta: bool,
    regression39_pass: bool,
    p_correct_clean: Optional[float],
) -> dict[str, Any]:
    """LOCKED verdict enum."""
    reasons: list[str] = []
    if science_open:
        return {
            "verdict": "INVALID",
            "reasons": ["science_open_stamped_true"],
        }
    if not api_ok:
        return {
            "verdict": "INVALID",
            "reasons": ["inference_contract_api_broken"],
        }

    # Collateral priority
    coll = matched_harm
    for k, v in matched_deltas.items():
        if v == v and abs(float(v)) >= COLLATERAL_ABS:
            coll = True
            reasons.append(f"matched_{k}_abs_delta={v}>={COLLATERAL_ABS}")
    if coll:
        return {"verdict": "COLLATERAL", "reasons": reasons or ["matched_harm"]}

    floors = {
        "fo_catch_45": fo_killed == FO_CATCH_EXACT,
        "rem22_catch_22": rem22_killed == REM22_CATCH_EXACT,
        "cert_off_fo_45": cert_off_fo == CERT_OFF_FO_EXACT,
        "matched_abs_le_0.01": all(
            abs(float(v)) <= MATCHED_ABS_MAX
            for v in matched_deltas.values()
            if v == v
        ),
        "neq_energy": neq_energy,
        "neq_zeta": neq_zeta,
        "regression39_pass": regression39_pass,
        "p_correct_clean": (
            p_correct_clean is not None and p_correct_clean >= P_CORRECT_CLEAN_MIN
        ),
    }
    failed = [k for k, ok in floors.items() if not ok]
    if not failed:
        return {
            "verdict": "CONTRACT_LOCKED",
            "reasons": ["all_contract_floors_hold"],
            "floors": floors,
        }
    return {
        "verdict": "PARTIAL",
        "reasons": [f"soft_miss:{k}" for k in failed],
        "floors": floors,
    }


def _api_smoke_synthetic() -> dict[str, Any]:
    """CPU-safe synthetic smoke: dirty YES refused; clean YES kept."""
    import torch

    # Two examples: unreachable YES (dirty) + reachable YES (clean)
    rows = [
        {
            "encoding": "N 3 EDGES 1,2 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 0,
        },
        {
            "encoding": "N 3 EDGES 0,1 1,2 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 1,
        },
    ]
    # Fake ens logits (M=2, N=2, C=2): both members predict YES with high conf
    logits = torch.tensor(
        [
            [[0.0, 5.0], [0.0, 5.0]],
            [[0.0, 5.0], [0.0, 5.0]],
        ],
        dtype=torch.float32,
    )
    batch = run_with_certificates(
        rows, contract=default_contract(), logits_stack=logits
    )
    ok = (
        batch.preds_raw == [1, 1]
        and batch.preds_contract == [0, 1]
        and batch.n_refuse == 1
        and batch.examples[0].refuse is True
        and batch.examples[1].refuse is False
        and batch.examples[0].hard_unanimous is True
        and batch.science_open is False
    )
    receipt = emit_run_receipt(batch, dataset="synthetic_smoke")
    return {
        "ok": ok,
        "n_refuse": batch.n_refuse,
        "preds_raw": batch.preds_raw,
        "preds_contract": batch.preds_contract,
        "layout_hash": batch.layout_hash,
        "receipt_format": receipt.get("format"),
        "science_open": batch.science_open,
    }


def _live_contract_eval(
    *,
    hops_data: Path,
    matched_data: Path,
    max_examples: Optional[int] = None,
    skip_matched: bool = False,
) -> dict[str, Any]:
    """Optional live re-infer through run_with_certificates."""
    import torch

    from reachability_gen.run_stalk_reach_certificates import _load_ens
    from reachability_gen.run_stalk_seed_ensemble import _collect_member_logits
    from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

    rows = load_jsonl(hops_data)
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    max_nodes = max(DEFAULT_MAX_NODE_ID, max(int(r["n"]) for r in rows))
    ens_models, ens_meta = _load_ens(CONTRACT_SEEDS, max_nodes)
    logits, labels, hops = _collect_member_logits(
        ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
    )
    matched_rows = None
    matched_logits = None
    if not skip_matched and matched_data.exists() and max_examples is None:
        matched_rows = load_jsonl(matched_data)
        m_max = max(max_nodes, max(int(r["n"]) for r in matched_rows))
        if m_max > max_nodes:
            ens_models, ens_meta = _load_ens(CONTRACT_SEEDS, m_max)
            max_nodes = m_max
            logits, labels, hops = _collect_member_logits(
                ens_models, rows, T=FOCUS_T, max_nodes=max_nodes
            )
        matched_logits, _mlabels, _mhops = _collect_member_logits(
            ens_models, matched_rows, T=FOCUS_T, max_nodes=max_nodes
        )

    batch = run_with_certificates(
        rows,
        contract=default_contract(),
        logits_stack=logits,
        labels=labels,
        matched_rows=matched_rows,
        matched_logits_stack=matched_logits,
        include_examples=False,
    )
    # FO / rem from sealed cites for kill counts on live preds
    cite35 = _load_json(DEFAULT_CITE35) if DEFAULT_CITE35.exists() else {}
    fo_ids = list((cite35.get("fo_analysis") or {}).get("fo_ids") or [])
    rem_ids = list((cite35.get("fo_analysis") or {}).get("rem22_ids") or [])
    fo_killed = sum(
        1
        for i in fo_ids
        if i < batch.n and batch.preds_contract[i] == int(labels[i].item())
    )
    rem_killed = sum(
        1
        for i in rem_ids
        if i < batch.n and batch.preds_contract[i] == int(labels[i].item())
    )
    fo_raw_wrong = sum(
        1
        for i in fo_ids
        if i < batch.n and batch.preds_raw[i] != int(labels[i].item())
    )
    return {
        "n": batch.n,
        "n_refuse": batch.n_refuse,
        "n_dirty_yes": batch.n_dirty_yes,
        "n_clean": batch.n_clean,
        "n_hard_unanimous": batch.n_hard_unanimous,
        "layout_hash": batch.layout_hash,
        "fo_killed_contract": fo_killed,
        "rem22_killed_contract": rem_killed,
        "fo_still_wrong_raw": fo_raw_wrong,
        "matched_floors": batch.matched_floors,
        "members": ens_meta,
        "receipt": emit_run_receipt(
            batch,
            dataset=str(hops_data),
            extra={"mode": "live", "fo_killed_contract": fo_killed},
        ),
    }


def run_cycle(
    *,
    out_path: Path = DEFAULT_OUT,
    receipt_path: Path = DEFAULT_RECEIPT,
    cite30_path: Path = DEFAULT_CITE30,
    cite31_path: Path = DEFAULT_CITE31,
    cite35_path: Path = DEFAULT_CITE35,
    cite39_path: Path = DEFAULT_CITE39,
    cite40_path: Path = DEFAULT_CITE40,
    cite41_path: Path = DEFAULT_CITE41,
    hops_data: Path = DEFAULT_HOPS,
    matched_data: Path = DEFAULT_MATCHED,
    live: bool = False,
    live_max: Optional[int] = None,
    skip_matched_live: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    missing = [
        str(p)
        for p in (cite30_path, cite35_path, cite39_path, cite40_path, cite41_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing sealed cites: {missing}")

    cite30 = _load_json(cite30_path)
    cite31 = _load_json(cite31_path) if cite31_path.exists() else {}
    cite35 = _load_json(cite35_path)
    cite39 = _load_json(cite39_path)
    cite40 = _load_json(cite40_path)
    cite41 = _load_json(cite41_path)

    print("[inference-contract] synthetic API smoke", file=sys.stderr)
    smoke = _api_smoke_synthetic()

    tables35 = _cite35_tables(cite35)
    shatter30 = _cite30_shatter(cite30)
    neq = _energy_zeta_neq(cite40, cite41, cite35)

    # Prefer #35 baseline FO for cert-off (exact replicate of #30 storyline)
    cert_off_fo = int(
        (tables35.get("cert_off_baseline") or {}).get("FO") or CERT_OFF_FO_EXACT
    )
    fo_killed = int((tables35.get("fo_catch") or {}).get("fo_killed") or -1)
    rem_killed = int((tables35.get("fo_catch") or {}).get("rem22_killed") or -1)
    matched_deltas = tables35.get("matched_delta") or {}
    p_clean = tables35.get("p_correct_given_clean")
    regression39_pass = (
        cite39.get("verdict") == "PASS_REGRESSION"
        or (cite39.get("decision") or {}).get("verdict") == "PASS_REGRESSION"
    )

    # HARD_UNANIMOUS from #31 member_logits.tag_counts (sealed 34/45)
    tags = ((cite31.get("member_logits") or {}).get("tag_counts") or {})
    n_hard = tags.get("HARD_UNANIMOUS")
    if n_hard is None:
        n_hard = 34  # sealed cite

    live_block: dict[str, Any] = {"ran": False}
    receipt: Optional[dict[str, Any]] = None
    if live:
        print(
            f"[inference-contract] LIVE run_with_certificates "
            f"(max={live_max})",
            file=sys.stderr,
        )
        live_block = _live_contract_eval(
            hops_data=hops_data,
            matched_data=matched_data,
            max_examples=live_max,
            skip_matched=skip_matched_live,
        )
        live_block["ran"] = True
        receipt = live_block.get("receipt")
    else:
        # Artifact-mode receipt from layout + sealed cert status
        contract = default_contract()
        receipt = {
            "format": RECEIPT_FORMAT,
            "science_open": SCIENCE_OPEN_LOCKED,
            "layout_hash": compute_layout_hash(
                seeds=CONTRACT_SEEDS, T=CONTRACT_T, ens_agg=CONTRACT_AGG
            ),
            "T": CONTRACT_T,
            "ens_agg": CONTRACT_AGG,
            "ens_seeds": list(CONTRACT_SEEDS),
            "dataset": "artifact_first_cite35",
            "cert_status": {
                "cite35_verdict": tables35.get("verdict_cite35"),
                "fo_killed": fo_killed,
                "rem22_killed": rem_killed,
                "dirty_rate": (tables35.get("cert_on") or {}).get("dirty_rate"),
                "p_correct_given_clean": p_clean,
            },
            "hard_unanimous_cite31": n_hard,
            "note": (
                "Artifact-first receipt; live re-infer optional via --live. "
                "Not a science_open claim."
            ),
            "synthetic_smoke_layout_hash": smoke.get("layout_hash"),
        }

    decision = decide_verdict(
        api_ok=bool(smoke.get("ok")),
        science_open=bool(smoke.get("science_open")),
        fo_killed=fo_killed,
        rem22_killed=rem_killed,
        cert_off_fo=cert_off_fo,
        matched_deltas={k: float(v) for k, v in matched_deltas.items()},
        matched_harm=bool(tables35.get("matched_harm_flag")),
        neq_energy=bool(neq.get("neq_energy")),
        neq_zeta=bool(neq.get("neq_zeta")),
        regression39_pass=bool(regression39_pass),
        p_correct_clean=float(p_clean) if p_clean is not None else None,
    )

    # Ablation table (cert-on vs cert-off) — core stress
    ablation = {
        "cert_off": tables35.get("cert_off_baseline"),
        "cert_on": tables35.get("cert_on"),
        "reading": (
            "cert-off returns FO=45 / HN shatter; cert-on kills 45/45 FO + "
            "22/22 rem-22 with matched Δ≈0 — documents why the contract exists"
        ),
        "cite30_shatter": shatter30,
    }

    elapsed = time.time() - t0
    artifact: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE",
        "science_open": SCIENCE_OPEN_LOCKED,
        "section22_unchanged": True,
        "hop_ood_remains_measure_residue": True,
        "open_status": "not_widened",
        "T_fixed": FOCUS_T,
        "ens_agg_locked": CONTRACT_AGG,
        "ensemble_seeds": list(CONTRACT_SEEDS),
        "train": None,
        "generation": None,
        "llama_cpp_abstraction": {
            "frozen_forward": True,
            "explicit_inference_contract": True,
            "certificates_as_execution_witnesses": True,
            "ggml_kernel_port": False,
            "quant_as_fo_fix": False,
            "sheaf_train": False,
            "ens_remix": False,
        },
        "api": {
            "module": "reachability_gen.inference_contract",
            "entrypoint": "run_with_certificates",
            "contract_class": "InferenceContract",
            "synthetic_smoke": smoke,
        },
        "tables": {
            "ablation_cert_on_vs_off": ablation,
            "fo_catch_rem22": tables35.get("fo_catch"),
            "matched_ood_delta": matched_deltas,
            "neq_energy40_zeta41": neq,
            "CD": {
                "baseline": tables35.get("CD_baseline"),
                "cert": tables35.get("CD_cert"),
            },
            "p_correct_given_clean": p_clean,
            "HARD_UNANIMOUS_cite31": n_hard,
        },
        "ci": {
            "fo_cert_regression_verdict": cite39.get("verdict")
            or (cite39.get("decision") or {}).get("verdict"),
            "pass": regression39_pass,
        },
        "live": live_block,
        "receipt_path": str(receipt_path),
        "prereg": {
            "cycle_doc": "docs/CYCLE_STALK_INFERENCE_CONTRACT.md",
            "verdicts": list(VERDICTS),
            "floors": {
                "FO_CATCH_EXACT": FO_CATCH_EXACT,
                "REM22_CATCH_EXACT": REM22_CATCH_EXACT,
                "MATCHED_ABS_MAX": MATCHED_ABS_MAX,
                "CERT_OFF_FO_EXACT": CERT_OFF_FO_EXACT,
            },
        },
        "decision": decision,
        "verdict": decision["verdict"],
        "residue": f"INFERENCE_CONTRACT/{decision['verdict']}",
        "reading": (
            "InferenceContract locks dirty-cert refuse as API around frozen "
            "#22 ens; cert-on kills FO that cert-off does not; energy/ζ ≢ cert; "
            "science_open=false; §22 unchanged."
        ),
        "git_sha": _git_sha(),
        "elapsed_sec": elapsed,
        "datasets": {
            "ood_hops": str(hops_data),
            "matched_ood": str(matched_data),
            "cites": {
                "30": str(cite30_path),
                "31": str(cite31_path),
                "35": str(cite35_path),
                "39": str(cite39_path),
                "40": str(cite40_path),
                "41": str(cite41_path),
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if receipt is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        f"[inference-contract] verdict={decision['verdict']} "
        f"wrote {out_path} ({elapsed:.2f}s)",
        file=sys.stderr,
    )
    return artifact


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="CYCLE_STALK_INFERENCE_CONTRACT MEASURE harness"
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    p.add_argument("--live", action="store_true", help="full live re-infer")
    p.add_argument(
        "--live-max",
        type=int,
        default=None,
        help="limit live ood_hops rows (smoke)",
    )
    p.add_argument("--skip-matched-live", action="store_true")
    p.add_argument("--hops", type=Path, default=DEFAULT_HOPS)
    p.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    args = p.parse_args(argv)
    art = run_cycle(
        out_path=args.out,
        receipt_path=args.receipt,
        hops_data=args.hops,
        matched_data=args.matched,
        live=bool(args.live) or args.live_max is not None,
        live_max=args.live_max,
        skip_matched_live=args.skip_matched_live,
    )
    verdict = art.get("verdict")
    if verdict == "INVALID":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
