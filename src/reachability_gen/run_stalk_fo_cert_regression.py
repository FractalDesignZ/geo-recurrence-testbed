"""CYCLE_STALK_FO_CERT_REGRESSION — MEASURE/CI hygiene regression gate.

Locks sealed storyline invariants from #30/#31/#35/#38 via artifact reads
(CPU-safe; no train; no generation). science_open=false (not widened).

Usage::

    python -m reachability_gen.run_stalk_fo_cert_regression
    reachability-stalk-fo-cert-regression
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

CYCLE = "CYCLE_STALK_FO_CERT_REGRESSION"
FOCUS_T = 16

DEFAULT_CITE30 = Path("artifacts/stalk_hop_ood_hn.json")
DEFAULT_CITE31 = Path("artifacts/stalk_hn_fail_open_autopsy.json")
DEFAULT_CITE35 = Path("artifacts/stalk_reach_certificates.json")
DEFAULT_CITE38 = Path("artifacts/stalk_energy_selector.json")
DEFAULT_OUT = Path("artifacts/stalk_fo_cert_regression.json")

VERDICTS = ("PASS_REGRESSION", "FAIL")

# --- Gate A: HARD_UNANIMOUS / oracle wall (#31 / #38) ---
FO_TOTAL_LOCKED = 45
HARD_UNANIMOUS_MIN_COUNT = 30
HARD_UNANIMOUS_MIN_RATE = 0.70
MIXED_PRED_MAX = 0
ORACLE_FO_KILLED_MAX = 0
ORACLE_REM22_KILLED_MAX = 0

# --- Gate B: certificate refuse (#35) ---
FO_CATCH_EXACT = 45
REM22_CATCH_EXACT = 22
P_CORRECT_CLEAN_MIN = 0.99
MATCHED_COLLATERAL_ABS_MAX = 0.01
CERT_VERDICT_LOCKED = "CERT_FO_CATCH"

# --- Gate C: optional #30 smoke (baseline without cert) ---
CITE30_FO_EXACT = 45
CITE30_HN_MAX = 0.10
HN_SHATTER_MAX = 0.50
CITE30_K16_MIN = 0.95
CITE30_VERDICT_NEEDLE = "HN_SHATTER_CONFIRMED"

# Sealed cite anchors (documentation / exact-match helpers)
CITE31_HARD_UNANIMOUS_SEALED = 34
CITE30_HN_SEALED = 0.06666666666666667
CITE30_K16_SEALED = 0.9875


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object, got {type(data)}")
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


def check_gate_a_hard_unanimous_oracle(
    cite31: dict[str, Any],
    cite38: dict[str, Any],
) -> dict[str, Any]:
    """Gate A: HARD_UNANIMOUS among FO stays high; oracle cannot kill FO."""
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    tags = (cite31.get("member_logits") or {}).get("tag_counts") or {}
    n_hard = int(tags.get("HARD_UNANIMOUS", -1))
    n_soft = int(tags.get("SOFT_AGREE", 0))
    n_mixed = int(tags.get("MIXED_PRED", 0))
    fo_ids = cite31.get("example_id_list_FO_HN") or []
    fo_total = len(fo_ids) if fo_ids else n_hard + n_soft + n_mixed
    rate = (n_hard / fo_total) if fo_total > 0 else 0.0

    ok_fo = fo_total == FO_TOTAL_LOCKED
    checks.append(
        {
            "name": "fo_total",
            "ok": ok_fo,
            "value": fo_total,
            "floor": FO_TOTAL_LOCKED,
        }
    )
    if not ok_fo:
        reasons.append(f"fo_total={fo_total}!={FO_TOTAL_LOCKED}")

    ok_count = n_hard >= HARD_UNANIMOUS_MIN_COUNT
    checks.append(
        {
            "name": "hard_unanimous_count",
            "ok": ok_count,
            "value": n_hard,
            "floor": HARD_UNANIMOUS_MIN_COUNT,
            "sealed_cite": CITE31_HARD_UNANIMOUS_SEALED,
        }
    )
    if not ok_count:
        reasons.append(
            f"HARD_UNANIMOUS={n_hard}<{HARD_UNANIMOUS_MIN_COUNT}"
        )

    ok_rate = rate >= HARD_UNANIMOUS_MIN_RATE
    checks.append(
        {
            "name": "hard_unanimous_rate",
            "ok": ok_rate,
            "value": rate,
            "floor": HARD_UNANIMOUS_MIN_RATE,
        }
    )
    if not ok_rate:
        reasons.append(f"HARD_UNANIMOUS_rate={rate:.4f}<{HARD_UNANIMOUS_MIN_RATE}")

    ok_mixed = n_mixed <= MIXED_PRED_MAX
    checks.append(
        {
            "name": "mixed_pred",
            "ok": ok_mixed,
            "value": n_mixed,
            "floor": MIXED_PRED_MAX,
        }
    )
    if not ok_mixed:
        reasons.append(f"MIXED_PRED={n_mixed}>{MIXED_PRED_MAX}")

    gap = cite38.get("selector_oracle_gap") or {}
    oracle_fo = int(gap.get("oracle_fo_killed", -1))
    oracle_rem = int(gap.get("oracle_rem22_killed", -1))
    # Fallback: arms.oracle_member
    if oracle_fo < 0:
        arm = (cite38.get("arms") or {}).get("oracle_member") or {}
        oracle_fo = int(arm.get("fo_killed_of_baseline45", -1))
    if oracle_rem < 0:
        arm = (cite38.get("arms") or {}).get("oracle_member") or {}
        oracle_rem = int(arm.get("rem22_killed", -1))

    ok_oracle_fo = oracle_fo == ORACLE_FO_KILLED_MAX
    checks.append(
        {
            "name": "oracle_fo_killed",
            "ok": ok_oracle_fo,
            "value": oracle_fo,
            "floor": ORACLE_FO_KILLED_MAX,
            "note": "oracle member cannot kill FO (HARD_UNANIMOUS wall)",
        }
    )
    if not ok_oracle_fo:
        reasons.append(f"oracle_fo_killed={oracle_fo}!={ORACLE_FO_KILLED_MAX}")

    ok_oracle_rem = oracle_rem == ORACLE_REM22_KILLED_MAX
    checks.append(
        {
            "name": "oracle_rem22_killed",
            "ok": ok_oracle_rem,
            "value": oracle_rem,
            "floor": ORACLE_REM22_KILLED_MAX,
        }
    )
    if not ok_oracle_rem:
        reasons.append(
            f"oracle_rem22_killed={oracle_rem}!={ORACLE_REM22_KILLED_MAX}"
        )

    # science_open must stay false on cite artifacts
    so31 = bool(cite31.get("science_open", True))
    so38 = bool(cite38.get("science_open", True))
    ok_so = (not so31) and (not so38)
    checks.append(
        {
            "name": "science_open_false_cites",
            "ok": ok_so,
            "value": {"cite31": so31, "cite38": so38},
            "floor": False,
        }
    )
    if not ok_so:
        reasons.append("science_open not false on cite31/38")

    passed = all(c["ok"] for c in checks)
    return {
        "gate": "A_HARD_UNANIMOUS_ORACLE",
        "pass": passed,
        "checks": checks,
        "reasons": reasons,
        "summary": {
            "fo_total": fo_total,
            "HARD_UNANIMOUS": n_hard,
            "SOFT_AGREE": n_soft,
            "MIXED_PRED": n_mixed,
            "hard_unanimous_rate": rate,
            "oracle_fo_killed": oracle_fo,
            "oracle_rem22_killed": oracle_rem,
        },
    }


def check_gate_b_cert_refuse(cite35: dict[str, Any]) -> dict[str, Any]:
    """Gate B: #35 cert refuse FO catch / P(correct|clean) / matched collateral."""
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    decision = cite35.get("decision") or {}
    fo_killed = int(decision.get("fo_killed", -1))
    rem22_killed = int(decision.get("rem22_killed", -1))
    verdict = str(decision.get("verdict") or cite35.get("verdict") or "")

    ok_fo = fo_killed == FO_CATCH_EXACT
    checks.append(
        {
            "name": "fo_killed_cert",
            "ok": ok_fo,
            "value": fo_killed,
            "floor": FO_CATCH_EXACT,
        }
    )
    if not ok_fo:
        reasons.append(f"fo_killed={fo_killed}!={FO_CATCH_EXACT}")

    ok_rem = rem22_killed == REM22_CATCH_EXACT
    checks.append(
        {
            "name": "rem22_killed_cert",
            "ok": ok_rem,
            "value": rem22_killed,
            "floor": REM22_CATCH_EXACT,
        }
    )
    if not ok_rem:
        reasons.append(f"rem22_killed={rem22_killed}!={REM22_CATCH_EXACT}")

    ok_verdict = verdict == CERT_VERDICT_LOCKED
    checks.append(
        {
            "name": "cert_verdict",
            "ok": ok_verdict,
            "value": verdict,
            "floor": CERT_VERDICT_LOCKED,
        }
    )
    if not ok_verdict:
        reasons.append(f"verdict={verdict!r}!={CERT_VERDICT_LOCKED!r}")

    p_clean = cite35.get("p_correct_given_clean") or {}
    p_post = float(p_clean.get("post_policy", -1.0))
    ok_p = p_post >= P_CORRECT_CLEAN_MIN
    checks.append(
        {
            "name": "p_correct_given_clean",
            "ok": ok_p,
            "value": p_post,
            "floor": P_CORRECT_CLEAN_MIN,
        }
    )
    if not ok_p:
        reasons.append(f"P(correct|clean)={p_post}<{P_CORRECT_CLEAN_MIN}")

    collateral = cite35.get("collateral") or {}
    harm = bool(collateral.get("harm", True))
    ok_harm = harm is False
    checks.append(
        {
            "name": "collateral_harm_false",
            "ok": ok_harm,
            "value": harm,
            "floor": False,
        }
    )
    if not ok_harm:
        reasons.append("collateral.harm is true")

    matched = collateral.get("matched_ood_T16") or {}
    deltas = matched.get("deltas_cert") or {}
    delta_vals = {
        "overall_acc": float(deltas.get("overall_acc", 99.0)),
        "hard_neg_acc": float(deltas.get("hard_neg_acc", 99.0)),
        "K16": float(deltas.get("K16", 99.0)),
    }
    ok_deltas = all(
        abs(v) <= MATCHED_COLLATERAL_ABS_MAX for v in delta_vals.values()
    )
    checks.append(
        {
            "name": "matched_ood_deltas_cert",
            "ok": ok_deltas,
            "value": delta_vals,
            "floor_abs": MATCHED_COLLATERAL_ABS_MAX,
        }
    )
    if not ok_deltas:
        reasons.append(f"matched deltas exceed ±{MATCHED_COLLATERAL_ABS_MAX}: {delta_vals}")

    so35 = bool(cite35.get("science_open", True))
    ok_so = not so35
    checks.append(
        {
            "name": "science_open_false_cite35",
            "ok": ok_so,
            "value": so35,
            "floor": False,
        }
    )
    if not ok_so:
        reasons.append("science_open not false on cite35")

    passed = all(c["ok"] for c in checks)
    return {
        "gate": "B_CERT_REFUSE",
        "pass": passed,
        "checks": checks,
        "reasons": reasons,
        "summary": {
            "fo_killed": fo_killed,
            "rem22_killed": rem22_killed,
            "verdict": verdict,
            "p_correct_given_clean": p_post,
            "matched_deltas_cert": delta_vals,
            "harm": harm,
        },
    }


def check_gate_c_smoke_cite30(cite30: dict[str, Any]) -> dict[str, Any]:
    """Gate C: baseline FO / HN shatter still present without cert (#30)."""
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    audit = (cite30.get("audit") or {}).get("baseline_prob_mean_T16") or {}
    fail_mode = audit.get("fail_mode") or {}
    fo = int(fail_mode.get("FAIL_OPEN", -1))
    ens = audit.get("ensemble") or {}
    # hop artifact may nest metrics under ensemble keys differently
    hn = ens.get("hard_neg_acc")
    k16 = ens.get("K16")
    if hn is None or k16 is None:
        # fallback: top-level replicate / hn_verdict
        hn_v = cite30.get("hn_verdict") or {}
        hn = hn_v.get("baseline_HN", hn)
        rep = cite30.get("replicate_cite28_hops") or {}
        cite = rep.get("cite_28_hops") or {}
        if hn is None:
            hn = cite.get("ens_hard_neg_acc")
        if k16 is None:
            k16 = cite.get("ens_K16")
    # Another common layout: metrics under audit.baseline...ensemble metrics
    if hn is None:
        by_hop = ens.get("by_hop") or {}
        hn_slot = by_hop.get("-1") or by_hop.get(-1) or {}
        hn = hn_slot.get("acc_mean")
    if k16 is None:
        by_hop = ens.get("by_hop") or {}
        k_slot = by_hop.get("16") or by_hop.get(16) or {}
        k16 = k_slot.get("acc_mean")

    # Direct from comparison if present in later replicates
    if fo < 0:
        fo = int((cite30.get("replicate_cite28_hops") or {}).get("FAIL_OPEN", -1))

    # Prefer explicit fail_mode from audit; else search top-level verdict smoke
    if fo < 0:
        # last resort: known sealed count from hn_verdict note path
        fo = -1

    hn_f = float(hn) if hn is not None else -1.0
    k16_f = float(k16) if k16 is not None else -1.0

    # If audit missing FO, try reading from nested fail_mode elsewhere
    if fo < 0:
        for key in ("fail_mode",):
            fm = cite30.get(key)
            if isinstance(fm, dict) and "FAIL_OPEN" in fm:
                fo = int(fm["FAIL_OPEN"])
                break

    # Parse from arms-like structure used when reverse-citing
    if fo < 0 and "FAIL_OPEN" in str(cite30.get("verdict", "")):
        # cannot recover count; leave failing
        pass

    ok_fo = fo == CITE30_FO_EXACT
    checks.append(
        {
            "name": "baseline_FO",
            "ok": ok_fo,
            "value": fo,
            "floor": CITE30_FO_EXACT,
        }
    )
    if not ok_fo:
        reasons.append(f"baseline_FO={fo}!={CITE30_FO_EXACT}")

    ok_hn = (0.0 <= hn_f <= CITE30_HN_MAX) and (hn_f < HN_SHATTER_MAX)
    checks.append(
        {
            "name": "baseline_HN_shatter",
            "ok": ok_hn,
            "value": hn_f,
            "floor_max": CITE30_HN_MAX,
            "shatter_max": HN_SHATTER_MAX,
            "sealed_cite": CITE30_HN_SEALED,
        }
    )
    if not ok_hn:
        reasons.append(
            f"baseline_HN={hn_f} not in [0,{CITE30_HN_MAX}] or >= shatter {HN_SHATTER_MAX}"
        )

    ok_k16 = k16_f >= CITE30_K16_MIN
    checks.append(
        {
            "name": "baseline_K16",
            "ok": ok_k16,
            "value": k16_f,
            "floor": CITE30_K16_MIN,
            "sealed_cite": CITE30_K16_SEALED,
        }
    )
    if not ok_k16:
        reasons.append(f"baseline_K16={k16_f}<{CITE30_K16_MIN}")

    verdict = str(cite30.get("verdict") or "")
    ok_v = CITE30_VERDICT_NEEDLE in verdict
    checks.append(
        {
            "name": "cite30_verdict_needle",
            "ok": ok_v,
            "value": verdict,
            "floor": CITE30_VERDICT_NEEDLE,
        }
    )
    if not ok_v:
        reasons.append(
            f"verdict missing {CITE30_VERDICT_NEEDLE!r}: {verdict!r}"
        )

    so30 = bool(cite30.get("science_open", True))
    ok_so = not so30
    checks.append(
        {
            "name": "science_open_false_cite30",
            "ok": ok_so,
            "value": so30,
            "floor": False,
        }
    )
    if not ok_so:
        reasons.append("science_open not false on cite30")

    passed = all(c["ok"] for c in checks)
    return {
        "gate": "C_SMOKE_CITE30",
        "pass": passed,
        "checks": checks,
        "reasons": reasons,
        "summary": {
            "FAIL_OPEN": fo,
            "HN": hn_f,
            "K16": k16_f,
            "verdict": verdict,
        },
    }


def decide_verdict(
    gate_a: dict[str, Any],
    gate_b: dict[str, Any],
    gate_c: Optional[dict[str, Any]],
    *,
    require_smoke: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not gate_a.get("pass"):
        reasons.extend(gate_a.get("reasons") or ["gate_A_fail"])
    if not gate_b.get("pass"):
        reasons.extend(gate_b.get("reasons") or ["gate_B_fail"])
    smoke_ok = True
    if require_smoke:
        if gate_c is None:
            smoke_ok = False
            reasons.append("gate_C_missing")
        elif not gate_c.get("pass"):
            smoke_ok = False
            reasons.extend(gate_c.get("reasons") or ["gate_C_fail"])

    passed = bool(gate_a.get("pass")) and bool(gate_b.get("pass")) and smoke_ok
    return {
        "verdict": "PASS_REGRESSION" if passed else "FAIL",
        "pass": passed,
        "require_smoke": require_smoke,
        "reasons": reasons,
        "science_open": False,
    }


def run_regression(
    *,
    cite30_path: Path = DEFAULT_CITE30,
    cite31_path: Path = DEFAULT_CITE31,
    cite35_path: Path = DEFAULT_CITE35,
    cite38_path: Path = DEFAULT_CITE38,
    out_path: Path = DEFAULT_OUT,
    require_smoke: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    cite31 = _load_json(cite31_path)
    cite35 = _load_json(cite35_path)
    cite38 = _load_json(cite38_path)
    cite30: Optional[dict[str, Any]] = None
    if require_smoke:
        cite30 = _load_json(cite30_path)

    gate_a = check_gate_a_hard_unanimous_oracle(cite31, cite38)
    gate_b = check_gate_b_cert_refuse(cite35)
    gate_c = check_gate_c_smoke_cite30(cite30) if cite30 is not None else None
    decision = decide_verdict(gate_a, gate_b, gate_c, require_smoke=require_smoke)

    thresholds = {
        "FO_TOTAL_LOCKED": FO_TOTAL_LOCKED,
        "HARD_UNANIMOUS_MIN_COUNT": HARD_UNANIMOUS_MIN_COUNT,
        "HARD_UNANIMOUS_MIN_RATE": HARD_UNANIMOUS_MIN_RATE,
        "MIXED_PRED_MAX": MIXED_PRED_MAX,
        "ORACLE_FO_KILLED_MAX": ORACLE_FO_KILLED_MAX,
        "ORACLE_REM22_KILLED_MAX": ORACLE_REM22_KILLED_MAX,
        "FO_CATCH_EXACT": FO_CATCH_EXACT,
        "REM22_CATCH_EXACT": REM22_CATCH_EXACT,
        "P_CORRECT_CLEAN_MIN": P_CORRECT_CLEAN_MIN,
        "MATCHED_COLLATERAL_ABS_MAX": MATCHED_COLLATERAL_ABS_MAX,
        "CERT_VERDICT_LOCKED": CERT_VERDICT_LOCKED,
        "CITE30_FO_EXACT": CITE30_FO_EXACT,
        "CITE30_HN_MAX": CITE30_HN_MAX,
        "HN_SHATTER_MAX": HN_SHATTER_MAX,
        "CITE30_K16_MIN": CITE30_K16_MIN,
        "CITE30_VERDICT_NEEDLE": CITE30_VERDICT_NEEDLE,
        "FOCUS_T": FOCUS_T,
    }

    payload: dict[str, Any] = {
        "cycle": CYCLE,
        "mode": "MEASURE_CI_HYGIENE",
        "science_open": False,
        "section22_unchanged": True,
        "hop_ood_remains_measure_residue": True,
        "train": False,
        "generation": False,
        "T_fixed": FOCUS_T,
        "git_sha": _git_sha(),
        "elapsed_sec": time.time() - t0,
        "datasets": {
            "cite30": str(cite30_path),
            "cite31": str(cite31_path),
            "cite35": str(cite35_path),
            "cite38": str(cite38_path),
        },
        "prereg": {
            "thresholds": thresholds,
            "verdicts": list(VERDICTS),
            "non_goals": [
                "no_train",
                "no_generation",
                "no_science_open_widen",
                "section22_unchanged",
                "T_fixed_16",
                "no_hop_ood_open_claim",
                "artifact_first_ci_gate",
            ],
        },
        "gates": {
            "A": gate_a,
            "B": gate_b,
            "C": gate_c,
        },
        "decision": decision,
        "verdict": decision["verdict"],
        "open_status": {
            "science_open": False,
            "section22_widened": False,
            "claim_hop_ood_open": False,
        },
        "reading": (
            "Regression lock on sealed #30/#31/#35/#38 storyline: "
            "HARD_UNANIMOUS FO wall + oracle FO killed=0; "
            "cert refuse 45/45 with P(correct|clean)≈1 and matched Δ≈0; "
            "optional #30 FO/HN shatter smoke. MEASURE/CI only; science_open=false."
        ),
        "residue": (
            "HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/CERT_FO_CATCH/"
            "HARD_UNANIMOUS_ORACLE_WALL/PASS_REGRESSION"
            if decision["pass"]
            else "REGRESSION_FAIL"
        ),
    }

    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    return payload


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "CYCLE_STALK_FO_CERT_REGRESSION — artifact regression gate "
            "(science_open=false; no train)"
        )
    )
    p.add_argument("--cite30", type=Path, default=DEFAULT_CITE30)
    p.add_argument("--cite31", type=Path, default=DEFAULT_CITE31)
    p.add_argument("--cite35", type=Path, default=DEFAULT_CITE35)
    p.add_argument("--cite38", type=Path, default=DEFAULT_CITE38)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Omit Gate C (#30 baseline FO/HN shatter smoke)",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write artifacts/stalk_fo_cert_regression.json",
    )
    args = p.parse_args(argv)

    result = run_regression(
        cite30_path=args.cite30,
        cite31_path=args.cite31,
        cite35_path=args.cite35,
        cite38_path=args.cite38,
        out_path=args.out,
        require_smoke=not args.skip_smoke,
        write=not args.no_write,
    )
    verdict = result["verdict"]
    print(f"{CYCLE} verdict={verdict} science_open=false")
    for gname, gate in (result.get("gates") or {}).items():
        if gate is None:
            print(f"  gate {gname}: skipped")
            continue
        status = "PASS" if gate.get("pass") else "FAIL"
        print(f"  gate {gname} ({gate.get('gate')}): {status}")
        if not gate.get("pass"):
            for r in gate.get("reasons") or []:
                print(f"    - {r}")
    if not args.no_write:
        print(f"wrote {args.out}")
    return 0 if verdict == "PASS_REGRESSION" else 1


if __name__ == "__main__":
    sys.exit(main())
