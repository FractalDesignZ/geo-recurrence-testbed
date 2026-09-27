# CYCLE_SHEAF_INFERENCE — aspirational MEASURE plan

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | PLAN (not MEASURE yet) |
| **Cycle** | `CYCLE_SHEAF_INFERENCE` |
| **science_open** | **false** (until measured) |
| **Prior seal** | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` (`1d6c3da` / PR #2) |

---

## Motivation

`CYCLE_STALK_LOCALIZATION` opened a **scoped** claim: stalk locality + discrete depth resolves zero-shot path-length gen up to K=16 on directed graphs — but **only** with a hard `A_ij` mask. Residue: topology was handed, not learned.

## Goal (aspirational)

Infer sheaf / restriction maps **F_{u→v}** from edge tokens **without** hard adjacency mask, while preserving:

1. **Zero leakage** on disconnect (Gate0-style stalk-ablation / ‖h_t‖≈0).
2. **Stability** under discrete cycle depth (causal horizon T vs K).
3. Param parity hygiene vs FF baseline (±5%).

## Non-goals (fail-closed)

- Soft ACT reintroduction without a separate MEASURE.
- Unrestricted MLP loops as a substitute for local stalk/probe.
- Stamping `science_open=true` before Gate0/1 artifacts exist.
- Claiming universal graphs beyond a declared matched substrate.

## MEASURE sketch (not executed here)

1. Replace hard `A_ij` with edge-token → F_{u→v} inference path; keep local stalk@s / probe@t.
2. Gate0: balanced overfit + disconnect ablation (expect max_l2 ≤ atol).
3. Gate1: id_2k train + `covariate_matched_ood` discrete T∈{6,8,12,16}; prereg hard-neg / K16 thresholds TBD before run.
4. Seal only after cited artifacts; default `science_open=false`.

No training in this stub.
