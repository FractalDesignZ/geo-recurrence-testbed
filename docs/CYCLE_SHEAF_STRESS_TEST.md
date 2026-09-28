# CYCLE_SHEAF_STRESS_TEST — MEASURE plan (stub)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | MEASURE (plan + RED_TEST cell pointer) |
| **Cycle** | `CYCLE_SHEAF_STRESS_TEST` |
| **science_open** | **false** (until measured + human seal; harness never self-stamps true) |
| **Prior seal** | sheaf inference seal — `docs/SESSION-SEAL-SHEAF-INFERENCE.md` (PR #3 / MEASURE `f374e7c`) |
| **Baseline stalk seal** | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` @ `b144dac` (hard-mask claim; separate) |
| **Out of scope (full stress)** | Full K∈{20,24,32} × density grid + retrain arm — still deferred; **RED_TEST** single cell is the first executed slice |

Fail-closed. Do not stamp OPEN from this document alone.

---

## Motivation

`CYCLE_SHEAF_INFERENCE` opened a **scoped** claim: learned directed restriction maps + stalk-local discrete diffusion resolve zero-shot path-length gen up to **K=16** on directed graphs **without** an external A oracle at eval — on `covariate_matched_ood` (token_len band [45,70], synthetic ER digraphs). Residue: sealed evidence does **not** cover K&gt;16 or denser regimes.

## Goal

Stress `SheafInferCore` beyond the sealed envelope:

1. Longer causal hops: **K ∈ {20, 24, 32}**.
2. Denser ER digraphs: edge probability **p up to 0.15** (see density protocol below).
3. Preserve **zero leakage** on hard-neg / disconnect and **causal horizon** (T&lt;K → fail on that hop; T≥K → high acc).
4. Keep **no A oracle at eval** and param-parity hygiene.

---

## Density protocol (vs current generators)

| Item | Current sealed substrate | Stress proposal |
|------|--------------------------|-----------------|
| Graph family | Synthetic ER digraphs (`er_digraph`) | Same family unless a later MEASURE names another |
| ID split `P_VALUES` | `(0.15, 0.25, 0.35)` in `splits.py` | Unchanged for ID unless retrain arm requires it |
| Matched-OOD draws | Mostly sparse effective p (often ≪0.15; covariate band-driven) | Explicitly include cells with **p ∈ {0.08, 0.10, 0.12, 0.15}** (cap **0.15**) when searching for hop-K positives |
| How density applies | ER edge Bernoulli(p) on digraph of n nodes | Same: raise p in the hop-search grid; do **not** reinterpret p as undirected density |
| Seq-len matching | Sealed OOD used token_len band **[45, 70]** (covariate match to ID whitespace mean ~57) | **Still required** if claiming matched-OOD continuity: either (a) keep band [45,70] and accept that high-p / high-K may need smaller n / edge-list compression, or (b) declare a **new** band and treat as a separate substrate (fail-closed vs sealed claim) |
| n support | Sealed OOD n mostly 10–22 | May need larger n for K∈{20,24,32}; document n grid in the MEASURE artifact |

If band matching and K=32 are jointly infeasible under p≤0.15, **STOP** and report the feasibility wall rather than silently dropping the covariate constraint.

---

## MEASURE protocol (proposed)

### Controls

| Control | Requirement |
|---------|-------------|
| Freeze vs retrain | Arm A: **freeze** `artifacts/sheaf_infer_gate1_best.pt` from sheaf seal (eval-only stress). Arm B: **retrain** SheafInferCore under same hparam family if freeze fails length/density. Report both; do not mix. |
| Param parity | ±5% of FF 121218 → window [115157, 127279]; hard-fail outside |
| A oracle | **`hard_A_oracle_eval=false`** always |
| Gate default | Prefer sealed default: `gate_detach_diffusion=true`; document if STE/Gumbel end-to-end is ablated |
| Discrete T | Eval T grid must include values **≥ each K** under test (e.g. T∈{20,24,32} and/or T=K for each K column) |

### Prereg success criteria (propose; mirror prior floors)

Concrete thresholds for a later OPEN consideration (report PASS/FAIL honestly; `science_open=false` until human seal):

| Criterion | Threshold |
|-----------|-----------|
| Hard-neg floor | hard-neg accuracy ≥ **0.95** at T_max (T≥32 recommended) |
| Long-hop ceiling | **K32 @ T≥32** accuracy ≥ **0.75** |
| Mid hops | K20 @ T≥20 and K24 @ T≥24 each ≥ **0.75** (report all) |
| Causal horizon | For each K∈{20,24,32}: when **T&lt;K**, hop-stratified acc on that K ≈ **0** (or clearly collapsed); when **T≥K**, acc meets the floors above |
| Density slice | At least one eval slice with **p=0.15** cells present in the report (or documented impossibility + STOP) |
| Disconnect / leakage | Gate0-style ‖h_t‖≈0 on disconnect under `A_hat` still holds on a stress overfit or documented subsample |

### Kill / STOP conditions

- Hard-neg &lt; 0.95 at T_max under freeze **and** retrain → **STOP** (leakage / topology failure).
- K32@T≥32 &lt; 0.75 after freeze+retrain with documented hparam parity → **STOP** length-gen claim extension.
- Causal horizon inverted (T&lt;K still high on that K) without an explained substrate bug → **STOP** (suggests shortcut / oracle leak).
- Param count outside ±5% FF window → **STOP** until architecture rebalanced.
- A oracle accidentally enabled at eval → **STOP** / invalidate run.
- Covariate band abandoned without declaring a new substrate → **STOP** (would silently widen vs sealed claim).

---

## Non-goals (fail-closed)

- Stamping `science_open=true` from this stub.
- Rewriting `docs/SESSION-SEAL-SHEAF-INFERENCE.md` or the stalk seal.
- Soft ACT reintroduction.
- Claiming universal graphs, non-ER families, or K&gt;32 without a further cycle.
- Implementing code in this document.

## Artifacts (expected when MEASURE runs)

| Slice | Path (proposed) |
|-------|-----------------|
| Stress OOD report | `artifacts/sheaf_stress_matched_ood.json` (or successor name; full grid TBD) |
| Generation report | `artifacts/sheaf_stress_ood_generation_report.json` |
| Retrain ckpt (if any) | `artifacts/sheaf_stress_gate1_best.pt` |
| **RED_TEST result** | `artifacts/sheaf_infer_red_test.json` |
| **RED_TEST gen report** | `artifacts/sheaf_red_test_generation_report.json` |

## RED_TEST cell (executed MEASURE pointer)

| Field | Value |
|-------|-------|
| **Cycle slice** | `CYCLE_SHEAF_RED_TEST` (single cell under this stress plan) |
| **Cell** | K=20, p_requested=0.15, T∈{20,24}, frozen Gate1 ckpt |
| **Runner** | `src/reachability_gen/run_sheaf_red_test.py` |
| **Gen helper** | `src/reachability_gen/gen_sheaf_red_test.py` |
| **Result artifact** | `artifacts/sheaf_infer_red_test.json` |
| **Gen report** | `artifacts/sheaf_red_test_generation_report.json` |
| **Data** | `data/sheaf_red_test_k20.jsonl` |
| **science_open** | **false** (stress / red-test still fail-closed; do not widen §6 sheaf claim) |

Path-backbone construction keeps seq_len in sealed band [45,70]; pure ER@p=0.15 under band is documented infeasible for K=20. See artifact for prereg floors, Â FPR/FNR, attribution mode, and PASS/FAIL.

## DENSITY_STRESS cell (executed MEASURE pointer)

| Field | Value |
|-------|-------|
| **Cycle slice** | `CYCLE_SHEAF_DENSITY_STRESS` |
| **Cell** | K=8, n=16, true ER p=0.15, T∈{8,12}, frozen Gate1 |
| **Doc** | `docs/CYCLE_SHEAF_DENSITY_STRESS.md` |
| **Result** | `artifacts/sheaf_infer_density_stress.json` — **INVALID** (seq_len mean≈111 ∉ [45,70]; no path-backbone) |
| **science_open** | **false** |

## science_open

**false** until a human seal cites stress Gate artifacts. Harness stamps `science_open: false` always. RED_TEST / DENSITY_STRESS do **not** open science.
