# SESSION SEAL — CYCLE_SHEAF_DENSE_CONTEXT (dense band [100,140])

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | FREEZE / human seal |
| **Cycle** | `CYCLE_SHEAF_DENSE_CONTEXT` |
| **science_open** | **true** (scoped claim only — §6) |
| **Repo** | https://github.com/FractalDesignZ/geo-recurrence-testbed |
| **Branch** | `cycle/sheaf-dense-context` |
| **MEASURE SHA (pre-merge tip)** | `e145113` (`e14511360f4039fb4425e0fe873a171ed158db3d`) |
| **Merge target** | PR #6 → `main` |
| **Merge SHA (post-merge)** | `353bd61` (`353bd6197fb61e0eed8475d9d13e1fcd5648cb6d`) |
| **Prior** | PR #5 merge `a6665bc` (DENSITY_STRESS INVALID); sheaf §6 sparse [45,70] K≤16 OPEN unchanged |
| **Verdict class** | OPEN (scoped science) + FAIL-CLOSED elsewhere |

Fail-closed outside the single claim in §6. Append-only pointer on `docs/SESSION-SEAL-SHEAF-INFERENCE.md`. Evidence = cited artifact paths; do not invent metrics. Harness never self-stamps `science_open=true` (`science_open: false` in JSON).

---

## 1. Scope sealed

| Slice | Artifact | Status |
|-------|----------|--------|
| Two-cell dense-context MEASURE | `artifacts/sheaf_infer_dense_context.json` | SEALED — **Prereg PASS** / mode **PASS** |
| Generation report | `artifacts/sheaf_dense_context_generation_report.json` | SEALED |
| Cycle doc | `docs/CYCLE_SHEAF_DENSE_CONTEXT.md` | SEALED |
| Frozen Gate1 ckpt | `artifacts/sheaf_infer_gate1_best.pt` | SEALED (no retrain; param_count **123206**) |

Substrate: seq_len band **[100,140]** (orthogonal to sealed matched-OOD [45,70]). Cell1 true ER digraph **n=16 p=0.15 K=8**; Cell2 sequence-matched sparse ER **n=32 p=0.0352** (tuned). Discrete T∈{8,12}; hard-neg R_out BFS+DFS; **no** hard A oracle at eval.

---

## 2. Prereg PASS — both cells

Cite: `artifacts/sheaf_infer_dense_context.json` → `prereg` / `attribution` / `cell1` / `cell2`.

| Field | Observed |
|-------|----------|
| **Verdict** | **MEASURE PASS** → human seal opens §6 only |
| seq_len Cell1 / Cell2 | **110.79** / **116.16** (both ∈ [100,140]) |
| \|E\| mean Cell1 / Cell2 | **34.93** / **36.72** |
| Cell1 p_emp mean | **≈0.1455** (p_requested=0.15; no backbone) |
| Cell2 p_requested (tuned) | **0.0352** (p_emp mean ≈0.0370) |
| ckpt | frozen Gate1; param_count **123206**; `hard_A_oracle_eval: false` |
| harness `science_open` | **false** |

### Â vs gold (compiler clean)

| Cell | FPR(Â) | FNR(Â) |
|------|--------|--------|
| Cell1 dense | **0.000** | **0.000** |
| Cell2 matched-sparse | **0.000** | **0.000** |

### Reachability tables

**Cell1 Dense** (n=16 p=0.15 K=8):

| T | Overall | Hard-neg | K8 |
|---|---------|----------|-----|
| 8 | 1.000 | **1.000** | **1.000** |
| 12 | 1.000 | **1.000** | **1.000** |

**Cell2 Matched sparse** (n=32 p=0.0352 K=8):

| T | Overall | Hard-neg | K8 |
|---|---------|----------|-----|
| 8 | 1.000 | **1.000** | **1.000** |
| 12 | 1.000 | **1.000** | **1.000** |

---

## 3. Envelope map

| Envelope slice | Status | Cite |
|----------------|--------|------|
| K≤16 sparse matched-OOD seq_len∈[45,70] | **OPEN** (unchanged) | sheaf seal §6 / PR #3 merge `af8e49f` |
| K=20 sparse / path-backbone RED_TEST | **MEASURE** only | PR #4 merge `7a02dea`; sheaf seal §12 |
| True ER p=0.15 @ n=16 under sealed [45,70] | **INVALID** (feasibility wall) | PR #5 merge `a6665bc`; `docs/SESSION-SEAL-SHEAF-DENSITY-FEASIBILITY.md` |
| Dense ER p=0.15 n=16 K=8 vs matched-sparse @ seq_len∈[100,140], T∈{8,12}, frozen Gate1 | **OPEN** (this seal §6) | PR #6 / `artifacts/sheaf_infer_dense_context.json` |
| Dense K≥16 | **NOT OPEN** | unmeasured |
| NL / CoT / soft ACT / Spaces | **NOT OPEN** | aspirational / rejected |
| Depth × Density frontier | **residue next** | §8 |

---

## 4. Explicit NON-claims

`science_open=false` / fail-closed on all of the following:

| NON-claim | Why closed |
|-----------|------------|
| Dense **K≥16** | Sealed dense cell is **K=8 only**; longer hops unmeasured on dense ER |
| NL / CoT / o1 metaphors as mechanism | Not evidence; rejected as sealed claims |
| Widen of sheaf §6 **K≤16 sparse [45,70]** OPEN | This seal is **orthogonal** (new band + density); does **not** rewrite or widen that claim |
| PR #4 K=20 RED_TEST as science OPEN | Remains **MEASURE** residue only |
| PR #5 density-stress under [45,70] | Remains **INVALID** / feasibility wall; telemetry SUSPENDED |
| Retrain / architecture change | Frozen Gate1 ckpt only; zero-shot |
| Universality / alternate graph families beyond cited ER cells | Unmeasured |
| Mandelbrot analogy as science | Aspirational only |

---

## 5. Evidence vs aspiration

| Label | Content |
|-------|---------|
| **Evidence** | Prereg PASS; Â FPR=FNR=0 both cells; hard-neg=1.000 / K8=1.000 at T∈{8,12}; seq_len means in [100,140]; frozen Gate1; cited JSON |
| **Aspiration** | Dense K≥16; Depth×Density joint frontier; NL/CoT bridge; universal graphs |

---

## 6. Scoped science OPEN (single claim)

| Field | Value |
|-------|-------|
| **science_open** | **true** |
| **Claim** | Learned cellular sheaf inference + discrete stalk diffusion show zero-shot generalization to seq_len∈[100,140] and dense directed ER (p=0.15, n=16, K=8) vs sequence-matched sparse control, with Â FPR=0, FNR=0, hard-neg=1.000, K8=1.000 at T∈{8,12} on frozen Gate1 ckpt. |
| **Substrate** | Band [100,140]; Cell1 true ER digraph n=16 p=0.15 K=8; Cell2 matched-sparse n=32 p=0.0352; discrete T∈{8,12}; frozen `artifacts/sheaf_infer_gate1_best.pt`; no hard A at eval |
| **Evidence** | §2 tables; `artifacts/sheaf_infer_dense_context.json` |
| **SHAs** | MEASURE pre-merge tip `e145113`; merge commit `353bd61` via PR [#6](https://github.com/FractalDesignZ/geo-recurrence-testbed/pull/6) → `main` |

Do **not** generalize this OPEN beyond the cited cells, K=8, band [100,140], and frozen Gate1 architecture. Does **not** widen the sparse [45,70] K≤16 OPEN.

---

## 7. Fail-closed invariants preserved

- Artifact harness `science_open: false`; human seal opens **only** §6.
- No A oracle at eval; gold edges for Â FPR/FNR comparison only (compiler hygiene).
- Evidence = cited artifact paths; tip `e145113` / merge `353bd61` / PR #6 → `main`.
- Outside §6 claim: **fail-closed**.
- Do **not** rewrite sheaf seal §1–§14 body; append-only pointer only.
- PR #5 INVALID stands; PR #4 K20 remains MEASURE.

---

## 8. Next frontier — Depth × Density residue

**Residue:** Depth×Density frontier (joint stress: longer hops **and** dense ER under an admissible band) — MEASURE plan only until measured; `science_open=false` until a later human seal.

Prior sparse K=20 MEASURE and this dense K=8 OPEN remain separate envelopes; combining them is the next cycle, not this seal.
