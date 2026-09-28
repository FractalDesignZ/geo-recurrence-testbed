# CYCLE_SHEAF_DENSE_CONTEXT — MEASURE (fail-closed)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | MEASURE (two-cell RED_TEST) |
| **Cycle** | `CYCLE_SHEAF_DENSE_CONTEXT` |
| **science_open** | **false** (do not widen sheaf §6 / K≤16; harness never self-stamps true) |
| **Prior** | PR #5 merge `a6665bc` (DENSITY_STRESS INVALID / feasibility wall); seal `docs/SESSION-SEAL-SHEAF-DENSITY-FEASIBILITY.md` |
| **Ckpt** | frozen `artifacts/sheaf_infer_gate1_best.pt` (param_count **123206**, no retrain) |

Fail-closed. Evidence = cited artifact paths only. Do not invent metrics. No HF.

---

## Design

Re-band to **[100,140]** (mean target ~110–120) after the sealed [45,70] feasibility wall
(n=16 p=0.15 → |E|≈36 → ~111 tokens). Contrast dense vs matched-sparse at matched context length.

| Cell | Spec |
|------|------|
| **1 Dense** | n=16, true ER digraph **p=0.15**, K=8, T∈{8,12} |
| **2 Matched sparse** | n=32, ER **p≈0.036** tuned so &#124;E&#124; and seq_len match Cell1 empirically in-band; K=8, T∈{8,12} |
| Samples | 128/cell = 64 pos (exact hop=8) + 64 hard-neg |
| Hard-neg | Rejection-sampled; assert `v ∉ R_out(u)` via full BFS **and** DFS |
| Seq-len | mean **must** ∈ [100,140] for **both** cells; else **INVALID** |

## Prereg floors (written before interpret)

- FPR(Â) ≤ 0.02, FNR(Â) ≤ 0.02 (each cell)
- hard-neg ≥ 0.950 at T=8 and T=12
- K8 pos ≥ 0.950 at T=8 and T=12

## Attribution

| Mode | Meaning |
|------|---------|
| A | Cell1 Â fail, Cell2 Â pass → dense hallucination |
| Encoder Context Ceiling | Both Â fail (zero-shot 110+ tokens; do **not** claim architecture cannot learn — note retrain as next cycle) |
| C | Â both clean, Cell1 reachability drops |
| PASS | both cells pass all floors (MEASURE PASS; **no** science_open widen) |
| INVALID | seq_len mean ∉ [100,140] for either cell |

## Artifacts

| Slice | Path |
|-------|------|
| Result | `artifacts/sheaf_infer_dense_context.json` |
| Gen report | `artifacts/sheaf_dense_context_generation_report.json` |
| Data Cell1 | `data/sheaf_dense_context_cell1.jsonl` |
| Data Cell2 | `data/sheaf_dense_context_cell2.jsonl` |
| Runner | `python -m reachability_gen.run_sheaf_dense_context` |
| Gen helper | `python -m reachability_gen.gen_sheaf_dense_context` |

## Result (from artifact)

| Field | Observed |
|-------|----------|
| **Verdict** | **MEASURE PASS** |
| **mode** | **PASS** |
| seq_len Cell1 / Cell2 | **110.79** / **116.16** (band [100,140]; both in-band) |
| &#124;E&#124; mean Cell1 / Cell2 | **34.93** / **36.72** |
| Cell2 p_requested (tuned) | **0.0352** (init 0.036; p_emp mean ≈0.0370) |
| Cell1 p_emp mean | **≈0.1455** (near 0.15; no backbone) |
| Cell1 FPR/FNR(Â) | **0.000 / 0.000** |
| Cell2 FPR/FNR(Â) | **0.000 / 0.000** |
| science_open | **false** |

### Cell1 Dense — reachability

| T | Overall | Hard-neg | K8 pos |
|---|---------|----------|--------|
| 8 | 1.000 | 1.000 | 1.000 |
| 12 | 1.000 | 1.000 | 1.000 |

### Cell2 Matched sparse — reachability

| T | Overall | Hard-neg | K8 pos |
|---|---------|----------|--------|
| 8 | 1.000 | 1.000 | 1.000 |
| 12 | 1.000 | 1.000 | 1.000 |

**Interpretation (fail-closed):** both cells pass all floors under frozen Gate1 at ~110–116 tokens.
This is **MEASURE PASS** only — do **not** widen sheaf §6 / K≤16 science OPEN.
Prior density-stress under sealed [45,70] remains INVALID (orthogonal feasibility wall).

Cite: `artifacts/sheaf_infer_dense_context.json`, `artifacts/sheaf_dense_context_generation_report.json`.
