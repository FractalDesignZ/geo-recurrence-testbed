# CYCLE_SHEAF_DENSITY_STRESS — MEASURE (fail-closed)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | MEASURE (single cell) |
| **Cycle** | `CYCLE_SHEAF_DENSITY_STRESS` |
| **science_open** | **false** (do not widen sheaf §6 claim; harness never self-stamps true) |
| **Prior** | PR #4 merge `7a02dea` (RED_TEST K=20 MEASURE residue); sheaf seal §6 remains K≤16 only |
| **Ckpt** | frozen `artifacts/sheaf_infer_gate1_best.pt` (param_count **123206**, no retrain) |

Fail-closed. Evidence = cited artifact paths only. Do not invent metrics.

---

## Cell

| Param | Value |
|-------|-------|
| K | 8 |
| n | 16 |
| Graph | **true** ER digraph Bernoulli(**p=0.15**) |
| Path-backbone | **Forbidden** (no density drop to force band) |
| T | {8, 12} |
| Samples | 128 = 64 pos (exact hop=8) + 64 hard-neg |
| Hard-neg | Rejection-sampled; assert `v ∉ R_out(u)` via full BFS **and** DFS |
| Seq-len | mean **must** ∈ [45, 70]; else **INVALID** and stop |

## Prereg floors (written before interpret)

- FPR(Â) ≤ 0.02, FNR(Â) ≤ 0.02
- hard-neg ≥ 0.950
- positive reachability ≥ 0.950 at T=8 and T=12

## Attribution

| Mode | Meaning |
|------|---------|
| A | FPR(Â) > 0.02 |
| B | FNR(Â) > 0.02 |
| C | Â clean but accuracy drops below floors |
| PASS | all floors met |
| INVALID | seq_len mean ∉ [45,70] under true p=0.15 (no backbone fake) |

## Artifacts

| Slice | Path |
|-------|------|
| Result | `artifacts/sheaf_infer_density_stress.json` |
| Gen report | `artifacts/sheaf_density_stress_generation_report.json` |
| Data | `data/sheaf_density_stress_p015.jsonl` |
| Runner | `python -m reachability_gen.run_sheaf_density_stress` |
| Gen helper | `python -m reachability_gen.gen_sheaf_density_stress` |

## Result (from artifact)

| Field | Observed |
|-------|----------|
| **Verdict** | **INVALID** |
| seq_len mean | **111.07** (band [45,70]; min=78 max=168) |
| p_empirical mean | **0.1459** (near p_requested=0.15; no backbone) |
| FPR(Â) / FNR(Â) | 0.000 / 0.000 (telemetry only; not a PASS under INVALID) |
| mode | **INVALID** (attribution suspended) |
| science_open | **false** |

| T | Overall | Hard-neg | K8 pos |
|---|---------|----------|--------|
| 8 | 1.000 | 1.000 | 1.000 |
| 12 | 1.000 | 1.000 | 1.000 |

**INVALID reason:** true ER digraph p=0.15 @ n=16 yields token_len mean ≈111 ∉ [45,70]. Band requires |E|∈[13,21] while E[|E|]≈36. Do **not** fake density with path-backbone. Feasibility wall documented; prereg.pass=false / fail_closed.

Cite: `artifacts/sheaf_infer_density_stress.json`, `artifacts/sheaf_density_stress_generation_report.json`.
