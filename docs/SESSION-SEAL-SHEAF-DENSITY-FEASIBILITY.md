# SESSION SEAL — SHEAF DENSITY FEASIBILITY WALL (MEASURE residue)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | MEASURE residue / fail-closed |
| **Cycle** | `CYCLE_SHEAF_DENSITY_STRESS` (PR #5) |
| **science_open** | **false** (no scientific promotion; do not widen sheaf §6 / K≤16) |
| **Repo** | https://github.com/FractalDesignZ/geo-recurrence-testbed |
| **Prior** | PR #4 merge `7a02dea` (K=20 RED_TEST MEASURE residue); sheaf seal §6 remains K≤16 only |
| **Artifact** | `artifacts/sheaf_infer_density_stress.json` |
| **Doc** | `docs/CYCLE_SHEAF_DENSITY_STRESS.md` |

Fail-closed. Evidence = cited artifact paths only. Do not invent metrics.
Append-only pointer lives on `docs/SESSION-SEAL-SHEAF-INFERENCE.md` §13 — prior seal body is **not** rewritten.

---

## 1. Feasibility wall (orthogonal to sealed band [45,70])

True ER digraph Bernoulli(**p=0.15**) at **n=16**:

| Quantity | Value |
|----------|-------|
| E[&#124;E&#124;] | ≈ **36** (= p · n(n−1) = 0.15 · 240) |
| token / seq_len mean (observed) | **~111** (cite artifact: **111.07**; gen report band note ≈114) |
| Sealed matched-OOD band | **[45, 70]** (requires &#124;E&#124;∈[13,21]) |
| Relation | **Orthogonal** — true p=0.15 @ n=16 cannot land in [45,70] without faking density |

**Do not** path-backbone / density-drop to force the sealed band. Feasibility wall documented; density stress under [45,70] is **INVALID**.

---

## 2. PR #5 telemetry — SUSPENDED / INVALID

| Field | Value |
|-------|-------|
| **PR** | [#5](https://github.com/FractalDesignZ/geo-recurrence-testbed/pull/5) — `cycle/sheaf-density-stress` → `main` |
| **Verdict** | **INVALID** (seq_len mean ∉ [45,70]) |
| **Telemetry status** | **SUSPENDED / INVALID** — not a PASS; not a scientific claim |
| **Attribution** | suspended (`mode=INVALID`) |
| **prereg.pass** | **false** / fail_closed |
| **science_open** | **false** |

Observed telemetry under INVALID (cite only; **no promotion**):

| Slice | Observed |
|-------|----------|
| seq_len mean | **111.07** (min=78 max=168; band [45,70]) |
| p_empirical mean | **≈0.1459** (near p_requested=0.15; no backbone) |
| FPR(Â) / FNR(Â) | 0.000 / 0.000 (telemetry only) |
| T=8 / T=12 tables | overall=hard-neg=K8=**1.000** (telemetry only) |

Ckpt: frozen `artifacts/sheaf_infer_gate1_best.pt` (param_count **123206**, no retrain).

---

## 3. Explicit non-claims

- **No** widen of sheaf §6 / K≤16 OPEN (`docs/SESSION-SEAL-SHEAF-INFERENCE.md`).
- **No** claim that dense ER p=0.15 is compiler-clean or reachability-safe under the sealed [45,70] substrate.
- **No** scientific promotion of INVALID-run tables or Â FPR/FNR.
- Path-backbone remains **forbidden** for true-density MEASURE under the sealed band.

---

## 4. Next MEASURE (handoff)

`CYCLE_SHEAF_DENSE_CONTEXT` — re-band to **[100,140]** (mean target ~110–120), keep frozen Gate1 ckpt, Cell1 dense n=16 p=0.15 vs Cell2 matched sparse n=32 ER p≈0.036 (&#124;E&#124; / seq_len matched), T∈{8,12}, 128/cell. Fail-closed; `science_open=false`.
