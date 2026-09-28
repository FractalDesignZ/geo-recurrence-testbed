# CYCLE_STALK_PARK_ACCEPT_MEASURE — MEASURE close (science_open=false)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-28 (CDT) |
| **science_open** | **false** (not widened; do not reopen) |
| **Mode** | MEASURE / PARK documentation — **no new training** |
| **Base** | `main` `0a69b3f` (after PR #19 AUDIT-STALK-STABILIZE-V3) |
| **Spec source** | `docs/AUDIT-STALK-STABILIZE-V3.md` §3 |
| **Verdict** | **`PARK_ACCEPT_MEASURE`** — accept **PR #14** hard-Â stalk **0.5/0.5** as best MEASURE corridor; walk away clean |

**One-line claim:** Hard-Â stalk **#14** (0.5·HN+0.5·overall select, 60ep cosine, uniform ID) remains the best stabilize evidence — **MEASURE_STILL** (mean HN **0.957** / K16 **0.863** PASS; seed **2/5**); seed-fragile (#14 2/5, #18 envelope 3/10); V2/V3/ObjV1 **STOP_FRAGILE**; select/curriculum/seed-panel chase **closed**; `science_open=false`.

---

## 1. Why this park (no new train)

| Closed line | PR | Verdict | Why closed |
|-------------|-----|---------|------------|
| Select HN-heavy V2 | #15 `f94f9f7` | **STOP_FRAGILE** | K16 mean 0.423; HN weight → under-propagation |
| Select equal + K16 aux V3 | #16 `8866414` | **STOP_FRAGILE** | K16↑ (0.968) traded HN↓ (0.838); inverted V2 |
| Train curriculum ObjV1 | #17 `4560d24` | **STOP_FRAGILE** | Seed 0/5; HN mean 0.927 FAIL |
| Seed panel freeze #14 | #18 `8850590` | **MEASURE_ENVELOPE** | n=10 rate 3/10; mean HN 0.935 FAIL — seed-fragile |
| Audit V3 → park pointer | #19 `0a69b3f` | **PARK_SELECT_LINE** | Prefer #14; no V4 select mix |

Further select-weight / curriculum / seed-panel chase would re-litigate the same Pareto without new attribution. Honest park > lonely OPEN.

---

## 2. Accepted corridor — PR #14 MEASURE_STILL

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_STABILIZE_MULTI_SEED` |
| **Merge** | PR #14 `61314a0` |
| **Artifact** | `artifacts/stalk_stabilize_multi_seed.json` |
| **Recipe** | hard-Â stalk; 5 seeds; 60 ep; cosine LR 1.5e-3→1.5e-4; joint **0.5·HN+0.5·overall** @ T16 ID-val select; **uniform ID** (no upsample) |
| **Band** | matched-OOD T∈{6,8,12,16} `[45,70]` |
| **Verdict** | **`MEASURE_STILL`** |
| **science_open** | **false** (not widened; no PASS_CANDIDATE) |

### Matched-OOD T16 (cite artifact)

| Metric | mean±std | Floor | Status |
|--------|----------|-------|--------|
| overall | **0.929±0.035** | — | — |
| hard-neg | **0.957±0.061** | ≥0.95 | **PASS** |
| K16 | **0.863±0.143** | ≥0.75 | **PASS** |
| seed PASS | **2/5** | ≥4/5 | **FAIL** (→ MEASURE_STILL, not OPEN) |

Seed-fragility confirmed by #18 envelope under the same frozen recipe: n=10 seed PASS **3/10** (rate 0.30); mean HN **0.935** dips below floor. #14 n=5 means were optimistic; corridor stays MEASURE.

---

## 3. Explicit accept / STOP / closed

| Decision | Status |
|----------|--------|
| Accept #14 0.5/0.5 as best hard-Â stalk MEASURE corridor | **YES** |
| Document seed-fragile (#14 2/5; #18 3/10) | **YES** |
| V2 / V3 / ObjV1 select-or-curriculum lines | **STOP** (do not reopen) |
| Select-weight chase / train upsample / seed-panel harden | **CLOSED** |
| science_open | **false** (do not widen) |
| New training this PR | **NONE** |
| Next active cell | Orthogonal only — ledger priority **6** aux ablation (no stalk select re-chase) |

### STOP reopen (reject as re-chase)

Any new PR that changes HN/K16/overall select weights or train upsample **without** an orthogonal hypothesis → reject. Future PASS path only via orthogonal cell + human seal; never auto-widen.

---

## 4. How far we got (walk-away summary)

| Milestone | Outcome |
|-----------|---------|
| Single-seed Gate1 OPEN (hist.) | Demoted MEASURE (PR #12 / seal §13) — not INVALID |
| Untrained control | Mid scores; bake-in **not** proven |
| #14 harden 0.5/0.5 | Best corridor: mean floors PASS; seed 2/5 |
| V2 / V3 select mixes | STOP — Pareto flip HN↔K16 |
| ObjV1 train curriculum | STOP — 0/5 |
| #18 seed envelope n=10 | MEASURE_ENVELOPE — confirms fragility |
| This park | Accept #14; close select/curriculum; walk away |

---

## 5. Artifacts cited (no new runs)

| Path | Role |
|------|------|
| `artifacts/stalk_stabilize_multi_seed.json` | #14 MEASURE_STILL |
| `artifacts/stalk_stabilize_v2.json` | #15 STOP_FRAGILE |
| `artifacts/stalk_stabilize_v3.json` | #16 STOP_FRAGILE |
| `artifacts/stalk_objective_v1.json` | #17 STOP_FRAGILE |
| `artifacts/stalk_seed_stability.json` | #18 MEASURE_ENVELOPE |
| `docs/AUDIT-STALK-STABILIZE-V3.md` | Park spec |
| `docs/SESSION-SEAL-STALK-LOCALIZATION.md` §14–§19 | Append-only seal trail |
| `docs/LEDGER-OPEN-MEASURE-STOP.md` | Dimensional ledger |

---

## 6. Policy

- Fail-closed. Prefer #14 over V2/V3/ObjV1.
- OOD floors only; no ID overfit as proof.
- Do **not** set `science_open=true`.
- Append-only seals; do not rewrite prior bodies.
