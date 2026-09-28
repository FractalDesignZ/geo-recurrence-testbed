# LEDGER — OPEN / MEASURE / STOP (dimensional)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-28 (CDT) |
| **Repo** | https://github.com/FractalDesignZ/geo-recurrence-testbed |
| **Base after PR #18** | `8850590` (merge CYCLE_STALK_SEED_STABILITY MEASURE_ENVELOPE); audit stalk V3 → PARK |
| **Hygiene** | PR #12 demotes stalk §6 OPEN → **MEASURE**; PR #14 MEASURE_STILL 2/5 (best corridor); PR #15/#16/#17 STOP_FRAGILE; PR #18 seed-envelope n=10 MEASURE_ENVELOPE 3/10; **AUDIT V3** parks select-line — prefer #14 0.5/0.5 unchanged |
| **science_open policy** | Fail-closed. Harness never self-stamps `true`. Human seal only. |
| **Purpose** | Single dimensional table of seals/PRs and **clear next cells**. |

Do not invent metrics. Cite artifacts. Prefer truth over prior OPEN.

---

## 1. Dimensional ledger (seals / PRs)

Axes abbreviated: **Attr** = attribution class; **Band** = seq_len / substrate; **Gate** = gate coupling; **Aux** = edge-recon; **Init** = bake-in vs neutral.

| Cell / PR | Cycle | Attr | science_open | Init | Aux | Gate detach | Band / K | Verdict | Artifact / seal |
|-----------|-------|------|--------------|------|-----|-------------|----------|---------|-----------------|
| **PR #1** `8f0b482` | `CYCLE_FRACTAL_CORE_GENESIS` | engineering | **false** | n/a (FractalCore) | n/a | n/a | ID + OOD early | OPEN-candidate eng. only | `docs/SESSION-SEAL-FRACTAL-CORE.md` |
| **PR #2** `b144dac` | `CYCLE_STALK_LOCALIZATION` | was **OPEN** → **MEASURE** (demoted) | was true §6 → **demoted false** | hard-A oracle stalk | n/a | n/a | matched-OOD [45,70] K≤16 | **MEASURE** (hist. single-seed; see PR #12 §13) | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` §6+§13 / `artifacts/fractal_core_stalk_*` |
| **PR #3** `af8e49f` | `CYCLE_SHEAF_INFERENCE` | was OPEN → **INVALID** | was true §6 → **INVALIDATED** | **bake-in** (edge+4 / energy±5) | ON (train) | **True** | [45,70] K≤16 | **INVALID** (learned claim) | `docs/SESSION-SEAL-SHEAF-INFERENCE.md` §6 + §16 |
| **PR #4** `7a02dea` | `CYCLE_SHEAF_RED_TEST` | MEASURE | **false** | bake-in (frozen Gate1) | ON | True | K=20 sparse | MEASURE residue (not OPEN) | sheaf seal §12 / red-test artifacts |
| **PR #5** `a6665bc` | `CYCLE_SHEAF_DENSITY_STRESS` | **INVALID** | **false** | bake-in frozen | ON | True | true ER p=0.15 vs [45,70] | **INVALID** feasibility wall | `docs/SESSION-SEAL-SHEAF-DENSITY-FEASIBILITY.md` |
| **PR #6** `353bd61` | `CYCLE_SHEAF_DENSE_CONTEXT` | was OPEN → **INVALID** | was true §6 → **INVALIDATED** | bake-in frozen Gate1 | ON | True | dense [100,140] K=8 | **INVALIDATED** as learned | `docs/SESSION-SEAL-SHEAF-DENSE-CONTEXT.md` §6 + §8 |
| **PR #7** `034a074` | AUDIT untrained control | **INVALIDATION** | **false** | documents bake-in | — | — | matched-OOD / RED / dense | `SEALS_COMPROMISED_INIT_BAKE_IN` | `docs/AUDIT-SHEAF-UNTRAINED-CONTROL.md` / `artifacts/sheaf_untrained_control_audit.json` |
| **PR #8** `e0877eb` | `CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN` | **MEASURE** | **false** | **neutral** | **ON** | **True** | [45,70] T16 | `MEASURE_CANDIDATE_PASS_FLOORS` | `artifacts/sheaf_neutral_init_retrain.json` |
| **PR #9** `2834256` | `CYCLE_SHEAF_NO_AUX_EDGE_RECON` | **STOP** residue | **false** | **neutral** | **OFF** (w=0) | **True** | [45,70] T16 | `STOP_LEARNING_FAIL` | `artifacts/sheaf_no_aux_edge_recon.json` |
| **PR #10** `af152bd` | `CYCLE_SHEAF_STE_NO_AUX` | **STOP** residue | **false** | **neutral** | **OFF** | **False** (STE) | [45,70] T16; 60 ep | `STOP_LEARNING_FAIL` (unstable **1/3** seeds) | `artifacts/sheaf_ste_no_aux.json` |
| **PR #11** `7f1037a` | Stalk untrained control | was OPEN contingent → superseded | was true §6 → demoted via PR #12 | sealed stalk | n/a | n/a | matched-OOD T∈{6,8,12,16} | `OPEN_STILL_CONTINGENT_NEEDS_MULTI_SEED` (u≈0.63; **not** bake-in) | `artifacts/stalk_untrained_control_audit.json` |
| **PR #12** `69d61a9` | `CYCLE_STALK_MULTI_SEED_RECONFIRM` | **MEASURE** (demoted from OPEN) | **false** (demoted; not widened) | hard-A stalk | n/a | n/a | matched-OOD T∈{6,8,12,16}; ≥3 seeds | `OPEN_CONTINGENT_AT_RISK` → **DEMOTION MEASURE** (1/3; mean HN 0.918 / K16 0.654) | `artifacts/stalk_multi_seed_reconfirm.json` / seal §13 |
| **PR #14** `61314a0` | `CYCLE_STALK_STABILIZE_MULTI_SEED` | **MEASURE** | **false** (not widened) | hard-A stalk harden | n/a | n/a | matched-OOD T∈{6,8,12,16}; **5 seeds**; 60ep cosine | `MEASURE_STILL` (mean HN **0.957** / K16 **0.863** PASS; seed **2/5** &lt;4/5) | `artifacts/stalk_stabilize_multi_seed.json` / seal §14 / `docs/CYCLE_STALK_STABILIZE_MULTI_SEED.md` |
| **PR #15** `f94f9f7` | `CYCLE_STALK_STABILIZE_V2` | **MEASURE** | **false** (not widened) | hard-A stalk V2 | n/a | n/a | matched-OOD T∈{6,8,12,16}; **5 seeds**; 90ep; gated 0.7·HN | `STOP_FRAGILE` (seed **1/5**; K16 mean **0.423**; HN-heavy under-propagation) | `artifacts/stalk_stabilize_v2.json` / seal §15 / `docs/CYCLE_STALK_STABILIZE_V2.md` |
| **PR #16** `8866414` | `CYCLE_STALK_STABILIZE_V3` | **MEASURE** | **false** (not widened) | hard-A stalk V3 | n/a | n/a | matched-OOD T∈{6,8,12,16}; **5 seeds**; 60ep; equal HN+K16+ov select-aux | `STOP_FRAGILE` (seed **1/5**; HN mean **0.838**; K16 **0.968** rescued) | `artifacts/stalk_stabilize_v3.json` / seal §16 / `docs/CYCLE_STALK_STABILIZE_V3.md` |
| **PR #17** `4560d24` | `CYCLE_STALK_OBJECTIVE_V1` | **MEASURE** | **false** (not widened) | hard-A stalk objective | n/a | n/a | matched-OOD T∈{6,8,12,16}; **5 seeds**; 60ep; #14 select + HN/longhop train curriculum | `STOP_FRAGILE` (seed **0/5**; HN mean **0.927**; K16 **0.845**) | `artifacts/stalk_objective_v1.json` / seal §17 / `docs/CYCLE_STALK_OBJECTIVE_V1.md` |
| **PR #18** `8850590` | `CYCLE_STALK_SEED_STABILITY` | **MEASURE** | **false** (not widened) | hard-A stalk #14 freeze | n/a | n/a | matched-OOD T∈{6,8,12,16}; **10 seeds** (0..4 reconfirm + 5..9 new); 60ep; #14 select **frozen** (NO upsample) | `MEASURE_ENVELOPE` (seed **3/10**; HN mean **0.935** FAIL; K16 **0.792** PASS) | `artifacts/stalk_seed_stability.json` / seal §18 / `docs/CYCLE_STALK_SEED_STABILITY.md` |
| **AUDIT** (this PR) | `AUDIT-STALK-STABILIZE-V3` / `CYCLE_STALK_PARK_ACCEPT_MEASURE` | **MEASURE** park | **false** | — | — | — | cite #14/#16/#18 | `PARK_SELECT_LINE` — accept #14; no V4 select | `docs/AUDIT-STALK-STABILIZE-V3.md` |

### 1.1 Status legend

| Label | Meaning |
|-------|---------|
| **OPEN** | Human-sealed scoped science claim still standing |
| **INVALID / INVALIDATED** | Prior OPEN revoked or cell never admissible (feasibility / bake-in) |
| **MEASURE** | Numeric floors may pass; no science_open stamp |
| **STOP** | Prereg / learning floors failed; residue documented |
| **DEMOTION** | Prior OPEN no longer live; append-only status → MEASURE (not bake-in INVALID) |
| **Stalk MEASURE (demoted)** | Former hard-A stalk OPEN (PR #2) — **not INVALID** (untrained mid); **demoted** to MEASURE on multi-seed fragility (PR #12 / seal §13) |

### 1.2 Sheaf learned OPEN status

| Claim | Status |
|-------|--------|
| Sheaf §6 sparse [45,70] K≤16 "**learned** Â" | **INVALID** (PR #7 / seal §16) |
| Dense-context §6 "**learned**" dense band | **INVALIDATED** (dense seal §8) |
| Stalk §6 hard-A locality | **MEASURE** (**demoted**; PR #14 best corridor 2/5; V2/V3/ObjV1 STOP; PR #18 n=10 envelope 3/10; **AUDIT V3 PARK** select-line; **not INVALID**; science_open false) |
| Feasibility true p=0.15 under [45,70] | **INVALID** (PR #5) |
| With-aux neutral retrain floors (PR #8) | **MEASURE** candidate only (`science_open=false`) |
| No-aux detach (PR #9) | **STOP** residue |
| STE no-aux (PR #10) | **STOP** residue (unstable 1/3) |

---

## 2. Clear next cells

| Priority | Cell | Knobs | Pass → | Fail → | Notes |
|----------|------|-------|--------|--------|-------|
| **1 (done)** | `CYCLE_SHEAF_STE_NO_AUX` (PR #10 `af152bd`) | STE + detach=False + no-aux + 60 ep | — | **STOP** (unstable **1/3**) | Artifact `sheaf_ste_no_aux.json`; mean T16 overall 0.766±0.231, hard-neg 0.679±0.278, K16 0.804±0.339 |
| **2 (done)** | Stalk untrained control (PR #11 `7f1037a`) | FractalCore/stalk = sealed OPEN; matched-OOD T∈{6,8,12,16} | — | bake-in would INVALIDATE | **Verdict:** OPEN still contingent (u T16 overall 0.633 / hard-neg 0.600 vs sealed 0.977 / 1.000; agree 0.610). No revoke. |
| **3 (done)** | `CYCLE_STALK_MULTI_SEED_RECONFIRM` (PR #12) | ≥3 seeds; hard-A stalk; T∈{6,8,12,16}; prereg mean HN≥0.95 & K16≥0.75 | — | **DEMOTION MEASURE** | Measurement `OPEN_CONTINGENT_AT_RISK` (1/3); mean HN 0.918±0.142 / K16 0.654±0.524 → **demote** live §6 OPEN to MEASURE (seal §13). Not INVALID. |
| **3b (done)** | `CYCLE_STALK_STABILIZE_MULTI_SEED` (PR #14 `61314a0`) | 5 seeds; 60ep cosine; joint 0.5 ID-val T16; floors + seed ≥4/5 | PASS_CANDIDATE (still science_open=false) | **MEASURE_STILL** | Means PASS (HN 0.957±0.061 / K16 0.863±0.143); seed **2/5**. Prefer this corridor over V2. |
| **3c (done)** | `CYCLE_STALK_STABILIZE_V2` (PR #15 `f94f9f7`) | 90ep; LR→1e-4; gated **0.7·HN+0.3·ov** (ov≥0.85) | PASS_CANDIDATE | **STOP_FRAGILE** | Seed **1/5**; K16 mean **0.423**; HN weight reintroduced under-propagation. Do **not** raise HN weight further. |
| **3d (done)** | `CYCLE_STALK_STABILIZE_V3` (PR #16 `8866414`) | 60ep #14 corridor; equal **(1/3)HN+(1/3)K16_sel+(1/3)ov**; select-aux longhop | PASS_CANDIDATE | **STOP_FRAGILE** | Seed **1/5**; K16 **0.968** rescued; HN mean **0.838** fails. Inverted V2 failure. Prefer #14. |
| **4 (done)** | Accept MEASURE / orthogonal (non-select) pointer | prefer **#14 0.5/0.5** | — | — | Select-weight chase closed (V2/V3 STOP). Prefer #14 corridor. |
| **4b (done)** | `CYCLE_STALK_OBJECTIVE_V1` (PR #17 `4560d24`) | #14 select + HN×2/hop≥5×2 train curriculum | — | **STOP_FRAGILE** 0/5 | HN mean 0.927 FAIL; K16 0.845 PASS; worse seed rate than #14. Prefer #14 MEASURE_STILL. |
| **4c (done)** | `CYCLE_STALK_SEED_STABILITY` (PR #18) | freeze #14; seeds **0..9**; NO new select/upsample | — | **MEASURE_ENVELOPE** 3/10 | HN mean 0.935 FAIL; K16 0.792 PASS; rate 0.30. Prefer #14 unchanged. |
| **5 (done)** | `CYCLE_STALK_PARK_ACCEPT_MEASURE` (AUDIT V3) | park select/curriculum/seed-panel; prefer **#14 0.5/0.5** | — | **PARK** | Audit `docs/AUDIT-STALK-STABILIZE-V3.md`: V3 K16↑ HN↓ STOP; do **not** reopen select mix. |
| **6 (active)** | Aux ablation attribution (orthogonal) | with-aux vs STE-no-aux under matched seeds; **no** stalk select re-chase | document necessity | STOP | Human review before any OPEN revive; `science_open=false` |
| 8 | Neutral-init RED_TEST / dense re-eval | frozen or retrained neutral ckpt | MEASURE | STOP | Do not revive bake-in OPEN |
| 9 | Depth×Density frontier | longer K **and** dense ER under admissible band | MEASURE plan | — | Orthogonal envelopes; `science_open=false` until seal |
| — | Revive sheaf learned OPEN | — | **blocked** | — | Requires human seal after clean attribution |

### 2.1 Explicit non-cells (closed)

- Path-backbone / density-drop to fake [45,70] under true p=0.15 → **INVALID**
- Stamping `science_open=true` from harness → forbidden
- Rewriting invalidated seal bodies → append-only only
- Treating PR #8 with-aux PASS as learned OPEN without human review of aux attribution

---

## 3. Quick citation map

| Need | Path |
|------|------|
| Stalk MEASURE (demoted) | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` §6 hist. + §13 DEMOTION |
| Sheaf INVALIDATION | `docs/SESSION-SEAL-SHEAF-INFERENCE.md` §16 |
| Dense INVALIDATED | `docs/SESSION-SEAL-SHEAF-DENSE-CONTEXT.md` §8 |
| Feasibility INVALID | `docs/SESSION-SEAL-SHEAF-DENSITY-FEASIBILITY.md` |
| MEASURE with-aux | `docs/CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN.md` / PR #8 |
| STOP no-aux | `docs/CYCLE_SHEAF_NO_AUX_EDGE_RECON.md` / PR #9 |
| STE STOP (PR #10) | `docs/CYCLE_SHEAF_STE_NO_AUX.md` / `artifacts/sheaf_ste_no_aux.json` |
| Stalk untrained control | `docs/AUDIT-STALK-UNTRAINED-CONTROL.md` / `artifacts/stalk_untrained_control_audit.json` |
| Stalk multi-seed reconfirm | `docs/CYCLE_STALK_MULTI_SEED_RECONFIRM.md` / `artifacts/stalk_multi_seed_reconfirm.json` |
| Stalk stabilize multi-seed | `docs/CYCLE_STALK_STABILIZE_MULTI_SEED.md` / `artifacts/stalk_stabilize_multi_seed.json` |
| Stalk stabilize V2 | `docs/CYCLE_STALK_STABILIZE_V2.md` / `artifacts/stalk_stabilize_v2.json` |
| Stalk stabilize V3 | `docs/CYCLE_STALK_STABILIZE_V3.md` / `artifacts/stalk_stabilize_v3.json` |
| Stalk objective V1 | `docs/CYCLE_STALK_OBJECTIVE_V1.md` / `artifacts/stalk_objective_v1.json` |
| Stalk seed stability | `docs/CYCLE_STALK_SEED_STABILITY.md` / `artifacts/stalk_seed_stability.json` |
| Stalk V3 audit / PARK | `docs/AUDIT-STALK-STABILIZE-V3.md` |
| This ledger | `docs/LEDGER-OPEN-MEASURE-STOP.md` |

---

## 4. Append rule

After each MEASURE/STOP merge, append one row to §1 and refresh §2 “active” pointer. Do not delete prior rows.
