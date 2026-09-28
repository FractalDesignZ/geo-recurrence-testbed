# LEDGER — OPEN / MEASURE / STOP (dimensional)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 (CDT) |
| **Repo** | https://github.com/FractalDesignZ/geo-recurrence-testbed |
| **Base after PR #9** | `2834256a12a7a1d9ff826738cd860051f71e19ac` |
| **science_open policy** | Fail-closed. Harness never self-stamps `true`. Human seal only. |
| **Purpose** | Single dimensional table of seals/PRs and **clear next cells**. |

Do not invent metrics. Cite artifacts. Prefer truth over prior OPEN.

---

## 1. Dimensional ledger (seals / PRs)

Axes abbreviated: **Attr** = attribution class; **Band** = seq_len / substrate; **Gate** = gate coupling; **Aux** = edge-recon; **Init** = bake-in vs neutral.

| Cell / PR | Cycle | Attr | science_open | Init | Aux | Gate detach | Band / K | Verdict | Artifact / seal |
|-----------|-------|------|--------------|------|-----|-------------|----------|---------|-----------------|
| **PR #1** `8f0b482` | `CYCLE_FRACTAL_CORE_GENESIS` | engineering | **false** | n/a (FractalCore) | n/a | n/a | ID + OOD early | OPEN-candidate eng. only | `docs/SESSION-SEAL-FRACTAL-CORE.md` |
| **PR #2** `b144dac` | `CYCLE_STALK_LOCALIZATION` | **OPEN** (scoped) | **true** §6 | hard-A oracle stalk | n/a | n/a | matched-OOD [45,70] K≤16 | **OPEN** stalk locality | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` / `artifacts/fractal_core_stalk_*` |
| **PR #3** `af8e49f` | `CYCLE_SHEAF_INFERENCE` | was OPEN → **INVALID** | was true §6 → **INVALIDATED** | **bake-in** (edge+4 / energy±5) | ON (train) | **True** | [45,70] K≤16 | **INVALID** (learned claim) | `docs/SESSION-SEAL-SHEAF-INFERENCE.md` §6 + §16 |
| **PR #4** `7a02dea` | `CYCLE_SHEAF_RED_TEST` | MEASURE | **false** | bake-in (frozen Gate1) | ON | True | K=20 sparse | MEASURE residue (not OPEN) | sheaf seal §12 / red-test artifacts |
| **PR #5** `a6665bc` | `CYCLE_SHEAF_DENSITY_STRESS` | **INVALID** | **false** | bake-in frozen | ON | True | true ER p=0.15 vs [45,70] | **INVALID** feasibility wall | `docs/SESSION-SEAL-SHEAF-DENSITY-FEASIBILITY.md` |
| **PR #6** `353bd61` | `CYCLE_SHEAF_DENSE_CONTEXT` | was OPEN → **INVALID** | was true §6 → **INVALIDATED** | bake-in frozen Gate1 | ON | True | dense [100,140] K=8 | **INVALIDATED** as learned | `docs/SESSION-SEAL-SHEAF-DENSE-CONTEXT.md` §6 + §8 |
| **PR #7** `034a074` | AUDIT untrained control | **INVALIDATION** | **false** | documents bake-in | — | — | matched-OOD / RED / dense | `SEALS_COMPROMISED_INIT_BAKE_IN` | `docs/AUDIT-SHEAF-UNTRAINED-CONTROL.md` / `artifacts/sheaf_untrained_control_audit.json` |
| **PR #8** `e0877eb` | `CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN` | **MEASURE** | **false** | **neutral** | **ON** | **True** | [45,70] T16 | `MEASURE_CANDIDATE_PASS_FLOORS` | `artifacts/sheaf_neutral_init_retrain.json` |
| **PR #9** `2834256` | `CYCLE_SHEAF_NO_AUX_EDGE_RECON` | **STOP** residue | **false** | **neutral** | **OFF** (w=0) | **True** | [45,70] T16 | `STOP_LEARNING_FAIL` | `artifacts/sheaf_no_aux_edge_recon.json` |
| **STE branch** (→PR) | `CYCLE_SHEAF_STE_NO_AUX` | **STOP** residue | **false** | **neutral** | **OFF** | **False** (STE) | [45,70] T16; 60 ep | `STOP_LEARNING_FAIL` (1/3 seeds) | `artifacts/sheaf_ste_no_aux.json` |

### 1.1 Status legend

| Label | Meaning |
|-------|---------|
| **OPEN** | Human-sealed scoped science claim still standing |
| **INVALID / INVALIDATED** | Prior OPEN revoked or cell never admissible (feasibility / bake-in) |
| **MEASURE** | Numeric floors may pass; no science_open stamp |
| **STOP** | Prereg / learning floors failed; residue documented |
| **Stalk OPEN** | Orthogonal hard-A stalk claim (PR #2) — **not** auto-invalidated by sheaf bake-in audit |

### 1.2 Sheaf learned OPEN status

| Claim | Status |
|-------|--------|
| Sheaf §6 sparse [45,70] K≤16 "**learned** Â" | **INVALID** (PR #7 / seal §16) |
| Dense-context §6 "**learned**" dense band | **INVALIDATED** (dense seal §8) |
| Stalk §6 hard-A locality | **OPEN** (standing; orthogonal) |
| Feasibility true p=0.15 under [45,70] | **INVALID** (PR #5) |
| With-aux neutral retrain floors (PR #8) | **MEASURE** candidate only (`science_open=false`) |
| No-aux detach (PR #9) | **STOP** residue |

---

## 2. Clear next cells

| Priority | Cell | Knobs | Pass → | Fail → | Notes |
|----------|------|-------|--------|--------|-------|
| **1 (done)** | `CYCLE_SHEAF_STE_NO_AUX` | STE + detach=False + no-aux + 60 ep | — | **STOP** (1/3 prereg) | See artifact; mean T16 overall 0.766±0.231, hard-neg 0.679±0.278, K16 0.804±0.339 |
| **2 (active)** | Gumbel-Sigmoid sweep | same as STE cell but `gate_mode=gumbel`, temp grid; still no-aux | MEASURE | STOP | STE STOP unstable; optional next |
| 3 | Aux ablation attribution | with-aux vs STE-no-aux under matched seeds | document necessity | — | Human review before any OPEN revive |
| 4 | Neutral-init RED_TEST / dense re-eval | frozen or retrained neutral ckpt | MEASURE | STOP | Do not revive bake-in OPEN |
| 5 | Depth×Density frontier | longer K **and** dense ER under admissible band | MEASURE plan | — | Orthogonal envelopes; `science_open=false` until seal |
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
| Stalk OPEN | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` |
| Sheaf INVALIDATION | `docs/SESSION-SEAL-SHEAF-INFERENCE.md` §16 |
| Dense INVALIDATED | `docs/SESSION-SEAL-SHEAF-DENSE-CONTEXT.md` §8 |
| Feasibility INVALID | `docs/SESSION-SEAL-SHEAF-DENSITY-FEASIBILITY.md` |
| MEASURE with-aux | `docs/CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN.md` / PR #8 |
| STOP no-aux | `docs/CYCLE_SHEAF_NO_AUX_EDGE_RECON.md` / PR #9 |
| STE cycle plan | `docs/CYCLE_SHEAF_STE_NO_AUX.md` |
| This ledger | `docs/LEDGER-OPEN-MEASURE-STOP.md` |

---

## 4. Append rule

After each MEASURE/STOP merge, append one row to §1 and refresh §2 “active” pointer. Do not delete prior rows.
