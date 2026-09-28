# AUDIT — Stalk-local FractalCore untrained control (fail-closed)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 (CDT) |
| **Base after PR #10** | `af152bdcfe77a788e01764e07d193bf59fa03066` |
| **Verdict** | **`OPEN_STILL_CONTINGENT_NEEDS_MULTI_SEED`** |
| **seals_invalidated** | **false** |
| **Artifact** | `artifacts/stalk_untrained_control_audit.json` |
| **Harness** | `python -m reachability_gen.run_stalk_untrained_control` |
| **Seal note** | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` §11 |

**First sentence:** Untrained stalk-local FractalCore scores **mid (~0.63 overall / 0.60 hard-neg)** on matched-OOD T16 vs sealed Gate1 **0.977 / 1.000** (agreement 0.610) — **bake-in NOT proven**; stalk §6 OPEN stays standing but **contingent** (needs multi-seed trained reconfirm). Do not silently widen `science_open`; only revoke if bake-in proven.

## Tables (matched-OOD)

| T | Untrained overall | Sealed overall | Untrained hard-neg | Sealed hard-neg | K8 u/t | K12 u/t | K16 u/t | Agree |
|---|-------------------|----------------|--------------------|-----------------|--------|---------|---------|-------|
| 6 | 0.633 | 0.502 | 0.600 | 1.000 | 1.000 / 0.000 | 0.000 / 0.0125 | 1.000 / 0.000 | 0.465 |
| 8 | 0.700 | 0.675 | 0.400 | 1.000 | 1.000 / 1.000 | 1.000 / 0.050 | 1.000 / 0.000 | 0.375 |
| 12 | 0.638 | 0.833 | 0.608 | 1.000 | 1.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 | 0.471 |
| **16** | **0.633** | **0.977** | **0.600** | **1.000** | 1.000 / 0.938 | 0.000 / 1.000 | 1.000 / 0.925 | **0.610** |

## Policy

- Orthogonal to sheaf bake-in audit (PR #7). Stalk OPEN is **not** auto-invalidated by sheaf INVALIDATION.
- Revoke stalk OPEN **only** if untrained ≈ sealed ≈ 1.0 with ~100% agreement (bake-in proven).
- Otherwise document contingency; keep `science_open` fail-closed outside the existing scoped seal.
