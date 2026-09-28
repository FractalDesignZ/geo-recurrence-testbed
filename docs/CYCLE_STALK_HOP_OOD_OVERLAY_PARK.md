# CYCLE_STALK_HOP_OOD_OVERLAY_PARK — park accept (science_open=false)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-28 (CDT) |
| **Mode** | **FREEZE / PARK documentation only** — no code, train, or eval |
| **Base** | `main` `a50894d` (after PR #33 merge) |
| **Scope** | Hop-OOD HN **overlay chase only** |
| **Verdict** | **`PARK_HOP_OOD_OVERLAY`** after `LOCAL_SOUND_WALL` |
| **science_open** | **false** for this cycle; no widening; existing scoped §22 ensemble OPEN unchanged |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |

**One-line claim:** PRs #30–#33 establish a hop-OOD HN FAIL_OPEN core that is partly removed by the sound `outdeg(s)==0` hygiene measure, but whose 22-example remainder has no actionable sound local cut. Freeze and park the **overlay chase** at `LOCAL_SOUND_WALL`; do not park the whole stalk corridor or §22.

## 1. Facts sealed (no re-run)

All facts below are carried forward from the merged PRs. This park performs no new evaluation.

| PR | Sealed result | Consequence |
|----|---------------|-------------|
| **#30** | `HN_SHATTER_CONFIRMED` + `FAIL_CLOSED_DOMINANT`; hop-OOD HN has **45 FAIL_OPEN** examples; gate and vote overlays do not repair the shatter | Name the residue `HN_FAIL_OPEN_CORE`; do not treat hop-OOD as OPEN |
| **#31** | The 45-example core is a `STRUCTURAL_CLUSTER`: isolated-`s` / hub-`t` | Characterization only; no learned or BFS gate is promoted |
| **#32** | Sound `outdeg(s)==0` force-unreach is `FO_PARTIAL`: **23/45** killed; HN **0.067→0.304**; matched-OOD Δ=**0** | Retain only as optional documented sound hygiene `MEASURE`, not as `science_open` |
| **#33** | The remaining **22** are `DIFFUSE_MULTI_HOP_LIKE_OK_FC`; verdict `LOCAL_SOUND_WALL`; best sound cut **5/22** | **STOP** further local-sound gates for this FO core |

**Sealed residue:** `HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL`.

## 2. Exact park scope

### Parked

- Further inference-overlay hunting for the hop-OOD HN FAIL_OPEN core.
- Further local-sound gates for the remaining 22 FO examples.
- Any attempt to turn the #30–#33 hop-OOD stress result into an OPEN claim.

### Explicitly not parked

- The whole stalk corridor.
- The matched-OOD ensemble claim in §22.
- The existing preference for #14 + #22 on matched-OOD.

The §22 matched-OOD ensemble `OPEN` is unchanged. This park is not a demotion or widening of §22; it is a stop on one out-of-distribution overlay line.

## 3. Outdeg-zero hygiene boundary

The sound `outdeg(s)==0` condition from #32 may remain documented as an optional inference **MEASURE** hygiene overlay: when its sound premise holds, force-unreach is valid. It is not promoted to `science_open`, does not repair the remaining FO core, and does not establish hop-OOD competence. Matched-OOD collateral remains Δ=0 from the sealed #32 result.

## 4. Explicit STOP

This is an explicit stop on further **local-sound gates** for this FO core.

- Do not chase another local-incidence predicate after `LOCAL_SOUND_WALL`.
- Do not use BFS/out-closure or a reachability oracle as an overlay gate.
- Do not implement the subthreshold 5/22 cut as a new overlay.
- Do not re-run evals, train, or reopen §22 as part of this park.

## 5. Named forks outside this chase

These are named only; none was started here:

1. **Multi-hop competence / path reasoning:** a separately preregistered question about whether the model can learn the multi-hop relation needed by the 22-example remainder, rather than another local gate.
2. **Train/retrain fork:** a separately attributed training experiment, with its own substrate, floors, and human seal; no training claim is made by this park.

Neither fork inherits OPEN status from §22 or from the hop-OOD overlay measurements.

## 6. Close

No code, train, or eval is required for this documentation-only park. The correct status is `science_open=false` for the cycle, with the existing scoped §22 matched-OOD ensemble OPEN unchanged. Prefer #14 + #22 on matched-OOD; leave the hop-OOD HN overlay residue named and frozen.
