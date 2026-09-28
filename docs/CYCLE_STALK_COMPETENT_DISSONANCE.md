# CYCLE_STALK_COMPETENT_DISSONANCE — MEASURE audit (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** competent-dissonance audit of frozen **#14/#18/#22** ens vs **#27** bag |
| **science_open** | **false** (always in harness; **not widened** beyond #24/#22 ensemble-at-eval scope) |
| **Trigger** | #27 bag raised pairwise disagree (0.431 ≥ #22 0.162) but missed HN; #26 showed #22 lift is **disagreement-localized**. Ask: is #22 **competent dissonance** (useful disagreement among accurate members) vs #27 **chaos** (disagreement among weak members)? #22 low *global* disagree ≠ echo chamber if lift rides on disagreement (§24). |
| **Base** | `main` tip after PR #27 (`07401f0`) |
| **Prior** | #14 MEASURE_STILL; #18 envelope; #20 park; #22/#24 ens OPEN scoped; #23 distill STOP; #25 SWA MEASURE; #26 audit LIFTS_ON_DISAGREEMENT + multi-hyp STOP; #27 bag MEASURE_LIFT |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

Eval-only audit on matched-OOD **T16** (and one cheap harder OOD stress) comparing:

1. Frozen **#14/#18** members = **#22** ensemble map (seeds 0..9)
2. Frozen **#27** DGE bag members (seeds 0..4)

**Do NOT train** bag noise / new members. **Do NOT** widen `science_open`.

## Bound (closed — do not reopen)

| Attempt | Change locus | Outcome |
|---------|--------------|---------|
| #14 | select 0.5 HN + 0.5 overall | **MEASURE_STILL** — freeze select |
| #15/#16/#17 | select / curriculum | **STOP_FRAGILE** — **CLOSED** |
| #18 | seed panel under #14 freeze | **MEASURE_ENVELOPE** |
| #20 | park accept #14 | select/curriculum **CLOSED** |
| #22/#24 | inference `prob_mean` ensemble | **scoped science_open** (map only) |
| #23 | soft KL distill | **STOP** |
| #25 | SWA | **MEASURE** |
| #26 | multi-hyp + JS | **STOP** + audit LIFTS_ON_DISAGREEMENT |
| #27 | DGE bag bootstrap+subspace | **MEASURE_LIFT** (disagree↑ HN miss) |

This cycle = **competent-dissonance audit only**. Not train. Not select re-chase. Not soft distill. Not widening §22.

## Metaphor

- **map ≠ location** — #22 is a competent seed-diversity *map*; #27 is an induced-diversity bag. Audit asks which bag has **competent dissonance** (members accurate *and* usefully disagree on hard slices) vs **chaos** (high disagree, weak members) vs **echo risk** (competent but no disagreement geometry).
- **negatives = mirror** — hard-neg disagree / member-acc on disagree-set mirrors whether diversity is protective or noisy.

## Narrative lock (fail-closed)

**#22 low global pairwise disagree ≠ proven echo chamber** when #26 showed lift is **disagreement-localized** (median-split lift_gap ≫ 0.02; zero-disagree examples already perfect for singles). Echo risk requires competence **without** hard-slice / localized disagreement geometry — not merely a moderate global rate.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_competent_dissonance
# or: reachability-stalk-competent-dissonance
```

| Item | Spec (locked) |
|------|----------------|
| Arms | **#22 map**: frozen #14 `stabilize_seed{0..4}` + #18 `seed_stability_seed{5..9}`; **#27 bag**: `bag_diversity_seed{0..4}` |
| Train | **NONE** — eval-only; do not train bag noise |
| Primary eval | matched-OOD **T16** |
| Secondary T | report T∈{6,8,12,16} pair-disagree summary (cheap) |
| Optional stress | **`data/ood_hops.jsonl`** @ T16 — alternate OOD draws (larger n); #22 shatter probe |
| Aggregator cite | ens `prob_mean` metrics cited; audit is member-disagreement geometry |

### Metrics (prereg — locked)

**(a) Global pair disagree** — mean_{i<j} fraction(argmax_i ≠ argmax_j) over all examples @ T16.

**(b) Slice disagree** — same pairwise rate restricted to:
- **hard-neg**: hop == −1
- **K16**: hop == 16
- **easy**: hop == 8 (shortest positive K in matched-OOD; locked label `easy=K8`)

**(c) Member accuracy on agree-set vs disagree-set**
- **agree-set**: examples with pairwise-disagreement-per-example == 0 (all members same argmax)
- **disagree-set**: pairwise-disagreement-per-example > 0
- Report mean-over-members of overall_acc on each set; also ens `prob_mean` acc on each set.

**(d) Competent-dissonance score — FORMULA LOCKED BEFORE RUN**

```
μ_acc     = mean over members of overall_acc @ matched-OOD T16
D_HN      = pairwise disagree rate on hard-neg slice
D_K16     = pairwise disagree rate on K16 slice
D_hard    = 0.5 * (D_HN + D_K16)          # hard-slice disagree
D_SAT     = 0.25                          # saturate excess chaos disagree
D_clip    = min(D_hard, D_SAT) / D_SAT    # ∈ [0, 1]
CD        = μ_acc * D_clip                # competent-dissonance score ∈ [0, 1]
CD_raw    = μ_acc * D_hard                # uncapped diagnostic
```

Rationale: CD rewards **competence × hard-slice disagreement**, but **does not** let over-dispersed chaos inflate the score past D_SAT. Pure echo (D_hard→0) → CD→0; pure chaos (high D, low μ) → low CD via μ.

**(e) Optional OOD stress** — same (a)–(d) on `ood_hops` @ T16 for #22 map (and #27 bag if cheap). Shatter note if #22 ens HN or K16 drops ≥0.10 abs vs matched-OOD, or CD collapses.

### Also report (secondary, from #26 pattern)

- Median-split lift_gap (ens − singles_mean) high vs low disagreement; `rides_on_disagreement` if any gap ≥ 0.02 on overall/HN/K16.

### Verdict map (fail-closed) — per arm; cycle compares arms

Thresholds locked:

| Symbol | Value |
|--------|-------|
| `ACC_FLOOR` | **0.85** (mean member overall @ T16) |
| `D_HARD_MIN` | **0.05** |
| `DIS_SET_ACC_MIN` | **0.65** (mean member overall on disagree-set) |
| `AUDIT_DELTA_EPS` | **0.02** (#26 rides threshold) |

| Label | Rule (all must hold) |
|-------|----------------------|
| **`COMPETENT`** | `μ_acc ≥ ACC_FLOOR` **and** (`D_hard ≥ D_HARD_MIN` **or** `rides_on_disagreement`) **and** mean member acc on disagree-set ≥ `DIS_SET_ACC_MIN` |
| **`ECHO_RISK`** | `μ_acc ≥ ACC_FLOOR` **and** `D_hard < D_HARD_MIN` **and** **not** `rides_on_disagreement` |
| **`CHAOS`** | else (typically `μ_acc < ACC_FLOOR`, or high disagree with weak disagree-set member acc) |

**Cycle comparative verdict** = label(#22) vs label(#27) + which arm has higher CD. Prefer COMPETENT map over CHAOS bag even if bag CD_raw is higher via uncapped D.

**Never** set `science_open=true`. Select/curriculum stay **CLOSED**. §22 scope **unchanged**.

## Explicit non-goals

- No training / bag noise / new members
- No soft distill / multi-hyp / SWA re-chase
- No new select weights / curriculum
- No matched-OOD peek for training (N/A — eval-only)
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen
- No claim that low global disagree alone proves echo chamber

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_competent_dissonance.json` |
| **Log** | `artifacts/stalk_competent_dissonance_run.log` |
| **Verdict (#22 / #27)** | *(filled after run)* |
| **science_open** | **false** (not widened) |
| **Prereg SHA** | *(this commit, before run)* |

*(tables filled after MEASURE run)*
