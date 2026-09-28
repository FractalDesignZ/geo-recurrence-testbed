# CYCLE_STALK_BAG_DIVERSITY — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — DGE-style **independent** hard-Â stalk bag: separate params per member + **data bootstrap** and/or **graph-subspace** diversity; eval `prob_mean` ensemble; report pairwise disagree vs #22 |
| **science_open** | **false** (always in harness; **not widened** beyond #24/#22 ensemble-at-eval scope) |
| **Trigger** | #22/#24 ens OPEN rides on disagreement (§24). Multi-hyp shared-trunk heads **collapsed** (STOP). Soft distill STOP; SWA MEASURE. Next: **true bag** with separate params + induced diversity (not multi-hyp, not soft distill, not SWA-only). |
| **Base** | `main` tip after PR #26 (`f210d0c`) |
| **Prior** | #14 MEASURE_STILL; #18 envelope; #20 park; #22/#24 ens OPEN scoped; #23 distill STOP; #25 SWA MEASURE; #26 audit LIFTS_ON_DISAGREEMENT + multi-hyp STOP |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

Train **M=5** independent sealed hard-Â stalk members (**separate full params**, DGE-style —
not shared-trunk multi-hyp heads). Induce diversity via:

1. **Data bootstrap** — with-replacement resample of ID train (size = |train|), seeded per member.
2. **Graph-subspace** — fixed per-member edge keep-mask on each example (`EDGE_KEEP_P=0.75`);
   train sees a neighborhood subspace; **select/eval use full hard A** (no drop).

Eval matched-OOD with `prob_mean` bag ensemble. Report **pairwise disagreement** vs frozen
#14/#18 (#22 map). Prereg floors on **bag ensemble** @ T16. **Not** soft distill. **Not**
SWA-only. **Not** multi-hyp heads.

## Bound (closed — do not reopen)

| Attempt | Change locus | Outcome |
|---------|--------------|---------|
| #14 | select 0.5 HN + 0.5 overall | **MEASURE_STILL** — freeze select |
| #15/#16/#17 | select / curriculum | **STOP_FRAGILE** — **CLOSED** |
| #18 | seed panel under #14 freeze | **MEASURE_ENVELOPE** |
| #20 | park accept #14 | select/curriculum **CLOSED** |
| #22/#24 | inference `prob_mean` ensemble | **scoped science_open** (map only) |
| #23 | soft KL distill ensemble→student | **STOP** — do **not** chase distill knobs |
| #25 | one-train SWA | **MEASURE** — not ens substitute |
| #26 | multi-hyp heads + JS | **STOP** (heads collapsed) + audit LIFTS_ON_DISAGREEMENT |

This cycle = **independent-param bag + bootstrap/graph-subspace diversity**. Not select
re-chase. Not soft distill. Not SWA-only. Not multi-hyp. Not widening §22.

## Metaphor

- **map ≠ location** — #22 is a seed-diversity *map*; this bag asks whether **induced**
  data/graph subspace diversity with **separate params** yields a map with similar
  (or stronger) disagreement geometry.
- **negatives = mirror** — hard-neg @ T16 mirrors whether the bag kept the reject boundary.

## Why bag diversity (not multi-hyp / soft distill / SWA)

| Option | Why not / why |
|--------|---------------|
| Soft distill (#23) | **STOP** — map did not compress under KL |
| SWA (#25) | **MEASURE** — K16↑ HN↓; not ens substitute |
| Multi-hyp heads (#26) | **STOP** — within-head disagree collapsed (~0.009) |
| Seed-only ens (#22) | Already OPEN scoped; this tests **induced** diversity vs seed-only |
| **DGE-style bag (this)** | Separate params + bootstrap + graph-subspace; eval `prob_mean` |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_bag_diversity
# or: reachability-stalk-bag-diversity
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c`, no soft ACT — **one full model per member** (separate params) |
| Train recipe | **Frozen #14**: 60 ep; cosine 1.5e-3→1.5e-4; clip 2.5; AdamW wd 0.01; T_train=**6**; CE on bootstrap rows |
| Diversity — bootstrap | With-replacement sample of ID **train** only; size=`len(train)`; RNG seed=`member_seed` |
| Diversity — graph-subspace | Per member, per example: keep each off-diag edge with prob **`EDGE_KEEP_P=0.75`** via deterministic hash(`member_seed`, `edge_hash`, `u`, `v`); self always kept; rewrite `encoding` for train rows only |
| Select | **Frozen #14**: lex `(0.5·HN+0.5·overall, overall, HN, −epoch)` @ **ID-val T16** on **full hard A** (no edge drop) — **no OOD peek** |
| Eval primary | Bag `prob_mean` (mean softmax over members → argmax) |
| Secondary | `logit_mean`, `majority_vote`; per-member singles |
| Members | seeds **0,1,2,3,4** (n=5) |
| Eval | matched-OOD T∈{6,8,12,16}; degree-balanced secondary |
| Compare | bag ens vs bag singles mean vs #22 ens (re-eval #14/#18) vs #14 seed0; **pairwise disagree bag vs #22** |

### Checkpoint policy

Writes per member:
- `artifacts/fractal_core_stalk_bag_diversity_seed{i}_best.pt`

### Prereg floors (match PR #12 / #14 / #22) — applied to **bag ensemble `prob_mean`** @ T16

| Floor | Threshold |
|-------|-----------|
| Bag ens hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Bag ens K16 @ matched-OOD T16 | **≥ 0.75** |

Lift reference = **mean of bag singles** on the same matched-OOD T16 eval.

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| Primary clears both floors **and** lifts both HN **and** K16 vs bag singles mean (Δ>0 each) | `PASS_CANDIDATE` | stays **false**; do **not** widen §22 |
| Primary lifts ≥0.01 absolute on at least one of {overall, HN, K16} vs bag singles mean (floors may miss) | `MEASURE_LIFT` | **false** |
| No ≥0.01 lift on overall/HN/K16 vs bag singles mean | `STOP_NO_LIFT` | **false** |

**Disagreement diagnostic** (not a floor gate): report bag pairwise disagree rate vs #22
re-eval pairwise disagree on the same OOD T16. Label note
`DISAGREE_GE_REF` if bag ≥ #22 rate; else `DISAGREE_LT_REF`.

**Never** set `science_open=true` from this harness. Select / curriculum remain
**CLOSED**. Ensemble OPEN stays **ensemble-at-eval only** (§22).

## Explicit non-goals

- No multi-hyp / multi-head shared trunk
- No soft KL / disagreement distill vs ensemble teacher
- No SWA-only primary (SWA may be cited as reference only)
- No new select weights / K16-in-select / HN>0.5
- No train upsample / curriculum / weighted CE
- No matched-OOD peek for training or ckpt selection
- No sheaf unsupervised revival
- No `science_open=true` from harness
- No claim that bag alone upgrades MEASURE → OPEN or widens §22

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_bag_diversity.json` |
| **Log** | `artifacts/stalk_bag_diversity_run.log` |
| **Ckpts** | `fractal_core_stalk_bag_diversity_seed{0..4}_best.pt` |
| **Verdict** | **`MEASURE_LIFT`** |
| **Disagreement diagnostic** | **`DISAGREE_GE_REF`** (bag pair **0.431** ≥ #22 **0.162**) |
| **science_open** | **false** (not widened; §22 ensemble scope unchanged) |
| **Elapsed** | ~767 s (~12.8 min CDT) |
| **Prereg SHA** | `c58ab73` (committed before runs) |
| **Floors (bag ens)** | HN **0.821 FAIL** (≥0.95); K16 **0.788 PASS** (≥0.75); member PASS **0/5** |

### Matched-OOD T16 — bag ens vs bag singles vs #22 vs #14

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| Bag singles mean±std (n=5) | 0.658±0.122 | 0.720±0.192 | 0.538±0.486 |
| **Bag ens `prob_mean` (primary)** | **0.760** | **0.821 FAIL** | **0.788 PASS** |
| Bag `logit_mean` (secondary) | 0.769 | 0.867 | 0.812 |
| Bag `majority_vote` (secondary) | 0.804 | 0.842 | 0.775 |
| #22 ens `prob_mean` (re-eval) | **0.996** | **1.000** | **1.000** |
| #14 seed0 (re-eval) | **0.975** | **1.000** | **0.863** |
| Δ bag ens − bag singles | **+0.102** | **+0.101** | **+0.250** |
| Δ bag ens − #22 | −0.236 | −0.179 | −0.212 |

### Pairwise disagreement @ T16 (vs #22)

| Bag | pair disagree | n members | vs #22 |
|-----|---------------|-----------|--------|
| **DGE bag (this)** | **0.431** | 5 | **≥ #22** → `DISAGREE_GE_REF` |
| #14/#18 ens (#22 map) | **0.162** | 10 | ref |

Induced bootstrap+subspace diversity **increases** disagreement vs seed-only #22, but
accuracy does **not** follow — HN floor miss; far below #22/#14.

### Per-member matched-OOD T16

| Seed | ov / HN / K16 | prereg |
|------|---------------|--------|
| 0 | 0.550 / 0.575 / 0.012 | FAIL |
| 1 | 0.506 / **1.000** / 0.000 | FAIL (K16) |
| 2 | 0.717 / 0.742 / 0.888 | FAIL (HN) |
| 3 | 0.779 / 0.775 / 0.850 | FAIL (HN) |
| 4 | 0.740 / 0.508 / 0.938 | FAIL (HN) |
| **mean** | **0.658 / 0.720 / 0.538** | **0/5** |

### Causal horizon (matched-OOD; bag ens vs singles mean)

| T | ens overall | ens HN | ens K16 | sing overall | sing HN | sing K16 |
|---|-------------|--------|---------|--------------|---------|----------|
| 6 | 0.535 | 0.704 | 0.013 | 0.597 | 0.673 | 0.410 |
| 8 | 0.794 | 0.658 | 0.950 | 0.637 | 0.618 | 0.583 |
| 12 | 0.650 | 0.738 | 0.087 | 0.607 | 0.647 | 0.425 |
| **16** | **0.760** | **0.821** | **0.788** | **0.658** | **0.720** | **0.538** |

### Reading (fail-closed)

1. **Diversity without competence:** Bootstrap + graph-subspace (EDGE_KEEP_P=0.75)
   with **separate params** yields pairwise disagree **0.431 ≫ #22 0.162**, and
   lifts vs own fragile singles (ΔHN/K16 **+0.10 / +0.25**) → **`MEASURE_LIFT`**.
2. **Floors miss:** Bag ens HN **0.821** fails ≥0.95; K16 **0.788** clears ≥0.75.
   Member PASS **0/5**. Far below #22 ens (1.000/1.000) and #14 seed0.
3. Disagreement alone is **not** sufficient — #22's seed-diversity map couples
   disagreement with accurate members; induced bag diversity over-dispersed into
   weak singles. Prefer #14 MEASURE_STILL + #22 ens overlay. Distill STOP; SWA
   MEASURE; multi-hyp STOP; select/curriculum CLOSED; **§22 not widened**.
   `science_open=false`.

