# CYCLE_STALK_EPISTEMIC_DISAGREEMENT — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — (1) epistemic/aleatoric-style disagreement audit of frozen #14/#18 ensemble vs SWA vs singles; (2) one structured-diversity train (multi-hyp heads + disagreement term; **NOT** soft distill) |
| **science_open** | **false** (always in harness; **not widened** beyond #24/#22 ensemble-at-eval scope) |
| **Trigger** | #22/#24 ens OPEN is inference overlay; #23 soft distill **STOP**; #25 SWA **MEASURE** (not single-model substitute). Ask: does #22 lift **ride on member disagreement**? If so, can structured diversity (multi-hyp) reproduce a durable location without soft KL? |
| **Base** | `main` tip after PR #25 (`b051e15`) |
| **Prior** | #14 MEASURE_STILL; #18 envelope; #20 park; #22/#24 ens OPEN scoped; #23 distill STOP; #25 SWA MEASURE |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

1. **Audit (eval-only):** On matched-OOD T16, measure pairwise prediction disagreement
   and epistemic-vs-aleatoric-style uncertainty decomposition for:
   - frozen #14/#18 hard-Â stalk ensemble members (seeds 0..9) — the #22 map
   - SWA ckpts (seeds 0..4 from #25) as a weight-average pseudo-bag
   - singles baseline (mean of members; epistemic ≈ 0 per model)
   Show whether #22 lift concentrates on high-disagreement examples.
2. **Train (if same PR):** One multi-hypothesis stalk — shared sealed trunk + **H=3**
   linear heads; train CE on each head + **encourage** pairwise JS disagreement
   among head softmaxes. **No** soft KL vs ensemble teacher. Prereg floors on
   head-`prob_mean` @ T16. Seeds **0,1,2**.

## Bound (closed — do not reopen)

| Attempt | Change locus | Outcome |
|---------|--------------|---------|
| #14 | select 0.5 HN + 0.5 overall | **MEASURE_STILL** — freeze select |
| #15/#16/#17 | select / curriculum | **STOP_FRAGILE** — **CLOSED** |
| #18 | seed panel under #14 freeze | **MEASURE_ENVELOPE** |
| #20 | park accept #14 | select/curriculum **CLOSED** |
| #22/#24 | inference `prob_mean` ensemble | **scoped science_open** (map only) |
| #23 | soft KL distill ensemble→student | **STOP** — do **not** chase distill knobs |
| #25 | one-train SWA | **MEASURE** — not single-model OPEN; §22 not widened |

This cycle = **disagreement audit** + **structured multi-hyp diversity** (encourage
disagreement, opposite of soft distill collapse). Not select/upsample re-chase.
Not soft distill. Not widening ensemble OPEN.

## Metaphor

- **map ≠ location** — #22 is a multi-seed *map*; audit asks whether the map's
  *disagreement geometry* carries the lift; multi-hyp asks whether structured
  head diversity can seed a *location* without copying the teacher.
- **negatives = mirror** — hard-neg @ T16 mirrors whether diversity kept the
  reject boundary (same floors as #12/#14/#22).

## Why multi-hyp (not soft distill / not SWA)

| Option | Why not / why |
|--------|---------------|
| Soft distill (#23) | **STOP** — map did not compress under KL |
| SWA (#25) | **MEASURE** — K16↑ HN↓; not ens substitute |
| Snapshot prediction ens of one train | Still multi-forward like #22 |
| **Multi-hyp heads + disagree term (this)** | Structured diversity inside one trunk; **encourage** JS (anti-distill); one forward at eval via head-`prob_mean` |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_epistemic_disagreement
# or: reachability-stalk-epistemic-disagreement
```

### Part A — Audit (eval-only; locked)

| Item | Spec (locked) |
|------|----------------|
| Members (#22 map) | Frozen #14 `stabilize_seed{0..4}_best.pt` + #18 `seed_stability_seed{5..9}_best.pt` |
| SWA bag | #25 `swa_persist_seed{0..4}_swa.pt` (if present; else skip SWA arm with note) |
| Eval | matched-OOD **T16** primary; report T∈{6,8,12,16} summary |
| Pairwise disagreement | mean_{i<j} fraction(argmax_i ≠ argmax_j) over examples |
| Epistemic / aleatoric style | `total = H(mean p)`; `aleatoric = mean_i H(p_i)`; `epistemic = total − aleatoric` (nats, mean over examples) |
| Lift×disagreement | Median-split examples by pairwise disagreement; report ens vs singles_mean accuracy **per bin**; Δ_high − Δ_low |
| Claim test | `#22 lift rides on disagreement` if Δ_high − Δ_low ≥ **0.02** absolute on overall **or** HN **or** K16 strata |

### Part B — Multi-hyp structured diversity train (locked)

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk trunk (local stalk@s / probe@t, **hard A**, discrete T, no `c`, no soft ACT) + **H=3** linear heads replacing single `head` |
| Train recipe | **Frozen #14**: 60 ep; cosine 1.5e-3→1.5e-4; clip 2.5; AdamW wd 0.01; T_train=**6**; uniform ID |
| Loss | `L = mean_h CE(z_h, y) − λ · mean_{i<j} JS(p_i, p_j)` with **λ=0.5** (encourage disagreement; **NOT** soft KL vs teacher) |
| Select | **Frozen #14**: lex `(0.5·HN+0.5·overall, overall, HN, −epoch)` @ **ID-val T16** on **head-prob_mean** — **no OOD peek** |
| Eval primary | head-`prob_mean` (mean softmax over H heads → argmax) |
| Seeds | **0,1,2** (n=3) |
| Eval | matched-OOD T∈{6,8,12,16}; degree-balanced secondary |
| Compare | multi-hyp mean±std vs #14 seed0 vs ens `prob_mean` vs SWA mean (cite) |

### Checkpoint policy

Writes per seed:
- `artifacts/fractal_core_stalk_epistemic_disagreement_seed{i}_best.pt`

### Prereg floors (match PR #12 / #14 / #22) — applied to **multi-hyp mean** @ T16

| Floor | Threshold |
|-------|-----------|
| Multi-hyp mean hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Multi-hyp mean K16 @ matched-OOD T16 | **≥ 0.75** |

### Verdict map (fail-closed)

**Audit verdict** (independent label):

| Outcome | Label |
|---------|-------|
| Δ_high − Δ_low ≥ 0.02 on overall or HN or K16 | `AUDIT_LIFTS_ON_DISAGREEMENT` |
| else | `AUDIT_LIFT_NOT_DISAGREEMENT_DOMINATED` |

**Train verdict** (floors on multi-hyp mean):

| Outcome | Label | science_open |
|---------|-------|--------------|
| Multi-hyp mean clears both floors | `PASS_CANDIDATE` | stays **false**; do **not** widen §22 |
| Floors miss but multi-hyp mean HN **or** K16 ≥ #14 seed0 (re-eval) | `MEASURE` | **false** |
| Floors miss and multi-hyp mean HN **and** K16 both **<** #14 seed0 | `STOP` | **false** |

**Combined cycle verdict** = train verdict (audit is diagnostic). Harness
`science_open=false` always.

**Never** set `science_open=true` from this harness. Select / curriculum remain
**CLOSED**. Ensemble OPEN stays **ensemble-at-eval only** (§22).

## Explicit non-goals

- No soft KL / disagreement distill vs ensemble teacher
- No new select weights / K16-in-select / HN>0.5
- No train upsample / curriculum / weighted CE
- No matched-OOD peek for training or ckpt selection
- No sheaf unsupervised revival
- No `science_open=true` from harness
- No claim that multi-hyp alone upgrades MEASURE → OPEN or widens §22

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_epistemic_disagreement.json` |
| **Log** | `artifacts/stalk_epistemic_disagreement_run.log` |
| **Ckpts** | `fractal_core_stalk_epistemic_disagreement_seed{0,1,2}_best.pt` |
| **Audit verdict** | **`AUDIT_LIFTS_ON_DISAGREEMENT`** |
| **Train verdict** | **`STOP`** |
| **science_open** | **false** (not widened; §22 ensemble scope unchanged) |
| **Elapsed** | ~440 s train+audit (~7.3 min CDT); re-eval ~38 s |
| **Prereg SHA** | `daaeb21` (committed before runs) |
| **Floors (multi-hyp mean)** | HN **0.942 FAIL** (≥0.95); K16 **0.712 FAIL** (≥0.75); seed PASS **0/3** |

### Part A — Matched-OOD T16 disagreement audit

| Arm | pair disagree | epistemic H | aleatoric H | ens ov/HN/K16 | singles mean ov/HN/K16 | Δ ens−sing | rides? |
|-----|---------------|-------------|-------------|---------------|------------------------|------------|--------|
| **#14/#18 ens (n=10)** | **0.162** | **0.192** | 0.080 | **0.996 / 1.000 / 1.000** | 0.903 / 0.935 / 0.792 | +0.093 / +0.065 / +0.208 | **YES** (median) |
| SWA bag (n=5) | 0.162 | 0.157 | 0.054 | 0.952 / 0.938 / 0.975 | 0.893 / 0.867 / 0.910 | +0.059 / +0.071 / +0.065 | **YES** (any↔none; med=0) |
| Multi-hyp within-head (mean) | **0.009** | ~0.000 | ~0.12 | — | — | — | collapsed |

**Lift × disagreement (#14/#18 ens; median split):**

| Bin | n | ens overall | singles mean overall | Δ overall | Δ HN | Δ K16 |
|-----|---|-------------|----------------------|-----------|------|-------|
| High disagreement | 254 | **0.992** | 0.817 | **+0.175** | **+0.164** | **+0.221** |
| Low disagreement | 226 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| **lift_gap (high−low)** | — | — | — | **+0.175** | **+0.164** | **+0.221** |

All gaps ≫ 0.02 → **#22 lift rides on disagreement**. On zero-disagreement examples, singles already perfect; ens adds nothing. On disagreeing examples, ens recovers ~17–22 pts vs singles mean.

Disagreement-by-T (ens): T6 0.305 / T8 0.278 / T12 0.244 / T16 0.162 (falls as horizon opens).

### Part B — Multi-hyp train @ matched-OOD T16

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| Multi-hyp mean±std (n=3) | 0.812±0.149 | **0.942±0.058** | **0.712±0.383** |
| #14 seed0 (re-eval) | **0.975** | **1.000** | **0.863** |
| Ensemble `prob_mean` | **0.996** | **1.000** | **1.000** |
| SWA bag `prob_mean` | 0.952 | 0.938 | 0.975 |
| Δ multi-hyp − #14 seed0 | −0.162 | **−0.058** | **−0.150** |

| Seed | ov / HN / K16 | within-head disagree | prereg |
|------|---------------|----------------------|--------|
| 0 | 0.846 / 0.883 / 0.875 | 0.017 | FAIL (HN) |
| 1 | 0.650 / **1.000** / 0.275 | 0.001 | FAIL (K16) |
| 2 | 0.942 / 0.942 / **0.988** | 0.010 | FAIL (HN) |
| **mean** | **0.812 / 0.942 / 0.712** | **0.009** | **0/3** |

Cross-seed bag of the 3 multi-hyp ckpts (secondary): ov 0.902 / HN 0.988 / K16 0.912 — again lifts via **across-seed** disagreement (pair rate 0.301), not within-head structure.

### Reading (fail-closed)

1. **Audit:** #22 `prob_mean` lift is **disagreement-dominated**. Epistemic-style entropy (0.192) exceeds aleatoric (0.080). Median-split lift gaps +0.16–0.22. SWA bag shows the same any↔none pattern but does not clear HN floors as a bag.
2. **Train:** Multi-hyp H=3 + λ·JS **collapsed** (within-head disagree ≈0.009; train JS→0). Mean misses both floors and loses to #14 seed0 on HN and K16 → **`STOP`**. Structured head diversity did **not** become a durable single-location substitute for the ensemble map.
3. Prefer #14 MEASURE_STILL + #22 ens overlay. Distill STOP; SWA MEASURE; select/curriculum CLOSED; **§22 not widened**. `science_open=false`.
