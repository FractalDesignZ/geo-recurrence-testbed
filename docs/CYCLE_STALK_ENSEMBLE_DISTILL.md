# CYCLE_STALK_ENSEMBLE_DISTILL — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — distill frozen #14/#18 ensemble (`prob_mean` teacher) into **ONE** hard-Â stalk student |
| **science_open** | **false** (always in harness; **not widened**; human seal only) |
| **Trigger** | PR #22 ensemble **PASS_CANDIDATE** (map). Map≠location: compress inference overlay into a single stalk **location**. Negatives=mirror: HN floor mirrors whether the student internalized the ensemble boundary. |
| **Base** | `main` tip after PR #22 (`3972756`) |
| **Prior** | #14 MEASURE_STILL; #18 MEASURE_ENVELOPE; #20 PARK; #22 ensemble PASS_CANDIDATE |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

Freeze teacher = inference ensemble of #14/#18 hard-Â stalk ckpts (primary
**`prob_mean`**). Distill into **one** hard-Â stalk student (same sealed
architecture). **No** new select weights / upsample / curriculum.
Prereg matched-OOD T16 floors HN≥0.95 and K16≥0.75. Multi-seed student
**(0,1,2)** if cheap. Compare **student vs ensemble vs #14 single**.

## Bound (closed — do not reopen)

| Attempt | Change locus | Outcome |
|---------|--------------|---------|
| #14 | select 0.5 HN + 0.5 overall | **MEASURE_STILL** — freeze select |
| #15/#16/#17 | select / curriculum | **STOP_FRAGILE** — **CLOSED** |
| #18 | seed panel under #14 freeze | **MEASURE_ENVELOPE** |
| #20 | park accept #14 | select/curriculum **CLOSED** |
| #22 | inference `prob_mean` ensemble | **PASS_CANDIDATE** (map) |

This cycle = **knowledge distill** of the frozen map into one student location.
Not a select-weight / train-upsample re-chase.

## Metaphor (user)

- **map ≠ location** — ensemble is a multi-seed *map*; student is a single *location*. Distill asks whether the map compresses.
- **negatives = mirror** — hard-neg @ T16 is the *mirror* that checks whether the student kept the ensemble's reject boundary.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_ensemble_distill
# or: reachability-stalk-ensemble-distill
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c` broadcast, no soft ACT |
| Teacher | Frozen #14 seeds 0..4 + #18 seeds 5..9 ckpts; soft target = **mean softmax** (`prob_mean`); **no grad** |
| Student | One FractalCore hard-Â stalk (same hparams as #14) |
| Distill loss | `L = α·CE(hard) + (1−α)·τ²·KL(log_softmax(z_s/τ) ‖ p_teacher)` with **α=0.5**, **τ=2.0** |
| Train T | **6** (same as #14) |
| Soft-label cache | Precompute teacher soft probs on ID **train** @ T=6 once (eval-mode) |
| Select | **Frozen #14**: lex `(0.5·HN+0.5·overall, overall, HN, −epoch)` @ **ID-val T16** — **no OOD peek**; **no new select** |
| Epochs / LR | 60 ep; cosine 1.5e-3→1.5e-4; clip 2.5; AdamW wd 0.01 |
| Student seeds | **0,1,2** (multi-seed if cheap; n=3) |
| Eval | matched-OOD T∈{6,8,12,16}; degree-balanced secondary |
| Compare | student mean±std vs ensemble `prob_mean` (re-eval) vs #14 seed0 single (re-eval) + #14 seeds0..4 mean |

### Checkpoint policy

Teacher ckpts must exist (same paths as #22). Student writes
`artifacts/fractal_core_stalk_ensemble_distill_seed{i}_best.pt`.

### Prereg floors (match PR #12 / #14 / #22) — applied to **student mean** @ T16

| Floor | Threshold |
|-------|-----------|
| Student mean hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Student mean K16 @ matched-OOD T16 | **≥ 0.75** |

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| Student mean clears both floors | `PASS_CANDIDATE` | stays **false**; flag human — do **not** auto-widen |
| Student mean misses ≥1 floor but mean HN **or** K16 ≥ #14 seed0 (same-run) | `MEASURE` | **false** |
| Student mean HN **and** K16 both **<** #14 seed0 (no transfer) | `STOP` | **false** |

**Never** set `science_open=true` from this harness. Select / curriculum remain **CLOSED**.

## Explicit non-goals

- No new select weights / K16-in-select / HN>0.5
- No train upsample / curriculum / weighted CE
- No matched-OOD peek for training or ckpt selection
- No sheaf unsupervised revival
- No `science_open=true` from harness
- No claim that distill alone upgrades MEASURE → OPEN without human seal

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_ensemble_distill.json` |
| **Log** | `artifacts/stalk_ensemble_distill_run.log` |
| **Ckpts** | `fractal_core_stalk_ensemble_distill_seed{0,1,2}_best.pt` |
| **Verdict** | **`STOP`** |
| **science_open** | **false** (not widened) |
| **Elapsed** | ~404 s (~6.7 min CDT) |
| **Prereg SHA** | `6117eb3` (committed before runs) |
| **Floors (student mean)** | HN **0.843 FAIL** (≥0.95); K16 **0.575 FAIL** (≥0.75); seed PASS **0/3** |

### Matched-OOD T16 — student vs ensemble vs #14

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| Student mean±std (n=3) | 0.810±0.050 | **0.843±0.146** | **0.575±0.447** |
| Ensemble `prob_mean` (re-eval) | **0.996** | **1.000** | **1.000** |
| #14 seed0 (re-eval) | **0.975** | **1.000** | **0.863** |
| #14 seeds 0..4 mean | 0.929±0.035 | 0.957±0.061 | 0.863±0.143 |
| Δ student − ensemble | −0.186 | −0.157 | −0.425 |
| Δ student − #14 seed0 | −0.165 | −0.157 | −0.288 |

### Per-seed student matched-OOD T16

| Seed | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|---------|----------|----|-----|-----|--------|
| 0 | 0.754 | 0.817 | 0.975 | 1.000 | **0.100** | FAIL (HN+K16) |
| 1 | 0.823 | **1.000** | 0.675 | 0.625 | 0.638 | FAIL (K16) |
| 2 | 0.852 | 0.713 | 0.988 | 1.000 | **0.988** | FAIL (HN) |
| **mean±std** | **0.810±0.050** | **0.843±0.146** | — | — | **0.575±0.447** | **0/3** |

### Reading (fail-closed)

ID-val select looked strong (seed0 best joint≈0.996) but matched-OOD did **not**
inherit the ensemble map. Seeds trade HN vs K16 (inverted fragility, same class
as V2/V3). Student mean HN **and** K16 both **below** #14 seed0 → verdict
**`STOP`** (no distill transfer). Ensemble remains inference overlay only;
select/curriculum stay **CLOSED**. Prefer honesty: map ≠ location — the map
did not compress under this prereg. `science_open=false`.
