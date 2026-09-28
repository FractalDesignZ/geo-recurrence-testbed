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

## Results (fill after run)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_ensemble_distill.json` |
| **Log** | `artifacts/stalk_ensemble_distill_run.log` |
| **Verdict** | _(pending)_ |
| **science_open** | **false** |
| **Prereg SHA** | _(this commit)_ |
