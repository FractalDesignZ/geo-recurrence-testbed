# CYCLE_STALK_SEED_ENSEMBLE — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — inference-time ensemble of frozen #14/#18 hard-Â stalk singles |
| **science_open** | **false** (always in harness; **not widened**; human seal only) |
| **Trigger** | Park accepted #14 MEASURE corridor (PR #20) but **do not assume tops out**. Seed-fragile singles (#14 2/5; #18 3/10) leave an **isomorphic** opportunity: ensemble the same frozen seeds at eval — **no** new select/upsample train. |
| **Base** | `main` tip after PR #21 (`00b844b`) |
| **Prior** | #14 MEASURE_STILL; #18 MEASURE_ENVELOPE; #20 PARK_ACCEPT_MEASURE (select/curriculum **CLOSED**) |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** (STOP/INVALID path) |

## Goal

Freeze hard-Â stalk **singles** from PR #14 (seeds 0..4) and PR #18 (seeds 5..9).
**No new select weights. No train upsample.** Eval matched-OOD T16 with a
**preregistered** ensemble aggregator. Report ensemble overall / HN / K16 vs
**mean of singles**; optional leave-one-out. Verdict whether ensemble **lifts**
the corridor without reopening select chase.

## Bound (closed — do not reopen)

| Attempt | Change locus | Outcome |
|---------|--------------|---------|
| #14 | select 0.5 HN + 0.5 overall | **MEASURE_STILL** 2/5 — **freeze singles** |
| #15/#16/#17 | select / curriculum | **STOP_FRAGILE** — **CLOSED** |
| #18 | seed panel n=10 under #14 freeze | **MEASURE_ENVELOPE** 3/10 — **use ckpts** |
| #20 | park accept #14 | **PARK_ACCEPT_MEASURE** — select/curriculum parked |

This cycle is **inference ensemble only**. Not a select-weight / train-upsample re-chase.

## Prior singles evidence (cite #18)

| Metric @ matched-OOD T16 | Singles mean±std (n=10) | Floor |
|--------------------------|-------------------------|-------|
| overall | 0.903±0.053 | — |
| hard-neg | 0.935±0.090 | ≥0.95 |
| K16 | 0.792±0.288 | ≥0.75 |
| seed PASS | 3/10 | — |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_seed_ensemble
# or: reachability-stalk-seed-ensemble
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c` broadcast, no soft ACT |
| Members | Existing ckpts only — seeds **0..4** = `#14` `fractal_core_stalk_stabilize_seed{i}_best.pt`; seeds **5..9** = `#18` `fractal_core_stalk_seed_stability_seed{i}_best.pt` |
| Train | **NONE** unless a required ckpt is missing (then fill with frozen #14 harden only — same as #18). Expected: all 10 present → **zero train**. |
| Eval | matched-OOD T∈{6,8,12,16} — **same floors / substrate as #12/#14/#18** |
| Degree-balanced | Secondary table |
| **Primary aggregator (prereg)** | **`prob_mean`** — mean of per-member softmax probabilities, then argmax |
| Secondary (report only) | `logit_mean` (mean logits → argmax); `majority_vote` (hard argmax vote; ties → class 0) |
| Leave-one-out | Optional; default **ON** — for each held-out seed, `prob_mean` over the other 9; report mean±std LOO |
| Singles baseline | Re-eval each member alone on same OOD; report mean±std; compare ensemble − singles_mean |

### Checkpoint policy (prereg)

1. Resolve paths for seeds 0..9 as above.
2. If any missing → train **only** missing seeds under frozen #14 harden (`train_fractal_id2k_harden`); do **not** change select/upsample.
3. If all present → **eval-only**.

### Prereg floors (match PR #12 / #14) — applied to **primary ensemble** @ T16

| Floor | Threshold |
|-------|-----------|
| Ensemble hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Ensemble K16 @ matched-OOD T16 | **≥ 0.75** |

Lift reference = **mean of singles** on the same matched-OOD T16 eval (not #14 n=5 alone).

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| Primary clears both floors **and** lifts both HN **and** K16 vs singles mean (Δ>0 each) | `PASS_CANDIDATE` | stays **false**; flag human — do **not** auto-widen |
| Primary lifts ≥0.01 absolute on at least one of {overall, HN, K16} vs singles mean (floors may miss) | `MEASURE_LIFT` | **false** |
| No ≥0.01 lift on overall/HN/K16 vs singles mean | `STOP_NO_LIFT` | **false** |

**Never** set `science_open=true` from this harness. OPEN revive = human-only.
Select / curriculum remain **CLOSED** — ensemble does not reopen them.

## Explicit non-goals

- No new select weights / K16-in-select / HN>0.5
- No train upsample / curriculum / weighted CE (except fill missing seeds under #14 freeze)
- No matched-OOD peek for training or member selection
- No sheaf unsupervised revival
- No `science_open=true` from harness
- No claim that ensemble alone upgrades MEASURE → OPEN without human seal

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_seed_ensemble.json` |
| **Log** | `artifacts/stalk_seed_ensemble_run.log` |
| **Ckpts** | #14 `stabilize_seed{0..4}_best.pt` + #18 `seed_stability_seed{5..9}_best.pt` (**all present**; **0** fill-trains) |
| **Verdict** | **`PASS_CANDIDATE`** |
| **science_open** | **false** (not widened; FLAG human) |
| **Primary** | `prob_mean` |
| **Elapsed** | ~77 s (~1.3 min CDT) — eval-only |
| **Prereg SHA** | `e040270` (committed before runs) |
| **Floors** | HN **1.000≥0.95 PASS**; K16 **1.000≥0.75 PASS** |
| **Lift vs singles mean** | Δoverall **+0.093**; ΔHN **+0.065**; ΔK16 **+0.208** |

### Matched-OOD T16 — ensemble vs singles mean

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| Singles mean±std (n=10, same eval) | 0.903±0.053 | 0.935±0.090 | 0.792±0.288 |
| **Ensemble `prob_mean` (primary)** | **0.996** | **1.000** | **1.000** |
| Ensemble `logit_mean` (secondary) | 0.990 | 1.000 | 0.975 |
| Ensemble `majority_vote` (secondary) | 0.990 | 1.000 | 0.975 |
| LOO `prob_mean` mean±std (hold 1 of 10) | 0.995±0.004 | **1.000±0.000** | **0.991±0.008** |
| Deg-bal `prob_mean` | 0.996 | 1.000 | 1.000 |

### Per-seed singles matched-OOD T16 (re-eval)

| Seed | source | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|--------|---------|----------|----|-----|-----|--------|
| 0 | #14 | **0.975** | **1.000** | 0.988 | 1.000 | **0.863** | **PASS** |
| 1 | #14 | 0.919 | **1.000** | 0.825 | 0.813 | **0.875** | **PASS** |
| 2 | #14 | 0.915 | 0.871 | 0.900 | 1.000 | 0.975 | FAIL (HN) |
| 3 | #14 | 0.885 | **1.000** | 0.750 | 0.938 | 0.625 | FAIL (K16) |
| 4 | #14 | 0.952 | 0.912 | 1.000 | 1.000 | 0.975 | FAIL (HN) |
| 5 | #18 | 0.865 | 0.912 | 0.938 | 0.975 | 0.538 | FAIL (HN+K16) |
| 6 | #18 | 0.812 | **0.992** | 0.925 | 0.863 | **0.112** | FAIL (K16) |
| 7 | #18 | 0.883 | 0.946 | 0.463 | 1.000 | **1.000** | FAIL (HN) |
| 8 | #18 | 0.854 | 0.717 | 1.000 | 1.000 | 0.975 | FAIL (HN) |
| 9 | #18 | **0.971** | **1.000** | 0.838 | 1.000 | **0.988** | **PASS** |
| **mean±std** | — | **0.903±0.053** | **0.935±0.090** | 0.863±0.163 | 0.959±0.068 | **0.792±0.288** | **3/10** |

### Leave-one-out `prob_mean` @ T16

| Held-out | overall | hard-neg | K16 |
|----------|---------|----------|-----|
| 0 | 0.994 | 1.000 | 0.988 |
| 1 | 0.994 | 1.000 | 1.000 |
| 2 | 0.996 | 1.000 | 0.988 |
| 3 | **1.000** | 1.000 | 1.000 |
| 4 | 0.994 | 1.000 | 0.988 |
| 5 | **1.000** | 1.000 | 1.000 |
| 6 | **1.000** | 1.000 | 1.000 |
| 7 | 0.996 | 1.000 | 0.988 |
| 8 | 0.990 | 1.000 | 0.988 |
| 9 | 0.992 | 1.000 | 0.975 |
| **mean±std** | **0.995±0.004** | **1.000±0.000** | **0.991±0.008** |

### Causal horizon (matched-OOD; primary vs singles mean)

| T | ens overall | ens HN | ens K16 | sing overall | sing HN | sing K16 |
|---|-------------|--------|---------|--------------|---------|----------|
| 6 | 0.569 | **1.000** | 0.000 | 0.673 | 0.934 | 0.309 |
| 8 | 0.752 | **1.000** | 0.012 | 0.736 | 0.941 | 0.319 |
| 12 | 0.833 | **1.000** | 0.088 | 0.815 | 0.952 | 0.479 |
| **16** | **0.996** | **1.000** | **1.000** | **0.903** | **0.935** | **0.792** |

At short T, ensemble is HN-conservative (overall dips vs singles at T=6); floors gate **T16** only.

### Reading (fail-closed)

Frozen #14/#18 singles remain seed-fragile (3/10 PASS; mean HN 0.935 FAIL). **Inference
`prob_mean` ensemble** clears both floors (HN/K16 = **1.000**) and lifts vs singles mean
(ΔHN **+0.065**, ΔK16 **+0.208**). Secondary aggregators and LOO agree. This is an
**inference overlay**, not a new select/train recipe — select/curriculum stay **CLOSED**.
Verdict **`PASS_CANDIDATE`**; harness keeps **`science_open=false`** (human seal only).
