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

## Results (fill after run)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_seed_ensemble.json` |
| **Log** | `artifacts/stalk_seed_ensemble_run.log` |
| **Ckpts** | #14 stabilize 0..4 + #18 seed_stability 5..9 (eval-only if present) |
| **Verdict** | _(pending)_ |
| **science_open** | **false** |
| **Primary** | `prob_mean` |
| **Prereg SHA** | _(commit before runs)_ |
