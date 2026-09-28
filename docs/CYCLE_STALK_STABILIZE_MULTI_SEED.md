# CYCLE_STALK_STABILIZE_MULTI_SEED — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — fail-closed / middle-out stabilize |
| **science_open** | **false** (always in harness; **not widened**; human seal only) |
| **Trigger** | PR #12 demotion: stalk multi-seed fragile (1/3 PASS; mean HN 0.918 / K16 0.654) |
| **Base** | `main` tip `12cb455` (after PR #12/#13 MEASURE demotion) |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Prior sealed** | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` §6 hist. + §13 DEMOTION |
| **Sheaf unsupervised** | **IGNORE this cycle** (STOP/INVALID path) |

## Goal (middle-out)

Stabilize the **known-good corridor**: supervised/hard `A_ij` + stalk-local FractalCore
diffusion. Pour compute into **multi-seed robustness** (prefer 5 seeds). Do **not** chase
unsupervised sheaf Â. Claim stays **MEASURE** until a human clears prereg for any OPEN
revive — harness never self-stamps `science_open=true`.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_stabilize_multi_seed
# or: reachability-stalk-stabilize-multi-seed
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c` broadcast, no soft ACT |
| Train data | `data/id_2k.jsonl` (unchanged) |
| Eval | matched-OOD T∈{6,8,12,16} — **same floors / same substrate as PR #12** |
| Degree-balanced | Secondary table if available (same construction as #12) |
| Untrained control | Fresh init per seed; compare mean overall ~0.63 baseline |
| **Seeds** | **0, 1, 2, 3, 4** (n=5) |
| **Epochs** | **60** (harden vs prior 30) |
| **LR** | Cosine anneal **1.5e-3 → 1.5e-4** over 60 ep (AdamW, wd=0.01) |
| **Grad clip** | **2.5** (unchanged bound30) |
| **T_train** | 6 (unchanged; discrete unroll) |
| **d / mlp** | 64 / ×10 (unchanged) |

### Checkpoint selection rule (prereg — NO test peeking)

ID `val` has hops ∈ {−1,2..6} only — **no K16 proxy on ID**.

Each epoch after train step:

1. Eval ID **val** at **T=16**.
2. Record `sel_hard_neg` (hop −1) and `sel_overall`.
3. `joint = 0.5 * sel_hard_neg + 0.5 * sel_overall`.
4. **Best ckpt** = argmax lexicographic `(joint, sel_overall, sel_hard_neg, −epoch)`  
   (prefer earlier epoch on ties).
5. Matched-OOD / degree-balanced / K16 are **never** used for selection — only for final report.

**Amendment (structural, before full 5-seed):** HN-primary alone was aborted after a
seed-0 pilot — maximizing HN without overall biases **under-propagation** (HN↑ while
positives collapse). Joint score is the honest reading of “early-stop on hard-neg+path
competence” given no K16 on ID. No matched-OOD floors were used to choose this amendment.
Aborted pilot log: `artifacts/stalk_stabilize_multi_seed_run_hn_primary_aborted.log`.

### Prereg floors (match PR #12)

| Floor | Threshold |
|-------|-----------|
| Mean hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Mean K16 @ matched-OOD T16 | **≥ 0.75** |
| Seed-wise PASS (HN≥0.95 **and** K16≥0.75) | Goal **≥ 4/5** (stretch **5/5**) |

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| Mean floors PASS **and** ≥4/5 seed PASS | `PASS_CANDIDATE_FOR_OPEN` | stays **false**; flag Fractal-1 / human — do **not** auto-widen |
| Mean floors miss **or** &lt;4/5 but not collapse chaos | `MEASURE_STILL` | **false** |
| Pathological fragility (e.g. ≤1/5 or K16 mean ≪0.5) | `STOP_FRAGILE` | **false** |

**Never** set `science_open=true` from this harness. OPEN revive = human-only after review.

## Harden knobs used (vs PR #12 reconfirm)

| Knob | PR #12 | This cycle |
|------|--------|------------|
| Seeds | 0..2 | **0..4** |
| Epochs | 30 | **60** |
| LR | fixed 1.5e-3 | **cosine 1.5e-3→1.5e-4** |
| Grad clip | 2.5 | 2.5 |
| Ckpt select | best ID-val overall @ T_train=6 | **joint 0.5·HN+0.5·overall @ T=16 ID val** (HN-primary aborted structurally) |
| Architecture / hard Â | sealed stalk | **unchanged** |

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_stabilize_multi_seed.json` |
| **Log** | `artifacts/stalk_stabilize_multi_seed_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` |
| **Verdict** | **`MEASURE_STILL`** (mean floors PASS; seed goal 2/5 &lt; 4/5) |
| **science_open** | **false** (not widened) |
| **Elapsed** | ~752 s (~12.5 min CDT) |
| **Individual prereg** | **2/5** seeds PASS |
| **Mean floors** | HN **0.957≥0.95 PASS**; K16 **0.863≥0.75 PASS** |

### Per-seed matched-OOD T16

| Seed | best_ep | sel joint HN/ov | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|---------|-----------------|---------|----------|----|-----|-----|--------|
| 0 | 58 | 0.990 / 0.995 | **0.975** | **1.000** | 0.988 | 1.000 | **0.863** | **PASS** |
| 1 | 25 | 0.990 / 0.888 | 0.919 | **1.000** | 0.825 | 0.813 | **0.875** | **PASS** |
| 2 | 26 | 0.945 / 0.960 | 0.915 | 0.871 | 0.900 | 1.000 | 0.975 | FAIL (HN) |
| 3 | 43 | 0.975 / 0.928 | 0.885 | **1.000** | 0.750 | 0.938 | 0.625 | FAIL (K16) |
| 4 | 37 | 1.000 / 0.998 | 0.952 | 0.913 | 1.000 | 1.000 | 0.975 | FAIL (HN) |
| **mean±std** | — | — | **0.929±0.035** | **0.957±0.061** | 0.893±0.107 | 0.950±0.081 | **0.863±0.143** | **2/5** |

### Untrained control (per seed, T16)

| Seed | u overall | u hard-neg | u K16 | agree vs trained |
|------|-----------|------------|-------|------------------|
| 0 | 0.633 | 0.600 | 1.000 | 0.608 |
| 1 | 0.608 | 0.883 | 0.000 | 0.627 |
| 2 | 0.635 | 0.271 | 1.000 | 0.625 |
| 3 | 0.346 | 0.692 | 0.000 | 0.460 |
| 4 | 0.219 | 0.438 | 0.000 | 0.225 |
| **mean±std** | **0.488±0.194** | 0.577±0.235 | — | **0.509±0.173** |

Untrained remains mid/low (not ≈ trained); bake-in still **not** proven. Seed1 K16 collapse from PR #12 (**0.05 → 0.875**) is fixed under harden.

### Degree-balanced T16 (secondary)

| Seed | overall | hard-neg | K16 |
|------|---------|----------|-----|
| 0 | 0.975 | 1.000 | 0.863 |
| 1 | 0.918 | 1.000 | 0.875 |
| 2 | 0.920 | 0.882 | 0.975 |
| 3 | 0.885 | 1.000 | 0.625 |
| 4 | 0.893 | 0.793 | 0.975 |
| **mean±std** | 0.918±0.035 | 0.935±0.094 | 0.863±0.143 |

### Causal horizon mean±std (matched-OOD)

| T | overall | hard-neg | K16 |
|---|---------|----------|-----|
| 6 | 0.662±0.121 | 0.959±0.064 | 0.202±0.446 |
| 8 | 0.718±0.085 | 0.965±0.049 | 0.210±0.442 |
| 12 | 0.829±0.096 | 0.971±0.040 | 0.360±0.421 |
| **16** | **0.929±0.035** | **0.957±0.061** | **0.863±0.143** |

### vs PR #12 reconfirm + sealed single-seed

| Metric @ T16 | Sealed OPEN (seed0) | PR #12 mean±std (n=3) | This cycle mean±std (n=5) |
|--------------|---------------------|------------------------|---------------------------|
| overall | 0.977 | 0.803±0.202 | **0.929±0.035** |
| hard-neg | 1.000 | 0.918±0.142 | **0.957±0.061** |
| K16 | 0.925 | 0.654±0.524 | **0.863±0.143** |
| seed PASS | 1/1 | 1/3 | **2/5** |

## Verdict

**`MEASURE_STILL`** — mean prereg floors **PASS**; seed-wise goal **FAIL** (2/5 &lt; 4/5).

Harden improved stability vs PR #12 (esp. seed1 K16; tighter std; means clear floors)
but does **not** clear PASS_CANDIDATE_FOR_OPEN. Prefer honesty: stay **MEASURE**;
**do not** widen `science_open`. Flag Fractal-1 only if a future cycle hits ≥4/5 with
means still clear.

Param count **117506** within ±5% of FF 121218 (parity ok all seeds).

## Policy

- Prefer honesty over lonely OPEN.
- Sheaf unsupervised path is out of scope.
- Append MEASURE to ledger; `science_open` remains **false**.
