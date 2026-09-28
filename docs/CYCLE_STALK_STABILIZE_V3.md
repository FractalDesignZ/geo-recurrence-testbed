# CYCLE_STALK_STABILIZE_V3 — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — fail-closed / middle-out continue |
| **science_open** | **false** (always in harness; **not widened**; human seal only) |
| **Trigger** | PR #15 `STOP_FRAGILE` (HN-heavy 0.7 select → K16 collapse); prefer #14 corridor |
| **Base** | `main` tip after PR #15 merge (`f94f9f7`); prereg `0e464b3` |
| **Prior cycles** | #14 `MEASURE_STILL` 2/5; #15 V2 `STOP_FRAGILE` 1/5 |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** (STOP/INVALID path) |

## Goal (middle-out)

Address bottleneck: **ID-val select missing long-hop (K16) signal**. Keep #14
train corridor (60 ep, cosine 1.5e-3→1.5e-4, clip 2.5); **change select primarily**
to equal-weight joint **HN + K16 + overall** with K16 from a selection-only
long-hop aux. **Do not** raise HN weight (>0.5 forbidden after V2). Harness never
self-stamps `science_open=true`. If ≥4/5 → `PASS_CANDIDATE_FOR_OPEN` only.

## Prior evidence (cite)

| Metric @ matched-OOD T16 | #14 (n=5) | #15 V2 (n=5) |
|--------------------------|-----------|--------------|
| overall mean±std | **0.929±0.035** | 0.813±0.157 |
| hard-neg mean±std | **0.957±0.061 PASS** | 0.955±0.101 |
| K16 mean±std | **0.863±0.143 PASS** | **0.423±0.477 FAIL** |
| seed PASS | **2/5** | **1/5 STOP_FRAGILE** |

V2 reading: HN weight 0.7 rescued seed2 HN but collapsed K16 on 0/1/3. Prefer #14 0.5/0.5 corridor; fix select by **adding K16**, not overweighting HN.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_stabilize_v3
# or: reachability-stalk-stabilize-v3
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c` broadcast, no soft ACT |
| Train data | `data/id_2k.jsonl` (unchanged) |
| Eval floors | matched-OOD T∈{6,8,12,16} — **same floors / substrate as #12/#14/#15** |
| Degree-balanced | Secondary table if available |
| Untrained control | Fresh init per seed (cheap; include) |
| **Seeds** | **0, 1, 2, 3, 4** (n=5) |
| **Epochs** | **60** (same as #14; not V2's 90) |
| **LR** | Cosine anneal **1.5e-3 → 1.5e-4** over 60 ep (AdamW, wd=0.01) |
| **Grad clip** | **2.5** |
| **T_train** | 6 |
| **d / mlp** | 64 / ×10 |

### Checkpoint selection rule (prereg — NO floors peeking)

ID `val` has hops ∈ {−1,2..6} only — **no native K16**. V3 supplies K16 via a
**selection-only** aux set (same seq_len band [45,70] as matched-OOD; different
generator seed; runtime overlap-filtered vs floors set):

- Aux path: `data/id_select_longhop.jsonl` (seed 168000; n=240 raw → 239 after filter)
- Generation report: `artifacts/id_select_longhop_generation_report.json`
- **Matched-OOD floors JSONL is never used for selection.**

Each epoch after train step:

1. Eval ID **val** at **T=16** → `HN_id`, `overall_id`.
2. Eval **select_longhop** at **T=16** → `K16_sel` = hop-16 accuracy.
3. `joint = (1/3)·HN_id + (1/3)·K16_sel + (1/3)·overall_id`.
4. **Best ckpt** = argmax lexicographic `(joint, min(HN_id, K16_sel), overall_id, −epoch)`  
   (prefer earlier on ties).
5. **HN weight = 1/3 ≤ 0.5** — no HN overweight (V2 0.7 rejected).
6. Floors on matched-OOD evaluated **only after** train/select complete.

**Rationale:** #14 overall@T16 was an incomplete long-hop proxy; V2 HN-heavy
reintroduced under-propagation. Equal-weight joint with a true K16 select signal
targets the documented bottleneck without peeking at the floors set.

### Prereg floors (match PR #12 / #14 / #15)

| Floor | Threshold |
|-------|-----------|
| Mean hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Mean K16 @ matched-OOD T16 | **≥ 0.75** |
| Seed-wise PASS (HN≥0.95 **and** K16≥0.75) | Goal **≥ 4/5** (stretch **5/5**) |

PASS definition identical to #14/#15.

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| Mean floors PASS **and** ≥4/5 seed PASS | `PASS_CANDIDATE_FOR_OPEN` | stays **false**; flag Fractal-1 / human — do **not** auto-widen |
| Mean floors miss **or** &lt;4/5 but not collapse chaos | `MEASURE_STILL` | **false** |
| Pathological fragility (e.g. ≤1/5 or K16 mean ≪0.5) | `STOP_FRAGILE` | **false** |

**Never** set `science_open=true` from this harness. OPEN revive = human-only after review.

## Harden knobs used (vs #14 / #15)

| Knob | PR #14 | PR #15 V2 | **This V3** |
|------|--------|-----------|-------------|
| Seeds | 0..4 | 0..4 | **0..4** |
| Epochs | 60 | 90 | **60** (#14) |
| LR | 1.5e-3→1.5e-4 | →1.0e-4 | **1.5e-3→1.5e-4** (#14) |
| Grad clip | 2.5 | 2.5 | **2.5** |
| Ckpt select | 0.5·HN+0.5·ov @ ID T16 | 0.7·HN+0.3·ov gated | **(1/3)·HN+(1/3)·K16_sel+(1/3)·ov** |
| K16 in select | no (ID missing) | no | **yes (select-aux)** |
| Architecture | sealed stalk | sealed | **unchanged** |

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_stabilize_v3.json` |
| **Log** | `artifacts/stalk_stabilize_v3_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_stabilize_v3_seed{0..4}_best.pt` |
| **Verdict** | **`STOP_FRAGILE`** (seed 1/5; HN mean 0.838 fails floor; K16 rescued) |
| **science_open** | **false** (not widened) |
| **Elapsed** | ~812 s (~13.5 min CDT) |
| **Individual prereg** | **1/5** seeds PASS |
| **Mean floors** | HN **0.838≥0.95 FAIL**; K16 **0.968≥0.75 PASS** |

### Per-seed matched-OOD T16

| Seed | best_ep | sel HN/K16/ov | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|---------|---------------|---------|----------|----|-----|-----|--------|
| 0 | 27 | 0.960 / 1.000 / 0.980 | 0.798 | 0.629 | 0.900 | 1.000 | **1.000** | FAIL (HN) |
| 1 | 38 | 1.000 / 0.949 / 0.823 | 0.929 | **1.000** | 0.800 | 0.887 | **0.887** | **PASS** |
| 2 | 43 | 0.930 / 1.000 / 0.955 | 0.929 | 0.871 | 0.963 | 1.000 | **1.000** | FAIL (HN) |
| 3 | 14 | 0.965 / 1.000 / 0.910 | 0.840 | 0.775 | 0.775 | 0.963 | **0.975** | FAIL (HN) |
| 4 | 37 | 1.000 / 1.000 / 0.998 | 0.952 | 0.912 | 1.000 | 1.000 | **0.975** | FAIL (HN) |
| **mean±std** | — | — | **0.890±0.067** | **0.838±0.142** | — | — | **0.968±0.046** | **1/5** |

### Untrained control (per seed, T16)

| Seed | u overall | u hard-neg | u K16 | agree vs trained |
|------|-----------|------------|-------|------------------|
| 0 | 0.633 | 0.600 | 1.000 | 0.673 |
| 1 | 0.608 | 0.883 | 0.000 | 0.642 |
| 2 | 0.635 | 0.271 | 1.000 | 0.640 |
| 3 | 0.346 | 0.692 | 0.000 | 0.298 |
| 4 | 0.219 | 0.438 | 0.000 | 0.225 |
| **mean±std** | **0.488±0.194** | 0.577±0.235 | — | **0.495±0.216** |

Untrained remains mid/low vs trained; bake-in still **not** proven. K16 rescue is a **selection** effect (select-aux), not bake-in.

### Degree-balanced T16 (secondary)

| Seed | overall | hard-neg | K16 |
|------|---------|----------|-----|
| 0 | 0.740 | 0.511 | 1.000 |
| 1 | 0.929 | 1.000 | 0.887 |
| 2 | 0.935 | 0.882 | 1.000 |
| 3 | 0.788 | 0.671 | 0.975 |
| 4 | 0.893 | 0.793 | 0.975 |
| **mean±std** | 0.857±0.087 | 0.771±0.192 | 0.967±0.046 |

### Causal horizon mean±std (matched-OOD)

| T | overall | hard-neg | K16 |
|---|---------|----------|-----|
| 6 | 0.674±0.063 | 0.859±0.134 | 0.200±0.440 |
| 8 | 0.704±0.078 | 0.843±0.152 | 0.210±0.442 |
| 12 | 0.851±0.063 | 0.838±0.154 | 0.695±0.388 |
| **16** | **0.890±0.067** | **0.838±0.142** | **0.968±0.046** |

### vs PR #14 / #15 / #12

| Metric @ T16 | PR #12 (n=3) | PR #14 (n=5) | PR #15 V2 (n=5) | **This V3 (n=5)** |
|--------------|--------------|--------------|-----------------|-------------------|
| overall | 0.803±0.202 | **0.929±0.035** | 0.813±0.157 | 0.890±0.067 |
| hard-neg | 0.918±0.142 | **0.957±0.061** | 0.955±0.101 | **0.838±0.142** |
| K16 | 0.654±0.524 | 0.863±0.143 | **0.423±0.477** | **0.968±0.046** |
| seed PASS | 1/3 | **2/5** | 1/5 | **1/5** |

### Reading (fail-closed)

Equal-weight joint **with true K16 select-aux** fully rescued mean K16
(0.423→**0.968**) and tightened K16 std, but **traded HN**: mean HN 0.957→**0.838**
(fails ≥0.95). Seed-wise still **1/5**. V3 inverted V2's failure mode
(V2: HN↑ K16↓; V3: K16↑ HN↓). Select-aux K16 is a real long-hop signal, but
weighting it equally with HN on a small aux pulls ckpts toward propagation at
the expense of hard-neg rejection.

**Prefer #14 joint 0.5/0.5 corridor** as best stabilize evidence (means clear both
floors; 2/5). Do **not** treat V3 as a path to OPEN. Orthogonal next knobs (if any)
should not raise HN weight and should not auto-widen.

Param count **117506** within ±5% of FF 121218 (parity ok all seeds). Select/floors
overlap dropped at runtime: **1** pair.

## Verdict

**`STOP_FRAGILE`** — seed-wise **1/5**; HN mean **0.838** fails floor. K16 mean
**0.968** clears floor (bottleneck partially addressed) but corridor remains fragile.

Stay **MEASURE** / stop this select-aux equal-weight line as a PASS path.
**`science_open=false`**. Do not widen. Retain #14 MEASURE_STILL as best stabilize.

## Policy

- Prefer honesty over lonely OPEN.
- Sheaf unsupervised path is out of scope.
- Append MEASURE/STOP to ledger; `science_open` remains **false**.
- If ≥4/5 clear → verdict `PASS_CANDIDATE_FOR_OPEN` for human only; harness keeps `science_open=false`.
