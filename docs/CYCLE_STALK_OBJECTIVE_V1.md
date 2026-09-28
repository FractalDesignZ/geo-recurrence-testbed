# CYCLE_STALK_OBJECTIVE_V1 — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — fail-closed / middle-out continue |
| **science_open** | **false** (always in harness; **not widened**; human seal only) |
| **Trigger** | Select-weight chase closed: V2 HN-heavy STOP_FRAGILE; V3 K16-joint STOP_FRAGILE (inverted). Prefer **#14 0.5/0.5** MEASURE corridor. |
| **Base** | `main` tip after PR #16 merge (`8866414`) |
| **Prior cycles** | #14 `MEASURE_STILL` 2/5 (best); #15 V2 `STOP_FRAGILE` 1/5; #16 V3 `STOP_FRAGILE` 1/5 |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** (STOP/INVALID path) |

## Goal (easy $)

Move **HN + longer-hop into training** (batch curriculum and/or loss), keep
**#14-style ID-val select** (0.5·HN + 0.5·overall; **NO K16 in select**;
**NO HN select weight > 0.5**). Same hard-Â stalk arch. Do **not** peek
matched-OOD for select. Harness never self-stamps `science_open=true`.

## Bound (closed)

| Attempt | Change locus | Outcome |
|---------|--------------|---------|
| #14 | select 0.5 HN + 0.5 overall | **MEASURE_STILL** 2/5; means PASS — **prefer** |
| #15 V2 | select HN-heavy 0.7 | **STOP_FRAGILE** K16↓ |
| #16 V3 | select equal HN+K16+ov | **STOP_FRAGILE** HN↓ (inverted) |

**Do not** reopen select-weight chase. Objective move is the remaining easy cell.

## Prior evidence (cite)

| Metric @ matched-OOD T16 | #14 (n=5) | #15 V2 | #16 V3 |
|--------------------------|-----------|--------|--------|
| overall mean±std | **0.929±0.035** | 0.813±0.157 | 0.890±0.067 |
| hard-neg mean±std | **0.957±0.061 PASS** | 0.955±0.101 | **0.838±0.142 FAIL** |
| K16 mean±std | **0.863±0.143 PASS** | **0.423±0.477 FAIL** | **0.968±0.046 PASS** |
| seed PASS | **2/5** | 1/5 | 1/5 |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_objective_v1
# or: reachability-stalk-objective-v1
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c` broadcast, no soft ACT |
| Train data | `data/id_2k.jsonl` train split only (ID) |
| Eval floors | matched-OOD T∈{6,8,12,16} — **same floors / substrate as #12/#14/#15/#16** |
| Degree-balanced | Secondary table if available |
| Untrained control | Fresh init per seed (cheap; include) |
| **Seeds** | **0, 1, 2, 3, 4** (n=5) |
| **Epochs** | **60** (#14 corridor) |
| **LR** | Cosine anneal **1.5e-3 → 1.5e-4** over 60 ep (AdamW, wd=0.01) |
| **Grad clip** | **2.5** |
| **T_train** | 6 |
| **d / mlp** | 64 / ×10 |

### Training objective (prereg — the only intentional change)

ID train hops ∈ {−1,2..6}. Curriculum + loss (ID only):

| Band | Rule | Sample weight |
|------|------|---------------|
| Hard-neg | `hop_distance == -1` | **2.0** |
| Longer-hop ID | `hop_distance >= 5` (hops 5,6) | **2.0** |
| Mid positives | hops 2,3,4 | **1.0** |

Each epoch:

1. Draw `n_draw = len(train)` indices via **multinomial** with replacement under the weights above (same #steps/epoch as #14).
2. Train with **mean-normalized sample-weighted CE** using the same per-example weights.
3. No matched-OOD rows in train. No select-aux longhop in select (V3 path closed for select).

### Checkpoint selection rule (prereg — NO floors peeking; #14 locked)

ID `val` has hops ∈ {−1,2..6} only — **no K16 on ID**.

Each epoch after train step:

1. Eval ID **val** at **T=16**.
2. Record `sel_hard_neg` (hop −1) and `sel_overall`.
3. `joint = 0.5 * sel_hard_neg + 0.5 * sel_overall`.
4. **Best ckpt** = argmax lexicographic `(joint, sel_overall, sel_hard_neg, −epoch)`  
   (prefer earlier epoch on ties).
5. **HN select weight = 0.5** — not raised (V2 rejected). **No K16 in select** (V3 rejected).
6. Matched-OOD / degree-balanced / K16 are **never** used for selection — only for final report.

### Prereg floors (match PR #12 / #14 / #15 / #16)

| Floor | Threshold |
|-------|-----------|
| Mean hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Mean K16 @ matched-OOD T16 | **≥ 0.75** |
| Seed-wise PASS (HN≥0.95 **and** K16≥0.75) | Goal **≥ 4/5** (stretch **5/5**) |

PASS definition identical to #14/#15/#16.

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| Mean floors PASS **and** ≥4/5 seed PASS | `PASS_CANDIDATE_FOR_OPEN` | stays **false**; flag Fractal-1 / human — do **not** auto-widen |
| Mean floors miss **or** &lt;4/5 but not collapse chaos | `MEASURE_STILL` | **false** |
| Pathological fragility (e.g. ≤1/5 or K16 mean ≪0.5) | `STOP_FRAGILE` | **false** |

**Never** set `science_open=true` from this harness. OPEN revive = human-only after review.

## Harden knobs used (vs #14 / #15 / #16)

| Knob | PR #14 | PR #15 V2 | PR #16 V3 | **This OBJECTIVE_V1** |
|------|--------|-----------|-----------|------------------------|
| Seeds | 0..4 | 0..4 | 0..4 | **0..4** |
| Epochs | 60 | 90 | 60 | **60** (#14) |
| LR | 1.5e-3→1.5e-4 | →1.0e-4 | 1.5e-3→1.5e-4 | **1.5e-3→1.5e-4** (#14) |
| Grad clip | 2.5 | 2.5 | 2.5 | **2.5** |
| Ckpt select | 0.5·HN+0.5·ov | 0.7·HN gated | (1/3) HN+K16+ov | **0.5·HN+0.5·ov (#14 locked)** |
| K16 in select | no | no | yes (aux) | **no** |
| Train objective | uniform ID | uniform ID | uniform ID | **HN×2 + hop≥5×2 curriculum + weighted CE** |
| Architecture | sealed stalk | sealed | sealed | **unchanged** |

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_objective_v1.json` |
| **Log** | `artifacts/stalk_objective_v1_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_objective_v1_seed{0..4}_best.pt` |
| **Verdict** | **`STOP_FRAGILE`** (seed **0/5**; mean HN **0.927** fails ≥0.95; K16 **0.845** PASS) |
| **science_open** | **false** (not widened) |
| **Elapsed** | ~746 s (~12.4 min CDT) |
| **Individual prereg** | **0/5** seeds PASS |
| **Mean floors** | HN **0.927≥0.95 FAIL**; K16 **0.845≥0.75 PASS** |
| **Prereg SHA** | `68a147a` (committed before runs) |

### Per-seed matched-OOD T16

| Seed | best_ep | sel HN/ov | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|---------|-----------|---------|----------|----|-----|-----|--------|
| 0 | 32 | 0.945 / 0.965 | **0.954** | 0.912 | 0.988 | 1.000 | **1.000** | FAIL (HN) |
| 1 | 11 | 0.970 / 0.917 | 0.835 | **1.000** | 0.650 | 0.725 | 0.637 | FAIL (K16) |
| 2 | 29 | 0.970 / 0.980 | 0.904 | 0.938 | 0.900 | 1.000 | 0.713 | FAIL (HN+K16) |
| 3 | 16 | 0.960 / 0.938 | 0.925 | 0.917 | 0.887 | 0.912 | **1.000** | FAIL (HN) |
| 4 | 25 | 0.975 / 0.895 | 0.875 | 0.867 | 0.787 | 0.988 | 0.875 | FAIL (HN) |
| **mean±std** | — | — | **0.899±0.046** | **0.927±0.048** | 0.842±0.129 | 0.925±0.118 | **0.845±0.166** | **0/5** |

### Untrained control (per seed, T16)

| Seed | u overall | u hard-neg | u K16 | agree vs trained |
|------|-----------|------------|-------|------------------|
| 0 | 0.633 | 0.600 | 1.000 | 0.650 |
| 1 | 0.608 | 0.883 | 0.000 | 0.681 |
| 2 | 0.635 | 0.271 | 1.000 | 0.577 |
| 3 | 0.346 | 0.692 | 0.000 | 0.346 |
| 4 | 0.219 | 0.438 | 0.000 | 0.302 |
| **mean±std** | **0.488±0.194** | 0.577±0.235 | — | **0.511±0.176** |

Untrained remains mid/low vs trained; bake-in still **not** proven.

### Degree-balanced T16 (secondary)

| Seed | overall | hard-neg | K16 |
|------|---------|----------|-----|
| 0 | 0.895 | 0.793 | 1.000 |
| 1 | 0.834 | 1.000 | 0.637 |
| 2 | 0.870 | 0.869 | 0.713 |
| 3 | 0.895 | 0.857 | 1.000 |
| 4 | 0.839 | 0.793 | 0.875 |
| **mean±std** | 0.867±0.029 | 0.862±0.085 | 0.845±0.166 |

### Causal horizon mean±std (matched-OOD)

| T | overall | hard-neg | K16 |
|---|---------|----------|-----|
| 6 | 0.647±0.095 | 0.890±0.088 | 0.205±0.438 |
| 8 | 0.696±0.135 | 0.878±0.132 | 0.265±0.418 |
| 12 | 0.860±0.089 | 0.922±0.053 | 0.700±0.378 |
| **16** | **0.899±0.046** | **0.927±0.048** | **0.845±0.166** |

### vs PR #14 / #15 / #16

| Metric @ T16 | PR #14 (n=5) | PR #15 V2 | PR #16 V3 | **This OBJECTIVE_V1** |
|--------------|--------------|-----------|-----------|------------------------|
| overall | **0.929±0.035** | 0.813±0.157 | 0.890±0.067 | 0.899±0.046 |
| hard-neg | **0.957±0.061 PASS** | 0.955±0.101 | 0.838±0.142 | **0.927±0.048 FAIL** |
| K16 | **0.863±0.143 PASS** | 0.423±0.477 | **0.968±0.046** | **0.845±0.166 PASS** |
| seed PASS | **2/5** | 1/5 | 1/5 | **0/5** |

### Reading (fail-closed)

ID-only HN×2 + hop≥5×2 curriculum + weighted CE with **#14 select locked** did
**not** clear seed goal (0/5) and **missed** mean HN floor (0.927 &lt; 0.95).
K16 mean still PASS but seed-wise fragile. vs #14: HN↓, K16≈, seed PASS 2/5→0/5.
Prefer honesty: **STOP_FRAGILE** this objective upsample line; retain **#14**
`MEASURE_STILL` as best stabilize evidence. Select-weight chase remains closed;
this train-curriculum easy-$ also fails. `science_open=false`.

## Explicit non-goals

- No HN select weight > 0.5
- No K16 / select-aux in checkpoint selection
- No matched-OOD peek for select or train
- No sheaf unsupervised revival
- No `science_open=true` from harness
