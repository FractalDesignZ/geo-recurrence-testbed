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

## Results (filled after runs)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_objective_v1.json` |
| **Log** | `artifacts/stalk_objective_v1_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_objective_v1_seed{0..4}_best.pt` |
| **Verdict** | _(pending)_ |
| **science_open** | **false** |

## Explicit non-goals

- No HN select weight > 0.5
- No K16 / select-aux in checkpoint selection
- No matched-OOD peek for select or train
- No sheaf unsupervised revival
- No `science_open=true` from harness
