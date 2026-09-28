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
2. Record `sel_hard_neg` = hard-neg acc (hop −1) and `sel_overall` = overall acc.
3. **Best ckpt** = argmax lexicographic `(sel_hard_neg, sel_overall, −epoch)`  
   i.e. maximize hard-neg@T16-val, then overall@T16-val, then prefer **earlier** epoch on ties.
4. Matched-OOD / degree-balanced / K16 are **never** used for selection — only for final report.

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
| Ckpt select | best ID-val overall @ T_train=6 | **lexicographic hard-neg then overall @ T=16 on ID val** |
| Architecture / hard Â | sealed stalk | **unchanged** |

## Results (filled after run)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_stabilize_multi_seed.json` |
| **Log** | `artifacts/stalk_stabilize_multi_seed_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` |
| **Verdict** | _TBD_ |
| **science_open** | **false** |
| **Individual prereg** | _TBD_ /5 |

### Per-seed matched-OOD T16

| Seed | val_sel HN | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|------------|---------|----------|----|-----|-----|--------|
| _TBD_ | | | | | | | |

### Untrained control (per seed, T16)

_TBD_

### Degree-balanced T16 (secondary)

_TBD_

## Policy

- Prefer honesty over lonely OPEN.
- Sheaf unsupervised path is out of scope.
- Append MEASURE (or STOP_FRAGILE) to ledger; `PASS_CANDIDATE_FOR_OPEN` still leaves `science_open=false`.
