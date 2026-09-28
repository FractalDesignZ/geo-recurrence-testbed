# CYCLE_STALK_SEED_STABILITY — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — fail-closed / seed-envelope under frozen #14 |
| **science_open** | **false** (always in harness; **not widened**; human seal only) |
| **Trigger** | Select-weight + ID-curriculum easy-$ closed (V2/V3/ObjV1 STOP). Prefer **#14 0.5/0.5** MEASURE corridor. Characterize seed envelope with **more seeds**, **no new select/upsample**. |
| **Base** | `main` tip after PR #17 merge (`4560d24`) |
| **Prior cycles** | #14 `MEASURE_STILL` 2/5 (best); #15/#16/#17 all STOP_FRAGILE |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** (STOP/INVALID path) |

## Goal (projection v3 easy $)

**Freeze #14 recipe exactly.** No new select weights. No train upsample /
curriculum. Expand seed panel to **0..9** (reconfirm 0..4 from #14 ckpts +
train 5..9 under identical harden). Report mean±std, seed PASS rate, and
cheap normal-approx 95% CI on matched-OOD T16. Prefer honesty: document
envelope; **do not** change the #14 corridor.

## Bound (closed)

| Attempt | Change locus | Outcome |
|---------|--------------|---------|
| #14 | select 0.5 HN + 0.5 overall; 60ep cosine; hard-Â | **MEASURE_STILL** 2/5; means PASS — **prefer / freeze** |
| #15 V2 | select HN-heavy 0.7 | **STOP_FRAGILE** K16↓ |
| #16 V3 | select equal HN+K16+ov | **STOP_FRAGILE** HN↓ |
| #17 ObjV1 | #14 select + HN/longhop train upsample | **STOP_FRAGILE** 0/5 HN↓ |

**Do not** reopen select-weight or train-upsample chase. This cycle is
**seed-count / envelope only**.

## Prior evidence (cite)

| Metric @ matched-OOD T16 | #14 (n=5) | #15 V2 | #16 V3 | #17 ObjV1 |
|--------------------------|-----------|--------|--------|-----------|
| overall mean±std | **0.929±0.035** | 0.813±0.157 | 0.890±0.067 | 0.899±0.046 |
| hard-neg mean±std | **0.957±0.061 PASS** | 0.955±0.101 | 0.838±0.142 | 0.927±0.048 |
| K16 mean±std | **0.863±0.143 PASS** | 0.423±0.477 | **0.968±0.046** | 0.845±0.166 |
| seed PASS | **2/5** | 1/5 | 1/5 | 0/5 |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_seed_stability
# or: reachability-stalk-seed-stability
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c` broadcast, no soft ACT |
| Train data | `data/id_2k.jsonl` (unchanged; **uniform** — no upsample) |
| Eval | matched-OOD T∈{6,8,12,16} — **same floors / substrate as #12/#14** |
| Degree-balanced | Secondary table if available |
| Untrained control | Optional (cheap; include by default) |
| **Seeds** | **0..9** (n=10). Seeds **0..4** = reconfirm #14 ckpts (no retrain). Seeds **5..9** = train under identical #14 harden. |
| **Epochs** | **60** (#14 frozen) |
| **LR** | Cosine anneal **1.5e-3 → 1.5e-4** over 60 ep (AdamW, wd=0.01) |
| **Grad clip** | **2.5** |
| **T_train** | 6 |
| **d / mlp** | 64 / ×10 |
| **Select** | **#14 locked**: `joint = 0.5·HN + 0.5·overall` @ T16 ID-val; **NO K16 in select**; **NO HN weight > 0.5** |
| **Train objective** | **uniform ID** (#14; no curriculum / upsample) |

### Checkpoint selection rule (prereg — NO floors peeking; #14 locked)

Identical to `CYCLE_STALK_STABILIZE_MULTI_SEED` / PR #14:

1. Eval ID **val** at **T=16**.
2. `joint = 0.5 * sel_hard_neg + 0.5 * sel_overall`.
3. **Best ckpt** = argmax lexicographic `(joint, sel_overall, sel_hard_neg, −epoch)`.
4. Matched-OOD / degree-balanced / K16 **never** used for selection.

Seeds 0..4: load existing `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt`
and re-eval only (reconfirm). Seeds 5..9: call the same
`train_fractal_id2k_harden` and write
`artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt`.

### Prereg floors (match PR #12 / #14)

| Floor | Threshold |
|-------|-----------|
| Mean hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Mean K16 @ matched-OOD T16 | **≥ 0.75** |
| Seed-wise PASS (HN≥0.95 **and** K16≥0.75) | Goal **≥ 8/10** (same 80% rate as ≥4/5; stretch 10/10) |

PASS definition identical to #14. Report **seed PASS rate** + mean±std.

### Cheap CI (prereg)

Normal-approx 95% CI on means: `mean ± 1.96 * std / sqrt(n)` for overall /
hard-neg / K16 @ T16 (n=10). Optional; does not gate verdict.

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| Mean floors PASS **and** ≥8/10 seed PASS | `PASS_CANDIDATE_FOR_OPEN` | stays **false**; flag Fractal-1 / human — do **not** auto-widen |
| Mean floors PASS **or** miss mildly; seed rate documents variance under frozen #14 | `MEASURE_ENVELOPE` | **false** — prefer #14 unchanged |
| Pathological fragility (e.g. ≤2/10 or K16 mean ≪0.5) | `STOP_FRAGILE` | **false** |

**Never** set `science_open=true` from this harness. OPEN revive = human-only.
**Prefer #14 unchanged** regardless of envelope label — this cycle does not
propose a new select/train recipe.

## Explicit non-goals

- No new select weights / K16-in-select / HN>0.5
- No train upsample / curriculum / weighted CE
- No matched-OOD peek for select or train
- No sheaf unsupervised revival
- No `science_open=true` from harness
- No claim that more seeds alone upgrades MEASURE → OPEN

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_seed_stability.json` |
| **Log** | `artifacts/stalk_seed_stability_run.log` |
| **Ckpts** | #14 `stabilize_seed{0..4}_best.pt` (reconfirm) + `seed_stability_seed{5..9}_best.pt` |
| **Verdict** | **`MEASURE_ENVELOPE`** (seed **3/10**; mean HN **0.935** fails ≥0.95; K16 **0.792** PASS) |
| **science_open** | **false** (not widened) |
| **Elapsed** | ~715 s (~11.9 min CDT) |
| **Individual prereg** | **3/10** seeds PASS (rate **0.30**; goal ≥8/10 **FAIL**) |
| **Mean floors** | HN **0.935≥0.95 FAIL**; K16 **0.792≥0.75 PASS** |
| **CI95 (normal)** | HN [0.879, 0.991]; K16 [0.614, 0.971]; overall [0.870, 0.936] |
| **Prereg SHA** | `59337cb` (committed before runs) |
| **Prefer** | **#14 unchanged** (`MEASURE_STILL` corridor) |

### Per-seed matched-OOD T16

| Seed | mode | best_ep | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|------|---------|---------|----------|----|-----|-----|--------|
| 0 | reconfirm #14 | 58 | **0.975** | **1.000** | 0.988 | 1.000 | **0.863** | **PASS** |
| 1 | reconfirm #14 | 25 | 0.919 | **1.000** | 0.825 | 0.813 | **0.875** | **PASS** |
| 2 | reconfirm #14 | 26 | 0.915 | 0.871 | 0.900 | 1.000 | 0.975 | FAIL (HN) |
| 3 | reconfirm #14 | 43 | 0.885 | **1.000** | 0.750 | 0.938 | 0.625 | FAIL (K16) |
| 4 | reconfirm #14 | 37 | 0.952 | 0.912 | 1.000 | 1.000 | 0.975 | FAIL (HN) |
| 5 | train frozen | 51 | 0.865 | 0.912 | 0.938 | 0.975 | 0.537 | FAIL (HN+K16) |
| 6 | train frozen | 31 | 0.812 | **0.992** | 0.925 | 0.863 | **0.113** | FAIL (K16) |
| 7 | train frozen | 2 | 0.883 | 0.946 | 0.463 | 1.000 | **1.000** | FAIL (HN) |
| 8 | train frozen | 28 | 0.854 | 0.717 | 1.000 | 1.000 | 0.975 | FAIL (HN) |
| 9 | train frozen | 50 | **0.971** | **1.000** | 0.838 | 1.000 | **0.988** | **PASS** |
| **mean±std** | — | — | **0.903±0.053** | **0.935±0.090** | 0.863±0.163 | 0.959±0.068 | **0.792±0.288** | **3/10** |

### Untrained control (per seed, T16)

| Seed | u overall | u hard-neg | u K16 | agree vs trained |
|------|-----------|------------|-------|------------------|
| 0 | 0.633 | 0.600 | 1.000 | 0.608 |
| 1 | 0.608 | 0.883 | 0.000 | 0.627 |
| 2 | 0.635 | 0.271 | 1.000 | 0.625 |
| 3 | 0.346 | 0.692 | 0.000 | 0.460 |
| 4 | 0.219 | 0.438 | 0.000 | 0.225 |
| 5 | 0.231 | 0.129 | 0.000 | 0.329 |
| 6 | 0.508 | 0.350 | 1.000 | 0.346 |
| 7 | 0.615 | 0.562 | 1.000 | 0.698 |
| 8 | 0.562 | 0.458 | 0.000 | 0.613 |
| 9 | 0.540 | 0.412 | 1.000 | 0.565 |
| **mean±std** | **0.490±0.164** | **0.480±0.216** | — | **0.510±0.159** |

Untrained remains mid/low vs trained; bake-in still **not** proven.

### Degree-balanced T16 (secondary)

| Seed | overall | hard-neg | K16 |
|------|---------|----------|-----|
| 0..4 | (same as #14 reconfirm) | | |
| 5 | 0.805 | 0.793 | 0.537 |
| 6 | 0.801 | 0.970 | 0.113 |
| 7 | 0.870 | 0.920 | 1.000 |
| 8 | 0.811 | 0.629 | 0.975 |
| 9 | 0.971 | 1.000 | 0.988 |
| **mean±std (n=10)** | **0.885±0.064** | **0.899±0.126** | **0.792±0.288** |

### Causal horizon mean±std (matched-OOD, n=10)

| T | overall | hard-neg | K16 |
|---|---------|----------|-----|
| 6 | 0.673±0.134 | 0.934±0.089 | 0.309±0.478 |
| 8 | 0.736±0.123 | 0.941±0.090 | 0.319±0.469 |
| 12 | 0.815±0.122 | 0.952±0.079 | 0.479±0.465 |
| **16** | **0.903±0.053** | **0.935±0.090** | **0.792±0.288** |

### vs PR #14 (n=5) under same recipe

| Metric @ T16 | PR #14 (n=5) | **This envelope (n=10)** |
|--------------|--------------|--------------------------|
| overall | **0.929±0.035** | 0.903±0.053 |
| hard-neg | **0.957±0.061 PASS** | **0.935±0.090 FAIL** |
| K16 | **0.863±0.143 PASS** | **0.792±0.288 PASS** |
| seed PASS | **2/5 (0.40)** | **3/10 (0.30)** |
| CI95 HN | — | [0.879, 0.991] |
| CI95 K16 | — | [0.614, 0.971] |

### Reading (fail-closed)

Frozen #14 recipe on a wider seed panel (**0..9**) yields **`MEASURE_ENVELOPE`**:
mean HN dips below floor (0.957→0.935), K16 mean still PASS but std widens
(0.143→0.288; seed6 K16 collapse 0.113), seed PASS rate **0.30** (3/10) vs
goal 0.80. New seeds 5..8 mostly fail; seed9 PASS. Envelope shows #14 n=5
means were optimistic; corridor remains seed-fragile. Prefer honesty: keep
**#14 unchanged** as best MEASURE corridor; do **not** invent new
select/upsample. `science_open=false`.
