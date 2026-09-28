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

## Results (filled after run)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_seed_stability.json` |
| **Log** | `artifacts/stalk_seed_stability_run.log` |
| **Ckpts** | #14 `stabilize_seed{0..4}_best.pt` (reconfirm) + `seed_stability_seed{5..9}_best.pt` |
| **Verdict** | _(pending run)_ |
| **science_open** | **false** |
| **Prereg SHA** | _(this commit before runs)_ |
