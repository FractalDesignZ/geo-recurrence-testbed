# CYCLE_STALK_SWA_PERSIST — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — SWA (stochastic weight averaging) on **one** #14-corridor hard-Â stalk train → **single** persistent weight vector |
| **science_open** | **false** (always in harness; **not widened** beyond #24/#22 ensemble-at-eval scope) |
| **Trigger** | #22/#24 ensemble OPEN is inference overlay only; #23 soft distill **STOP** (map≠location). Next merit = **single-model persistence** without soft KL. |
| **Base** | `main` tip after PR #24 seal (`ecd0489`) |
| **Prior** | #14 MEASURE_STILL; #18 envelope; #20 park; #22/#24 ens OPEN scoped; #23 distill STOP |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

Train sealed hard-Â stalk under **frozen #14** recipe. During the **second half** of
one train, accumulate a Polyak / SWA average of weights. Deliverable = **one**
SWA checkpoint per seed (true single model — not soft-distill student, not
inference ensemble). Prereg matched-OOD T16 floors HN≥0.95 and K16≥0.75 on
**SWA mean**. Multi-seed **0..4**. Compare SWA vs within-run #14 select vs
ensemble `prob_mean` vs #14 seed0.

## Bound (closed — do not reopen)

| Attempt | Change locus | Outcome |
|---------|--------------|---------|
| #14 | select 0.5 HN + 0.5 overall | **MEASURE_STILL** — freeze select |
| #15/#16/#17 | select / curriculum | **STOP_FRAGILE** — **CLOSED** |
| #18 | seed panel under #14 freeze | **MEASURE_ENVELOPE** |
| #20 | park accept #14 | select/curriculum **CLOSED** |
| #22/#24 | inference `prob_mean` ensemble | **scoped science_open** (map only) |
| #23 | soft KL distill ensemble→student | **STOP** — do **not** chase distill knobs |

This cycle = **weight-space averaging on one #14 train**. Not soft distill. Not
select-weight / train-upsample re-chase. Not widening ensemble OPEN.

## Metaphor

- **map ≠ location** — ensemble remains a multi-seed *map*; SWA asks whether
  trajectory diversity inside *one* train collapses to a durable *location*.
- **negatives = mirror** — hard-neg @ T16 mirrors whether the SWA location kept
  the reject boundary (same floors as #12/#14/#22).

## Why SWA (not soft distill / multi-head)

| Option | Why not / why |
|--------|---------------|
| Soft distill (#23) | **STOP** — map did not compress under KL |
| Disagreement-vs-teacher lightly | Distill-family; ledger: do not chase distill knobs |
| Multi-head / multi-hyp stalk | Architectural widen; defer |
| Snapshot **prediction** ensemble of one train | Still multi-forward at eval (like #22) |
| **SWA / weight average (this)** | One weight vector; no teacher; #14 select frozen as paired baseline |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_swa_persist
# or: reachability-stalk-swa-persist
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c` broadcast, no soft ACT |
| Train recipe | **Frozen #14**: 60 ep; cosine 1.5e-3→1.5e-4; clip 2.5; AdamW wd 0.01; T_train=**6**; uniform ID |
| Select (paired baseline) | **Frozen #14**: lex `(0.5·HN+0.5·overall, overall, HN, −epoch)` @ **ID-val T16** — **no OOD peek** |
| SWA | Polyak average of `state_dict` each epoch with `epoch >= SWA_START`; **SWA_START=31** (second half); no BN update (RMSNorm only) |
| Primary model | **SWA** weights at end of train (floors gate this) |
| Secondary | Within-run #14-select best ckpt (same trajectory) |
| Seeds | **0,1,2,3,4** (n=5) |
| Eval | matched-OOD T∈{6,8,12,16}; degree-balanced secondary |
| Compare | SWA mean±std vs within-run select mean vs ensemble `prob_mean` (re-eval #14/#18) vs #14 seed0 |

### Checkpoint policy

Writes per seed:
- `artifacts/fractal_core_stalk_swa_persist_seed{i}_swa.pt` — **primary**
- `artifacts/fractal_core_stalk_swa_persist_seed{i}_select.pt` — paired #14 select

### Prereg floors (match PR #12 / #14 / #22) — applied to **SWA mean** @ T16

| Floor | Threshold |
|-------|-----------|
| SWA mean hard-neg @ matched-OOD T16 | **≥ 0.95** |
| SWA mean K16 @ matched-OOD T16 | **≥ 0.75** |

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| SWA mean clears both floors | `PASS_CANDIDATE` | stays **false**; flag human — do **not** auto-widen; do **not** widen ensemble scope |
| Floors miss but SWA mean HN **or** K16 ≥ within-run select mean (same seeds) | `MEASURE` | **false** |
| Floors miss and SWA mean HN **and** K16 both **<** within-run select mean | `STOP` | **false** |

**Never** set `science_open=true` from this harness. Select / curriculum remain
**CLOSED**. Ensemble OPEN stays **ensemble-at-eval only** (§22).

## Explicit non-goals

- No soft KL / disagreement distill vs ensemble teacher
- No new select weights / K16-in-select / HN>0.5
- No train upsample / curriculum / weighted CE
- No matched-OOD peek for training or ckpt selection
- No multi-head architectural widen this cycle
- No sheaf unsupervised revival
- No `science_open=true` from harness
- No claim that SWA alone upgrades MEASURE → OPEN or widens §22

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_swa_persist.json` |
| **Log** | `artifacts/stalk_swa_persist_run.log` |
| **Ckpts** | `fractal_core_stalk_swa_persist_seed{i}_{swa,select}.pt` (i=0..4) |
| **Verdict** | **`MEASURE`** |
| **science_open** | **false** (not widened; §22 ensemble scope unchanged) |
| **Elapsed** | ~722 s (~12.0 min CDT) |
| **Prereg SHA** | `6089d31` (committed before runs) |
| **Floors (SWA mean)** | HN **0.867 FAIL** (≥0.95); K16 **0.910 PASS** (≥0.75); seed PASS **2/5** |

### Matched-OOD T16 — SWA vs within-run select vs ensemble vs #14

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| SWA mean±std (n=5) | 0.893±0.050 | **0.867±0.141** | **0.910±0.085** |
| Within-run #14 select mean±std | 0.929±0.035 | **0.957±0.061** | 0.863±0.143 |
| Ensemble `prob_mean` (re-eval) | **0.996** | **1.000** | **1.000** |
| #14 seed0 (re-eval) | **0.975** | **1.000** | **0.863** |
| Δ SWA − select | −0.036 | **−0.090** | **+0.047** |
| Δ SWA − ensemble | −0.103 | −0.133 | −0.090 |

### Per-seed SWA vs select matched-OOD T16

| Seed | SWA ov / HN / K16 | select ov / HN / K16 | SWA prereg |
|------|-------------------|----------------------|------------|
| 0 | 0.840 / 0.758 / 0.813 | 0.975 / **1.000** / 0.863 | FAIL (HN) |
| 1 | 0.940 / **1.000** / 0.863 | 0.919 / **1.000** / 0.875 | **PASS** |
| 2 | 0.838 / 0.688 / **1.000** | 0.915 / 0.871 / 0.975 | FAIL (HN) |
| 3 | 0.927 / **1.000** / 0.875 | 0.885 / **1.000** / 0.625 | **PASS** |
| 4 | 0.923 / 0.888 / **1.000** | 0.952 / 0.913 / 0.975 | FAIL (HN) |
| **mean±std** | **0.893±0.050 / 0.867±0.141 / 0.910±0.085** | **0.929±0.035 / 0.957±0.061 / 0.863±0.143** | **2/5** |

### Reading (fail-closed)

SWA (Polyak from ep31) **raises K16** vs within-run #14 select (+0.047) but
**drops HN** (−0.090) — same HN↔K16 trade class as V2/V3 inverted fragility.
Floors miss on HN → not PASS_CANDIDATE. K16 ≥ select mean → verdict **`MEASURE`**
(partial persist). Ensemble overlay still dominates (HN/K16=1.000). Soft distill
remains STOP; select/curriculum stay CLOSED; §22 ensemble-at-eval scope **not**
widened. Prefer honesty: one-train SWA is not a single-model substitute for the
ensemble map. `science_open=false`.
