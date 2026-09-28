# CYCLE_STALK_RED_COMPETENT_DISSONANCE — MEASURE audit (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** RED stress of frozen **#14/#18/#22** ens after #28 COMPETENT |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged) |
| **Trigger** | Gemini + Fractal-1 agreement on #22 after #28: #22 is COMPETENT on matched-OOD (D_HN/D_K16/CD locked). Next: strip easy mass, push held-out OOD **p/K outside train priors** (harder than `ood_hops` if feasible), classify **FAIL_OPEN** vs **FAIL_CLOSED**. |
| **Base** | `main` tip after PR #28 (`0a5890a`) |
| **Prior** | #28 COMPETENT_vs_CHAOS; #22 ens OPEN scoped; #27 bag CHAOS (do **not** train); select/curriculum CLOSED |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

1. **Re-report / extend** boundary disagree table stripping easy mass (hard-neg, K16) for #22 overlay — cite #28 numbers + optional matched-OOD refresh.
2. **New held-out RED OOD** pushing **p and/or K outside train priors**, harder than `ood_hops` when feasible.
3. Run frozen **#14+#22** ens only (no #27 bag).
4. On RED: global D, D_hard, CD, agree-set vs disagree-set ens accuracy; classify **FAIL_OPEN** vs **FAIL_CLOSED**.
5. `science_open` **not** widened. No bag train.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open |
| #23/#25/#26 | distill STOP; SWA MEASURE; multi-hyp STOP |
| #27 | bag MEASURE_LIFT / CHAOS — **do not train more bag noise** |
| #28 | COMPETENT_vs_CHAOS on matched-OOD |

This cycle = **RED competent-dissonance audit only**. Not train. Not §22 widen.

## Metaphor

- **map ≠ location** — #22 COMPETENT map on matched-OOD; RED asks whether that map **fails open** (unified confident wrong) or **fails closed** (high D / uncertainty) outside train priors.
- **negatives = mirror** — hard-neg D and disagree-set ens acc mirror protective vs brittle diversity.

## Cite #28 (matched-OOD T16 — #22 ens; strip easy)

Locked from `artifacts/stalk_competent_dissonance.json` / PR #28:

| Slice | Pairwise D | Role |
|-------|------------|------|
| hard-neg | **0.112** | hard mass |
| K16 | **0.337** | hard mass |
| easy K8 | 0.225 | stripped from D_hard |
| **D_hard** | **0.225** = 0.5·(D_HN+D_K16) | |
| **CD** | **0.811** | COMPETENT |
| global pair | 0.162 | ≠ echo (rides YES) |

Refresh on matched-OOD (same ckpts) must match within float noise; do not invent.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.gen_stalk_red_competent_dissonance
python -m reachability_gen.run_stalk_red_competent_dissonance
# or: reachability-gen-stalk-red-cd / reachability-stalk-red-competent-dissonance
```

| Item | Spec (locked) |
|------|----------------|
| Arms | Frozen **#14/#18** = **#22** ens map seeds **0..9** only |
| Train | **NONE** — no #27 bag train / no new members |
| Primary cite | matched-OOD boundary table (#28 + refresh) |
| RED eval | `data/stalk_red_competent_dissonance.jsonl` @ **T=16** |
| Aggregator | ens `prob_mean` |

### RED substrate (locked)

Push **outside train priors** (train: K≤6, ID p∈{0.08..0.35} on short hops):

| Cell | Hop K | Construction | p / n intent | Why harder than `ood_hops` |
|------|-------|--------------|--------------|---------------------------|
| **DENSE_K16** | 16 | path-backbone + distractors under token≤250 | empirical p **≫** ood_hops K16 (≤0.025); target p_emp ≥ **0.04** mean | denser long-hop than sparse ER ood_hops |
| **LONG_K20** | 20 | path-backbone + ≤few distractors; token≤250 | K=**20** ∉ train and ∉ ADR OOD_HOP_VALUES {8,12,16} | longer than ood_hops max K=16; T=16 under-horizon |

Quotas: ≥48 pos/cell (default **64**) + matching hard-neg for 50/50. Seed **210_000**. Not matched-OOD band — dilated tokens allowed (RED substrate, fail-closed vs §22).

### Metrics (prereg — locked)

**(0) Boundary refresh** — re-eval #22 ens on matched-OOD T16; report D_HN, D_K16, D_easy, D_hard, CD; assert close to #28 cite.

**(a) Global pair disagree** on RED @ T16.

**(b) Slice disagree** — hard-neg; K16; K20; easy absent on RED (N/A).

```
D_long  = 0.5 * (D_K16 + D_K20)     # both cells present
D_hard  = 0.5 * (D_HN + D_long)
CD      = μ_acc * min(D_hard, 0.25) / 0.25   # same #28 formula family
```

**(c) Agree-set vs disagree-set** — ens `prob_mean` overall (/HN/K16/K20 when defined) on each set; member mean overall.

**(d) FAIL_OPEN vs FAIL_CLOSED** on ens-**wrong** examples (locked):

| Symbol | Value |
|--------|-------|
| `CONF_THRESH` | **0.80** (ens max softmax) |
| `D_CLOSED_MIN` | **0.10** (per-example pairwise disagree) |
| `EPI_CLOSED_MIN` | **0.15** nats (per-example epistemic entropy) |

| Label | Rule on an ens-wrong example |
|-------|------------------------------|
| **`FAIL_OPEN`** | pairwise D_ex **== 0** **and** ens_max_prob **≥ CONF_THRESH** (unified confident wrong) |
| **`FAIL_CLOSED`** | else if D_ex **≥ D_CLOSED_MIN** **or** epi_ex **≥ EPI_CLOSED_MIN** **or** ens_max_prob **< CONF_THRESH** |
| **`FAIL_AMBIG`** | else (rare residual) |

Cycle RED verdict:

| Label | Rule |
|-------|------|
| **`FAIL_CLOSED_DOMINANT`** | among ens-wrong, FAIL_CLOSED rate ≥ FAIL_OPEN rate **and** FAIL_CLOSED ≥ 0.50 of wrongs (or n_wrong=0 → N/A_PERFECT) |
| **`FAIL_OPEN_DOMINANT`** | FAIL_OPEN rate > FAIL_CLOSED rate among wrongs |
| **`FAIL_MIXED`** | else |

**Never** set `science_open=true`. §22 scope **unchanged**.

## Explicit non-goals

- No training / bag noise / #27 chase
- No soft distill / multi-hyp / SWA re-chase
- No select / curriculum reopen
- No matched-OOD peek for training
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen from RED PASS or FAIL
- No claim that RED metrics extend §22 OPEN

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_red_competent_dissonance.json` |
| **Gen report** | `artifacts/stalk_red_competent_dissonance_generation_report.json` |
| **Data** | `data/stalk_red_competent_dissonance.jsonl` (n=256; K16=64, K20=64, HN=128) |
| **Log** | `artifacts/stalk_red_competent_dissonance_run.log` |
| **Prereg SHA** | `8d83560` (committed before runs) |
| **Results SHA** | `e4f9b365d859b079802ccf5749e7df36cb149a42` |
| **Boundary refresh** | **exact match** to #28 (Δ=0 within float) |
| **RED verdict** | **`FAIL_CLOSED_DOMINANT`** (61/61 ens-wrong = FAIL_CLOSED; 0 FAIL_OPEN) |
| **CD arm on RED** | **CHAOS** (μ_acc 0.774 < 0.85 — expected under RED; not COMPETENT claim) |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Elapsed** | ~18 s (~0.3 min CDT) |

### (0) Boundary strip-easy — #22 ens matched-OOD T16 (#28 cite + refresh)

| Source | global pair | D_HN | D_K16 | D_easy(K8) | D_hard | μ_acc | CD | verdict |
|--------|-------------|------|-------|------------|--------|-------|-----|---------|
| **#28 cite** | **0.162** | **0.112** | **0.337** | 0.225 | **0.225** | **0.903** | **0.811** | **COMPETENT** |
| **refresh** | **0.162** | **0.112** | **0.337** | 0.225 | **0.225** | **0.903** | **0.811** | **COMPETENT** |

Easy mass (K8) stripped from D_hard. Refresh Δ abs = 0 vs cite (ok).

### RED T16 — (a)(b)(d) disagree + CD (#22 ens only; no #27)

| Arm | global pair | D_HN | D_K16 | D_K20 | D_long | D_hard | μ_acc | CD | CD arm |
|-----|-------------|------|-------|-------|--------|--------|-------|-----|--------|
| **#22 ens** | **0.277** | 0.195 | 0.276 | **0.443** | **0.360** | **0.277** | 0.774 | **0.774** | CHAOS (RED) |

`D_long = 0.5·(D_K16+D_K20)`; `D_hard = 0.5·(D_HN+D_long)`. K16 denser than ood_hops (p_emp mean **0.084** ≫ 0.025); K20 outside ADR OOD hops.

### Ens accuracy @ RED T16

| Slice | ens `prob_mean` | singles mean |
|-------|-----------------|--------------|
| overall | **0.762** | 0.774 |
| hard-neg | **0.898** | 0.851 |
| K16 (dense) | **0.812** | 0.769 |
| K20 (long) | **0.438** | 0.627 |

K20 under T=16 causal horizon collapses (expected). Dense K16 holds better than ood_hops HN shatter residue (#28: ens HN 0.067 on ood_hops).

### (c) Agree-set vs disagree-set ens acc @ RED T16

| Set | n | member ov | ens ov | ens HN | ens K16 | ens K20 |
|-----|---|-----------|--------|--------|---------|---------|
| agree | 84 | **1.000** | **1.000** | 1.000 | 1.000 | 1.000 |
| disagree | 172 | 0.664 | **0.645** | 0.817 | 0.714 | 0.390 |

### FAIL_OPEN vs FAIL_CLOSED (ens-wrong; n_wrong=61)

| Tag | count | rate |
|-----|-------|------|
| **FAIL_OPEN** | **0** | **0.000** |
| **FAIL_CLOSED** | **61** | **1.000** |
| FAIL_AMBIG | 0 | 0.000 |

Uncertainty means: epistemic **0.319** > aleatoric **0.086**. **No unified confident wrongs** — errors carry disagreement and/or low confidence / high epi.

### Reading (fail-closed)

1. **Boundary confirmed:** #28 strip-easy table stands; #22 COMPETENT on matched-OOD unchanged.
2. **RED = FAIL_CLOSED_DOMINANT:** when p/K leave train priors (denser K16 + K20), ens errs **without** FAIL_OPEN — dissonance/uncertainty signal present. Prefer this over echo-chamber failure mode.
3. K20@T16 under-horizon residue (ens 0.438) is expected; do **not** widen §22. Dense K16 holds (0.812) better than ood_hops HN shatter.
4. No #27 bag train. `science_open=false`. Select/curriculum CLOSED.
