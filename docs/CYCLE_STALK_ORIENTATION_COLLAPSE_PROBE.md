# CYCLE_STALK_ORIENTATION_COLLAPSE_PROBE — MEASURE Dir-GNN-style orientation collapse (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** orientation / in–out collapse probe on frozen **#14/#18/#22** ens |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | #35 `CERT_FO_CATCH` + #36 `COLLATERAL_HARM`; named Phase 3 = whether rem-22 FO under large bidirectional cones reflects **orientation collapse** (in/out mixing) in stalk recurrence |
| **Base** | `main` tip after PR #36 (`3198371`) |
| **Prior** | #30 FO=45; #31 STRUCTURAL_CLUSTER; #32 rem-22; #33 LOCAL_SOUND_WALL; #34 park; #35 CERT_FO_CATCH; #36 COLLATERAL_HARM |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |
| **Train / PEFT / β-anneal / DEAR / tropical retrain** | **NONE** — eval-only on frozen ckpts |
| **Phase 4 energy selector** | **NOT started** (out of scope for this PR) |

## Goal

Test whether Dir-GNN-style **in-neighborhood vs out-neighborhood** orientation metrics on frozen stalk states separate FAIL_OPEN (esp. rem-22) from OK_HN / FC — as a **probe**, not an architecture change.

Hypothesis: rem-22 FO under large bidirectional cones reflects **orientation collapse** (in/out mixing) in stalk recurrence. If true, orientation metrics (high `cos(h_in,h_out)` / low `||h_in−h_out||`) concentrate on FO / rem-22 vs OK_HN. If null, falsify this explanatory lever for rem-22.

## Architecture scope (LOCKED — honest)

`FractalCore` uses **directed adjacency-masked** `nn.MultiheadAttention`:

- `A[i,j] = 0` iff edge `j → i` (key→query) or self; else `-inf`
- Messages flow **in-neighbors only**; **no** explicit Dir-GNN in/out channel split
- Attention weights are not exposed in the sealed forward path (`need_weights=False`)

**Therefore** this cycle does **not** claim true Dir-GNN in/out channels. Primary instrument = **directed incidence × final hidden** proxy (documented below). Optional reverse-mask / attn-mass arms are **out of scope** this PR (would alter inference or require invasive hooks).

### Exact proxy (LOCKED)

On frozen forward @ T=16 with `return_states=True` → `H = final_states[b, :n, :]` (pre-ln_f node slots):

For each node `i` with edge sets `N_in(i)={j: j→i}`, `N_out(i)={j: i→j}`:

```
h_in[i]  = mean_{j ∈ N_in(i)}  H[j]     (0-vector if empty)
h_out[i] = mean_{j ∈ N_out(i)} H[j]     (0-vector if empty)
```

At **target `t`** (primary) and **source `s`** (secondary):

| Metric | Definition |
|--------|------------|
| `cos_orient` | `cos(h_in, h_out)` — nan if either norm &lt; eps |
| `l2_orient` | `‖h_in − h_out‖₂` |
| `mass_in` / `mass_out` | `‖h_in‖₂` / `‖h_out‖₂` |
| `mass_ratio` | `mass_in / (mass_in + mass_out + eps)` |
| `deg_in` / `deg_out` | `|N_in|` / `|N_out|` (structural context) |

Ensemble aggregation: **mean over members** seeds `0..9` (#14+#18) of each scalar metric per example. Also report member-0 (#14 seed0) as a single-model check.

Helpers live in `reachability_gen.orientation`.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| #30–#33 | FO core + LOCAL_SOUND_WALL; overlay chase parked #34 |
| #35 | CERT_FO_CATCH (post-hoc checker) |
| #36 | COLLATERAL_HARM (tropical ens aggregation) |

This cycle = **MEASURE orientation collapse probe**. Not train. Not §22 widen. Not tropical retrain. Not new overlay chase. Not Phase 4 energy selector.

## Cite #30 / #32 / #35 / #36 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN (baseline) | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN baseline | **45** |
| rem-22 (post-outdeg0) | **22** |
| #35 cert HN / FO | **1.000** / **0** (reference only) |
| #36 tropical | COLLATERAL_HARM; FO 0/45 |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_orientation_collapse_probe
# or: reachability-stalk-orientation-collapse-probe
```

| Item | Spec (locked) |
|------|----------------|
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** @ **T=16** fixed |
| Strata | OK_HN / FO_HN (45) / rem-22 / FO_KILLED / FC_HN from #31/#32/#33 artifacts (row indices) |
| Prefer | reuse #22 ckpts + sealed FO / rem-22 ids |
| Artifact | `artifacts/stalk_orientation_collapse_probe.json` |

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29–#36)

| Symbol | Value |
|--------|-------|
| `CONF_THRESH` | **0.80** |
| `D_CLOSED_MIN` | **0.10** |
| `EPI_CLOSED_MIN` | **0.15** nats |

Strata IDs are **cited** from sealed artifacts (not re-derived FO labels as primary claim). Replicate #22 ens preds only as hygiene.

### Metrics (prereg — locked)

| Metric | Spec |
|--------|------|
| Stratum medians | `cos_orient_t`, `l2_orient_t`, `mass_ratio_t` (+ `_s` secondary) by OK_HN / FO_HN / rem-22 / FO_KILLED / FC_HN |
| AUROC | primary `cos_orient_t` and `l2_orient_t` scoring FO_HN vs OK_HN; rem-22 vs OK_HN |
| Empty-N rate | fraction of examples with empty in or out at t (nan cos) — feeds INCONCLUSIVE_ARCH |
| rem-22 separates? | rem-22 median outside OK_HN IQR on primary metric |

**Primary score direction (hypothesis):** higher `cos_orient_t` and lower `l2_orient_t` → more collapse → more FO. AUROC for cos uses raw score; AUROC for l2 uses **negated** l2 so higher=more FO.

### Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`ORIENT_SEPARATES_FO`** | AUROC(cos or −l2; FO_HN vs OK_HN) ≥ **0.75** **AND** rem-22 median outside OK_HN IQR on that primary metric **AND** empty-N@t cos-nan rate on rem-22 ≤ **0.50** |
| **`ORIENT_PARTIAL`** | AUROC ≥ **0.60** **or** rem-22 median outside OK IQR, but not full SEPARATES; empty-N ok |
| **`ORIENT_NULL`** | AUROC ∈ [0.40, 0.60] **and** rem-22 median inside OK_HN IQR on both cos and l2 — falsifies orientation-collapse explanatory lever under this proxy |
| **`INCONCLUSIVE_ARCH`** | cos-nan rate on rem-22 **or** OK_HN &gt; **0.50** (proxy unusable on bidirectional-cone hypothesis set) |

Still **MEASURE**, never OPEN. Even `ORIENT_SEPARATES_FO` does **not** widen `science_open` / does **not** claim hop-OOD OPEN / does **not** authorize Phase 4 in this PR.

### Prohibited defaults

- No train / PEFT / β-anneal / DEAR / tropical retrain
- No inventing Dir-GNN architecture / dual in-out MHA
- No §22 widen; no claiming hop-OOD OPEN
- No new overlay chase / local-sound gate reopen
- No Phase 4 energy selector in this PR
- No T beyond 16
- No hardcoded desired labels into model init
- Do not regenerate `ood_hops.jsonl`

## Explicit non-goals

- No training / bag noise / anneal
- No architecture change to true Dir-GNN
- No tropical retrain / in-attn rewrite
- No Phase 4 energy selector
- No soft distill / multi-hyp / SWA re-chase
- No select / curriculum reopen
- No T>16
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen
- No claim that hop-OOD metrics extend §22 OPEN

## Results (this run — fill after MEASURE)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_orientation_collapse_probe.json` |
| **Log** | `artifacts/stalk_orientation_collapse_probe_run.log` |
| **Prereg SHA** | _(fill)_ |
| **Harness SHA** | _(fill)_ |
| **Results SHA** | _(fill)_ |
| **Cycle verdict** | _(fill)_ |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Proxy used** | directed incidence × final hidden (no explicit in/out channels) |
| **rem-22 separates?** | _(fill)_ |
| **Elapsed** | _(fill)_ CDT |

### Table (stratum medians — fill)

| Stratum | n | cos_t med | l2_t med | mass_ratio_t med | cos nan rate |
|---------|---|-----------|----------|------------------|--------------|
| OK_HN | | | | | |
| FO_HN | | | | | |
| rem-22 | | | | | |
| FO_KILLED | | | | | |
| FC_HN | | | | | |

### AUROC (fill)

| Contrast | cos_t AUROC | (−l2_t) AUROC |
|----------|-------------|---------------|
| FO_HN vs OK_HN | | |
| rem-22 vs OK_HN | | |

### Reading (fill after run)

_(fail-closed reading)_
