# CYCLE_STALK_TROPICAL_ATTENTION_PROBE — MEASURE tropical / max-plus ens probe (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** tropical / max-plus probe on frozen **#14/#18/#22** ens |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | #35 `CERT_FO_CATCH` sealed post-hoc certificates; named Phase 2 fork = whether **tropical prediction** moves FO **before** refuse / certificates |
| **Base** | `main` tip after PR #35 (`86d1ff0`) |
| **Prior** | #30 FO=45; #31 STRUCTURAL_CLUSTER; #32 rem-22; #33 LOCAL_SOUND_WALL; #34 park; #35 CERT_FO_CATCH |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |
| **Train / PEFT / β-anneal / DEAR** | **NONE** — eval-only on frozen ckpts |

## Goal

Test whether max-plus / β→∞ hard routing of the **frozen** stalk ensemble reduces false-reach (FAIL_OPEN) vs `#22` `prob_mean` **before** post-hoc certificates — or falsify if FO unchanged.

Hypothesis: softmax sum-product aggregation bleeds mass across large non-intersecting cones (remainder-22 FO). Tropical max-plus / β→∞ hard member routing should reduce false-reach vs `prob_mean`, or falsify (`TROPICAL_NULL`) if rem-22 FO and hop-OOD HN FO are essentially unchanged.

**Do not claim tropical ≡ certificate.** Cert-on-top (#35) is reported as a secondary reference column only.

## Scope decision (LOCKED — smallest tropical change)

In-attention tropical rewrite (`y = max_j (s_ij + v_j)` inside `nn.MultiheadAttention`) on **frozen** softmax-trained weights is **impractical / not a fair port test**: it would invent a new inference operator over QK scores trained under sum-product softmax, not test the #22 ensemble map.

**Scoped probe (this cycle):** tropical **ensemble aggregation / logit mix** over the 10 frozen member logits — still a genuine falsifier of the "sum-product bleed → FO" hypothesis at the ens layer.

| Arm | Definition |
|-----|------------|
| **(0) baseline** | `#22` `prob_mean`: `argmax_c mean_m softmax(L_m)_c` |
| **(1) tropical primary — `logit_max`** | Max-plus over member logits: `score_c = max_m L[m,c]`; `pred = argmax_c score_c`. Confidence for FO labeling = `max_c softmax(score)_c`. |
| **(2) tropical secondary — `beta_inf_member`** | β→∞ hard member routing: pick member `m* = argmax_m max_c softmax(L_m)_c` (highest-confidence member); use that member's hard pred. Documented as optional ens-only tropical. |
| **(3) cert reference** | Cite sealed #35 `CERT_FO_CATCH` numbers — **do not re-claim**; not a tropical arm |

Exact helpers live in `reachability_gen.tropical` (`max_plus_aggregate`, `beta_inf_member_preds`).

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| #30–#33 | FO core + LOCAL_SOUND_WALL; overlay chase parked #34 |
| #35 | CERT_FO_CATCH (post-hoc checker); tropical **deferred → this cycle** |

This cycle = **MEASURE tropical ens aggregation probe**. Not train. Not β-anneal. Not DEAR. Not §22 widen. Not OPEN. Does **not** replace certificates.

## Cite #30 / #32 / #35 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN (baseline) | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN baseline | **45** |
| rem-22 (post-outdeg0) | **22** |
| #35 cert HN / FO | **1.000** / **0** (reference only) |
| Matched-OOD Δ (#35 cert) | **0.000** |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_tropical_attention_probe
# or: reachability-stalk-tropical-attention-probe
```

| Item | Spec (locked) |
|------|----------------|
| Arms | (0) baseline `#22` `prob_mean`; (1) `logit_max` max-plus; (2) `beta_inf_member`; (3) cert reference from #35 artifact (no re-infer required for claim) |
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** @ **T=16** fixed; matched-OOD (`data/covariate_matched_ood.jsonl` @ T=16) for collateral |
| Prefer | reuse #22 ckpts + #30/#31/#32 FO / rem-22 ids |
| Artifact | `artifacts/stalk_tropical_attention_probe.json` |

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29/#30/#31/#32/#35)

| Symbol | Value |
|--------|-------|
| `CONF_THRESH` | **0.80** |
| `D_CLOSED_MIN` | **0.10** |
| `EPI_CLOSED_MIN` | **0.15** nats |

| Label | Rule on ens-wrong |
|-------|-------------------|
| **`FAIL_OPEN`** | D_ex **== 0** **and** ens_max_prob **≥ CONF_THRESH** |
| **`FAIL_CLOSED`** | else if D_ex ≥ D_CLOSED_MIN **or** epi ≥ EPI_CLOSED_MIN **or** ens_max_prob < CONF_THRESH |
| **`FAIL_AMBIG`** | else |

Baseline FO id set = #30/#31 45-core (replicate). rem-22 = #32 `fo_still_wrong_ids`. Tropical FO counts use the **same** FO labeling rule on the tropical arm's preds + tropical confidence; primary kill metrics = how many of the **baseline 45 / rem-22** become correct under tropical preds.

### Metrics (prereg — locked)

| Metric | Spec |
|--------|------|
| HN / K16 / overall | baseline vs tropical arms @ ood_hops T16 |
| FO count | FAIL_OPEN under each arm; FO killed of baseline 45; rem-22 killed of 22 |
| Matched-OOD Δ | overall / HN / K16 vs baseline (collateral) |
| Cert reference | cite #35 table only |

### Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`TROPICAL_FO_LIFT`** | ≥16/45 baseline FO eliminated under primary tropical (`logit_max`) **AND** ≥8/22 rem-22 eliminated **AND** no COLLATERAL_HARM |
| **`TROPICAL_PARTIAL`** | ≥1 baseline FO killed but not TROPICAL_FO_LIFT; no COLLATERAL_HARM |
| **`TROPICAL_NULL`** | rem-22 FO killed == 0 **and** baseline FO killed ≤ 2 **and** \|Δ HN\| ≤ 0.05 vs baseline (essentially unchanged) — falsifies hypothesis |
| **`COLLATERAL_HARM`** | matched-OOD overall/HN/K16 drop ≥ **0.05** abs vs baseline |

Priority: if collateral harm fires → verdict **`COLLATERAL_HARM`**. Still **MEASURE**, never OPEN.

**Even `TROPICAL_FO_LIFT` does not widen `science_open` / does not claim hop-OOD OPEN / does not replace certificates.**

### Prohibited defaults

- No train / PEFT / β-anneal schedule / DEAR solver
- No inventing untrained tropical attention layers with random weights
- No claiming tropical ≡ certificate
- No T beyond 16
- No `science_open` widen; no §22 widen
- No claiming hop-OOD OPEN
- No hardcoded desired labels into model init
- Do not regenerate `ood_hops.jsonl`
- Do not reopen parked local-sound gate chase

## Explicit non-goals

- No training / bag noise / anneal
- No in-attention MHA rewrite on frozen ckpts (scoped out; document why)
- No soft distill / multi-hyp / SWA re-chase
- No select / curriculum reopen
- No T>16
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen
- No claim that hop-OOD metrics extend §22 OPEN
- No replacing #35 certificates

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_tropical_attention_probe.json` |
| **Log** | `artifacts/stalk_tropical_attention_probe_run.log` |
| **Prereg SHA** | `30f57de` |
| **Harness SHA** | `ffd1990` |
| **Results SHA** | `698112f54951a47d53e0a859cf3c64039281aac0` |
| **Cycle verdict** | **`COLLATERAL_HARM`** |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Tropical definition used** | scoped ens **`logit_max`** (primary) + **`beta_inf_member`** (secondary); **not** in-attn rewrite |
| **Cite #30 replicate** | **exact** (FO=45 / ens HN=0.067) |
| **Residue update** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL/CERT_FO_CATCH/COLLATERAL_HARM`** |
| **Elapsed** | ~16.4 s CDT |

### Table (baseline vs tropical)

| Arm | HN | K16 | overall | FO | rem22 FO | FO killed/45 | rem22 killed/22 |
|-----|----|-----|---------|----|----------|--------------|-----------------|
| (0) baseline #22 | **0.067** | **0.988** | **0.510** | **45** | **22** | — | — |
| (1) logit_max | **0.150** | **0.850** | **0.510** | **45** | — | **0**/45 | **0**/22 |
| (2) beta_inf_member | **0.150** | **0.887** | **0.521** | **45** | — | **0**/45 | **0**/22 |
| (3) cert ref #35 | **1.000** | **0.988** | **0.977** | **0** | **0** | **45**/45 | **22**/22 |

### Matched-OOD collateral (T16)

| Slice | baseline | logit_max | Δ | beta_inf | Δ |
|-------|----------|-----------|---|----------|---|
| overall | 0.996 | 0.892 | **-0.104** | 0.892 | **-0.104** |
| HN | 1.000 | 0.992 | **-0.008** | 0.983 | **-0.017** |
| K16 | 1.000 | 0.512 | **-0.488** | 0.538 | **-0.462** |

**COLLATERAL_HARM** fires (matched K16 drop ≈0.49; overall drop ≈0.10). Primary FO catch null (0/45; 0/22 rem). Tropical ≢ certificate.

### Reading (fail-closed)

1. **COLLATERAL_HARM:** tropical ens aggregation (`logit_max` / `beta_inf_member`) does **not** eliminate baseline FO (0/45) or rem-22 (0/22). Hop-OOD HN inches 0.067→0.150 without touching the FO core; K16 on hop-OOD drops (0.988→0.850).
2. Matched-OOD competence collapses under hard max routing (K16 1.000→0.512) — `prob_mean` soft averaging is load-bearing for §22 competence.
3. Falsifies the ens-layer "sum-product bleed → rem-22 FO" hypothesis as a repair lever: FO unchanged **and** collateral harm. Prefer #14+#22 `prob_mean` on matched-OOD; keep #35 certificates as the FO catch (post-hoc). Do **not** widen §22. Do **not** claim hop-OOD OPEN. Do **not** equate tropical with certificates. Still **MEASURE**.
