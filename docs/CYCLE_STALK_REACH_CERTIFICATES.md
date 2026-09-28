# CYCLE_STALK_REACH_CERTIFICATES — MEASURE post-hoc reach certificates (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** post-hoc structural certificates on frozen **#14/#18/#22** ens |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | #33 `LOCAL_SOUND_WALL` + #34 park of further *local-sound gates*; named fork = multi-hop / path-reasoning **certificates** (post-hoc), not another local gate and **not** tropical Phase 2 |
| **Base** | `main` tip after PR #34 (`3730558`) |
| **Prior** | #30 FO=45; #31 STRUCTURAL_CLUSTER; #32 outdeg0 FO_PARTIAL (23/45); #33 rem-22 LOCAL_SOUND_WALL; #34 park local-sound chase |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |
| **Tropical Phase 2** | **NOT started** (deferred; out of scope for this PR) |

## Goal

Post-hoc structural certificates for reachability under hard Â:

1. **If ens (frozen #22 `prob_mean`) predicts YES:** require a **path-witness** on Â (or verify path existence via BFS **only as checker**, never as model input / train target / init). **Clean** iff the witness validates on Â; else **dirty** → treat as **FAIL_CLOSED / reject** (force pred=0).
2. **If predicts NO:** mark NO as **clean** when checker BFS confirms unreachable; optional directed-cut / frontier certificate if cheap (same checker; no separate expensive min-cut required this cycle). Dirty NO (pred=0 but reachable) is counted dirty but **not** force-opened (fail-closed policy; do not oracle-correct YES).
3. **Primary success:** dirty concentrates on hop-OOD FAIL_OPEN (esp. remainder-22 after outdeg0) with **near-zero** matched-OOD collateral (same hygiene shape as #32 outdeg0).

Certificates are **post-hoc** on `(Â, prediction [, optional witness])` and **deterministic**. They are **not** baked into weights, training targets, PEFT, select, bag, JS, sheaf, or BFS-as-model-feature.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| #27 | bag MEASURE_LIFT / CHAOS — **do not train more bag noise** |
| #28–#29 | COMPETENT_vs_CHAOS; RED FAIL_CLOSED_DOMINANT |
| #30 | FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED; residue **`HN_FAIL_OPEN_CORE`** (45) |
| #31 | STRUCTURAL_CLUSTER — isolated-source / hub-target |
| #32 | FO_PARTIAL — outdeg(s)==0 kills 23/45; residue OUTDEG0_PARTIAL |
| #33 | LOCAL_SOUND_WALL — best local cut 5/22; stop *local-sound* overlay chase |
| #34 | PARK_HOP_OOD_OVERLAY — local-sound gates parked; outdeg0 optional hygiene |

This cycle = **MEASURE post-hoc certificates** (checker BFS allowed). Not train. Not §22 widen. Not tropical. Do **not** claim hop-OOD OPEN. Does **not** reopen the parked *local-sound gate* chase; certificates are a named multi-hop / path-witness fork.

## Metaphor

- **map ≠ location** — #22 COMPETENT map on matched-OOD; a path-witness is a *post-hoc location check* on Â, not a learned bake-in.
- **negatives = mirror** — dirty YES without a validating path is the structural shadow of FAIL_OPEN false-reachability.

## Cite #30 / #32 / #33 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN (baseline) | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN baseline | **45** |
| #32 gated FO | **22** (23 killed by outdeg0) |
| #32 gated HN | **0.304** |
| Matched-OOD Δ (#32) | **0.000** |
| Residue in | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL`** |

## Checker BFS policy (LOCKED — explicit)

| Allowed | Forbidden |
|---------|-----------|
| Post-hoc path-existence / path-reconstruction **checker** to validate a YES witness on Â | BFS / out-closure / reachability as **training target**, **init**, or **model input feature** |
| Post-hoc unreachable confirmation to mark NO clean | Baking checker labels into weights / PEFT / select / bag / JS / sheaf |
| Deterministic witness validation: consecutive edges of path ∈ Â | Regenerating `ood_hops.jsonl` |
| Optional outdeg0 hygiene arm (local incidence; #32) stacked with cert | Tropical Phase 2 / any tropical probe in this PR |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_reach_certificates
# or: reachability-stalk-reach-certificates
```

| Item | Spec (locked) |
|------|----------------|
| Arms | (0) baseline #22 `prob_mean`; (1) **outdeg0** optional hygiene (#32); (2) **cert** reject/force-closed dirty YES; (3) **outdeg0+cert** |
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** @ **T=16** fixed; matched-OOD (`data/covariate_matched_ood.jsonl` @ T=16) for collateral |
| Prefer | reuse #22 ckpts + #28/#30/#32 FO ids / eval paths; re-infer ens preds |
| Artifact | `artifacts/stalk_reach_certificates.json` |

### Certificate defs (LOCKED)

| Pred | Clean iff | Dirty action (policy) |
|------|-----------|------------------------|
| **YES** (pred=1) | Path-witness `s=v0→…→vk=t` validates on Â (edges present), **or** checker BFS proves existence and reconstructed witness validates | **Reject / force-closed** → pred:=0 |
| **NO** (pred=0) | Checker BFS confirms unreachable (`t ∉ R_out(s)`), or `s==t` is false for NO of non-self (self `s==t` is always reachable / not a NO gold) | Count dirty; **do not** force-open |

Witness validation is deterministic: every consecutive pair `(v_i, v_{i+1})` must appear in Â; path endpoints must be `(s,t)`.

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29/#30/#31/#32)

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

### Metrics (prereg — locked)

| Metric | Spec |
|--------|------|
| Clean / dirty rates | overall + by pred YES/NO; fraction of dirty that are baseline FO / rem-22 |
| P(correct \| clean) | accuracy among clean examples after policy |
| FO killed | of the **45** baseline FO; of the **22** remainder (post-outdeg0) |
| Acc after policy | HN / K16 / overall under each arm |
| Matched-OOD Δ | overall / HN / K16 vs baseline (collateral) |
| Concentration | dirty∩FO / dirty  and  FO∩dirty / FO |

### Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`CERT_FO_CATCH`** | ≥40/45 FO eliminated under cert (or outdeg0+cert) **AND** ≥16/22 rem-22 eliminated **AND** no COLLATERAL_HARM **AND** dirty concentrates on FO (dirty∩FO / dirty ≥ 0.50 **or** FO∩dirty/FO ≥ 0.80) |
| **`CERT_PARTIAL`** | ≥1 FO killed but not CERT_FO_CATCH; no COLLATERAL_HARM |
| **`CERT_NOISE`** | dirty does **not** concentrate on FO (fails concentration) **and** FO catch weak (&lt;8/45) — certificates fire off-target |
| **`COLLATERAL_HARM`** | matched-OOD overall/HN/K16 **or** OK_HN index-set / positives drop ≥ **0.05** abs vs baseline |

Priority: if collateral harm fires → verdict **`COLLATERAL_HARM`**. Still **MEASURE**, never OPEN.

**Even `CERT_FO_CATCH` does not widen `science_open` / does not claim hop-OOD OPEN.**

### Prohibited defaults

- No train / PEFT / select / bag / JS / sheaf / BFS-as-model-feature baked into weights
- No BFS checker used as training target or init
- No T beyond 16
- No `science_open` widen; no §22 widen
- No claiming hop-OOD OPEN
- No tropical Phase 2 / tropical probe code in this PR
- No hardcoded desired labels into model init
- Do not regenerate `ood_hops.jsonl`
- Do not reopen parked *local-sound gate* chase as a new local predicate (outdeg0 reuse as optional hygiene arm only)

## Explicit non-goals

- No training / bag noise / #27 chase
- No soft distill / multi-hyp / SWA re-chase
- No select / curriculum reopen
- No T>16
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen
- No claim that hop-OOD metrics extend §22 OPEN
- No tropical Phase 2
- No treating checker BFS as a learned model feature

## Results (this run — cite artifact)

_Pending harness + run. Fill after MEASURE execution._
