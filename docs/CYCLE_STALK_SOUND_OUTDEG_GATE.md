# CYCLE_STALK_SOUND_OUTDEG_GATE — MEASURE eval-only overlay (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** sound hard-Â out-degree gate overlay on frozen **#14/#18/#22** ens |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | #31 `STRUCTURAL_CLUSTER`: FO_HN = isolated-source / hub-target (median `n_reach_from_s=1`, `max_dist_from_s=0`, `n_reach_to_t=27`). OK_HN inverse. Members 34 HARD_UNANIMOUS / 11 SOFT_AGREE. Residue `HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER`. |
| **Base** | `main` tip after PR #31 (`c909d1b`) |
| **Prior** | #30 hop-OOD HN shatter (45 FAIL_OPEN); #31 structural autopsy |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

Test a **sound** hard-Â constraint overlay:

> If **out-degree of source `s` is 0** (local directed out-edges from `s` in the given Â), and target `t ≠ s`, then gold reachability is **false** → **force ens pred = unreachable** (class 0).

This uses **only local incidence of `s`** from the given Â — **NOT** a BFS/reachability oracle over the full graph, **NOT** training, **NOT** baking labels into init.

Optional note: for simple digraphs without self-loops, `|out-neighborhood(s)| == outdeg(s)`; the stricter `|out-neighborhood|==0` arm is **identical** to primary — keep **one primary**.

Abstain alternative (not primary): abstain when triggered, counted as fail-closed-correct on HN `label=0`. Primary = **force-unreach**.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| #27 | bag MEASURE_LIFT / CHAOS — **do not train more bag noise** |
| #28 | COMPETENT_vs_CHAOS; ood_hops HN shatter residue |
| #29 | RED FAIL_CLOSED_DOMINANT (0 OPEN) — did **not** pay ood_hops HN |
| #30 | FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED; residue **`HN_FAIL_OPEN_CORE`** (45) |
| #31 | STRUCTURAL_CLUSTER — isolated-source / hub-target |

This cycle = **MEASURE overlay only**. Not train. Not §22 widen. Do **not** claim hop-OOD OPEN even if FO_CORE_KILLED.

## Metaphor

- **map ≠ location** — #22 COMPETENT map on matched-OOD; sound local incidence is a hard-Â *constraint*, not a learned location.
- **negatives = mirror** — FO core is isolated-source false-reachability; outdeg(s)==0 is the local incidence shadow of that cluster (not full out-closure BFS).

## Cite #30 / #31 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN | **45** (all HN) |
| FAIL_CLOSED | **190** |
| Autopsy | **STRUCTURAL_CLUSTER** — FO median `n_reach_from_s=1`, `max_dist_from_s=0`; OK inverse |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_sound_outdeg_gate
# or: reachability-stalk-sound-outdeg-gate
```

| Item | Spec (locked) |
|------|----------------|
| Arms | (0) baseline #22 `prob_mean`; (1) **primary** `outdeg(s)==0 ∧ t≠s` → force pred=0 |
| Optional (2) | `|out-neighborhood(s)|==0` — identical to (1) on simple digraphs; report equivalence, do not dual-run |
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** @ **T=16** fixed; also matched-OOD slice (`data/covariate_matched_ood.jsonl` @ T=16) for collateral |
| Prefer | reuse #30/#31 FO ids + graph fields from jsonl; re-infer ens preds if not stored in artifact |
| Artifact | `artifacts/stalk_sound_outdeg_gate.json` |

### Outdeg definition (LOCKED — matches autopsy directed out-edges)

```
outdeg(s) = |{ v | (s,v) ∈ Â }|   # directed out-edges from s; no self-loops in Â
```

Equivalent: `len(adjacency_list(n, edges)[s])`. **Not** total degree (in+out). **Not** `|R_out(s)|` (BFS out-closure — prohibited as gate feature).

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29/#30/#31)

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
| Acc | overall / HN / K16 under baseline vs gated ens |
| Fail mode | FAIL_OPEN / FAIL_CLOSED counts on ens-wrong (baseline vs gated) |
| FO killed | how many of the **45** baseline FO become correct under gate |
| Collateral | OK_HN index-set acc delta; positives acc delta; matched-OOD overall/HN/K16 deltas |
| CD | competent dissonance if cheap (same #28 formula) |

### Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`FO_CORE_KILLED`** | ≥40/45 FO eliminated **AND** HN ens ≥0.80 |
| **`FO_PARTIAL`** | ≥1 FO killed but not FO_CORE_KILLED |
| **`FO_UNMOVED`** | 0 of 45 FO eliminated |
| **`COLLATERAL_HARM`** | matched-OOD overall/HN/K16 **or** OK_HN index-set / positives drop ≥ **0.05** abs vs baseline |

Priority: if collateral harm fires → verdict **`COLLATERAL_HARM`** (still MEASURE; STOP this overlay only if harm forces it — do not park whole stalk corridor otherwise).

**Even `FO_CORE_KILLED` does not widen `science_open` / does not claim hop-OOD OPEN.**

### Prohibited defaults

- No full BFS / reachability oracle as gate features beyond local `outdeg(s)`
- No training / bag retrain / multi-hyp / select / upsample / distill
- No T beyond 16
- No `science_open` widen; no §22 widen
- No claiming hop-OOD OPEN
- No parking whole stalk corridor unless collateral harm forces STOP on **this overlay**
- No hardcoded desired labels into model init
- Do not regenerate `ood_hops.jsonl`

## Explicit non-goals

- No training / bag noise / #27 chase
- No soft distill / multi-hyp / SWA re-chase
- No select / curriculum reopen
- No T>16
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen
- No claim that hop-OOD metrics extend §22 OPEN
- No BFS out-closure gate (`n_reach_from_s==1`) — that is autopsy structure, not this overlay

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_sound_outdeg_gate.json` |
| **Log** | `artifacts/stalk_sound_outdeg_gate_run.log` |
| **Prereg SHA** | `3f4d791` |
| **Harness SHA** | `6a75b1f` |
| **Results SHA** | _(stamp after results commit)_ |
| **Cycle verdict** | **`FO_PARTIAL`** |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Residue update** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL`** |
| **Elapsed** | ~16.3 s (~0.3 min CDT) |
| **Cite #30 replicate** | **exact** (FO=45 / FC=190 / ens HN=0.067) |
| **Soundness** | 59 triggers; **0** y≠0 violations; out-neighborhood ≡ outdeg on simple digraphs |

### Table (baseline vs gate)

| Arm | HN | K16 | overall | FO | FC | FO killed /45 |
|-----|----|-----|---------|----|----|---------------|
| (0) baseline #22 | **0.067** | **0.988** | **0.510** | **45** | **190** | — |
| (1) outdeg(s)==0 force-unreach | **0.304** | **0.988** | **0.629** | **22** | **156** | **23**/45 |

### Matched-OOD collateral (T16)

| Slice | baseline | gated | Δ |
|-------|----------|-------|---|
| overall | 0.996 | 0.996 | **0.000** |
| HN | 1.000 | 1.000 | **0.000** |
| K16 | 1.000 | 1.000 | **0.000** |

OK_HN index-set: 16/16 stay correct (Δ=0). Positives: Δ=0. Matched-OOD gate triggers=97 (already pred=0). **No COLLATERAL_HARM.**

### Reading (fail-closed)

1. **FO_PARTIAL:** local `outdeg(s)==0` kills **23/45** of the FO core (exactly the outdeg0 subset of the isolated-source cluster). Remaining **22** FO have outdeg(s)>0 but still trivial/small out-closure by autopsy — **not** reachable by local-incidence gate alone (BFS out-closure prohibited).
2. HN rises 0.067→0.304 (also absorbs some FC_HN with outdeg0); K16 holds; overall +0.119. CD unchanged (member geometry).
3. Matched-OOD competence **untouched** (Δ=0). Prefer #14+#22 on matched-OOD only. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Corridor not parked — overlay MEASURE only; residue unpaid.
