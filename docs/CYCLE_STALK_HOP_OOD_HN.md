# CYCLE_STALK_HOP_OOD_HN — MEASURE audit (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** overlay stress of frozen **#14/#18/#22** ens on hop-OOD HN shatter |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | #28 ood_hops stress: #22 ens HN **1.000→0.067** shatter; K16 holds ~1.000→0.988; CD 0.811→0.553. #29 RED FAIL_CLOSED_DOMINANT did **not** pay this residue. |
| **Base** | `main` tip after PR #29 (`5c5006a`) |
| **Prior** | #28 COMPETENT_vs_CHAOS + ood_hops shatter residue; #29 RED FAIL_CLOSED_DOMINANT (0 OPEN / 61 CLOSED) |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

Deep-cut **eval-only** of frozen **#14/#18/#22** `prob_mean` ensemble on **`data/ood_hops.jsonl` @ T=16** under the same FAIL_OPEN / FAIL_CLOSED + CD lens as #29, focused on **hard-neg (HN)** shatter — **repair-or-characterize** via cheap inference overlays only.

Optional cheap overlay ablations (**inference only** — no bag retrain, no select/curriculum reopen):

1. **Baseline** `#22` `prob_mean` — replicate #28 ood_hops numbers.
2. **Disagreement / confidence gate** — abstain when pairwise D high **OR** ens_max_prob low; measure FAIL_OPEN≈0 and whether calibrated reject improves HN without inventing accuracy OPEN.
3. **Secondary cheap knob** — `majority_vote` aggregator (already in codebase) vs `prob_mean`.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| #23/#25/#26 | distill STOP; SWA MEASURE; multi-hyp STOP |
| #27 | bag MEASURE_LIFT / CHAOS — **do not train more bag noise** |
| #28 | COMPETENT_vs_CHAOS; ood_hops HN shatter residue unpaid |
| #29 | RED FAIL_CLOSED_DOMINANT — did **not** pay ood_hops HN |

This cycle = **MEASURE overlay stress only**. Not train. Not §22 widen.

## Metaphor

- **map ≠ location** — #22 COMPETENT map on matched-OOD; hop-OOD HN asks whether the map **shatters open** (unified confident wrong on hard-neg) or can **fail closed / abstain** under a cheap gate.
- **negatives = mirror** — HN slice is the unpaid residue; K16 holding is not payment.

## Cite #28 (ood_hops T16 — #22 ens shatter)

Locked from `artifacts/stalk_competent_dissonance.json` / PR #28 §(e):

| Metric | Value |
|--------|-------|
| ens HN | **0.067** (shatter from matched 1.000) |
| ens K16 | **0.988** (holds) |
| ens overall | **0.510** |
| global pair | **0.286** |
| D_HN | **0.358** |
| D_K16 | **0.211** |
| D_hard | **0.285** |
| μ_acc | **0.553** |
| CD | **0.553** |
| stress verdict | CHAOS |

Baseline arm must replicate within float noise (cite tol **0.02** abs on key scalars).

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_hop_ood_hn
# or: reachability-stalk-hop-ood-hn
```

| Item | Spec (locked) |
|------|----------------|
| Arms | Frozen **#14/#18** = **#22** ens map seeds **0..9** only |
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** — no bag / select / curriculum / distill / multi-hyp |
| Eval substrate | **`data/ood_hops.jsonl`** (existing; do not regenerate) @ **T=16** fixed |
| Primary aggregator | ens `prob_mean` |
| Secondary knob | `majority_vote` (codebase already) |
| Gate overlay | abstain when `D_ex ≥ GATE_D_MIN` **OR** `ens_max_prob < GATE_CONF_MIN` |

### Prohibited defaults

- No training / bag retrain / multi-hyp / select / upsample / distill
- No T beyond 16
- No `science_open` widen; even HN_RECOVERED does **not** widen §22
- No hardcoded desired labels into model init
- Hop-OOD remains MEASURE residue until human seal

### Slices (locked)

- **overall**
- **HN** (hop_distance == −1)
- **K16** (hop == 16)
- existing hop buckets in ood_hops: **K12**, **K8** (report; not in D_hard)

### Metrics (prereg — locked)

**(a) Ens accuracy** — overall / HN / K16 / K12 / K8 under `prob_mean` (and majority_vote secondary).

**(b) Disagreement** — global D; D_HN; D_K16; D_hard = 0.5·(D_HN+D_K16) (same #28 family).

**(c) CD** — `CD = μ_acc · min(D_hard, 0.25) / 0.25` (same #28/#29).

**(d) Agree-set vs disagree-set** — ens `prob_mean` overall (/HN/K16) on each set; member mean overall.

**(e) FAIL_OPEN vs FAIL_CLOSED** on ens-**wrong** (same defs as #29):

| Symbol | Value |
|--------|-------|
| `CONF_THRESH` | **0.80** (ens max softmax) |
| `D_CLOSED_MIN` | **0.10** (per-example pairwise disagree) |
| `EPI_CLOSED_MIN` | **0.15** nats (per-example epistemic entropy) |

| Label | Rule on an ens-wrong example |
|-------|------------------------------|
| **`FAIL_OPEN`** | pairwise D_ex **== 0** **and** ens_max_prob **≥ CONF_THRESH** |
| **`FAIL_CLOSED`** | else if D_ex **≥ D_CLOSED_MIN** **or** epi_ex **≥ EPI_CLOSED_MIN** **or** ens_max_prob **< CONF_THRESH** |
| **`FAIL_AMBIG`** | else |

Cycle fail-mode verdict:

| Label | Rule |
|-------|------|
| **`FAIL_CLOSED_DOMINANT`** | among ens-wrong, FAIL_CLOSED rate ≥ FAIL_OPEN **and** FAIL_CLOSED ≥ 0.50 of wrongs (or n_wrong=0 → N/A_PERFECT) |
| **`FAIL_OPEN_DOMINANT`** | FAIL_OPEN rate > FAIL_CLOSED among wrongs |
| **`MIXED`** | else |
| **`N/A_PERFECT`** | n_wrong == 0 |

**(f) Gate overlay** (inference abstain; locked thresholds):

| Symbol | Value |
|--------|-------|
| `GATE_D_MIN` | **0.10** (abstain if D_ex ≥ this) |
| `GATE_CONF_MIN` | **0.80** (abstain if ens_max_prob < this) |

On accepted (non-abstained) set report: coverage; ens overall/HN/K16; FAIL_OPEN/CLOSED among accepted wrongs. Abstain is **not** counted as correct — no invented accuracy OPEN.

### HN verdicts (overlay; fail-closed) — orthogonal to fail-mode

| Label | Rule |
|-------|------|
| **`HN_SHATTER_CONFIRMED`** | baseline ens HN < 0.50 (replicate shatter) **and** no overlay reaches HN ≥ 0.90 on full (non-abstain) eval |
| **`HN_PARTIAL_RECOVER`** | gate accepted-set HN ≥ 0.50 **or** majority_vote HN lifts ≥ +0.10 abs vs baseline, but not HN_RECOVERED |
| **`HN_RECOVERED`** | some reported arm (baseline or vote full-set, **or** gate accepted-set with coverage ≥ 0.25) has ens HN ≥ **0.90** **and** fail-mode is FAIL_CLOSED_DOMINANT or N/A_PERFECT |

**Explicit:** even **`HN_RECOVERED` does not widen §22**. Hop-OOD remains MEASURE residue until human seal. Harness `science_open=false` always.

### Cycle composite verdict

`{fail_mode_verdict}+{hn_verdict}` e.g. `FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED`.

**Never** set `science_open=true`.

## Explicit non-goals

- No training / bag noise / #27 chase
- No soft distill / multi-hyp / SWA re-chase
- No select / curriculum reopen
- No T>16
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen from any HN verdict
- No claim that hop-OOD metrics extend §22 OPEN
- No hardcoded labels into model init

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_hop_ood_hn.json` |
| **Log** | `artifacts/stalk_hop_ood_hn_run.log` |
| **Prereg SHA** | `a40abb6` (committed before runs) |
| **Harness SHA** | `051743c` |
| **Results SHA** | _(stamped on results commit)_ |
| **Cycle verdict** | **`FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED`** |
| **CD arm** | **CHAOS** (μ_acc 0.553 < 0.85 — expected under hop-OOD) |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Elapsed** | ~12.7 s (~0.2 min CDT) |
| **Cite #28 replicate** | **exact match** (Δ=0 within float) |

### Baseline vs overlays — ood_hops T16 (#22 ens)

| Arm | overall | HN | K16 | FAIL_OPEN | FAIL_CLOSED | CD | notes |
|-----|---------|----|----|-----------|-------------|-----|-------|
| **baseline `prob_mean`** | **0.510** | **0.067** | **0.988** | **45** | **190** | **0.553** | shatter replicate |
| **majority_vote** | 0.498 | **0.067** | 0.962 | 45 | 196 | 0.553† | no HN lift |
| **gate accepted** (cov **0.279**) | 0.664 | **0.000** | 1.000 | **45** | **0** | n/a | concentrates FAIL_OPEN HN |

† CD/D from member geometry identical across aggregators (same hard argmaxes).

Gate rule: abstain if `D_ex ≥ 0.10` OR `ens_max_prob < 0.80`. Accepted n=134 / 480; HN accept 45/240 (all 45 are unified confident wrong — predict reachable on unreachable).

### Disagreement + CD (baseline; = #28 cite)

| Metric | Value |
|--------|-------|
| global pair | **0.286** |
| D_HN | **0.358** |
| D_K16 | **0.211** |
| D_hard | **0.285** |
| μ_acc | **0.553** |
| CD | **0.553** |

### Agree-set vs disagree-set ens acc @ T16

| Set | n | ens ov | ens HN | ens K16 |
|-----|---|--------|--------|---------|
| agree | 134 | 0.664 | **0.000** (45 HN) | 1.000 |
| disagree | 346 | 0.451 | 0.082 (195 HN) | 0.978 |

### FAIL_OPEN vs FAIL_CLOSED (ens-wrong; n_wrong=235)

| Tag | count | rate |
|-----|-------|------|
| **FAIL_OPEN** | **45** | **0.191** |
| **FAIL_CLOSED** | **190** | **0.809** |
| FAIL_AMBIG | 0 | 0.000 |

**All 45 FAIL_OPEN are hard-neg** (D_ex=0, ens_max_prob≥0.80, pred=1 / label=0). Bulk errors FAIL_CLOSED → cycle fail-mode **FAIL_CLOSED_DOMINANT**, but HN shatter has a **FAIL_OPEN core**.

### Reading (fail-closed)

1. **Shatter confirmed:** #28 ood_hops numbers replicate exactly; ens HN **0.067**, K16 holds **0.988**.
2. **Overlays do not repair HN:** vote HN unchanged; prereg gate **worsens** HN (0.000 on accepted) by keeping the agree/high-conf FAIL_OPEN hard-negs and dropping disagreeing mass.
3. **Residue named:** `HN_FAIL_OPEN_CORE` — 45 hop-OOD hard-neg unified confident wrongs. Distinct from #29 RED (0 FAIL_OPEN).
4. Prefer #14 + #22 ens on **matched-OOD only**. Do **not** widen §22. `science_open=false`. No train.
