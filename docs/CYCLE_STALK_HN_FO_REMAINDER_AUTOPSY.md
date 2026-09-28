# CYCLE_STALK_HN_FO_REMAINDER_AUTOPSY — MEASURE structural autopsy of outdeg>0 FO remainder (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** structural autopsy of the **22** remaining hop-OOD FAIL_OPEN after #32 outdeg0 gate (outdeg(s)>0) |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | #32 `FO_PARTIAL`: FO 45→22 killed; HN 0.067→0.304; matched-OOD Δ=0. Residue `HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL` — remaining **22** have **outdeg(s)>0**. |
| **Base** | `main` tip after PR #32 (`db8c2b1`) |
| **Prior** | #30 hop-OOD HN shatter (45 FO); #31 STRUCTURAL_CLUSTER; #32 sound outdeg0 FO_PARTIAL |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

1. Structural autopsy of the **remaining 22** FO (outdeg(s)>0) vs: the **23** killed FO, OK_HN, FC_HN on `ood_hops` @ T=16.
2. Ask: is there another **local-sound** constraint (computable from Â without full-graph BFS/reachability oracle) that covers a large fraction of the 22?
3. If a candidate local cut is found that is **sound** (when true ⇒ gold must be unreachable) and covers **≥8/22**, **implement and eval** it as overlay arm stacked on outdeg0 (same harness style as #32), report FO killed, HN, matched-OOD collateral.
4. If none: verdict **`LOCAL_SOUND_WALL`** — remaining FO require multi-hop reasoning; name residue and **stop overlay chase** for this core.

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
| #32 | FO_PARTIAL — outdeg(s)==0 kills 23/45; residue OUTDEG0_PARTIAL |

This cycle = **MEASURE autopsy (+ optional sound overlay if cut found)**. Not train. Not §22 widen. Do **not** claim hop-OOD OPEN.

## Metaphor

- **map ≠ location** — #22 COMPETENT map on matched-OOD; remainder asks whether another *local incidence* shadow covers the unpaid FO, or whether the unpaid core needs multi-hop location.
- **negatives = mirror** — the 22 are the outdeg>0 half of the isolated-source cluster; local-sound wall vs local cut is the mirror question.

## Cite #30 / #31 / #32 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN (baseline) | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN baseline | **45** |
| #32 gated FO | **22** (23 killed by outdeg0) |
| #32 gated HN | **0.304** |
| Matched-OOD Δ | **0.000** |
| Residue in | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL`** |

## Preregistered questions (LOCKED before runs)

1. **Contrast:** local + structural features of the **22** remainder FO vs **23** killed FO vs OK_HN vs FC_HN (medians / bins). Autopsy *may* report BFS-derived features (`n_reach_from_s`, `max_dist_from_s`, …) for characterization only.
2. **Local-sound candidate search:** among features **computable from Â without full-graph BFS/reachability oracle**, is there a constraint C such that:
   - **Sound:** C ⇒ gold unreachable (y must be 0 when t≠s / applicable)
   - **Coverage:** C holds on ≥ **8/22** of the remainder FO
3. **If cut found:** implement overlay stacked on outdeg0; eval FO killed / HN / matched-OOD collateral; gate verdict enum below.
4. **If none:** verdict `LOCAL_SOUND_WALL`; name residue; **stop overlay chase** for this FO core.

## Allowed / prohibited gate features (LOCKED)

### Allowed (local-sound candidates)

- `outdeg(s)`, `indeg(s)`, `outdeg(t)`, `indeg(t)` — directed local incidence
- `t ∈ N+(s)` (1-hop out-neighbor check); `s ∈ N-(t)` (1-hop in-neighbor check)
- `|N+(s)|`, `|N-(t)|` (1-hop neighborhood sizes)
- Simple local cuts over the 1-hop star of s and/or t (e.g. every out-neighbor of s has outdeg 0 — degrees of adjacent nodes only; **not** iterative BFS closure)

### Prohibited as gate features

- Full BFS / multi-hop reachability closure (`n_reach_from_s`, `|R_out(s)|`, `|R_in(t)|` as gate predicates)
- Gold reachability labels / anything that solves the task
- Training / baking labels into init
- Regenerating `ood_hops.jsonl`

**Note:** BFS-derived features remain allowed **for autopsy characterization tables only**, never as overlay gate predicates.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_hn_fo_remainder_autopsy
# or: reachability-stalk-hn-fo-remainder-autopsy
```

| Item | Spec (locked) |
|------|----------------|
| Arms | Autopsy of remainder vs killed/OK/FC; optional overlay stacked on #32 outdeg0 if cut ≥8/22 sound |
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** @ **T=16** fixed; matched-OOD (`data/covariate_matched_ood.jsonl`) for collateral if gate run |
| Prefer | offline on stored #30/#31/#32 artifacts + jsonl graphs; re-infer ens preds only if needed |
| Artifact | `artifacts/stalk_hn_fo_remainder_autopsy.json` |

### Cohorts (locked)

| Cohort | Definition |
|--------|------------|
| **FO_REMAINDER** | the **22** FO ids still wrong under #32 outdeg0 gate (`fo_still_wrong_ids`) |
| **FO_KILLED** | the **23** FO ids killed by outdeg0 (`fo_killed_ids`) |
| **OK_HN** | ens-correct ∩ hop==-1 (from #31) |
| **FC_HN** | ens-wrong ∩ hop==-1 ∩ FAIL_CLOSED (from #31) |

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29/#30/#31/#32)

| Symbol | Value |
|--------|-------|
| `CONF_THRESH` | **0.80** |
| `D_CLOSED_MIN` | **0.10** |
| `EPI_CLOSED_MIN` | **0.15** nats |

### Coverage threshold (LOCKED)

A local-sound cut is **actionable** iff it is sound **and** covers **≥ 8/22** of FO_REMAINDER. Below that → do not implement overlay; prefer `LOCAL_SOUND_WALL` / `DIFFUSE` / `INCONCLUSIVE` as appropriate.

### Autopsy / cycle verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`STRUCTURAL_CLUSTER_REMAINDER`** | Remainder FO still concentrate in an identifiable structural cluster vs controls (characterization) |
| **`DIFFUSE`** | Remainder scattered; no dominant structural bin vs controls |
| **`LOCAL_SOUND_CUT_FOUND`** | Sound local cut covers ≥8/22; overlay implemented (or ready) |
| **`LOCAL_SOUND_WALL`** | No sound local cut covers ≥8/22; remainder requires multi-hop reasoning; **stop overlay chase** for this core |
| **`INCONCLUSIVE`** | Evidence insufficient / mixed |

### Gate overlay verdicts (LOCKED — only if gate run)

| Label | Meaning |
|-------|---------|
| **`FO_REMAINDER_KILLED`** | ≥8/22 remainder FO eliminated **AND** no COLLATERAL_HARM |
| **`FO_REMAINDER_PARTIAL`** | ≥1 remainder FO killed but not FO_REMAINDER_KILLED; no COLLATERAL_HARM |
| **`COLLATERAL_HARM`** | matched-OOD overall/HN/K16 **or** OK_HN / positives drop ≥ **0.05** abs vs outdeg0-stacked baseline |

Priority: collateral harm → `COLLATERAL_HARM`. Still **MEASURE**, never OPEN.

**Even a full remainder kill does not widen `science_open` / does not claim hop-OOD OPEN.**

### Prohibited defaults

- No full BFS / reachability oracle as gate features
- No training / bag retrain / multi-hyp / select / upsample / distill
- No T beyond 16
- No `science_open` widen; no §22 widen
- No claiming hop-OOD OPEN
- No parking whole stalk corridor unless collateral harm forces STOP on **this overlay**
- No hardcoded desired labels into model init
- Do not regenerate `ood_hops.jsonl`
- Do not chase further overlays if `LOCAL_SOUND_WALL`

## Explicit non-goals

- No training / bag noise / #27 chase
- No soft distill / multi-hyp / SWA re-chase
- No select / curriculum reopen
- No T>16
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen
- No claim that hop-OOD metrics extend §22 OPEN
- No BFS-oracle gate (`n_reach_from_s==1`) as overlay feature
- No further overlay chase after LOCAL_SOUND_WALL

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_hn_fo_remainder_autopsy.json` |
| **Log** | `artifacts/stalk_hn_fo_remainder_autopsy_run.log` |
| **Prereg SHA** | `bd7e440` |
| **Harness SHA** | `a098584` |
| **Results SHA** | _(stamped after results commit)_ |
| **Cycle verdict** | **`LOCAL_SOUND_WALL`** |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Residue update** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL`** |
| **Gate overlay** | **not run** (best sound coverage 5/22 < 8) |
| **Elapsed** | ~0.09 s CDT |
| **Cite #32 replicate** | rem=22 / kil=23 / union=45 exact |

### Contrast medians (ood_hops T16)

| Feature | FO_REMAINDER (22) | FO_KILLED (23) | OK_HN (16) | FC_HN (179) |
|---------|-------------------|----------------|------------|-------------|
| **outdeg_s** | **1.5** | **0** | 2 | 1 |
| **indeg_t** | 1 | 2 | **0** | 1 |
| **n_reach_from_s** | **19** | **1** | **19** | **19** |
| **max_dist_from_s** | **6** | **0** | 6 | 6 |
| **n_reach_to_t** | 2 | **27** | 1 | 6 |
| frac_reach_from_s | 0.594 | 0.031 | 0.594 | 0.594 |

Remainder is **not** the isolated-source cluster (that was FO_KILLED). Remainder out-closure overlaps OK_HN/FC_HN (multi-hop unpaid hard-negs). Characterization tag: `DIFFUSE_MULTI_HOP_LIKE_OK_FC`.

### Local-sound cut search (gate features only)

| Cut | Sound (viol=0)? | Remainder coverage | Actionable (≥8/22)? |
|-----|-----------------|--------------------|---------------------|
| `C_outdeg0` | yes (59 trig) | **0/22** (already applied) | no |
| `C_indeg_t0` | yes (56 trig) | **0/22** | no |
| `C_deadend_nbrs` | yes (15 trig) | **5/22** | no |
| `C_outdeg1_deadend` | yes (15 trig) | **5/22** | no |

Best sound coverage **5/22 < 8** → **no overlay**. `indeg(t)==0` covers 16/16 OK_HN but **0** remainder (remainder have indeg_t≥1).

### Reading (fail-closed)

1. **LOCAL_SOUND_WALL:** no local-sound constraint computable from Â without full-graph BFS covers ≥8/22 of the outdeg>0 FO remainder.
2. The #32 outdeg0 gate already extracted the local-incidence half of the #31 isolated-source cluster. Remaining 22 require **multi-hop** reasoning (median `|R_out(s)|=19`, `max_dist=6`).
3. **Stop overlay chase** for this FO core. Prefer #14 + #22 on matched-OOD only. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Do **not** BFS-gate. No train. Corridor not parked — MEASURE wall only.
