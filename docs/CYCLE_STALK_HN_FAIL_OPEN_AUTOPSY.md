# CYCLE_STALK_HN_FAIL_OPEN_AUTOPSY — MEASURE structural autopsy (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** structural autopsy of the #30 `HN_FAIL_OPEN_CORE` (45 hop-OOD hard-neg FAIL_OPEN) |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | #30 `FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED`: ens HN **0.067**; **45** FAIL_OPEN / **190** FAIL_CLOSED; overlays do not repair HN; residue named **`HN_FAIL_OPEN_CORE`**. Distinct from #29 RED (0 FAIL_OPEN). |
| **Base** | `main` tip after PR #30 (`dd18ff8`) |
| **Prior** | #28 ood_hops shatter; #29 RED FAIL_CLOSED; #30 hop-OOD HN overlay stress |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |

## Goal

Structural autopsy of the **45** hop-OOD hard-neg examples where frozen **#14/#18/#22** ens is **FAIL_OPEN** (pairwise D_ex==0, ens_max_prob≥0.80, pred=1 / label=0). **Measure only** — characterize whether FAIL_OPEN concentrates in a structural cluster or is diffuse. Fixed **T=16**. No train. No science_open widen.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| #27 | bag MEASURE_LIFT / CHAOS — **do not train more bag noise** |
| #28 | COMPETENT_vs_CHAOS; ood_hops HN shatter residue |
| #29 | RED FAIL_CLOSED_DOMINANT (0 OPEN) — did **not** pay ood_hops HN |
| #30 | FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED; residue **`HN_FAIL_OPEN_CORE`** (45) |

This cycle = **MEASURE autopsy only**. Not train. Not repair. Not §22 widen. Do **not** park the whole stalk corridor — only autopsy this core.

## Metaphor

- **map ≠ location** — #22 COMPETENT map on matched-OOD; hop-OOD FAIL_OPEN asks *where* the map unifies into confident false-reachability.
- **negatives = mirror** — the 45 are the unpaid residue core; K16 holding and FAIL_CLOSED dominance do not dissolve them.

## Cite #30 (ood_hops T16 — #22 ens)

Locked from `artifacts/stalk_hop_ood_hn.json` / PR #30:

| Metric | Value |
|--------|-------|
| ens HN | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN | **45** (all HN; D=0, conf≥0.80, pred=1 label=0) |
| FAIL_CLOSED | **190** |
| CD | **0.553** |
| gate accepted HN | **0.000** (concentrates FAIL_OPEN) |
| majority_vote HN | **0.067** (no lift) |

## Preregistered questions (LOCKED before runs)

1. **Graph structure of the 45:** K/hops, |V|, |E|, density `p` (stored + empirical), token length, path/reachability vs distractors (R_out(s), R_in(t), deg(s)/deg(t), near-miss bridge count), hard-neg construction tags if present in jsonl (`arm_id`/`arm_meta` — expected None on ood_hops).
2. **Controls:** compare FAIL_OPEN vs FAIL_CLOSED wrongs **and** vs correct HN on the same `ood_hops` slice.
3. **Concentration:** is FAIL_OPEN concentrated in a structural cluster (e.g. near-miss hops, dense distractors, specific |V|/p) or scattered?
4. **Member logits:** all 10 members wrong+confident, or soft-agree at threshold (D_ex==0 with varying conf)?
5. **Verdict enum** (locked labels below) + named residue update.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_hn_fail_open_autopsy
# or: reachability-stalk-hn-fail-open-autopsy
```

| Item | Spec (locked) |
|------|----------------|
| Arms | Frozen **#14/#18** = **#22** ens map seeds **0..9** only |
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** (existing; do not regenerate) @ **T=16** fixed |
| Primary aggregator | ens `prob_mean` (same #30 baseline) |
| Prefer | extract 45 ids from #30 artifact if full FAIL_OPEN detail present; else re-infer (eval-only) for full wrong list + member logits |
| Artifact | `artifacts/stalk_hn_fail_open_autopsy.json` |

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29/#30)

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

### Cohorts (locked)

| Cohort | Definition |
|--------|------------|
| **FO_HN** | ens-wrong ∩ hop==-1 ∩ FAIL_OPEN (the 45 core) |
| **FC_wrong** | ens-wrong ∩ FAIL_CLOSED (control; may include HN + positives) |
| **FC_HN** | ens-wrong ∩ hop==-1 ∩ FAIL_CLOSED |
| **OK_HN** | ens-correct ∩ hop==-1 |

### Structural features (locked)

Per example from encoding + row fields:

- `n` (|V|), `n_edges` (|E|), `p` (stored), `p_emp` = |E|/(n·(n−1))
- `token_len`, `hop_distance`, `y`, `s`, `t`
- `deg_s`, `deg_t` (total in+out)
- `n_reach_from_s` = |R_out(s)|, `n_reach_to_t` = |R_in(t)| (nodes that reach t)
- `frac_reach_from_s` = n_reach_from_s / n, `frac_reach_to_t` = n_reach_to_t / n
- `near_miss_bridges` = count of nodes u∈R_out(s) with an out-edge to some v∉R_out(s) where t is reachable from v (one-edge near-miss)
- `max_dist_from_s` among reachable; construction tags (`arm_id`, `arm_meta`) if any

### Clustering / concentration (locked)

Report FO_HN vs controls on feature means/medians + histogram mass by `n`, `p` bin, `near_miss_bridges>0`. Concentration rule of thumb (not a science claim):

- **STRUCTURAL_CLUSTER** if ≥1 feature shows FO median outside IQR of OK_HN **and** FO shares a dominant bin (≥50% of FO in one `n` or `p` bin that OK_HN does not dominate)
- **DIFFUSE** if FO feature distributions overlap OK_HN/FC_HN without a dominant bin
- **DATA_ARTIFACT** if FO share a construction tag / encoding defect / label inconsistency
- **INCONCLUSIVE** otherwise

### Member-logit question (locked)

On FO_HN: for each example report `n_members_pred1`, `n_members_conf_ge_0.80` (member max-softmax ≥0.80), mean member conf on class-1. Labels:

| Tag | Rule |
|-----|------|
| **HARD_UNANIMOUS** | all 10 members pred=1 **and** all 10 member conf≥0.80 |
| **SOFT_AGREE** | all 10 pred=1 but <10 member conf≥0.80 |
| **MIXED_PRED** | not all members pred=1 (should not occur if D_ex==0) |

### Autopsy verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`STRUCTURAL_CLUSTER`** | FO concentrated in identifiable structural cluster vs controls |
| **`DIFFUSE`** | FO scattered; no dominant structural bin vs controls |
| **`DATA_ARTIFACT`** | FO explained by construction/label/encoding artifact |
| **`INCONCLUSIVE`** | evidence insufficient / mixed |

Cycle composite: `{autopsy_verdict}` e.g. `STRUCTURAL_CLUSTER`. Residue update must rename or refine **`HN_FAIL_OPEN_CORE`** (append subtype), not erase it.

**Never** set `science_open=true`. Do **not** claim HN accuracy repair.

### Prohibited defaults

- No training / bag retrain / multi-hyp / select / upsample / distill
- No T beyond 16
- No `science_open` widen; no §22 widen
- No claiming OPEN or repairing HN accuracy as science claim
- No parking the whole stalk corridor (autopsy this core only)
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
- No invented repair of HN accuracy

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_hn_fail_open_autopsy.json` |
| **Log** | `artifacts/stalk_hn_fail_open_autopsy_run.log` |
| **Prereg SHA** | `931bd78` |
| **Harness SHA** | `3980451` |
| **Results SHA** | _(this commit)_ |
| **Cycle verdict** | **`STRUCTURAL_CLUSTER`** |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Residue update** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER`** — isolated-source / hub-target false-reachability |
| **Elapsed** | ~6.7 s (~0.1 min CDT) |
| **Cite #30 replicate** | **exact** (FO=45 / FC=190 / ens HN=0.067) |

### Cohorts (ood_hops T16; #22 ens)

| Cohort | n | notes |
|--------|---|-------|
| **FO_HN** | **45** | all FAIL_OPEN; all hop=-1; pred=1 label=0 |
| **FC_HN** | **179** | FAIL_CLOSED hard-neg wrongs |
| **FC_wrong** | **190** | all FAIL_CLOSED (incl. 11 non-HN) |
| **OK_HN** | **16** | ens-correct hard-neg |

### Key structural contrasts (medians)

| Feature | FO_HN | OK_HN | FC_HN |
|---------|-------|-------|-------|
| n / p / \|E\| / token_len | 32 / 0.06 / 61 / 189 | **same** (fixed across all HN) | same |
| **n_reach_from_s** | **1** | **19** | 19 |
| **frac_reach_from_s** | **0.031** | **0.594** | 0.594 |
| **n_reach_to_t** | **27** | **1** | 6 |
| **frac_reach_to_t** | **0.844** | **0.031** | 0.188 |
| **max_dist_from_s** | **0** | **6** | 6 |
| deg_t | **4** | **1** | 3 |
| near_miss_bridges (missing) | 27 | 19 | 114 |

Construction tags: all `arm_id`/`arm_meta` = None (no DATA_ARTIFACT). `n`/`p` fixed for all HN — not discriminative. `frontier_exits`≡0 by out-closure definition.

### Concentration

FO dominant bin **`frac_reach_s<0.25`** (28/45) vs OK_HN dominant **`frac_reach_s≥0.25`** (14/16). Cluster = **isolated-source / hub-target**: s out-closure trivial (often only s), t in-closure large — ens unifies confident false-reachability. OK_HN is the **inverse** (large R_out(s), tiny R_in(t)).

### Member logits (FO_HN)

| Tag | count |
|-----|-------|
| **HARD_UNANIMOUS** | **34** |
| **SOFT_AGREE** | **11** |
| MIXED_PRED | 0 |

Mostly all-10 members wrong+confident; 11 soft-agree (all pred=1, some conf<0.80).

### Reading (fail-closed)

1. **STRUCTURAL_CLUSTER confirmed:** HN_FAIL_OPEN_CORE is not diffuse — it is the isolated-source / hub-target hard-neg pattern.
2. **Not a data artifact:** labels/encoding consistent; no construction tags.
3. **Member geometry:** predominantly HARD_UNANIMOUS — unified confident false-reachability, not soft thresholding.
4. Residue refined to **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER`**. Prefer #14 + #22 on matched-OOD only. Do **not** widen §22. No train. Do not park whole stalk corridor — this autopsy only.
