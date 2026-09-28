# CYCLE_STALK_STABILIZE_V2 — MEASURE (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — fail-closed / middle-out continue |
| **science_open** | **false** (always in harness; **not widened**; human seal only) |
| **Trigger** | PR #14 `MEASURE_STILL`: mean floors PASS; seed **2/5** (goal ≥4/5) |
| **Base** | `main` tip `61314a0` (merge of PR #14) |
| **Prior cycle** | `docs/CYCLE_STALK_STABILIZE_MULTI_SEED.md` / `artifacts/stalk_stabilize_multi_seed.json` |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** (STOP/INVALID path) |

## Goal (middle-out)

Raise **seed PASS fraction** toward ≥4/5 without changing claim shape:
supervised/hard `A_ij` + stalk-local FractalCore. Same prereg floors as #14.
Do **not** chase unsupervised sheaf Â. Harness never self-stamps `science_open=true`.

## Prior MEASURE_STILL (cite)

| Metric @ matched-OOD T16 | #14 (n=5) |
|--------------------------|-----------|
| overall mean±std | 0.929±0.035 |
| hard-neg mean±std | 0.957±0.061 **PASS** (≥0.95) |
| K16 mean±std | 0.863±0.143 **PASS** (≥0.75) |
| seed PASS (HN≥0.95 ∧ K16≥0.75) | **2/5** (goal ≥4/5 **FAIL**) |
| Untrained overall mean | ~0.488 |

Fails: seed2 HN 0.871; seed3 K16 0.625; seed4 HN 0.913.
HN-primary alone aborted structurally in #14 (under-propagation: HN↑ while positives/K16 collapse).

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_stabilize_v2
# or: reachability-stalk-stabilize-v2
```

| Item | Spec (locked) |
|------|----------------|
| Architecture | Sealed stalk: local stalk@s / probe@t, **hard A**, discrete T, no `c` broadcast, no soft ACT |
| Train data | `data/id_2k.jsonl` (unchanged) |
| Eval | matched-OOD T∈{6,8,12,16} — same floors / substrate as PR #12/#14 |
| Degree-balanced | Secondary table if available |
| Untrained control | Fresh init per seed (cheap; include) |
| **Seeds** | **0, 1, 2, 3, 4** (n=5; same set as #14 for one-protocol table) |
| **Epochs** | **90** (vs #14 60) |
| **LR** | Cosine anneal **1.5e-3 → 1.0e-4** over 90 ep (AdamW, wd=0.01) |
| **Grad clip** | **2.5** |
| **T_train** | 6 |
| **d / mlp** | 64 / ×10 |

### Checkpoint selection rule (prereg — NO test peeking)

ID `val` has hops ∈ {−1,2..6} only — **no K16 proxy on ID**.

Each epoch after train step:

1. Eval ID **val** at **T=16**.
2. Record `sel_hard_neg` and `sel_overall`.
3. **Eligibility gate** (structural fix for HN-primary abort): epoch is eligible iff
   `sel_overall ≥ 0.85`. (Blocks under-propagation ckpts that max HN while path competence collapses.)
4. Among eligible epochs: `joint = 0.7 * sel_hard_neg + 0.3 * sel_overall`.
5. **Best ckpt** = argmax lexicographic `(joint, sel_hard_neg, sel_overall, −epoch)`  
   (stronger HN weight than #14's 0.5/0.5; prefer earlier on ties).
6. If **no** epoch ever eligible: fall back to argmax `(sel_overall, sel_hard_neg, −epoch)` and record `selection_fallback=true`.
7. Matched-OOD / degree-balanced / K16 are **never** used for selection — only for final report.

**Rationale vs #14 abort:** pure HN-primary selected ep10 with HN=1.0 / ov≈0.78 → OOD K16=0 / overall≈0.50.
Gate keeps path competence; 0.7 HN weight targets seed2/seed4 HN misses without reopening under-propagation.

**Not used this cycle (documented out):** patience early-stop (seed0 #14 best was late ep58 — patience risks cutting late consolidation); 7-seed expand (prefer same 5-seed table under one protocol); HN-primary without gate (structurally aborted).

### Prereg floors (match PR #12 / #14)

| Floor | Threshold |
|-------|-----------|
| Mean hard-neg @ matched-OOD T16 | **≥ 0.95** |
| Mean K16 @ matched-OOD T16 | **≥ 0.75** |
| Seed-wise PASS (HN≥0.95 **and** K16≥0.75) | Goal **≥ 4/5** (stretch **5/5**) |

PASS definition identical to #14.

### Verdict map (fail-closed)

| Outcome | Label | science_open |
|---------|-------|--------------|
| Mean floors PASS **and** ≥4/5 seed PASS | `PASS_CANDIDATE_FOR_OPEN` | stays **false**; flag Fractal-1 / human — do **not** auto-widen |
| Mean floors miss **or** &lt;4/5 but not collapse chaos | `MEASURE_STILL` | **false** |
| Pathological fragility (e.g. ≤1/5 or K16 mean ≪0.5) | `STOP_FRAGILE` | **false** |

**Never** set `science_open=true` from this harness. OPEN revive = human-only after review.

## Harden knobs used (vs PR #14)

| Knob | PR #14 | This V2 |
|------|--------|---------|
| Seeds | 0..4 | **0..4** (same) |
| Epochs | 60 | **90** |
| LR | cosine 1.5e-3→1.5e-4 | **cosine 1.5e-3→1.0e-4** |
| Grad clip | 2.5 | 2.5 |
| Ckpt select | joint 0.5·HN+0.5·overall @ T16 ID-val | **joint 0.7·HN+0.3·overall** among epochs with **overall≥0.85** |
| Architecture / hard Â | sealed stalk | **unchanged** |

## Results (filled after run)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_stabilize_v2.json` |
| **Log** | `artifacts/stalk_stabilize_v2_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_stabilize_v2_seed{0..4}_best.pt` |
| **Verdict** | _TBD_ |
| **science_open** | **false** |
| **Elapsed** | _TBD_ |
| **Individual prereg** | _TBD_ |
| **Mean floors** | _TBD_ |

### Per-seed matched-OOD T16

_Filled after run._

## Policy

- Prefer honesty over lonely OPEN.
- Sheaf unsupervised path is out of scope.
- Append MEASURE to ledger; `science_open` remains **false**.
- If ≥4/5 clear → verdict `PASS_CANDIDATE_FOR_OPEN` for human only; harness keeps `science_open=false`.
