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

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_stabilize_v2.json` |
| **Log** | `artifacts/stalk_stabilize_v2_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_stabilize_v2_seed{0..4}_best.pt` |
| **Verdict** | **`STOP_FRAGILE`** (seed 1/5; K16 mean 0.423≪0.75; under-propagation return) |
| **science_open** | **false** (not widened) |
| **Elapsed** | ~1039 s (~17.3 min CDT) |
| **Individual prereg** | **1/5** seeds PASS |
| **Mean floors** | HN **0.955≥0.95 PASS**; K16 **0.423≥0.75 FAIL** |

### Per-seed matched-OOD T16

| Seed | best_ep | sel joint HN/ov | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|---------|-----------------|---------|----------|----|-----|-----|--------|
| 0 | 53 | 0.998 / 1.000 / 0.995 | 0.838 | **1.000** | 1.000 | 0.963 | **0.062** | FAIL (K16) |
| 1 | 25 | 0.973 / 0.975 / 0.968 | 0.802 | **1.000** | 0.800 | 0.938 | **0.075** | FAIL (K16) |
| 2 | 39 | 0.964 / 0.965 / 0.963 | **0.983** | **1.000** | 1.000 | 1.000 | **0.900** | **PASS** |
| 3 | 16 | 0.966 / 0.975 / 0.945 | 0.560 | **1.000** | 0.188 | 0.087 | **0.088** | FAIL (K16) |
| 4 | 24 | 0.931 / 0.935 / 0.922 | 0.883 | 0.775 | 1.000 | 0.988 | **0.988** | FAIL (HN) |
| **mean±std** | — | — | **0.813±0.157** | **0.955±0.101** | — | — | **0.423±0.477** | **1/5** |

### Untrained control (per seed, T16)

| Seed | u overall | u hard-neg | u K16 | agree vs trained |
|------|-----------|------------|-------|------------------|
| 0 | 0.633 | 0.600 | 1.000 | 0.483 |
| 1 | 0.608 | 0.883 | 0.000 | 0.785 |
| 2 | 0.635 | 0.271 | 1.000 | 0.619 |
| 3 | 0.346 | 0.692 | 0.000 | 0.785 |
| 4 | 0.219 | 0.438 | 0.000 | 0.265 |
| **mean±std** | **0.488±0.194** | 0.577±0.235 | — | **0.587±0.220** |

Untrained remains mid/low vs trained where trained still propagates; bake-in still **not** proven. K16 collapse on trained seeds 0/1/3 is a **selection** failure (HN-heavy), not bake-in.

### Degree-balanced T16 (secondary)

| Seed | overall | hard-neg | K16 |
|------|---------|----------|-----|
| 0 | 0.836 | 1.000 | 0.062 |
| 1 | 0.801 | 1.000 | 0.075 |
| 2 | 0.983 | 1.000 | 0.900 |
| 3 | 0.558 | 1.000 | 0.087 |
| 4 | 0.834 | 0.675 | 0.988 |
| **mean±std** | 0.802±0.153 | 0.935±0.145 | 0.422±0.477 |

### Causal horizon mean±std (matched-OOD)

| T | overall | hard-neg | K16 |
|---|---------|----------|-----|
| 6 | 0.540±0.068 | 0.960±0.053 | 0.003±0.006 |
| 8 | 0.621±0.077 | 0.957±0.058 | 0.020±0.019 |
| 12 | 0.721±0.151 | 0.964±0.076 | 0.210±0.372 |
| **16** | **0.813±0.157** | **0.955±0.101** | **0.423±0.477** |

### vs PR #14 stabilize + PR #12

| Metric @ T16 | PR #12 (n=3) | PR #14 (n=5) | **This V2 (n=5)** |
|--------------|--------------|--------------|-------------------|
| overall | 0.803±0.202 | **0.929±0.035** | 0.813±0.157 |
| hard-neg | 0.918±0.142 | **0.957±0.061** | 0.955±0.101 |
| K16 | 0.654±0.524 | **0.863±0.143** | **0.423±0.477** |
| seed PASS | 1/3 | **2/5** | **1/5** |

### Reading (fail-closed)

Stronger HN weight (0.7) with overall≥0.85 gate **did** rescue seed2 HN (0.871→1.000) but **reintroduced under-propagation** on seeds 0/1/3: ID-val overall stayed ≥0.85 while matched-OOD **K16 collapsed** (0.06–0.09). Gate on ID overall is **insufficient** as a K16 proxy. Seed4 still misses HN (0.775).

**Prefer #14 joint 0.5/0.5 corridor** over this V2 selection for any future harden. Do **not** chase further HN-primary variants without a true long-hop ID proxy (none exists on current ID val hops).

Param count **117506** within ±5% of FF 121218 (parity ok all seeds). No selection_fallback fired (all seeds had eligible epochs).

## Verdict

**`STOP_FRAGILE`** — seed-wise **1/5**; K16 mean **0.423** fails floor and is deep-fragile (≪0.5 with ≤1/5 pass). Mean HN still clears 0.95 but is hollow without propagation.

Stay **MEASURE** / stop this knob line. **`science_open=false`**. Do not widen. Flag: next middle-out should **not** increase HN weight further; prefer #14 protocol or orthogonal knobs (e.g. longer train under **0.5/0.5**, or accept MEASURE_STILL).

## Policy

- Prefer honesty over lonely OPEN.
- Sheaf unsupervised path is out of scope.
- Append MEASURE to ledger; `science_open` remains **false**.
- If ≥4/5 clear → verdict `PASS_CANDIDATE_FOR_OPEN` for human only; harness keeps `science_open=false`.
