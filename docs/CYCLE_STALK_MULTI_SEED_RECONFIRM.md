# CYCLE_STALK_MULTI_SEED_RECONFIRM — MEASURE results (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — fail-closed |
| **science_open** | **false** (always in harness; human seal only) |
| **Trigger** | PR #11 stalk untrained control → `OPEN_STILL_CONTINGENT_NEEDS_MULTI_SEED` |
| **Base after PR #11** | `7f1037a750cb078d66521dd2ca867010f0d85f7e` |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sealed single-seed OPEN** | PR #2 / `docs/SESSION-SEAL-STALK-LOCALIZATION.md` §6 |

## Goal

Reconfirm stalk-local FractalCore (hard `A_ij`, stalk@s / probe@t, no `c` broadcast,
no soft ACT) across **≥3 seeds** on matched-OOD T∈{6,8,12,16}. Report mean±std
overall / hard-neg / K16; untrained control each seed; degree-balanced secondary.
Compare to sealed single-seed OPEN Gate1. **Do not silently widen** §6 claim.

## Protocol

```bash
python -m reachability_gen.run_stalk_multi_seed_reconfirm
# or: reachability-stalk-multi-seed-reconfirm
```

| Item | Spec |
|------|------|
| Architecture | Sealed stalk OPEN: local stalk/probe, hard A, discrete T, no soft ACT |
| Train | `data/id_2k.jsonl` × 30 ep; lr=1.5e-3; clip=2.5; d=64; mlp×10; T_train=6 |
| Seeds | 0, 1, 2 |
| Eval | matched-OOD T∈{6,8,12,16}; degree-balanced if available |
| Untrained | Fresh init per seed vs trained agreement |
| **Prereg (mean)** | hard-neg ≥ **0.95** AND K16@T16 ≥ **0.75** |
| PASS → | MEASURE reconfirm note only (no science_open widen) |
| FAIL → | contingent OPEN **at risk**; append note |

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_multi_seed_reconfirm.json` |
| **Log** | `artifacts/stalk_multi_seed_reconfirm_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_reconfirm_seed{0,1,2}_best.pt` |
| **Verdict** | **`OPEN_CONTINGENT_AT_RISK`** |
| **science_open** | **false** |
| **Elapsed** | ~213 s (CDT) |
| **Individual prereg** | **1/3** seeds pass |

### Per-seed matched-OOD T16

| Seed | val | overall | hard-neg | K8 | K12 | K16 | prereg |
|------|-----|---------|----------|----|-----|-----|--------|
| 0 | 0.985 | **0.977** | **1.000** | 0.938 | 1.000 | **0.925** | **PASS** |
| 1 | 0.973 | 0.581 | 1.000 | 0.363 | 0.075 | **0.050** | FAIL (K16) |
| 2 | 0.965 | 0.850 | **0.754** | 0.913 | 0.938 | 0.988 | FAIL (hard-neg) |
| **mean±std** | 0.974±0.010 | **0.803±0.202** | **0.918±0.142** | 0.738±0.325 | 0.671±0.517 | **0.654±0.524** | 1/3 |

Prereg means: hard-neg **0.918 < 0.95** FAIL; K16 **0.654 < 0.75** FAIL.

### Untrained control (per seed, T16)

| Seed | u overall | u hard-neg | u K16 | agree vs trained |
|------|-----------|------------|-------|------------------|
| 0 | 0.633 | 0.600 | 1.000 | 0.610 |
| 1 | 0.608 | 0.883 | 0.000 | 0.719 |
| 2 | 0.635 | 0.271 | 1.000 | 0.648 |
| **mean±std** | **0.626±0.015** | 0.585±0.307 | — | **0.659±0.055** |

Untrained remains mid (~0.63); bake-in still **not** proven. Instability is in **trained** multi-seed generalization, not init bake-in.

### Degree-balanced T16 (secondary)

| Seed | overall | hard-neg | K16 |
|------|---------|----------|-----|
| 0 | 0.977 | 1.000 | 0.925 |
| 1 | 0.579 | 1.000 | 0.050 |
| 2 | 0.832 | 0.717 | 0.988 |
| **mean±std** | 0.796±0.202 | 0.906±0.163 | 0.654±0.524 |

### Causal horizon mean±std (matched-OOD)

| T | overall | hard-neg | K16 |
|---|---------|----------|-----|
| 6 | 0.594±0.081 | 0.908±0.159 | 0.004±0.007 |
| 8 | 0.673±0.130 | 0.906±0.164 | 0.308±0.523 |
| 12 | 0.761±0.140 | 0.911±0.154 | 0.342±0.560 |
| **16** | **0.803±0.202** | **0.918±0.142** | **0.654±0.524** |

### vs sealed single-seed OPEN (seed-0 Gate1)

| Metric @ T16 | Sealed OPEN | Reconfirm mean±std | Δ |
|--------------|-------------|--------------------|---|
| overall | **0.977** | 0.803±0.202 | −0.174 |
| hard-neg | **1.000** | 0.918±0.142 | −0.082 |
| K16 | **0.925** | 0.654±0.524 | −0.271 |

Seed 0 reproduces the sealed run; seeds 1–2 do **not**. Single-seed OPEN is **not** robust under reconfirm.

## Verdict

**`OPEN_CONTINGENT_AT_RISK`** — prereg mean floors miss; only **1/3** seeds pass.
Stalk §6 OPEN remains **standing but contingent at risk**. Append-only note on seal.
**Do not** silently widen `science_open`. Prefer truth over prior OPEN. Human review
required before any claim update or revoke.

Param count **117506** within ±5% of FF 121218 (parity ok all seeds).
