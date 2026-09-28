# CYCLE_SHEAF_STE_NO_AUX — MEASURE plan + results (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — fail-closed |
| **science_open** | **false** (always in harness) |
| **Trigger** | PR #9 STOP (`STOP_LEARNING_FAIL`) — no-aux + `gate_detach_diffusion=True` collapsed OOD |
| **PR #9 merge SHA** | `2834256a12a7a1d9ff826738cd860051f71e19ac` |
| **PR #8 with-aux** | `e0877eb` / `artifacts/sheaf_neutral_init_retrain.json` |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |

## Goal

Retrain SheafInferCore with **identical** Gates A–D / neutral_init / param parity /
`edge_recon_weight=0` as PR #9, except enable **end-to-end STE** (or Gumbel-Sigmoid)
into Φ: **`gate_detach_diffusion=False`**. Longer budget (60 ep default). Compare
matched-OOD T16 to PR #8 (with-aux) and PR #9 (no-aux detach).

## Protocol

```bash
python -m reachability_gen.run_sheaf_ste_no_aux
# or: reachability-sheaf-ste-no-aux
# optional: --gate-mode gumbel --epochs 90
```

| Gate | Requirement |
|------|-------------|
| **A** | Neutral untrained matched-OOD reachability ≈0.50; Â FNR ≈ chance |
| **B** | Neutral init (no edge+4 / absent−4 / energy±5); STE/Gumbel OK |
| **C** | Degree-balanced hard-negs; reach-cue ≤0.52 on balanced set |
| **D** | ≥3 seeds; same prereg floors: hard-neg≥0.95 **and** K16≥0.75 at T16 |

| Knob | PR #9 (STOP) | This cycle |
|------|--------------|------------|
| neutral_init | True | True |
| edge_recon_weight | 0 | 0 |
| gate_detach_diffusion | **True** | **False** |
| gate_mode | ste (detached) | **ste** (or gumbel) |
| epochs | 30 | **60** (longer) |

PASS → MEASURE candidate. FAIL → STOP residue. Never stamp `science_open`.

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/sheaf_ste_no_aux.json` |
| **Log** | `artifacts/sheaf_ste_no_aux_run.log` |
| **Verdict** | **`STOP_LEARNING_FAIL`** |
| **science_open** | **false** |
| **Elapsed** | ~501 s (CDT) |
| **Knobs** | neutral_init; edge_recon_weight=0; gate_detach_diffusion=**False**; gate_mode=**ste**; epochs=**60** |

### Gates A–C

| Gate | Result |
|------|--------|
| **A** | reach T16=**0.500**; Â FNR≈0.508; bake-in cleared. **PASS** |
| **B** | neutral_init flags cleared; STE end-to-end enabled. **PASS** |
| **C** | degree-balanced cue max ≤0.52. **PASS** |

### Gate D — per-seed matched-OOD T16

| Seed | val | overall | hard-neg | K8 | K12 | K16 | FPR | FNR | Â FPR | Â FNR | prereg |
|------|-----|---------|----------|----|-----|-----|-----|-----|-------|-------|--------|
| 0 | 0.508 | 0.538 | 0.517 | 0.713 | 0.550 | 0.413 | 0.483 | 0.442 | 1.000 | 0.290 | FAIL |
| 1 | 0.558 | 0.760 | 0.521 | 1.000 | 1.000 | 1.000 | 0.479 | 0.000 | 1.000 | 0.853 | FAIL (hard-neg) |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.001 | **PASS** |
| **mean±std** | 0.688±0.271 | **0.766±0.231** | **0.679±0.278** | 0.904±0.166 | 0.850±0.260 | **0.804±0.339** | 0.321±0.278 | 0.147±0.255 | 0.667±0.577 | 0.381±0.434 | 1/3 |

Param count **123206** within ±5% of FF 121218 (parity ok all seeds).

Untrained vs trained seed0 T16 prediction agreement = **0.479** (learning signal present; not bake-in).

### vs PR #8 / #9

| Metric (T16 mean) | STE no-aux (this) | PR #9 no-aux detach | PR #8 with-aux |
|-------------------|-------------------|---------------------|----------------|
| overall | 0.766 | 0.500 | 1.000 |
| hard-neg | 0.679 | 1.000† | 1.000 |
| K16 | 0.804 | 0.000 | 1.000 |
| FNR | 0.147 | 1.000 | 0 |
| Â FNR | 0.381 | 0.358 | 0 |

† PR #9 hard-neg=1.0 trivial under always-unreach collapse.

STE end-to-end **partially** recovers path-length gen vs PR #9 collapse (seed2 full PASS; seed1 K*=1 but hard-neg≈chance; seed0 weak) — **not** stable across ≥3 seeds → prereg FAIL.

## Residue / fail-closed

Under `neutral_init` + `edge_recon_weight=0` + `gate_detach_diffusion=False` + STE + 60 ep, learning is **unstable** across seeds (1/3 prereg). Do **not** stamp `science_open`. Aux edge-recon (PR #8) remains the only stable MEASURE floor-passer under locked hparams. Optional next: Gumbel temp sweep or longer budget — still science_open=false.

- Verdict: **`STOP_LEARNING_FAIL`**
- Do **not** stamp `science_open`
- Do **not** revive PR #8 / prior seals as OPEN without human review

## Artifacts

| Path | Contents |
|------|----------|
| `artifacts/sheaf_ste_no_aux.json` | Full Gates A–D, per-seed, mean±std, verdict |
| `artifacts/sheaf_ste_no_aux_reach_cue_audit.json` | Gate C cue audit |
| `artifacts/sheaf_ste_no_aux_gate1_seed{0,1,2}_best.pt` | Per-seed checkpoints |
| `artifacts/sheaf_ste_no_aux_run.log` | Full run log |
