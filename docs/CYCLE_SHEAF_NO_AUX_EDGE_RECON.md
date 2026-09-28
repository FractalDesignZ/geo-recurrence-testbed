# CYCLE_SHEAF_NO_AUX_EDGE_RECON — MEASURE plan + results (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only — fail-closed |
| **science_open** | **false** (always in harness) |
| **Trigger** | PR #8 with-aux MEASURE candidate — isolate aux edge-recon necessity |
| **PR #8 merge SHA** | `e0877ebf52a16efd81df1c80ca6fb3f1c3289a31` |
| **With-aux baseline** | `artifacts/sheaf_neutral_init_retrain.json` |

## Goal

Retrain SheafInferCore with **identical** Gates A–D / hparams / param parity as
`CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN`, except **aux edge-recon loss OFF**
(`edge_recon_weight=0`). Train on reachability CE only. Compare matched-OOD T16
(overall, hard-neg, K8/12/16, Â FPR/FNR) to PR #8 with-aux.

## Protocol

```bash
python -m reachability_gen.run_sheaf_no_aux_edge_recon
# or: reachability-sheaf-no-aux-edge-recon
```

| Gate | Requirement |
|------|-------------|
| **A** | Neutral untrained matched-OOD reachability ≈0.50; Â FNR ≈ chance |
| **B** | Neutral init (no edge+4 / absent−4 / energy±5) |
| **C** | Degree-balanced hard-negs; reach-cue ≤0.52 on balanced set |
| **D** | ≥3 seeds; same prereg floors as neutral retrain: hard-neg≥0.95 **and** K16≥0.75 at T16 |

If floors fail → **STOP** with residue. Never stamp `science_open`.

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/sheaf_no_aux_edge_recon.json` |
| **Log** | `artifacts/sheaf_no_aux_edge_recon_run.log` |
| **Verdict** | **`STOP_LEARNING_FAIL`** |
| **science_open** | **false** |
| **Elapsed** | ~238 s (CDT) |

### Gates A–C

| Gate | Result |
|------|--------|
| **A** | reach T16=**0.500**; Â FNR≈0.508; bake-in cleared. **PASS** |
| **B** | neutral_init flags cleared. **PASS** |
| **C** | degree-balanced cue max=`out_s` **0.503** ≤0.52. **PASS** |

### Gate0 (aux OFF)

Balanced overfit 150 steps: final acc≈0.53, y1≈0.06 — **did not** meet CE&lt;1e-3 + per-class. Documented (no recon floor with aux OFF).

### Gate D — per-seed matched-OOD T16

| Seed | val | overall | hard-neg | K8 | K12 | K16 | FPR | FNR | Â FPR | Â FNR |
|------|-----|---------|----------|----|-----|-----|-----|-----|-------|-------|
| 0 | 0.648 | 0.500 | 1.000 | 0 | 0 | 0 | 0 | 1 | 0 | 0.508 |
| 1 | 0.678 | 0.500 | 1.000 | 0 | 0 | 0 | 0 | 1 | 0 | 0.311 |
| 2 | 0.753 | 0.500 | 1.000 | 0 | 0 | 0 | 0 | 1 | 0 | 0.255 |
| **mean±std** | 0.692±0.054 | **0.500±0** | 1.000±0 | **0±0** | **0±0** | **0±0** | 0±0 | **1±0** | 0±0 | **0.358±0.133** |

Param count **123206** within ±5% of FF 121218 (parity ok all seeds).

Hard-neg=1.0 is **trivial** under always-predict-unreach collapse (FNR=1); prereg still requires K16≥0.75 → **FAIL**.

Untrained vs trained seed0 T16 prediction agreement = **1.0** (both collapse to unreach on OOD — no OOD learning signal).

### Comparison to PR #8 with-aux

| Metric (T16 mean) | no-aux | with-aux (PR#8) | Δ |
|-------------------|--------|-----------------|---|
| overall | 0.500 | 1.000 | −0.500 |
| hard-neg | 1.000 | 1.000 | 0 |
| K16 | 0.000 | 1.000 | −1.000 |
| K8 | 0.000 | 1.000 | −1.000 |
| FPR | 0 | 0 | 0 |
| FNR | 1.000 | 0 | +1.000 |
| Â FPR | 0 | 0 | 0 |
| Â FNR | 0.358 | 0 | +0.358 |
| val | 0.692 | 1.000 | −0.308 |

### Quick stalk untrained (report only)

FractalCore stalk untrained matched-OOD T16 overall=**0.633**, hard-neg=**0.600** (param 117506). Context only — not a sheaf Gate.

## Residue / fail-closed

Under `neutral_init` + `gate_detach_diffusion`, **aux edge-recon is necessary** for Â recovery and OOD path-length gen in the locked 30-ep / lr=1.5e-3 budget. CE-only training plateaus on majority/ID cues and collapses OOD to always-unreach.

- Verdict: **`STOP_LEARNING_FAIL`**
- Do **not** stamp `science_open`
- Do **not** revive PR #8 / prior seals as OPEN without human review of aux attribution
- Next MEASURE (if any): longer budget / different gate coupling — still science_open=false until floors + attribution clear

## Artifacts

| Path | Contents |
|------|----------|
| `artifacts/sheaf_no_aux_edge_recon.json` | Full Gates A–D, per-seed, mean±std, PR#8 deltas, stalk control |
| `artifacts/sheaf_no_aux_reach_cue_audit.json` | Gate C cue audit |
| `artifacts/sheaf_no_aux_gate1_seed{0,1,2}_best.pt` | Per-seed checkpoints |
| `artifacts/sheaf_no_aux_edge_recon_run.log` | Full run log |

