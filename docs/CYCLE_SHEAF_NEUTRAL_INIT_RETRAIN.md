# CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN — MEASURE plan (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only until human seal |
| **science_open** | **false** (always in harness) |
| **Trigger** | `SEALS_COMPROMISED_INIT_BAKE_IN` — `artifacts/sheaf_untrained_control_audit.json` |
| **Prior INVALIDATION** | `docs/SESSION-SEAL-SHEAF-INFERENCE.md` §16; dense-context seal §8 |

## Goal

Retrain SheafInferCore **without** reachability-baking init so that any Gate1 / OOD / RED_TEST success is attributable to learning, not `_init_specials`.

## Neutral init (proposed; lock before run)

| Knob | Current (INVALID) | Neutral proposal |
|------|-------------------|------------------|
| edge_encoder last | W=0, bias=+4 | W ~ N(0,0.02) or Xavier; **bias=0** (or −1) |
| absent_bias | −4 | keep −4 **or** 0 (document); must not alone imply listed=ON |
| self logit | +8 hardcoded | keep (structural self-loop) **or** learn; document |
| Φ W_msg / W_out | I | Xavier / small; MLP not forced 0 |
| residual_alpha | 1.0 | keep 1.0 or 0.5; document |
| head | energy col ±1, bias ±5 | standard Linear(d+1,2) Xavier **or** Linear(d,2) without energy oracle bias |

## Protocol

1. Implement `neutral_init=True` flag on `SheafInferCore` (default **False** preserves reproducibility of compromised seals as historical artifacts).
2. Run Gate0 overfit + Gate1 id_2k (same hparams as sealed: d=64, mlp×12, T=6, lr=1.5e-3, 30 ep) with `neutral_init=True`.
3. **Mandatory** `run_sheaf_untrained_control` before claiming any PASS:
   - Expect untrained matched-OOD T16 overall ~ chance / ≪ trained.
   - Prediction agreement with trained ≪ 1.0.
4. Re-eval matched-OOD prereg floors (hard-neg≥0.95, K16≥0.75 at T16) only if step 3 passes.
5. Optional: RED_TEST K20 + dense-context with **new** ckpt (not frozen compromised Gate1).
6. Harness stamps `science_open: false`. Human seal may open only if untrained_control + prereg both pass.

## Fail-closed

- If neutral-init trained fails prereg → STOP; do not revive learned OPEN from bake-in seals.
- If untrained still ≈ trained → bake-in not fully removed; iterate init; no OPEN.
- Do not rewrite invalidated seal bodies; append MEASURE residue only.
