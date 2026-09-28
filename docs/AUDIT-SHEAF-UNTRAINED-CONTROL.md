# AUDIT — SheafInferCore untrained control (fail-closed)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Audited main SHA** | `4083d43` (`4083d4316601ab9ca94058cc61f25bab8d7cb66b`) |
| **Verdict** | **`SEALS_COMPROMISED_INIT_BAKE_IN`** |
| **Artifact** | `artifacts/sheaf_untrained_control_audit.json` |
| **Harness** | `python -m reachability_gen.run_sheaf_untrained_control` |
| **Seal INVALIDATION** | `docs/SESSION-SEAL-SHEAF-INFERENCE.md` §16; `docs/SESSION-SEAL-SHEAF-DENSE-CONTEXT.md` §8 |
| **Next MEASURE** | `docs/CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN.md` |

**First sentence:** Untrained SheafInferCore reproduces sealed Gate1 / K16 / hard-neg **1.000** with **100%** prediction agreement vs `artifacts/sheaf_infer_gate1_best.pt` because `_init_specials` hard-codes listed edges ON (bias=+4) and energy readout = reachability indicator — **science_open "learned" seals are INVALID**.

Do not invent. Prefer truth over prior OPEN.
