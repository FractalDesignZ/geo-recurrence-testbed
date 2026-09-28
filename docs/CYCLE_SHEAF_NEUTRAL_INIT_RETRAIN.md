# CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN — MEASURE plan (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE only until human seal |
| **science_open** | **false** (always in harness) |
| **Trigger** | `SEALS_COMPROMISED_INIT_BAKE_IN` — `artifacts/sheaf_untrained_control_audit.json` |
| **Prior INVALIDATION** | `docs/SESSION-SEAL-SHEAF-INFERENCE.md` §16; dense-context seal §8 |
| **PR #7 merge** | `034a074` (INVALIDATION + untrained control landed on main) |

## Goal

Retrain SheafInferCore **without** reachability-baking init so that any Gate1 / OOD / RED_TEST success is attributable to learning, not `_init_specials`.

## Neutral init (locked)

| Knob | Legacy (INVALID) | Neutral (`neutral_init=True`) |
|------|------------------|-------------------------------|
| edge_encoder last | W=0, bias=+4 | W ~ N(0,0.02); **bias=0** |
| edge_encoder first | default | Xavier; bias=0 |
| absent_bias | −4 | **0** |
| self logit | +8 hardcoded | **keep** (structural self-loop) |
| Φ W_msg / W_out | I | Xavier |
| Φ MLP | forced 0 | N(0,0.02) |
| stalk_proj | I | Xavier |
| residual_alpha | 1.0 | **keep 1.0** |
| head | energy col ±1, bias ±5 | Xavier Linear(d+1,2); bias=0 |

Default `neutral_init=False` preserves reproducibility of compromised seals as historical artifacts.

## Gates A–D (fail-closed)

| Gate | Requirement |
|------|-------------|
| **A** | Neutral untrained matched-OOD reachability ≈0.50; Â FNR ≈ chance (listed edges not all ON). `absent_bias=0` ⇒ hard-gate Â FPR≈0 by construction (document; not bake-in). Disqualify if reachability above chance. |
| **B** | Bake-in removed: no edge+4, absent−4, energy±5. STE/Gumbel OK for learning Â. |
| **C** | Degree-balanced hard-negs (match endpoint out_s×in_t bins to positives). Reach-cue audit: single-feature degree/reach rules ≤0.52 on **degree-balanced** eval. Artifact: `artifacts/sheaf_reach_cue_audit.json`. Raw OOD may still carry degree cues — document; prefer balanced secondary metrics for trust. |
| **D** | ≥3 seeds; report mean±std. Param parity ~121k ±5%. |

## Protocol (harness)

```bash
python -m reachability_gen.run_sheaf_neutral_init_retrain
# or: reachability-sheaf-neutral-init-retrain
```

1. Prove Gate A on neutral untrained.
2. Train SheafInferCore from neutral init on `data/id_2k.jsonl` (d=64, mlp×12, T=6, lr=1.5e-3, 30 ep); log untrained vs best.
3. Eval matched-OOD at T∈{6,8,12,16} with reach FPR/FNR + Â FPR/FNR; prereg: hard-neg≥0.95 and K16≥0.75 at T16. If K16 incomplete, document K8 scope — prefer complete gates over fake OPEN.
4. Artifacts JSON; docs update; branch+PR.

## Fail-closed

- If Gate A/B/C fail → STOP; no OPEN.
- If neutral-init trained fails prereg → STOP with residue; do not revive learned OPEN from bake-in seals.
- If untrained still ≈ trained → bake-in not fully removed; iterate init; no OPEN.
- Harness stamps `science_open: false`. Human seal may open only if untrained≪trained and prereg both pass.
- Do not rewrite invalidated seal bodies; append MEASURE residue only.

## Artifacts

| Path | Contents |
|------|----------|
| `artifacts/sheaf_neutral_init_retrain.json` | Full Gates A–D report, per-seed tables, mean±std, verdict |
| `artifacts/sheaf_reach_cue_audit.json` | Gate C cue ceilings + degree-balanced construction |
| `artifacts/sheaf_neutral_gate1_seed{0,1,2}_best.pt` | Per-seed checkpoints |

## Results (this run — cite artifact; science_open=false)

| Field | Value |
|-------|-------|
| **PR #7 merge SHA** | `034a0742a143cd0f16384509e529a78282b8884c` |
| **Artifact** | `artifacts/sheaf_neutral_init_retrain.json` |
| **Verdict** | `MEASURE_CANDIDATE_PASS_FLOORS` |
| **science_open** | **false** |

### Gate A (neutral untrained, matched-OOD)

| Metric | Value |
|--------|-------|
| reach T16 | **0.500** |
| reach FPR / FNR | 0.0 / 1.0 |
| Â FPR / FNR | 0.0 / **0.508** |
| bake-in removed | edge+4 / absent−4 / energy±5 all cleared |

### Gate C (reach-cue)

| Set | max degree/reach rule | acc | ≤0.52? |
|-----|----------------------|-----|--------|
| matched-OOD raw | `in_t` | 0.738 | **no** (document) |
| degree-balanced OOD | `out_s` | **0.503** | **yes** |

### Gate D — per-seed matched-OOD T16 + mean±std

| Seed | val | T16 overall | hard-neg | K16 | FPR | FNR | untrained val |
|------|-----|-------------|----------|-----|-----|-----|---------------|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0.570 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0.675 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0.565 |
| **mean±std** | 1.000±0 | 1.000±0 | 1.000±0 | 1.000±0 | 0±0 | 0±0 | 0.603±0.062 |

Untrained vs trained seed0 T16 prediction agreement = **0.500** (untrained≪trained).

### Gate0 residue

Acc/recon/per-class PASS by ~step 10; CE < 1e-3 **not** met in 150-step legacy budget (needs ~787 @ lr=3e-2). Documented — not silent PASS.

### Human seal gate

Do **not** stamp `science_open=true` without human review. Aux edge-recon recovers Â in epoch 1 (supervised gold edges, train-only) — distinct from init bake-in, but attribution still MEASURE.

