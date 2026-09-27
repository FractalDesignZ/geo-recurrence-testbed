# CYCLE_SHEAF_INFERENCE — MEASURE plan + status

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | MEASURE |
| **Cycle** | `CYCLE_SHEAF_INFERENCE` |
| **science_open** | **false** (until measured + human seal; harness never self-stamps true) |
| **Prior seal** | stalk seal `b144dac` (PR #2 / `docs/SESSION-SEAL-STALK-LOCALIZATION.md`) |
| **Branch** | `cycle/sheaf-inference` |

---

## Motivation

`CYCLE_STALK_LOCALIZATION` opened a **scoped** claim: stalk locality + discrete depth resolves zero-shot path-length gen up to K=16 on directed graphs — but **only** with a hard `A_ij` mask. Residue: topology was handed, not learned.

## Goal

Infer sheaf / restriction maps **F_{u→v}** from edge tokens **without** hard adjacency mask at eval, while preserving:

1. **Zero leakage** on disconnect (Gate0 ‖h_t‖≈0 under inferred `A_hat`).
2. **Stability** under discrete cycle depth (causal horizon T vs K).
3. Param parity hygiene vs FF baseline (±5% of 121218 → [115157, 127279]).

## Architecture (`SheafInferCore`)

1. **Edge token encoder** → `E_hat_u→v` logits (listed tokens); `absent_bias` prior for non-tokens.
2. **Hard gate**: `gate=1{σ(E)>θ}` with θ=0.5; train supports STE / Gumbel-Sigmoid. Default: **detach gate into diffusion** so CE cannot open phantom edges via STE; encoder trained by **aux edge-recon BCE (train only)**.
3. **Stalk diffusion**: local init (`z_s=stalk`, else 0); discrete T unroll over inferred gate with shared bias-free Φ (identity-init message pass + residual MLP).
4. **Readout**: `[z_t ; ‖z_t‖]` → 2-way head (energy channel).
5. **No hard A oracle at eval** — adjacency inferred from tokens only.

## Non-goals (fail-closed)

- Soft ACT reintroduction without a separate MEASURE.
- Unrestricted MLP loops as a substitute for local stalk/probe.
- Stamping `science_open=true` before Gate0/1 artifacts exist / human seal.
- Claiming universal graphs beyond a declared matched substrate.
- End-to-end STE into Φ without documenting disconnect softening.

## MEASURE protocol

### Gate 0
- `tests/test_sheaf_infer_core.py` + `python -m reachability_gen.overfit_sheaf --balanced`
- Param band assert ±5% of 121218
- Balanced 32: reachability CE < 1e-3 and edge recon offdiag acc ≥ 0.99 within ≤150 steps
- Disconnect: ‖h_t‖=0 under `A_hat` across T∈{6,8,12,16} (document if STE softens)

### Gate 1
- Train 30 ep `id_2k`; eval `covariate_matched_ood` T∈{6,8,12,16}
- Prereg for later OPEN consideration: hard-neg ≥ 0.95 AND K16 ≥ 0.75 @ T=16 — report PASS/FAIL honestly
- Artifact: `artifacts/sheaf_infer_matched_ood.json`

## Artifacts

| Slice | Path |
|-------|------|
| Gate0 overfit | `artifacts/sheaf_infer_overfit.json` |
| Gate1 matched-OOD | `artifacts/sheaf_infer_matched_ood.json` |
| Gate1 ckpt | `artifacts/sheaf_infer_gate1_best.pt` |
| Gate1 log | `artifacts/sheaf_infer_gate1_run.log` |

## science_open

**false** until a human seal cites Gate0/1 artifacts. Harness stamps `science_open: false` always.
