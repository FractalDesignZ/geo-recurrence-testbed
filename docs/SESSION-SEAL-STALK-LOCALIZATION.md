# SESSION SEAL — CYCLE_STALK_LOCALIZATION (stalk-local FractalCore Gate0/1)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | MEASURE / FREEZE |
| **Cycle** | `CYCLE_STALK_LOCALIZATION` |
| **science_open** | **true** (scoped claim only — §6) |
| **Repo** | https://github.com/FractalDesignZ/geo-recurrence-testbed |
| **Branch** | `cycle/stalk-localization` |
| **MEASURE SHA** | `1d6c3da` (`1d6c3da87a8387bf83813620f079fc0844fd1c06`) |
| **Merge target** | PR #2 → `main` |
| **Verdict class** | OPEN (scoped science) + FAIL-CLOSED elsewhere |

**Current status (2026-09-27 CDT):** live claim = **`MEASURE` / demoted** (see **§13**). Header/`§6` `science_open=true` is **historical** single-seed seal only — not a live OPEN. Measurement §12 = `OPEN_CONTINGENT_AT_RISK`; demotion prefers honesty over lonely OPEN. **Not INVALID** (PR #11 untrained mid).

Fail-closed outside the single claim in §6. Append-only. Mandelbrot / sheaf metaphor remains aspirational except where metrics are cited.

---

## 1. Scope sealed

| Slice | Artifact | Status |
|-------|----------|--------|
| Gate0 balanced overfit + disconnect stalk-ablation | `artifacts/fractal_core_stalk_overfit.json` | SEALED — **PASS** |
| Gate1 id_2k × 30 ep + matched-OOD (discrete T) | `artifacts/fractal_core_stalk_gate1_matched_ood.json` | SEALED — **Prereg PASS** |
| Gate1 best checkpoint | `artifacts/fractal_core_stalk_gate1_best.pt` | SEALED |
| Gate1 run log | `artifacts/fractal_core_stalk_gate1_run.log` | SEALED |

Architecture (from artifact `architecture`): local stalk@`s` / probe@`t`, intermediates=0, **no** `c` broadcast, **no** soft ACT, discrete `T ∈ {6,8,12,16}`, hard adjacency mask. Param count **117506** within ±5% of FF 121218.

Substrate: `covariate_matched_ood` (`data/covariate_matched_ood.jsonl`, n=480) + hard `A_ij` mask. Prior cycle seal: `docs/SESSION-SEAL-FRACTAL-CORE.md` (broadcast/`soft ACT` leakage → this cycle).

---

## 2. Gate0 — PASS + disconnect invariant

Balanced 32 (16 y=0 hard-neg + 16 y=1), T=6, discrete halt (`adaptive_halt=false`).

| Metric | Value |
|--------|-------|
| final_acc | **1.0** |
| final_loss | **9.28e-4** (&lt; 1e-3) |
| passed_at | step 24 / 100 |
| param_count | 117506 |
| within ±5% of FF 121218 | **true** (ratio 0.969) |
| disconnect stalk-ablation | **ok** across T∈{6,8,12,16} |

**Invariant (Gate0):** stalk-ablation L2 at target for disconnected pairs — `max_l2 = 0.0` for every T (atol=1e-3). Interpretable as ‖h_t‖ ≈ 0 on disconnect under local potential (no `c` broadcast). Source: `artifacts/fractal_core_stalk_overfit.json` → `disconnect_leak` / embedded `gate0_overfit` in Gate1 artifact. `science_open=false` on Gate0 engineering.

---

## 3. Gate1 — ID best

Train: `data/id_2k.jsonl`, 30 epochs, bound30 recurrent hparams (lr=1.5e-3, clip=2.5, d=64, mlp×10, T_train=6, discrete).

| Metric | Value |
|--------|-------|
| best_epoch | **29** |
| best_val_acc | **0.985** |
| param_parity | within_5pct **true** |
| run_id | `fractal-stalk-gate1-ce91d0d032` |
| science_open (harness) | false |

---

## 4. Prereg PASS — matched-OOD @ T=16

Cite: `artifacts/fractal_core_stalk_gate1_matched_ood.json` → `prereg` / `ood_eval.dynamic.dynamic_T16_unroll`.

| Prereg threshold | Observed | Status |
|------------------|----------|--------|
| hard-neg @ T16 ≥ 0.98 | **1.000** (240/240) | **PASS** |
| K16 @ T16 ≥ 0.80 | **0.925** (74/80) | **PASS** |
| `prereg.pass` | true | |

Harness stamp on artifact remains `science_open: false` (never self-stamps true). Human seal may open **only** the scoped claim in §6.

---

## 5. Causal horizon — T vs K (discrete unroll)

Cite: same artifact `ood_eval.fixed.fixed_T6_unroll` + `ood_eval.dynamic.dynamic_T{8,12,16}_unroll`. Hard-neg = hop −1.

| T | Overall | Hard-neg | K8 | K12 | K16 |
|---|---------|---------|----|-----|-----|
| 6 | 0.502 | **1.000** | 0.000 | 0.0125 | 0.000 |
| 8 | 0.675 | **1.000** | **1.000** | 0.050 | 0.000 |
| 12 | 0.833 | **1.000** | **1.000** | **1.000** | 0.000 |
| **16** | **0.977** | **1.000** | 0.938 | **1.000** | **0.925** |

**Causal horizon:** when **T &lt; K**, positive acc ≈ 0 (under-propagation); when **T ≥ K**, positive acc is high (near-ceiling for K≤12; K16@T16 = 0.925). Hard-neg stays **1.000** at every T — zero leakage under hard mask + local stalk/probe.

---

## 6. Scoped science OPEN (single claim)

| Field | Value |
|-------|-------|
| **science_open** | **true** |
| **Claim** | Topological stalk locality + discrete cycle depth resolves zero-shot path length generalization up to K=16 on directed graphs. |
| **Substrate** | `covariate_matched_ood` + **hard** `A_ij` mask; stalk-local FractalCore (no `c` broadcast, no soft ACT) |
| **Evidence** | Prereg PASS (§4); causal horizon table (§5); Gate0 disconnect ‖h_t‖=0 (§2) |
| **SHAs** | MEASURE `1d6c3da`; merge via PR #2 → `main` |

Do **not** generalize this OPEN beyond the cited substrate and architecture.

---

## 7. Explicit residue / NOT open

`science_open=false` on all of the following (fail-closed):

| Residue | Why closed |
|---------|------------|
| Learning topology **without** hard mask | Topology was handed via hard `A_ij`; not inferred |
| Soft ACT | Stripped this cycle; not re-licensed |
| Unrestricted MLP loops | Not the stalk-local discrete-T design |
| Universal / arbitrary graphs | Only `covariate_matched_ood` matched covariate OOD |
| Mandelbrot analogy as science | Aspirational only (`mandelbrot_analogy` in artifact) |
| Prior Gate2 / Geo-Loop length-gen | STOP — see `docs/SESSION-SEAL-GATE2.md` |

---

## 8. Limitation for next cycle

**Topology is handed via the hard `A_ij` mask.** Zero-leakage and the causal horizon are conditional on that oracle adjacency. Next frontier must remove the hard mask while preserving zero leakage + stability — see `CYCLE_SHEAF_INFERENCE` (§9 / `docs/CYCLE_SHEAF_INFERENCE.md`).

---

## 9. Next frontier — `CYCLE_SHEAF_INFERENCE`

Formulate (MEASURE plan only until measured):

- Infer restriction / sheaf maps **F_{u→v}** from edge tokens **without** hard `A_ij` mask.
- Preserve **zero leakage** on disconnect and **stability** under discrete depth.
- `science_open=false` for that cycle until Gate0/1 metrics exist.

Stub: `docs/CYCLE_SHEAF_INFERENCE.md`.

---

## 10. Fail-closed invariants preserved

- Artifact harness never self-stamps `science_open=true` (`science_open: false` in JSON).
- `_verify_param_parity` ±5% of FF 121218; hard-fail outside window.
- Masks (this cycle) from parsed graph edges only — never token co-occurrence as adjacency substitute.
- Evidence = cited artifact paths above; SHAs `1d6c3da` / PR #2 → `main`.
- Outside §6 claim: **fail-closed**.

---

## 11. Untrained-control contingent note (2026-09-27 CDT)

| Field | Value |
|-------|-------|
| **Audit** | `artifacts/stalk_untrained_control_audit.json` |
| **Harness** | `python -m reachability_gen.run_stalk_untrained_control` |
| **Config** | Matches sealed stalk OPEN: local stalk@s / probe@t, no `c` broadcast, no soft ACT, hard `A_ij`, discrete T∈{6,8,12,16} |
| **Verdict** | **`OPEN_STILL_CONTINGENT_NEEDS_MULTI_SEED`** |
| **seals_invalidated** | **false** |

**Matched-OOD T=16 (cite artifact `focus_summary`):**

| Arm | Overall | Hard-neg | K8 | K12 | K16 | Agreement |
|-----|---------|----------|----|-----|-----|-----------|
| Untrained | **0.633** | **0.600** | 1.000 | 0.000 | 1.000 | — |
| Sealed ckpt | **0.977** | **1.000** | 0.938 | 1.000 | 0.925 | **0.610** |

Untrained is **mid (~0.6)**, not ≈ sealed 1.0 — **bake-in NOT proven**. Sheaf-style init/oracle INVALIDATION does **not** apply here.

**science_open policy:** do **not** silently widen; do **not** revoke §6 OPEN. Stalk OPEN remains **standing but contingent** — needs **multi-seed trained reconfirm** before treating the single-seed Gate1 seal as fully robust. Prefer truth over prior OPEN; only revoke if a future bake-in proof lands.


---

## 12. Multi-seed reconfirm — OPEN contingent AT RISK (2026-09-27 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_MULTI_SEED_RECONFIRM` |
| **Artifact** | `artifacts/stalk_multi_seed_reconfirm.json` |
| **Harness** | `python -m reachability_gen.run_stalk_multi_seed_reconfirm` |
| **Base after PR #11** | `7f1037a750cb078d66521dd2ca867010f0d85f7e` |
| **Verdict** | **`OPEN_CONTINGENT_AT_RISK`** |
| **science_open** | **false** (harness); §6 human seal **not widened** |
| **Prereg (mean)** | hard-neg≥0.95 **FAIL** (0.918±0.142); K16≥0.75 **FAIL** (0.654±0.524) |
| **Individual** | **1/3** seeds pass (seed 0 only — matches sealed) |

**Matched-OOD T16 mean±std vs sealed single-seed:**

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| Sealed OPEN (seed 0 Gate1) | **0.977** | **1.000** | **0.925** |
| Reconfirm mean±std (seeds 0–2) | **0.803±0.202** | **0.918±0.142** | **0.654±0.524** |
| Untrained mean±std | 0.626±0.015 | 0.585±0.307 | — |

Untrained mid ≈0.63 — bake-in still **not** proven. Failure mode is **seed instability**
of trained stalk (seed 1 K16 collapse; seed 2 hard-neg drop), not init oracle.

**Policy (pre-demotion note):** Prefer truth over prior OPEN. Do **not** silently widen `science_open`.
§12 measured `OPEN_CONTINGENT_AT_RISK`; **human demotion to MEASURE** is recorded in **§13** (append-only). Cite
`docs/CYCLE_STALK_MULTI_SEED_RECONFIRM.md`.

---

## 13. DEMOTION — science_open OPEN → MEASURE (append-only; 2026-09-27 CDT)

| Field | Value |
|-------|-------|
| **Label** | **DEMOTION** / MEASURE (not INVALID; not a rewrite of §1–§12) |
| **Date** | 2026-09-27 (CDT) |
| **Trigger** | PR #12 `CYCLE_STALK_MULTI_SEED_RECONFIRM` multi-seed fragility |
| **Artifact** | `artifacts/stalk_multi_seed_reconfirm.json` |
| **Prior untrained control** | PR #11 — bake-in **not** proven (u≈0.63); seals_invalidated=false |
| **Verdict now** | **`MEASURE`** — live `science_open=true` claim **demoted** |
| **Not** | **INVALID** — failure mode is seed instability, not init bake-in |

### 13.1 Evidence cited (PR #12)

| Metric @ matched-OOD T16 | Sealed single-seed | Reconfirm mean±std (seeds 0–2) |
|--------------------------|--------------------|--------------------------------|
| overall | 0.977 | **0.803±0.202** |
| hard-neg | 1.000 | **0.918±0.142** (prereg mean ≥0.95 **FAIL**) |
| K16 | 0.925 | **0.654±0.524** (prereg mean ≥0.75 **FAIL**) |
| individual prereg | PASS (seed 0) | **1/3** seeds PASS |

Untrained mean overall **0.626±0.015** — mid; bake-in still not proven. Prefer honesty over lonely OPEN.

### 13.2 Science status — DEMOTE (not revoke-as-INVALID)

| Prior | Status now |
|-------|------------|
| §6 `science_open=true` scoped stalk locality OPEN | **DEMOTED → MEASURE** (no live science_open claim) |
| §6 / §4 / §5 numeric tables | Remain **historical** evidence of the single-seed run |
| PR #11 contingent standing | Superseded by multi-seed FAIL → demotion |
| Claim widening | **None** — `science_open` stays false/narrow; human-only widen later |

**Do not silently defend prior OPEN.** Multi-seed means miss floors; only 1/3 seeds pass.
Not INVALID: untrained control passed (no bake-in). Demotion = honesty that single-seed
OPEN is not robust enough to remain a live science_open claim.

### 13.3 Non-rewrite rule

Prior seal body (§1–§12) is **not** rewritten. This §13 is append-only DEMOTION.
Ledger: `docs/LEDGER-OPEN-MEASURE-STOP.md`. Cycle note: `docs/CYCLE_STALK_MULTI_SEED_RECONFIRM.md`.
