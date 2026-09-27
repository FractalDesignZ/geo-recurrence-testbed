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
