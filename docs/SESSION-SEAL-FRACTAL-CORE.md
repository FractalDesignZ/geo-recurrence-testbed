# SESSION SEAL — CYCLE_FRACTAL_CORE_GENESIS (FractalCore Gate0/1)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | MEASURE / FREEZE |
| **Cycle** | `CYCLE_FRACTAL_CORE_GENESIS` |
| **science_open** | **false** |
| **Repo** | https://github.com/FractalDesignZ/geo-recurrence-testbed |
| **Branch** | `cycle/fractal-core-genesis` |
| **Verdict class** | OPEN-candidate (**engineering only**) — not science OPEN |

Fail-closed. Append-only. This seal does **not** reopen science. Mandelbrot / boundary metaphor remains aspirational; evidence is metrics artifacts only.

---

## 1. Scope sealed

| Slice | Artifact | Status |
|-------|----------|--------|
| Gate0 balanced overfit (16/16 hard-neg) | `artifacts/fractal_core_overfit.json` | SEALED — **PASS** |
| Gate1 id_2k × 30 ep + matched-OOD | `artifacts/fractal_core_gate1_matched_ood.json` | SEALED |
| Gate1 best checkpoint | `artifacts/fractal_core_gate1_best.pt` | SEALED |
| Gate1 run log | `artifacts/fractal_core_gate1_run.log` | SEALED |

Substrate: `src/reachability_gen/models/fractal_core.py` — node-slot encoding, adjacency-masked attention, broadcast `+c` boundary, ACT soft halt. Wired into arms / param parity / Gate0–1 runners.

---

## 2. Gate0 — PASS

Balanced 32 (16 y=0 hard-neg + 16 y=1), T=6, adaptive halt.

| Metric | Value |
|--------|-------|
| final_acc | **1.0** |
| final_loss | **9.50e-4** (&lt; 1e-3) |
| passed_at | step 26 / 100 |
| param_count | 117507 |
| within ±5% of FF 121218 | **true** (window [115157, 127279], ratio 0.969) |
| science_open | false |

Source: `artifacts/fractal_core_overfit.json` / embedded `gate0_overfit` in Gate1 artifact.

---

## 3. Gate1 — ID best

Train: `data/id_2k.jsonl`, 30 epochs, bound30 recurrent hparams (lr=1.5e-3, clip=2.5, d=64, mlp×10, T_train=6, adaptive_halt).

| Metric | Value |
|--------|-------|
| best_epoch | **25** |
| best_val_acc | **0.97** |
| val hard-neg (−1) | 0.995 |
| val K=2…6 | 0.95 / 1.00 / 0.85 / 0.85 / 0.90 |
| param_parity | within_5pct **true** |
| science_open | false |

---

## 4. Matched-OOD tables (`data/covariate_matched_ood.jsonl`, n=480)

### Fixed T=6

| Mode | Overall | Hard-neg | K8 | K12 | K16 |
|------|---------|----------|----|-----|-----|
| adaptive | 0.329 | 0.596 | 0.125 | 0.0625 | 0.000 |
| unroll | 0.540 | 0.638 | 0.375 | 0.2125 | 0.0125 |

### Dynamic unroll / adaptive (T ∈ {8,12,16})

| Mode | T | Overall | Hard-neg | K8 | K12 | **K16** |
|------|---|---------|----------|----|-----|---------|
| adaptive | 8 | 0.348 | 0.633 | 0.125 | 0.0625 | 0.000 |
| adaptive | 12 | 0.348 | 0.633 | 0.125 | 0.0625 | 0.000 |
| adaptive | 16 | 0.348 | 0.633 | 0.125 | 0.0625 | 0.000 |
| **unroll** | 8 | 0.471 | 0.700 | 0.375 | 0.2125 | 0.0125 |
| **unroll** | 12 | 0.450 | 0.754 | 0.2125 | 0.2125 | 0.0125 |
| **unroll** | **16** | **0.554** | **~0.763** | 0.125 | 0.0875 | **0.825** |

Headline recoveries (unroll):

- **K16 @ T16 unroll pos = 0.825** (66/80)
- **Hard-neg miss ~0.763** vs prereg aspirational **1.0** (183/240 at T16 unroll; fixed-T6 adaptive observed hard-neg **0.596**)

Prereg aspirational (from artifact): hard-neg OOD = 1.0 if mask algebraically forbids disconnected pairs; recover K≥8 positives via unroll. **Report honestly: hard-neg ≠ 1.0.**

---

## 5. Leakage (documented — not science OPEN)

`ood_eval.mask_leakage` (`algebraically_perfect=false`):

> With adjacency-only + self mask and target-node readout, unreachable (s,t) cannot receive source-seeded signal along edges. Hard-neg acc &lt; 1.0 ⇒ residual leakage: **broadcast boundary prompt `c(s,t)` on all nodes**, shared Φ weights, **soft ACT mixing**, and/or finite-T under-propagation — not a fake mask.

Primary leakage channels called out for next cycle:

1. **`c` broadcast** — invariant (s,t) prompt added to every node slot each cycle
2. **Soft ACT** — mixture of Head(z_k[target]) across halt steps

`science_open=false` on leakage section.

---

## 6. OPEN candidates — engineering only

| Candidate | Class | science_open |
|-----------|-------|--------------|
| FractalCore plumbing (node-slot + adjacency mask + Gate0/1 runners) | engineering | **false** |
| Param parity ±5% vs FF ~121218 | accounting hygiene | **false** |
| Gate0 overfit PASS | eng gate | **false** |
| ID val ≈ 0.97 @ epoch 25 | eng signal | **false** |
| K16@T16 unroll recovery 0.825 | eng observation | **false** |

**No science OPEN.** Do not treat hard-neg miss or Mandelbrot analogy as evidence of learnability / geometry.

Next engineering direction (out of this seal): kill global (s,t) broadcast; local stalk potential; strip soft ACT → discrete T only — see cycle `CYCLE_STALK_LOCALIZATION`.

---

## 7. Fail-closed invariants preserved

- `science_open=false` always (harness never self-stamps true)
- `_verify_param_parity` ±5% of FF 121218; hard-fail outside window
- Masks from parsed graph edges only (never token co-occurrence)
- Evidence = cited artifact paths above

