# CYCLE_STALK_SEED_ENSEMBLE — human seal (scoped science_open)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-28 (CDT) |
| **Mode** | HUMAN SEAL — append-only; **no new training** |
| **Cycle** | `CYCLE_STALK_SEED_ENSEMBLE` (PR #22) |
| **science_open** | **true** (scoped claim only — this note + seal §22) |
| **Base** | `main` `16e97fc` (after PR #23 distill STOP) |
| **Evidence** | `artifacts/stalk_seed_ensemble.json` (verdict `PASS_CANDIDATE`; harness `science_open: false`) |
| **Merge** | PR #22 `3972756`; this seal PR stamps live OPEN |
| **Prereg / results** | prereg `e040270`; results `f774c69` |

**Exact claim (narrow):** Inference-time `prob_mean` ensemble of frozen hard-Â stalk checkpoints from PR #14 (seeds 0..4) and PR #18 (seeds 5..9) clears matched-OOD T16 hard-neg (≥0.95) and K16 (≥0.75) floors with positive lift vs the mean of those singles on `covariate_matched_ood` + hard `A_ij` — **ensemble-at-eval only**.

---

## 1. Merit (why seal)

| Evidence | Value | Gate |
|----------|-------|------|
| Ensemble `prob_mean` T16 HN | **1.000** | ≥0.95 **PASS** |
| Ensemble `prob_mean` T16 K16 | **1.000** | ≥0.75 **PASS** |
| Lift vs singles mean | Δoverall **+0.093**; ΔHN **+0.065**; ΔK16 **+0.208** | dual lift **PASS** |
| LOO `prob_mean` | HN **1.000±0.000**; K16 **0.991±0.008** | agrees |
| Train this cycle | **0** fill-trains (all #14/#18 ckpts present) | eval-only |
| Select / curriculum | remain **CLOSED** (#20 park) | not reopened |

Harness correctly left `science_open=false` (`PASS_CANDIDATE`). Human seal opens **only** the narrow ensemble-at-eval claim.

---

## 2. Explicit NON-claims (fail-closed)

| NON-claim | Status |
|-----------|--------|
| Single hard-Â stalk location / seed is OPEN | **NO** — singles remain **MEASURE_STILL** / seed-fragile (#14 **2/5**; #18 **3/10**) |
| Distill / one-student compression of the ensemble map | **STOP** (PR #23 / seal §21) — map≠location |
| Select-weight / curriculum / seed-panel reopen | **CLOSED** |
| Sheaf unsupervised / learned-Â revive | **NOT opened** (STOP/INVALID path) |
| Soft ACT / no hard mask / arbitrary graphs | **NOT** licensed |
| Hist. §6 single-seed Gate1 OPEN | remains **demoted** (§13); this seal does **not** revive it |
| Auto-widen from harness | **forbidden** |

---

## 3. Scope sealed

| Slice | Status |
|-------|--------|
| Primary aggregator | **`prob_mean`** only (logit_mean / majority_vote = secondary report) |
| Members | Frozen #14 `stabilize_seed{0..4}_best.pt` + #18 `seed_stability_seed{5..9}_best.pt` |
| Substrate | `covariate_matched_ood` + hard `A_ij`; discrete T∈{6,8,12,16}; floors gate **T16** |
| Architecture | Sealed stalk-local FractalCore (no `c` broadcast, no soft ACT) |
| Training | **None** for this seal stamp |

Do **not** generalize beyond cited members, aggregator, substrate, and architecture.

---

## 4. Citations

- Cycle note: `docs/CYCLE_STALK_SEED_ENSEMBLE.md`
- Seal append: `docs/SESSION-SEAL-STALK-LOCALIZATION.md` §22
- Ledger: `docs/LEDGER-OPEN-MEASURE-STOP.md`
- Distill STOP (non-claim): `docs/CYCLE_STALK_ENSEMBLE_DISTILL.md` / PR #23
- Park / select CLOSED: `docs/CYCLE_STALK_PARK_ACCEPT_MEASURE.md` / §19
