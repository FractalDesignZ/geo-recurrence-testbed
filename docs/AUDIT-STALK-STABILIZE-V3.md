# AUDIT — CYCLE_STALK_STABILIZE_V3 (metric evaluator / systems architect)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-28 (CDT) |
| **Base** | `main` after PR #18 `8850590` |
| **Cycle audited** | `CYCLE_STALK_STABILIZE_V3` (PR #16 `8866414`) |
| **Artifact** | `artifacts/stalk_stabilize_v3.json` + `stalk_stabilize_v3_run.log` |
| **Ckpts** | `artifacts/fractal_core_stalk_stabilize_v3_seed{0..4}_best.pt` (**all present**) |
| **Harness verdict** | **`STOP_FRAGILE`** (seed **1/5**; HN mean **0.838** fails ≥0.95; K16 **0.968** clears ≥0.75) |
| **science_open** | **false** (not widened; this audit does **not** set `science_open=true`) |
| **Auditor stance** | Fail-closed. Prefer #14 MEASURE_STILL corridor. No ID-overfit as OOD proof. |

**First sentence:** Equal-weight HN+K16+overall select-aux **rescued OOD K16** (V2 0.423→**0.968**) but **traded hard-neg** (0.957→**0.838**), inverting V2's failure mode; seed PASS stays **1/5** — park further select-mix chase; keep **#14** as best stabilize evidence.

---

## 1. stalk_stabilize_v3 Performance Synthesis

### 1.1 Seed table (matched-OOD T16 — floors; cite artifact)

| Seed | best_ep | ID-sel HN / K16_aux / ov | **Overall** | **Hard-neg** | K8 | K12 | **K16** | Final train_loss | prereg |
|------|---------|-------------------------|-------------|--------------|----|-----|---------|------------------|--------|
| 0 | 27 | 0.960 / 1.000 / 0.980 | 0.798 | **0.629** | 0.900 | 1.000 | **1.000** | 0.0123 | FAIL (HN) |
| 1 | 38 | 1.000 / 0.949 / 0.823 | 0.929 | **1.000** | 0.800 | 0.887 | **0.887** | 0.0250 | **PASS** |
| 2 | 43 | 0.930 / 1.000 / 0.955 | 0.929 | 0.871 | 0.963 | 1.000 | **1.000** | 0.0786 | FAIL (HN) |
| 3 | 14 | 0.965 / 1.000 / 0.910 | 0.840 | 0.775 | 0.775 | 0.963 | **0.975** | 0.0624 | FAIL (HN) |
| 4 | 37 | 1.000 / 1.000 / 0.998 | 0.952 | 0.912 | 1.000 | 1.000 | **0.975** | 0.0057 | FAIL (HN) |
| **mean±std** | — | — | **0.890±0.067** | **0.838±0.142** | — | — | **0.968±0.046** | — | **1/5** |
| **worst** | — | — | 0.798 (s0) | **0.629 (s0)** | — | — | 0.887 (s1) | — | — |

- **Loss:** per-seed **final-epoch train CE** from artifact (`train.final_epoch_sel.train_loss`). No aggregate val CE summary key at cycle top-level; hop-stratified `loss_mean` exists under `matched_ood_by_T` (HN/unreachable hop−1 loss is the dominant failure channel on FAIL seeds).
- **OOD gen:** floors = matched-OOD T∈{6,8,12,16} (`data/covariate_matched_ood.jsonl`). Select-aux longhop (`data/id_select_longhop.jsonl`, n=239 after 1 overlap drop) used **only** for ckpt select — **not** floors. ID val hops ∈ {−1,2..6} — **no native K16**; do not treat ID sel ov as OOD proof.
- **FLOPs:** **N/A honest.** ADR-001 / `flops.py` define schematic FF/attn/`F_block`; V3 harness does **not** emit `cumulative_flops` or stalk-vs-FF FLOP tables. Param parity only (below). No invented FLOP estimate claimed as measured.

### 1.2 Multi-seed mean / var / worst (brief)

| Metric @ T16 | mean | std | worst seed |
|--------------|------|-----|------------|
| overall | 0.890 | 0.067 | 0.798 (seed0) |
| hard-neg | 0.838 | 0.142 | **0.629 (seed0)** |
| K16 | 0.968 | 0.046 | 0.887 (seed1) |

Variance is **HN-dominated**; K16 is tight after select-aux. Seed PASS rate **0.20** (1/5).

### 1.3 Param invariance (`param_match` / state_dict)

| Seed | ckpt present | `state_dict` numel | FF baseline | ratio | ±5% |
|------|--------------|--------------------|-------------|-------|-----|
| 0..4 | **yes** (5/5) | **117506** | 121218 | 0.9694 | **PASS** |

Shapes consistent (`src_emb`/`tgt_emb` (64,), `node_emb`/`pos_emb`/`stalk_proj`/`probe_proj` 64×64). Artifact `param_parity.within_5pct=true` all seeds. **science_open=false** on parity (hygiene only).

### 1.4 Cross-check honesty (#14 / #15 / #16 / #18 on main)

| Metric @ T16 | #14 stabilize | #15 V2 | **#16 V3** | #17 ObjV1 | #18 seed env (n=10) |
|--------------|---------------|--------|------------|-----------|---------------------|
| overall | **0.929±0.035** | 0.813±0.157 | 0.890±0.067 | 0.899±0.046 | 0.903±0.053 |
| hard-neg | **0.957±0.061 PASS** | 0.955±0.101 | **0.838±0.142 FAIL** | 0.927±0.048 | 0.935±0.090 FAIL |
| K16 | 0.863±0.143 PASS | **0.423±0.477 FAIL** | **0.968±0.046 PASS** | 0.845±0.166 | 0.792±0.288 PASS |
| seed PASS | **2/5** | 1/5 | **1/5** | 0/5 | 3/10 |
| verdict | **MEASURE_STILL** | STOP_FRAGILE | **STOP_FRAGILE** | STOP_FRAGILE | MEASURE_ENVELOPE |

Artifact numbers match cycle docs / ledger within rounding. **#14 remains best stabilize corridor** (both mean floors PASS; highest seed PASS among stabilize attempts).

### 1.5 Init vs opt stall (log-supported)

- **Not init stall:** untrained T16 overall mean **0.488±0.194** vs trained **0.890**; agreement **0.495** — bake-in **not** proven (consistent with stalk untrained control).
- **Not opt dead:** all seeds run 60 ep; final train_acc ≈ 0.97–0.998; grad_clip sat rates low (**0.6–1.4%**). Early joint peaks (seed3 best_ep=**14**) are **selection** dynamics, not gradient death.
- Select-aux K16 oscillates 0↔1 during train (log); joint can lock a high-K16 / weaker-HN epoch that later OOD HN fails — **selection-induced HN fragility**, not under-training.

### 1.6 Diameter / dense gap / compute vs FF

| Probe | Status |
|-------|--------|
| Diameter gap | **N/A** — not instrumented this cycle |
| Dense ER gap | **N/A** — stalk band stays matched-OOD [45,70]; dense cells are sheaf (INVALID/STOP path) |
| Compute vs FF/attn | **N/A** — no harness FLOP emit for FractalCore stalk; ADR schematic only for FF/attn/`F_block` |

### 1.7 Degree-balanced T16 (secondary; not floors)

| Seed | overall | hard-neg | K16 |
|------|---------|----------|-----|
| mean±std | 0.857±0.088 | 0.771±0.189 | 0.967±0.046 |

HN worse under deg-balance than matched-OOD — reinforces HN as the binding failure, not K16.

---

## 2. Observed Failure Modes & Boundary Intersections

### 2.1 Primary failure mode (evidence-backed)

**Select-weight Pareto flip:** V2 (HN-heavy) → K16↓; V3 (equal + K16_aux) → **K16↑ HN↓**. Same architecture / #14 train corridor; only select changed. Mean HN **0.838 < 0.95**; seed0 HN **0.629** is catastrophic for floors. Seed-wise still **1/5**.

### 2.2 Boundary intersections

| Boundary | Finding |
|----------|---------|
| **ID vs OOD** | ID-sel HN/K16_aux/ov look strong (sel means HN≈0.97 / K16_aux≈0.99) while OOD HN collapses on 4/5 seeds — **ID/select overfit is not OOD proof**. Floors correctly use matched-OOD only. |
| **HN vs long-hop** | Binding trade-off under equal weights; unreachable hop−1 CE elevated on FAIL seeds (e.g. seed0 T16 hop−1 `loss_mean`≈1.46 vs seed1≈0.0006). |
| **Untrained** | Mid scores; not bake-in INVALID. |
| **Param** | Invariant; not a capacity confound. |

### 2.3 Tensor / grad / cue instrumentation

| Signal | Instrumented? |
|--------|---------------|
| Grad clip saturation | **Yes** (low rates; no explosion) |
| Per-layer / activation tensors | **Not instrumented** |
| Gradient norms beyond clip sat | **Not instrumented** |
| `reach_cue_audit` (degree/position/shortcut ceilings) | **Not run for V3** — only sheaf cycles have `artifacts/*reach_cue_audit.json`. Deg-balanced secondary eval exists; full cue ceiling audit **absent** this cycle. Say so: **not instrumented for stalk V3**. |

---

## 3. Next Cycle Specification

### Chosen: `CYCLE_STALK_PARK_ACCEPT_MEASURE` (honest park — **not** another select mix)

| Field | Spec |
|-------|------|
| **Name** | `CYCLE_STALK_PARK_ACCEPT_MEASURE` |
| **One-line hyp** | Select / curriculum / seed-panel chase is closed (V2/V3/ObjV1 STOP; #18 envelope fragile); **accept #14 0.5/0.5 MEASURE_STILL** as best stabilize corridor and park further stalk harden until an **orthogonal** cell. |
| **Why not V4 select** | V3 already inverted V2; another weight mix would re-chase the same Pareto without new attribution. Prefer honesty over lonely OPEN. |
| **Why not sheaf_ste_no_aux** | Ledger marks PR #10 **done STOP** (unstable 1/3); not an open next cell. |
| **science_open** | **false** always; harness must not self-stamp `true`. |
| **Mode** | MEASURE / PARK documentation — **no new train required** for seal of this park. |

### Floors / STOP (for the park seal)

| Gate | Threshold / action |
|------|-------------------|
| Seal | Append ledger: stalk select-line **PARKED**; best corridor = PR #14 |
| STOP reopen select | Any new PR that changes HN/K16/ov select weights or train upsample **without** orthogonal hyp → reject as re-chase |
| Future PASS path | Only via orthogonal cell (below) + human seal; never auto-widen |

### CLI / files (park PR = this audit)

```bash
# No new train. Documentation only:
#   docs/AUDIT-STALK-STABILIZE-V3.md   (this file)
#   docs/LEDGER-OPEN-MEASURE-STOP.md   (pointer §2)
```

Optional deferred orthogonal (ledger priority 6 — **not** this PR's execute):

```bash
# Future only — Aux ablation attribution under matched seeds
# (with-aux vs no-aux / STE residue); science_open=false
```

### Target metrics for seal / ledger update

| Target | Value |
|--------|-------|
| Document V3 | STOP_FRAGILE; K16↑ HN↓; 1/5 |
| Prefer corridor | **#14** MEASURE_STILL (HN 0.957 / K16 0.863 / 2/5) |
| Envelope cite | #18 MEASURE_ENVELOPE 3/10 |
| science_open | **false** |
| Next active cell | PARK stalk select-line; orthogonal = aux ablation / human-reviewed sheaf path — **not** V4 select |

---

## 4. Missing artifacts

| Item | Status |
|------|--------|
| `stalk_stabilize_v3.json` / `_run.log` | **present** |
| ckpts seed0..4 | **present** (117506 params each) |
| V3 `reach_cue_audit` JSON | **missing** (not run) |
| Per-cycle FLOP table vs FF | **missing** (N/A) |
| Diameter / dense stalk gap | **missing** (N/A) |

---

## 5. Policy

- Fail-closed. Prefer #14 over V2/V3/ObjV1.
- OOD floors only; no ID overfit as proof.
- Do **not** set `science_open=true`.
- Sheaf unsupervised STOP/INVALID path stays out of scope for stalk park.
