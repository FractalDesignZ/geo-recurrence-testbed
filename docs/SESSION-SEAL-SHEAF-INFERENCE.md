# SESSION SEAL — CYCLE_SHEAF_INFERENCE (SheafInferCore Gate0/1)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | FREEZE / human seal |
| **Cycle** | `CYCLE_SHEAF_INFERENCE` |
| **science_open** | **true** (scoped claim only — §6) |
| **Repo** | https://github.com/FractalDesignZ/geo-recurrence-testbed |
| **Branch** | `cycle/sheaf-inference` |
| **MEASURE SHA (pre-merge tip)** | `f374e7c` (`f374e7c2414e70345bbb6fe9a68dafb1dc95f77f`) |
| **Merge target** | PR #3 → `main` |
| **Merge SHA (post-merge)** | `af8e49f` (`af8e49fc026349d01efa4fda1f151c67cd453a04`) |
| **Prior stalk seal** | `docs/SESSION-SEAL-STALK-LOCALIZATION.md` @ `b144dac` (append-only; not rewritten) |
| **Verdict class** | OPEN (scoped science) + FAIL-CLOSED elsewhere |

Fail-closed outside the single claim in §6. Append-only. Mandelbrot / sheaf metaphor remains aspirational except where metrics are cited. Evidence = cited artifact paths; do not invent metrics.

---

## 1. Scope sealed

| Slice | Artifact | Status |
|-------|----------|--------|
| Gate0 balanced overfit + disconnect under `A_hat` | `artifacts/sheaf_infer_overfit.json` | SEALED — **PASS** |
| Gate1 id_2k × 30 ep + matched-OOD (discrete T) | `artifacts/sheaf_infer_matched_ood.json` | SEALED — **Prereg PASS** |
| Gate1 best checkpoint | `artifacts/sheaf_infer_gate1_best.pt` | SEALED |
| Gate1 run log | `artifacts/sheaf_infer_gate1_run.log` | SEALED |

Architecture (from artifact `architecture` + Gate0): edge-token encoder → `E_hat` → hard gate θ=0.5 → inferred `A_hat`; **default `gate_detach_diffusion=true`** into stalk-local discrete diffusion; shared bias-free Φ; local stalk@`s` / else 0; discrete `T ∈ {6,8,12,16}`; **no hard A oracle at eval**; aux edge-recon BCE **train-only**. Param count **123206** within ±5% of FF 121218 (ratio ≈1.016).

Substrate: `covariate_matched_ood` (`data/covariate_matched_ood.jsonl`, n=480; token_len band [45,70]) on synthetic ER digraphs / matched seq-len band. Prior cycle seal: stalk locality with hard `A_ij` mask (`docs/SESSION-SEAL-STALK-LOCALIZATION.md` @ `b144dac`) — residue was handed topology; this cycle removes the eval oracle.

---

## 2. Gate0 — PASS + disconnect invariant

Balanced 32 (16 y=0 hard-neg + 16 y=1), T=6, discrete halt (`adaptive_halt=false`). Cite: `artifacts/sheaf_infer_overfit.json`.

| Metric | Value |
|--------|-------|
| final_acc | **1.0** |
| final_loss | **~2.28e-5** (&lt; 1e-3) |
| final_edge_recon_acc | **1.0** (≥ 0.99) |
| passed_at | step 1 / 150 |
| param_count | **123206** |
| within ±5% of FF 121218 | **true** (ratio ≈1.016; window [115157, 127279]) |
| hard_A_oracle_eval | **false** |
| disconnect ‖h_t‖ under `A_hat` | **ok** — max_l2 = **0.0** across T∈{6,8,12,16} (atol=1e-3) |
| edge recon (disconnect eval) | **1.0** offdiag across those T |

**Invariant (Gate0):** for disconnected pairs, ‖h_t‖ = 0 under inferred hard gate / `A_hat` at every sealed T. `ste_softening_note` = null. Harness `science_open=false` on Gate0 engineering.

---

## 3. Gate1 — ID best

Train: `data/id_2k.jsonl`, 30 epochs, bound30 recurrent hparams (lr=1.5e-3, clip=2.5, d=64, mlp×12, T_train=6, discrete). Cite: `artifacts/sheaf_infer_matched_ood.json` → `train`.

| Metric | Value |
|--------|-------|
| best_epoch | **1** |
| best_val_acc | **1.0** |
| param_count | 123206 |
| param_parity | within_5pct **true** |
| run_id | `sheaf-infer-gate1-23a9f0978a` |
| hard_A_oracle_eval | **false** |
| science_open (harness) | false |

---

## 4. Prereg PASS — matched-OOD @ T=16

Cite: `artifacts/sheaf_infer_matched_ood.json` → `prereg` / `ood_eval.dynamic.dynamic_T16_unroll`.

| Prereg threshold | Observed | Status |
|------------------|----------|--------|
| hard-neg @ T16 ≥ 0.95 | **1.000** (240/240) | **PASS** |
| K16 @ T16 ≥ 0.75 | **1.000** (80/80) | **PASS** |
| `prereg.pass` | true | |

Harness stamp on artifact remains `science_open: false` (never self-stamps true). Human seal may open **only** the scoped claim in §6.

---

## 5. Causal horizon — T vs K (discrete unroll)

Cite: same artifact `ood_eval.fixed.fixed_T6_unroll` + `ood_eval.dynamic.dynamic_T{8,12,16}_unroll` (dynamic T6 matches fixed). Hard-neg = hop −1.

| T | Overall | Hard-neg | K8 | K12 | K16 |
|---|---------|---------|----|-----|-----|
| 6 | 0.500 | **1.000** | **0.000** | **0.000** | **0.000** |
| 8 | 0.667 | **1.000** | **1.000** | **0.000** | **0.000** |
| 12 | 0.833 | **1.000** | **1.000** | **1.000** | **0.000** |
| **16** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

**Causal horizon confirmation:** when **T &lt; K**, hop-stratified accuracy on that K column is **0.000** (under-propagation); when **T ≥ K**, accuracy is **1.000** at T=16 for all sealed K∈{8,12,16}. Hard-neg stays **1.000** at every T — zero leakage under inferred hard gate (no A oracle at eval).

---

## 6. Scoped science OPEN (single claim)

| Field | Value |
|-------|-------|
| **science_open** | **true** |
| **Claim** | Learned directed restriction maps (edge-token → Ê → hard-gated Â) + stalk-local discrete cycle diffusion resolve zero-shot path-length generalization up to K=16 on directed graphs without an external adjacency oracle at eval (covariate-matched OOD, this substrate). |
| **Substrate** | `covariate_matched_ood` (synthetic ER digraphs, token_len band [45,70]); SheafInferCore with train-only aux edge recon + default `gate_detach_diffusion`; **no** hard A at eval |
| **Evidence** | Prereg PASS (§4); causal horizon table (§5); Gate0 disconnect ‖h_t‖=0 + edge recon 1.0 (§2) |
| **SHAs** | MEASURE pre-merge tip `f374e7c`; merge commit `af8e49f` via PR #3 → `main` |

Do **not** generalize this OPEN beyond the cited substrate, K≤16, and architecture.

---

## 7. Explicit residue / NOT open

`science_open=false` on all of the following (fail-closed):

| Residue | Why closed |
|---------|------------|
| K &gt; 16 path-length gen | Sealed evidence stops at K16@T16; longer hops unmeasured |
| Denser / alternate graph families | Only this matched ER / seq-len-band substrate; no claim on denser graphs beyond sealed draw |
| Aux edge recon at eval | Train-only gold-edge BCE; eval uses inferred gate only |
| STE / Gumbel as required mechanism | Available in code; reported numbers use default **gate_detach_diffusion** (STE mode for gate construction, detached into Φ) |
| Soft ACT | Stripped; not re-licensed |
| Unrestricted MLP loops | Not the stalk-local discrete-T design |
| Stalk OPEN with hard mask | Separate sealed claim — `docs/SESSION-SEAL-STALK-LOCALIZATION.md` @ `b144dac` (append-only; not rewritten here) |
| Mandelbrot analogy as science | Aspirational only (`mandelbrot_analogy` in artifact) |
| Prior Gate2 / Geo-Loop length-gen | STOP — see `docs/SESSION-SEAL-GATE2.md` |

---

## 8. Evidence vs aspiration

| Label | Content |
|-------|---------|
| **Evidence** | Gate0/1 JSON metrics cited above; causal horizon T vs K; prereg PASS; param parity hygiene; `hard_A_oracle_eval: false` |
| **Aspiration** | Sheaf / Mandelbrot metaphor; universal graphs; K&gt;16; density stress — deferred to next cycle |

---

## 9. Next frontier — `CYCLE_SHEAF_STRESS_TEST`

Stress learned sheaf beyond sealed K≤16 / current matched density. MEASURE plan only until measured; `science_open=false` until a later human seal.

Stub: `docs/CYCLE_SHEAF_STRESS_TEST.md`.

---

## 10. Fail-closed invariants preserved

- Artifact harness never self-stamps `science_open=true` (`science_open: false` in JSON).
- `_verify_param_parity` ±5% of FF 121218; hard-fail outside window.
- No A oracle at eval; gold edges for aux recon in training only.
- Evidence = cited artifact paths above; SHAs `f374e7c` / merge `af8e49f` / PR #3 → `main`.
- Outside §6 claim: **fail-closed**.
- Do not rewrite `docs/SESSION-SEAL-STALK-LOCALIZATION.md`.

---

## 11. External AUDIT commentary triage (append-only)

| Field | Value |
|-------|-------|
| **Label** | external AUDIT commentary triage |
| **Date** | 2026-09-27 |
| **Scoped validation accepted** | MEASURE / seal body at SHAs **`f374e7c`** / merge **`af8e49f`** (PR #3) — Gate0/1 metrics and §6 scoped claim as written |
| **Rejected as sealed claims** | universality; permanent confound elimination; CoT / o1 metaphors as mechanism; Spaces-first deployment; soft ACT halt; NL Llama bridge |
| **Note** | Commentary outside the cited artifacts is not evidence. Prior seal body (§1–§10) is **not** rewritten. Fail-closed elsewhere. |
| **Next MEASURE** | `CYCLE_SHEAF_RED_TEST` single cell (K=20, p=0.15, T∈{20,24}) — see `docs/CYCLE_SHEAF_STRESS_TEST.md` / `artifacts/sheaf_infer_red_test.json` |

---

## 12. MEASURE residue — CYCLE_SHEAF_RED_TEST K=20 (append-only)

| Field | Value |
|-------|-------|
| **Label** | MEASURE residue (not science OPEN) |
| **Date** | 2026-09-27 |
| **Slice** | `CYCLE_SHEAF_RED_TEST` single cell — K=20 sparse / path-backbone under sealed seq_len band [45,70] |
| **Result** | Prereg **PASS** / attribution **PASS** (cite `artifacts/sheaf_infer_red_test.json`) |
| **science_open claim** | **Unchanged** — §6 remains **K≤16 only** on the sealed matched-OOD substrate. Do **not** widen OPEN from this RED_TEST PASS. |
| **Density** | **p=0.15 still unverified.** Path-backbone keeps band; empirical p ≪ 0.15. Pure ER@p=0.15 × band × K=20 documented infeasible in gen report. |
| **Note** | K=20 sparse RED_TEST PASS is **MEASURE residue** only. Prior seal body (§1–§11) is **not** rewritten. Fail-closed elsewhere. |
| **Next MEASURE** | `CYCLE_SHEAF_DENSITY_STRESS` — true ER digraph p=0.15, K=8, n=16, T∈{8,12}, no path-backbone density drop |

---

## 13. MEASURE residue — CYCLE_SHEAF_DENSITY_STRESS feasibility wall (append-only)

| Field | Value |
|-------|-------|
| **Label** | MEASURE residue / feasibility wall (not science OPEN) |
| **Date** | 2026-09-27 |
| **Slice** | `CYCLE_SHEAF_DENSITY_STRESS` — true ER p=0.15 @ n=16 → &#124;E&#124;≈36 → seq_len≈111 ∉ sealed [45,70] → **INVALID** |
| **Seal** | `docs/SESSION-SEAL-SHEAF-DENSITY-FEASIBILITY.md` (PR #5 telemetry **SUSPENDED/INVALID**; no scientific promotion) |
| **science_open claim** | **Unchanged** — §6 remains **K≤16 only**. Do **not** widen OPEN. |
| **Note** | Prior seal body (§1–§12) is **not** rewritten. Fail-closed elsewhere. |
| **Next MEASURE** | `CYCLE_SHEAF_DENSE_CONTEXT` — band [100,140]; dense vs matched-sparse control |

---

## 14. MEASURE residue — CYCLE_SHEAF_DENSE_CONTEXT (append-only)

| Field | Value |
|-------|-------|
| **Label** | MEASURE residue (not science OPEN) |
| **Date** | 2026-09-27 |
| **Slice** | `CYCLE_SHEAF_DENSE_CONTEXT` — band [100,140]; Cell1 dense n=16 p=0.15 vs Cell2 matched-sparse n=32 p≈0.0352; frozen Gate1 |
| **Result** | Prereg **PASS** / attribution **PASS** (cite `artifacts/sheaf_infer_dense_context.json`) |
| **seq_len** | Cell1 **110.79** / Cell2 **116.16** (both ∈ [100,140]) |
| **science_open claim** | **Unchanged** — §6 remains **K≤16 only**. Do **not** widen OPEN from this MEASURE PASS. |
| **Note** | Prior seal body (§1–§13) is **not** rewritten. Fail-closed elsewhere. Feasibility wall under sealed [45,70] remains INVALID (PR #5 / §13). |

---

## 15. SESSION SEAL pointer — CYCLE_SHEAF_DENSE_CONTEXT (append-only)

| Field | Value |
|-------|-------|
| **Label** | SESSION SEAL pointer (append-only; body §1–§14 **not** rewritten) |
| **Date** | 2026-09-27 |
| **Slice** | `CYCLE_SHEAF_DENSE_CONTEXT` — band [100,140]; Cell1 dense n=16 p=0.15 K=8 vs Cell2 matched-sparse; frozen Gate1 |
| **Seal** | `docs/SESSION-SEAL-SHEAF-DENSE-CONTEXT.md` |
| **Artifact** | `artifacts/sheaf_infer_dense_context.json` |
| **PR** | [#6](https://github.com/FractalDesignZ/geo-recurrence-testbed/pull/6) tip `e145113`; merge `353bd61` → `main` |
| **science_open (dense-context seal)** | **true** — scoped claim on that seal §6 only (dense K=8 @ [100,140] vs matched-sparse) |
| **This seal §6 (sparse [45,70] K≤16)** | **Unchanged** — do **not** widen from dense-context OPEN |
| **NON-claims carried** | not dense K≥16; not NL/CoT; PR #4 K20 remains MEASURE; PR #5 INVALID stands |
| **Residue next** | Depth × Density frontier |
| **Note** | Prior seal body (§1–§14) is **not** rewritten. Fail-closed elsewhere. |

---

## 16. INVALIDATION — untrained init bake-in (append-only; FAIL-CLOSED)

| Field | Value |
|-------|-------|
| **Label** | **INVALIDATION** / RESTRICT (not a rewrite of §1–§15 body text) |
| **Date** | 2026-09-27 |
| **Trigger** | External Opus probe (GammaHunter sister) + local reproduction |
| **Audit artifact** | `artifacts/sheaf_untrained_control_audit.json` |
| **Harness** | `python -m reachability_gen.run_sheaf_untrained_control` (standing) |
| **Base SHA audited** | `4083d43` (main after PR #6 merge `353bd61`) |
| **Verdict** | **`SEALS_COMPROMISED_INIT_BAKE_IN`** |

### 16.1 Init that bakes reachability (`SheafInferCore._init_specials`)

| Knob | Init | Effect at eval (θ=0.5) |
|------|------|-------------------------|
| `edge_encoder` last Linear | **W=0, bias=+4.0** | Listed edge tokens → logit 4 → σ≈0.982 → **gate ON** |
| `absent_bias` | **−4.0** | Non-listed cells → **gate OFF** |
| Diagonal self logit | **+8.0** (hardcoded in `encode_edge_logits`) | Self always gated on |
| `residual_alpha` | **1.0** | Full φ replace |
| `SheafDiffusionPhi` | **W_msg=I, W_out=I, MLP=0** | Pure diffusion along gated Â from step 0 |
| `stalk_proj` | **I** | Local stalk@s nonzero |
| Readout head | **[z_t; ‖z_t‖]**; energy col (−1,+1); bias **(+5,−5)** | ‖z_t‖=0 → y=0; ‖z_t‖>0 → y=1 |

Untrained model = discrete BFS/diffusion reachability oracle on listed edge tokens. Training is not required for sealed metrics.

### 16.2 Reproduction tables (untrained vs sealed ckpt)

Cite: `artifacts/sheaf_untrained_control_audit.json`. Prediction agreement = fraction of identical argmax preds.

**matched-OOD** (`data/covariate_matched_ood.jsonl`, n=480):

| T | U overall | U hard-neg | U K8 / K12 / K16 | Trained overall | Agree | U FPR / FNR |
|---|-----------|------------|------------------|-----------------|-------|-------------|
| 6 | 0.500 | **1.000** | 0 / 0 / 0 | 0.500 | **1.000** | 0 / 1.000 |
| 8 | 0.667 | **1.000** | 1 / 0 / 0 | 0.667 | **1.000** | 0 / 0.667 |
| 12 | 0.833 | **1.000** | 1 / 1 / 0 | 0.833 | **1.000** | 0 / 0.333 |
| **16** | **1.000** | **1.000** | **1 / 1 / 1** | **1.000** | **1.000** | **0 / 0** |

**RED_TEST K20** (`data/sheaf_red_test_k20.jsonl`, n=128): T∈{20,24} — untrained overall/hard-neg/K20 = **1.000**; agree **1.000**; FPR/FNR **0**.

**Dense-context** Cell1 + Cell2: T∈{8,12} — untrained overall/hard-neg/K8 = **1.000**; agree **1.000**; FPR/FNR **0**.

Sealed Gate1 best epoch was **1** with train_acc=1.000 from epoch 1 (`artifacts/sheaf_infer_gate1_run.log`) — consistent with bake-in.

### 16.3 Science status — REVOKE / NARROW

| Prior | Status now |
|-------|------------|
| §6 `science_open=true` claim: "**Learned** directed restriction maps …" | **INVALIDATED** as a *learned* claim |
| Allowed residual claim (if any) | **init+architecture** implements discrete reachability on listed edge tokens under T≥K; **not** evidence of learned Â from data |
| Prereg numeric PASS (§4) | Numbers remain true of the *system*; attribution to learning is false |
| RED_TEST / dense-context MEASURE PASS using frozen Gate1 | Same contamination — metrics = untrained oracle |

**Do not silently defend prior OPEN.** Prefer truth: revoke learned OPEN; narrow to architecture/init demonstration only pending neutral-init retrain.

### 16.4 Stalk seal note

`FractalCore` stalk (`docs/SESSION-SEAL-STALK-LOCALIZATION.md`) uses hard A oracle + `Linear(d,2)` head **without** this energy-bias / edge-bias=+4 pattern. **Not the same classifier bake-in.** Stalk claim is orthogonal; not auto-invalidated by this audit. Still require standing untrained controls on future sheaf variants.

### 16.5 Required next MEASURE — neutral-init RED_TEST

Propose `CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN` (see `docs/CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN.md`):

- Neutralize bake-in: edge last bias ~0 (or small negative); absent_bias kept or symmetric; head Xavier/zeros **without** energy±5 oracle; optionally random W_msg.
- Retrain Gate1 protocol on `id_2k`; re-eval matched-OOD / RED_TEST / dense-context.
- Fail-closed: `science_open=false` until untrained_control shows untrained ≪ trained and trained meets prereg floors.
- Standing harness must remain in CI / MEASURE checklist.

### 16.6 Non-rewrite rule

Prior seal body (§1–§15) is **not** rewritten. This §16 is append-only INVALIDATION. Downstream dense-context seal §6 learned OPEN is likewise invalidated by pointer — see that seal's INVALIDATION appendix.

---

## 17. MEASURE residue — CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN (append-only; science_open=false)

| Field | Value |
|-------|-------|
| **Label** | MEASURE residue (not a science OPEN) |
| **Date** | 2026-09-27 |
| **Trigger** | §16 INVALIDATION; PR #7 merge `034a074` |
| **Plan** | `docs/CYCLE_SHEAF_NEUTRAL_INIT_RETRAIN.md` |
| **Harness** | `python -m reachability_gen.run_sheaf_neutral_init_retrain` |
| **Artifact** | `artifacts/sheaf_neutral_init_retrain.json` (+ `artifacts/sheaf_reach_cue_audit.json`) |
| **science_open** | **false** — harness never self-stamps true; human seal required |

Gates A–D (neutral init, ≥3 seeds, degree-balanced reach-cue ≤0.52). If floors fail → STOP with residue. If floors pass → MEASURE candidate only. Cite artifact for numbers; do not invent.

