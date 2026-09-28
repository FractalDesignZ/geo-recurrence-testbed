# SESSION SEAL — CYCLE_STALK_LOCALIZATION (stalk-local FractalCore Gate0/1)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | MEASURE / FREEZE |
| **Cycle** | `CYCLE_STALK_LOCALIZATION` |
| **science_open** | **true** (scoped claim only — **§22** ensemble-at-eval; hist. §6 demoted) |
| **Repo** | https://github.com/FractalDesignZ/geo-recurrence-testbed |
| **Branch** | `cycle/stalk-localization` |
| **MEASURE SHA** | `1d6c3da` (`1d6c3da87a8387bf83813620f079fc0844fd1c06`) |
| **Merge target** | PR #2 → `main` |
| **Verdict class** | OPEN (scoped science) + FAIL-CLOSED elsewhere |

**Current status (2026-09-28 CDT):** live scoped OPEN = **§22** `CYCLE_STALK_SEED_ENSEMBLE` inference **`prob_mean`** overlay (PR #22/#24) — **ensemble-at-eval only**. Singles remain **MEASURE_STILL**/fragile (#14 2/5; #18 3/10). Distill **STOP** (§21 / PR #23). SWA persist **MEASURE** (§23 / PR #25). Epistemic disagreement audit **LIFTS_ON_DISAGREEMENT** + multi-hyp **STOP** (§24). Bag diversity **MEASURE_LIFT** (§25). Competent dissonance **COMPETENT_vs_CHAOS** (§26) — #22 **COMPETENT** (CD **0.811**; low global disagree ≠ echo); #27 **CHAOS** (CD **0.658**). RED competent dissonance **FAIL_CLOSED_DOMINANT** (§27) — 0 FAIL_OPEN / 61 FAIL_CLOSED on denser-K16+K20 RED. Hop-OOD HN **FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED** (§28) — ens HN **0.067**; **45** FAIL_OPEN HN core; gate/vote no repair. HN FAIL_OPEN autopsy **STRUCTURAL_CLUSTER** (§29) — isolated-source / hub-target; residue **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER`**. Sound outdeg gate **FO_PARTIAL** (§30) — **23/45** FO killed; HN **0.067→0.304**; matched-OOD Δ=0; residue **`…/OUTDEG0_PARTIAL`**. FO remainder autopsy **LOCAL_SOUND_WALL** (§31) — best local cut **5/22**; stop overlay chase; residue **`…/LOCAL_SOUND_WALL`**. Hop-OOD overlay **PARK** (§32). Reach certificates **CERT_FO_CATCH** (§33) — **45/45** FO + **22/22** rem-22; HN **0.067→1.000**; matched-OOD Δ=0; checker BFS post-hoc only. Tropical ens probe **COLLATERAL_HARM** (§34) — FO **0**/45 + rem22 **0**/22; matched K16 **1.000→0.512**; ens-layer max-plus falsified as FO repair; tropical ≢ cert; §22 **not** widened. Orientation collapse probe **INCONCLUSIVE_ARCH** (§35) — OK_HN cos nan_rate **1.0** (empty_in_t); cos AUROC undefined; (−l2) confounded; §22 **not** widened. Energy selector **COLLATERAL_HARM** (§36) — E_free FO **0**/45 + rem22 **0**/22; matched K16 **1.000→0.062**; oracle FO **0**/45 HARD_UNANIMOUS; cert_refuse tracks #35; §22 **not** widened. Select/curriculum **CLOSED**. Sheaf unsupervised **NOT** opened. Hist. §6 single-seed OPEN remains **demoted** (§13). Fail-closed outside §22 claim.

Fail-closed outside the live scoped claim in **§22** (hist. §6 demoted). Append-only. Mandelbrot / sheaf metaphor remains aspirational except where metrics are cited.

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

---

## 14. Stabilize multi-seed — MEASURE_STILL (append-only; 2026-09-27 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_STABILIZE_MULTI_SEED` |
| **Artifact** | `artifacts/stalk_stabilize_multi_seed.json` |
| **Harness** | `python -m reachability_gen.run_stalk_stabilize_multi_seed` |
| **Base** | `main` `12cb455` (after PR #12/#13) |
| **Verdict** | **`MEASURE_STILL`** |
| **science_open** | **false** (not widened; no PASS_CANDIDATE) |
| **Harden** | 5 seeds; 60 ep; cosine LR 1.5e-3→1.5e-4; joint 0.5·HN+0.5·overall @ T16 ID-val select |
| **Prereg mean** | hard-neg≥0.95 **PASS** (0.957±0.061); K16≥0.75 **PASS** (0.863±0.143) |
| **Seed-wise** | **2/5** PASS (goal ≥4/5 **FAIL**) |

### 14.1 Matched-OOD T16 mean±std

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| Sealed OPEN (hist. seed0) | 0.977 | 1.000 | 0.925 |
| PR #12 reconfirm (n=3) | 0.803±0.202 | 0.918±0.142 | 0.654±0.524 |
| **This stabilize (n=5)** | **0.929±0.035** | **0.957±0.061** | **0.863±0.143** |
| Untrained mean | 0.488±0.194 | 0.577±0.235 | — |

Seed1 K16 collapse **0.05 → 0.875** under harden. Means clear PR #12 floors; seed fraction does not. Stay MEASURE. Cite `docs/CYCLE_STALK_STABILIZE_MULTI_SEED.md`.

---

## 15. Stabilize V2 — STOP_FRAGILE (append-only; 2026-09-27 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_STABILIZE_V2` |
| **Artifact** | `artifacts/stalk_stabilize_v2.json` |
| **Harness** | `python -m reachability_gen.run_stalk_stabilize_v2` |
| **Base** | `main` `61314a0` (PR #14 merge) + prereg `6db3341` |
| **Verdict** | **`STOP_FRAGILE`** |
| **science_open** | **false** (not widened) |
| **Harden V2** | 5 seeds; 90 ep; cosine 1.5e-3→1.0e-4; gated joint **0.7·HN+0.3·overall** @ T16 ID-val (overall≥0.85) |
| **Prereg mean** | hard-neg≥0.95 **PASS** (0.955±0.101); K16≥0.75 **FAIL** (0.423±0.477) |
| **Seed-wise** | **1/5** PASS (goal ≥4/5 **FAIL**; ≤1/5 → STOP_FRAGILE) |

### 15.1 Matched-OOD T16 mean±std

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| PR #14 stabilize (n=5) | 0.929±0.035 | 0.957±0.061 | 0.863±0.143 |
| **This V2 (n=5)** | **0.813±0.157** | **0.955±0.101** | **0.423±0.477** |
| Untrained mean | 0.488±0.194 | 0.577±0.235 | — |

HN-heavy selection rescued seed2 HN but collapsed K16 on seeds 0/1/3 (ID overall gate insufficient as long-hop proxy). Prefer honesty: **STOP** this knob line; retain #14 MEASURE_STILL as best stabilize evidence. Cite `docs/CYCLE_STALK_STABILIZE_V2.md`.

### 15.2 Non-rewrite rule

Prior seal body + §13 DEMOTION + §14 MEASURE_STILL are **not** rewritten. This §15 is append-only STOP_FRAGILE.

---

## 16. Stabilize V3 — STOP_FRAGILE (append-only; 2026-09-27 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_STABILIZE_V3` |
| **Artifact** | `artifacts/stalk_stabilize_v3.json` |
| **Harness** | `python -m reachability_gen.run_stalk_stabilize_v3` |
| **Base** | `main` `f94f9f7` (PR #15 merge) + prereg `0e464b3` |
| **Verdict** | **`STOP_FRAGILE`** |
| **science_open** | **false** (not widened) |
| **Harden V3** | 5 seeds; 60 ep (#14 corridor); cosine 1.5e-3→1.5e-4; equal-weight joint **(1/3)·HN+(1/3)·K16_sel+(1/3)·overall** @ T16 (K16 from select-aux `id_select_longhop`; HN_weight=1/3≤0.5) |
| **Prereg mean** | hard-neg≥0.95 **FAIL** (0.838±0.142); K16≥0.75 **PASS** (0.968±0.046) |
| **Seed-wise** | **1/5** PASS (goal ≥4/5 **FAIL**; ≤1/5 → STOP_FRAGILE) |

### 16.1 Matched-OOD T16 mean±std

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| PR #14 stabilize (n=5) | 0.929±0.035 | 0.957±0.061 | 0.863±0.143 |
| PR #15 V2 (n=5) | 0.813±0.157 | 0.955±0.101 | 0.423±0.477 |
| **This V3 (n=5)** | **0.890±0.067** | **0.838±0.142** | **0.968±0.046** |
| Untrained mean | 0.488±0.194 | 0.577±0.235 | — |

K16 select-aux rescued mean K16 (0.423→0.968) but traded HN (0.957→0.838). Inverted V2 failure mode. Prefer honesty: **STOP** this select-aux equal-weight line; retain #14 MEASURE_STILL as best stabilize evidence. Cite `docs/CYCLE_STALK_STABILIZE_V3.md`.

### 16.2 Non-rewrite rule

Prior seal body + §13 DEMOTION + §14 MEASURE_STILL + §15 STOP_FRAGILE are **not** rewritten. This §16 is append-only STOP_FRAGILE.

---

## 17. Objective V1 — STOP_FRAGILE (append-only; 2026-09-27 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_OBJECTIVE_V1` |
| **Artifact** | `artifacts/stalk_objective_v1.json` |
| **Harness** | `python -m reachability_gen.run_stalk_objective_v1` |
| **Base** | `main` `8866414` (PR #16 merge) + prereg `68a147a` |
| **Verdict** | **`STOP_FRAGILE`** |
| **science_open** | **false** (not widened) |
| **Objective V1** | 5 seeds; 60 ep (#14 corridor); cosine 1.5e-3→1.5e-4; **#14 select locked** (0.5·HN+0.5·ov; NO K16); train multinomial HN×2 + hop≥5×2 + weighted CE |
| **Prereg mean** | hard-neg≥0.95 **FAIL** (0.927±0.048); K16≥0.75 **PASS** (0.845±0.166) |
| **Seed-wise** | **0/5** PASS (goal ≥4/5 **FAIL**; ≤1/5 → STOP_FRAGILE) |

### 17.1 Matched-OOD T16 mean±std

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| PR #14 stabilize (n=5) | **0.929±0.035** | **0.957±0.061** | **0.863±0.143** |
| PR #15 V2 (n=5) | 0.813±0.157 | 0.955±0.101 | 0.423±0.477 |
| PR #16 V3 (n=5) | 0.890±0.067 | 0.838±0.142 | 0.968±0.046 |
| **This Obj V1 (n=5)** | **0.899±0.046** | **0.927±0.048** | **0.845±0.166** |
| Untrained mean | 0.488±0.194 | 0.577±0.235 | — |

ID train curriculum (HN+longhop upsample) missed mean HN floor and collapsed seed PASS to 0/5. Prefer honesty: **STOP** this objective line; retain #14 MEASURE_STILL as best stabilize evidence. Cite `docs/CYCLE_STALK_OBJECTIVE_V1.md`.

### 17.2 Non-rewrite rule

Prior seal body + §13 DEMOTION + §14 MEASURE_STILL + §15/§16 STOP_FRAGILE are **not** rewritten. This §17 is append-only STOP_FRAGILE.

## 18. Seed stability — MEASURE_ENVELOPE (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_SEED_STABILITY` |
| **Artifact** | `artifacts/stalk_seed_stability.json` |
| **Harness** | `python -m reachability_gen.run_stalk_seed_stability` |
| **Base** | `main` `4560d24` (PR #17 merge) + prereg `59337cb` |
| **Verdict** | **`MEASURE_ENVELOPE`** |
| **science_open** | **false** (not widened) |
| **Recipe** | **#14 frozen** exactly: 0.5·HN+0.5·ov select; 60ep cosine; hard-Â; **uniform ID** (NO upsample). Seeds **0..4** reconfirm #14 ckpts; **5..9** train identical harden. |
| **Prereg mean** | hard-neg≥0.95 **FAIL** (0.935±0.090); K16≥0.75 **PASS** (0.792±0.288) |
| **Seed-wise** | **3/10** PASS (rate 0.30; goal ≥8/10 **FAIL**) |
| **CI95** | HN [0.879, 0.991]; K16 [0.614, 0.971]; overall [0.870, 0.936] |

### 18.1 Matched-OOD T16 mean±std

| Arm | overall | hard-neg | K16 | seed PASS |
|-----|---------|----------|-----|-----------|
| PR #14 stabilize (n=5) | **0.929±0.035** | **0.957±0.061** | **0.863±0.143** | **2/5** |
| PR #15 V2 / #16 V3 / #17 ObjV1 | STOP_FRAGILE | | | ≤1/5 |
| **This envelope (n=10)** | **0.903±0.053** | **0.935±0.090** | **0.792±0.288** | **3/10** |
| Untrained mean (n=10) | 0.490±0.164 | 0.480±0.216 | — | — |

Wider seed panel under frozen #14 dips mean HN below floor and widens K16 std (seed6 K16=0.113). Prefer honesty: **MEASURE_ENVELOPE**; retain **#14 unchanged** as best MEASURE corridor. Cite `docs/CYCLE_STALK_SEED_STABILITY.md`.

### 18.2 Non-rewrite rule

Prior seal body + §13 DEMOTION + §14 MEASURE_STILL + §15/§16/§17 STOP_FRAGILE are **not** rewritten. This §18 is append-only MEASURE_ENVELOPE.


## 19. PARK — accept #14 MEASURE close (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_PARK_ACCEPT_MEASURE` |
| **Spec** | `docs/AUDIT-STALK-STABILIZE-V3.md` §3 |
| **Note** | `docs/CYCLE_STALK_PARK_ACCEPT_MEASURE.md` |
| **Base** | `main` `0a69b3f` (after PR #19) |
| **Verdict** | **`PARK_ACCEPT_MEASURE`** |
| **science_open** | **false** (not widened) |
| **Accepted corridor** | PR #14 `61314a0` hard-Â stalk **0.5·HN+0.5·overall** — **MEASURE_STILL** (HN **0.957±0.061** / K16 **0.863±0.143** PASS; seed **2/5**) |
| **Seed-fragile** | Cite #14 **2/5** + #18 envelope **3/10** (mean HN 0.935 FAIL) |
| **Closed** | V2/V3/ObjV1 **STOP_FRAGILE**; select / curriculum / seed-panel chase **PARKED** |
| **Training** | **None** this cycle |

### 19.1 Walk-away claim

Hard-Â stalk **#14** remains the best stabilize MEASURE corridor. Means cleared floors; seed fraction did not (≥4/5). Later select mixes and train curriculum inverted or missed floors. Wider seed panel under frozen #14 confirms fragility. Prefer honesty: **accept MEASURE_STILL**, park further stalk harden until an **orthogonal** cell. Do **not** reopen V4 select. `science_open=false`.

### 19.2 Non-rewrite rule

Prior seal body + §13 DEMOTION + §14 MEASURE_STILL + §15/§16/§17 STOP_FRAGILE + §18 MEASURE_ENVELOPE are **not** rewritten. This §19 is append-only PARK_ACCEPT_MEASURE.

## 20. Seed ensemble — PASS_CANDIDATE (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_SEED_ENSEMBLE` |
| **Artifact** | `artifacts/stalk_seed_ensemble.json` |
| **Harness** | `python -m reachability_gen.run_stalk_seed_ensemble` |
| **Base** | `main` `00b844b` (after PR #21) + prereg `e040270` |
| **Verdict** | **`PASS_CANDIDATE`** |
| **science_open** | **false** (not widened; FLAG human) |
| **Recipe** | Inference ensemble of **frozen #14/#18** hard-Â stalk singles (seeds 0..9). Primary **`prob_mean`**; secondary logit_mean / majority_vote. **No** new select/upsample; **0** fill-trains. |
| **Prereg ensemble floors** | hard-neg≥0.95 **PASS** (1.000); K16≥0.75 **PASS** (1.000) |
| **Lift vs singles mean** | Δoverall **+0.093**; ΔHN **+0.065**; ΔK16 **+0.208** |
| **LOO** | HN **1.000±0.000**; K16 **0.991±0.008**; overall 0.995±0.004 |

### 20.1 Matched-OOD T16

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| Singles mean±std (n=10) | 0.903±0.053 | 0.935±0.090 | 0.792±0.288 |
| **Ensemble `prob_mean`** | **0.996** | **1.000** | **1.000** |
| logit_mean / majority_vote | 0.990 | 1.000 | 0.975 |
| LOO mean±std | 0.995±0.004 | 1.000±0.000 | 0.991±0.008 |

Seed-fragile singles (3/10) do **not** top out the corridor under inference ensembling.
Select/curriculum remain **CLOSED**. Cite `docs/CYCLE_STALK_SEED_ENSEMBLE.md`.

### 20.2 Non-rewrite rule

Prior seal body + §13 DEMOTION + §14 MEASURE_STILL + §15/§16/§17 STOP_FRAGILE + §18 MEASURE_ENVELOPE + §19 PARK_ACCEPT_MEASURE are **not** rewritten. This §20 is append-only PASS_CANDIDATE (science_open=false).

## 21. Ensemble distill — STOP (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_ENSEMBLE_DISTILL` |
| **Artifact** | `artifacts/stalk_ensemble_distill.json` |
| **Harness** | `python -m reachability_gen.run_stalk_ensemble_distill` |
| **Base** | `main` `3972756` (PR #22) + prereg `6117eb3` |
| **Verdict** | **`STOP`** |
| **science_open** | **false** (not widened) |
| **Recipe** | Distill frozen #14/#18 **`prob_mean`** teacher → one hard-Â stalk student; α=0.5 τ=2.0; #14 select freeze; seeds **0,1,2** |
| **Prereg student mean** | hard-neg≥0.95 **FAIL** (0.843±0.146); K16≥0.75 **FAIL** (0.575±0.447); seed **0/3** |
| **Compare T16** | ens HN/K16 **1.000/1.000**; #14 seed0 **1.000/0.863**; student **0.843/0.575** |

### 21.1 Matched-OOD T16

| Arm | overall | hard-neg | K16 |
|-----|---------|----------|-----|
| Student mean±std (n=3) | 0.810±0.050 | 0.843±0.146 | 0.575±0.447 |
| Ensemble `prob_mean` | **0.996** | **1.000** | **1.000** |
| #14 seed0 | **0.975** | **1.000** | **0.863** |

Map≠location: ensemble map did **not** compress into one student under this
prereg. Negatives=mirror: HN floor fails mean. Prefer #14 MEASURE_STILL + #22
inference overlay; distill **STOP**. Cite `docs/CYCLE_STALK_ENSEMBLE_DISTILL.md`.

### 21.2 Non-rewrite rule

Prior seal body + §13–§20 are **not** rewritten. This §21 is append-only STOP
(science_open=false).

## 22. Seed ensemble — scoped science_open (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_SEED_ENSEMBLE` |
| **Artifact** | `artifacts/stalk_seed_ensemble.json` |
| **Seal note** | `docs/CYCLE_STALK_SEED_ENSEMBLE_SEAL.md` |
| **Harness** | `python -m reachability_gen.run_stalk_seed_ensemble` |
| **Base** | `main` `16e97fc` (after PR #23) + MEASURE merge PR #22 `3972756` |
| **Prereg / results** | `e040270` / `f774c69` |
| **Harness verdict** | **`PASS_CANDIDATE`** (harness kept `science_open=false`) |
| **science_open** | **true** (human seal; scoped claim only) |
| **Recipe** | Inference ensemble of **frozen #14/#18** hard-Â stalk singles (seeds 0..9). Primary **`prob_mean`**. **No** new select/upsample; **0** fill-trains. |
| **Prereg ensemble floors** | hard-neg≥0.95 **PASS** (1.000); K16≥0.75 **PASS** (1.000) |
| **Lift vs singles mean** | Δoverall **+0.093**; ΔHN **+0.065**; ΔK16 **+0.208** |
| **LOO** | HN **1.000±0.000**; K16 **0.991±0.008**; overall 0.995±0.004 |

### 22.1 Exact claim (narrow)

| Field | Value |
|-------|-------|
| **science_open** | **true** |
| **Claim** | Inference-time `prob_mean` ensemble of frozen hard-Â stalk checkpoints from PR #14 (seeds 0..4) and PR #18 (seeds 5..9) clears matched-OOD T16 hard-neg (≥0.95) and K16 (≥0.75) floors with positive lift vs the mean of those singles on `covariate_matched_ood` + hard `A_ij` — **ensemble-at-eval only**. |
| **Substrate** | `covariate_matched_ood` + **hard** `A_ij` mask; stalk-local FractalCore (no `c` broadcast, no soft ACT); discrete T∈{6,8,12,16}; floors gate **T16** |
| **Evidence** | Artifact `stalk_seed_ensemble.json`; seal note; §20 PASS_CANDIDATE table |
| **SHAs** | MEASURE merge PR #22 `3972756`; this seal on `main` after PR #23 `16e97fc` |

### 22.2 Explicit NON-claims

| NON-claim | Status |
|-----------|--------|
| Singles / one stalk location OPEN | **NO** — remain **MEASURE_STILL**/fragile (#14 2/5; #18 3/10) |
| Distill / student compression | **STOP** (§21 / PR #23) — map≠location |
| Select / curriculum reopen | **CLOSED** (§19) |
| Sheaf unsupervised | **NOT opened** |
| Hist. §6 single-seed OPEN revive | **NO** — stays demoted (§13) |

### 22.3 Non-rewrite rule

Prior seal body + §13 DEMOTION + §14–§21 are **not** rewritten. This §22 is append-only scoped science_open (ensemble-at-eval only).

## 23. SWA persist — MEASURE (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_SWA_PERSIST` |
| **Artifact** | `artifacts/stalk_swa_persist.json` |
| **Cycle note** | `docs/CYCLE_STALK_SWA_PERSIST.md` |
| **Harness** | `python -m reachability_gen.run_stalk_swa_persist` |
| **Base** | `main` `ecd0489` (after PR #24 ensemble seal) |
| **Prereg / results** | `6089d31` / `7767bca` |
| **Harness verdict** | **`MEASURE`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Recipe** | Frozen #14 harden + Polyak SWA `ep≥31`; primary=SWA; paired #14 lex select; seeds **0..4**. **Not** soft distill. |
| **SWA mean floors** | hard-neg **0.867 FAIL** (≥0.95); K16 **0.910 PASS** (≥0.75); seed PASS **2/5** |
| **vs within-run select** | ΔHN **−0.090**; ΔK16 **+0.047** (HN↔K16 trade) |
| **vs ensemble** | far below (ens HN/K16 **1.000**) |

### 23.1 Reading

One-train SWA is **not** a single-model substitute for the §22 ensemble map.
Partial K16 persist vs select; HN regresses. Prefer #14 select + #22 overlay.
Select/curriculum stay CLOSED. Distill remains STOP.

### 23.2 Non-rewrite rule

Prior seal body + §13–§22 are **not** rewritten. This §23 is append-only MEASURE
(`science_open=false`).

## 24. Epistemic disagreement — MEASURE/STOP (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_EPISTEMIC_DISAGREEMENT` |
| **Artifact** | `artifacts/stalk_epistemic_disagreement.json` |
| **Cycle note** | `docs/CYCLE_STALK_EPISTEMIC_DISAGREEMENT.md` |
| **Harness** | `python -m reachability_gen.run_stalk_epistemic_disagreement` |
| **Base** | `main` `b051e15` (after PR #25 SWA) |
| **Prereg / results** | `daaeb21` / `9f1ee3e` |
| **Audit verdict** | **`AUDIT_LIFTS_ON_DISAGREEMENT`** |
| **Train verdict** | **`STOP`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Audit** | #14/#18 ens T16 pairwise disagree **0.162**; epistemic H **0.192** > aleatoric **0.080**; median lift_gap Δov/HN/K16 **+0.175 / +0.164 / +0.221** |
| **Multi-hyp** | H=3; CE−λ·JS (λ=0.5); seeds **0,1,2**; mean HN **0.942 FAIL** / K16 **0.712 FAIL**; within-head disagree **~0.009** (collapsed); 0/3 |

### 24.1 Reading

#22 lift **rides on disagreement**. Multi-hyp structured diversity did **not**
persist (heads collapsed; both floors miss vs #14 seed0). Prefer #14 + #22
overlay. Distill STOP; SWA MEASURE; select/curriculum CLOSED.

### 24.2 Non-rewrite rule

Prior seal body + §13–§23 are **not** rewritten. This §24 is append-only
MEASURE/STOP (`science_open=false`).

## 25. Bag diversity — MEASURE_LIFT (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_BAG_DIVERSITY` |
| **Artifact** | `artifacts/stalk_bag_diversity.json` |
| **Cycle note** | `docs/CYCLE_STALK_BAG_DIVERSITY.md` |
| **Harness** | `python -m reachability_gen.run_stalk_bag_diversity` |
| **Base** | `main` `f210d0c` (after PR #26 epistemic disagreement) |
| **Prereg / results** | `c58ab73` / `9cabd4c` |
| **Harness verdict** | **`MEASURE_LIFT`** |
| **Disagreement diagnostic** | **`DISAGREE_GE_REF`** (bag pair **0.431** ≥ #22 **0.162**) |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Recipe** | Frozen #14 harden + **bootstrap** ID train + **graph-subspace** EDGE_KEEP_P=0.75; **separate params** per member (DGE-style); NOT multi-hyp / soft distill / SWA-only |
| **Bag ens floors** | hard-neg **0.821 FAIL** (≥0.95); K16 **0.788 PASS** (≥0.75); member PASS **0/5** |
| **vs bag singles** | ΔHN **+0.101**; ΔK16 **+0.250**; Δoverall **+0.102** |
| **vs ensemble #22** | far below (ens HN/K16 **1.000**) |

### 25.1 Reading

Induced bag diversity raises pairwise disagreement above the #22 seed map but
does **not** clear HN floors — weak singles + over-dispersion. Prefer #14 + #22
overlay. Distill STOP; SWA MEASURE; multi-hyp STOP; select/curriculum CLOSED.

### 25.2 Non-rewrite rule

Prior seal body + §13–§24 are **not** rewritten. This §25 is append-only
MEASURE_LIFT (`science_open=false`).

## 26. Competent dissonance — MEASURE audit (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_COMPETENT_DISSONANCE` |
| **Artifact** | `artifacts/stalk_competent_dissonance.json` |
| **Cycle note** | `docs/CYCLE_STALK_COMPETENT_DISSONANCE.md` |
| **Harness** | `python -m reachability_gen.run_stalk_competent_dissonance` |
| **Base** | `main` `07401f0` (after PR #27 bag diversity) |
| **Prereg / results** | `3e20a4d` / `9c4e8f9` |
| **Cycle verdict** | **`COMPETENT_vs_CHAOS`** |
| **#22 / #27** | **`COMPETENT`** (CD **0.811**) / **`CHAOS`** (CD **0.658**) |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Recipe** | Eval-only; no train / no bag noise; CD = μ_acc · min(D_hard, 0.25)/0.25; D_hard = 0.5·(D_HN+D_K16) |
| **Matched-OOD T16** | #22 global pair **0.162**, D_hard **0.225** (K16 **0.337**); #27 global **0.431**, D_hard **0.466**; disagree-set member ov **0.817** vs **0.585** |
| **ood_hops stress** | #22 ens HN **1.000→0.067** shatter residue (K16 holds); do **not** widen §22 |

### 26.1 Reading

#22 is **competent dissonance** on sealed matched-OOD — low global disagree ≠ echo when lift is disagreement-localized (§24) and hard-slice D_hard is material. #27 is **chaos** (weak members + over-dispersion). Prefer #14 + #22 overlay; do not train bag noise. Select/curriculum CLOSED.

### 26.2 Non-rewrite rule

Prior seal body + §13–§25 are **not** rewritten. This §26 is append-only MEASURE
(`science_open=false`).

## 27. RED competent dissonance — MEASURE audit (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_RED_COMPETENT_DISSONANCE` |
| **Artifact** | `artifacts/stalk_red_competent_dissonance.json` |
| **Gen report** | `artifacts/stalk_red_competent_dissonance_generation_report.json` |
| **Data** | `data/stalk_red_competent_dissonance.jsonl` |
| **Cycle note** | `docs/CYCLE_STALK_RED_COMPETENT_DISSONANCE.md` |
| **Harness** | `python -m reachability_gen.run_stalk_red_competent_dissonance` |
| **Base** | `main` `0a5890a` (after PR #28) |
| **Prereg / results** | `8d83560` / `e4f9b365d859b079802ccf5749e7df36cb149a42` |
| **Cycle verdict** | **`FAIL_CLOSED_DOMINANT`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Recipe** | Eval-only #22 ens; cite#28 strip-easy boundary + refresh; RED denser-K16 (p_emp≫ood_hops) + LONG_K20; FAIL_OPEN vs FAIL_CLOSED; **no** #27 bag train |
| **Boundary** | refresh Δ=0 vs #28 (D_HN **0.112** / D_K16 **0.337** / CD **0.811** COMPETENT) |
| **RED T16** | ens ov **0.762** / HN **0.898** / K16 **0.812** / K20 **0.438**; D_hard **0.277**; CD **0.774**; **0** FAIL_OPEN / **61** FAIL_CLOSED |

### 27.1 Reading

#22 COMPETENT map on matched-OOD **fails closed** under RED (denser K16 + K20 outside train/ADR-OOD priors): ens-wrong examples carry disagreement and/or uncertainty — **not** unified confident wrongs. K20@T16 under-horizon residue expected; do **not** widen §22. Prefer #14 + #22 overlay; no bag train.

### 27.2 Non-rewrite rule

Prior seal body + §13–§26 are **not** rewritten. This §27 is append-only MEASURE
(`science_open=false`).

## 28. Hop-OOD HN overlay stress — MEASURE audit (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_HOP_OOD_HN` |
| **Artifact** | `artifacts/stalk_hop_ood_hn.json` |
| **Log** | `artifacts/stalk_hop_ood_hn_run.log` |
| **Cycle note** | `docs/CYCLE_STALK_HOP_OOD_HN.md` |
| **Harness** | `python -m reachability_gen.run_stalk_hop_ood_hn` |
| **Base** | `main` `5c5006a` (after PR #29) |
| **Prereg / harness / results** | `a40abb6` / `051743c` / `4da28fc7ac42a798d00f7fe200c81b2dd6e6f950` |
| **Cycle verdict** | **`FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Recipe** | Eval-only #22 ens on `ood_hops`@T16; FAIL_OPEN/CLOSED + CD; gate abstain + majority_vote; **no** train / bag / select |
| **Cite #28 replicate** | exact (ens HN **0.067** / K16 **0.988** / CD **0.553**) |
| **FAIL mode** | **45** FAIL_OPEN / **190** FAIL_CLOSED (all OPEN = HN unified confident wrong) |
| **Overlays** | vote HN **0.067** (no lift); gate accepted HN **0.000** (concentrates FAIL_OPEN core) |

### 28.1 Reading

#28 hop-OOD HN shatter **confirmed**. Cheap inference overlays do **not** repair HN; the prereg gate keeps the agree/high-conf FAIL_OPEN hard-neg core. Residue: **`HN_FAIL_OPEN_CORE`** (45). Prefer #14 + #22 on matched-OOD only. Do **not** widen §22. No bag train.

### 28.2 Non-rewrite rule

Prior seal body + §13–§27 are **not** rewritten. This §28 is append-only MEASURE
(`science_open=false`).

## 29. HN FAIL_OPEN autopsy — MEASURE (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_HN_FAIL_OPEN_AUTOPSY` |
| **Artifact** | `artifacts/stalk_hn_fail_open_autopsy.json` |
| **Log** | `artifacts/stalk_hn_fail_open_autopsy_run.log` |
| **Cycle note** | `docs/CYCLE_STALK_HN_FAIL_OPEN_AUTOPSY.md` |
| **Harness** | `python -m reachability_gen.run_stalk_hn_fail_open_autopsy` |
| **Base** | `main` `dd18ff8` (after PR #30) |
| **Prereg / harness / results** | `931bd78` / `3980451` / `81d7911be8ad9507385bee63e3187bf0441b4fbe` |
| **Cycle verdict** | **`STRUCTURAL_CLUSTER`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Recipe** | Eval-only structural autopsy of #30 FO=45 @ T=16; FO vs FC_HN vs OK_HN; member logits; **no** train |
| **Cite #30 replicate** | exact (FO **45** / FC **190** / ens HN **0.067**) |
| **Cluster** | isolated-source / hub-target (FO med R_out(s)=**1**, R_in(t)=**27**; OK inverse) |
| **Members** | HARD_UNANIMOUS **34** / SOFT_AGREE **11** |
| **Residue** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER`** |

### 29.1 Reading

#30 HN_FAIL_OPEN_CORE is a **structural cluster**, not diffuse: ens unifies confident false-reachability when s is out-isolated and t is an in-hub. Prefer #14 + #22 on matched-OOD only. Do **not** widen §22. No bag train. Corridor not parked — autopsy only.

### 29.2 Non-rewrite rule

Prior seal body + §13–§28 are **not** rewritten. This §29 is append-only MEASURE
(`science_open=false`).

## 30. Sound outdeg gate — MEASURE (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_SOUND_OUTDEG_GATE` |
| **Artifact** | `artifacts/stalk_sound_outdeg_gate.json` |
| **Log** | `artifacts/stalk_sound_outdeg_gate_run.log` |
| **Cycle note** | `docs/CYCLE_STALK_SOUND_OUTDEG_GATE.md` |
| **Harness** | `python -m reachability_gen.run_stalk_sound_outdeg_gate` |
| **Base** | `main` `c909d1b` (after PR #31) |
| **Prereg / harness / results** | `3f4d791` / `6a75b1f` / `32420d09bf3d141ec137a511c1034645d5304bb5` |
| **Cycle verdict** | **`FO_PARTIAL`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Recipe** | Eval-only sound local `outdeg(s)==0 ∧ t≠s` → force ens pred=0 on #22 `prob_mean` @ T=16; matched-OOD collateral; **no** train; **no** BFS oracle |
| **Cite #30 replicate** | exact (FO **45** / FC **190** / ens HN **0.067**) |
| **ood_hops gated** | HN **0.304** / K16 **0.988** / ov **0.629**; FO **22** / FC **156**; **FO killed 23/45** |
| **Matched-OOD collateral** | overall/HN/K16 Δ=**0.000** (97 triggers already pred=0) |
| **Residue** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL`** |

### 30.1 Reading

Local incidence outdeg(s)==0 kills the outdeg0 half of the isolated-source FO cluster (**23/45**). Remaining 22 FO have outdeg>0 — outside this sound local gate (BFS out-closure prohibited). Matched-OOD competence untouched. Prefer #14 + #22 on matched-OOD only. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Corridor not parked.

### 30.2 Non-rewrite rule

Prior seal body + §13–§29 are **not** rewritten. This §30 is append-only MEASURE
(`science_open=false`).

## 31. HN FO remainder autopsy — MEASURE (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_HN_FO_REMAINDER_AUTOPSY` |
| **Artifact** | `artifacts/stalk_hn_fo_remainder_autopsy.json` |
| **Log** | `artifacts/stalk_hn_fo_remainder_autopsy_run.log` |
| **Cycle note** | `docs/CYCLE_STALK_HN_FO_REMAINDER_AUTOPSY.md` |
| **Harness** | `python -m reachability_gen.run_stalk_hn_fo_remainder_autopsy` |
| **Base** | `main` `db8c2b1` (after PR #32) |
| **Prereg / harness / results** | `bd7e440` / `a098584` / `c8b034631c31d86daf4a575c88b40a7f0a14e3b0` |
| **Cycle verdict** | **`LOCAL_SOUND_WALL`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Recipe** | Offline autopsy of **22** FO remainder (outdeg>0) vs killed/OK/FC; local-sound cut hunt (indeg_t0 / deadend-nbrs); coverage ≥8/22 required for overlay; **no** train; **no** BFS gate |
| **Cite #32** | rem **22** / kil **23** / union **45** exact |
| **Best sound coverage** | **5/22** (`C_deadend_nbrs`); indeg_t0 **0/22** |
| **Gate overlay** | **not run** |
| **Residue** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL`** |

### 31.1 Reading

#32 outdeg0 extracted the local-incidence half of the isolated-source FO cluster. Remaining **22** have outdeg>0 and median `|R_out(s)|=19` / `max_dist=6` — overlap OK_HN/FC_HN multi-hop geometry. No sound local cut covers ≥8/22. **Stop overlay chase** for this FO core. Prefer #14 + #22 on matched-OOD only. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Do **not** BFS-gate.

### 31.2 Non-rewrite rule

Prior seal body + §13–§30 are **not** rewritten. This §31 is append-only MEASURE
(`science_open=false`).

## 32. Hop-OOD overlay chase — PARK (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_HOP_OOD_OVERLAY_PARK` |
| **Cycle note** | `docs/CYCLE_STALK_HOP_OOD_OVERLAY_PARK.md` |
| **PR** | **#34** |
| **Base** | `main` `a50894d` (after PR #33) |
| **Verdict** | **`PARK_HOP_OOD_OVERLAY`** after `LOCAL_SOUND_WALL` |
| **science_open** | **false** for this cycle; no widening; existing scoped §22 ensemble OPEN unchanged |
| **Scope** | Park the hop-OOD **overlay chase only**; do not park the whole stalk corridor or §22 |
| **Facts** | #30 HN_SHATTER_CONFIRMED + FAIL_CLOSED_DOMINANT (45 FAIL_OPEN); #31 STRUCTURAL_CLUSTER (isolated-s / hub-t); #32 FO_PARTIAL (23/45 killed by sound outdeg0; HN 0.067→0.304; matched Δ=0); #33 remainder 22 `DIFFUSE_MULTI_HOP_LIKE_OK_FC`, `LOCAL_SOUND_WALL`, best sound cut 5/22 |
| **Residue** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL`** |
| **Hygiene** | `outdeg(s)==0` remains optional sound **MEASURE** hygiene only; not `science_open` |

### 32.1 Reading

The #30–#33 hop-OOD overlay chase is frozen at `LOCAL_SOUND_WALL`. Stop further local-sound gates for this FO core; do not BFS-gate, train, re-run evals, or widen §22. The matched-OOD ensemble OPEN is unchanged. Named forks outside this chase are multi-hop competence / path reasoning and a separately attributed train/retrain experiment; neither started here.

### 32.2 Non-rewrite rule

Prior seal body + §13–§31 are **not** rewritten. This §32 is append-only documentation PARK (`science_open=false` for this cycle).

## 33. Reach certificates — CERT_FO_CATCH (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_REACH_CERTIFICATES` |
| **Artifact** | `artifacts/stalk_reach_certificates.json` |
| **Log** | `artifacts/stalk_reach_certificates_run.log` |
| **Cycle note** | `docs/CYCLE_STALK_REACH_CERTIFICATES.md` |
| **Harness** | `python -m reachability_gen.run_stalk_reach_certificates` |
| **Base** | `main` `3730558` (after PR #34) |
| **Prereg / harness / results** | `759357f` / `b856104` / `1abd8904194bcc5eb72fa036328d543a4cb542b2` |
| **Cycle verdict** | **`CERT_FO_CATCH`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Tropical Phase 2** | **not started** |
| **Recipe** | Post-hoc path-witness / checker-BFS certificates on frozen #14/#18/#22 ens; dirty YES → force-closed; dirty NO kept; arms baseline / outdeg0 / cert / outdeg0+cert; **no** train; BFS **checker post-hoc only** |
| **Cite #30/#32** | FO **45** exact; rem22 **22** exact; matched Δ=#32 shape |
| **FO catch** | **45/45** (cert); rem-22 **22/22**; HN **0.067→1.000**; K16 holds |
| **Matched-OOD Δ** | **0.000** (dirty_rate **0.004**) |
| **Residue** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL/CERT_FO_CATCH`** |

### 33.1 Reading

Post-hoc certificates (path-witness validation / checker BFS) force-close dirty YES claims without a validating path on Â. Catches the full hop-OOD FAIL_OPEN core including the #33 rem-22 multi-hop remainder. Matched-OOD competence untouched (Δ=0). Checker BFS is **not** a model feature / train target / init. Prefer #14 + #22 on matched-OOD only. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Tropical Phase 2 **not** started. Still **MEASURE**.

### 33.2 Non-rewrite rule

Prior seal body + §13–§32 are **not** rewritten. This §33 is append-only MEASURE
(`science_open=false`).

## 34. Tropical ens aggregation probe — COLLATERAL_HARM (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_TROPICAL_ATTENTION_PROBE` |
| **Artifact** | `artifacts/stalk_tropical_attention_probe.json` |
| **Log** | `artifacts/stalk_tropical_attention_probe_run.log` |
| **Cycle note** | `docs/CYCLE_STALK_TROPICAL_ATTENTION_PROBE.md` |
| **Harness** | `python -m reachability_gen.run_stalk_tropical_attention_probe` |
| **Base** | `main` `86d1ff0` (after PR #35) |
| **Prereg / harness / results** | `30f57de` / `ffd1990` / `698112f54951a47d53e0a859cf3c64039281aac0` |
| **Cycle verdict** | **`COLLATERAL_HARM`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Tropical definition** | scoped ens **`logit_max`** (`score_c = max_m L[m,c]`) + **`beta_inf_member`**; **not** in-attn MHA rewrite |
| **Recipe** | Eval-only on frozen #14/#18/#22 ens; ood_hops@T16 + matched-OOD collateral; **no** train / anneal / DEAR |
| **FO catch** | **0/45** baseline FO; rem-22 **0/22**; HN 0.067→0.150 (FO core untouched) |
| **Matched-OOD Δ** | overall **−0.104**; K16 **−0.488** (COLLATERAL_HARM) |
| **Cert reference** | #35 CERT_FO_CATCH (45/45) — tropical ≢ certificate |
| **Residue** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL/CERT_FO_CATCH/COLLATERAL_HARM`** |

### 34.1 Reading

Tropical ens aggregation does not move the hop-OOD FAIL_OPEN core and harms matched-OOD K16 competence that `prob_mean` carries. The ens-layer "sum-product bleed → rem-22 FO" hypothesis is falsified as a repair lever under this scoped probe. Prefer #14 + #22 `prob_mean` on matched-OOD; keep #35 post-hoc certificates for FO catch. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Do **not** equate tropical with certificates. Still **MEASURE**.

### 34.2 Non-rewrite rule

Prior seal body + §13–§33 are **not** rewritten. This §34 is append-only MEASURE
(`science_open=false`).

## 35. Orientation collapse probe — INCONCLUSIVE_ARCH (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_ORIENTATION_COLLAPSE_PROBE` |
| **Artifact** | `artifacts/stalk_orientation_collapse_probe.json` |
| **Log** | `artifacts/stalk_orientation_collapse_probe_run.log` |
| **Cycle note** | `docs/CYCLE_STALK_ORIENTATION_COLLAPSE_PROBE.md` |
| **Harness** | `python -m reachability_gen.run_stalk_orientation_collapse_probe` |
| **Base** | `main` `3198371` (after PR #36) |
| **Prereg / harness / results** | `a422eef` / `5d6c1a8` / `de5d1f5f4b7468960fa241274f1523e935708179` |
| **Cycle verdict** | **`INCONCLUSIVE_ARCH`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Proxy** | directed incidence × final hidden; **no** explicit Dir-GNN in/out channels; attn weights unused |
| **Recipe** | Eval-only on frozen #14/#18/#22 ens; ood_hops@T16 strata OK_HN/FO_HN/rem-22/FO_KILLED/FC_HN; **no** train / Phase4 / tropical retrain |
| **Key numbers** | OK_HN cos nan_rate **1.00** (empty_in_t=1.0); FO/rem cos_t ≈ **0.91**; (−l2_t) AUROC **1.000** confounded |
| **rem-22 separates?** | cos **no** (OK control unusable); l2 **yes but confounded** by empty-in |
| **Residue** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL/CERT_FO_CATCH/COLLATERAL_HARM/INCONCLUSIVE_ARCH`** |

### 35.1 Reading

Dir-GNN-style orientation proxy is **inconclusive** on this architecture/strata: OK_HN targets are all in-empty, so cos(h_in,h_out) cannot score FO vs OK. FO/rem-22 live in bidirectional target cones (high cos≈0.91) but that alone does not seal orientation collapse as the rem-22 lever. l2 gaps vs OK are structural empty-in confounds. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Do **not** start Phase 4 from this probe. Prefer #14 + #22 `prob_mean` on matched-OOD; keep #35 certificates for FO catch. Still **MEASURE**.

### 35.2 Non-rewrite rule

Prior seal body + §13–§34 are **not** rewritten. This §35 is append-only MEASURE
(`science_open=false`).

## 36. Energy selector — COLLATERAL_HARM (append-only; 2026-09-28 CDT)

| Field | Value |
|-------|-------|
| **Cycle** | `CYCLE_STALK_ENERGY_SELECTOR` |
| **Artifact** | `artifacts/stalk_energy_selector.json` |
| **Log** | `artifacts/stalk_energy_selector_run.log` |
| **Cycle note** | `docs/CYCLE_STALK_ENERGY_SELECTOR.md` |
| **Harness** | `python -m reachability_gen.run_stalk_energy_selector` |
| **Base** | `main` `39e3aa7` (after PR #37) |
| **Prereg / harness / results** | `eecc4b7` / `79f7196` / `bbae5a757674b2c4a593a94d1b32ea410fbe92a7` |
| **Cycle verdict** | **`COLLATERAL_HARM`** |
| **science_open** | **false** (not widened; §22 ensemble-at-eval scope **unchanged**) |
| **Exact E_free** | `10·E_sound + 1·E_cone + 0.1·(−log(p+1e-8))` (cert-free; no BFS; no orientation) |
| **Exact E_cert** | `10·E_disagree + 0.1·(−log(p+1e-8))` (cert-energy secondary) |
| **Recipe** | Eval-only on frozen #14/#18/#22 ens; energy_argmin / weighted / oracle / cert / refuse; ood_hops@T16 + matched-OOD; **no** train / orientation / tropical anneal |
| **FO catch (E_free)** | **0/45** baseline FO; rem-22 **0/22**; oracle also **0/45** (HARD_UNANIMOUS) |
| **HN / gap** | HN 0.067→0.812; gap_close **+0.081** (non-FO routing); hop-OOD K16 **0.988→0.438** |
| **Matched-OOD Δ** | overall **−0.327**; K16 **−0.938** (COLLATERAL_HARM) |
| **Cert refuse** | 45/45 FO + 22/22 rem-22 (tracks #35; not cert-free value) |
| **Energy↔conf** | r≈**−0.049**; sound/cone differs — not just-confidence |
| **Residue** | **`HN_FAIL_OPEN_CORE/STRUCTURAL_CLUSTER/OUTDEG0_PARTIAL/LOCAL_SOUND_WALL/CERT_FO_CATCH/COLLATERAL_HARM/INCONCLUSIVE_ARCH/COLLATERAL_HARM`** |

### 36.1 Reading

Cert-free energy member selection does **not** repair the hop-OOD FAIL_OPEN core (generation wall: no member correct on the 45 FO) and **harms** matched-OOD K16 competence that `prob_mean` carries. HN/gap movement is non-FO routing under a collapsed K16. Cert-tied argmin alone ≢ #35; refuse-when-all-disagree tracks certificates. Prefer #14 + #22 `prob_mean` on matched-OOD; keep #35 post-hoc certificates for FO catch. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Still **MEASURE**.

### 36.2 Non-rewrite rule

Prior seal body + §13–§35 are **not** rewritten. This §36 is append-only MEASURE
(`science_open=false`).
