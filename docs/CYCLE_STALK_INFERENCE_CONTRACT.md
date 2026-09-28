# CYCLE_STALK_INFERENCE_CONTRACT — MEASURE inference-contract lock (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** formalization of frozen-forward **inference contract** around #22 ens + #35 certificates (llama.cpp-abstraction frame) |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | After #35 CERT_FO_CATCH + #39 PASS_REGRESSION + #40 ENERGY_NULL + #41 COLLATERAL_HARM: lock certificates as **execution witnesses** into a first-class API, not one-off scripts |
| **Base** | `main` tip after PR #41 (`c13aec6`) |
| **Prior** | #22 ens `prob_mean`; #30 FO=45 HN shatter; #31 HARD_UNANIMOUS; #35 CERT_FO_CATCH; #39 FO/cert regression; #40/#41 energy/ζ ≢ cert |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Train / PEFT / select / bag / JS / sheaf / ggml** | **NONE** — no llama.cpp/ggml port; no quant-as-FO-fix; no sheaf train; no ens remix |

## Goal (llama.cpp analogy → code)

Map the **frozen forward interpreter + explicit inference contract + certificates as execution witnesses** frame onto stalk/#22 ens eval:

| llama.cpp-ish concept | This repo lock |
|-----------------------|----------------|
| Frozen GGUF weights | Frozen #14/#18/#22 ckpts (seeds 0..9) |
| Fixed decode schedule | Fixed **T=16** + ens agg **`prob_mean`** (#22) |
| Explicit forward artifacts | Â, predicted reach, member votes, **HARD_UNANIMOUS** |
| Post-hoc witness / checker | Reachability certificate (clean/dirty) from #35 |
| Refuse / abort | dirty cert ⇒ **FAIL_CLOSED** / no YES claim (API, not one-off) |
| CI gate | Keep/extend #39 fo_cert regression (unanimity + cert + shatter smoke) |

**Not** a ggml/kernel port. **Not** quant as magic FO fix. **Not** sheaf train. **Not** ens remix. **Not** reopen zeta/energy refuse.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) — **do not widen** |
| #30–#34 | FO core + LOCAL_SOUND_WALL; overlay parked |
| #35 | CERT_FO_CATCH — **reference** refuse |
| #36–#38 | tropical / orientation / energy-selector — not FO repair |
| #39 | PASS_REGRESSION CI lock |
| #40 | ENERGY_NULL — Id coboundary ≢ cert |
| #41 | COLLATERAL_HARM — ζ ≢ cert |

## Metaphor

- **map ≠ location** — #22 COMPETENT map predicts; the contract is the *interpreter schedule + witness check*, not a new learned map.
- **negatives = mirror** — dirty YES without a path-witness is the structural shadow that FAIL_OPEN false-reachability casts; refuse closes it.

## Cite #30 / #35 / #39 / #40 / #41 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN (baseline / cert-off) | **0.067** |
| ens K16 | **0.988** |
| FAIL_OPEN baseline | **45** |
| rem-22 | **22** |
| #35 cert FO / rem22 | **45/45** / **22/22**; matched Δ=**0** |
| #39 regression | **PASS_REGRESSION** |
| #40 energy refuse | **0**/45 FO (≢ cert) |
| #41 ζ refuse | **0**/45 FO (≢ cert) |

## Preregistered protocol (LOCKED before runs)

```bash
# Unit / API
pytest -q tests/test_stalk_inference_contract.py tests/test_stalk_fo_cert_regression.py

# MEASURE harness (artifact-first default; optional --live)
python -m reachability_gen.run_stalk_inference_contract
# or: reachability-stalk-inference-contract
```

| Item | Spec (locked) |
|------|----------------|
| Contract module | `reachability_gen.inference_contract` |
| Entrypoint | `InferenceContract` + `run_with_certificates` |
| Checkpoints | `#14` seeds 0..4 + `#18` seeds 5..9 (same as #22/#35) |
| T / agg | **T=16** fixed; **`prob_mean`** only (no ens remix) |
| Refuse | dirty YES → force pred=0; dirty NO → keep (fail-closed; no oracle open) |
| Train | **NONE** |
| Gen | **NONE** — do not regenerate `ood_hops.jsonl` |
| Primary tables | reuse sealed #30/#35/#40/#41 (+ optional live re-infer) |
| Artifact | `artifacts/stalk_inference_contract.json` |
| Receipt (optional) | `artifacts/stalk_inference_contract_receipt.json` (GGUF-like auditability) |

### Contract surface (LOCKED)

`run_with_certificates` **always** emits per example / batch:

| Field | Meaning |
|-------|---------|
| `pred_raw` | ens `prob_mean` before cert |
| `pred_contract` | after refuse policy |
| `refuse` | True iff dirty YES forced closed |
| `cert` | `ReachCertificate` (clean/dirty/kind/witness) |
| `HARD_UNANIMOUS` | all members pred=1 and conf≥0.80 (#31) |
| `member_votes` | per-member hard preds |
| `matched_floors` | matched-OOD Δ vs baseline when matched rows supplied |

### MEASURE suite (LOCKED)

| Arm | Spec |
|-----|------|
| **A** | Formalize API; unit tests prove refuse + receipt |
| **B** | Tables: FO catch rem-22; matched-OOD Δ; ≢ vs energy(#40)/ζ(#41); CD if present in cites |
| **C** | Stress: **cert-on vs cert-off** ablation on hop-OOD (expect FO returns without cert) |
| **D** | Optional: machine-readable run receipt (layout hash, T, ens seeds, cert status) |
| **E** | Ledger + seal; honest verdict |

### Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`CONTRACT_LOCKED`** | API present + cert-on FO catch **45/45** + rem22 **22/22** + matched abs Δ ≤ **0.01** + cert-off FO **==45** (FO returns) + energy/ζ refuse ≢ cert (cite #40/#41 FO refuse ==0) + `science_open=false` + #39 gate still PASS |
| **`PARTIAL`** | API works; some table floors soft-miss; no matched harm |
| **`INVALID`** | Contract broken (refuse inverted / floors inverted / science_open stamped true) |
| **`COLLATERAL`** | matched-OOD overall/HN/K16 abs Δ ≥ **0.05** under cert-on |

Priority: `INVALID` > `COLLATERAL` > else floors → `CONTRACT_LOCKED` / `PARTIAL`.

**Even `CONTRACT_LOCKED` does not widen `science_open` / does not claim hop-OOD OPEN / does not widen §22.**

### Prohibited defaults

- No llama.cpp / ggml kernel port
- No quant-as-FO-fix claim
- No sheaf train / unsupervised sheaf
- No tropical ens / ens agg remix
- No zeta/energy refuse reopen as primary
- No T>16; no curriculum/distill/JS
- No `science_open=true`; no §22 widen
- No claiming hop-OOD OPEN
- Do not regenerate `ood_hops.jsonl`

## Explicit non-goals

- No ggml/PEFT/sheaf/tropical/ζ/energy chase
- No train / bag / select / curriculum reopen
- No `science_open=true`
- No hop-OOD OPEN claim

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_inference_contract.json` |
| **Receipt** | `artifacts/stalk_inference_contract_receipt.json` |
| **Cycle verdict** | **`CONTRACT_LOCKED`** |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Train / gen** | **none** |
| **Live elapsed** | ~14.8 s CDT |
| **Base SHA** | `c13aec6` (post-#41) |

### Ablation (cert-on vs cert-off) — hop-OOD T16

| Arm | FO | HN | K16 | overall | rem22 killed |
|-----|----|----|-----|---------|--------------|
| cert-off (#35 baseline / #30) | **45** | **0.067** | **0.988** | **0.510** | — |
| cert-on (`run_with_certificates`) | **0** | **1.000** | **0.988** | **0.977** | **22**/22 |

Live contract: `fo_still_wrong_raw=45`; `fo_killed_contract=45`; `rem22_killed_contract=22`; `n_refuse=224` dirty YES.

### Matched-OOD Δ (live + sealed)

| Slice | Δ (cert − baseline) |
|-------|---------------------|
| overall | **0.000** |
| HN | **0.000** |
| K16 | **0.000** |

### ≢ vs energy(#40) / ζ(#41)

| Instrument | FO refuse killed | Verdict |
|------------|------------------|---------|
| #35 cert | **45**/45 | CERT_FO_CATCH |
| #40 Id coboundary energy | **0**/45 | ENERGY_NULL |
| #41 spectral ζ | **0**/45 | COLLATERAL_HARM |

### CI / HARD_UNANIMOUS

| Gate | Value |
|------|-------|
| #39 fo_cert regression | **PASS_REGRESSION** |
| #31 HARD_UNANIMOUS among FO | **34**/45 |
| P(correct|clean) | **1.000** |

### Reading (fail-closed)

1. **CONTRACT_LOCKED:** `InferenceContract` + `run_with_certificates` are the API lock of #35 refuse around frozen #22 ens. Cert-on kills FO that cert-off does not; matched Δ=0; energy/ζ ≢ cert.
2. Receipt JSON is auditability only (layout hash + T + seeds + cert status) — not new science.
3. Do **not** widen §22. Do **not** claim hop-OOD OPEN. Still **MEASURE**.
