# CYCLE_STALK_ENERGY_SELECTOR — MEASURE eval-only stalk energy selector (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** energy-based member selection on frozen **#14/#18/#22** ens |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | #37 `INCONCLUSIVE_ARCH` (orientation proxy unusable); Phase 4 = independent import — separate **generation** (#22 members) from **selection** via explicit energy over Â-local / certificate-related terms |
| **Base** | `main` tip after PR #37 (`39e3aa7`) |
| **Prior** | #22 ens `prob_mean`; #26 multi-hyp JS **STOP**; #27 bag MEASURE_LIFT/CHAOS; #35 CERT_FO_CATCH; #36 COLLATERAL_HARM; #37 INCONCLUSIVE_ARCH |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **IGNORE this cycle** |
| **Train / PEFT / tropical anneal / DEAR / multi-hyp JS / bag** | **NONE** — eval-only on frozen ckpts |
| **Orientation metrics** | **NOT used** (#37 inconclusive; do not depend) |

## Goal

Separate **generation** (frozen #22 ens members) from **selection**. Score each member prediction with an **explicit energy** `E(Â, outputs)` — **not** a trained head, **not** multi-hyp JS (#26 STOP), **not** bag chaos (#27).

Primary scientific question: does **cert-free** energy (degree / cone / local-incidence factors only) beat `prob_mean` **toward the oracle member upper bound** on hop-OOD / rem-22?

Secondary: cert-tied energy (checker disagreement) — expected to track #35; if energy ≈ certificate, label that arm **cert-energy** and keep cert-free as the primary scientific arm. Certificates remain reference; energy must add value **before** refuse or close selector–oracle gap.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| #26 | multi-hyp JS **STOP** — do not train disagreement heads |
| #27 | bag MEASURE_LIFT / CHAOS — do not train more bag noise |
| #30–#33 | FO core + LOCAL_SOUND_WALL; overlay chase parked #34 |
| #35 | CERT_FO_CATCH (post-hoc checker) — reference |
| #36 | COLLATERAL_HARM (tropical ens aggregation) |
| #37 | INCONCLUSIVE_ARCH — **no orientation metrics in this cycle** |

This cycle = **MEASURE energy selector** (eval-only). Not train. Not §22 widen. Not OPEN. Not tropical anneal. Not orientation reopen.

## Metaphor

- **map ≠ location** — #22 COMPETENT map generates member hypotheses; energy scores *where* each lands on Â-local structure.
- **negatives = mirror** — high energy on YES-with-empty-local-cone is the structural shadow of false-reach.

## Cite #30 / #32 / #35 / #37 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN (baseline) | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN baseline | **45** |
| rem-22 (post-outdeg0) | **22** |
| #35 cert HN / FO | **1.000** / **0** (reference) |
| #37 orientation | INCONCLUSIVE_ARCH — **not used** |

## Exact energy formula (LOCKED — prereg BEFORE run)

Helpers live in `reachability_gen.energy_selector`.

### Local Â features (cert-free; **no BFS**)

For row with digraph Â, source `s`, target `t`:

```
out_s = |{v : (s,v) ∈ Â}|
in_t  = |{u : (u,t) ∈ Â}|
```

### Per-member outputs

Member `m` logits `L_m ∈ ℝ²` (class 0=NO, 1=YES):

```
pred_m = argmax_c L_m[c]
p_m    = max_c softmax(L_m)_c          # confidence
```

### Cert-free energy (PRIMARY scientific)

```
E_sound(m) = 1[out_s==0 ∧ s≠t ∧ pred_m==1]
           + 1[in_t==0  ∧ s≠t ∧ pred_m==1]

E_cone(m)  = 1[pred_m==1] / (1 + out_s + in_t)

E_conf(m)  = −log(p_m + ε)             # ε = 1e-8

E_free(m)  = α·E_sound(m) + β·E_cone(m) + γ·E_conf(m)
```

**Locked weights:** `α = 10.0`, `β = 1.0`, `γ = 0.1`, `ε = 1e-8`.

Interpretation: local sound mismatch dominates; sparse-cone YES is penalized; mild confidence preference among otherwise-equal members. **No checker BFS. No orientation metrics.**

### Cert-tied energy (SECONDARY — labeled cert-energy)

Checker BFS `reach = reachable_bfs(Â, s, t)` **post-hoc only** (same policy as #35; never model input / train):

```
E_disagree(m) = 1[(pred_m==1 ∧ ¬reach) ∨ (pred_m==0 ∧ reach)]

E_cert(m)     = α·E_disagree(m) + γ·E_conf(m)
```

Same `α=10`, `γ=0.1`. If this arm ≈ #35 certificate policy, report as **cert-energy** (expected); it is **not** the primary scientific claim.

### Selection rules (LOCKED)

| Rule | Definition |
|------|------------|
| **energy_argmin** | `m* = argmin_m E(m)`; pred = `pred_{m*}` (ties → lowest member index) |
| **energy_weighted** (report) | `w_m ∝ exp(−E(m)/τ)`, `τ=1.0`; pred = `argmax_c Σ_m w_m softmax(L_m)_c` |
| **oracle_member** | if any member correct on gold `y`, pick lowest-index correct member; else member 0 |

Primary selection for verdicts = **energy_argmin** under **E_free**.

### Optional refuse (document only; cert-aware)

If `min_m E_cert(m) ≥ α` (all members disagree with checker) → force pred=0 (fail-closed refuse). Reported as `cert_energy_refuse` overlay; certificates remain reference — energy must still show value on **cert-free** before claiming selector progress.

## Arms (LOCKED)

| # | Arm |
|---|-----|
| 0 | baseline `#22` `prob_mean` |
| 1 | **energy_argmin** under **E_free** (primary scientific) |
| 1b | energy_weighted under E_free (report) |
| 2 | **oracle_member** (upper bound) |
| 3 | energy_argmin under **E_cert** (cert-energy; secondary) |
| 3b | cert_energy_refuse overlay (document) |
| — | #35 cert reference column (cite artifact; no re-claim) |

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_energy_selector
# or: reachability-stalk-energy-selector
```

| Item | Spec (locked) |
|------|----------------|
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** @ **T=16** fixed; matched-OOD for collateral |
| Prefer | reuse #22 ckpts + #30/#31/#32 FO / rem-22 ids |
| Artifact | `artifacts/stalk_energy_selector.json` |

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29–#37)

| Symbol | Value |
|--------|-------|
| `CONF_THRESH` | **0.80** |
| `D_CLOSED_MIN` | **0.10** |
| `EPI_CLOSED_MIN` | **0.15** nats |

### Metrics (prereg — locked)

| Metric | Spec |
|--------|------|
| HN / K16 / overall | ood_hops T16 per arm |
| FO count | FAIL_OPEN under arm; FO killed of baseline 45; rem-22 killed of 22 |
| Matched-OOD Δ | overall / HN / K16 vs baseline (collateral) |
| Selector–oracle gap | `gap_arm = oracle_overall − arm_overall` (also HN); **beats mean** if `gap_free < gap_mean` by ≥ `GAP_CLOSE_MIN` abs **or** FO/rem catches below |
| Energy vs confidence | Pearson `r(E_free, −log p_m)` over (member, example); **falsify** energy≈confidence if `|r| ≥ CONF_CORR_MAX` **and** sound/cone terms never differentiate members |

### Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`ENERGY_BEATS_MEAN`** | Cert-free energy_argmin closes selector–oracle overall gap by ≥ **0.02** abs **OR** kills ≥ **8**/45 FO **OR** ≥ **4**/22 rem-22 vs baseline; **AND** no COLLATERAL_HARM; **AND** energy not falsified as ≈confidence alone |
| **`ENERGY_PARTIAL`** | Some FO kill (≥1) or gap close ≥0.005 or HN Δ≥0.01 vs baseline under E_free, but not BEATS; no COLLATERAL_HARM |
| **`ENERGY_NULL`** | Cert-free: FO killed ≤2, rem22 killed =0, \|ΔHN\|≤0.05, gap close &lt;0.005 — does not beat mean toward oracle |
| **`ENERGY_IS_CERT`** | Cert-free null/partial **and** cert-tied energy_argmin (or refuse) tracks #35 (≥40/45 FO + ≥16/22 rem22) — only working energy is certificate-tied |
| **`COLLATERAL_HARM`** | matched-OOD overall/HN/K16 drop ≥ **0.05** abs vs baseline under primary E_free arm (priority) |

Still **MEASURE**, never OPEN. Even `ENERGY_BEATS_MEAN` does **not** widen `science_open` / does **not** claim hop-OOD OPEN.

### Prohibited defaults

- No train / PEFT / β-anneal / DEAR / tropical anneal
- No multi-hyp JS train / bag train
- No orientation metrics (#37 inconclusive)
- No §22 widen; no claiming hop-OOD OPEN
- No `science_open=true`
- No T beyond 16
- No regenerating `ood_hops.jsonl`
- No baking checker into weights / model input

## Explicit non-goals

- No training / bag / JS / SWA / distill re-chase
- No tropical anneal / in-attn rewrite
- No orientation reopen
- No select / curriculum reopen
- No T>16
- No sheaf unsupervised revival
- No `science_open=true` / §22 widen
- No claim that hop-OOD metrics extend §22 OPEN

## Results (this run — cite artifact)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_energy_selector.json` |
| **Log** | `artifacts/stalk_energy_selector_run.log` |
| **Prereg SHA** | `eecc4b7` |
| **Harness SHA** | `79f7196` |
| **Results SHA** | `bbae5a757674b2c4a593a94d1b32ea410fbe92a7` |
| **Cycle verdict** | **`COLLATERAL_HARM`** |
| **science_open** | **false** (not widened; §22 unchanged) |
| **Exact E_free** | `10·E_sound + 1·E_cone + 0.1·(−log(p+1e-8))` |
| **Exact E_cert** | `10·E_disagree + 0.1·(−log(p+1e-8))` |
| **Cite #30 replicate** | **exact** (FO=45 / HN=0.067 / K16=0.988 / ov=0.510) |
| **Elapsed** | **16.7 s** CDT |
| **Energy↔conf** | Pearson r(**−0.049**); sound/cone differs=**true** — not just-confidence |

### Table (baseline vs arms) — ood_hops T16

| Arm | HN | K16 | overall | FO | FO killed /45 | rem22 killed /22 |
|-----|----|-----|---------|----|---------------|------------------|
| (0) baseline #22 `prob_mean` | **0.067** | **0.988** | **0.510** | **45** | — | — |
| (1) **E_free** energy_argmin (primary) | **0.812** | **0.438** | **0.592** | **45** | **0**/45 | **0**/22 |
| (1b) E_free energy_weighted | 0.358 | 0.950 | 0.646 | 45 | 0/45 | 0/22 |
| (2) oracle_member (upper bound) | **0.812** | **1.000** | **0.906** | 45 | **0**/45 | **0**/22 |
| (3) E_cert energy_argmin (cert-energy) | 0.812 | 1.000 | 0.906 | 45 | 0/45 | 0/22 |
| (3b) cert_energy_refuse | **1.000** | **1.000** | **1.000** | **0** | **45**/45 | **22**/22 |
| #35 cert reference | 1.000 | 0.988 | 0.977 | 0 | 45/45 | 22/22 |

Selector–oracle gap: `gap_mean=0.396` → `gap_free=0.315` (**gap_close=+0.081**). Oracle itself kills **0**/45 FO (HARD_UNANIMOUS generation wall).

### Matched-OOD collateral (T16) — primary E_free

| Slice | baseline | E_free argmin | Δ |
|-------|----------|---------------|---|
| overall | 0.996 | 0.669 | **−0.327** |
| HN | 1.000 | 1.000 | 0.000 |
| K16 | 1.000 | 0.062 | **−0.938** |

**COLLATERAL_HARM** (K16/overall drops ≫ 0.05). E_cert argmin matched Δ≈0 (no harm) but FO catch 0/45 without refuse.

### Reading (fail-closed)

1. **COLLATERAL_HARM:** Cert-free `E_free` energy_argmin destroys matched-OOD K16 (**1.000→0.062**) and overall (**0.996→0.669**). Priority verdict — same hygiene shape as #36 tropical ens harm.
2. **FO / rem-22 null under member selection:** E_free and E_cert argmin kill **0**/45 FO and **0**/22 rem-22. Oracle also **0**/45 — FO core is HARD_UNANIMOUS (#31); **no member is correct**, so selection cannot repair generation failure.
3. **HN / gap illusion:** E_free HN **0.067→0.812** (matches oracle HN) and gap_close **+0.081** come from non-FO HN routing; hop-OOD K16 collapses (**0.988→0.438**). Not a FO repair.
4. **Cert-energy vs certificate:** E_cert argmin alone ≢ #35 (FO 0/45). `cert_energy_refuse` (force NO when all members disagree with checker) tracks #35-style FO catch (45/45 + 22/22) — that is refuse/certificate policy, not cert-free energy value.
5. Energy ≠ confidence (r≈−0.05; sound/cone differentiate). Do **not** widen §22. Do **not** claim hop-OOD OPEN. Prefer #14+#22 on matched-OOD; keep #35 certificates for FO catch. Still **MEASURE**.
