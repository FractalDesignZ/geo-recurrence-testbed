# CYCLE_STALK_SHEAF_ENERGY_FO_PROBE — MEASURE Dirichlet/coboundary energy FO probe (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** sheaf / coboundary Dirichlet energy probe on frozen **#14/#18/#22** ens |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | After #35 CERT_FO_CATCH + #38 energy-selector COLLATERAL_HARM + #39 FO/cert regression: instrument Kant/sheaf **A2/A4** as **identity-restriction** coboundary energy on sealed stalk wall — **without** sheaf training |
| **Base** | `main` tip after PR #39 (`73a458a`) |
| **Prior** | #22 ens `prob_mean`; #30 FO=45 HN shatter; #35 CERT_FO_CATCH (dirty concentrates on FO); #36 tropical COLLATERAL_HARM; #37 orientation INCONCLUSIVE; #38 energy selector COLLATERAL_HARM; #39 PASS_REGRESSION |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **NOT opened** — no restriction-map learning; no sheaf train; STOP sheaf arms stay closed |
| **Train / PEFT / select / bag / JS** | **NONE** — eval-only on frozen ckpts |

## Goal

Hypothesis: **high sheaf / coboundary Dirichlet energy** of local sections (final stalk node states) on hard Â concentrates on hop-OOD **FAIL_OPEN** the way dirty certificates do (#35) — axiom **A2** (consistency / gluing) and **A4** (Dirichlet regularity) instrumented as a **MEASURE probe**, not a trained sheaf.

Primary scientific question: does identity-restriction coboundary energy `E_cob` separate FO / rem-22 from OK / FC (AUROC), and can an **energy-threshold refuse** match #35 cert refuse without matched-OOD collateral harm?

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| Sheaf learned OPEN | **INVALID** / STOP (bake-in / no-aux / STE) — **do not reopen** |
| #30–#34 | FO core + LOCAL_SOUND_WALL; overlay chase parked |
| #35 | CERT_FO_CATCH — **reference** dirty concentration |
| #36–#38 | tropical / orientation / energy-selector — not FO repair |
| #39 | PASS_REGRESSION CI lock |

This cycle = **MEASURE energy probe** (eval-only). Not train. Not §22 widen. Not OPEN. Not sheaf unsupervised. Not restriction-map learning. Not reopen STOP sheaf arms.

## Metaphor

- **map ≠ location** — #22 COMPETENT map generates sections on Â; coboundary energy is a *post-hoc location consistency check* (A2), not a learned sheaf.
- **negatives = mirror** — high Dirichlet energy of YES sections on hop-OOD FO is the structural shadow that dirty certificates catch (#35).

## Cite #30 / #32 / #35 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN (baseline) | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN baseline | **45** |
| rem-22 (post-outdeg0) | **22** |
| #35 cert FO / rem22 | **45/45** / **22/22**; matched Δ=0 |
| #35 dirty∩FO | **45**; dirty_n=235 |

## Exact energy formula (LOCKED — prereg BEFORE run)

Helpers live in `reachability_gen.sheaf_energy`.

### Local sections (sealed stalk wall)

FractalCore `final_states` `H ∈ ℝ^{M×d}` (`return_states=True`); use `H[:n]` for graph of size `n`. **No** learned restriction maps. **No** sheaf Laplacian training.

### Identity restriction (honest proxy — full sheaf Laplacian unavailable)

Kant/sheaf **A2** (consistency): adjacent stalks agree under restriction. With **ρ = Id** (sealed; no F-learning):

```
E_cob(H; Â) = Σ_{(u→v) ∈ Â} ||H[u] − H[v]||₂²
Ẽ_cob       = E_cob / max(|E|, 1)     # per-edge mean
```

This is the sheaf-Laplacian quadratic form `Hᵀ L_Id H` under identity restrictions — the simplest honest coboundary energy when restriction-map learning is forbidden.

### Scalar Dirichlet (A4 regularity proxies)

Mass field and target-alignment field (reach-logit-derived honest scalars from sections; **not** a trained head):

```
f(i) = ||H[i]||₂
E_dir(f; Â) = Σ_{(u→v) ∈ Â} (f(u) − f(v))²
Ẽ_dir       = E_dir / max(|E|, 1)

a(i) = cos(H[i], H[t])     # 0 if either near-zero (eps=1e-8)
E_align(a; Â) = Σ_{(u→v) ∈ Â} (a(u) − a(v))²
Ẽ_align       = E_align / max(|E|, 1)
```

### Ensemble aggregation (LOCKED)

```
E_* = mean_{m ∈ #22} Ẽ_*(H_m)     # primary report = E_cob
```

**Primary scientific score** = `E_cob`. `E_dir` / `E_align` reported as secondary scalars.

### Optional energy-threshold refuse (LOCKED)

Calibrate threshold on **matched-OOD** only (no FO peek):

```
τ = quantile_{0.90}( E_cob(matched-OOD) )
```

Policy: if ens `prob_mean` pred=YES **and** `E_cob > τ` → force pred=0 (fail-closed refuse). Reported as `energy_refuse` arm vs #35 `cert` reference.

## Arms (LOCKED)

| # | Arm |
|---|-----|
| 0 | baseline `#22` `prob_mean` |
| 1 | **energy_refuse** under `E_cob > τ` (optional; primary scientific refuse) |
| — | #35 cert reference column (cite artifact; no re-claim) |

## Strata / metrics (LOCKED)

| Item | Spec |
|------|------|
| Strata | OK_HN / FO_HN (45) / rem-22 / FC_HN from #31/#32/#33 artifacts |
| Stratify | mean/median `E_cob`, `E_dir`, `E_align` by stratum |
| AUROC | `E_cob` → FO_HN vs OK∪FC (and vs OK alone; rem-22 vs OK∪FC) |
| Cert dirty compare | fraction of FO with `E_cob > τ` vs #35 dirty∩FO; Spearman/Pearson if both scored |
| Matched-OOD | Δ overall / HN / K16 under energy_refuse vs baseline |

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29–#39)

| Symbol | Value |
|--------|-------|
| `CONF_THRESH` | **0.80** |
| `D_CLOSED_MIN` | **0.10** |
| `EPI_CLOSED_MIN` | **0.15** nats |

## Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`ENERGY_TRACKS_CERT`** | energy_refuse FO_killed ≥ **40**/45 **AND** rem22_killed ≥ **16**/22 **AND** matched abs Δ ≤ **0.05** on overall/HN/K16 **AND** AUROC(`E_cob`→FO vs OK∪FC) ≥ **0.75** |
| **`ENERGY_PARTIAL`** | AUROC ≥ **0.60** **OR** FO_killed ≥ **8** **OR** rem22 ≥ **4**, without full TRACKS_CERT; matched abs Δ ≤ **0.05** |
| **`ENERGY_NULL`** | AUROC ∈ [0.40, 0.60] **and** FO_killed ≤ **2** **and** rem22 ≤ **2** — falsifies FO-concentration hypothesis under this proxy |
| **`COLLATERAL_HARM`** | matched Δ drop ≥ **0.05** on overall / HN / K16 under energy_refuse (priority over PARTIAL) |

Still **MEASURE**, never OPEN. Passing TRACKS_CERT does **not** widen `science_open` / does **not** claim hop-OOD OPEN / does **not** open sheaf train.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_sheaf_energy_fo_probe
# or: reachability-stalk-sheaf-energy-fo-probe
pytest -q tests/test_stalk_sheaf_energy_fo_probe.py
```

| Item | Spec (locked) |
|------|----------------|
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** @ **T=16** fixed; matched-OOD for τ + collateral |
| Prefer | reuse #22 ckpts + #30/#31/#32/#33 FO / rem-22 / OK / FC ids |
| Artifact | `artifacts/stalk_sheaf_energy_fo_probe.json` |

### Prohibited defaults

- No train / PEFT / select / bag / JS / sheaf unsupervised / restriction-map learning
- No `science_open` widen; no §22 widen
- No reopen STOP sheaf arms
- No claiming hop-OOD OPEN
- No baking BFS / energy into weights

## Explicit non-goals

- No sheaf training / restriction-map learning
- No energy member-selection chase reopen (#38 COLLATERAL_HARM)
- No tropical / orientation reopen
- No `science_open=true`

## Results (fill after run)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_sheaf_energy_fo_probe.json` |
| **Cycle verdict** | *(pending)* |
| **science_open** | **false** |
| **E formula** | `E_cob = mean_m Σ_{(u→v)} ‖H_m[u]−H_m[v]‖₂² / \|E\|` (Id restriction) |
| **AUROC E_cob→FO** | *(pending)* |
| **FO concentration** | *(pending)* |
| **energy_refuse FO/rem22** | *(pending)* |
| **Matched-OOD Δ** | *(pending)* |
| **vs #35 cert** | *(pending)* |
| **Base SHA** | `73a458a` |
| **Train / sheaf** | **none** |
