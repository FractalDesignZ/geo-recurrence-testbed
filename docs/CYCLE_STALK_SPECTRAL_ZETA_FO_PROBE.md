# CYCLE_STALK_SPECTRAL_ZETA_FO_PROBE — MEASURE graph heat / finite spectral-ζ FO probe (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | MEASURE — **eval-only** finite spectral zeta / heat of graph Laplacian on hard Â as FO refuse instrument on frozen **#14/#18/#22** ens |
| **science_open** | **false** (always in harness; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | After #35 CERT_FO_CATCH + #40 ENERGY_NULL: port attached conformal spectral-zeta FEM note (Dirichlet conformal-invariant; ζ sensitive to ρ) → **discrete** graph heat/ζ on Â (not continuum FEM); instruments remain refuse/rank only (#38 HARD_UNANIMOUS wall) |
| **Base** | `main` tip after PR #40 (`34d059b`) |
| **Prior** | #22 ens `prob_mean`; #30 FO=45 HN shatter; #35 CERT_FO_CATCH; #38 energy-selector COLLATERAL_HARM / HARD_UNANIMOUS; #40 ENERGY_NULL (Id coboundary ≢ cert) |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Sheaf unsupervised** | **NOT opened** — no restriction-map learning; no sheaf train; STOP sheaf arms stay closed |
| **Train / PEFT / select / bag / JS** | **NONE** — eval-only on frozen ckpts; ens aggregation **unchanged** (`prob_mean`) |

## Goal

Hypothesis: **finite spectral zeta** of the (normalized or combinatorial) Laplacian on hard attention graph Â concentrates on hop-OOD **FAIL_OPEN** the way dirty certificates do (#35) — a structural spectral refuse instrument, not a trained sheaf and not continuum FEM.

Primary scientific question: does ζ_Â(s=2) (ρ=Id primary) separate FO / rem-22 from OK / FC (AUROC), and can a **zeta-threshold refuse** match #35 cert refuse without matched-OOD collateral harm?

Port note (user attachment): continuum FEM checks show Dirichlet energy conformal-invariant and ζ sensitive to ρ. **This cycle ports the discrete analog only** — graph heat Tr(e^{-tL}) / finite ζ on Â used by certs. No mesh, no FEM, no learned ρ.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #14–#20 | select/curriculum CLOSED; park MEASURE_STILL |
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| Sheaf learned OPEN | **INVALID** / STOP — **do not reopen** |
| #30–#34 | FO core + LOCAL_SOUND_WALL; overlay chase parked |
| #35 | CERT_FO_CATCH — **reference** dirty concentration |
| #36–#38 | tropical / orientation / energy-selector — not FO repair |
| #39 | PASS_REGRESSION CI lock |
| #40 | ENERGY_NULL — Id coboundary ≢ cert |

This cycle = **MEASURE spectral-ζ probe** (eval-only). Not train. Not §22 widen. Not OPEN. Not sheaf unsupervised. Not remix ens aggregation. Not continuum FEM.

## Metaphor

- **map ≠ location** — #22 COMPETENT map predicts on Â; spectral ζ of Â is a *post-hoc location geometry check*, not a learned sheaf.
- **negatives = mirror** — anomalous ζ of hard Â on hop-OOD FO is the structural shadow that dirty certificates catch (#35).

## Cite #30 / #32 / #35 / #40 (ood_hops T16 — #22 ens)

| Metric | Value |
|--------|-------|
| ens HN (baseline) | **0.067** |
| ens K16 | **0.988** |
| ens overall | **0.510** |
| FAIL_OPEN baseline | **45** |
| rem-22 (post-outdeg0) | **22** |
| #35 cert FO / rem22 | **45/45** / **22/22**; matched Δ=0 |
| #40 E_cob AUROC FO vs OK∪FC | **0.317**; refuse **0/45** |

## Exact L / ζ / heat (LOCKED — prereg BEFORE run)

Helpers live in `reachability_gen.spectral_zeta`.

### Graph Â (same substrate as #35 certs)

Directed edge list from the instance encoding (hard A_ij). **No** attention soft weights. **No** model states in the primary score (pure Â spectrum).

### Symmetrization (LOCKED)

```
A_sym[i,j] = 1  if (i→j)∈Â or (j→i)∈Â, else 0   (i≠j; no self-loops)
D_ii       = Σ_j A_sym[i,j]
L_comb     = D − A_sym                                 # combinatorial (primary)
L_norm     = I − D^{+}½ A_sym D^{+}½   (D^{+}½=0 on isolates)  # normalized (secondary)
```

Directed random-walk Laplacian is **not** primary (certs use undirected path existence via directed BFS; spectral refuse uses undirected skeleton for SPD L).

### Zero-mode handling (LOCKED)

```
λ_i  = eigenvalues of L (ascending; torch.linalg.eigh)
EPS_ZERO = 1e-8
ζ(s) = Σ_{λ_i > EPS_ZERO} λ_i^{−s}     # omit zero modes
n_zero = |{λ_i ≤ EPS_ZERO}|
```

Prefer **s=2** (matches attached FEM note).

### Optional ρ (LOCKED — MEASURE only; no learned ρ)

| Variant | Spec |
|---------|------|
| **ρ=Id (PRIMARY)** | standard eigenproblem `L_comb v = λ v` |
| **ρ=outdeg (secondary)** | `ρ_i = max(outdeg_Â(i), 1)`; generalized `L_comb v = λ diag(ρ) v` ≡ eig of `ρ^{−½} L ρ^{−½}` |
| **edge-mass (secondary)** | undirected edge weight `w_{ij}=1` same as Id for unweighted Â — report identical to primary when unweighted; if cheap outdeg-mass: `w_{ij}=(outdeg(i)+outdeg(j))/2` on symmetrized support → weighted combinatorial L |

**Do not invent learned ρ.** Sheaf unsupervised stays closed.

### Heat (optional secondary)

```
H(t) = Tr(e^{−t L}) = Σ_i exp(−t λ_i)    # includes zero modes
```

Report at fixed **t ∈ {0.5, 1.0}** on primary L_comb (ρ=Id).

### Primary scientific score (LOCKED)

```
score = ζ_comb(s=2; ρ=Id) on A_sym
```

Secondary reported: ζ_norm(2), ζ_comb_ρ_outdeg(2), H(0.5), H(1.0), n_zero, n_nodes, n_edges_sym.

### Ensemble aggregation (LOCKED — unchanged)

```
pred = #22 ens prob_mean     # DO NOT change ens aggregation
```

Zeta is a **post-hoc refuse/rank instrument** on top of frozen ens preds — not a remix.

### Optional zeta-threshold refuse (LOCKED)

Calibrate threshold on **matched-OOD** only (no FO peek):

```
τ = quantile_{0.90}( ζ_comb(matched-OOD) )
```

Policy: if ens `prob_mean` pred=YES **and** `ζ_comb > τ` → force pred=0 (fail-closed refuse). Reported as `zeta_refuse` arm vs #35 `cert` reference.

## Arms (LOCKED)

| # | Arm |
|---|-----|
| 0 | baseline `#22` `prob_mean` |
| 1 | **zeta_refuse** under `ζ_comb > τ` (primary scientific refuse) |
| — | #35 cert reference column (cite artifact; no re-claim) |

## Strata / metrics (LOCKED)

| Item | Spec |
|------|------|
| Strata | OK_HN / FO_HN (45) / rem-22 / FC_HN from #31/#32/#33 artifacts (same hop-OOD as #35/#40) |
| Stratify | mean/median `ζ_comb`, `ζ_norm`, `ζ_ρ`, H(0.5), H(1.0) by stratum |
| Primary AUROC | `ζ_comb` → FO_HN vs OK∪FC; also vs OK alone; rem-22 vs OK∪FC |
| Cert dirty compare | FO ∩ refuse vs FO ∩ dirty (#35); Spearman/Pearson score vs FO-proxy |
| Matched-OOD | Δ overall / HN / K16 under zeta_refuse vs baseline |
| ≢ vs #35 | if refuse∩FO ≪ dirty∩FO → instrument ≢ cert |

### FAIL_OPEN / FAIL_CLOSED defs (LOCKED — same as #29–#40)

| Symbol | Value |
|--------|-------|
| `CONF_THRESH` | **0.80** |
| `D_CLOSED_MIN` | **0.10** |
| `EPI_CLOSED_MIN` | **0.15** nats |

## Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`ZETA_FO_CATCH`** | zeta_refuse FO_killed ≥ **40**/45 **AND** rem22_killed ≥ **16**/22 **AND** matched abs Δ ≤ **0.05** on overall/HN/K16 **AND** AUROC(ζ→FO vs OK∪FC) ≥ **0.75** |
| **`ZETA_PARTIAL`** | AUROC ≥ **0.60** **OR** FO_killed ≥ **8** **OR** rem22 ≥ **4**, without full FO_CATCH; matched abs Δ ≤ **0.05** |
| **`ZETA_NULL`** | AUROC ∈ [0.40, 0.60] **or** AUROC < 0.60 with FO_killed ≤ **2** **and** rem22 ≤ **2** — falsifies FO-concentration under this proxy |
| **`COLLATERAL_HARM`** | matched Δ drop ≥ **0.05** on overall / HN / K16 under zeta_refuse (priority over PARTIAL) |
| **`INVALID`** | cite floors break (FO≠45 / rem≠22 / baseline HN/K16 mismatch) **OR** science_open widened **OR** ens agg changed **OR** sheaf train attempted |

Still **MEASURE**, never OPEN. Passing ZETA_FO_CATCH does **not** widen `science_open` / does **not** claim hop-OOD OPEN / does **not** open sheaf train.

## Preregistered protocol (LOCKED before runs)

```bash
python -m reachability_gen.run_stalk_spectral_zeta_fo_probe
# or: reachability-stalk-spectral-zeta-fo-probe
pytest -q tests/test_stalk_spectral_zeta_fo_probe.py
```

| Item | Spec (locked) |
|------|----------------|
| Checkpoints | `#14` `artifacts/fractal_core_stalk_stabilize_seed{0..4}_best.pt` + `#18` `artifacts/fractal_core_stalk_seed_stability_seed{5..9}_best.pt` |
| Train | **NONE** |
| Eval substrate | **`data/ood_hops.jsonl`** @ **T=16** fixed; matched-OOD for τ + collateral |
| Prefer | reuse #22 ckpts + #30/#31/#32/#33 FO / rem-22 / OK / FC ids; local CPU |
| Artifact | `artifacts/stalk_spectral_zeta_fo_probe.json` |

### Prohibited defaults (KILL)

- No train / PEFT / select / bag / JS / sheaf unsupervised / restriction-map learning
- No `science_open` widen; no §22 widen
- No changing ens aggregation (keep `prob_mean`)
- No reopen STOP sheaf arms
- No claiming hop-OOD OPEN
- No continuum FEM / mesh; no inventing learned ρ
- No baking BFS / ζ into weights

## Explicit non-goals

- No sheaf training / restriction-map learning
- No energy / tropical / orientation reopen
- No `science_open=true`
- No FEM continuum claim

## Results (after run)

_Pending run._
