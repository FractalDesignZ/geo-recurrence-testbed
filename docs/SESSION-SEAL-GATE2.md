# SESSION SEAL — Gate 2 / bound30 (FREEZE)

| Field | Value |
|-------|-------|
| **Date** | 2026-09-27 |
| **Mode** | FREEZE |
| **science_open** | **false** |
| **Repo** | https://github.com/ZuluYokohama/geo-recurrence-testbed |
| **SHA bound30** | `3c809a0` (Bound30 rematch: RMSNorm state bound + recurrent clip/LR match) |
| **SHA Gate2** | `984cca81ff2d8454418f5ca1c9056211acf8f8b0` (Gate 2 OOD-hop stress on bound30 best ckpts) |
| **Verdict class** | STOP (computed from artifacts; not narrated) |

Fail-closed. Append-only. This seal does **not** reopen science.

---

## 1. Scope sealed

| Slice | Artifact | Status |
|-------|----------|--------|
| ID 2k bound30 rematch (FF / Geo / Loop, 30 epochs, RMSNorm) | `artifacts/id_2k_rematch_bound30.json` | SEALED |
| Gate 1 best-epoch hop breakdown | `artifacts/id_2k_rematch_bound30_gate1_best_hop.json` | SEALED |
| Gate 2 OOD-hop stress (fixed + dynamic T) | `artifacts/id_2k_rematch_bound30_gate2_ood.json` | SEALED |
| OOD generation report | `artifacts/ood_hops_generation_report.json` | SEALED |
| ID generation report | `artifacts/id_2k_generation_report.json` | cited for ID n |

No model-code change. No retrain. MEASURE plumbing only.

---

## 2. Post-mortem (evidence)

### Gate 2 fixed-depth OOD (`data/ood_hops.jsonl`, n=480, 50/50)

Hard-neg hop=`-1` accuracy collapses to **0.000** for FF, Geo, and Loop. Overall accuracy sits near chance (~0.43–0.50). Positives remain high, **including FF**.

| Arm | Overall | Hard-neg (−1) | K=8 | K=12 | K=16 | Source |
|-----|---------|---------------|-----|------|------|--------|
| FF L=2 | 0.427 | **0.000** | 0.725 | 0.9625 | 0.875 | `gate2_ood.fixed_depth.ff` |
| Geo T=6 | 0.4875 | **0.000** | 1.000 | 1.000 | 0.925 | `gate2_ood.fixed_depth.geo` |
| Loop T=6 | 0.496 | **0.000** | 0.9875 | 0.9875 | 1.000 | `gate2_ood.fixed_depth.loop` |

Prereg one-liner (artifact): *STOP: all arms near chance on OOD overall (hard-neg acc≈0) — recurrence length generalization / discrimination on this substrate not supported; positives alone do not license OPEN.*

### Dynamic unroll (Geo / Loop only, T∈{8,12,16})

- **Geo:** flat in T — overall **0.4875** at T=8/12/16; hard-neg remains **0.000**; OOD positives unchanged vs T=6 (K8/12=1.0, K16=0.925).
- **Loop:** positives degrade with larger T — overall T=8 → 0.4625, T=12 → 0.275, T=16 → 0.19375; hard-neg ≈0 (0.000 / 0.000 / 0.025).

### Covariate-shift risk (explicit)

| Axis | ID (`id_2k`) | OOD (`ood_hops`) |
|------|--------------|------------------|
| n | support short `{8,12,16}` + long `{16,20,24,28,32}`; used `{8,16,20,24,28,32}` (`id_2k_generation_report.json`) | stated range **[32, 64]** (`gate2_ood.dataset.n_range`); hop supports include 24–64 (`ood_hops_generation_report.json`) |
| p | code supports short `{0.15,0.25,0.35}` + long `{0.08,0.10,0.12,0.15,0.20}` (`gen_id_2k.py`); data uniques `{0.08…0.35}` (**p not listed in ID gen report**) | stated range **[0.015, 0.06]** (`gate2_ood.dataset.p_range`); hop supports include up to 0.10 on K=8 |
| seq_len (encoding tokens) | mean **57.36**, max **109** (computed from `data/id_2k.jsonl`) | mean **203.225**, max **246**, min 117 (`ood_hops_generation_report.token_len`) |

Gate 2 therefore confounds hop-OOD with **n / p / seq_len** shift. Length-gen failure is **not** isolated. Label: **evidence of collapse under this joint shift; aspiration** to attribute collapse to hop length alone is **not** licensed.

### ID late collapse (separate from OOD all-positive bias)

Gate 1 Geo best@epoch 11 → epoch 30: hard-neg **0.91 → 0.21** while positives go to 1.0 (`gate1_best_hop.arms.geo`). This ID late all-positive bias is **distinct** from OOD hard-neg=0 / overall~chance.

---

## 3. Closed hypotheses (STOP)

Status computed from cited numbers. **science_open=false.**

| # | Claim (aspirational prior) | Status | Tie |
|---|----------------------------|--------|-----|
| H1 | Geometric / looped recurrence **length-generalizes** and discriminates on OOD hops K∈{8,12,16} beyond ID train hops {2…6} | **Falsified** | Gate 2: hard-neg=0.000 all arms; overall FF 0.427 / Geo 0.4875 / Loop 0.496 (`gate2_ood.fixed_depth` + `interpretation_stop`) |
| H2 | Geo+τ at peak ID checkpoint is a hop-capable / τ-attractor regime that should transfer under extra unroll | **Falsified** | Gate 1 Geo@11 hop table: overall 0.8475 but K=4 **0.575**, K=5 0.65, K=6 0.75 (weak mid-hops); Gate 2: Geo flat in T with neg still 0; Loop positives degrade with T (`gate1_best_hop`, `gate2_ood.dynamic_unroll`) |
| H3 | FF L=2 hits a **receptive-field wall** and cannot solve ID multi-hop reachability for K≤6 | **Disproven** | Gate 1 FF best@15: overall **0.9525**; by hop K=2…6: 1.00 / 1.00 / 0.95 / 1.00 / 0.975 (`gate1_best_hop.arms.ff.at_best_epoch`) |

---

## 4. Validated engineering invariants (OPEN *candidates* only)

Engineering, **not** scientific OPEN. `science_open=false` on each.

| Invariant | Evidence | Label |
|-----------|----------|-------|
| ±5% param parity | FF 121218 / Geo 121858 (ratio 1.0053) / Loop 120834 (ratio 0.9968); `within_5pct=true` | evidenced (`bound30.param_match`) |
| F_block compute accounting | ADR-001 + `flops.py` schematic; CoT debits K_used | evidenced (scaffold) |
| Deterministic generation | ID seed 42000; OOD seed 84000; `verify_ok=true` both reports | evidenced |
| RMSNorm residual bound | `z←RMSNorm(z+α·(Φ−z))`, α=0.5; Geo `mean_z_norm` ≤≈8.31 ≈√d; eliminates fixed30 `‖z‖~820` runaway | evidenced (`bound30.geo.drift_summary.diameter_check`; prior fixed30 seal note) |

---

## 5. Architectural preconditions for future work (**aspiration**)

Labeled **aspiration**, not evidence:

- Prefer **spectral / Laplacian / energy-bounded / merit-halting** recurrence over unconstrained MLP residual unroll.
- Do not treat RMSNorm + α-mix as a geometric attractor certificate.
- Any new length-gen claim requires matched covariates (see §7) before MEASURE reopen.

---

## 6. Explicit residue

1. **Gate 2 does not isolate length-generalization failure from covariate shift** on n / p / seq_len (see §2 table).
2. **ID Geo late collapse** (neg 0.91→0.21 at ep30 vs best@11) is **separate** from OOD all-positive bias (hard-neg=0, positives high including FF).
3. Positives-alone metrics never license OPEN under prereg (`interpretation_stop.prereg_rules.hardneg_collapse_overall_chance`).

---

## 7. Kill / next-cycle conditions

- **Do not** reopen a length-gen claim without matched n/p/seq_len OOD **or** an ablation that holds covariate fixed while varying hop only.
- **Do not** claim τ attractor / damping advantage without cycle-level probes that survive hard-neg discrimination (ID + matched OOD).
- **Do not** narrate Geo>FF from OOD positives while hard-neg=0 and FF also high on positives.

---

## 8. Pointers

### CLIs

```bash
python -m reachability_gen.run_id_2k_rematch_bound30   # or: reachability-id-2k-rematch-bound30
python -m reachability_gen.gen_ood_hops                # or: reachability-gen-ood-hops
python -m reachability_gen.run_ood_gate2               # or: reachability-ood-gate2
```

Gate 1 best-hop JSON is a non-destructive aggregation from existing val logs (see artifact `note`); no separate console script.

### Artifacts

- `artifacts/id_2k_rematch_bound30.json`
- `artifacts/id_2k_rematch_bound30_gate1_best_hop.json`
- `artifacts/id_2k_rematch_bound30_gate2_ood.json`
- `artifacts/ood_hops_generation_report.json`
- `artifacts/id_2k_generation_report.json`
- Checkpoints: `artifacts/id_2k_rematch_bound30_{ff,geo,loop}_best.pt`

### Could not verify from artifacts alone

- ID **p_support** is absent from `id_2k_generation_report.json` (cited from `gen_id_2k.py` + data uniques).
- ID **seq_len** distribution is absent from the ID gen report (computed from `data/id_2k.jsonl` encodings for this seal).

---

**Terminal:** FREEZE. **science_open=false.** Scientific claims H1–H3 closed as above. Engineering invariants remain OPEN *candidates* only.
