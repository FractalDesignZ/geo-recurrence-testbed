# geo-recurrence-testbed / reachability_gen

**RESEARCH / MEASURE plumbing only.** This package generates labeled
directed-graph reachability datasets (JSONL), schematic FLOP / arm-interface /
eval-harness code, and optional real torch **feed-forward** / **geometric** / **Euclidean-loop**
training paths for overfit sanity checks and param-matched rematch. CoT remains a stub. It does **not** make science OPEN claims about learnability or geometry.
Parameter matching and FLOP tables are accounting hygiene — they never stamp
science OPEN.

Task: directed simple graph; `y = 1` iff target `t` is reachable from source `s`.


**ADR-001** (`docs/ADR-001-metrics-and-compute.md`) freezes encoding, FLOP
formulas, param parity, hop stratification, and the `RunMetricRecord` logging
schema as immutable runtime assertions in `adr_invariants.py`. Status: Accepted
for scaffold; **MEASURE plumbing — not OPEN**.

## Setup

Stdlib-only runtime (Python ≥ 3.10). Optional venv + pytest for tests:

```bash
cd /workspace/geo-recurrence-testbed
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Torch is **optional**. Without it, the eval harness writes the full ADR-001
schema with placeholder stub metrics (schema CI stays green). With torch
(CPU is fine):

```bash
pip install -e ".[torch]"
# or: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

the feed-forward arm attaches a real `FeedForward` module so metrics JSONL gets
**real CE loss / accuracy**. FF trajectories stay empty by design
(`drift_trajectory=[]`, `terminal_drift=null`, `perturbation_delta=null`).
The geometric arm can attach a real weight-tied `GeometricRecurrent`
(`Phi(z_t, c; τ_t)`, fixed `T=6` for the ID-hop gate) with real
`drift_trajectory` / `terminal_drift` telemetry.

Or without install:

```bash
export PYTHONPATH=src
```

## Generate

```bash
# help
python3 -m reachability_gen.generate --help

# tiny train set (2 examples per (n,p) cell → 2×3×3 = 18 rows)
python3 -m reachability_gen.generate --split train --n-per-cell 2 --out data/train_tiny.jsonl

# all splits
python3 -m reachability_gen.generate --all-splits --n-per-cell 4
```

Console script (after editable install): `reachability-gen --help`.

Each JSONL row includes `hop_distance` (shortest-path hops; **`-1` when `y=0`**)
and `is_ood` (`hop_distance > K_TRAIN_MAX` with `K_TRAIN_MAX=6`). Positive
examples prefer stratified hop buckets toward ID `[2,6]` (and OOD `{8,12,16}`
on `ood_size` / `--prefer-ood-hops` when the pool permits).

## FLOP accounting + arm stubs

Schematic inference FLOPs (testbed compute table; constants documented in
`src/reachability_gen/flops.py` and checksummed against ADR-001):

| Arm | Inference FLOPs |
|-----|-----------------|
| Feed-forward | `L * F_block(m, d)` |
| Geometric / Euclidean loop | `T * F_block(m, d)` (+ optional cheap τ embed for Geo) |
| CoT | `F_prefill(m) + Σ_{k=1..K_used} F_decode(m+k)` |

Rules encoded in code:

- CoT debits **realized** `K_used`, never the cap `K`.
- `F_block = 24*m*d^2 + 4*m^2*d` (attention + MLP expansion 4, MAC=2, no bias/norm FLOPs).
- Non-embedding params for Geo / FF / Loop should match within **±5%**
  (`PARAM_TOL=0.05` in `adr_invariants.py`); the checker always sets
  `science_open=False`.

Arm Protocol stubs (`arms.py`): `FeedForwardArm`, `GeometricRecurrentArm`,
`EuclideanLoopArm`, `ChainOfThoughtArm` — `forward_stub` returns placeholder
logits; FF/Geo/Loop can attach real torch modules. `forward(..., return_trajectory=bool)`
returns `(logits, trajectory)` when `return_trajectory=True`.

### FLOP demo CLI

```bash
python3 -m reachability_gen.flops_demo
python3 -m reachability_gen.flops_demo --m 48 --d 64 --T 4 --L 1 --K-used 8 --no-tau
```

Prints a tiny comparison table for one dummy context length. No training.

## Eval harness (schema writer)

```bash
python3 -m reachability_gen.eval_demo
python3 -m reachability_gen.eval_demo --examples data/train_tiny.jsonl \
  --out artifacts/metrics.jsonl --limit 4
```

`ExperimentHarness` (`src/reachability_gen/harness/runner.py`):

- Asserts Geo/FF/Loop param parity (±5%).
- Runs stub forwards for Geo/Loop/CoT; real torch FF forward when attached.
- Writes ADR-001 `RunMetricRecord` rows to JSONL (`log_eval_step`).
- `measure_phase_diagnostics` stub fills drift fields for recurrent arms
  (`EPS_SIGMA=1e-4`); FF leaves them empty/null by design.
- Emits user TypedDict keys (`run_id`, `param_count`, `cumulative_flops`, …).
- Never self-stamps `science_open=True`.

## Spec summary

| Item | Value |
|------|--------|
| Train/val/test `n` | `{8, 12, 16}` |
| `ood_size` `n` | `{24, 32}` |
| Edge prob `p` | `{0.15, 0.25, 0.35}` (Bernoulli on ordered pairs, no self-loops) |
| Labels | ≈50/50 via stratified rejection; `reject_rate` logged |
| Encoding v0 | Canonical sorted edge-list + `QUERY s t` (no scratchpad) |
| ID hops | `K ∈ [2, 6]` |
| OOD hops | `K ∈ {8, 12, 16}`; `is_ood = K > 6` |
| Unreachable hop | `hop_distance = -1` (ADR allows null for consumers) |
| Seeds | Fixed table per split in `splits.py` |

### Balancing note

For each example we sample an ER digraph, compute the reachability matrix, and
draw `(s,t)` from the target-label pool (with hop-bucket preference for `y=1`).
Graphs with an empty pool are rejected. At large `n` and high `p` (especially
`ood_size` with `p=0.35`) ER digraphs are nearly strongly connected, so `y=0`
pools are often empty; after `--max-rejects` the generator falls back to an
unconstrained sample so JSONL still completes. In-distribution cells (`n≤16`)
typically keep exact 50/50.

### Why edge-list encoding (not adjacency)?

See docstring in `src/reachability_gen/encode.py`: unique canonical order for
stable `edge_hash`, compact for sparse ER digraphs, flat token stream without
row separators. **Encoding remains locked** — arm stubs do not change it.

## Feed-forward overfit gate (torch)

Real `FeedForward` (`models/feedforward.py`): L unshared transformer blocks →
logits `[B, 2]`. `FeedForwardTrainer`: AdamW + CE + grad clip 1.0.

**Balanced gate (default):** exactly 16 y=1 with K∈[2,6] + 16 y=0 *hard
negatives* (deg(s)≥1, deg(t)≥1, unreachable). Pass: per-class acc=1.0 and
CE loss < 1e-3 within ≤100 steps. Regenerates data as needed; logs reject
reasons.

```bash
# Balanced overfit gate (default)
python -m reachability_gen.overfit_ff --balanced --examples data/train_tiny.jsonl --steps 100
# Legacy (non-balanced): loss < 0.05, overall acc 1.0
python -m reachability_gen.overfit_ff --no-balanced --examples data/train_tiny.jsonl

# After PASS: full ID train + hop-stratified val → artifacts/ff_id_train_*.json*
python -m reachability_gen.train_ff_id --epochs 8

# CI-ish script (installs torch CPU if needed, pytest, balanced overfit gate)
bash scripts/ci_ff_overfit.sh
```


## Geometric recurrent overfit gate (torch)

Real `GeometricRecurrent` (`models/geometric.py`): weight-tied `Phi(z_t, c; τ_t)`
for `T` cycles (default / gate: **T=6** = max ID hop). Context `c` is the same
tokenized edge-list encoding as FF. `forward(..., return_trajectory=True)` →
`(logits [B,2], list of z_t)` for drift.

**Balanced gate (identical construction to FF):** reuses
`overfit_ff.ensure_balanced_batch` (same seed → same 16+16 examples). Pass:
per-class acc=1.0, CE < 1e-3 within ≤100 steps, and `drift_trajectory` length
`T-1` (or `T`) finite and non-zero (fail on NaN/Inf / all-zero).

```bash
python -m reachability_gen.overfit_geo --balanced --examples data/train_tiny.jsonl --steps 100
# or: reachability-overfit-geo --balanced
bash scripts/ci_geo_overfit.sh
```

Do **not** treat this gate as science OPEN. Stop after the overfit result —
large-scale ID train for Geo is out of scope for this scaffold step.



## ID 2k FF vs Geo comparison (MEASURE)

Fixed 2000-example ID set (`data/id_2k.jsonl`): 1600 train / 400 val, exact
1000/1000 labels, hop-uniform positives K∈{2..6}, **all** negatives hard
(`deg(s)≥1`, `deg(t)≥1`, unreachable). Expanded `n` support for scarce hop-6.

```bash
# regenerate 2k + train FF L=2 and Geo T=6 (tau) on the same splits
python -m reachability_gen.run_id_2k_comparison
# or: reachability-id-2k-comparison

# data only / verify
python -m reachability_gen.gen_id_2k
python -m reachability_gen.gen_id_2k --verify-only data/id_2k.jsonl
```

Writes `artifacts/id_2k_comparison.json` with hop-stratified val
(`acc_mean` / `loss_mean`; Geo also `mean_drift_trajectory` /
`mean_terminal_drift` / damp-vs-expand). **`science_open=false` always.**

## ID 2k param-matched rematch (MEASURE)

Three arms on the **exact** existing `data/id_2k.jsonl` (never regenerated here):

1. FF L=2 (~121218 params)
2. Geo T=6 + τ, width/MLP scaled into ±5% of FF
3. Euclidean loop T=6 (same arch as Geo, τ disabled), also in that window

Hard-fails before Geo/Loop train if either recurrent arm is outside ±5% of FF
(`assert_param_parity`). Shared hyperparams: AdamW, lr, batch, grad clip 1.0,
seed, epochs/early-stop. **`science_open=false` always.**

```bash
python -m reachability_gen.run_id_2k_rematch
# reuse prior FF block from comparison (still trains Geo+Loop):
python -m reachability_gen.run_id_2k_rematch --reuse-ff-from-comparison
```

Writes `artifacts/id_2k_rematch.json` with hop-stratified val for
K∈{-1,2,3,4,5,6} (acc/loss; Geo+Loop also mean drift / terminal_drift /
perturbation_delta / damp regime) and a `param_match` section with
`within_5pct` for both vs FF.




## ID 2k fixed-30 rematch (MEASURE; no early stop)

Same three arms / exact `data/id_2k.jsonl` / ±5% parity (recurrent **mlp×10**),
but **epochs=30 locked with NO early stopping**. Still tracks/saves the best
val-acc checkpoint, then always finishes all 30 epochs.

Drift-audit on recurrent update::

    z_{t+1} = z_t + α · (Φ(h_t) - z_t),  α = 0.5

Outer stream LN (``LN(z+α·v)``) blocked learning with this Pre-LN Phi, so the
run keeps the α-mix and reports **raw + LN-normalized** δ_t (protocol option
b), plus mean ``||z_t||₂`` per cycle and grad-clip saturation rate.
**`science_open=false` always.**

```bash
python -m reachability_gen.run_id_2k_rematch_fixed30
# or: reachability-id-2k-rematch-fixed30
```

Writes `artifacts/id_2k_rematch_fixed30.json`.


## ID 2k bound-30 rematch (MEASURE; clip/LR + RMSNorm)

Same three arms / exact `data/id_2k.jsonl` / ±5% parity / **epochs=30 no
early-stop**, plus Phase A+B plumbing:

**Phase A — effective LR / clip** (chosen values; FF control unchanged):

| Arm | lr | grad clip |
|-----|-----|-----------|
| FF | `3e-3` | `1.0` |
| Geo / Loop | `1.5e-3` (=0.5×FF) | `2.5` (=2.5×FF) |

Rationale: fixed30 under shared clip=1.0 lr=3e-3 had FF sat≈7.7%, Geo≈16%,
Loop≈21%. With RMSNorm, lr=3e-3 stuck Geo at chance; scaled recurrent lr +
raised max_norm brings sat ≈ FF while remaining learnable.

**Phase B — state bound:** after α=0.5 residual mix apply **RMSNorm** on `z`
(stream LN blocked learning). Logs mean `||z_t||₂` and raw + LN- +
RMS-normalized drift. **`science_open=false` always.**

```bash
python -m reachability_gen.run_id_2k_rematch_bound30
# or: reachability-id-2k-rematch-bound30
```

Writes `artifacts/id_2k_rematch_bound30.json`.


## OOD hops dataset + Gate 2 stress (MEASURE)

Held-out hop-OOD set (`data/ood_hops.jsonl`): positives with
`hop_distance ∈ {8,12,16}` (ADR-001 `OOD_HOP_VALUES`), hard negatives
(`deg(s)≥1`, `deg(t)≥1`, unreachable), 50/50 class balance, `is_ood=True`
on positives. Encodings filtered to fit bound30 `max_len=257`.
**`science_open=false` always.**

```bash
# generate OOD hops JSONL + report (jsonl gitignored like id_2k)
python -m reachability_gen.gen_ood_hops
python -m reachability_gen.gen_ood_hops --verify-only data/ood_hops.jsonl

# Gate 2: inference-only stress on bound30 best ckpts
# Fixed L=2 / T=6 for all arms; dynamic T∈{8,12,16} for Geo/Loop
python -m reachability_gen.run_ood_gate2
# or: reachability-ood-gate2
```

Writes `artifacts/ood_hops_generation_report.json` and
`artifacts/id_2k_rematch_bound30_gate2_ood.json` (tables + prereg
`interpretation_stop`; never stamps OPEN). No retraining.




## Covariate-matched OOD (MEASURE; Gate2 residue)

Resolves the Gate2 confound (hop K∈{8,12,16} vs seq_len dilation: ID
whitespace-mean~57 / prior OOD split-token mean~203). New held-out set
(`data/covariate_matched_ood.jsonl`) keeps positives at hop ∈{8,12,16} and
hard negatives (50/50), but **strictly** constrains model-visible encoding
token length (`split_encoding_tokens`) to **[45, 70]** (target mean ~57) via
path-backbone + few non-shortcut distractors (ER grid search documented in
the generation report; pure ER under the band rarely yields K=16).
**`science_open=false` always.**

```bash
# generate matched OOD JSONL + report (jsonl gitignored)
python -m reachability_gen.gen_covariate_matched_ood
python -m reachability_gen.gen_covariate_matched_ood --verify-only data/covariate_matched_ood.jsonl

# zero-retrain eval on bound30 best ckpts (fixed L=2/T=6; dynamic T∈{8,12,16})
python -m reachability_gen.run_covariate_matched_ood
# or: reachability-ood-covariate-matched
```

Writes `artifacts/covariate_matched_ood_generation_report.json` and
`artifacts/id_2k_rematch_bound30_gate2_matched_ood.json` (hop tables +
prereg STOP/OPEN-candidate framing; hard-neg collapse still primary).
No retraining. Band is fail-closed: do not silently widen past 70.



## Cycle: SHEAF_INFERENCE (MEASURE)

Follow-on to STALK_LOCALIZATION. Infers `A_hat` from **edge tokens** (hard gate
θ=0.5; STE available; default gate-detach into diffusion). Local stalk init
(`s=1`, else `0`); shared bias-free Φ; discrete `T ∈ {6,8,12,16}`. **No hard
adjacency oracle at eval.** Gold edges may be used for **aux reconstruction
loss in training only**. Param parity ±5% of FF ~121218. `science_open=false`.
Baseline residue: stalk seal `b144dac`.

```bash
python -m reachability_gen.overfit_sheaf --balanced --out artifacts/sheaf_infer_overfit.json
python -m reachability_gen.run_sheaf_infer_gate1  # artifacts/sheaf_infer_matched_ood.json
```

Prereg (report honestly): hard-neg ≥0.95 AND K16≥0.75 at T=16 on covariate-matched OOD.

## Cycle: STALK_MULTI_SEED_RECONFIRM (MEASURE)

Follow-on to stalk untrained control (PR #11). Retrain stalk-local FractalCore
≥3 seeds (hard A, stalk-local, no soft ACT); matched-OOD T∈{6,8,12,16};
mean±std overall/hard-neg/K16; untrained per seed; degree-balanced secondary.
Prereg (mean): hard-neg≥0.95 and K16@T16≥0.75. `science_open=false` always.

```bash
python -m reachability_gen.run_stalk_multi_seed_reconfirm
```

Verdict this run: **`OPEN_CONTINGENT_AT_RISK` → DEMOTION `MEASURE`** (1/3 seeds; means miss floors).
Stalk §6 live OPEN demoted to MEASURE (seal §13); not INVALID. Do not silently widen.
See `docs/CYCLE_STALK_MULTI_SEED_RECONFIRM.md`.

## Cycle: STALK_STABILIZE_MULTI_SEED (MEASURE)

Middle-out harden of hard-Â stalk corridor (5 seeds, 60 ep, cosine LR, joint ID-val
T16 selection). Mean floors PASS (HN 0.957 / K16 0.863); seed-wise **2/5** →
**`MEASURE_STILL`**. `science_open=false` (not widened).

```bash
python -m reachability_gen.run_stalk_stabilize_multi_seed
```

Artifact: `artifacts/stalk_stabilize_multi_seed.json`. See `docs/CYCLE_STALK_STABILIZE_MULTI_SEED.md`.

## Cycle: STALK_STABILIZE_V2 (MEASURE → STOP_FRAGILE)

Continue from PR #14 MEASURE_STILL. Knobs: 90 ep, cosine 1.5e-3→1.0e-4, gated
joint **0.7·HN+0.3·overall** @ T16 ID-val (overall≥0.85). Result: seed **1/5**,
K16 mean **0.423** → **`STOP_FRAGILE`** (HN-heavy under-propagation). Prefer #14
0.5/0.5 corridor. `science_open=false`.

```bash
python -m reachability_gen.run_stalk_stabilize_v2
```

Artifact: `artifacts/stalk_stabilize_v2.json`. See `docs/CYCLE_STALK_STABILIZE_V2.md`.


## Cycle: STALK_PARK_ACCEPT_MEASURE (MEASURE close)

Honest park after AUDIT V3 (PR #19). **No new training.** Accept hard-Â stalk
**PR #14** recipe (0.5·HN+0.5·overall, 60ep, uniform ID) as best MEASURE
corridor — **MEASURE_STILL** (HN 0.957 / K16 0.863 PASS; seed **2/5**).
Seed-fragile (#14 2/5; #18 envelope 3/10). V2/V3/ObjV1 **STOP_FRAGILE**.
Select / curriculum / seed-panel chase **CLOSED**. `science_open=false`.

See `docs/CYCLE_STALK_PARK_ACCEPT_MEASURE.md` / seal §19 / ledger priority 5 done.
Next active = orthogonal aux ablation (ledger priority 6) — not V4 select.

## Cycle: STALK_SEED_ENSEMBLE (PASS_CANDIDATE → scoped science_open)

Do **not** assume #14 tops out. Seed-fragile singles (#14 2/5; #18 3/10) leave
an isomorphic opportunity: **ensemble** frozen #14/#18 hard-Â stalk ckpts at
eval (primary **`prob_mean`**; secondary logit_mean / majority_vote). **No** new
select/upsample (0 fill-trains). Matched-OOD T16: ens overall **0.996**, HN
**1.000**, K16 **1.000** vs singles mean 0.903 / 0.935 / 0.792 (ΔHN **+0.065**,
ΔK16 **+0.208**). LOO HN 1.000±0 / K16 0.991±0.008. Verdict
**`PASS_CANDIDATE`**. Human seal: **scoped `science_open=true`** — ensemble-at-eval
only (singles remain MEASURE_STILL/fragile; distill STOP; select/curriculum CLOSED;
sheaf unsupervised NOT opened).

```bash
python -m reachability_gen.run_stalk_seed_ensemble
# or: reachability-stalk-seed-ensemble
```

Artifact: `artifacts/stalk_seed_ensemble.json`. See `docs/CYCLE_STALK_SEED_ENSEMBLE.md` /
`docs/CYCLE_STALK_SEED_ENSEMBLE_SEAL.md` / seal §22.


## Cycle: STALK_SWA_PERSIST (MEASURE)

After #22/#24 ensemble **OPEN** (map only) and #23 soft distill **STOP**, try
**single-model persistence** via Polyak **SWA** on frozen #14 harden (ep≥31;
seeds **0..4**). Primary = SWA weights; paired #14 lex select. Matched-OOD T16:
SWA mean HN **0.867** FAIL / K16 **0.910** PASS (seed **2/5**); vs select
ΔHN **−0.090** / ΔK16 **+0.047** → **`MEASURE`** (partial; HN↔K16 trade).
Ensemble still dominates (1.000/1.000). §22 scope **not** widened.
`science_open=false`.

```bash
python -m reachability_gen.run_stalk_swa_persist
# or: reachability-stalk-swa-persist
```

Artifact: `artifacts/stalk_swa_persist.json`. See `docs/CYCLE_STALK_SWA_PERSIST.md`.


## Cycle: STALK_COMPETENT_DISSONANCE (COMPETENT_vs_CHAOS)

Eval-only audit: frozen **#14/#18/#22** ens vs **#27** bag — global/slice pairwise
disagree, member acc on agree vs disagree sets, competent-dissonance
`CD = μ_acc · min(D_hard, 0.25)/0.25`, optional `ood_hops` shatter probe.
Verdicts: **COMPETENT** / **ECHO_RISK** / **CHAOS**. **No train.** `science_open=false`.

```bash
python -m reachability_gen.run_stalk_competent_dissonance
# or: reachability-stalk-competent-dissonance
```

Artifact: `artifacts/stalk_competent_dissonance.json`. See `docs/CYCLE_STALK_COMPETENT_DISSONANCE.md`.

## Cycle: STALK_BAG_DIVERSITY (MEASURE_LIFT)

After #26 multi-hyp STOP (heads collapsed) and audit showing #22 lift rides on
disagreement: train **DGE-style independent** hard-Â stalk bag (separate params;
bootstrap + EDGE_KEEP_P=0.75 graph-subspace; members **0..4**). Eval `prob_mean`.
Bag ens HN **0.821** FAIL / K16 **0.788** PASS; lifts vs bag singles
(ΔHN/K16 **+0.101 / +0.250**) → **`MEASURE_LIFT`**. Pairwise disagree **0.431**
≥ #22 **0.162** (`DISAGREE_GE_REF`) but far below #22/#14 accuracy. Not multi-hyp,
not soft distill, not SWA-only. §22 **not** widened. `science_open=false`.

```bash
python -m reachability_gen.run_stalk_bag_diversity
# or: reachability-stalk-bag-diversity
```

Artifact: `artifacts/stalk_bag_diversity.json`. See `docs/CYCLE_STALK_BAG_DIVERSITY.md`.

## Cycle: STALK_EPISTEMIC_DISAGREEMENT (MEASURE → STOP)

After #22/#24 ens OPEN and #25 SWA MEASURE: (1) audit whether #22 lift rides on
member disagreement; (2) one multi-hyp (H=3) train with CE−λ·JS (**not** soft
distill). Audit: **`AUDIT_LIFTS_ON_DISAGREEMENT`** — median-split lift gaps
Δoverall/HN/K16 **+0.175 / +0.164 / +0.221** on high-disagreement examples;
epistemic H 0.192 > aleatoric 0.080. Multi-hyp mean HN **0.942** / K16 **0.712**
FAIL floors; within-head disagree collapsed (~0.009); **0/3** → **`STOP`**.
§22 **not** widened. `science_open=false`.

```bash
python -m reachability_gen.run_stalk_epistemic_disagreement
# or: reachability-stalk-epistemic-disagreement
```

Artifact: `artifacts/stalk_epistemic_disagreement.json`. See `docs/CYCLE_STALK_EPISTEMIC_DISAGREEMENT.md`.

## Cycle: STALK_ENSEMBLE_DISTILL (MEASURE → STOP)

After #22 ensemble **PASS_CANDIDATE** (map), distill frozen #14/#18
**`prob_mean`** teacher into **ONE** hard-Â stalk student (location).
Negatives=mirror (HN floor). Select stays **#14 0.5/0.5** (CLOSED).
α=0.5, τ=2.0, student seeds **0,1,2**. Result: student mean HN **0.843** /
K16 **0.575** FAIL floors; **0/3** seed PASS; both below #14 seed0
(HN 1.000 / K16 0.863) and far below ensemble (1.000/1.000) → **`STOP`**.
Map did not compress. Prefer #14 + #22 overlay. `science_open=false`.

```bash
python -m reachability_gen.run_stalk_ensemble_distill
# or: reachability-stalk-ensemble-distill
```

Artifact: `artifacts/stalk_ensemble_distill.json`. See `docs/CYCLE_STALK_ENSEMBLE_DISTILL.md`.

## Cycle: STALK_SEED_STABILITY (MEASURE → MEASURE_ENVELOPE)

Freeze PR #14 recipe exactly (0.5/0.5 select, 60ep, hard-Â, uniform ID). **NO**
new select/upsample. Seeds **0..9** (reconfirm 0..4 + train 5..9). Matched-OOD
T16: mean HN **0.935±0.090** (FAIL ≥0.95), K16 **0.792±0.288** (PASS), seed
PASS **3/10** (rate 0.30). CI95 HN [0.879, 0.991]. Verdict
**`MEASURE_ENVELOPE`**. Prefer #14 MEASURE_STILL unchanged. `science_open=false`.

```bash
python -m reachability_gen.run_stalk_seed_stability
# or: reachability-stalk-seed-stability
```

Artifact: `artifacts/stalk_seed_stability.json`. See `docs/CYCLE_STALK_SEED_STABILITY.md`.

## Cycle: STALK_OBJECTIVE_V1 (MEASURE → STOP_FRAGILE)

Easy $ after select-weight chase closed (V2/V3 STOP). Kept **#14 0.5/0.5**
ID-val select; moved HN + hop≥5 into training (upsample×2 + weighted CE).
Result: seed **0/5**, HN mean **0.927** FAIL / K16 **0.845** PASS →
**`STOP_FRAGILE`**. Prefer #14 MEASURE_STILL. `science_open=false`.

```bash
python -m reachability_gen.run_stalk_objective_v1
```

Artifact: `artifacts/stalk_objective_v1.json`. See `docs/CYCLE_STALK_OBJECTIVE_V1.md`.

## Cycle: STALK_STABILIZE_V3 (MEASURE → STOP_FRAGILE)

Continue from PR #15 STOP_FRAGILE / prefer #14 corridor. Knobs: 60 ep, cosine
1.5e-3→1.5e-4, equal-weight joint **(1/3)·HN+(1/3)·K16_sel+(1/3)·overall** @ T16
(K16 from selection-only longhop aux; HN_weight=1/3). Result: seed **1/5**,
HN mean **0.838** FAIL / K16 **0.968** PASS → **`STOP_FRAGILE`** (inverted V2
failure). Prefer #14 0.5/0.5 corridor. `science_open=false`.

```bash
python -m reachability_gen.run_stalk_stabilize_v3
```

Artifact: `artifacts/stalk_stabilize_v3.json`. See `docs/CYCLE_STALK_STABILIZE_V3.md`.

## Cycle: STALK_LOCALIZATION (MEASURE)

Follow-on to FRACTAL_CORE_GENESIS. Kills global `(s,t)` broadcast `+c` and soft
ACT. Local potentials: stalk at source slot `s`, probe at target slot `t`,
intermediates `0`. Discrete `T ∈ {6,8,12,16}` only. Param parity ±5% of FF
~121218. `science_open=false`.

```bash
python -m reachability_gen.overfit_fractal --balanced --out artifacts/fractal_core_stalk_overfit.json
python -m reachability_gen.run_fractal_core_gate1  # writes artifacts/fractal_core_stalk_gate1_matched_ood.json
```

Prereg (report honestly): hard-neg ≥0.98 AND K16≥0.80 at T=16 on covariate-matched OOD.

## Cycle: FRACTAL_CORE_GENESIS (MEASURE)

Branch: `cycle/fractal-core-genesis`. Adds **FractalCore** — adjacency-masked
weight-tied recurrence with boundary re-injection `+c` and ACT-style adaptive
halting. **`science_open=false` always.** Mandelbrot / fractal boundary
language in docs is **aspirational analogy only**; evidence is metrics in
`artifacts/` (never the metaphor).

Node-slot encoding helpers recover directed adjacency from the locked
edge-list `encoding` (token indices alone cannot); attention uses
`A_ij = -inf` when no directed edge (self allowed). Param parity via
`_verify_param_parity` ±5% of FF ~121218.

```bash
# Gate 0 — balanced 16/16 overfit (fail-closed)
python -m reachability_gen.overfit_fractal --balanced \
  --out artifacts/fractal_core_overfit.json

# Gate 1 — id_2k ×30 epochs (bound30 recurrent lr/clip) + matched OOD
python -m reachability_gen.run_fractal_core_gate1
# writes artifacts/fractal_core_gate1_matched_ood.json
```

## PR / review lane

Copilot (and similar) are **authoring-side**. **CodeRabbit** is **review-side**
hygiene for MaxOp OPEN|STOP cycles (`science_open=false` by default).

- Config: [`.coderabbit.yaml`](.coderabbit.yaml) — auto-review on PRs to `main`
  (drafts skipped; WIP titles skipped); path instructions for seals/ADRs,
  `artifacts/*.json`, `gen_*.py` / `run_*.py`, and `tests/`.
- Template: [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)
  — Mode, cycle, substrate, prereg criteria, evidence paths, `science_open`
  default checkbox, STOP / OPEN-candidate / RESIDUE, test plan
  (`pytest -m "not slow"`).

**Install (one-liner):** install the [CodeRabbit GitHub app](https://github.com/apps/coderabbitai)
on this repo for reviews to fire. This repo only ships the config; the app is
not installed from here.

Direct pushes to `main` remain possible; **PRs are preferred for research
cycles** so review-lane checks (seals append-only, no invented OPEN, artifact
cite hygiene) can run before merge.

## Tests


```bash
cd /workspace/geo-recurrence-testbed
source .venv/bin/activate   # if using venv
pytest -q                   # schema / FLOP / stub tests (no torch required)
pytest -q tests/test_ff_torch.py -m "not slow"   # when torch installed
pytest -q tests/test_geo_torch.py -m "not slow"  # geo arm + drift
```

## Logging fields

### Example JSONL (generator)

Required: `split`, `seed`, `n`, `p`, `edge_hash`, `s`, `t`, `y`,
`hop_distance`, `is_ood`.

Also written: `encoding`, optional empty `arm_id` / `arm_meta`, diagnostics
`reject_rate` / `n_attempts`.

### Metrics JSONL (harness `RunMetricRecord`)

Required (user TypedDict): `run_id`, `seed`, `arm`, `step`, `epoch`,
`param_count`, `d_model`, `seq_len`, `cycles_or_depth`, `tokens_decoded`,
`cumulative_flops`, `hop_distance`, `is_ood`, `loss`, `accuracy`,
`drift_trajectory`, `terminal_drift`, `perturbation_delta`.

For arm `ff-*` with torch attached: `loss` / `accuracy` are real CE metrics;
`drift_trajectory=[]`, `terminal_drift=null`, `perturbation_delta=null`,
`tokens_decoded=null` (**FF trajectories empty by design** — see ADR-001).

Optional example-linkage extras (documented in ADR-001): `split`, `n`, `p`,
`edge_hash`, `s`, `t`, `y`. See ADR-001 §7 migration table for renames
(`params`→`param_count`, `flops`→`cumulative_flops`,
`z_drift_series`→`drift_trajectory`, `T_or_L_or_K`→`cycles_or_depth`).
