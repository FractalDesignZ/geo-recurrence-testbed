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
