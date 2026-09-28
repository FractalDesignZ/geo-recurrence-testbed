# CYCLE_STALK_FO_CERT_REGRESSION — MEASURE / CI hygiene (science_open=false)

| Field | Value |
|-------|-------|
| **Mode** | **MEASURE / CI hygiene** — **regression gate** locking sealed storyline (#30/#31/#35/#38); **not** `science_open` widen |
| **science_open** | **false** (always; **not widened**; §22 unchanged — matched-OOD only) |
| **Trigger** | After #38 energy selector `COLLATERAL_HARM` + oracle FO wall: lock FO HARD_UNANIMOUS + #35 cert refuse so future PRs cannot silently regress the sealed storyline |
| **Base** | `main` tip after PR #38 (`a8b8a68`) |
| **Prior** | #30 HN shatter + 45 FO; #31 STRUCTURAL_CLUSTER HARD_UNANIMOUS 34/45; #35 CERT_FO_CATCH 45/45; #38 oracle FO killed **0** |
| **Ledger** | `docs/LEDGER-OPEN-MEASURE-STOP.md` |
| **Train / gen** | **NONE** — no generation, no train fork, no ckpt rewrite |

## Goal

Wire a **regression gate** (artifact-first; optional CPU re-eval) that fails CI / local check if sealed storyline invariants drift:

1. **Hop-OOD FO HARD_UNANIMOUS wall (#31/#38):** among the known **45** FAIL_OPEN (ood_hops T16, frozen #22 ens), HARD_UNANIMOUS rate stays **high**; **oracle member cannot kill FO** (`oracle_fo_killed = 0` from #38).
2. **Certificate refuse (#35):** FO catch **45/45** shape; `P(correct|clean) ≈ 1`; matched-OOD collateral **≈ 0** (Δ floors from #35).
3. **Optional smoke (#30):** baseline FO count **45** / HN shatter still present **without** cert.

This cycle does **not** start generation or train. Prefer reusing sealed artifacts under `artifacts/`.

## Bound (closed — do not reopen)

| Attempt | Outcome |
|---------|---------|
| #22/#24 | ens `prob_mean` scoped science_open (matched-OOD) |
| #30 | FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED; residue `HN_FAIL_OPEN_CORE` (45) |
| #31 | STRUCTURAL_CLUSTER; HARD_UNANIMOUS **34**/45 |
| #35 | CERT_FO_CATCH; 45/45 FO + 22/22 rem-22; P(correct\|clean)=1; matched Δ=0 |
| #36–#38 | tropical / orientation / energy — not FO repair; #38 oracle FO **0**/45 |

## Locked thresholds (prereg — regression floors)

Sealed cite values in parentheses. Floors are **fail-closed** (must hold for `PASS_REGRESSION`).

### Gate A — HARD_UNANIMOUS / oracle wall (cite #31 / #38)

| Metric | Floor (locked) | Sealed cite |
|--------|----------------|-------------|
| `FO_TOTAL` | **== 45** | 45 (#30/#31/#35/#38) |
| `HARD_UNANIMOUS` count among FO | **≥ 30** | 34 (#31) |
| `HARD_UNANIMOUS` rate among FO | **≥ 0.70** | 34/45 ≈ 0.756 |
| `MIXED_PRED` among FO | **== 0** | 0 (#31; D_ex==0) |
| `oracle_fo_killed` (#38) | **== 0** | 0 |
| `oracle_rem22_killed` (#38) | **== 0** | 0 |

### Gate B — Certificate refuse (cite #35)

| Metric | Floor (locked) | Sealed cite |
|--------|----------------|-------------|
| `FO_killed` under cert arm | **== 45** | 45 |
| `rem22_killed` under cert arm | **== 22** | 22 |
| `P(correct \| clean)` post-policy | **≥ 0.99** | 1.0 |
| matched-OOD Δ overall / HN / K16 (cert vs baseline) | **abs ≤ 0.01** | 0.0 |
| #35 verdict | **`CERT_FO_CATCH`** | CERT_FO_CATCH |
| collateral `harm` | **false** | false |

### Gate C — Optional smoke baseline without cert (cite #30)

| Metric | Floor (locked) | Sealed cite |
|--------|----------------|-------------|
| baseline `FAIL_OPEN` | **== 45** | 45 |
| baseline ens HN | **≤ 0.10** and **< 0.50** (`HN_SHATTER_MAX`) | 0.067 |
| baseline ens K16 | **≥ 0.95** | 0.988 |
| #30 verdict contains | `HN_SHATTER_CONFIRMED` | FAIL_CLOSED_DOMINANT+HN_SHATTER_CONFIRMED |

### Verdicts (LOCKED)

| Label | Meaning |
|-------|---------|
| **`PASS_REGRESSION`** | Gates A+B pass (C optional but default-on); all floors hold; `science_open=false` |
| **`FAIL`** | Any mandatory floor fails |

Still **MEASURE / CI hygiene**, never OPEN. Passing does **not** widen `science_open` / does **not** claim hop-OOD OPEN.

## Preregistered protocol

```bash
# Artifact gate (default; CPU; no train; CI-safe)
python -m reachability_gen.run_stalk_fo_cert_regression
# or: reachability-stalk-fo-cert-regression
# or: scripts/ci_fo_cert_regression.sh
pytest -q tests/test_stalk_fo_cert_regression.py
```

| Item | Spec (locked) |
|------|----------------|
| Primary path | Read sealed artifacts `#30` / `#31` / `#35` / `#38` |
| Optional | `--require-smoke` (default true); `--skip-smoke` to omit Gate C |
| Train | **NONE** |
| Gen | **NONE** — do not regenerate `ood_hops.jsonl` |
| T | locked **16** (storyline substrate) |
| Artifact out | `artifacts/stalk_fo_cert_regression.json` |

### Artifact inputs (LOCKED)

| Cite | Path |
|------|------|
| #30 | `artifacts/stalk_hop_ood_hn.json` |
| #31 | `artifacts/stalk_hn_fail_open_autopsy.json` |
| #35 | `artifacts/stalk_reach_certificates.json` |
| #38 | `artifacts/stalk_energy_selector.json` |

### Prohibited defaults

- No train / PEFT / select / bag / JS / sheaf / generation fork
- No `science_open` widen; no §22 widen
- No claiming hop-OOD OPEN
- No rewriting sealed #30/#31/#35/#38 result bodies (append-only elsewhere)
- Do not treat this gate as a science cell that reopens energy/tropical/orientation chase

## Explicit non-goals

- No generation / train fork in this PR
- No tropical / orientation / energy re-chase
- No `science_open=true`
- No hop-OOD OPEN claim

## Metaphor

- **map ≠ location** — regression locks the sealed *location checks* (HARD_UNANIMOUS wall + cert refuse), not a new map train.
- **negatives = mirror** — FO HARD_UNANIMOUS is the generation wall; cert refuse is the post-hoc mirror that catches it without matched collateral.

## Results (fill after gate run)

| Field | Value |
|-------|-------|
| **Artifact** | `artifacts/stalk_fo_cert_regression.json` |
| **Cycle verdict** | **`PASS_REGRESSION`** |
| **science_open** | **false** |
| **Gates** | A PASS / B PASS / C PASS |
| **Gate A** | FO=45; HARD_UNANIMOUS=34/45 (rate=0.756); oracle_fo_killed=0 |
| **Gate B** | FO_killed=45/45; rem22=22/22; P(correct\|clean)=1.0; matched Δ={'K16': 0.0, 'hard_neg_acc': 0.0, 'overall_acc': 0.0} |
| **Gate C** | FO=45; HN=0.067; K16=0.988 |
| **Base SHA** | `a8b8a688eb5ed2164a104c2230cbfe3140eae70b` |
| **Train / gen** | **none** |
