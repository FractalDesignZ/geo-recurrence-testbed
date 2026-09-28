#!/usr/bin/env bash
# CI hygiene: CYCLE_STALK_FO_CERT_REGRESSION artifact gate (CPU; no train; no gen).
# science_open=false — locks sealed #30/#31/#35/#38 storyline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pip install -q -e ".[dev]" 2>/dev/null || pip install -q -e .

echo "== pytest FO/cert regression + inference contract =="
pytest -q tests/test_stalk_fo_cert_regression.py tests/test_stalk_inference_contract.py

echo "== console regression gate =="
python -m reachability_gen.run_stalk_fo_cert_regression

echo "== inference contract MEASURE (artifact-first) =="
python -m reachability_gen.run_stalk_inference_contract

echo "CI FO/cert + inference-contract OK (science_open=false)"
