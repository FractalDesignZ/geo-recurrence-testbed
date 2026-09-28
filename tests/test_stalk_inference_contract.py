"""Inference contract: frozen forward + cert refuse API; science_open=false."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.inference_contract import (  # noqa: E402
    CONTRACT_AGG,
    CONTRACT_SEEDS,
    CONTRACT_T,
    RECEIPT_FORMAT,
    SCIENCE_OPEN_LOCKED,
    InferenceContract,
    certify_and_refuse,
    compute_layout_hash,
    default_contract,
    emit_run_receipt,
    run_with_certificates,
)
from reachability_gen.run_stalk_inference_contract import (  # noqa: E402
    CYCLE,
    FO_CATCH_EXACT,
    VERDICTS,
    decide_verdict,
    run_cycle,
)


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_INFERENCE_CONTRACT"
    assert CONTRACT_T == 16
    assert CONTRACT_AGG == "prob_mean"
    assert tuple(CONTRACT_SEEDS) == tuple(range(10))
    assert SCIENCE_OPEN_LOCKED is False
    assert set(VERDICTS) == {
        "CONTRACT_LOCKED",
        "PARTIAL",
        "INVALID",
        "COLLATERAL",
    }
    assert FO_CATCH_EXACT == 45


def test_default_contract_locked():
    c = default_contract()
    assert c.T == 16
    assert c.ens_agg == "prob_mean"
    assert c.force_closed_dirty_yes is True
    assert c.force_open_dirty_no is False
    assert c.science_open is False


def test_contract_rejects_science_open_true():
    with pytest.raises(ValueError, match="science_open"):
        InferenceContract(science_open=True)


def test_contract_rejects_ens_remix():
    with pytest.raises(ValueError, match="ens_agg"):
        InferenceContract(ens_agg="logit_mean")


def test_contract_rejects_T_not_16():
    with pytest.raises(ValueError, match="T locked"):
        InferenceContract(T=32)


def test_contract_rejects_force_open_dirty_no():
    with pytest.raises(ValueError, match="force_open_dirty_no"):
        InferenceContract(force_open_dirty_no=True)


def test_contract_rejects_disable_refuse():
    with pytest.raises(ValueError, match="force_closed_dirty_yes"):
        InferenceContract(force_closed_dirty_yes=False)


def test_certify_and_refuse_dirty_yes():
    pred, cert, refuse = certify_and_refuse(
        3, [(1, 2)], 0, 2, pred=1
    )
    assert refuse is True
    assert pred == 0
    assert cert.clean is False
    assert cert.kind == "dirty_yes"


def test_certify_and_refuse_clean_yes():
    pred, cert, refuse = certify_and_refuse(
        3, [(0, 1), (1, 2)], 0, 2, pred=1
    )
    assert refuse is False
    assert pred == 1
    assert cert.clean is True


def test_run_with_certificates_refuse_and_hard_unanimous():
    rows = [
        {
            "encoding": "N 3 EDGES 1,2 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 0,
        },
        {
            "encoding": "N 3 EDGES 0,1 1,2 QUERY 0 2",
            "n": 3,
            "s": 0,
            "t": 2,
            "y": 1,
        },
    ]
    logits = torch.tensor(
        [
            [[0.0, 5.0], [0.0, 5.0]],
            [[0.0, 5.0], [0.0, 5.0]],
        ],
        dtype=torch.float32,
    )
    batch = run_with_certificates(
        rows, contract=default_contract(), logits_stack=logits
    )
    assert batch.preds_raw == [1, 1]
    assert batch.preds_contract == [0, 1]
    assert batch.n_refuse == 1
    assert batch.examples[0].refuse is True
    assert batch.examples[0].hard_unanimous is True
    assert batch.examples[1].refuse is False
    assert batch.science_open is False
    receipt = emit_run_receipt(batch, dataset="unit")
    assert receipt["format"] == RECEIPT_FORMAT
    assert receipt["science_open"] is False
    assert receipt["T"] == 16
    assert receipt["ens_agg"] == "prob_mean"
    assert "layout_hash" in receipt


def test_layout_hash_stable():
    a = compute_layout_hash()
    b = compute_layout_hash()
    assert a == b
    assert len(a) == 64


def test_decide_verdict_contract_locked():
    d = decide_verdict(
        api_ok=True,
        science_open=False,
        fo_killed=45,
        rem22_killed=22,
        cert_off_fo=45,
        matched_deltas={"overall_acc": 0.0, "hard_neg_acc": 0.0, "K16": 0.0},
        matched_harm=False,
        neq_energy=True,
        neq_zeta=True,
        regression39_pass=True,
        p_correct_clean=1.0,
    )
    assert d["verdict"] == "CONTRACT_LOCKED"


def test_decide_verdict_invalid_science_open():
    d = decide_verdict(
        api_ok=True,
        science_open=True,
        fo_killed=45,
        rem22_killed=22,
        cert_off_fo=45,
        matched_deltas={},
        matched_harm=False,
        neq_energy=True,
        neq_zeta=True,
        regression39_pass=True,
        p_correct_clean=1.0,
    )
    assert d["verdict"] == "INVALID"


def test_decide_verdict_collateral():
    d = decide_verdict(
        api_ok=True,
        science_open=False,
        fo_killed=45,
        rem22_killed=22,
        cert_off_fo=45,
        matched_deltas={"K16": -0.35},
        matched_harm=False,
        neq_energy=True,
        neq_zeta=True,
        regression39_pass=True,
        p_correct_clean=1.0,
    )
    assert d["verdict"] == "COLLATERAL"


@pytest.mark.skipif(
    not Path("artifacts/stalk_reach_certificates.json").exists(),
    reason="sealed cites missing",
)
def test_run_cycle_artifact_first(tmp_path: Path):
    out = tmp_path / "stalk_inference_contract.json"
    receipt = tmp_path / "receipt.json"
    art = run_cycle(out_path=out, receipt_path=receipt, live=False)
    assert art["science_open"] is False
    assert art["section22_unchanged"] is True
    assert art["verdict"] in VERDICTS
    assert art["verdict"] == "CONTRACT_LOCKED"
    assert out.exists()
    assert receipt.exists()
    rec = json.loads(receipt.read_text())
    assert rec["format"] == RECEIPT_FORMAT
    assert rec["science_open"] is False
    tables = art["tables"]
    assert tables["fo_catch_rem22"]["fo_killed"] == 45
    assert tables["fo_catch_rem22"]["rem22_killed"] == 22
    abl = tables["ablation_cert_on_vs_off"]
    assert abl["cert_off"]["FO"] == 45
    assert abl["cert_on"]["FO"] == 0
    neq = tables["neq_energy40_zeta41"]
    assert neq["neq_energy"] is True
    assert neq["neq_zeta"] is True
    assert art["llama_cpp_abstraction"]["ggml_kernel_port"] is False
