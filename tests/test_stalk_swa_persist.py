"""Stalk SWA persist: science_open fail-closed; SWA_START; #14 select freeze."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.run_stalk_swa_persist import (  # noqa: E402
    CYCLE,
    DEFAULT_SEEDS,
    PREREG_HARD_NEG,
    PREREG_K16,
    SWA_START,
    _select_ckpt,
    _swa_ckpt,
    _swa_update,
)
from reachability_gen.models.fractal_core import FractalCore  # noqa: E402
from reachability_gen.run_fractal_core_gate1 import DEFAULT_D, DEFAULT_MLP, _n_heads  # noqa: E402


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_SWA_PERSIST"
    assert PREREG_HARD_NEG == 0.95
    assert PREREG_K16 == 0.75
    assert SWA_START == 31
    assert tuple(DEFAULT_SEEDS) == (0, 1, 2, 3, 4)


def test_ckpt_path_helpers():
    assert _swa_ckpt(0).name == "fractal_core_stalk_swa_persist_seed0_swa.pt"
    assert _select_ckpt(2).name == "fractal_core_stalk_swa_persist_seed2_select.pt"


def test_swa_update_polyak_mean():
    model = FractalCore(
        d=DEFAULT_D,
        T=6,
        n_heads=_n_heads(DEFAULT_D),
        mlp_expansion=DEFAULT_MLP,
        max_nodes=64,
        max_T=16,
        use_tau=True,
        apply_cycle_rmsnorm=True,
    )
    # Seed swa with zeros-like of current
    swa = {k: torch.zeros_like(v).cpu() for k, v in model.state_dict().items()}
    # Force a float param to known value
    float_keys = [k for k, v in model.state_dict().items() if v.is_floating_point()]
    assert float_keys
    k0 = float_keys[0]
    with torch.no_grad():
        model.state_dict()[k0].fill_(2.0)
    _swa_update(swa, model, n_averaged=1)
    assert torch.allclose(swa[k0], torch.full_like(swa[k0], 2.0))
    with torch.no_grad():
        model.state_dict()[k0].fill_(4.0)
    _swa_update(swa, model, n_averaged=2)
    # mean of 2 and 4 = 3
    assert torch.allclose(swa[k0], torch.full_like(swa[k0], 3.0))
