"""Stalk RED competent dissonance: science_open fail-closed; FAIL_OPEN/CLOSED; no train."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.gen_stalk_red_competent_dissonance import (  # noqa: E402
    K16,
    K20,
    MAX_TOKENS_ACCEPT,
    RED_HOPS,
    RED_SEED,
    generate_red,
)
from reachability_gen.run_stalk_red_competent_dissonance import (  # noqa: E402
    CITE_28,
    CONF_THRESH,
    CYCLE,
    DEFAULT_ENSEMBLE_SEEDS,
    D_CLOSED_MIN,
    EPI_CLOSED_MIN,
    FOCUS_T,
)
from reachability_gen.tokenize import split_encoding_tokens  # noqa: E402


def test_cycle_constants_fail_closed():
    assert CYCLE == "CYCLE_STALK_RED_COMPETENT_DISSONANCE"
    assert FOCUS_T == 16
    assert CONF_THRESH == 0.80
    assert D_CLOSED_MIN == 0.10
    assert EPI_CLOSED_MIN == 0.15
    assert tuple(DEFAULT_ENSEMBLE_SEEDS) == tuple(range(10))
    assert tuple(RED_HOPS) == (16, 20)
    assert RED_SEED == 210_000
    assert MAX_TOKENS_ACCEPT == 250
    assert CITE_28["verdict"] == "COMPETENT"
    assert abs(CITE_28["CD"] - 0.8111400496856206) < 1e-9


def test_generate_red_tiny_quotas(tmp_path):
    """Smoke: tiny quotas fill K16+K20+hard-neg; science_open false."""
    examples, report = generate_red(seed=RED_SEED, pos_per_cell=4, max_draws=20_000)
    assert report["science_open"] is False
    assert report["verify_ok"] is True
    assert report["hop_counts"][str(K16)] == 4
    assert report["hop_counts"][str(K20)] == 4
    assert report["hop_counts"]["-1"] == 8
    assert report["p_empirical"]["K16"]["harder_density"] is True
    assert all(
        len(split_encoding_tokens(ex.encoding)) <= MAX_TOKENS_ACCEPT for ex in examples
    )
    out = tmp_path / "red.jsonl"
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    assert sum(1 for _ in out.open()) == 16


def test_no_bag_train_flags():
    import reachability_gen.run_stalk_red_competent_dissonance as m

    assert m.DEFAULT_ENSEMBLE_SEEDS == tuple(range(10))
    doc = m.__doc__ or ""
    assert "No train" in doc or "no train" in doc.lower()
    assert "pr27" in doc.lower() or "#27" in doc
