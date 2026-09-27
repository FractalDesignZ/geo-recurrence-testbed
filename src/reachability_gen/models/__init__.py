"""Optional torch models (RESEARCH / MEASURE plumbing)."""

from __future__ import annotations

__all__: list[str] = []

try:
    from reachability_gen.models.feedforward import FeedForward, TransformerBlock

    __all__ += ["FeedForward", "TransformerBlock"]
except ImportError:  # pragma: no cover - torch optional
    pass

try:
    from reachability_gen.models.geometric import (
        DEFAULT_RESIDUAL_ALPHA,
        GeometricRecurrent,
        drift_from_trajectory,
        mean_z_norms_from_trajectory,
        trajectory_finite_nonzero,
    )

    __all__ += [
        "DEFAULT_RESIDUAL_ALPHA",
        "GeometricRecurrent",
        "drift_from_trajectory",
        "mean_z_norms_from_trajectory",
        "trajectory_finite_nonzero",
    ]
except ImportError:  # pragma: no cover - torch optional
    pass

try:
    from reachability_gen.models.euclidean_loop import EuclideanLoop

    __all__ += ["EuclideanLoop"]
except ImportError:  # pragma: no cover - torch optional
    pass


try:
    from reachability_gen.models.fractal_core import (
        DEFAULT_DISCONNECT_LEAK_ATOL,
        DEFAULT_HALT_EPS,
        DISCRETE_T_VALUES,
        MANDELBROT_ANALOGY_NOTE,
        FractalCore,
        MaskedTransformerBlock,
        build_adjacency_attn_mask,
        build_node_slot_batch,
        _verify_param_parity,
    )

    __all__ += [
        "DEFAULT_DISCONNECT_LEAK_ATOL",
        "DEFAULT_HALT_EPS",
        "DISCRETE_T_VALUES",
        "MANDELBROT_ANALOGY_NOTE",
        "FractalCore",
        "MaskedTransformerBlock",
        "build_adjacency_attn_mask",
        "build_node_slot_batch",
        "_verify_param_parity",
    ]
except ImportError:  # pragma: no cover - torch optional
    pass


try:
    from reachability_gen.models.sheaf_infer_core import (
        DEFAULT_ABSENT_BIAS,
        DEFAULT_EDGE_RECON_WEIGHT,
        DEFAULT_GATE_THETA,
        SheafDiffusionPhi,
        SheafInferCore,
        build_sheaf_batch,
        ste_hard_gate,
    )

    __all__ += [
        "DEFAULT_ABSENT_BIAS",
        "DEFAULT_EDGE_RECON_WEIGHT",
        "DEFAULT_GATE_THETA",
        "SheafDiffusionPhi",
        "SheafInferCore",
        "build_sheaf_batch",
        "ste_hard_gate",
    ]
except ImportError:  # pragma: no cover - torch optional
    pass

