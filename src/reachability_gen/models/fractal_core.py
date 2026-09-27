"""FractalCore: adjacency-masked recurrent reachability arm (MEASURE plumbing).

Architecture (documented formulas; Mandelbrot analogy is **aspirational** only —
evidence comes from metrics artifacts, never from the metaphor)::

    Mask construction (node-slot sequence)
    --------------------------------------
    Positions ``0 .. n-1`` are graph nodes (batch-padded to ``M = max_n``).
    Additive attention mask ``A`` (float) for directed message-passing::

        A[b, i, j] = 0      if j → i is an edge, or i == j (self),
                            or (optional) either index is a documented
                            special slot (none by default in v0)
        A[b, i, j] = -inf   otherwise

    Padded node slots are excluded via ``key_padding_mask`` (True = pad).
    The locked edge-list string encoding cannot recover adjacency at token
    indices (tokens are ``N``, ``EDGES``, ``u``, ``,``, ``v``, ``QUERY``, …).
    FractalCore therefore uses :func:`build_node_slot_batch` to parse
    ``encoding`` → ``(n, edges, s, t)`` and build node-slot tensors + ``A``.
    Masks are never fabricated from token co-occurrence.

    Boundary injection (+c)
    -----------------------
    Invariant prompt embedding ``c`` from query ``(s, t)`` (and optional
    residual mix α)::

        h_t = z_t + τ_t                         # optional cycle embed
        φ_t = Φ(h_t; Mask=A)                    # masked Pre-LN block
        z_{t+1} = RMSNorm(z_t + α · (φ_t - z_t) + c)

    with α = 0.5 by default (same residual step as Geo bound30).

    Adaptive halting (ACT-style continuous merit gate)
    --------------------------------------------------
    After each cycle, from the **target-node** state ``z_t[t]``::

        u_t = σ(W_halt · z_t[target] + b) ∈ (0, 1)
        p_1 = u_1
        p_k = u_k · (1 - Σ_{j<k} p_j)     for k < N
        N = min{k : Σ_{j≤k} u_j ≥ 1-ε} ∪ {T_max}
        p_N = R_N = 1 - Σ_{j<N} p_j       (remainder)

    Soft train/eval output::

        logits = Σ_k p_k · Head(z_k[target])

    Halt diagnostics (mean halt step, mean ponder, u-trajectory) are exposed
    in metrics; they are MEASURE plumbing only.

RESEARCH / MEASURE — ``science_open=false`` always. No science OPEN claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reachability_gen.encode import parse_instance
from reachability_gen.models.feedforward import TransformerBlock
from reachability_gen.models.geometric import DEFAULT_RESIDUAL_ALPHA
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

# ACT remainder threshold (Graves-style).
DEFAULT_HALT_EPS: float = 0.01
# Aspirational metaphor only — never used as evidence.
MANDELBROT_ANALOGY_NOTE: str = (
    "Mandelbrot-style boundary re-injection is an aspirational analogy for "
    "the +c residual; scientific claims require metrics artifacts only."
)


def build_adjacency_attn_mask(
    n: int,
    edges: Sequence[tuple[int, int]],
    *,
    max_n: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build additive attn mask ``[max_n, max_n]``: 0 allow, -inf deny.

    Allows self and incoming edges ``j → i`` (key j → query i). Positions
    ``n .. max_n-1`` are left as -inf (pad); callers should also pass
    ``key_padding_mask``.
    """
    if max_n < n:
        raise ValueError(f"max_n={max_n} < n={n}")
    neg = torch.finfo(dtype).min / 2  # large negative, stable for softmax
    A = torch.full((max_n, max_n), neg, device=device, dtype=dtype)
    if n > 0:
        idx = torch.arange(n, device=device)
        A[idx, idx] = 0.0  # self
        for u, v in edges:
            ui, vi = int(u), int(v)
            if not (0 <= ui < n and 0 <= vi < n):
                raise ValueError(f"edge ({ui},{vi}) out of range for n={n}")
            # key=u (source) → query=v (target): message along directed edge
            A[vi, ui] = 0.0
    return A


def build_node_slot_batch(
    examples: Sequence[Mapping[str, Any]],
    *,
    max_n: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    """Parse encodings → node-slot tensors for FractalCore.

    Returns dict with:
      - ``node_ids``: LongTensor ``[B, M]`` (node index or 0 for pad)
      - ``node_mask``: LongTensor ``[B, M]`` (1 = real node, 0 = pad)
      - ``attn_mask``: FloatTensor ``[B, M, M]`` additive (0 / -inf)
      - ``s_idx`` / ``t_idx``: LongTensor ``[B]``
      - ``labels``: LongTensor ``[B]`` when ``y`` present
      - ``n_nodes``: LongTensor ``[B]``
    """
    parsed: list[tuple[int, list[tuple[int, int]], int, int]] = []
    labels_list: list[int] = []
    has_y = True
    for ex in examples:
        enc = ex.get("encoding")
        if enc:
            n, edges, s, t = parse_instance(str(enc))
        else:
            n = int(ex["n"])
            edges = [(int(a), int(b)) for a, b in ex["edges"]]
            s, t = int(ex["s"]), int(ex["t"])
        n = int(ex.get("n", n))
        s = int(ex.get("s", s))
        t = int(ex.get("t", t))
        parsed.append((n, edges, s, t))
        if "y" in ex:
            labels_list.append(int(ex["y"]))
        else:
            has_y = False

    M = int(max_n) if max_n is not None else max(n for n, _, _, _ in parsed)
    if M < 1:
        M = 1
    B = len(parsed)
    node_ids = torch.zeros(B, M, dtype=torch.long, device=device)
    node_mask = torch.zeros(B, M, dtype=torch.long, device=device)
    attn = torch.zeros(B, M, M, dtype=torch.float32, device=device)
    s_idx = torch.zeros(B, dtype=torch.long, device=device)
    t_idx = torch.zeros(B, dtype=torch.long, device=device)
    n_nodes = torch.zeros(B, dtype=torch.long, device=device)

    for b, (n, edges, s, t) in enumerate(parsed):
        if n > M:
            raise ValueError(f"n={n} exceeds max_n={M}")
        if not (0 <= s < n and 0 <= t < n):
            raise ValueError(f"query ({s},{t}) out of range for n={n}")
        node_ids[b, :n] = torch.arange(n, device=device)
        node_mask[b, :n] = 1
        attn[b] = build_adjacency_attn_mask(
            n, edges, max_n=M, device=device, dtype=torch.float32
        )
        s_idx[b] = s
        t_idx[b] = t
        n_nodes[b] = n

    out: dict[str, torch.Tensor] = {
        "node_ids": node_ids,
        "node_mask": node_mask,
        "attn_mask": attn,
        "s_idx": s_idx,
        "t_idx": t_idx,
        "n_nodes": n_nodes,
    }
    if has_y and len(labels_list) == B:
        out["labels"] = torch.tensor(labels_list, dtype=torch.long, device=device)
    return out


class MaskedTransformerBlock(nn.Module):
    """Pre-LN block with additive adjacency attn_mask (+ optional key pad)."""

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        *,
        mlp_expansion: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d)
        hidden = mlp_expansion * d
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        self.dropout = nn.Dropout(dropout)
        self.n_heads = int(n_heads)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``attn_mask``: ``[M,M]`` or ``[B,M,M]`` additive; broadcast to heads."""
        h = self.ln1(x)
        mask = attn_mask
        if mask is not None and mask.dim() == 3:
            # MHA expects [B*heads, L, S] or [L, S]. Expand per head.
            bsz, mlen, _ = mask.shape
            mask = (
                mask.unsqueeze(1)
                .expand(bsz, self.n_heads, mlen, mlen)
                .reshape(bsz * self.n_heads, mlen, mlen)
            )
        a, _ = self.attn(
            h,
            h,
            h,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(a)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class FractalCore(nn.Module):
    """Weight-tied masked recurrent core with +c boundary and ACT halt.

    Parameters
    ----------
    d, T, n_heads, mlp_expansion :
        Width / max cycles / heads / MLP expansion (default mlp×10 for
        parity with bound30 recurrent arms vs FF ~121218).
    max_nodes :
        Node-slot capacity (pad length). Must cover dataset ``n``.
    halt_eps :
        ACT remainder threshold ε.
    residual_alpha :
        Outer residual step α (default 0.5).
    """

    def __init__(
        self,
        d: int = 64,
        T: int = 6,
        *,
        n_heads: int = 4,
        mlp_expansion: int = 10,
        max_nodes: int = DEFAULT_MAX_NODE_ID,
        max_T: Optional[int] = None,
        dropout: float = 0.0,
        halt_eps: float = DEFAULT_HALT_EPS,
        residual_alpha: float = DEFAULT_RESIDUAL_ALPHA,
        use_tau: bool = True,
        apply_cycle_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")
        self.d = int(d)
        self.T = int(T)
        self.n_heads = int(n_heads)
        self.mlp_expansion = int(mlp_expansion)
        self.max_nodes = int(max_nodes)
        self.max_T = int(max_T) if max_T is not None else max(self.T, 16)
        self.halt_eps = float(halt_eps)
        self.residual_alpha = float(residual_alpha)
        self.use_tau = bool(use_tau)
        self.apply_cycle_rmsnorm = bool(apply_cycle_rmsnorm)
        if self.T > self.max_T:
            raise ValueError(f"T={self.T} exceeds max_T={self.max_T}")

        # Node identity + positional slots (node-centric, not edge-list tokens).
        self.node_emb = nn.Embedding(self.max_nodes, d)
        self.pos_emb = nn.Embedding(self.max_nodes, d)
        self.src_emb = nn.Parameter(torch.zeros(d))
        self.tgt_emb = nn.Parameter(torch.zeros(d))
        # Boundary prompt c from (s, t) node embeddings.
        self.c_proj = nn.Linear(2 * d, d)
        self.phi = MaskedTransformerBlock(
            d, n_heads, mlp_expansion=mlp_expansion, dropout=dropout
        )
        if self.use_tau:
            self.tau_emb = nn.Embedding(self.max_T, d)
        else:
            self.tau_emb = None  # type: ignore[assignment]
        self.cycle_rmsnorm = nn.RMSNorm(d) if apply_cycle_rmsnorm else None
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        self.halt_gate = nn.Linear(d, 1)
        self._init_specials()

    def _init_specials(self) -> None:
        nn.init.normal_(self.src_emb, std=0.02)
        nn.init.normal_(self.tgt_emb, std=0.02)

    def _gather_node(
        self, z: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """Gather per-batch node states: ``z`` [B,M,d], ``idx`` [B] → [B,d]."""
        bsz = z.shape[0]
        return z[torch.arange(bsz, device=z.device), idx]

    def _build_c(
        self,
        node_ids: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Invariant prompt ``c`` [B, 1, d] from (s, t) embeddings."""
        s_e = self.node_emb(s_idx.clamp(0, self.max_nodes - 1))
        t_e = self.node_emb(t_idx.clamp(0, self.max_nodes - 1))
        c = self.c_proj(torch.cat([s_e, t_e], dim=-1))  # [B, d]
        return c.unsqueeze(1)

    def _initial_state(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z_0 and broadcast prompt c."""
        bsz, mlen = node_ids.shape
        if mlen > self.max_nodes:
            raise ValueError(
                f"sequence length {mlen} exceeds max_nodes={self.max_nodes}"
            )
        device = node_ids.device
        pos = torch.arange(mlen, device=device).unsqueeze(0).expand(bsz, -1)
        # Clamp pad slots to 0 for embedding lookup; mask zeros them out later.
        ids_clamped = node_ids.clamp(0, self.max_nodes - 1)
        z = self.node_emb(ids_clamped) + self.pos_emb(pos)
        # Mark source / target nodes (additive role features).
        batch_ix = torch.arange(bsz, device=device)
        z = z.clone()
        z[batch_ix, s_idx] = z[batch_ix, s_idx] + self.src_emb
        z[batch_ix, t_idx] = z[batch_ix, t_idx] + self.tgt_emb
        c = self._build_c(node_ids, s_idx, t_idx)
        if self.cycle_rmsnorm is not None:
            z = self.cycle_rmsnorm(z)
        # Zero pad slots for cleanliness.
        z = z * node_mask.unsqueeze(-1).to(dtype=z.dtype)
        return z, c

    def _cycle_update(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        device = z.device
        if self.use_tau and self.tau_emb is not None:
            tau_t = self.tau_emb(
                torch.tensor(t, device=device, dtype=torch.long)
            ).view(1, 1, -1)
            h = z + tau_t
        else:
            h = z
        phi_out = self.phi(
            h, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        z_new = z + self.residual_alpha * (phi_out - z) + c
        if self.cycle_rmsnorm is not None:
            z_new = self.cycle_rmsnorm(z_new)
        return z_new

    def forward(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        *,
        return_trajectory: bool = False,
        return_halt: bool = False,
        T: Optional[int] = None,
        adaptive_halt: bool = True,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[dict[str, Any]]]:
        """Forward with optional ACT soft halt.

        Returns
        -------
        logits, trajectory, halt_info
            ``logits``: ``[B, 2]``. Trajectory is list of target-pooled
            ``[B, d]`` states when requested. ``halt_info`` holds diagnostics.
        """
        if node_ids.dim() != 2:
            raise ValueError(f"node_ids must be [B, M], got {tuple(node_ids.shape)}")
        cycles = int(self.T if T is None else T)
        if cycles < 1:
            raise ValueError(f"T must be >= 1, got {cycles}")
        if self.use_tau and cycles > self.max_T:
            raise ValueError(f"T={cycles} exceeds tau table max_T={self.max_T}")

        z, c = self._initial_state(node_ids, node_mask, s_idx, t_idx)
        # Fold pad into additive attn_mask (float) so MHA sees one mask type.
        key_padding_mask = node_mask == 0  # [B, M] True=pad
        neg = torch.finfo(attn_mask.dtype).min / 2
        pad_cols = key_padding_mask.unsqueeze(1).expand_as(attn_mask)
        pad_rows = key_padding_mask.unsqueeze(2).expand_as(attn_mask)
        attn_mask = attn_mask.masked_fill(pad_cols | pad_rows, neg)
        key_padding_mask_arg = None  # pad already in attn_mask

        trajectory: list[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(self._gather_node(z, t_idx))

        # Soft ACT accumulators
        bsz = node_ids.shape[0]
        device = node_ids.device
        halt_still = torch.ones(bsz, device=device, dtype=z.dtype)
        probs: list[torch.Tensor] = []
        u_list: list[torch.Tensor] = []
        logits_acc = torch.zeros(bsz, 2, device=device, dtype=z.dtype)
        cum_u = torch.zeros(bsz, device=device, dtype=z.dtype)
        halt_step = torch.full(
            (bsz,), float(cycles), device=device, dtype=z.dtype
        )

        for t in range(cycles):
            z = self._cycle_update(
                z,
                c,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask_arg,
                t=t,
            )
            z = z * node_mask.unsqueeze(-1).to(dtype=z.dtype)
            tgt = self._gather_node(self.ln_f(z), t_idx)  # [B, d]
            step_logits = self.head(tgt)  # [B, 2]
            u_t = torch.sigmoid(self.halt_gate(tgt)).squeeze(-1)  # [B]
            u_list.append(u_t)

            if adaptive_halt:
                # p_t = u_t * remaining mass (halt_still)
                p_t = u_t * halt_still
                # If last step, dump remainder into p_T
                is_last = t == cycles - 1
                if is_last:
                    p_t = halt_still
                probs.append(p_t)
                logits_acc = logits_acc + p_t.unsqueeze(-1) * step_logits
                cum_u = cum_u + u_t
                # Mark first time cum_u >= 1-eps (for diagnostics).
                newly = (cum_u >= 1.0 - self.halt_eps) & (halt_step >= float(cycles))
                halt_step = torch.where(
                    newly, torch.full_like(halt_step, float(t + 1)), halt_step
                )
                if not is_last:
                    halt_still = halt_still - p_t
                    # Numerical floor
                    halt_still = halt_still.clamp(min=0.0)
            else:
                # Fixed unroll: final step only (like Geo).
                logits_acc = step_logits
                probs.append(torch.ones(bsz, device=device, dtype=z.dtype))

            if return_trajectory:
                trajectory.append(tgt)

        if not adaptive_halt:
            # already set to last step logits
            pass

        halt_info: Optional[dict[str, Any]] = None
        if return_halt or adaptive_halt:
            u_stack = torch.stack(u_list, dim=1) if u_list else torch.zeros(bsz, 0)
            p_stack = torch.stack(probs, dim=1) if probs else torch.zeros(bsz, 0)
            ponder = (p_stack * torch.arange(1, p_stack.shape[1] + 1, device=device).float()).sum(dim=1) if p_stack.numel() else torch.zeros(bsz)
            halt_info = {
                "u_trajectory": u_stack.detach(),
                "p_trajectory": p_stack.detach(),
                "halt_step": halt_step.detach(),
                "mean_halt_step": float(halt_step.mean().item()) if bsz else float("nan"),
                "mean_ponder": float(ponder.mean().item()) if bsz else float("nan"),
                "mean_u_final": float(u_stack[:, -1].mean().item()) if u_stack.numel() else float("nan"),
                "halt_eps": self.halt_eps,
                "adaptive_halt": bool(adaptive_halt),
                "T": cycles,
                "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
            }

        if return_trajectory and return_halt:
            return logits_acc, trajectory, halt_info
        if return_trajectory:
            return logits_acc, trajectory, halt_info
        if return_halt:
            return logits_acc, None, halt_info
        return logits_acc, None, halt_info

    def forward_from_examples(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        max_n: Optional[int] = None,
        return_trajectory: bool = False,
        return_halt: bool = False,
        T: Optional[int] = None,
        adaptive_halt: bool = True,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[dict[str, Any]]]:
        """Convenience: encode examples then forward."""
        batch = build_node_slot_batch(
            examples, max_n=max_n or self.max_nodes, device=next(self.parameters()).device
        )
        return self.forward(
            batch["node_ids"],
            batch["node_mask"],
            batch["attn_mask"],
            batch["s_idx"],
            batch["t_idx"],
            return_trajectory=return_trajectory,
            return_halt=return_halt,
            T=T,
            adaptive_halt=adaptive_halt,
        )

    def param_count(self) -> int:
        """Total trainable params (matches rematch accounting vs FF ~121218)."""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def non_embedding_param_count(self) -> int:
        """Exclude node/pos embeddings (tau + halt + c_proj counted)."""
        emb_ids = {id(p) for p in self.node_emb.parameters()}
        emb_ids |= {id(p) for p in self.pos_emb.parameters()}
        total = 0
        for p in self.parameters():
            if id(p) not in emb_ids:
                total += p.numel()
        return int(total)


def _verify_param_parity(
    fractal_count: int,
    *,
    ff_baseline: int = 121_218,
    tol: float = 0.05,
) -> dict[str, Any]:
    """Hard check: FractalCore total params within ±tol of FF baseline.

    Always returns ``science_open=False``. Raises ``AssertionError`` on miss
    when used as a gate; callers may also inspect the dict.
    """
    lo = int(ff_baseline * (1.0 - tol))
    hi = int(round(ff_baseline * (1.0 + tol)))
    ok = lo <= int(fractal_count) <= hi
    ratio = fractal_count / ff_baseline if ff_baseline else float("nan")
    section = {
        "ff_baseline_params": int(ff_baseline),
        "fractal_param_count": int(fractal_count),
        "window": [lo, hi],
        "tolerance": tol,
        "fractal_over_ff_ratio": ratio,
        "within_5pct": ok,
        "science_open": False,
        "notes": (
            "Param parity is MEASURE accounting hygiene — never stamps science OPEN."
        ),
    }
    if not ok:
        raise AssertionError(
            f"FractalCore params {fractal_count} outside ±{tol:.0%} of FF "
            f"{ff_baseline} (window [{lo}, {hi}])"
        )
    return section


__all__ = [
    "DEFAULT_HALT_EPS",
    "MANDELBROT_ANALOGY_NOTE",
    "FractalCore",
    "MaskedTransformerBlock",
    "build_adjacency_attn_mask",
    "build_node_slot_batch",
    "_verify_param_parity",
]
