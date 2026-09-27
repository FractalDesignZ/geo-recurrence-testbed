"""FractalCore: adjacency-masked recurrent reachability arm (MEASURE plumbing).

CYCLE_STALK_LOCALIZATION architecture
-------------------------------------
Local stalk / probe potentials — **no** global ``(s, t)`` broadcast::

    potential[i] = stalk_proj(e_s)   if i == s   (source stalk)
    potential[i] = probe_proj(e_t)   if i == t   (target probe)
    potential[i] = 0                 otherwise   (intermediates)

Cycle update (discrete fixed-T unroll only — soft ACT removed)::

    h_t = z_t + τ_t                         # optional cycle embed
    φ_t = Φ(h_t; Mask=A)                    # masked Pre-LN block
    z_{t+1} = RMSNorm(z_t + α · (φ_t - z_t) + potential)

Readout from target slot ``z_T[t]`` after exactly ``T`` cycles.
Protocol discrete depths: ``T ∈ {6, 8, 12, 16}``.

Mask construction (unchanged)
-----------------------------
Positions ``0 .. n-1`` are graph nodes (batch-padded to ``M = max_n``).
Additive attention mask ``A`` (float) for directed message-passing::

    A[b, i, j] = 0      if j → i is an edge, or i == j (self)
    A[b, i, j] = -inf   otherwise

Masks are built from parsed graph edges only — never from token co-occurrence.

RESEARCH / MEASURE — ``science_open=false`` always. No science OPEN claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from reachability_gen.encode import parse_instance
from reachability_gen.models.geometric import DEFAULT_RESIDUAL_ALPHA
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

# Discrete protocol depths (Gate0/1). Forward still accepts any T>=1 for tests.
DISCRETE_T_VALUES: tuple[int, ...] = (6, 8, 12, 16)

# Aspirational metaphor only — never used as evidence.
MANDELBROT_ANALOGY_NOTE: str = (
    "Mandelbrot-style boundary re-injection is an aspirational analogy for "
    "local stalk/probe potentials; scientific claims require metrics artifacts only."
)

# Default L2 tolerance: stalk-ablation delta at target for disconnected pairs.
DEFAULT_DISCONNECT_LEAK_ATOL: float = 1e-3


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
    """Weight-tied masked recurrent core with **local** stalk/probe potentials.

    Soft ACT removed. Discrete fixed-T unroll only. No global ``(s,t)``
    broadcast across node slots.

    Parameters
    ----------
    d, T, n_heads, mlp_expansion :
        Width / cycles / heads / MLP expansion (default mlp×10 for
        parity with bound30 recurrent arms vs FF ~121218).
    max_nodes :
        Node-slot capacity (pad length). Must cover dataset ``n``.
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
        residual_alpha: float = DEFAULT_RESIDUAL_ALPHA,
        use_tau: bool = True,
        apply_cycle_rmsnorm: bool = True,
        # Legacy kwargs accepted then ignored (stalk localization kill-list).
        halt_eps: float = 0.0,
        adaptive_halt: bool = False,
    ) -> None:
        super().__init__()
        del halt_eps, adaptive_halt  # soft ACT stripped
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
        self.max_T = int(max_T) if max_T is not None else max(self.T, max(DISCRETE_T_VALUES))
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
        # Local potentials: stalk at s, probe at t — NOT broadcast c(s,t).
        self.stalk_proj = nn.Linear(d, d)
        self.probe_proj = nn.Linear(d, d)
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
        self._init_specials()

    def _init_specials(self) -> None:
        nn.init.normal_(self.src_emb, std=0.02)
        nn.init.normal_(self.tgt_emb, std=0.02)
        # Near-identity local potentials so early cycles stay stable.
        nn.init.eye_(self.stalk_proj.weight)
        nn.init.zeros_(self.stalk_proj.bias)
        nn.init.eye_(self.probe_proj.weight)
        nn.init.zeros_(self.probe_proj.bias)

    def _gather_node(
        self, z: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """Gather per-batch node states: ``z`` [B,M,d], ``idx`` [B] → [B,d]."""
        bsz = z.shape[0]
        return z[torch.arange(bsz, device=z.device), idx]

    def _local_potential(
        self,
        node_ids: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        *,
        zero_stalk: bool = False,
        zero_probe: bool = False,
    ) -> torch.Tensor:
        """Build ``[B, M, d]`` potential: stalk@s, probe@t, else 0."""
        bsz, mlen = node_ids.shape
        device = node_ids.device
        dtype = self.stalk_proj.weight.dtype
        pot = torch.zeros(bsz, mlen, self.d, device=device, dtype=dtype)
        batch_ix = torch.arange(bsz, device=device)
        s_ids = s_idx.clamp(0, self.max_nodes - 1)
        t_ids = t_idx.clamp(0, self.max_nodes - 1)
        if not zero_stalk:
            stalk = self.stalk_proj(self.node_emb(s_ids))  # [B, d]
            pot[batch_ix, s_idx] = stalk
        if not zero_probe:
            probe = self.probe_proj(self.node_emb(t_ids))  # [B, d]
            pot[batch_ix, t_idx] = pot[batch_ix, t_idx] + probe
        return pot

    def _initial_state(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        """z_0 with role marks at s/t only (no broadcast c)."""
        bsz, mlen = node_ids.shape
        if mlen > self.max_nodes:
            raise ValueError(
                f"sequence length {mlen} exceeds max_nodes={self.max_nodes}"
            )
        device = node_ids.device
        pos = torch.arange(mlen, device=device).unsqueeze(0).expand(bsz, -1)
        ids_clamped = node_ids.clamp(0, self.max_nodes - 1)
        z = self.node_emb(ids_clamped) + self.pos_emb(pos)
        batch_ix = torch.arange(bsz, device=device)
        z = z.clone()
        z[batch_ix, s_idx] = z[batch_ix, s_idx] + self.src_emb
        z[batch_ix, t_idx] = z[batch_ix, t_idx] + self.tgt_emb
        if self.cycle_rmsnorm is not None:
            z = self.cycle_rmsnorm(z)
        z = z * node_mask.unsqueeze(-1).to(dtype=z.dtype)
        return z

    def _cycle_update(
        self,
        z: torch.Tensor,
        potential: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
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
        z_new = z + self.residual_alpha * (phi_out - z) + potential
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
        adaptive_halt: bool = False,  # ignored; soft ACT stripped
        zero_stalk: bool = False,
        zero_probe: bool = False,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[dict[str, Any]]]:
        """Discrete fixed-T forward (no soft ACT).

        Returns
        -------
        logits, trajectory, info
            ``logits``: ``[B, 2]`` from final target state. Trajectory is
            list of target-pooled ``[B, d]`` when requested.
        """
        del adaptive_halt  # stripped
        if node_ids.dim() != 2:
            raise ValueError(f"node_ids must be [B, M], got {tuple(node_ids.shape)}")
        cycles = int(self.T if T is None else T)
        if cycles < 1:
            raise ValueError(f"T must be >= 1, got {cycles}")
        if self.use_tau and cycles > self.max_T:
            raise ValueError(f"T={cycles} exceeds tau table max_T={self.max_T}")

        z = self._initial_state(node_ids, node_mask, s_idx, t_idx)
        potential = self._local_potential(
            node_ids, s_idx, t_idx, zero_stalk=zero_stalk, zero_probe=zero_probe
        )
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

        final_states: Optional[torch.Tensor] = None
        for t in range(cycles):
            z = self._cycle_update(
                z,
                potential,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask_arg,
                t=t,
            )
            z = z * node_mask.unsqueeze(-1).to(dtype=z.dtype)
            if return_trajectory:
                trajectory.append(self._gather_node(self.ln_f(z), t_idx))

        final_states = z
        tgt = self._gather_node(self.ln_f(z), t_idx)
        logits = self.head(tgt)

        info: Optional[dict[str, Any]] = None
        if return_halt or return_states:
            bsz = node_ids.shape[0]
            info = {
                "adaptive_halt": False,
                "T": cycles,
                "discrete_T": True,
                "mean_halt_step": float(cycles),
                "mean_ponder": float(cycles),  # fixed unroll: ponder = T
                "local_potential": True,
                "broadcast_c": False,
                "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
                "halt_step": torch.full(
                    (bsz,), float(cycles), device=node_ids.device, dtype=z.dtype
                ),
            }
            if return_states and final_states is not None:
                info["final_states"] = final_states.detach()
                info["target_hidden"] = tgt.detach()

        if return_trajectory and return_halt:
            return logits, trajectory, info
        if return_trajectory:
            return logits, trajectory, info
        if return_halt:
            return logits, None, info
        return logits, None, info

    def forward_from_examples(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        max_n: Optional[int] = None,
        return_trajectory: bool = False,
        return_halt: bool = False,
        T: Optional[int] = None,
        adaptive_halt: bool = False,
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
            adaptive_halt=False,
        )

    @torch.no_grad()
    def disconnected_target_leak(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        T_values: Sequence[int] = DISCRETE_T_VALUES,
        max_n: Optional[int] = None,
        atol: float = DEFAULT_DISCONNECT_LEAK_ATOL,
    ) -> dict[str, Any]:
        """Stalk-ablation leak at target for y=0 (disconnected) pairs.

        For each ``T``, compare ``||h_t(full) - h_t(zero_stalk)||_2`` on
        hard-neg examples. Under local potentials + adjacency mask, source
        stalk cannot reach unreachable ``t``, so the delta should stay
        ~0 (within ``atol``) across ``T``.
        """
        negs = [ex for ex in examples if int(ex.get("y", 1)) == 0]
        if not negs:
            return {
                "n_neg": 0,
                "ok": False,
                "reason": "no y=0 examples",
                "atol": atol,
                "science_open": False,
            }
        batch = build_node_slot_batch(
            negs, max_n=max_n or self.max_nodes, device=next(self.parameters()).device
        )
        was_training = self.training
        self.eval()
        by_T: dict[str, Any] = {}
        all_ok = True
        for T in T_values:
            _, _, info_full = self.forward(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                T=int(T),
                return_halt=True,
                return_states=True,
                zero_stalk=False,
            )
            _, _, info_abl = self.forward(
                batch["node_ids"],
                batch["node_mask"],
                batch["attn_mask"],
                batch["s_idx"],
                batch["t_idx"],
                T=int(T),
                return_halt=True,
                return_states=True,
                zero_stalk=True,
            )
            assert info_full is not None and info_abl is not None
            h_full = info_full["target_hidden"]
            h_abl = info_abl["target_hidden"]
            delta = (h_full - h_abl).norm(dim=-1)  # [B]
            mean_l2 = float(delta.mean().item())
            max_l2 = float(delta.max().item())
            ok_T = max_l2 <= atol
            all_ok = all_ok and ok_T
            by_T[str(T)] = {
                "mean_l2": mean_l2,
                "max_l2": max_l2,
                "n": int(delta.numel()),
                "ok": bool(ok_T),
            }
        if was_training:
            self.train()
        return {
            "n_neg": len(negs),
            "atol": atol,
            "T_values": list(T_values),
            "by_T": by_T,
            "ok": bool(all_ok),
            "note": (
                "Stalk-ablation L2 at target for disconnected pairs; "
                f"expect max_l2 <= {atol} across T (local potential, no c broadcast)."
            ),
            "science_open": False,
        }

    def param_count(self) -> int:
        """Total trainable params (matches rematch accounting vs FF ~121218)."""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def non_embedding_param_count(self) -> int:
        """Exclude node/pos embeddings (tau + stalk/probe counted)."""
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


# Back-compat alias (soft ACT removed; value unused).
DEFAULT_HALT_EPS: float = 0.0


__all__ = [
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
