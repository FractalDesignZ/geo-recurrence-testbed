"""Real torch geometric recurrent arm: weight-tied Phi(z_t, c; tau_t).

T cycles of one shared transformer block with cycle (tau) embeddings.
Context ``c`` is the tokenized edge-list encoding (same tokenizer as FF).
``forward(..., return_trajectory=True)`` returns ``(logits [B,2], list[z_t])``
for drift telemetry (delta_t = ||z_{t+1}-z_t||_2).

RESEARCH / MEASURE plumbing only — no science OPEN claims.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from reachability_gen.models.feedforward import TransformerBlock


class GeometricRecurrent(nn.Module):
    """Weight-tied Phi reused T times → binary reachability logits ``[B, 2]``.

    Parameters
    ----------
    vocab_size :
        Embedding table size (from :func:`tokenize.build_vocab`).
    d :
        Model width.
    T :
        Number of weight-tied recurrence cycles (fixed to 6 for ID-hop gate).
    n_heads :
        Attention heads (must divide ``d``).
    max_len :
        Maximum positional embedding length.
    mlp_expansion :
        MLP width multiplier (default 4, matches FLOP schematic).
    pad_id :
        Padding token id (excluded from mean-pool).
    use_tau :
        If True, add a learned cycle embedding ``tau_t`` each step.
    max_T :
        Capacity of the tau embedding table (default ``max(T, 16)``).
    """

    def __init__(
        self,
        vocab_size: int,
        d: int = 64,
        T: int = 6,
        *,
        n_heads: int = 4,
        max_len: int = 256,
        mlp_expansion: int = 4,
        pad_id: int = 0,
        dropout: float = 0.0,
        use_tau: bool = True,
        max_T: Optional[int] = None,
    ) -> None:
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        self.vocab_size = int(vocab_size)
        self.d = int(d)
        self.T = int(T)
        self.n_heads = int(n_heads)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.mlp_expansion = int(mlp_expansion)
        self.use_tau = bool(use_tau)
        self.max_T = int(max_T) if max_T is not None else max(self.T, 16)
        if self.T > self.max_T:
            raise ValueError(f"T={self.T} exceeds max_T={self.max_T}")

        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d)
        # Weight-tied Phi: one shared transformer block.
        self.phi = TransformerBlock(
            d, n_heads, mlp_expansion=mlp_expansion, dropout=dropout
        )
        if self.use_tau:
            self.tau_emb = nn.Embedding(self.max_T, d)
        else:
            self.tau_emb = None  # type: ignore[assignment]
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def _pool(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Masked mean-pool → ``[B, d]`` latent used for drift / head."""
        if attention_mask is not None:
            mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (x * mask).sum(dim=1) / denom
        return x.mean(dim=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_trajectory: bool = False,
        T: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """Forward pass.

        Parameters
        ----------
        token_ids :
            ``LongTensor [B, M]`` (tokenized edge-list + QUERY).
        attention_mask :
            Optional ``[B, M]`` with 1 = real token, 0 = pad.
        return_trajectory :
            If True, also return ``[z_0, z_1, ..., z_T]`` pooled latents
            (length ``T+1``) for drift ``δ_t = ||z_{t+1}-z_t||_2``.
        T :
            Override cycle count (default ``self.T``).

        Returns
        -------
        logits, trajectory
            ``logits`` has shape ``[B, 2]``. ``trajectory`` is a list of
            ``[B, d]`` tensors when ``return_trajectory`` else ``None``.
        """
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B, M], got {tuple(token_ids.shape)}")
        bsz, mlen = token_ids.shape
        if mlen > self.max_len:
            raise ValueError(
                f"sequence length {mlen} exceeds max_len={self.max_len}"
            )
        cycles = int(self.T if T is None else T)
        if cycles < 1:
            raise ValueError(f"T must be >= 1, got {cycles}")
        if self.use_tau and cycles > self.max_T:
            raise ValueError(f"T={cycles} exceeds tau table max_T={self.max_T}")

        device = token_ids.device
        pos = torch.arange(mlen, device=device).unsqueeze(0).expand(bsz, -1)
        # Context c from tokenized encoding; z_0 := c.
        c = self.tok_emb(token_ids) + self.pos_emb(pos)
        z_seq = c

        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        trajectory: list[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(self._pool(z_seq, attention_mask))

        for t in range(cycles):
            if self.use_tau and self.tau_emb is not None:
                tau_t = self.tau_emb(
                    torch.tensor(t, device=device, dtype=torch.long)
                )  # [d]
                h = z_seq + tau_t.view(1, 1, -1)
            else:
                h = z_seq
            # Phi(z_t, c; tau_t): weight-tied block; c is baked into z_0 / residual path.
            z_seq = self.phi(h, key_padding_mask=key_padding_mask)
            if return_trajectory:
                trajectory.append(self._pool(z_seq, attention_mask))

        x = self.ln_f(z_seq)
        pooled = self._pool(x, attention_mask)
        logits = self.head(pooled)  # [B, 2]
        if return_trajectory:
            return logits, trajectory
        return logits, None

    def non_embedding_param_count(self) -> int:
        """Count params excluding tok/pos embeddings (tau counted as non-emb)."""
        emb_ids = {id(p) for p in self.tok_emb.parameters()}
        emb_ids |= {id(p) for p in self.pos_emb.parameters()}
        total = 0
        for p in self.parameters():
            if id(p) not in emb_ids:
                total += p.numel()
        return int(total)


def drift_from_trajectory(
    trajectory: list[torch.Tensor],
) -> list[float]:
    """Compute δ_t = mean_batch ||z_{t+1}-z_t||_2 for consecutive latents.

    Returns a Python list of length ``len(trajectory)-1``.
    """
    if len(trajectory) < 2:
        return []
    drifts: list[float] = []
    for t in range(len(trajectory) - 1):
        z0 = trajectory[t]
        z1 = trajectory[t + 1]
        # Per-example L2, then mean over batch.
        delta = (z1 - z0).float().reshape(z0.shape[0], -1)
        norms = torch.linalg.vector_norm(delta, ord=2, dim=-1)
        drifts.append(float(norms.mean().item()))
    return drifts


def trajectory_finite_nonzero(drifts: list[float]) -> tuple[bool, str]:
    """Gate helper: fail on NaN/Inf or all-zero drifts."""
    if not drifts:
        return False, "empty drift_trajectory"
    for i, d in enumerate(drifts):
        if d != d:  # NaN
            return False, f"NaN at drift[{i}]"
        if d == float("inf") or d == float("-inf"):
            return False, f"Inf at drift[{i}]"
    if all(abs(d) < 1e-12 for d in drifts):
        return False, "all-zero drifts"
    return True, "ok"


__all__ = [
    "GeometricRecurrent",
    "drift_from_trajectory",
    "trajectory_finite_nonzero",
]
