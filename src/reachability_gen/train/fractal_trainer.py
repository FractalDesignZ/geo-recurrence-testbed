"""AdamW + CE trainer for FractalCore (MEASURE plumbing).

Matches bound30 recurrent defaults where sensible: lr=1.5e-3, grad_clip=2.5.
No science OPEN claims.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from reachability_gen.models.fractal_core import FractalCore
from reachability_gen.models.geometric import drift_from_trajectory


class FractalTrainer:
    """Single-step AdamW trainer with CE loss and configurable grad clip."""

    def __init__(
        self,
        model: FractalCore,
        *,
        lr: float = 1.5e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 2.5,
        device: Optional[torch.device] = None,
        adaptive_halt: bool = True,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.grad_clip = float(grad_clip)
        self.adaptive_halt = bool(adaptive_halt)
        self.opt = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.last_pre_clip_grad_norm: float = 0.0
        self.last_halt_info: dict[str, Any] = {}

    def train_step(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        labels: torch.Tensor,
        *,
        T: Optional[int] = None,
    ) -> tuple[float, float]:
        """One AdamW step. Returns ``(loss, accuracy)``."""
        self.model.train()
        node_ids = node_ids.to(self.device)
        node_mask = node_mask.to(self.device)
        attn_mask = attn_mask.to(self.device)
        s_idx = s_idx.to(self.device)
        t_idx = t_idx.to(self.device)
        labels = labels.to(self.device)

        self.opt.zero_grad(set_to_none=True)
        logits, _, halt = self.model(
            node_ids,
            node_mask,
            attn_mask,
            s_idx,
            t_idx,
            return_halt=True,
            T=T,
            adaptive_halt=self.adaptive_halt,
        )
        self.last_halt_info = halt or {}
        loss = self.loss_fn(logits, labels)
        loss.backward()
        pre_clip = clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.last_pre_clip_grad_norm = float(
            pre_clip.item() if hasattr(pre_clip, "item") else pre_clip
        )
        self.opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def eval_step(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        labels: torch.Tensor,
        *,
        return_drift: bool = False,
        T: Optional[int] = None,
        adaptive_halt: Optional[bool] = None,
    ) -> tuple[float, float] | tuple[float, float, list[float], dict[str, Any]]:
        """Eval CE + accuracy; optionally drift + halt diagnostics."""
        self.model.eval()
        node_ids = node_ids.to(self.device)
        node_mask = node_mask.to(self.device)
        attn_mask = attn_mask.to(self.device)
        s_idx = s_idx.to(self.device)
        t_idx = t_idx.to(self.device)
        labels = labels.to(self.device)
        ah = self.adaptive_halt if adaptive_halt is None else bool(adaptive_halt)
        logits, traj, halt = self.model(
            node_ids,
            node_mask,
            attn_mask,
            s_idx,
            t_idx,
            return_trajectory=return_drift,
            return_halt=True,
            T=T,
            adaptive_halt=ah,
        )
        self.last_halt_info = halt or {}
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        if return_drift:
            drifts = drift_from_trajectory(list(traj or []))
            return float(loss.item()), float(acc.item()), drifts, halt or {}
        return float(loss.item()), float(acc.item())


__all__ = ["FractalTrainer"]
