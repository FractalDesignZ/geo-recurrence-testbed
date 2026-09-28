"""AdamW + CE (+ aux edge recon) trainer for SheafInferCore (MEASURE).

Matches bound30 recurrent defaults: lr=1.5e-3, grad_clip=2.5.
Gold edges used ONLY for aux BCE reconstruction — never as eval A oracle.
No science OPEN claims.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from reachability_gen.models.geometric import drift_from_trajectory
from reachability_gen.models.sheaf_infer_core import (
    DEFAULT_EDGE_RECON_WEIGHT,
    SheafInferCore,
)


class SheafTrainer:
    """Single-step AdamW trainer: CE reachability + optional edge recon aux."""

    def __init__(
        self,
        model: SheafInferCore,
        *,
        lr: float = 1.5e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 2.5,
        device: Optional[torch.device] = None,
        edge_recon_weight: float = DEFAULT_EDGE_RECON_WEIGHT,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.grad_clip = float(grad_clip)
        self.edge_recon_weight = float(edge_recon_weight)
        self.opt = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.last_pre_clip_grad_norm: float = 0.0
        self.last_halt_info: dict[str, Any] = {}
        self.last_edge_recon_acc: float = float("nan")
        self.last_ce: float = float("nan")
        self.last_recon: float = float("nan")

    def train_step(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        labels: torch.Tensor,
        gold_adj: Optional[torch.Tensor] = None,
        *,
        T: Optional[int] = None,
    ) -> tuple[float, float]:
        """One AdamW step. Returns ``(total_loss, accuracy)``."""
        self.model.train()
        node_ids = node_ids.to(self.device)
        node_mask = node_mask.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_mask = edge_mask.to(self.device)
        s_idx = s_idx.to(self.device)
        t_idx = t_idx.to(self.device)
        labels = labels.to(self.device)
        if gold_adj is not None:
            gold_adj = gold_adj.to(self.device)

        self.opt.zero_grad(set_to_none=True)
        logits, _, halt = self.model(
            node_ids,
            node_mask,
            edge_index,
            edge_mask,
            s_idx,
            t_idx,
            return_halt=True,
            return_edge_logits=True,
            T=T,
        )
        self.last_halt_info = halt or {}
        ce = self.loss_fn(logits, labels)
        recon = torch.zeros((), device=self.device)
        if gold_adj is not None and self.edge_recon_weight > 0:
            assert halt is not None and "edge_logits" in halt
            recon = self.model.edge_recon_loss(
                halt["edge_logits"], gold_adj, node_mask
            )
            with torch.no_grad():
                self.last_edge_recon_acc = self.model.edge_recon_accuracy(
                    halt["edge_logits"], gold_adj, node_mask
                )["offdiag_acc"]
        loss = ce + self.edge_recon_weight * recon
        self.last_ce = float(ce.item())
        self.last_recon = float(recon.item()) if recon.numel() else 0.0
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
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        labels: torch.Tensor,
        gold_adj: Optional[torch.Tensor] = None,
        *,
        return_drift: bool = False,
        T: Optional[int] = None,
    ) -> tuple[float, float] | tuple[float, float, list[float], dict[str, Any]]:
        self.model.eval()
        node_ids = node_ids.to(self.device)
        node_mask = node_mask.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_mask = edge_mask.to(self.device)
        s_idx = s_idx.to(self.device)
        t_idx = t_idx.to(self.device)
        labels = labels.to(self.device)
        logits, traj, halt = self.model(
            node_ids,
            node_mask,
            edge_index,
            edge_mask,
            s_idx,
            t_idx,
            return_trajectory=return_drift,
            return_halt=True,
            return_edge_logits=gold_adj is not None,
            T=T,
        )
        self.last_halt_info = halt or {}
        # Eval metric: CE only (reachability). Aux recon is train-only signal.
        loss = self.loss_fn(logits, labels)
        if gold_adj is not None and halt is not None and "edge_logits" in halt:
            self.last_edge_recon_acc = self.model.edge_recon_accuracy(
                halt["edge_logits"], gold_adj.to(self.device), node_mask
            )["offdiag_acc"]
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        if return_drift:
            drifts = drift_from_trajectory(list(traj or []))
            return float(loss.item()), float(acc.item()), drifts, halt or {}
        return float(loss.item()), float(acc.item())


__all__ = ["SheafTrainer"]
