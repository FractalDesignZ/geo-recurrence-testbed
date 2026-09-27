"""AdamW + CE trainer for the Euclidean loop arm (MEASURE plumbing).

Same recipe as GeometricTrainer: AdamW, grad clip 1.0, optional drift
telemetry (trajectory / terminal_drift / perturbation_delta). No τ.
No science OPEN claims.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from reachability_gen.models.euclidean_loop import EuclideanLoop
from reachability_gen.models.geometric import (
    GeometricRecurrent,
    drift_from_trajectory,
)

RecurrentModel = Union[EuclideanLoop, GeometricRecurrent]


class LoopTrainer:
    """Single-step AdamW trainer with CE loss and grad clip 1.0."""

    def __init__(
        self,
        model: RecurrentModel,
        *,
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.grad_clip = float(grad_clip)
        self.opt = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[float, float]:
        """One AdamW step. Returns ``(loss, accuracy)`` as Python floats."""
        self.model.train()
        token_ids = token_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.opt.zero_grad(set_to_none=True)
        logits, _ = self.model(token_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def eval_step(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_drift: bool = False,
    ) -> tuple[float, float] | tuple[float, float, list[float]]:
        """Eval-only CE + accuracy; optionally return drift_trajectory."""
        self.model.eval()
        token_ids = token_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits, traj = self.model(
            token_ids, attention_mask, return_trajectory=return_drift
        )
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        if return_drift:
            drifts = drift_from_trajectory(list(traj or []))
            return float(loss.item()), float(acc.item()), drifts
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def drift_telemetry(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        eps_sigma: Optional[float] = None,
    ) -> dict:
        """Compute drift_trajectory, terminal_drift, optional perturbation_delta."""
        self.model.eval()
        token_ids = token_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        _, traj = self.model(token_ids, attention_mask, return_trajectory=True)
        traj_list = list(traj or [])
        drifts = drift_from_trajectory(traj_list)
        terminal = drifts[-1] if drifts else None
        perturbation_delta = None
        if eps_sigma is not None and traj_list:
            emb = self.model.tok_emb(token_ids)
            noise = torch.randn_like(emb) * float(eps_sigma)
            pos = torch.arange(
                token_ids.shape[1], device=token_ids.device
            ).unsqueeze(0).expand(token_ids.shape[0], -1)
            noisy = emb + noise + self.model.pos_emb(pos)
            clean_z0 = traj_list[0]
            if attention_mask is not None:
                mask = attention_mask.to(dtype=noisy.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                noisy_z0 = (noisy * mask).sum(dim=1) / denom
            else:
                noisy_z0 = noisy.mean(dim=1)
            delta = (noisy_z0 - clean_z0).float().reshape(clean_z0.shape[0], -1)
            norms = torch.linalg.vector_norm(delta, ord=2, dim=-1)
            perturbation_delta = float(norms.mean().item())
        return {
            "drift_trajectory": drifts,
            "terminal_drift": terminal,
            "perturbation_delta": perturbation_delta,
            "trajectory_len": len(traj_list),
        }


__all__ = ["LoopTrainer"]
