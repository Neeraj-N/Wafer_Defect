"""
Loss functions for the imbalanced wafer-defect problem.

The default training loss in this repo is class-weighted cross-entropy, which
already compensates for the ~85% `none` majority. Focal loss (Lin et al.,
2017) is offered as an alternative that additionally *down-weights easy,
confidently-correct examples* so the gradient keeps focusing on the hard
minority patterns (Scratch, Loc, Edge-Loc) even late in training. Enable it
with `python -m src.train --loss focal`.

`gamma=0` recovers ordinary (optionally weighted) cross-entropy, which the
unit tests check.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # per-sample CE (already applies class weights if provided)
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        # pt = model probability assigned to the true class = exp(-ce_unweighted).
        # Using exp(-ce) is exact when no weights are set and a mild, standard
        # approximation when they are; the modulating term (1-pt)^gamma is what
        # matters here.
        logp = F.log_softmax(logits, dim=1)
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        loss = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
