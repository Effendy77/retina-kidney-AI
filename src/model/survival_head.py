# src/model/survival_head.py

import torch
from torch import nn


class SurvivalHead(nn.Module):
    """
    Simple linear survival head that maps backbone features to a single log-risk score.
    """

    def __init__(self, in_features: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, in_features]
        returns: [batch_size] log-risk scores
        """
        out = self.net(x)
        return out.view(-1)
