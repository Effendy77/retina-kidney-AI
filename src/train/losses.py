# src/train/losses.py

import torch
from torch import nn


class CoxPHLoss(nn.Module):
    """
    Negative partial log-likelihood for Cox proportional hazards model.

    Assumes:
      - y_time:  survival / follow-up times (float tensor, shape [N])
      - y_event: event indicator (1 = event, 0 = censored) (float/bool tensor, shape [N])
      - pred_risk: model output, higher = higher risk (shape [N])
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_risk: torch.Tensor, y_time: torch.Tensor, y_event: torch.Tensor) -> torch.Tensor:
        # Ensure 1D
        pred_risk = pred_risk.view(-1)
        y_time = y_time.view(-1)
        y_event = y_event.view(-1)

        # Sort by descending time
        sorted_indices = torch.argsort(y_time, descending=True)
        pred_risk = pred_risk[sorted_indices]
        y_event = y_event[sorted_indices]

        # Compute log partial likelihood
        # log L = sum_{i: event} [ r_i - log(sum_{j in R_i} exp(r_j)) ]
        risk = pred_risk
        hazard_ratio = torch.exp(risk)
        # cumulative sum of hazard_ratio gives denominator of risk set for each time
        log_cumsum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0) + self.eps)

        # Only events contribute
        log_likelihood = torch.sum((risk - log_cumsum_hazard) * y_event)

        # Negative average log-likelihood
        # use total number of events for normalization
        num_events = torch.clamp(y_event.sum(), min=1.0)
        loss = -log_likelihood / num_events
        return loss
