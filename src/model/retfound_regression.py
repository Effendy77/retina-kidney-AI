# src/model/retfound_regression.py

import torch
from torch import nn
from src.model.retfound_backbone import RETFoundBackbone


class RETFoundRegression(nn.Module):
    """
    RETFound-based regression model to predict a continuous value (e.g., eGFR).

    Structure:
        - Vision transformer backbone (RETFound / ViT from timm)
        - Linear regression head
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Feature extractor from timm or RETFound
        self.backbone = RETFoundBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
        )

        in_dim = self.backbone.get_output_dim()

        # Regression head
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, 1))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] input images
        Returns: [B] continuous prediction (e.g., eGFR)
        """
        feats = self.backbone(x)   # [B, D]
        out = self.head(feats)     # [B, 1]
        return out.view(-1)        # flatten to [B]
