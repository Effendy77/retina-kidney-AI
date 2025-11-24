# src/model/survival_model.py

import torch
from torch import nn
from src.model.retfound_backbone import RETFoundBackbone
from src.model.survival_head import SurvivalHead


class SurvivalModel(nn.Module):
    """
    Full ESRD survival model:
    - RETFound backbone → feature vector
    - Linear survival head → log-risk
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.backbone = RETFoundBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
        )

        in_dim = self.backbone.get_output_dim()
        self.head = SurvivalHead(in_features=in_dim, dropout=dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)            # [B, out_dim]
        logits = self.head(feats)               # [B]
        return logits
