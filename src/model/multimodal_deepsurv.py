# src/model/multimodal_deepsurv.py

import torch
import torch.nn as nn

# We reuse the encoders you already defined
from src.model.multimodal_fusion import (
    ImageEncoder,
    VesselMaskEncoder,
    TabularEncoder,
)


class MultimodalDeepSurv(nn.Module):
    """
    DeepSurv-style multimodal survival model:

      - ImageEncoder (RETFound ViT-L, CLS embedding)
      - VesselMaskEncoder (1-channel CNN)
      - TabularEncoder (MLP)

    Outputs:
      - risk score (higher = higher hazard)
    """

    def __init__(
        self,
        weight_path: str,
        num_tabular_features: int,
        fusion_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Encoders (same as regression)
        self.image_encoder = ImageEncoder(weight_path)
        self.mask_encoder = VesselMaskEncoder()
        self.tabular_encoder = TabularEncoder(num_tabular_features)

        total_dim = (
            self.image_encoder.embed_dim +  # e.g. 1024
            128 +                           # mask
            128                             # tabular
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # DeepSurv head -> single risk score
        self.head = nn.Linear(fusion_dim, 1)

    def forward(self, image, mask, tabular):
        img_feat = self.image_encoder(image)      # [B, EMBED]
        mask_feat = self.mask_encoder(mask)       # [B,128]
        tab_feat = self.tabular_encoder(tabular)  # [B,128]

        if mask_feat.ndim > 2:
            mask_feat = mask_feat.view(mask_feat.size(0), -1)

        fused = torch.cat([img_feat, mask_feat, tab_feat], dim=1)
        fused = self.fusion(fused)
        risk = self.head(fused).squeeze(-1)       # [B]

        return risk
