import torch
import torch.nn as nn
import torch.serialization
import argparse

from src.model.retfound_backbone import RETFoundBackbone


# ============================================================
# 1) IMAGE ENCODER — RETFound ViT-L (MAE)
# ============================================================

class ImageEncoderV2(nn.Module):
    """
    V2 Image encoder using the RETFound ViT-L backbone.
    """

    def __init__(self, weight_path, model_name="vit_large_patch16_224"):
        super().__init__()

        # Build ViT-L backbone
        self.backbone = RETFoundBackbone(
            model_name=model_name,
            pretrained=False
        )

        # Load RETFound MAE weights
        torch.serialization.add_safe_globals([argparse.Namespace])

        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        state_dict = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()

        self.backbone.model.load_state_dict(state_dict, strict=False)

        # RETFound ViT-L = 1024 embedding dimension
        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        return self.backbone(x)   # [B,1024]


# ============================================================
# 2) VESSEL MASK ENCODER — 1 Channel CNN → 128-d
# ============================================================

class VesselMaskEncoderV2(nn.Module):
    """
    Encodes vessel masks (binary 1-channel images) into 128-d vector.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),   # 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, 2, 1),  # 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),  # 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # → 64
        )

        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)   # [B,128]


# ============================================================
# 3) TABULAR ENCODER — MLP for 10 features → 128-d
# ============================================================

class TabularEncoderV2(nn.Module):
    """
    Encodes 10 tabular features into a 128-d vector.
    """

    def __init__(self, input_dim=10, hidden_dim=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)   # [B,128]


# ============================================================
# 4) MULTIMODAL FUSION MODEL (Image + Mask + Tabular)
# ============================================================

class MultimodalKidneyModelV2(nn.Module):
    """
    Multimodal model for eGFR regression:
        - Image encoder → 1024 dims
        - Mask encoder  → 128 dims
        - Tabular encoder → 128 dims
        - Fusion MLP → regression head
    """

    def __init__(
        self,
        weight_path,
        num_tabular_features=10,
        fusion_dim=1024,
        dropout=0.2
    ):
        super().__init__()

        # Encoders
        self.image_encoder = ImageEncoderV2(weight_path)
        self.mask_encoder = VesselMaskEncoderV2()
        self.tabular_encoder = TabularEncoderV2(
            input_dim=num_tabular_features
        )

        # Feature dimensions
        self.total_feature_dim = (
            self.image_encoder.embed_dim +  # 1024
            128 +                           # mask
            128                             # tabular
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(self.total_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Regression output
        self.head = nn.Linear(fusion_dim, 1)

    def forward(self, image, mask, tabular):
        img_feat = self.image_encoder(image)
        mask_feat = self.mask_encoder(mask)
        tab_feat = self.tabular_encoder(tabular)

        fused = torch.cat([img_feat, mask_feat, tab_feat], dim=1)
        fused = self.fusion(fused)

        return self.head(fused)  # [B,1]
