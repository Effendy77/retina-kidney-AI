import torch
import torch.nn as nn
import timm
import os


# ============================================================
# 1. Image Encoder (RETFound backbone)
# ============================================================

class RetfoundImageEncoderV3(nn.Module):
    """
    RETFound ViT-L encoder wrapper for Survival_v3.
    Extracts CLS token embedding.
    """

    def __init__(self, weight_path: str):
        super().__init__()

        # ViT-L/16 backbone
        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        # Load pretrained RETFound MAE weights
        if weight_path is not None and os.path.exists(weight_path):
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            state = torch.load(weight_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state, strict=False)
            print("[RetfoundImageEncoderV3] Loaded RETFound pretrained weights.")

        # Feature dimension (ViT-L = 1024, ViT-B = 768)
        self.embed_dim = self.model.num_features

    def forward(self, x):
        # Use forward_features to extract all patch embeddings
        feat = self.model.forward_features(x)  # [B, seq, embed]
        return feat[:, 0]                      # CLS token embedding
        

# ============================================================
# 2. Vessel Mask Encoder (same as v2, 1-channel mask → 128-D)
# ============================================================

class VesselMaskEncoderV3(nn.Module):
    """
    Light CNN encoder for 1-channel vessel masks.
    Output embedding: 128-D
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        feat = self.encoder(x).view(x.size(0), -1)
        return self.fc(feat)


# ============================================================
# 3. Tabular Encoder (11 features → 128-D)
# ============================================================

class TabularEncoderV3(nn.Module):
    """
    MLP for 11 tabular clinical + retinal features.
    Output: 128-D embedding
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Multimodal DeepSurv (Image + Mask + Tabular)
# ============================================================

class MultimodalDeepSurvV3(nn.Module):
    """
    Multimodal ESRD survival model:
        - RETFound image encoder
        - Vessel mask encoder
        - Tabular encoder (11 features)
        - Fusion MLP → Cox proportional hazards head
    """

    def __init__(
        self,
        weight_path: str,
        num_tabular_features: int,
        fusion_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Encoders
        self.image_encoder = RetfoundImageEncoderV3(weight_path)
        self.mask_encoder = VesselMaskEncoderV3()
        self.tabular_encoder = TabularEncoderV3(num_tabular_features)

        # Total fused embedding size
        fused_dim = (
            self.image_encoder.embed_dim +  # 1024 (ViT-L)
            128 +                           # mask encoder
            128                             # tabular encoder
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cox PH head → log-risk
        self.head = nn.Linear(fusion_dim, 1)


    def forward(self, image, mask, tabular):
        img_feat = self.image_encoder(image)      # [B, embed_dim]
        mask_feat = self.mask_encoder(mask)       # [B, 128]
        tab_feat = self.tabular_encoder(tabular)  # [B, 128]

        fused = torch.cat([img_feat, mask_feat, tab_feat], dim=1)
        fused = self.fusion(fused)
        log_risk = self.head(fused).squeeze(-1)

        # return raw log-risk for Cox PH loss
        return log_risk
