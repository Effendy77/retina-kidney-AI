import torch
import torch.nn as nn
import timm
import os


# ============================================================
# 1. Image Encoder (RETFound backbone)
# ============================================================

class RetfoundImageEncoderV2(nn.Module):
    """
    Wraps RETFound ViT backbone.
    Returns CLS token embedding.
    """

    def __init__(self, weight_path: str):
        super().__init__()
        # Load RETFound ViT-L (or ViT-B depending on your weights)
        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,    # no classification head
        )

        # Load pretrained RETFound weights
        if weight_path is not None:
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            state = torch.load(weight_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state, strict=False)
            print(f"[RetfoundImageEncoderV2] Loaded pretrained RETFound weights.")


        self.embed_dim = self.model.num_features

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        Returns: [B, embed_dim]
        """
        # Some timm models return [B, seq, embed] or [B, embed] depending on forward mode
        # Use forward_features to ensure we get sequence, then extract CLS token
        if hasattr(self.model, "forward_features"):
            feat = self.model.forward_features(x)  # [B, seq, embed]
            return feat[:, 0] if feat.ndim == 3 else feat  # CLS token or direct embedding
        return self.model(x)   # fallback: timm may return CLS token by default


# ============================================================
# 2. Vessel Mask Encoder (simple CNN)
# ============================================================

class VesselMaskEncoderV2(nn.Module):
    """
    Light CNN for 1-channel vessel mask.
    Output: 128-dim embedding
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        """
        x: [B, 1, 224, 224]
        """
        feat = self.encoder(x).view(x.size(0), -1)
        return self.fc(feat)


# ============================================================
# 3. Tabular Encoder (MLP)
# ============================================================

class TabularEncoderV2(nn.Module):
    """
    MLP for tabular clinical features.
    Input: N features (e.g., 7)
    Output: 128-dim embedding
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
# 4. DeepSurv Fusion Model (Multimodal)
# ============================================================

class MultimodalDeepSurvV2(nn.Module):
     """
     Fusion of:
         - RETFound image encoder
         - Vessel mask encoder
         - Tabular feature encoder
     Followed by Cox survival head outputting log-risk.
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
         self.image_encoder = RetfoundImageEncoderV2(weight_path)
         self.mask_encoder = VesselMaskEncoderV2()
         self.tabular_encoder = TabularEncoderV2(num_tabular_features)

         fused_dim = (
             self.image_encoder.embed_dim +  # ViT-L = 1024 or ViT-B = 768
             128 +                           # mask encoder
             128                             # tabular encoder
         )

         # Fusion MLP
         self.fusion = nn.Sequential(
             nn.Linear(fused_dim, fusion_dim),
             nn.ReLU(),
             nn.Dropout(dropout),
         )

         # DeepSurv head → log-risk score (raw logits)
         self.head = nn.Linear(fusion_dim, 1)

     def forward(self, image, mask, tabular):
         img_feat = self.image_encoder(image)         # [B, embed]
         mask_feat = self.mask_encoder(mask)          # [B, 128]
         tab_feat = self.tabular_encoder(tabular)     # [B, 128]

         fused = torch.cat([img_feat, mask_feat, tab_feat], dim=1)
         fused = self.fusion(fused)
         log_risk = self.head(fused).squeeze(-1)      # [B] raw logits

         # For Cox loss: return log_risk directly
         # For inference: return exp(log_risk) as hazard ratio
         return log_risk
