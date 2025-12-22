import torch
import torch.nn as nn
import torch.serialization
import argparse

from egfr_ablation_v2.src.model.retfound_backbone import RETFoundBackbone


# ============================================================
# 1) IMAGE ENCODER — RETFound ViT-L (MAE)
# ============================================================

class ImageEncoderV2(nn.Module):
    def __init__(self, weight_path, model_name="vit_large_patch16_224"):
        super().__init__()

        self.backbone = RETFoundBackbone(
            model_name=model_name,
            pretrained=False
        )

        # Allow loading older checkpoints that store argparse.Namespace
        torch.serialization.add_safe_globals([argparse.Namespace])

        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

        # Robustly extract state_dict from common checkpoint formats
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt.state_dict()

        self.backbone.model.load_state_dict(state_dict, strict=False)
        self.embed_dim = self.backbone.embed_dim  # typically 1024 for vit_large

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# 2) VESSEL MASK ENCODER — 1 Channel CNN → 128-d
# ============================================================

class VesselMaskEncoderV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ============================================================
# 3) TABULAR ENCODER — MLP → 128-d
# ============================================================

class TabularEncoderV2(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ============================================================
# 4) MULTIMODAL FUSION MODEL (Ablation-enabled)
# ============================================================

class MultimodalKidneyModelV2(nn.Module):
    """
    Ablation-ready multimodal model:
        - image branch (RETFound)
        - mask branch (CNN)
        - tabular branch (MLP)
        - fusion MLP → regression
      Ablation is performed by zeroing embeddings, not changing architecture.
    """

    def __init__(
        self,
        weight_path,
        num_tabular_features=10,
        fusion_dim=1024,
        dropout=0.2,
        use_image=True,
        use_mask=True,
        use_tabular=True,
        use_retinal_features=True,  # informational only; dataset controls tabular columns
    ):
        super().__init__()

        self.use_image = use_image
        self.use_mask = use_mask
        self.use_tabular = use_tabular
        self.use_retinal_features = use_retinal_features

        self.image_encoder = ImageEncoderV2(weight_path)
        self.mask_encoder = VesselMaskEncoderV2()
        self.tabular_encoder = TabularEncoderV2(num_tabular_features)

        self.total_feature_dim = (
            self.image_encoder.embed_dim +  # e.g. 1024
            128 +                           # mask
            128                             # tabular
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.total_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(fusion_dim, 1)

    def forward(self, image, mask, tabular):
        # Robustly determine batch/device
        B = tabular.size(0) if tabular is not None else image.size(0)
        device = tabular.device if tabular is not None else image.device

        # IMAGE
        if self.use_image:
            img_feat = self.image_encoder(image)
        else:
            img_feat = torch.zeros((B, self.image_encoder.embed_dim), device=device)

        # MASK
        if self.use_mask:
            mask_feat = self.mask_encoder(mask)
        else:
            mask_feat = torch.zeros((B, 128), device=device)

        # TABULAR
        if self.use_tabular:
            tab_feat = self.tabular_encoder(tabular)
        else:
            tab_feat = torch.zeros((B, 128), device=device)

        fused = torch.cat([img_feat, mask_feat, tab_feat], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)
