import torch
import torch.nn as nn
import torch.serialization
import argparse
from src.model.retfound_backbone import RETFoundBackbone


# ---------------------------------------------------------
# 1) IMAGE ENCODER (RETFound ViT-L)
# ---------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    Image encoder using a RETFound-compatible ViT-L backbone.
    """

    def __init__(self, weight_path, model_name="vit_large_patch16_224"):
        super().__init__()

        # build ViT-L backbone
        self.backbone = RETFoundBackbone(
            model_name=model_name,
            pretrained=False
        )

        # load RETFound weights (MAE)
        torch.serialization.add_safe_globals([argparse.Namespace])

        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

        # checkpoint is a raw OrderedDict
        state_dict = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()

        # load weights into ViT-L backbone
        self.backbone.model.load_state_dict(state_dict, strict=False)

        self.embed_dim = self.backbone.embed_dim   # 1024

    def forward(self, x):
        return self.backbone(x)   # [B,1024]


# ---------------------------------------------------------
# 2) MASK ENCODER (1-channel vessel mask)
# ---------------------------------------------------------

class VesselMaskEncoder(nn.Module):
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

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(64, 128)

    def forward(self, m):
        x = self.net(m)
        x = x.view(x.size(0), -1)
        return self.fc(x)   # [B,128]


# ---------------------------------------------------------
# 3) TABULAR ENCODER (MLP)
# ---------------------------------------------------------

class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------------------------------
# 4) MULTIMODAL FUSION MODEL
# ---------------------------------------------------------

class MultimodalKidneyModel(nn.Module):
    def __init__(self, weight_path, num_tabular_features,
                 fusion_dim=1024, output_type="regression"):

        super().__init__()

        self.image_encoder = ImageEncoder(weight_path)      # 1024 dim
        self.mask_encoder = VesselMaskEncoder()             # 128 dim
        self.tabular_encoder = TabularEncoder(num_tabular_features)  # 128 dim

        # total = 1024 + 128 + 128 = 1280
        self.total_dim = self.image_encoder.embed_dim + 128 + 128

        self.fusion = nn.Sequential(
            nn.Linear(self.total_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # heads (same shape output)
        self.head = nn.Linear(fusion_dim, 1)
        self.output_type = output_type

    def forward(self, image, mask, tabular):

        img_feat = self.image_encoder(image)     # [B,1024]
        mask_feat = self.mask_encoder(mask)      # [B,128]
        tab_feat = self.tabular_encoder(tabular) # [B,128]

        fused = torch.cat([img_feat, mask_feat, tab_feat], dim=1)
        fused = self.fusion(fused)

        return self.head(fused)
