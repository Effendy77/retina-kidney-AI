import torch
import torch.nn as nn
import timm


class RETFoundBackbone(nn.Module):
    def __init__(self, model_name="vit_large_patch16_224", pretrained=False):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        self.embed_dim = self.model.num_features
        assert self.embed_dim in (1024, 768, 1408), \
            f"Unexpected embedding dimension: {self.embed_dim}"

    def forward_features(self, x):
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)

        if feats.ndim == 3:
            return feats[:, 0]   # CLS TOKEN

        return feats

    def forward(self, x):
        return self.forward_features(x)
