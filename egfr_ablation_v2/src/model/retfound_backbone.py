import torch
import torch.nn as nn
import timm


class RETFoundBackbone(nn.Module):
    """
    Generic RETFound backbone wrapper for ViT-L / ViT-B / ViT-H.
    Works with RETFound MAE or RETFound supervised checkpoints.
    """

    def __init__(self, model_name="vit_large_patch16_224", pretrained=False):
        super().__init__()

        # Create timm model with classification head removed
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0   # ensures CLS token output
        )

        # Some models need explicit reset
        if hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier(0)

        # Embedding dimension of CLS token
        self.embed_dim = getattr(self.model, "num_features", None)

        assert self.embed_dim in (768, 1024, 1408), \
            f"[ERROR] Unexpected embedding dimension: {self.embed_dim}"

    def forward_features(self, x):
        """
        Extract CLS token embedding.
        RETFound MAE returns [B, 197, D]
        ViT returns [B, D]
        """
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)

        # MAE-style output → take CLS
        if feats.ndim == 3:
            return feats[:, 0]

        # ViT-style output → already CLS embedding
        return feats

    def forward(self, x):
        return self.forward_features(x)
