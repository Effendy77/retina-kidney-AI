# src/inference/predict_egfr.py

import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional

from src.model.retfound_regression import RETFoundRegression


def load_image(image_path: str, image_size: int = 224):
    """
    Load and preprocess a single fundus image for eGFR prediction.
    """
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # You can add normalization here later if desired
    ])

    tensor = transform(img)  # [C, H, W]
    tensor = tensor.unsqueeze(0)  # [1, C, H, W]
    return tensor


def load_regression_model(
    checkpoint_path: Optional[str] = None,
    backbone_name: str = "vit_base_patch16_224",
    device: Optional[torch.device] = None,
):
    """
    Create a RETFoundRegression model and optionally load a checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RETFoundRegression(
        backbone_name=backbone_name,
        pretrained=True,
        dropout=0.1,
    )

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        # support either full state dict or nested key
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_egfr_single(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    backbone_name: str = "vit_base_patch16_224",
    device: Optional[torch.device] = None,
) -> float:
    """
    Run eGFR prediction on a single image.

    Returns:
        predicted_eGFR (float)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_regression_model(
        checkpoint_path=checkpoint_path,
        backbone_name=backbone_name,
        device=device,
    )

    tensor = load_image(image_path).to(device)

    pred = model(tensor)  # [1]
    return float(pred.item())
