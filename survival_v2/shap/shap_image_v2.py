# survival_v2/shap/shap_image_v2.py

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from captum.attr import Saliency

from survival_v2.src.model.multimodal_deepsurv_v2 import MultimodalDeepSurvV2
from survival_v2.src.datasets.multimodal_survival_dataset_v2 import MultimodalSurvivalDatasetV2
from survival_v2.src.utils.shap_utils_v2 import parse_eid


def main():

    # =======================================================
    # Paths
    # =======================================================
    csv_path = "data/survival_multimodal_master.csv"
    ckpt_path = "survival_v2/checkpoints_single_v2/best_model.pth"
    fold_dir  = "survival_v2/checkpoints_single_v2_5fold"
    out_dir   = "survival_v2/checkpoints_single_v2/shap_image_v2"

    # Validate paths exist
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # =======================================================
    # Device
    # =======================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    # =======================================================
    # Load dataset
    # =======================================================
    dataset = MultimodalSurvivalDatasetV2(csv_path)
    num_tab = len(dataset.tabular_features)

    # Build EID → idx mapping (use raw EIDs for consistent lookup)
        # Build EID → idx mapping USING CLEANED INTEGER EIDs
    eid_to_idx = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        raw_eid = sample.get("eid")
        if raw_eid is not None:
            clean_eid = parse_eid(raw_eid)   # -> int
            eid_to_idx[clean_eid] = i


    # =======================================================
    # Load model
    # =======================================================
    model = MultimodalDeepSurvV2(
        weight_path="retfound/RETFound_mae_natureCFP.pth",
        num_tabular_features=num_tab,
        fusion_dim=512,
        dropout=0.2,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print(f">>> Loaded model checkpoint: {ckpt_path}")

    # =======================================================
    # Collect risk scores from folds → find top EIDs
    # =======================================================
    pooled = []
    for k in range(1, 6):
        fp = os.path.join(fold_dir, f"fold{k}_risk_scores.csv")
        if os.path.exists(fp):
            pooled.append(pd.read_csv(fp))

    if len(pooled) == 0:
        print(">>> No risk files found. Exiting.")
        return

    pooled = pd.concat(pooled)
    pooled = pooled.sort_values("risk_score", ascending=False)

        # Use cleaned integer EIDs for matching
    top_eids = []
    for raw in pooled["eid"].tolist():
        clean = parse_eid(raw)   # 'tensor(5238560)' -> 5238560
        if clean not in top_eids:
            top_eids.append(clean)
        if len(top_eids) >= 10:
            break

    print(">>> Selected top EIDs (display):", top_eids)

    
    # Compute mean baselines from background samples
    n_bg = min(20, len(dataset))
    bg_idx = np.random.choice(len(dataset), size=n_bg, replace=False)
    mean_mask = torch.zeros((1, 1, 224, 224), device=device)
    mean_tab = torch.zeros((1, num_tab), device=device)
    for i in bg_idx:
        mean_mask += dataset[i]["mask"].to(device)
        mean_tab += dataset[i]["tabular"].to(device)
    mean_mask /= n_bg
    mean_tab /= n_bg

    # =======================================================
    # Captum Saliency explainer
    # =======================================================
    def model_image_only(x):
        """ Forward wrapper taking ONLY an image as input """
        B = x.size(0)
        mask = mean_mask.expand(B, -1, -1, -1)
        tab  = mean_tab.expand(B, -1)
        out = model(x, mask, tab)
        return out

    saliency = Saliency(model_image_only)

    # =======================================================
    # Process each top EID
    # =======================================================
    df = pd.read_csv(csv_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # CRITICAL: Match model's expected ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    for eid in top_eids:

        if eid not in eid_to_idx:
            print(f"--- EID {eid} not found in dataset. Skipped.")
            continue

        idx = eid_to_idx[eid]

        # df['eid'] is almost certainly numeric UKBB eid
        img_rows = df[df["eid"] == eid]["fundus_image"].values
        if len(img_rows) == 0:
            print(f"--- No image found for EID {eid}. Skipped.")
            continue
        img_path = img_rows[0]

        # Load image
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            pil_resized = pil_img.resize((224, 224))
            x_img = transform(pil_img).unsqueeze(0).to(device)

        # Compute saliency (gradients)
        grad = saliency.attribute(x_img)
        grad = grad.squeeze().detach().cpu().numpy()  # [3,H,W]

        # Convert to saliency heatmap
        heat = np.mean(np.abs(grad), axis=0)          # [H,W]
        heat = heat / (heat.max() + 1e-8)

        # =======================================================
        # Save overlays
        # =======================================================

        # Overlay
        plt.figure(figsize=(6,6))
        plt.imshow(pil_resized)
        plt.imshow(heat, cmap="jet", alpha=0.45)
        plt.axis("off")
        out_overlay = os.path.join(
            out_dir, f"eid_{eid}_image_shap_overlay_v2.png"
        )
        plt.savefig(out_overlay, dpi=300, bbox_inches="tight")
        plt.close()

        # Raw heatmap
        plt.figure(figsize=(6,6))
        plt.imshow(heat, cmap="jet")
        plt.axis("off")
        out_raw = os.path.join(
            out_dir, f"eid_{eid}_image_shap_raw_v2.png"
        )
        plt.savefig(out_raw, dpi=300, bbox_inches="tight")
        plt.close()

        print(f">>> Saved SHAP image for EID {parse_eid(eid)}")

    print(f">>> Image SHAP complete → {out_dir}")


if __name__ == "__main__":
    main()
