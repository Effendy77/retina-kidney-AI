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

from survival_v3.src.model.multimodal_deepsurv_v3 import MultimodalDeepSurvV3
from survival_v3.src.datasets.multimodal_survival_dataset_v3 import MultimodalSurvivalDatasetV3
from survival_v3.src.utils.shap_utils_v3 import parse_eid


def main():

    # =======================================================
    # PATHS (UPDATED FOR Survival_v3)
    # =======================================================
    csv_path  = "data/survival_multimodal_master_v3.csv"
    ckpt_path = "survival_v3/experiments/single_run/best_model.pth"
    fold_dir  = "survival_v3/experiments/single_run_5fold"
    out_dir   = "survival_v3/checkpoints_single_v3/shap_image_v3"

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
    # Load dataset (v3)
    # =======================================================
    dataset = MultimodalSurvivalDatasetV3(csv_path)
    num_tab = len(dataset.tabular_features)

    # Map EID → dataset index
    eid_to_idx = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        raw_eid = sample.get("eid")
        if raw_eid is not None:
            clean_eid = parse_eid(raw_eid)
            eid_to_idx[clean_eid] = i

    # =======================================================
    # Load model (v3)
    # =======================================================
    model = MultimodalDeepSurvV3(
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
    # Collect risk scores from 5-fold to pick top EIDs
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

    # Select top-10 highest risk individuals
    top_eids = []
    for raw in pooled["eid"].tolist():
        clean = parse_eid(raw)
        if clean not in top_eids:
            top_eids.append(clean)
        if len(top_eids) >= 10:
            break

    print(">>> Selected top EIDs:", top_eids)

    # =======================================================
    # Compute mean background mask + mean tabular features
    # =======================================================
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
    # Captum Saliency (IMAGE CHANNEL ONLY)
    # =======================================================
    def model_image_only(x):
        """
        Forward wrapper for image-only SHAP.
        Mask and tabular replaced with averaged baseline values.
        """
        B = x.size(0)
        mask = mean_mask.expand(B, -1, -1, -1)
        tab  = mean_tab.expand(B, -1)
        return model(x, mask, tab)

    saliency = Saliency(model_image_only)

    # =======================================================
    # Load CSV for image paths
    # =======================================================
    df = pd.read_csv(csv_path)

    # Transforms (match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # =======================================================
    # PROCESS EACH TOP EID
    # =======================================================
    for eid in top_eids:

        if eid not in eid_to_idx:
            print(f"--- EID {eid} not in dataset. Skipped.")
            continue

        idx = eid_to_idx[eid]

        # Fetch image path from CSV
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

        # Compute saliency gradients
        grad = saliency.attribute(x_img)
        grad = grad.squeeze().detach().cpu().numpy()   # [3,H,W]

        heat = np.mean(np.abs(grad), axis=0)          # reduce to [H,W]
        heat = heat / (heat.max() + 1e-8)

        # =======================================================
        # Save SHAP overlays (v3 filenames)
        # =======================================================
        # Overlay
        plt.figure(figsize=(6,6))
        plt.imshow(pil_resized)
        plt.imshow(heat, cmap="jet", alpha=0.45)
        plt.axis("off")
        out_overlay = os.path.join(out_dir, f"eid_{eid}_image_shap_overlay_v3.png")
        plt.savefig(out_overlay, dpi=300, bbox_inches="tight")
        plt.close()

        # Raw heatmap
        plt.figure(figsize=(6,6))
        plt.imshow(heat, cmap="jet")
        plt.axis("off")
        out_raw = os.path.join(out_dir, f"eid_{eid}_image_shap_raw_v3.png")
        plt.savefig(out_raw, dpi=300, bbox_inches="tight")
        plt.close()

        print(f">>> Saved SHAP for EID {eid}")

    print(f"\n>>> Image SHAP complete → {out_dir}\n")


if __name__ == "__main__":
    main()
