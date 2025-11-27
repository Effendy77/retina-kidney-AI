# survival_v2/shap/shap_mask_v2.py

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from captum.attr import Saliency

from survival_v2.src.datasets.multimodal_survival_dataset_v2 import (
    MultimodalSurvivalDatasetV2,
)
from survival_v2.src.model.multimodal_deepsurv_v2 import (
    MultimodalDeepSurvV2,
)
from survival_v2.src.utils.shap_utils_v2 import parse_eid


def main():
    # =======================================================
    # Paths
    # =======================================================
    csv_path = "data/survival_multimodal_master.csv"
    ckpt_path = "survival_v2/checkpoints_single_v2/best_model.pth"
    fold_dir  = "survival_v2/checkpoints_single_v2_5fold"
    out_dir   = "survival_v2/checkpoints_single_v2/shap_mask_v2"

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[ERROR] Missing CSV: {csv_path}")
        return
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Missing checkpoint: {ckpt_path}")
        return

    # =======================================================
    # Device + dataset
    # =======================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    dataset = MultimodalSurvivalDatasetV2(csv_path)
    num_tab = len(dataset.tabular_features)

    # Build mapping: CLEAN EID (int) -> dataset index
    eid_to_idx = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        raw_eid = sample.get("eid")
        if raw_eid is not None:
            clean = parse_eid(raw_eid)
            eid_to_idx[clean] = i

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
    print(f">>> Loaded model: {ckpt_path}")

    # =======================================================
    # Read fold risk scores -> top EIDs (cleaned)
    # =======================================================
    pooled = []
    for k in range(1, 6):
        fp = os.path.join(fold_dir, f"fold{k}_risk_scores.csv")
        if os.path.exists(fp):
            pooled.append(pd.read_csv(fp))

    if len(pooled) == 0:
        print(">>> No fold risk scores found, exiting.")
        return

    pooled = pd.concat(pooled, axis=0)
    pooled = pooled.sort_values("risk_score", ascending=False)

    top_eids = []
    for raw in pooled["eid"].tolist():
        clean = parse_eid(raw)
        if clean not in top_eids:
            top_eids.append(clean)
        if len(top_eids) >= 10:
            break

    print(">>> Top mask SHAP EIDs:", top_eids)

    # =======================================================
    # Background statistics: mean image + mean tabular
    # =======================================================
    n_bg = min(20, len(dataset))
    bg_idx = np.random.choice(len(dataset), size=n_bg, replace=False)

    mean_img = torch.zeros((1, 3, 224, 224), device=device)
    mean_tab = torch.zeros((1, num_tab), device=device)

    for i in bg_idx:
        s = dataset[i]
        mean_img += s["image"].to(device)
        mean_tab += s["tabular"].to(device)

    mean_img /= n_bg
    mean_tab /= n_bg

    # =======================================================
    # Saliency wrapper: vary only the mask
    # =======================================================
    def model_mask_only(mask_tensor: torch.Tensor):
        """
        Forward wrapper taking only a vessel mask [B,1,224,224]
        and using mean image + mean tabular as fixed context.
        """
        B = mask_tensor.size(0)
        img = mean_img.expand(B, -1, -1, -1)
        tab = mean_tab.expand(B, -1)
        out = model(img, mask_tensor, tab)
        return out

    saliency = Saliency(model_mask_only)

    # =======================================================
    # Compute SHAP/Saliency for each top EID
    # =======================================================
    for eid in top_eids:
        if eid not in eid_to_idx:
            print(f"--- EID {eid} not in dataset, skipped.")
            continue

        idx = eid_to_idx[eid]
        sample = dataset[idx]
        mask = sample["mask"]
        # ensure tensor on device and batch/channel dims: expect [1,1,H,W]
        mask = mask.to(device)
        if mask.dim() == 3:           # [1,H,W] -> [1,1,H,W]
            mask = mask.unsqueeze(1)
        if mask.dim() == 2:           # [H,W] -> [1,1,H,W]
            mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.contiguous()

        # Compute gradients w.r.t. mask (defensive)
        mask.requires_grad_(True)
        try:
            attr = saliency.attribute(mask)
        except Exception as exc:
            print(f"[WARN] Saliency failed for EID {eid}: {exc}")
            continue
        attr = attr.squeeze().detach().cpu().numpy()  # [H,W] if mask is 1-ch

        # Convert to heatmap
        if attr.ndim == 3:
            # In case it ends up [C,H,W], average over channels
            attr = np.mean(np.abs(attr), axis=0)
        else:
            attr = np.abs(attr)

        heat = attr / (attr.max() + 1e-8)  # [H,W], 0–1

        # Also get the original mask for overlay
        mask_np = sample["mask"].squeeze().cpu().numpy()

        # Paths
        raw_path = os.path.join(out_dir, f"eid_{eid}_mask_shap_raw_v2.png")
        overlay_path = os.path.join(out_dir, f"eid_{eid}_mask_shap_overlay_v2.png")

        # Raw heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(heat, cmap="jet")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(raw_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        # Overlay on vessel mask
        plt.figure(figsize=(5, 5))
        plt.imshow(mask_np, cmap="gray")
        plt.imshow(heat, cmap="jet", alpha=0.45)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"[OK] Saved mask SHAP for eid {eid}")

    print(f"\n>>> All mask SHAP saved in {out_dir}")


if __name__ == "__main__":
    main()
