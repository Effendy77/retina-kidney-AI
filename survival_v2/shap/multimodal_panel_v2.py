# survival_v2/shap/multimodal_panel_v2.py

import os
import re
import matplotlib
matplotlib.use("Agg")   # Force non-GUI backend for image saving
import matplotlib.pyplot as plt
from PIL import Image

from survival_v2.src.utils.shap_utils_v2 import parse_eid


def extract_eid_from_filename(filename):
    """
    Extract EID from patterns like:
    - eid_123_image_shap_overlay_v2.png
    - eid_123_mask_shap_raw_v2.png
    """
    match = re.search(r"eid_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def main():
    # ---------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------
    root = "survival_v2/checkpoints_single_v2"

    image_dir = os.path.join(root, "shap_image_v2")
    mask_dir = os.path.join(root, "shap_mask_v2")
    tab_dir  = os.path.join(root, "shap_tabular_v2")

    out_dir = os.path.join(root, "shap_multimodal_panel_v2")
    os.makedirs(out_dir, exist_ok=True)

    print(">>> Building full multimodal SHAP panels (3×2)...")

    # ---------------------------------------------------------
    # FIND FILES FOR EACH MODALITY
    # ---------------------------------------------------------
    # Ensure directories exist and list PNGs (case-insensitive)
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        print(f">>> ERROR: Missing SHAP directories: {image_dir} or {mask_dir}")
        return
    try:
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])
        mask_files  = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(".png")])
    except Exception as e:
        print(f">>> Error listing files: {e}")
        return

    # Tabular plots: beeswarm + bar
    beeswarm_path = os.path.join(tab_dir, "tabular_shap_summary_beeswarm_v2.png")
    bar_path      = os.path.join(tab_dir, "tabular_shap_summary_bar_v2.png")

    if not (os.path.exists(beeswarm_path) and os.path.exists(bar_path)):
        print(">>> WARNING: Tabular SHAP plots not found. Skipping multimodal panels.")
        return

    # Build EID → file dicts
    image_by_eid = {}
    mask_by_eid = {}

    for f in image_files:
        if f.endswith(".png"):
            eid = extract_eid_from_filename(f)
            if eid is not None:
                image_by_eid.setdefault(eid, []).append(f)

    for f in mask_files:
        if f.endswith(".png"):
            eid = extract_eid_from_filename(f)
            if eid is not None:
                mask_by_eid.setdefault(eid, []).append(f)

    # EIDs for which both modalities exist
    common_eids = sorted(set(image_by_eid.keys()) & set(mask_by_eid.keys()))

    print(f">>> Found {len(common_eids)} EIDs with both image + mask SHAP.")

    # ---------------------------------------------------------
    # BUILD PANELS
    # ---------------------------------------------------------
    for eid in common_eids:
        img_files = image_by_eid[eid]
        mask_files = mask_by_eid[eid]

        # Preferred file selection
        img_overlay = next((f for f in img_files if "overlay" in f), img_files[0])
        img_raw     = next((f for f in img_files if "raw" in f), img_files[-1])

        mask_overlay = next((f for f in mask_files if "overlay" in f), mask_files[0])
        mask_raw     = next((f for f in mask_files if "raw" in f), mask_files[-1])

        # Load modality images using context managers to ensure files are closed
        with Image.open(os.path.join(image_dir, img_overlay)) as pil_img_overlay, \
             Image.open(os.path.join(image_dir, img_raw))     as pil_img_raw, \
             Image.open(os.path.join(mask_dir, mask_overlay))  as pil_mask_overlay, \
             Image.open(os.path.join(mask_dir, mask_raw))      as pil_mask_raw, \
             Image.open(beeswarm_path)                         as pil_beeswarm, \
             Image.open(bar_path)                              as pil_bar:

            # -----------------------------------------------------
            # 3×2 MULTIMODAL PANEL
            # -----------------------------------------------------
            fig, axes = plt.subplots(3, 2, figsize=(14, 18))

            # Row 1 — Fundus
            axes[0, 0].imshow(pil_img_overlay)
            axes[0, 0].set_title("Fundus + SHAP Overlay")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(pil_img_raw)
            axes[0, 1].set_title("Fundus SHAP Heatmap")
            axes[0, 1].axis("off")

            # Row 2 — Mask
            axes[1, 0].imshow(pil_mask_overlay)
            axes[1, 0].set_title("Mask + SHAP Overlay")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(pil_mask_raw)
            axes[1, 1].set_title("Mask SHAP Heatmap")
            axes[1, 1].axis("off")

            # Row 3 — Tabular Feature SHAP
            axes[2, 0].imshow(pil_bar)
            axes[2, 0].set_title("Tabular SHAP — Feature Importance")
            axes[2, 0].axis("off")

            axes[2, 1].imshow(pil_beeswarm)
            axes[2, 1].set_title("Tabular SHAP — Beeswarm Distribution")
            axes[2, 1].axis("off")

            plt.tight_layout()

            # Save final panel
            out_path = os.path.join(out_dir, f"eid_{eid}_multimodal_panel_v2.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f">>> Saved multimodal SHAP panel for EID {eid} → {out_path}")

    print(f">>> All multimodal SHAP panels saved to {out_dir}")


if __name__ == "__main__":
    main()
