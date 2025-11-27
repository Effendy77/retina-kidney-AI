# survival_v2/shap/combined_shap_v2.py

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
    out_dir = os.path.join(root, "shap_combined_v2")

    os.makedirs(out_dir, exist_ok=True)

    # Basic existence checks
    if not os.path.isdir(image_dir):
        print(f">>> Image SHAP directory not found: {image_dir}")
        return
    if not os.path.isdir(mask_dir):
        print(f">>> Mask SHAP directory not found: {mask_dir}")
        return

    print(">>> Combining image + mask SHAP into multimodal 2×2 panels...")

    # ---------------------------------------------------------
    # FIND SHAP FILES
    # ---------------------------------------------------------
    try:
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(".png")])
    except Exception as e:
        print(f">>> Error listing SHAP directories: {e}")
        return

    # Group by EID
    image_by_eid = {}
    mask_by_eid = {}

    for f in image_files:
        eid = extract_eid_from_filename(f)
        if eid is not None:
            image_by_eid.setdefault(eid, []).append(f)

    for f in mask_files:
        eid = extract_eid_from_filename(f)
        if eid is not None:
            mask_by_eid.setdefault(eid, []).append(f)

    # ---------------------------------------------------------
    # PROCESS EIDS APPEARING IN BOTH IMAGE + MASK SHAP SETS
    # ---------------------------------------------------------
    common_eids = sorted(set(image_by_eid.keys()) & set(mask_by_eid.keys()))

    print(f">>> Found {len(common_eids)} EIDs with both image and mask SHAP.")

    for eid in common_eids:
        # -----------------------------------------------------
        # SELECT FILES FOR THIS EID
        # -----------------------------------------------------
        img_files = image_by_eid[eid]
        mask_files = mask_by_eid[eid]

        # Prefer overlay first, raw second
        img_overlay = next((f for f in img_files if "overlay" in f), img_files[0])
        img_raw     = next((f for f in img_files if "raw" in f), img_files[-1])

        mask_overlay = next((f for f in mask_files if "overlay" in f), mask_files[0])
        mask_raw     = next((f for f in mask_files if "raw" in f), mask_files[-1])

        # -----------------------------------------------------
        # LOAD IMAGES
        # -----------------------------------------------------
        # use context manager to ensure files are closed promptly
        with Image.open(os.path.join(image_dir, img_overlay)) as pil_img_overlay, \
             Image.open(os.path.join(image_dir, img_raw)) as pil_img_raw, \
             Image.open(os.path.join(mask_dir, mask_overlay)) as pil_mask_overlay, \
             Image.open(os.path.join(mask_dir, mask_raw)) as pil_mask_raw:

            # -----------------------------------------------------
            # CREATE MULTIMODAL PANEL (2×2)
            # -----------------------------------------------------
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(pil_img_overlay)
            axes[0, 0].set_title(f"Fundus + SHAP Overlay")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(pil_img_raw)
            axes[0, 1].set_title("Fundus SHAP Heatmap")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(pil_mask_overlay)
            axes[1, 0].set_title("Mask + SHAP Overlay")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(pil_mask_raw)
            axes[1, 1].set_title("Mask SHAP Heatmap")
            axes[1, 1].axis("off")

            plt.tight_layout()

            # -----------------------------------------------------
            # SAVE
            # -----------------------------------------------------
            out_path = os.path.join(out_dir, f"eid_{eid}_combined_shap_v2.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f">>> Saved combined SHAP for EID {eid} → {out_path}")

    print(f">>> Combined SHAP panels saved to: {out_dir}")


if __name__ == "__main__":
    main()
