import os
import re
import matplotlib
matplotlib.use("Agg")   # Force non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image

from survival_v3.src.utils.shap_utils_v3 import parse_eid


def extract_eid_from_filename(filename):
    """
    Extract EID from filename patterns such as:
    - eid_123_image_shap_overlay_v3.png
    - eid_123_mask_shap_raw_v3.png
    """
    match = re.search(r"eid_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def main():
    # ---------------------------------------------------------
    # PATHS (UPDATED TO v3)
    # ---------------------------------------------------------
    root = "survival_v3/checkpoints_single_v3"

    image_dir = os.path.join(root, "shap_image_v3")
    mask_dir  = os.path.join(root, "shap_mask_v3")
    tab_dir   = os.path.join(root, "shap_tabular_v3")

    out_dir = os.path.join(root, "shap_multimodal_panel_v3")
    os.makedirs(out_dir, exist_ok=True)

    print(">>> Building multimodal SHAP panels (3×2) for Survival_v3 ...")

    # ---------------------------------------------------------
    # TABULAR SHAP FILES (BEESWARM + BAR)
    # Using the corrected v3 filenames you created earlier
    # ---------------------------------------------------------
    beeswarm_path = os.path.join(tab_dir, "tabular_shap_beeswarm_v3.png")
    bar_path      = os.path.join(tab_dir, "tabular_shap_barplot_v3.png")

    if not os.path.exists(beeswarm_path) or not os.path.exists(bar_path):
        print(">>> WARNING: Tabular SHAP images not found; skipping multimodal panel creation.")
        return

    # ---------------------------------------------------------
    # LIST PNG FILES IN IMAGE & MASK DIRS
    # ---------------------------------------------------------
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        print(f">>> ERROR: Missing SHAP directories: {image_dir} or {mask_dir}")
        return

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])
    mask_files  = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(".png")])

    # ---------------------------------------------------------
    # BUILD MAP OF EID → FILE LIST
    # ---------------------------------------------------------
    image_by_eid = {}
    mask_by_eid  = {}

    for f in image_files:
        eid = extract_eid_from_filename(f)
        if eid is not None:
            image_by_eid.setdefault(eid, []).append(f)

    for f in mask_files:
        eid = extract_eid_from_filename(f)
        if eid is not None:
            mask_by_eid.setdefault(eid, []).append(f)

    # EIDs that have both image + mask SHAP
    common_eids = sorted(set(image_by_eid.keys()) & set(mask_by_eid.keys()))
    print(f">>> Found {len(common_eids)} EIDs with both image + mask SHAP.")

    # ---------------------------------------------------------
    # BUILD MULTIMODAL PANELS
    # ---------------------------------------------------------
    for eid in common_eids:
        img_list  = image_by_eid[eid]
        mask_list = mask_by_eid[eid]

        # Prefer overlay first, fallback to raw if missing
        img_overlay = next((x for x in img_list if "overlay" in x), img_list[0])
        img_raw     = next((x for x in img_list if "raw" in x), img_list[-1])

        mask_overlay = next((x for x in mask_list if "overlay" in x), mask_list[0])
        mask_raw     = next((x for x in mask_list if "raw" in x), mask_list[-1])

        with Image.open(os.path.join(image_dir, img_overlay)) as pil_img_overlay, \
             Image.open(os.path.join(image_dir, img_raw))     as pil_img_raw, \
             Image.open(os.path.join(mask_dir, mask_overlay)) as pil_mask_overlay, \
             Image.open(os.path.join(mask_dir, mask_raw))     as pil_mask_raw, \
             Image.open(bar_path)                            as pil_bar, \
             Image.open(beeswarm_path)                       as pil_beeswarm:

            # ---------------------------------------------------------
            # Create a 3×2 panel
            # ---------------------------------------------------------
            fig, axes = plt.subplots(3, 2, figsize=(14, 18))

            # Row 1 — Fundus SHAP
            axes[0, 0].imshow(pil_img_overlay)
            axes[0, 0].set_title("Fundus + SHAP Overlay")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(pil_img_raw)
            axes[0, 1].set_title("Fundus SHAP Heatmap")
            axes[0, 1].axis("off")

            # Row 2 — Vessel Mask SHAP
            axes[1, 0].imshow(pil_mask_overlay)
            axes[1, 0].set_title("Mask + SHAP Overlay")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(pil_mask_raw)
            axes[1, 1].set_title("Mask SHAP Heatmap")
            axes[1, 1].axis("off")

            # Row 3 — Tabular SHAP visualizations
            axes[2, 0].imshow(pil_bar)
            axes[2, 0].set_title("Tabular SHAP — Feature Importance")
            axes[2, 0].axis("off")

            axes[2, 1].imshow(pil_beeswarm)