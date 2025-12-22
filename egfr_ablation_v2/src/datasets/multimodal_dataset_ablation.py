import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

BASELINE_FEATURES = ["age", "sex", "diabetes", "hypertension", "dm_htn_combined"]
QRISK_FEATURES    = ["qrisk3"]  # or whatever your column name is
RETINAL_FEATURES  = ["fractal_dim", "vessel_density", "eccentricity", "mean_width_px"]



class MultimodalKidneyDatasetV2(Dataset):
    """
    Ablation-ready dataset (baseline-anchored ablation_v2):
      - Loads LEFT-eye RGB fundus image
      - Loads LEFT-eye vessel mask
      - Loads tabular features controlled by:
            tabular_mode: qrisk_only | baseline | baseline_plus_qrisk
            use_retinal_features: append handcrafted retinal scalars (4 cols)
      - Supports ablation flags: use_image, use_mask, use_tabular
      - Returns eGFR regression target
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        mask_root: str,
        use_image: bool = True,
        use_mask: bool = True,
        use_tabular: bool = True,
        use_retinal_features: bool = False,
        tabular_mode: str = "baseline_plus_qrisk",
        silence_qrisk: bool = False,
    ):
        super().__init__()

        # ------------------------------
        # Build tabular feature list
        # ------------------------------
        baseline_cols = ["age", "sex", "diabetes", "hypertension", "dm_htn_combined"]

        if tabular_mode == "qrisk_only":
            self.tabular_features = ["qrisk3"]
        elif tabular_mode == "baseline":
            self.tabular_features = baseline_cols
        elif tabular_mode == "baseline_plus_qrisk":
            self.tabular_features = baseline_cols + ["qrisk3"]
        else:
            raise ValueError(
                f"Unknown tabular_mode={tabular_mode}. "
                f"Choose from: qrisk_only, baseline, baseline_plus_qrisk"
            )

        self.retinal_scalar_cols = ["fractal_dim", "vessel_density", "eccentricity", "mean_width_px"]
        if use_retinal_features:
            self.tabular_features += self.retinal_scalar_cols

        # ------------------------------
        # Validate paths (ALWAYS)
        # ------------------------------
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Image root not a directory: {image_root}")
        if not os.path.isdir(mask_root):
            raise FileNotFoundError(f"Mask root not a directory: {mask_root}")

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # ------------------------------
        # Validate required columns
        # ------------------------------
        required_cols = ["fundus_image", "vessel_mask", "egfr"]
        required_cols.extend(self.tabular_features)

        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Store paths
        self.image_root = image_root
        self.mask_root = mask_root

        # Ablation flags
        self.use_image = use_image
        self.use_mask = use_mask
        self.use_tabular = use_tabular
        self.use_retinal_features = use_retinal_features
        self.tabular_mode = tabular_mode
        self.silence_qrisk = silence_qrisk

        # ------------------------------
        # Image transforms
        # ------------------------------
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.mask_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        print(f"[INFO] Loaded dataset: {len(self.df)} samples")
        print(
            f"[INFO] Ablation flags: image={use_image}, mask={use_mask}, tabular={use_tabular}, "
            f"retinal_features={use_retinal_features}, tabular_mode={tabular_mode}, "
            f"silence_qrisk={silence_qrisk}"
        )
        print(f"[INFO] Tabular features ({len(self.tabular_features)}): {self.tabular_features}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Sample ID
        sample_id = row["eid"] if "eid" in row else idx

        # --------------------------
        # IMAGE
        # --------------------------
        img_path = row["fundus_image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"[WARN] Missing image → {img_path}, filling zeros.")
            img = Image.new("RGB", (224, 224), color=0)

        image_tensor = self.img_transform(img)
        if not self.use_image:
            image_tensor = torch.zeros_like(image_tensor)

        # --------------------------
        # MASK
        # --------------------------
        mask_path = row["vessel_mask"]
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(self.mask_root, mask_path)

        try:
            mask = Image.open(mask_path).convert("L")
        except Exception:
            print(f"[WARN] Missing mask → {mask_path}, filling zeros.")
            mask = Image.new("L", (224, 224), color=0)

        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        if not self.use_mask:
            mask_tensor = torch.zeros_like(mask_tensor)

        ## --------------------------
# TABULAR
# --------------------------
        tab_values = row[self.tabular_features].astype(float).values

# missingness mask (1 = missing)
        tab_missing = torch.tensor(np.isnan(tab_values).astype(np.float32))

# values with NaNs replaced by 0 (real imputation can be fold-fitted in trainer later)
        tab = torch.tensor(np.nan_to_num(tab_values, nan=0.0), dtype=torch.float32)

# Optional: silence qrisk while keeping shape
        if self.silence_qrisk and "qrisk3" in self.tabular_features:
            tab[self.tabular_features.index("qrisk3")] = 0.0

        if not self.use_tabular:
            tab = torch.zeros_like(tab)
            tab_missing = torch.zeros_like(tab_missing)

        # --------------------------
        # TARGET
        # --------------------------
        target = torch.tensor(float(row["egfr"]), dtype=torch.float32)

        return {
            "sample_id": sample_id,
            "image": image_tensor,
            "mask": mask_tensor,
            "tabular": tab,
            "egfr": target,
        }
