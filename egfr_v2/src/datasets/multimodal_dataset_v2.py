import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MultimodalKidneyDatasetV2(Dataset):
    """
    V2 dataset:
      - Loads LEFT-eye RGB fundus image
      - Loads LEFT-eye vessel mask (binary)
      - Loads updated tabular features (10 cols)
      - Loads eGFR as regression target
    """

    def __init__(self, csv_path, image_root, mask_root):
        super().__init__()

        self.df = pd.read_csv(csv_path)

        self.image_root = image_root
        self.mask_root = mask_root

        # ------------------------------
        # 10 TABULAR FEATURES (V2)
        # ------------------------------
        self.tabular_features = [
            "age",
            "sex",
            "diabetes",
            "hypertension",
            "qrisk3",
            "dm_htn_combined",
            "fractal_dim",
            "vessel_density",
            "eccentricity",
            "mean_width_px"
        ]

        # ------------------------------
        # IMAGE TRANSFORM (RETFound-ready)
        # ------------------------------
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # ------------------------------
        # MASK TRANSFORM (1-channel)
        # ------------------------------
        self.mask_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),  # → shape [1, H, W]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -----------------------------------------------------------
        # IMAGE PATH (LEFT EYE ALREADY ENFORCED IN CLEANv2 CSV)
        # -----------------------------------------------------------
        img_path = row["fundus_image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        # Load RGB fundus
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        # -----------------------------------------------------------
        # VESSEL MASK PATH (LEFT EYE)
        # -----------------------------------------------------------
        mask_path = row["vessel_mask"]
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(self.mask_root, mask_path)

        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()   # ensure binary [0,1]

        # -----------------------------------------------------------
        # TABULAR FEATURES
        # -----------------------------------------------------------
        tab_series = row[self.tabular_features].astype(float).fillna(0.0)
        tab = torch.tensor(tab_series.values, dtype=torch.float32)

        # -----------------------------------------------------------
        # eGFR REGRESSION TARGET
        # -----------------------------------------------------------
        target = torch.tensor(row["egfr"], dtype=torch.float32)

        return {
            "image": img,
            "mask": mask,
            "tabular": tab,
            "egfr": target,
        }
