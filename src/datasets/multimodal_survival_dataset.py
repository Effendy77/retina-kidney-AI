# src/datasets/multimodal_survival_dataset.py

import os
from typing import List, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MultimodalSurvivalDataset(Dataset):
    """
    Multimodal survival dataset combining:
      - fundus_image (RGB)
      - vessel_mask (1-channel)
      - tabular clinical covariates
      - time_to_event, event (for ESRD survival)
    """

    def __init__(
        self,
        csv_path: str,
        tabular_features: Optional[List[str]] = None,
    ):
        super().__init__()

        self.df = pd.read_csv(csv_path)

        # ---- Tabular covariates ----
        if tabular_features is None:
            self.tabular_features = [
                "age",
                "sex",
                "diabetes",
                "hypertension",
                "dm_htn_combined",
                "qrisk3",
                "egfr",
            ]
        else:
            self.tabular_features = tabular_features

        # Basic sanity: drop rows with missing required fields
        required_cols = (
            ["fundus_image", "vessel_mask", "time_to_event", "event"]
            + self.tabular_features
        )
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)

        # ---- Transforms ----
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
            T.ToTensor(),   # 1-channel float in [0,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ---- Image ----
        img_path = row["fundus_image"]
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        # ---- Vessel mask ----
        mask_path = row["vessel_mask"]
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        # ---- Tabular covariates ----
        tab = row[self.tabular_features].astype(float).values
        tab = torch.tensor(tab, dtype=torch.float32)

        # ---- Survival target ----
        time = torch.tensor(row["time_to_event"], dtype=torch.float32)
        event = torch.tensor(row["event"], dtype=torch.float32)

        sample = {
            "image": img,
            "mask": mask,
            "tabular": tab,
            "time": time,
            "event": event,
        }

        if "eid" in self.df.columns:
            sample["eid"] = int(row["eid"])

        return sample
