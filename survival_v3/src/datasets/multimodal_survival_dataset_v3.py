import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MultimodalSurvivalDatasetV3(Dataset):
    """
    Multimodal ESRD survival dataset (v3):

      • RGB retinal image  (RETFound)
      • 1-channel vessel mask
      • 11 tabular clinical + retinal-engineered features
      • Survival labels: time_to_event, event_occurred
      • Includes EID for downstream inference joining
    """

    def __init__(self, csv_path: str):
        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # ------------------------------------------------------
        # Define ALL tabular features used in survival_v3
        # ------------------------------------------------------
        self.tabular_features = [
            "age",
            "sex",
            "diabetes",
            "hypertension",
            "egfr",
            "qrisk3",
            "dm_htn_combined",
            "fractal_dim",
            "vessel_density",
            "eccentricity",
            "mean_width_px",
        ]

        # ------------------------------------------------------
        # Ensure required core columns exist
        # ------------------------------------------------------
        required = (
            ["fundus_image", "vessel_mask", "time_to_event", "event_occurred"]
            + self.tabular_features
        )

        missing_cols = [c for c in required if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Drop rows missing required fields
        self.df = self.df.dropna(subset=required).reset_index(drop=True)

        # ------------------------------------------------------
        # Image transforms (same as survival_v2)
        # ------------------------------------------------------
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
            T.ToTensor(),     # → [0,1] float mask
        ])


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ------------------------------------------------------
        # Load RGB fundus image
        # ------------------------------------------------------
        img_path = row["fundus_image"]
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        # ------------------------------------------------------
        # Load vessel mask (1-channel)
        # ------------------------------------------------------
        mask_path = row["vessel_mask"]
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        # ------------------------------------------------------
        # Tabular features → tensor [11]
        # ------------------------------------------------------
        tab_values = row[self.tabular_features].astype(float).values
        tab_tensor = torch.tensor(tab_values, dtype=torch.float32)

        # ------------------------------------------------------
        # Survival labels
        # ------------------------------------------------------
        time = torch.tensor(row["time_to_event"], dtype=torch.float32)
        event = torch.tensor(row["event_occurred"], dtype=torch.float32)

        sample = {
            "image": image,
            "mask": mask,
            "tabular": tab_tensor,
            "time": time,
            "event": event,
        }

        # Optional EID
        if "eid" in row:
            sample["eid"] = int(row["eid"])

        return sample
