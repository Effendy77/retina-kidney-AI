import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MultimodalSurvivalDatasetV2(Dataset):
    """
    Multimodal ESRD survival dataset (v2):
    
      • RGB retinal image
      • 1-channel vessel mask (binary)
      • Tabular features (7 selected clinical features)
      • Survival labels (time_to_event, event_occurred)
    """

    def __init__(self, csv_path: str):
        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # ------------------------------------------------------
        # Define the fixed tabular clinical features (Option 2)
        # ------------------------------------------------------
        self.tabular_features = [
            "age",
            "sex",
            "diabetes",
            "hypertension",
            "egfr",
            "qrisk3",
            "dm_htn_combined",
        ]

        # ------------------------------------------------------
        # Basic cleaning: remove rows missing any required info
        # ------------------------------------------------------
        required = (
            ["fundus_image", "vessel_mask", "time_to_event", "event_occurred"]
            + self.tabular_features
        )

        self.df = self.df.dropna(subset=required).reset_index(drop=True)

        # ------------------------------------------------------
        # Image transforms
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
            T.ToTensor(),     # converts to [0,1]
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
        # Load vessel mask (convert to 1-channel float)
        # ------------------------------------------------------
        mask_path = row["vessel_mask"]
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()     # binarize (0 or 1)

        # ------------------------------------------------------
        # Tabular features → tensor [7]
        # ------------------------------------------------------
        tab = torch.tensor(
            row[self.tabular_features].astype(float).values,
            dtype=torch.float32,
        )

        # ------------------------------------------------------
        # Survival labels
        # ------------------------------------------------------
        time = torch.tensor(row["time_to_event"], dtype=torch.float32)
        event = torch.tensor(row["event_occurred"], dtype=torch.float32)

        sample = {
            "image": image,
            "mask": mask,
            "tabular": tab,
            "time": time,
            "event": event,
        }

        # Add eid for later joining of predictions
        if "eid" in row:
            sample["eid"] = int(row["eid"])

        return sample
