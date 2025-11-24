# src/datasets/fundus_survival_dataset.py

import os
import pandas as pd
from PIL import Image
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


class FundusSurvivalDataset(Dataset):
    """
    Generic survival dataset for fundus images.

    Expects a CSV with at least columns:
      - image_path : path to the image (absolute or relative to image_root)
      - time       : follow-up time
      - event      : 1 if event occurred, 0 if censored

    You can add more columns later (e.g., clinical covariates).
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform: Optional[Callable] = None,
        time_col: str = "time",
        event_col: str = "event",
        image_col: str = "image_path",
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.time_col = time_col
        self.event_col = event_col
        self.image_col = image_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = row[self.image_col]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        time = torch.tensor(row[self.time_col], dtype=torch.float32)
        event = torch.tensor(row[self.event_col], dtype=torch.float32)

        return {
            "image": image,
            "time": time,
            "event": event,
        }
