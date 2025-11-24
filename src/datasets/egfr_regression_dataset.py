# src/datasets/egfr_regression_dataset.py

import os
import pandas as pd
from PIL import Image
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset


class EGFRRegressionDataset(Dataset):
    """
    Dataset for eGFR regression from retinal images.

    Expected CSV format:
        image_path, egfr
        1001_21015_0.0.png, 92.3
        1002_21015_0.0.png, 74.1
        ...

    Parameters
    ----------
    csv_path : str
        Path to metadata CSV.
    image_root : str
        Directory containing images.
    transform : Callable
        Torchvision transform pipeline.
    image_col : str
        CSV column containing filenames.
    label_col : str
        CSV column containing target eGFR values.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform: Optional[Callable] = None,
        image_col: str = "image_path",
        label_col: str = "egfr",
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row[self.image_col]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        egfr_value = float(row[self.label_col])
        egfr_value = torch.tensor(egfr_value, dtype=torch.float32)

        return {
            "image": image,
            "label": egfr_value,
        }
