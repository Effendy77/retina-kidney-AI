# src/data/collate.py

from typing import List, Dict
import torch


def survival_collate_fn(batch: List[Dict]):
    """
    Collate function for FundusSurvivalDataset.

    batch: list of dicts with keys: 'image', 'time', 'event'
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    times = torch.stack([item["time"] for item in batch], dim=0)
    events = torch.stack([item["event"] for item in batch], dim=0)
    return {
        "image": images,
        "time": times,
        "event": events,
    }

