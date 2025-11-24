# src/eval/eval_survival.py

import numpy as np
import torch
from tqdm import tqdm
from src.utils.metrics import concordance_index


@torch.no_grad()
def evaluate_survival_model(model, dataloader, device):
    """
    Evaluate a survival model on a dataloader:
      - returns C-index
      - returns numpy arrays of times, events, predicted logits
    """
    model.eval()
    all_times = []
    all_events = []
    all_preds = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        times = batch["time"].cpu().numpy()
        events = batch["event"].cpu().numpy()

        preds = model(images).cpu().numpy()

        all_times.append(times)
        all_events.append(events)
        all_preds.append(preds)

    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    all_preds = np.concatenate(all_preds)

    cindex = concordance_index(all_times, all_preds, all_events)
    return {
        "c_index": float(cindex),
        "times": all_times,
        "events": all_events,
        "preds": all_preds,
    }
