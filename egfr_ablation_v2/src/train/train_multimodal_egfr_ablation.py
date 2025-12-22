# egfr_ablation_v2/src/train/train_multimodal_egfr_ablation.py

import os
import random
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from egfr_ablation_v2.src.utils.metrics_regression_ablation import rmse, mae, r2_score


# ============================================================
# Reproducibility helpers
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode can slow training but helps reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Fold-safe tabular preprocessing (fit on TRAIN only)
# ============================================================

@torch.no_grad()
def fit_tabular_preprocess(
    train_loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit per-feature mean/std on the training fold only, using missingness masks if provided.
    Returns:
        mean: (D,)
        std:  (D,)
    """
    tabs: List[torch.Tensor] = []
    miss: List[torch.Tensor] = []

    for batch in train_loader:
        x = batch["tabular"].float()
        tabs.append(x)

        if "tabular_missing" in batch and batch["tabular_missing"] is not None:
            m = batch["tabular_missing"].float()
        else:
            # fallback if dataset didn't provide mask
            m = torch.isnan(x).float()
        miss.append(m)

    X = torch.cat(tabs, dim=0)   # (N, D)
    M = torch.cat(miss, dim=0)   # (N, D) 1 = missing

    # Compute mean using only observed entries
    denom = (1.0 - M).sum(dim=0).clamp_min(1.0)
    mean = (X * (1.0 - M)).sum(dim=0) / denom

    # Impute for variance computation
    X_imp = X.clone()
    if M.any():
        mean_b = mean.unsqueeze(0).expand_as(X_imp)
        X_imp[M.bool()] = mean_b[M.bool()]

    # Standard deviation (avoid zeros)
    var = X_imp.var(dim=0, unbiased=False).clamp_min(1e-6)
    std = var.sqrt()

    return mean.to(device), std.to(device)


def preprocess_tabular(
    tab: torch.Tensor,
    tab_missing: Optional[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Apply train-fold mean-imputation + standardization to a batch.
    Args:
        tab:         (B, D)
        tab_missing: (B, D) with 1=missing, or None
        mean/std:    (D,)
    Returns:
        tab_z: (B, D)
    """
    tab = tab.float()

    if tab_missing is None:
        tab_missing = torch.isnan(tab).float()
    else:
        tab_missing = tab_missing.float()

    tab_imp = tab.clone()
    if tab_missing.any():
        mean_b = mean.unsqueeze(0).expand_as(tab_imp)
        tab_imp[tab_missing.bool()] = mean_b[tab_missing.bool()]

    tab_z = (tab_imp - mean) / std
    return tab_z


# ============================================================
# TRAIN A SINGLE FOLD
# ============================================================

def train_one_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    lr: float,
    epochs: int,
    save_path: str,
    pred_output_path: str,
    metric_output_path: str,
    seed: int = 42,
    weight_decay: float = 1e-5,
    grad_clip: Optional[float] = None,
) -> Dict:
    """
    Train for one fold, select best epoch by Val RMSE, save:
      - best model weights to save_path
      - best-epoch predictions on val to pred_output_path
      - per-epoch metric history to metric_output_path

    Returns dict with best metrics.
    """
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(pred_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metric_output_path), exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -------------------------------
    # Fit fold-safe tabular preprocess
    # -------------------------------
    tab_mean, tab_std = fit_tabular_preprocess(train_loader, device)

    best_rmse = float("inf")
    best_epoch = -1
    best_metrics = {"rmse": None, "mae": None, "r2": None}
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}] =====================")

        # ---------------------------------------
        # TRAIN
        # ---------------------------------------
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc="Training", ncols=100)
        for batch in pbar:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)

            tab_missing = batch.get("tabular_missing", None)
            if tab_missing is not None:
                tab_missing = tab_missing.to(device)

            tab = preprocess_tabular(tab, tab_missing, tab_mean, tab_std)

            target = batch["egfr"].to(device).float()
            if target.ndim > 1:
                target = target.view(-1)

            optimizer.zero_grad()
            pred = model(img, mask, tab)

            # allow model to output (B,) or (B,1)
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            elif pred.ndim > 1:
                pred = pred.view(-1)

            loss = criterion(pred, target)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")

        # ---------------------------------------
        # VALIDATION
        # ---------------------------------------
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", ncols=100):
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                tab = batch["tabular"].to(device)

                tab_missing = batch.get("tabular_missing", None)
                if tab_missing is not None:
                    tab_missing = tab_missing.to(device)

                tab = preprocess_tabular(tab, tab_missing, tab_mean, tab_std)

                target = batch["egfr"].to(device).float()
                if target.ndim > 1:
                    target = target.view(-1)

                pred = model(img, mask, tab)
                if pred.ndim == 2 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)
                elif pred.ndim > 1:
                    pred = pred.view(-1)

                val_preds.extend(pred.detach().cpu().numpy().tolist())
                val_targets.extend(target.detach().cpu().numpy().tolist())

        # ---------------------------------------
        # METRICS
        # ---------------------------------------
        fold_rmse = rmse(val_targets, val_preds)
        fold_mae = mae(val_targets, val_preds)
        fold_r2 = r2_score(val_targets, val_preds)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val RMSE:  {fold_rmse:.4f}")
        print(f"Val MAE:   {fold_mae:.4f}")
        print(f"Val R2:    {fold_r2:.4f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "r2": fold_r2,
            }
        )

        # ---------------------------------------
        # SAVE BEST MODEL
        # ---------------------------------------
        if fold_rmse < best_rmse:
            best_rmse = fold_rmse
            best_epoch = epoch
            best_metrics = {"rmse": fold_rmse, "mae": fold_mae, "r2": fold_r2}
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] New best model saved at {save_path} (epoch={best_epoch})")

    # ============================================================
    # FINAL EVALUATION OF BEST MODEL (on VAL)
    # ============================================================
    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    final_preds = []
    final_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Val Eval", ncols=100):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)

            tab_missing = batch.get("tabular_missing", None)
            if tab_missing is not None:
                tab_missing = tab_missing.to(device)

            tab = preprocess_tabular(tab, tab_missing, tab_mean, tab_std)

            target = batch["egfr"].to(device).float()
            if target.ndim > 1:
                target = target.view(-1)

            pred = model(img, mask, tab)
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            elif pred.ndim > 1:
                pred = pred.view(-1)

            final_preds.extend(pred.detach().cpu().numpy().tolist())
            final_targets.extend(target.detach().cpu().numpy().tolist())

    # Save predictions
    pd.DataFrame({"target": final_targets, "pred": final_preds}).to_csv(pred_output_path, index=False)

    # Save metric history
    pd.DataFrame(history).to_csv(metric_output_path, index=False)

    print(f"[INFO] Best epoch: {best_epoch} | Best RMSE={best_metrics['rmse']:.4f} "
          f"MAE={best_metrics['mae']:.4f} R2={best_metrics['r2']:.4f}")
    print(f"[INFO] Predictions saved → {pred_output_path}")
    print(f"[INFO] Metrics history saved → {metric_output_path}")

    return {
        "best_epoch": best_epoch,
        "best_rmse": best_metrics["rmse"],
        "best_mae": best_metrics["mae"],
        "best_r2": best_metrics["r2"],
        "save_path": save_path,
        "pred_output_path": pred_output_path,
        "metric_output_path": metric_output_path,
    }
