import torch, pandas as pd
import numpy as np
import sys, os

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from src.datasets.multimodal_dataset import MultimodalKidneyDataset
from src.model.multimodal_fusion import MultimodalKidneyModel


def evaluate_fold(fold, full_ds, val_idx, fold_ckpt_path, backbone_ckpt_path,
                  num_tab_features, device="cuda"):
    """Evaluate one fold checkpoint on its validation set and save metrics CSV."""
    val_ds = Subset(full_ds, val_idx)
    loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Initialise model with backbone weights
    model = MultimodalKidneyModel(
        weight_path=backbone_ckpt_path,
        num_tabular_features=num_tab_features,
        output_type="regression"
    ).to(device)

    if not os.path.exists(fold_ckpt_path):
        print(f"[!] Checkpoint not found: {fold_ckpt_path}")
        return None

    # Load the trained fold weights
    state = torch.load(fold_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            tab = batch["tabular"].to(device)
            y = batch["egfr"].to(device).unsqueeze(1)

            pred = model(img, mask, tab)
            preds.extend(pred.cpu().numpy().ravel())
            trues.extend(y.cpu().numpy().ravel())

    errors = np.array(trues) - np.array(preds)
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((np.array(trues) - np.mean(trues))**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    df = pd.DataFrame([{
        "fold": fold, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2
    }])

    os.makedirs("checkpoints_5fold", exist_ok=True)
    out_path = f"checkpoints_5fold/fold{fold}_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Load dataset once
    full_ds = MultimodalKidneyDataset(
        csv_path=os.path.join(project_root, "/home/fendy77/projects/retina-kidney-AI/data/multimodal_master_CLEAN.csv"),
        image_root="/home/fendy77/data/retina_images",
        mask_root=os.path.join(project_root, "/home/fendy77/projects/retina-kidney-AI/data/masks_raw_binary")
    )
    num_tab_features = len(full_ds.tabular_features)

    # Path to backbone weights (RETFound pretraining)
    backbone_weights = os.path.join(project_root, "/home/fendy77/projects/retina-kidney-AI/retfound/RETFound_mae_natureCFP.pth")

    # Recreate the same folds (same random_state as training)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for fold, (_, val_idx) in enumerate(kf.split(full_ds)):
        fold_ckpt = f"checkpoints_5fold/fold{fold}_best.pth"
        df = evaluate_fold(fold, full_ds, val_idx, fold_ckpt,
                           backbone_weights, num_tab_features)
        if df is not None:
            results.append(df)

    if results:
        summary = pd.concat(results, ignore_index=True)
        print("\n=== 5-Fold CV Summary ===")
        print(summary.to_string(index=False))
        print(f"\nMean RMSE: {summary['rmse'].mean():.4f} ± {summary['rmse'].std():.4f}")
        print(f"Mean MAE:  {summary['mae'].mean():.4f} ± {summary['mae'].std():.4f}")
        print(f"Mean R²:   {summary['r2'].mean():.4f} ± {summary['r2'].std():.4f}")

        summary.to_csv("checkpoints_5fold/cv_summary.csv", index=False)
        print("Saved → checkpoints_5fold/cv_summary.csv")
