import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from survival_v3.src.model.multimodal_deepsurv_v3 import MultimodalDeepSurvV3
from survival_v3.src.datasets.multimodal_survival_dataset_v3 import MultimodalSurvivalDatasetV3
from survival_v3.src.utils.shap_utils_v3 import parse_eid


def main():

    # =======================================================
    # PATHS (UPDATED FOR v3)
    # =======================================================
    csv_path  = "data/survival_multimodal_master_v3.csv"
    ckpt_path = "survival_v3/experiments/single_run/best_model.pth"
    fold_dir  = "survival_v3/experiments/single_run_5fold"
    out_dir   = "survival_v3/checkpoints_single_v3/shap_tabular_v3"

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # =======================================================
    # DEVICE
    # =======================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    # =======================================================
    # LOAD DATASET (v3)
    # =======================================================
    dataset = MultimodalSurvivalDatasetV3(csv_path)

    num_tab = len(dataset.tabular_features)
    feature_names = dataset.tabular_features   # Must match model input order

    # Build mapping: cleaned EID → dataset index
    eid_to_idx = {}
    for i in range(len(dataset)):
        eid_clean = parse_eid(dataset[i]["eid"])
        eid_to_idx[eid_clean] = i

    # =======================================================
    # LOAD MODEL
    # =======================================================
    model = MultimodalDeepSurvV3(
        weight_path="retfound/RETFound_mae_natureCFP.pth",
        num_tabular_features=num_tab,
        fusion_dim=512,
        dropout=0.2,
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print(f">>> Loaded model checkpoint: {ckpt_path}")

    # =======================================================
    # SELECT TOP RISK INDIVIDUALS FROM 5-FOLD
    # =======================================================
    pooled = []
    for k in range(1, 6):
        fp = os.path.join(fold_dir, f"fold{k}_risk_scores.csv")
        if os.path.exists(fp):
            pooled.append(pd.read_csv(fp))

    if len(pooled) == 0:
        print(">>> No fold risk scores found. Exiting.")
        return

    pooled = pd.concat(pooled)
    pooled = pooled.sort_values("risk_score", ascending=False)

    # Pick top 10 EIDs
    top_eids = []
    for raw in pooled["eid"].tolist():
        clean = parse_eid(raw)
        if clean not in top_eids:
            top_eids.append(clean)
        if len(top_eids) >= 10:
            break

    print(">>> Selected EIDs for tabular SHAP:", top_eids)

    # =======================================================
    # PREPARE TABULAR SAMPLES FOR SHAP
    # =======================================================
    tab_eval = []
    for eid in top_eids:
        if eid not in eid_to_idx:
            continue
        idx = eid_to_idx[eid]
        tab_eval.append(dataset[idx]["tabular"].numpy())

    if len(tab_eval) == 0:
        print(">>> No tabular samples found. Exiting.")
        return

    tab_eval = np.stack(tab_eval)   # [K, F]

    # =======================================================
    # BACKGROUND BASELINE (50 random samples)
    # =======================================================
    n_bg = min(50, len(dataset))
    bg_idx = np.random.choice(len(dataset), size=n_bg, replace=False)

    bg = np.stack([dataset[i]["tabular"].numpy() for i in bg_idx])

    # Use mean tabular features as background baseline
    bg_mean = bg.mean(axis=0, keepdims=True)   # [1, F]

    # =======================================================
    # DEFINE TABULAR-ONLY MODEL WRAPPER FOR SHAP
    # =======================================================
    # NOTE:
    # For tabular SHAP, image + mask are set to MEAN BASELINES
    # consistent with SHAP for image and mask.
    # =======================================================

    # Build image baseline and mask baseline
    mean_img = torch.zeros((1, 3, 224, 224), device=device)
    mean_mask = torch.zeros((1, 1, 224, 224), device=device)

    def model_tab(x_np):
        """
        x_np: NumPy array of shape [B, F]
        """
        x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
        B = x_t.size(0)

        img = mean_img.expand(B, -1, -1, -1)
        mask = mean_mask.expand(B, -1, -1, -1)

        with torch.no_grad():
            out = model(img, mask, x_t)
        return out.cpu().numpy()

    # =======================================================
    # SHAP KernelExplainer
    # =======================================================
    explainer = shap.KernelExplainer(model_tab, bg)

    print(">>> Running SHAP KernelExplainer… (nsamples=100)")
    shap_values = explainer.shap_values(tab_eval, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # =======================================================
    # SAVE SHAP RAW FILES
    # =======================================================
    np.save(os.path.join(out_dir, "tabular_shap_values.npy"), shap_values)
    np.save(os.path.join(out_dir, "tabular_shap_samples.npy"), tab_eval)

    # =======================================================
    # OPTIONAL quick preview plots (v3)
    # Main plotting done by plot_tabular_shap_v3.py
    # =======================================================
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(12,6))
    plt.bar(range(len(order)), mean_abs[order])
    plt.xticks(range(len(order)), np.array(feature_names)[order], rotation=45, ha="right")
    plt.ylabel("Mean |SHAP|")
    plt.title("Tabular SHAP (Survival_v3) — Quick Preview")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tabular_shap_barplot_v3.png"), dpi=300)
    plt.close()

    shap.summary_plot(
        shap_values, tab_eval, feature_names=feature_names,
        show=False
    )
    plt.savefig(os.path.join(out_dir, "tabular_shap_beeswarm_v3.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> Tabular SHAP saved in {out_dir}")


if __name__ == "__main__":
    main()
