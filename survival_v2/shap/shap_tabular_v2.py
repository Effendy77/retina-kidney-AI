import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")           # IMPORTANT for saving PNGs in headless Linux
import matplotlib.pyplot as plt

from survival_v2.src.model.multimodal_deepsurv_v2 import MultimodalDeepSurvV2
from survival_v2.src.datasets.multimodal_survival_dataset_v2 import MultimodalSurvivalDatasetV2
from survival_v2.src.utils.shap_utils_v2 import parse_eid


def main():

    # ===============================
    # PATHS
    # ===============================
    csv_path = "data/survival_multimodal_master.csv"
    ckpt_path = "survival_v2/checkpoints_single_v2/best_model.pth"
    fold_dir = "survival_v2/checkpoints_single_v2_5fold"
    out_dir = "survival_v2/checkpoints_single_v2/shap_tabular_v2"

    # Validate paths exist
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ===============================
    # DEVICE
    # ===============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    # ===============================
    # LOAD DATASET
    # ===============================
    dataset = MultimodalSurvivalDatasetV2(csv_path)

    # The dataset defines: dataset.tabular_features → list of names
    num_tab = len(dataset.tabular_features)
    feature_names = dataset.tabular_features

    # Build mapping: EID → dataset index
    eid_to_idx = {}
    for i in range(len(dataset)):
        eid_raw = dataset[i]["eid"]
        eid_clean = parse_eid(eid_raw)
        eid_to_idx[eid_clean] = i

    # ===============================
    # LOAD MODEL
    # ===============================
    model = MultimodalDeepSurvV2(
        weight_path="retfound/RETFound_mae_natureCFP.pth",
        num_tabular_features=num_tab,
        fusion_dim=512,
        dropout=0.2,
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print(f">>> Loaded model from {ckpt_path}")

    # ===============================
    # SELECT TOP EIDs FROM 5-FOLD RISK
    # ===============================
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

    # Pick 10 EIDs
    top_eids_raw = pooled["eid"].tolist()
    top_eids = []
    for e in top_eids_raw:
        clean = parse_eid(e)
        if clean not in top_eids:
            top_eids.append(clean)
        if len(top_eids) >= 10:
            break

    print(">>> Selected 10 EIDs for tabular SHAP:", top_eids)

    # ===============================
    # PREPARE SAMPLES
    # ===============================
    tab_eval = []
    for eid in top_eids:
        if eid not in eid_to_idx:
            continue
        idx = eid_to_idx[eid]
        sample = dataset[idx]
        tab_eval.append(sample["tabular"].numpy())

    if len(tab_eval) == 0:
        print(">>> No tabular samples found → exiting.")
        return

    tab_eval = np.stack(tab_eval)        # [K, F]

    # background: random samples (use mean for realistic baseline)
    n_bg = min(50, len(dataset))
    bg_idx = np.random.choice(len(dataset), size=n_bg, replace=False)
    bg = []
    for i in bg_idx:
        s = dataset[i]
        bg.append(s["tabular"].numpy())
    bg = np.stack(bg)
    
    # Use mean tabular as baseline (more realistic than zeros)
    mean_tab = bg.mean(axis=0, keepdims=True)  # [1, F]

    # ===============================
    # DEFINE TABULAR-ONLY MODEL WRAPPER
    # ===============================
    def model_tab(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
        B = x_t.size(0)
        # Use mean image & mask baselines (consistent with other SHAP scripts)
        mean_img = torch.zeros((B, 3, 224, 224), device=device)  # or load from dataset mean
        mean_mask = torch.zeros((B, 1, 224, 224), device=device)
        with torch.no_grad():
            out = model(mean_img, mean_mask, x_t)
        return out.cpu().numpy()

    # ===============================
    # SHAP KernelExplainer (model-agnostic)
    # ===============================
    explainer = shap.KernelExplainer(model_tab, bg)

    print(">>> Running SHAP KernelExplainer…")
    print(f">>> Background samples: {bg.shape[0]}, Evaluation samples: {tab_eval.shape[0]}")

    # Reduce nsamples for speed (200 is high for KernelExplainer)
    shap_values = explainer.shap_values(tab_eval, nsamples=100)

    if isinstance(shap_values, list):       # if list returned
        shap_values = shap_values[0]
    
    # Validate output shape
    if shap_values.shape != tab_eval.shape:
        print(f"[WARNING] SHAP shape mismatch: {shap_values.shape} vs eval {tab_eval.shape}")

    # ===============================
    # SAVE RAW SHAP
    # ===============================
    np.save(os.path.join(out_dir, "tabular_shap_values.npy"), shap_values)
    np.save(os.path.join(out_dir, "tabular_shap_samples.npy"), tab_eval)

    # ===============================
    # BAR PLOT (mean abs SHAP)
    # ===============================
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(order)), mean_abs[order])
    plt.xticks(range(len(order)), np.array(feature_names)[order], rotation=45, ha='right')
    plt.ylabel("Mean |SHAP|")
    plt.title("Mean |SHAP| for Tabular Features")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tabular_shap_barplot_v2.png"), dpi=300)
    plt.close()

    # ===============================
    # BEESWARM PLOT
    # ===============================
    shap.summary_plot(
        shap_values, tab_eval, feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tabular_shap_beeswarm_v2.png"), dpi=300)
    plt.close()

    print(f">>> Tabular SHAP complete → {out_dir}")


if __name__ == "__main__":
    main()
