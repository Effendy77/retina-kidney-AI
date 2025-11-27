import os
import json
import pandas as pd

def aggregate_5fold_results(root_dir="survival_v2/checkpoints_single_v2_5fold"):
    """
    Aggregates per-fold metrics and C-index for survival 5-fold CV (v2).
    
    Expected structure:
        root_dir/
            fold1_metrics.csv
            fold2_metrics.csv
            ...
            fold5_metrics.csv
            fold1_risk_scores.csv
            ...
            fold5_risk_scores.csv
            fold1_best.pth
            ...
    
    Output:
        - summary JSON
        - summary CSV
    """

    fold_cindex = []
    metrics_list = []

    print(">>> Aggregating survival 5-fold results from:", root_dir)

    for fold in range(1, 6):
        metrics_path = os.path.join(root_dir, f"fold{fold}_metrics.csv")

        if not os.path.exists(metrics_path):
            print(f"[Warning] Missing file: {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)

        # last row contains best val_cindex OR early-stop last epoch
        best_c = df["val_cindex"].max()
        fold_cindex.append(best_c)

        metrics_list.append({
            "fold": fold,
            "best_val_cindex": float(best_c),
            "best_epoch": int(df.loc[df["val_cindex"].idxmax(), "epoch"]),
        })

        print(f"  Fold {fold}: Best C-index = {best_c:.4f}")

    # -----------------------
    # Summary statistics
    # -----------------------
    if len(fold_cindex) == 0:
        print("[Error] No folds found.")
        return

    mean_c = float(sum(fold_cindex) / len(fold_cindex))
    std_c = float(pd.Series(fold_cindex).std())

    summary = {
        "num_folds": len(fold_cindex),
        "fold_cindex": fold_cindex,
        "mean_cindex": mean_c,
        "std_cindex": std_c,
    }

    # -----------------------
    # Save outputs
    # -----------------------
    summary_json_path = os.path.join(root_dir, "summary_cindex_v2.json")
    summary_csv_path = os.path.join(root_dir, "summary_cindex_v2.csv")

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    pd.DataFrame(metrics_list).to_csv(summary_csv_path, index=False)

    print("\n===== Aggregation Complete =====")
    print(f"Mean C-index: {mean_c:.4f}")
    print(f"Std  C-index: {std_c:.4f}")
    print(f"Saved summary JSON →  {summary_json_path}")
    print(f"Saved summary CSV  →  {summary_csv_path}")


if __name__ == "__main__":
    aggregate_5fold_results()
