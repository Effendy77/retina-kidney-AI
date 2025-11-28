import os
import argparse
import pandas as pd
import numpy as np

from survival_v3.src.train.train_survival_v3 import c_index_v3


# ------------------------------------------------------------
# Utility: Brier score at time t
# (Simple implementation for preparation; full IBS can be done later)
# ------------------------------------------------------------
def brier_score_at_time(times, events, preds, t):
    """
    Binary indicator: event before t.
    """
    y_t = ((times <= t) & (events == 1)).astype(float)
    return np.mean((preds - y_t) ** 2)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_root",
        type=str,
        default="survival_v3/experiments/single_run_5fold",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="survival_v3/experiments/single_run_5fold/eval_v3",
    )
    args = parser.parse_args()

    exp_root = args.exp_root
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f">>> Loading fold results from: {exp_root}")

    # --------------------------------------------------------
    # Load all risk score files
    # --------------------------------------------------------
    risk_files = sorted([
        os.path.join(exp_root, f)
        for f in os.listdir(exp_root)
        if f.endswith("_risk_scores.csv")
    ])

    if len(risk_files) == 0:
        raise RuntimeError("No fold risk score files found.")

    all_df = []
    for path in risk_files:
        try:
            df = pd.read_csv(path)
            df["fold"] = int(path.split("fold")[1].split("_")[0])
            all_df.append(df)
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"Warning: Failed to load {path} — {e}")

    full = pd.concat(all_df, ignore_index=True)

    # --------------------------------------------------------
    # Compute global C-index
    # --------------------------------------------------------
    times = full["time_to_event"].values
    events = full["event_occurred"].values
    preds = full["risk_score"].values

    c_global = c_index_v3(
        torch.tensor(preds),
        torch.tensor(times),
        torch.tensor(events),
    )

    print(f">>> Combined C-index: {c_global:.4f}")

    # --------------------------------------------------------
    # Compute Brier score curve
    # --------------------------------------------------------
    max_time = int(np.percentile(times, 95))
    bs_curve = []
    for t in range(1, max_time + 1):
        bs = brier_score_at_time(times, events, preds, t)
        bs_curve.append([t, bs])

    bs_df = pd.DataFrame(bs_curve, columns=["time", "brier_score"])
    bs_df.to_csv(os.path.join(out_dir, "brier_curve_v3.csv"), index=False)

    # --------------------------------------------------------
    # Save unified risk dataframe
    # --------------------------------------------------------
    full.to_csv(
        os.path.join(out_dir, "all_folds_risk_scores_v3.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------
    summary = {
        "Combined C-index": [c_global],
        "Num folds": [len(risk_files)],
        "Num samples": [len(full)],
    }
    pd.DataFrame(summary).to_csv(
        os.path.join(out_dir, "summary_v3.csv"),
        index=False,
    )

    print(f"\n>>> Evaluation artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
