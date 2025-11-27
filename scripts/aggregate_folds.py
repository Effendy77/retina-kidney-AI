import pandas as pd
import glob
import os

def load_fold_metrics(folder):
    csvs = sorted(glob.glob(os.path.join(folder, "fold*_metrics.csv")))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No fold metrics found in: {folder}")

    dfs = []
    for f in csvs:
        df = pd.read_csv(f)
        df['fold'] = int(os.path.basename(f).split("_")[0].replace("fold", ""))
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


if __name__ == "__main__":
    df = load_fold_metrics("checkpoints_5fold")
    print("\nAggregated 5-fold results:\n")
    print(df)

    df.to_csv("five_fold_summary.csv", index=False)
    print("\nSaved → five_fold_summary.csv")
