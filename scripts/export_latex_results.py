import pandas as pd

df = pd.read_csv("five_fold_summary.csv")

mean = df.mean()
std = df.std()

latex = f"""
\\begin{{table}}[ht]
\\centering
\\begin{{tabular}}{{lccc}}
\\hline
Metric & Mean & SD \\\\ 
\\hline
MAE  & {mean['mae']:.3f} & {std['mae']:.3f} \\\\
RMSE & {mean['rmse']:.3f} & {std['rmse']:.3f} \\\\
MSE  & {mean['mse']:.3f} & {std['mse']:.3f} \\\\
\\hline
\\end{{tabular}}
\\caption{{5-Fold Cross-Validation Performance Summary}}
\\end{{table}}
"""

with open("five_fold_results.tex", "w") as f:
    f.write(latex)

print("Saved LaTeX table → five_fold_results.tex")
