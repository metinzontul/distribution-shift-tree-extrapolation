"""Supplementary Tables S1-S3 (constructed feature-range arm, first set of experiments).

Usage: python analysis/supplementary_tables.py [--out analysis/supplementary]
Reads first_set/results/wp3_analysis/_long_covariate.csv and _cell_meta.csv and writes
S1.csv/.md, S2.csv/.md and S3.csv/.md. Every value is the mean over seeds of nOOD;
gap = nOOD(CatBoost) - nOOD(FT-Transformer).

  S1  nOOD of every model at s = 50 and the in-distribution gap (training-region ID test set)
  S2  saturation control: gap under the extrapolation and saturation splits at s = 50,
      their differential and the slope ratios (complete version of Table 6)
  S3  gap by severity, onset (first severity with gap > 0.05), Spearman rho and
      rank reversal (complete version of Table 7)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "first_set" / "results" / "wp3_analysis"
EXCLUDE = {"air_quality", "housing"}
SEVS = [0, 15, 30, 50]
MODELS = ["catboost", "xgboost", "lightgbm", "ft_transformer", "mlp", "ridge", "tabpfn"]
NOTICEABLE = 0.05


def fmt(x):
    return "–" if pd.isna(x) else f"{x:.3f}".replace("-", "−")


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(["Dataset"] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [v if isinstance(v, str) else fmt(v) for v in row.values]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "analysis" / "supplementary"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    cov = pd.read_csv(BASE / "_long_covariate.csv")
    cov = cov[~cov.dataset.isin(EXCLUDE)]
    meta = pd.read_csv(BASE / "_cell_meta.csv")
    meta = meta[~meta.dataset.isin(EXCLUDE)].drop_duplicates(["arm", "dataset"]).set_index(["arm", "dataset"])
    M = cov.groupby(["arm", "severity", "model", "dataset"]).nood.mean()

    def gap(arm, s):
        return M.loc[(arm, s, "catboost")] - M.loc[(arm, s, "ft_transformer")]

    order = lambda idx: sorted(idx, key=str.lower)

    # S1
    s1 = pd.DataFrame({m: M.loc[("extrapolation", 50, m)] if m in set(cov.model) else np.nan for m in MODELS})
    id_ = cov[(cov.arm == "extrapolation") & (cov.severity == 50)].groupby(["dataset", "model"]).nood_id.mean().unstack()
    s1["id_gap"] = id_["catboost"] - id_["ft_transformer"]
    s1 = s1.loc[order(s1.index)]
    s1.columns = ["CatBoost", "XGBoost", "LightGBM", "FT-Transformer", "MLP", "Ridge", "TabPFN", "ID gap"]

    # S2
    ext, sat = meta.loc["extrapolation"], meta.loc["saturation"]
    s2 = pd.DataFrame({"Shift feature (extrap.)": ext.feature, "Saturation feature": sat.feature,
                       "Gap (extrap.)": gap("extrapolation", 50), "Gap (satur.)": gap("saturation", 50)})
    s2["Differential"] = s2["Gap (extrap.)"] - s2["Gap (satur.)"]
    s2["Slope ratio (extrap.)"] = ext.slope_ratio
    s2["Slope ratio (satur.)"] = sat.slope_ratio
    s2 = s2.sort_values("Differential", ascending=False)

    # S3
    s3 = pd.DataFrame({f"s = {s}": gap("extrapolation", s) for s in SEVS})
    vals = s3.values
    onset = []
    for row in vals:
        hit = [s for s, v in zip(SEVS, row) if v > NOTICEABLE]
        onset.append(str(hit[0]) if hit else "never")
    s3["Onset"] = onset
    s3["Spearman rho"] = [spearmanr(SEVS, r).correlation for r in vals]
    s3["Reversal"] = np.where((vals[:, 0] < 0) & (s3["Onset"] != "never"), "yes", "no")
    s3 = s3.loc[order(s3.index)]

    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        df.to_csv(out / f"{name}.csv", index_label="dataset")
        (out / f"{name}.md").write_text(to_markdown(df), encoding="utf-8")
        print(f"wrote {name}: {len(df)} datasets")


if __name__ == "__main__":
    main()
