"""Statistics of the second set of experiments (Tables 3 and 5, in-distribution comparison).

Usage: python analysis/second_set_stats.py
Reads second_set/results/*.csv. Per-dataset values are means over seeds; gaps are summarised
across datasets by median, 95% percentile-bootstrap CI (10,000 resamples), % positive,
two-sided Wilcoxon signed-rank p and matched-pairs rank-biserial r.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parents[1] / "second_set" / "results"
d = pd.concat([pd.read_csv(RES / "main_results.csv"), pd.read_csv(RES / "ft_id_results.csv")])
d["severity"] = d.severity.fillna(-1)
agg = d.groupby(["dataset", "kind", "severity", "model"])[["nood", "nood_full", "rmse"]].mean().reset_index()
agg["logr"] = np.log(agg.rmse)


def piv(kind, sev, col="nood"):
    return agg[(agg.kind == kind) & (agg.severity == sev)].pivot(index="dataset", columns="model", values=col)


def rank_biserial(x):
    x = x[x != 0]; r = pd.Series(np.abs(x)).rank().values
    return (r[x > 0].sum() - r[x < 0].sum()) / r.sum()


def boot_ci(x, B=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    return np.percentile(np.median(rng.choice(x, (B, len(x))), axis=1), [2.5, 97.5])


def summary(g):
    lo, hi = boot_ci(g)
    return (f"n={len(g)} median={np.median(g):+.3f} CI[{lo:+.3f},{hi:+.3f}] "
            f"pos={100*(g>0).mean():.0f}% p={wilcoxon(g).pvalue:.3f} r={rank_biserial(g.values):+.2f}")


def gap(a, b, col="nood", kind="extrap", sev=50):
    p = piv(kind, sev, col).dropna(subset=[a, b]); return p[a] - p[b]


def iqr(x):
    return f"{np.median(x):.3f} [{np.percentile(x,25):.2f}, {np.percentile(x,75):.2f}]"


if __name__ == "__main__":
    print("Table 3, ID column (median nOOD [IQR])")
    for m, v in piv("id", -1).items(): print(f"  {m:16s} {iqr(v.dropna())}")
    print("Table 3, extrapolation s=50")
    for m, v in piv("extrap", 50).items(): print(f"  {m:16s} {iqr(v.dropna())}")
    print("\nIn-distribution CatBoost - FT-Transformer")
    for col in ["nood", "nood_full", "logr"]:
        print(f"  {col:9s}", summary(gap("catboost", "ft_transformer", col, "id", -1)))
    print("\nTable 5")
    for s in [0, 15, 30, 50]:
        print(f"  s={s:2d} CatBoost-Ridge      ", summary(gap("catboost", "ridge", sev=s)))
        print(f"  s={s:2d} CatBoost-LGBMlinear ", summary(gap("catboost", "lgbm_linear", sev=s)))
    print("\nRobustness at s=50")
    for col in ["nood_full", "logr"]:
        print(f"  {col:9s} CatBoost-Ridge      ", summary(gap("catboost", "ridge", col)))
        print(f"  {col:9s} CatBoost-LGBMlinear ", summary(gap("catboost", "lgbm_linear", col)))
