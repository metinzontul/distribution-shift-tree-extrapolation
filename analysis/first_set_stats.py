"""Additional statistics of the first set of experiments (Table 4 CIs, Holm, mixed model,
in-distribution comparison, difference-in-differences, log-ratio check, TOST).

Usage: python analysis/first_set_stats.py
Reads first_set/results/wp3_analysis/_long_covariate.csv and _long_temporal.csv.
"""
from pathlib import Path
import warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, spearmanr, rankdata
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1] / "first_set" / "results" / "wp3_analysis"
EXCLUDE = {"air_quality", "housing"}   # see first_set/cwr and the paper, Section 3.3
SEVS = [0, 15, 30, 50]


def rank_biserial(x):
    x = np.asarray(x); nz = x[x != 0]; r = rankdata(np.abs(nz))
    return (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()


def boot_ci(x, B=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    return np.percentile(np.median(rng.choice(x, (B, len(x))), axis=1), [2.5, 97.5])


def summary(g, name):
    lo, hi = boot_ci(g)
    print(f"{name:28s} n={len(g)} median={np.median(g):+.3f} CI[{lo:+.3f},{hi:+.3f}] "
          f"pos={100*np.mean(g>0):.1f}% p={wilcoxon(g).pvalue:.4f} r={rank_biserial(g):+.2f} "
          f"dz={np.mean(g)/np.std(g, ddof=1):+.2f}")


cov = pd.read_csv(BASE / "_long_covariate.csv")
cov = cov[~cov.dataset.isin(EXCLUDE)]
ex = cov[cov.arm == "extrapolation"]
M = ex.groupby(["dataset", "severity", "model"])["nood"].mean().unstack("model")
G = (M["catboost"] - M["ft_transformer"]).unstack("severity")[SEVS]
print(f"datasets: {len(G)}")
for s in SEVS:
    summary(G[s].values, f"Table 4, s={s}")

I = ex[ex.severity == 50].groupby(["dataset", "model"]).nood_id.mean().unstack()
gid = (I["catboost"] - I["ft_transformer"]).reindex(G.index)
summary(gid.values, "ID gap (training region)")
summary((G[50] - gid).values, "DiD: s=50 minus ID")

L = ex.assign(l=np.log(ex.nood)).groupby(["dataset", "severity", "model"]).l.mean().unstack()
summary((L["catboost"] - L["ft_transformer"]).xs(50, level="severity").values, "log RMSE ratio, s=50")

# Holm over the three primary tests
rhos = [spearmanr(SEVS, G.loc[d].values).correlation for d in G.index]
sa = cov[cov.arm == "saturation"].groupby(["dataset", "severity", "model"]).nood.mean().unstack()
gs = (sa["catboost"] - sa["ft_transformer"]).xs(50, level="severity").reindex(G.index)
ps = sorted([("severity trend", wilcoxon(rhos).pvalue), ("gap at s=50", wilcoxon(G[50]).pvalue),
             ("specificity contrast", wilcoxon(G[50] - gs).pvalue)], key=lambda t: t[1])
prev = 0.0
for i, (nm, p) in enumerate(ps):
    adj = min(max((3 - i) * p, prev), 1.0); prev = adj
    print(f"Holm {nm:22s} p={p:.5f} p_holm={adj:.5f}")

# mixed model, dataset-level gaps, random intercept and slope
D = G.stack().rename("gap").reset_index(); D["sev10"] = D.severity / 10
for label, data in [("all", D), ("without diamonds, video_transcoding",
                                   D[~D.dataset.isin(["diamonds", "video_transcoding"])])]:
    r = smf.mixedlm("gap ~ sev10", data, groups=data.dataset, re_formula="~sev10").fit(
        reml=True, method=["lbfgs", "powell"])
    lo, hi = r.conf_int().loc["sev10"]
    print(f"mixed model ({label}): beta1={r.params['sev10']:.4f} per 10 pp, 95% CI [{lo:.4f}, {hi:.4f}]")
slopes = [np.polyfit(np.array(SEVS) / 10, G.loc[d].values, 1)[0] for d in G.index]
summary(np.array(slopes), "per-dataset OLS slope")

# TOST on the temporal arm (margin +-0.05, per-seed gaps)
t = pd.read_csv(BASE / "_long_temporal.csv")
p = t.pivot_table(index=["dataset", "seed"], columns="model", values="nood")
g = (p.catboost - p.ft_transformer).rename("g").reset_index()
for d, x in g.groupby("dataset"):
    x = x.g.values; n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    p_tost = max(1 - stats.t.cdf((m + 0.05) / se, n - 1), stats.t.cdf((m - 0.05) / se, n - 1))
    h = stats.t.ppf(0.95, n - 1) * se
    print(f"TOST {d:18s} mean={m:+.4f} 90% CI [{m-h:+.3f}, {m+h:+.3f}] p={p_tost:.2e}")
