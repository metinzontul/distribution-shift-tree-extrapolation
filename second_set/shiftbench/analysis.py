"""Dataset-level statistics for the revision.

Unit of analysis = dataset (seeds are averaged first). Provides the
manuscript's statistics (median gap, % positive, two-sided Wilcoxon,
rank-biserial r, d_z, per-dataset Spearman trend, Holm) plus the additions
requested in the review:
  * bootstrap CIs for the median (resampling datasets);
  * scale-free log-RMSE-ratio gap (independent of any normaliser);
  * gap under the full-data normaliser (nood_full);
  * difference-in-differences against the ID split;
  * linear mixed-effects severity trend;
  * TOST equivalence tests (per dataset over seeds, and across datasets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ loading
def load_results(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["severity"] = df["severity"].fillna(-1.0)
    df["log_rmse"] = np.log(df["rmse"])
    return df


def per_seed_gap(df: pd.DataFrame, a: str, b: str, metric: str = "nood") -> pd.DataFrame:
    """Paired per-seed differences a - b (metric='log_rmse' gives log RMSE ratio)."""
    idx = ["dataset", "kind", "severity", "seed"]
    wa = df[df.model == a].set_index(idx)[metric]
    wb = df[df.model == b].set_index(idx)[metric]
    g = (wa - wb).dropna().rename("gap").reset_index()
    return g


def dataset_gap(df, a, b, metric="nood") -> pd.DataFrame:
    """Seed-averaged gap per (dataset, kind, severity)."""
    g = per_seed_gap(df, a, b, metric)
    return g.groupby(["dataset", "kind", "severity"], as_index=False)["gap"].mean()


# ------------------------------------------------------------- statistics
def rank_biserial(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def boot_median_ci(x, n_boot=10_000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return tuple(np.quantile(meds, [alpha / 2, 1 - alpha / 2]))


def summarize(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {"n": len(x)}
    if len(x) == 0:
        return out
    lo, hi = boot_median_ci(x)
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    out.update(median=np.median(x), q25=np.quantile(x, .25), q75=np.quantile(x, .75),
               ci_lo=lo, ci_hi=hi, pct_pos=100 * np.mean(x > 0), mean=x.mean(), sd=sd,
               dz=x.mean() / sd if sd and sd > 0 else np.nan, r_rb=rank_biserial(x))
    try:
        out["p_wilcoxon"] = stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue
    except ValueError:
        out["p_wilcoxon"] = np.nan
    return out


def holm(pvals: dict) -> dict:
    items = sorted(((p, k) for k, p in pvals.items() if np.isfinite(p)))
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ------------------------------------------------------------- analyses
def severity_table(df, a, b, metric="nood", kind="extrap") -> pd.DataFrame:
    g = dataset_gap(df[df.kind == kind], a, b, metric)
    rows = [{"severity": s, **summarize(gs["gap"])} for s, gs in g.groupby("severity")]
    return pd.DataFrame(rows)


def severity_trend(df, a, b, metric="nood", kind="extrap") -> dict:
    """Per-dataset Spearman(severity, gap) + linear mixed model
    gap ~ severity + (severity | dataset)."""
    g = dataset_gap(df[df.kind == kind], a, b, metric)
    rhos = []
    for _, gd in g.groupby("dataset"):
        if gd["severity"].nunique() >= 3 and gd["gap"].std() > 0:
            rhos.append(stats.spearmanr(gd["severity"], gd["gap"]).statistic)
    out = {"spearman": summarize(np.array(rhos))}
    try:
        import warnings
        import statsmodels.formula.api as smf
        gg = g.assign(sev=g["severity"] / 50.0)
        _w = warnings.catch_warnings()
        _w.__enter__()
        warnings.simplefilter("ignore")
        try:
            fit = smf.mixedlm("gap ~ sev", gg, groups=gg["dataset"], re_formula="~sev").fit(reml=True)
        except Exception:
            fit = smf.mixedlm("gap ~ sev", gg, groups=gg["dataset"]).fit(reml=True)
        finally:
            _w.__exit__(None, None, None)
        out["mixed"] = {"slope_per_50pp": fit.params["sev"], "se": fit.bse["sev"],
                        "p": fit.pvalues["sev"], "intercept": fit.params["Intercept"]}
    except Exception as ex:
        out["mixed"] = {"error": str(ex)}
    return out


def specificity(df, diag, a, b, metric="nood", k1="extrap", k2="saturation", severity=50.0):
    """Paired (within-dataset) comparison of the gap under two split designs.
    Datasets where both splits use the same feature+direction are structural
    ties and are reported but excluded from the paired test."""
    g = dataset_gap(df[df.kind.isin([k1, k2]) & (df.severity == severity)], a, b, metric)
    w = g.pivot(index="dataset", columns="kind", values="gap").dropna()
    d = (w[k1] - w[k2])
    degenerate = []
    if diag is not None and "degenerate" in diag.columns:
        dd = diag[(diag.kind == k2) & (diag.severity == severity)]
        degenerate = dd.loc[dd["degenerate"].astype(str) == "True", "dataset"].tolist()
    d_eff = d.drop(index=[x for x in degenerate if x in d.index])
    out = {"paired": summarize(d_eff.to_numpy()), "degenerate": degenerate,
           "median_gap_" + k1: np.median(w[k1]), "median_gap_" + k2: np.median(w[k2])}
    try:
        out["p_mannwhitney"] = stats.mannwhitneyu(w[k1], w[k2], alternative="two-sided").pvalue
    except ValueError:
        out["p_mannwhitney"] = np.nan
    return out, d


def diff_in_diff(df, a, b, metric="nood", kind="extrap", severity=50.0):
    """gap(kind, severity) - gap(id): shift-specific part of the gap."""
    g = dataset_gap(df, a, b, metric)
    s = g[(g.kind == kind) & (g.severity == severity)].set_index("dataset")["gap"]
    i = g[g.kind == "id"].set_index("dataset")["gap"]
    d = (s - i).dropna()
    return {"id_gap": summarize(i.to_numpy()), "did": summarize(d.to_numpy())}, d


def tost(x, margin: float) -> dict:
    """Two one-sided t-tests for H0: |mean| >= margin. Equivalent if p_tost < alpha."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return {"n": len(x), "mean": float(np.mean(x)) if len(x) else np.nan, "p_tost": np.nan}
    p_low = stats.ttest_1samp(x, -margin, alternative="greater").pvalue
    p_up = stats.ttest_1samp(x, margin, alternative="less").pvalue
    ci = stats.t.interval(0.90, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    return {"n": len(x), "mean": x.mean(), "ci90_lo": ci[0], "ci90_hi": ci[1],
            "p_tost": max(p_low, p_up)}


def equivalence(df, a, b, margin=0.05, metric="nood", kind="temporal") -> pd.DataFrame:
    """Per-dataset TOST over seeds plus an across-dataset TOST on seed means."""
    g = per_seed_gap(df[df.kind == kind], a, b, metric)
    rows = [{"dataset": ds, **tost(gd["gap"], margin)} for ds, gd in g.groupby("dataset")]
    means = g.groupby("dataset")["gap"].mean()
    rows.append({"dataset": "ACROSS-DATASETS", **tost(means.to_numpy(), margin)})
    return pd.DataFrame(rows)


def model_levels(df, metric="nood") -> pd.DataFrame:
    """Median [IQR] of seed-averaged metric per model and split context (Table 2)."""
    m = df.groupby(["dataset", "kind", "severity", "model"], as_index=False)[metric].mean()
    out = m.groupby(["kind", "severity", "model"])[metric].agg(
        median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75), n="count")
    return out.reset_index()


# ------------------------------------------------------------ report
def _fmt(df: pd.DataFrame, digits=3) -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "–"
            return f"{v:.2e}" if (abs(v) < 1e-3 and v != 0) else f"{v:.{digits}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def full_report(df, diag=None, pairs=(("catboost", "ft_transformer"),), max_sev=50.0,
                margin=0.05) -> str:
    keep = ["severity", "n", "median", "ci_lo", "ci_hi", "pct_pos", "p_wilcoxon", "r_rb", "dz"]
    out = ["# shiftbench analysis report\n"]
    kinds = set(df.kind)
    out.append("## Model levels (median nOOD, seed-averaged)\n")
    out.append(_fmt(model_levels(df)))
    primary_p = {}
    for a, b in pairs:
        out.append(f"\n## Gap: {a} − {b}\n")
        for kind in [k for k in ["extrap", "reverse", "plateau", "saturation", "saturation_robust",
                                 "saturation_matched", "synthetic"] if k in kinds]:
            for metric, label in [("nood", "nOOD (train-region σ)"), ("nood_full", "nOOD (full-data σ)"),
                                  ("log_rmse", "log RMSE ratio (scale-free)")]:
                t = severity_table(df, a, b, metric, kind)
                if len(t):
                    out.append(f"\n**{kind} — {label}**\n")
                    out.append(_fmt(t[[c for c in keep if c in t.columns]]))
            tr = severity_trend(df, a, b, "nood", kind)
            sp = tr["spearman"]
            out.append(f"\nPer-dataset Spearman trend ({kind}): median {sp.get('median', np.nan):.2f}, "
                       f"{sp.get('pct_pos', np.nan):.0f}% positive, p = {sp.get('p_wilcoxon', np.nan):.2g}. "
                       f"Mixed model: {tr['mixed']}")
            if kind == "extrap":
                t = severity_table(df, a, b, "nood", kind)
                if (t.severity == max_sev).any():
                    primary_p[f"{a}-{b} gap@{max_sev:g}"] = float(t.loc[t.severity == max_sev, "p_wilcoxon"].iloc[0])
                primary_p[f"{a}-{b} trend"] = sp.get("p_wilcoxon", np.nan)
        for k2 in [k for k in ["saturation", "saturation_robust", "saturation_matched", "plateau",
                               "reverse"] if k in kinds and "extrap" in kinds]:
            res, _ = specificity(df, diag, a, b, "nood", "extrap", k2, max_sev)
            p = res["paired"]
            out.append(f"\n**Specificity extrap vs {k2} (sev {max_sev:g})**: paired median diff "
                       f"{p.get('median', np.nan):.3f} [{p.get('ci_lo', np.nan):.3f}, {p.get('ci_hi', np.nan):.3f}], "
                       f"n = {p['n']}, Wilcoxon p = {p.get('p_wilcoxon', np.nan):.3g}, r = {p.get('r_rb', np.nan):.2f}; "
                       f"Mann-Whitney p = {res['p_mannwhitney']:.3g}; degenerate: {res['degenerate']}")
            primary_p[f"{a}-{b} specificity vs {k2}"] = p.get("p_wilcoxon", np.nan)
        if "id" in kinds and "extrap" in kinds:
            res, _ = diff_in_diff(df, a, b, "nood", "extrap", max_sev)
            out.append(f"\n**ID reference**: median ID gap {res['id_gap'].get('median', np.nan):.3f}; "
                       f"diff-in-diff (extrap@{max_sev:g} − ID) median {res['did'].get('median', np.nan):.3f} "
                       f"[{res['did'].get('ci_lo', np.nan):.3f}, {res['did'].get('ci_hi', np.nan):.3f}], "
                       f"p = {res['did'].get('p_wilcoxon', np.nan):.3g}")
        if "temporal" in kinds:
            out.append(f"\n**Temporal equivalence (TOST, margin ±{margin})**\n")
            out.append(_fmt(equivalence(df, a, b, margin)))
    if primary_p:
        out.append("\n## Holm-adjusted p-values (all primary tests above)\n")
        adj = holm(primary_p)
        out.append(_fmt(pd.DataFrame([{"test": k, "p_raw": v, "p_holm": adj.get(k, np.nan)}
                                      for k, v in primary_p.items()])))
    return "\n".join(out) + "\n"
