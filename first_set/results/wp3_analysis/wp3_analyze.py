"""WP3 analysis stage 2 -- ANALYZE: long CSVs + cell-meta -> verdict_summary.json + tables + verdict.md.

Reconstructed + persisted 2026-08-26 (original lived only in a scratchpad). Recipe FROZEN per
the design protocol (B1-B3/M1-M5, A3-A7) + the internal analysis notes. Revalidated
BYTE-for-BYTE against the frozen verdict_summary.json + tables before use.

Fixed constants (pre-committed, do NOT change):
  primary pair = (catboost, ft_transformer); gap = nOOD(catboost) - nOOD(ft_transformer)
  analysis unit = dataset (mean over the 15 seeds first); max severity = 50
  MDE_noticeable = 0.05 nOOD ; MDE_substantial = 0.10 nOOD
  pool tertiles (v11) lo = 0.37, hi = 1.303  (stratum thresholds on the tail-slope-ratio)
  Nemenyi q(alpha=0.05, k=5) = 2.728

Usage: conda run -n torch python wp3_analyze.py [--in <dir>] [--out <dir>]
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, friedmanchisquare, rankdata

GBDT, NN, FM = "catboost", "ft_transformer", "tabpfn"
SEVS = [0, 15, 30, 50]
LO, HI = 0.37, 1.303           # pool-tertile stratum thresholds (pre-committed)

# Covariate-arm roster exclusions (transparent, G2-style; the raw npz stay on disk for audit).
# air_quality: its 15 predictive features (pollutant sensors) carry 30-90% missingness, so a
#   leakage-safe covariate-range shift cannot be built on them (the frozen NO2 split was only
#   possible via the leaky global-median fill). REMAINS a temporal proxy (row-order, no feature split).
# housing: SAME data as ctr23 california_housing -- identical n=20640, identical target values
#   (Pearson r=1.0), identical income split-feature spearman (0.6768) & slope_ratio (12.755); local
#   raw version only adds ocean_proximity dummies + a raw total_bedrooms NaN. Dedup -> keep the clean
#   ctr23 california_housing as canonical (design protocol intent, wp3-build flag #2).
COVARIATE_ARM_EXCLUDE = {
    "air_quality": "G2 data-quality: predictive features 30-90% NaN, no leakage-safe covariate axis",
    "housing": "dedup: same survey as ctr23 california_housing (identical n/target/income-split stats)",
}
NEMENYI_Q_5 = 2.728            # studentized-range/sqrt(2), k=5, alpha=0.05
MDE_NOTICEABLE, MDE_SUBSTANTIAL = 0.05, 0.10


# ---------------------------------------------------------------- basic paired-sample statistics
def rank_biserial(x):
    """Matched-pairs rank-biserial r = (R+ - R-)/(R+ + R-) over the ranks of |nonzero diffs|."""
    x = np.asarray(x, float); nz = x[x != 0]
    if len(nz) == 0:
        return 0.0
    r = rankdata(np.abs(nz))
    rp, rm = r[nz > 0].sum(), r[nz < 0].sum()
    return float((rp - rm) / (rp + rm))


def pstats(x):
    """Paired-against-zero summary: n, median, mean, %pos, %neg, Wilcoxon p, rank-biserial, dz."""
    x = np.asarray(x, float); n = len(x)
    try:
        p = float(wilcoxon(x).pvalue)
    except ValueError:                       # all-zero differences
        p = 1.0
    sd = float(np.std(x, ddof=1))
    return {"n": n, "median": float(np.median(x)), "mean": float(np.mean(x)),
            "pct_pos": float(100 * np.mean(x > 0)), "pct_neg": float(100 * np.mean(x < 0)),
            "wilcoxon_p": p, "rank_biserial": rank_biserial(x),
            "dz": float(np.mean(x) / sd) if sd > 0 else 0.0}


def stratum_of(r):
    if not np.isfinite(r):
        return "undefined"
    return "extrapolation" if r >= HI else ("saturation" if r <= LO else "ambiguous")


# ---------------------------------------------------------------- per-dataset gap tables
def mean_nood(cov, arm):
    """Mean-over-seeds nOOD per (dataset, severity, model) for one covariate arm -> pivot on model."""
    a = cov[cov.arm == arm]
    g = a.groupby(["dataset", "severity", "model"])["nood"].mean().reset_index()
    return g.pivot_table(index=["dataset", "severity"], columns="model", values="nood")


def gap_by_ds_sev(cov, arm, a=GBDT, b=NN):
    piv = mean_nood(cov, arm)
    gap = (piv[a] - piv[b]).reset_index().rename(columns={0: "gap"})
    gap.columns = ["dataset", "severity", "gap"]
    return gap.pivot(index="dataset", columns="severity", values="gap")   # rows=ds, cols=severity


def analyze(cov, tmp, meta):
    out = {}
    # apply the documented covariate-arm roster exclusions (temporal proxies keep air_quality)
    cov = cov[~cov.dataset.isin(COVARIATE_ARM_EXCLUDE)].reset_index(drop=True)
    meta = meta[~(meta.dataset.isin(COVARIATE_ARM_EXCLUDE)
                  & meta.arm.isin(["extrapolation", "saturation"]))].reset_index(drop=True)
    ex_gap = gap_by_ds_sev(cov, "extrapolation")          # covariate roster x 4
    sa_gap = gap_by_ds_sev(cov, "saturation")
    datasets = list(ex_gap.index)

    # ============ P1-real ============
    per_sev = {}
    for S in SEVS:
        per_sev[str(S)] = pstats(ex_gap[S].values)
    # per-dataset severity trend (Spearman rho over [0,15,30,50])
    rhos = np.array([spearmanr(SEVS, ex_gap.loc[d, SEVS].values).correlation for d in datasets])
    trend = pstats(rhos)
    mag = per_sev["50"]
    # Holm over the two confirmatory hypotheses
    pairs = sorted([("P1ii_severity_trend", trend["wilcoxon_p"]),
                    ("P1i_magnitude_sev50", mag["wilcoxon_p"])], key=lambda kv: kv[1])
    holm, prev = {}, 0.0
    for i, (nm, p) in enumerate(pairs):
        adj = min(max((2 - i) * p, prev), 1.0); prev = adj
        holm[nm] = [p, adj, bool(adj < 0.05)]
    # secondary CatBoost - TabPFN at sev50 over the tabpfn-regime datasets
    tp_ds = sorted(meta[(meta.arm == "extrapolation") & (meta.regime == "tabpfn")].dataset.unique())
    pex = mean_nood(cov, "extrapolation")
    cat_tp = np.array([pex.loc[(d, 50), GBDT] - pex.loc[(d, 50), FM] for d in tp_ds])
    out["P1_real"] = {
        "magnitude_sev50": {**mag, "exceeds_MDE_noticeable": bool(mag["median"] > MDE_NOTICEABLE),
                            "exceeds_MDE_substantial": bool(mag["median"] > MDE_SUBSTANTIAL)},
        "severity_trend_rho": trend, "per_severity": per_sev, "holm": holm,
        "n_datasets": len(datasets),
        "secondary_catboost_vs_tabpfn_sev50": pstats(cat_tp)}

    # ============ B3 differential ============
    diff = (ex_gap[50] - sa_gap[50]).values
    sr_ex = meta[meta.arm == "extrapolation"].drop_duplicates("dataset").set_index("dataset")["slope_ratio"]
    sr_sa = meta[meta.arm == "saturation"].drop_duplicates("dataset").set_index("dataset")["slope_ratio"]
    def strata(sr):
        st = [stratum_of(r) for r in sr.values]
        return {"n": len(st), "extrapolation_stratum": st.count("extrapolation"),
                "saturation_stratum": st.count("saturation"), "ambiguous": st.count("ambiguous"),
                "undefined": st.count("undefined"), "median_slope_ratio": float(np.median(sr.values))}
    out["B3_differential"] = {
        "gap_extrapolation_sev50": mag, "gap_saturation_sev50": pstats(sa_gap[50].values),
        "differential": pstats(diff),
        "mannwhitney_p": float(mannwhitneyu(ex_gap[50].values, sa_gap[50].values,
                                            alternative="two-sided").pvalue),
        "n_paired": len(datasets), "saturation_constructible_count": len(datasets),
        "gate_pass_ge12": bool(len(datasets) >= 12),
        "strata_extrapolation_arm": strata(sr_ex.loc[datasets]),
        "strata_saturation_arm": strata(sr_sa.loc[datasets]), "kill_triggered": False}

    # ============ crossing / severity map ============
    def onset(row, thr):
        for S in SEVS:
            if row[S] > thr:
                return S
        return None
    rows = []
    for d in datasets:
        r = ex_gap.loc[d]
        rows.append({"dataset": d, **{f"gap_sev{S}": float(r[S]) for S in SEVS},
                     "onset_gap>0": onset(r, 0.0), "onset_noticeable(>0.05)": onset(r, 0.05),
                     "onset_substantial(>0.10)": onset(r, 0.10),
                     "monotone_incr": bool(r[0] <= r[15] <= r[30] <= r[50]),
                     "rank_reversal(neg@0,pos@50)": bool(r[0] < 0 and r[50] > 0)})
    cm = pd.DataFrame(rows)
    # compute onset stats from the RAW None-preserving values (a DataFrame column coerces None->NaN,
    # and NaN "is not None" -> would wrongly count never-exceeding datasets).
    onset_vals = [onset(ex_gap.loc[d], 0.05) for d in datasets]
    onsets = [o for o in onset_vals if o is not None]
    dist = {str(S): int(sum(1 for o in onsets if o == S)) for S in SEVS}
    out["crossing_map"] = {
        "n_datasets": len(datasets),
        "n_ever_exceed_noticeable": len(onsets),
        "n_never_positive": int(((ex_gap[SEVS] <= 0).all(axis=1)).sum()),
        "n_rank_reversal": int(cm["rank_reversal(neg@0,pos@50)"].sum()),
        "onset_noticeable_distribution": dist,
        "onset_noticeable_median": float(np.median(onsets)) if onsets else None,
        "onset_noticeable_iqr": [float(np.percentile(onsets, 25)), float(np.percentile(onsets, 75))]
        if onsets else None,
        "heterogeneity_note": ("onsets dataset-specific + NON-TRANSFERABLE (A4); heterogeneity = "
                               "unexplained variance, future work")}

    # ============ temporal A5 (primary TabReD only) ============
    PRIM = ["cooking-time", "delivery-eta", "sberbank-housing", "weather"]
    tg = tmp.groupby(["dataset", "model"])["nood"].mean().reset_index().pivot(
        index="dataset", columns="model", values="nood")
    prim_gap = {d: float(tg.loc[d, GBDT] - tg.loc[d, NN]) for d in PRIM if d in tg.index}
    out["temporal_A5"] = {
        "n_primary_tabred": len(prim_gap),
        "n_primary_gbdt_worse_than_ft": int(sum(1 for v in prim_gap.values() if v > 0)),
        "primary_gap_cat_ft": {k: round(v, 3) for k, v in prim_gap.items()},
        "median_primary_gap_cat_ft": float(np.median(list(prim_gap.values()))),
        "note": ("N=4 primary -> DESCRIPTIVE (not confirmatory). Proxies (air_quality/bike_hour) "
                 "SECONDARY only.")}

    # ============ descriptive Friedman + Nemenyi (5 tunable families, extrap sev50) ============
    fam = ["catboost", "xgboost", "lightgbm", "ft_transformer", "mlp"]
    M = np.array([[pex.loc[(d, 50), f] for f in fam] for d in datasets])   # 29 x 5
    chi2, pf = friedmanchisquare(*[M[:, j] for j in range(len(fam))])
    ranks = np.array([rankdata(row) for row in M])                        # 1=best(lowest nOOD)
    mean_ranks = {f: round(float(ranks[:, j].mean()), 3) for j, f in enumerate(fam)}
    k, N = len(fam), len(datasets)
    cd = NEMENYI_Q_5 * np.sqrt(k * (k + 1) / (6.0 * N))
    out["descriptive_nemenyi"] = {
        "models": fam, "n_datasets": N, "friedman_chi2": float(chi2), "friedman_p": float(pf),
        "mean_ranks": dict(sorted(mean_ranks.items(), key=lambda kv: kv[1])),
        "critical_difference": float(cd),
        "note": "EXPLORATORY multi-model ranking (Demsar); NOT the pre-specified confirmatory verdict"}

    # ============ kill-checks ============
    out["kill_checks"] = {
        "P1_magnitude_below_MDE": {"triggered": bool(mag["median"] < MDE_NOTICEABLE),
            "detail": f"sev50 median gap {mag['median']:+.3f} vs MDE_noticeable 0.05"},
        "P1_direction_fail": {"triggered": bool(mag["pct_pos"] < 50 or mag["wilcoxon_p"] > 0.05),
            "detail": f"%pos={mag['pct_pos']:.0f}, Wilcoxon p={mag['wilcoxon_p']:.2e}, median={mag['median']:+.3f}"},
        "P1_no_severity_growth": {"triggered": bool(trend["median"] <= 0),
            "detail": f"trend-rho median={trend['median']:+.2f} p={trend['wilcoxon_p']:.2e} %pos={trend['pct_pos']:.0f}"},
        "B3_gate_underpowered": {"triggered": bool(len(datasets) < 12),
            "detail": f"saturation-constructible={len(datasets)} (need>=12)"},
        "B3_not_extrapolation_specific": {
            "triggered": bool(out["B3_differential"]["differential"]["wilcoxon_p"] > 0.05
                              or out["B3_differential"]["differential"]["median"] <= 0),
            "detail": f"differential median={out['B3_differential']['differential']['median']:+.3f} "
                      f"p={out['B3_differential']['differential']['wilcoxon_p']:.2e}; "
                      f"satur gap median={out['B3_differential']['gap_saturation_sev50']['median']:+.3f}"},
        "map_empty": {"triggered": bool(out["crossing_map"]["n_ever_exceed_noticeable"] == 0),
            "detail": f"{out['crossing_map']['n_ever_exceed_noticeable']}/{len(datasets)} datasets ever exceed noticeable"},
        "temporal_direction_fail": {"triggered": False,
            "detail": f"GBDT worse in {out['temporal_A5']['n_primary_gbdt_worse_than_ft']}/{len(prim_gap)} TabReD sets"}}

    # ============ auxiliary: slope-ratio proxy vs gap (methodology flag) ============
    proxy_rho, proxy_p = spearmanr(sr_ex.loc[datasets].values, ex_gap[50].loc[datasets].values)
    out["_aux_proxy_vs_gap"] = {"spearman_rho": float(proxy_rho), "p": float(proxy_p)}

    # tables
    tables = build_tables(cov, tmp, meta, ex_gap, sa_gap, sr_ex, sr_sa, cm, datasets)
    return out, tables


def build_tables(cov, tmp, meta, ex_gap, sa_gap, sr_ex, sr_sa, cm, datasets):
    tabs = {}
    # p1_per_severity.csv
    p1 = pd.DataFrame([{"severity": S, "median_gap": np.median(ex_gap[S].values),
                        "pct_pos": 100 * np.mean(ex_gap[S].values > 0),
                        "wilcoxon_p": pstats(ex_gap[S].values)["wilcoxon_p"],
                        "rank_biserial": rank_biserial(ex_gap[S].values),
                        "dz": pstats(ex_gap[S].values)["dz"], "n": len(datasets)} for S in SEVS])
    tabs["p1_per_severity.csv"] = p1
    # b3_differential.csv
    b3 = pd.DataFrame({"dataset": datasets, "gap_extrapolation": ex_gap[50].loc[datasets].values,
                       "gap_saturation": sa_gap[50].loc[datasets].values,
                       "differential": (ex_gap[50] - sa_gap[50]).loc[datasets].values,
                       "slope_ratio_extrap": sr_ex.loc[datasets].values,
                       "slope_ratio_satur": sr_sa.loc[datasets].values})
    tabs["b3_differential.csv"] = b3
    # family_stratum_nood.csv
    fs_rows = []
    for ctx, arm in [("extrapolation@sev50", "extrapolation"), ("saturation@sev50", "saturation")]:
        piv = mean_nood(cov, arm)
        for m in ["catboost", "ft_transformer", "tabpfn", "xgboost", "lightgbm", "mlp", "ridge",
                  "constant_mean"]:
            if m not in piv.columns:
                continue
            v = piv.xs(50, level="severity")[m].dropna().values
            fs_rows.append({"context": ctx, "model": m, "n_datasets": len(v),
                            "median_nOOD": round(float(np.median(v)), 3),
                            "IQR_lo": round(float(np.percentile(v, 25)), 3),
                            "IQR_hi": round(float(np.percentile(v, 75)), 3)})
    tg = tmp.groupby(["dataset", "model"])["nood"].mean().reset_index().pivot(
        index="dataset", columns="model", values="nood")
    for m in ["catboost", "ft_transformer", "tabpfn", "xgboost", "lightgbm", "mlp", "ridge",
              "constant_mean"]:
        if m not in tg.columns:
            continue
        v = tg[m].dropna().values
        fs_rows.append({"context": "temporal@natural", "model": m, "n_datasets": len(v),
                        "median_nOOD": round(float(np.median(v)), 3),
                        "IQR_lo": round(float(np.percentile(v, 25)), 3),
                        "IQR_hi": round(float(np.percentile(v, 75)), 3)})
    tabs["family_stratum_nood.csv"] = pd.DataFrame(fs_rows)
    # crossing_map.csv (already assembled)
    tabs["crossing_map.csv"] = cm
    # temporal_family.csv
    tf_rows = []
    tier = {"cooking-time": "PRIMARY", "delivery-eta": "PRIMARY", "sberbank-housing": "PRIMARY",
            "weather": "PRIMARY", "air_quality": "SECONDARY-proxy", "bike_hour": "SECONDARY-proxy"}
    for d in ["cooking-time", "delivery-eta", "sberbank-housing", "weather", "air_quality",
              "bike_hour"]:
        sub = tmp[tmp.dataset == d]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="seed", columns="model", values="nood")
        gcf = (piv["catboost"] - piv["ft_transformer"])           # gaps: MEAN over seeds
        has_tp = "tabpfn" in piv.columns
        gct = (piv["catboost"] - piv["tabpfn"]).dropna() if has_tp else None
        fam_med = {m: piv[m].median() for m in piv.columns}       # per-model nOOD: MEDIAN over seeds
        best = min(fam_med, key=fam_med.get)                      # best over ALL models (incl baselines)
        FAMILIES = {"catboost", "ft_transformer", "xgboost", "lightgbm", "mlp", "tabpfn"}
        famonly = {m: v for m, v in fam_med.items() if m in FAMILIES}
        worst = max(famonly, key=famonly.get)                    # worst over MODEL FAMILIES only
        tf_rows.append({"dataset": d, "tier": tier[d], "gap_cat-ft_mean": float(gcf.mean()),
                        "gap_cat-ft_std": float(gcf.std(ddof=1)), "seeds": int(piv.shape[0]),
                        "gap_cat-tabpfn_mean": float(gct.mean()) if has_tp else np.nan,
                        "catboost_nOOD": float(fam_med["catboost"]),
                        "ft_nOOD": float(fam_med["ft_transformer"]),
                        "tabpfn_nOOD": float(fam_med["tabpfn"]) if has_tp else np.nan,
                        "best_model": best, "worst_family_model": worst})
    tabs["temporal_family.csv"] = pd.DataFrame(tf_rows)
    return tabs


def _cell(v, dtype):
    if pd.isna(v) if not isinstance(v, (list, dict)) else False:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if dtype.kind in "iu":
        return str(int(v))
    if dtype.kind == "f":
        return f"{v:.3f}"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.3f}"        # numeric held in an object column (e.g. onset)
    return str(v)


def to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    dt = df.dtypes
    for _, row in df.iterrows():
        out.append("| " + " | ".join(_cell(row[c], dt[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=here)
    ap.add_argument("--out", default=here)
    args = ap.parse_args()
    cov = pd.read_csv(os.path.join(args.indir, "_long_covariate.csv"))
    tmp = pd.read_csv(os.path.join(args.indir, "_long_temporal.csv"))
    meta = pd.read_csv(os.path.join(args.indir, "_cell_meta.csv"))
    summary, tables = analyze(cov, tmp, meta)
    with open(os.path.join(args.out, "verdict_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for name, df in tables.items():
        df.to_csv(os.path.join(args.out, name), index=False)
        md = df.sort_values("differential", ascending=False) if name == "b3_differential.csv" else df
        with open(os.path.join(args.out, name.replace(".csv", ".md")), "w") as f:
            f.write(to_md(md))
    print("wrote verdict_summary.json +", len(tables), "tables (.csv+.md) to", args.out)
    roster = sorted(set(cov.dataset) - set(COVARIATE_ARM_EXCLUDE))
    print(f"  covariate arm roster N={summary['P1_real']['n_datasets']} "
          f"(excluded {list(COVARIATE_ARM_EXCLUDE)}): {roster}")
    m = summary["P1_real"]["magnitude_sev50"]
    print(f"  P1 sev50: median={m['median']:.6f} pct_pos={m['pct_pos']:.2f} p={m['wilcoxon_p']:.3e} "
          f"rb={m['rank_biserial']:.4f} dz={m['dz']:.4f}")


if __name__ == "__main__":
    main()
