"""Does dataset SIZE confound the cat-FT extrapolation gap?
TabPFN ran on only 14/27 sets (all <=10000 rows). Test whether size explains gap.
No new experiments: uses existing b3_differential.csv (gap_extrapolation == sev50 gap)
and _cell_meta.csv (n rows, has_tabpfn). Publication-quality scatter in make_pub_figures style.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
ANA = os.path.join(ROOT, "results", "wp3_analysis")
OUT_CANON = os.path.join(ANA, "fig_size_vs_gap.png")
OUT_PUB = os.path.join(ANA, "pub", "fig_size_vs_gap.png")

# ---------- load ----------
b3 = pd.read_csv(os.path.join(ANA, "b3_differential.csv"))
meta = pd.read_csv(os.path.join(ANA, "_cell_meta.csv"))
g = (meta.groupby("dataset")
     .agg(n=("n", "first"), has_tabpfn=("has_tabpfn", "max"))
     .reset_index())
df = b3.merge(g, on="dataset", how="left").copy()
assert df["n"].notna().all(), "missing n for some dataset"
df["gap"] = df["gap_extrapolation"].astype(float)
df["log10n"] = np.log10(df["n"].astype(float))
df["ran"] = df["has_tabpfn"].astype(bool)           # TabPFN feasible/ran
df["degenerate"] = df["gap_extrapolation"].round(6).eq(df["gap_saturation"].round(6))

N = len(df)
small = df[df["ran"]]       # TabPFN ran, n<=10000
large = df[~df["ran"]]      # TabPFN did not run, n>=11934

# ---------- helpers ----------
def spearman_boot(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rhos = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if np.std(x[b]) == 0 or np.std(y[b]) == 0:
            continue
        rhos.append(stats.spearmanr(x[b], y[b]).statistic)
    rhos = np.array(rhos)
    return np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)

def report_spearman(sub, label):
    x = sub["n"].to_numpy(float); y = sub["gap"].to_numpy(float)
    r = stats.spearmanr(x, y)
    lo, hi = spearman_boot(x, y)
    print(f"[Spearman {label}] n={len(sub):2d}  rho={r.statistic:+.3f}  "
          f"p={r.pvalue:.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")
    # pearson on log10n too, for context
    rp = stats.pearsonr(sub["log10n"].to_numpy(float), y)
    print(f"             pearson(log10 n, gap) r={rp.statistic:+.3f} p={rp.pvalue:.3f}")
    return r

def report_mwu(a, b, label):
    av = a["gap"].to_numpy(float); bv = b["gap"].to_numpy(float)
    U, p = stats.mannwhitneyu(av, bv, alternative="two-sided")
    n1, n2 = len(av), len(bv)
    # rank-biserial effect size (small vs large); r in [-1,1]
    r_rb = 1.0 - 2.0 * U / (n1 * n2)   # U is U for first group (a); sign: >0 => a tends larger
    # common-language effect size (P(a>b))
    cles = U / (n1 * n2)
    print(f"[MWU {label}] small(n<=10k, ran): n={n1} median={np.median(av):+.3f} "
          f"IQR[{np.percentile(av,25):+.3f},{np.percentile(av,75):+.3f}] mean={av.mean():+.3f}")
    print(f"           large(n>=12k, not-run): n={n2} median={np.median(bv):+.3f} "
          f"IQR[{np.percentile(bv,25):+.3f},{np.percentile(bv,75):+.3f}] mean={bv.mean():+.3f}")
    print(f"           U={U:.1f}  p={p:.3f}  rank-biserial r={r_rb:+.3f}  CLES P(small>large)={cles:.3f}")
    return U, p, r_rb

print("=" * 78)
print(f"TOTAL datasets: {N}  |  TabPFN ran (n<=10000): {len(small)}  |  not-run (n>=11934): {len(large)}")
print(f"size range ran:    [{small.n.min():.0f}, {small.n.max():.0f}]")
print(f"size range not-run:[{large.n.min():.0f}, {large.n.max():.0f}]")
print(f"degenerate (extrap==satur split): {sorted(df[df.degenerate].dataset)}")
print("=" * 78)

print("\n--- Q1: Spearman(n_rows, cat-FT extrapolation gap sev50) ---")
report_spearman(df, "ALL-27")
nd = df[~df["degenerate"]]
report_spearman(nd, "drop-degenerate-25")

print("\n--- Q2: 14 (TabPFN-feasible, small) vs 13 (large) ---")
report_mwu(small, large, "ALL")
report_mwu(small[~small.degenerate], large[~large.degenerate], "drop-degenerate")

# also: median gap both groups vs the 0.05/0.10 thresholds
print(f"\nmedian gap all-27 = {df.gap.median():+.3f}; "
      f"share gap>0.05: small={ (small.gap>0.05).mean():.2f} large={ (large.gap>0.05).mean():.2f}")

# =========================== PUBLICATION FIGURE ===========================
plt.rcParams.update({
    "savefig.dpi": 600, "figure.dpi": 110, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 12, "axes.titlepad": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "legend.fontsize": 9.5, "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc", "legend.borderpad": 0.6,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.color": "#333333", "ytick.color": "#333333",
})
OK = dict(blue="#0072B2", vermillion="#D55E00", orange="#E69F00", green="#009E73")
GREY = "#9aa0a6"

fig, ax = plt.subplots(figsize=(7.0, 4.8))
YLO, YHI = -0.72, 0.78                      # zoom to the bulk; off-scale annotated
XLO, XHI = 2.84, 4.40                        # log10 rows ~ [690, 25000]
thr = np.log10(10000)                        # TabPFN feasibility boundary
ax.set_xlim(XLO, XHI)

# feasibility shading
ax.axvspan(XLO, thr, color=OK["green"], alpha=0.05, zorder=0)
ax.axvline(thr, color="#888888", lw=1.3, ls="--", zorder=2)
ax.text(thr - 0.028, YHI - 0.04, "TabPFN limit (~10k rows)",
        rotation=90, va="top", ha="right", fontsize=8.5, color="#666666")
ax.axhline(0, color="#666666", lw=1.0, zorder=2)
ax.axhline(0.05, color=OK["orange"], lw=1.3, ls="--", zorder=2)
ax.axhline(0.10, color=OK["vermillion"], lw=1.3, ls=":", zorder=2)

def plot_group(sub, color, marker, label):
    yv = sub["gap"].to_numpy(float)
    xv = sub["log10n"].to_numpy(float)
    inb = (yv >= YLO) & (yv <= YHI)
    ax.scatter(xv[inb], yv[inb], s=70, color=color, marker=marker,
               edgecolor="white", linewidth=0.8, zorder=5, label=label)
    # off-scale: clip to boundary, hollow marker + arrow
    for xi, yi, name in zip(xv[~inb], yv[~inb], sub["dataset"].to_numpy()[~inb]):
        yc = YHI - 0.015 if yi > YHI else YLO + 0.015
        ax.scatter([xi], [yc], s=80, facecolor="none", edgecolor=color,
                   marker=marker, linewidth=1.6, zorder=6)
        ax.annotate(f"{name}\n({yi:+.2f})", (xi, yc),
                    textcoords="offset points",
                    xytext=(-16, -20 if yi > YHI else 20),
                    ha="right", va="top" if yi > YHI else "bottom",
                    fontsize=7.5, color=color, linespacing=0.95,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0,
                                    connectionstyle="arc3,rad=0.2"))

plot_group(small, OK["blue"], "o", f"TabPFN ran  (n\u226410,000;  {len(small)} sets)")
plot_group(large, OK["vermillion"], "^", f"TabPFN not run  (n\u226511,934;  {len(large)} sets)")

# group-median guide lines (only within feasibility/large x-ranges)
ax.hlines(small.gap.median(), small.log10n.min(), thr, color=OK["blue"],
          lw=2.0, alpha=0.55, zorder=3)
ax.hlines(large.gap.median(), thr, large.log10n.max(), color=OK["vermillion"],
          lw=2.0, alpha=0.55, zorder=3)

r_all = stats.spearmanr(df["n"], df["gap"])
U, p = stats.mannwhitneyu(small["gap"], large["gap"], alternative="two-sided")
ax.text(0.015, 0.025,
        f"Spearman(rows, gap) \u03c1 = {r_all.statistic:+.2f}  (p = {r_all.pvalue:.2f}, n={N})\n"
        f"small vs large medians: {small.gap.median():+.2f} vs {large.gap.median():+.2f}"
        f"  (Mann\u2013Whitney p = {p:.2f})",
        transform=ax.transAxes, fontsize=8.8, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.92))

ax.set_ylim(YLO, YHI)
ax.set_xlabel("dataset size  (log\u2081\u2080 rows)")
ax.set_ylabel("extrapolation gap, severity 50  (CatBoost \u2212 FT-Transformer)")
pass  # title removed

# x ticks as readable row counts
ax.set_xticks([np.log10(1000), np.log10(3000), np.log10(10000), np.log10(20000)])
ax.set_xticklabels(["1,000", "3,000", "10,000", "20,000"])

thr_o = Line2D([0], [0], color=OK["orange"], lw=1.3, ls="--", label="noticeable (0.05)")
thr_v = Line2D([0], [0], color=OK["vermillion"], lw=1.3, ls=":", label="substantial (0.10)")
h, l = ax.get_legend_handles_labels()
ax.legend(handles=h + [thr_o, thr_v], loc="upper left", ncol=1)

for pth in (OUT_CANON, OUT_PUB):
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    assert not os.path.exists(pth) or "size_vs_gap" in pth
    fig.savefig(pth)
plt.close(fig)
print("\nwrote figure ->", OUT_CANON)
print("wrote figure ->", OUT_PUB)
