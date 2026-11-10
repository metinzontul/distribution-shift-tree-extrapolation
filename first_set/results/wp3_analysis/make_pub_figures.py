"""Publication-quality regeneration of the 5 WP3 figures.
Okabe-Ito CVD-safe palette, 600 dpi, consistent legends/fonts/grid, zoomed axes with
off-scale outliers annotated (not stretching the axis), no internal jargon in titles.
Outputs to results/wp3_analysis/pub/ (inspect before overwriting canonical paths)."""
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tabred_shift_characterization")
OUT = os.path.join(HERE, "pub")
os.makedirs(OUT, exist_ok=True)

# ---- shared style ----
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
OK = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7")
GREY = "#9aa0a6"
INK = "#222222"
SEVS = [0, 15, 30, 50]
SEVCOLS = ["gap_sev0", "gap_sev15", "gap_sev30", "gap_sev50"]

cross = pd.read_csv(os.path.join(HERE, "crossing_map.csv"))
p1 = pd.read_csv(os.path.join(HERE, "p1_per_severity.csv"))
b3 = pd.read_csv(os.path.join(HERE, "b3_differential.csv"))
temp = pd.read_csv(os.path.join(HERE, "temporal_family.csv"))
sc = pd.read_csv(os.path.join(TAB, "shift_characterization.csv"))
cpd = pd.read_csv(os.path.join(TAB, "construct_per_dataset.csv"))

def offscale_note(ax, txt):
    ax.text(0.985, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, style="italic", color="#555555")

# ================= FIG 1: aggregate gap vs severity (median + IQR band) =================
fig, ax = plt.subplots(figsize=(6.4, 4.4))
G = cross[SEVCOLS].to_numpy()  # (29, 4)
med = np.median(G, axis=0); q25 = np.percentile(G, 25, axis=0); q75 = np.percentile(G, 75, axis=0)
# faint per-dataset spaghetti (context)
for row in G:
    ax.plot(SEVS, row, color=GREY, lw=0.7, alpha=0.20, zorder=1)
ax.fill_between(SEVS, q25, q75, color=OK["blue"], alpha=0.18, lw=0, zorder=2,
                label="interquartile range (25-75%)")
ax.plot(SEVS, med, color=OK["blue"], lw=2.4, marker="o", ms=7, zorder=4,
        label=f"median across {len(cross)} datasets")
ax.axhline(0.05, color=OK["orange"], lw=1.6, ls="--", zorder=3, label="noticeable threshold (0.05)")
ax.axhline(0.10, color=OK["vermillion"], lw=1.6, ls=":", zorder=3, label="substantial threshold (0.10)")
ax.axhline(0, color="#666666", lw=1.0, zorder=2)
ax.set_ylim(-0.55, 1.0)
ax.set_xticks(SEVS)
ax.set_xlabel("shift severity (train-to-test percentile gap)")
ax.set_ylabel("nOOD gap  (CatBoost - FT-Transformer)")
pass  # title removed
ax.legend(loc="upper left")
n_off = int((G[:, -1] > 1.0).sum() + (G[:, -1] < -0.55).sum())
offscale_note(ax, f"{n_off} datasets off-scale (max +4.16, min -3.60)")
fig.savefig(os.path.join(OUT, "fig_gap_severity_curves.png")); plt.close(fig)

# ================= FIG 2: per-dataset trajectories, rank-reversals highlighted =================
rr_col = [c for c in cross.columns if c.startswith("rank_reversal")][0]
cross["_rr"] = cross[rr_col].astype(str).str.strip().str.lower().eq("true")
rev = cross[cross["_rr"]].copy()
oth = cross[~cross["_rr"]].copy()
fig, ax = plt.subplots(figsize=(6.6, 4.6))
for _, r in oth.iterrows():
    ax.plot(SEVS, r[SEVCOLS].to_numpy(), color=GREY, lw=0.8, alpha=0.30, zorder=1)
hi_colors = [OK["blue"], OK["vermillion"], OK["green"], OK["orange"], OK["purple"]]
rev = rev.sort_values("gap_sev50", ascending=False)
handles = []
for i, (_, r) in enumerate(rev.iterrows()):
    c = hi_colors[i % len(hi_colors)]
    y = r[SEVCOLS].to_numpy()
    ax.plot(SEVS, y, color=c, lw=2.2, marker="o", ms=5.5, zorder=4)
    handles.append(Line2D([0], [0], color=c, lw=2.2, marker="o", ms=5.5, label=r["dataset"]))
ax.axhline(0, color="#444444", lw=1.1, zorder=2)
ax.axhline(0.05, color="#888888", lw=1.2, ls=":", zorder=2)
ax.set_ylim(-0.32, 0.42)
ax.set_xticks(SEVS)
ax.set_xlabel("shift severity (train-to-test percentile gap)")
ax.set_ylabel("nOOD gap  (CatBoost - FT-Transformer)")
pass  # title removed
grey_h = Line2D([0], [0], color=GREY, lw=0.8, alpha=0.6, label=f"other {len(oth)} datasets")
notice_h = Line2D([0], [0], color="#888888", lw=1.2, ls=":", label="noticeable (0.05)")
ax.legend(handles=[grey_h] + handles + [notice_h], loc="upper left",
          title="rank-reversal datasets", title_fontsize=9)
offscale_note(ax, "axis zoomed to the crossing region")
fig.savefig(os.path.join(OUT, "fig_per_dataset_crossing.png")); plt.close(fig)

# ================= FIG 3: B3 differential scatter (zoomed to data) =================
fig, ax = plt.subplots(figsize=(5.6, 5.2))
deg = b3["gap_extrapolation"].round(6).eq(b3["gap_saturation"].round(6))
main = b3[~deg]; degen = b3[deg]
lim = (-0.85, 1.15)
xs = np.linspace(lim[0], lim[1], 10)
ax.fill_between(xs, xs, lim[1], color=OK["blue"], alpha=0.06, zorder=0)
ax.text(-0.1, 0.95, "extrapolation-specific\n(gap larger under extrapolation)", fontsize=8.5,
        color=OK["blue"], ha="left", va="top")
ax.plot(lim, lim, color="#444444", lw=1.4, ls="--", zorder=2, label="extrapolation = saturation")
ax.axhline(0, color="#bbbbbb", lw=0.9, zorder=1); ax.axvline(0, color="#bbbbbb", lw=0.9, zorder=1)
ax.scatter(main["gap_saturation"], main["gap_extrapolation"], s=55, color=OK["blue"],
           edgecolor="white", linewidth=0.8, zorder=4, label=f"dataset (n={len(main)})")
ax.scatter(degen["gap_saturation"].clip(*lim), degen["gap_extrapolation"].clip(*lim),
           s=60, marker="s", facecolor="none", edgecolor=OK["vermillion"], linewidth=1.6,
           zorder=4, label="degenerate (same feature both splits)")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("gap under saturation-split")
ax.set_ylabel("gap under extrapolation-split")
pass  # title removed
ax.legend(loc="lower right")
ax.text(0.03, 0.985, "diamonds: off-scale at (+4.16, +4.16), on diagonal",
        transform=ax.transAxes, fontsize=8, style="italic", color="#666666", ha="left", va="top")
fig.savefig(os.path.join(OUT, "fig_b3_differential.png")); plt.close(fig)

# ================= FIG 4: temporal family bars (3 primary families) =================
order = ["cooking-time", "delivery-eta", "sberbank-housing", "weather", "air_quality", "bike_hour"]
temp = temp.set_index("dataset").loc[order].reset_index()
fams = [("catboost_nOOD", "CatBoost (GBDT)", OK["blue"]),
        ("ft_nOOD", "FT-Transformer (NN)", OK["vermillion"]),
        ("tabpfn_nOOD", "TabPFN (FM)", OK["green"])]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
x = np.arange(len(order)); w = 0.26
for j, (col, lab, c) in enumerate(fams):
    vals = temp[col].to_numpy(dtype=float)
    ax.bar(x + (j - 1) * w, np.nan_to_num(vals), width=w, color=c, label=lab,
           edgecolor="white", linewidth=0.6)
disp = {"cooking-time": "cooking", "delivery-eta": "delivery", "sberbank-housing": "sberbank",
        "weather": "weather", "air_quality": "air quality", "bike_hour": "bike hour"}
labels = [f"{disp.get(d, d)}\n({'primary' if t=='PRIMARY' else 'proxy'})"
          for d, t in zip(temp["dataset"], temp["tier"])]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.axvline(3.5, color="#cccccc", lw=1.0, ls="--")
ax.set_ylabel("median nOOD (out-of-distribution)")
pass  # title removed
ax.legend(loc="upper left", ncol=3, columnspacing=1.2)
ax.set_ylim(0, np.nanmax(temp[[f[0] for f in fams]].to_numpy(dtype=float)) * 1.18)
fig.savefig(os.path.join(OUT, "fig_temporal_family.png")); plt.close(fig)

# ================= FIG 5: shift-type characterization (3 panels, shared legend) =================
prim = ["sberbank-housing", "cooking-time", "weather", "delivery-eta"]
scx = sc.set_index("dataset")
short = {"sberbank-housing": "sberbank", "cooking-time": "cooking", "weather": "weather", "delivery-eta": "delivery"}
def constrow(col):
    return float(scx.loc[[i for i in scx.index if str(i).startswith("CONSTRUCT")][0], col])
panels = [
    ("OOR_mm_Yfeat", "construct", "A. Out-of-range on the target-driving feature",
     "out-of-range fraction (train min-max)", False, "split_OOR_mm"),
    ("OOR_mm_mean", "construct", "B. Mean out-of-range across all features",
     "out-of-range fraction (train min-max)", False, "mean_OOR_mm"),
    ("PSI_med", "construct", "C. Covariate-shift magnitude (median PSI)",
     "median per-feature PSI", True, "PSI_med"),
]
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
for ax, (col, _, title, ylab, logy, cpd_col) in zip(axes, panels):
    tvals = [float(scx.loc[d, col]) for d in prim]
    cval = constrow(col)
    xs = np.arange(len(prim) + 1)
    ax.bar(xs[:-1], tvals, width=0.62, color=OK["blue"], edgecolor="white", linewidth=0.6, zorder=3)
    ax.bar(xs[-1], cval, width=0.62, color=OK["orange"], alpha=0.55, edgecolor=OK["orange"],
           linewidth=1.0, zorder=2)
    jit = (np.random.default_rng(0).random(len(cpd)) - 0.5) * 0.34
    ax.scatter(np.full(len(cpd), xs[-1]) + jit, cpd[cpd_col], s=16, color=OK["orange"],
               edgecolor="#7a4a00", linewidth=0.4, alpha=0.85, zorder=4)
    for xi, v in zip(xs[:-1], tvals):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5, color=INK)
    if logy:
        ax.set_yscale("log")
        ax.axhline(0.25, color=OK["vermillion"], lw=1.3, ls=":", zorder=1)
        ax.text(0.02, 0.25, "PSI=0.25 (large)", fontsize=8, color=OK["vermillion"], va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([short[d] for d in prim] + [f"construct\n({len(cpd)} sets)"], fontsize=9)
    ax.set_ylabel(ylab, fontsize=10.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
handles = [Patch(facecolor=OK["blue"], label="TabReD (natural temporal shift)"),
           Patch(facecolor=OK["orange"], alpha=0.55, label=f"construct extrapolation, severity 50 (per-dataset dots; bar = median of {len(cpd)})")]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04),
           frameon=True, edgecolor="#cccccc")
pass  # suptitle removed
_unused = dict(t="Natural temporal shift is not feature-range extrapolation, unlike the construct positive test",
             fs=12.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(OUT, "fig_oor_tabred_vs_construct.png")); plt.close(fig)

print("OK wrote 5 figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
