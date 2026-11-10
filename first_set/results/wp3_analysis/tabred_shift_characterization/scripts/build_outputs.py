"""Build the characterization tables (.csv + .md) and the out-of-range comparison figure."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[4])
OUT = os.path.join(ROOT, "results", "wp3_analysis", "tabred_shift_characterization")
TAB = json.load(open(os.path.join(OUT, "tabred_metrics.json")))
CON = json.load(open(os.path.join(OUT, "construct_metrics.json")))

SETS = ["sberbank-housing", "cooking-time", "weather", "delivery-eta"]
con_inc = {k: v for k, v in CON.items() if v.get("included")}

# ---- construct summary distributions ----
c_split = np.array([v["split_feature_oor"]["oor_mm"] for v in con_inc.values()])
c_maxf = np.array([v["oor_minmax"]["max_feat_rate"] for v in con_inc.values()])
c_mean = np.array([v["oor_minmax"]["mean_rate"] for v in con_inc.values()])
c_meanpp = np.array([v["oor_p1p99"]["mean_rate"] for v in con_inc.values()])
c_psi = np.array([v["covariate_shift"]["psi_median"] for v in con_inc.values()
                  if v["covariate_shift"]["psi_median"] is not None])


def med(a):
    return float(np.median(a))


# ============================== MAIN TABLE ==============================
hdr = ["dataset", "n_test", "F_num", "OOR_mm_Yfeat", "OOR_mm_maxfeat", "OOR_mm_mean",
       "OOR_p1p99_mean", "PSI_med", "SMD_med", "Y_SMD", "Y_PSI", "Y_stdratio", "ridge_ood/tr"]
rows = []
for s in SETS:
    t = TAB[s]
    rows.append([
        s, t["n_test"], t["F_used"],
        round(t["top_spearman_feature"]["oor_mm"], 3),
        round(t["oor_minmax"]["max_feat_rate"], 3),
        round(t["oor_minmax"]["mean_rate"], 4),
        round(t["oor_p1p99"]["mean_rate"], 4),
        round(t["covariate_shift"]["psi_median"], 3),
        round(t["covariate_shift"]["smd_median"], 3),
        round(t["y_shift"]["smd_Y"], 3),
        round(t["y_shift"]["psi_Y"], 3),
        round(t["y_shift"]["std_ratio_Y"], 3),
        round(t["ridge_probe"]["ood_over_train"], 2),
    ])
# construct reference row (median over the leakage-safe construct roster; Y-feat == split feature)
rows.append([
    "CONSTRUCT sev50 (median, n=%d)" % len(con_inc), "-", "-",
    round(med(c_split), 3), round(med(c_maxf), 3), round(med(c_mean), 4),
    round(med(c_meanpp), 4), round(med(c_psi), 3), "-", "-", "-", "-", "-"])

with open(os.path.join(OUT, "shift_characterization.csv"), "w", newline="") as f:
    import csv
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(rows)


def md_table(hdr, rows):
    out = "| " + " | ".join(hdr) + " |\n"
    out += "| " + " | ".join("---" for _ in hdr) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


md = "# TabReD natural temporal shift -- characterization (post-hoc / exploratory)\n\n"
md += ("> Numeric (x_num) features only; authored `default` temporal split (fit=train period, "
       "OOD=test period). OOR = out-of-range fraction of OOD-test cells vs train support. "
       "`_mm` = train [min,max]; `_p1p99` = train [p1,p99]. `Yfeat` = the single most Y-correlated "
       "(max |Spearman|) feature = the mechanism's designed extrapolation axis; the construct arm "
       "forces this feature to 100% out-of-range at sev50. PSI/SMD = per-feature covariate-shift "
       "magnitude (median over features). Y_* = target/prior shift. ridge_ood/tr = crude linear "
       "concept-drift probe (OOD RMSE / train RMSE).\n\n")
md += md_table(hdr, rows) + "\n"

# ---- covariate-shift + Y-shift detail table ----
hdr2 = ["dataset", "PSI_mean", "PSI_max", "frac_PSI>0.25", "SMD_max", "frac_SMD>0.5",
        "Y_mean_tr", "Y_mean_te", "Y_q50_shift_std", "Y_q90_shift_std", "Y_KS"]
rows2 = []
for s in SETS:
    t = TAB[s]; cs = t["covariate_shift"]; ys = t["y_shift"]
    rows2.append([s, round(cs["psi_mean"], 3), round(cs["psi_max"], 2),
                  round(cs["frac_psi_over025"], 3), round(cs["smd_max"], 2),
                  round(cs["frac_smd_over05"], 3), round(ys["mean_Y_train"], 3),
                  round(ys["mean_Y_test"], 3), round(ys["q50_shift_std"], 3),
                  round(ys["q90_shift_std"], 3), round(ys["ks_Y"], 3)])
md += "## Covariate-shift magnitude and target/prior shift (detail)\n\n"
md += md_table(hdr2, rows2) + "\n"

# ---- construct per-dataset appendix ----
hdr3 = ["dataset", "F", "abs_spearman", "split_OOR_mm", "maxfeat_OOR_mm", "mean_OOR_mm", "PSI_med"]
rows3 = []
for k, v in sorted(con_inc.items(), key=lambda kv: -kv[1]["oor_minmax"]["mean_rate"]):
    rows3.append([k, v["F_used"], round(v["abs_spearman"], 2),
                  round(v["split_feature_oor"]["oor_mm"], 3),
                  round(v["oor_minmax"]["max_feat_rate"], 3),
                  round(v["oor_minmax"]["mean_rate"], 4),
                  round(v["covariate_shift"]["psi_median"], 3) if v["covariate_shift"]["psi_median"] is not None else "-"])
md += ("## Construct extrapolation-arm sev50 out-of-range, per dataset (calibration; n=%d)\n\n"
       % len(con_inc))
md += md_table(hdr3, rows3) + "\n"
with open(os.path.join(OUT, "construct_per_dataset.csv"), "w", newline="") as f:
    import csv
    w = csv.writer(f); w.writerow(hdr3); w.writerows(rows3)

with open(os.path.join(OUT, "shift_characterization.md"), "w", encoding="utf-8") as f:
    f.write(md)
print("WROTE tables: shift_characterization.{csv,md}, construct_per_dataset.csv")

# ============================== FIGURE ==============================
labels = ["sberbank", "cooking", "weather", "delivery"]
y_split = [TAB[s]["top_spearman_feature"]["oor_mm"] for s in SETS]
y_mean = [TAB[s]["oor_minmax"]["mean_rate"] for s in SETS]
y_psi = [TAB[s]["covariate_shift"]["psi_median"] for s in SETS]

BLUE = "#3b6fb0"
ORANGE = "#e08214"
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))

def tabred_construct_panel(a, tvals, cpoints, cmed, title, ylab, ymax=None, logy=False):
    x = np.arange(len(labels))
    a.bar(x, tvals, color=BLUE, width=0.7, label="TabReD (natural temporal)")
    for xi, v in zip(x, tvals):
        a.text(xi, v + (0.012 if not logy else v * 0.1 + 1e-4),
               f"{v:.3f}" if v < 0.1 else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    cx = len(labels) + 0.4
    # construct: median bar + jittered scatter of per-dataset values
    a.bar([cx], [cmed], color=ORANGE, width=0.7, alpha=0.45,
          label="construct sev50 median (n=%d)" % len(cpoints))
    jit = (np.random.default_rng(0).random(len(cpoints)) - 0.5) * 0.5
    a.scatter(cx + jit, cpoints, s=14, color=ORANGE, edgecolor="k", linewidth=0.3, zorder=5)
    a.set_xticks(list(x) + [cx])
    a.set_xticklabels(labels + ["construct\n(%d sets)" % len(cpoints)], fontsize=9)
    a.set_title(title, fontsize=11)
    a.set_ylabel(ylab, fontsize=10)
    if logy:
        a.set_yscale("log")
    elif ymax:
        a.set_ylim(0, ymax)
    a.axvline(len(labels) - 0.35, color="0.7", ls="--", lw=0.8)
    a.grid(axis="y", alpha=0.3)

tabred_construct_panel(ax[0], y_split, c_split, med(c_split),
                       "A. OOR on the most Y-relevant feature\n(mechanism's designed axis)",
                       "out-of-range rate  [train min,max]", ymax=1.15)
ax[0].legend(fontsize=7.5, loc="center left")
tabred_construct_panel(ax[1], y_mean, c_mean, med(c_mean),
                       "B. Mean per-cell OOR (all numeric features)",
                       "mean out-of-range rate  [train min,max]", ymax=None)
ax[1].set_ylim(0, max(0.85, max(c_mean) * 1.1))
tabred_construct_panel(ax[2], y_psi, c_psi, med(c_psi),
                       "C. Covariate-shift magnitude (median PSI)\n(shift is real, but in-support)",
                       "median per-feature PSI", logy=True)
ax[2].axhline(0.25, color="red", ls=":", lw=1, label="PSI=0.25 (large)")
ax[2].legend(fontsize=7.5, loc="upper left")

fig.suptitle("Natural temporal shift (TabReD) is NOT feature-range extrapolation, unlike the "
             "construct positive test", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_oor_tabred_vs_construct.png"), dpi=150, bbox_inches="tight")
print("WROTE fig_oor_tabred_vs_construct.png")

# print headline contrast
print("\n=== HEADLINE CONTRAST ===")
print(f"Y-relevant-feature OOR[min,max]: TabReD = {[round(v,3) for v in y_split]} (all 0.0) "
      f"vs construct median = {med(c_split):.2f} (all {len(con_inc)} = 1.00)")
print(f"mean-cell OOR[min,max]: TabReD = {[round(v,4) for v in y_mean]} "
      f"vs construct median = {med(c_mean):.3f} (mean {c_mean.mean():.3f})")
print(f"median PSI: TabReD = {[round(v,3) for v in y_psi]} vs construct median = {med(c_psi):.2f}")
