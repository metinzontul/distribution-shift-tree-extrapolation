"""Robustness: does out-of-range depend on the train-support subsample size?
The models actually fit on a capped train subsample (max_fit_train=5000). A smaller train has
narrower min/max => could inflate out-of-range. Check that the top-Y-feature and mean OOR stay
near ZERO (not construct-level) across train caps {full, 5000, 3000}.
"""
import sys, os, json
import numpy as np
from scipy.stats import spearmanr

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[4])
sys.path.insert(0, os.path.join(ROOT, "code"))
from cwr.wp3_data import _maybe_extract_tabred, tabred_split_indices

SETS = ["sberbank-housing", "cooking-time", "weather", "delivery-eta"]
OUT = os.path.join(ROOT, "results", "wp3_analysis", "tabred_shift_characterization")


def impute(A, med):
    A = A.copy()
    A[~np.isfinite(A)] = np.nan
    m = np.isnan(A)
    if m.any():
        A[m] = np.take(med, np.where(m)[1])
    return A


res = {}
for s in SETS:
    d = _maybe_extract_tabred(s)
    sp = tabred_split_indices(s)
    xnum = np.load(d / "x_num.npy", mmap_mode="r")
    y = np.load(d / "y.npy").astype(np.float64)
    tr, te = sp["train"], sp["test"]
    Xtr_full = np.asarray(xnum[tr], np.float32)
    te_s = np.sort(np.random.default_rng(7).choice(te, min(80000, len(te)), replace=False))
    Xte = np.asarray(xnum[te_s], np.float32)
    ytr = y[tr]
    Xtr_full[~np.isfinite(Xtr_full)] = np.nan
    with np.errstate(all="ignore"):
        med = np.where(np.isfinite(np.nanmedian(Xtr_full, 0)), np.nanmedian(Xtr_full, 0), 0.0).astype(np.float32)
    Xtr_full = impute(Xtr_full, med)
    Xte = impute(Xte, med)
    keep = (np.ptp(Xtr_full, 0) > 1e-12)
    Xtr_full, Xte = Xtr_full[:, keep], Xte[:, keep]
    F = Xtr_full.shape[1]
    ns = min(30000, len(tr))
    si = np.random.default_rng(11).choice(len(tr), ns, replace=False)
    rhos = np.nan_to_num([abs(spearmanr(Xtr_full[si, j], ytr[si]).correlation) for j in range(F)])
    top = int(np.argmax(rhos))
    row = {}
    for cap in ("full", 5000, 3000):
        if cap == "full":
            Xtr = Xtr_full
        else:
            idx = np.random.default_rng(1).choice(len(tr), min(cap, len(tr)), replace=False)
            Xtr = Xtr_full[idx]
        lo, hi = Xtr.min(0), Xtr.max(0)
        oor = (Xte < lo[None, :]) | (Xte > hi[None, :])
        row[str(cap)] = {"mean_oor": float(oor.mean()), "top_feat_oor": float(oor[:, top].mean()),
                         "max_feat_oor": float(oor.mean(0).max())}
    res[s] = row
    print(f"{s:18s} topSpearman rho={rhos[top]:.3f}  "
          + "  ".join(f"{c}:mean={row[str(c)]['mean_oor']:.4f}/top={row[str(c)]['top_feat_oor']:.4f}"
                       for c in ("full", 5000, 3000)))

json.dump(res, open(os.path.join(OUT, "robustness_traincap.json"), "w"), indent=2)
print("WROTE robustness_traincap.json")
