"""TabReD natural temporal-shift CHARACTERIZATION (post-hoc / exploratory).

For each of the 4 TabReD sets (authored `default` temporal split: fit=train period, ID=val,
OOD=test period), measure -- on the numeric (x_num) features -- how much the OOD-test block leaves
the TRAIN-period support (feature-range extrapolation), plus the general covariate-shift magnitude
(PSI / standardised mean-diff) and the target/prior shift. Numeric-only: x_bin is binary (no range
extrapolation) and x_cat/x_meta are not features (x_meta col0 = timestamp).

Train support = FULL train-period block (true data support), imputed with train-column medians
(leakage-safe, matching load_tabred). Test = FULL OOD test block, same train-median imputation.
"""
import sys, os, json, time
import numpy as np
from scipy.stats import spearmanr, ks_2samp
from sklearn.linear_model import Ridge

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[4])
sys.path.insert(0, os.path.join(ROOT, "code"))
from cwr.wp3_data import _maybe_extract_tabred, tabred_split_indices

OUT = os.path.join(ROOT, "results", "wp3_analysis", "tabred_shift_characterization")
os.makedirs(OUT, exist_ok=True)
SETS = ["sberbank-housing", "cooking-time", "weather", "delivery-eta"]
RNG = np.random.default_rng(42)


def _read_rows(mmap_arr, idx, cap=None, rng=RNG):
    idx = np.asarray(idx, np.int64)
    if cap and len(idx) > cap:
        idx = np.sort(rng.choice(idx, int(cap), replace=False))
    return np.asarray(mmap_arr[idx], np.float32)


def psi(train, test, bins=10):
    qs = np.quantile(train, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return np.nan
    qs = qs.astype(np.float64)
    qs[0], qs[-1] = -np.inf, np.inf
    tr_c, _ = np.histogram(train, qs)
    te_c, _ = np.histogram(test, qs)
    tr_p = tr_c / max(tr_c.sum(), 1)
    te_p = te_c / max(te_c.sum(), 1)
    eps = 1e-6
    tr_p = np.clip(tr_p, eps, None)
    te_p = np.clip(te_p, eps, None)
    return float(np.sum((te_p - tr_p) * np.log(te_p / tr_p)))


def oor_block(Xtr, Xte, lo, hi):
    """Per-feature out-of-range rate of test vs [lo,hi] (arrays over features)."""
    below = Xte < lo[None, :]
    above = Xte > hi[None, :]
    oor = below | above                      # (n_test, F) bool
    per_feat = oor.mean(axis=0)              # fraction of test points oor, per feature
    per_row_any = oor.any(axis=1).mean()     # frac test rows with >=1 feature oor
    per_row_count = oor.sum(axis=1).mean()   # mean #features oor per test row
    return per_feat, float(per_row_any), float(per_row_count)


def characterize(set_name):
    t0 = time.time()
    d = _maybe_extract_tabred(set_name)
    sp = tabred_split_indices(set_name)
    xnum = np.load(d / "x_num.npy", mmap_mode="r")
    y_all = np.load(d / "y.npy").astype(np.float64)

    tr_idx, va_idx, te_idx = sp["train"], sp["val"], sp["test"]
    Xtr = _read_rows(xnum, tr_idx)                       # FULL train (support)
    ytr = y_all[tr_idx]

    # --- test/val row indices sampled so X and y align exactly ---
    def sample_idx(idx, cap):
        idx = np.asarray(idx, np.int64)
        if cap and len(idx) > cap:
            idx = np.sort(np.random.default_rng(7).choice(idx, cap, replace=False))
        return idx
    te_s = sample_idx(te_idx, 80000)
    va_s = sample_idx(va_idx, 80000)
    Xte = np.asarray(xnum[te_s], np.float32)
    Xva = np.asarray(xnum[va_s], np.float32)
    yte = y_all[te_s]
    yva = y_all[va_s]

    # inf -> nan, train-median imputation (leakage-safe)
    for A in (Xtr, Xte, Xva):
        A[~np.isfinite(A)] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    for A in (Xtr, Xte, Xva):
        m = np.isnan(A)
        if m.any():
            A[m] = np.take(med, np.where(m)[1])

    F_all = Xtr.shape[1]
    ptp = Xtr.max(axis=0) - Xtr.min(axis=0)
    std_tr = Xtr.std(axis=0)
    keep = (ptp > 1e-12) & (std_tr > 1e-12)      # drop train-constant/degenerate features
    Xtr, Xte, Xva = Xtr[:, keep], Xte[:, keep], Xva[:, keep]
    F = int(keep.sum())
    std_tr = std_tr[keep]

    # --- out-of-range: [min,max] and [p1,p99] ---
    lo_mm, hi_mm = Xtr.min(axis=0), Xtr.max(axis=0)
    lo_pp = np.percentile(Xtr, 1, axis=0)
    hi_pp = np.percentile(Xtr, 99, axis=0)
    pf_mm, any_mm, cnt_mm = oor_block(Xtr, Xte, lo_mm, hi_mm)
    pf_pp, any_pp, cnt_pp = oor_block(Xtr, Xte, lo_pp, hi_pp)
    # ID-val reference (should be near baseline)
    pf_mm_v, any_mm_v, _ = oor_block(Xtr, Xva, lo_mm, hi_mm)

    # --- Spearman(X,Y) on train sample -> top Y-relevant feature (mechanism axis analog) ---
    ns = min(30000, Xtr.shape[0])
    si = np.random.default_rng(11).choice(Xtr.shape[0], ns, replace=False)
    rhos = np.array([abs(spearmanr(Xtr[si, j], ytr[si]).correlation) for j in range(F)])
    rhos = np.nan_to_num(rhos, nan=0.0)
    top = int(np.argmax(rhos))

    # --- covariate-shift magnitude (numeric) ---
    smd = np.abs(Xte.mean(axis=0) - Xtr.mean(axis=0)) / std_tr
    psis = np.array([psi(Xtr[:, j], Xte[:, j]) for j in range(F)])
    psis_f = psis[np.isfinite(psis)]

    # --- Y (target/prior) shift ---
    s_ytr = float(ytr.std()) or 1.0
    q = lambda a, p: float(np.percentile(a, p))
    y_metrics = {
        "smd_Y": float((yte.mean() - ytr.mean()) / s_ytr),
        "std_ratio_Y": float(yte.std() / s_ytr),
        "q10_shift_std": (q(yte, 10) - q(ytr, 10)) / s_ytr,
        "q50_shift_std": (q(yte, 50) - q(ytr, 50)) / s_ytr,
        "q90_shift_std": (q(yte, 90) - q(ytr, 90)) / s_ytr,
        "psi_Y": psi(ytr, yte),
        "ks_Y": float(ks_2samp(ytr, yte).statistic),
        "mean_Y_train": float(ytr.mean()), "mean_Y_test": float(yte.mean()),
        "std_Y_train": s_ytr, "std_Y_test": float(yte.std()),
    }

    # --- light concept-drift probe (CRUDE, labeled): ridge train vs OOD nRMSE ---
    cap = 80000
    ti = np.random.default_rng(3).choice(Xtr.shape[0], min(cap, Xtr.shape[0]), replace=False)
    mu, sd = Xtr[ti].mean(0), Xtr[ti].std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Ztr = (Xtr[ti] - mu) / sd
    Zte = (Xte - mu) / sd
    rg = Ridge(alpha=1.0).fit(Ztr, ytr[ti])
    rmse_tr = float(np.sqrt(np.mean((rg.predict(Ztr) - ytr[ti]) ** 2)))
    rmse_te = float(np.sqrt(np.mean((rg.predict(Zte) - yte) ** 2)))
    ridge = {"nRMSE_train": rmse_tr / s_ytr, "nRMSE_ood": rmse_te / s_ytr,
             "ood_over_train": (rmse_te / max(rmse_tr, 1e-9))}

    # top-5 most out-of-range features (min-max) + their Y-relevance
    order = np.argsort(-pf_mm)[:5]
    top_oor_feats = [{"feat_idx": int(j), "oor_mm": float(pf_mm[j]), "oor_pp": float(pf_pp[j]),
                      "smd": float(smd[j]), "psi": float(psis[j]), "abs_spearman_Y": float(rhos[j])}
                     for j in order]

    res = {
        "set": set_name, "n_full": int(xnum.shape[0]),
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx)),
        "F_numeric_all": F_all, "F_used": F, "F_degenerate": F_all - F,
        # OUT-OF-RANGE (MAIN)
        "oor_minmax": {"mean_rate": float(pf_mm.mean()), "max_feat_rate": float(pf_mm.max()),
                       "frac_rows_any": any_mm, "mean_nfeat_oor": cnt_mm,
                       "frac_feats_over50": float((pf_mm > 0.5).mean()),
                       "frac_feats_over20": float((pf_mm > 0.2).mean())},
        "oor_p1p99": {"mean_rate": float(pf_pp.mean()), "max_feat_rate": float(pf_pp.max()),
                      "frac_rows_any": any_pp, "mean_nfeat_oor": cnt_pp,
                      "frac_feats_over50": float((pf_pp > 0.5).mean())},
        "oor_idval_minmax": {"mean_rate": float(pf_mm_v.mean()), "frac_rows_any": any_mm_v},
        "top_spearman_feature": {"feat_idx": top, "abs_spearman_Y": float(rhos[top]),
                                 "oor_mm": float(pf_mm[top]), "oor_pp": float(pf_pp[top]),
                                 "smd": float(smd[top]), "psi": float(psis[top])},
        "top5_oor_features": top_oor_feats,
        # COVARIATE SHIFT MAGNITUDE
        "covariate_shift": {"smd_mean": float(smd.mean()), "smd_median": float(np.median(smd)),
                            "smd_max": float(smd.max()), "frac_smd_over05": float((smd > 0.5).mean()),
                            "frac_smd_over10": float((smd > 1.0).mean()),
                            "psi_mean": float(psis_f.mean()), "psi_median": float(np.median(psis_f)),
                            "psi_max": float(psis_f.max()), "frac_psi_over01": float((psis_f > 0.1).mean()),
                            "frac_psi_over025": float((psis_f > 0.25).mean())},
        # TARGET / PRIOR SHIFT
        "y_shift": y_metrics,
        # CONCEPT PROBE (crude)
        "ridge_probe": ridge,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[{set_name}] F={F} oor_mm mean={pf_mm.mean():.4f} max={pf_mm.max():.3f} "
          f"any={any_mm:.3f} | oor_pp mean={pf_pp.mean():.4f} | topSpear rho={rhos[top]:.3f} "
          f"oor_mm={pf_mm[top]:.3f} | PSI med={np.median(psis_f):.3f} smd med={np.median(smd):.3f} | "
          f"Y smd={y_metrics['smd_Y']:+.3f} psiY={y_metrics['psi_Y']:.3f} | "
          f"ridge ood/tr={ridge['ood_over_train']:.2f}  ({res['seconds']}s)")
    return res


def main():
    out = {}
    for s in SETS:
        out[s] = characterize(s)
    path = os.path.join(OUT, "tabred_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
