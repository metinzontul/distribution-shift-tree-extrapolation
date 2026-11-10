"""WP3-REVISION Experiment B helpers: natural out-of-range (OOR) + drift characterization. (internal review). All quantities are TRAIN-only-defined (leakage-safe):
  * per-feature marginal OOR-fraction  = share of TEST rows whose feature value falls outside the
    TRAIN min/max (primary, descriptive, low-risk; DA-B2 marginal-primary);
  * target-relevant feature            = train-only argmax |Spearman(X_j, Y)|;
  * PSI (population stability index)    = train-vs-test marginal drift per feature (train-only bins);
  * label shift                        = Y mean/var + KS(train, test) (DA-B2 multi-shift visibility);
  * concept-shift proxy                = train-fit ridge probe test/train RMSE ratio;
  * multivariate OOR (Mahalanobis)     = share of TEST rows beyond the train 99th-pct Mahalanobis
    distance (DA-B2 bonus; marginal stays primary).

These never touch the model's predictions; they describe the shift. The cell runner stores the
target-feature OOR membership of the PREDICTED test rows so the within-test OOR-stratified differential
(cat-FT gap on OOR-subset vs in-range-subset) can be computed downstream by data-stats-analyst.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, spearmanr


def oor_fraction_per_feature(X_train, X_test):
    """Per-feature marginal OOR fraction + train range. X_* are 2-D float arrays, same columns.
    Returns (frac[F], lo[F], hi[F]) where frac_j = mean_test(x_j < lo_j or x_j > hi_j)."""
    Xtr = np.asarray(X_train, float); Xte = np.asarray(X_test, float)
    lo = np.nanmin(Xtr, axis=0); hi = np.nanmax(Xtr, axis=0)
    below = Xte < lo[None, :]
    above = Xte > hi[None, :]
    frac = np.nanmean(below | above, axis=0)
    return frac, lo, hi


def target_relevant_feature(X_train, y_train, feature_names):
    """Train-only argmax |Spearman(X_j, Y)|; returns (name, idx, abs_spearman). Fully-observed
    features only (a NaN-carrying column is ineligible, mirroring wp3_shift)."""
    Xtr = np.asarray(X_train, float); y = np.asarray(y_train, float)
    best_idx, best_rho = None, -1.0
    for j in range(Xtr.shape[1]):
        xv = Xtr[:, j]
        if not np.all(np.isfinite(xv)) or np.ptp(xv) < 1e-12:
            continue
        r = spearmanr(xv, y).correlation
        if r == r and abs(r) > best_rho:
            best_idx, best_rho = j, abs(float(r))
    if best_idx is None:
        return None, None, np.nan
    return feature_names[best_idx], int(best_idx), float(best_rho)


def psi_per_feature(X_train, X_test, bins=10, eps=1e-6):
    """Population Stability Index per feature, using TRAIN-only quantile bin edges. Constant features
    -> PSI 0. Returns psi[F] (NaN-safe)."""
    Xtr = np.asarray(X_train, float); Xte = np.asarray(X_test, float)
    F = Xtr.shape[1]
    out = np.zeros(F, float)
    qs = np.linspace(0, 100, bins + 1)
    for j in range(F):
        a = Xtr[:, j][np.isfinite(Xtr[:, j])]
        b = Xte[:, j][np.isfinite(Xte[:, j])]
        if len(a) == 0 or len(b) == 0 or np.ptp(a) < 1e-12:
            out[j] = 0.0
            continue
        edges = np.unique(np.percentile(a, qs))
        if len(edges) < 2:
            out[j] = 0.0
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        tr_frac = np.histogram(a, edges)[0].astype(float); tr_frac /= tr_frac.sum()
        te_frac = np.histogram(b, edges)[0].astype(float); te_frac /= te_frac.sum()
        tr_frac = np.clip(tr_frac, eps, None); te_frac = np.clip(te_frac, eps, None)
        out[j] = float(np.sum((te_frac - tr_frac) * np.log(te_frac / tr_frac)))
    return out


def label_shift(y_train, y_test):
    """Y distribution drift: mean/var train vs test, mean shift in train-sigma units, KS stat/p."""
    a = np.asarray(y_train, float); b = np.asarray(y_test, float)
    sd = float(np.std(a)) or 1.0
    ks = ks_2samp(a, b)
    return {"mean_train": float(np.mean(a)), "mean_test": float(np.mean(b)),
            "var_train": float(np.var(a)), "var_test": float(np.var(b)),
            "mean_shift_in_sigma": float((np.mean(b) - np.mean(a)) / sd),
            "var_ratio_test_over_train": float((np.var(b) + 1e-12) / (np.var(a) + 1e-12)),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}


def concept_probe_ratio(Z_train, y_train, Z_test, y_test, alpha=1.0):
    """Concept-shift proxy = test/train RMSE ratio of a train-fit ridge linear probe on the SAME
    standardized features both models see. >1 => the linear X->Y relationship degrades on test."""
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha).fit(np.asarray(Z_train, float), np.asarray(y_train, float))
    def _rmse(Z, y):
        p = m.predict(np.asarray(Z, float))
        return float(np.sqrt(np.mean((p - np.asarray(y, float)) ** 2)))
    r_tr = _rmse(Z_train, y_train); r_te = _rmse(Z_test, y_test)
    return {"ridge_rmse_train": r_tr, "ridge_rmse_test": r_te,
            "ridge_rmse_ratio_test_over_train": float((r_te + 1e-12) / (r_tr + 1e-12))}


def mahalanobis_oor_fraction(X_train, X_test, q=99.0, ridge=1e-6):
    """Multivariate OOR (DA-B2 bonus): fraction of TEST rows whose Mahalanobis distance (train
    mean/cov) exceeds the train q-th percentile. Marginal per-feature OOR stays primary."""
    Xtr = np.asarray(X_train, float); Xte = np.asarray(X_test, float)
    mu = np.nanmean(Xtr, axis=0)
    cov = np.cov(np.nan_to_num(Xtr - mu, nan=0.0), rowvar=False)
    cov = np.atleast_2d(cov) + ridge * np.eye(Xtr.shape[1])
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return {"maha_oor_fraction": float("nan"), "maha_threshold": float("nan")}
    def _d2(X):
        d = np.nan_to_num(X - mu, nan=0.0)
        return np.einsum("ij,jk,ik->i", d, inv, d)
    d2_tr = _d2(Xtr); thr = float(np.percentile(d2_tr, q))
    frac = float(np.mean(_d2(Xte) > thr))
    return {"maha_oor_fraction": frac, "maha_threshold": thr, "maha_q": q}
