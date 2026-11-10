"""Shift diagnostics: out-of-range fraction (OOR), population stability index
(PSI) and standardised mean difference (SMD), per feature."""
from __future__ import annotations

import numpy as np
import pandas as pd


def oor(train: np.ndarray, test: np.ndarray, q=(0.0, 1.0)) -> float:
    """Share of test values outside the training [q_lo, q_hi] quantile range.
    q=(0,1) is the min-max range; q=(0.01,0.99) the outlier-robust variant."""
    train, test = _clean(train), _clean(test)
    if len(train) == 0 or len(test) == 0:
        return np.nan
    lo, hi = np.quantile(train, q[0]), np.quantile(train, q[1])
    return float(np.mean((test < lo) | (test > hi)))


def psi(train: np.ndarray, test: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI with quantile bins from the training data. Test values beyond the
    training range fall into the outer bins. Note: PSI is unbounded and very
    large values (>5) usually reflect near-empty bins, so report the median
    over features rather than the max."""
    train, test = _clean(train), _clean(test)
    if len(train) == 0 or len(test) == 0:
        return np.nan
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(train, edges)[0] / len(train)
    r = np.histogram(test, edges)[0] / len(test)
    p, r = np.clip(p, eps, None), np.clip(r, eps, None)
    return float(np.sum((r - p) * np.log(r / p)))


def smd(train: np.ndarray, test: np.ndarray) -> float:
    train, test = _clean(train), _clean(test)
    if len(train) < 2 or len(test) < 2:
        return np.nan
    pooled = np.sqrt((train.var(ddof=1) + test.var(ddof=1)) / 2)
    return float((test.mean() - train.mean()) / pooled) if pooled > 0 else 0.0


def _clean(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def shift_report(X: pd.DataFrame, y: np.ndarray, num_cols: list, train_idx, test_idx,
                 key_feature: str | None = None) -> dict:
    """Summary used in Table 7 / Fig. 6: OOR on the key (target-correlated)
    feature, mean / max per-feature OOR, robust OOR, median & max PSI, and
    label shift (SMD of y)."""
    rows = []
    for c in num_cols:
        a, b = X[c].to_numpy()[train_idx], X[c].to_numpy()[test_idx]
        rows.append({"feature": c, "oor": oor(a, b), "oor_p1p99": oor(a, b, (0.01, 0.99)),
                     "psi": psi(a, b), "smd": smd(a, b)})
    f = pd.DataFrame(rows)
    out = {
        "oor_mean": f["oor"].mean() if len(f) else np.nan,
        "oor_max": f["oor"].max() if len(f) else np.nan,
        "oor_p1p99_mean": f["oor_p1p99"].mean() if len(f) else np.nan,
        "psi_median": f["psi"].median() if len(f) else np.nan,
        "psi_max": f["psi"].max() if len(f) else np.nan,
        "abs_smd_median": f["smd"].abs().median() if len(f) else np.nan,
        "label_smd": smd(y[train_idx], y[test_idx]),
    }
    if key_feature is not None and key_feature in X.columns:
        out["key_feature"] = key_feature
        out["oor_key"] = float(f.loc[f.feature == key_feature, "oor"].iloc[0])
    return out


def top_feature_by_train_correlation(X: pd.DataFrame, y: np.ndarray, num_cols: list, train_idx):
    """Target-relevant feature for the temporal arm, chosen on training rows only."""
    from scipy.stats import spearmanr
    best, best_r = None, -1.0
    for c in num_cols:
        a = X[c].to_numpy(dtype=float)[train_idx]
        m = np.isfinite(a)
        if m.sum() < 10 or np.nanstd(a) == 0:
            continue
        r = abs(spearmanr(a[m], y[train_idx][m]).statistic)
        if np.isfinite(r) and r > best_r:
            best, best_r = c, r
    return best, best_r
