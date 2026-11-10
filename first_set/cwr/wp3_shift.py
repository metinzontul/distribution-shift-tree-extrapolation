"""WP3 shift construction + observational strata (design protocol §9 B1/B2/B3, §10 A3/A5/A7).

ALL selection here is TRAIN-only / X-only or explicitly-disclosed shift-design (M3 leakage order):
  (1) split-feature  = argmax_f |Spearman(X_f, Y)| over the whole dataset. Uses Y, but ONLY to
      DEFINE the shift axis (disclosed construct, not model tuning) -- M3 step 1.
  (2) cutpoints      = percentiles of the split feature, X-only            -- M3 step 2.
  (3) sigma_Y^ref    = TRAIN-region Y-std, dataset-fixed                    -- M3 step 4.
  (4) tail-slope-ratio proxy / strata = TRAIN-region local slopes of Y-on-split-feature.

Shift builders
--------------
B2 covariate = FIXED-TRAIN SLIDING-TEST-WINDOW (severity <-> train-size decoupled):
  train  = split-feature lower `train_upper`% (FIXED size, all severities);
  test   = a `win`%-wide window whose lower edge sweeps `lower_edges`, so the ID->OOD *gap* moves
           while n_train and n_test stay fixed. severity = lower_edge - train_upper.

A3-fix dual split (fills the falsification stratum BY DESIGN):
  extrapolation-split = the max-|Spearman| feature (tail-slope-ratio high -> mechanism expects a
                        POSITIVE GBDT-NN gap);
  saturation-split    = the feature whose TRAIN-region tail-slope-ratio is LOWEST (most saturated;
                        outer region flattens -> tree extrapolation is fine -> NULL/NEG gap).
  The dataset-INTERNAL paired differential gap(extrap) - gap(satur) is the B3 mechanism-specificity
  test. GATE: #datasets with a constructible saturation-split (min ratio < `satur_max`) is reported;
  <12 -> differential is declared "N/A" (low power), never a silent pass.

B1 sigma_Y^ref: the pre-reg wording is "median-split lower-50% TRAIN-region Y-std". Under the B2
  fixed-train protocol the only strictly train-only, leakage-safe, dataset-fixed realization is the
  lower-`sigma_ref_pct`% region (the training side); lower-50% would include the severity-0 OOD
  window ([30,50] pct) and leak. Default `sigma_ref_pct=30` == the B2 train region; exposed as a
  knob so methodology can override to 50 if it rules the marginal-scale leak acceptable .
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

# --------------------------------------------------------------------------------------------------
# Feature candidates + split-feature selection (M1)
# --------------------------------------------------------------------------------------------------


def continuous_columns(X, min_unique=20):
    """Inclusion-eligible continuous features: >= `min_unique` distinct values (design §9 minor)."""
    return [c for c in X.columns if int(X[c].nunique()) >= min_unique]


def _abs_spearman(xv, yv):
    """|Spearman(x, y)| for split-feature ranking, under the explicit missing-data rule of the
    leakage-free covariate arm (the loader now PRESERVES NaN): a split AXIS must be FULLY OBSERVED,
    because the shift is built from percentile windows over that axis and a partially-missing feature
    cannot define clean, leakage-safe train/test windows (the missing rows have no window). So any
    feature carrying a NaN/inf is INELIGIBLE (returns NaN -> skipped). This mirrors tail_slope_ratio's
    rule, so extrapolation- and saturation-split candidacy use the SAME criterion. NaN-free inputs
    (WP2, filled loaders, every clean feature) are byte-identical to a plain |Spearman|."""
    xv = np.asarray(xv, float); yv = np.asarray(yv, float)
    if not np.all(np.isfinite(xv)):
        return np.nan                      # partially-missing feature: ineligible as split axis
    if np.ptp(xv) < 1e-12:
        return np.nan
    r = spearmanr(xv, yv).correlation
    return float(abs(r)) if r == r else np.nan


def select_split_feature(X, y, candidates=None):
    """max-|Spearman(X_f, Y)| feature; tie-break = LOWEST column index (M1). Returns (name, rho).
    Iterating in column order + strict-greater keeps the earliest (lowest-index) feature on ties."""
    cands = set(candidates if candidates is not None else continuous_columns(X))
    best_name, best_rho = None, -1.0
    for c in X.columns:                       # column order -> lowest index wins ties
        if c not in cands:
            continue
        rho = _abs_spearman(X[c].values, y.values)
        if np.isfinite(rho) and rho > best_rho + 1e-12:
            best_name, best_rho = c, rho
    return best_name, (best_rho if best_name is not None else np.nan)


# --------------------------------------------------------------------------------------------------
# B2 fixed-train sliding-test windows
# --------------------------------------------------------------------------------------------------

DEFAULT_LOWER_EDGES = (30, 45, 60, 80)     # -> severity (gap) = {0, 15, 30, 50}
TRAIN_UPPER = 30                            # train = split-feature lower 30% (fixed size)
WIN = 20                                    # test window width (percentile points)


def covariate_windows(xv, train_upper=TRAIN_UPPER, win=WIN, lower_edges=DEFAULT_LOWER_EDGES):
    """B2 masks over the split-feature values `xv` (full dataset). train = lower `train_upper`%
    (fixed); each test window = [p(edge), p(edge+win)]. Returns list of per-severity dicts with
    boolean masks into `xv`. severity = edge - train_upper (the ID->OOD percentile gap)."""
    xv = np.asarray(xv, float)
    q = lambda p: np.percentile(xv, p)
    train_mask = xv <= q(train_upper)
    out = []
    for e in lower_edges:
        lo, hi = q(e), q(e + win)
        # at edge==train_upper the window abuts train; include its lower boundary so severity-0 is non-empty
        test_mask = ((xv >= lo) if e <= train_upper else (xv > lo)) & (xv <= hi)
        out.append({"severity": int(e - train_upper), "lower_edge": int(e),
                    "train_mask": train_mask, "test_mask": test_mask,
                    "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())})
    return out


# --------------------------------------------------------------------------------------------------
# B1 reference scale + B3 tail-behaviour proxy / strata (all TRAIN-only)
# --------------------------------------------------------------------------------------------------


def sigma_y_ref(xv, yv, sigma_ref_pct=TRAIN_UPPER):
    """B1 normalizer: Y-std over the lower `sigma_ref_pct`% TRAIN region of the split feature.
    Dataset-fixed (independent of severity), strictly train-only. See module docstring re: the
    50% -> 30% reconciliation (leakage-safe realization of the pre-reg wording)."""
    xv = np.asarray(xv, float); yv = np.asarray(yv, float)
    m = xv <= np.percentile(xv, sigma_ref_pct)
    s = float(np.std(yv[m]))
    return s if s > 0 else float(np.std(yv))   # degenerate guard (should not trigger post-inclusion)


def tail_slope_ratio(xv, yv, train_upper=TRAIN_UPPER):
    """B3 proxy = slope_outer / slope_inner of Y-on-(split feature), TRAIN region only.

    The train region (lower `train_upper`%) is median-split on the feature into an inner (nearer the
    centre) and outer (nearer the future OOD boundary) half; a local OLS slope is fit in each. Ratio
    ~1 => the trend is sustained toward the boundary (extrapolation risk, tree extrapolates flat and
    fails); ratio ~0 => the trend flattens (saturation, tree extrapolation is fine). Returns NaN when
    the inner slope is ~0 (DA A3 guard: ratio explodes) or a half is too small."""
    xv = np.asarray(xv, float); yv = np.asarray(yv, float)
    # A shift axis needs clean percentile windows, so a feature carrying ANY missing value is
    # ineligible (returns NaN -> excluded from the saturation-split search). NaN-free features are
    # unaffected. This is the explicit form of the pre-fix behaviour (percentile over NaN was NaN).
    if not np.all(np.isfinite(xv)):
        return np.nan
    m = xv <= np.percentile(xv, train_upper)
    x, yy = xv[m], yv[m]
    if len(x) < 12:
        return np.nan
    med = np.median(x)
    inner, outer = x <= med, x > med
    if inner.sum() < 5 or outer.sum() < 5 or np.ptp(x[inner]) < 1e-12 or np.ptp(x[outer]) < 1e-12:
        return np.nan
    s_in = float(np.polyfit(x[inner], yy[inner], 1)[0])
    s_out = float(np.polyfit(x[outer], yy[outer], 1)[0])
    if abs(s_in) < 1e-9:
        return np.nan
    return s_out / s_in


def calibrate_thresholds(ratios):
    """A3-fix: derive the stratum boundaries (and the A7 flag threshold) from the DATASET-POOL
    tertiles of the tail-slope-ratio distribution, NOT the arbitrary 0.7/0.3. Returns (lo, hi) =
    (33rd, 67th percentile) over finite ratios. Pre-committed once the pool is fixed."""
    r = np.asarray([x for x in ratios if np.isfinite(x)], float)
    if len(r) < 3:
        return (0.3, 0.7)                     # fallback to the pre-reg defaults if pool too small
    return (float(np.percentile(r, 33.3)), float(np.percentile(r, 66.7)))


def stratum_of(ratio, lo, hi):
    """extrapolation (ratio>=hi) / saturation (ratio<=lo) / ambiguous (between). NaN -> 'undefined'."""
    if not np.isfinite(ratio):
        return "undefined"
    if ratio >= hi:
        return "extrapolation"
    if ratio <= lo:
        return "saturation"
    return "ambiguous"


def extrapolation_risk_flag(ratio, hi):
    """A7 prescriptive flag (single split-direction extrapolation-risk bit, NOT a family predictor):
    train-only tail-slope-ratio >= calibrated `hi` -> avoid the tree family / prefer NN-FM in THIS
    split direction. Returns None when the proxy is undefined."""
    if not np.isfinite(ratio):
        return None
    return bool(ratio >= hi)


# --------------------------------------------------------------------------------------------------
# A3-fix dual split builders (extrapolation vs saturation)
# --------------------------------------------------------------------------------------------------


def build_extrapolation_split(X, y, candidates=None, train_upper=TRAIN_UPPER, win=WIN,
                              lower_edges=DEFAULT_LOWER_EDGES):
    """Extrapolation arm: split feature = max-|Spearman|. Returns a spec dict (feature, rho,
    slope_ratio, windows, sigma_ref) or None if no continuous candidate exists."""
    cands = candidates if candidates is not None else continuous_columns(X)
    if not cands:
        return None
    feat, rho = select_split_feature(X, y, cands)
    if feat is None:
        return None
    xv, yv = X[feat].values, np.asarray(y, float)
    return {"arm": "extrapolation", "feature": feat, "spearman": float(rho),
            "slope_ratio": tail_slope_ratio(xv, yv, train_upper),
            "sigma_ref": sigma_y_ref(xv, yv), "windows": covariate_windows(xv, train_upper, win, lower_edges)}


def build_saturation_split(X, y, candidates=None, satur_max=0.5, train_upper=TRAIN_UPPER, win=WIN,
                           lower_edges=DEFAULT_LOWER_EDGES):
    """A3-fix falsification arm: split feature = the continuous feature whose TRAIN-region
    tail-slope-ratio is LOWEST (most saturated). Constructible only if that minimum ratio is
    < `satur_max`; otherwise returns None (contributes to the B3 gate count). Returns the same spec
    shape as build_extrapolation_split plus the constructibility flag."""
    cands = candidates if candidates is not None else continuous_columns(X)
    yv = np.asarray(y, float)
    ratios = {c: tail_slope_ratio(X[c].values, yv, train_upper) for c in cands}
    finite = {c: r for c, r in ratios.items() if np.isfinite(r)}
    if not finite:
        return None
    feat = min(finite, key=lambda c: finite[c])       # lowest slope-ratio = most saturated
    if finite[feat] >= satur_max:
        return None                                    # no saturated feature -> not constructible
    xv = X[feat].values
    return {"arm": "saturation", "feature": feat, "spearman": float(_abs_spearman(xv, yv)),
            "slope_ratio": float(finite[feat]), "sigma_ref": sigma_y_ref(xv, yv),
            "windows": covariate_windows(xv, train_upper, win, lower_edges)}


# --------------------------------------------------------------------------------------------------
# A5 temporal split (natural shift, primary evidence where available)
# --------------------------------------------------------------------------------------------------


def temporal_split(ds, order_values=None, train_frac=0.6, val_frac=0.15):
    """Natural time split (A5): rows sorted by `order_values` (a timestamp/order key); the earliest
    `train_frac` is train, the latest 1-train_frac is OOD test, with an ID-val slice carved from the
    END of the train block (temporal ID-val, no future leakage). Returns masks or None when the
    dataset carries no usable temporal key (caller then reports 'no natural temporal axis').

    A5 SECONDARY/fallback path only (air_quality / bike_hour row-order proxies). The A5-PRIMARY
    evidence uses the authored TabReD temporal split via temporal_split_from_indices()."""
    if order_values is None:
        return None
    order = np.asarray(order_values, float)
    if np.ptp(order) < 1e-12:
        return None
    n = len(order)
    rank = np.argsort(np.argsort(order))               # temporal rank 0..n-1
    cut = int(train_frac * n)
    train_mask = rank < cut
    test_mask = rank >= cut
    nv = max(int(val_frac * cut), 1)
    idval_mask = (rank >= cut - nv) & (rank < cut)      # ID-val = last slice of train (temporal)
    return {"arm": "temporal", "train_mask": train_mask, "test_mask": test_mask,
            "idval_mask": idval_mask, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}


def temporal_split_from_indices(train_idx, val_idx, test_idx, n):
    """A5-PRIMARY natural time split from an AUTHORED split (TabReD `splits/default/`): the dataset
    ships three contiguous, chronologically-ordered index blocks (train = earliest period, val = mid
    period, test = future period). We map them onto the harness temporal-arm contract:
      fit-train = train block   (earliest);  ID-val = val block (near-train future, in-period drift);
      OOD-test  = test block    (future period, the natural ID->OOD temporal shift).
    Returns the same mask dict shape as temporal_split(). ID-val is the AUTHORED val block (NOT carved
    from the train tail): with train < val < test contiguous, fit/ID stay strictly before the test
    period, so the future-leakage guard in run_temporal_cell holds by construction."""
    tm = np.zeros(n, dtype=bool); tm[np.asarray(train_idx, int)] = True
    vm = np.zeros(n, dtype=bool); vm[np.asarray(val_idx, int)] = True
    te = np.zeros(n, dtype=bool); te[np.asarray(test_idx, int)] = True
    return {"arm": "temporal", "train_mask": tm, "idval_mask": vm, "test_mask": te,
            "n_train": int(tm.sum()), "n_test": int(te.sum())}
