"""WP2 robustness metrics (design.md §6 + frozen Lock-in 6/7/10 + 2026-07-30 AMENDMENT).

Metric families (frozen-protocol-shift-robustness.md AMENDMENT, methodology
`wp2-recenter-approval.md`):

  * normalized RMSE (nID / nOOD)  : RMSE / std(y_fit_train). The reference scale NEVER uses the test
                                    target (no leak). `nOOD` == `absOOD` (P1 primary metric, LINE-FREE
                                    -> immune to reference-line fragility).
  * raw ID->OOD gap               : nOOD - nID. Co-primary, line-free (Lock-in 3 / §6 AMENDMENT:
                                    promoted from secondary). A constant predictor can fake a small
                                    gap, so it is ALWAYS reported next to absOOD.
  * OOD/ID ratio                  : nOOD / nID. S1 line-free framing ("difficulty not robustness":
                                    geometry inflates absOOD + gap in proportion to ID difficulty, so
                                    the ratio stays ~constant across rotations).
  * robustified effective robustness : per-CONDITION reference ID->OOD line fit at the model x seed
                                    level (>=150 pts) with a ROBUST estimator (Theil-Sen default,
                                    Huber optional) over a PRE-COMMITTED anchor set (baselines
                                    constant_mean + ridge INCLUDED, Lock-in 7 AMENDMENT). A model's
                                    effective robustness = predicted_nOOD - actual_nOOD (positive =
                                    below the line = more robust than the population ID->OOD trend).
                                    Replaces the old 5-archetype centroid OLS fit (dropped: 5-point
                                    fragility, slope ranged [-0.48, 2.34] and flipped sign per cell).
  * obliqueness score             : Lock-in 6 surrogate, 1 - shallowTreeR2(X) / shallowTreeR2(PCA-X),
                                    TRAIN only. High = more oblique (rotated splits help more).

These operate on the RAW arrays the sweep writes to results/wp2_synth*/*.npz (default =
results/wp2_confirmatory for the confirmatory tuned sweep). Kept separate from the runner so the
analysis layer (data-stats-analyst) can recompute without retraining.
"""
from __future__ import annotations

import numpy as np

# Pre-committed anchor set for the robustified effrob reference line (Lock-in 7 AMENDMENT).
# The reference line is fit over ALL of these model families' (model x seed) points within a
# condition; baselines constant_mean + ridge are INCLUDED so the low-error / high-error ends of the
# line are anchored and cannot be dragged degenerately by the family under evaluation. The oracle is
# NEVER an anchor (Lock-in 10, diagnostic only). Frozen before the confirmatory run.
ANCHOR_MODELS = ("constant_mean", "ridge", "catboost", "ft_transformer", "tabpfn")


def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def norm_rmse(y_true, y_pred, y_fit_std):
    """RMSE normalized by the fit-train target std (leakage-free reference scale, design.md §6).

    Applied to the OOD arrays this IS nOOD == absOOD (P1 primary, line-free); applied to the ID
    arrays it is nID. The reference std comes from fit-train only, never the test target."""
    return rmse(y_true, y_pred) / (float(y_fit_std) + 1e-12)


def raw_gap(id_norm, ood_norm):
    """Line-free co-primary (design.md §6 AMENDMENT): normalized ID->OOD gap = nOOD - nID. Report
    alongside absOOD so a constant predictor's fake robustness (flat gap, huge absOOD) stays visible."""
    return np.asarray(ood_norm, float) - np.asarray(id_norm, float)


def ood_id_ratio(id_norm, ood_norm, eps=1e-9):
    """Line-free S1 metric: nOOD / nID. Used for the geometry 'difficulty not robustness' framing
    (OOD/ID ratio ~constant across rotations => geometry inflates difficulty, not fragility)."""
    x = np.asarray(id_norm, float)
    return np.asarray(ood_norm, float) / (x + eps)


def obliqueness_score(X_train, y_train, max_depth=4, random_state=0):
    """Lock-in 6 estimator (train only). Matches validate_synth.gt_obliqueness depth. Uses PCA as the
    unknown-rotation surrogate (real data has no ground-truth R)."""
    from sklearn.decomposition import PCA
    from sklearn.tree import DecisionTreeRegressor

    def tr2(F):
        return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state).fit(F, y_train).score(F, y_train)

    Xr = PCA(n_components=X_train.shape[1], random_state=random_state).fit_transform(X_train)
    return 1.0 - tr2(X_train) / max(tr2(Xr), 1e-6)


# --------------------------------------------------------------------------------------------------
# Robustified effective robustness (Lock-in 7 AMENDMENT: model x seed level + robust fit + anchors)
# --------------------------------------------------------------------------------------------------


def robust_reference_line(id_norm, ood_norm, method="theilsen"):
    """Fit a ROBUST ID->OOD reference line over a pooled anchor population.

    `id_norm`, `ood_norm` are 1-D arrays of the anchor set's (model x seed) points within ONE
    condition (>=150 points with 30 seeds x 5 anchor models). Returns (intercept, slope, ok) where
    `ok` is False for a degenerate population (fewer than 3 points or ~no spread in x).

      method="theilsen" (default) : scipy.stats.theilslopes -- median-of-slopes, breakdown 29%,
                                    robust to the fat OOD tail that broke the old 5-point OLS.
      method="huber"              : sklearn HuberRegressor -- robust linear alternative.
    """
    x = np.asarray(id_norm, float); y = np.asarray(ood_norm, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) < 1e-9:
        return (np.nan, np.nan, False)
    if method == "theilsen":
        from scipy.stats import theilslopes
        slope, intercept, _lo, _hi = theilslopes(y, x)
    elif method == "huber":
        from sklearn.linear_model import HuberRegressor
        hr = HuberRegressor().fit(x.reshape(-1, 1), y)
        slope, intercept = float(hr.coef_[0]), float(hr.intercept_)
    else:
        raise ValueError(f"method must be 'theilsen' or 'huber', got {method!r}")
    return (float(intercept), float(slope), True)


def effrob_residual(id_norm, ood_norm, intercept, slope):
    """Per-point effective robustness against a fixed line: predicted_nOOD - actual_nOOD.
    Positive = below the line = more robust than the population ID->OOD trend."""
    x = np.asarray(id_norm, float); y = np.asarray(ood_norm, float)
    return (intercept + slope * x) - y


def robust_effective_robustness(anchor_id, anchor_ood, eval_id, eval_ood, method="theilsen"):
    """Convenience: fit the robust reference line on the ANCHOR points, then return each EVAL point's
    effective robustness residual. `anchor_*` = the pre-committed anchor set's (model x seed) points
    for one condition; `eval_*` = the points to score (may equal the anchors). Falls back to the
    line-free raw gap -(ood - id) with ok=False if the anchor population is degenerate."""
    intercept, slope, ok = robust_reference_line(anchor_id, anchor_ood, method=method)
    if not ok:
        return -raw_gap(eval_id, eval_ood), (np.nan, np.nan, False)
    return effrob_residual(eval_id, eval_ood, intercept, slope), (intercept, slope, True)


# --------------------------------------------------------------------------------------------------
# Deprecated exploratory metric (default sweep only; DO NOT use for the confirmatory verdict)
# --------------------------------------------------------------------------------------------------


def pooled_effective_robustness(id_norm, ood_norm):
    """DEPRECATED for the confirmatory verdict (Lock-in 7 AMENDMENT). Non-robust OLS ID->OOD line;
    retained only so the already-run exploratory DEFAULT sweep analysis (results/wp2_analysis)
    reproduces. The confirmatory sweep uses robust_effective_robustness (Theil-Sen + anchors)."""
    x = np.asarray(id_norm, float); y = np.asarray(ood_norm, float)
    if len(x) < 3 or np.ptp(x) < 1e-9:
        return -(y - x), (np.nan, np.nan)
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    return pred - y, (a, b)
