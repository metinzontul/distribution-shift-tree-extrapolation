"""WP3 metrics (design protocol §9 B1, §10 A6).

Primary output = the full GAP-SEVERITY CURVE (A6: reader applies their own threshold), normalized by
a dataset-fixed, train-only reference scale sigma_Y^ref (B1). Secondary = robustified effective
robustness + absolute OOD error (reused from wp2_metrics so the analysis layer stays consistent).

  nOOD          = OOD-RMSE / sigma_Y^ref                              (B1 primary, line-free)
  MDE_noticeable = 0.05 * sigma_Y^ref  ;  MDE_substantial = 0.10 * sigma_Y^ref   (A6 illustrative)
  gap-severity  = per-severity nOOD per model -> the curve the reader thresholds
  paired family gap = nOOD(GBDT) - nOOD(NN) per severity                (M2 confirmatory quantity)

These operate on the RAW arrays the pipeline writes to results/wp3_*/*.npz, kept separate from the
runner so data-stats-analyst can recompute without retraining. This module does NOT interpret the OOD
magnitudes (confirmatory verdict / severity map is the analysis layer's, on the full run).
"""
from __future__ import annotations

import numpy as np

# Secondary metrics reused verbatim from WP2 (robust Theil-Sen effrob line + line-free gap/ratio).
from .wp2_metrics import raw_gap, robust_effective_robustness, robust_reference_line  # noqa: F401

MDE_NOTICEABLE = 0.05     # A6 illustrative "smallest noticeable deployment gap"
MDE_SUBSTANTIAL = 0.10    # A6 illustrative "substantial" marker


def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nood(y_true, y_pred, sigma_y_ref):
    """B1 primary metric: OOD-RMSE normalized by the dataset-fixed train-only reference scale."""
    return rmse(y_true, y_pred) / (float(sigma_y_ref) + 1e-12)


def abs_ood(y_true, y_pred):
    """Secondary: un-normalized OOD RMSE (reported alongside nOOD; Pillar-2 caveat)."""
    return rmse(y_true, y_pred)


def mde(sigma_y_ref, frac=MDE_NOTICEABLE):
    """Minimum deployment-relevant effect in nOOD units (B1/A6). Since nOOD already divides by
    sigma_Y^ref, an nOOD gap threshold is simply `frac` (0.05 noticeable / 0.10 substantial); the
    same value in absolute RMSE units is `frac * sigma_Y^ref`, returned here for convenience."""
    return {"nood_threshold": float(frac), "abs_threshold": float(frac) * float(sigma_y_ref)}


def family_gap(nood_gbdt, nood_nn):
    """M2 confirmatory quantity: paired family gap = nOOD(GBDT) - nOOD(NN). Positive = GBDT worse
    (extrapolation-mechanism direction). Normalizer-immune (both divide by the same sigma_Y^ref)."""
    return float(nood_gbdt) - float(nood_nn)


def gap_severity_curve(cell_records, gbdt="catboost", nn="ft_transformer"):
    """A6 primary output assembler. `cell_records` = list of per-severity dicts
    {severity, nood: {model: value}} for ONE (dataset, arm, seed). Returns severities + the paired
    GBDT-NN gap per severity + per-model nOOD curves, plus the crossing severity (first severity at
    which the gap turns positive, i.e. GBDT becomes worse than NN) or None if it never crosses.

    Does NOT threshold or interpret; it only structures the raw curve for the analysis layer."""
    recs = sorted(cell_records, key=lambda r: r["severity"])
    sev = [r["severity"] for r in recs]
    gap = [family_gap(r["nood"][gbdt], r["nood"][nn]) for r in recs]
    models = sorted({m for r in recs for m in r["nood"]})
    curves = {m: [r["nood"].get(m, np.nan) for r in recs] for m in models}
    crossing = next((sev[i] for i, g in enumerate(gap) if g > 0), None)
    return {"severity": sev, "family_gap": gap, "nood_curves": curves,
            "gbdt": gbdt, "nn": nn, "crossing_severity": crossing}
