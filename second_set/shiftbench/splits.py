"""Split constructions for the covariate arm.

kinds
-----
extrap              Train on the lowest `train_frac` of the target-correlated
                    feature, test on a band of width `band` whose lower edge
                    sits `severity` percentile points above the training top
                    (paper: train 30 %, band 20 pp, severity 0/15/30/50).
reverse             Same feature, opposite direction (train high, test low).
saturation          Paper control: (feature, direction) with the lowest
                    train-only near/far slope ratio ("paper" rule = raw argmin).
saturation_robust   Same, but ignores sign-flipped ratios and takes the ratio
                    closest to 0 (fixes the instability noted by the review).
saturation_matched  Robust rule restricted to features whose |rho| is at least
                    `matched_tol` x |rho| of the extrapolation feature, i.e. a
                    leverage-matched control.
plateau             Same feature/direction as extrap, but the target is made to
                    flatten beyond the training boundary via an isotonic fit
                    (semi-synthetic real-data analogue of the synthetic plateau
                    control; holds feature leverage inside the training range).
id                  Random split with the same train / test sizes (in-distribution
                    reference).

feature selection
-----------------
selection="full"     paper protocol: |Spearman| on all rows (design-level step).
selection="holdout"  a random `holdout_frac` of rows is used ONLY to choose the
                     shift / saturation features and is then discarded, so the
                     test rows never inform the split design (review item 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from .data import Dataset

SEVERITIES = (0, 15, 30, 50)
ALL_KINDS = ("extrap", "reverse", "saturation", "saturation_robust",
             "saturation_matched", "plateau", "id")


@dataclass
class Split:
    dataset: str
    kind: str
    severity: float
    feature: Optional[str]
    direction: Optional[str]
    train_idx: np.ndarray
    test_idx: np.ndarray
    y: Optional[np.ndarray] = None  # target override (plateau); None = original
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        sev = "na" if np.isnan(self.severity) else int(self.severity)
        return f"{self.dataset}|{self.kind}|{sev}"


# ---------------------------------------------------------------- helpers
def candidate_features(ds: Dataset, min_unique: int = 20, max_missing: float = 0.3) -> list:
    out = []
    for c in ds.num_cols:
        col = ds.X[c]
        if col.isna().mean() <= max_missing and col.nunique(dropna=True) >= min_unique:
            out.append(c)
    return out


def abs_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.std(x[m]) == 0:
        return 0.0, 1.0
    r = spearmanr(x[m], y[m]).statistic
    return (abs(r), np.sign(r)) if np.isfinite(r) else (0.0, 1.0)


def percentile_rank(x: np.ndarray, tiebreak_seed: int = 0) -> np.ndarray:
    """Ranks scaled to [0,1); deterministic random tie-break; NaN stays NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    ok = np.where(np.isfinite(x))[0]
    jitter = np.random.default_rng(tiebreak_seed).random(len(ok))
    order = np.lexsort((jitter, x[ok]))
    out[ok[order]] = np.arange(len(ok)) / len(ok)
    return out


def band_split(x: np.ndarray, rows: np.ndarray, train_frac: float, severity_pp: float,
               band: float, direction: str = "up"):
    """Indices (into the full dataset) for a feature-range split over `rows`."""
    v = x[rows] if direction == "up" else -x[rows]
    pr = percentile_rank(v)
    lo = train_frac + severity_pp / 100.0
    hi = lo + band
    if hi > 1.0 + 1e-9:
        raise ValueError(f"band [{lo:.2f},{hi:.2f}] exceeds the feature range")
    train = rows[pr < train_frac]
    test = rows[(pr >= lo) & (pr < hi + 1e-12)]
    return train, test


def slope_ratio(x: np.ndarray, y: np.ndarray, direction: str = "up") -> float:
    """Near-half / far-half OLS slope inside the training range. 'near' is the
    half adjacent to the train/test boundary. <1 flattening, >1 steepening."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 20:
        return np.nan
    mid = np.median(x)
    lower, upper = x <= mid, x > mid
    if lower.sum() < 5 or upper.sum() < 5:
        return np.nan

    def slope(mask):
        xs = x[mask]
        if np.std(xs) == 0:
            return 0.0
        return np.polyfit(xs, y[mask], 1)[0]

    near, far = (slope(upper), slope(lower)) if direction == "up" else (slope(lower), slope(upper))
    if far == 0:
        return np.inf
    return float(near / far)


def plateau_target(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, sign: float,
                   direction: str = "up") -> np.ndarray:
    """y' = y - g(x) + g(clip(x at training boundary)), g = isotonic fit of y on x.
    Inside the training range y' == y; beyond it the feature's effect is frozen."""
    y_new = y.copy()
    ok = np.isfinite(x)
    iso = IsotonicRegression(increasing=bool(sign >= 0), out_of_bounds="clip")
    iso.fit(x[ok], y[ok])
    if direction == "up":
        b = np.nanmax(x[train_idx])
        xc = np.minimum(x[ok], b)
    else:
        b = np.nanmin(x[train_idx])
        xc = np.maximum(x[ok], b)
    y_new[ok] = y[ok] - iso.predict(x[ok]) + iso.predict(xc)
    return y_new


# ---------------------------------------------------------------- main API
def build_splits(ds: Dataset, kinds=("extrap", "saturation"), severities=SEVERITIES,
                 train_frac: float = 0.30, band: float = 0.20, selection: str = "full",
                 holdout_frac: float = 0.30, matched_tol: float = 0.75,
                 split_seed: int = 0, min_unique: int = 20) -> list[Split]:
    rng = np.random.default_rng(split_seed)
    all_rows = np.arange(ds.n)
    if selection == "holdout":
        perm = rng.permutation(ds.n)
        n_sel = int(round(holdout_frac * ds.n))
        sel_rows, pool = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])
    elif selection == "full":
        sel_rows, pool = all_rows, all_rows
    else:
        raise ValueError(selection)

    cands = candidate_features(ds, min_unique=min_unique)
    if not cands:
        raise ValueError(f"{ds.name}: no numeric feature with >= {min_unique} unique values")
    X = {c: ds.X[c].to_numpy(dtype=float) for c in cands}
    y = ds.y

    # extrapolation feature: max |Spearman| on selection rows
    rhos = {c: abs_spearman(X[c][sel_rows], y[sel_rows]) for c in cands}
    f_ext = max(rhos, key=lambda c: rhos[c][0])
    rho_ext, sign_ext = rhos[f_ext]

    # saturation candidates: train-only slope ratio on the selection rows
    sat_table = []
    for c in cands:
        for d in ("up", "down"):
            tr, _ = band_split(X[c], sel_rows[np.isfinite(X[c][sel_rows])], train_frac, 0, band, d)
            sat_table.append({"feature": c, "direction": d, "abs_rho": rhos[c][0],
                              "ratio": slope_ratio(X[c][tr], y[tr], d)})

    def pick_sat(rule: str):
        rows = [r for r in sat_table if np.isfinite(r["ratio"])]
        rows = [r for r in rows if not (r["feature"] == f_ext and r["direction"] == "up")] or rows
        if rule == "paper":
            return min(rows, key=lambda r: r["ratio"])
        pos = [r for r in rows if r["ratio"] >= 0] or rows
        if rule == "robust":
            return min(pos, key=lambda r: abs(r["ratio"]))
        if rule == "matched":
            # a *different* feature of comparable leverage (the same feature in the
            # opposite direction is already covered by kind="reverse")
            pos = [r for r in pos if r["feature"] != f_ext] or pos
            strong = [r for r in pos if r["abs_rho"] >= matched_tol * rho_ext]
            if strong:
                return min(strong, key=lambda r: abs(r["ratio"]))
            flat = [r for r in pos if r["ratio"] < 1] or pos
            return min(flat, key=lambda r: abs(r["abs_rho"] - rho_ext))  # nearest leverage
        raise ValueError(rule)

    def valid(c):
        return pool[np.isfinite(X[c][pool])]

    out: list[Split] = []
    base_meta = {"extrap_feature": f_ext, "extrap_abs_rho": rho_ext, "selection": selection}
    for kind in kinds:
        if kind == "id":
            tr0, te0 = band_split(X[f_ext], valid(f_ext), train_frac, 0, band, "up")
            rows = valid(f_ext)
            perm = rng.permutation(rows)
            tr, te = np.sort(perm[:len(tr0)]), np.sort(perm[len(tr0):len(tr0) + len(te0)])
            out.append(Split(ds.name, "id", np.nan, None, None, tr, te, meta=dict(base_meta)))
            continue
        if kind in ("extrap", "reverse", "plateau"):
            feat, d = f_ext, ("down" if kind == "reverse" else "up")
            meta = dict(base_meta, abs_rho=rho_ext)
        elif kind.startswith("saturation"):
            rule = {"saturation": "paper", "saturation_robust": "robust",
                    "saturation_matched": "matched"}[kind]
            r = pick_sat(rule)
            feat, d = r["feature"], r["direction"]
            meta = dict(base_meta, abs_rho=r["abs_rho"], slope_ratio=r["ratio"],
                        degenerate=(feat == f_ext and d == "up"))
        else:
            raise ValueError(f"unknown split kind {kind}")
        if kind in ("extrap", "plateau"):
            tr_e, _ = band_split(X[f_ext], valid(f_ext), train_frac, 0, band, "up")
            meta["slope_ratio"] = slope_ratio(X[f_ext][tr_e], y[tr_e], "up")
        for sev in severities:
            tr, te = band_split(X[feat], valid(feat), train_frac, sev, band, d)
            y_over = plateau_target(X[feat], y, tr, sign_ext, d) if kind == "plateau" else None
            out.append(Split(ds.name, kind, float(sev), feat, d, tr, te, y=y_over, meta=dict(meta)))
    return out
