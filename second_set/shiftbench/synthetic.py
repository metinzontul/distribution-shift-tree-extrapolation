"""Fully specified synthetic generator (review item 9: the manuscript must state
the data-generating process exactly; this is one explicit choice to adapt).

Latent z ~ U(0,1)^d. The shift coordinate is z0.
Regime r in {0,1,2} = tertile of z1 (three latent sub-populations).

    y = A[r] * h(z0) + sum_{j=1..3} B[r, j] * sin(2*pi*z_{j+1}) + eps,
    eps ~ N(0, noise^2)

h ("rising")  = 3*z0 + 1.5*z0^2         target keeps rising beyond training
h ("plateau") = 3*min(z0, t) + 1.5*min(z0, t)^2, t = training-boundary of z0
                                          (reverse control: flat beyond training)

Geometry: "axis" -> features X = z; "oblique" -> X = z @ Q with Q a random
orthogonal matrix, so the informative structure is not axis-aligned.

Splits: train = lowest `train_frac` (default 10 %) of z0; test band of width
`band` (10 %) with lower edge train_frac + s * (1 - train_frac - band), so
s = 0 is adjacent and s = 1 is the top decile (lowest vs highest tenth).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset
from .splits import Split, percentile_rank

A = np.array([1.0, 1.5, 0.7])
B = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.3], [0.4, 0.1, 0.5]])
SYN_SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)


def make_synthetic(n=5000, d=8, geometry="axis", shape="rising", noise=0.1,
                   train_frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.random((n, d))
    r = np.digitize(z[:, 1], [1 / 3, 2 / 3])
    t = np.quantile(z[:, 0], train_frac)
    z0 = z[:, 0] if shape == "rising" else np.minimum(z[:, 0], t)
    h = 3 * z0 + 1.5 * z0 ** 2
    y = A[r] * h + sum(B[r, j] * np.sin(2 * np.pi * z[:, j + 1]) for j in range(3))
    y = y + rng.normal(0, noise, n)
    if geometry == "axis":
        X = z
    elif geometry == "oblique":
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        X = z @ Q
    else:
        raise ValueError(geometry)
    cols = [f"x{i}" for i in range(d)]
    name = f"syn_{geometry}_{shape}_s{seed}"
    ds = Dataset(name=name, X=pd.DataFrame(X, columns=cols), y=y, num_cols=cols, cat_cols=[])
    return ds, z[:, 0]


def synthetic_splits(ds: Dataset, shift_coord: np.ndarray, severities=SYN_SEVERITIES,
                     train_frac=0.10, band=0.10) -> list[Split]:
    pr = percentile_rank(shift_coord)
    rows = np.arange(ds.n)
    out = []
    for s in severities:
        lo = train_frac + s * (1 - train_frac - band)
        tr, te = rows[pr < train_frac], rows[(pr >= lo) & (pr < lo + band + 1e-12)]
        out.append(Split(ds.name, "synthetic", float(s), "z0", "up", tr, te, meta={}))
    return out
