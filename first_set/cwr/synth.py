"""Synthetic regime data with sign-flipping feature effects.

Each sample belongs to a latent regime; within regime r the target is linear, y = X @ beta_r,
but beta_r differs (and flips sign) across regimes (controlled by sign_flip). The regime boundary
can be axis-aligned (easy for trees) or oblique (a random hyperplane mix of features, hard for
greedy axis-aligned trees). This is the data type where explicit local regression should have an
edge over a global GBM, IF the gating can recover the regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import norm


def make_regime_data(n=20000, d=8, k=3, oblique=True, sign_flip=1.0, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    if oblique:
        W = rng.normal(size=(d, k))          # regime = argmax of oblique linear scores
        regime = (X @ W).argmax(axis=1)
    else:
        edges = np.quantile(X[:, 0], np.linspace(0, 1, k + 1))  # regime = bins of feature 0
        regime = np.clip(np.digitize(X[:, 0], edges[1:-1]), 0, k - 1)

    base = rng.normal(size=d)
    y = np.zeros(n)
    for r in range(k):
        signs = rng.choice([-1.0, 1.0], size=d)
        beta_r = base + sign_flip * signs * np.abs(rng.normal(size=d))
        m = regime == r
        y[m] = X[m] @ beta_r
    y = y + noise * np.std(y) * rng.normal(size=n)
    return X, y, regime


def make_regime_data_nonlinear(n=20000, d=8, k=3, oblique=True, noise=0.2, seed=0, rotation=None):
    """Like make_regime_data but each regime has a DIFFERENT NONLINEAR relationship (curvy, not a
    straight line), on a regime-specific direction.

    rotation in [0,1] is a continuous 'obliqueness' knob for the regime boundary: the regime is
    argmax over k of X @ W, where W interpolates from an axis-aligned basis (rotation=0, easy for
    greedy trees) to a fully random oblique mix (rotation=1, hard for trees). If rotation is None it
    falls back to the old boolean `oblique` (True->1, False->0)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    if rotation is None:
        rotation = 1.0 if oblique else 0.0
    # Grinsztajn-style: rotate the coordinates, then define the regime by an axis-aligned partition
    # in the rotated frame. rotation=0 -> identity (partition on raw features, easy for greedy trees);
    # rotation=1 -> random rotation (partition on an oblique feature mix, hard for greedy trees).
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    R, _ = np.linalg.qr((1.0 - rotation) * np.eye(d) + rotation * Q)
    Xr = X @ R
    # axis-aligned k-way partition in the rotated frame: bins of the first two rotated coords
    b0 = np.digitize(Xr[:, 0], np.quantile(Xr[:, 0], np.linspace(0, 1, int(np.ceil(k / 2)) + 1)[1:-1]))
    b1 = (Xr[:, 1] > np.median(Xr[:, 1])).astype(int) if k > 1 else 0
    regime = np.clip(b0 * (2 if k > 1 else 1) + b1, 0, k - 1)
    y = np.zeros(n)
    for r in range(k):
        m = regime == r
        z = X[m] @ rng.normal(size=d)  # regime-specific single index
        if r % 3 == 0:
            yr = np.sin(2.5 * z)
        elif r % 3 == 1:
            yr = 0.5 * z ** 2 - z
        else:
            yr = np.tanh(2.0 * z)
        y[m] = yr
    y = y + noise * np.std(y) * rng.normal(size=n)
    return X, y, regime


# =====================================================================================
# Controlled shift injection (WP1, journal shift-robustness design protocol).
# Contract: design.md §5.1 (exact DGP) + frozen-protocol Lock-in 4 (covariate operational),
# Lock-in 5 (code freeze). Geometry (rotation) and shift are ORTHOGONAL knobs by construction:
# all structure (regime partition, regime function, extrapolation axis, shift) is defined in a
# STRUCTURAL FRAME S ~ N(0, I); the observed features are X = S @ R with R orthogonal. Because S
# is isotropic and R is orthogonal, the observed marginal is N(0, I) for every rotation and any
# shift defined on S is distributionally invariant to R. rotation=0 -> X = S (axis-aligned, tree-
# friendly); rotation=1 -> oblique mix (hard for greedy axis-aligned trees), same as the legacy
# `rotation` knob above. Regime partition uses structural coords {0, 1}; the extrapolation /
# covariate-shift axis is a SEPARATE coord (e_idx = 2), so covariate region-restriction does not
# perturb P(regime) (clean covariate shift). Requires d >= 3.
# =====================================================================================

# Frozen shift constants (part of the pre-specified config; snapshotted in results/frozen/).
_MEAN_MAX = 3.0            # covariate/mean: test-side mean offset (std units) at severity=1
_REGION_MAX = 0.40        # covariate/region: max half-gap (quantile units) at severity=1
_CONCEPT_MAX = np.pi / 2  # concept: max rotation of the regime direction at severity=1
_TARGET_TILT = 2.0        # target: max exponential tilt of regime mixture weights at severity=1
_DEFAULT_KNEE_Q = 0.85    # plateau knee quantile when no region restriction is active

SHIFT_KINDS = ("none", "covariate", "concept", "target")
EXTRAP_REGIMES = ("linear", "plateau", "oscillate")
COVARIATE_MODES = ("region", "mean")


@dataclass
class ShiftSpec:
    """One controlled shift cell. `kind` selects the shift family; `severity` in [0, 1] sweeps its
    magnitude (0 = no shift = i.i.d. train/test). See module header + design.md §5.1."""
    kind: str = "none"                 # none | covariate | concept | target
    severity: float = 0.0              # continuous magnitude, 0 -> i.i.d.
    covariate_mode: str = "region"     # region (primary, Lock-in 4) | mean (secondary)
    extrap: str = "linear"             # linear | plateau | oscillate (out-of-support target)
    extrap_amp: float = 1.0            # amplitude of the additive extrapolation term

    def __post_init__(self):
        if self.kind not in SHIFT_KINDS:
            raise ValueError(f"kind must be one of {SHIFT_KINDS}, got {self.kind!r}")
        if self.covariate_mode not in COVARIATE_MODES:
            raise ValueError(f"covariate_mode must be one of {COVARIATE_MODES}")
        if self.extrap not in EXTRAP_REGIMES:
            raise ValueError(f"extrap must be one of {EXTRAP_REGIMES}")


@dataclass
class _World:
    """Deterministic DGP parameters (fixed by seed). Shared by train and test EXCEPT the regime
    directions `v`, which differ between train- and test-world only under concept shift."""
    d: int
    k: int
    R: np.ndarray             # (d, d) orthogonal rotation: X = S @ R
    e_idx: int                # structural coord used for extrapolation / covariate shift
    edges0: np.ndarray        # theoretical normal quantile edges partitioning coord 0
    v: list                   # per-regime single-index directions in structural frame (d,)
    nl: list                  # per-regime nonlinearity id (0=sin, 1=quad, 2=tanh)
    extrap: str
    amp: float
    knee: float               # plateau knee location along e_idx


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _rotation_matrix(d, rotation, rng):
    """Same identity->random interpolation as make_regime_data_nonlinear (Grinsztajn-style)."""
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    R, _ = np.linalg.qr((1.0 - rotation) * np.eye(d) + rotation * Q)
    return R


def _build_world(d, k, rotation, extrap, amp, rng):
    if d < 3:
        raise ValueError("shift injection requires d >= 3 (coord 2 reserved for the shift axis)")
    R = _rotation_matrix(d, rotation, rng)
    n0 = int(np.ceil(k / 2))
    edges0 = norm.ppf(np.linspace(0, 1, n0 + 1)[1:-1]) if n0 > 1 else np.array([])
    v = [_unit(rng.standard_normal(d)) for _ in range(k)]
    nl = [r % 3 for r in range(k)]
    knee = float(norm.ppf(_DEFAULT_KNEE_Q))
    return _World(d=d, k=k, R=R, e_idx=2, edges0=edges0, v=v, nl=nl,
                  extrap=extrap, amp=float(amp), knee=knee)


def _regime_of(S, w: _World):
    """Latent regime = axis-aligned partition of the structural frame (coords 0 and 1)."""
    b0 = np.digitize(S[:, 0], w.edges0) if w.edges0.size else np.zeros(len(S), int)
    b1 = (S[:, 1] > 0.0).astype(int) if w.k > 1 else np.zeros(len(S), int)
    return np.clip(b0 * (2 if w.k > 1 else 1) + b1, 0, w.k - 1)


def _target_signal(S, w: _World, regime):
    """Noise-free target E[Y|X] under world w. Regime-specific single-index nonlinearity plus an
    additive extrapolation term g(coord e_idx) whose out-of-support behavior is set by w.extrap."""
    y = np.zeros(len(S))
    for r in range(w.k):
        m = regime == r
        if not m.any():
            continue
        z = S[m] @ w.v[r]
        if w.nl[r] == 0:
            y[m] = np.sin(2.5 * z)
        elif w.nl[r] == 1:
            y[m] = 0.5 * z ** 2 - z
        else:
            y[m] = np.tanh(2.0 * z)
    t = S[:, w.e_idx]
    if w.extrap == "linear":
        g = t
    elif w.extrap == "plateau":
        g = np.minimum(t, w.knee)   # flat beyond the (train) knee -> trees are RIGHT there
    else:  # oscillate
        g = np.sin(2.0 * t)
    return y + w.amp * g


def _perturb_concept(w: _World, severity, rng):
    """Test-world for concept shift: rotate each regime direction toward an orthogonal direction by
    angle severity*(pi/2). P(X) and the regime partition are untouched; only E[Y|X] changes."""
    a = severity * _CONCEPT_MAX
    v2 = []
    for vr in w.v:
        u = rng.standard_normal(w.d)
        u = _unit(u - (u @ vr) * vr)           # component orthogonal to vr
        v2.append(_unit(np.cos(a) * vr + np.sin(a) * u))
    return replace(w, v=v2)


def _sample_structural(rng, n, d):
    return rng.standard_normal((n, d))


def _covariate_region(rng, n_train, n_test, d, e_idx, severity):
    """Primary covariate operational (Lock-in 4): restrict train to the low sub-region of coord
    e_idx, test to the disjoint high sub-region. Returns (S_train, S_test, knee)."""
    w = severity * _REGION_MAX
    q_lo = float(norm.ppf(0.5 - w))            # train upper bound
    q_hi = float(norm.ppf(0.5 + w))            # test lower bound
    frac = max(0.5 - w, 1e-3)
    tr, te = [], []
    need_tr, need_te = n_train, n_test
    while need_tr > 0 or need_te > 0:
        pool = int(np.ceil(max(need_tr, need_te) / frac) * 1.6) + 1000
        S = rng.standard_normal((pool, d))
        t = S[:, e_idx]
        if need_tr > 0:
            got = S[t <= q_lo][:need_tr]
            tr.append(got); need_tr -= len(got)
        if need_te > 0:
            got = S[t >= q_hi][:need_te]
            te.append(got); need_te -= len(got)
    return np.vstack(tr)[:n_train], np.vstack(te)[:n_test], q_lo


def _target_reweight(rng, n_train, n_test, d, w: _World, severity):
    """Target/label shift: keep P(X|regime) and E[Y|X] fixed, tilt the regime mixture weights so
    P(regime) (hence P(Y)) shifts. Train and test are drawn from SEPARATE i.i.d. pools and then
    sub-sampled per regime to hit the target proportions -> within-regime P(X) stays identical."""
    tilt = np.arange(w.k) - (w.k - 1) / 2.0
    scale = max(np.abs(tilt).max(), 1.0)
    w_te = np.exp(severity * _TARGET_TILT * tilt / scale)
    w_te /= w_te.sum()
    w_tr = np.full(w.k, 1.0 / w.k)             # train: uniform mixture

    def draw(weights, n):
        counts = np.floor(weights * n).astype(int)
        while counts.sum() < n:                # distribute rounding remainder to largest weights
            counts[np.argmax(weights * n - counts)] += 1
        parts, need = [None] * w.k, counts.copy()
        while need.sum() > 0:
            pool = int(np.ceil(need.sum() / min(w_tr.min(), w_te.min())) * 1.5) + 1000
            S = rng.standard_normal((pool, d))
            reg = _regime_of(S, w)
            for r in range(w.k):
                if need[r] <= 0:
                    continue
                got = S[reg == r][:need[r]]
                parts[r] = got if parts[r] is None else np.vstack([parts[r], got])
                need[r] -= len(got)
        out = np.vstack([p for p in parts if p is not None])
        rng.shuffle(out)
        return out

    return draw(w_tr, n_train), draw(w_te, n_test)


def make_shifted_regime_data(spec: ShiftSpec = None, n_train=8000, n_test=4000, d=8, k=3,
                             rotation=0.0, noise=0.2, seed=0, return_worlds=False, **spec_kw):
    """Matched (train, test) draw under a controlled distribution shift.

    Geometry (rotation) and shift (spec) are orthogonal knobs: rotation sets the observed coordinate
    frame (X = S @ R), the shift is applied in the structural frame S, so a fixed spec produces the
    SAME shift under any rotation ("same shift, two geometries"). See module header + design.md §5.1.

    Returns a dict with X_train, y_train, X_test, y_test, regime_train, regime_test, shift_dir (the
    true covariate-shift direction in observed space), regime_coords, and meta. If return_worlds,
    also world_train / world_test (for the E[Y|X] conditional probe used by the validation suite).
    """
    if spec is None:
        spec = ShiftSpec(**spec_kw)
    rng = np.random.default_rng(seed)
    world = _build_world(d, k, rotation, spec.extrap, spec.extrap_amp, rng)
    world_tr = world
    world_te = _perturb_concept(world, spec.severity, rng) if spec.kind == "concept" else world

    sev = float(spec.severity)
    kind = "none" if sev <= 0 else spec.kind
    if kind == "none":
        Str, Ste = _sample_structural(rng, n_train, d), _sample_structural(rng, n_test, d)
    elif kind == "covariate":
        if spec.covariate_mode == "region":
            Str, Ste, knee = _covariate_region(rng, n_train, n_test, d, world.e_idx, sev)
            world_tr = replace(world_tr, knee=knee)   # plateau knee at the train support boundary
            world_te = replace(world_te, knee=knee)
        else:  # mean
            Str, Ste = _sample_structural(rng, n_train, d), _sample_structural(rng, n_test, d)
            Ste[:, world.e_idx] += sev * _MEAN_MAX
    elif kind == "concept":
        Str, Ste = _sample_structural(rng, n_train, d), _sample_structural(rng, n_test, d)
    elif kind == "target":
        Str, Ste = _target_reweight(rng, n_train, n_test, d, world, sev)

    reg_tr, reg_te = _regime_of(Str, world), _regime_of(Ste, world)

    # Noise scale from a fixed reference pool under the train world (deterministic; no test leakage).
    Sref = _sample_structural(rng, 20000, d)
    nscale = noise * float(np.std(_target_signal(Sref, world_tr, _regime_of(Sref, world))))

    y_tr = _target_signal(Str, world_tr, reg_tr) + nscale * rng.standard_normal(n_train)
    y_te = _target_signal(Ste, world_te, reg_te) + nscale * rng.standard_normal(n_test)
    X_tr, X_te = Str @ world.R, Ste @ world.R

    out = {
        "X_train": X_tr, "y_train": y_tr, "regime_train": reg_tr,
        "X_test": X_te, "y_test": y_te, "regime_test": reg_te,
        "shift_dir": world.R[world.e_idx, :].copy(),   # obs-space direction of the covariate axis
        "regime_coords": world.R[[0, 1], :].copy(),
        "meta": {"shift": kind, "severity": sev, "covariate_mode": spec.covariate_mode,
                 "extrap": spec.extrap, "extrap_amp": spec.extrap_amp, "rotation": rotation,
                 "d": d, "k": k, "seed": seed, "n_train": n_train, "n_test": n_test,
                 "noise_scale": nscale},
    }
    if return_worlds:
        out["world_train"], out["world_test"] = world_tr, world_te
        out["_target_signal"] = _target_signal
        out["_regime_of"] = _regime_of
    return out
