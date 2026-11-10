"""Generator-validation suite for the controlled shift injection (WP1 design protocol guarantee).

Quantitatively verifies that each shift kind produces WHAT IT CLAIMS and that geometry (rotation)
and shift are ORTHOGONAL knobs. This is the design protocol's "the DGP does what it says" evidence
(design.md §5.1 + frozen-protocol Lock-in 4). No OOD model numbers here -- pure distributional checks.

Signatures verified (measured on the OBSERVED features X the model sees, projected onto the true
structural axes via the returned R rows):

  shift            KS(coord0, regime axis)  KS(coord2, shift axis)  |dP(regime)|  dE[Y|X]
  covariate/*      small                    BIG                     small         ~0
  concept          small                    small                   small         BIG
  target           (moderate)               small                   BIG           ~0
  none             small                    small                   small         ~0

Run:  python experiments/validate_synth.py     (from code/, or anywhere -- path is bootstrapped)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from scipy.stats import ks_2samp
from sklearn.tree import DecisionTreeRegressor

from cwr.synth import ShiftSpec, make_shifted_regime_data

# Decision thresholds (fixed; report is deterministic given seeds).
BIG_KS = 0.30
SMALL_KS = 0.10
REGIME_SHIFT = 0.03      # max |P_train(regime) - P_test(regime)|
COND_BIG = 0.05          # normalized mean |E[Y|X]_train - E[Y|X]_test|
COND_SMALL = 0.012


def regime_prop_gap(reg_tr, reg_te, k):
    p_tr = np.array([(reg_tr == r).mean() for r in range(k)])
    p_te = np.array([(reg_te == r).mean() for r in range(k)])
    return float(np.abs(p_tr - p_te).max())


def conditional_delta(out, n_grid=6000):
    """Normalized mean |E[Y|X]_train - E[Y|X]_test| on a COMMON structural grid. Isolates concept
    (the only shift that changes the target function) from covariate/target (function unchanged)."""
    wt, we = out["world_train"], out["world_test"]
    tgt, reg_of = out["_target_signal"], out["_regime_of"]
    rng = np.random.default_rng(12345)
    S = rng.standard_normal((n_grid, wt.d))
    reg = reg_of(S, wt)
    yt = tgt(S, wt, reg)
    ye = tgt(S, we, reg)
    return float(np.mean(np.abs(yt - ye)) / (np.std(yt) + 1e-9))


def diagnostics(spec, rotation, seed, n_train=8000, n_test=6000, d=8, k=3):
    out = make_shifted_regime_data(spec, n_train=n_train, n_test=n_test, d=d, k=k,
                                   rotation=rotation, seed=seed, return_worlds=True)
    Xtr, Xte = out["X_train"], out["X_test"]
    ax0 = out["regime_coords"][0]           # observed-space direction of structural coord 0
    ax2 = out["shift_dir"]                   # observed-space direction of structural coord 2
    ks0 = ks_2samp(Xtr @ ax0, Xte @ ax0).statistic
    ks2 = ks_2samp(Xtr @ ax2, Xte @ ax2).statistic
    ksy = ks_2samp(out["y_train"], out["y_test"]).statistic
    gap = regime_prop_gap(out["regime_train"], out["regime_test"], k)
    cond = conditional_delta(out)
    return dict(ks0=ks0, ks2=ks2, ksy=ksy, gap=gap, cond=cond)


def gt_obliqueness(spec, rotation, seed, d=8, k=3):
    """Ground-truth obliqueness of the observed features: 1 - treeR2(X) / treeR2(S), where S is the
    TRUE structural frame (X @ R^T). ~0 when structure is axis-aligned in X (rotation 0), large when
    oblique (rotation 1). Used to show geometry depends on rotation, NOT on the shift kind. (Lock-in
    6 uses a PCA surrogate on real data where R is unknown; here R is known so we use it directly.)"""
    out = make_shifted_regime_data(spec, n_train=6000, n_test=10, d=d, k=k,
                                   rotation=rotation, seed=seed, return_worlds=True)
    X, y, R = out["X_train"], out["y_train"], out["world_train"].R
    S = X @ R.T

    def tr2(F):
        return DecisionTreeRegressor(max_depth=4, random_state=0).fit(F, y).score(F, y)

    return 1.0 - tr2(X) / max(tr2(S), 1e-6)


def verdict(ok):
    return "PASS" if ok else "**FAIL**"


def main():
    seeds = [0, 1, 2]
    cases = [
        ("covariate/region", ShiftSpec("covariate", 0.7, covariate_mode="region")),
        ("covariate/mean", ShiftSpec("covariate", 0.7, covariate_mode="mean")),
        ("concept", ShiftSpec("concept", 0.7)),
        ("target", ShiftSpec("target", 0.7)),
        ("none", ShiftSpec("none", 0.0)),
    ]
    all_ok = True

    print("=" * 92)
    print("1) SHIFT SIGNATURES  (mean over seeds; measured on observed X projected onto true axes)")
    print("=" * 92)
    hdr = f"{'case':18s}{'rot':>5}{'KS_coord0':>11}{'KS_coord2':>11}{'KS_y':>8}{'|dPreg|':>9}{'dE[Y|X]':>10}   verdict"
    print(hdr)
    for name, spec in cases:
        for rot in (0.0, 1.0):
            ds = [diagnostics(spec, rot, s) for s in seeds]
            m = {kk: float(np.mean([dd[kk] for dd in ds])) for kk in ds[0]}
            # expected signature per kind
            if spec.kind == "covariate":
                ok = (m["ks2"] > BIG_KS and m["ks0"] < SMALL_KS and m["gap"] < REGIME_SHIFT
                      and m["cond"] < COND_SMALL)
            elif spec.kind == "concept":
                ok = (m["cond"] > COND_BIG and m["ks2"] < SMALL_KS and m["ks0"] < SMALL_KS
                      and m["gap"] < REGIME_SHIFT)
            elif spec.kind == "target":
                ok = (m["gap"] > REGIME_SHIFT and m["ks2"] < SMALL_KS and m["cond"] < COND_SMALL)
            else:  # none
                ok = (m["ks2"] < SMALL_KS and m["ks0"] < SMALL_KS and m["gap"] < REGIME_SHIFT
                      and m["cond"] < COND_SMALL)
            all_ok &= ok
            print(f"{name:18s}{rot:>5.1f}{m['ks0']:>11.3f}{m['ks2']:>11.3f}{m['ksy']:>8.3f}"
                  f"{m['gap']:>9.3f}{m['cond']:>10.3f}   {verdict(ok)}")

    print()
    print("=" * 92)
    print("2) ORTHOGONALITY-A  (fixed shift, vary rotation -> shift magnitude must be INVARIANT)")
    print("=" * 92)
    spec = ShiftSpec("covariate", 0.6, covariate_mode="region")
    rows = []
    for rot in (0.0, 0.5, 1.0):
        ds = [diagnostics(spec, rot, s) for s in seeds]
        rows.append((rot, float(np.mean([d["ks2"] for d in ds])),
                     float(np.mean([d["gap"] for d in ds]))))
    ks2_vals = [r[1] for r in rows]
    ortho_a = (max(ks2_vals) - min(ks2_vals)) < 0.05
    all_ok &= ortho_a
    print(f"{'rotation':>10}{'KS_coord2(shift axis)':>24}{'|dPreg|':>10}")
    for rot, k2, gp in rows:
        print(f"{rot:>10.1f}{k2:>24.3f}{gp:>10.3f}")
    print(f"  -> covariate magnitude spread across rotations = {max(ks2_vals)-min(ks2_vals):.3f} "
          f"(must be <0.05)  {verdict(ortho_a)}")

    print()
    print("=" * 92)
    print("3) ORTHOGONALITY-B  (geometry is a property of rotation, NOT of the shift kind)")
    print("=" * 92)
    # Reference: pure-R obliqueness on a SHIFT-FREE probe -> the DGP-level geometry (exact, no shift
    # sample confound). Then the estimate FROM each shifted train sample clusters around it with
    # estimator/sample variance; the rotation effect must DOMINATE that cross-shift spread.
    ref0 = float(np.mean([gt_obliqueness(ShiftSpec("none", 0.0), 0.0, s) for s in seeds]))
    ref1 = float(np.mean([gt_obliqueness(ShiftSpec("none", 0.0), 1.0, s) for s in seeds]))
    print(f"  reference (shift-free, pure R):  obliq@rot=0={ref0:.3f}   obliq@rot=1={ref1:.3f}")
    shift_specs = [("covariate", ShiftSpec("covariate", 0.6)), ("concept", ShiftSpec("concept", 0.6)),
                   ("target", ShiftSpec("target", 0.6)), ("none", ShiftSpec("none", 0.0))]
    print(f"{'shift':>12}{'obliq@rot=0':>14}{'obliq@rot=1':>14}")
    obl0, obl1 = [], []
    for nm, sp in shift_specs:
        o0 = float(np.mean([gt_obliqueness(sp, 0.0, s) for s in seeds]))
        o1 = float(np.mean([gt_obliqueness(sp, 1.0, s) for s in seeds]))
        obl0.append(o0); obl1.append(o1)
        print(f"{nm:>12}{o0:>14.3f}{o1:>14.3f}")
    spread0 = max(obl0) - min(obl0)
    spread1 = max(obl1) - min(obl1)
    min_rise = min(o1 - o0 for o0, o1 in zip(obl0, obl1))
    rises = all(o1 > o0 + 0.05 for o0, o1 in zip(obl0, obl1))
    # Honest criterion: obliqueness ~0 at rot0 for ALL shifts (exact DGP invariance), rises for all,
    # and the rotation-induced change dominates the cross-shift estimator spread at rot1.
    dominates = min_rise > spread1
    ortho_b = spread0 < 0.05 and rises and dominates
    all_ok &= ortho_b
    print(f"  -> rot0 spread={spread0:.3f} (DGP-level, ~0); rises-for-all={rises}; "
          f"rotation effect (min rise={min_rise:.3f}) > cross-shift spread ({spread1:.3f}) "
          f"= {dominates}  {verdict(ortho_b)}")

    print()
    print("=" * 92)
    print("4) SEVERITY MONOTONICITY  (each shift's magnitude increases with its severity knob)")
    print("=" * 92)
    sevs = [0.0, 0.25, 0.5, 0.75, 1.0]
    cov = [np.mean([diagnostics(ShiftSpec("covariate", s), 0.0, sd)["ks2"] for sd in seeds]) for s in sevs]
    con = [np.mean([diagnostics(ShiftSpec("concept", s), 0.0, sd)["cond"] for sd in seeds]) for s in sevs]
    tar = [np.mean([diagnostics(ShiftSpec("target", s), 0.0, sd)["gap"] for sd in seeds]) for s in sevs]
    mono = (lambda a: all(x <= y + 1e-6 for x, y in zip(a, a[1:])))
    m_ok = mono(cov) and mono(con) and mono(tar)
    all_ok &= m_ok
    print(f"{'severity':>10}" + "".join(f"{s:>8.2f}" for s in sevs))
    print(f"{'cov KS2':>10}" + "".join(f"{v:>8.3f}" for v in cov))
    print(f"{'con dEYX':>10}" + "".join(f"{v:>8.3f}" for v in con))
    print(f"{'tar dPreg':>10}" + "".join(f"{v:>8.3f}" for v in tar))
    print(f"  -> all monotone non-decreasing in severity  {verdict(m_ok)}")

    print()
    print("=" * 92)
    print(f"OVERALL: {verdict(all_ok)}")
    print("=" * 92)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
