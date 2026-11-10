"""WP3 model zoo (real-data mechanism-replication + severity-threshold map).

Extends the WP2 family representatives with the WP3 design protocol §9/§10 secondary set.

  M2 PRIMARY confirmatory pair (pre-committed, best-of FORBIDDEN):
      GBDT-primary = CatBoost   vs   NN-primary = FT-Transformer.
  Secondary (descriptive / robustness only, never the confirmatory verdict):
      XGBoost, LightGBM (extra GBDT) · MLP (extra NN) · TabPFN v2 (FM, regime-1 only).
  Baselines: constant-mean, ridge.

Everything reuses the WP2 leakage-safe wrappers (cwr.wp2_models) where they already exist; this
module only ADDS XGBoost / LightGBM / MLP plus their equal-budget HPO tuners so the C8 rule
("eşit 8-config HPO") covers every tunable family with an IDENTICAL protocol (same seed, same 20%
train-carved validation split, same random-config draw). TabPFN is excluded from HPO by design.

Dual-regime (M3): regime='tabpfn' (n<=~10k, TabPFN included) vs regime='large' (TabPFN dropped;
GBDT+NN only). Cross-regime comparison is NEVER made (design pin). Env: conda `torch` (GPU).
"""
from __future__ import annotations

import numpy as np

# Reuse the frozen WP2 wrappers + their identical-protocol tuners (do NOT re-implement).
from .wp2_models import (CatBoostGBDT, ConstantMean, FTTransformerNN, RidgeLinear, TabPFNv2,
                         _val_rmse, tune_catboost, tune_ft)

# --------------------------------------------------------------------------------------------------
# Extra GBDT representatives: XGBoost + LightGBM (secondary; CPU by default, fast on tabular)
# --------------------------------------------------------------------------------------------------

_XGB_SPACE = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "reg_lambda": [1.0, 3.0, 5.0, 9.0],
}
_LGB_SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "reg_lambda": [1.0, 3.0, 5.0, 9.0],
}
_MLP_SPACE = {
    "hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128)],
    "alpha": [1e-5, 1e-4, 1e-3],
    "learning_rate_init": [3e-4, 1e-3],
}


class XGBoostGBDT:
    """Secondary GBDT (WP3 §9). Fixed n_estimators (leakage-safe, no test-touched early stop);
    depth/lr/lambda are the tuned axes. device='cuda' uses the GPU hist tree method."""

    def __init__(self, n_estimators=600, random_state=0, device="cpu", params=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.params = params or {"max_depth": 6, "learning_rate": 0.05, "reg_lambda": 3.0}

    def fit(self, X, y):
        from xgboost import XGBRegressor
        self.m_ = XGBRegressor(n_estimators=self.n_estimators, tree_method="hist",
                               device=self.device, subsample=0.8, colsample_bytree=0.8,
                               random_state=self.random_state, verbosity=0, n_jobs=0, **self.params)
        self.m_.fit(np.asarray(X, np.float32), np.asarray(y, float))
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(np.asarray(X, np.float32)), float)


class LightGBMGBDT:
    """Secondary GBDT (WP3 §9). Fixed n_estimators; num_leaves/lr/lambda tuned."""

    def __init__(self, n_estimators=600, random_state=0, params=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.params = params or {"num_leaves": 31, "learning_rate": 0.05, "reg_lambda": 3.0}

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        self.m_ = LGBMRegressor(n_estimators=self.n_estimators, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, random_state=self.random_state, verbose=-1,
                                n_jobs=0, **self.params)
        self.m_.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(np.asarray(X, float)), float)


# --------------------------------------------------------------------------------------------------
# Extra NN representative: MLP (secondary). Target standardized with fit-train stats (leakage-safe).
# --------------------------------------------------------------------------------------------------


class MLPNN:
    """Secondary NN (WP3 §9) = sklearn MLPRegressor. Standardized X expected (caller scales);
    target is standardized with FIT-TRAIN stats only for optimizer stability, de-standardized at
    predict. Early stopping uses a validation split carved from FIT-TRAIN ONLY (sklearn
    validation_fraction is train-internal, never touches test)."""

    def __init__(self, hidden_layer_sizes=(256, 128), alpha=1e-4, learning_rate_init=1e-3,
                 max_iter=300, random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        y = np.asarray(y, float)
        self.y_mu_, self.y_sd_ = float(y.mean()), float(y.std() + 1e-8)
        ys = (y - self.y_mu_) / self.y_sd_
        self.m_ = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                               learning_rate_init=self.learning_rate_init, max_iter=self.max_iter,
                               early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                               random_state=self.random_state)
        self.m_.fit(np.asarray(X, float), ys)
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(np.asarray(X, float)), float) * self.y_sd_ + self.y_mu_


# --------------------------------------------------------------------------------------------------
# Equal-budget HPO for the extra families (C8: identical protocol to tune_catboost / tune_ft)
# --------------------------------------------------------------------------------------------------


def _draw(rng, values):
    """Draw one value by index (handles tuple configs like hidden_layer_sizes)."""
    return values[int(rng.integers(len(values)))]


def _tune(space, ctor, X, y, budget, random_state=0, return_curve=False):
    """Generic random-search tuner, IDENTICAL protocol to tune_catboost/tune_ft (C8): same seed,
    same 20% train-carved validation split, same random-config draw -> family-fair HPO. `ctor(p)`
    builds a leakage-safe model for a config dict `p`. Returns best config (or (best, curve))."""
    rng = np.random.default_rng(random_state)
    n = len(X); perm = rng.permutation(n); nv = max(int(0.2 * n), 1)
    va, tr = perm[:nv], perm[nv:]
    best, best_p, curve = float("inf"), None, []
    for _ in range(budget):
        p = {k: _draw(rng, v) for k, v in space.items()}
        r = _val_rmse(ctor(p), X[tr], y[tr], X[va], y[va])
        curve.append({"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
                      "val_rmse": r})
        if r < best:
            best, best_p = r, p
    return (best_p, curve) if return_curve else best_p


def tune_xgb(X, y, budget, random_state=0, device="cpu", return_curve=False):
    return _tune(_XGB_SPACE, lambda p: XGBoostGBDT(random_state=random_state, device=device, params=p),
                 X, y, budget, random_state, return_curve)


def tune_lgb(X, y, budget, random_state=0, return_curve=False):
    return _tune(_LGB_SPACE, lambda p: LightGBMGBDT(random_state=random_state, params=p),
                 X, y, budget, random_state, return_curve)


def tune_mlp(X, y, budget, random_state=0, return_curve=False):
    return _tune(_MLP_SPACE, lambda p: MLPNN(random_state=random_state, **p),
                 X, y, budget, random_state, return_curve)


# --------------------------------------------------------------------------------------------------
# Factory (dual-regime) + tuner registry
# --------------------------------------------------------------------------------------------------

# The two families whose paired gap is the M2 confirmatory verdict (pre-committed representatives).
PRIMARY_PAIR = ("catboost", "ft_transformer")
SECONDARY_MODELS = ("xgboost", "lightgbm", "mlp", "tabpfn")
BASELINE_MODELS = ("constant_mean", "ridge")
# Tunable families (equal 8-config HPO). TabPFN is excluded from HPO by design.
TUNABLE = ("catboost", "ft_transformer", "xgboost", "lightgbm", "mlp")

TABPFN_REGIME_MAX_N = 10000   # M3 dual-regime cutoff: TabPFN only in regime-1 (n<=~10k)
TABPFN_MAX_FEATURES = 500     # TabPFN v2 feature ceiling: >500 cols -> drop TabPFN (e.g. maps-routing 984)


def build_wp3_models(regime, device="cuda", tabpfn_max_train=3000, random_state=0, tuned=None,
                     n_features=None):
    """Family + baseline models for one cell. `regime` in {'tabpfn','large'} controls whether the
    FM representative (TabPFN v2) is present (M3 dual-regime; NO cross-regime comparison). `tuned`
    is an optional {family: best_config} map from the equal-budget HPO; missing families use the
    frozen default config. CatBoost/XGB/LGB run CPU (fast); FT-Transformer/TabPFN use `device`.

    TabPFN is additionally gated on `n_features`: TabPFN v2 does not scale past ~500 features, so a
    high-dimensional set (maps-routing, 984 features) drops TabPFN even in the tabpfn regime; the
    GBDT/NN families stay. This is a per-dataset feasibility gate on the SECONDARY FM family, never a
    change to the CatBoost-vs-FT confirmatory pair."""
    tuned = tuned or {}
    models = {
        "constant_mean": ConstantMean(),
        "ridge": RidgeLinear(alpha=1.0),
        "catboost": CatBoostGBDT(random_state=random_state, params=tuned.get("catboost")),
        "ft_transformer": FTTransformerNN(device=device, random_state=random_state,
                                          **(tuned.get("ft_transformer") or {})),
        "xgboost": XGBoostGBDT(random_state=random_state, params=tuned.get("xgboost")),
        "lightgbm": LightGBMGBDT(random_state=random_state, params=tuned.get("lightgbm")),
        "mlp": MLPNN(random_state=random_state, **(tuned.get("mlp") or {})),
    }
    tabpfn_feat_ok = (n_features is None) or (int(n_features) <= TABPFN_MAX_FEATURES)
    if regime == "tabpfn" and tabpfn_feat_ok:
        models["tabpfn"] = TabPFNv2(device=device, max_train=tabpfn_max_train,
                                    random_state=random_state)
    return models


def tune_all(Z_fit, y_fit, budget, device="cuda", random_state=0):
    """Run the equal-budget HPO for every tunable family (C8) and return
    ({family: best_config}, {family: tuning_curve}). TabPFN excluded by design."""
    best, curves = {}, {}
    best["catboost"], curves["catboost"] = tune_catboost(Z_fit, y_fit, budget,
                                                          random_state=random_state, return_curve=True)
    best["ft_transformer"], curves["ft_transformer"] = tune_ft(Z_fit, y_fit, budget, device=device,
                                                               random_state=random_state, return_curve=True)
    best["xgboost"], curves["xgboost"] = tune_xgb(Z_fit, y_fit, budget, random_state=random_state,
                                                  return_curve=True)
    best["lightgbm"], curves["lightgbm"] = tune_lgb(Z_fit, y_fit, budget, random_state=random_state,
                                                    return_curve=True)
    best["mlp"], curves["mlp"] = tune_mlp(Z_fit, y_fit, budget, random_state=random_state,
                                          return_curve=True)
    return best, curves
