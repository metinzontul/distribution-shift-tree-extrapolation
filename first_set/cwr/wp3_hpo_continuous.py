"""WP3-REVISION continuous-space HPO ablation (internal review).

PURPOSE (descriptive / exploratory ONLY, NOT a mechanism proof): answer the reviewer's "does a much
larger HPO budget close the CatBoost-FT extrapolation gap?" with a budget sweep {8, 50, 100}.

Gate-mandated design (methodology BLOCKER1 + DA-A3), implemented here:
  * ONE authoritative CONTINUOUS search space per family (literature-based ranges) -- NOT the frozen
    discrete 8-grid (which saturates at large budget -> "more search" would be fake).
  * SAME algorithm as main text: RANDOM SEARCH (NOT TPE/Bayesian) so only the trial count changes
    across budgets -> no algorithm x budget confound. This reuses the EXACT `_tune` protocol from
    wp2_models (same seed, same 20% fit-train-carved val split computed BEFORE sampling, same per-
    budget random draw); only the per-config SAMPLER changes from discrete-list to continuous-dist.
  * Every budget {8, 50, 100} is drawn FRESH from this continuous space (budget-8 is NOT the frozen
    grid-8; only #trials differs, space fixed).
  * FT gets a genuinely HIGH-DIMENSIONAL space (n_blocks, d_block, 3 dropouts, ffn-multiplier, lr,
    weight_decay = 8 dims) vs GBDT's 3 dims -- this is the point: at equal #trials the NN space is
    sampled more sparsely (DA-A3), so the sweep shows whether more trials help the NN.
  * Config keys are exactly those the frozen model ctors accept (wp2_models / wp3_models), so
    build_wp3_models(tuned=...) consumes them unchanged (FTTransformerNN gained an `arch` override).

Literature ranges:
  FT-Transformer  = gorishniy2021 (Revisiting Deep Learning for Tabular Data) tuning space:
      n_blocks UniformInt[1,4]; d_block LogUniformInt[64,512] (snapped to a multiple of 8 for the
      8-head attention); attention/ffn dropout Uniform[0,0.5]; residual dropout Uniform[0,0.2];
      ffn_d_hidden_multiplier Uniform[2/3,8/3]; lr LogUniform[1e-5,1e-3]; weight_decay LogUniform[1e-6,1e-3].
  MLP  (standard) = n_layers UniformInt[1,3]; width LogUniformInt[32,512] (uniform per layer);
      alpha LogUniform[1e-6,1e-2]; learning_rate_init LogUniform[1e-4,1e-2].
  CatBoost (standard GBDT) = depth UniformInt[4,10]; learning_rate LogUniform[0.02,0.3];
      l2_leaf_reg LogUniform[1,10].
  XGBoost  = max_depth UniformInt[3,10]; learning_rate LogUniform[0.02,0.3]; reg_lambda LogUniform[1,10].
  LightGBM = num_leaves LogUniformInt[15,255]; learning_rate LogUniform[0.02,0.3]; reg_lambda LogUniform[1,10].

RealMLP (holzmuller2024, pytabkit) is an EXTRA strong-default NN (reviewer's "fragile NN" target),
run with its meta-tuned default (NOT in the HPO sweep). Wrapper below; injected via extra models.
"""
from __future__ import annotations

import numpy as np

from .wp2_models import (CatBoostGBDT, FTTransformerNN, _val_rmse)
from .wp3_models import LightGBMGBDT, MLPNN, XGBoostGBDT

# --------------------------------------------------------------------------------------------------
# Continuous draw helpers (every draw consumes the SAME rng as _tune, preserving the protocol)
# --------------------------------------------------------------------------------------------------


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _logu_int(rng, lo, hi):
    return int(round(_logu(rng, lo, hi)))


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _int(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))      # inclusive upper


def _snap(v, step):
    return int(max(step, round(v / step) * step))


# --------------------------------------------------------------------------------------------------
# Per-family continuous samplers (rng -> config dict directly usable by the frozen ctors)
# --------------------------------------------------------------------------------------------------


def _sample_catboost(rng):
    return {"depth": _int(rng, 4, 10), "learning_rate": _logu(rng, 0.02, 0.3),
            "l2_leaf_reg": _logu(rng, 1.0, 10.0)}


def _sample_xgb(rng):
    return {"max_depth": _int(rng, 3, 10), "learning_rate": _logu(rng, 0.02, 0.3),
            "reg_lambda": _logu(rng, 1.0, 10.0)}


def _sample_lgb(rng):
    return {"num_leaves": _logu_int(rng, 15, 255), "learning_rate": _logu(rng, 0.02, 0.3),
            "reg_lambda": _logu(rng, 1.0, 10.0)}


def _sample_mlp(rng):
    n_layers = _int(rng, 1, 3)
    width = _logu_int(rng, 32, 512)
    return {"hidden_layer_sizes": tuple([width] * n_layers),
            "alpha": _logu(rng, 1e-6, 1e-2), "learning_rate_init": _logu(rng, 1e-4, 1e-2)}


def _sample_ft(rng):
    n_blocks = _int(rng, 1, 4)
    arch = {"d_block": _snap(_logu_int(rng, 64, 512), 8),
            "attention_dropout": _u(rng, 0.0, 0.5), "ffn_dropout": _u(rng, 0.0, 0.5),
            "residual_dropout": _u(rng, 0.0, 0.2), "ffn_d_hidden_multiplier": _u(rng, 2.0 / 3.0, 8.0 / 3.0)}
    return {"n_blocks": n_blocks, "lr": _logu(rng, 1e-5, 1e-3),
            "weight_decay": _logu(rng, 1e-6, 1e-3), "arch": arch}


# --------------------------------------------------------------------------------------------------
# Generic continuous random-search tuner -- IDENTICAL protocol to wp2_models._tune / tune_catboost /
# tune_ft (same seed, same 20% fit-train val split computed BEFORE sampling, same per-budget draw).
# Only the sampler is continuous instead of discrete-list. (DA-A3: no algorithm x budget confound.)
# --------------------------------------------------------------------------------------------------


def _tune_continuous(sampler, ctor, X, y, budget, random_state=0, return_curve=False):
    rng = np.random.default_rng(random_state)
    n = len(X); perm = rng.permutation(n); nv = max(int(0.2 * n), 1)   # val split: fixed per (n, seed)
    va, tr = perm[:nv], perm[nv:]
    best, best_p, curve = float("inf"), None, []
    for _ in range(budget):
        p = sampler(rng)
        r = _val_rmse(ctor(p), X[tr], y[tr], X[va], y[va])
        curve.append({"config": _jsonable_config(p), "val_rmse": r})
        if r < best:
            best, best_p = r, p
    return (best_p, curve) if return_curve else best_p


def _jsonable_config(p):
    out = {}
    for k, v in p.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def tune_catboost_continuous(X, y, budget, random_state=0, device="cpu", return_curve=False):
    return _tune_continuous(_sample_catboost,
                            lambda p: CatBoostGBDT(random_state=random_state, params=p),
                            X, y, budget, random_state, return_curve)


def tune_xgb_continuous(X, y, budget, random_state=0, device="cpu", return_curve=False):
    return _tune_continuous(_sample_xgb,
                            lambda p: XGBoostGBDT(random_state=random_state, device=device, params=p),
                            X, y, budget, random_state, return_curve)


def tune_lgb_continuous(X, y, budget, random_state=0, device="cpu", return_curve=False):
    return _tune_continuous(_sample_lgb,
                            lambda p: LightGBMGBDT(random_state=random_state, params=p),
                            X, y, budget, random_state, return_curve)


def tune_mlp_continuous(X, y, budget, random_state=0, device="cpu", return_curve=False):
    return _tune_continuous(_sample_mlp,
                            lambda p: MLPNN(random_state=random_state, **p),
                            X, y, budget, random_state, return_curve)


def tune_ft_continuous(X, y, budget, random_state=0, device="cuda", return_curve=False):
    return _tune_continuous(_sample_ft,
                            lambda p: FTTransformerNN(device=device, random_state=random_state, **p),
                            X, y, budget, random_state, return_curve)


def tune_all_continuous(Z_fit, y_fit, budget, device="cuda", random_state=0):
    """Continuous-space equal-budget random search for every tunable family (drop-in for
    wp3_models.tune_all in the HPO ablation). Returns ({family: best_config}, {family: curve})."""
    best, curves = {}, {}
    best["catboost"], curves["catboost"] = tune_catboost_continuous(Z_fit, y_fit, budget, random_state=random_state, return_curve=True)
    best["ft_transformer"], curves["ft_transformer"] = tune_ft_continuous(Z_fit, y_fit, budget, device=device, random_state=random_state, return_curve=True)
    best["xgboost"], curves["xgboost"] = tune_xgb_continuous(Z_fit, y_fit, budget, random_state=random_state, return_curve=True)
    best["lightgbm"], curves["lightgbm"] = tune_lgb_continuous(Z_fit, y_fit, budget, random_state=random_state, return_curve=True)
    best["mlp"], curves["mlp"] = tune_mlp_continuous(Z_fit, y_fit, budget, random_state=random_state, return_curve=True)
    return best, curves


# --------------------------------------------------------------------------------------------------
# RealMLP (holzmuller2024, pytabkit) -- EXTRA strong-default NN (NOT tuned). Reviewer's "fragile NN"
# target: a modern, meta-tuned-default MLP that should be robust without per-dataset HPO.
# --------------------------------------------------------------------------------------------------


class RealMLPNN:
    """RealMLP tuned-default regressor (pytabkit). Strong default, no per-dataset HPO. Standardized X
    expected (caller scales); RealMLP does its own internal preprocessing + y-handling + train-only
    early-stopping holdout (leakage-safe). variant='td' = full meta-tuned default (holzmuller2024);
    'td_s' = the faster simplified default."""

    def __init__(self, device="cuda", random_state=0, variant="td_s"):
        self.device = device
        self.random_state = random_state
        self.variant = variant

    def fit(self, X, y):
        from pytabkit import RealMLP_TD_Regressor, RealMLP_TD_S_Regressor
        cls = RealMLP_TD_Regressor if self.variant == "td" else RealMLP_TD_S_Regressor
        self.m_ = cls(device=self.device, random_state=self.random_state, n_cv=1, n_refit=0,
                      verbosity=0)
        self.m_.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(np.asarray(X, float)), float).ravel()


def realmlp_factory(variant="td_s"):
    """Returns a (random_state, device, n_features) -> RealMLPNN factory for run_cell extra models."""
    return lambda random_state, device, n_features=None: RealMLPNN(
        device=device, random_state=random_state, variant=variant)
