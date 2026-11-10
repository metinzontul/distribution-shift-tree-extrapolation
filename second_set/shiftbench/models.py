"""Model families, search grids (manuscript Table 1) and the shared random search.

Every tunable family draws `n_trials` configurations from its grid, fits each on
80 % of the training rows, scores RMSE on the remaining 20 % (in-distribution
validation) and keeps the best fitted model. Neural nets carve a further 15 %
early-stopping split from their 80 %. TabPFN / constant / ridge are untuned.

Added for the revision:
  lgbm_linear      LightGBM with linear leaves (piecewise-linear GBDT)
  ft_transformer_x FT-Transformer with an extended grid (width, dropout, batch)
  realmlp          RealMLP-TD via `pytabkit` (optional)
  tabpfn           any installed `tabpfn` version (v2 / 2.5); row cap is a
                   single global policy applied identically in every arm
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------- grids
GRIDS: dict[str, dict] = {
    "catboost": {"depth": [4, 6, 8, 10], "learning_rate": [0.03, 0.05, 0.1, 0.2],
                 "l2_leaf_reg": [1, 3, 5, 9]},
    "xgboost": {"max_depth": [3, 5, 7, 9], "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "reg_lambda": [1, 3, 5, 9]},
    "lightgbm": {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.03, 0.05, 0.1, 0.2],
                 "reg_lambda": [1, 3, 5, 9]},
    "lgbm_linear": {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.03, 0.05, 0.1, 0.2],
                    "reg_lambda": [1, 3, 5, 9], "linear_lambda": [0.0, 0.01, 0.1, 1.0]},
    "ft_transformer": {"lr": [3e-5, 1e-4, 3e-4, 1e-3], "weight_decay": [1e-6, 1e-5, 1e-4],
                       "n_blocks": [2, 3, 4]},
    "ft_transformer_x": {"lr": [3e-5, 1e-4, 3e-4, 1e-3], "weight_decay": [1e-6, 1e-5, 1e-4],
                         "n_blocks": [2, 3, 4], "d_block": [96, 192, 256],
                         "dropout": [0.0, 0.1, 0.2], "batch_size": [128, 256, 512]},
    "mlp": {"hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128)],
            "alpha": [1e-5, 1e-4, 1e-3], "learning_rate_init": [3e-4, 1e-3]},
}
UNTUNED = {"ridge", "constant", "tabpfn", "realmlp"}
PRIMARY = ["catboost", "ft_transformer", "tabpfn"]
DEFAULT_MODELS = ["catboost", "xgboost", "lightgbm", "lgbm_linear", "ft_transformer",
                  "mlp", "ridge", "constant"]


@dataclass
class RunConfig:
    n_trials: int = 8
    val_frac: float = 0.20
    n_threads: int = 4
    device: str = "auto"              # FT-Transformer: "auto" | "cpu" | "cuda"
    ft_max_epochs: int = 100
    ft_min_epochs: int = 20
    ft_patience: int = 16
    ft_batch_size: int = 256
    mlp_max_iter: int = 300
    mlp_standardize_target: bool = True
    tabpfn_max_rows: int = 10_000     # training-row cap for TabPFN
    tabpfn_policy: str = "skip"       # "skip" (paper covariate arm) | "subsample"
    gbdt_trees: dict = None

    def trees(self, name):
        d = self.gbdt_trees or {"catboost": 1000, "xgboost": 600, "lightgbm": 600, "lgbm_linear": 600}
        return d[name]


# --------------------------------------------------------------- factories
def _catboost(p, seed, cfg):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=cfg.trees("catboost"), random_seed=seed, verbose=0,
                             thread_count=cfg.n_threads, allow_writing_files=False, **p)


def _xgboost(p, seed, cfg):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=cfg.trees("xgboost"), random_state=seed, tree_method="hist",
                        n_jobs=cfg.n_threads, verbosity=0, **p)


def _lightgbm(p, seed, cfg, linear=False):
    from lightgbm import LGBMRegressor
    name = "lgbm_linear" if linear else "lightgbm"
    return LGBMRegressor(n_estimators=cfg.trees(name), random_state=seed, n_jobs=cfg.n_threads,
                         verbose=-1, linear_tree=linear, **p)


def _mlp(p, seed, cfg):
    m = MLPRegressor(max_iter=cfg.mlp_max_iter, early_stopping=True, validation_fraction=0.15,
                     n_iter_no_change=16, random_state=seed, **p)
    return TransformedTargetRegressor(m, transformer=StandardScaler()) if cfg.mlp_standardize_target else m


def _ridge(p, seed, cfg):
    return Ridge(alpha=1.0)


def _constant(p, seed, cfg):
    return DummyRegressor(strategy="mean")


def _ft(p, seed, cfg):
    return FTTransformerRegressor(seed=seed, device=cfg.device, max_epochs=cfg.ft_max_epochs,
                                  min_epochs=cfg.ft_min_epochs, patience=cfg.ft_patience,
                                  batch_size=p.pop("batch_size", cfg.ft_batch_size), **p)


def _tabpfn(p, seed, cfg):
    return TabPFNWrapper(seed=seed, max_rows=cfg.tabpfn_max_rows, policy=cfg.tabpfn_policy)


def _realmlp(p, seed, cfg):
    from pytabkit import RealMLP_TD_Regressor
    return RealMLP_TD_Regressor(random_state=seed, n_threads=cfg.n_threads, verbosity=0)


FACTORIES: dict[str, Callable] = {
    "catboost": _catboost, "xgboost": _xgboost, "lightgbm": _lightgbm,
    "lgbm_linear": lambda p, s, c: _lightgbm(p, s, c, linear=True),
    "ft_transformer": _ft, "ft_transformer_x": _ft, "mlp": _mlp, "ridge": _ridge,
    "constant": _constant, "tabpfn": _tabpfn, "realmlp": _realmlp,
}


# ------------------------------------------------------------ FT-Transformer
class FTTransformerRegressor:
    """FT-Transformer (Gorishniy et al., 2021) via `rtdl_revisiting_models`.
    AdamW, constant LR, standardised target, early stopping on a 15 % split."""

    def __init__(self, lr=1e-4, weight_decay=1e-5, n_blocks=3, d_block=192, dropout=None,
                 batch_size=256, max_epochs=100, min_epochs=20, patience=16, seed=0,
                 device="auto", es_frac=0.15):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _device(self):
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def fit(self, X, y):
        import torch
        from rtdl_revisiting_models import FTTransformer

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = self._device()
        X = np.array(X, dtype=np.float32, copy=True)
        y = np.array(y, dtype=np.float32, copy=True)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        ys = (y - self.y_mean_) / self.y_std_
        perm = rng.permutation(len(y))
        n_es = max(1, int(round(self.es_frac * len(y))))
        es, tr = perm[:n_es], perm[n_es:]

        kw = FTTransformer.get_default_kwargs()
        kw.pop("_is_default", None)
        kw.update(n_blocks=self.n_blocks, d_block=self.d_block)
        if self.dropout is not None:
            kw.update(attention_dropout=self.dropout, ffn_dropout=self.dropout)
        model = FTTransformer(n_cont_features=X.shape[1], cat_cardinalities=[], d_out=1, **kw).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(ys).to(dev)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.max_epochs):
            model.train()
            for b in np.array_split(rng.permutation(tr), max(1, int(np.ceil(len(tr) / self.batch_size)))):
                bi = torch.from_numpy(b).to(dev)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(Xt[bi], None).squeeze(-1), yt[bi])
                loss.backward()
                opt.step()
            val = float(np.mean((self._predict_std(model, Xt[torch.from_numpy(es).to(dev)]) - ys[es]) ** 2))
            if val < best - 1e-7:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if epoch + 1 >= self.min_epochs and bad >= self.patience:
                break
        model.load_state_dict(best_state)
        self.model_, self.dev_, self.epochs_ = model, dev, epoch + 1
        return self

    def _predict_std(self, model, Xt):
        import torch
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 4096):
                out.append(model(Xt[i:i + 4096], None).squeeze(-1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        import torch
        Xt = torch.from_numpy(np.array(X, dtype=np.float32, copy=True)).to(self.dev_)
        return self._predict_std(self.model_, Xt) * self.y_std_ + self.y_mean_


# ------------------------------------------------------------------- TabPFN
class SkipModel(Exception):
    """Raised when a model is not applicable to a split (e.g. TabPFN row cap)."""


class TabPFNWrapper:
    def __init__(self, seed=0, max_rows=10_000, policy="skip"):
        self.seed, self.max_rows, self.policy = seed, max_rows, policy

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        if len(y) > self.max_rows:
            if self.policy == "skip":
                raise SkipModel(f"{len(y)} training rows > cap {self.max_rows}")
            idx = np.random.default_rng(self.seed).choice(len(y), self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        self.m_ = TabPFNRegressor(random_state=self.seed, ignore_pretraining_limits=True)
        self.m_.fit(X, y)
        return self

    def predict(self, X):
        out = [self.m_.predict(X[i:i + 5000]) for i in range(0, len(X), 5000)]
        return np.concatenate(out)


# ------------------------------------------------------------- random search
def sample_configs(name: str, n: int, seed: int) -> list[dict]:
    grid = GRIDS[name]
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(combos), size=min(n, len(combos)), replace=False)
    return [dict(zip(keys, combos[i])) for i in pick]


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fit_with_search(name: str, X, y, seed: int, cfg: RunConfig):
    """Return (fitted model, chosen params, info dict)."""
    t0 = time.time()
    if name in UNTUNED or name not in GRIDS:
        m = FACTORIES[name]({}, seed, cfg).fit(X, y)
        return m, {}, {"fit_seconds": time.time() - t0, "val_rmse": np.nan}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = max(1, int(round(cfg.val_frac * len(y))))
    va, tr = perm[:n_val], perm[n_val:]
    best = (np.inf, None, None)
    for p in sample_configs(name, cfg.n_trials, seed):
        m = FACTORIES[name](dict(p), seed, cfg).fit(X[tr], y[tr])
        s = rmse(m.predict(X[va]), y[va])
        if s < best[0]:
            best = (s, m, p)
    return best[1], best[2], {"fit_seconds": time.time() - t0, "val_rmse": best[0]}
