"""WP2 model zoo for the shift-robustness sweep (frozen Lock-in 1 aile temsilcileri).

Family representatives (pre-committed, frozen-protocol Lock-in 1):
  GBDT = CatBoost, NN = FT-Transformer (GPU), FM = TabPFN v2 (GPU).
Plus the design.md §4.1 controls/baselines: constant-mean, ridge, and a diagnostic ORACLE
(Bayes-optimal noise-free signal; Lock-in 10 -> reported separately, never as a "family").

Every wrapper is a leakage-safe sklearn-style fit(X, y)/predict(X):
  * fit sees ONLY fit-train (the caller carves ID-test out of the train side and passes fit-train);
  * NN early-stopping validation is carved from fit-train ONLY (design.md §7);
  * TabPFN enforces its N-limit by train-only subsampling; test is never touched.

Authoritative default configs (design.md §4.4): FT-Transformer = gorishniy2021 defaults
(rtdl_revisiting_models get_default_kwargs); CatBoost = standard grid center; TabPFN = no HPO.
Optional random-search HPO (equal #config budget across the two tunable families) is available via
`hpo_budget`; TabPFN is excluded from HPO by design and uses its recommended settings.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

# --------------------------------------------------------------------------------------------------
# Baselines + oracle
# --------------------------------------------------------------------------------------------------


class ConstantMean:
    """Degenerate-case anchor (design.md §6): its fake 'robustness' must stay visible in the metric."""

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_, dtype=float)


class RidgeLinear:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        self.m_ = Ridge(alpha=self.alpha).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(X), float)


class OracleSignal:
    """Diagnostic upper bound (Lock-in 10): the true noise-free E[Y|X] evaluated in the TEST world.

    Not learned from data; it uses the DGP's own target function. Reported as a separate diagnostic
    ceiling, NEVER inside the family FWER comparison set. The sweep passes precomputed oracle values
    (from cwr.synth world objects), so this wrapper just stores/returns them.
    """

    def __init__(self, y_id_signal, y_ood_signal):
        self.y_id_signal = np.asarray(y_id_signal, float)
        self.y_ood_signal = np.asarray(y_ood_signal, float)

    def fit(self, X, y):
        return self


# --------------------------------------------------------------------------------------------------
# GBDT representative: CatBoost
# --------------------------------------------------------------------------------------------------

_CATBOOST_SPACE = {
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 9.0],
}


class CatBoostGBDT:
    """GBDT family representative (Lock-in 1). CPU by default (fast on small tabular; GPU optional)."""

    def __init__(self, iterations=1000, random_state=0, task_type="CPU", params=None):
        self.iterations = iterations
        self.random_state = random_state
        self.task_type = task_type
        self.params = params or {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0}

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        # allow_writing_files=False: project lives on Google Drive (G:\My Drive); CatBoost's default
        # catboost_info/tmp/*.tmp writes intermittently fail there (Error 433). Disabling file output
        # is result-neutral (only stops training logs). Fixes temporal-arm crash (2026-08-16).
        self.m_ = CatBoostRegressor(iterations=self.iterations, verbose=0, allow_writing_files=False,
                                    random_state=self.random_state, task_type=self.task_type,
                                    **self.params)
        self.m_.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(X), float)


# --------------------------------------------------------------------------------------------------
# NN representative: FT-Transformer (GPU), gorishniy2021 defaults
# --------------------------------------------------------------------------------------------------

_FT_OPT_SPACE = {
    "lr": [3e-5, 1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-6, 1e-5, 1e-4],
    "n_blocks": [2, 3, 4],
}

# Memory budget for FT-Transformer batching: attention activations scale as batch * (n_feat+1)^2,
# so the per-forward batch is capped at _FT_ATTN_BUDGET / (n_feat+1)^2 (never above the configured
# batch_size). Tuned so a ~1000-feature set uses batch ~30 (fits a 16 GB GPU) while low-dimensional
# sets keep the full batch (WP2 behaviour unchanged). See FTTransformerNN.fit.
_FT_ATTN_BUDGET = 3.0e7


class FTTransformerNN:
    """NN family representative (Lock-in 1) = gorishniy2021 FT-Transformer via rtdl_revisiting_models.

    Standardized continuous features are expected (caller applies the per-family scaler). The target
    is standardized with FIT-TRAIN statistics only (leakage-safe) for optimizer stability, then
    de-standardized at predict. Early stopping uses a validation split carved from FIT-TRAIN ONLY.
    """

    def __init__(self, device="cuda", n_blocks=3, lr=1e-4, weight_decay=1e-5, batch_size=256,
                 max_epochs=100, patience=16, min_epochs=20, val_frac=0.15, random_state=0,
                 arch=None):
        self.device = device
        self.n_blocks = n_blocks
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_epochs = min_epochs
        self.val_frac = val_frac
        self.random_state = random_state
        # arch: optional dict of backbone overrides (d_block, attention_dropout, ffn_dropout,
        # residual_dropout, ffn_d_hidden_multiplier) on top of gorishniy2021 defaults. Used ONLY by
        # the WP3-revision continuous-space HPO ablation so FT has a genuinely high-dimensional search
        # space. arch=None (every frozen WP2/WP3 path) -> byte-identical default-config behaviour.
        self.arch = arch

    def fit(self, X, y):
        import torch
        import rtdl_revisiting_models as rtdl
        torch.manual_seed(self.random_state)
        dev = self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        self._dev = dev
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float64)
        self.y_mu_, self.y_sd_ = float(y.mean()), float(y.std() + 1e-8)
        ys = ((y - self.y_mu_) / self.y_sd_).astype(np.float32)

        rng = np.random.default_rng(self.random_state)
        n = len(X)
        perm = rng.permutation(n)
        n_val = max(int(round(self.val_frac * n)), 1)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        dk = rtdl.FTTransformer.get_default_kwargs(self.n_blocks)
        if self.arch:
            # Override gorishniy2021 default backbone keys (continuous-HPO ablation only). d_block must
            # stay divisible by attention_n_heads (8); caller snaps it. Unknown/None entries ignored.
            for k, v in self.arch.items():
                if v is not None and k in dk:
                    dk[k] = v
        net = rtdl.FTTransformer(n_cont_features=X.shape[1], cat_cardinalities=[], d_out=1, **dk).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # FT-Transformer attends over (n_features+1) tokens, so activation memory scales with
        # batch * n_features^2. Cap the batch by a memory budget so a high-dimensional set (e.g.
        # maps-routing, 984 features -> 985^2 attention) does not OOM. For typical feature counts the
        # budget is far above `batch_size`, so the effective batch (and WP2 behaviour) is unchanged.
        self._bs = max(4, min(self.batch_size, int(_FT_ATTN_BUDGET / (X.shape[1] + 1) ** 2)))
        Xtr = torch.tensor(X[tr_idx]).to(dev); ytr = torch.tensor(ys[tr_idx]).view(-1, 1).to(dev)
        Xva = torch.tensor(X[val_idx]).to(dev); yva = torch.tensor(ys[val_idx]).view(-1, 1).to(dev)
        ntr = len(tr_idx)

        best_val, best_state, bad = float("inf"), None, 0
        for ep in range(self.max_epochs):
            net.train()
            order = rng.permutation(ntr)
            for i in range(0, ntr, self._bs):
                b = order[i:i + self._bs]
                opt.zero_grad()
                loss = ((net(Xtr[b], None) - ytr[b]) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vpred = torch.cat([net(Xva[j:j + self._bs], None)
                                   for j in range(0, len(Xva), self._bs)])
                vmse = float(((vpred - yva) ** 2).mean().item())
            if vmse < best_val - 1e-5:
                best_val = vmse
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if ep + 1 >= self.min_epochs and bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        self._epochs_run = ep + 1
        return self

    def predict(self, X):
        import torch
        X = np.asarray(X, np.float32)
        bs = min(getattr(self, "_bs", 4096), 4096)     # same attention-memory cap as training
        with torch.no_grad():
            out = []
            for i in range(0, len(X), bs):
                xb = torch.tensor(X[i:i + bs]).to(self._dev)
                out.append(self._net(xb, None).cpu().numpy().ravel())
        p = np.concatenate(out) if out else np.array([])
        return p * self.y_sd_ + self.y_mu_


# --------------------------------------------------------------------------------------------------
# FM representative: TabPFN v2 (GPU)
# --------------------------------------------------------------------------------------------------


class TabPFNv2:
    """FM family representative (Lock-in 1). GPU (device=cuda). No HPO by design (§4.4).

    Enforces the N-limit by leakage-safe train-only subsampling; never touches test.
    """

    def __init__(self, device="cuda", max_train=3000, random_state=0):
        self.device = device
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        X = np.asarray(X, float); y = np.asarray(y, float)
        if len(X) > self.max_train:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self.m_ = TabPFNRegressor(device=self.device, random_state=self.random_state,
                                  ignore_pretraining_limits=True)
        self.m_.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(np.asarray(X, float)), float)


# --------------------------------------------------------------------------------------------------
# HPO (equal #config random search on fit-train ID-validation; leakage-safe)
# --------------------------------------------------------------------------------------------------


def _val_rmse(model, Xtr, ytr, Xva, yva):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr)
        p = model.predict(Xva)
    return float(np.sqrt(np.mean((p - yva) ** 2)))


def tune_catboost(X, y, budget, random_state=0, return_curve=False):
    """Random search over _CATBOOST_SPACE; returns best params dict (or (best, curve) if
    return_curve). IDENTICAL protocol to tune_ft (C8): same budget, same 20% fit-train val split
    drawn with the same seed convention, same random-config draw -> family-fair HPO. The `curve` is
    the per-config trial log [{config, val_rmse}...] so the tuning effort can be audited/plotted."""
    rng = np.random.default_rng(random_state)
    n = len(X); perm = rng.permutation(n); nv = max(int(0.2 * n), 1)
    va, tr = perm[:nv], perm[nv:]
    best, best_p, curve = float("inf"), None, []
    for _ in range(budget):
        p = {k: rng.choice(v).item() for k, v in _CATBOOST_SPACE.items()}
        r = _val_rmse(CatBoostGBDT(random_state=random_state, params=p), X[tr], y[tr], X[va], y[va])
        curve.append({"config": p, "val_rmse": r})
        if r < best:
            best, best_p = r, p
    return (best_p, curve) if return_curve else best_p


def tune_ft(X, y, budget, device="cuda", random_state=0, return_curve=False):
    """Random search over _FT_OPT_SPACE; returns best kwargs (or (best, curve) if return_curve).
    IDENTICAL protocol to tune_catboost (C8): same budget, same 20% fit-train val split, same
    random-config draw. The `curve` is the per-config trial log [{config, val_rmse}...]."""
    rng = np.random.default_rng(random_state)
    n = len(X); perm = rng.permutation(n); nv = max(int(0.2 * n), 1)
    va, tr = perm[:nv], perm[nv:]
    best, best_k, curve = float("inf"), None, []
    for _ in range(budget):
        k = {kk: (rng.choice(v).item()) for kk, v in _FT_OPT_SPACE.items()}
        m = FTTransformerNN(device=device, random_state=random_state, **k)
        r = _val_rmse(m, X[tr], y[tr], X[va], y[va])
        curve.append({"config": k, "val_rmse": r})
        if r < best:
            best, best_k = r, k
    return (best_k, curve) if return_curve else best_k


# --------------------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------------------

FAMILY_MODELS = ("catboost", "ft_transformer", "tabpfn")
BASELINE_MODELS = ("constant_mean", "ridge")


def build_models(device="cuda", tabpfn_max_train=3000, random_state=0):
    """Default-config family + baseline models (no HPO). Oracle is injected by the sweep separately."""
    return {
        "constant_mean": ConstantMean(),
        "ridge": RidgeLinear(alpha=1.0),
        "catboost": CatBoostGBDT(random_state=random_state),
        "ft_transformer": FTTransformerNN(device=device, random_state=random_state),
        "tabpfn": TabPFNv2(device=device, max_train=tabpfn_max_train, random_state=random_state),
    }
