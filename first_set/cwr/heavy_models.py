"""Heavy-model wrappers (FT-Transformer, TabPFN v2) for the shift-robustness study.

WP1 provides only the SKELETON + availability guards; the full training/HPO loops are filled in WP2
(design.md §4.1 aile temsilcileri; frozen Lock-in 1: NN=FT-Transformer, FM=TabPFN v2). Each wrapper
exposes a leakage-safe sklearn-style fit(X, y)/predict(X). When a backend library is missing the
wrapper stays importable and reports exactly what to install, so the pipeline can guard-and-skip.

Environment probe (this env): torch present (CPU-only), tabpfn present; NO FT-Transformer library
(rtdl / rtdl_revisiting_models / pytorch_tabular all missing).
"""
from __future__ import annotations

import importlib.util

import numpy as np


def _have(mod):
    return importlib.util.find_spec(mod) is not None


def heavy_status():
    """Report availability + install hint for each heavy family (no side effects)."""
    torch = _have("torch")
    ft_lib = next((m for m in ("rtdl_revisiting_models", "rtdl") if _have(m)), None)
    return {
        "FT-Transformer": {
            "available": bool(torch and ft_lib),
            "backend": ft_lib,
            "needs": None if (torch and ft_lib) else
                     ("pip install rtdl_revisiting_models" if torch else "pip install torch + rtdl_revisiting_models"),
            "note": "WP2: implement gorishniy2021 FT-Transformer via rtdl_revisiting_models "
                    "(or a small custom torch module). torch is present but CPU-only here.",
        },
        "TabPFNv2": {
            "available": _have("tabpfn") and torch,
            "backend": "tabpfn" if _have("tabpfn") else None,
            "needs": None if (_have("tabpfn") and torch) else "pip install tabpfn",
            "note": "TabPFN v2 regression; N<=~10000 / features<=~500 (CPU<=1000 -> GPU). "
                    "First call may download model weights.",
        },
    }


class FTTransformerWrapper:
    """Skeleton NN representative (frozen Lock-in 1). WP2 fills the training loop. Kept importable so
    the pipeline can probe availability and guard-skip when no backend is installed."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._status = heavy_status()["FT-Transformer"]

    @property
    def available(self):
        return self._status["available"]

    def fit(self, X, y):
        raise NotImplementedError(
            "FTTransformerWrapper is a WP1 skeleton. WP2 implements the training loop. "
            f"Missing backend: {self._status['needs']}. {self._status['note']}")

    def predict(self, X):
        raise NotImplementedError("FTTransformerWrapper.predict not implemented until WP2.")


class TabPFNRegressorWrapper:
    """FM representative (frozen Lock-in 1 = TabPFN v2). Real, runnable when tabpfn is installed.
    Enforces the N-limit by (leakage-safe) train-side subsampling only; never touches test."""

    def __init__(self, device="cpu", max_train=1000, random_state=0, **kwargs):
        self.device = device
        self.max_train = max_train
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self._status = heavy_status()["TabPFNv2"]

    @property
    def available(self):
        return self._status["available"]

    def fit(self, X, y):
        if not self.available:
            raise NotImplementedError(f"TabPFN unavailable: {self._status['needs']}")
        from tabpfn import TabPFNRegressor
        X = np.asarray(X, float); y = np.asarray(y, float)
        if len(X) > self.max_train:                       # train-only subsample (no test leakage)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self._model = TabPFNRegressor(device=self.device, random_state=self.random_state,
                                      ignore_pretraining_limits=True, **self.kwargs)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("call fit before predict")
        return np.asarray(self._model.predict(np.asarray(X, float)), float)
