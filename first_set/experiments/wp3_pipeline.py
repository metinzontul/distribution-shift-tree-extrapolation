"""Real-data pipeline of the first set of experiments (constructed and temporal shift arms).

Per cell = (dataset x arm x seed). B2 makes train FIXED (split-feature lower 30%) and only the test
window slide, so a cell is: ONE leakage-safe fit on the fixed train region, then OOD evaluation on
every severity window. Protocol (M3 leakage order):
  1. split feature = max|Spearman(X,Y)| (extrapolation arm) / lowest train-slope-ratio (saturation
     arm) -- disclosed shift-design;
  2. cutpoints = split-feature percentiles, X-only; test windows forced DISJOINT from train;
  3. ID-test carved from the train region; ONE StandardScaler fit on fit-train ONLY;
  4. equal 8-config HPO (C8) for every tunable family on a fit-train-carved val split (train-only);
  5. sigma_Y^ref = dataset-fixed train-only reference; nOOD = OOD-RMSE / sigma_Y^ref;
  6. write RAW arrays (y_true / y_pred, ID + per-severity OOD) + meta + tuning curves to
     results/wp3_<arm>/<dataset>__seed<k>.npz.

Dual-regime (M3): n<=10k -> regime 'tabpfn' (TabPFN included); else 'large' (GBDT+NN only). No
cross-regime comparison. Env: conda `torch` (GPU). Portable: --device cpu, --out redirect.

A5 temporal arm (PRIMARY natural shift) runs on the 5 real TabReD sets via the authored `default`
temporal split (fit=train period, ID=val period, OOD=future test period); air_quality/bike_hour
row-order are SECONDARY proxies. maps-routing (984 feat) drops TabPFN (feature ceiling); GBDT/NN stay.

  Smoke     : conda run -n torch python experiments/wp3_pipeline.py --smoke
  Projection: conda run -n torch python experiments/wp3_pipeline.py --project-effn
  Full run  : conda run -n torch python experiments/wp3_pipeline.py --arm all --seeds 15
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Reduce CUDA fragmentation for the high-dimensional temporal cells (maps-routing, 984 features).
# Set before torch is imported (models import it lazily inside fit).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from sklearn.preprocessing import StandardScaler

from cwr import RESULTS_DIR
from cwr import wp3_data as D
from cwr import wp3_shift as S
from cwr.wp3_metrics import mde, nood
from cwr.wp3_models import (PRIMARY_PAIR, TABPFN_MAX_FEATURES, TABPFN_REGIME_MAX_N, TUNABLE,
                           build_wp3_models, tune_all)

# Covariate-arm smoke: small (regime 'tabpfn') + one large (regime 'large') so both regimes fire.
SMOKE_COV_DATASETS = [("concrete_compressive_strength", "ctr23", 44959),
                      ("airfoil_self_noise", "ctr23", 44957), ("miami_housing", "ctr23", 44983)]
# Temporal-arm smoke: TabReD PRIMARY (sberbank = smallest/fastest + TabPFN-feasible; weather = 100
# feat, TabPFN-feasible) + air_quality row-order proxy (SECONDARY). maps-routing (984 feat) is
# checked for TabPFN-EXCLUSION via metadata (its 1.3 GB x_num is not trained in the smoke).
SMOKE_TABRED = ["sberbank-housing", "weather"]
SMOKE_PROXY = [("air_quality", "local", None)]


def resolve_arms(arm):
    if arm == "both":
        return ["extrapolation", "saturation"]
    if arm == "all":
        return ["extrapolation", "saturation", "temporal"]
    return [arm]
MIN_FIT_TRAIN = 200          # B2: n_train<200 -> that arm is N/A (reported)


def regime_of(n):
    return "tabpfn" if n <= TABPFN_REGIME_MAX_N else "large"


def regime_temporal(n_fit_capped, n_features):
    """Dual-regime for the A5 temporal arm (M3). Because every model in a temporal cell trains on the
    SAME capped fit-train subsample, TabPFN feasibility is governed by that capped size (<=~10k) AND
    the feature count (<=~500) -- NOT the full dataset N. This is a fair single-regime comparison on
    the subsample (no TabPFN-on-subsample vs GBDT-on-full mismatch); TabPFN is simply infeasible when
    features exceed the ceiling (maps-routing 984 -> 'large')."""
    return ("tabpfn" if (n_fit_capped <= TABPFN_REGIME_MAX_N and n_features <= TABPFN_MAX_FEATURES)
            else "large")


def _cap_ordered(idx, cap, rng):
    """Seeded subsample of a local index array to <=`cap`, sorted ascending so the chronological
    order inside the (time-ordered) block is preserved (design pin: temporal-sira korunarak)."""
    idx = np.asarray(idx, np.int64)
    if cap and len(idx) > cap:
        idx = rng.choice(idx, int(cap), replace=False)
    return np.sort(idx)


# --------------------------------------------------------------------------------------------------
# One leakage-safe cell (fit once on fixed train region; OOD-evaluate every severity window)
# --------------------------------------------------------------------------------------------------


def run_cell(ds, spec, seed, out_dir, device, hpo_budget, id_frac, max_fit_train, tabpfn_max_train,
             pool_thresholds=None, verbose=True, tune_fn=None, extra_model_factories=None):
    # tune_fn: optional (Z_fit, y_fit, budget, device, random_state) -> (tuned, curves); None ->
    #   the frozen equal-budget wp3_models.tune_all. The WP3-revision HPO ablation injects the
    #   continuous-space random-search tuner here (same leakage body, only the sampler changes).
    # extra_model_factories: optional {name: (random_state, device, n_features) -> model} of EXTRA
    #   untuned models (e.g. RealMLP) fit/predicted alongside the family set. Both default to None ->
    #   this call is byte-identical to the frozen covariate-arm behaviour.
    t_cell = time.time()
    xv = ds.X[spec["feature"]].values.astype(float)
    y = np.asarray(ds.y, float)
    X = ds.X.values.astype(float)
    train_mask = spec["windows"][0]["train_mask"]           # FIXED across severity (B2)
    train_idx = np.where(train_mask)[0]

    # --- ID-test carved from the train region; optional train-only cap for cost (leakage-safe) ---
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(train_idx)
    if max_fit_train and len(perm) > max_fit_train:
        perm = perm[:max_fit_train]                          # train-only subsample; test untouched
    n_id = max(int(round(id_frac * len(perm))), 1)
    id_idx, fit_idx = perm[:n_id], perm[n_id:]
    if len(fit_idx) < MIN_FIT_TRAIN:
        return {"skipped": True, "reason": f"n_fit={len(fit_idx)}<{MIN_FIT_TRAIN}",
                "dataset": ds.name, "arm": spec["arm"], "seed": seed}

    # --- leakage guards ---
    assert set(fit_idx.tolist()).isdisjoint(set(id_idx.tolist())), "fit/ID overlap"
    regime = regime_of(ds.n)

    # --- leakage-safe imputation of any PRESERVED NaN with the FIT-TRAIN median only (M3) ---
    # The covariate loader keeps NaN (impute=False) so the marginal never leaks across the split;
    # here we fill with a fit-train-only median (same rows the scaler is fit on), then apply that
    # SAME median to ID/OOD rows. NaN-free X skips this block entirely -> byte-identical to the old
    # global-fill path. An all-NaN fit column -> 0 (matches the TabReD train-only rule).
    if np.isnan(X).any():
        med = np.nanmedian(X[fit_idx], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = X.copy()
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(med, np.where(nan_mask)[1])

    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_id, y_id = X[id_idx], y[id_idx]
    scaler = StandardScaler().fit(X_fit)                     # M3 step 3: fit-train ONLY
    Z_fit, Z_id = scaler.transform(X_fit), scaler.transform(X_id)
    sigma_ref = spec["sigma_ref"]                            # M3 step 4: dataset-fixed train-only

    # --- equal 8-config HPO for every tunable family (C8), train-only val split ---
    tuned, curves = ({}, {})
    if hpo_budget and hpo_budget > 0:
        _tuner = tune_fn or tune_all
        tuned, curves = _tuner(Z_fit, y_fit, hpo_budget, device=device, random_state=seed)

    models = build_wp3_models(regime, device=device, tabpfn_max_train=tabpfn_max_train,
                              random_state=seed, tuned=tuned)
    if extra_model_factories:
        for _nm, _fac in extra_model_factories.items():
            models[_nm] = _fac(seed, device, X.shape[1])

    # --- fit once on the fixed train region; evaluate ID + every severity OOD window ---
    preds = {"y_id_true": y_id}
    per_sev_meta, timings, rows = [], {}, {}
    fitted = {}
    for name, model in models.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Z_fit, y_fit)
        fitted[name] = model
        timings[name] = round(time.time() - t0, 3)
        preds[f"{name}__y_id_pred"] = model.predict(Z_id)

    for w in spec["windows"]:
        test_mask = w["test_mask"] & ~train_mask             # force DISJOINT from train (no leak)
        test_idx = np.where(test_mask)[0]
        sev = w["severity"]
        if len(test_idx) == 0:
            per_sev_meta.append({"severity": sev, "n_test": 0, "note": "empty window"})
            continue
        assert set(test_idx.tolist()).isdisjoint(set(train_idx.tolist())), "OOD/train overlap"
        Z_ood = scaler.transform(X[test_idx])
        y_ood = y[test_idx]
        preds[f"ood_sev{sev}_true"] = y_ood
        nood_row = {}
        for name, model in fitted.items():
            p = model.predict(Z_ood)
            preds[f"{name}__ood_sev{sev}_pred"] = p
            nood_row[name] = nood(y_ood, p, sigma_ref)
        rows[sev] = nood_row
        per_sev_meta.append({"severity": sev, "lower_edge": w["lower_edge"], "n_test": len(test_idx)})

    # --- stratum (provisional default thresholds; final via pool tertiles in projection/analysis) ---
    lo, hi = pool_thresholds if pool_thresholds else (0.3, 0.7)
    stratum = S.stratum_of(spec["slope_ratio"], lo, hi)
    flag = S.extrapolation_risk_flag(spec["slope_ratio"], hi)

    meta = {"dataset": ds.name, "source": spec.get("source"), "arm": spec["arm"], "seed": seed,
            "n": ds.n, "d": ds.d, "regime": regime, "feature": spec["feature"],
            "spearman": spec["spearman"], "slope_ratio": spec["slope_ratio"], "stratum": stratum,
            "extrap_risk_flag": flag, "pool_thresholds": [lo, hi], "sigma_y_ref": sigma_ref,
            "mde": mde(sigma_ref), "n_train_region": int(train_mask.sum()),
            "n_fit": len(fit_idx), "n_id": len(id_idx), "id_frac": id_frac,
            "max_fit_train": max_fit_train, "severities": per_sev_meta,
            "hpo_budget": int(hpo_budget or 0), "tuned_configs": {k: _jsonable(v) for k, v in tuned.items()},
            "tuning_curves": {k: curves[k] for k in curves}, "tunable_families": list(TUNABLE),
            "primary_pair": list(PRIMARY_PAIR), "models": list(models.keys()),
            "device": device, "timings": timings, "cell_seconds": round(time.time() - t_cell, 3),
            "leakage_checks": ("split-feature disclosed; cutpoints X-only; test DISJOINT from train; "
                               "scaler/HPO/fit on fit-train only; sigma_ref train-only dataset-fixed")}

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{ds.name}__{spec['arm']}__seed{seed}"
    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), meta=json.dumps(meta),
                        **{k: np.asarray(v, float) for k, v in preds.items()})
    with open(os.path.join(out_dir, "_timing.jsonl"), "a") as f:
        f.write(json.dumps({"tag": tag, "dataset": ds.name, "arm": spec["arm"], "seed": seed,
                            "regime": regime, "cell_seconds": meta["cell_seconds"],
                            "timings": timings, "hpo_budget": meta["hpo_budget"]}) + "\n")
    if verbose:
        maxsev = max(rows) if rows else None
        rr = "  ".join(f"{n}:{rows[maxsev][n]:.3f}" for n in models) if maxsev is not None else "-"
        print(f"[{tag}] {meta['cell_seconds']:5.1f}s regime={regime} feat={spec['feature'][:16]} "
              f"slopeR={spec['slope_ratio']:.2f} nOOD@sev{maxsev}: {rr}")
    return {"skipped": False, "meta": meta, "nood_rows": rows, "tag": tag}


def _jsonable(v):
    if isinstance(v, dict):
        return {k: (list(x) if isinstance(x, tuple) else x) for k, x in v.items()}
    return v


# --------------------------------------------------------------------------------------------------
# A5 temporal cell (natural shift, PRIMARY evidence): single time-ordered train/test split
# --------------------------------------------------------------------------------------------------


def run_temporal_cell(ds, seed, out_dir, device, hpo_budget, max_fit_train, tabpfn_max_train,
                      order=None, splits=None, sigma_ref_fixed=None, n_features=None, verbose=True):
    """A5 temporal arm (natural ID->OOD shift). Two split sources:
      * splits given (TabReD, PRIMARY): authored `default` blocks -- fit-train = train (earliest
        period), ID-val = val (mid period), OOD = test (future period). sigma_ref_fixed = full
        train-period Y-std.
      * order given (air_quality/bike_hour, SECONDARY proxy): row-order fractional split, ID-val
        carved from the train tail.
    Each period is seeded-subsampled to `max_fit_train` with TEMPORAL ORDER PRESERVED (design pin);
    the SAME fit-train subsample feeds every model so TabPFN inclusion is a fair single-regime call
    (feature-count-gated). Same leakage-safe fit/scaler/HPO discipline as run_cell."""
    t_cell = time.time()
    if splits is not None:
        sp = S.temporal_split_from_indices(splits["train"], splits["val"], splits["test"], ds.n)
        temporal_source = "tabred_default"
    else:
        sp = S.temporal_split(ds, order_values=order)
        temporal_source = "row_order_proxy"
    if sp is None:
        return {"skipped": True, "reason": "no temporal axis", "dataset": ds.name, "arm": "temporal"}
    y = np.asarray(ds.y, float); X = ds.X.values.astype(float)
    train_idx0 = np.where(sp["train_mask"] & ~sp["idval_mask"])[0]    # fit-train (train minus ID-val)
    id_idx0 = np.where(sp["idval_mask"])[0]                           # ID-val (mid/tail period)
    test_idx0 = np.where(sp["test_mask"])[0]                          # OOD = future block
    if len(train_idx0) < MIN_FIT_TRAIN or len(test_idx0) == 0:
        return {"skipped": True, "reason": f"n_fit={len(train_idx0)}<{MIN_FIT_TRAIN} or empty test",
                "dataset": ds.name, "arm": "temporal"}

    # per-period seeded subsample, temporal order preserved (design pin); all periods capped equally
    rng = np.random.default_rng(1000 + seed)
    train_idx = _cap_ordered(train_idx0, max_fit_train, rng)
    id_idx = _cap_ordered(id_idx0, max_fit_train, rng)
    test_idx = _cap_ordered(test_idx0, max_fit_train, rng)
    assert set(train_idx.tolist()).isdisjoint(test_idx.tolist()), "temporal leak: train/test overlap"
    assert set(train_idx.tolist()).isdisjoint(id_idx.tolist()), "temporal leak: train/ID overlap"
    assert max(train_idx.max(), id_idx.max()) < test_idx.min(), "future leakage: train/id after test start"

    nfeat = int(n_features if n_features is not None else X.shape[1])
    if splits is not None:
        regime = regime_temporal(len(train_idx), nfeat)              # capped-fit + feature gated
    else:
        regime = regime_of(ds.n)                                     # proxy: full-N regime

    # --- leakage-safe imputation of any PRESERVED NaN with the TRAIN-PERIOD median only ---
    # TabReD (splits path) already imputes train-only in load_tabred -> no NaN here. Row-order proxies
    # loaded with impute=False keep NaN; fill with the fit-train (earliest-period) median, then apply
    # that SAME median to ID/OOD (future) rows. NaN-free X skips this -> byte-identical to before.
    if np.isnan(X).any():
        med = np.nanmedian(X[train_idx], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = X.copy()
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(med, np.where(nan_mask)[1])

    X_fit, y_fit = X[train_idx], y[train_idx]
    X_id, y_id = X[id_idx], y[id_idx]
    scaler = StandardScaler().fit(X_fit)                             # M3 step 3: fit-train ONLY
    Z_fit, Z_id, Z_ood = scaler.transform(X_fit), scaler.transform(X_id), scaler.transform(X[test_idx])
    # sigma_Y^ref (temporal) = full train-period Y-std, train-only, dataset-fixed (TabReD passes it in)
    sigma_ref = (float(sigma_ref_fixed) if sigma_ref_fixed
                 else (float(np.std(y[train_idx0])) or float(np.std(y))))

    tuned, curves = ({}, {})
    if hpo_budget and hpo_budget > 0:
        tuned, curves = tune_all(Z_fit, y_fit, hpo_budget, device=device, random_state=seed)
    models = build_wp3_models(regime, device=device, tabpfn_max_train=tabpfn_max_train,
                              random_state=seed, tuned=tuned, n_features=nfeat)
    preds = {"y_id_true": y_id, "ood_true": y[test_idx]}
    timings, nood_row = {}, {}
    for name, model in models.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Z_fit, y_fit)
        timings[name] = round(time.time() - t0, 3)
        preds[f"{name}__y_id_pred"] = model.predict(Z_id)
        p = model.predict(Z_ood)
        preds[f"{name}__ood_pred"] = p
        nood_row[name] = nood(y[test_idx], p, sigma_ref)

    meta = {"dataset": ds.name, "source": getattr(ds, "source", None), "arm": "temporal",
            "temporal_source": temporal_source, "seed": seed, "n": ds.n,
            "n_full": int(getattr(ds, "n_full", ds.n)), "d": ds.d, "n_features": nfeat,
            "regime": regime, "tabpfn_feasible": bool(nfeat <= TABPFN_MAX_FEATURES),
            "feature": None, "spearman": None, "slope_ratio": None,
            "stratum": "temporal", "extrap_risk_flag": None, "pool_thresholds": None,
            "sigma_y_ref": sigma_ref, "mde": mde(sigma_ref),
            "severities": [{"severity": "natural", "n_test": len(test_idx)}],
            "n_train_region": int(sp["n_train"]), "n_test_region": int(sp["n_test"]),
            "n_fit": len(train_idx), "n_id": len(id_idx), "n_ood": len(test_idx),
            "id_frac": None, "max_fit_train": max_fit_train, "hpo_budget": int(hpo_budget or 0),
            "tuned_configs": {k: _jsonable(v) for k, v in tuned.items()},
            "tuning_curves": {k: curves[k] for k in curves}, "tunable_families": list(TUNABLE),
            "primary_pair": list(PRIMARY_PAIR), "models": list(models.keys()), "device": device,
            "timings": timings, "cell_seconds": round(time.time() - t_cell, 3),
            "leakage_checks": ("temporal order: fit/ID strictly before test period; scaler/HPO/fit "
                               "train-only; per-period seeded subsample (order preserved); "
                               "sigma_ref = full train-period Y-std (train-only, dataset-fixed)")}
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{ds.name}__temporal__seed{seed}"
    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), meta=json.dumps(meta),
                        **{k: np.asarray(v, float) for k, v in preds.items()})
    with open(os.path.join(out_dir, "_timing.jsonl"), "a") as f:
        f.write(json.dumps({"tag": tag, "dataset": ds.name, "arm": "temporal", "seed": seed,
                            "regime": regime, "cell_seconds": meta["cell_seconds"],
                            "timings": timings, "hpo_budget": meta["hpo_budget"]}) + "\n")
    if verbose:
        rr = "  ".join(f"{n}:{nood_row[n]:.3f}" for n in models)
        print(f"[{tag}] {meta['cell_seconds']:5.1f}s regime={regime} TEMPORAL nOOD: {rr}")
    return {"skipped": False, "meta": meta, "nood_rows": {"natural": nood_row}, "tag": tag}


# --------------------------------------------------------------------------------------------------
# Smoke self-check (finiteness + leakage guards + equal-HPO; does NOT interpret OOD magnitudes)
# --------------------------------------------------------------------------------------------------


def verify_written_cells(out_dir):
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not files:
        print("  [verify] no npz written"); return False
    finite_ok, hpo_ok, leak_ok = True, True, True
    for f in files:
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        # equal-HPO: every tunable family got the SAME #configs == budget (C8)
        if m["hpo_budget"] > 0:
            ncfg = {k: len(m["tuning_curves"][k]) for k in m["tunable_families"]}
            if len(set(ncfg.values())) != 1 or set(ncfg.values()) != {m["hpo_budget"]}:
                hpo_ok = False
        # finiteness of every written prediction array + primary pair present
        for k in z.files:
            if k.endswith("_pred") or k.endswith("_true"):
                if not np.all(np.isfinite(z[k])):
                    finite_ok = False
        for fam in m["primary_pair"]:
            if f"{fam}__y_id_pred" not in z.files:
                leak_ok = False
        # leakage sanity: sigma_ref>0, n_fit>=MIN, severities present
        if not (m["sigma_y_ref"] > 0 and m["n_fit"] >= MIN_FIT_TRAIN):
            leak_ok = False
    print(f"  [verify] cells={len(files)}  pred/true-finite={finite_ok}  "
          f"equal-HPO(all tunable=={json.loads(str(np.load(files[0], allow_pickle=True)['meta']))['hpo_budget']}cfg)={hpo_ok}  "
          f"primary-pair+sigma_ref+n_fit-guards={leak_ok}")
    return finite_ok and hpo_ok and leak_ok


# --------------------------------------------------------------------------------------------------
# Effective-N projection (structural counts; NO training) + threshold calibration
# --------------------------------------------------------------------------------------------------


def project_effective_n(max_rows=15000, seed=42, limit=0, verbose=True):
    """Load every registry dataset (capped), apply inclusion, build both arms, compute slope-ratios
    and TabPFN-feasibility, then report the effective-N that drives the confirmatory-vs-descriptive
    call (M2) and the B3 saturation-gate. No models are trained."""
    reg = D.registry()
    if limit:
        reg = reg[:limit]
    rows, ratios = [], []
    for (name, source, did, n_hint) in reg:
        rec = {"name": name, "source": source, "n_hint": n_hint}
        try:
            ds = D.load_one(name, source, did=did, max_rows=max_rows, seed=seed)
        except Exception as e:
            rec.update({"loaded": False, "error": repr(e)[:120]}); rows.append(rec)
            if verbose: print(f"  {name:28s} [{source}] LOAD-FAIL {repr(e)[:60]}")
            continue
        inc = D.check_inclusion(ds)
        rec.update({"loaded": True, "n": ds.n, "d": ds.d, "n_continuous": inc["n_continuous"],
                    "sigma_y": round(inc["sigma_y"], 4), "included": inc["included"],
                    "reasons": inc["reasons"]})
        if inc["included"]:
            ex = S.build_extrapolation_split(ds.X, ds.y, candidates=inc["continuous_cols"])
            sa = S.build_saturation_split(ds.X, ds.y, candidates=inc["continuous_cols"])
            n_train_region = int(ex["windows"][0]["n_train"]) if ex else 0
            # fit-train after id carve ~ 0.75 * capped train region
            n_fit_est = int(0.75 * min(n_train_region, max_rows))
            rec.update({
                "split_feature": ex["feature"] if ex else None,
                "spearman": round(ex["spearman"], 3) if ex else None,
                "slope_ratio": round(ex["slope_ratio"], 3) if ex and np.isfinite(ex["slope_ratio"]) else None,
                "n_train_region": n_train_region, "n_fit_est": n_fit_est,
                "covariate_ok": bool(ex and n_fit_est >= MIN_FIT_TRAIN),
                "saturation_constructible": bool(sa is not None),
                "saturation_feature": sa["feature"] if sa else None,
                "saturation_slope_ratio": round(sa["slope_ratio"], 3) if sa else None,
                "tabpfn_feasible": bool(ds.n <= TABPFN_REGIME_MAX_N),
                "temporal_axis": D.temporal_order(ds) is not None,
                "regime": regime_of(ds.n)})
            if ex and np.isfinite(ex["slope_ratio"]):
                ratios.append(ex["slope_ratio"])
        rows.append(rec)
        if verbose:
            tag = ("INCL" if inc["included"] else "EXCL " + ",".join(inc["reasons"]))
            extra = ""
            if inc["included"]:
                extra = (f" cov={'Y' if rows[-1]['covariate_ok'] else 'N'} "
                         f"satur={'Y' if rows[-1]['saturation_constructible'] else 'N'} "
                         f"tabpfn={'Y' if rows[-1]['tabpfn_feasible'] else 'N'} "
                         f"temporal={'Y' if rows[-1]['temporal_axis'] else 'N'} "
                         f"slopeR={rows[-1]['slope_ratio']}")
            print(f"  {name:28s} [{source}] n={rec.get('n','?'):>6} {tag}{extra}")

    lo, hi = S.calibrate_thresholds(ratios)
    for r in rows:                                     # assign strata with pool tertiles
        if r.get("included") and r.get("slope_ratio") is not None:
            r["stratum"] = S.stratum_of(r["slope_ratio"], lo, hi)

    inc_rows = [r for r in rows if r.get("included")]
    cov = [r for r in inc_rows if r.get("covariate_ok")]
    satur = [r for r in inc_rows if r.get("saturation_constructible")]
    both = [r for r in inc_rows if r.get("covariate_ok") and r.get("saturation_constructible")]
    tabpfn = [r for r in inc_rows if r.get("tabpfn_feasible")]
    temporal_proxy = [r for r in inc_rows if r.get("temporal_axis")]     # air_quality/bike_hour proxies
    ex_strat = [r for r in cov if r.get("stratum") == "extrapolation"]
    sa_strat = [r for r in cov if r.get("stratum") == "saturation"]

    # A5-PRIMARY temporal count = the 5 real TabReD sets (mmap metadata, no full load). These are the
    # natural-shift primary evidence; the row-order proxies stay SECONDARY. TabPFN feasibility per set
    # is governed by the feature count (maps-routing 984 > 500 -> excluded).
    tabred = D.tabred_status()
    tabred_ok = [s for s in tabred["available"]]
    tabred_tabpfn = [s for s in tabred_ok if tabred["meta"][s].get("tabpfn_feasible")]

    summary = {
        "n_registry": len(rows), "n_loaded": sum(r.get("loaded", False) for r in rows),
        "n_included": len(inc_rows),
        "N_covariate_effective": len(cov),          # M2-i confirmatory effective-N (max-severity paired)
        "N_saturation_constructible": len(satur),   # B3 gate input
        "N_both_arms": len(both),
        "B3_gate_pass": len(satur) >= 12,           # <12 -> differential = "N/A"
        "N_tabpfn_regime1": len(tabpfn), "N_large_regime2": len(inc_rows) - len(tabpfn),
        "N_temporal_primary_tabred": len(tabred_ok),    # A5-PRIMARY (real TabReD, not proxy)
        "N_temporal_tabred_tabpfn": len(tabred_tabpfn),  # TabReD sets where TabPFN is feature-feasible
        "N_temporal_proxy": len(temporal_proxy),         # SECONDARY row-order fallback
        "pool_thresholds": {"lo": round(lo, 3), "hi": round(hi, 3)},
        "N_extrapolation_stratum": len(ex_strat), "N_saturation_stratum": len(sa_strat),
        "confirmatory_verdict": "CONFIRMATORY" if len(cov) >= 20 else "DESCRIPTIVE",
        "tabred": tabred,
        "gretton_deduped_out": D.GRETTON_DEDUPED_OUT,
    }
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------


def build_specs_for(ds, arms, inc):
    specs = []
    if "extrapolation" in arms:
        ex = S.build_extrapolation_split(ds.X, ds.y, candidates=inc["continuous_cols"])
        if ex:
            ex["source"] = None; specs.append(ex)
    if "saturation" in arms:
        sa = S.build_saturation_split(ds.X, ds.y, candidates=inc["continuous_cols"])
        if sa:
            sa["source"] = None; specs.append(sa)
    return specs


def main():
    ap = argparse.ArgumentParser(description="WP3 real-data pipeline (build/smoke/projection).")
    ap.add_argument("--smoke", action="store_true", help="GPU end-to-end + leakage + equal-HPO check")
    ap.add_argument("--project-effn", action="store_true", help="effective-N projection (no training)")
    ap.add_argument("--arm", default="both",
                    choices=["extrapolation", "saturation", "temporal", "both", "all"])
    ap.add_argument("--datasets", default="", help="comma list of names (default: registry/smoke set)")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--hpo-budget", type=int, default=8)
    ap.add_argument("--id-frac", type=float, default=0.25)
    ap.add_argument("--max-rows", type=int, default=20000, help="train-safe row cap per dataset")
    ap.add_argument("--max-fit-train", type=int, default=5000, help="train-only fit cap (cost)")
    ap.add_argument("--tabred-pool", type=int, default=20000,
                    help="TabReD per-period pool cap (memory-safe mmap load; per-seed fit draws "
                         "max-fit-train from this pool). Keeps maps-routing's 1.3GB x_num off RAM.")
    ap.add_argument("--tabpfn-max-train", type=int, default=3000)
    ap.add_argument("--project-cap", type=int, default=15000, help="row cap for the projection load")
    ap.add_argument("--project-limit", type=int, default=0, help="cap #datasets in projection")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(RESULTS_DIR / "wp3"))
    args = ap.parse_args()

    if args.project_effn:
        print("=" * 100 + "\nEFFECTIVE-N PROJECTION (structural; no training)\n" + "=" * 100)
        res = project_effective_n(max_rows=args.project_cap, limit=args.project_limit)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "effective_n_projection.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        s = res["summary"]
        print("-" * 100)
        for k, v in s.items():
            if k not in ("tabred", "gretton_deduped_out"):
                print(f"  {k:32s} {v}")
        print(f"  TabReD available: {s['tabred']['available']}  missing: {s['tabred']['missing']}")
        print(f"  projection -> {os.path.join(args.out, 'effective_n_projection.json')}")
        return

    if args.smoke:
        cov_pairs = SMOKE_COV_DATASETS
        tabred_names = list(SMOKE_TABRED)
        proxy_pairs = list(SMOKE_PROXY)
        arms = ["extrapolation", "saturation", "temporal"]    # smoke exercises all three arms
        seeds = list(range(args.seed0, args.seed0 + 2))
        out_dir = args.out + "_smoke"
        args.max_fit_train = min(args.max_fit_train, 1500)   # keep FT/GPU fast for the smoke
        args.tabred_pool = min(args.tabred_pool, 4000)
    else:
        names = [n.strip() for n in args.datasets.split(",") if n.strip()]
        arms = resolve_arms(args.arm)
        reg = {n: (n, src, did) for (n, src, did, _) in D.registry()}   # covariate backbone
        cov_pairs = ([reg[n] for n in names if n in reg] if names else list(reg.values()))
        all_tabred = D.TABRED_SETS
        tabred_names = [n for n in all_tabred if (not names or n in names)]
        proxy_pairs = [reg[n] for n in ("air_quality", "bike_hour")
                       if n in reg and (not names or n in names)]        # SECONDARY temporal proxies
        seeds = list(range(args.seed0, args.seed0 + args.seeds))
        out_dir = args.out
    do_cov = any(a in ("extrapolation", "saturation") for a in arms)
    do_temporal = "temporal" in arms

    n_cells_planned = ((len(cov_pairs) if do_cov else 0) + (len(tabred_names) + len(proxy_pairs)
                       if do_temporal else 0))
    print(f"WP3 {'SMOKE' if args.smoke else 'RUN'}  arms={arms}  cov-datasets={len(cov_pairs) if do_cov else 0}"
          f"  tabred-temporal={len(tabred_names) if do_temporal else 0}"
          f"  proxy-temporal={len(proxy_pairs) if do_temporal else 0}  seeds={seeds[0]}..{seeds[-1]}"
          f"  hpo={args.hpo_budget}  device={args.device}  out={out_dir}")
    print("=" * 100)
    t_all, per_cell, n_skip = time.time(), [], 0

    def _seed_loop(tag_arm, name, runner):
        nonlocal n_skip
        for s in seeds:
            tag = f"{name}__{tag_arm}__seed{s}"
            if os.path.exists(os.path.join(out_dir, f"{tag}.npz")):
                n_skip += 1; continue
            r = runner(s)
            if r.get("skipped"):
                print(f"  [{tag}] SKIP {r['reason']}")
            else:
                per_cell.append(r["meta"]["cell_seconds"])

    # ---- covariate arms (extrapolation / saturation) on the CTR23+gretton+local backbone ----
    if do_cov:
        for (name, source, did) in cov_pairs:
            try:
                # impute=False: preserve NaN so run_cell imputes with a fit-train-only median
                # (leakage-safe). NaN-free datasets are byte-identical to the old global-fill path.
                ds = D.load_one(name, source, did=did, max_rows=args.max_rows, impute=False)
            except Exception as e:
                print(f"  {name} LOAD-FAIL {repr(e)[:80]}"); continue
            inc = D.check_inclusion(ds)
            if not inc["included"]:
                print(f"  {name} EXCLUDED {inc['reasons']}"); continue
            for spec in build_specs_for(ds, arms, inc):
                _seed_loop(spec["arm"], name, lambda s, ds=ds, spec=spec: run_cell(
                    ds, spec, s, out_dir, args.device, args.hpo_budget, args.id_frac,
                    args.max_fit_train, args.tabpfn_max_train))

    # ---- A5 temporal arm: TabReD PRIMARY (authored default split) + row-order proxies SECONDARY ----
    if do_temporal:
        for name in tabred_names:
            try:
                ds = D.load_tabred(name, pool_cap=args.tabred_pool)   # memory-safe compact load
            except Exception as e:
                print(f"  {name} [tabred] LOAD-FAIL {repr(e)[:90]}"); continue
            print(f"  {name} [tabred] n_full={ds.n_full} pooled={ds.n} feat={ds.n_features} "
                  f"tabpfn={'Y' if ds.n_features <= TABPFN_MAX_FEATURES else 'N(>500feat)'} "
                  f"splits(tr/val/te)={len(ds.splits['train'])}/{len(ds.splits['val'])}/{len(ds.splits['test'])}")
            _seed_loop("temporal", name, lambda s, ds=ds: run_temporal_cell(
                ds, s, out_dir, args.device, args.hpo_budget, args.max_fit_train,
                args.tabpfn_max_train, splits=ds.splits, sigma_ref_fixed=ds.sigma_ref,
                n_features=ds.n_features))
        for (name, source, did) in proxy_pairs:
            try:
                # impute=False: preserve NaN so run_temporal_cell imputes with a train-period-only
                # median (leakage-safe). NaN-free proxies (bike_hour) stay byte-identical.
                ds = D.load_one(name, source, did=did, max_rows=args.max_rows, impute=False)
            except Exception as e:
                print(f"  {name} [proxy] LOAD-FAIL {repr(e)[:80]}"); continue
            order = D.temporal_order(ds)
            if order is None:
                print(f"  {name} temporal-proxy SKIP (no natural temporal axis)"); continue
            _seed_loop("temporal", name, lambda s, ds=ds, order=order: run_temporal_cell(
                ds, s, out_dir, args.device, args.hpo_budget, args.max_fit_train,
                args.tabpfn_max_train, order=order))
    total = time.time() - t_all
    print("=" * 100)
    if per_cell:
        avg = float(np.mean(per_cell))
        print(f"DONE {len(per_cell)} cells ({n_skip} resumed) in {total/60:.1f} min. "
              f"avg {avg:.1f}s/cell (min {np.min(per_cell):.1f}, max {np.max(per_cell):.1f}).")
    print(f"raw arrays -> {out_dir}  (timing: {os.path.join(out_dir, '_timing.jsonl')})")
    if args.smoke:
        print("-" * 100)
        ok = verify_written_cells(out_dir)
        print(f"SMOKE self-check {'PASS' if ok else 'FAIL'} "
              "(finiteness + leakage guards + equal-HPO; OOD magnitudes NOT interpreted -- full run is separate).")


if __name__ == "__main__":
    main()
