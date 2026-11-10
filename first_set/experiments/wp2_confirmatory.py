"""WP2 CONFIRMATORY tuned synthetic sweep (single-shot binding; 2026-07-30 design protocol amendment).

Blueprint: methodology `wp2-recenter-approval.md` (P1/P2/S1, C1-C9) +
frozen-protocol-shift-robustness.md AMENDMENT (Lock-in 1/3/7). This is the ONE binding tuned run; a
third revision is forbidden (C4). Hypotheses/metric/thresholds are FROZEN; results do NOT change them.

Focused cells (NOT the exploratory full factorial):
  * P1 [Kanit C, metric = nOOD line-free]:
        covariate x target-regime{linear, plateau} x geometry{rot0, rot1} x severity{medium-high}.
        plateau is the pre-specified reverse-control (extrapolation neutralised => GBDT <= NN+MDE).
  * P2 [Kanit D, synthetic calibration]:
        target(=label/prior-probability) shift x geometry{rot0, rot1} x severity sweep. The frozen
        DGP's `target` shift already IS label shift (tilts P(regime)->P(Y), keeps E[Y|X] & P(X|regime)
        fixed), so there is no separate label mechanism to add without unfreezing synth.py; unit count
        (C6) is raised via the finer severity sweep + both rotations. Predictor inputs recorded per
        cell: obliqueness (train-only, Lock-in 6) + target-shift magnitude (severity).

Protocol (leakage-free, design.md §4/§7; identical to the frozen exploratory runner except cells):
  1. make_shifted_regime_data(return_worlds) -> matched (train, OOD-test) on the structural frame;
  2. carve an ID-test out of the TRAIN side (same distribution, leakage-free);
  3. ONE StandardScaler fit on fit-train ONLY (per realization; test never seen);
  4. TUNED HPO = 8 config/model, IDENTICAL protocol for GBDT (CatBoost) and NN (FT-Transformer)
     (C8); the per-config tuning curve is saved. TabPFN has no HPO by design (declared). Then train
     the Lock-in 1 representatives + baselines + diagnostic ORACLE;
  5. write RAW arrays (y_true / y_pred per model, ID + OOD) + cell meta + obliqueness + tuning curves
     to results/wp2_confirmatory/<condition>__seed<k>.npz (SEPARATE from the default sweep dir);
     append a timing row to _timing.jsonl.

FRESH seeds: default 100-129 (30 seeds), OUT of the default 0-29 set (C3 -> genuine out-of-sample).

Env: conda env `torch` (GPU, RTX 4070 Ti SUPER). Run via `conda run -n torch python ...`.
  Smoke : python experiments/wp2_confirmatory.py --smoke
  Full  : python experiments/wp2_confirmatory.py --phase both --seeds 30 --seed0 100 --hpo-budget 8
Portable: --device cpu falls back cleanly; --out redirects results (lab-machine offload).
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
from sklearn.preprocessing import StandardScaler

from cwr import RESULTS_DIR
from cwr.synth import ShiftSpec, make_shifted_regime_data
from cwr.wp2_metrics import (ANCHOR_MODELS, norm_rmse, obliqueness_score, ood_id_ratio, raw_gap,
                             robust_effective_robustness)
from cwr.wp2_models import (CatBoostGBDT, ConstantMean, FTTransformerNN, RidgeLinear, TabPFNv2,
                            tune_catboost, tune_ft)

# --------------------------------------------------------------------------------------------------
# Focused confirmatory condition builders
# --------------------------------------------------------------------------------------------------

FRESH_SEED0 = 100                    # C3: fresh seed block 100..129 (default sweep used 0..29)
P1_SEVERITIES = (0.5, 0.75, 1.0)     # medium-high (Kanit C lives at wider extrapolation gaps)
P2_SEVERITIES = (0.2, 0.4, 0.6, 0.8, 1.0)  # finer sweep -> more (cell x family) units for P2 (C6)
ROTATIONS = (0.0, 1.0)               # rot0 (axis-aligned) vs rot1 (oblique); pooled is analysis-level
DEFAULT_SEEDS = list(range(0, 30))   # provenance guard reference (must NOT overlap the fresh block)


def _cond(shift, rotation, severity, extrap, phase):
    return {"shift": shift, "rotation": rotation, "severity": severity, "extrap": extrap,
            "phase": phase}


def p1_conditions(severities=P1_SEVERITIES, rotations=ROTATIONS):
    """P1 [Kanit C]: covariate x {linear, plateau} x geometry x severity{medium-high}."""
    conds = []
    for rot in rotations:
        for sev in severities:
            for ex in ("linear", "plateau"):
                conds.append(_cond("covariate", rot, sev, ex, "P1"))
    return conds


def p2_conditions(severities=P2_SEVERITIES, rotations=ROTATIONS):
    """P2 [Kanit D]: target(=label) shift x geometry x severity sweep. extrap is inert (no OOD
    region under target shift), so linear only."""
    conds = []
    for rot in rotations:
        for sev in severities:
            conds.append(_cond("target", rot, sev, "linear", "P2"))
    return conds


def all_conditions(phase):
    if phase == "p1":
        return p1_conditions()
    if phase == "p2":
        return p2_conditions()
    return p1_conditions() + p2_conditions()


def smoke_conditions():
    """A tiny leakage/pipeline/equal-HPO exercise: one P1 direct + one P1 reverse-control + two P2."""
    return [
        _cond("covariate", 0.0, 1.0, "linear", "P1"),    # Kanit C direct (rot0)
        _cond("covariate", 1.0, 1.0, "plateau", "P1"),   # reverse-control (rot1)
        _cond("target", 0.0, 1.0, "linear", "P2"),        # target shift rot0
        _cond("target", 1.0, 0.4, "linear", "P2"),        # target shift rot1, lower severity
    ]


def cond_tag(c):
    return (f"{c['phase']}_{c['shift']}_rot{c['rotation']}_sev{c['severity']}_{c['extrap']}"
            .replace(".", "p"))


# --------------------------------------------------------------------------------------------------
# One leakage-free confirmatory cell (tuned; identical HPO protocol GBDT/NN; tuning curve saved)
# --------------------------------------------------------------------------------------------------


def run_cell(cond, seed, n_train, n_test, d, k, id_frac, device, hpo_budget, out_dir,
             tabpfn_max_train, verbose=True):
    t_cell = time.time()
    spec = ShiftSpec(kind=cond["shift"], severity=cond["severity"],
                     covariate_mode="region", extrap=cond["extrap"])
    data = make_shifted_regime_data(spec, n_train=n_train, n_test=n_test, d=d, k=k,
                                    rotation=cond["rotation"], seed=seed, return_worlds=True)
    X_tr_all, y_tr_all = data["X_train"], data["y_train"]
    X_ood, y_ood = data["X_test"], data["y_test"]
    reg_tr, reg_ood = data["regime_train"], data["regime_test"]
    w_tr, w_te = data["world_train"], data["world_test"]
    tgt = data["_target_signal"]

    # --- ID-test carved from the TRAIN side (same distribution, leakage-free) ---
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(X_tr_all))
    n_id = max(int(round(id_frac * len(perm))), 1)
    id_idx, fit_idx = perm[:n_id], perm[n_id:]
    X_fit, y_fit = X_tr_all[fit_idx], y_tr_all[fit_idx]
    X_id, y_id = X_tr_all[id_idx], y_tr_all[id_idx]

    # --- leakage guards ---
    assert set(fit_idx.tolist()).isdisjoint(set(id_idx.tolist())), "fit/ID index overlap"
    assert not np.shares_memory(X_ood, X_tr_all), "OOD must be a physically separate array"
    assert seed not in DEFAULT_SEEDS, f"seed {seed} is in the default block 0-29 (C3 requires fresh seeds)"

    # --- oracle: true noise-free signal (diagnostic upper bound, Lock-in 10) ---
    R = w_tr.R
    S_tr_all = X_tr_all @ R.T
    sig_tr_all = tgt(S_tr_all, w_tr, reg_tr)
    S_ood = X_ood @ R.T
    sig_ood = tgt(S_ood, w_te, reg_ood)
    oracle_id = sig_tr_all[id_idx]
    oracle_ood = sig_ood

    # --- per-realization scaler, fit on fit-train ONLY (design.md §7) ---
    scaler = StandardScaler().fit(X_fit)
    Z_fit, Z_id, Z_ood = scaler.transform(X_fit), scaler.transform(X_id), scaler.transform(X_ood)
    y_fit_std = float(np.std(y_fit))

    # --- TUNED HPO: 8 config/model, IDENTICAL protocol for GBDT and NN (C8); TabPFN excluded ---
    cat_params, ft_kwargs = None, {}
    cat_curve, ft_curve = [], []
    if hpo_budget and hpo_budget > 0:
        cat_params, cat_curve = tune_catboost(Z_fit, y_fit, hpo_budget, random_state=seed,
                                              return_curve=True)
        ft_kwargs, ft_curve = tune_ft(Z_fit, y_fit, hpo_budget, device=device, random_state=seed,
                                      return_curve=True)

    models = {
        "constant_mean": ConstantMean(),
        "ridge": RidgeLinear(alpha=1.0),
        "catboost": CatBoostGBDT(random_state=seed, params=cat_params),
        "ft_transformer": FTTransformerNN(device=device, random_state=seed, **ft_kwargs),
        "tabpfn": TabPFNv2(device=device, max_train=tabpfn_max_train, random_state=seed),
    }

    preds = {"y_id_true": y_id, "y_ood_true": y_ood,
             "oracle__y_id_pred": oracle_id, "oracle__y_ood_pred": oracle_ood}
    timings, rows = {}, []
    for name, model in models.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Z_fit, y_fit)
            p_id, p_ood = model.predict(Z_id), model.predict(Z_ood)
        timings[name] = round(time.time() - t0, 3)
        preds[f"{name}__y_id_pred"] = p_id
        preds[f"{name}__y_ood_pred"] = p_ood
        rows.append((name, norm_rmse(y_id, p_id, y_fit_std), norm_rmse(y_ood, p_ood, y_fit_std)))
    rows.append(("oracle", norm_rmse(y_id, oracle_id, y_fit_std),
                 norm_rmse(y_ood, oracle_ood, y_fit_std)))

    obliq = obliqueness_score(X_fit, y_fit)

    meta = {**cond, "seed": seed, "n_train": n_train, "n_test": n_test, "d": d, "k": k,
            "id_frac": id_frac, "n_fit": len(fit_idx), "n_id": len(id_idx), "n_ood": len(X_ood),
            "y_fit_std": y_fit_std, "obliqueness": float(obliq),
            "hpo_budget": int(hpo_budget or 0), "tuned": bool(hpo_budget and hpo_budget > 0),
            "device": device, "cat_params": cat_params, "ft_kwargs": ft_kwargs,
            "cat_tuning_curve": cat_curve, "ft_tuning_curve": ft_curve,
            "anchor_models": list(ANCHOR_MODELS),
            "shift_severity_effective": data["meta"].get("severity"),
            "noise_scale": data["meta"].get("noise_scale"),
            "timings": timings, "cell_seconds": round(time.time() - t_cell, 3),
            "seed_block": "confirmatory-fresh-100s",
            "leakage_checks": ("scaler+models fit on fit-train only; ID/fit disjoint; OOD separate "
                               "array; seed disjoint from default 0-29")}

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cond_tag(cond)}__seed{seed}"
    npz_path = os.path.join(out_dir, f"{tag}.npz")
    np.savez_compressed(npz_path, meta=json.dumps(meta),
                        **{kk: np.asarray(vv, float) for kk, vv in preds.items()})

    with open(os.path.join(out_dir, "_timing.jsonl"), "a") as f:
        f.write(json.dumps({"tag": tag, **{k: meta[k] for k in ("phase", "shift", "rotation",
                "severity", "extrap", "seed", "cell_seconds", "obliqueness", "hpo_budget")},
                "timings": timings}) + "\n")

    if verbose:
        rr = "  ".join(f"{n}:{oo:.3f}" for n, _, oo in rows)
        print(f"[{tag}] {meta['cell_seconds']:5.1f}s  nOOD  {rr}")
    return meta, rows, npz_path


# --------------------------------------------------------------------------------------------------
# Metric / leakage self-check on the written cells (does NOT interpret OOD numbers; full run is separate)
# --------------------------------------------------------------------------------------------------


def verify_written_cells(out_dir):
    """Load the cells just written, recompute the AMENDMENT metrics (nOOD, gap, ratio, robust
    Theil-Sen effrob over the pre-committed anchors) and re-assert leakage guards. Reports PASS/FAIL
    on FINITENESS + guard integrity only; it deliberately does NOT interpret the OOD magnitudes."""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not files:
        print("  [verify] no npz written"); return False
    by_cond = {}
    ok_all = True
    for f in files:
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        yfs = m["y_fit_std"]; y_id, y_ood = z["y_id_true"], z["y_ood_true"]
        # leakage-guard integrity: OOD size matches meta, arrays present & finite
        rec = {"models": {}, "meta": m}
        for name in list(ANCHOR_MODELS) + ["oracle"]:
            nid = norm_rmse(y_id, z[f"{name}__y_id_pred"], yfs)
            nood = norm_rmse(y_ood, z[f"{name}__y_ood_pred"], yfs)
            if not (np.isfinite(nid) and np.isfinite(nood)):
                ok_all = False
            rec["models"][name] = (nid, nood)
        by_cond.setdefault(m["phase"] + "|" + cond_tag(m), []).append(rec)

    # equal-HPO integrity: catboost & ft got the SAME number of tuning configs (C8)
    hpo_ok = True
    for f in files:
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        if m.get("tuned"):
            nc, nf = len(m["cat_tuning_curve"]), len(m["ft_tuning_curve"])
            if nc != nf or nc != m["hpo_budget"]:
                hpo_ok = False
    # robust effrob over anchors per condition (finiteness only)
    effrob_finite = True
    for cond, recs in by_cond.items():
        aid, aood = [], []
        for r in recs:
            for name in ANCHOR_MODELS:
                nid, nood = r["models"][name]
                aid.append(nid); aood.append(nood)
        if len(aid) >= 3:
            er, (a, b, okfit) = robust_effective_robustness(aid, aood, aid, aood, method="theilsen")
            if okfit and not np.all(np.isfinite(er)):
                effrob_finite = False
        # line-free metrics finiteness
        for r in recs:
            for name in ANCHOR_MODELS:
                nid, nood = r["models"][name]
                if not (np.isfinite(raw_gap(nid, nood)) and np.isfinite(ood_id_ratio(nid, nood))):
                    effrob_finite = False

    print(f"  [verify] cells={len(files)}  metrics-finite={ok_all}  "
          f"equal-HPO(GBDT==NN configs)={hpo_ok}  robust-effrob+gap+ratio-finite={effrob_finite}  "
          f"anchor-pts/cond>=150-at-30seed(here {len(next(iter(by_cond.values())))*len(ANCHOR_MODELS)}/cond)")
    return ok_all and hpo_ok and effrob_finite


# --------------------------------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="WP2 CONFIRMATORY tuned sweep (single-shot binding).")
    ap.add_argument("--phase", default="both", choices=["p1", "p2", "both"])
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline/leakage/equal-HPO exercise")
    ap.add_argument("--seeds", type=int, default=30, help="number of seeds per condition")
    ap.add_argument("--seed0", type=int, default=FRESH_SEED0, help="first seed (fresh block, C3)")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--id-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hpo-budget", type=int, default=8, help="tuned: 8 config/model (C8), GBDT==NN")
    ap.add_argument("--tabpfn-max-train", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="cap number of cells; 0 = no cap")
    ap.add_argument("--out", default=str(RESULTS_DIR / "wp2_confirmatory"))
    args = ap.parse_args()

    if args.smoke:
        conds = smoke_conditions()
        seeds = list(range(args.seed0, args.seed0 + 2))     # 2 fresh seeds for smoke
        out_dir = args.out + "_smoke"
    else:
        conds = all_conditions(args.phase)
        seeds = list(range(args.seed0, args.seed0 + args.seeds))
        out_dir = args.out
    cells = [(c, s) for c in conds for s in seeds]
    if args.limit:
        cells = cells[:args.limit]

    # provenance guard: refuse to write over the default block (C3)
    overlap = sorted(set(seeds) & set(DEFAULT_SEEDS))
    assert not overlap, f"C3 violation: seeds {overlap} overlap the default 0-29 block"

    print(f"CONFIRMATORY  smoke={args.smoke}  phase={args.phase}  conditions={len(conds)}  "
          f"seeds={seeds[0]}..{seeds[-1]} (n={len(seeds)})  cells={len(cells)}  device={args.device}  "
          f"hpo_budget={args.hpo_budget}")
    print(f"per-cell: n_train={args.n_train} n_test={args.n_test} d={args.d} k={args.k} "
          f"id_frac={args.id_frac}  out={out_dir}")
    print("=" * 100)

    t_all = time.time()
    per_cell = []
    n_skipped = 0
    for i, (c, s) in enumerate(cells):
        # resume: deterministic seeds -> an already-written cell is identical; skip it (C3/binding safe)
        existing = os.path.join(out_dir, f"{cond_tag(c)}__seed{s}.npz")
        if os.path.exists(existing):
            n_skipped += 1
            continue
        meta, _, _ = run_cell(c, s, args.n_train, args.n_test, args.d, args.k, args.id_frac,
                              args.device, args.hpo_budget, out_dir, args.tabpfn_max_train)
        per_cell.append(meta["cell_seconds"])
        if len(per_cell) % 10 == 0:
            done = i + 1
            avg = float(np.mean(per_cell))
            eta = avg * (len(cells) - done) / 60.0
            print(f"  ... {done}/{len(cells)} cells ({n_skipped} resumed), avg {avg:.1f}s/cell, ETA {eta:.1f} min")

    total = time.time() - t_all
    avg = float(np.mean(per_cell)) if per_cell else float("nan")
    print("=" * 100)
    if per_cell:
        print(f"DONE {len(cells)} cells ({n_skipped} resumed/skipped, {len(per_cell)} run) in {total/60:.1f} min. "
              f"avg {avg:.2f}s/cell (min {np.min(per_cell):.1f}, max {np.max(per_cell):.1f}).")
    else:
        print(f"DONE all {len(cells)} cells already present ({n_skipped} skipped, nothing to run).")
    print(f"raw arrays -> {out_dir}  (timing log: {os.path.join(out_dir, '_timing.jsonl')})")

    if args.smoke:
        print("-" * 100)
        ok = verify_written_cells(out_dir)
        print(f"SMOKE self-check {'PASS' if ok else 'FAIL'} (finiteness + leakage guards + equal-HPO; "
              f"OOD magnitudes NOT interpreted -- full run is separate).")
    print("NOTE: single-shot binding confirmatory run; hypotheses/metric/thresholds are FROZEN.")


if __name__ == "__main__":
    main()
