#!/usr/bin/env python3
"""T4: single ordinal big-event model replacing the per-threshold binaries.

One multiclass LightGBM over ordered follower-magnitude bins:
    0: <5.0   1: [5.0,5.5)   2: [5.5,6.0)   3: [6.0,6.5)   4: >=6.5
P(>=M5.5) = P2+P3+P4, P(>=M6.0) = P3+P4, P(>=M6.5) = P4 — monotone by
construction (a distribution's tail can't cross itself), so the inference
clamp disappears. The rare bins share the trunk with the abundant ones,
which is exactly where the per-threshold models were data-starved (M6.5).

Same 41 features and 100km/30d labels as train_big_event_model.py, same
splits — results are directly comparable. Deployment bar: beat the
per-threshold reference on FIRST-EVENT AUC at M5.5/M6.0 and not regress
P@0.1%. If it loses, delete models/big_event_ordinal.txt (the scorer
auto-prefers it when present).

Reference (per-threshold + T3, first-event):
    M5.5 AUC 0.6724 P@0.1%=23.9% | M6.0 AUC 0.7148 P@0.1%=14.9% | M6.5 AUC 0.474

RESULT (2026-07-25 run): LOST — not deployed. M5.5 first-event 0.6403,
M6.0 first-event 0.6887 with P@0.1% collapsing to 3.0% (vs 14.9%). M6.5
"won" 0.470 vs 0.436 but both are below chance. Multiclass splits capacity
across 5 bins and dilutes exactly the top-of-ranking sharpness the watch
depends on; the clamp problem it targeted is already solved by isotonic
calibration. Keep the per-threshold ensemble. Revisit only after a
pre-2015 catalog backfill (T5) changes the data regime.

Also fits isotonic calibration for each tail on the validation slice and
saves models/big_event_ordinal_calib.npz (curves + watch/elevated bands).

Usage: python3 -u scripts/train_ordinal_model.py
"""
import os, sys, time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from train_event_model import (load_events, compute_catalog_features,
                               assign_zones, CATALOG_FEAT_NAMES)
from train_big_event_model import (compute_regional, LABEL_WINDOW_DAYS,
                                   FIRST_EVENT_GATE_MAG, REGIONAL_FEAT_NAMES,
                                   ALL_FEAT_NAMES, precision_at_k)

MODELS_DIR = os.path.join(ROOT, "models")
BIN_EDGES = [5.0, 5.5, 6.0, 6.5]           # 5 ordered classes
TAILS = {"5.5": 2, "6.0": 3, "6.5": 4}     # tail prob = sum(classes >= idx)
REFERENCE = {"5.5": (0.6724, 0.239), "6.0": (0.7148, 0.149),
             "6.5": (0.4357, None)}


def to_classes(fwd_max):
    cls = np.zeros(len(fwd_max), dtype=np.int32)
    for i, edge in enumerate(BIN_EDGES):
        cls[fwd_max >= edge] = i + 1
    return cls


def main():
    ids, epochs, mags, depths, lats, lons = load_events()

    fwd_max, reg = compute_regional(epochs, mags, depths, lats, lons)
    print("\nCatalog features...")
    cat = compute_catalog_features(epochs, mags, depths, lats, lons)
    feats = np.hstack([cat, reg])
    zones = assign_zones(lats, lons)

    t_cut = epochs.max() - LABEL_WINDOW_DAYS * 86400
    keep = epochs <= t_cut
    feats, fwd_max, zones = feats[keep], fwd_max[keep], zones[keep]
    reg_max_prior = reg[keep][:, REGIONAL_FEAT_NAMES.index("r100_max_mag_30d")]
    first_event = reg_max_prior < FIRST_EVENT_GATE_MAG

    cls = to_classes(fwd_max)
    n = len(cls)
    tr, va = int(n * 0.70), int(n * 0.85)
    print(f"\nAfter trim: {n:,} events | split {tr:,}/{va-tr:,}/{n-va:,}")
    print("Class distribution (train):",
          dict(zip(*np.unique(cls[:tr], return_counts=True))))

    # Mild sqrt class weighting — lets the rare tail bins register without
    # the probability distortion of full inverse-frequency weighting.
    counts = np.bincount(cls[:tr], minlength=5).astype(float)
    w = np.sqrt(counts.max() / np.maximum(counts, 1))
    sample_w = w[cls[:tr]]
    print("Class weights:", np.round(w, 2).tolist())

    params = {
        "objective": "multiclass", "num_class": 5, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 63, "max_depth": 7,
        "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 50, "lambda_l1": 0.1, "lambda_l2": 1.0,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(feats[:tr], cls[:tr], weight=sample_w,
                         feature_name=ALL_FEAT_NAMES, free_raw_data=False)
    dval = lgb.Dataset(feats[tr:va], cls[tr:va],
                       feature_name=ALL_FEAT_NAMES, free_raw_data=False)

    print("\nTraining ordinal (multiclass) model...")
    t0 = time.time()
    model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100),
                                 lgb.log_evaluation(50)])
    print(f"Trained in {time.time()-t0:.0f}s, best iter {model.best_iteration}")

    # ── Evaluate tails on test, all-triggers and first-event ─────────────
    pred_te = model.predict(feats[va:])           # (n_te, 5)
    fe_te = first_event[va:]
    print(f"\n{'='*66}\n  ORDINAL MODEL vs PER-THRESHOLD REFERENCE (test)\n{'='*66}")
    for tkey, cidx in TAILS.items():
        tail = pred_te[:, cidx:].sum(axis=1)
        y = (fwd_max[va:] >= float(tkey)).astype(int)
        auc_all = roc_auc_score(y, tail) if 0 < y.sum() < len(y) else float("nan")
        y_fe, tail_fe = y[fe_te], tail[fe_te]
        auc_fe = roc_auc_score(y_fe, tail_fe) \
            if 0 < y_fe.sum() < len(y_fe) else float("nan")
        base, pk = precision_at_k(y_fe, tail_fe)
        ref_auc, ref_pk = REFERENCE[tkey]
        pk01 = pk[0][2]
        verdict = "BEATS" if auc_fe > ref_auc else "loses to"
        print(f"\n  M{tkey}: all-trigger AUC {auc_all:.4f} | "
              f"FIRST-EVENT AUC {auc_fe:.4f} ({verdict} ref {ref_auc:.4f})")
        print(f"        first-event P@0.1%={pk01*100:.1f}% (base {base*100:.2f}%)"
              + (f" | ref P@0.1%={ref_pk*100:.1f}%" if ref_pk else ""))

    # ── Calibrate tails on validation, derive bands ──────────────────────
    print("\nFitting tail calibration (validation slice)...")
    pred_va = model.predict(feats[tr:va])
    calib = {}
    for tkey, cidx in TAILS.items():
        tail = pred_va[:, cidx:].sum(axis=1)
        y = (fwd_max[tr:va] >= float(tkey)).astype(float)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(tail, y)
        calib[f"be{tkey.replace('.','')}_x"] = iso.X_thresholds_
        calib[f"be{tkey.replace('.','')}_y"] = iso.y_thresholds_

    # Meaning-based bands (see fit_calibration.py): volume-matched bands are
    # period-dependent; fixed calibrated meaning is not.
    calib["be60_watch"] = np.array(0.10)
    calib["be60_elevated"] = np.array(0.30)
    print("  Bands (calibrated P(M6+/100km/30d)): "
          "WATCH >= 0.10 (~4x base), ELEVATED >= 0.30 (~12x base)")

    model.save_model(os.path.join(MODELS_DIR, "big_event_ordinal.txt"))
    np.savez(os.path.join(MODELS_DIR, "big_event_ordinal_calib.npz"), **calib)
    print(f"\nSaved: models/big_event_ordinal.txt + big_event_ordinal_calib.npz")
    print("The scorer prefers the ordinal model when this file exists.")
    print("If it lost to the reference above, DELETE both files to keep "
          "the per-threshold ensemble.")


if __name__ == "__main__":
    main()
