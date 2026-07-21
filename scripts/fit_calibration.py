#!/usr/bin/env python3
"""Fit isotonic probability calibration for all deployed event-level models.

The models were trained with scale_pos_weight (good for ranking, but it
inflates raw probabilities 2-4x — verified live: events shown 0.59 realized
0.16). This fits an isotonic curve per model on the VALIDATION slice (never
seen in training, never used for early stopping thresholds beyond AUC), so
displayed probabilities become empirical frequencies.

Also emits suggested watch/elevated bands for the big-event watch, chosen to
keep current alert volume (raw-score p99 / p99.7 of the validation
distribution) but expressed in calibrated probability.

Output: models/probability_calibration.npz
Usage:  python3 -u scripts/fit_calibration.py
"""
import os, sys
import numpy as np
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from train_event_model import (load_events, compute_labels,
                               compute_catalog_features, CATALOG_FEAT_NAMES)
from train_mag_models import compute_max_follower
from train_big_event_model import compute_regional, LABEL_WINDOW_DAYS

MODELS_DIR = os.path.join(ROOT, "models")
OUT = os.path.join(MODELS_DIR, "probability_calibration.npz")


def fit_iso(pred, y, name):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(pred, y)
    x, ycal = iso.X_thresholds_, iso.y_thresholds_
    # Report the distortion at a few raw levels
    probe = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
    mapped = np.interp(probe, x, ycal)
    print(f"  {name:<24} n={len(y):,} base={y.mean():.4f}  " +
          "  ".join(f"{p:.1f}->{m:.3f}" for p, m in zip(probe, mapped)))
    return x, ycal


def main():
    ids, epochs, mags, depths, lats, lons = load_events()
    n_full = len(epochs)

    print("\nCatalog features (shared)...")
    cat = compute_catalog_features(epochs, mags, depths, lats, lons)

    out = {}

    # ── Escalation model + GK/7d magnitude threshold models ──────────────
    print("\nEscalation labels (GK/7d, M+1.0)...")
    esc_labels = compute_labels(epochs, mags, lats, lons)
    print("Max-follower magnitudes (GK/7d)...")
    max_fol = compute_max_follower(epochs, mags, lats, lons)

    tr, va = int(n_full * 0.70), int(n_full * 0.85)
    val = slice(tr, va)

    print("\nFitting calibration curves (validation slice):")
    m = lgb.Booster(model_file=os.path.join(MODELS_DIR, "event_escalation_lgb.txt"))
    pred = m.predict(cat[val])
    out["esc_x"], out["esc_y"] = fit_iso(pred, esc_labels[val], "escalation")

    for t in [5.0, 5.5, 6.0]:
        path = os.path.join(MODELS_DIR, f"mag_threshold_m{int(t*10)}.txt")
        if not os.path.exists(path):
            continue
        m = lgb.Booster(model_file=path)
        pred = m.predict(cat[val])
        y = (max_fol[val] >= t).astype(float)
        out[f"mag{int(t*10)}_x"], out[f"mag{int(t*10)}_y"] = \
            fit_iso(pred, y, f"mag_threshold_m{int(t*10)}")

    # ── Big-event models (100km/30d labels, 41 features) ─────────────────
    print("\nBig-event labels + regional features (100km/30d)...")
    fwd_max, reg = compute_regional(epochs, mags, depths, lats, lons)
    feats41 = np.hstack([cat, reg])

    t_cut = epochs.max() - LABEL_WINDOW_DAYS * 86400
    keep = epochs <= t_cut
    feats41, fwd_max = feats41[keep], fwd_max[keep]
    nb = len(fwd_max)
    btr, bva = int(nb * 0.70), int(nb * 0.85)
    bval = slice(btr, bva)

    be_preds = {}
    for t in [5.5, 6.0]:
        path = os.path.join(MODELS_DIR, f"big_event_m{int(t*10)}.txt")
        if not os.path.exists(path):
            continue
        m = lgb.Booster(model_file=path)
        pred = m.predict(feats41[bval])
        be_preds[t] = pred
        y = (fwd_max[bval] >= t).astype(float)
        out[f"be{int(t*10)}_x"], out[f"be{int(t*10)}_y"] = \
            fit_iso(pred, y, f"big_event_m{int(t*10)}")

    # ── Suggested bands: keep today's alert volume, calibrated scale ─────
    if 6.0 in be_preds:
        raw99 = float(np.percentile(be_preds[6.0], 99.0))
        raw997 = float(np.percentile(be_preds[6.0], 99.7))
        watch_cal = float(np.interp(raw99, out["be60_x"], out["be60_y"]))
        elev_cal = float(np.interp(raw997, out["be60_x"], out["be60_y"]))
        out["be60_watch"] = np.array(watch_cal)
        out["be60_elevated"] = np.array(elev_cal)
        print(f"\n  Suggested bands (calibrated P(M6+/100km/30d)):")
        print(f"    WATCH    >= {watch_cal:.3f}  (raw-score p99   = {raw99:.3f})")
        print(f"    ELEVATED >= {elev_cal:.3f}  (raw-score p99.7 = {raw997:.3f})")

    np.savez(OUT, **out)
    print(f"\nSaved: {OUT}")
    print("Restart the realtime services to activate "
          "(scorers auto-load this file when present).")


if __name__ == "__main__":
    main()
