#!/usr/bin/env python3
"""Train per-threshold magnitude models.

Instead of a static lookup table, train a separate LightGBM classifier for each
magnitude threshold (M5.0, M5.5, M6.0, M6.5, M7.0). Each model uses the same
25 sequence features but predicts P(≥Mx follows within GK/7d) directly.

The probabilities reflect the full sequence dynamics — pattern, event count,
rate acceleration, magnitude trend all influence each threshold independently.
As new quakes hit and features change, the predicted probabilities update.

Output: models/mag_threshold_m{50,55,60,65,70}.txt
Usage: python3 -u scripts/train_mag_models.py
"""
import os, sys, math, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_event_model import (
    load_events, compute_catalog_features, assign_zones,
    train_and_eval, CATALOG_FEAT_NAMES,
    gk_radius, haversine, GK_CAP_KM, WINDOW_DAYS,
)

THRESHOLDS = [5.0, 5.5, 6.0, 6.5, 7.0]
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models")


def compute_max_follower(epochs, mags, lats, lons):
    """For each event, find the max magnitude within GK-radius/7d after it."""
    n = len(epochs)
    max_follower = np.zeros(n, dtype=np.float32)
    window_s = WINDOW_DAYS * 86400

    print("Computing max follower magnitudes...")
    t0 = time.time()

    for i in range(n):
        r_km = gk_radius(mags[i])
        dlat_deg = r_km / 111.0
        dlon_deg = r_km / (111.0 * max(0.1, math.cos(math.radians(float(lats[i])))))

        j_end = np.searchsorted(epochs, epochs[i] + window_s, side='right')
        best = 0.0
        for j in range(i + 1, j_end):
            if mags[j] <= best:
                continue
            if abs(lats[j] - lats[i]) > dlat_deg:
                continue
            if abs(lons[j] - lons[i]) > dlon_deg:
                continue
            d = haversine(float(lats[i]), float(lons[i]),
                          float(lats[j]), float(lons[j]))
            if d <= r_km:
                best = float(mags[j])

        max_follower[i] = best

        if (i + 1) % 50000 == 0:
            print(f"  {i+1:,}/{n:,} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    print(f"  Done in {time.time()-t0:.0f}s")
    return max_follower


def main():
    ids, epochs, mags, depths, lats, lons = load_events()

    max_follower = compute_max_follower(epochs, mags, lats, lons)

    print("\nComputing catalog features...")
    cat_feats = compute_catalog_features(epochs, mags, depths, lats, lons)

    zones = assign_zones(lats, lons)

    n = len(epochs)
    tr = int(n * 0.70); va = int(n * 0.85)
    print(f"\nTemporal split: train {tr:,} | val {va-tr:,} | test {n-va:,}")

    print(f"\nMax follower distribution:")
    for thresh in THRESHOLDS:
        cnt = (max_follower >= thresh).sum()
        print(f"  ≥M{thresh:.1f}: {cnt:,} events ({cnt/n*100:.2f}%)")

    results = {}
    for thresh in THRESHOLDS:
        labels = (max_follower >= thresh).astype(np.float32)

        if labels[va:].sum() < 20:
            print(f"\n  ≥M{thresh:.1f}: skipping — only {labels[va:].sum():.0f} test positives")
            continue

        model, macro, pooled, aucs = train_and_eval(
            cat_feats[:tr], labels[:tr],
            cat_feats[tr:va], labels[tr:va],
            cat_feats[va:], labels[va:],
            zones[va:], CATALOG_FEAT_NAMES,
            f"≥M{thresh:.1f} follower within GK/7d")

        out_path = os.path.join(MODELS_DIR, f"mag_threshold_m{int(thresh*10)}.txt")
        model.save_model(out_path)
        results[thresh] = {"macro": macro, "pooled": pooled}
        print(f"  Saved: {out_path}")

    print("\n" + "=" * 65)
    print("  MAGNITUDE THRESHOLD MODELS SUMMARY")
    print("=" * 65)
    for thresh in THRESHOLDS:
        if thresh in results:
            r = results[thresh]
            print(f"  ≥M{thresh:.1f}: macro {r['macro']:.4f}  pooled {r['pooled']:.4f}")
        else:
            print(f"  ≥M{thresh:.1f}: skipped (insufficient data)")


if __name__ == "__main__":
    main()
