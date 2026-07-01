#!/usr/bin/env python3
"""Build magnitude exceedance table from historical escalation data.

For each sequence-magnitude bin, computes: given that escalation happened
(M+1.0 larger event followed within GK-radius/7d), what fraction of followers
reached various absolute magnitude thresholds?

Output: models/mag_exceedance_table.json

Used by event_scorer.py to report per-magnitude probabilities:
  P(≥M6.0) = escalation_prob × P(follower ≥ 6.0 | escalation, seq_max bin)

Usage: python3 -u scripts/build_mag_probs.py
"""
import os, sys, math, sqlite3, json
import numpy as np

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "quakewatch.db")
MIN_MAG = 2.5
YEAR_MIN = "2015"
LABEL_DELTA = 1.0
WINDOW_DAYS = 7
GK_CAP_KM = 300
W7D = 7 * 86400

THRESHOLDS = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

BINS = [
    (2.5, 3.5),
    (3.5, 4.5),
    (4.5, 5.5),
    (5.5, 6.5),
    (6.5, 10.0),
]


def gk_radius(mag):
    return min(GK_CAP_KM, 10 ** (0.1238 * mag + 0.983))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(min(1, math.sqrt(a)))


def main():
    conn = sqlite3.connect(DB, timeout=60)
    rows = conn.execute(
        "SELECT id, timestamp, magnitude, depth_km, lat, lon FROM earthquakes "
        "WHERE magnitude >= ? AND substr(timestamp,1,4) >= ? ORDER BY timestamp",
        (MIN_MAG, YEAR_MIN)).fetchall()
    conn.close()

    from datetime import datetime
    epochs, mags, lats, lons = [], [], [], []
    for eid, ts, mag, dep, lat, lon in rows:
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            epochs.append(t.timestamp()); mags.append(mag)
            lats.append(lat); lons.append(lon)
        except Exception:
            continue

    epochs = np.array(epochs)
    mags = np.array(mags, dtype=np.float32)
    lats = np.array(lats, dtype=np.float32)
    lons = np.array(lons, dtype=np.float32)
    n = len(epochs)
    window_s = WINDOW_DAYS * 86400
    print(f"Loaded {n:,} events")

    max_follower = np.full(n, np.nan, dtype=np.float32)
    max_prior_7d = np.zeros(n, dtype=np.float32)

    print("Computing follower + prior magnitudes...")
    for i in range(n):
        r_km = gk_radius(mags[i])
        dlat_deg = r_km / 111.0
        dlon_deg = r_km / (111.0 * max(0.1, math.cos(math.radians(float(lats[i])))))

        # Forward: max follower within GK/7d
        j_end = np.searchsorted(epochs, epochs[i] + window_s, side='right')
        best_fwd = 0.0
        for j in range(i + 1, j_end):
            if mags[j] <= best_fwd:
                continue
            if abs(lats[j] - lats[i]) > dlat_deg:
                continue
            if abs(lons[j] - lons[i]) > dlon_deg:
                continue
            d = haversine(float(lats[i]), float(lons[i]),
                          float(lats[j]), float(lons[j]))
            if d <= r_km:
                best_fwd = float(mags[j])

        if best_fwd >= mags[i] + LABEL_DELTA:
            max_follower[i] = best_fwd

        # Backward: max prior within GK/7d (for sequence max binning)
        j_start = np.searchsorted(epochs, epochs[i] - W7D, side='left')
        best_bwd = 0.0
        for j in range(j_start, i):
            if mags[j] <= best_bwd:
                continue
            if abs(lats[j] - lats[i]) > dlat_deg:
                continue
            if abs(lons[j] - lons[i]) > dlon_deg:
                continue
            d = haversine(float(lats[i]), float(lons[i]),
                          float(lats[j]), float(lons[j]))
            if d <= r_km:
                best_bwd = float(mags[j])

        max_prior_7d[i] = best_bwd

        if (i + 1) % 50000 == 0:
            esc_so_far = (~np.isnan(max_follower[:i+1])).sum()
            print(f"  {i+1:,}/{n:,}  ({esc_so_far:,} escalations so far)")
            sys.stdout.flush()

    seq_max = np.maximum(mags, max_prior_7d)
    escalated = ~np.isnan(max_follower)
    n_esc = escalated.sum()
    print(f"\nEscalations: {n_esc:,} / {n:,} ({n_esc/n*100:.1f}%)")

    # Build exceedance table
    table = {"thresholds": THRESHOLDS, "bins": []}

    print("\nExceedance by sequence-max bin:")
    for lo, hi in BINS:
        in_bin = (seq_max >= lo) & (seq_max < hi)
        esc_in_bin = in_bin & escalated
        n_total = int(in_bin.sum())
        n_esc_bin = int(esc_in_bin.sum())

        if n_esc_bin < 20:
            print(f"  [{lo}, {hi}): {n_total:,} events, {n_esc_bin} escalations (too few, skipping)")
            continue

        follower_mags = max_follower[esc_in_bin]
        exceedance = {}
        for thresh in THRESHOLDS:
            frac = float((follower_mags >= thresh).sum()) / n_esc_bin
            exceedance[str(thresh)] = round(frac, 4)

        bin_entry = {
            "lo": lo, "hi": hi,
            "n_events": n_total,
            "n_escalations": n_esc_bin,
            "base_rate": round(n_esc_bin / max(n_total, 1), 4),
            "exceedance": exceedance,
        }
        table["bins"].append(bin_entry)

        label = f"[{lo}, {hi})" if hi < 10 else f"[{lo}+)"
        print(f"\n  {label}: {n_total:,} events, {n_esc_bin:,} escalations ({bin_entry['base_rate']*100:.1f}%)")
        for thresh in THRESHOLDS:
            p = exceedance[str(thresh)]
            if p >= 0.0005:
                print(f"    ≥M{thresh:.1f}: {p*100:6.2f}%  ({int(follower_mags[follower_mags >= thresh].shape[0])} events)")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models", "mag_exceedance_table.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
