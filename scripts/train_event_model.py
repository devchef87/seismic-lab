#!/usr/bin/env python3
"""Event-triggered earthquake escalation model.

Paradigm shift: instead of hourly zone snapshots, each training sample is a
SINGLE EARTHQUAKE. When a new event arrives, score it immediately:
  - What's the sequence context? (catalog lookback)
  - What are environmental conditions right now?
  - Label: did a larger event follow within GK-radius / 7 days?

Two-stage evaluation:
  Stage 1: Catalog-only (sequence features) — the base probability
  Stage 2: Catalog + environmental snapshot — does environment add lift?
"""
import os, sys, math, sqlite3, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import lightgbm as lgb

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "quakewatch.db")
MIN_MAG = 2.5          # trigger threshold (need small events for early sequence detection)
YEAR_MIN = "2015"
LABEL_DELTA = 1.0      # follower must be trigger_mag + this
WINDOW_DAYS = 7
GK_CAP_KM = 300        # cap GK radius for very large events


def gk_radius(mag):
    return min(GK_CAP_KM, 10 ** (0.1238 * mag + 0.983))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(min(1, math.sqrt(a)))


def haversine_vec(lat1, lon1, lats, lons):
    R = 6371.0
    dlat = np.radians(lats - lat1); dlon = np.radians(lons - lon1)
    a = np.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        np.cos(np.radians(lats)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def load_events():
    conn = sqlite3.connect(DB, timeout=60)
    rows = conn.execute(
        "SELECT id, timestamp, magnitude, depth_km, lat, lon FROM earthquakes "
        "WHERE magnitude >= ? AND substr(timestamp,1,4) >= ? ORDER BY timestamp",
        (MIN_MAG, YEAR_MIN)).fetchall()
    conn.close()
    from datetime import datetime
    ids, epochs, mags, depths, lats, lons = [], [], [], [], [], []
    for eid, ts, mag, dep, lat, lon in rows:
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ids.append(eid); epochs.append(t.timestamp())
            mags.append(mag); depths.append(dep or 10.0)
            lats.append(lat); lons.append(lon)
        except Exception:
            continue
    print(f"Loaded {len(ids):,} events (M{MIN_MAG}+, {YEAR_MIN}+)")
    return (ids, np.array(epochs), np.array(mags, dtype=np.float32),
            np.array(depths, dtype=np.float32),
            np.array(lats, dtype=np.float32), np.array(lons, dtype=np.float32))


def compute_labels(epochs, mags, lats, lons):
    """For each event, did a larger event (mag + LABEL_DELTA) follow within GK/7d?"""
    n = len(epochs)
    labels = np.zeros(n, dtype=np.float32)
    window_s = WINDOW_DAYS * 86400

    for i in range(n):
        r_km = gk_radius(mags[i])
        threshold = mags[i] + LABEL_DELTA
        j_end = np.searchsorted(epochs, epochs[i] + window_s, side='right')

        for j in range(i + 1, j_end):
            if mags[j] < threshold:
                continue
            dlat = abs(lats[j] - lats[i])
            if dlat > r_km / 111.0:
                continue
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            if d <= r_km:
                labels[i] = 1.0
                break

        if (i + 1) % 100000 == 0:
            print(f"  labels: {i+1:,}/{n:,}")
            sys.stdout.flush()

    print(f"  Labels: {labels.sum():.0f}/{n:,} positive ({labels.mean()*100:.2f}%)")
    return labels


CATALOG_FEAT_NAMES = [
    # Event properties
    "trigger_mag", "trigger_depth",
    # Sequence counts (trailing window, GK radius)
    "n_events_24h", "n_events_72h", "n_events_7d", "n_events_30d",
    "n_m4_events_7d",
    # Sequence magnitudes
    "max_mag_72h", "max_mag_7d", "mag_range_7d", "median_mag_7d",
    # Sequence shape
    "consec_up", "rumble_ratio", "mag_trend_72h",
    "double_tap_count", "depth_to_peak",
    # Activity ratios
    "active_ratio_7d_30d", "rate_accel_6h_24h",
    # Timing
    "hours_since_last", "hours_since_m4",
    # Seismological
    "b_value_14d", "energy_ratio_24h_48h",
    "moment_7d_log", "moment_accel",
    "depth_trend_7d",
]

ENV_FEAT_NAMES = [
    "sw_speed", "sw_density", "imf_bz", "imf_bt",
    "dst", "kp",
    "dart_loading", "dart_residual_std",
    "tidal_potential", "tidal_strain",
]

ALL_FEAT_NAMES = CATALOG_FEAT_NAMES + ENV_FEAT_NAMES


def compute_catalog_features(epochs, mags, depths, lats, lons):
    """Per-event catalog/sequence features. Vectorized where possible."""
    n = len(epochs)
    nf = len(CATALOG_FEAT_NAMES)
    feats = np.zeros((n, nf), dtype=np.float32)
    fi = {name: i for i, name in enumerate(CATALOG_FEAT_NAMES)}

    H1 = 3600
    W24 = 24 * H1; W72 = 72 * H1; W7D = 7 * 86400; W14D = 14 * 86400; W30D = 30 * 86400

    print("  Computing catalog features...")
    t0 = time.time()

    for i in range(n):
        feats[i, fi["trigger_mag"]] = mags[i]
        feats[i, fi["trigger_depth"]] = depths[i]

        r_km = gk_radius(mags[i])
        t_i = epochs[i]

        # Backward window: find all events within GK radius and various time windows
        j_start = np.searchsorted(epochs, t_i - W30D, side='left')
        if j_start >= i:
            # No prior events in 30d window
            feats[i, fi["hours_since_last"]] = 999.0
            feats[i, fi["hours_since_m4"]] = 999.0
            continue

        # Rough lat/lon box
        cand_slice = slice(j_start, i)
        dlat_mask = np.abs(lats[cand_slice] - lats[i]) <= r_km / 111.0
        dlon_deg = r_km / (111.0 * max(0.1, math.cos(math.radians(lats[i]))))
        dlon_mask = np.abs(lons[cand_slice] - lons[i]) <= dlon_deg
        box_mask = dlat_mask & dlon_mask

        if not box_mask.any():
            feats[i, fi["hours_since_last"]] = 999.0
            feats[i, fi["hours_since_m4"]] = 999.0
            continue

        cand_idx = np.where(box_mask)[0] + j_start
        dists = haversine_vec(lats[i], lons[i], lats[cand_idx], lons[cand_idx])
        nearby_mask = dists <= r_km
        nearby = cand_idx[nearby_mask]

        if len(nearby) == 0:
            feats[i, fi["hours_since_last"]] = 999.0
            feats[i, fi["hours_since_m4"]] = 999.0
            continue

        nb_epochs = epochs[nearby]
        nb_mags = mags[nearby]
        nb_depths = depths[nearby]
        dt = t_i - nb_epochs  # seconds ago

        # Time windows
        m24 = dt <= W24; m72 = dt <= W72; m7d = dt <= W7D
        m4_mask = nb_mags >= 4.0

        # Counts
        feats[i, fi["n_events_24h"]] = m24.sum()
        feats[i, fi["n_events_72h"]] = m72.sum()
        feats[i, fi["n_events_7d"]] = m7d.sum()
        feats[i, fi["n_events_30d"]] = len(nearby)
        feats[i, fi["n_m4_events_7d"]] = (m7d & m4_mask).sum()

        # Magnitudes in 7d
        if m7d.sum() > 0:
            m7_mags = nb_mags[m7d]
            feats[i, fi["max_mag_72h"]] = nb_mags[m72].max() if m72.any() else 0
            feats[i, fi["max_mag_7d"]] = m7_mags.max()
            feats[i, fi["mag_range_7d"]] = m7_mags.max() - m7_mags.min()
            feats[i, fi["median_mag_7d"]] = np.median(m7_mags)

            # Sequence shape (chronological order within 7d)
            seq_mags = m7_mags  # already in time order (nearby preserves sort)
            if len(seq_mags) >= 2:
                # Consecutive up
                run = 1; best = 1
                for k in range(1, len(seq_mags)):
                    if seq_mags[k] >= seq_mags[k-1] - 0.05:
                        run += 1; best = max(best, run)
                    else:
                        run = 1
                feats[i, fi["consec_up"]] = best

                # Rumble ratio
                med = np.median(seq_mags)
                if med > 0:
                    feats[i, fi["rumble_ratio"]] = seq_mags.max() / med

                # Depth to peak
                feats[i, fi["depth_to_peak"]] = np.argmax(seq_mags)

            # Mag trend (72h)
            if m72.sum() >= 3:
                m72_mags = nb_mags[m72]
                x = np.arange(len(m72_mags), dtype=np.float32)
                xm = x.mean()
                denom = ((x - xm) ** 2).sum()
                if denom > 0:
                    feats[i, fi["mag_trend_72h"]] = ((x - xm) * (m72_mags - m72_mags.mean())).sum() / denom

            # Double-tap
            if len(seq_mags) >= 3:
                taps = 0
                for k in range(len(seq_mags) - 2):
                    if abs(seq_mags[k] - seq_mags[k+1]) <= 0.3 and \
                       seq_mags[k+2] >= seq_mags[k] + 1.0:
                        taps += 1
                feats[i, fi["double_tap_count"]] = taps

            # B-value (14d)
            m14d = dt <= W14D
            if m14d.sum() >= 10:
                bm = nb_mags[m14d]
                mc = MIN_MAG
                above = bm[bm >= mc]
                if len(above) >= 5:
                    feats[i, fi["b_value_14d"]] = np.log10(np.e) / (above.mean() - mc + 0.01)

            # Energy ratio (24h vs prior 24h)
            e24 = np.sum(10 ** (1.5 * nb_mags[m24])) if m24.any() else 0
            m24_48 = (dt > W24) & (dt <= 2 * W24)
            e_prior = np.sum(10 ** (1.5 * nb_mags[m24_48])) if m24_48.any() else 0.01
            feats[i, fi["energy_ratio_24h_48h"]] = e24 / (e_prior + 0.01)

            # Moment
            moment_7d = np.sum(10 ** (1.5 * m7_mags + 9.1))
            feats[i, fi["moment_7d_log"]] = np.log10(moment_7d + 1)
            moment_prior = np.sum(10 ** (1.5 * nb_mags[~m7d] + 9.1)) if (~m7d).any() else 1
            feats[i, fi["moment_accel"]] = moment_7d / (moment_prior + 1)

            # Depth trend
            if m7d.sum() >= 3:
                d7 = nb_depths[m7d]
                x = np.arange(len(d7), dtype=np.float32)
                xm = x.mean()
                denom = ((x - xm) ** 2).sum()
                if denom > 0:
                    feats[i, fi["depth_trend_7d"]] = ((x - xm) * (d7 - d7.mean())).sum() / denom

        # Activity ratios
        n7 = m7d.sum(); n30 = len(nearby)
        feats[i, fi["active_ratio_7d_30d"]] = n7 / max(1, n30)
        n6h = (dt <= 6 * H1).sum(); n24h = m24.sum()
        rate_6h = n6h; rate_prior_18h = max(n24h - n6h, 0) / 3.0
        feats[i, fi["rate_accel_6h_24h"]] = rate_6h / (rate_prior_18h + 1.0)

        # Timing
        feats[i, fi["hours_since_last"]] = dt.min() / 3600.0
        m4_times = dt[m4_mask]
        feats[i, fi["hours_since_m4"]] = m4_times.min() / 3600.0 if len(m4_times) > 0 else 999.0

        if (i + 1) % 100000 == 0:
            print(f"    {i+1:,}/{n:,} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    print(f"  Catalog features done in {time.time()-t0:.0f}s")
    return feats


def compute_env_features(epochs, lats, lons):
    """Environmental snapshot at each event time. Queries the samples table for
    the nearest-in-time value of each signal."""
    n = len(epochs)
    nf = len(ENV_FEAT_NAMES)
    feats = np.full((n, nf), np.nan, dtype=np.float32)
    fi = {name: i for i, name in enumerate(ENV_FEAT_NAMES)}

    conn = sqlite3.connect(DB, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")

    # Load global signals as hourly time series (reuse for all events)
    print("  Loading environmental signals...")
    t0 = time.time()

    signal_map = {
        "sw_speed": ("solar_wind_speed", None),
        "sw_density": ("solar_wind_density", None),
        "imf_bz": ("imf_bz_gsm", None),
        "imf_bt": ("imf_bt", None),
        "dst": ("dst_index", None),
        "kp": ("kp_index", None),
        "tidal_potential": ("tidal_potential", None),
        "tidal_strain": ("tidal_strain_rate", None),
    }

    # Build hourly index from min to max event time
    t_min = epochs.min(); t_max = epochs.max()
    h_start = int(t_min // 3600) * 3600
    h_end = int(t_max // 3600 + 1) * 3600
    n_hours = (h_end - h_start) // 3600
    hour_epochs = np.arange(h_start, h_end, 3600, dtype=np.float64)

    full_idx = pd.date_range(
        pd.Timestamp(h_start, unit="s", tz="UTC"),
        pd.Timestamp(h_end, unit="s", tz="UTC"),
        freq="h")[:-1]

    from datetime import datetime, timezone
    ts_lo = datetime.fromtimestamp(t_min - 86400, tz=timezone.utc).strftime("%Y-%m-%dT%H")
    ts_hi = datetime.fromtimestamp(t_max + 86400, tz=timezone.utc).strftime("%Y-%m-%dT%H")

    for feat_name, (metric, source) in signal_map.items():
        t1 = time.time()
        # Aggregate to hourly in SQL to avoid loading millions of rows
        q = ("SELECT substr(timestamp,1,13) AS hr, AVG(value) FROM samples "
             "WHERE metric=? AND timestamp >= ? AND timestamp <= ?")
        params = [metric, ts_lo, ts_hi]
        if source:
            q += " AND source=?"
            params.append(source)
        q += " GROUP BY hr ORDER BY hr"
        rows = conn.execute(q, params).fetchall()
        if not rows:
            print(f"    {feat_name}: no data")
            continue
        vals = pd.Series(dtype=np.float64)
        for hr_str, v in rows:
            try:
                t = pd.Timestamp(hr_str + ":00:00", tz="UTC")
                vals[t] = v
            except Exception:
                continue
        if vals.empty:
            print(f"    {feat_name}: empty after parse")
            continue
        hourly = vals.reindex(full_idx).interpolate(limit=6).ffill(limit=12).values
        print(f"    {feat_name}: {len(rows):,} hourly bins, {time.time()-t1:.1f}s")

        # Map each event to its hour
        event_hour_idx = ((epochs - h_start) / 3600).astype(int)
        event_hour_idx = np.clip(event_hour_idx, 0, len(hourly) - 1)
        feats[:, fi[feat_name]] = hourly[event_hour_idx]

    # DART — needs per-zone nearest station, skip for now (use NaN)
    # Will be zone-specific in production

    conn.close()
    nan_rate = np.isnan(feats).mean()
    print(f"  Environmental features done in {time.time()-t0:.0f}s (NaN rate: {nan_rate:.1%})")
    return feats


def assign_zones(lats, lons):
    """Simple zone assignment for macro-AUC computation."""
    zones = []
    for lat, lon in zip(lats, lons):
        if lat > 50 and lon < -130:
            z = "alaska"
        elif 32 <= lat <= 42 and -125 <= lon <= -114:
            z = "california"
        elif -10 <= lat <= 12 and -90 <= lon <= -75:
            z = "south_america_n"
        elif -60 <= lat <= -10 and -90 <= lon <= -60:
            z = "south_america"
        elif 25 <= lat <= 50 and 125 <= lon <= 155:
            z = "japan_kurils"
        elif -15 <= lat <= 10 and 95 <= lon <= 145:
            z = "indonesia"
        elif -50 <= lat <= -33 and 165 <= lon <= 180:
            z = "new_zealand"
        elif 5 <= lat <= 22 and 118 <= lon <= 130:
            z = "philippines"
        elif 25 <= lat <= 45 and 65 <= lon <= 105:
            z = "himalaya"
        elif 10 <= lat <= 25 and -100 <= lon <= -80:
            z = "mexico_ca"
        elif 30 <= lat <= 45 and -10 <= lon <= 45:
            z = "mediterranean"
        elif 10 <= lat <= 25 and -85 <= lon <= -58:
            z = "caribbean"
        elif 45 <= lat <= 62 and 155 <= lon <= 170:
            z = "kamchatka"
        else:
            z = "other"
        zones.append(z)
    return np.array(zones)


def train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, Z_te, feat_names, tag):
    print(f"\n{'='*65}")
    print(f"  {tag}")
    print(f"  Train: {len(y_tr):,} ({y_tr.mean()*100:.2f}% pos)")
    print(f"  Val:   {len(y_va):,} ({y_va.mean()*100:.2f}% pos)")
    print(f"  Test:  {len(y_te):,} ({y_te.mean()*100:.2f}% pos)")
    print(f"  Features: {len(feat_names)}")
    print(f"{'='*65}")

    dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, y_va, feature_name=feat_names, free_raw_data=False)

    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 63, "max_depth": 7,
        "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 50, "lambda_l1": 0.1, "lambda_l2": 1.0,
        "scale_pos_weight": (1 - y_tr.mean()) / max(y_tr.mean(), 0.001),
        "verbose": -1,
    }

    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval], callbacks=[
                          lgb.early_stopping(100), lgb.log_evaluation(50)])

    pred = model.predict(X_te)
    pooled = roc_auc_score(y_te, pred)
    ap = average_precision_score(y_te, pred)

    # Per-zone AUC
    aucs = {}
    for z in set(Z_te):
        if z == "other":
            continue
        mm = Z_te == z; s = y_te[mm].sum()
        if mm.sum() >= 50 and 0 < s < mm.sum():
            aucs[z] = roc_auc_score(y_te[mm], pred[mm])

    macro = float(np.mean(list(aucs.values()))) if aucs else 0
    print(f"\n  >>> {tag}")
    print(f"  Pooled AUC: {pooled:.4f} | Macro AUC: {macro:.4f} | AP: {ap:.4f}")

    print(f"\n  Per-zone:")
    for z in sorted(aucs, key=lambda z: -aucs[z]):
        print(f"    {z:<20} {aucs[z]:.4f}")

    # Feature importance
    imp = dict(zip(feat_names, model.feature_importance(importance_type="gain")))
    tot = sum(imp.values()) or 1
    top = sorted(imp, key=lambda f: -imp[f])[:15]
    print(f"\n  Top 15 features:")
    for f in top:
        print(f"    {f:<30} {100*imp[f]/tot:.1f}%")

    return model, macro, pooled, aucs


def main():
    ids, epochs, mags, depths, lats, lons = load_events()

    # Labels
    print("\nComputing labels (did M+1.0 follow within GK/7d?)...")
    labels = compute_labels(epochs, mags, lats, lons)

    # Features
    print("\nComputing catalog features...")
    cat_feats = compute_catalog_features(epochs, mags, depths, lats, lons)

    print("\nComputing environmental features...")
    env_feats = compute_env_features(epochs, lats, lons)

    # Zone assignment
    zones = assign_zones(lats, lons)
    print(f"Zone distribution: {dict(zip(*np.unique(zones, return_counts=True)))}")

    # Temporal split: 70/15/15
    n = len(epochs)
    tr = int(n * 0.70); va = int(n * 0.85)
    print(f"\nTemporal split: train {tr:,} | val {va-tr:,} | test {n-va:,}")

    # Filter to M4+ triggers only for evaluation (smaller events are mostly noise)
    # But train on everything so the model sees the full sequence
    m4_mask = mags >= 4.0
    print(f"M4+ events: {m4_mask.sum():,} / {n:,}")

    # Stage 1: Catalog-only
    model_cat, macro_cat, pooled_cat, aucs_cat = train_and_eval(
        cat_feats[:tr], labels[:tr],
        cat_feats[tr:va], labels[tr:va],
        cat_feats[va:], labels[va:],
        zones[va:], CATALOG_FEAT_NAMES,
        "STAGE 1: Catalog/Sequence Only (event-level)")

    # Stage 2: Full model (catalog + environment) — skip if env not loaded
    if env_feats is not None:
        all_feats = np.hstack([cat_feats, env_feats])
        model_full, macro_full, pooled_full, aucs_full = train_and_eval(
            all_feats[:tr], labels[:tr],
            all_feats[tr:va], labels[tr:va],
            all_feats[va:], labels[va:],
            zones[va:], ALL_FEAT_NAMES,
            "STAGE 2: Catalog + Environment (event-level)")
    else:
        macro_full = macro_cat; pooled_full = pooled_cat; aucs_full = aucs_cat

    # Stage 3: M4+ only (realistic operational threshold)
    m4_tr = m4_mask[:tr]; m4_va = m4_mask[tr:va]; m4_te = m4_mask[va:]
    if m4_te.sum() >= 100:
        model_m4, macro_m4, pooled_m4, aucs_m4 = train_and_eval(
            cat_feats[:tr][m4_tr], labels[:tr][m4_tr],
            cat_feats[tr:va][m4_va], labels[tr:va][m4_va],
            cat_feats[va:][m4_te], labels[va:][m4_te],
            zones[va:][m4_te], CATALOG_FEAT_NAMES,
            "STAGE 3: Catalog-only, M4+ triggers only")
    else:
        macro_m4 = 0; aucs_m4 = {}

    # Summary
    print("\n" + "=" * 65)
    print("  EVENT-LEVEL MODEL SUMMARY")
    print("=" * 65)
    print(f"  Stage 1 (catalog only):       macro {macro_cat:.4f}  pooled {pooled_cat:.4f}")
    if env_feats is not None:
        print(f"  Stage 2 (catalog + env):      macro {macro_full:.4f}  pooled {pooled_full:.4f}")
    if m4_te.sum() >= 100:
        print(f"  Stage 3 (catalog, M4+ only):  macro {macro_m4:.4f}  pooled {pooled_m4:.4f}")
    if env_feats is not None:
        print(f"\n  Environment lift: {macro_full - macro_cat:+.4f} macro")
        print(f"\n  Per-zone comparison:")
        all_zones = sorted(set(list(aucs_cat.keys()) + list(aucs_full.keys())))
        for z in all_zones:
            c = aucs_cat.get(z, float('nan'))
            f = aucs_full.get(z, float('nan'))
            delta = f - c if not (np.isnan(c) or np.isnan(f)) else float('nan')
            print(f"    {z:<20} cat {c:.4f}  full {f:.4f}  Δ{delta:+.4f}")

    # Save the best model
    best = model_cat  # start with catalog since it's the base
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models", "event_escalation_lgb.txt")
    best.save_model(out_path)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
