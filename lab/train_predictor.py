"""Train time-windowed prediction models and save for real-time serving.

Builds Gradient Boosting classifiers for 24h, 48h, 72h forecast horizons,
plus a magnitude regression model for estimating expected event size.

Run:  python3 lab/train_predictor.py [--windows 24,48,72] [--min-mag 6.0]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
import joblib
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings("ignore")

from lab.experiment import (
    FEATURE_DEFS, extract_features, compute_b_value, get_conn
)

UTC = timezone.utc
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def build_windowed_dataset(conn, window_hours=72, min_mag=6.0, neg_ratio=10):
    """Build dataset with labels based on whether M6+ occurs within window_hours AFTER the feature snapshot."""

    print(f"\n  Building {window_hours}h forecast dataset (M{min_mag}+)")

    quakes = conn.execute(
        "SELECT * FROM earthquakes WHERE magnitude >= ? ORDER BY timestamp",
        (min_mag,)
    ).fetchall()
    quakes = [dict(q) for q in quakes]

    # Deduplicate (same minute, same degree box)
    seen = set()
    unique = []
    for q in quakes:
        key = f"{q['timestamp'][:16]}_{round(q['lat'],0)}_{round(q['lon'],0)}"
        if key not in seen:
            seen.add(key)
            unique.append(q)
    quakes = unique
    print(f"  {len(quakes)} target events (M{min_mag}+)")

    feature_names = [f[0] for f in FEATURE_DEFS]
    X_pos, y_pos, meta_pos = [], [], []

    for i, q in enumerate(quakes):
        if i % 25 == 0:
            print(f"  Extracting positives: {i+1}/{len(quakes)}...", end="\r")

        # Feature snapshot taken window_hours BEFORE the event
        t_event = datetime.fromisoformat(q["timestamp"].replace("Z", "+00:00"))
        if t_event.tzinfo is None:
            t_event = t_event.replace(tzinfo=UTC)
        t_snapshot = t_event - timedelta(hours=window_hours)

        try:
            features = extract_features(conn, q["lat"], q["lon"],
                                        t_snapshot.isoformat(), hours=72)
            vec = [features.get(f, np.nan) for f in feature_names]
            X_pos.append(vec)
            y_pos.append(1)
            meta_pos.append({
                "mag": q["magnitude"], "place": q["place"],
                "ts": q["timestamp"], "lead_hours": window_hours,
            })
        except Exception:
            pass

    print(f"\n  Extracted {len(X_pos)} positive samples")

    # Negatives: random windows where NO M5.5+ happens in next window_hours
    from predict import PLATE_BOUNDARY_SEGMENTS
    boundary_points = []
    for seg in PLATE_BOUNDARY_SEGMENTS:
        for j in range(len(seg) - 1):
            mx = (seg[j][0] + seg[j+1][0]) / 2
            my = (seg[j][1] + seg[j+1][1]) / 2
            boundary_points.append((mx, my))

    ts_range = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM earthquakes").fetchone()
    t_min = datetime.fromisoformat(ts_range[0])
    t_max = datetime.fromisoformat(ts_range[1])

    n_neg = len(X_pos) * neg_ratio
    X_neg, y_neg, ts_neg = [], [], []
    rng = np.random.RandomState(42)
    attempts = 0

    print(f"  Generating {n_neg} negative samples (10:1 ratio, {window_hours}h/5° exclusion)...")

    while len(X_neg) < n_neg and attempts < n_neg * 20:
        attempts += 1
        days_range = (t_max - t_min).days
        rand_day = t_min + timedelta(days=rng.randint(30, max(31, days_range - 7)))
        rand_hour = rng.randint(0, 23)
        t_sample = rand_day.replace(hour=rand_hour, minute=0, second=0,
                                     tzinfo=UTC)

        bp = boundary_points[rng.randint(0, len(boundary_points))]
        lat = bp[0] + rng.uniform(-2, 2)
        lon = bp[1] + rng.uniform(-2, 2)

        # Exclusion: no M5.5+ within 5° and forecast window
        t_excl_end = (t_sample + timedelta(hours=window_hours)).isoformat()
        check = conn.execute(
            "SELECT COUNT(*) FROM earthquakes WHERE magnitude >= 5.5 "
            "AND timestamp >= ? AND timestamp <= ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (t_sample.isoformat(), t_excl_end, lat - 5, lat + 5, lon - 5, lon + 5)
        ).fetchone()[0]
        if check > 0:
            continue

        try:
            features = extract_features(conn, lat, lon,
                                        t_sample.isoformat(), hours=72)
            vec = [features.get(f, np.nan) for f in feature_names]
            X_neg.append(vec)
            y_neg.append(0)
            ts_neg.append(t_sample.isoformat())
        except Exception:
            pass

        if len(X_neg) % 100 == 0:
            print(f"  Negatives: {len(X_neg)}/{n_neg}...", end="\r")

    print(f"\n  Generated {len(X_neg)} negatives ({attempts} attempts)")

    X = np.array(X_pos + X_neg, dtype=float)
    y = np.array(y_pos + y_neg, dtype=int)
    mags = [m["mag"] for m in meta_pos] + [0.0] * len(X_neg)
    timestamps = [m["ts"] for m in meta_pos] + ts_neg

    print(f"  Dataset: {X.shape[0]} × {X.shape[1]} | Pos: {sum(y)} | Neg: {len(y)-sum(y)}")
    return X, y, np.array(mags), feature_names, meta_pos, timestamps


def _add_top_k_interactions(X, feature_names, importances, k=15):
    """Generate pairwise product features from the top-K most important features."""
    ranked_idx = np.argsort(importances)[::-1][:k]
    interaction_cols = []
    interaction_names = []
    for i in range(len(ranked_idx)):
        for j in range(i + 1, len(ranked_idx)):
            fi, fj = ranked_idx[i], ranked_idx[j]
            prod = X[:, fi] * X[:, fj]
            interaction_cols.append(prod)
            interaction_names.append(f"{feature_names[fi]}_X_{feature_names[fj]}")
    if interaction_cols:
        X_aug = np.column_stack([X] + interaction_cols)
        return X_aug, feature_names + interaction_names
    return X, feature_names


def _temporal_cv_splits(timestamps, n_splits=5):
    """Rolling temporal cross-validation: train on past, test on future."""
    sorted_idx = np.argsort(timestamps)
    n = len(sorted_idx)
    min_train = n // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        test_start = min_train + i * ((n - min_train) // n_splits)
        test_end = min_train + (i + 1) * ((n - min_train) // n_splits)
        train_idx = sorted_idx[:test_start]
        test_idx = sorted_idx[test_start:test_end]
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def train_classifier(X, y, feature_names, window_hours, timestamps=None):
    """Train GBM + LightGBM ensemble with temporal CV and auto-interactions."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
    import lightgbm as lgb

    # Drop features with >70% NaN
    nan_pct = np.isnan(X).mean(axis=0)
    keep = nan_pct < 0.7
    X_kept = X[:, keep]
    kept_names = [n for n, k in zip(feature_names, keep) if k]
    dropped = [n for n, k in zip(feature_names, keep) if not k]
    if dropped:
        print(f"  Dropped {len(dropped)} features: {', '.join(dropped)}")

    # Impute once for all models
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_kept)

    # Class weighting for imbalanced data
    n_pos = int(sum(y))
    n_neg_count = len(y) - n_pos
    pos_weight = n_neg_count / max(1, n_pos)
    sample_weights = np.where(y == 1, pos_weight, 1.0)
    print(f"  Class weighting: scale_pos_weight={pos_weight:.1f} ({n_pos} pos, {n_neg_count} neg)")

    # ── Phase 1: Initial GBM to get feature importance for interactions (#5) ──
    print(f"  Phase 1: Initial GBM for feature ranking ({len(kept_names)} features)...")
    gbm_init = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    gbm_init.fit(X_imp, y, sample_weight=sample_weights)
    init_importances = gbm_init.feature_importances_

    # ── Phase 2: Add top-K pairwise interactions (#5) ──
    X_aug, aug_names = _add_top_k_interactions(X_imp, kept_names, init_importances, k=15)
    n_interactions = len(aug_names) - len(kept_names)
    print(f"  Phase 2: Added {n_interactions} interaction features → {len(aug_names)} total")

    # ── Phase 3: Temporal rolling CV (#7) ──
    if timestamps and len(timestamps) == len(y):
        ts_epochs = []
        for ts in timestamps:
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=UTC)
                ts_epochs.append(t.timestamp())
            except:
                ts_epochs.append(0)
        ts_epochs = np.array(ts_epochs)
        splits = _temporal_cv_splits(ts_epochs, n_splits=5)
        cv_type = "temporal"
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(X_aug, y))
        cv_type = "stratified"

    print(f"  Phase 3: {cv_type} CV ({len(splits)} folds)")

    # ── Train GBM + LightGBM ensemble (#6) ──
    gbm_probs = np.zeros(len(y))
    lgb_probs = np.zeros(len(y))
    gbm_preds = np.zeros(len(y), dtype=int)
    fold_aucs_gbm = []
    fold_aucs_lgb = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X_aug[train_idx], X_aug[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        w_tr = sample_weights[train_idx]

        # GBM (with sample weights for class balance)
        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        gbm.fit(X_tr, y_tr, sample_weight=w_tr)
        gbm_p = gbm.predict_proba(X_te)[:, 1]
        gbm_probs[test_idx] = gbm_p
        gbm_preds[test_idx] = gbm.predict(X_te)

        # LightGBM (with scale_pos_weight for class balance)
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight,
            random_state=42, verbose=-1)
        lgb_clf.fit(X_tr, y_tr)
        lgb_p = lgb_clf.predict_proba(X_te)[:, 1]
        lgb_probs[test_idx] = lgb_p

        if len(np.unique(y_te)) > 1:
            fold_aucs_gbm.append(roc_auc_score(y_te, gbm_p))
            fold_aucs_lgb.append(roc_auc_score(y_te, lgb_p))
            print(f"    Fold {fold_i+1}: GBM={fold_aucs_gbm[-1]:.4f} LGB={fold_aucs_lgb[-1]:.4f} "
                  f"(train={len(train_idx)} test={len(test_idx)})")

    # Ensemble: average probabilities
    ensemble_probs = (gbm_probs + lgb_probs) / 2.0

    # Find optimal threshold via PR curve (maximize F1)
    prec_arr, rec_arr, thresholds = precision_recall_curve(y, ensemble_probs)
    f1_scores = 2 * prec_arr[:-1] * rec_arr[:-1] / np.maximum(0.001, prec_arr[:-1] + rec_arr[:-1])
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_f1_idx])
    ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)

    auc_gbm = roc_auc_score(y, gbm_probs)
    auc_lgb = roc_auc_score(y, lgb_probs)
    auc_ens = roc_auc_score(y, ensemble_probs)

    cm = confusion_matrix(y, ensemble_preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\n  ── Results ({cv_type} CV) ──")
    print(f"  GBM AUC:      {auc_gbm:.4f} (mean fold: {np.mean(fold_aucs_gbm):.4f})")
    print(f"  LightGBM AUC: {auc_lgb:.4f} (mean fold: {np.mean(fold_aucs_lgb):.4f})")
    print(f"  Ensemble AUC: {auc_ens:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f} (max F1={f1:.3f})")
    print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Also report at fixed 0.5 for comparison
    cm50 = confusion_matrix(y, (ensemble_probs >= 0.5).astype(int))
    tn50, fp50, fn50, tp50 = cm50.ravel()
    p50 = tp50 / max(1, tp50 + fp50)
    r50 = tp50 / max(1, tp50 + fn50)
    print(f"  (at 0.5: P={p50:.3f} R={r50:.3f} TP={tp50} FP={fp50} FN={fn50} TN={tn50})")

    # Fit final models on all data (with class weighting)
    gbm_final = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    gbm_final.fit(X_aug, y, sample_weight=np.where(y == 1, pos_weight, 1.0))

    lgb_final = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=pos_weight,
        random_state=42, verbose=-1)
    lgb_final.fit(X_aug, y)

    # Feature importance (ensemble average)
    gbm_imp = gbm_final.feature_importances_
    lgb_imp = lgb_final.feature_importances_ / max(1, lgb_final.feature_importances_.sum())
    gbm_imp_norm = gbm_imp / max(1e-10, gbm_imp.sum())
    avg_imp = (gbm_imp_norm + lgb_imp) / 2.0

    ranked = sorted(zip(aug_names, avg_imp), key=lambda x: x[1], reverse=True)
    print(f"\n  Top features (ensemble importance):")
    for i, (fname, imp) in enumerate(ranked[:15]):
        cat = next((f[1] for f in FEATURE_DEFS if f[0] == fname), "auto")
        pct = imp / max(1e-10, sum(v for _, v in ranked)) * 100
        print(f"    {i+1:2d}. {fname:35s} [{cat:8s}]  {pct:.1f}%")

    # Bundle ensemble for serving
    ensemble_model = {
        "imputer": imputer,
        "gbm": gbm_final,
        "lgb": lgb_final,
        "feature_mask": keep.tolist(),
        "kept_names": kept_names,
        "aug_names": aug_names,
        "interaction_indices": _get_interaction_indices(kept_names, init_importances, k=15),
    }

    metadata = {
        "window_hours": window_hours,
        "cv_type": cv_type,
        "auc_gbm": round(auc_gbm, 4),
        "auc_lgb": round(auc_lgb, 4),
        "auc_ensemble": round(auc_ens, 4),
        "auc": round(auc_ens, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "class_weighted": True,
        "scale_pos_weight": round(pos_weight, 2),
        "n_positive": int(sum(y)),
        "n_negative": int(len(y) - sum(y)),
        "n_base_features": len(kept_names),
        "n_interaction_features": n_interactions,
        "n_features": len(aug_names),
        "feature_names": aug_names,
        "feature_mask": keep.tolist(),
        "trained_at": datetime.now(UTC).isoformat(),
    }

    return ensemble_model, metadata


def _get_interaction_indices(feature_names, importances, k=15):
    """Return the index pairs used for interaction features."""
    ranked_idx = np.argsort(importances)[::-1][:k]
    pairs = []
    for i in range(len(ranked_idx)):
        for j in range(i + 1, len(ranked_idx)):
            pairs.append((int(ranked_idx[i]), int(ranked_idx[j])))
    return pairs


ZONE_DEFS = [
    {"id": "socal",       "lat": [31, 36],  "lon": [-121, -114]},
    {"id": "norcal",      "lat": [36, 50],  "lon": [-131, -119]},
    {"id": "alaska",      "lat": [50, 65],  "lon": [-180, -130]},
    {"id": "japan",       "lat": [28, 46],  "lon": [128, 148]},
    {"id": "indonesia",   "lat": [-12, 8],  "lon": [94, 136]},
    {"id": "chile_peru",  "lat": [-46, 2],  "lon": [-82, -65]},
    {"id": "mediterranean","lat": [33, 42], "lon": [-6, 45]},
    {"id": "mexico_ca",   "lat": [7, 25],   "lon": [-115, -77]},
    {"id": "himalaya",    "lat": [24, 40],  "lon": [65, 100]},
    {"id": "caribbean",   "lat": [10, 22],  "lon": [-85, -60]},
    {"id": "iceland",     "lat": [55, 68],  "lon": [-25, -12]},
    {"id": "philippines", "lat": [4, 26],   "lon": [118, 128]},
]


def _zone_for_quake(lat, lon):
    """Return zone_id for an earthquake location, or None if outside all zones."""
    for z in ZONE_DEFS:
        if z["lat"][0] <= lat <= z["lat"][1] and z["lon"][0] <= lon <= z["lon"][1]:
            return z["id"]
    return None


def _compute_zone_stats(conn):
    """Compute historical magnitude stats per zone for use as features."""
    stats = {}
    for z in ZONE_DEFS:
        rows = conn.execute(
            "SELECT magnitude FROM earthquakes "
            "WHERE magnitude >= 5.5 AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (z["lat"][0], z["lat"][1], z["lon"][0], z["lon"][1])
        ).fetchall()
        mags = np.array([r[0] for r in rows])
        if len(mags) >= 3:
            stats[z["id"]] = {
                "hist_max": float(mags.max()),
                "hist_p75": float(np.percentile(mags, 75)),
                "hist_p90": float(np.percentile(mags, 90)),
                "hist_std": float(mags.std()),
                "hist_median": float(np.median(mags)),
                "hist_n": len(mags),
            }
        else:
            stats[z["id"]] = {
                "hist_max": 7.0, "hist_p75": 6.3, "hist_p90": 6.5,
                "hist_std": 0.4, "hist_median": 6.0, "hist_n": len(mags),
            }
    return stats


def build_magnitude_dataset(conn, min_mag=5.5):
    """Build zone-aware magnitude training set from all M5.5+ quakes in known zones."""
    from lab.experiment import extract_features

    zone_stats = _compute_zone_stats(conn)
    zone_ids = [z["id"] for z in ZONE_DEFS]

    quakes = conn.execute(
        "SELECT id, timestamp, magnitude, lat, lon, place FROM earthquakes "
        "WHERE magnitude >= ? ORDER BY timestamp",
        (min_mag,)
    ).fetchall()
    quakes = [dict(q) for q in quakes]

    feature_names = [f[0] for f in FEATURE_DEFS]
    zone_feature_names = ["zone_hist_max", "zone_hist_p75", "zone_hist_p90",
                          "zone_hist_std", "zone_hist_median"]
    zone_onehot_names = [f"zone_{z}" for z in zone_ids]
    all_names = feature_names + zone_feature_names + zone_onehot_names

    X_rows, y_mags, timestamps, zones_out = [], [], [], []
    skipped = 0

    print(f"  Building zone-aware magnitude dataset from {len(quakes)} M{min_mag}+ events")

    for i, q in enumerate(quakes):
        if i % 50 == 0:
            print(f"  Processing: {i+1}/{len(quakes)} ({len(X_rows)} matched)...", end="\r")

        zone_id = _zone_for_quake(q["lat"], q["lon"])
        if zone_id is None:
            skipped += 1
            continue

        t_event = datetime.fromisoformat(q["timestamp"].replace("Z", "+00:00"))
        if t_event.tzinfo is None:
            t_event = t_event.replace(tzinfo=UTC)
        t_snapshot = t_event - timedelta(hours=72)

        try:
            features = extract_features(conn, q["lat"], q["lon"],
                                        t_snapshot.isoformat(), hours=72)
            vec = [features.get(f, np.nan) for f in feature_names]

            zs = zone_stats[zone_id]
            vec += [zs["hist_max"], zs["hist_p75"], zs["hist_p90"],
                    zs["hist_std"], zs["hist_median"]]

            onehot = [1.0 if z == zone_id else 0.0 for z in zone_ids]
            vec += onehot

            X_rows.append(vec)
            y_mags.append(q["magnitude"])
            timestamps.append(q["timestamp"])
            zones_out.append(zone_id)
        except Exception:
            pass

    print(f"\n  Extracted {len(X_rows)} samples ({skipped} outside zones)")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_mags, dtype=float)

    zone_counts = defaultdict(int)
    for z in zones_out:
        zone_counts[z] += 1
    print(f"  Zone distribution:")
    for z in zone_ids:
        if zone_counts[z] > 0:
            mags_z = [m for m, zid in zip(y_mags, zones_out) if zid == z]
            print(f"    {z:15s}  n={zone_counts[z]:4d}  "
                  f"range={min(mags_z):.1f}–{max(mags_z):.1f}")

    return X, y, all_names, timestamps, zones_out


def _resample_magnitude_data(X, y, zones, target_per_bucket=1200):
    """Oversample rare magnitude buckets via row duplication + jitter."""
    buckets = [(5.5, 6.0), (6.0, 6.5), (6.5, 7.0), (7.0, 7.5), (7.5, 10.0)]
    X_out, y_out, z_out = [], [], []
    rng = np.random.RandomState(42)

    for lo, hi in buckets:
        mask = (y >= lo) & (y < hi)
        n = mask.sum()
        if n == 0:
            continue
        X_b, y_b = X[mask], y[mask]
        z_b = [z for z, m in zip(zones, mask) if m]

        if n >= target_per_bucket:
            X_out.append(X_b)
            y_out.extend(y_b)
            z_out.extend(z_b)
        else:
            repeats = max(1, target_per_bucket // n)
            for _ in range(repeats):
                noise = rng.normal(0, 0.01, X_b.shape)
                noise[:, np.isnan(X_b).any(axis=0)] = 0
                X_out.append(X_b + noise)
                y_out.extend(y_b)
                z_out.extend(z_b)

    return np.vstack(X_out), np.array(y_out), z_out


def train_magnitude_model(X, y, mags, feature_names, conn=None):
    """Train zone-aware magnitude regression with balanced resampling."""
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import GradientBoostingRegressor
    import lightgbm as lgb

    if conn is not None:
        X_mag, y_mag, mag_names, ts_mag, zones_mag = build_magnitude_dataset(conn)
        if len(X_mag) < 20:
            print(f"  Too few samples ({len(X_mag)}) for magnitude regression")
            return None, None
    else:
        pos_mask = y == 1
        X_mag = X[pos_mask]
        y_mag = mags[pos_mask]
        mag_names = feature_names
        ts_mag = None
        zones_mag = None
        if len(X_mag) < 20:
            print(f"  Too few positive samples ({len(X_mag)}) for magnitude regression")
            return None, None

    nan_pct = np.isnan(X_mag).mean(axis=0)
    keep = nan_pct < 0.7
    X_kept = X_mag[:, keep]
    kept_names = [n for n, k in zip(mag_names, keep) if k]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_kept)

    zone_stats = _compute_zone_stats(conn) if conn else {}

    print(f"\n  ── Magnitude Regression (zone-aware, resampled) ──")
    print(f"  Samples: {len(y_mag)} | Features: {len(kept_names)}")

    # Distribution before resampling
    for label, lo, hi in [("M5.5-6.0", 5.5, 6.0), ("M6.0-6.5", 6.0, 6.5),
                           ("M6.5-7.0", 6.5, 7.0), ("M7.0-7.5", 7.0, 7.5),
                           ("M7.5+", 7.5, 10.0)]:
        n = ((y_mag >= lo) & (y_mag < hi)).sum()
        print(f"    {label}: {n}")

    # --- Temporal CV on ORIGINAL data (no resampling leak) ---
    if ts_mag:
        ts_epochs = []
        for ts in ts_mag:
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=UTC)
                ts_epochs.append(t.timestamp())
            except Exception:
                ts_epochs.append(0)
        ts_epochs = np.array(ts_epochs)
        splits = _temporal_cv_splits(ts_epochs, n_splits=5)
        cv_type = "temporal"
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X_imp))
        cv_type = "kfold"

    print(f"  CV: {cv_type} ({len(splits)} folds)")
    print(f"  Resampling: balanced bucket duplication + jitter within each train fold")

    cv_preds = np.full(len(y_mag), np.nan)
    fold_maes = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr_raw, X_te = X_imp[train_idx], X_imp[test_idx]
        y_tr_raw, y_te = y_mag[train_idx], y_mag[test_idx]
        z_tr = [zones_mag[i] for i in train_idx] if zones_mag else ["unk"] * len(train_idx)

        # Resample train fold to balance magnitude buckets
        X_tr, y_tr, _ = _resample_magnitude_data(X_tr_raw, y_tr_raw, z_tr)

        gbm = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            min_samples_leaf=8, subsample=0.8, loss="huber",
            random_state=42)
        gbm.fit(X_tr, y_tr)

        lgb_reg = lgb.LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            num_leaves=40, min_child_samples=8, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1)
        lgb_reg.fit(X_tr, y_tr)

        pred = (gbm.predict(X_te) + lgb_reg.predict(X_te)) / 2.0

        # Floor: clamp to zone minimum observed magnitude
        if zones_mag:
            for j, ti in enumerate(test_idx):
                zid = zones_mag[ti]
                if zid in zone_stats:
                    z_min = zone_stats[zid].get("hist_median", 5.5)
                    pred[j] = max(pred[j], z_min)

        cv_preds[test_idx] = pred

        fold_mae = np.mean(np.abs(y_te - pred))
        fold_maes.append(fold_mae)
        print(f"    Fold {fold_i+1}: MAE={fold_mae:.3f} "
              f"(train={len(X_tr)} [{len(X_tr_raw)}+resample] test={len(test_idx)})")

    # Only evaluate samples that were in a test fold
    evaluated = ~np.isnan(cv_preds)
    cv_residuals_full = np.zeros(len(y_mag))
    cv_residuals_full[evaluated] = y_mag[evaluated] - cv_preds[evaluated]

    eval_mae = np.mean(np.abs(y_mag[evaluated] - cv_preds[evaluated]))
    eval_rmse = np.sqrt(np.mean((y_mag[evaluated] - cv_preds[evaluated]) ** 2))

    print(f"\n  CV MAE: {eval_mae:.3f} | RMSE: {eval_rmse:.3f}")
    print(f"  Evaluated: {evaluated.sum()}/{len(y_mag)} samples")
    print(f"  Magnitude range: {y_mag.min():.1f} — {y_mag.max():.1f}")

    buckets = [("M5.5-6.0", 5.5, 6.0), ("M6.0-6.5", 6.0, 6.5),
               ("M6.5-7.0", 6.5, 7.0), ("M7.0+", 7.0, 10.0)]
    print(f"\n  Bucket performance:")
    for label, lo, hi in buckets:
        mask = evaluated & (y_mag >= lo) & (y_mag < hi)
        if mask.sum() > 0:
            bucket_mae = np.mean(np.abs(y_mag[mask] - cv_preds[mask]))
            bucket_mean_pred = np.mean(cv_preds[mask])
            bucket_mean_actual = np.mean(y_mag[mask])
            print(f"    {label:10s}  n={mask.sum():4d}  MAE={bucket_mae:.3f}  "
                  f"avg_pred={bucket_mean_pred:.2f}  avg_actual={bucket_mean_actual:.2f}")

    # Final models on ALL data, resampled
    z_all = zones_mag if zones_mag else ["unk"] * len(y_mag)
    X_final, y_final, _ = _resample_magnitude_data(X_imp, y_mag, z_all)

    print(f"\n  Final training: {len(X_final)} samples (resampled from {len(X_imp)})")

    gbm_final = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        min_samples_leaf=8, subsample=0.8, loss="huber",
        random_state=42)
    gbm_final.fit(X_final, y_final)

    lgb_final = lgb.LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.03,
        num_leaves=40, min_child_samples=8, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1)
    lgb_final.fit(X_final, y_final)

    # Feature importance
    gbm_imp = gbm_final.feature_importances_
    lgb_imp = lgb_final.feature_importances_ / max(1, lgb_final.feature_importances_.sum())
    gbm_imp_norm = gbm_imp / max(1e-10, gbm_imp.sum())
    avg_imp = (gbm_imp_norm + lgb_imp) / 2.0

    ranked = sorted(zip(kept_names, avg_imp), key=lambda x: x[1], reverse=True)
    print(f"\n  Top magnitude features:")
    for i, (fname, imp) in enumerate(ranked[:15]):
        pct = imp / max(1e-10, sum(v for _, v in ranked)) * 100
        print(f"    {i+1:2d}. {fname:35s}  {pct:.1f}%")

    mag_model = {
        "imputer": imputer,
        "gbm": gbm_final,
        "lgb": lgb_final,
        "feature_mask": keep.tolist(),
        "kept_names": kept_names,
        "zone_aware": True,
        "zone_stats": zone_stats,
    }

    metadata = {
        "type": "magnitude_regression_zone_aware",
        "n_samples": len(y_mag),
        "n_resampled": len(X_final),
        "cv_mae": round(eval_mae, 3),
        "cv_rmse": round(eval_rmse, 3),
        "cv_type": cv_type,
        "n_features": len(kept_names),
        "feature_names": kept_names,
        "feature_mask": keep.tolist(),
        "resampled": True,
        "trained_at": datetime.now(UTC).isoformat(),
    }

    return mag_model, metadata


def save_model(pipeline, metadata, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    meta_path = os.path.join(MODEL_DIR, f"{name}.json")
    joblib.dump(pipeline, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {model_path}")
    print(f"         {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train QuakeWatch prediction models")
    parser.add_argument("--windows", type=str, default="24,48,72",
                        help="Comma-separated forecast windows in hours")
    parser.add_argument("--min-mag", type=float, default=6.0)
    parser.add_argument("--neg-ratio", type=int, default=10)
    parser.add_argument("--magnitude-only", action="store_true",
                        help="Only retrain the magnitude model, skip forecast models")
    args = parser.parse_args()

    conn = get_conn()

    if args.magnitude_only:
        print(f"\n{'='*70}")
        print(f"  TRAINING ZONE-AWARE MAGNITUDE MODEL (magnitude-only mode)")
        print(f"{'='*70}")
        mag_model, mag_meta = train_magnitude_model(
            None, None, None, None, conn=conn)
        if mag_model:
            save_model(mag_model, mag_meta, "magnitude_est")
    else:
        windows = [int(w) for w in args.windows.split(",")]
        for wh in windows:
            print(f"\n{'='*70}")
            print(f"  TRAINING {wh}h FORECAST MODEL")
            print(f"  Improvements: 10:1 neg ratio, {wh}h/5° exclusion, GBM+LGB ensemble,")
            print(f"  auto-interactions (top-15 pairwise), temporal rolling CV, class weighting")
            print(f"{'='*70}")

            X, y, mags, feature_names, meta, timestamps = build_windowed_dataset(
                conn, window_hours=wh, min_mag=args.min_mag, neg_ratio=args.neg_ratio
            )

            model, metadata = train_classifier(X, y, feature_names, wh, timestamps)
            save_model(model, metadata, f"forecast_{wh}h")

            if wh == 72:
                print(f"\n{'='*70}")
                print(f"  TRAINING ZONE-AWARE MAGNITUDE MODEL")
                print(f"{'='*70}")
                mag_model, mag_meta = train_magnitude_model(
                    X, y, mags, feature_names, conn=conn)
                if mag_model:
                    save_model(mag_model, mag_meta, "magnitude_est")

    conn.close()
    print(f"\n  All models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
