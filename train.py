"""SeismicLab — ML training pipeline.

Two models:
1. Magnitude regression: given an earthquake just happened, predict its magnitude
   from pre-event signal conditions (less useful but validates signal correlation)
2. Binary classification: given current signal conditions, predict whether a M4.5+
   earthquake will occur within the next 6 hours (the actual prediction goal)

Uses XGBoost with the OMNI-aligned feature matrix.
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seismiclab.train")

DATA_DIR = Path(__file__).parent / "data"


def train_signal_correlation():
    """Train a model to find correlations between pre-event signals and
    earthquake magnitude. This is our validation step — if space weather
    and geomagnetic signals have ANY predictive power for seismic events,
    this model should outperform random baseline."""

    from ingest.store import QuakeStore
    from ingest.features import build_earthquake_features, save_dataset

    store = QuakeStore()

    log.info("=" * 60)
    log.info("PHASE 1: Building feature matrix (M4.0+ earthquakes, 2015-2026)")
    log.info("=" * 60)

    names, X, y, info = build_earthquake_features(
        store, min_mag=4.0, start="2015-01-01"
    )

    if X.shape[0] == 0:
        log.error("No data — run backfill first")
        return

    save_dataset(names, X, y, info, DATA_DIR / "features_magnitude.npz")

    log.info("=" * 60)
    log.info("PHASE 2: Feature selection (drop high-NaN columns)")
    log.info("=" * 60)

    # drop features with >50% NaN
    nan_frac = np.isnan(X).mean(axis=0)
    keep_mask = nan_frac < 0.5
    X_clean = X[:, keep_mask]
    names_clean = [n for n, k in zip(names, keep_mask) if k]

    log.info(f"Kept {X_clean.shape[1]} / {X.shape[1]} features (>50% coverage)")

    # impute remaining NaNs with column median
    for col in range(X_clean.shape[1]):
        mask = np.isnan(X_clean[:, col])
        if mask.any():
            median = np.nanmedian(X_clean[:, col])
            X_clean[mask, col] = median

    log.info("=" * 60)
    log.info("PHASE 3: Train/test split (2015-2024 train, 2025-2026 test)")
    log.info("=" * 60)

    # temporal split
    split_date = "2025-01-01"
    train_mask = np.array([i["timestamp"] < split_date for i in info])
    test_mask = ~train_mask

    X_train, X_test = X_clean[train_mask], X_clean[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    log.info(f"Train: {X_train.shape[0]} events (2015-2024)")
    log.info(f"Test:  {X_test.shape[0]} events (2025-2026)")
    log.info(f"Train mag range: {y_train.min():.1f} - {y_train.max():.1f}")
    log.info(f"Test mag range:  {y_test.min():.1f} - {y_test.max():.1f}")

    log.info("=" * 60)
    log.info("PHASE 4: Training XGBoost regression")
    log.info("=" * 60)

    try:
        import xgboost as xgb
    except ImportError:
        log.error("XGBoost not installed. Run: pip install xgboost")
        return

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    log.info("=" * 60)
    log.info("PHASE 5: Evaluation")
    log.info("=" * 60)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # baseline: always predict mean magnitude
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    log.info(f"Train MAE: {train_mae:.3f}  RMSE: {train_rmse:.3f}  R²: {train_r2:.3f}")
    log.info(f"Test  MAE: {test_mae:.3f}  RMSE: {test_rmse:.3f}  R²: {test_r2:.3f}")
    log.info(f"Baseline MAE: {baseline_mae:.3f}  RMSE: {baseline_rmse:.3f}")
    log.info(f"Improvement over baseline: {(1 - test_mae/baseline_mae)*100:.1f}% MAE")

    log.info("=" * 60)
    log.info("PHASE 6: Feature importance (top 30)")
    log.info("=" * 60)

    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:30]

    for rank, idx in enumerate(top_idx, 1):
        log.info(f"  {rank:2d}. {names_clean[idx]}: {importance[idx]:.4f}")

    # save model
    model_path = DATA_DIR / "model_magnitude.json"
    model.save_model(str(model_path))
    log.info(f"Model saved to {model_path}")

    # save results
    results = {
        "task": "magnitude_regression",
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "n_features": int(X_clean.shape[1]),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "baseline_mae": float(baseline_mae),
        "baseline_rmse": float(baseline_rmse),
        "improvement_pct": float((1 - test_mae / baseline_mae) * 100),
        "top_features": [
            {"name": names_clean[idx], "importance": float(importance[idx])}
            for idx in top_idx
        ],
    }

    results_path = DATA_DIR / "results_magnitude.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved to {results_path}")

    return results


def train_binary_predictor(target_mag=5.0, window_hours=24, **kwargs):
    """Train a global binary classifier with foreshock + space weather features.

    Uses event-centered sampling: positive samples from pre-earthquake
    conditions, negative samples from confirmed quiet periods."""

    from ingest.store import QuakeStore
    from ingest.features import build_global_prediction_features, save_dataset

    store = QuakeStore()

    log.info("=" * 60)
    log.info(f"GLOBAL BINARY: M{target_mag}+ within {window_hours}h (foreshock + space weather)")
    log.info("=" * 60)

    cache_path = DATA_DIR / "features_binary.npz"
    if cache_path.exists() and not kwargs.get("rebuild", False):
        log.info(f"Loading cached features from {cache_path}")
        data = np.load(str(cache_path), allow_pickle=True)
        names = list(data["feature_names"])
        X = data["X"]
        y = data["y"]
        timestamps = json.loads(str(data["eq_info"]))
        log.info(f"Loaded {X.shape[0]} samples × {X.shape[1]} features")
    else:
        names, X, y, timestamps = build_global_prediction_features(
            store, target_mag=target_mag, window_hours=window_hours,
        )
        if X.shape[0] == 0:
            log.error("No data")
            return
        save_dataset(names, X, y, timestamps, cache_path)

    # sort by timestamp so train/test split is temporal
    ts_keys = []
    for t in timestamps:
        ts_str = t.get("timestamp", "") if isinstance(t, dict) else str(t)
        ts_keys.append(ts_str)
    sort_idx = np.argsort(ts_keys)
    X = X[sort_idx]
    y = y[sort_idx]
    timestamps = [timestamps[i] for i in sort_idx]

    # drop high-NaN features
    nan_frac = np.isnan(X).mean(axis=0)
    keep_mask = nan_frac < 0.5
    X_clean = X[:, keep_mask]
    names_clean = [n for n, k in zip(names, keep_mask) if k]

    log.info(f"Kept {X_clean.shape[1]} / {X.shape[1]} features")

    # impute NaNs
    for col in range(X_clean.shape[1]):
        mask = np.isnan(X_clean[:, col])
        if mask.any():
            X_clean[mask, col] = np.nanmedian(X_clean[:, col])

    # temporal split (80/20)
    split_idx = int(len(timestamps) * 0.8)
    X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    pos_train = y_train.sum()
    pos_test = y_test.sum()
    log.info(f"Train: {len(y_train)} samples, {pos_train} positive ({pos_train/len(y_train)*100:.2f}%)")
    log.info(f"Test:  {len(y_test)} samples, {pos_test} positive ({pos_test/len(y_test)*100:.2f}%)")

    # handle class imbalance
    scale_pos = (len(y_train) - pos_train) / max(pos_train, 1)

    try:
        import xgboost as xgb
    except ImportError:
        log.error("XGBoost not installed. Run: pip install xgboost")
        return

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=scale_pos,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    from sklearn.metrics import (
        classification_report, roc_auc_score, average_precision_score,
        precision_recall_curve
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    log.info("\nClassification Report:")
    log.info("\n" + classification_report(y_test, y_pred, labels=[0, 1], target_names=["No quake", "Quake"], zero_division=0))

    if pos_test > 0:
        auc_roc = roc_auc_score(y_test, y_prob)
        auc_pr = average_precision_score(y_test, y_prob)
        log.info(f"AUC-ROC: {auc_roc:.4f}")
        log.info(f"AUC-PR:  {auc_pr:.4f}")
        log.info(f"Baseline (random) AUC-PR: {pos_test/len(y_test):.4f}")
    else:
        auc_roc = 0
        auc_pr = 0

    # feature importance
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:20]
    log.info("\nTop 20 features:")
    for rank, idx in enumerate(top_idx, 1):
        log.info(f"  {rank:2d}. {names_clean[idx]}: {importance[idx]:.4f}")

    model_path = DATA_DIR / "model_binary.json"
    model.save_model(str(model_path))
    log.info(f"Model saved to {model_path}")

    results = {
        "task": "binary_classification",
        "target": f"M{target_mag}+ within {window_hours}h",
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_positive": int(pos_train),
        "test_positive": int(pos_test),
        "n_features": int(X_clean.shape[1]),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "top_features": [
            {"name": names_clean[idx], "importance": float(importance[idx])}
            for idx in top_idx
        ],
    }

    results_path = DATA_DIR / "results_binary.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("magnitude", "both"):
        train_signal_correlation()

    if mode in ("binary", "both"):
        train_binary_predictor(target_mag=5.0, window_hours=24)
