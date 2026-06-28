"""SeismicLab — Feature alignment layer.

Builds ML training matrices by aligning multi-signal time series
to hourly windows around earthquake events.

For each earthquake (M4.0+), we look at the state of all signals
in configurable windows before the event (e.g. 1h, 2h, 6h, 12h, 24h)
and compute statistical features (mean, std, min, max, slope, delta).

The output is a flat feature matrix suitable for XGBoost / random forest.
"""

import sqlite3
import json
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass

from .store import QuakeStore

log = logging.getLogger("seismiclab.features")
UTC = timezone.utc

LOOKBACK_HOURS = [1, 2, 4, 6, 12, 24, 48]

SIGNAL_METRICS = [
    # (source_filter, metric, description)
    ("omni_historical", "imf_bz_gsm", "IMF Bz (GSM)"),
    ("omni_historical", "imf_bt", "IMF |B| total"),
    ("omni_historical", "solar_wind_speed", "Solar wind speed"),
    ("omni_historical", "solar_wind_density", "Solar wind density"),
    ("omni_historical", "solar_wind_temp", "Solar wind temperature"),
    ("omni_historical", "kp_index", "Kp geomagnetic index"),
    ("omni_historical", "dst_index", "Dst storm index"),
    ("gfz_kp", "kp_index", "GFZ definitive Kp"),
    ("noaa_swpc", "kp_index", "NOAA realtime Kp"),
    ("noaa_swpc", "dst_index", "NOAA realtime Dst"),
    ("noaa_swpc", "solar_wind_speed", "NOAA realtime SW speed"),
    ("noaa_swpc", "solar_wind_density", "NOAA realtime SW density"),
    ("noaa_swpc", "imf_bz", "NOAA realtime IMF Bz"),
    ("noaa_swpc", "imf_bt", "NOAA realtime IMF Bt"),
    ("noaa_tides", "water_level", "Tide gauge water level"),
    ("usgs_magnetometer", "mag_x", "Magnetometer X"),
    ("usgs_magnetometer", "mag_y", "Magnetometer Y"),
    ("usgs_magnetometer", "mag_z", "Magnetometer Z"),
    ("usgs_magnetometer", "mag_h", "Magnetometer H"),
    ("usgs_groundwater", "water_level_depth", "Groundwater depth"),
    ("noaa_goes", "proton_flux", "GOES proton flux"),
    ("noaa_goes", "electron_flux", "GOES electron flux"),
    ("nws_weather", "barometric_pressure", "Barometric pressure"),
    ("goes_magnetometer", "goes_mag_hp", "GOES satellite mag Hp"),
    ("goes_magnetometer", "goes_mag_he", "GOES satellite mag He"),
    ("goes_magnetometer", "goes_mag_hn", "GOES satellite mag Hn"),
    ("goes_magnetometer", "goes_mag_total", "GOES satellite mag total"),
    ("intermagnet", "imag_x", "INTERMAGNET X component"),
    ("intermagnet", "imag_y", "INTERMAGNET Y component"),
    ("intermagnet", "imag_z", "INTERMAGNET Z component"),
    ("intermagnet", "imag_f", "INTERMAGNET F total"),
]


def _compute_window_features(values: list[float]) -> dict:
    """Compute statistical features for a window of values."""
    if not values or len(values) < 2:
        return {
            "mean": values[0] if values else np.nan,
            "std": 0.0 if values else np.nan,
            "min": values[0] if values else np.nan,
            "max": values[0] if values else np.nan,
            "range": 0.0 if values else np.nan,
            "slope": np.nan,
            "delta": np.nan,
            "count": len(values),
        }

    arr = np.array(values, dtype=np.float64)
    n = len(arr)

    # linear slope via least squares
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = ((x - x_mean) ** 2).sum()
    slope = ((x - x_mean) * (arr - y_mean)).sum() / denom if denom > 0 else 0.0

    return {
        "mean": float(y_mean),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "slope": float(slope),
        "delta": float(arr[-1] - arr[0]),
        "count": n,
    }


def build_earthquake_features(store: QuakeStore, min_mag: float = 4.0,
                               start: str = None, end: str = None) -> tuple:
    """Build feature matrix aligned to earthquake events.

    Returns (feature_names, X, y, eq_info) where:
      - feature_names: list of str
      - X: numpy array (n_events, n_features)
      - y: numpy array (n_events,) — magnitude
      - eq_info: list of dicts with earthquake metadata
    """
    earthquakes = store.query_earthquakes(start=start, end=end, min_mag=min_mag, limit=50000)
    log.info(f"Building features for {len(earthquakes)} earthquakes (M{min_mag}+)")

    if not earthquakes:
        return [], np.array([]), np.array([]), []

    # determine which signals actually have data
    available_signals = _find_available_signals(store)
    log.info(f"Available signals: {len(available_signals)}")

    # build feature names
    feature_names = []
    # earthquake properties
    feature_names.extend(["eq_lat", "eq_lon", "eq_depth_km", "eq_hour_utc", "eq_day_of_year"])

    # signal features: {source}_{metric}_{window}h_{stat}
    for source, metric, _ in available_signals:
        key = f"{source}__{metric}".replace(".", "_")
        for hours in LOOKBACK_HOURS:
            for stat in ["mean", "std", "min", "max", "range", "slope", "delta"]:
                feature_names.append(f"{key}__{hours}h__{stat}")

    log.info(f"Feature vector size: {len(feature_names)}")

    # build rows
    X_rows = []
    y_values = []
    eq_infos = []
    db_path = store.db_path

    for i, eq in enumerate(earthquakes):
        if i % 100 == 0 and i > 0:
            log.info(f"Processing earthquake {i}/{len(earthquakes)}")

        eq_time = eq["timestamp"]
        try:
            eq_dt = datetime.fromisoformat(eq_time)
        except ValueError:
            continue

        row = []

        # earthquake intrinsic features
        row.append(eq.get("lat", 0.0))
        row.append(eq.get("lon", 0.0))
        row.append(eq.get("depth_km") or 0.0)
        row.append(eq_dt.hour)
        row.append(eq_dt.timetuple().tm_yday)

        # signal features
        for source, metric, _ in available_signals:
            signal_features = _extract_signal_features(
                db_path, source, metric, eq_dt
            )
            for hours in LOOKBACK_HOURS:
                feats = signal_features.get(hours, {})
                for stat in ["mean", "std", "min", "max", "range", "slope", "delta"]:
                    row.append(feats.get(stat, np.nan))

        X_rows.append(row)
        y_values.append(eq["magnitude"])
        eq_infos.append({
            "id": eq["id"],
            "timestamp": eq_time,
            "magnitude": eq["magnitude"],
            "lat": eq.get("lat"),
            "lon": eq.get("lon"),
            "place": eq.get("place", ""),
        })

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_values, dtype=np.float64)

    log.info(f"Feature matrix: {X.shape[0]} events × {X.shape[1]} features")

    # report coverage
    nan_pct = np.isnan(X).sum() / X.size * 100
    log.info(f"NaN coverage: {nan_pct:.1f}%")

    return feature_names, X, y, eq_infos


def _find_available_signals(store: QuakeStore) -> list:
    """Find which signal metrics actually have data in the DB."""
    available = []
    with store._conn() as conn:
        for source, metric, desc in SIGNAL_METRICS:
            count = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE source = ? AND metric = ? LIMIT 1",
                (source, metric)
            ).fetchone()[0]
            if count > 0:
                available.append((source, metric, desc))
    return available


FILL_VALUE_FILTERS = {
    ("omni_historical", "solar_wind_speed"): lambda v: 100 < v < 900,
    ("omni_historical", "solar_wind_temp"): lambda v: 1000 < v < 5000000,
    ("omni_historical", "solar_wind_density"): lambda v: 0 < v < 200,
    ("omni_historical", "dst_index"): lambda v: -600 < v < 100,
    ("omni_historical", "kp_index"): lambda v: 0 <= v < 10,
    ("omni_historical", "imf_bt"): lambda v: 0 <= v < 100,
    ("omni_historical", "imf_bz_gsm"): lambda v: abs(v) < 50,
    ("omni_historical", "imf_bz_gse"): lambda v: abs(v) < 50,
    ("noaa_swpc", "dst_index"): lambda v: -600 < v < 100,
}


def _extract_signal_features(db_path: Path, source: str, metric: str,
                              eq_dt: datetime) -> dict:
    """Extract windowed features for one signal around one earthquake."""
    max_lookback = max(LOOKBACK_HOURS)
    start_dt = eq_dt - timedelta(hours=max_lookback)

    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        rows = conn.execute(
            "SELECT timestamp, value FROM samples "
            "WHERE source = ? AND metric = ? AND timestamp >= ? AND timestamp < ? "
            "ORDER BY timestamp",
            (source, metric, start_dt.isoformat(), eq_dt.isoformat())
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {}

    val_filter = FILL_VALUE_FILTERS.get((source, metric))

    points = []
    for ts_str, val in rows:
        try:
            ts = datetime.fromisoformat(ts_str)
            if val_filter and not val_filter(val):
                continue
            points.append((ts, val))
        except ValueError:
            continue

    results = {}
    for hours in LOOKBACK_HOURS:
        cutoff = eq_dt - timedelta(hours=hours)
        window_vals = [v for t, v in points if t >= cutoff]
        results[hours] = _compute_window_features(window_vals)

    return results


def _extract_seismic_rate_features(db_path: Path, eq_dt: datetime,
                                    lat: float, lon: float,
                                    radius_deg: float = 5.0) -> dict:
    """Count foreshock activity in windows before an event.

    This is the strongest known precursor signal — elevated small quake
    rates often precede larger events. We compute counts and rates
    at multiple magnitude thresholds and time windows, both globally
    and within a radius of the target location.
    """
    windows = [1, 2, 6, 12, 24, 48, 72, 168]  # hours
    mag_thresholds = [2.5, 3.0, 3.5, 4.0]

    conn = sqlite3.connect(str(db_path), timeout=10)
    features = {}

    try:
        max_lookback = max(windows)
        start_dt = eq_dt - timedelta(hours=max_lookback)

        # global seismic rate
        rows = conn.execute(
            "SELECT timestamp, magnitude, lat, lon FROM earthquakes "
            "WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
            (start_dt.isoformat(), eq_dt.isoformat())
        ).fetchall()

        for window_h in windows:
            cutoff = eq_dt - timedelta(hours=window_h)

            for min_mag in mag_thresholds:
                # global count
                global_count = sum(1 for ts, mag, _, _ in rows
                                   if ts >= cutoff.isoformat() and mag >= min_mag)
                features[f"seismic_global_M{min_mag}_{window_h}h_count"] = global_count
                features[f"seismic_global_M{min_mag}_{window_h}h_rate"] = global_count / window_h

                # regional count (within radius_deg)
                if lat is not None and lon is not None:
                    regional_count = sum(
                        1 for ts, mag, rlat, rlon in rows
                        if ts >= cutoff.isoformat() and mag >= min_mag
                        and rlat is not None and rlon is not None
                        and abs(rlat - lat) <= radius_deg
                        and abs(rlon - lon) <= radius_deg
                    )
                    features[f"seismic_regional_M{min_mag}_{window_h}h_count"] = regional_count
                    features[f"seismic_regional_M{min_mag}_{window_h}h_rate"] = regional_count / window_h

            # max magnitude in window (any threshold)
            window_mags = [mag for ts, mag, _, _ in rows if ts >= cutoff.isoformat()]
            features[f"seismic_global_max_mag_{window_h}h"] = max(window_mags) if window_mags else 0.0
            features[f"seismic_global_mean_mag_{window_h}h"] = (
                sum(window_mags) / len(window_mags) if window_mags else 0.0
            )

            # b-value estimate (Gutenberg-Richter slope) for 48h+ windows
            if window_h >= 48 and len(window_mags) >= 10:
                m_arr = np.array(window_mags)
                b_value = np.log10(np.e) / (m_arr.mean() - m_arr.min() + 0.01)
                features[f"seismic_b_value_{window_h}h"] = b_value

    finally:
        conn.close()

    return features


SEISMIC_RATE_FEATURE_NAMES = []
for window_h in [1, 2, 6, 12, 24, 48, 72, 168]:
    for min_mag in [2.5, 3.0, 3.5, 4.0]:
        SEISMIC_RATE_FEATURE_NAMES.append(f"seismic_global_M{min_mag}_{window_h}h_count")
        SEISMIC_RATE_FEATURE_NAMES.append(f"seismic_global_M{min_mag}_{window_h}h_rate")
        SEISMIC_RATE_FEATURE_NAMES.append(f"seismic_regional_M{min_mag}_{window_h}h_count")
        SEISMIC_RATE_FEATURE_NAMES.append(f"seismic_regional_M{min_mag}_{window_h}h_rate")
    SEISMIC_RATE_FEATURE_NAMES.append(f"seismic_global_max_mag_{window_h}h")
    SEISMIC_RATE_FEATURE_NAMES.append(f"seismic_global_mean_mag_{window_h}h")
for window_h in [48, 72, 168]:
    SEISMIC_RATE_FEATURE_NAMES.append(f"seismic_b_value_{window_h}h")


def build_global_prediction_features(store: QuakeStore, target_mag: float = 5.0,
                                      window_hours: int = 24,
                                      sample_interval_hours: int = 6,
                                      start: str = None, end: str = None) -> tuple:
    """Build features for global prediction with foreshock + space weather signals.

    Approach: for each M5.0+ earthquake, create a positive sample from
    the time just before it. Create negative samples from random quiet
    periods (no M{target_mag}+ within window_hours).

    Includes:
    - Space weather features (OMNI: Bz, Bt, solar wind, Kp, Dst)
    - Foreshock features (seismic rates at multiple mag thresholds/windows)
    - Gutenberg-Richter b-value estimates
    - Temporal features (hour, day of year)
    - Location features (lat, lon, depth for event-centered samples)
    """
    # Get target earthquakes
    all_eq = store.query_earthquakes(
        start=start, end=end, min_mag=target_mag, limit=50000
    )
    log.info(f"Global prediction: {len(all_eq)} target earthquakes (M{target_mag}+)")

    if not all_eq:
        return [], np.array([]), np.array([]), []

    # Parse and sort by time
    eq_events = []
    for eq in all_eq:
        try:
            dt = datetime.fromisoformat(eq["timestamp"])
            eq_events.append((dt, eq))
        except ValueError:
            continue
    eq_events.sort(key=lambda x: x[0])

    available_signals = _find_available_signals(store)
    log.info(f"Available signals: {len(available_signals)}")

    # Build feature names (no depth — unknown at prediction time)
    feature_names = [
        "eq_lat", "eq_lon",
        "hour_utc", "day_of_year", "day_of_week", "month",
    ]

    # Seismic rate features
    feature_names.extend(SEISMIC_RATE_FEATURE_NAMES)

    # Space weather features (wider windows for prediction)
    sw_windows = [1, 2, 4, 6, 12, 24, 48]
    sw_stats = ["mean", "std", "min", "max", "slope", "delta"]
    for source, metric, _ in available_signals:
        key = f"{source}__{metric}".replace(".", "_")
        for hours in sw_windows:
            for stat in sw_stats:
                feature_names.append(f"{key}__{hours}h__{stat}")

    log.info(f"Feature vector: {len(feature_names)} features")

    db_path = store.db_path
    X_rows = []
    y_labels = []
    sample_info = []

    # Create positive samples: 1 sample per target earthquake
    # Use the time just before the earthquake
    log.info("Building positive samples (pre-earthquake conditions)...")
    for i, (eq_dt, eq) in enumerate(eq_events):
        if i % 200 == 0 and i > 0:
            log.info(f"  Positive samples: {i}/{len(eq_events)}")

        lat = eq.get("lat", 0.0)
        lon = eq.get("lon", 0.0)

        row = [lat, lon, eq_dt.hour,
               eq_dt.timetuple().tm_yday, eq_dt.weekday(), eq_dt.month]

        # seismic rate features
        seismic_feats = _extract_seismic_rate_features(db_path, eq_dt, lat, lon)
        for fname in SEISMIC_RATE_FEATURE_NAMES:
            row.append(seismic_feats.get(fname, 0.0))

        # space weather features
        for source, metric, _ in available_signals:
            sig_feats = _extract_signal_features(db_path, source, metric, eq_dt)
            for hours in sw_windows:
                f = sig_feats.get(hours, {})
                for stat in sw_stats:
                    row.append(f.get(stat, np.nan))

        X_rows.append(row)
        y_labels.append(1)
        sample_info.append({
            "id": eq["id"], "timestamp": eq["timestamp"],
            "magnitude": eq["magnitude"], "lat": lat, "lon": lon,
            "type": "positive",
        })

    # Create negative samples: for each positive sample's location+time,
    # pick a nearby time/location where no M{target}+ occurred regionally.
    # Global quiet periods don't exist at M5.0+ (~5/day worldwide),
    # so negatives must be regional: no target-mag earthquake within
    # window_hours AND within radius_deg of the sample point.
    log.info("Building negative samples (regionally quiet periods)...")
    neg_count = 0
    neg_target = len(eq_events) * 3  # 3:1 negative:positive ratio
    radius_deg = 10.0  # ~1000 km

    # Build sorted list of earthquake times + locations for fast regional check
    eq_time_loc = [(dt, eq.get("lat", 0.0), eq.get("lon", 0.0))
                   for dt, eq in eq_events]

    # Strategy: iterate through time, using seismically active regions
    # (from positive samples) but offset in time to find quiet windows
    import bisect
    eq_times_sorted = [dt for dt, _ in eq_events]

    t_start = eq_events[0][0] + timedelta(days=7)
    t_end = eq_events[-1][0] - timedelta(days=1)

    sample_dt = t_start
    pos_idx = 0  # cycle through positive sample locations
    while sample_dt < t_end and neg_count < neg_target:
        # Use location from a positive sample (cycle through them)
        _, ref_eq = eq_events[pos_idx % len(eq_events)]
        ref_lat = ref_eq.get("lat", 0.0)
        ref_lon = ref_eq.get("lon", 0.0)
        pos_idx += 1

        # Check if any target earthquake is within window_hours AND radius
        too_close = False
        # Binary search for nearby-in-time earthquakes
        lo_dt = sample_dt - timedelta(hours=window_hours)
        hi_dt = sample_dt + timedelta(hours=window_hours)
        lo_idx = bisect.bisect_left(eq_times_sorted, lo_dt)
        hi_idx = bisect.bisect_right(eq_times_sorted, hi_dt)

        for j in range(lo_idx, min(hi_idx, len(eq_time_loc))):
            _, elat, elon = eq_time_loc[j]
            if (abs(elat - ref_lat) <= radius_deg
                    and abs(elon - ref_lon) <= radius_deg):
                too_close = True
                break

        if not too_close:
            row = [ref_lat, ref_lon, sample_dt.hour,
                   sample_dt.timetuple().tm_yday, sample_dt.weekday(), sample_dt.month]

            seismic_feats = _extract_seismic_rate_features(
                db_path, sample_dt, ref_lat, ref_lon
            )
            for fname in SEISMIC_RATE_FEATURE_NAMES:
                row.append(seismic_feats.get(fname, 0.0))

            for source, metric, _ in available_signals:
                sig_feats = _extract_signal_features(db_path, source, metric, sample_dt)
                for hours in sw_windows:
                    f = sig_feats.get(hours, {})
                    for stat in sw_stats:
                        row.append(f.get(stat, np.nan))

            X_rows.append(row)
            y_labels.append(0)
            sample_info.append({
                "timestamp": sample_dt.isoformat(), "type": "negative",
                "lat": ref_lat, "lon": ref_lon,
            })
            neg_count += 1

            if neg_count % 500 == 0:
                log.info(f"  Negative samples: {neg_count}/{neg_target}")

        sample_dt += timedelta(hours=sample_interval_hours)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int32)

    pos = y.sum()
    neg = len(y) - pos
    log.info(f"Dataset: {len(y)} samples — {pos} positive, {neg} negative "
             f"({pos/len(y)*100:.1f}% / {neg/len(y)*100:.1f}%)")

    return feature_names, X, y, sample_info


def save_dataset(feature_names: list, X: np.ndarray, y: np.ndarray,
                 eq_info: list, path: Path):
    """Save feature matrix to npz for ML training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        feature_names=np.array(feature_names),
        X=X,
        y=y,
        eq_info=np.array(json.dumps(eq_info)),
    )
    log.info(f"Saved dataset to {path} ({X.shape[0]} × {X.shape[1]})")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    store = QuakeStore()
    mode = sys.argv[1] if len(sys.argv) > 1 else "magnitude"

    if mode == "magnitude":
        names, X, y, info = build_earthquake_features(store, min_mag=4.0)
        save_dataset(names, X, y, info,
                     Path(__file__).parent.parent / "data" / "features_magnitude.npz")
    elif mode == "binary":
        names, X, y, ts = build_binary_classification_features(
            store, target_mag=4.5, window_hours=6
        )
        save_dataset(names, X, y, ts,
                     Path(__file__).parent.parent / "data" / "features_binary.npz")
