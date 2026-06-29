"""QuakeWatch — ST-GNN Data Pipeline

Builds spatio-temporal graph datasets for earthquake prediction.
Graph structure:
  - 24 seismic station nodes, edges by geographic proximity
  - Node features: 5-min seismic metrics + rolling statistics
  - Global features: solar wind, geomagnetic, tidal
  - Target: M5+ event within 1 hour in any monitored zone

Temporal resolution: 5-minute windows
Lookback: 72 steps (6 hours)
Prediction horizon: 12 steps (1 hour)
"""

import os
import sys
import math
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UTC = timezone.utc
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "quakewatch.db")

STATIONS = [
    {"key": "IU.COLA", "lat": 64.87, "lon": -147.86},
    {"key": "IU.COR",  "lat": 44.59, "lon": -123.30},
    {"key": "IU.TUC",  "lat": 32.31, "lon": -110.78},
    {"key": "IU.ANMO", "lat": 34.95, "lon": -106.46},
    {"key": "IU.TEIG", "lat": 20.23, "lon": -88.28},
    {"key": "IU.SJG",  "lat": 18.11, "lon": -66.15},
    {"key": "II.JTS",  "lat": 10.29, "lon": -84.95},
    {"key": "II.NNA",  "lat": -11.99, "lon": -76.84},
    {"key": "IU.LCO",  "lat": -29.01, "lon": -70.70},
    {"key": "II.ESK",  "lat": 55.32, "lon": -3.21},
    {"key": "IU.ANTO", "lat": 39.87, "lon": 30.50},
    {"key": "IU.GNI",  "lat": 40.15, "lon": 44.74},
    {"key": "II.MBAR", "lat": -0.60, "lon": 30.74},
    {"key": "II.AAK",  "lat": 42.64, "lon": 74.49},
    {"key": "II.DGAR", "lat": -7.41, "lon": 72.45},
    {"key": "IU.ULN",  "lat": 47.87, "lon": 107.05},
    {"key": "IU.INCN", "lat": 37.48, "lon": 126.62},
    {"key": "IU.MAJO", "lat": 36.54, "lon": 138.20},
    {"key": "II.ERM",  "lat": 42.02, "lon": 143.16},
    {"key": "IU.DAV",  "lat": 7.07, "lon": 125.58},
    {"key": "II.KAPI", "lat": -5.01, "lon": 119.75},
    {"key": "IU.GUMO", "lat": 13.59, "lon": 144.87},
    {"key": "IU.CTAO", "lat": -20.09, "lon": 146.25},
    {"key": "II.TAU",  "lat": -42.91, "lon": 147.32},
]

NUM_STATIONS = len(STATIONS)
STATION_KEYS = [s["key"] for s in STATIONS]

LOOKBACK_STEPS = 72       # 6 hours at 5-min resolution
HORIZON_STEPS = 12        # 1 hour ahead
STEP_MINUTES = 5
EDGE_DISTANCE_KM = 4000   # connect stations within this radius
MIN_MAG = 5.0

ZONES = [
    {"id": "socal",        "lat_range": [31, 36],  "lon_range": [-121, -114]},
    {"id": "norcal",       "lat_range": [36, 50],  "lon_range": [-131, -119]},
    {"id": "alaska",       "lat_range": [50, 65],  "lon_range": [-180, -130]},
    {"id": "japan",        "lat_range": [28, 46],  "lon_range": [128, 148]},
    {"id": "indonesia",    "lat_range": [-12, 8],  "lon_range": [94, 136]},
    {"id": "chile_peru",   "lat_range": [-46, 2],  "lon_range": [-82, -65]},
    {"id": "mediterranean","lat_range": [33, 42],  "lon_range": [-6, 45]},
    {"id": "mexico_ca",    "lat_range": [7, 25],   "lon_range": [-115, -77]},
    {"id": "himalaya",     "lat_range": [24, 40],  "lon_range": [65, 100]},
    {"id": "caribbean",    "lat_range": [10, 22],  "lon_range": [-85, -60]},
    {"id": "iceland",      "lat_range": [55, 68],  "lon_range": [-25, -12]},
    {"id": "philippines",  "lat_range": [4, 26],   "lon_range": [118, 128]},
]
NUM_ZONES = len(ZONES)

NODE_FEATURES = ["amp_min", "amp_max", "amp_mean", "sta_lta_ratio", "triggered",
                 "amp_range", "amp_rms_1h", "sta_lta_max_1h", "trigger_count_1h",
                 "amp_trend_1h", "noise_floor_delta"]
NUM_NODE_FEATURES = len(NODE_FEATURES)

GLOBAL_FEATURES = ["kp_index", "dst_index", "imf_bz", "solar_wind_speed",
                    "solar_wind_density", "xray_flux", "neutron_count", "ief"]
NUM_GLOBAL_FEATURES = len(GLOBAL_FEATURES)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def build_edge_index(distance_km=EDGE_DISTANCE_KM):
    """Build adjacency based on geographic distance. Returns (2, E) int tensor."""
    src, dst = [], []
    for i, si in enumerate(STATIONS):
        for j, sj in enumerate(STATIONS):
            if i == j:
                continue
            d = haversine_km(si["lat"], si["lon"], sj["lat"], sj["lon"])
            if d <= distance_km:
                src.append(i)
                dst.append(j)
    return np.array([src, dst], dtype=np.int64)


def build_edge_weights(edge_index):
    """Inverse-distance edge weights, normalized per node."""
    weights = []
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        d = haversine_km(STATIONS[i]["lat"], STATIONS[i]["lon"],
                         STATIONS[j]["lat"], STATIONS[j]["lon"])
        weights.append(1.0 / max(d, 100))
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.max()
    return weights


def load_station_metrics(conn, start, end):
    """Load all station metrics into a DataFrame aligned to 5-minute bins."""
    rows = conn.execute("""
        SELECT timestamp, station, amp_min, amp_max, amp_mean,
               sta_lta_ratio, triggered
        FROM station_metrics
        WHERE timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (start.isoformat(), end.isoformat())).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "station", "amp_min", "amp_max",
                                      "amp_mean", "sta_lta_ratio", "triggered"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["time_bin"] = df["timestamp"].dt.floor(f"{STEP_MINUTES}min")

    agg = df.groupby(["time_bin", "station"]).agg({
        "amp_min": "min",
        "amp_max": "max",
        "amp_mean": "mean",
        "sta_lta_ratio": "max",
        "triggered": "max",
    }).reset_index()

    return agg


def compute_rolling_features(station_df):
    """Add rolling 1-hour features for a single station's time series."""
    df = station_df.sort_values("time_bin").copy()
    window = 12  # 12 × 5min = 1 hour

    df["amp_range"] = df["amp_max"] - df["amp_min"]

    df["amp_rms_1h"] = df["amp_mean"].rolling(window, min_periods=1).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True)
    df["sta_lta_max_1h"] = df["sta_lta_ratio"].rolling(window, min_periods=1).max()
    df["trigger_count_1h"] = df["triggered"].rolling(window, min_periods=1).sum()

    df["amp_trend_1h"] = df["amp_mean"].rolling(window, min_periods=2).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)

    baseline = df["amp_mean"].rolling(window * 6, min_periods=window).mean()
    df["noise_floor_delta"] = df["amp_mean"] - baseline

    df = df.fillna(0)
    return df


def load_global_features(conn, start, end):
    """Load global geophysical features aligned to 5-minute bins."""
    time_bins = pd.date_range(start, end, freq=f"{STEP_MINUTES}min", tz=UTC)
    gf = pd.DataFrame({"time_bin": time_bins})

    source_map = {
        "kp_index": ("kp_index", "gfz_kp"),
        "dst_index": ("dst_index", "kyoto_dst"),
        "imf_bz": ("imf_bz", "noaa_swpc"),
        "solar_wind_speed": ("solar_wind_speed", "noaa_swpc"),
        "solar_wind_density": ("solar_wind_density", "noaa_swpc"),
        "xray_flux": ("xray_flux", "goes_xray"),
        "neutron_count": ("neutron_count", "nmdb_cosmic"),
        "ief": ("ief", "derived_ief"),
    }

    for feat_name, (metric, source) in source_map.items():
        rows = conn.execute("""
            SELECT timestamp, value FROM samples
            WHERE metric = ? AND source = ?
            AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
        """, (metric, source, start.isoformat(), end.isoformat())).fetchall()

        if not rows:
            gf[feat_name] = 0.0
            continue

        sdf = pd.DataFrame(rows, columns=["timestamp", feat_name])
        sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True)
        sdf["time_bin"] = sdf["timestamp"].dt.floor(f"{STEP_MINUTES}min")
        sdf = sdf.groupby("time_bin").agg({feat_name: "mean"}).reset_index()
        gf = gf.merge(sdf, on="time_bin", how="left")
        gf[feat_name] = gf[feat_name].ffill().fillna(0)

    return gf


def load_earthquake_targets(conn, start, end, min_mag=MIN_MAG):
    """Load earthquake events and create per-zone binary targets at 5-min resolution."""
    time_bins = pd.date_range(start, end, freq=f"{STEP_MINUTES}min", tz=UTC)
    targets = pd.DataFrame({"time_bin": time_bins})

    quakes = conn.execute("""
        SELECT timestamp, lat, lon, magnitude FROM earthquakes
        WHERE magnitude >= ? AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (min_mag, start.isoformat(), end.isoformat())).fetchall()

    for zone in ZONES:
        col = f"target_{zone['id']}"
        targets[col] = 0

    for q in quakes:
        try:
            qt = pd.Timestamp(q[0], tz=UTC)
        except Exception:
            continue
        qlat, qlon, qmag = q[1], q[2], q[3]

        for zone in ZONES:
            if (zone["lat_range"][0] <= qlat <= zone["lat_range"][1] and
                zone["lon_range"][0] <= qlon <= zone["lon_range"][1]):
                event_bin = qt.floor(f"{STEP_MINUTES}min")
                horizon_start = event_bin - timedelta(minutes=HORIZON_STEPS * STEP_MINUTES)
                mask = (targets["time_bin"] >= horizon_start) & (targets["time_bin"] < event_bin)
                col = f"target_{zone['id']}"
                targets.loc[mask, col] = 1

    return targets


def build_dataset(start=None, end=None, save_path=None):
    """Build the full ST-GNN dataset.

    Returns dict with:
        node_features:  (T, N, F_node) — per-station features over time
        global_features: (T, F_global) — global geophysical features
        targets:        (T, Z) — per-zone binary targets
        edge_index:     (2, E) — graph adjacency
        edge_weight:    (E,) — edge weights
        time_bins:      (T,) — timestamps
        station_keys:   list of station keys
        zone_ids:       list of zone ids
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    if start is None:
        row = conn.execute("SELECT MIN(timestamp) FROM station_metrics").fetchone()
        start = pd.Timestamp(row[0], tz=UTC).floor(f"{STEP_MINUTES}min")
    else:
        start = pd.Timestamp(start, tz=UTC)

    if end is None:
        row = conn.execute("SELECT MAX(timestamp) FROM station_metrics").fetchone()
        end = pd.Timestamp(row[0], tz=UTC).floor(f"{STEP_MINUTES}min")
    else:
        end = pd.Timestamp(end, tz=UTC)

    print(f"  Building dataset: {start} → {end}")
    print(f"  Stations: {NUM_STATIONS}, Zones: {NUM_ZONES}")

    # 1. Station metrics
    print("  Loading station metrics...")
    metrics_df = load_station_metrics(conn, start, end)
    if metrics_df is None or metrics_df.empty:
        raise ValueError("No station metrics found in the specified range")

    time_bins = pd.date_range(start, end, freq=f"{STEP_MINUTES}min", tz=UTC)
    T = len(time_bins)
    print(f"  Time steps: {T:,} ({T * STEP_MINUTES / 60:.0f} hours)")

    # 2. Build per-station feature matrices with rolling features
    print("  Computing per-station features...")
    node_data = np.zeros((T, NUM_STATIONS, NUM_NODE_FEATURES), dtype=np.float32)

    for si, skey in enumerate(STATION_KEYS):
        sdf = metrics_df[metrics_df["station"] == skey].copy()
        if sdf.empty:
            continue
        sdf = compute_rolling_features(sdf)

        merged = pd.DataFrame({"time_bin": time_bins}).merge(sdf, on="time_bin", how="left")
        merged = merged.ffill().fillna(0)

        for fi, feat in enumerate(NODE_FEATURES):
            if feat in merged.columns:
                node_data[:, si, fi] = merged[feat].values[:T]

    # 3. Global features
    print("  Loading global features...")
    gf = load_global_features(conn, start, end)
    global_data = np.zeros((T, NUM_GLOBAL_FEATURES), dtype=np.float32)
    for fi, feat in enumerate(GLOBAL_FEATURES):
        if feat in gf.columns:
            vals = gf[feat].values[:T]
            global_data[:len(vals), fi] = vals

    # 4. Targets
    print("  Building earthquake targets...")
    targets_df = load_earthquake_targets(conn, start, end)
    target_data = np.zeros((T, NUM_ZONES), dtype=np.float32)
    for zi, zone in enumerate(ZONES):
        col = f"target_{zone['id']}"
        if col in targets_df.columns:
            vals = targets_df[col].values[:T]
            target_data[:len(vals), zi] = vals

    conn.close()

    # 5. Graph structure
    edge_index = build_edge_index()
    edge_weight = build_edge_weights(edge_index)

    # 6. Normalize features (per-feature z-score)
    print("  Normalizing features...")
    node_mean = node_data.mean(axis=(0, 1), keepdims=True)
    node_std = node_data.std(axis=(0, 1), keepdims=True)
    node_std[node_std < 1e-8] = 1.0
    node_data = (node_data - node_mean) / node_std

    global_mean = global_data.mean(axis=0, keepdims=True)
    global_std = global_data.std(axis=0, keepdims=True)
    global_std[global_std < 1e-8] = 1.0
    global_data = (global_data - global_mean) / global_std

    pos_rate = target_data.mean()
    pos_per_zone = target_data.mean(axis=0)
    print(f"  Overall positive rate: {pos_rate:.3f}")
    for zi, zone in enumerate(ZONES):
        print(f"    {zone['id']:20s} {pos_per_zone[zi]:.4f}")

    dataset = {
        "node_features": node_data,
        "global_features": global_data,
        "targets": target_data,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "time_bins": time_bins.values,
        "station_keys": STATION_KEYS,
        "zone_ids": [z["id"] for z in ZONES],
        "norm_params": {
            "node_mean": node_mean, "node_std": node_std,
            "global_mean": global_mean, "global_std": global_std,
        },
    }

    if save_path:
        np.savez_compressed(save_path, **{
            k: v for k, v in dataset.items()
            if isinstance(v, np.ndarray)
        })
        print(f"  Saved: {save_path}")

    return dataset


def create_sequences(dataset, lookback=LOOKBACK_STEPS, horizon=HORIZON_STEPS):
    """Slice the dataset into (lookback, horizon) sequence pairs.

    Returns:
        X_node:   (S, lookback, N, F_node)
        X_global: (S, lookback, F_global)
        Y:        (S, Z) — binary target: any positive in the horizon window
    """
    node = dataset["node_features"]
    glob = dataset["global_features"]
    tgt = dataset["targets"]
    T = node.shape[0]

    S = T - lookback - horizon + 1
    if S <= 0:
        raise ValueError(f"Not enough timesteps: {T} < {lookback + horizon}")

    X_node = np.zeros((S, lookback, NUM_STATIONS, NUM_NODE_FEATURES), dtype=np.float32)
    X_global = np.zeros((S, lookback, NUM_GLOBAL_FEATURES), dtype=np.float32)
    Y = np.zeros((S, NUM_ZONES), dtype=np.float32)

    for i in range(S):
        X_node[i] = node[i:i + lookback]
        X_global[i] = glob[i:i + lookback]
        Y[i] = tgt[i + lookback:i + lookback + horizon].max(axis=0)

    return X_node, X_global, Y


def temporal_split(n_samples, train_frac=0.7, val_frac=0.15):
    """Temporal train/val/test split indices."""
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_samples),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--save", type=str, default="data/stgnn_dataset.npz")
    args = parser.parse_args()

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.save
    )

    print("=" * 60)
    print("  ST-GNN DATA PIPELINE — QUAKEWATCH")
    print("=" * 60)

    ds = build_dataset(start=args.start, end=args.end, save_path=save_path)

    print(f"\n  Creating sequences (lookback={LOOKBACK_STEPS}, horizon={HORIZON_STEPS})...")
    X_node, X_global, Y = create_sequences(ds)
    print(f"  Sequences: {X_node.shape[0]:,}")
    print(f"  X_node:   {X_node.shape}")
    print(f"  X_global: {X_global.shape}")
    print(f"  Y:        {Y.shape}")
    print(f"  Positive samples: {(Y.max(axis=1) > 0).sum():,} / {Y.shape[0]:,} "
          f"({100 * (Y.max(axis=1) > 0).mean():.1f}%)")
