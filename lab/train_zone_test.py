"""SeismicLab — Zone-Focused ST-GNN Test Training (v3)

v3 fixes from v2:
- Normalization computed from training split only (no data leak)
- Replaced oversampling with class-weighted focal loss (no memorization)
- Removed fully-connected GAT (6 nodes @ 6000km = meaningless graph)
- Station embeddings mean-pooled instead
- Smoother LR schedule (ReduceLROnPlateau, no restarts)
- Much stronger regularization (dropout 0.5, weight_decay 0.01)
- Smaller model (hidden=32, ~45K params vs 143K)
- Larger batch size (256) for smoother gradients

Stations: TEIG, SJG, ANMO, TUC, COR, COLA
DART buoys: 43413, 42407, 41420, 41421, 42409
Target zones: indonesia, japan_kurils, south_america, mexico_ca, himalaya, alaska
"""

import os
import sys
import math
import time
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from datetime import datetime, timedelta, timezone

torch.set_float32_matmul_precision("medium")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UTC = timezone.utc
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "quakewatch.db")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Seismic stations ---
STATIONS = [
    {"key": "IU.TEIG", "lat": 20.23, "lon": -88.28, "name": "Teoloyucan, Mexico"},
    {"key": "IU.SJG",  "lat": 18.11, "lon": -66.15, "name": "San Juan, Puerto Rico"},
    {"key": "IU.ANMO", "lat": 34.95, "lon": -106.46, "name": "Albuquerque, NM"},
    {"key": "IU.TUC",  "lat": 32.31, "lon": -110.78, "name": "Tucson, Arizona"},
    {"key": "IU.COR",  "lat": 44.59, "lon": -123.30, "name": "Corvallis, Oregon"},
    {"key": "IU.COLA", "lat": 64.87, "lon": -147.86, "name": "College, Alaska"},
]
NUM_STATIONS = len(STATIONS)
STATION_KEYS = [s["key"] for s in STATIONS]

# --- DART buoys near zones ---
DART_BUOYS = [
    {"id": "43413", "lat": 10.927, "lon": -100.012},
    {"id": "42407", "lat": 15.276, "lon": -68.191},
    {"id": "41420", "lat": 23.433, "lon": -67.386},
    {"id": "41421", "lat": 23.445, "lon": -63.851},
    {"id": "42409", "lat": 25.797, "lon": -89.288},
]
NUM_DART = len(DART_BUOYS)
DART_FEATURES_PER = 3

# --- Target zones (6 most active, no dateline wrapping) ---
ZONES = [
    {"id": "indonesia",    "lat_range": [-12, 8],   "lon_range": [95, 140]},
    {"id": "japan_kurils", "lat_range": [25, 50],   "lon_range": [128, 155]},
    {"id": "south_america","lat_range": [-60, 7],   "lon_range": [-82, -60]},
    {"id": "mexico_ca",    "lat_range": [7, 25],    "lon_range": [-115, -77]},
    {"id": "himalaya",     "lat_range": [25, 42],   "lon_range": [65, 100]},
    {"id": "alaska",       "lat_range": [50, 72],   "lon_range": [-180, -130]},
]
NUM_ZONES = len(ZONES)

# --- Seismic node features (5 raw + 6 derived) ---
NODE_FEATURES = ["amp_min", "amp_max", "amp_mean", "sta_lta_ratio", "triggered",
                 "amp_range", "amp_rms_1h", "sta_lta_max_1h", "trigger_count_1h",
                 "amp_trend_1h", "noise_floor_delta"]
NUM_NODE_FEATURES = len(NODE_FEATURES)

# --- Global features ---
GLOBAL_METRICS = [
    "solar_wind_speed", "solar_wind_density", "solar_wind_temp",
    "imf_bt", "imf_bz_gsm",
    "kp_index", "dst_index",
    "tidal_potential", "tidal_strain_rate",
    "neutron_count", "ief",
]
NUM_GLOBAL_RAW = len(GLOBAL_METRICS)
NUM_GLOBAL_FEATURES = NUM_GLOBAL_RAW + 2

# --- Training params ---
LOOKBACK_STEPS = 288      # 24 hours
HORIZON_STEPS = 288       # 24 hours
STEP_MINUTES = 5
MIN_MAG = 5.0


def compute_rolling_features(station_df):
    df = station_df.sort_values("time_bin").copy()
    window = 12

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


def build_dataset():
    conn = sqlite3.connect(DB_PATH, timeout=30)

    start = pd.Timestamp("2021-07-01", tz=UTC)
    row = conn.execute("SELECT MAX(timestamp) FROM station_metrics WHERE station IN ({})".format(
        ",".join(f"'{s}'" for s in STATION_KEYS)
    )).fetchone()
    end = pd.Timestamp(row[0], tz=UTC).floor(f"{STEP_MINUTES}min")

    print(f"  Data range: {start} → {end}")
    total_hours = (end - start).total_seconds() / 3600
    print(f"  Duration: {total_hours:.0f} hours ({total_hours/24:.0f} days)")

    time_bins = pd.date_range(start, end, freq=f"{STEP_MINUTES}min", tz=UTC)
    T = len(time_bins)
    print(f"  Time steps: {T:,} ({T * STEP_MINUTES / 60:.0f} hours)")

    # ── 1. Station metrics ──
    print("\n  [1/4] Loading station metrics...")
    rows = conn.execute("""
        SELECT timestamp, station, amp_min, amp_max, amp_mean, sta_lta_ratio, triggered
        FROM station_metrics
        WHERE timestamp >= ? AND timestamp < ?
        AND station IN ({})
        ORDER BY timestamp
    """.format(",".join(f"'{s}'" for s in STATION_KEYS)),
        (start.isoformat(), end.isoformat())).fetchall()

    if not rows:
        raise ValueError("No station metrics found")

    df = pd.DataFrame(rows, columns=["timestamp", "station", "amp_min", "amp_max",
                                      "amp_mean", "sta_lta_ratio", "triggered"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["time_bin"] = df["timestamp"].dt.floor(f"{STEP_MINUTES}min")

    agg = df.groupby(["time_bin", "station"]).agg({
        "amp_min": "min", "amp_max": "max", "amp_mean": "mean",
        "sta_lta_ratio": "max", "triggered": "max",
    }).reset_index()

    print("  Computing per-station features...")
    node_data = np.zeros((T, NUM_STATIONS, NUM_NODE_FEATURES), dtype=np.float32)

    for si, skey in enumerate(STATION_KEYS):
        sdf = agg[agg["station"] == skey].copy()
        if sdf.empty:
            print(f"    WARNING: no data for {skey}")
            continue
        sdf = compute_rolling_features(sdf)
        merged = pd.DataFrame({"time_bin": time_bins}).merge(sdf, on="time_bin", how="left")
        merged = merged.ffill().fillna(0)
        for fi, feat in enumerate(NODE_FEATURES):
            if feat in merged.columns:
                node_data[:, si, fi] = merged[feat].values[:T]
        print(f"    {skey}: {len(sdf):,} bins")

    del df, agg, rows

    # ── 2. Global features ──
    print(f"\n  [2/4] Loading global features ({NUM_GLOBAL_RAW} metrics)...")
    global_data = np.zeros((T, NUM_GLOBAL_FEATURES), dtype=np.float32)

    for gi, metric in enumerate(GLOBAL_METRICS):
        rows = conn.execute("""
            SELECT timestamp, value FROM samples
            WHERE metric = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
        """, (metric, start.isoformat(), end.isoformat())).fetchall()

        if not rows:
            print(f"    {metric}: NO DATA")
            continue

        mdf = pd.DataFrame(rows, columns=["timestamp", "value"])
        mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], format="ISO8601", utc=True)
        mdf["time_bin"] = mdf["timestamp"].dt.floor(f"{STEP_MINUTES}min")
        mdf = mdf.groupby("time_bin")["value"].mean().reset_index()
        merged = pd.DataFrame({"time_bin": time_bins}).merge(mdf, on="time_bin", how="left")
        merged = merged.ffill().bfill().fillna(0)
        global_data[:, gi] = merged["value"].values[:T]
        coverage = (merged["value"] != 0).sum() / T * 100
        print(f"    {metric}: {len(mdf):,} bins ({coverage:.0f}% coverage)")

    sw_speed_idx = GLOBAL_METRICS.index("solar_wind_speed")
    sw_density_idx = GLOBAL_METRICS.index("solar_wind_density")
    imf_bz_idx = GLOBAL_METRICS.index("imf_bz_gsm")

    global_data[:, NUM_GLOBAL_RAW] = (
        global_data[:, sw_density_idx] * global_data[:, sw_speed_idx] ** 2 / 1e12
    )
    bz = global_data[:, imf_bz_idx]
    global_data[:, NUM_GLOBAL_RAW + 1] = (
        global_data[:, sw_speed_idx] * np.abs(np.minimum(bz, 0))
    )
    print(f"    + sw_dynamic_pressure (derived)")
    print(f"    + imf_coupling (derived)")
    print(f"    Total global features: {NUM_GLOBAL_FEATURES}")

    del rows

    # ── 3. DART buoy data ──
    print(f"\n  [3/4] Loading DART buoy data ({NUM_DART} buoys)...")
    dart_data = np.zeros((T, NUM_DART, DART_FEATURES_PER), dtype=np.float32)

    for di, buoy in enumerate(DART_BUOYS):
        rows = conn.execute("""
            SELECT timestamp, height_m FROM dart_readings
            WHERE station_id = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
        """, (buoy["id"], start.isoformat(), end.isoformat())).fetchall()

        if not rows:
            print(f"    {buoy['id']}: NO DATA")
            continue

        ddf = pd.DataFrame(rows, columns=["timestamp", "height_m"])
        ddf["timestamp"] = pd.to_datetime(ddf["timestamp"], format="ISO8601", utc=True)
        ddf["time_bin"] = ddf["timestamp"].dt.floor(f"{STEP_MINUTES}min")
        ddf = ddf.groupby("time_bin")["height_m"].mean().reset_index()
        merged = pd.DataFrame({"time_bin": time_bins}).merge(ddf, on="time_bin", how="left")
        merged = merged.ffill().bfill().fillna(0)

        height = merged["height_m"].values[:T].astype(np.float32)
        dart_data[:, di, 0] = height
        dart_data[1:, di, 1] = np.diff(height) * 1000
        h_series = pd.Series(height)
        dart_data[:, di, 2] = h_series.rolling(12, min_periods=1).std().fillna(0).values

        coverage = (height != 0).sum() / T * 100
        print(f"    {buoy['id']}: {len(ddf):,} bins ({coverage:.0f}% coverage)")

    del rows

    # ── 4. Earthquake targets ──
    print(f"\n  [4/4] Loading earthquake targets (M{MIN_MAG}+)...")
    quakes = conn.execute("""
        SELECT timestamp, lat, lon, magnitude FROM earthquakes
        WHERE magnitude >= ? AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (MIN_MAG, start.isoformat(), end.isoformat())).fetchall()

    targets = pd.DataFrame({"time_bin": time_bins})
    for zone in ZONES:
        targets[f"target_{zone['id']}"] = 0

    event_count = {z["id"]: 0 for z in ZONES}
    for q in quakes:
        try:
            qt = pd.Timestamp(q[0], tz=UTC)
        except Exception:
            continue
        qlat, qlon = q[1], q[2]
        for zone in ZONES:
            if (zone["lat_range"][0] <= qlat <= zone["lat_range"][1] and
                zone["lon_range"][0] <= qlon <= zone["lon_range"][1]):
                event_bin = qt.floor(f"{STEP_MINUTES}min")
                horizon_start = event_bin - timedelta(minutes=HORIZON_STEPS * STEP_MINUTES)
                mask = (targets["time_bin"] >= horizon_start) & (targets["time_bin"] < event_bin)
                targets.loc[mask, f"target_{zone['id']}"] = 1
                event_count[zone["id"]] += 1

    target_data = np.zeros((T, NUM_ZONES), dtype=np.float32)
    for zi, zone in enumerate(ZONES):
        col = f"target_{zone['id']}"
        target_data[:len(targets), zi] = targets[col].values[:T]

    conn.close()

    for zone in ZONES:
        print(f"    {zone['id']}: {event_count[zone['id']]} M{MIN_MAG}+ events")

    return {
        "node_features": node_data,
        "global_features": global_data,
        "dart_features": dart_data,
        "targets": target_data,
        "time_bins": time_bins.values,
    }


class ZoneDataset(Dataset):
    def __init__(self, node_features, global_features, dart_features, targets, start_idx, end_idx):
        self.node = node_features
        self.glob = global_features
        self.dart = dart_features
        self.targets = targets
        self.start = start_idx
        self.end = end_idx
        self.length = end_idx - start_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = self.start + idx
        s = slice(i, i + LOOKBACK_STEPS)
        x_node = torch.from_numpy(self.node[s].copy())
        x_glob = torch.from_numpy(self.glob[s].copy())
        x_dart = torch.from_numpy(self.dart[s].copy())
        y_window = self.targets[i + LOOKBACK_STEPS:i + LOOKBACK_STEPS + HORIZON_STEPS]
        y = torch.from_numpy(y_window.max(axis=0).copy())
        return x_node, x_glob, x_dart, y


class TemporalEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim=32, num_layers=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        # 288 → 72 → 18 steps — GRU only sees 18 steps
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.drop(F.gelu(self.bn1(self.conv1(x.transpose(1, 2)))))
        h = self.drop(F.gelu(self.bn2(self.conv2(h))))
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        return self.norm(h[:, -1, :])


class ZoneSTGNN(nn.Module):
    """v3: simplified — no GAT (graph was fully connected = meaningless),
    mean-pool station embeddings instead."""
    def __init__(self, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.seismic_enc = TemporalEncoder(NUM_NODE_FEATURES, hidden_dim, dropout=dropout)
        self.global_enc = TemporalEncoder(NUM_GLOBAL_FEATURES, hidden_dim, dropout=dropout)
        self.dart_enc = TemporalEncoder(DART_FEATURES_PER, hidden_dim, dropout=dropout)

        # Zone cross-attention over pooled sources
        # Keys: 1 (mean station) + 1 (global) + NUM_DART = 7
        self.zone_query = nn.Parameter(torch.randn(NUM_ZONES, hidden_dim) * 0.02)
        self.zone_attn = nn.MultiheadAttention(hidden_dim, num_heads=4,
                                                batch_first=True, dropout=dropout)
        self.zone_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_node, x_global, x_dart):
        B, T, N, Feat = x_node.shape

        # Seismic: encode each station, then mean-pool
        x_flat = x_node.permute(0, 2, 1, 3).reshape(B * N, T, Feat)
        station_emb = self.seismic_enc(x_flat).view(B, N, self.hidden_dim)
        station_pooled = station_emb.mean(dim=1, keepdim=True)  # (B, 1, H)

        # Global
        global_emb = self.global_enc(x_global).unsqueeze(1)  # (B, 1, H)

        # DART
        Nd = x_dart.shape[2]
        dart_flat = x_dart.permute(0, 2, 1, 3).reshape(B * Nd, T, DART_FEATURES_PER)
        dart_emb = self.dart_enc(dart_flat).view(B, Nd, self.hidden_dim)  # (B, 5, H)

        all_emb = torch.cat([station_pooled, global_emb, dart_emb], dim=1)  # (B, 7, H)

        zone_q = self.zone_query.unsqueeze(0).expand(B, -1, -1)
        zone_emb, _ = self.zone_attn(zone_q, all_emb, all_emb)
        zone_emb = self.zone_norm(zone_emb + zone_q.expand_as(zone_emb))

        logits = self.classifier(zone_emb).squeeze(-1)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
        )
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x_node, x_glob, x_dart, y in loader:
        x_node = x_node.to(DEVICE)
        x_glob = x_glob.to(DEVICE)
        x_dart = x_dart.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_node, x_glob, x_dart)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x_node.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for x_node, x_glob, x_dart, y in loader:
        x_node = x_node.to(DEVICE)
        x_glob = x_glob.to(DEVICE)
        x_dart = x_dart.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x_node, x_glob, x_dart)
        loss = criterion(logits, y)
        total_loss += loss.item() * x_node.size(0)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    metrics = {"loss": avg_loss}

    any_pred = preds.max(axis=1)
    any_target = targets.max(axis=1)
    if any_target.sum() > 0 and any_target.sum() < len(any_target):
        metrics["auc"] = roc_auc_score(any_target, any_pred)
        fpr, tpr, thresholds = roc_curve(any_target, any_pred)
        j_scores = tpr - fpr
        best_thresh = float(thresholds[np.argmax(j_scores)])
        metrics["threshold"] = best_thresh
        binary_pred = (any_pred > best_thresh).astype(int)
        metrics["precision"] = precision_score(any_target, binary_pred, zero_division=0)
        metrics["recall"] = recall_score(any_target, binary_pred, zero_division=0)
        metrics["f1"] = f1_score(any_target, binary_pred, zero_division=0)
    else:
        metrics["auc"] = 0.0

    zone_aucs = {}
    for zi in range(NUM_ZONES):
        if targets[:, zi].sum() > 0 and targets[:, zi].sum() < len(targets):
            zone_aucs[ZONES[zi]["id"]] = roc_auc_score(targets[:, zi], preds[:, zi])
    metrics["zone_aucs"] = zone_aucs

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Zone-focused ST-GNN v3")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print("  ZONE TEST v3 — ST-GNN (6 zones)")
    print("  Fixes: train-only norm, no oversampling, no GAT, stronger reg")
    print("  24h lookback → 24h prediction horizon, M5.0+")
    print("=" * 70)

    print(f"\n  Building dataset...")
    dataset = build_dataset()

    node_data = dataset["node_features"]
    global_data = dataset["global_features"]
    dart_data = dataset["dart_features"]
    target_data = dataset["targets"]
    T = node_data.shape[0]
    n_seq = T - LOOKBACK_STEPS - HORIZON_STEPS + 1

    # Temporal split
    train_end = int(n_seq * 0.7)
    val_end = int(n_seq * 0.85)

    # ── Normalize using TRAINING DATA ONLY ──
    print("\n  Normalizing (train-only stats)...")
    train_node = node_data[:train_end + LOOKBACK_STEPS]
    node_mean = train_node.mean(axis=(0, 1), keepdims=True)
    node_std = train_node.std(axis=(0, 1), keepdims=True)
    node_std[node_std < 1e-8] = 1.0
    node_data = (node_data - node_mean) / node_std

    train_glob = global_data[:train_end + LOOKBACK_STEPS]
    global_mean = train_glob.mean(axis=0, keepdims=True)
    global_std = train_glob.std(axis=0, keepdims=True)
    global_std[global_std < 1e-8] = 1.0
    global_data = (global_data - global_mean) / global_std

    train_dart = dart_data[:train_end + LOOKBACK_STEPS]
    dart_mean = train_dart.mean(axis=(0, 1), keepdims=True)
    dart_std = train_dart.std(axis=(0, 1), keepdims=True)
    dart_std[dart_std < 1e-8] = 1.0
    dart_data = (dart_data - dart_mean) / dart_std

    pos_rate = target_data.mean()
    print(f"\n  Overall positive rate: {pos_rate:.4f}")
    for zi, zone in enumerate(ZONES):
        print(f"    {zone['id']:15s} {target_data[:, zi].mean():.5f}")

    print(f"\n  Sequences: {n_seq:,} (lookback={LOOKBACK_STEPS} [{LOOKBACK_STEPS*5/60:.0f}h], "
          f"horizon={HORIZON_STEPS} [{HORIZON_STEPS*5/60:.0f}h])")
    print(f"  Features: {NUM_NODE_FEATURES} seismic/station × {NUM_STATIONS} stations "
          f"+ {NUM_GLOBAL_FEATURES} global + {DART_FEATURES_PER}/buoy × {NUM_DART} DART")

    train_ds = ZoneDataset(node_data, global_data, dart_data, target_data, 0, train_end)
    val_ds = ZoneDataset(node_data, global_data, dart_data, target_data, train_end, val_end)
    test_ds = ZoneDataset(node_data, global_data, dart_data, target_data, val_end, n_seq)

    # Compute class weight from training labels (no oversampling)
    train_labels = np.zeros(len(train_ds), dtype=np.float32)
    for i in range(len(train_ds)):
        y_window = target_data[i + LOOKBACK_STEPS:i + LOOKBACK_STEPS + HORIZON_STEPS]
        train_labels[i] = y_window.max()
    pos_count = train_labels.sum()
    neg_count = len(train_labels) - pos_count
    pos_weight = min(neg_count / max(pos_count, 1), 10.0)  # cap at 10x
    pos_pct = pos_count / len(train_labels) * 100

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  Positives in train: {int(pos_count):,} ({pos_pct:.2f}%)")
    print(f"  Class weight: pos_weight={pos_weight:.1f}x (capped at 10x)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)

    model = ZoneSTGNN(
        hidden_dim=args.hidden,
        dropout=args.dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")
    print(f"  Hidden: {args.hidden}, dropout: {args.dropout}")
    print(f"  Device: {DEVICE}")

    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    best_val_auc = 0
    patience_counter = 0
    best_path = os.path.join(MODEL_DIR, "stgnn_zone_test_v3.pt")

    print(f"\n  Training for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"  {'Epoch':>5} {'Train':>10} {'Val':>9} {'AUC':>7} {'Thr':>5} "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'LR':>12}")
    print("  " + "-" * 78)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val = evaluate(model, val_loader, criterion)
        scheduler.step(val.get("auc", 0))

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        thresh = val.get('threshold', 0.5)
        print(f"  {epoch:>5d} {train_loss:>10.5f} {val['loss']:>9.5f} "
              f"{val.get('auc', 0):>7.4f} {thresh:>5.3f} "
              f"{val.get('precision', 0):>6.3f} "
              f"{val.get('recall', 0):>6.3f} "
              f"{val.get('f1', 0):>6.3f} "
              f"{lr:>12.6f}  ({elapsed:.1f}s)")

        if val.get("auc", 0) > best_val_auc:
            best_val_auc = val["auc"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": best_val_auc,
                "val_metrics": val,
                "stations": STATION_KEYS,
                "dart_buoys": [b["id"] for b in DART_BUOYS],
                "zones": [z["id"] for z in ZONES],
                "norm_params": {
                    "node_mean": node_mean, "node_std": node_std,
                    "global_mean": global_mean, "global_std": global_std,
                    "dart_mean": dart_mean, "dart_std": dart_std,
                },
                "args": vars(args),
            }, best_path)
            print(f"         ↑ best AUC={best_val_auc:.4f} → {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final test
    print(f"\n  {'='*70}")
    print(f"  FINAL TEST EVALUATION")
    print(f"  {'='*70}")

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, criterion)
    print(f"  Test Loss:      {test_metrics['loss']:.5f}")
    print(f"  Test AUC:       {test_metrics.get('auc', 0):.4f}")
    print(f"  Test Precision: {test_metrics.get('precision', 0):.4f}")
    print(f"  Test Recall:    {test_metrics.get('recall', 0):.4f}")
    print(f"  Test F1:        {test_metrics.get('f1', 0):.4f}")

    print(f"\n  Per-zone AUC:")
    for zone_id, auc in test_metrics.get("zone_aucs", {}).items():
        print(f"    {zone_id:15s} {auc:.4f}")

    print(f"\n  Model saved: {best_path}")
    print(f"  Best val AUC: {best_val_auc:.4f} (epoch {ckpt['epoch']})")


if __name__ == "__main__":
    main()
