"""SeismicLab — Zone-Focused ST-GNN Test Training (v2)

Full-feature run: seismic + solar/geomagnetic + tidal + DART buoy data.
24h lookback, 6h prediction horizon.

Stations: TEIG, SJG, ANMO, TUC, COR, COLA
DART buoys: 43413 (E. Pacific), 42407 (Caribbean), 41420/41421 (N. Caribbean), 42409 (Gulf)
Target zones: mexico_ca, caribbean
Data range: 2021-07-01 → present
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from datetime import datetime, timedelta, timezone

torch.set_float32_matmul_precision("medium")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UTC = timezone.utc
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "seismiclab.db")
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
DART_FEATURES_PER = 3  # height, rate_of_change, rolling_std

# --- Target zones ---
ZONES = [
    {"id": "mexico_ca",  "lat_range": [7, 25],  "lon_range": [-115, -77]},
    {"id": "caribbean",  "lat_range": [10, 22], "lon_range": [-85, -60]},
]
NUM_ZONES = len(ZONES)

# --- Seismic node features (5 raw + 6 derived) ---
NODE_FEATURES = ["amp_min", "amp_max", "amp_mean", "sta_lta_ratio", "triggered",
                 "amp_range", "amp_rms_1h", "sta_lta_max_1h", "trigger_count_1h",
                 "amp_trend_1h", "noise_floor_delta"]
NUM_NODE_FEATURES = len(NODE_FEATURES)

# --- Global features from samples table ---
GLOBAL_METRICS = [
    "solar_wind_speed", "solar_wind_density", "solar_wind_temp",
    "imf_bt", "imf_bz_gsm",
    "kp_index", "dst_index",
    "tidal_potential", "tidal_strain_rate",
    "neutron_count", "ief",
]
NUM_GLOBAL_RAW = len(GLOBAL_METRICS)
NUM_GLOBAL_FEATURES = NUM_GLOBAL_RAW + 2  # +sw_dynamic_pressure, +imf_coupling

# --- Training params ---
LOOKBACK_STEPS = 288      # 24 hours
HORIZON_STEPS = 72        # 6 hours
STEP_MINUTES = 5
MIN_MAG = 5.0
EDGE_DISTANCE_KM = 6000


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def build_edge_index():
    src, dst = [], []
    for i, si in enumerate(STATIONS):
        for j, sj in enumerate(STATIONS):
            if i == j:
                continue
            d = haversine_km(si["lat"], si["lon"], sj["lat"], sj["lon"])
            if d <= EDGE_DISTANCE_KM:
                src.append(i)
                dst.append(j)
    return np.array([src, dst], dtype=np.int64)


def build_edge_weights(edge_index):
    weights = []
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        d = haversine_km(STATIONS[i]["lat"], STATIONS[i]["lon"],
                         STATIONS[j]["lat"], STATIONS[j]["lon"])
        weights.append(1.0 / max(d, 100))
    weights = np.array(weights, dtype=np.float32)
    if weights.max() > 0:
        weights /= weights.max()
    return weights


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

    # Derived global features
    sw_speed_idx = GLOBAL_METRICS.index("solar_wind_speed")
    sw_density_idx = GLOBAL_METRICS.index("solar_wind_density")
    imf_bz_idx = GLOBAL_METRICS.index("imf_bz_gsm")

    # Dynamic pressure: ~0.5 * n * v^2 (normalized)
    global_data[:, NUM_GLOBAL_RAW] = (
        global_data[:, sw_density_idx] * global_data[:, sw_speed_idx] ** 2 / 1e12
    )
    # IMF coupling: v * |Bz_south| (only when Bz is southward / negative)
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
    dart_ids = [b["id"] for b in DART_BUOYS]

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
        # Rate of change (mm/step)
        dart_data[1:, di, 1] = np.diff(height) * 1000
        # Rolling 1h std
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

    # ── Normalize ──
    print("\n  Normalizing...")
    node_mean = node_data.mean(axis=(0, 1), keepdims=True)
    node_std = node_data.std(axis=(0, 1), keepdims=True)
    node_std[node_std < 1e-8] = 1.0
    node_data = (node_data - node_mean) / node_std

    global_mean = global_data.mean(axis=0, keepdims=True)
    global_std = global_data.std(axis=0, keepdims=True)
    global_std[global_std < 1e-8] = 1.0
    global_data = (global_data - global_mean) / global_std

    dart_mean = dart_data.mean(axis=(0, 1), keepdims=True)
    dart_std = dart_data.std(axis=(0, 1), keepdims=True)
    dart_std[dart_std < 1e-8] = 1.0
    dart_data = (dart_data - dart_mean) / dart_std

    pos_rate = target_data.mean()
    print(f"\n  Overall positive rate: {pos_rate:.4f}")
    for zi, zone in enumerate(ZONES):
        print(f"    {zone['id']:15s} {target_data[:, zi].mean():.5f}")

    # Graph
    edge_index = build_edge_index()
    edge_weight = build_edge_weights(edge_index)
    print(f"  Graph: {NUM_STATIONS} nodes, {edge_index.shape[1]} edges")

    mem_mb = (node_data.nbytes + global_data.nbytes + dart_data.nbytes + target_data.nbytes) / 1e6
    print(f"  Memory: {mem_mb:.0f} MB")

    return {
        "node_features": node_data,
        "global_features": global_data,
        "dart_features": dart_data,
        "targets": target_data,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "time_bins": time_bins.values,
        "norm_params": {
            "node_mean": node_mean, "node_std": node_std,
            "global_mean": global_mean, "global_std": global_std,
            "dart_mean": dart_mean, "dart_std": dart_std,
        },
    }


class ZoneDataset(Dataset):
    """Lazy windowing dataset — indexes into base arrays on the fly."""
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
    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        # Strided convs: 288 → 144 → 72 steps before GRU
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, T, Fin = x.shape
        h = self.drop(F.gelu(self.bn1(self.conv1(x.transpose(1, 2)))))
        h = self.drop(F.gelu(self.bn2(self.conv2(h))))
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        return self.norm(h[:, -1, :])


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        B, N, _ = x.shape
        h = self.W(x)
        src, dst = edge_index[0], edge_index[1]
        e_src = self.a_src(h[:, src])
        e_dst = self.a_dst(h[:, dst])
        e = self.leaky(e_src + e_dst).squeeze(-1)
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(0)
        alpha = torch.full((B, N, N), float('-inf'), device=x.device)
        alpha[:, dst, src] = e
        alpha = torch.softmax(alpha, dim=-1)
        alpha = self.dropout(alpha)
        return torch.bmm(alpha, h)


class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, head_dim, dropout) for _ in range(num_heads)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        head_outs = [head(x, edge_index, edge_weight) for head in self.heads]
        out = torch.cat(head_outs, dim=-1)
        return self.norm(self.dropout(out) + x if out.shape == x.shape else self.norm(out))


class ZoneSTGNN(nn.Module):
    """Multi-source ST-GNN: seismic nodes + global features + DART buoys."""
    def __init__(self, hidden_dim=48, num_gat_heads=4, num_gat_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Three temporal encoders for different data sources
        self.seismic_enc = TemporalEncoder(NUM_NODE_FEATURES, hidden_dim, dropout=dropout)
        self.global_enc = TemporalEncoder(NUM_GLOBAL_FEATURES, hidden_dim, dropout=dropout)
        self.dart_enc = TemporalEncoder(DART_FEATURES_PER, hidden_dim, dropout=dropout)

        # GAT over seismic station embeddings
        self.gat_layers = nn.ModuleList([
            MultiHeadGAT(hidden_dim, hidden_dim, num_gat_heads, dropout)
            for _ in range(num_gat_layers)
        ])

        # Zone cross-attention: queries = zone params, keys/values = all source embeddings
        # Total keys: NUM_STATIONS + 1 (global) + NUM_DART = 12
        self.zone_query = nn.Parameter(torch.randn(NUM_ZONES, hidden_dim))
        self.zone_attn = nn.MultiheadAttention(hidden_dim, num_heads=4,
                                                batch_first=True, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_node, x_global, x_dart, edge_index, edge_weight=None):
        B, T, N, Feat = x_node.shape

        # Seismic: (B, T, N_stations, F) → (B, N_stations, H)
        x_flat = x_node.permute(0, 2, 1, 3).reshape(B * N, T, Feat)
        station_emb = self.seismic_enc(x_flat).view(B, N, self.hidden_dim)

        for gat in self.gat_layers:
            station_emb = station_emb + gat(station_emb, edge_index, edge_weight)

        # Global: (B, T, F_global) → (B, 1, H)
        global_emb = self.global_enc(x_global).unsqueeze(1)

        # DART: (B, T, N_dart, F_dart) → (B, N_dart, H)
        Nd = x_dart.shape[2]
        dart_flat = x_dart.permute(0, 2, 1, 3).reshape(B * Nd, T, DART_FEATURES_PER)
        dart_emb = self.dart_enc(dart_flat).view(B, Nd, self.hidden_dim)

        # Concat all source embeddings as keys/values
        all_emb = torch.cat([station_emb, global_emb, dart_emb], dim=1)

        # Zone cross-attention
        zone_q = self.zone_query.unsqueeze(0).expand(B, -1, -1)
        zone_emb, _ = self.zone_attn(zone_q, all_emb, all_emb)

        logits = self.classifier(zone_emb).squeeze(-1)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def train_epoch(model, loader, optimizer, criterion, ei, ew):
    model.train()
    total_loss = 0
    for x_node, x_glob, x_dart, y in loader:
        x_node = x_node.to(DEVICE)
        x_glob = x_glob.to(DEVICE)
        x_dart = x_dart.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_node, x_glob, x_dart, ei, ew)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x_node.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, ei, ew):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for x_node, x_glob, x_dart, y in loader:
        x_node = x_node.to(DEVICE)
        x_glob = x_glob.to(DEVICE)
        x_dart = x_dart.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x_node, x_glob, x_dart, ei, ew)
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
    parser = argparse.ArgumentParser(description="Zone-focused ST-GNN v2 — full features")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    print("=" * 70)
    print("  ZONE TEST v2 — ST-GNN (Mexico/CA + Caribbean)")
    print("  Full features: seismic + solar/geomag + tidal + DART")
    print("  24h lookback → 6h prediction horizon, M5.0+")
    print("=" * 70)

    print(f"\n  Building dataset...")
    dataset = build_dataset()

    node_data = dataset["node_features"]
    global_data = dataset["global_features"]
    dart_data = dataset["dart_features"]
    target_data = dataset["targets"]
    T = node_data.shape[0]
    n_seq = T - LOOKBACK_STEPS - HORIZON_STEPS + 1
    print(f"\n  Sequences: {n_seq:,} (lookback={LOOKBACK_STEPS} [{LOOKBACK_STEPS*5/60:.0f}h], "
          f"horizon={HORIZON_STEPS} [{HORIZON_STEPS*5/60:.0f}h])")
    print(f"  Features: {NUM_NODE_FEATURES} seismic/station × {NUM_STATIONS} stations "
          f"+ {NUM_GLOBAL_FEATURES} global + {DART_FEATURES_PER}/buoy × {NUM_DART} DART")

    # Temporal split
    train_end = int(n_seq * 0.7)
    val_end = int(n_seq * 0.85)

    train_ds = ZoneDataset(node_data, global_data, dart_data, target_data, 0, train_end)
    val_ds = ZoneDataset(node_data, global_data, dart_data, target_data, train_end, val_end)
    test_ds = ZoneDataset(node_data, global_data, dart_data, target_data, val_end, n_seq)

    # Weighted sampler
    print("  Computing sample weights for oversampling...")
    train_labels = np.zeros(len(train_ds), dtype=np.float32)
    for i in range(len(train_ds)):
        y_window = target_data[i + LOOKBACK_STEPS:i + LOOKBACK_STEPS + HORIZON_STEPS]
        train_labels[i] = y_window.max()
    pos_count = train_labels.sum()
    neg_count = len(train_labels) - pos_count
    weight_pos = neg_count / max(pos_count, 1)
    sample_weights = np.where(train_labels > 0, weight_pos, 1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    pos_pct = pos_count / len(train_labels) * 100
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  Positives in train: {int(pos_count):,} ({pos_pct:.2f}%) — oversampling {weight_pos:.1f}x")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)

    ei = torch.from_numpy(dataset["edge_index"]).to(DEVICE)
    ew = torch.from_numpy(dataset["edge_weight"]).to(DEVICE)

    model = ZoneSTGNN(
        hidden_dim=args.hidden,
        num_gat_heads=args.gat_heads,
        num_gat_layers=args.gat_layers,
        dropout=args.dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")
    print(f"  Hidden: {args.hidden}, GAT heads: {args.gat_heads}, GAT layers: {args.gat_layers}")
    print(f"  Device: {DEVICE}")

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_auc = 0
    patience_counter = 0
    best_path = os.path.join(MODEL_DIR, "stgnn_zone_test_v2.pt")

    print(f"\n  Training for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"  {'Epoch':>5} {'Train':>10} {'Val':>9} {'AUC':>7} {'Thr':>5} "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'MX_CA':>7} {'CARIB':>7} {'LR':>10}")
    print("  " + "-" * 90)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, ei, ew)
        val = evaluate(model, val_loader, criterion, ei, ew)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        mx_auc = val.get("zone_aucs", {}).get("mexico_ca", 0)
        cb_auc = val.get("zone_aucs", {}).get("caribbean", 0)

        thresh = val.get('threshold', 0.5)
        print(f"  {epoch:>5d} {train_loss:>10.5f} {val['loss']:>9.5f} "
              f"{val.get('auc', 0):>7.4f} {thresh:>5.3f} "
              f"{val.get('precision', 0):>6.3f} "
              f"{val.get('recall', 0):>6.3f} "
              f"{val.get('f1', 0):>6.3f} "
              f"{mx_auc:>7.4f} {cb_auc:>7.4f} "
              f"{lr:>10.6f}  ({elapsed:.1f}s)")

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
                "norm_params": dataset["norm_params"],
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

    test_metrics = evaluate(model, test_loader, criterion, ei, ew)
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
