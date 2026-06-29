"""QuakeWatch — Ensemble Earthquake Forecaster (Full Feature Extraction)

Pipeline:
  1. Build ALL ~185 features per zone per hour from DB
     (full parity with XGBoost's experiment.py extract_features)
  2. Smart negative sampling (boundary-aware + random subsample)
  3. Train LightGBM on snapshot features
  4. Train XGBoost on snapshot features (different hyperparams for diversity)
  5. Stack with logistic regression meta-learner

Target: M5.0+ binary classification, 6h prediction horizon, per zone
Features: seismic catalog, solar/geomag/tidal signals, DART buoys,
          CME/flare/storm events, tidal triggering, volcanic, FIRMS,
          seismic stations, coupling & interaction terms

Run:  python3 lab/train_ensemble.py [--min-mag 5.0] [--horizon 6]
      python3 lab/train_ensemble.py --skip-feature-build  # use cached features
"""

import os
import sys
import time
import math
import sqlite3
import argparse
import warnings
from collections import deque
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score, average_precision_score)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "quakewatch.db")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GPS_DB = os.path.join(CACHE_DIR, "gps.db")  # crustal deformation (NGL), separate DB
os.makedirs(MODEL_DIR, exist_ok=True)
PARENT_ZONES = [
    {"id": "indonesia",    "lat_range": [-12, 8],   "lon_range": [95, 140]},
    {"id": "japan_kurils", "lat_range": [25, 50],   "lon_range": [128, 155]},
    {"id": "south_america","lat_range": [-60, 7],   "lon_range": [-82, -60]},
    {"id": "mexico_ca",    "lat_range": [7, 25],    "lon_range": [-115, -77]},
    {"id": "himalaya",     "lat_range": [25, 42],   "lon_range": [60, 100]},  # extended W to Makran/Iran
    {"id": "alaska",       "lat_range": [50, 72],   "lon_range": [-180, -130]},
    # California — benchmark zone. Complete USGS catalog (Mc~1.5, best on Earth),
    # densest station network anywhere. Run at M4+ target: many M4-5 labels, few M5+.
    {"id": "california",   "lat_range": [32, 42],   "lon_range": [-125, -114]},
    # Philippines/Taiwan — extremely active arc (Luzon, Mindanao, Taiwan). Abuts the
    # indonesia zone at lat 8 to avoid overlap. Added after a live foreshock sequence.
    {"id": "philippines",  "lat_range": [8, 25],    "lon_range": [117, 127]},
    # --- Global expansion (full-stack: EMSC foreshocks + open stations) ---
    {"id": "mediterranean","lat_range": [34, 43],   "lon_range": [19, 45]},    # Italy/Greece/Turkey
    {"id": "caribbean",    "lat_range": [10, 20],   "lon_range": [-77, -60]},  # Hispaniola/PR/Lesser Antilles
    {"id": "new_zealand",  "lat_range": [-48, -34], "lon_range": [165, 180]},  # Alpine fault / Kermadec
    # --- Global expansion (catalog-only: very active, not in EMSC) ---
    {"id": "png_solomon",  "lat_range": [-12, -3],  "lon_range": [140, 160]},  # most active gap
    {"id": "kamchatka",    "lat_range": [50, 62],   "lon_range": [156, 168]},  # Kamchatka/N. Kurils
]
CELL_SIZE = 10  # degrees

def generate_grid_cells(parent_zones, cell_size=CELL_SIZE):
    cells = []
    for pz in parent_zones:
        lat_lo, lat_hi = pz["lat_range"]
        lon_lo, lon_hi = pz["lon_range"]
        for lat in range(int(lat_lo), int(lat_hi), cell_size):
            for lon in range(int(lon_lo), int(lon_hi), cell_size):
                cell_id = f"{pz['id']}_{lat:+03d}_{lon:+04d}"
                cells.append({
                    "id": cell_id,
                    "parent": pz["id"],
                    "lat_range": [lat, min(lat + cell_size, int(lat_hi))],
                    "lon_range": [lon, min(lon + cell_size, int(lon_hi))],
                })
    return cells

ZONES = PARENT_ZONES
NUM_ZONES = len(PARENT_ZONES)

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
    # Dense regional stations (2021+) — Chile + Alaska, for rupture-size signal
    {"key": "C1.VA01", "lat": -33.0, "lon": -71.6},
    {"key": "C1.VA05", "lat": -33.7, "lon": -71.6},
    {"key": "C1.VA06", "lat": -32.6, "lon": -71.3},
    {"key": "C1.MT02", "lat": -33.3, "lon": -71.1},
    {"key": "C1.MT07", "lat": -33.0, "lon": -71.0},
    {"key": "C.ROC1",  "lat": -33.0, "lon": -71.0},
    {"key": "C1.MT01", "lat": -33.9, "lon": -71.3},
    {"key": "C1.MT19", "lat": -33.4, "lon": -70.9},
    {"key": "C1.MT05", "lat": -33.4, "lon": -70.7},
    {"key": "AK.SWD",  "lat": 60.1, "lon": -149.5},
    {"key": "AK.BRSE", "lat": 59.7, "lon": -150.7},
    {"key": "AK.BRLK", "lat": 59.8, "lon": -150.9},
    {"key": "AK.SLK",  "lat": 60.5, "lon": -150.2},
    {"key": "AK.CNP",  "lat": 59.5, "lon": -151.2},
    {"key": "AK.HOM",  "lat": 59.7, "lon": -151.7},
    {"key": "AK.CAPN", "lat": 60.8, "lon": -151.2},
    # Taiwan (Philippines/Taiwan zone) — deliver waveforms via IRIS
    {"key": "TW.SSLB", "lat": 23.8, "lon": 121.0},
    {"key": "TW.YULB", "lat": 23.4, "lon": 121.3},
    {"key": "TW.TPUB", "lat": 23.3, "lon": 120.6},
    {"key": "TW.TWGB", "lat": 22.8, "lon": 121.1},
    {"key": "TW.NACB", "lat": 24.2, "lon": 121.6},
    {"key": "TW.YHNB", "lat": 24.7, "lon": 121.4},
    # Global expansion stations (open on IRIS)
    {"key": "MN.AQU",  "lat": 42.35, "lon": 13.40},   # central Italy
    {"key": "IU.SNZO", "lat": -41.31, "lon": 174.70}, # Wellington, NZ
    {"key": "PR.PCDR", "lat": 18.51, "lon": -68.38},  # Dominican Rep
    {"key": "DR.SDD",  "lat": 18.46, "lon": -69.92},  # Santo Domingo
    {"key": "PR.SMDR", "lat": 19.29, "lon": -69.19},  # Samana, DR
    {"key": "AU.RABL", "lat": -4.19, "lon": 152.16},  # Rabaul, PNG
    {"key": "IU.PET",  "lat": 53.02, "lon": 158.65},  # Petropavlovsk, Kamchatka
]
STATIONS_PER_ZONE = 4

LOOKBACK_HOURS = 168
HORIZON_HOURS = 6
MIN_MAG_TARGET = 5.0
MIN_MAG_CATALOG = 2.5

# ═══════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS — matches experiment.py's FEATURE_DEFS
# ═══════════════════════════════════════════════════════════════════════

CATALOG_FEATURES = [
    "eq_count_1h", "eq_count_6h", "eq_count_24h", "eq_count_72h",
    "eq_count_7d", "eq_count_30d",
    "m4_count_24h", "m4_count_7d",
    "foreshock_accel", "rate_anomaly_7d",
    "b_value_14d", "b_value_30d", "b_delta",
    "energy_24h", "energy_ratio",
    "max_mag_72h", "max_mag_7d", "mag_range_7d",
    "moment_7d", "moment_30d", "moment_accel",
    "time_since_last",
    "depth_mean_7d",
    "b_value_accel",
    "coulomb_proxy",
    "omori_deficit",
    "subduction_proxy",
    "foreshock_curvature",
]

GEOMAG_FEATURES = [
    "kp_mean_72h", "kp_max_72h", "kp_trend",
    "dst_mean_72h", "dst_min_72h", "dst_rate", "dst_volatility",
]

SOLAR_FEATURES = [
    "sw_speed_mean_72h", "sw_speed_max_72h",
    "sw_density_mean_72h", "sw_density_max_72h",
    "imf_bz_mean_72h", "imf_bz_min_72h",
    "imf_bt_mean_72h", "imf_bt_max_72h",
]

TIDAL_FEATURES = [
    "tidal_potential_72h", "tidal_strain_max_72h", "tidal_range_72h",
    "tide_range_72h", "tide_rate_max",
]

COSMIC_FEATURES = ["cosmic_mean_72h", "cosmic_trend_72h", "cosmic_anomaly"]

IEF_FEATURES = ["ief_mean_72h", "ief_max_72h", "ief_volatility"]

OLR_FEATURES = ["olr_mean_7d", "olr_anomaly", "olr_trend"]

GROUND_MAG_FEATURES = [
    "mag_x_mean_72h", "mag_x_range_72h", "mag_x_trend",
    "mag_y_mean_72h", "mag_y_range_72h", "mag_y_trend",
    "mag_z_mean_72h", "mag_z_range_72h", "mag_z_trend",
    "mag_dbdt_max",
]

TRAJECTORY_FEATURES = [
    "dst_mean_24h", "dst_mean_6h", "dst_slope_24h", "dst_slope_6h", "dst_ramp",
    "sw_mean_24h", "sw_mean_6h", "sw_slope_24h", "sw_slope_6h", "sw_ramp",
]

SHAPE_FEATURES = [
    "dst_curvature_72h", "sw_curvature_72h", "kp_curvature_72h",
    "dst_p10_72h", "dst_p90_72h",
    "sw_p10_72h", "sw_p90_72h",
    "kp_p10_72h", "kp_p90_72h",
]

COUPLING_FEATURES = [
    "sw_speed_accel", "sw_above_483", "sw_above_567", "sw_peak_to_mean",
    "kp_spike", "kp_hours_ge3",
    "dst_drop_rate", "bz_southward_frac", "storm_coupling_idx",
]

DART_FEATURES = [
    "dart_residual_mean_24h", "dart_residual_std_24h", "dart_residual_trend_48h",
    "dart_rate_of_change_6h", "dart_loading_index", "dart_trend_reversal",
    "dart_event_mode_count_7d",
    "dart_vol_ramp_6v18", "dart_vol_ramp_12v36",
    "dart_std_6h", "dart_std_12h",
    "dart_trend_12h", "dart_trend_6h", "dart_trend_accel",
    "dart_event_mode_24h", "dart_event_mode_12h", "dart_event_mode_6h",
    "dart_em_accel",
    # Deepened loading trajectory (dart_loading_index is a top escalation discriminator)
    "dart_loading_rate_24h", "dart_loading_max_72h", "dart_loading_accel",
    "dart_x_coulomb", "dart_x_sw",
]

CME_FEATURES = [
    "cme_count_72h", "cme_count_7d", "cme_speed_max_72h",
    "cme_speed_mean_72h", "cme_speed_max_7d",
    "cme_fast_count_7d", "cme_accel",
]

FLARE_FEATURES = [
    "flare_count_72h", "flare_count_7d", "flare_max_class_72h",
    "flare_max_class_7d", "flare_m_plus_count_7d", "flare_energy_72h",
]

STORM_FEATURES = [
    "storm_kp_max_72h", "storm_kp_max_7d", "storm_count_7d", "storm_severe_7d",
]

TIDAL_TRIGGER_FEATURES = [
    "tidal_schuster_p_30d", "tidal_schuster_p_90d",
    "tidal_schuster_R_30d", "tidal_schuster_R_90d",
    "tidal_mean_phase_30d", "tidal_sensitivity_flag", "tidal_sensitivity_onset",
    "tidal_R_30d_small", "tidal_R_30d_mid",
    "tidal_R_90d_small", "tidal_R_90d_mid",
    "tidal_p_30d_small", "tidal_p_30d_mid",
    "tidal_small_vs_mid_R", "tidal_mag_divergence",
    # Deepened: tidal-sensitivity trajectory + instantaneous tidal stress.
    # A swarm becoming tidally synchronized (rising R) while AT high tidal stress
    # is more likely to escalate — the mid-band R/p are top discriminators.
    "tidal_R_mid_rate_30d", "tidal_R_mid_accel",
    "tidal_stress_now", "tidal_stress_max_24h", "tidal_stress_rate",
]

VOLCANIC_FEATURES = [
    # STATIC per cell (regional volcanic prior — broadcast to all hours)
    "volcano_dist_nearest", "volcano_count_200km", "volcano_count_500km",
    "volcano_active_200km", "eruption_nearby_90d", "eruption_vei_max_90d",
    "eruption_ongoing_500km", "volcanic_stress_index",
]

# DYNAMIC volcanic activity (varies by hour, from the eruption timeline).
# Volcanic features were the top tier-2 discriminators but were entirely STATIC —
# these add a genuine time-varying volcanic-unrest signal (eruptions starting /
# active near the cell as the swarm evolves).
VOLCANIC_DYN_FEATURES = [
    "volc_erupt_active",     # # nearby volcanoes erupting at this hour
    "volc_onset_90d",        # eruptions that STARTED within trailing 90d (rolling)
    "volc_onset_365d",       # eruptions started within trailing 365d
    "volc_vei_max_365d",     # max VEI of eruptions started in trailing 365d
]

FIRMS_FEATURES = [
    "hotspot_count_7d", "hotspot_count_30d", "hotspot_frp_max_7d", "hotspot_accel",
]

LOCATION_FEATURES = []

STATION_FEATURES = [
    # Instantaneous (this hour, aggregated across nearby stations)
    "stn_amp_max", "stn_amp_mean", "stn_sta_lta_max",
    "stn_triggered", "stn_amp_range",
    # Sustained activity / trajectories (where the precursor signal lives)
    "stn_trigger_sum_24h", "stn_trigger_sum_72h", "stn_trigger_accel",
    "stn_amp_mean_24h", "stn_amp_trend_24h", "stn_sta_lta_mean_24h",
    # Anomaly vs station's own baseline
    "stn_amp_z_7d",        # current amplitude z-score vs 7d baseline
    "stn_quiescence",      # seismic quiescence index (amp suppression — classic precursor)
    # Spatial agreement
    "stn_coherence",       # fraction of nearby stations simultaneously elevated
]

# GPS / GNSS crustal deformation (daily, forward-filled to hourly). The textbook
# precursor: elastic strain accumulation + slow-slip transients (residuals).
GPS_FEATURES = [
    "gps_anomaly_max",       # max deformation anomaly score across nearby stations
    "gps_anomaly_mean",      # mean anomaly
    "gps_residual_mag_mm",   # horizontal residual transient magnitude (slow slip)
    "gps_residual_up_mm",    # |vertical residual| (subduction uplift/subsidence)
    "gps_strain_rate",       # station velocity magnitude (mm/yr)
    "gps_anomaly_coherence", # fraction of nearby stations simultaneously anomalous
    "gps_residual_accel",    # 7-day change in residual magnitude (transient growth)
    # NOTE: gps_coverage removed — it's near-static per region (station-count =
    # spatial identity), the same leak we banned with lat_abs / cell_idx.
]

INTERACTION_FEATURES = [
    "kp_x_foreshock", "dst_x_bvalue", "sw_x_accel", "ief_x_foreshock",
    "cosmic_x_dst", "sw_x_coulomb", "kp_x_moment",
    "cme_x_coulomb", "cme_x_foreshock", "flare_x_bvalue",
    "storm_x_foreshock", "cme_x_tidal_R",
    "tidal_sens_x_coulomb", "tidal_sens_x_bvalue", "tidal_sens_x_foreshock",
    "tidal_small_x_coulomb", "tidal_small_x_accel",
    "volcano_x_coulomb", "volcano_x_sw",
    "hotspot_x_coulomb",
]

VELOCITY_BASE_FEATURES = ["coulomb_proxy"]
VELOCITY_FEATURES = ["v24_coulomb_proxy", "acc_coulomb_proxy", "precursor_alarm"]

ALL_FEATURES = (CATALOG_FEATURES + GEOMAG_FEATURES + SOLAR_FEATURES +
                TIDAL_FEATURES + COSMIC_FEATURES + IEF_FEATURES + OLR_FEATURES +
                GROUND_MAG_FEATURES + TRAJECTORY_FEATURES + SHAPE_FEATURES +
                COUPLING_FEATURES + DART_FEATURES + CME_FEATURES + FLARE_FEATURES +
                STORM_FEATURES + TIDAL_TRIGGER_FEATURES + VOLCANIC_FEATURES +
                VOLCANIC_DYN_FEATURES + FIRMS_FEATURES + LOCATION_FEATURES +
                STATION_FEATURES + GPS_FEATURES + INTERACTION_FEATURES + VELOCITY_FEATURES)
NUM_FEATURES = len(ALL_FEATURES)
FEAT_IDX = {name: i for i, name in enumerate(ALL_FEATURES)}


# ═══════════════════════════════════════════════════════════════════════
# PART 1A: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _zone_center(zone):
    clat = (zone["lat_range"][0] + zone["lat_range"][1]) / 2
    clon = (zone["lon_range"][0] + zone["lon_range"][1]) / 2
    return clat, clon


def _get_zone_stations(zone, n=STATIONS_PER_ZONE):
    clat, clon = _zone_center(zone)
    dists = [(s["key"], _haversine(clat, clon, s["lat"], s["lon"])) for s in STATIONS]
    dists.sort(key=lambda x: x[1])
    return [d[0] for d in dists[:n]]


def _load_signal_hourly(conn, metric, hours, source=None):
    """Load a signal metric from DB and resample to hourly aligned to `hours`."""
    n = len(hours)
    if source:
        rows = conn.execute(
            "SELECT timestamp, value FROM samples WHERE metric = ? AND source = ? "
            "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp",
            (metric, source, hours[0].isoformat(), hours[-1].isoformat())
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT timestamp, value FROM samples WHERE metric = ? "
            "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp",
            (metric, hours[0].isoformat(), hours[-1].isoformat())
        ).fetchall()
    if not rows:
        return np.full(n, np.nan, dtype=np.float32)
    sdf = pd.DataFrame(rows, columns=["timestamp", "value"])
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], format="ISO8601", utc=True)
    sdf = sdf.groupby(sdf["timestamp"].dt.floor("h"))["value"].mean().reset_index()
    hours_ts = pd.DatetimeIndex(hours)
    merged = pd.DataFrame({"timestamp": hours_ts}).merge(sdf, on="timestamp", how="left")
    merged["value"] = merged["value"].ffill(limit=6).bfill(limit=6)
    return merged["value"].values[:n].astype(np.float32)


def _rolling_slope(values, window):
    """O(T) rolling linear regression slope using cumulative sums."""
    n = len(values)
    w = window
    S = w * (w - 1) / 2.0
    denom = w * w * (w * w - 1) / 12.0
    if denom == 0:
        return np.zeros(n, dtype=np.float32)
    v = np.nan_to_num(values.astype(np.float64), nan=0.0)
    cumsum_v = np.cumsum(v)
    s1 = np.zeros(n, dtype=np.float64)
    s1[:w] = cumsum_v[:w]
    s1[w:] = cumsum_v[w:] - cumsum_v[:-w]
    jv = np.arange(n, dtype=np.float64) * v
    cumsum_jv = np.cumsum(jv)
    u = np.zeros(n, dtype=np.float64)
    u[:w] = cumsum_jv[:w]
    u[w:] = cumsum_jv[w:] - cumsum_jv[:-w]
    p = np.arange(n, dtype=np.float64)
    ty = u - (p - w + 1) * s1
    slope = (w * ty - S * s1) / denom
    slope[:w - 1] = 0.0
    return slope.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# PART 1B: CATALOG FEATURES (sliding windows on earthquake catalog)
# ═══════════════════════════════════════════════════════════════════════

def _sliding_count(event_epochs, hour_epochs, window_sec, event_mask=None):
    n = len(hour_epochs)
    result = np.zeros(n, dtype=np.float32)
    left = right = 0
    count = 0
    ne = len(event_epochs)
    for i in range(n):
        h = hour_epochs[i]
        while right < ne and event_epochs[right] <= h:
            if event_mask is None or event_mask[right]:
                count += 1
            right += 1
        while left < right and event_epochs[left] < h - window_sec:
            if event_mask is None or event_mask[left]:
                count -= 1
            left += 1
        result[i] = max(0, count)
    return result


def _sliding_max_mag(event_epochs, event_mags, hour_epochs, window_sec):
    n = len(hour_epochs)
    result = np.zeros(n, dtype=np.float32)
    dq = deque()
    left = right = 0
    ne = len(event_epochs)
    for i in range(n):
        h = hour_epochs[i]
        while right < ne and event_epochs[right] <= h:
            while dq and event_mags[dq[-1]] <= event_mags[right]:
                dq.pop()
            dq.append(right)
            right += 1
        while dq and event_epochs[dq[0]] < h - window_sec:
            dq.popleft()
        result[i] = event_mags[dq[0]] if dq else 0.0
    return result


def _sliding_min_mag(event_epochs, event_mags, hour_epochs, window_sec):
    n = len(hour_epochs)
    result = np.full(n, 99.0, dtype=np.float32)
    dq = deque()
    left = right = 0
    ne = len(event_epochs)
    for i in range(n):
        h = hour_epochs[i]
        while right < ne and event_epochs[right] <= h:
            while dq and event_mags[dq[-1]] >= event_mags[right]:
                dq.pop()
            dq.append(right)
            right += 1
        while dq and event_epochs[dq[0]] < h - window_sec:
            dq.popleft()
        result[i] = event_mags[dq[0]] if dq else 0.0
    return result


def _sliding_energy(event_epochs, event_mags, hour_epochs, window_sec):
    n = len(hour_epochs)
    result = np.zeros(n, dtype=np.float32)
    left = right = 0
    energy_sum = 0.0
    ne = len(event_epochs)
    for i in range(n):
        h = hour_epochs[i]
        while right < ne and event_epochs[right] <= h:
            energy_sum += 10.0 ** (1.5 * event_mags[right])
            right += 1
        while left < right and event_epochs[left] < h - window_sec:
            energy_sum -= 10.0 ** (1.5 * event_mags[left])
            energy_sum = max(0.0, energy_sum)
            left += 1
        result[i] = np.log10(energy_sum + 1.0) if energy_sum > 0 else 0.0
    return result


def _sliding_b_value(event_epochs, event_mags, hour_epochs, window_sec, mc=2.5):
    n = len(hour_epochs)
    result = np.full(n, np.nan, dtype=np.float32)
    left = right = 0
    mag_sum = 0.0
    mag_count = 0
    ne = len(event_epochs)
    for i in range(n):
        h = hour_epochs[i]
        while right < ne and event_epochs[right] <= h:
            if event_mags[right] >= mc:
                mag_sum += event_mags[right]
                mag_count += 1
            right += 1
        while left < right and event_epochs[left] < h - window_sec:
            if event_mags[left] >= mc:
                mag_sum -= event_mags[left]
                mag_count -= 1
            left += 1
        mag_count = max(0, mag_count)
        if mag_count >= 15:
            mean_m = mag_sum / mag_count
            if mean_m > mc + 0.01:
                result[i] = 1.0 / (np.log(10) * (mean_m - mc))
    return result


def _sliding_moment(event_epochs, event_mags, hour_epochs, window_sec):
    n = len(hour_epochs)
    result = np.zeros(n, dtype=np.float32)
    left = right = 0
    moment_sum = 0.0
    ne = len(event_epochs)
    for i in range(n):
        h = hour_epochs[i]
        while right < ne and event_epochs[right] <= h:
            moment_sum += 10.0 ** (1.5 * event_mags[right] + 9.05)
            right += 1
        while left < right and event_epochs[left] < h - window_sec:
            moment_sum -= 10.0 ** (1.5 * event_mags[left] + 9.05)
            moment_sum = max(0.0, moment_sum)
            left += 1
        result[i] = np.log10(moment_sum) if moment_sum > 0 else 0.0
    return result


def _sliding_depth_mean(event_epochs, event_depths, hour_epochs, window_sec):
    n = len(hour_epochs)
    result = np.full(n, np.nan, dtype=np.float32)
    left = right = 0
    depth_sum = 0.0
    depth_count = 0
    ne = len(event_epochs)
    for i in range(n):
        h = hour_epochs[i]
        while right < ne and event_epochs[right] <= h:
            if event_depths[right] > 0:
                depth_sum += event_depths[right]
                depth_count += 1
            right += 1
        while left < right and event_epochs[left] < h - window_sec:
            if event_depths[left] > 0:
                depth_sum -= event_depths[left]
                depth_count -= 1
            left += 1
        depth_count = max(0, depth_count)
        if depth_count > 0:
            result[i] = depth_sum / depth_count
    return result


def _time_since_last(event_epochs, hour_epochs):
    n = len(hour_epochs)
    result = np.full(n, 720.0, dtype=np.float32)
    j = 0
    ne = len(event_epochs)
    for i in range(n):
        while j < ne and event_epochs[j] <= hour_epochs[i]:
            j += 1
        if j > 0:
            result[i] = (hour_epochs[i] - event_epochs[j - 1]) / 3600.0
    return result


def _sliding_deep_fraction(event_epochs, event_depths, hour_epochs, window_sec, depth_thresh=70.0):
    """Fraction of events deeper than threshold (subduction proxy)."""
    n = len(hour_epochs)
    result = np.zeros(n, dtype=np.float32)
    left = right = 0
    deep_count = 0
    total_count = 0
    ne = len(event_epochs)
    for i in range(n):
        h = hour_epochs[i]
        while right < ne and event_epochs[right] <= h:
            total_count += 1
            if event_depths[right] > depth_thresh:
                deep_count += 1
            right += 1
        while left < right and event_epochs[left] < h - window_sec:
            total_count -= 1
            if event_depths[left] > depth_thresh:
                deep_count -= 1
            left += 1
        total_count = max(0, total_count)
        deep_count = max(0, deep_count)
        result[i] = deep_count / max(1, total_count)
    return result


def build_zone_catalog_features(conn, zone, hour_epochs):
    """Build all catalog-derived features for one zone."""
    rows = conn.execute("""
        SELECT timestamp, magnitude, depth_km FROM earthquakes
        WHERE magnitude >= ?
        AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
        ORDER BY timestamp
    """, (MIN_MAG_CATALOG, zone["lat_range"][0], zone["lat_range"][1],
          zone["lon_range"][0], zone["lon_range"][1])).fetchall()

    n = len(hour_epochs)
    nf = len(CATALOG_FEATURES)
    feats = np.zeros((n, nf), dtype=np.float32)
    fi = {name: i for i, name in enumerate(CATALOG_FEATURES)}

    if not rows:
        return feats

    event_times, event_mags, event_depths = [], [], []
    for r in rows:
        try:
            t = pd.Timestamp(r[0], tz="UTC").timestamp()
            event_times.append(t)
            event_mags.append(float(r[1]))
            event_depths.append(float(r[2]) if r[2] else 0.0)
        except Exception:
            continue

    ee = np.array(event_times)
    em = np.array(event_mags)
    ed = np.array(event_depths)
    sort_idx = np.argsort(ee)
    ee, em, ed = ee[sort_idx], em[sort_idx], ed[sort_idx]

    H1 = 3600
    m4_mask = em >= 4.0

    # Counts
    feats[:, fi["eq_count_1h"]] = _sliding_count(ee, hour_epochs, 1 * H1)
    feats[:, fi["eq_count_6h"]] = _sliding_count(ee, hour_epochs, 6 * H1)
    feats[:, fi["eq_count_24h"]] = _sliding_count(ee, hour_epochs, 24 * H1)
    feats[:, fi["eq_count_72h"]] = _sliding_count(ee, hour_epochs, 72 * H1)
    feats[:, fi["eq_count_7d"]] = _sliding_count(ee, hour_epochs, 7 * 86400)
    feats[:, fi["eq_count_30d"]] = _sliding_count(ee, hour_epochs, 30 * 86400)
    feats[:, fi["m4_count_24h"]] = _sliding_count(ee, hour_epochs, 24 * H1, m4_mask)
    feats[:, fi["m4_count_7d"]] = _sliding_count(ee, hour_epochs, 7 * 86400, m4_mask)

    # Foreshock acceleration
    rate_6h = feats[:, fi["eq_count_6h"]]
    rate_24h = feats[:, fi["eq_count_24h"]]
    rate_prior_18h = np.maximum(rate_24h - rate_6h, 0)
    feats[:, fi["foreshock_accel"]] = rate_6h / (rate_prior_18h / 3.0 + 1.0)

    # Rate anomaly
    rate_7d = feats[:, fi["eq_count_7d"]] / 7.0
    rate_30d = feats[:, fi["eq_count_30d"]] / 30.0
    feats[:, fi["rate_anomaly_7d"]] = rate_7d / np.maximum(rate_30d, 0.01)

    # B-values
    b14 = _sliding_b_value(ee, em, hour_epochs, 14 * 86400)
    b30 = _sliding_b_value(ee, em, hour_epochs, 30 * 86400)
    median_b = np.nanmedian(b14)
    if np.isnan(median_b):
        median_b = 1.0
    b14_clean = np.where(np.isnan(b14), median_b, b14)
    b30_clean = np.where(np.isnan(b30), median_b, b30)
    feats[:, fi["b_value_14d"]] = b14_clean
    feats[:, fi["b_value_30d"]] = b30_clean
    feats[:, fi["b_delta"]] = b14_clean - b30_clean

    # B-value acceleration (trend of b-value over 4 weekly windows)
    b_weekly = []
    for w in range(4):
        bw = _sliding_b_value(ee, em, hour_epochs, 14 * 86400)
        b_weekly.append(bw)
    # Approximate as slope of (b14 - b30) — measures how fast b is changing
    feats[:, fi["b_value_accel"]] = _rolling_slope(b14_clean, 4 * 7 * 24)

    # Energy
    feats[:, fi["energy_24h"]] = _sliding_energy(ee, em, hour_epochs, 24 * H1)
    energy_48h = _sliding_energy(ee, em, hour_epochs, 48 * H1)
    energy_prior_24h = np.maximum(energy_48h - feats[:, fi["energy_24h"]], 0.01)
    feats[:, fi["energy_ratio"]] = feats[:, fi["energy_24h"]] / (energy_prior_24h + 0.01)

    # Magnitudes
    max72 = _sliding_max_mag(ee, em, hour_epochs, 72 * H1)
    max7d = _sliding_max_mag(ee, em, hour_epochs, 7 * 86400)
    min7d = _sliding_min_mag(ee, em, hour_epochs, 7 * 86400)
    feats[:, fi["max_mag_72h"]] = max72
    feats[:, fi["max_mag_7d"]] = max7d
    feats[:, fi["mag_range_7d"]] = np.where(max7d > 0, max7d - min7d, 0.0)

    # Moment
    feats[:, fi["moment_7d"]] = _sliding_moment(ee, em, hour_epochs, 7 * 86400)
    feats[:, fi["moment_30d"]] = _sliding_moment(ee, em, hour_epochs, 30 * 86400)
    moment_prior_7d = _sliding_moment(ee, em, hour_epochs, 14 * 86400) - feats[:, fi["moment_7d"]]
    moment_prior_7d = np.maximum(moment_prior_7d, 0.01)
    feats[:, fi["moment_accel"]] = feats[:, fi["moment_7d"]] / (moment_prior_7d + 0.01)

    # Time since last
    feats[:, fi["time_since_last"]] = _time_since_last(ee, hour_epochs)

    # Depth
    feats[:, fi["depth_mean_7d"]] = _sliding_depth_mean(ee, ed, hour_epochs, 7 * 86400)

    # Coulomb proxy: distance-decayed stress from M5+ events within 60d
    m5_mask = em >= 5.0
    m5_epochs = ee[m5_mask]
    m5_mags = em[m5_mask]
    clat, clon = _zone_center(zone)
    # Pre-compute moments for M5+ events
    m5_moments = np.array([10.0 ** (1.5 * m + 9.1) for m in m5_mags])
    # For each hour, sum M0/r^3 for M5+ events in last 60d
    coulomb = np.zeros(n, dtype=np.float32)
    # Get lat/lon for M5+ events from DB
    m5_rows = conn.execute("""
        SELECT timestamp, magnitude, lat, lon FROM earthquakes
        WHERE magnitude >= 5.0 AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
        ORDER BY timestamp
    """, (zone["lat_range"][0] - 2, zone["lat_range"][1] + 2,
          zone["lon_range"][0] - 2, zone["lon_range"][1] + 2)).fetchall()
    if m5_rows:
        m5e, m5m, m5lat, m5lon = [], [], [], []
        for r in m5_rows:
            try:
                t = pd.Timestamp(r[0], tz="UTC").timestamp()
                m5e.append(t)
                m5m.append(float(r[1]))
                m5lat.append(float(r[2]))
                m5lon.append(float(r[3]))
            except Exception:
                continue
        m5e = np.array(m5e)
        m5m_arr = np.array(m5m)
        m5lat_arr = np.array(m5lat)
        m5lon_arr = np.array(m5lon)
        m5_moment = 10.0 ** (1.5 * m5m_arr + 9.1)
        m5_r_km = np.array([max(1.0, _haversine(clat, clon, la, lo))
                            for la, lo in zip(m5lat_arr, m5lon_arr)])
        m5_contribution = m5_moment / m5_r_km ** 3
        # Sliding window: sum contributions in last 60d
        window_60d = 60 * 86400
        left = right = 0
        csum = 0.0
        ne_m5 = len(m5e)
        for i in range(n):
            h = hour_epochs[i]
            while right < ne_m5 and m5e[right] <= h:
                csum += m5_contribution[right]
                right += 1
            while left < right and m5e[left] < h - window_60d:
                csum -= m5_contribution[left]
                left += 1
            coulomb[i] = np.log10(max(csum, 0) + 1e-30) if csum > 0 else 0.0
    feats[:, fi["coulomb_proxy"]] = coulomb

    # Omori deficit (simplified: compare recent rate to expected after largest event)
    feats[:, fi["omori_deficit"]] = 0.0  # non-trivial to vectorize, leave as baseline

    # Subduction proxy
    feats[:, fi["subduction_proxy"]] = _sliding_deep_fraction(ee, ed, hour_epochs, 30 * 86400)

    # Foreshock curvature: 2nd derivative of 6h-binned seismicity rate
    # Approximate using second difference of eq_count_6h
    c6h = feats[:, fi["eq_count_6h"]]
    curv = np.zeros(n, dtype=np.float32)
    curv[12:] = c6h[12:] - 2 * c6h[6:-6] + c6h[:-12]
    feats[:, fi["foreshock_curvature"]] = curv

    return feats


# ═══════════════════════════════════════════════════════════════════════
# PART 1C: SIGNAL FEATURES (rolling stats on hourly time series)
# ═══════════════════════════════════════════════════════════════════════

def build_signal_features(conn, hours):
    """Build all signal-derived features: geomag, solar, tidal, cosmic, IEF, OLR,
    ground mag, trajectory, shape, coupling. Returns dict of feature arrays."""
    n = len(hours)

    # Load all raw signals
    print("    Loading signal time series...")
    raw = {}
    signal_loads = [
        ("kp", "kp_index", None), ("dst", "dst_index", None),
        ("sw_speed", "solar_wind_speed", None), ("sw_density", "solar_wind_density", None),
        ("imf_bz", "imf_bz_gsm", None), ("imf_bt", "imf_bt", None),
        ("cosmic", "neutron_count", None), ("ief", "ief", None),
        ("tidal_pot", "tidal_potential", None), ("tidal_strain", "tidal_strain_rate", None),
        ("tide", "water_level", None),
        ("olr", "olr", None),
        ("mag_x", "mag_x", "intermagnet_hist"), ("mag_y", "mag_y", "intermagnet_hist"),
        ("mag_z", "mag_z", "intermagnet_hist"),
    ]
    for key, metric, source in signal_loads:
        raw[key] = _load_signal_hourly(conn, metric, hours, source)

    result = {}

    # Helper: rolling stats
    def rs(series, window=72):
        s = pd.Series(series, dtype=np.float64)
        r = s.rolling(window, min_periods=1)
        return {
            "mean": r.mean().values.astype(np.float32),
            "max": r.max().values.astype(np.float32),
            "min": r.min().values.astype(np.float32),
            "std": r.std(ddof=0).values.astype(np.float32),
            "range": (r.max() - r.min()).values.astype(np.float32),
        }

    # Geomag
    kp_s = rs(raw["kp"])
    result["kp_mean_72h"] = kp_s["mean"]
    result["kp_max_72h"] = kp_s["max"]
    result["kp_trend"] = _rolling_slope(raw["kp"], 72)
    dst_s = rs(raw["dst"])
    result["dst_mean_72h"] = dst_s["mean"]
    result["dst_min_72h"] = dst_s["min"]
    diff_dst = np.abs(np.diff(np.nan_to_num(raw["dst"], nan=0.0), prepend=0))
    result["dst_rate"] = pd.Series(diff_dst).rolling(72, min_periods=1).max().values.astype(np.float32)
    result["dst_volatility"] = dst_s["std"]

    # Solar
    sw_s = rs(raw["sw_speed"])
    result["sw_speed_mean_72h"] = sw_s["mean"]
    result["sw_speed_max_72h"] = sw_s["max"]
    swd_s = rs(raw["sw_density"])
    result["sw_density_mean_72h"] = swd_s["mean"]
    result["sw_density_max_72h"] = swd_s["max"]
    bz_s = rs(raw["imf_bz"])
    result["imf_bz_mean_72h"] = bz_s["mean"]
    result["imf_bz_min_72h"] = bz_s["min"]
    bt_s = rs(raw["imf_bt"])
    result["imf_bt_mean_72h"] = bt_s["mean"]
    result["imf_bt_max_72h"] = bt_s["max"]

    # Tidal
    tp_s = rs(raw["tidal_pot"])
    result["tidal_potential_72h"] = tp_s["mean"]
    ts_s = rs(raw["tidal_strain"])
    result["tidal_strain_max_72h"] = ts_s["max"]
    result["tidal_range_72h"] = tp_s["range"]
    tide_s = rs(raw["tide"])
    result["tide_range_72h"] = tide_s["range"]
    diff_tide = np.abs(np.diff(np.nan_to_num(raw["tide"], nan=0.0), prepend=0))
    result["tide_rate_max"] = pd.Series(diff_tide).rolling(72, min_periods=1).max().values.astype(np.float32)

    # Cosmic
    cos_s = rs(raw["cosmic"])
    result["cosmic_mean_72h"] = cos_s["mean"]
    result["cosmic_trend_72h"] = _rolling_slope(raw["cosmic"], 72)
    cos_30d = rs(raw["cosmic"], 720)
    anomaly_denom = np.where(cos_30d["std"] > 0, cos_30d["std"], 1.0)
    result["cosmic_anomaly"] = ((cos_s["mean"] - cos_30d["mean"]) / anomaly_denom).astype(np.float32)

    # IEF
    ief_s = rs(raw["ief"])
    result["ief_mean_72h"] = ief_s["mean"]
    result["ief_max_72h"] = ief_s["max"]
    result["ief_volatility"] = ief_s["std"]

    # OLR
    olr_s = rs(raw["olr"], 168)
    result["olr_mean_7d"] = olr_s["mean"]
    result["olr_trend"] = _rolling_slope(raw["olr"], 168)
    olr_30d = rs(raw["olr"], 720)
    olr_denom = np.where(olr_30d["std"] > 0, olr_30d["std"], 1.0)
    result["olr_anomaly"] = ((olr_s["mean"] - olr_30d["mean"]) / olr_denom).astype(np.float32)

    # Ground magnetometer
    for comp in ["x", "y", "z"]:
        key = f"mag_{comp}"
        ms = rs(raw[key])
        result[f"mag_{comp}_mean_72h"] = ms["mean"]
        result[f"mag_{comp}_range_72h"] = ms["range"]
        result[f"mag_{comp}_trend"] = _rolling_slope(raw[key], 72)
    diff_x = np.abs(np.diff(np.nan_to_num(raw["mag_x"], nan=0.0), prepend=0))
    diff_y = np.abs(np.diff(np.nan_to_num(raw["mag_y"], nan=0.0), prepend=0))
    diff_z = np.abs(np.diff(np.nan_to_num(raw["mag_z"], nan=0.0), prepend=0))
    dbdt = np.maximum(np.maximum(diff_x, diff_y), diff_z)
    result["mag_dbdt_max"] = pd.Series(dbdt).rolling(72, min_periods=1).max().values.astype(np.float32)

    # Trajectory (multi-timescale sub-windows)
    for prefix, key in [("dst", "dst"), ("sw", "sw_speed")]:
        s = raw[key]
        s24 = rs(s, 24)
        s6 = rs(s, 6)
        result[f"{prefix}_mean_24h"] = s24["mean"]
        result[f"{prefix}_mean_6h"] = s6["mean"]
        result[f"{prefix}_slope_24h"] = _rolling_slope(s, 24)
        result[f"{prefix}_slope_6h"] = _rolling_slope(s, 6)
        slope_24 = _rolling_slope(s, 24)
        slope_prior = np.roll(_rolling_slope(s, 48), 24)
        result[f"{prefix}_ramp"] = (slope_24 - slope_prior).astype(np.float32)

    # Shape features (curvature + percentiles)
    for prefix, key in [("dst", "dst"), ("sw", "sw_speed"), ("kp", "kp")]:
        s = pd.Series(np.nan_to_num(raw[key], nan=0.0), dtype=np.float64)
        r72 = s.rolling(72, min_periods=10)
        result[f"{prefix}_curvature_72h"] = r72.apply(
            lambda x: 2 * np.polyfit(np.arange(len(x)), x, 2)[0] if len(x) >= 5 else 0.0,
            raw=False
        ).values.astype(np.float32)
        result[f"{prefix}_p10_72h"] = r72.quantile(0.1).values.astype(np.float32)
        result[f"{prefix}_p90_72h"] = r72.quantile(0.9).values.astype(np.float32)

    # Coupling features
    sw_arr = np.nan_to_num(raw["sw_speed"], nan=0.0)
    sw_24h = pd.Series(sw_arr).rolling(24, min_periods=1).mean().values
    sw_prior_48h = pd.Series(sw_arr).rolling(72, min_periods=1).mean().values
    result["sw_speed_accel"] = np.where(sw_prior_48h > 0, sw_24h / np.maximum(sw_prior_48h, 1.0), 1.0).astype(np.float32)
    sw_72 = pd.Series(sw_arr).rolling(72, min_periods=1)
    result["sw_above_483"] = sw_72.apply(lambda x: np.mean(x >= 483), raw=True).values.astype(np.float32)
    result["sw_above_567"] = sw_72.apply(lambda x: np.mean(x >= 567), raw=True).values.astype(np.float32)
    sw_mean_72 = sw_s["mean"]
    result["sw_peak_to_mean"] = np.where(sw_mean_72 > 0, sw_s["max"] / np.maximum(sw_mean_72, 1.0), 1.0).astype(np.float32)

    kp_arr = np.nan_to_num(raw["kp"], nan=0.0)
    result["kp_spike"] = (kp_s["max"] - kp_s["mean"]).astype(np.float32)
    kp_72 = pd.Series(kp_arr).rolling(72, min_periods=1)
    result["kp_hours_ge3"] = kp_72.apply(lambda x: np.mean(x >= 3.0), raw=True).values.astype(np.float32)

    result["dst_drop_rate"] = (dst_s["min"] - dst_s["max"]).astype(np.float32)

    bz_arr = np.nan_to_num(raw["imf_bz"], nan=0.0)
    bz_72 = pd.Series(bz_arr).rolling(72, min_periods=1)
    result["bz_southward_frac"] = bz_72.apply(lambda x: np.mean(x < 0), raw=True).values.astype(np.float32)

    sw_z = (sw_mean_72 - 428.0) / 97.0
    kp_z = (kp_s["mean"] - 1.4) / 1.4
    result["storm_coupling_idx"] = (sw_z * kp_z).astype(np.float32)

    # Store raw signals for later use in interactions
    result["_raw"] = raw

    return result


# ═══════════════════════════════════════════════════════════════════════
# PART 1D: DART BUOY FEATURES
# ═══════════════════════════════════════════════════════════════════════

def _get_nearest_dart(zone, conn):
    """Find nearest DART buoy to zone center with >10K readings."""
    from lab.dart_ingest import DART_STATIONS
    clat, clon = _zone_center(zone)
    best = None
    best_dist = 9999
    for sid, info in DART_STATIONS.items():
        d = _haversine(clat, clon, info["lat"], info["lon"])
        if d < best_dist:
            cnt = conn.execute("SELECT COUNT(*) FROM dart_readings WHERE station_id = ?",
                               (sid,)).fetchone()[0]
            if cnt > 10000:
                best_dist = d
                best = sid
    return best, best_dist


def build_zone_dart_features(conn, zone, hours):
    """Build DART buoy features for one zone from nearest buoy."""
    n = len(hours)
    nf = len(DART_FEATURES)
    feats = np.full((n, nf), np.nan, dtype=np.float32)
    fi = {name: i for i, name in enumerate(DART_FEATURES)}

    try:
        station_id, dist = _get_nearest_dart(zone, conn)
    except Exception:
        return feats

    if not station_id or dist > 3000:
        return feats

    rows = conn.execute("""
        SELECT timestamp, height_m, mode FROM dart_readings
        WHERE station_id = ? AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (station_id, hours[0].isoformat(), hours[-1].isoformat())).fetchall()

    if len(rows) < 100:
        return feats

    df = pd.DataFrame(rows, columns=["timestamp", "height_m", "mode"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df["hour"] = df["timestamp"].dt.floor("h")

    # Detrend: subtract 48h rolling median
    hourly = df.groupby("hour").agg(
        height_mean=("height_m", "mean"),
        height_std=("height_m", "std"),
        event_mode=("mode", lambda x: (x > 1).sum() if hasattr(x, '__iter__') else 0),
    ).reset_index()

    hours_ts = pd.DatetimeIndex(hours)
    merged = pd.DataFrame({"hour": hours_ts}).merge(hourly, on="hour", how="left")
    h = merged["height_mean"].values.astype(np.float64)
    h_std = merged["height_std"].values.astype(np.float64)
    em_count = merged["event_mode"].fillna(0).values.astype(np.float64)

    # Detrend
    baseline = pd.Series(h).rolling(48, min_periods=6, center=True).median().values
    residual = h - np.nan_to_num(baseline, nan=0.0)

    s_res = pd.Series(residual)
    feats[:, fi["dart_residual_mean_24h"]] = s_res.rolling(24, min_periods=1).mean().values.astype(np.float32)
    feats[:, fi["dart_residual_std_24h"]] = s_res.rolling(24, min_periods=1).std().values.astype(np.float32)
    feats[:, fi["dart_residual_trend_48h"]] = _rolling_slope(residual.astype(np.float32), 48)
    feats[:, fi["dart_rate_of_change_6h"]] = _rolling_slope(residual.astype(np.float32), 6)

    # Loading index: max - min of residual over 7d
    r7d = s_res.rolling(168, min_periods=1)
    feats[:, fi["dart_loading_index"]] = (r7d.max() - r7d.min()).values.astype(np.float32)

    # Trend reversal: days 3-2 trend vs days 1-0 trend
    slope_recent = _rolling_slope(residual.astype(np.float32), 48)
    slope_prior = np.roll(_rolling_slope(residual.astype(np.float32), 96), 48)
    feats[:, fi["dart_trend_reversal"]] = (slope_recent - slope_prior).astype(np.float32)

    # Event mode counts
    s_em = pd.Series(em_count)
    feats[:, fi["dart_event_mode_count_7d"]] = s_em.rolling(168, min_periods=1).sum().values.astype(np.float32)
    feats[:, fi["dart_event_mode_24h"]] = s_em.rolling(24, min_periods=1).sum().values.astype(np.float32)
    feats[:, fi["dart_event_mode_12h"]] = s_em.rolling(12, min_periods=1).sum().values.astype(np.float32)
    feats[:, fi["dart_event_mode_6h"]] = s_em.rolling(6, min_periods=1).sum().values.astype(np.float32)

    # EM acceleration
    em12 = feats[:, fi["dart_event_mode_12h"]]
    em_prior = np.roll(em12, 12)
    feats[:, fi["dart_em_accel"]] = em12 / np.maximum(em_prior, 1.0)

    # Volatility features
    s_hstd = pd.Series(np.nan_to_num(h_std, nan=0.0))
    std_6h = s_hstd.rolling(6, min_periods=1).mean().values
    std_12h = s_hstd.rolling(12, min_periods=1).mean().values
    std_18h_prior = np.roll(s_hstd.rolling(18, min_periods=1).mean().values, 6)
    std_24h_prior = np.roll(s_hstd.rolling(24, min_periods=1).mean().values, 12)
    feats[:, fi["dart_std_6h"]] = std_6h.astype(np.float32)
    feats[:, fi["dart_std_12h"]] = std_12h.astype(np.float32)
    feats[:, fi["dart_vol_ramp_6v18"]] = (std_6h / np.maximum(std_18h_prior, 1e-6)).astype(np.float32)
    feats[:, fi["dart_vol_ramp_12v36"]] = (std_12h / np.maximum(std_24h_prior, 1e-6)).astype(np.float32)

    feats[:, fi["dart_trend_12h"]] = _rolling_slope(residual.astype(np.float32), 12)
    feats[:, fi["dart_trend_6h"]] = _rolling_slope(residual.astype(np.float32), 6)
    trend_12 = feats[:, fi["dart_trend_12h"]]
    trend_prior = np.roll(trend_12, 12)
    feats[:, fi["dart_trend_accel"]] = (trend_12 - trend_prior).astype(np.float32)

    # Deepened loading trajectory — loading index is a top escalation discriminator
    loading = pd.Series(np.nan_to_num(feats[:, fi["dart_loading_index"]], nan=0.0))
    load_rate = loading - loading.shift(24)
    feats[:, fi["dart_loading_rate_24h"]] = load_rate.values.astype(np.float32)
    feats[:, fi["dart_loading_max_72h"]] = loading.rolling(72, min_periods=1).max().values.astype(np.float32)
    feats[:, fi["dart_loading_accel"]] = (load_rate - load_rate.shift(24)).values.astype(np.float32)

    # Interaction placeholders (filled in build_full_dataset)
    feats[:, fi["dart_x_coulomb"]] = 0.0
    feats[:, fi["dart_x_sw"]] = 0.0

    return feats


# ═══════════════════════════════════════════════════════════════════════
# PART 1E: CME/FLARE/STORM EVENT FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_event_features(conn, hours, hour_epochs):
    """Build CME, flare, and storm features (global, shared across zones)."""
    n = len(hour_epochs)

    # Load CME events
    cme_rows = conn.execute("""
        SELECT timestamp, value FROM samples
        WHERE source = 'nasa_donki' AND metric = 'cme_speed'
        AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (hours[0].isoformat(), hours[-1].isoformat())).fetchall()

    cme_epochs, cme_speeds = [], []
    for r in cme_rows:
        try:
            t = pd.Timestamp(r[0], tz="UTC").timestamp()
            cme_epochs.append(t)
            cme_speeds.append(float(r[1]))
        except Exception:
            continue
    cme_epochs = np.array(cme_epochs) if cme_epochs else np.array([])
    cme_speeds = np.array(cme_speeds) if cme_speeds else np.array([])

    result = {}

    H1 = 3600
    if len(cme_epochs) > 0:
        result["cme_count_72h"] = _sliding_count(cme_epochs, hour_epochs, 72 * H1)
        result["cme_count_7d"] = _sliding_count(cme_epochs, hour_epochs, 7 * 86400)
        result["cme_speed_max_72h"] = _sliding_max_mag(cme_epochs, cme_speeds, hour_epochs, 72 * H1)
        cme_mean = _sliding_energy(cme_epochs, np.log10(np.maximum(cme_speeds, 1)) / 1.5, hour_epochs, 72 * H1)
        result["cme_speed_mean_72h"] = np.where(result["cme_count_72h"] > 0, cme_mean, 0.0).astype(np.float32)
        result["cme_speed_max_7d"] = _sliding_max_mag(cme_epochs, cme_speeds, hour_epochs, 7 * 86400)
        fast_mask = cme_speeds > 1000
        result["cme_fast_count_7d"] = _sliding_count(cme_epochs, hour_epochs, 7 * 86400, fast_mask)
        c72 = result["cme_count_72h"]
        c_prior = np.roll(c72, 72)
        result["cme_accel"] = np.where(c_prior > 0, c72 / np.maximum(c_prior, 1.0), 0.0).astype(np.float32)
    else:
        for f in CME_FEATURES:
            result[f] = np.zeros(n, dtype=np.float32)

    # Load flare events
    flare_rows = conn.execute("""
        SELECT timestamp, value FROM samples
        WHERE source = 'nasa_donki' AND metric = 'solar_flare'
        AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (hours[0].isoformat(), hours[-1].isoformat())).fetchall()

    flare_epochs, flare_classes = [], []
    for r in flare_rows:
        try:
            t = pd.Timestamp(r[0], tz="UTC").timestamp()
            flare_epochs.append(t)
            flare_classes.append(float(r[1]))
        except Exception:
            continue
    flare_epochs = np.array(flare_epochs) if flare_epochs else np.array([])
    flare_classes = np.array(flare_classes) if flare_classes else np.array([])

    if len(flare_epochs) > 0:
        result["flare_count_72h"] = _sliding_count(flare_epochs, hour_epochs, 72 * H1)
        result["flare_count_7d"] = _sliding_count(flare_epochs, hour_epochs, 7 * 86400)
        result["flare_max_class_72h"] = _sliding_max_mag(flare_epochs, flare_classes, hour_epochs, 72 * H1)
        result["flare_max_class_7d"] = _sliding_max_mag(flare_epochs, flare_classes, hour_epochs, 7 * 86400)
        m_plus = flare_classes >= 4.0
        result["flare_m_plus_count_7d"] = _sliding_count(flare_epochs, hour_epochs, 7 * 86400, m_plus)
        result["flare_energy_72h"] = _sliding_energy(flare_epochs, flare_classes, hour_epochs, 72 * H1)
    else:
        for f in FLARE_FEATURES:
            result[f] = np.zeros(n, dtype=np.float32)

    # Storm features (from Kp >= 5)
    kp_rows = conn.execute("""
        SELECT timestamp, value FROM samples WHERE metric = 'kp_index'
        AND timestamp >= ? AND timestamp < ? ORDER BY timestamp
    """, (hours[0].isoformat(), hours[-1].isoformat())).fetchall()

    storm_epochs, storm_kp = [], []
    for r in kp_rows:
        try:
            v = float(r[1])
            if v >= 5.0:
                t = pd.Timestamp(r[0], tz="UTC").timestamp()
                storm_epochs.append(t)
                storm_kp.append(v)
        except Exception:
            continue
    storm_epochs = np.array(storm_epochs) if storm_epochs else np.array([])
    storm_kp = np.array(storm_kp) if storm_kp else np.array([])

    if len(storm_epochs) > 0:
        result["storm_kp_max_72h"] = _sliding_max_mag(storm_epochs, storm_kp, hour_epochs, 72 * H1)
        result["storm_kp_max_7d"] = _sliding_max_mag(storm_epochs, storm_kp, hour_epochs, 7 * 86400)
        result["storm_count_7d"] = _sliding_count(storm_epochs, hour_epochs, 7 * 86400)
        severe = storm_kp >= 7.0
        result["storm_severe_7d"] = _sliding_count(storm_epochs, hour_epochs, 7 * 86400, severe)
    else:
        for f in STORM_FEATURES:
            result[f] = np.zeros(n, dtype=np.float32)

    return result


# ═══════════════════════════════════════════════════════════════════════
# PART 1F: TIDAL TRIGGERING (Schuster test)
# ═══════════════════════════════════════════════════════════════════════

def build_zone_tidal_triggering(conn, zone, hour_epochs, tidal_times, tidal_values):
    """Compute rolling Schuster test statistics per zone using O(T+E) sliding windows."""
    n = len(hour_epochs)
    nf = len(TIDAL_TRIGGER_FEATURES)
    feats = np.full((n, nf), np.nan, dtype=np.float32)
    fi = {name: i for i, name in enumerate(TIDAL_TRIGGER_FEATURES)}

    if len(tidal_times) == 0:
        return feats

    # Load zone events
    def load_events(min_mag, max_mag=None):
        if max_mag:
            rows = conn.execute("""
                SELECT timestamp FROM earthquakes
                WHERE magnitude >= ? AND magnitude < ?
                AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
                ORDER BY timestamp
            """, (min_mag, max_mag, zone["lat_range"][0], zone["lat_range"][1],
                  zone["lon_range"][0], zone["lon_range"][1])).fetchall()
        else:
            rows = conn.execute("""
                SELECT timestamp FROM earthquakes
                WHERE magnitude >= ?
                AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
                ORDER BY timestamp
            """, (min_mag, zone["lat_range"][0], zone["lat_range"][1],
                  zone["lon_range"][0], zone["lon_range"][1])).fetchall()

        epochs, phases = [], []
        for (ts,) in rows:
            try:
                t = pd.Timestamp(ts, tz="UTC").timestamp()
                idx = np.searchsorted(tidal_times, t)
                if 0 < idx < len(tidal_values):
                    v0, v1 = tidal_values[idx - 1], tidal_values[idx]
                    derivative = v1 - v0
                    t0, t1 = tidal_times[idx - 1], tidal_times[idx]
                    frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.5
                    value = v0 + frac * (v1 - v0)
                    phase = math.atan2(derivative, value)
                    epochs.append(t)
                    phases.append(phase)
            except Exception:
                continue
        return np.array(epochs), np.array(phases)

    def sliding_schuster(event_epochs, event_phases, window_sec):
        """O(T+E) sliding Schuster test."""
        rn = len(hour_epochs)
        p_out = np.full(rn, 1.0, dtype=np.float32)
        r_out = np.zeros(rn, dtype=np.float32)
        phase_out = np.zeros(rn, dtype=np.float32)

        if len(event_epochs) == 0:
            return p_out, r_out, phase_out

        left = right = 0
        cos_sum = 0.0
        sin_sum = 0.0
        count = 0
        ne = len(event_epochs)

        for i in range(rn):
            h = hour_epochs[i]
            while right < ne and event_epochs[right] <= h:
                cos_sum += math.cos(event_phases[right])
                sin_sum += math.sin(event_phases[right])
                count += 1
                right += 1
            while left < right and event_epochs[left] < h - window_sec:
                cos_sum -= math.cos(event_phases[left])
                sin_sum -= math.sin(event_phases[left])
                count -= 1
                left += 1
            count = max(0, count)
            if count >= 10:
                R = math.sqrt(cos_sum ** 2 + sin_sum ** 2) / count
                D_sq = (cos_sum ** 2 + sin_sum ** 2) / count
                p_out[i] = math.exp(-D_sq)
                r_out[i] = R
                phase_out[i] = math.atan2(sin_sum, cos_sum)
        return p_out, r_out, phase_out

    # All magnitudes
    ee_all, ep_all = load_events(2.5)
    p30, R30, ph30 = sliding_schuster(ee_all, ep_all, 30 * 86400)
    p90, R90, _ = sliding_schuster(ee_all, ep_all, 90 * 86400)

    feats[:, fi["tidal_schuster_p_30d"]] = p30
    feats[:, fi["tidal_schuster_p_90d"]] = p90
    feats[:, fi["tidal_schuster_R_30d"]] = R30
    feats[:, fi["tidal_schuster_R_90d"]] = R90
    feats[:, fi["tidal_mean_phase_30d"]] = ph30
    feats[:, fi["tidal_sensitivity_flag"]] = (p30 < 0.05).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        onset = np.where((p90 > 0) & (p30 > 0), np.log10(p90) - np.log10(p30), 0.0)
    feats[:, fi["tidal_sensitivity_onset"]] = onset.astype(np.float32)

    # Small (M2.5-4) and mid (M4-5)
    ee_s, ep_s = load_events(2.5, 4.0)
    ee_m, ep_m = load_events(4.0, 5.0)
    _, R30s, _ = sliding_schuster(ee_s, ep_s, 30 * 86400)
    _, R30m, _ = sliding_schuster(ee_m, ep_m, 30 * 86400)
    _, R90s, _ = sliding_schuster(ee_s, ep_s, 90 * 86400)
    _, R90m, _ = sliding_schuster(ee_m, ep_m, 90 * 86400)
    p30s, _, _ = sliding_schuster(ee_s, ep_s, 30 * 86400)
    p30m, _, _ = sliding_schuster(ee_m, ep_m, 30 * 86400)

    feats[:, fi["tidal_R_30d_small"]] = R30s
    feats[:, fi["tidal_R_30d_mid"]] = R30m
    feats[:, fi["tidal_R_90d_small"]] = R90s
    feats[:, fi["tidal_R_90d_mid"]] = R90m
    feats[:, fi["tidal_p_30d_small"]] = p30s
    feats[:, fi["tidal_p_30d_mid"]] = p30m
    feats[:, fi["tidal_small_vs_mid_R"]] = np.where(R30m > 0.001, R30s / np.maximum(R30m, 0.001),
                                                     np.where(R30s > 0.01, 10.0, 1.0)).astype(np.float32)
    feats[:, fi["tidal_mag_divergence"]] = (R30s - R30m).astype(np.float32)

    # ── Deepened: tidal-sensitivity trajectory (is the swarm becoming tidally synced?) ──
    R30m_s = pd.Series(R30m)
    rate = R30m_s - R30m_s.shift(168)               # 7-day change in mid-band R
    feats[:, fi["tidal_R_mid_rate_30d"]] = rate.values.astype(np.float32)
    feats[:, fi["tidal_R_mid_accel"]] = (rate - rate.shift(168)).values.astype(np.float32)

    # ── Deepened: instantaneous tidal stress at each hour (interp of tidal potential) ──
    if len(tidal_times) > 1:
        stress = np.abs(np.interp(hour_epochs, tidal_times, tidal_values,
                                  left=np.nan, right=np.nan)).astype(np.float32)
        s_stress = pd.Series(stress)
        feats[:, fi["tidal_stress_now"]] = stress
        feats[:, fi["tidal_stress_max_24h"]] = s_stress.rolling(24, min_periods=1).max().values.astype(np.float32)
        feats[:, fi["tidal_stress_rate"]] = s_stress.diff().values.astype(np.float32)

    return feats


# ═══════════════════════════════════════════════════════════════════════
# PART 1G: VOLCANIC & FIRMS FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_zone_volcanic_features(conn, zone):
    """Static + slow-changing volcanic features for a zone."""
    clat, clon = _zone_center(zone)
    nf = len(VOLCANIC_FEATURES)
    fi = {name: i for i, name in enumerate(VOLCANIC_FEATURES)}
    feats = np.zeros(nf, dtype=np.float32)

    try:
        volcanoes = conn.execute(
            "SELECT volcano_number, lat, lon, last_eruption_year FROM volcanoes "
            "WHERE ABS(lat - ?) < 10 AND ABS(lon - ?) < 10", (clat, clon)
        ).fetchall()
    except Exception:
        feats[fi["volcano_dist_nearest"]] = 9999.0
        return feats

    if not volcanoes:
        feats[fi["volcano_dist_nearest"]] = 9999.0
        return feats

    dists = []
    for v in volcanoes:
        d = _haversine(clat, clon, v[1], v[2])
        dists.append((d, v))
    dists.sort(key=lambda x: x[0])

    feats[fi["volcano_dist_nearest"]] = dists[0][0]
    feats[fi["volcano_count_200km"]] = sum(1 for d, _ in dists if d < 200)
    feats[fi["volcano_count_500km"]] = sum(1 for d, _ in dists if d < 500)
    feats[fi["volcano_active_200km"]] = sum(
        1 for d, v in dists if d < 200 and v[3] and v[3] >= 2000)

    # Eruption features (use recent data)
    nearby_vnums = [v[0] for d, v in dists if d < 500]
    if nearby_vnums:
        placeholders = ",".join("?" * len(nearby_vnums))
        eruptions = conn.execute(
            f"SELECT volcano_number, vei, start_date, continuing "
            f"FROM eruptions WHERE volcano_number IN ({placeholders}) "
            f"AND start_year >= 2020", nearby_vnums
        ).fetchall()
        feats[fi["eruption_nearby_90d"]] = len(eruptions)
        feats[fi["eruption_vei_max_90d"]] = max((e[1] or 0 for e in eruptions), default=0)
        feats[fi["eruption_ongoing_500km"]] = sum(1 for e in eruptions if e[3])

    # Volcanic stress index
    stress = 0.0
    for d, v in dists:
        if d < 1000 and v[3] and v[3] >= 1900:
            stress += 1.0 / max(d, 1.0)
    feats[fi["volcanic_stress_index"]] = stress

    return feats


def build_zone_volcanic_dynamic(conn, zone, hour_epochs):
    """Time-varying volcanic unrest from the eruption timeline (vs the static
    regional prior above). Eruptions starting / active near the cell as the swarm
    evolves — a genuine dynamic volcanic-escalation signal."""
    n = len(hour_epochs)
    fi = {name: i for i, name in enumerate(VOLCANIC_DYN_FEATURES)}
    feats = np.zeros((n, len(VOLCANIC_DYN_FEATURES)), dtype=np.float32)
    clat, clon = _zone_center(zone)

    try:
        vols = conn.execute(
            "SELECT volcano_number, lat, lon FROM volcanoes "
            "WHERE ABS(lat - ?) < 8 AND ABS(lon - ?) < 8", (clat, clon)).fetchall()
    except Exception:
        return feats
    near = [v[0] for v in vols if _haversine(clat, clon, v[1], v[2]) < 500]
    if not near:
        return feats

    ph = ",".join("?" * len(near))
    erupts = conn.execute(
        f"SELECT start_date, end_date, vei, continuing FROM eruptions "
        f"WHERE volcano_number IN ({ph})", near).fetchall()

    def _parse(d):
        if not d:
            return None
        try:
            return pd.to_datetime(str(d)[:8], format="%Y%m%d", utc=True).timestamp()
        except Exception:
            return None

    starts = []
    spans = []  # (start_ep, end_ep, vei)
    data_end = hour_epochs[-1]
    for sd, ed, vei, cont in erupts:
        s = _parse(sd)
        if s is None:
            continue
        e = _parse(ed)
        if e is None:
            e = data_end if cont else s + 30 * 86400  # ongoing -> to now; else ~1 month
        starts.append(s)
        spans.append((s, max(e, s), int(vei) if vei is not None else 0))

    if starts:
        starts = np.sort(np.array(starts))
        feats[:, fi["volc_onset_90d"]] = (np.searchsorted(starts, hour_epochs, "right")
                                          - np.searchsorted(starts, hour_epochs - 90 * 86400, "left"))
        feats[:, fi["volc_onset_365d"]] = (np.searchsorted(starts, hour_epochs, "right")
                                           - np.searchsorted(starts, hour_epochs - 365 * 86400, "left"))
    for s, e, vei in spans:
        lo = np.searchsorted(hour_epochs, s, "left"); hi = np.searchsorted(hour_epochs, e, "right")
        if hi > lo:
            feats[lo:min(hi, n), fi["volc_erupt_active"]] += 1.0
        lo2 = np.searchsorted(hour_epochs, s, "left"); hi2 = np.searchsorted(hour_epochs, s + 365 * 86400, "right")
        if min(hi2, n) > lo2:
            idx = np.arange(lo2, min(hi2, n))
            np.maximum.at(feats[:, fi["volc_vei_max_365d"]], idx, float(vei))
    return feats


def build_zone_firms_features(conn, zone, hour_epochs):
    """FIRMS thermal hotspot features per zone."""
    n = len(hour_epochs)
    nf = len(FIRMS_FEATURES)
    feats = np.zeros((n, nf), dtype=np.float32)
    fi = {name: i for i, name in enumerate(FIRMS_FEATURES)}

    clat, clon = _zone_center(zone)
    lat_lo, lat_hi = clat - 1.8, clat + 1.8
    lon_lo, lon_hi = clon - 1.8, clon + 1.8

    try:
        rows = conn.execute("""
            SELECT acq_date, frp FROM thermal_anomalies
            WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
            ORDER BY acq_date
        """, (lat_lo, lat_hi, lon_lo, lon_hi)).fetchall()
    except Exception:
        return feats

    if not rows:
        return feats

    event_epochs_list, frps = [], []
    for r in rows:
        try:
            t = pd.Timestamp(r[0], tz="UTC").timestamp()
            event_epochs_list.append(t)
            frps.append(float(r[1]) if r[1] else 0.0)
        except Exception:
            continue

    if not event_epochs_list:
        return feats

    he = np.array(event_epochs_list)
    hf = np.array(frps)
    sort_idx = np.argsort(he)
    he, hf = he[sort_idx], hf[sort_idx]

    feats[:, fi["hotspot_count_7d"]] = _sliding_count(he, hour_epochs, 7 * 86400)
    feats[:, fi["hotspot_count_30d"]] = _sliding_count(he, hour_epochs, 30 * 86400)
    feats[:, fi["hotspot_frp_max_7d"]] = _sliding_max_mag(he, hf, hour_epochs, 7 * 86400)

    c7 = feats[:, fi["hotspot_count_7d"]]
    c_prior = feats[:, fi["hotspot_count_30d"]] - c7
    feats[:, fi["hotspot_accel"]] = np.where(c_prior > 0, c7 / np.maximum(c_prior / (23.0 / 7.0), 1.0), 0.0).astype(np.float32)

    return feats


# ═══════════════════════════════════════════════════════════════════════
# PART 1H: STATION FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_zone_station_features(conn, zone, hours):
    """Seismic features from nearby broadband stations.

    Computes instantaneous aggregates, sustained-activity trajectories,
    per-station amplitude anomalies, and multi-station coherence — the
    sustained/coherent patterns are where the precursor signal lives, not
    the single-hour snapshot.
    """
    station_keys = _get_zone_stations(zone)
    n = len(hours)
    fi = {name: i for i, name in enumerate(STATION_FEATURES)}
    feats = np.full((n, len(STATION_FEATURES)), np.nan, dtype=np.float32)
    hours_ts = pd.DatetimeIndex(hours)
    placeholders = ",".join("?" * len(station_keys))

    rows = conn.execute(f"""
        SELECT timestamp, station, amp_max, amp_mean, sta_lta_ratio, triggered
        FROM station_metrics WHERE station IN ({placeholders})
        AND timestamp >= ? AND timestamp < ? ORDER BY timestamp
    """, (*station_keys, hours[0].isoformat(), hours[-1].isoformat())).fetchall()

    if not rows:
        return feats

    df = pd.DataFrame(rows, columns=["timestamp", "station", "amp_max", "amp_mean",
                                      "sta_lta_ratio", "triggered"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df["hour"] = df["timestamp"].dt.floor("h")

    # ── Per-station hourly aggregation (for coherence + per-station baselines) ──
    per_stn = df.groupby(["hour", "station"]).agg(
        amp_max=("amp_max", "max"),
        amp_mean=("amp_mean", "mean"),
        sta_lta_max=("sta_lta_ratio", "max"),
        triggered=("triggered", "sum"),
    ).reset_index()

    # ── Cross-station instantaneous aggregates ──
    hourly = per_stn.groupby("hour").agg(
        amp_max=("amp_max", "max"),
        amp_mean=("amp_mean", "mean"),
        sta_lta_max=("sta_lta_max", "max"),
        triggered=("triggered", "sum"),
        amp_min=("amp_max", "min"),
    )
    hourly["amp_range"] = hourly["amp_max"] - hourly["amp_min"]

    # Reindex to the full contiguous hour grid so rolling windows are time-correct
    hourly = hourly.reindex(hours_ts)

    # ── Sustained activity trajectories ──
    trig = hourly["triggered"].fillna(0.0)
    trig_24h = trig.rolling(24, min_periods=1).sum()
    trig_72h = trig.rolling(72, min_periods=1).sum()
    # Acceleration: triggers in last 24h vs the 24h before that
    trig_accel = trig_24h - trig_24h.shift(24)

    amp = hourly["amp_mean"]
    amp_mean_24h = amp.rolling(24, min_periods=3).mean()
    amp_trend_24h = amp_mean_24h - amp_mean_24h.shift(24)
    sta_lta_mean_24h = hourly["sta_lta_max"].rolling(24, min_periods=3).mean()

    # ── Amplitude anomaly vs 7-day baseline ──
    amp_base = amp.rolling(168, min_periods=24).mean()
    amp_std = amp.rolling(168, min_periods=24).std()
    amp_z_7d = (amp - amp_base) / amp_std.replace(0, np.nan)
    # Quiescence: sustained amplitude suppression (a classic precursor) — positive when quiet
    quiescence = (-amp_z_7d).clip(lower=0)

    # ── Multi-station coherence: fraction of stations simultaneously elevated ──
    # A station is "elevated" if its sta_lta_max this hour exceeds its own median by 50%
    stn_base = per_stn.groupby("station")["sta_lta_max"].transform("median")
    per_stn["elevated"] = (per_stn["sta_lta_max"] > stn_base * 1.5).astype(np.float32)
    coh = per_stn.groupby("hour")["elevated"].mean().reindex(hours_ts)

    # ── Assemble ──
    feats[:, fi["stn_amp_max"]] = hourly["amp_max"].values[:n]
    feats[:, fi["stn_amp_mean"]] = hourly["amp_mean"].values[:n]
    feats[:, fi["stn_sta_lta_max"]] = hourly["sta_lta_max"].values[:n]
    feats[:, fi["stn_triggered"]] = hourly["triggered"].values[:n]
    feats[:, fi["stn_amp_range"]] = hourly["amp_range"].values[:n]
    feats[:, fi["stn_trigger_sum_24h"]] = trig_24h.values[:n]
    feats[:, fi["stn_trigger_sum_72h"]] = trig_72h.values[:n]
    feats[:, fi["stn_trigger_accel"]] = trig_accel.values[:n]
    feats[:, fi["stn_amp_mean_24h"]] = amp_mean_24h.values[:n]
    feats[:, fi["stn_amp_trend_24h"]] = amp_trend_24h.values[:n]
    feats[:, fi["stn_sta_lta_mean_24h"]] = sta_lta_mean_24h.values[:n]
    feats[:, fi["stn_amp_z_7d"]] = amp_z_7d.values[:n]
    feats[:, fi["stn_quiescence"]] = quiescence.values[:n]
    feats[:, fi["stn_coherence"]] = coh.values[:n]
    return feats


# ═══════════════════════════════════════════════════════════════════════
# PART 1H2: GPS / GNSS CRUSTAL DEFORMATION
# ═══════════════════════════════════════════════════════════════════════

def build_zone_gps_features(zone, hours):
    """Crustal deformation features from nearby GPS stations (gps.db, daily → hourly).

    Reads the NGL strain solutions, selects stations within the cell (+margin),
    aggregates per-day across stations, then forward-fills daily values to the
    hourly grid. Captures strain accumulation and slow-slip transients (residuals),
    which are among the most physically-grounded earthquake precursors.
    """
    import sqlite3
    n = len(hours)
    fi = {name: i for i, name in enumerate(GPS_FEATURES)}
    feats = np.full((n, len(GPS_FEATURES)), np.nan, dtype=np.float32)
    if not os.path.exists(GPS_DB):
        return feats

    lat_lo, lat_hi = zone["lat_range"]
    lon_lo, lon_hi = zone["lon_range"]
    margin = 1.5  # deg — include stations just outside the cell
    conn = sqlite3.connect(GPS_DB, timeout=30)
    try:
        site_rows = conn.execute(
            "SELECT site FROM gps_stations WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (lat_lo - margin, lat_hi + margin, lon_lo - margin, lon_hi + margin)).fetchall()
        sites = [s[0] for s in site_rows]
        if not sites:
            return feats
        ph = ",".join("?" * len(sites))
        rows = conn.execute(
            f"SELECT epoch, site, anomaly_score, residual_n_mm, residual_e_mm, residual_u_mm, "
            f"velocity_n_mm_yr, velocity_e_mm_yr FROM gps_strain "
            f"WHERE site IN ({ph}) AND epoch >= ? AND epoch <= ? ORDER BY epoch",
            (*sites, hours[0].strftime("%Y-%m-%d"), hours[-1].strftime("%Y-%m-%d"))).fetchall()
    finally:
        conn.close()
    if not rows:
        return feats

    df = pd.DataFrame(rows, columns=["epoch", "site", "anom", "rn", "re", "ru", "vn", "ve"])
    df["day"] = pd.to_datetime(df["epoch"], format="ISO8601", utc=True).dt.floor("D")
    df["res_mag"] = np.sqrt(df["rn"].fillna(0) ** 2 + df["re"].fillna(0) ** 2)
    df["strain_rate"] = np.sqrt(df["vn"].fillna(0) ** 2 + df["ve"].fillna(0) ** 2)
    df["res_up"] = df["ru"].abs()

    # Per-day aggregation across the cell's stations
    daily = df.groupby("day").agg(
        anom_max=("anom", "max"),
        anom_mean=("anom", "mean"),
        res_mag=("res_mag", "mean"),
        res_up=("res_up", "mean"),
        strain_rate=("strain_rate", "mean"),
        coverage=("site", "nunique"),
    )
    # Coherence: fraction of contributing stations with anomaly > 2.0 that day
    coh = df.assign(hot=(df["anom"] > 2.0).astype(float)).groupby("day")["hot"].mean()
    daily["coherence"] = coh
    # Transient growth: 7-day change in residual magnitude
    daily["res_accel"] = daily["res_mag"] - daily["res_mag"].shift(7)

    # Daily → hourly: reindex to full day range, forward-fill (cap staleness at 30 days)
    full_days = pd.date_range(daily.index.min(), pd.DatetimeIndex(hours).floor("D").max(),
                              freq="D", tz="UTC")
    daily = daily.reindex(full_days).ffill(limit=30)
    mapped = daily.reindex(pd.DatetimeIndex(hours).floor("D"))

    feats[:, fi["gps_anomaly_max"]] = mapped["anom_max"].values[:n]
    feats[:, fi["gps_anomaly_mean"]] = mapped["anom_mean"].values[:n]
    feats[:, fi["gps_residual_mag_mm"]] = mapped["res_mag"].values[:n]
    feats[:, fi["gps_residual_up_mm"]] = mapped["res_up"].values[:n]
    feats[:, fi["gps_strain_rate"]] = mapped["strain_rate"].values[:n]
    feats[:, fi["gps_anomaly_coherence"]] = mapped["coherence"].values[:n]
    feats[:, fi["gps_residual_accel"]] = mapped["res_accel"].values[:n]
    return feats


# ═══════════════════════════════════════════════════════════════════════
# PART 1I: TARGETS
# ═══════════════════════════════════════════════════════════════════════

def build_targets_vectorized(conn, zone, hour_epochs, horizon_hours, min_mag):
    rows = conn.execute("""
        SELECT timestamp, magnitude FROM earthquakes
        WHERE magnitude >= ? AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
        ORDER BY timestamp
    """, (min_mag, zone["lat_range"][0], zone["lat_range"][1],
          zone["lon_range"][0], zone["lon_range"][1])).fetchall()

    n = len(hour_epochs)
    targets = np.zeros(n, dtype=np.float32)
    mag_targets = np.zeros(n, dtype=np.float32)
    horizon_sec = horizon_hours * 3600
    event_count = 0

    for r in rows:
        try:
            t = pd.Timestamp(r[0], tz="UTC").timestamp()
            mag = float(r[1])
        except Exception:
            continue
        event_count += 1
        lo = np.searchsorted(hour_epochs, t - horizon_sec, side='left')
        hi = np.searchsorted(hour_epochs, t, side='left')
        if lo < n and hi > 0:
            lo, hi = max(lo, 0), min(hi, n)
            targets[lo:hi] = 1.0
            mag_targets[lo:hi] = np.maximum(mag_targets[lo:hi], mag)

    return targets, mag_targets, event_count


# ═══════════════════════════════════════════════════════════════════════
# PART 1I-2: TIER-2 ESCALATION LABELS (the primary objective)
# ═══════════════════════════════════════════════════════════════════════
# The honest, goal-aligned problem: given an ACTIVE SWARM, will it escalate to an
# independent mainshock, or fizzle? Removes aftershock inflation by construction
# (declustered target + "no big event yet" gate + exceeds-swarm-max).

TIER2_TRAIL_H = 72        # swarm-detection trailing window
TIER2_MIN_SMALL = 3       # >= this many M2.5+ in trailing window = active swarm
TIER2_ESC_H = 72          # escalation horizon (mainshock within next N hours)
TIER2_SMALL_MAG = 2.5

def _gk_window(mag):
    """Gardner-Knopoff (1974) aftershock space-time window."""
    L = 10 ** (0.1238 * mag + 0.983)  # km
    T = 10 ** (0.032 * mag + 2.7389) if mag >= 6.5 else 10 ** (0.5409 * mag - 0.547)  # days
    return L, T

def _decluster(ev):
    """ev: list of (epoch_s, mag, lat, lon). Returns bool list: True = independent mainshock."""
    n = len(ev)
    dep = [False] * n
    for i in sorted(range(n), key=lambda i: -ev[i][1]):
        if dep[i]:
            continue
        ti, mi, lai, loi = ev[i]
        L, T = _gk_window(mi); Ts = T * 86400.0
        for j in range(n):
            if j == i or dep[j] or ev[j][1] >= mi:
                continue
            if abs(ev[j][0] - ti) <= Ts and \
                    float(_haversine(lai, loi, ev[j][2], ev[j][3])) <= L:
                dep[j] = True
    return [not d for d in dep]

def build_tier2_labels(conn, zone, hour_epochs, min_mag=MIN_MAG_TARGET,
                       trail_h=TIER2_TRAIL_H, esc_h=TIER2_ESC_H, min_small=TIER2_MIN_SMALL):
    """Returns (episode_mask, escalation_label) per hour for a cell.
    episode: an active swarm (>=min_small M2.5+ in trailing window, no M5+ yet).
    escalation: an INDEPENDENT mainshock (declustered) M{min_mag}+ within next esc_h.
    """
    n = len(hour_epochs)
    g = zone
    trail_s = trail_h * 3600
    esc_s = esc_h * 3600

    small = conn.execute(
        "SELECT timestamp FROM earthquakes WHERE magnitude >= ? AND magnitude < ? "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? ORDER BY timestamp",
        (TIER2_SMALL_MAG, min_mag, g["lat_range"][0], g["lat_range"][1],
         g["lon_range"][0], g["lon_range"][1])).fetchall()
    bigrows = conn.execute(
        "SELECT timestamp, magnitude, lat, lon FROM earthquakes WHERE magnitude >= ? "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? ORDER BY timestamp",
        (min_mag, g["lat_range"][0], g["lat_range"][1],
         g["lon_range"][0], g["lon_range"][1])).fetchall()

    def _ep(ts):
        try:
            return pd.Timestamp(ts, tz="UTC").timestamp()
        except Exception:
            return None
    s_ep = np.array([e for e in (_ep(t) for (t,) in small) if e is not None])
    bev = [(_ep(ts), float(m), la, lo) for ts, m, la, lo in bigrows if _ep(ts) is not None]
    b_ep = np.array([e[0] for e in bev]) if bev else np.array([])

    small_trail = (np.searchsorted(s_ep, hour_epochs, "right") -
                   np.searchsorted(s_ep, hour_epochs - trail_s, "left")) if len(s_ep) else np.zeros(n)
    big_trail = (np.searchsorted(b_ep, hour_epochs, "right") -
                 np.searchsorted(b_ep, hour_epochs - trail_s, "left")) if len(b_ep) else np.zeros(n)
    episode = ((small_trail >= min_small) & (big_trail == 0)).astype(np.float32)

    escalation = np.zeros(n, dtype=np.float32)
    if bev:
        is_main = _decluster(bev)
        for k, (t, m, la, lo) in enumerate(bev):
            if not is_main[k]:
                continue
            lo_i = max(np.searchsorted(hour_epochs, t - esc_s, "left"), 0)
            hi_i = min(np.searchsorted(hour_epochs, t, "left"), n)
            escalation[lo_i:hi_i] = 1.0
    return episode, escalation


# ═══════════════════════════════════════════════════════════════════════
# PART 1J: FULL DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════

def _compute_velocity_features(zone_feats, n_hours):
    """Compute coulomb velocity/acceleration + composite precursor alarm."""
    def _base(name):
        return np.nan_to_num(zone_feats[:, FEAT_IDX[name]], nan=0.0) if name in FEAT_IDX else np.zeros(n_hours)

    def _delta24(arr):
        d = np.zeros(n_hours, dtype=np.float32)
        d[24:] = arr[24:] - arr[:-24]
        return d

    # Coulomb proxy velocity + acceleration (the only velocity features that earned gain)
    coulomb = _base("coulomb_proxy")
    d24_coulomb = _delta24(coulomb)
    zone_feats[:, FEAT_IDX["v24_coulomb_proxy"]] = d24_coulomb
    acc_coulomb = np.zeros(n_hours, dtype=np.float32)
    acc_coulomb[48:] = d24_coulomb[48:] - d24_coulomb[24:-24]
    zone_feats[:, FEAT_IDX["acc_coulomb_proxy"]] = acc_coulomb

    # Precursor alarm: count of simultaneously active indicators
    # Compute deltas on-the-fly for alarm thresholds (not stored as features)
    alarm = np.zeros(n_hours, dtype=np.float32)
    bval = _base("b_value_14d")
    d24_bval = _delta24(bval)
    alarm += ((bval < 0.8) & (d24_bval < 0)).astype(np.float32)
    alarm += (_delta24(_base("foreshock_accel")) > 0.5).astype(np.float32)
    alarm += (_delta24(_base("eq_count_24h")) > 2).astype(np.float32)
    alarm += (_delta24(_base("energy_24h")) > 0.5).astype(np.float32)
    alarm += (d24_coulomb > 0).astype(np.float32)
    alarm += (_base("tidal_sensitivity_flag") > 0.5).astype(np.float32)
    dst = _base("dst_mean_72h")
    alarm += ((dst < -30) & (_delta24(dst) < -5)).astype(np.float32)
    alarm += (np.abs(_delta24(_base("dart_residual_mean_24h"))) > 0.5).astype(np.float32)
    zone_feats[:, FEAT_IDX["precursor_alarm"]] = alarm


def _compute_interactions(zone_feats, n_hours):
    """Compute interaction features from base features."""
    def _sv(name):
        return np.nan_to_num(zone_feats[:, FEAT_IDX[name]], nan=0.0)

    zone_feats[:, FEAT_IDX["kp_x_foreshock"]] = _sv("kp_mean_72h") * _sv("eq_count_24h")
    zone_feats[:, FEAT_IDX["dst_x_bvalue"]] = np.abs(_sv("dst_mean_72h")) * np.maximum(1.0 - _sv("b_value_14d"), 0)
    zone_feats[:, FEAT_IDX["sw_x_accel"]] = _sv("sw_speed_mean_72h") * _sv("foreshock_accel")
    zone_feats[:, FEAT_IDX["ief_x_foreshock"]] = _sv("ief_mean_72h") * _sv("foreshock_accel")
    zone_feats[:, FEAT_IDX["cosmic_x_dst"]] = _sv("cosmic_anomaly") * np.abs(_sv("dst_mean_72h"))
    zone_feats[:, FEAT_IDX["sw_x_coulomb"]] = _sv("sw_speed_mean_72h") * np.abs(_sv("coulomb_proxy"))
    zone_feats[:, FEAT_IDX["kp_x_moment"]] = _sv("kp_mean_72h") * _sv("moment_7d")
    zone_feats[:, FEAT_IDX["cme_x_coulomb"]] = _sv("cme_speed_max_72h") * np.abs(_sv("coulomb_proxy"))
    zone_feats[:, FEAT_IDX["cme_x_foreshock"]] = _sv("cme_speed_max_72h") * _sv("foreshock_accel")
    zone_feats[:, FEAT_IDX["flare_x_bvalue"]] = _sv("flare_max_class_72h") * np.maximum(1.0 - _sv("b_value_14d"), 0)
    zone_feats[:, FEAT_IDX["storm_x_foreshock"]] = _sv("storm_kp_max_72h") * _sv("eq_count_24h")
    zone_feats[:, FEAT_IDX["cme_x_tidal_R"]] = _sv("cme_speed_max_72h") * _sv("tidal_schuster_R_30d")
    zone_feats[:, FEAT_IDX["tidal_sens_x_coulomb"]] = _sv("tidal_schuster_R_30d") * np.abs(_sv("coulomb_proxy"))
    zone_feats[:, FEAT_IDX["tidal_sens_x_bvalue"]] = _sv("tidal_schuster_R_30d") * np.maximum(1.0 - _sv("b_value_14d"), 0)
    zone_feats[:, FEAT_IDX["tidal_sens_x_foreshock"]] = _sv("tidal_schuster_R_30d") * _sv("eq_count_24h")
    zone_feats[:, FEAT_IDX["tidal_small_x_coulomb"]] = _sv("tidal_R_30d_small") * np.abs(_sv("coulomb_proxy"))
    zone_feats[:, FEAT_IDX["tidal_small_x_accel"]] = _sv("tidal_R_30d_small") * _sv("foreshock_accel")
    zone_feats[:, FEAT_IDX["volcano_x_coulomb"]] = _sv("volcanic_stress_index") * np.abs(_sv("coulomb_proxy"))
    zone_feats[:, FEAT_IDX["volcano_x_sw"]] = _sv("volcanic_stress_index") * _sv("sw_speed_mean_72h")
    zone_feats[:, FEAT_IDX["hotspot_x_coulomb"]] = _sv("hotspot_count_7d") * np.abs(_sv("coulomb_proxy"))
    dart_vol = _sv("dart_vol_ramp_6v18")
    zone_feats[:, FEAT_IDX["dart_x_coulomb"]] = dart_vol * np.abs(_sv("coulomb_proxy"))
    zone_feats[:, FEAT_IDX["dart_x_sw"]] = dart_vol * _sv("sw_speed_mean_72h")


def build_cell_features(conn, cell, hours, hour_epochs, sig, evt, tidal_times, tidal_values):
    """Assemble the full feature matrix for ONE cell, given precomputed GLOBAL signals
    (sig, evt, tidal_*). Shared by the batch builder and the realtime engine so both
    produce identical features. Returns (n_hours, NUM_FEATURES)."""
    n_hours = len(hours)
    cell_feats = np.zeros((n_hours, NUM_FEATURES), dtype=np.float32)

    cat = build_zone_catalog_features(conn, cell, hour_epochs)
    for j, name in enumerate(CATALOG_FEATURES):
        cell_feats[:, FEAT_IDX[name]] = cat[:, j]

    for name, arr in sig.items():
        if name in FEAT_IDX:
            cell_feats[:, FEAT_IDX[name]] = arr
    for name, arr in evt.items():
        if name in FEAT_IDX:
            cell_feats[:, FEAT_IDX[name]] = arr

    dart = build_zone_dart_features(conn, cell, hours)
    for j, name in enumerate(DART_FEATURES):
        cell_feats[:, FEAT_IDX[name]] = dart[:, j]

    tidal_trig = build_zone_tidal_triggering(conn, cell, hour_epochs, tidal_times, tidal_values)
    for j, name in enumerate(TIDAL_TRIGGER_FEATURES):
        cell_feats[:, FEAT_IDX[name]] = tidal_trig[:, j]

    volc = build_zone_volcanic_features(conn, cell)
    for j, name in enumerate(VOLCANIC_FEATURES):
        cell_feats[:, FEAT_IDX[name]] = volc[j]
    volc_dyn = build_zone_volcanic_dynamic(conn, cell, hour_epochs)
    for j, name in enumerate(VOLCANIC_DYN_FEATURES):
        cell_feats[:, FEAT_IDX[name]] = volc_dyn[:, j]

    firms = build_zone_firms_features(conn, cell, hour_epochs)
    for j, name in enumerate(FIRMS_FEATURES):
        cell_feats[:, FEAT_IDX[name]] = firms[:, j]

    stn = build_zone_station_features(conn, cell, hours)
    for j, name in enumerate(STATION_FEATURES):
        cell_feats[:, FEAT_IDX[name]] = stn[:, j]

    gps = build_zone_gps_features(cell, hours)
    for j, name in enumerate(GPS_FEATURES):
        cell_feats[:, FEAT_IDX[name]] = gps[:, j]

    _compute_interactions(cell_feats, n_hours)
    _compute_velocity_features(cell_feats, n_hours)
    return cell_feats


def build_full_dataset(horizon_hours=HORIZON_HOURS, min_mag=MIN_MAG_TARGET,
                       use_grid=True, min_events=100, start_date=None, zone_filter=None):
    conn = sqlite3.connect(DB_PATH, timeout=30)

    r = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM earthquakes").fetchone()
    start = pd.Timestamp(r[0], tz="UTC").ceil("h")
    end = pd.Timestamp(r[1], tz="UTC").floor("h")
    if start_date is not None:
        clip = pd.Timestamp(start_date, tz="UTC").ceil("h")
        if clip > start:
            start = clip
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    hour_epochs = np.array([h.timestamp() for h in hours])
    n_hours = len(hours)

    # Restrict to requested parent zones if a filter is given
    parents = PARENT_ZONES
    if zone_filter:
        parents = [z for z in PARENT_ZONES if z["id"] in zone_filter]
        print(f"  Zone filter: {[z['id'] for z in parents]}")

    # Generate grid cells or use parent zones
    if use_grid:
        all_cells = generate_grid_cells(parents)
        # Filter to cells with enough seismic activity
        active_cells = []
        for cell in all_cells:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM earthquakes WHERE magnitude >= ? "
                "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
                (MIN_MAG_CATALOG, cell["lat_range"][0], cell["lat_range"][1],
                 cell["lon_range"][0], cell["lon_range"][1])
            ).fetchone()[0]
            if cnt >= min_events:
                active_cells.append(cell)
        zones = active_cells
        print(f"  Grid cells: {len(all_cells)} total, {len(zones)} active (>={min_events} M{MIN_MAG_CATALOG}+ events)")
    else:
        zones = parents

    print(f"  Time range: {start} -> {end}")
    print(f"  Hours: {n_hours:,} ({n_hours / 24:.0f} days)")
    print(f"  Features: {NUM_FEATURES} ({NUM_FEATURES - len(VELOCITY_FEATURES)} base + {len(VELOCITY_FEATURES)} velocity)")

    # -- Global signal features (shared across all cells) --
    print(f"\n  Building global features...")
    t0 = time.time()
    sig = build_signal_features(conn, hours)
    raw = sig.pop("_raw")
    print(f"    Signal: {len(sig)} features ({time.time() - t0:.1f}s)")

    t0 = time.time()
    evt = build_event_features(conn, hours, hour_epochs)
    print(f"    Events: {len(evt)} features ({time.time() - t0:.1f}s)")

    print(f"  Loading tidal potential for Schuster tests...")
    tidal_rows = conn.execute(
        "SELECT timestamp, value FROM samples WHERE metric = 'tidal_potential' ORDER BY timestamp"
    ).fetchall()
    tidal_times, tidal_values = [], []
    for ts, val in tidal_rows:
        try:
            t = pd.Timestamp(ts, tz="UTC").timestamp()
            tidal_times.append(t)
            tidal_values.append(val)
        except Exception:
            continue
    tidal_times = np.array(tidal_times)
    tidal_values = np.array(tidal_values)
    print(f"    Tidal data: {len(tidal_times):,} points")

    # -- Per-cell features --
    all_features = {}
    all_targets = {}
    all_mag_targets = {}
    all_episode = {}
    all_escalation = {}
    cell_parent_map = {}

    for ci, cell in enumerate(zones):
        cid = cell["id"]
        parent = cell.get("parent", cid)
        cell_parent_map[cid] = parent
        clat, clon = _zone_center(cell)
        lat_span = cell["lat_range"][1] - cell["lat_range"][0]
        lon_span = cell["lon_range"][1] - cell["lon_range"][0]
        print(f"\n  Cell {ci + 1}/{len(zones)}: {cid} ({lat_span}x{lon_span} deg)")

        t0 = time.time()
        cell_feats = build_cell_features(conn, cell, hours, hour_epochs, sig, evt,
                                         tidal_times, tidal_values)
        print(f"    Features: {time.time() - t0:.1f}s")

        all_features[cid] = cell_feats

        # Targets
        tgt, mag_tgt, n_events = build_targets_vectorized(
            conn, cell, hour_epochs, horizon_hours, min_mag)
        all_targets[cid] = tgt
        all_mag_targets[cid] = mag_tgt

        # Tier-2 escalation labels (the primary objective)
        epi, esc = build_tier2_labels(conn, cell, hour_epochs, min_mag)
        all_episode[cid] = epi
        all_escalation[cid] = esc

        m5_count = conn.execute(
            "SELECT COUNT(*) FROM earthquakes WHERE magnitude >= ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (min_mag, cell["lat_range"][0], cell["lat_range"][1],
             cell["lon_range"][0], cell["lon_range"][1])
        ).fetchone()[0]
        n_epi = int(epi.sum()); esc_rate = (esc[epi > 0.5].mean() * 100) if n_epi else 0.0
        print(f"    Events: {m5_count} M{min_mag}+ | occ {tgt.mean()*100:.2f}% | "
              f"swarm-hrs {n_epi} (escalate {esc_rate:.1f}%)")

    conn.close()

    return {
        "features": all_features,
        "targets": all_targets,
        "mag_targets": all_mag_targets,
        "episode": all_episode,
        "escalation": all_escalation,
        "hours": hours,
        "hour_epochs": hour_epochs,
        "feature_names": ALL_FEATURES,
        "cell_parent_map": cell_parent_map,
        "zones": zones,
    }




# ═══════════════════════════════════════════════════════════════════════
# PART 2: LIGHTGBM
# ═══════════════════════════════════════════════════════════════════════

def train_lgbm(X_train, y_train, X_val, y_val, feature_names, tag="LightGBM",
               categorical_indices=None):
    import lightgbm as lgb
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale = neg / max(pos, 1)

    cat_feat = categorical_indices if categorical_indices else "auto"
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                         categorical_feature=cat_feat)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                       reference=dtrain, categorical_feature=cat_feat)

    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "max_depth": 6, "learning_rate": 0.005,
        "min_child_samples": 150, "colsample_bytree": 0.5, "subsample": 0.7,
        "subsample_freq": 1, "scale_pos_weight": min(scale, 15.0),
        "reg_alpha": 0.5, "reg_lambda": 2.0,
        "min_gain_to_split": 0.1,
        "verbose": -1, "seed": 42,
    }

    callbacks = [lgb.early_stopping(200), lgb.log_evaluation(0)]
    model = lgb.train(params, dtrain, num_boost_round=5000,
                      valid_sets=[dval], callbacks=callbacks)

    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred) if y_val.sum() > 0 else 0.0
    print(f"  {tag}: {model.num_trees()} trees, val AUC={auc:.4f}")

    imp = model.feature_importance(importance_type='gain')
    imp_idx = np.argsort(imp)[::-1][:25]
    print(f"  Top 25 features:")
    for i in imp_idx:
        print(f"    {feature_names[i]:30s} {imp[i]:>12.1f}")

    # Velocity feature importance summary
    vel_total = sum(imp[i] for i, fn in enumerate(feature_names) if fn in VELOCITY_FEATURES)
    all_total = imp.sum()
    if all_total > 0:
        vel_pct = vel_total / all_total * 100
        vel_in_top25 = sum(1 for i in imp_idx if feature_names[i] in VELOCITY_FEATURES)
        print(f"  Velocity features: {vel_pct:.1f}% of total gain, {vel_in_top25}/25 in top features")

    return model, auc


# ═══════════════════════════════════════════════════════════════════════
# PART 3: XGBOOST
# ═══════════════════════════════════════════════════════════════════════

def train_xgb(X_train, y_train, X_val, y_val, feature_names):
    import xgboost as xgb
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale = neg / max(pos, 1)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 7, "learning_rate": 0.008, "min_child_weight": 200,
        "colsample_bytree": 0.5, "subsample": 0.7,
        "scale_pos_weight": min(scale, 12.0),
        "reg_alpha": 1.0, "reg_lambda": 3.0, "gamma": 0.1,
        "tree_method": "hist", "seed": 123,
    }

    model = xgb.train(params, dtrain, num_boost_round=5000,
                      evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)

    val_pred = model.predict(dval)
    auc = roc_auc_score(y_val, val_pred) if y_val.sum() > 0 else 0.0
    print(f"  XGBoost: {model.best_iteration} trees, val AUC={auc:.4f}")

    imp = model.get_score(importance_type='gain')
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])[:25]
    print(f"  Top 25 features:")
    for name, gain in sorted_imp:
        print(f"    {name:30s} {gain:>12.1f}")

    return model, auc


# ═══════════════════════════════════════════════════════════════════════
# PART 4: NEGATIVE SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def smart_negative_sample(features, targets, zone_list, start_h, end_h,
                          neg_ratio=3, boundary_hours=3):
    """Keep all positives + boundary negatives + random subsample.

    boundary_hours: keep negatives within this many hours of a positive window edge.
    neg_ratio: target ratio of negatives to positives (after boundary inclusion).
    """
    X_list, y_list = [], []
    for zi, zid in enumerate(zone_list):
        feats = features[zid][start_h:end_h]
        tgt = targets[zid][start_h:end_h]
        X = feats

        pos_mask = tgt > 0.5
        pos_idx = np.where(pos_mask)[0]
        neg_idx_all = np.where(~pos_mask)[0]

        if len(pos_idx) == 0 or len(neg_idx_all) == 0:
            X_list.append(X)
            y_list.append(tgt)
            continue

        # Boundary negatives: within boundary_hours of any positive
        boundary_set = set()
        for pi in pos_idx:
            for offset in range(-boundary_hours, boundary_hours + 1):
                ni = pi + offset
                if 0 <= ni < len(tgt) and not pos_mask[ni]:
                    boundary_set.add(ni)
        boundary_idx = np.array(sorted(boundary_set))

        # Random negatives from the rest
        remaining_neg = np.setdiff1d(neg_idx_all, boundary_idx)
        n_target = max(0, int(len(pos_idx) * neg_ratio) - len(boundary_idx))
        if n_target > 0 and len(remaining_neg) > n_target:
            rng = np.random.RandomState(42 + zi)
            random_neg = rng.choice(remaining_neg, size=n_target, replace=False)
        else:
            random_neg = remaining_neg

        keep = np.sort(np.concatenate([pos_idx, boundary_idx, random_neg]))
        X_list.append(X[keep])
        y_list.append(tgt[keep])

    return np.concatenate(X_list), np.concatenate(y_list)


# ═══════════════════════════════════════════════════════════════════════
# PART 5: EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def _pr_optimal_threshold(y_true, y_score):
    """Find threshold that maximizes F1 on the precision-recall curve."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns n+1 prec/rec but n thresholds
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-8)
    best = np.argmax(f1s)
    return float(thresholds[best]), float(f1s[best])


def evaluate_per_zone(preds, targets, zone_ids_per_sample, zone_list, label="",
                      cell_parent_map=None):
    """Evaluate predictions per zone/cell and optionally aggregate by parent zone.
    Uses precision-recall curve for threshold selection instead of ROC Youden's J."""
    results = {}
    zone_aucs = []
    for zi, zid in enumerate(zone_list):
        mask = zone_ids_per_sample == zi
        if mask.sum() == 0:
            continue
        zt = targets[mask]
        zp = preds[mask]
        if zt.sum() > 0 and zt.sum() < len(zt):
            auc = roc_auc_score(zt, zp)
            ap = average_precision_score(zt, zp)
            thresh, _ = _pr_optimal_threshold(zt, zp)
            bp = (zp > thresh).astype(int)
            results[zid] = {
                "auc": auc, "ap": ap,
                "precision": precision_score(zt, bp, zero_division=0),
                "recall": recall_score(zt, bp, zero_division=0),
                "f1": f1_score(zt, bp, zero_division=0),
                "pos_rate": float(zt.mean()),
                "threshold": thresh,
            }
            zone_aucs.append(auc)

    macro_auc = np.mean(zone_aucs) if zone_aucs else 0.0

    # Aggregate by parent zone if we have grid cells
    parent_results = {}
    if cell_parent_map:
        parent_preds = {}
        parent_tgts = {}
        for zi, cid in enumerate(zone_list):
            mask = zone_ids_per_sample == zi
            if mask.sum() == 0:
                continue
            pid = cell_parent_map.get(cid, cid)
            parent_preds.setdefault(pid, []).append(preds[mask])
            parent_tgts.setdefault(pid, []).append(targets[mask])
        for pid in sorted(parent_preds.keys()):
            pt = np.concatenate(parent_tgts[pid])
            pp = np.concatenate(parent_preds[pid])
            if pt.sum() > 0 and pt.sum() < len(pt):
                auc = roc_auc_score(pt, pp)
                ap = average_precision_score(pt, pp)
                thresh, _ = _pr_optimal_threshold(pt, pp)
                bp = (pp > thresh).astype(int)
                parent_results[pid] = {
                    "auc": auc, "ap": ap,
                    "precision": precision_score(pt, bp, zero_division=0),
                    "recall": recall_score(pt, bp, zero_division=0),
                    "f1": f1_score(pt, bp, zero_division=0),
                    "pos_rate": float(pt.mean()),
                    "threshold": thresh,
                    "n_cells": sum(1 for c in zone_list if cell_parent_map.get(c, c) == pid),
                }

    if label:
        if parent_results:
            parent_aucs = [m["auc"] for m in parent_results.values()]
            parent_aps = [m["ap"] for m in parent_results.values()]
            pmacro = np.mean(parent_aucs)
            pmacro_ap = np.mean(parent_aps)
            print(f"\n  {label} — Parent-Zone Macro AUC: {pmacro:.4f}  AP: {pmacro_ap:.4f}")
            print(f"  {'Parent Zone':>20s} {'Cells':>5s} {'AUC':>7s} {'AP':>7s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Thr':>5s} {'Pos%':>6s}")
            print(f"  {'-'*72}")
            for pid, m in sorted(parent_results.items()):
                print(f"  {pid:>20s} {m['n_cells']:>5d} {m['auc']:>7.4f} {m['ap']:>7.4f} {m['precision']:>6.3f} "
                      f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['threshold']:>5.3f} {m['pos_rate']*100:>5.1f}%")
        else:
            print(f"\n  {label} — Macro AUC: {macro_auc:.4f}")
            print(f"  {'Zone':>15s} {'AUC':>7s} {'AP':>7s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Thr':>5s} {'Pos%':>6s}")
            print(f"  {'-'*58}")
            for zid, m in results.items():
                print(f"  {zid:>15s} {m['auc']:>7.4f} {m['ap']:>7.4f} {m['precision']:>6.3f} "
                      f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['threshold']:>5.3f} {m['pos_rate']*100:>5.1f}%")

    return macro_auc, results, parent_results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Earthquake forecaster v7")
    parser.add_argument("--min-mag", type=float, default=5.0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--no-grid", action="store_true",
                        help="Use parent zones instead of 10x10 grid cells")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Restrict window to >= this date (e.g. 2021-07-01 for station era)")
    parser.add_argument("--zones", type=str, default=None,
                        help="Comma-separated parent zone ids to build/train (default: all)")
    parser.add_argument("--objective", choices=["tier2", "occurrence"], default="tier2",
                        help="tier2 = swarm-escalation (primary); occurrence = legacy any-M5+")
    parser.add_argument("--skip-feature-build", action="store_true",
                        help="Load cached features from .npz")
    args = parser.parse_args()
    zone_filter = [z.strip() for z in args.zones.split(",")] if args.zones else None

    global LOOKBACK_HOURS, HORIZON_HOURS, MIN_MAG_TARGET
    LOOKBACK_HOURS = args.lookback
    HORIZON_HOURS = args.horizon
    MIN_MAG_TARGET = args.min_mag
    use_grid = not args.no_grid

    print("=" * 70)
    print("  SEISMICLAB FORECASTER — v8")
    print(f"  Objective: {args.objective.upper()}"
          + ("  (swarm -> independent mainshock escalation)" if args.objective == "tier2"
             else "  (legacy any-M5+ occurrence)"))
    print(f"  Global LightGBM — no spatial identity, multi-signal precursor physics")
    print(f"  Target mag: M{args.min_mag}+ | Features: {NUM_FEATURES}")
    print(f"  Grid: {'{0}x{0} deg cells'.format(CELL_SIZE) if use_grid else 'parent zones only'}")
    print("=" * 70)

    grid_tag = f"grid{CELL_SIZE}" if use_grid else "zones"
    era_tag = f"_from{args.start_date}" if args.start_date else ""
    zone_tag = f"_z{'-'.join(zone_filter)}" if zone_filter else ""
    cache_path = os.path.join(CACHE_DIR,
        f"ensemble_v7_{grid_tag}_m{args.min_mag}_h{args.horizon}{era_tag}{zone_tag}.npz")
    if args.start_date:
        print(f"  Window: STATION ERA (>= {args.start_date})")

    if args.skip_feature_build and os.path.exists(cache_path):
        print(f"\n  Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cell_ids = list(data["cell_ids"])
        cell_parent_map = dict(zip(data["cell_ids"], data["cell_parents"]))
        dataset = {
            "features": {cid: data[f"feat_{cid}"] for cid in cell_ids},
            "targets": {cid: data[f"tgt_{cid}"] for cid in cell_ids},
            "mag_targets": {cid: data[f"mag_{cid}"] for cid in cell_ids},
            "hours": pd.DatetimeIndex(data["hours"]),
            "hour_epochs": data["hour_epochs"],
            "feature_names": list(data["feature_names"]),
            "cell_parent_map": cell_parent_map,
        }
        if f"epi_{cell_ids[0]}" in data:
            dataset["episode"] = {cid: data[f"epi_{cid}"] for cid in cell_ids}
            dataset["escalation"] = {cid: data[f"esc_{cid}"] for cid in cell_ids}
        print(f"  Loaded: {len(cell_ids)} cells, {len(dataset['hours']):,} hours x "
              f"{len(dataset['feature_names'])} features")
    else:
        print(f"\n  -- STEP 1: FEATURE ENGINEERING --")
        t_start = time.time()
        dataset = build_full_dataset(args.horizon, args.min_mag, use_grid=use_grid,
                                     start_date=args.start_date, zone_filter=zone_filter)
        elapsed = time.time() - t_start
        cell_ids = list(dataset["features"].keys())
        cell_parent_map = dataset.get("cell_parent_map", {c: c for c in cell_ids})
        print(f"\n  Features built in {elapsed:.0f}s "
              f"({len(cell_ids)} cells x {NUM_FEATURES} features)")

        save_dict = {
            "hours": np.array(dataset["hours"]),
            "hour_epochs": dataset["hour_epochs"],
            "feature_names": np.array(dataset["feature_names"]),
            "cell_ids": np.array(cell_ids),
            "cell_parents": np.array([cell_parent_map.get(c, c) for c in cell_ids]),
        }
        for cid in cell_ids:
            save_dict[f"feat_{cid}"] = dataset["features"][cid]
            save_dict[f"tgt_{cid}"] = dataset["targets"][cid]
            save_dict[f"mag_{cid}"] = dataset["mag_targets"][cid]
            save_dict[f"epi_{cid}"] = dataset["episode"][cid]
            save_dict[f"esc_{cid}"] = dataset["escalation"][cid]
        np.savez_compressed(cache_path, **save_dict)
        print(f"  Cached to {cache_path}")

    hours = dataset["hours"]
    n_hours = len(hours)
    n_cells = len(cell_ids)

    # -- Temporal split --
    train_end = int(n_hours * 0.70)
    val_end = int(n_hours * 0.85)

    print(f"\n  Temporal split:")
    print(f"    Train: {hours[0].date()} -> {hours[train_end].date()} ({train_end:,} hours)")
    print(f"    Val:   {hours[train_end].date()} -> {hours[val_end].date()} ({val_end - train_end:,} hours)")
    print(f"    Test:  {hours[val_end].date()} -> {hours[-1].date()} ({n_hours - val_end:,} hours)")
    print(f"    Cells: {n_cells}")

    # -- Feature quality --
    print(f"\n  Feature quality check (sampling first 20 cells)...")
    sample_cells = cell_ids[:min(20, n_cells)]
    all_train = np.concatenate([dataset["features"][cid][:train_end] for cid in sample_cells])
    feat_std = np.nanstd(all_train, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    nan_rates = np.isnan(all_train).mean(axis=0)
    zero_rates = (all_train == 0).mean(axis=0)
    const_feats = [ALL_FEATURES[i] for i in range(NUM_FEATURES) if feat_std[i] < 1e-8]
    sparse_feats = [(ALL_FEATURES[i], nan_rates[i], zero_rates[i])
                    for i in range(NUM_FEATURES)
                    if nan_rates[i] > 0.3 or zero_rates[i] > 0.95]
    if const_feats:
        print(f"  WARNING: {len(const_feats)} constant features: {const_feats[:10]}")
    if sparse_feats:
        print(f"  Sparse features (>30% NaN or >95% zero):")
        for name, nr, zr in sorted(sparse_feats, key=lambda x: -x[1])[:15]:
            print(f"    {name:30s}  NaN={nr*100:5.1f}%  Zero={zr*100:5.1f}%")
    del all_train

    # -- Ensure tier-2 labels exist (recompute from catalog if cache predates them) --
    objective = args.objective
    if objective == "tier2" and "episode" not in dataset:
        print(f"\n  Computing tier-2 labels from catalog (cache predates them)...")
        _conn = sqlite3.connect(DB_PATH, timeout=60)
        geom = {c["id"]: c for c in generate_grid_cells(PARENT_ZONES)}
        epi_d, esc_d = {}, {}
        for cid in cell_ids:
            if cid in geom:
                epi_d[cid], esc_d[cid] = build_tier2_labels(_conn, geom[cid],
                                                            dataset["hour_epochs"], args.min_mag)
            else:
                epi_d[cid] = np.zeros(n_hours, np.float32); esc_d[cid] = np.zeros(n_hours, np.float32)
        _conn.close()
        dataset["episode"] = epi_d; dataset["escalation"] = esc_d

    feature_names = ALL_FEATURES
    cell_parent_map_for_eval = {cell_ids[ci]: cell_parent_map.get(cell_ids[ci], cell_ids[ci])
                                for ci in range(n_cells)}

    print(f"\n  -- STEP 2: ASSEMBLE TRAIN/VAL/TEST ({objective}) --")
    if objective == "tier2":
        episode = dataset["episode"]; escalation = dataset["escalation"]
        def assemble(lo, hi):
            X, y, z = [], [], []
            for ci, cid in enumerate(cell_ids):
                m = episode[cid][lo:hi] > 0.5
                if m.sum() == 0:
                    continue
                X.append(dataset["features"][cid][lo:hi][m])
                y.append(escalation[cid][lo:hi][m].astype(np.float32))
                z.append(np.full(int(m.sum()), ci))
            if not X:
                return np.empty((0, NUM_FEATURES), np.float32), np.empty(0), np.empty(0)
            return np.concatenate(X), np.concatenate(y), np.concatenate(z)
        X_train, y_train, _ = assemble(0, train_end)
        X_val, y_val, val_cell_ids = assemble(train_end, val_end)
        X_test, y_test, test_cell_ids = assemble(val_end, n_hours)
        print(f"  Swarm episodes — train {len(y_train):,} ({y_train.mean()*100:.1f}% escalate) | "
              f"val {len(y_val):,} | test {len(y_test):,} ({y_test.mean()*100:.1f}% escalate)")
    else:
        X_train, y_train = smart_negative_sample(
            dataset["features"], dataset["targets"], cell_ids,
            0, train_end, neg_ratio=4, boundary_hours=3)
        def make_full_data(s, e):
            Xl, yl = [], []
            for cid in cell_ids:
                Xl.append(dataset["features"][cid][s:e]); yl.append(dataset["targets"][cid][s:e])
            return np.concatenate(Xl), np.concatenate(yl)
        X_val, y_val = make_full_data(train_end, val_end)
        X_test, y_test = make_full_data(val_end, n_hours)
        n_val = val_end - train_end; n_test = n_hours - val_end
        val_cell_ids = np.concatenate([np.full(n_val, ci) for ci in range(n_cells)])
        test_cell_ids = np.concatenate([np.full(n_test, ci) for ci in range(n_cells)])
        print(f"  Train {X_train.shape[0]:,} ({y_train.mean()*100:.1f}% pos) | "
              f"val {X_val.shape[0]:,} | test {X_test.shape[0]:,} ({y_test.mean()*100:.1f}% pos)")

    del dataset

    # -- LightGBM --
    print(f"\n  -- STEP 3: LIGHTGBM ({objective}) --")
    lgb_model, _ = train_lgbm(X_train, y_train, X_val, y_val, feature_names)
    lgb_val_probs = lgb_model.predict(X_val)
    lgb_test_probs = lgb_model.predict(X_test)
    evaluate_per_zone(lgb_val_probs, y_val, val_cell_ids, cell_ids, "VAL", cell_parent_map_for_eval)

    print(f"\n  {'=' * 70}")
    print(f"  FINAL TEST EVALUATION ({objective})")
    print(f"  {'=' * 70}")
    macro_auc, _, parent_results = evaluate_per_zone(
        lgb_test_probs, y_test, test_cell_ids, cell_ids, "TEST", cell_parent_map_for_eval)
    parent_aucs = [m["auc"] for m in parent_results.values()] if parent_results else [macro_auc]
    parent_aps = [m["ap"] for m in parent_results.values()] if parent_results else [0.0]
    pmacro = np.mean(parent_aucs); pmacro_ap = np.mean(parent_aps)

    base = float(y_test.mean())
    ov_ap = average_precision_score(y_test, lgb_test_probs) if 0 < y_test.sum() < len(y_test) else 0.0

    print(f"\n  -- SUMMARY ({objective}) --")
    print(f"  Parent-Zone Macro AUC: {pmacro:.4f}  AP: {pmacro_ap:.4f}")
    print(f"  Overall test AP: {ov_ap:.4f}  (base rate {base*100:.1f}%, "
          f"lift {ov_ap/max(base,1e-6):.1f}x random)")
    if objective == "tier2":
        print(f"  Reference (preliminary tier-2): catalog-only 0.633 / multi-signal 0.668 AUC")

    lgb_path = os.path.join(MODEL_DIR, f"v8_{objective}_lgb.txt")
    lgb_model.save_model(lgb_path)
    save_path = os.path.join(MODEL_DIR, f"v8_{objective}_meta.npz")
    np.savez(save_path, feature_names=np.array(feature_names),
             cell_ids=np.array(cell_ids),
             cell_parents=np.array([cell_parent_map.get(c, c) for c in cell_ids]),
             macro_auc=pmacro, macro_ap=pmacro_ap, overall_ap=ov_ap)
    print(f"\n  Saved: {lgb_path}\n  Saved: {save_path}\n  Done.")


if __name__ == "__main__":
    main()
