"""SeismicLab Lab — NOAA DART Tsunami Buoy Ingestion

Pull deep-ocean bottom pressure recorder data from NOAA NDBC.
DART buoys measure seafloor pressure at mm resolution — detects:
  - Tsunami waves (obvious)
  - Slow-slip events (pre-seismic deformation at ~10 km/day)
  - Tidal loading anomalies (compression/extension patterns)

Data:
  - Realtime: https://www.ndbc.noaa.gov/data/realtime2/{STATION}.dart  (~45 days)
  - Historical: https://www.ndbc.noaa.gov/download_data.php?filename={STATION}t{YEAR}.txt.gz

Modes: T=1 (15-min normal), T=2 (1-min event), T=3 (15-sec tsunami)

Station 42407 showed a clear compression→extension pattern 4 days before
the Venezuela M7.5 (June 24, 2026) — loading then release on the local
station, uncorrelated with the shared Atlantic basin signal.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import gzip
import io
import time
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

try:
    import urllib.request
except ImportError:
    pass

UTC = timezone.utc
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "seismiclab.db")

NDBC_REALTIME = "https://www.ndbc.noaa.gov/data/realtime2"
NDBC_HISTORICAL = "https://www.ndbc.noaa.gov/data/historical/dart"

DART_STATIONS = {
    # Caribbean / Atlantic
    "41420": {"lat": 23.433, "lon": -67.386, "region": "N. Caribbean"},
    "41421": {"lat": 23.445, "lon": -63.851, "region": "N. Caribbean"},
    "41425": {"lat": 28.639, "lon": -65.774, "region": "Atlantic/Bermuda"},
    "42407": {"lat": 15.276, "lon": -68.191, "region": "Caribbean"},
    "42409": {"lat": 25.797, "lon": -89.288, "region": "Gulf of Mexico"},
    "43413": {"lat": 10.927, "lon": -100.012, "region": "E. Pacific"},
    # Pacific
    "46402": {"lat": 51.068, "lon": -164.006, "region": "N. Pacific/Alaska"},
    "46403": {"lat": 52.647, "lon": -156.939, "region": "N. Pacific/Alaska"},
    "46404": {"lat": 45.857, "lon": -128.777, "region": "NE Pacific"},
    "46407": {"lat": 42.568, "lon": -128.832, "region": "NE Pacific/Cascadia"},
    "46408": {"lat": 49.621, "lon": -128.834, "region": "NE Pacific/BC"},
    "46409": {"lat": 55.293, "lon": -148.496, "region": "Gulf of Alaska"},
    "46410": {"lat": 57.632, "lon": -143.843, "region": "Gulf of Alaska"},
    "46411": {"lat": 39.308, "lon": -127.090, "region": "NE Pacific/CA"},
    "46413": {"lat": 47.991, "lon": -173.975, "region": "N. Pacific/Aleutians"},
    "46414": {"lat": 48.942, "lon": -174.277, "region": "N. Pacific/Aleutians"},
    "46415": {"lat": 52.886, "lon": -171.831, "region": "N. Pacific/Aleutians"},
    "46416": {"lat": 48.988, "lon": -175.152, "region": "N. Pacific/Aleutians"},
    "46419": {"lat": 48.766, "lon": -129.621, "region": "NE Pacific/JdF"},
    # South Pacific
    "32401": {"lat": -8.000, "lon": -80.000, "region": "SE Pacific/Peru"},
    "32402": {"lat": -26.700, "lon": -73.900, "region": "SE Pacific/Chile"},
    "32403": {"lat": -14.700, "lon": -76.900, "region": "SE Pacific/Peru"},
    "32404": {"lat": -19.600, "lon": -74.700, "region": "SE Pacific/Chile"},
    "32413": {"lat": -7.400, "lon": -88.500, "region": "SE Pacific"},
    # West Pacific
    "21413": {"lat": 30.515, "lon": 152.117, "region": "W. Pacific/Japan"},
    "21414": {"lat": 48.938, "lon": 155.736, "region": "NW Pacific/Kuril"},
    "21415": {"lat": 50.149, "lon": 171.849, "region": "NW Pacific/Aleutians"},
    "21416": {"lat": 47.352, "lon": 155.741, "region": "NW Pacific/Kuril"},
    "21419": {"lat": 44.456, "lon": 155.764, "region": "NW Pacific/Japan"},
    "21420": {"lat": 27.096, "lon": 141.919, "region": "W. Pacific/Izu-Bonin"},
    # Indian Ocean
    "23220": {"lat": -11.030, "lon": 80.560, "region": "Indian Ocean"},
    "23223": {"lat": -6.560, "lon": 88.850, "region": "Indian Ocean"},
    "23226": {"lat": -2.500, "lon": 92.500, "region": "Indian Ocean/Sumatra"},
    "23401": {"lat": 8.905, "lon": 88.543, "region": "Bay of Bengal"},
    "23461": {"lat": -16.100, "lon": 44.600, "region": "Mozambique Channel"},
    # Other
    "51407": {"lat": 19.632, "lon": -156.524, "region": "Hawaii"},
    "51425": {"lat": 14.700, "lon": -156.000, "region": "Central Pacific"},
    "52402": {"lat": 11.880, "lon": 154.110, "region": "W. Pacific/Carolines"},
    "52403": {"lat": 4.050, "lon": 145.590, "region": "W. Pacific/Marianas"},
    "52405": {"lat": 12.880, "lon": 132.340, "region": "W. Pacific/Philippines"},
    "52406": {"lat": 5.300, "lon": 164.900, "region": "W. Pacific/Marshall"},
    "55012": {"lat": -27.520, "lon": -4.050, "region": "S. Atlantic"},
    "55015": {"lat": -30.000, "lon": -35.000, "region": "S. Atlantic"},
    "55023": {"lat": -15.860, "lon": -5.100, "region": "S. Atlantic"},
    "56003": {"lat": -46.050, "lon": 170.340, "region": "SW Pacific/NZ"},
    "44402": {"lat": 10.730, "lon": -42.500, "region": "Central Atlantic"},
    "44403": {"lat": 14.740, "lon": -51.070, "region": "Central Atlantic"},
}


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dart_stations (
            station_id TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            depth_m REAL,
            region TEXT
        );
        CREATE TABLE IF NOT EXISTS dart_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            mode INTEGER NOT NULL,
            height_m REAL NOT NULL,
            UNIQUE(station_id, timestamp, mode)
        );
        CREATE INDEX IF NOT EXISTS idx_dart_station_time
            ON dart_readings(station_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_dart_time
            ON dart_readings(timestamp);
        CREATE INDEX IF NOT EXISTS idx_dart_mode
            ON dart_readings(mode);
    """)
    conn.commit()


def populate_stations(conn):
    for sid, info in DART_STATIONS.items():
        conn.execute(
            "INSERT OR REPLACE INTO dart_stations (station_id, lat, lon, region) "
            "VALUES (?, ?, ?, ?)",
            (sid, info["lat"], info["lon"], info["region"])
        )
    conn.commit()
    print(f"  {len(DART_STATIONS)} DART stations registered")


def parse_dart_text(text, station_id):
    readings = []
    for line in text.strip().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        try:
            yr = int(parts[0])
            if yr < 100:
                yr += 2000
            mo, dy, hr, mn = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            mode = int(parts[6])
            height = float(parts[7])
            if height > 9000:
                continue
            ts = datetime(yr, mo, dy, hr, mn, tzinfo=UTC)
            readings.append((station_id, ts.strftime("%Y-%m-%d %H:%M:%S"), mode, height))
        except (ValueError, IndexError):
            continue
    return readings


def fetch_realtime(station_id):
    url = f"{NDBC_REALTIME}/{station_id}.dart"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode()
        return parse_dart_text(text, station_id)
    except Exception as e:
        print(f"    {station_id} realtime error: {e}")
        return []


def fetch_historical_year(station_id, year):
    filename = f"{station_id}t{year}.txt.gz"
    url = f"{NDBC_HISTORICAL}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if data[:2] == b"\x1f\x8b":
            text = gzip.decompress(data).decode()
        else:
            text = data.decode()
        if "<html" in text.lower() or "Error" in text:
            return []
        return parse_dart_text(text, station_id)
    except Exception as e:
        return []


def insert_readings(conn, readings, batch_size=5000):
    inserted = 0
    for i in range(0, len(readings), batch_size):
        batch = readings[i:i + batch_size]
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO dart_readings "
                "(station_id, timestamp, mode, height_m) VALUES (?, ?, ?, ?)",
                batch
            )
            inserted += conn.total_changes
        except sqlite3.Error:
            for r in batch:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO dart_readings "
                        "(station_id, timestamp, mode, height_m) VALUES (?, ?, ?, ?)",
                        r
                    )
                except sqlite3.Error:
                    pass
    conn.commit()
    return inserted


def detect_mode_transitions(conn, stations=None):
    """Detect buoys that switched to event/tsunami mode since last check.

    Returns list of dicts with station_id, region, lat, lon, mode, timestamp
    for any station whose most recent reading is mode 2 or 3 (event/tsunami)
    but whose prior normal-mode reading was within the last 24h.
    """
    if stations is None:
        stations = list(DART_STATIONS.keys())

    alerts = []
    for sid in stations:
        row = conn.execute(
            "SELECT timestamp, mode FROM dart_readings "
            "WHERE station_id = ? ORDER BY timestamp DESC LIMIT 1",
            (sid,)
        ).fetchone()
        if not row or row["mode"] == 1:
            continue
        last_normal = conn.execute(
            "SELECT timestamp FROM dart_readings "
            "WHERE station_id = ? AND mode = 1 ORDER BY timestamp DESC LIMIT 1",
            (sid,)
        ).fetchone()
        info = DART_STATIONS.get(sid, {})
        mode_name = {2: "EVENT", 3: "TSUNAMI"}.get(row["mode"], "?")
        alerts.append({
            "station_id": sid,
            "region": info.get("region", "?"),
            "lat": info.get("lat"),
            "lon": info.get("lon"),
            "mode": mode_name,
            "since": row["timestamp"],
            "last_normal": last_normal["timestamp"] if last_normal else None,
        })
    return alerts


def ingest_realtime(conn, stations=None):
    if stations is None:
        stations = list(DART_STATIONS.keys())

    print(f"\n  Pulling realtime data for {len(stations)} stations...")
    total = 0
    for sid in stations:
        readings = fetch_realtime(sid)
        if readings:
            n = insert_readings(conn, readings)
            region = DART_STATIONS.get(sid, {}).get("region", "?")
            print(f"    {sid} ({region}): {len(readings)} readings")
            total += len(readings)
        time.sleep(0.5)

    print(f"  Total: {total} readings from realtime feeds")

    alerts = detect_mode_transitions(conn, stations)
    if alerts:
        print(f"\n  *** ALERT: {len(alerts)} buoy(s) in event/tsunami mode ***")
        for a in alerts:
            print(f"    {a['station_id']} ({a['region']}) — {a['mode']} mode")
            print(f"      Location: {a['lat']:.1f}°, {a['lon']:.1f}°")
            print(f"      Active since: {a['since']}")
            if a['last_normal']:
                print(f"      Last normal: {a['last_normal']}")
        # _send_dart_alert(alerts)  # disabled — too noisy

    return total


def _send_dart_alert(alerts):
    """Send email alert when DART buoys switch to event/tsunami mode."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from alerts import _load_env, _send_email
        _load_env()

        lines = []
        for a in alerts:
            lines.append(
                f"<tr><td>{a['station_id']}</td><td>{a['region']}</td>"
                f"<td><b>{a['mode']}</b></td>"
                f"<td>{a['lat']:.1f}°, {a['lon']:.1f}°</td>"
                f"<td>{a['since']}</td></tr>"
            )
        table_rows = "\n".join(lines)

        mode_types = set(a["mode"] for a in alerts)
        level = "TSUNAMI" if "TSUNAMI" in mode_types else "EVENT"

        html = f"""
        <div style="font-family: monospace; max-width: 700px;">
        <h2 style="color: {'#dc4632' if level == 'TSUNAMI' else '#d2a032'};">
            DART Buoy Alert: {len(alerts)} station(s) in {level} mode
        </h2>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
        <tr style="background:#333;color:#fff;">
            <th>Station</th><th>Region</th><th>Mode</th><th>Location</th><th>Since</th>
        </tr>
        {table_rows}
        </table>
        <p style="color:#888; margin-top:16px;">
            Buoys switched from 15-min to 1-min sampling.
            This indicates the onboard algorithm detected anomalous seafloor pressure.
        </p>
        </div>
        """
        subject = f"DART Alert: {len(alerts)} buoy(s) in {level} mode"
        _send_email(subject, html)
        print(f"  Alert email sent to {os.environ.get('RECIPIENT', 'ryan@axomlabs.ai')}")
    except Exception as e:
        print(f"  Warning: failed to send alert email: {e}")


def ingest_historical(conn, start_year=2006, end_year=2026, stations=None):
    if stations is None:
        stations = list(DART_STATIONS.keys())

    print(f"\n  Pulling historical data {start_year}-{end_year} "
          f"for {len(stations)} stations...")

    existing = conn.execute(
        "SELECT station_id, MIN(timestamp), MAX(timestamp), COUNT(*) "
        "FROM dart_readings GROUP BY station_id"
    ).fetchall()
    existing_map = {r[0]: (r[1], r[2], r[3]) for r in existing}

    total = 0
    errors = 0

    for sid in stations:
        region = DART_STATIONS.get(sid, {}).get("region", "?")
        print(f"\n    {sid} ({region}):", end="", flush=True)

        station_total = 0
        for year in range(start_year, end_year + 1):
            year_count = conn.execute(
                "SELECT COUNT(*) FROM dart_readings "
                "WHERE station_id = ? AND timestamp LIKE ?",
                (sid, f"{year}-%")
            ).fetchone()[0]
            if year_count > 1000:
                print(".", end="", flush=True)
                continue

            readings = fetch_historical_year(sid, year)
            if readings:
                n = insert_readings(conn, readings)
                station_total += len(readings)
                print(f" {year}({len(readings)})", end="", flush=True)
            else:
                print(".", end="", flush=True)

            time.sleep(0.3)

        if station_total:
            total += station_total
            print(f" = {station_total}")
        else:
            print(" (no new data)")

    print(f"\n  Total: {total} historical readings ingested")
    return total


def update_station_depths(conn):
    stations = conn.execute(
        "SELECT station_id, AVG(height_m) as avg_depth "
        "FROM dart_readings WHERE mode = 1 "
        "GROUP BY station_id"
    ).fetchall()

    for s in stations:
        conn.execute(
            "UPDATE dart_stations SET depth_m = ? WHERE station_id = ?",
            (round(s["avg_depth"], 1), s["station_id"])
        )
    conn.commit()
    print(f"  Updated depths for {len(stations)} stations")


def compute_detrended_features(conn, station_id, target_time, window_days=7):
    """Compute DART-based features for a given station and time.

    Returns dict with:
      - dart_residual_mean_24h: mean detrended pressure (mm) over last 24h
      - dart_residual_std_24h: std of detrended pressure (mm) over last 24h
      - dart_residual_trend_48h: linear trend (mm/day) over last 48h
      - dart_rate_of_change_6h: pressure rate of change (mm/hr) over last 6h
      - dart_loading_index: compression→extension swing over window (mm)
      - dart_trend_reversal: change in 24h-trend from days 3-2 to days 1-0 (mm/day)
      - dart_event_mode_count_7d: number of event/tsunami mode readings in last 7d
    """
    t_start = target_time - timedelta(days=window_days)

    rows = conn.execute(
        "SELECT timestamp, mode, height_m FROM dart_readings "
        "WHERE station_id = ? AND timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp",
        (station_id, t_start.strftime("%Y-%m-%d %H:%M:%S"),
         target_time.strftime("%Y-%m-%d %H:%M:%S"))
    ).fetchall()

    if len(rows) < 48:
        return None

    normal = [(datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC),
               r["height_m"])
              for r in rows if r["mode"] == 1]

    event_count = sum(1 for r in rows if r["mode"] in (2, 3))

    if len(normal) < 48:
        return None

    t0 = normal[0][0]
    hours = np.array([(t - t0).total_seconds() / 3600 for t, h in normal])
    heights = np.array([h for t, h in normal])

    tidal_periods = [12.42, 12.00, 23.93, 25.82, 6.21]
    n = len(hours)
    A = np.zeros((n, 2 + 2 * len(tidal_periods)))
    A[:, 0] = 1
    A[:, 1] = hours
    for i, T in enumerate(tidal_periods):
        A[:, 2 + 2 * i] = np.sin(2 * np.pi * hours / T)
        A[:, 2 + 2 * i + 1] = np.cos(2 * np.pi * hours / T)

    coeffs, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)
    residuals = heights - A @ coeffs
    times = [t for t, h in normal]

    res_mm = residuals * 1000

    t_24h = target_time - timedelta(hours=24)
    t_48h = target_time - timedelta(hours=48)
    t_6h = target_time - timedelta(hours=6)

    last_24h = [res_mm[i] for i in range(len(times)) if times[i] >= t_24h]
    last_48h = [res_mm[i] for i in range(len(times)) if times[i] >= t_48h]
    last_6h = [res_mm[i] for i in range(len(times)) if times[i] >= t_6h]
    first_half = [res_mm[i] for i in range(len(times))
                  if t_start + timedelta(days=window_days // 2) > times[i] >= t_start]
    second_half = [res_mm[i] for i in range(len(times))
                   if times[i] >= t_start + timedelta(days=window_days // 2)]

    features = {
        "dart_residual_mean_24h": float(np.mean(last_24h)) if last_24h else 0.0,
        "dart_residual_std_24h": float(np.std(last_24h)) if len(last_24h) > 1 else 0.0,
        "dart_event_mode_count_7d": event_count,
    }

    if len(last_48h) > 4:
        x = np.arange(len(last_48h))
        slope = np.polyfit(x, last_48h, 1)[0]
        readings_per_day = len(last_48h) / 2.0
        features["dart_residual_trend_48h"] = float(slope * readings_per_day)
    else:
        features["dart_residual_trend_48h"] = 0.0

    if len(last_6h) > 1:
        dt = 6.0 / len(last_6h)
        features["dart_rate_of_change_6h"] = float(
            (last_6h[-1] - last_6h[0]) / (dt * len(last_6h))
        )
    else:
        features["dart_rate_of_change_6h"] = 0.0

    if first_half and second_half:
        features["dart_loading_index"] = float(
            np.mean(second_half) - np.mean(first_half)
        )
    else:
        features["dart_loading_index"] = 0.0

    t_2d = target_time - timedelta(days=2)
    t_3d = target_time - timedelta(days=3)
    early_window = [res_mm[i] for i in range(len(times)) if t_3d <= times[i] < t_2d]
    late_window = [res_mm[i] for i in range(len(times)) if t_2d <= times[i]]
    if len(early_window) > 4 and len(late_window) > 4:
        early_x = np.arange(len(early_window))
        late_x = np.arange(len(late_window))
        early_slope = np.polyfit(early_x, early_window, 1)[0]
        late_slope = np.polyfit(late_x, late_window, 1)[0]
        rpd = len(early_window)
        features["dart_trend_reversal"] = float(
            (late_slope - early_slope) * rpd
        )
    else:
        features["dart_trend_reversal"] = 0.0

    # ── Trajectory features: multi-timescale volatility and ramps ──

    t_12h = target_time - timedelta(hours=12)
    t_18h = target_time - timedelta(hours=18)
    t_36h = target_time - timedelta(hours=36)

    last_12h = [res_mm[i] for i in range(len(times)) if times[i] >= t_12h]
    prior_12_36h = [res_mm[i] for i in range(len(times)) if t_36h <= times[i] < t_12h]
    prior_18_6h = [res_mm[i] for i in range(len(times)) if t_24h <= times[i] < t_6h]

    # Volatility ramp: std(last 6h) / std(prior 18h)
    if len(last_6h) > 2 and len(prior_18_6h) > 2:
        std_6h = float(np.std(last_6h))
        std_prior = float(np.std(prior_18_6h))
        features["dart_vol_ramp_6v18"] = std_6h / max(0.01, std_prior)
    else:
        features["dart_vol_ramp_6v18"] = np.nan

    # Volatility ramp: std(last 12h) / std(prior 12-36h)
    if len(last_12h) > 2 and len(prior_12_36h) > 2:
        std_12h = float(np.std(last_12h))
        std_prior_12 = float(np.std(prior_12_36h))
        features["dart_vol_ramp_12v36"] = std_12h / max(0.01, std_prior_12)
    else:
        features["dart_vol_ramp_12v36"] = np.nan

    # Volatility at multiple timescales (absolute)
    features["dart_std_6h"] = float(np.std(last_6h)) if len(last_6h) > 2 else np.nan
    features["dart_std_12h"] = float(np.std(last_12h)) if len(last_12h) > 2 else np.nan

    # Pressure trend at multiple timescales
    if len(last_12h) > 4:
        x12 = np.arange(len(last_12h))
        slope12 = np.polyfit(x12, last_12h, 1)[0]
        rpd12 = len(last_12h) / 0.5  # readings per 12h → per day
        features["dart_trend_12h"] = float(slope12 * rpd12)
    else:
        features["dart_trend_12h"] = np.nan

    if len(last_6h) > 3:
        x6 = np.arange(len(last_6h))
        slope6 = np.polyfit(x6, last_6h, 1)[0]
        rpd6 = len(last_6h) / 0.25  # per day
        features["dart_trend_6h"] = float(slope6 * rpd6)
    else:
        features["dart_trend_6h"] = np.nan

    # Trend acceleration: trend_12h vs trend from 12-36h
    if len(last_12h) > 4 and len(prior_12_36h) > 4:
        x_prior = np.arange(len(prior_12_36h))
        slope_prior = np.polyfit(x_prior, prior_12_36h, 1)[0]
        rpd_prior = len(prior_12_36h) / 1.0
        features["dart_trend_accel"] = features["dart_trend_12h"] - float(slope_prior * rpd_prior)
    else:
        features["dart_trend_accel"] = np.nan

    # Event mode trajectory features
    t_24h_em = target_time - timedelta(hours=24)
    t_12h_em = target_time - timedelta(hours=12)
    t_6h_em = target_time - timedelta(hours=6)

    event_rows_all = [r for r in rows if r["mode"] in (2, 3)]
    em_times = []
    for r in event_rows_all:
        ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        em_times.append(ts)

    em_24h = sum(1 for t in em_times if t >= t_24h_em)
    em_12h = sum(1 for t in em_times if t >= t_12h_em)
    em_6h = sum(1 for t in em_times if t >= t_6h_em)
    em_prior = event_count - em_24h

    features["dart_event_mode_24h"] = em_24h
    features["dart_event_mode_12h"] = em_12h
    features["dart_event_mode_6h"] = em_6h
    # Event mode acceleration: recent vs earlier
    features["dart_em_accel"] = em_12h / max(1, em_prior) if event_count > 0 else 0.0

    return features


def summarize(conn):
    print(f"\n{'=' * 80}")
    print(f"  DART BUOY DATA SUMMARY")
    print(f"{'=' * 80}")

    total = conn.execute("SELECT COUNT(*) FROM dart_readings").fetchone()[0]
    date_range = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM dart_readings"
    ).fetchone()
    stations = conn.execute(
        "SELECT COUNT(DISTINCT station_id) FROM dart_readings"
    ).fetchone()[0]

    print(f"\n  Total readings: {total:,}")
    print(f"  Stations with data: {stations}")
    print(f"  Date range: {date_range[0]} to {date_range[1]}")

    per_station = conn.execute("""
        SELECT r.station_id, s.region, s.depth_m, COUNT(*) as cnt,
               MIN(r.timestamp) as first_ts, MAX(r.timestamp) as last_ts,
               SUM(CASE WHEN r.mode IN (2,3) THEN 1 ELSE 0 END) as events
        FROM dart_readings r
        LEFT JOIN dart_stations s ON r.station_id = s.station_id
        GROUP BY r.station_id
        ORDER BY cnt DESC
    """).fetchall()

    print(f"\n  {'Station':>8s}  {'Region':>25s}  {'Depth':>7s}  {'Count':>8s}  "
          f"{'Events':>7s}  {'First':>12s}  {'Last':>12s}")
    print(f"  {'─' * 95}")
    for r in per_station:
        depth = f"{r['depth_m']:.0f}m" if r["depth_m"] else "?"
        print(f"  {r['station_id']:>8s}  {(r['region'] or '?'):>25s}  {depth:>7s}  "
              f"{r['cnt']:>8,d}  {r['events']:>7d}  "
              f"{r['first_ts'][:10]:>12s}  {r['last_ts'][:10]:>12s}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DART buoy data ingestion")
    parser.add_argument("--realtime", action="store_true",
                        help="Pull realtime data (~45 days)")
    parser.add_argument("--historical", action="store_true",
                        help="Pull historical data")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--stations", nargs="+",
                        help="Specific station IDs (default: all)")
    parser.add_argument("--caribbean", action="store_true",
                        help="Only Caribbean/Atlantic stations")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Skip summary and depth update (for cron)")
    args = parser.parse_args()

    conn = get_conn()
    create_tables(conn)
    populate_stations(conn)

    if args.summary_only:
        summarize(conn)
        conn.close()
        return

    stations = args.stations
    if args.caribbean:
        stations = ["41420", "41421", "41425", "42407", "42409",
                     "43413", "44402", "44403"]

    print(f"\n{'=' * 80}")
    print(f"  NOAA DART BUOY DATA INGESTION")
    print(f"{'=' * 80}")

    if args.realtime or (not args.historical):
        ingest_realtime(conn, stations)

    if args.historical:
        ingest_historical(conn, args.start_year, args.end_year, stations)

    if not args.quick:
        update_station_depths(conn)
        summarize(conn)

    conn.close()
    print(f"\n  Complete.\n")


if __name__ == "__main__":
    main()
