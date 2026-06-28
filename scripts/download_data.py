#!/usr/bin/env python3
"""
download_data.py — Download Parquet files from HuggingFace Hub and rebuild
the SeismicLab SQLite database locally.

Usage:
    python scripts/download_data.py                       # Full download + build
    python scripts/download_data.py --since 2026-06-01    # Incremental (only recent data)
    python scripts/download_data.py --db data/custom.db   # Custom output path
"""

import argparse
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger("download_data")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HF_REPO = "AxomLabs/seismic-lab-data"
DEFAULT_DB = "data/seismiclab.db"

# ---------------------------------------------------------------------------
# Schema — mirrors the original SeismicLab DB exactly
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    metric TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    lat REAL,
    lon REAL,
    meta TEXT,
    ingested_at TEXT NOT NULL,
    UNIQUE(source, metric, timestamp, lat, lon)
);

CREATE TABLE IF NOT EXISTS earthquakes (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    magnitude REAL NOT NULL,
    depth_km REAL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    place TEXT,
    type TEXT
);

CREATE TABLE IF NOT EXISTS volcanoes (
    volcano_number INTEGER PRIMARY KEY,
    name TEXT,
    lat REAL,
    lon REAL,
    elevation_m INTEGER,
    volcano_type TEXT,
    tectonic_setting TEXT,
    region TEXT,
    subregion TEXT,
    last_eruption_year INTEGER,
    rock_type TEXT
);

CREATE TABLE IF NOT EXISTS eruptions (
    activity_id INTEGER PRIMARY KEY,
    volcano_number INTEGER,
    volcano_name TEXT,
    lat REAL,
    lon REAL,
    vei INTEGER,
    start_date TEXT,
    end_date TEXT,
    start_year INTEGER,
    continuing INTEGER,
    FOREIGN KEY (volcano_number) REFERENCES volcanoes(volcano_number)
);

CREATE TABLE IF NOT EXISTS thermal_anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat REAL,
    lon REAL,
    brightness REAL,
    frp REAL,
    acq_date TEXT,
    acq_time TEXT,
    satellite TEXT,
    instrument TEXT,
    confidence TEXT,
    daynight TEXT,
    source TEXT,
    nearest_volcano_id INTEGER,
    nearest_volcano_dist_km REAL
);

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

CREATE TABLE IF NOT EXISTS station_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    station TEXT NOT NULL,
    amp_min REAL NOT NULL,
    amp_max REAL NOT NULL,
    amp_mean REAL NOT NULL,
    sta_lta_ratio REAL NOT NULL,
    triggered INTEGER NOT NULL DEFAULT 0,
    sample_rate REAL,
    UNIQUE(station, timestamp)
);

CREATE TABLE IF NOT EXISTS event_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    earthquake_id TEXT UNIQUE,
    analyzed_at TEXT,
    magnitude REAL,
    lat REAL,
    lon REAL,
    depth_km REAL,
    place TEXT,
    event_timestamp TEXT,
    event_type TEXT,
    risk_score REAL,
    risk_level TEXT,
    mainshock_mag REAL,
    mainshock_dist_km REAL,
    hours_since_mainshock REAL,
    omori_expected_rate REAL,
    omori_observed_rate REAL,
    omori_residual REAL,
    sequence_count_24h INTEGER,
    sequence_count_72h INTEGER,
    mag_trend TEXT,
    migration_speed_km_day REAL,
    migration_bearing REAL,
    doublet_flag INTEGER DEFAULT 0,
    coulomb_loading REAL,
    dart_elevated_count INTEGER,
    dart_max_deviation REAL,
    volcanic_hotspots_500km INTEGER,
    volcanic_max_frp REAL,
    background_rate_30d REAL,
    rate_anomaly REAL,
    b_value_14d REAL,
    b_value_trend TEXT,
    summary TEXT,
    watch_items TEXT,
    llm_analysis TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    zone_name TEXT NOT NULL,
    window_hours INTEGER NOT NULL,
    probability REAL NOT NULL,
    magnitude_est REAL,
    magnitude_lo REAL,
    magnitude_hi REAL,
    threat_score REAL,
    tidal_stress REAL,
    top_features TEXT,
    outcome TEXT DEFAULT "pending",
    resolved_at TEXT,
    actual_mag REAL,
    actual_event_id TEXT,
    lead_time_hours REAL
);

CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    probability REAL NOT NULL,
    threat_score REAL,
    magnitude_est REAL
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    sample_count INTEGER,
    new_count INTEGER,
    duration_ms REAL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS gps_stations (
    site TEXT PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    height_m REAL,
    zone TEXT,
    first_epoch TEXT,
    last_epoch TEXT,
    n_epochs INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS gps_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site TEXT NOT NULL,
    epoch TEXT NOT NULL,
    delta_n_mm REAL,
    delta_e_mm REAL,
    delta_u_mm REAL,
    sig_n_mm REAL,
    sig_e_mm REAL,
    sig_u_mm REAL,
    UNIQUE(site, epoch)
);

CREATE TABLE IF NOT EXISTS gps_strain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site TEXT NOT NULL,
    epoch TEXT NOT NULL,
    velocity_n_mm_yr REAL,
    velocity_e_mm_yr REAL,
    residual_n_mm REAL,
    residual_e_mm REAL,
    residual_u_mm REAL,
    anomaly_score REAL,
    UNIQUE(site, epoch)
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_samples_source_ts ON samples(source, timestamp);
CREATE INDEX IF NOT EXISTS idx_samples_metric_ts ON samples(metric, timestamp);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(timestamp);
CREATE INDEX IF NOT EXISTS idx_samples_src_met_ts ON samples(source, metric, timestamp);
CREATE INDEX IF NOT EXISTS idx_samples_metric_ts_loc ON samples(metric, timestamp, lat, lon);
CREATE INDEX IF NOT EXISTS idx_samples_donki ON samples(source, metric, timestamp) WHERE source = "nasa_donki";

CREATE INDEX IF NOT EXISTS idx_eq_ts ON earthquakes(timestamp);
CREATE INDEX IF NOT EXISTS idx_eq_mag ON earthquakes(magnitude);
CREATE INDEX IF NOT EXISTS idx_eq_loc_ts ON earthquakes(lat, lon, timestamp);

CREATE INDEX IF NOT EXISTS idx_thermal_date ON thermal_anomalies(acq_date);
CREATE INDEX IF NOT EXISTS idx_thermal_loc ON thermal_anomalies(lat, lon);
CREATE INDEX IF NOT EXISTS idx_thermal_volcano ON thermal_anomalies(nearest_volcano_id);
CREATE INDEX IF NOT EXISTS idx_thermal_date_loc ON thermal_anomalies(acq_date, lat, lon);

CREATE INDEX IF NOT EXISTS idx_dart_station_time ON dart_readings(station_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_dart_time ON dart_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_dart_mode ON dart_readings(mode);

CREATE INDEX IF NOT EXISTS idx_sm_station_ts ON station_metrics(station, timestamp);
CREATE INDEX IF NOT EXISTS idx_sm_ts ON station_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_ea_timestamp ON event_analyses(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_ea_risk ON event_analyses(risk_score);

CREATE INDEX IF NOT EXISTS idx_pred_zone ON predictions(zone_id, issued_at);
CREATE INDEX IF NOT EXISTS idx_pred_outcome ON predictions(outcome);
CREATE INDEX IF NOT EXISTS idx_pred_expires ON predictions(expires_at);

CREATE INDEX IF NOT EXISTS idx_snap_zone_ts ON prediction_snapshots(zone_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_gps_pos_site_epoch ON gps_positions(site, epoch);
CREATE INDEX IF NOT EXISTS idx_gps_strain_site ON gps_strain(site, epoch);
"""

# ---------------------------------------------------------------------------
# Column maps — columns to insert for each table (excluding autoincrement id
# where the parquet carries the original id)
# ---------------------------------------------------------------------------

TABLE_COLUMNS = {
    "samples": [
        "id", "source", "metric", "timestamp", "value",
        "unit", "lat", "lon", "meta", "ingested_at",
    ],
    "earthquakes": [
        "id", "timestamp", "magnitude", "depth_km",
        "lat", "lon", "place", "type",
    ],
    "volcanoes": [
        "volcano_number", "name", "lat", "lon", "elevation_m",
        "volcano_type", "tectonic_setting", "region", "subregion",
        "last_eruption_year", "rock_type",
    ],
    "eruptions": [
        "activity_id", "volcano_number", "volcano_name", "lat", "lon",
        "vei", "start_date", "end_date", "start_year", "continuing",
    ],
    "thermal_anomalies": [
        "id", "lat", "lon", "brightness", "frp", "acq_date", "acq_time",
        "satellite", "instrument", "confidence", "daynight", "source",
        "nearest_volcano_id", "nearest_volcano_dist_km",
    ],
    "dart_stations": ["station_id", "lat", "lon", "depth_m", "region"],
    "dart_readings": ["id", "station_id", "timestamp", "mode", "height_m"],
    "station_metrics": [
        "id", "timestamp", "station", "amp_min", "amp_max",
        "amp_mean", "sta_lta_ratio", "triggered", "sample_rate",
    ],
    "event_analyses": [
        "id", "earthquake_id", "analyzed_at", "magnitude", "lat", "lon",
        "depth_km", "place", "event_timestamp", "event_type", "risk_score",
        "risk_level", "mainshock_mag", "mainshock_dist_km",
        "hours_since_mainshock", "omori_expected_rate", "omori_observed_rate",
        "omori_residual", "sequence_count_24h", "sequence_count_72h",
        "mag_trend", "migration_speed_km_day", "migration_bearing",
        "doublet_flag", "coulomb_loading", "dart_elevated_count",
        "dart_max_deviation", "volcanic_hotspots_500km", "volcanic_max_frp",
        "background_rate_30d", "rate_anomaly", "b_value_14d", "b_value_trend",
        "summary", "watch_items", "llm_analysis",
    ],
    "predictions": [
        "id", "issued_at", "expires_at", "zone_id", "zone_name",
        "window_hours", "probability", "magnitude_est", "magnitude_lo",
        "magnitude_hi", "threat_score", "tidal_stress", "top_features",
        "outcome", "resolved_at", "actual_mag", "actual_event_id",
        "lead_time_hours",
    ],
    "prediction_snapshots": [
        "id", "timestamp", "zone_id", "probability",
        "threat_score", "magnitude_est",
    ],
    "ingest_log": [
        "id", "source", "fetched_at", "sample_count",
        "new_count", "duration_ms", "error",
    ],
}

# Batch insert size
INSERT_BATCH = 50_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _year_from_filename(name: str) -> int | None:
    """Extract a single year from a filename like '2024.parquet'."""
    m = re.match(r"^(\d{4})\.parquet$", name)
    return int(m.group(1)) if m else None


def _year_range_from_filename(name: str) -> tuple[int, int] | None:
    """Extract year range from '2021-2023.parquet'."""
    m = re.match(r"^(\d{4})-(\d{4})\.parquet$", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _file_passes_since(filename: str, since_year: int) -> bool:
    """Return True if this parquet file might contain data at or after since_year."""
    yr = _year_from_filename(filename)
    if yr is not None:
        return yr >= since_year

    rng = _year_range_from_filename(filename)
    if rng is not None:
        return rng[1] >= since_year

    # Single-file tables — always include
    return True


def _insert_df(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    since: str | None = None,
) -> int:
    """Insert a DataFrame into the given table, using INSERT OR IGNORE to
    handle duplicates during incremental loads."""
    cols = TABLE_COLUMNS.get(table)
    if cols is None:
        log.warning("No column map for table %s — skipping", table)
        return 0

    # Filter to only columns that exist in both the dataframe and the schema
    available = [c for c in cols if c in df.columns]
    if not available:
        return 0

    df = df[available]

    # For incremental mode, filter rows by timestamp if the table has one
    if since:
        ts_col = None
        if "timestamp" in df.columns:
            ts_col = "timestamp"
        elif "acq_date" in df.columns:
            ts_col = "acq_date"
        elif "issued_at" in df.columns:
            ts_col = "issued_at"
        elif "fetched_at" in df.columns:
            ts_col = "fetched_at"

        if ts_col:
            before = len(df)
            df = df[df[ts_col] >= since]
            if len(df) < before:
                log.info("    filtered %s -> %s rows (since %s)",
                         f"{before:,}", f"{len(df):,}", since)

    if df.empty:
        return 0

    placeholders = ", ".join(["?"] * len(available))
    col_names = ", ".join(available)
    sql = f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})"

    rows_inserted = 0
    for start in range(0, len(df), INSERT_BATCH):
        batch = df.iloc[start : start + INSERT_BATCH]
        # Convert NaN to None for SQLite
        records = [
            tuple(None if pd.isna(v) else v for v in row)
            for row in batch.itertuples(index=False, name=None)
        ]
        conn.executemany(sql, records)
        rows_inserted += len(records)

    return rows_inserted


# ---------------------------------------------------------------------------
# Download + Build
# ---------------------------------------------------------------------------

def download_and_build(db_path: str, since: str | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    since_year = int(since[:4]) if since else None

    # Download snapshot
    log.info("Downloading dataset from HuggingFace: %s", HF_REPO)
    cache_dir = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns="*.parquet",
    )
    log.info("Downloaded to cache: %s", cache_dir)

    # Create DB
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-512000")  # 512MB cache

    log.info("Creating schema ...")
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    t0 = time.time()
    grand_total = 0

    # ---- Partitioned directories -------------------------------------------
    partitioned = {
        "samples": "samples",
        "station_metrics": "station_metrics",
        "dart_readings": "dart_readings",
    }

    for table, subdir in partitioned.items():
        dir_path = os.path.join(cache_dir, subdir)
        if not os.path.isdir(dir_path):
            log.warning("Directory not found: %s — skipping %s", dir_path, table)
            continue

        files = sorted(f for f in os.listdir(dir_path) if f.endswith(".parquet"))
        if since_year:
            files = [f for f in files if _file_passes_since(f, since_year)]

        log.info("Loading %s (%d files) ...", table, len(files))
        table_total = 0
        for fname in files:
            fpath = os.path.join(dir_path, fname)
            df = pd.read_parquet(fpath)
            n = _insert_df(conn, table, df, since=since)
            table_total += n
            log.info("  %s/%s — %s rows inserted", subdir, fname, f"{n:,}")
            conn.commit()

        grand_total += table_total
        log.info("  %s total: %s rows", table, f"{table_total:,}")

    # ---- Single-file tables ------------------------------------------------
    single_tables = [
        "earthquakes",
        "thermal_anomalies",
        "dart_stations",
        "volcanoes",
        "eruptions",
        "event_analyses",
        "predictions",
        "prediction_snapshots",
        "ingest_log",
    ]

    for table in single_tables:
        fpath = os.path.join(cache_dir, f"{table}.parquet")
        if not os.path.isfile(fpath):
            log.warning("File not found: %s — skipping", fpath)
            continue

        log.info("Loading %s ...", table)
        df = pd.read_parquet(fpath)
        n = _insert_df(conn, table, df, since=since)
        conn.commit()
        grand_total += n
        log.info("  %s — %s rows inserted", table, f"{n:,}")

    # ---- Create indexes after bulk insert (faster) -------------------------
    log.info("Creating indexes ...")
    conn.executescript(INDEXES_SQL)
    conn.commit()

    conn.execute("PRAGMA optimize")
    conn.close()

    elapsed = time.time() - t0
    log.info(
        "Build complete — %s total rows inserted into %s in %.1fs",
        f"{grand_total:,}", db_path, elapsed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download seismic-lab-data from HuggingFace and rebuild SQLite DB."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"Output SQLite database path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only download/import data after this date (e.g. 2026-06-01). "
             "Skips parquet files entirely outside the date range.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = os.path.abspath(args.db)
    log.info("Target DB: %s", db_path)
    if args.since:
        log.info("Incremental mode — since %s", args.since)

    download_and_build(db_path, since=args.since)


if __name__ == "__main__":
    main()
