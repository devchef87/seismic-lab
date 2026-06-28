#!/usr/bin/env python3
"""
export_data.py — Export the SeismicLab SQLite database to compressed Parquet
files, optionally pushing to HuggingFace Hub.

Output structure:
    output_dir/
    ├── samples/
    │   ├── 2015.parquet ... 2026.parquet
    ├── station_metrics/
    │   ├── 2021.parquet ... 2026.parquet
    ├── dart_readings/
    │   ├── 2006-2010.parquet, 2011-2015.parquet, ...
    ├── earthquakes.parquet
    ├── thermal_anomalies.parquet
    ├── dart_stations.parquet
    ├── volcanoes.parquet
    ├── eruptions.parquet
    ├── event_analyses.parquet
    ├── predictions.parquet
    ├── prediction_snapshots.parquet
    ├── ingest_log.parquet

Usage:
    python scripts/export_data.py
    python scripts/export_data.py --db /path/to/seismiclab.db --output export/
    python scripts/export_data.py --push
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("export_data")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DB = "data/seismiclab.db"
DEFAULT_OUTPUT = "export/"
HF_REPO = "Groovedev/seismic-lab-data"
CHUNK_SIZE = 500_000

# ---------------------------------------------------------------------------
# Year partitioning helpers
# ---------------------------------------------------------------------------
DART_BINS = [
    ("2006-2010", "2006-01-01", "2010-12-31 23:59:59"),
    ("2011-2015", "2011-01-01", "2015-12-31 23:59:59"),
    ("2016-2020", "2016-01-01", "2020-12-31 23:59:59"),
    ("2021-2023", "2021-01-01", "2023-12-31 23:59:59"),
    ("2024-2026", "2024-01-01", "2026-12-31 23:59:59"),
]


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        log.error("Database not found: %s", db_path)
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_partitioned_by_year(
    conn: sqlite3.Connection,
    table: str,
    ts_col: str,
    output_dir: str,
    year_start: int,
    year_end: int,
    columns: str = "*",
) -> int:
    """Export a table partitioned into one Parquet file per year."""
    dest = os.path.join(output_dir, table)
    _ensure_dir(dest)
    total = 0

    for year in range(year_start, year_end + 1):
        lo = f"{year}-01-01"
        hi = f"{year + 1}-01-01"
        query = f"SELECT {columns} FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ?"
        frames = []
        for chunk in pd.read_sql_query(query, conn, params=(lo, hi), chunksize=CHUNK_SIZE):
            frames.append(chunk)

        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)
        row_count = len(df)
        if row_count == 0:
            continue

        out_path = os.path.join(dest, f"{year}.parquet")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        total += row_count
        log.info("  %s/%d.parquet — %s rows", table, year, f"{row_count:,}")

    return total


def export_dart_readings_binned(conn: sqlite3.Connection, output_dir: str) -> int:
    """Export dart_readings into multi-year bins."""
    dest = os.path.join(output_dir, "dart_readings")
    _ensure_dir(dest)
    total = 0

    for label, lo, hi in DART_BINS:
        query = "SELECT * FROM dart_readings WHERE timestamp >= ? AND timestamp <= ?"
        frames = []
        for chunk in pd.read_sql_query(query, conn, params=(lo, hi), chunksize=CHUNK_SIZE):
            frames.append(chunk)

        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)
        row_count = len(df)
        if row_count == 0:
            continue

        out_path = os.path.join(dest, f"{label}.parquet")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        total += row_count
        log.info("  dart_readings/%s.parquet — %s rows", label, f"{row_count:,}")

    return total


def export_single_table(
    conn: sqlite3.Connection,
    table: str,
    output_dir: str,
) -> int:
    """Export a smaller table as a single Parquet file."""
    frames = []
    for chunk in pd.read_sql_query(f"SELECT * FROM {table}", conn, chunksize=CHUNK_SIZE):
        frames.append(chunk)

    if not frames:
        log.info("  %s.parquet — 0 rows (empty)", table)
        return 0

    df = pd.concat(frames, ignore_index=True)
    row_count = len(df)
    out_path = os.path.join(output_dir, f"{table}.parquet")
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    log.info("  %s.parquet — %s rows", table, f"{row_count:,}")
    return row_count


# ---------------------------------------------------------------------------
# Push to HuggingFace
# ---------------------------------------------------------------------------

def push_to_hub(output_dir: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    log.info("Creating / verifying HuggingFace repo: %s", HF_REPO)
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True, private=False)

    log.info("Uploading %s -> %s ...", output_dir, HF_REPO)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message="Update seismic-lab-data export",
    )
    log.info("Push complete: https://huggingface.co/datasets/%s", HF_REPO)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SeismicLab SQLite DB to compressed Parquet files."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"Path to SQLite database (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory for Parquet files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push exported files to HuggingFace Hub after export",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = os.path.abspath(args.db)
    output_dir = os.path.abspath(args.output)
    _ensure_dir(output_dir)

    log.info("Database : %s", db_path)
    log.info("Output   : %s", output_dir)

    conn = _connect(db_path)
    t0 = time.time()
    grand_total = 0

    # ---- Partitioned tables ------------------------------------------------

    log.info("Exporting samples (partitioned by year) ...")
    n = export_partitioned_by_year(
        conn, "samples", "timestamp", output_dir,
        year_start=2002, year_end=2026,
    )
    grand_total += n
    log.info("  samples total: %s rows", f"{n:,}")

    log.info("Exporting station_metrics (partitioned by year) ...")
    n = export_partitioned_by_year(
        conn, "station_metrics", "timestamp", output_dir,
        year_start=2021, year_end=2026,
    )
    grand_total += n
    log.info("  station_metrics total: %s rows", f"{n:,}")

    log.info("Exporting dart_readings (binned) ...")
    n = export_dart_readings_binned(conn, output_dir)
    grand_total += n
    log.info("  dart_readings total: %s rows", f"{n:,}")

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
        log.info("Exporting %s ...", table)
        n = export_single_table(conn, table, output_dir)
        grand_total += n

    conn.close()

    elapsed = time.time() - t0
    log.info("Export complete — %s total rows in %.1fs", f"{grand_total:,}", elapsed)

    # ---- Optional push -----------------------------------------------------

    if args.push:
        push_to_hub(output_dir)


if __name__ == "__main__":
    main()
