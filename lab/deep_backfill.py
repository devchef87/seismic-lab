"""Deep historical backfill — fetch 5 years of seismic waveforms from FDSN/IRIS.

Downloads BHZ waveforms for all 24 stations from 2015-01-01 to present,
computes 5-minute aggregate metrics (amp_min, amp_max, amp_mean, sta_lta_ratio),
and stores in the station_metrics table.

Designed for quality:
- 6-hour waveform chunks (fewer requests, more reliable)
- 5-minute metric windows within each chunk
- 6 parallel workers with per-worker rate limiting
- Full resume support — checks existing data per station and skips
- Retry with exponential backoff on FDSN errors
"""

import os
import sys
import sqlite3
import time
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deep_backfill")

UTC = timezone.utc

BACKFILL_START = datetime(2015, 1, 1, tzinfo=UTC)
CHUNK_HOURS = 6
METRIC_WINDOW_SEC = 300  # 5-minute aggregation
MAX_WORKERS = 6
STA_SEC = 2.0
LTA_SEC = 50.0
MAX_RETRIES = 3

STATIONS = [
    {"net": "IU", "sta": "COLA", "loc": "00", "cha": "BHZ", "name": "College, Alaska"},
    {"net": "IU", "sta": "COR",  "loc": "00", "cha": "BHZ", "name": "Corvallis, Oregon"},
    {"net": "IU", "sta": "TUC",  "loc": "00", "cha": "BHZ", "name": "Tucson, Arizona"},
    {"net": "IU", "sta": "ANMO", "loc": "00", "cha": "BHZ", "name": "Albuquerque, NM"},
    {"net": "IU", "sta": "TEIG", "loc": "00", "cha": "BHZ", "name": "Teoloyucan, Mexico"},
    {"net": "IU", "sta": "SJG",  "loc": "00", "cha": "BHZ", "name": "San Juan, Puerto Rico"},
    {"net": "II", "sta": "JTS",  "loc": "00", "cha": "BHZ", "name": "Las Juntas, Costa Rica"},
    {"net": "II", "sta": "NNA",  "loc": "00", "cha": "BHZ", "name": "Nana, Peru"},
    {"net": "IU", "sta": "LCO",  "loc": "00", "cha": "BHZ", "name": "Las Campanas, Chile"},
    {"net": "II", "sta": "ESK",  "loc": "00", "cha": "BHZ", "name": "Eskdalemuir, Scotland"},
    {"net": "IU", "sta": "ANTO", "loc": "00", "cha": "BHZ", "name": "Ankara, Turkey"},
    {"net": "IU", "sta": "GNI",  "loc": "00", "cha": "BHZ", "name": "Garni, Armenia"},
    {"net": "II", "sta": "MBAR", "loc": "00", "cha": "BHZ", "name": "Mbarara, Uganda"},
    {"net": "II", "sta": "AAK",  "loc": "00", "cha": "BHZ", "name": "Ala Archa, Kyrgyzstan"},
    {"net": "II", "sta": "DGAR", "loc": "00", "cha": "BHZ", "name": "Diego Garcia"},
    {"net": "IU", "sta": "ULN",  "loc": "00", "cha": "BHZ", "name": "Ulaanbaatar, Mongolia"},
    {"net": "IU", "sta": "INCN", "loc": "00", "cha": "BHZ", "name": "Incheon, Korea"},
    {"net": "IU", "sta": "MAJO", "loc": "00", "cha": "BHZ", "name": "Matsushiro, Japan"},
    {"net": "II", "sta": "ERM",  "loc": "00", "cha": "BHZ", "name": "Erimo, Japan"},
    {"net": "IU", "sta": "DAV",  "loc": "00", "cha": "BHZ", "name": "Davao, Philippines"},
    {"net": "II", "sta": "KAPI", "loc": "00", "cha": "BHZ", "name": "Kappang, Indonesia"},
    {"net": "IU", "sta": "GUMO", "loc": "00", "cha": "BHZ", "name": "Guam"},
    {"net": "IU", "sta": "CTAO", "loc": "00", "cha": "BHZ", "name": "Charters Towers, Australia"},
    {"net": "II", "sta": "TAU",  "loc": "00", "cha": "BHZ", "name": "Hobart, Tasmania"},
]

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quakewatch.db")


def get_existing_coverage(station_key):
    """Return set of date strings (YYYY-MM-DD) that already have data for this station."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    rows = conn.execute(
        "SELECT DISTINCT DATE(timestamp) FROM station_metrics WHERE station = ?",
        (station_key,)
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def compute_metrics(trace, window_sec=METRIC_WINDOW_SEC):
    """Compute 5-minute aggregate metrics from a seismic trace."""
    from obspy import UTCDateTime

    sr = trace.stats.sampling_rate
    raw = trace.data.astype(float)
    start_time = trace.stats.starttime

    sta_n = max(1, int(STA_SEC * sr))
    lta_n = max(1, int(LTA_SEC * sr))

    window_samples = int(window_sec * sr)
    metrics = []

    for i in range(0, len(raw) - window_samples + 1, window_samples):
        segment = raw[i:i + window_samples]
        abs_seg = np.abs(segment)

        ts = start_time + (i / sr)
        ts_iso = datetime.fromtimestamp(ts.timestamp, tz=UTC).replace(
            second=0, microsecond=0
        ).isoformat()

        if len(abs_seg) >= lta_n:
            sta_arr = np.convolve(abs_seg, np.ones(sta_n) / sta_n, mode='valid')
            lta_arr = np.convolve(abs_seg, np.ones(lta_n) / lta_n, mode='valid')
            min_len = min(len(sta_arr), len(lta_arr))
            sta_arr = sta_arr[-min_len:]
            lta_arr = lta_arr[-min_len:]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(lta_arr > 0, sta_arr / lta_arr, 0)
            max_ratio = float(np.max(ratio))
        else:
            max_ratio = 0.0

        metrics.append({
            "timestamp": ts_iso,
            "amp_min": float(segment.min()),
            "amp_max": float(segment.max()),
            "amp_mean": float(np.mean(abs_seg)),
            "sta_lta_ratio": round(max_ratio, 4),
            "triggered": 1 if max_ratio > 3.5 else 0,
            "sample_rate": sr,
        })

    return metrics


def store_metrics(station_key, metrics):
    """Batch insert metrics into station_metrics table with retry on lock."""
    if not metrics:
        return 0

    rows = [(m["timestamp"], station_key, m["amp_min"], m["amp_max"],
             m["amp_mean"], m["sta_lta_ratio"], m["triggered"], m["sample_rate"])
            for m in metrics]

    for attempt in range(10):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=120)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executemany(
                "INSERT OR IGNORE INTO station_metrics "
                "(timestamp, station, amp_min, amp_max, amp_mean, sta_lta_ratio, triggered, sample_rate) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            conn.commit()
            conn.close()
            return len(metrics)
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 9:
                time.sleep(0.5 + attempt * 0.5)
                continue
            raise
    return 0


def fetch_chunk(client, stn, chunk_start, chunk_end):
    """Fetch a single waveform chunk from FDSN with retry."""
    from obspy import UTCDateTime

    t1 = UTCDateTime(chunk_start.timestamp())
    t2 = UTCDateTime(chunk_end.timestamp())

    for attempt in range(MAX_RETRIES):
        try:
            st = client.get_waveforms(
                stn["net"], stn["sta"], stn["loc"], stn["cha"],
                t1, t2, attach_response=False,
            )
            if st and len(st) > 0:
                return st.merge(fill_value=0)[0]
            return None
        except Exception as e:
            err = str(e)
            if "204" in err or "No data" in err:
                return None
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                return None
    return None


def backfill_station(stn, station_idx, total_stations):
    """Backfill a single station from BACKFILL_START to now."""
    from obspy.clients.fdsn import Client
    import warnings
    warnings.filterwarnings("ignore")

    key = f"{stn['net']}.{stn['sta']}"
    client = Client("IRIS")

    existing_days = get_existing_coverage(key)
    now = datetime.now(UTC)
    total_days = (now - BACKFILL_START).days

    chunk_delta = timedelta(hours=CHUNK_HOURS)
    current = BACKFILL_START
    chunks_done = 0
    chunks_skipped = 0
    total_metrics = 0
    total_chunks = int((now - BACKFILL_START).total_seconds() / (CHUNK_HOURS * 3600))

    log.info(f"[{station_idx}/{total_stations}] {key} ({stn['name']}) — "
             f"{total_chunks} chunks, {len(existing_days)} days already covered")

    while current < now:
        chunk_end = min(current + chunk_delta, now)
        day_str = current.strftime("%Y-%m-%d")

        if day_str in existing_days:
            same_day_chunks = 0
            while current < now and current.strftime("%Y-%m-%d") == day_str:
                current += chunk_delta
                same_day_chunks += 1
            chunks_skipped += same_day_chunks
            continue

        trace = fetch_chunk(client, stn, current, chunk_end)

        if trace is not None:
            metrics = compute_metrics(trace)
            stored = store_metrics(key, metrics)
            total_metrics += stored

        chunks_done += 1
        current = chunk_end

        if chunks_done % 100 == 0:
            pct = (chunks_done + chunks_skipped) / total_chunks * 100
            log.info(f"  {key}: {pct:.1f}% — {chunks_done} fetched, "
                     f"{chunks_skipped} skipped, {total_metrics:,} metrics stored")

        time.sleep(0.3)

    log.info(f"  {key} DONE — {total_metrics:,} metrics stored, "
             f"{chunks_done} fetched, {chunks_skipped} skipped")
    return key, total_metrics


def main():
    log.info("=" * 70)
    log.info("  DEEP SEISMIC BACKFILL — 5 YEARS")
    log.info(f"  Range: {BACKFILL_START.date()} → now")
    log.info(f"  Stations: {len(STATIONS)}")
    log.info(f"  Chunk size: {CHUNK_HOURS}h, Metric window: {METRIC_WINDOW_SEC // 60}min")
    log.info(f"  Workers: {MAX_WORKERS}")
    log.info(f"  DB: {DB_PATH}")
    log.info("=" * 70)

    start_time = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for i, stn in enumerate(STATIONS):
            futures[pool.submit(backfill_station, stn, i + 1, len(STATIONS))] = stn
            if i < MAX_WORKERS:
                time.sleep(2)

        for future in as_completed(futures):
            stn = futures[future]
            key = f"{stn['net']}.{stn['sta']}"
            try:
                station_key, count = future.result()
                results[station_key] = count
            except Exception as e:
                log.error(f"  {key} FAILED: {e}")
                results[key] = -1

    elapsed = time.time() - start_time
    total = sum(v for v in results.values() if v > 0)
    failed = sum(1 for v in results.values() if v < 0)

    log.info("=" * 70)
    log.info("  BACKFILL COMPLETE")
    log.info(f"  Total metrics stored: {total:,}")
    log.info(f"  Stations completed: {len(results) - failed}/{len(STATIONS)}")
    if failed:
        log.info(f"  Stations failed: {failed}")
    log.info(f"  Elapsed: {elapsed / 3600:.1f} hours")
    log.info("=" * 70)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.execute(
        "SELECT station, COUNT(*), MIN(timestamp), MAX(timestamp) "
        "FROM station_metrics GROUP BY station ORDER BY station"
    )
    log.info("\n  Final station coverage:")
    for r in cur.fetchall():
        log.info(f"    {r[0]:<12} {r[1]:>8,} rows  {r[2][:10]} → {r[3][:10]}")
    conn.close()


if __name__ == "__main__":
    main()
