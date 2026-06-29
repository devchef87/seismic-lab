"""Backfill dense regional stations near Chile and Alaska — 2021+ only.

Size prediction (M6+) correlates with waveform amplitude/coherence (stn_* features),
but our GSN stations average 1,130km from events — too far to resolve rupture size.
These regional stations sit 33-128km from the action. Chile is the priority: worst
size AUC (0.41) and densest available coverage (C/C1 networks).

Runs separately from the main deep_backfill (which is still filling 2015-2021 for the
original 24 GSN stations). 2021+ only — matches the data-complete training window and
halves the work. Idempotent (skips already-covered days).
"""

import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import deep_backfill as db

UTC = timezone.utc

# Restrict to the data-complete era; be polite while the main backfill runs
db.BACKFILL_START = datetime(2021, 7, 1, tzinfo=UTC)

# Curated closest broadband stations (IRIS), net/sta/loc/cha + coords for cell assignment
NEW_STATIONS = [
    # --- Chile (C/C1/GE/G networks) — the priority ---
    {"net": "C1", "sta": "VA01", "loc": "", "cha": "BHZ", "lat": -33.0, "lon": -71.6, "name": "Valparaiso VA01, Chile"},
    {"net": "C1", "sta": "VA05", "loc": "", "cha": "BHZ", "lat": -33.7, "lon": -71.6, "name": "Valparaiso VA05, Chile"},
    {"net": "C1", "sta": "VA06", "loc": "", "cha": "BHZ", "lat": -32.6, "lon": -71.3, "name": "Valparaiso VA06, Chile"},
    {"net": "C1", "sta": "MT02", "loc": "", "cha": "BHZ", "lat": -33.3, "lon": -71.1, "name": "Maule MT02, Chile"},
    {"net": "C1", "sta": "MT07", "loc": "", "cha": "BHZ", "lat": -33.0, "lon": -71.0, "name": "Maule MT07, Chile"},
    {"net": "C",  "sta": "ROC1", "loc": "", "cha": "BHZ", "lat": -33.0, "lon": -71.0, "name": "ROC1, Chile"},
    {"net": "C1", "sta": "MT01", "loc": "", "cha": "BHZ", "lat": -33.9, "lon": -71.3, "name": "Maule MT01, Chile"},
    {"net": "C1", "sta": "MT19", "loc": "", "cha": "BHZ", "lat": -33.4, "lon": -70.9, "name": "Maule MT19, Chile"},
    {"net": "C1", "sta": "MT05", "loc": "", "cha": "BHZ", "lat": -33.4, "lon": -70.7, "name": "Maule MT05, Chile"},
    # --- Alaska (AK network) — incremental, already our best zone ---
    {"net": "AK", "sta": "SWD",  "loc": "", "cha": "BHZ", "lat": 60.1, "lon": -149.5, "name": "Seward, Alaska"},
    {"net": "AK", "sta": "BRSE", "loc": "", "cha": "BHZ", "lat": 59.7, "lon": -150.7, "name": "Bear Cove, Alaska"},
    {"net": "AK", "sta": "BRLK", "loc": "", "cha": "BHZ", "lat": 59.8, "lon": -150.9, "name": "Bradley Lake, Alaska"},
    {"net": "AK", "sta": "SLK",  "loc": "", "cha": "BHZ", "lat": 60.5, "lon": -150.2, "name": "Skilak, Alaska"},
    {"net": "AK", "sta": "CNP",  "loc": "", "cha": "BHZ", "lat": 59.5, "lon": -151.2, "name": "Coal Point, Alaska"},
    {"net": "AK", "sta": "HOM",  "loc": "", "cha": "BHZ", "lat": 59.7, "lon": -151.7, "name": "Homer, Alaska"},
    {"net": "AK", "sta": "CAPN", "loc": "", "cha": "BHZ", "lat": 60.8, "lon": -151.2, "name": "Captain Cook, Alaska"},
]


def main():
    db.log.info("=" * 70)
    db.log.info("  REGIONAL STATION BACKFILL — Chile + Alaska, 2021+")
    db.log.info(f"  Range: {db.BACKFILL_START.date()} -> now")
    db.log.info(f"  New stations: {len(NEW_STATIONS)}")
    db.log.info("=" * 70)

    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:  # polite: main backfill still running
        futures = {}
        for i, stn in enumerate(NEW_STATIONS):
            futures[pool.submit(db.backfill_station, stn, i + 1, len(NEW_STATIONS))] = stn
            if i < 3:
                import time; time.sleep(2)
        for future in as_completed(futures):
            stn = futures[future]
            key = f"{stn['net']}.{stn['sta']}"
            try:
                k, count = future.result()
                results[k] = count
            except Exception as e:
                db.log.error(f"  {key} FAILED: {e}")
                results[key] = -1

    total = sum(v for v in results.values() if v > 0)
    failed = sum(1 for v in results.values() if v < 0)
    db.log.info("=" * 70)
    db.log.info(f"  REGIONAL BACKFILL COMPLETE: {total:,} metrics, "
                f"{len(results) - failed}/{len(NEW_STATIONS)} stations OK")
    db.log.info("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    main()
