"""Backfill open IRIS stations for the global-expansion zones — 2021+.

Dense Mediterranean (Italy IV) and New Zealand (GeoNet) networks aren't on IRIS —
they'd need ORFEUS/GEONET FDSN clients (follow-up). What IRIS serves: Caribbean
(PR/DR networks, real coverage) plus single GSN stations for Med/NZ/PNG/Kamchatka.
Catalog foreshocks for all these zones come from EMSC separately.
"""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import deep_backfill as db

db.BACKFILL_START = datetime(2021, 7, 1, tzinfo=timezone.utc)

NEW_STATIONS = [
    {"net": "MN", "sta": "AQU",  "loc": "",   "cha": "BHZ", "lat": 42.35, "lon": 13.40,  "name": "L'Aquila, Italy"},
    {"net": "IU", "sta": "SNZO", "loc": "00", "cha": "BHZ", "lat": -41.31, "lon": 174.70, "name": "Wellington, NZ"},
    {"net": "PR", "sta": "PCDR", "loc": "",   "cha": "BHZ", "lat": 18.51, "lon": -68.38, "name": "Punta Cana, DR"},
    {"net": "DR", "sta": "SDD",  "loc": "",   "cha": "BHZ", "lat": 18.46, "lon": -69.92, "name": "Santo Domingo, DR"},
    {"net": "PR", "sta": "SMDR", "loc": "",   "cha": "BHZ", "lat": 19.29, "lon": -69.19, "name": "Samana, DR"},
    {"net": "AU", "sta": "RABL", "loc": "00", "cha": "BHZ", "lat": -4.19, "lon": 152.16, "name": "Rabaul, PNG"},
    {"net": "IU", "sta": "PET",  "loc": "00", "cha": "BHZ", "lat": 53.02, "lon": 158.65, "name": "Petropavlovsk, Kamchatka"},
]


def main():
    db.log.info("=" * 70)
    db.log.info("  GLOBAL-EXPANSION STATION BACKFILL — 7 stations, 2021+")
    db.log.info("=" * 70)
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(db.backfill_station, s, i + 1, len(NEW_STATIONS)): s
                   for i, s in enumerate(NEW_STATIONS)}
        for f in as_completed(futures):
            s = futures[f]; key = f"{s['net']}.{s['sta']}"
            try:
                k, n = f.result(); results[k] = n
            except Exception as e:
                db.log.error(f"  {key} FAILED: {e}"); results[key] = -1
    total = sum(v for v in results.values() if v > 0)
    db.log.info(f"  DONE: {total:,} metrics, {sum(1 for v in results.values() if v>0)}/{len(NEW_STATIONS)} OK")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    main()
