"""Live EMSC small-event poller — keeps the offshore foreshock catalog current.

USGS misses M2.5-4 events offshore; the tier-2 swarm gate needs them. This pulls
the last few days of EMSC per zone on a short interval and dedups into the catalog,
so the realtime engine's swarm detection works in Indonesia / Chile / Japan / etc.

  python ingest_emsc_live.py --loop 300   # every 5 min
  python ingest_emsc_live.py              # once
"""
import os, sys, time, argparse
import pandas as pd
from urllib.parse import urlencode

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> import ingest package
from ingest.backfill import (_parse_emsc_text, _load_existing_events, _is_duplicate,
                             EMSC_ZONES, EMSC_EVENT_URL)
from ingest.sources import _fetch_text
from ingest.store import QuakeStore


def fetch_recent(days=7):
    store = QuakeStore()
    end = pd.Timestamp.utcnow(); start = end - pd.Timedelta(days=days)
    total_new = 0
    for z in EMSC_ZONES:
        params = {"format": "text",
                  "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
                  "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
                  "minmag": 2.5, "minlat": z["lat"][0], "maxlat": z["lat"][1],
                  "minlon": z["lon"][0], "maxlon": z["lon"][1], "limit": 20000}
        try:
            txt = _fetch_text(f"{EMSC_EVENT_URL}?{urlencode(params)}", timeout=60)
        except Exception as e:
            print(f"  [warn] {z['id']}: {e}"); continue
        events = _parse_emsc_text(txt)
        if not events:
            continue
        existing = _load_existing_events(store, start, end, z["lat"], z["lon"])
        new = [e for e in events if not _is_duplicate(e, existing, 50, 90)]
        total_new += store.bulk_insert_earthquakes(new)
    print(f"  [{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}] EMSC live: +{total_new} new small events")
    return total_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0)
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    while True:
        try:
            fetch_recent(args.days)
        except Exception as e:
            print(f"  [ERROR] {e}")
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
