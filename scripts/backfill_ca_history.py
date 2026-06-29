#!/usr/bin/env python3
"""One-off: backfill the pre-2015 California catalog from USGS ComCat.

Our DB starts at 2015; the CA network has been dense and complete to ~M2.5 since the
mid-80s, so there's a deep, clean catalog we can use to give the tier-2 model many more
escalation examples in California (Loma Prieta '89, Landers '92, Northridge '94, Hector
Mine '99, El Mayor '10, ...). M2.5+ (need small events for the swarm gate). Monthly
chunks to stay well under the 20k/query cap. INSERT OR IGNORE on id (dedup-safe)."""
import json, sqlite3, time, os
import urllib.request, urllib.parse
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quakewatch.db")
BBOX = dict(minlatitude=32, maxlatitude=42, minlongitude=-125, maxlongitude=-114)
START_YEAR, END_YEAR = 1990, 2015   # 1985-89 already landed; resume 1990+
MIN_MAG = 2.5
URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch(t0, t1):
    p = dict(format="geojson", starttime=t0, endtime=t1, minmagnitude=MIN_MAG,
             orderby="time-asc", **BBOX)
    req = urllib.request.Request(URL + "?" + urllib.parse.urlencode(p),
                                 headers={"User-Agent": "seismic-lab/ca-backfill"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["features"]


def main():
    conn = sqlite3.connect(DB, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000;")
    months = [(y, m) for y in range(START_YEAR, END_YEAR) for m in range(1, 13)]
    total_seen = total_new = total_m5 = 0
    for y, m in months:
        t0 = f"{y}-{m:02d}-01T00:00:00"
        t1 = f"{y+(m==12)}-{(m % 12)+1:02d}-01T00:00:00"
        for attempt in range(3):
            try:
                feats = fetch(t0, t1)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  {y}-{m:02d}: FAILED {e}"); feats = []
                time.sleep(2)
        rows = []
        for f in feats:
            pr = f["properties"]; c = f.get("geometry", {}).get("coordinates") or [None, None, None]
            if pr.get("mag") is None or c[0] is None:
                continue
            ts = datetime.fromtimestamp(pr["time"] / 1000, tz=timezone.utc).isoformat()
            rows.append((f["id"], ts, pr["mag"], c[2], c[1], c[0],
                         pr.get("place", ""), pr.get("type", "earthquake")))
        cur = conn.executemany(
            "INSERT OR IGNORE INTO earthquakes (id,timestamp,magnitude,depth_km,lat,lon,place,type) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        total_seen += len(rows); total_new += cur.rowcount
        total_m5 += sum(1 for r in rows if r[2] and r[2] >= 5)
        if m == 12:
            print(f"  {y}: cumulative {total_new:,} new rows, {total_m5} M5+ seen")
        time.sleep(0.15)
    print(f"\nDONE: {total_seen:,} events fetched, {total_new:,} new inserted, {total_m5} M5+ in range")
    span = conn.execute("SELECT MIN(substr(timestamp,1,10)), MAX(substr(timestamp,1,10)), COUNT(*), "
                        "SUM(magnitude>=5) FROM earthquakes WHERE lat BETWEEN 32 AND 42 "
                        "AND lon BETWEEN -125 AND -114").fetchone()
    print(f"CA region now: {span[0]} -> {span[1]} | {span[2]:,} events | {span[3]} M5+")


if __name__ == "__main__":
    main()
