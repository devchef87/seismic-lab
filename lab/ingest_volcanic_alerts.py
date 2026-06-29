"""Live volcanic alert ingest — USGS HANS real-time volcano alert levels.

Pulls current elevated volcanoes (NORMAL/ADVISORY/WATCH/WARNING + aviation color)
into a `volcano_alerts` snapshot table. The realtime engine reads this to flag
swarms near a restless volcano as CONTEXT (an operational overlay, not a model
input — there's no historical alert archive to train on; that would need TROPOMI
SO2 backfill). US-only coverage today (AVO/CVO/HVO); global needs more sources.

  python ingest_volcanic_alerts.py --loop 1800   # every 30 min
"""
import os, sys, json, time, argparse
from urllib.request import urlopen, Request
import sqlite3
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(_HERE), "data", "quakewatch.db")
HANS = "https://volcanoes.usgs.gov/hans-public/api/volcano/getCapElevated"


def ensure_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS volcano_alerts (
        vnum TEXT PRIMARY KEY, volcano_name TEXT, lat REAL, lon REAL,
        alert_level TEXT, color_code TEXT, observatory TEXT, updated TEXT)""")


def fetch():
    req = Request(HANS, headers={"User-Agent": "SeismicLab/1.0", "Accept": "application/json"})
    data = json.loads(urlopen(req, timeout=30).read())
    now = pd.Timestamp.utcnow().isoformat()
    conn = sqlite3.connect(DB, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_table(conn)
    conn.execute("DELETE FROM volcano_alerts")  # current snapshot
    rows = 0
    for v in data:
        try:
            conn.execute("INSERT OR REPLACE INTO volcano_alerts VALUES (?,?,?,?,?,?,?,?)",
                         (str(v.get("vnum")), v.get("volcano_name_appended", ""),
                          v.get("latitude"), v.get("longitude"),
                          v.get("alert_level"), v.get("color_code"),
                          v.get("obs_abbr", ""), now))
            rows += 1
        except Exception:
            pass
    conn.commit(); conn.close()
    levels = {}
    for v in data:
        levels[v.get("alert_level")] = levels.get(v.get("alert_level"), 0) + 1
    print(f"  [{now[:16]}] volcanic alerts: {rows} elevated volcanoes {levels}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0)
    args = ap.parse_args()
    while True:
        try:
            fetch()
        except Exception as e:
            print(f"  [ERROR] {e}")
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
