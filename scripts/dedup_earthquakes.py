#!/usr/bin/env python3
"""De-duplicate the earthquakes catalog.

Two reporting agencies feed the table (USGS + EMSC), plus a legacy EMSC ingest
path. The same physical quake therefore lands multiple times under different id
schemes, with slightly different time / location / magnitude. INSERT OR IGNORE
only dedups on the `id` primary key, so these cross-source twins slip through and
the model double-counts events (inflating seismicity rates and the swarm gate).

This collapses spatiotemporal twins to ONE canonical row per physical event.

Matching: two events are the same physical quake if they come from DIFFERENT
ingest-source groups (USGS / EMSC-current / EMSC-legacy) AND are within:
    |Δt| <= 30 s,  distance <= 60 km,  |Δmag| <= 1.5
Same-source pairs are never merged (a single agency doesn't re-report an event,
except the legacy path, which is a different group and so IS merged).

Canonical preference (lowest rank kept): USGS authoritative (us*) < USGS regional
< EMSC current (emsc:) < EMSC legacy (emsc_). USGS wins overlaps because the
M5+ targets we predict want authoritative magnitudes; EMSC-only small events
(no twin) are always kept — that coverage is the swarm signal.

Reversible: losers are copied to `earthquakes_dupes_removed` before deletion.

Usage:
    python scripts/dedup_earthquakes.py            # dry run (report + samples)
    python scripts/dedup_earthquakes.py --apply     # perform the migration
"""
import argparse
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quakewatch.db")

DT_MAX_S = 30.0
DIST_MAX_KM = 50.0
DMAG_MAX = 1.0


def src_group(i):
    # one group per reporting contributor. Two events from DIFFERENT contributors that
    # are spatiotemporally coincident are the same physical quake (USGS global `us`,
    # the tsunami centers `at`/`pt`, regional nets `ak`/`hv`/`pr`/..., and EMSC all
    # independently solve the same event). Same-contributor pairs are distinct events.
    if i.startswith("emsc:"):
        return "emsc"        # current EMSC ingest
    if i.startswith("emsc_"):
        return "emscL"       # legacy EMSC ingest
    return i[:2]             # USGS-family network code: us, ak, at, ci, nc, hv, pr, tx, ...


def keep_rank(i):
    if i.startswith("emsc_"):
        return 3
    if i.startswith("emsc:"):
        return 2
    if i.startswith("us"):
        return 0             # USGS authoritative (global Mww)
    return 1                 # USGS regional network


def to_epoch(ts):
    # timestamps are ISO text, e.g. "2026-06-26T17:43:02" or "...+00:00" (± fractional)
    try:
        d = datetime.fromisoformat(ts)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except Exception:
        return None


def haversine_km(la1, lo1, la2, lo2):
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1)
    dl = math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform the migration (default: dry run)")
    args = ap.parse_args()

    conn = sqlite3.connect(DB, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000;")
    rows = conn.execute(
        "SELECT id, timestamp, magnitude, lat, lon, place FROM earthquakes"
    ).fetchall()
    print(f"loaded {len(rows):,} events")

    # build (epoch, ...) list, drop unparseable
    evs = []
    for r in rows:
        ep = to_epoch(r[1])
        if ep is None or r[3] is None or r[4] is None:
            continue
        evs.append((ep, r[0], float(r[2]) if r[2] is not None else 0.0, float(r[3]), float(r[4]), r[5]))
    evs.sort(key=lambda e: e[0])
    print(f"parsed {len(evs):,} events with valid time+location")

    uf = UF()
    pairs = []           # sample matched pairs for eyeballing
    n_pairs = 0
    j = 0
    for i in range(len(evs)):
        ei = evs[i]
        # window start: first event within DT_MAX_S before ei
        while evs[i][0] - evs[j][0] > DT_MAX_S:
            j += 1
        for k in range(j, i):
            ek = evs[k]
            if src_group(ei[1]) == src_group(ek[1]):
                continue
            if abs(ei[2] - ek[2]) > DMAG_MAX:
                continue
            if haversine_km(ei[3], ei[4], ek[3], ek[4]) > DIST_MAX_KM:
                continue
            uf.union(ei[1], ek[1])
            n_pairs += 1
            if len(pairs) < 15:
                pairs.append((ek, ei))

    # group by canonical root, choose keeper
    groups = {}
    meta = {e[1]: e for e in evs}
    for e in evs:
        root = uf.find(e[1])
        groups.setdefault(root, []).append(e[1])

    losers = []
    multi = 0
    for root, ids in groups.items():
        if len(ids) < 2:
            continue
        multi += 1
        ids_sorted = sorted(ids, key=lambda x: keep_rank(x))
        keep = ids_sorted[0]
        for lid in ids_sorted[1:]:
            losers.append((lid, keep))

    print(f"\nmatched pairs: {n_pairs:,}")
    print(f"merged physical events (groups with a twin): {multi:,}")
    print(f"duplicate rows to remove: {len(losers):,}")
    print(f"catalog after dedup: {len(rows) - len(losers):,}")

    print("\n--- sample matched pairs (kept <- removed) ---")
    for a, b in pairs:
        ka, kb = (a, b) if keep_rank(a[1]) <= keep_rank(b[1]) else (b, a)
        dt = abs(a[0] - b[0])
        dd = haversine_km(a[3], a[4], b[3], b[4])
        print(f"  KEEP {ka[1]:<24} M{ka[2]:.1f} ({ka[3]:.2f},{ka[4]:.2f}) {ka[5]}")
        print(f"  drop {kb[1]:<24} M{kb[2]:.1f} ({kb[3]:.2f},{kb[4]:.2f})  Δt={dt:.0f}s Δd={dd:.0f}km")
        print()

    if not args.apply:
        print("DRY RUN — nothing changed. Re-run with --apply to migrate.")
        return

    print("APPLYING migration...")
    loser_ids = [l[0] for l in losers]
    keep_for = {l[0]: l[1] for l in losers}
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS earthquakes_dupes_removed (
            id TEXT, timestamp TEXT, magnitude REAL, depth_km REAL,
            lat REAL, lon REAL, place TEXT, type TEXT,
            kept_id TEXT, removed_at TEXT
        )
    """)
    now = datetime.now(timezone.utc).isoformat()
    moved = 0
    for i in range(0, len(loser_ids), 500):
        chunk = loser_ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        src = cur.execute(
            f"SELECT id, timestamp, magnitude, depth_km, lat, lon, place, type "
            f"FROM earthquakes WHERE id IN ({ph})", chunk).fetchall()
        cur.executemany(
            "INSERT INTO earthquakes_dupes_removed "
            "(id,timestamp,magnitude,depth_km,lat,lon,place,type,kept_id,removed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(*s, keep_for.get(s[0], ""), now) for s in src])
        cur.execute(f"DELETE FROM earthquakes WHERE id IN ({ph})", chunk)
        moved += len(src)
    conn.commit()
    print(f"moved+deleted {moved:,} duplicate rows -> earthquakes_dupes_removed")
    print(f"earthquakes now: {conn.execute('SELECT COUNT(*) FROM earthquakes').fetchone()[0]:,}")


if __name__ == "__main__":
    main()
