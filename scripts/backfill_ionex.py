#!/usr/bin/env python3
"""Backfill historical ionospheric TEC (IGS Global Ionosphere Maps, IONEX) from NASA
CDDIS, so the tier-2 model can train on TEC over its 2021-07+ window (the live GloTEC
feed only archives ~1 month). Auth via Earthdata bearer token in ~/.earthdata_token.

IGS GIM: 2.5deg lat x 5deg lon, 2-hourly (13 maps/day). We downsample to a ~20deg grid
and store raw TEC into `samples` (source=igs_ionex, metric=tec) — idempotent.
Decompresses both .gz (new IGS long names) and .Z (pre-2022 short names) via `gzip -dc`.
"""
import os, sys, re, subprocess, sqlite3, urllib.request
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quakewatch.db")
TOKEN = open(os.path.expanduser("~/.earthdata_token")).read().strip()
BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"
START = datetime(2021, 7, 1, tzinfo=UTC)
END = datetime.now(UTC)


def candidates(d):
    yyyy, doy, yy = d.year, d.timetuple().tm_yday, d.year % 100
    return [  # try final, then rapid, then legacy short names
        f"IGS0OPSFIN_{yyyy}{doy:03d}0000_01D_02H_GIM.INX.gz",
        f"IGS0OPSRAP_{yyyy}{doy:03d}0000_01D_02H_GIM.INX.gz",
        f"igsg{doy:03d}0.{yy:02d}i.Z",
        f"igrg{doy:03d}0.{yy:02d}i.Z",
    ]


def download(d):
    yyyy, doy = d.year, d.timetuple().tm_yday
    for name in candidates(d):
        url = f"{BASE}/{yyyy}/{doy:03d}/{name}"
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + TOKEN,
                                                       "User-Agent": "seismic-lab"})
            raw = urllib.request.urlopen(req, timeout=90).read()
            if len(raw) < 500:
                continue
            txt = subprocess.run(["gzip", "-dc"], input=raw, capture_output=True).stdout.decode("latin-1")
            if "TEC MAP" in txt:
                return txt, name
        except Exception:
            continue
    return None, None


def parse_ionex(txt):
    lines = txt.splitlines()
    exp, lons = -1, None
    for ln in lines:
        if "EXPONENT" in ln:
            try: exp = int(ln.split()[0])
            except Exception: pass
        elif "LON1 / LON2 / DLON" in ln:
            a, b, dd = map(float, ln.split()[:3])
            lons = [round(a + dd * k, 2) for k in range(int(round((b - a) / dd)) + 1)]
        if "END OF HEADER" in ln:
            break
    if not lons:
        return []
    scale = 10.0 ** exp
    maps, epoch, grid, k = [], None, {}, 0
    while k < len(lines):
        ln = lines[k]
        if "EPOCH OF CURRENT MAP" in ln:
            p = ln.split()
            epoch = datetime(*(int(float(x)) for x in p[:6]), tzinfo=UTC)
            grid = {}
        elif "LAT/LON1/LON2/DLON/H" in ln:
            m = re.match(r"\s*(-?\d+\.?\d*)", ln)
            lat = float(m.group(1)) if m else None
            vals, k = [], k + 1
            while len(vals) < len(lons) and k < len(lines):
                row = lines[k]
                if any(t in row for t in ("LAT/LON1", "END OF", "START OF", "EPOCH OF")):
                    break
                vals += [int(x) for x in row.split()]
                k += 1
            k -= 1  # step back so the outer k+=1 re-lands on the marker line we stopped at
            if lat is not None:
                for lon, v in zip(lons, vals):
                    if v != 9999:
                        grid[(lat, lon)] = round(v * scale, 2)
        elif "END OF TEC MAP" in ln and epoch is not None:
            maps.append((epoch, grid)); epoch = None
        k += 1
    return maps


def keep(lat, lon):  # ~20deg grid: every 8th lat (2.5->20), every 4th lon (5->20)
    return int(round((lat + 87.5) / 2.5)) % 8 == 0 and int(round((lon + 180) / 5)) % 4 == 0


def main():
    one_day = "--one" in sys.argv
    conn = sqlite3.connect(DB, timeout=120); conn.execute("PRAGMA busy_timeout=120000")
    d, total, days_ok, days_miss = START, 0, 0, 0
    while d <= END:
        txt, name = download(d)
        if txt:
            rows = []
            for epoch, grid in parse_ionex(txt):
                ts = epoch.isoformat()
                for (lat, lon), tec in grid.items():
                    if keep(lat, lon):
                        rows.append(("igs_ionex", "tec", ts, tec, "TECU", lat, lon, None, ts))
            conn.executemany(
                "INSERT OR IGNORE INTO samples (source,metric,timestamp,value,unit,lat,lon,meta,ingested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)", rows)
            conn.commit(); total += len(rows); days_ok += 1
        else:
            days_miss += 1
        if d.timetuple().tm_yday % 30 == 0 or one_day:
            print(f"  {d.date()} | {days_ok} days ok, {days_miss} missing | {total:,} TEC rows", flush=True)
        if one_day:
            print("ONE-DAY TEST done. file:", name, "rows:", total); return
        d += timedelta(days=1)
    print(f"\nDONE: {days_ok} days, {days_miss} missing, {total:,} TEC rows inserted")


if __name__ == "__main__":
    main()
