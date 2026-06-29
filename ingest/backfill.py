"""QuakeWatch — Historical data backfill engine.

Pulls years of historical data from archival endpoints, chunked to respect
rate limits and API constraints. Designed to build the training dataset
for earthquake prediction ML.
"""

import json
import time
import logging
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path

from .sources import Sample, _fetch_json, _fetch_text, NASA_API_KEY
from .store import QuakeStore

log = logging.getLogger("quakewatch.backfill")
UTC = timezone.utc


# ---------------------------------------------------------------------------
# USGS Earthquake Catalog — goes back to 1900+
# Chunk by month, M2.5+ globally
# ---------------------------------------------------------------------------

def backfill_earthquakes(store: QuakeStore, start_year: int = 2015,
                         end_year: int = 2026, min_mag: float = 2.5):
    start = datetime(start_year, 1, 1, tzinfo=UTC)
    end = datetime(end_year, 7, 1, tzinfo=UTC)
    now = datetime.now(UTC)
    if end > now:
        end = now

    current = start
    total_samples = 0
    total_new = 0

    while current < end:
        chunk_end = current + timedelta(days=30)
        if chunk_end > end:
            chunk_end = end

        try:
            params = {
                "format": "geojson",
                "starttime": current.strftime("%Y-%m-%d"),
                "endtime": chunk_end.strftime("%Y-%m-%d"),
                "minmagnitude": min_mag,
                "limit": 20000,
                "orderby": "time",
            }
            url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?{urlencode(params)}"
            data = _fetch_json(url, timeout=60)

            samples = []
            for feat in data.get("features", []):
                props = feat["properties"]
                coords = feat["geometry"]["coordinates"]
                ts = datetime.fromtimestamp(props["time"] / 1000, tz=UTC).isoformat()

                samples.append(Sample(
                    source="usgs_earthquake", metric="magnitude",
                    timestamp=ts, value=props["mag"], unit="Mw",
                    lat=coords[1], lon=coords[0],
                    meta={"depth_km": coords[2], "place": props.get("place", ""),
                          "type": props.get("type", ""), "id": feat["id"]},
                ))
                if coords[2] is not None:
                    samples.append(Sample(
                        source="usgs_earthquake", metric="depth",
                        timestamp=ts, value=coords[2], unit="km",
                        lat=coords[1], lon=coords[0],
                        meta={"magnitude": props["mag"], "id": feat["id"]},
                    ))

            new = store.ingest(samples, source_name="usgs_earthquake_backfill")
            total_samples += len(samples)
            total_new += new
            log.info(f"EQ backfill {current.strftime('%Y-%m')}: {len(samples)} samples ({new} new)")

        except Exception as e:
            log.error(f"EQ backfill {current.strftime('%Y-%m')} failed: {e}")
            time.sleep(2)

        current = chunk_end
        time.sleep(0.5)

    log.info(f"EQ backfill complete: {total_samples:,} samples, {total_new:,} new")
    return {"source": "usgs_earthquake", "total": total_samples, "new": total_new}


# ---------------------------------------------------------------------------
# EMSC Earthquake Catalog — regional foreshock-band completeness
# USGS ComCat completeness is ~M4.5 offshore (Indonesia/Chile/Japan trenches).
# EMSC (seismicportal.eu) aggregates regional networks (BMKG, CSN, JMA, ...)
# with Mc ~2.5-3.0, giving 30-350x more small events — the foreshock signal.
# We pull per-zone bounding boxes, monthly, and dedup shared mainshocks against
# the existing USGS catalog by spatiotemporal proximity.
# ---------------------------------------------------------------------------

import math

EMSC_EVENT_URL = "https://www.seismicportal.eu/fdsnws/event/1/query"

# Target zone bounding boxes (match PARENT_ZONES in lab/train_ensemble.py)
EMSC_ZONES = [
    {"id": "indonesia",     "lat": (-12, 8),  "lon": (95, 140)},
    {"id": "japan_kurils",  "lat": (25, 50),  "lon": (128, 155)},
    {"id": "south_america", "lat": (-60, 7),  "lon": (-82, -60)},
    {"id": "mexico_ca",     "lat": (7, 25),   "lon": (-115, -77)},
    {"id": "himalaya",      "lat": (25, 42),  "lon": (65, 100)},
    {"id": "alaska",        "lat": (50, 72),  "lon": (-180, -130)},
    {"id": "philippines",   "lat": (8, 25),   "lon": (117, 127)},
    {"id": "mediterranean", "lat": (34, 43),  "lon": (19, 45)},
    {"id": "caribbean",     "lat": (10, 20),  "lon": (-77, -60)},
    {"id": "new_zealand",   "lat": (-48, -34), "lon": (165, 180)},
    {"id": "png_solomon",   "lat": (-12, -3), "lon": (140, 160)},
    {"id": "kamchatka",     "lat": (50, 62),  "lon": (156, 168)},
]


def _eq_haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _iso_to_epoch(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _parse_emsc_text(text):
    """Parse EMSC FDSN text (pipe-delimited):
    EventID|Time|Lat|Lon|Depth|Author|Catalog|Contributor|ContributorID|MagType|Mag|MagAuthor|LocationName
    """
    events = []
    for line in text.strip().split("\n"):
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 13:
            continue
        try:
            eid = parts[0].strip()
            dt = datetime.fromisoformat(parts[1].strip().replace("Z", "+00:00"))
            ts = dt.astimezone(UTC).isoformat()
            lat = float(parts[2]); lon = float(parts[3])
            depth = float(parts[4]) if parts[4].strip() else None
            mag = float(parts[10])
            place = parts[12].strip()
        except (ValueError, IndexError):
            continue
        events.append({
            "id": f"emsc:{eid}", "timestamp": ts, "magnitude": mag,
            "depth_km": depth, "lat": lat, "lon": lon,
            "place": place, "type": "earthquake", "_epoch": dt.timestamp(),
        })
    return events


def _load_existing_events(store, t0, t1, zlat, zlon):
    """Existing catalog events in this bbox+window, as (epoch, lat, lon) for dedup."""
    with store._conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, lat, lon FROM earthquakes
            WHERE timestamp >= ? AND timestamp < ?
            AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
        """, (t0.isoformat(), t1.isoformat(),
              zlat[0], zlat[1], zlon[0], zlon[1])).fetchall()
    out = []
    for ts, lat, lon in rows:
        ep = _iso_to_epoch(ts)
        if ep is not None:
            out.append((ep, lat, lon))
    return out


def _is_duplicate(ev, existing, radius_km, time_s):
    """Spatiotemporal dedup. Only M>=3.5 EMSC events can have a USGS twin
    (USGS offshore Mc ~4.5); smaller events are EMSC-only, so skip the check."""
    if ev["magnitude"] < 3.5:
        return False
    ep = ev["_epoch"]
    for (eep, elat, elon) in existing:
        if abs(eep - ep) <= time_s and \
                _eq_haversine_km(ev["lat"], ev["lon"], elat, elon) <= radius_km:
            return True
    return False


def backfill_earthquakes_emsc(store: QuakeStore, start_year: int = 2015,
                              end_year: int = 2026, min_mag: float = 2.5,
                              zones=None, dedup_radius_km: float = 50.0,
                              dedup_time_s: float = 90.0):
    """Backfill regional foreshock-band events from EMSC, deduped against USGS."""
    if zones is None:
        zones = EMSC_ZONES

    total_fetched = total_new = total_dup = 0

    for zone in zones:
        zlat, zlon = zone["lat"], zone["lon"]
        start = datetime(start_year, 1, 1, tzinfo=UTC)
        end = datetime(end_year, 7, 1, tzinfo=UTC)
        now = datetime.now(UTC)
        if end > now:
            end = now

        log.info(f"EMSC zone {zone['id']} lat={zlat} lon={zlon}")
        current = start
        zone_new = zone_fetched = 0

        while current < end:
            chunk_end = current + timedelta(days=30)
            if chunk_end > end:
                chunk_end = end

            params = {
                "format": "text",
                "starttime": current.strftime("%Y-%m-%dT%H:%M:%S"),
                "endtime": chunk_end.strftime("%Y-%m-%dT%H:%M:%S"),
                "minmag": min_mag,
                "minlat": zlat[0], "maxlat": zlat[1],
                "minlon": zlon[0], "maxlon": zlon[1],
                "limit": 20000,
            }
            url = f"{EMSC_EVENT_URL}?{urlencode(params)}"
            try:
                text = _fetch_text(url, timeout=90)
            except Exception as e:
                log.error(f"  EMSC {zone['id']} {current:%Y-%m} fetch failed: {e}")
                current = chunk_end
                time.sleep(1.0)
                continue

            events = _parse_emsc_text(text)
            zone_fetched += len(events)
            total_fetched += len(events)

            if events:
                existing = _load_existing_events(store, current, chunk_end, zlat, zlon)
                new_events = []
                for ev in events:
                    if _is_duplicate(ev, existing, dedup_radius_km, dedup_time_s):
                        total_dup += 1
                    else:
                        new_events.append(ev)
                inserted = store.bulk_insert_earthquakes(new_events)
                zone_new += inserted
                total_new += inserted
                log.info(f"  {zone['id']} {current:%Y-%m}: {len(events)} fetched, "
                         f"{inserted} new")

            current = chunk_end
            time.sleep(0.3)

        log.info(f"  zone {zone['id']} done: {zone_fetched:,} fetched, {zone_new:,} new")

    log.info(f"EMSC backfill complete: {total_fetched:,} fetched, "
             f"{total_new:,} new, {total_dup:,} dups skipped")
    return {"source": "emsc_earthquake", "total": total_fetched,
            "new": total_new, "dups": total_dup}


# ---------------------------------------------------------------------------
# GFZ Potsdam Kp Index — definitive Kp since 1932
# Single text file download, parse fixed-width format
# ---------------------------------------------------------------------------

def backfill_kp_index(store: QuakeStore, start_year: int = 2015):
    url = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
    log.info("Downloading GFZ Potsdam Kp archive...")
    text = _fetch_text(url, timeout=120)

    samples = []
    cutoff = datetime(start_year, 1, 1, tzinfo=UTC)

    for line in text.strip().split("\n"):
        if line.startswith("#") or len(line.strip()) < 20:
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            dt = datetime(year, month, day, tzinfo=UTC)
            if dt < cutoff:
                continue

            # 8 three-hour Kp values per day (columns vary by format)
            # Format: YYYY MM DD days days_m Bsr dB Kp1 Kp2 Kp3 Kp4 Kp5 Kp6 Kp7 Kp8 ...
            # Kp values start at index 7
            for i in range(8):
                idx = 7 + i
                if idx >= len(parts):
                    break
                kp_val = float(parts[idx])
                if kp_val < 0:
                    continue
                kp_val = kp_val / 10.0  # GFZ stores Kp*10
                hour = i * 3
                ts = datetime(year, month, day, hour, tzinfo=UTC)
                samples.append(Sample(
                    source="gfz_kp", metric="kp_index",
                    timestamp=ts.isoformat(), value=kp_val, unit="Kp",
                    meta={"definitive": True},
                ))
        except (ValueError, IndexError):
            continue

    new = store.ingest(samples, source_name="gfz_kp_backfill")
    log.info(f"GFZ Kp backfill: {len(samples):,} samples, {new:,} new")
    return {"source": "gfz_kp", "total": len(samples), "new": new}


# ---------------------------------------------------------------------------
# NOAA CO-OPS Historical Tides — chunk by 30 days (API limit)
# ---------------------------------------------------------------------------

TIDE_STATIONS = {
    "9414290": ("San Francisco", 37.81, -122.47),
    "9410660": ("Los Angeles", 33.72, -118.27),
    "9413450": ("Monterey", 36.61, -121.89),
    "9415020": ("Point Reyes", 38.00, -122.98),
}

def backfill_tides(store: QuakeStore, start_year: int = 2020):
    start = datetime(start_year, 1, 1, tzinfo=UTC)
    end = datetime.now(UTC)
    total_samples = 0
    total_new = 0

    for station_id, (name, lat, lon) in TIDE_STATIONS.items():
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=5), end)  # API caps ~6 days at 6-min
            try:
                params = {
                    "begin_date": current.strftime("%Y%m%d"),
                    "end_date": chunk_end.strftime("%Y%m%d"),
                    "station": station_id,
                    "product": "water_level",
                    "datum": "MLLW",
                    "units": "metric",
                    "time_zone": "gmt",
                    "format": "json",
                    "application": "QuakeWatch",
                }
                url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?{urlencode(params)}"
                data = _fetch_json(url, timeout=30)

                samples = []
                for entry in data.get("data", []):
                    try:
                        ts = datetime.strptime(entry["t"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                        val = float(entry["v"])
                        samples.append(Sample(
                            source="noaa_tides", metric="water_level",
                            timestamp=ts.isoformat(), value=val, unit="m",
                            lat=lat, lon=lon,
                            meta={"station": station_id, "name": name},
                        ))
                    except (ValueError, KeyError):
                        continue

                new = store.ingest(samples, source_name=f"tides_{station_id}")
                total_samples += len(samples)
                total_new += new
                log.info(f"Tides {name} {current.strftime('%Y-%m')}: {len(samples)} samples ({new} new)")

            except Exception as e:
                log.warning(f"Tides {name} {current.strftime('%Y-%m')}: {e}")

            current = chunk_end
            time.sleep(0.3)

    log.info(f"Tides backfill complete: {total_samples:,} samples, {total_new:,} new")
    return {"source": "noaa_tides", "total": total_samples, "new": total_new}


# ---------------------------------------------------------------------------
# USGS Magnetometer Historical — chunk by 7 days (large responses)
# ---------------------------------------------------------------------------

MAGNETOMETER_STATIONS = {
    "FRN": (37.09, -119.72),
    "BOU": (40.14, -105.24),
    "TUC": (32.17, -110.73),
    "CMO": (64.87, -147.86),
    "HON": (21.32, -158.00),
    "SJG": (18.11, -66.15),
    "FRD": (38.20, -77.37),
    "NEW": (48.27, -117.12),
    "BSL": (30.35, -89.64),
    "SIT": (57.06, -135.33),
    "BRW": (71.32, -156.62),
    "SHU": (55.35, -160.47),
}


def backfill_magnetometer(store: QuakeStore, stations: list = None,
                          start_year: int = 2020):
    if stations is None:
        stations = list(MAGNETOMETER_STATIONS.keys())

    start = datetime(start_year, 1, 1, tzinfo=UTC)
    end = datetime.now(UTC)

    total_samples = 0
    total_new = 0

    for station in stations:
        lat, lon = MAGNETOMETER_STATIONS.get(station, (None, None))
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=7), end)
            try:
                params = {
                    "id": station,
                    "starttime": current.strftime("%Y-%m-%d"),
                    "endtime": chunk_end.strftime("%Y-%m-%d"),
                    "type": "variation",
                    "elements": "H,D,Z",
                    "sampling_period": "3600",
                    "format": "json",
                }
                url = f"https://geomag.usgs.gov/ws/data/?{urlencode(params)}"
                data = _fetch_json(url, timeout=60)

                times = data.get("times", [])
                samples = []
                for element_data in data.get("values", []):
                    element = element_data.get("id", "")
                    values = element_data.get("values", [])
                    for ts_str, val in zip(times, values):
                        if val is None:
                            continue
                        samples.append(Sample(
                            source="usgs_magnetometer", metric=f"mag_{element.lower()}",
                            timestamp=ts_str, value=float(val), unit="nT",
                            lat=lat, lon=lon,
                            meta={"station": station, "sampling": "hourly"},
                        ))

                new = store.ingest(samples, source_name=f"mag_{station}")
                total_samples += len(samples)
                total_new += new
                log.info(f"Mag {station} {current.strftime('%Y-%m-%d')}: {len(samples)} samples ({new} new)")

            except Exception as e:
                log.warning(f"Mag {station} {current.strftime('%Y-%m-%d')}: {e}")

            current = chunk_end
            time.sleep(0.3)

    log.info(f"Mag backfill complete ({len(stations)} stations): {total_samples:,} samples, {total_new:,} new")
    return {"source": "usgs_magnetometer", "total": total_samples, "new": total_new}


# ---------------------------------------------------------------------------
# INTERMAGNET Historical — Kakioka, Memambetsu, Vassouras
# Chunk by 7 days via HAPI endpoint
# ---------------------------------------------------------------------------

INTERMAGNET_STATIONS = {
    "kak": ("Kakioka", 36.23, 140.19),
    "mmb": ("Memambetsu", 43.91, 144.19),
    "vss": ("Vassouras", -22.40, -43.65),
}


def backfill_intermagnet(store: QuakeStore, start_year: int = 2020):
    start = datetime(start_year, 1, 1, tzinfo=UTC)
    end = datetime.now(UTC) - timedelta(hours=6)

    total_samples = 0
    total_new = 0

    for station, (name, lat, lon) in INTERMAGNET_STATIONS.items():
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=7), end)
            try:
                t_min = current.strftime("%Y-%m-%dT%H:%M:%SZ")
                t_max = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                url = (f"https://imag-data.bgs.ac.uk/GIN_V1/hapi/data"
                       f"?id={station}/adjusted/PT1M/xyzf"
                       f"&time.min={t_min}&time.max={t_max}&format=json")
                data = _fetch_json(url, timeout=60)

                samples = []
                for record in data.get("data", []):
                    ts = record[0]
                    xyz = record[1] if len(record) > 1 else []
                    f_total = record[2] if len(record) > 2 else None

                    if isinstance(xyz, list) and len(xyz) == 3:
                        for i, comp in enumerate(("x", "y", "z")):
                            if xyz[i] is not None:
                                samples.append(Sample(
                                    source="intermagnet", metric=f"imag_{comp}",
                                    timestamp=ts, value=float(xyz[i]), unit="nT",
                                    lat=lat, lon=lon,
                                    meta={"station": station, "name": name},
                                ))
                    if f_total is not None:
                        samples.append(Sample(
                            source="intermagnet", metric="imag_f",
                            timestamp=ts, value=float(f_total), unit="nT",
                            lat=lat, lon=lon,
                            meta={"station": station, "name": name},
                        ))

                new = store.ingest(samples, source_name=f"imag_{station}")
                total_samples += len(samples)
                total_new += new
                if samples:
                    log.info(f"INTERMAGNET {station} {current.strftime('%Y-%m-%d')}: {len(samples)} samples ({new} new)")

            except Exception as e:
                log.warning(f"INTERMAGNET {station} {current.strftime('%Y-%m-%d')}: {e}")

            current = chunk_end
            time.sleep(0.5)

    log.info(f"INTERMAGNET backfill complete: {total_samples:,} samples, {total_new:,} new")
    return {"source": "intermagnet", "total": total_samples, "new": total_new}


# ---------------------------------------------------------------------------
# NASA DONKI Historical — CME, Flares, Storms (chunk by 30 days)
# ---------------------------------------------------------------------------

def backfill_donki(store: QuakeStore, start_year: int = 2015):
    start = datetime(start_year, 1, 1, tzinfo=UTC)
    end = datetime.now(UTC)

    total_samples = 0
    total_new = 0

    endpoints = [
        ("CME", "cme_speed", lambda cme: _parse_cme(cme)),
        ("FLR", "solar_flare", lambda flr: _parse_flare(flr)),
        ("GST", "geomag_storm_kp", lambda gst: _parse_storm(gst)),
    ]

    for ep_name, metric, parser in endpoints:
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            try:
                url = (f"https://api.nasa.gov/DONKI/{ep_name}"
                       f"?startDate={current.strftime('%Y-%m-%d')}"
                       f"&endDate={chunk_end.strftime('%Y-%m-%d')}"
                       f"&api_key={NASA_API_KEY}")
                data = _fetch_json(url, timeout=30)

                samples = []
                for item in data:
                    parsed = parser(item)
                    samples.extend(parsed)

                new = store.ingest(samples, source_name=f"donki_{ep_name}")
                total_samples += len(samples)
                total_new += new
                if samples:
                    log.info(f"DONKI {ep_name} {current.strftime('%Y-%m')}: {len(samples)} samples ({new} new)")

            except Exception as e:
                log.warning(f"DONKI {ep_name} {current.strftime('%Y-%m')}: {e}")
                time.sleep(5)

            current = chunk_end
            time.sleep(1.5)

    log.info(f"DONKI backfill complete: {total_samples:,} samples, {total_new:,} new")
    return {"source": "nasa_donki", "total": total_samples, "new": total_new}


def _parse_cme(cme):
    ts = cme.get("startTime", "")
    if not ts:
        return []
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
    except ValueError:
        return []
    speed = None
    for analysis in cme.get("cmeAnalyses", []):
        if analysis.get("speed"):
            speed = analysis["speed"]
            break
    if not speed:
        return []
    return [Sample(
        source="nasa_donki", metric="cme_speed",
        timestamp=dt.isoformat(), value=float(speed), unit="km/s",
    )]


def _parse_flare(flare):
    ts = flare.get("beginTime", "")
    if not ts:
        return []
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
    except ValueError:
        return []
    class_type = flare.get("classType", "")
    if not class_type:
        return []
    class_map = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
    letter = class_type[0].upper()
    base = class_map.get(letter, 0)
    try:
        mag = float(class_type[1:]) if len(class_type) > 1 else 1.0
    except ValueError:
        mag = 1.0
    return [Sample(
        source="nasa_donki", metric="solar_flare",
        timestamp=dt.isoformat(), value=base + mag / 10.0, unit="class",
        meta={"class_type": class_type},
    )]


def _parse_storm(storm):
    samples = []
    for idx in storm.get("allKpIndex", []):
        ts = idx.get("observedTime", "")
        kp = idx.get("kpIndex")
        if not ts or kp is None:
            continue
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        samples.append(Sample(
            source="nasa_donki", metric="geomag_storm_kp",
            timestamp=dt.isoformat(), value=float(kp), unit="Kp",
        ))
    return samples


# ---------------------------------------------------------------------------
# NASA OMNI Solar Wind Historical (hourly, via OMNIWeb CGI)
# This pulls hourly-averaged solar wind from the OMNI2 dataset
# Available from 1963 to present
# ---------------------------------------------------------------------------

def backfill_omni_solar_wind(store: QuakeStore, start_year: int = 2015):
    """Download OMNI2 hourly data files from SPDF. Each file is one year.

    OMNI2 55-column format (0-indexed, verified against 2024 data):
      [0-2]: Year, DOY, Hour
      [8]:  |B| avg (nT), fill=999.9
      [14]: Bz GSE (nT), fill=999.9
      [16]: Bz GSM (nT), fill=999.9
      [22]: Proton temp (K), fill=9999999.
      [23]: Proton density (N/cm3), fill=999.9
      [24]: Flow speed (km/s), fill=9999.
      [38]: Kp*10, fill=999
      [40]: Dst (nT), fill=99999
    """
    end_year = datetime.now(UTC).year
    total_samples = 0
    total_new = 0

    for year in range(start_year, end_year + 1):
        try:
            url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
            log.info(f"Downloading OMNI2 {year}...")
            text = _fetch_text(url, timeout=60)

            samples = []
            for line in text.strip().split("\n"):
                parts = line.split()
                if len(parts) < 41:
                    continue
                try:
                    yr = int(parts[0])
                    doy = int(parts[1])
                    hr = int(parts[2])
                    dt = datetime(yr, 1, 1, hr, tzinfo=UTC) + timedelta(days=doy - 1)
                    ts = dt.isoformat()

                    field_defs = [
                        (8,  "imf_bt",              "nT",     999.9),
                        (14, "imf_bz_gse",          "nT",     999.9),
                        (16, "imf_bz_gsm",          "nT",     999.9),
                        (23, "solar_wind_density",   "p/cm3",  999.9),
                        (24, "solar_wind_speed",     "km/s",   9999.),
                    ]

                    for col, metric, unit, fill in field_defs:
                        val = float(parts[col])
                        if abs(val - fill) < 0.1 or val > fill * 0.99:
                            continue
                        samples.append(Sample(
                            source="omni_historical", metric=metric,
                            timestamp=ts, value=val, unit=unit,
                        ))

                    # Proton temperature
                    temp = float(parts[22])
                    if temp < 9000000 and temp > 100:
                        samples.append(Sample(
                            source="omni_historical", metric="solar_wind_temp",
                            timestamp=ts, value=temp, unit="K",
                        ))

                    # Kp (stored as Kp*10)
                    kp_raw = int(parts[38])
                    if 0 <= kp_raw < 100:
                        samples.append(Sample(
                            source="omni_historical", metric="kp_index",
                            timestamp=ts, value=kp_raw / 10.0, unit="Kp",
                        ))

                    # Dst
                    dst = int(parts[40])
                    if abs(dst) < 1000:
                        samples.append(Sample(
                            source="omni_historical", metric="dst_index",
                            timestamp=ts, value=float(dst), unit="nT",
                        ))

                except (ValueError, IndexError):
                    continue

            new = store.ingest(samples, source_name=f"omni_{year}")
            total_samples += len(samples)
            total_new += new
            log.info(f"OMNI {year}: {len(samples):,} samples ({new:,} new)")

        except Exception as e:
            log.warning(f"OMNI {year}: {e}")

        time.sleep(0.5)

    log.info(f"OMNI backfill complete: {total_samples:,} samples, {total_new:,} new")
    return {"source": "omni_solar_wind", "total": total_samples, "new": total_new}


# ---------------------------------------------------------------------------
# Master backfill runner
# ---------------------------------------------------------------------------

def run_full_backfill(store: QuakeStore = None, start_year: int = 2015):
    if store is None:
        store = QuakeStore()

    log.info(f"Starting full historical backfill from {start_year}")
    results = []

    # 1. Earthquakes — the target variable, need the most depth
    log.info("=" * 60)
    log.info("PHASE 1: USGS Earthquake Catalog")
    log.info("=" * 60)
    results.append(backfill_earthquakes(store, start_year=start_year))

    # 2. GFZ Kp — definitive planetary geomagnetic index
    log.info("=" * 60)
    log.info("PHASE 2: GFZ Potsdam Kp Index")
    log.info("=" * 60)
    results.append(backfill_kp_index(store, start_year=start_year))

    # 3. OMNI solar wind — the big one, hourly space weather back to 2015
    log.info("=" * 60)
    log.info("PHASE 3: NASA OMNI Solar Wind")
    log.info("=" * 60)
    results.append(backfill_omni_solar_wind(store, start_year=start_year))

    # 4. NOAA Tides — California coastal stations
    log.info("=" * 60)
    log.info("PHASE 4: NOAA Tide Gauges")
    log.info("=" * 60)
    results.append(backfill_tides(store, start_year=max(start_year, 2018)))

    # 5. USGS Magnetometer — 12 global stations
    log.info("=" * 60)
    log.info("PHASE 5: USGS Magnetometer (12 stations)")
    log.info("=" * 60)
    results.append(backfill_magnetometer(store, start_year=max(start_year, 2020)))

    # 6. NASA DONKI — CME, flares, storms
    log.info("=" * 60)
    log.info("PHASE 6: NASA DONKI (CME, Flares, Storms)")
    log.info("=" * 60)
    results.append(backfill_donki(store, start_year=start_year))

    # 7. INTERMAGNET — Kakioka, Memambetsu, Vassouras
    log.info("=" * 60)
    log.info("PHASE 7: INTERMAGNET (3 observatories)")
    log.info("=" * 60)
    results.append(backfill_intermagnet(store, start_year=max(start_year, 2020)))

    log.info("=" * 60)
    log.info("BACKFILL COMPLETE")
    total = sum(r.get("total", 0) for r in results)
    new = sum(r.get("new", 0) for r in results)
    log.info(f"Total: {total:,} samples, {new:,} new")
    log.info("=" * 60)

    stats = store.get_stats()
    log.info(f"Database now has {stats['total_samples']:,} total samples, "
             f"{stats['earthquake_count']:,} earthquakes")

    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    # Subcommand: `python -m ingest.backfill emsc [start_year] [end_year] [zone]`
    if len(sys.argv) > 1 and sys.argv[1] == "emsc":
        from .store import QuakeStore
        sy = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
        ey = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
        only = sys.argv[4] if len(sys.argv) > 4 else None
        zs = [z for z in EMSC_ZONES if z["id"] == only] if only else None
        store = QuakeStore()
        backfill_earthquakes_emsc(store, start_year=sy, end_year=ey, zones=zs)
    else:
        start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
        run_full_backfill(start_year=start_year)
