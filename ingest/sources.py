"""SeismicLab — Data source definitions and fetchers for earthquake prediction engine.

Each source returns normalized records with UTC timestamps for alignment.
All APIs verified working from DGX Spark (2026-06-23).
"""

import json
import os
import time
import logging
import csv
import io
import sqlite3
import math
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")

log = logging.getLogger("seismiclab.sources")

UTC = timezone.utc


@dataclass
class Sample:
    source: str
    metric: str
    timestamp: str  # ISO 8601 UTC
    value: float
    unit: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    meta: dict = field(default_factory=dict)


def _fetch_json(url, timeout=30, retries=2):
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "SeismicLab/1.0", "Accept": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries and "429" in str(e):
                time.sleep(2 * (attempt + 1))
                continue
            raise


def _fetch_text(url, timeout=30):
    req = Request(url, headers={"User-Agent": "SeismicLab/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


# ---------------------------------------------------------------------------
# 1. SEISMIC — USGS Earthquake Catalog (FDSN)
# ---------------------------------------------------------------------------

def fetch_usgs_earthquakes(start: datetime = None, end: datetime = None,
                           min_mag: float = 2.5, limit: int = 2000) -> list[Sample]:
    if not end:
        end = datetime.now(UTC)
    if not start:
        start = end - timedelta(days=30)

    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_mag,
        "limit": limit,
        "orderby": "time",
    }
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?{urlencode(params)}"
    data = _fetch_json(url)

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
    log.info(f"USGS earthquakes: {len(samples)} samples ({start.date()} to {end.date()})")
    return samples


# ---------------------------------------------------------------------------
# 1b. SEISMIC — EMSC (Seismic Portal, FDSN) — faster than USGS
# ---------------------------------------------------------------------------

def fetch_emsc_earthquakes(min_mag: float = 2.5, limit: int = 300) -> list[Sample]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=6)
    params = {
        "format": "json",
        "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmag": min_mag,
        "limit": limit,
        "orderby": "time",
    }
    url = f"https://www.seismicportal.eu/fdsnws/event/1/query?{urlencode(params)}"
    data = _fetch_json(url, timeout=20)

    samples = []
    for feat in data.get("features", []):
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        mag = props.get("mag")
        if mag is None:
            continue
        ts_raw = props.get("time", "")
        if not ts_raw:
            continue
        try:
            if ts_raw.endswith("Z"):
                ts_raw = ts_raw[:-1] + "+00:00"
            ts = datetime.fromisoformat(ts_raw).astimezone(UTC).isoformat()
        except (ValueError, TypeError):
            continue

        eq_id = props.get("source_id") or props.get("unid") or feat.get("id", "")
        place = props.get("flynn_region", "") or props.get("region", "")

        samples.append(Sample(
            source="emsc_earthquake", metric="magnitude",
            timestamp=ts, value=float(mag), unit="Mw",
            lat=coords[1], lon=coords[0],
            meta={"depth_km": coords[2] if len(coords) > 2 else None,
                  "place": place, "type": "earthquake", "id": f"emsc_{eq_id}"},
        ))
    log.info(f"EMSC earthquakes: {len(samples)} samples (last 6h)")
    return samples


# ---------------------------------------------------------------------------
# 2. SPACE WEATHER — NOAA SWPC (Kp, Dst, Solar Wind)
# ---------------------------------------------------------------------------

def fetch_kp_index(days: int = 30) -> list[Sample]:
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    data = _fetch_json(url)

    cutoff = datetime.now(UTC) - timedelta(days=days)
    samples = []
    for entry in data:
        ts = datetime.strptime(entry["time_tag"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
        if ts < cutoff:
            continue
        samples.append(Sample(
            source="noaa_swpc", metric="kp_index",
            timestamp=ts.isoformat(), value=float(entry["kp_index"]),
            unit="Kp",
        ))
    log.info(f"NOAA Kp index: {len(samples)} samples")
    return samples


def fetch_dst_index() -> list[Sample]:
    url = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
    data = _fetch_json(url)

    samples = []
    for entry in data:
        try:
            ts_str = entry.get("time_tag", "")
            val = entry.get("dst")
            if not ts_str or val is None:
                continue
            ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
            samples.append(Sample(
                source="noaa_swpc", metric="dst_index",
                timestamp=ts.isoformat(), value=float(val), unit="nT",
            ))
        except (ValueError, KeyError):
            continue
    log.info(f"NOAA Dst index: {len(samples)} samples")
    return samples


def fetch_solar_wind() -> list[Sample]:
    url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
    data = _fetch_json(url)

    samples = []
    for row in data[1:]:
        try:
            ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=UTC)
            density = float(row[1]) if row[1] else None
            speed = float(row[2]) if row[2] else None
            temp = float(row[3]) if row[3] else None

            if density is not None:
                samples.append(Sample(
                    source="noaa_swpc", metric="solar_wind_density",
                    timestamp=ts.isoformat(), value=density, unit="p/cm3",
                ))
            if speed is not None:
                samples.append(Sample(
                    source="noaa_swpc", metric="solar_wind_speed",
                    timestamp=ts.isoformat(), value=speed, unit="km/s",
                ))
            if temp is not None:
                samples.append(Sample(
                    source="noaa_swpc", metric="solar_wind_temp",
                    timestamp=ts.isoformat(), value=temp, unit="K",
                ))
        except (ValueError, IndexError):
            continue
    log.info(f"NOAA solar wind: {len(samples)} samples")
    return samples


def fetch_mag_field() -> list[Sample]:
    url = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
    data = _fetch_json(url)

    samples = []
    for row in data[1:]:
        try:
            ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=UTC)
            bz = float(row[3]) if row[3] else None
            bt = float(row[6]) if len(row) > 6 and row[6] else None

            if bz is not None:
                samples.append(Sample(
                    source="noaa_swpc", metric="imf_bz",
                    timestamp=ts.isoformat(), value=bz, unit="nT",
                ))
            if bt is not None:
                samples.append(Sample(
                    source="noaa_swpc", metric="imf_bt",
                    timestamp=ts.isoformat(), value=bt, unit="nT",
                ))
        except (ValueError, IndexError):
            continue
    log.info(f"NOAA IMF/Mag: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# 3. SPACE WEATHER — NASA DONKI (CME, Solar Flares, Geomagnetic Storms)
# ---------------------------------------------------------------------------

def fetch_donki_cme(days: int = 30) -> list[Sample]:
    start = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://api.nasa.gov/DONKI/CME?startDate={start}&api_key={NASA_API_KEY}"
    data = _fetch_json(url)

    samples = []
    for cme in data:
        ts = cme.get("startTime", "")
        if not ts:
            continue
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        except ValueError:
            continue

        speed = None
        for analysis in cme.get("cmeAnalyses", []):
            if analysis.get("speed"):
                speed = analysis["speed"]
                break

        if speed:
            samples.append(Sample(
                source="nasa_donki", metric="cme_speed",
                timestamp=dt.isoformat(), value=float(speed), unit="km/s",
                meta={"type": cme.get("note", "")[:200]},
            ))
    log.info(f"NASA DONKI CME: {len(samples)} samples")
    return samples


def fetch_donki_flares(days: int = 30) -> list[Sample]:
    start = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://api.nasa.gov/DONKI/FLR?startDate={start}&api_key={NASA_API_KEY}"
    data = _fetch_json(url)

    class_map = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
    samples = []
    for flare in data:
        ts = flare.get("beginTime", "")
        if not ts:
            continue
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        except ValueError:
            continue

        class_type = flare.get("classType", "")
        if class_type:
            letter = class_type[0].upper()
            base = class_map.get(letter, 0)
            try:
                magnitude = float(class_type[1:]) if len(class_type) > 1 else 1.0
            except ValueError:
                magnitude = 1.0
            value = base + (magnitude / 10.0)

            samples.append(Sample(
                source="nasa_donki", metric="solar_flare",
                timestamp=dt.isoformat(), value=value, unit="class",
                meta={"class_type": class_type},
            ))
    log.info(f"NASA DONKI flares: {len(samples)} samples")
    return samples


def fetch_donki_storms(days: int = 30) -> list[Sample]:
    start = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://api.nasa.gov/DONKI/GST?startDate={start}&api_key={NASA_API_KEY}"
    data = _fetch_json(url)

    samples = []
    for storm in data:
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
                meta={"source": idx.get("source", "")},
            ))
    log.info(f"NASA DONKI storms: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# 4. GROUNDWATER — USGS NWIS (California wells)
# ---------------------------------------------------------------------------

def fetch_groundwater(state: str = "CA", days: int = 1) -> list[Sample]:
    params = {
        "format": "json",
        "stateCd": state,
        "parameterCd": "72019",  # depth to water level
        "siteStatus": "active",
        "period": f"P{days}D",
    }
    url = f"https://waterservices.usgs.gov/nwis/iv/?{urlencode(params)}"
    data = _fetch_json(url, timeout=60)

    samples = []
    site_count = 0
    for ts_data in data.get("value", {}).get("timeSeries", []):
        site_info = ts_data.get("sourceInfo", {})
        geo = site_info.get("geoLocation", {}).get("geogLocation", {})
        lat = geo.get("latitude")
        lon = geo.get("longitude")
        site_name = site_info.get("siteName", "")
        site_codes = site_info.get("siteCode", [{}])
        site_id = site_codes[0].get("value", "") if site_codes else ""
        site_count += 1

        for val_entry in ts_data.get("values", [{}])[0].get("value", []):
            try:
                v = float(val_entry["value"])
                if v < -900:
                    continue
                ts = val_entry["dateTime"]
                samples.append(Sample(
                    source="usgs_groundwater", metric="water_level_depth",
                    timestamp=ts, value=v, unit="ft_below_surface",
                    lat=lat, lon=lon,
                    meta={"site_id": site_id, "site_name": site_name},
                ))
            except (ValueError, KeyError):
                continue

    log.info(f"USGS groundwater: {len(samples)} samples from {site_count} sites")
    return samples


# ---------------------------------------------------------------------------
# 5. MAGNETOMETER — USGS Geomagnetism (multi-station)
# ---------------------------------------------------------------------------

MAGNETOMETER_STATIONS = {
    "FRN": (37.09, -119.72),   # Fresno, CA
    "BOU": (40.14, -105.24),   # Boulder, CO
    "TUC": (32.17, -110.73),   # Tucson, AZ
    "CMO": (64.87, -147.86),   # College, AK
    "HON": (21.32, -158.00),   # Honolulu, HI
    "SJG": (18.11, -66.15),    # San Juan, PR (Caribbean — near Venezuela)
    "GUA": (13.59, 144.87),    # Guam (Pacific)
    "FRD": (38.20, -77.37),    # Fredericksburg, VA
    "NEW": (48.27, -117.12),   # Newport, WA
    "BSL": (30.35, -89.64),    # Stennis, MS
    "SIT": (57.06, -135.33),   # Sitka, AK
    "BRW": (71.32, -156.62),   # Barrow, AK (Arctic)
    "DED": (70.36, -148.79),   # Deadhorse, AK
    "SHU": (55.35, -160.47),   # Shumagin, AK
}


def fetch_magnetometer(stations: list[str] = None, days: int = 7) -> list[Sample]:
    if stations is None:
        stations = list(MAGNETOMETER_STATIONS.keys())

    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    all_samples = []
    for station in stations:
        try:
            params = {
                "id": station,
                "starttime": start.strftime("%Y-%m-%d"),
                "endtime": end.strftime("%Y-%m-%d"),
                "type": "variation",
                "elements": "X,Y,Z",
                "sampling_period": "60",
                "format": "json",
            }
            url = f"https://geomag.usgs.gov/ws/data/?{urlencode(params)}"
            data = _fetch_json(url, timeout=30)

            lat, lon = MAGNETOMETER_STATIONS.get(station, (None, None))
            times = data.get("times", [])

            for element_data in data.get("values", []):
                element = element_data.get("id", "")
                values = element_data.get("values", [])

                for ts_str, val in zip(times, values):
                    if val is None:
                        continue
                    try:
                        all_samples.append(Sample(
                            source="usgs_magnetometer", metric=f"mag_{element.lower()}",
                            timestamp=ts_str, value=float(val), unit="nT",
                            lat=lat, lon=lon,
                            meta={"station": station},
                        ))
                    except (ValueError, TypeError):
                        continue
            log.info(f"  Magnetometer {station}: {len(times)} timestamps")
        except Exception as e:
            log.warning(f"  Magnetometer {station}: {e}")
            continue

    log.info(f"USGS magnetometer: {len(all_samples)} samples from {len(stations)} stations")
    return all_samples


# ---------------------------------------------------------------------------
# 5b. GOES SATELLITE MAGNETOMETER
# ---------------------------------------------------------------------------

def fetch_goes_magnetometer() -> list[Sample]:
    url = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-1-day.json"
    data = _fetch_json(url)

    samples = []
    for entry in data:
        ts = entry.get("time_tag")
        if not ts:
            continue
        sat = entry.get("satellite", "")
        for component in ("He", "Hp", "Hn", "total"):
            val = entry.get(component)
            if val is not None:
                samples.append(Sample(
                    source="goes_magnetometer", metric=f"goes_mag_{component.lower()}",
                    timestamp=ts, value=float(val), unit="nT",
                    meta={"satellite": sat},
                ))
    log.info(f"GOES magnetometer: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# 5c. INTERMAGNET — Global Observatory Network (HAPI)
# ---------------------------------------------------------------------------

INTERMAGNET_STATIONS = {
    "kak": ("Kakioka", 36.23, 140.19, "Japan"),
    "mmb": ("Memambetsu", 43.91, 144.19, "Japan"),
    "vss": ("Vassouras", -22.40, -43.65, "Brazil"),
}


def fetch_intermagnet(stations: list[str] = None, hours: int = 24) -> list[Sample]:
    if stations is None:
        stations = list(INTERMAGNET_STATIONS.keys())

    end = datetime.now(UTC) - timedelta(hours=6)  # ~6h processing latency
    start = end - timedelta(hours=hours)

    all_samples = []
    for station in stations:
        info = INTERMAGNET_STATIONS.get(station)
        if not info:
            continue
        name, lat, lon, country = info

        try:
            dataset = f"{station}/adjusted/PT1M/xyzf"
            t_min = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            t_max = end.strftime("%Y-%m-%dT%H:%M:%SZ")
            url = f"https://imag-data.bgs.ac.uk/GIN_V1/hapi/data?id={dataset}&time.min={t_min}&time.max={t_max}&format=json"
            data = _fetch_json(url, timeout=30)

            for record in data.get("data", []):
                ts = record[0]
                xyz = record[1] if len(record) > 1 else []
                f_total = record[2] if len(record) > 2 else None

                if isinstance(xyz, list) and len(xyz) == 3:
                    for i, comp in enumerate(("x", "y", "z")):
                        if xyz[i] is not None:
                            all_samples.append(Sample(
                                source="intermagnet", metric=f"imag_{comp}",
                                timestamp=ts, value=float(xyz[i]), unit="nT",
                                lat=lat, lon=lon,
                                meta={"station": station, "name": name, "country": country},
                            ))
                if f_total is not None:
                    all_samples.append(Sample(
                        source="intermagnet", metric="imag_f",
                        timestamp=ts, value=float(f_total), unit="nT",
                        lat=lat, lon=lon,
                        meta={"station": station, "name": name, "country": country},
                    ))
            log.info(f"  INTERMAGNET {station} ({name}): {len(data.get('data', []))} timestamps")
        except Exception as e:
            log.warning(f"  INTERMAGNET {station}: {e}")
            continue

    log.info(f"INTERMAGNET: {len(all_samples)} samples from {len(stations)} stations")
    return all_samples


# ---------------------------------------------------------------------------
# 6. ATMOSPHERIC — NWS Barometric Pressure
# ---------------------------------------------------------------------------

NWS_STATIONS = {
    "KFAT": ("Fresno, CA", 36.78, -119.72),
    "KOAK": ("Oakland, CA", 37.72, -122.22),
    "KLAX": ("Los Angeles, CA", 33.94, -118.41),
    "KSFO": ("San Francisco, CA", 37.62, -122.37),
}

def fetch_barometric_pressure(station_ids: list[str] = None) -> list[Sample]:
    if not station_ids:
        station_ids = list(NWS_STATIONS.keys())

    samples = []
    for station_id in station_ids:
        try:
            url = f"https://api.weather.gov/stations/{station_id}/observations?limit=48"
            data = _fetch_json(url, timeout=15)

            info = NWS_STATIONS.get(station_id, (station_id, None, None))

            for obs in data.get("features", []):
                props = obs.get("properties", {})
                ts = props.get("timestamp")
                pressure = props.get("barometricPressure", {})

                if ts and pressure and pressure.get("value") is not None:
                    val = pressure["value"] / 100.0  # Pa -> hPa
                    samples.append(Sample(
                        source="nws_weather", metric="barometric_pressure",
                        timestamp=ts, value=val, unit="hPa",
                        lat=info[1], lon=info[2],
                        meta={"station": station_id, "name": info[0]},
                    ))

                temp = props.get("temperature", {})
                if ts and temp and temp.get("value") is not None:
                    samples.append(Sample(
                        source="nws_weather", metric="temperature",
                        timestamp=ts, value=temp["value"], unit="C",
                        lat=info[1], lon=info[2],
                        meta={"station": station_id, "name": info[0]},
                    ))
        except Exception as e:
            log.warning(f"NWS station {station_id}: {e}")
            continue

    log.info(f"NWS barometric pressure: {len(samples)} samples from {len(station_ids)} stations")
    return samples


# ---------------------------------------------------------------------------
# 7. TIDAL — NOAA CO-OPS Tide Gauges
# ---------------------------------------------------------------------------

TIDE_STATIONS = {
    "9414290": ("San Francisco", 37.81, -122.47),
    "9410660": ("Los Angeles", 33.72, -118.27),
    "9413450": ("Monterey", 36.61, -121.89),
    "9415020": ("Point Reyes", 38.00, -122.98),
}

def fetch_tidal_data(station_ids: list[str] = None, days: int = 7) -> list[Sample]:
    if not station_ids:
        station_ids = list(TIDE_STATIONS.keys())

    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    samples = []
    for station_id in station_ids:
        try:
            params = {
                "begin_date": start.strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
                "station": station_id,
                "product": "water_level",
                "datum": "MLLW",
                "units": "metric",
                "time_zone": "gmt",
                "format": "json",
                "application": "SeismicLab",
            }
            url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?{urlencode(params)}"
            data = _fetch_json(url, timeout=20)

            info = TIDE_STATIONS.get(station_id, (station_id, None, None))

            for entry in data.get("data", []):
                try:
                    ts = datetime.strptime(entry["t"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                    val = float(entry["v"])
                    samples.append(Sample(
                        source="noaa_tides", metric="water_level",
                        timestamp=ts.isoformat(), value=val, unit="m",
                        lat=info[1], lon=info[2],
                        meta={"station": station_id, "name": info[0]},
                    ))
                except (ValueError, KeyError):
                    continue
        except Exception as e:
            log.warning(f"Tide station {station_id}: {e}")
            continue

    log.info(f"NOAA tides: {len(samples)} samples from {len(station_ids)} stations")
    return samples


# ---------------------------------------------------------------------------
# 8. NOAA Solar Flare List (XRS)
# ---------------------------------------------------------------------------

def fetch_xray_flares() -> list[Sample]:
    url = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
    data = _fetch_json(url)

    samples = []
    for flare in data:
        ts = flare.get("begin_time") or flare.get("max_time")
        xray_class = flare.get("max_class", "")
        if not ts or not xray_class:
            continue

        class_map = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
        letter = xray_class[0].upper()
        base = class_map.get(letter, 0)
        try:
            mag = float(xray_class[1:]) if len(xray_class) > 1 else 1.0
        except ValueError:
            mag = 1.0

        samples.append(Sample(
            source="noaa_xray", metric="xray_flare",
            timestamp=ts, value=base + mag / 10.0, unit="class",
            meta={"class": xray_class},
        ))
    log.info(f"NOAA X-ray flares: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# 8b. GOES X-RAY FLUX — Continuous 1-min measurements (not just flare events)
# ---------------------------------------------------------------------------

def fetch_goes_xray_flux() -> list[Sample]:
    url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json"
    data = _fetch_json(url)

    samples = []
    for entry in data:
        ts = entry.get("time_tag")
        flux = entry.get("flux")
        if ts and flux is not None and flux > 0:
            samples.append(Sample(
                source="goes_xray", metric="xray_flux",
                timestamp=ts, value=float(flux), unit="W/m2",
                meta={"energy": entry.get("energy", "")},
            ))
    log.info(f"GOES X-ray flux: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# 8c. OLR — Daily Outgoing Longwave Radiation from NOAA interpolated dataset
# ---------------------------------------------------------------------------

def fetch_olr_recent() -> list[Sample]:
    """Fetch recent OLR from NOAA CDR daily means via OPeNDAP for key seismic zones."""
    try:
        import xarray as xr
    except ImportError:
        log.warning("xarray not installed — OLR fetch disabled")
        return []

    url = "https://psl.noaa.gov/thredds/dodsC/Datasets/interp_OLR/olr.day.mean.nc"
    zones = [
        (-22, -70, "chile"), (37, 140, "japan"), (-2, 115, "indonesia"),
        (33.5, -117.5, "socal"), (16, -96, "mexico"), (38, 28, "turkey"),
        (32, 82, "himalaya"), (16, -72, "caribbean"), (15, 123, "philippines"),
    ]

    samples = []
    try:
        ds = xr.open_dataset(url)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        for lat, lon, name in zones:
            try:
                olr_lon = lon % 360
                ts = ds.olr.sel(lat=lat, lon=olr_lon, method="nearest")
                ts = ts.sel(time=slice(str(start.date()), str(end.date())))
                for t, val in zip(ts.time.values, ts.values):
                    if hasattr(val, 'item'):
                        val = val.item()
                    if val != val:  # NaN check
                        continue
                    samples.append(Sample(
                        source="noaa_olr", metric="olr",
                        timestamp=str(t)[:19], value=float(val), unit="W/m2",
                        lat=lat, lon=lon,
                    ))
            except Exception as e:
                log.warning(f"OLR fetch for {name} failed: {e}")
                continue
        ds.close()
    except Exception as e:
        log.error(f"OLR dataset open failed: {e}")
    log.info(f"OLR recent: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# 8d. GRACE-FO Gravity Anomaly — Monthly mascon data
# ---------------------------------------------------------------------------

def fetch_grace_recent() -> list[Sample]:
    """Fetch recent GRACE-FO gravity data from JPL mascon RL06.1v3."""
    try:
        import xarray as xr
    except ImportError:
        log.warning("xarray not installed — GRACE fetch disabled")
        return []

    grace_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "grace_mascon.nc")
    if not os.path.exists(grace_path):
        log.info("GRACE mascon file not found — skipping")
        return []

    zones = [
        (-22, -70, "chile"), (37, 140, "japan"), (-2, 115, "indonesia"),
        (33.5, -117.5, "socal"), (16, -96, "mexico"), (38, 28, "turkey"),
        (32, 82, "himalaya"), (16, -72, "caribbean"),
    ]

    samples = []
    try:
        ds = xr.open_dataset(grace_path)
        var_name = None
        for v in ['lwe_thickness', 'ewh', 'mascon']:
            if v in ds.data_vars:
                var_name = v
                break
        if not var_name:
            var_name = list(ds.data_vars)[0]

        for lat, lon, name in zones:
            try:
                ts = ds[var_name].sel(lat=lat, lon=lon % 360, method="nearest")
                last_12 = ts.isel(time=slice(-12, None))
                for t, val in zip(last_12.time.values, last_12.values):
                    if hasattr(val, 'item'):
                        val = val.item()
                    if val != val:
                        continue
                    samples.append(Sample(
                        source="grace_gravity", metric="gravity_anomaly",
                        timestamp=str(t)[:19], value=float(val), unit="cm_eqw",
                        lat=lat, lon=lon,
                    ))
            except Exception as e:
                log.warning(f"GRACE extract for {name} failed: {e}")
                continue
        ds.close()
    except Exception as e:
        log.error(f"GRACE open failed: {e}")
    log.info(f"GRACE recent: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# 9. ELECTRON FLUX / PROTON FLUX — GOES Satellite
# ---------------------------------------------------------------------------

def fetch_electron_flux() -> list[Sample]:
    url = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-3-day.json"
    data = _fetch_json(url)

    samples = []
    for entry in data:
        ts = entry.get("time_tag")
        flux = entry.get("flux")
        energy = entry.get("energy", "")
        if ts and flux is not None:
            samples.append(Sample(
                source="noaa_goes", metric="electron_flux",
                timestamp=ts, value=float(flux), unit="particles/cm2-s-sr",
                meta={"energy": energy},
            ))
    log.info(f"GOES electron flux: {len(samples)} samples")
    return samples


def fetch_proton_flux() -> list[Sample]:
    url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json"
    data = _fetch_json(url)

    samples = []
    for entry in data:
        ts = entry.get("time_tag")
        flux = entry.get("flux")
        energy = entry.get("energy", "")
        if ts and flux is not None:
            samples.append(Sample(
                source="noaa_goes", metric="proton_flux",
                timestamp=ts, value=float(flux), unit="particles/cm2-s-sr",
                meta={"energy": energy},
            ))
    log.info(f"GOES proton flux: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# FIRMS — Near Real-Time Satellite Thermal Anomalies
# ---------------------------------------------------------------------------

FIRMS_KEY = os.environ.get("FIRMS_API_KEY", "")
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
_FIRMS_DB = Path(__file__).parent.parent / "data" / "seismiclab.db"


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_firms_nrt() -> list[Sample]:
    conn = sqlite3.connect(str(_FIRMS_DB), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.row_factory = sqlite3.Row

    volcanoes = conn.execute(
        "SELECT volcano_number, name, lat, lon FROM volcanoes "
        "WHERE last_eruption_year >= 2000"
    ).fetchall()
    if not volcanoes:
        conn.close()
        return []

    used = set()
    bboxes = []
    volc_list = [(v["volcano_number"], v["name"], v["lat"], v["lon"]) for v in volcanoes]

    for vnum, vname, vlat, vlon in volc_list:
        if vnum in used:
            continue
        group = [(vnum, vname, vlat, vlon)]
        used.add(vnum)
        for vnum2, vname2, vlat2, vlon2 in volc_list:
            if vnum2 in used:
                continue
            if abs(vlat - vlat2) < 3.0 and abs(vlon - vlon2) < 3.0:
                group.append((vnum2, vname2, vlat2, vlon2))
                used.add(vnum2)
        lats = [v[2] for v in group]
        lons = [v[3] for v in group]
        pad = 1.0
        bbox = (
            round(min(lons) - pad, 2), round(min(lats) - pad, 2),
            round(max(lons) + pad, 2), round(max(lats) + pad, 2),
        )
        bboxes.append((bbox, group))

    nrt_sources = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
    total_new = 0
    volcano_stats = {}

    for bbox, group in bboxes:
        west, south, east, north = bbox
        bbox_str = f"{west},{south},{east},{north}"
        for src in nrt_sources:
            url = f"{FIRMS_BASE}/{FIRMS_KEY}/{src}/{bbox_str}/1"
            try:
                req = Request(url, headers={"User-Agent": "SeismicLab/1.0"})
                with urlopen(req, timeout=30) as resp:
                    text = resp.read().decode()
            except Exception as e:
                log.warning(f"FIRMS NRT {src} bbox {bbox_str}: {e}")
                continue

            if "transaction_limit" in text:
                log.warning("FIRMS rate limited — stopping NRT fetch")
                break

            if "Invalid" in text or "Error" in text or "<html" in text.lower():
                continue

            rows = list(csv.DictReader(io.StringIO(text)))
            for row in rows:
                try:
                    rlat = float(row.get("latitude", 0))
                    rlon = float(row.get("longitude", 0))
                    brightness = float(
                        row.get("brightness") or row.get("bright_ti4") or 0)
                    frp = float(row.get("frp", 0))
                except (ValueError, TypeError):
                    continue

                best_vid, best_dist = None, 100.0
                for vnum, vname, vlat, vlon in group:
                    d = _haversine_km(rlat, rlon, vlat, vlon)
                    if d < best_dist:
                        best_vid, best_dist = vnum, d

                if best_vid is None:
                    continue

                acq_date = row.get("acq_date", "")
                acq_time = row.get("acq_time", "")

                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO thermal_anomalies "
                        "(lat,lon,brightness,frp,acq_date,acq_time,satellite,"
                        "instrument,confidence,daynight,source,nearest_volcano_id,"
                        "nearest_volcano_dist_km) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (rlat, rlon, brightness, frp, acq_date, acq_time,
                         row.get("satellite", ""), row.get("instrument", ""),
                         row.get("confidence", ""), row.get("daynight", ""),
                         src, best_vid, round(best_dist, 1)),
                    )
                    if conn.total_changes:
                        total_new += 1
                except sqlite3.IntegrityError:
                    pass

                vkey = best_vid
                if vkey not in volcano_stats:
                    vname_match = next(
                        (n for v, n, la, lo in group if v == best_vid), str(best_vid)
                    )
                    volcano_stats[vkey] = {
                        "name": vname_match, "count": 0,
                        "max_brightness": 0, "max_frp": 0,
                        "lat": next(la for v, n, la, lo in group if v == best_vid),
                        "lon": next(lo for v, n, la, lo in group if v == best_vid),
                    }
                volcano_stats[vkey]["count"] += 1
                volcano_stats[vkey]["max_brightness"] = max(
                    volcano_stats[vkey]["max_brightness"], brightness)
                volcano_stats[vkey]["max_frp"] = max(
                    volcano_stats[vkey]["max_frp"], frp)

            time.sleep(0.5)

    conn.commit()
    conn.close()

    samples = []
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for vid, st in volcano_stats.items():
        if st["count"] > 0:
            samples.append(Sample(
                source="firms_nrt", metric="hotspot_count",
                timestamp=now_iso, value=float(st["count"]),
                unit="detections", lat=st["lat"], lon=st["lon"],
                meta={"volcano": st["name"], "max_brightness": st["max_brightness"],
                       "max_frp": st["max_frp"]},
            ))

    log.info(f"FIRMS NRT: {total_new} new hotspots across {len(volcano_stats)} volcanoes")
    return samples


# ---------------------------------------------------------------------------
# SOURCE REGISTRY — all sources with fetch functions and poll intervals
# ---------------------------------------------------------------------------

SOURCE_REGISTRY = {
    "usgs_earthquake": {
        "fn": fetch_usgs_earthquakes,
        "interval_min": 5,
        "description": "USGS earthquake catalog (mag, depth, location)",
    },
    "emsc_earthquake": {
        "fn": fetch_emsc_earthquakes,
        "interval_min": 2,
        "description": "EMSC Seismic Portal — faster event detection (2-5 min lag)",
    },
    "kp_index": {
        "fn": fetch_kp_index,
        "interval_min": 15,
        "description": "Planetary Kp geomagnetic index",
    },
    "dst_index": {
        "fn": fetch_dst_index,
        "interval_min": 60,
        "description": "Disturbance Storm Time index",
    },
    "solar_wind": {
        "fn": fetch_solar_wind,
        "interval_min": 5,
        "description": "ACE/DSCOVR solar wind plasma (density, speed, temp)",
    },
    "imf_mag": {
        "fn": fetch_mag_field,
        "interval_min": 5,
        "description": "Interplanetary magnetic field (Bz, Bt)",
    },
    "donki_cme": {
        "fn": fetch_donki_cme,
        "interval_min": 60 if NASA_API_KEY != "DEMO_KEY" else 480,
        "description": "NASA DONKI coronal mass ejections",
    },
    "donki_flares": {
        "fn": fetch_donki_flares,
        "interval_min": 60 if NASA_API_KEY != "DEMO_KEY" else 480,
        "description": "NASA DONKI solar flare events",
    },
    "donki_storms": {
        "fn": fetch_donki_storms,
        "interval_min": 60 if NASA_API_KEY != "DEMO_KEY" else 480,
        "description": "NASA DONKI geomagnetic storms",
    },
    "groundwater": {
        "fn": fetch_groundwater,
        "interval_min": 30,
        "description": "USGS California groundwater levels",
    },
    "magnetometer": {
        "fn": fetch_magnetometer,
        "interval_min": 10,
        "description": "USGS Fresno magnetometer (X, Y, Z)",
    },
    "barometric_pressure": {
        "fn": fetch_barometric_pressure,
        "interval_min": 15,
        "description": "NWS station barometric pressure",
    },
    "tidal": {
        "fn": fetch_tidal_data,
        "interval_min": 15,
        "description": "NOAA tide gauge water levels",
    },
    "xray_flares": {
        "fn": fetch_xray_flares,
        "interval_min": 15,
        "description": "NOAA GOES X-ray flare list",
    },
    "electron_flux": {
        "fn": fetch_electron_flux,
        "interval_min": 15,
        "description": "GOES integral electron flux",
    },
    "proton_flux": {
        "fn": fetch_proton_flux,
        "interval_min": 15,
        "description": "GOES integral proton flux",
    },
    "goes_magnetometer": {
        "fn": fetch_goes_magnetometer,
        "interval_min": 5,
        "description": "GOES satellite magnetometer (He, Hp, Hn, total)",
    },
    "intermagnet": {
        "fn": fetch_intermagnet,
        "interval_min": 15,
        "description": "INTERMAGNET global observatory network (10 stations, XYZ+F)",
    },
    "firms_nrt": {
        "fn": fetch_firms_nrt,
        "interval_min": 30,
        "description": "NASA FIRMS satellite thermal anomalies near active volcanoes",
    },
    "goes_xray_flux": {
        "fn": fetch_goes_xray_flux,
        "interval_min": 5,
        "description": "GOES X-ray flux continuous 1-min measurements",
    },
    "olr_recent": {
        "fn": fetch_olr_recent,
        "interval_min": 360,
        "description": "NOAA OLR daily means for seismic zones (last 30 days)",
    },
    "grace_recent": {
        "fn": fetch_grace_recent,
        "interval_min": 1440,
        "description": "GRACE-FO monthly gravity anomalies for seismic zones",
    },
}
