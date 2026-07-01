"""QuakeWatch — Server with real-time ingestion, prediction, and streaming dashboard."""

import asyncio
import json as json_mod
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

load_dotenv()

import threading
from ingest import QuakeStore, IngestEngine, SOURCE_REGISTRY
from ingest.backfill import run_full_backfill
from predict import QuakePredictor, scan_hotspots
from threat import ThreatDetector
from alerts import AlertScanner
from seedlink import SeedLinkMonitor
from prediction_engine import PredictionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("quakewatch")

UTC = timezone.utc
store = QuakeStore()
engine = IngestEngine(store)
predictor = None
threat_detector = None
seedlink_monitor = None
prediction_engine = None

DASHBOARD_SIGNALS = [
    ("noaa_swpc", "kp_index", "Kp Index", ""),
    ("noaa_swpc", "dst_index", "Dst Index", "nT"),
    ("noaa_swpc", "solar_wind_speed", "Solar Wind", "km/s"),
]

app = FastAPI(title="QuakeWatch")
dashboard_dir = Path(__file__).parent / "dashboard"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (dashboard_dir / "static" / "index.html").read_text()


@app.get("/api/stats")
async def api_stats():
    return store.get_stats()


@app.get("/api/config")
async def api_config():
    return {"maptiler_key": os.getenv("MAPTILER_API_KEY", "")}


@app.get("/api/earthquakes")
async def api_earthquakes(
    start: str = Query(None), end: str = Query(None),
    min_mag: float = Query(None), limit: int = Query(500),
):
    return store.query_earthquakes(start=start, end=end, min_mag=min_mag, limit=limit)


@app.get("/api/event-context")
async def api_event_context(
    lat: float = Query(...), lon: float = Query(...),
    time: str = Query(...), mag: float = Query(None),
):
    """Shareable event context: foreshocks, aftershocks, conditions snapshot, DART."""
    return await asyncio.to_thread(lambda: _build_event_context(lat, lon, time, mag))


def _build_event_context(lat, lon, time_str, mag):
    import sqlite3
    from datetime import datetime, timedelta
    try:
        evt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    except Exception:
        return {"error": "invalid time format"}
    evt_naive = evt.replace(tzinfo=None)

    radius = 3.0
    fore_hours, after_hours = 72, 168
    fore_start = (evt_naive - timedelta(hours=fore_hours)).isoformat()
    after_end = (evt_naive + timedelta(hours=after_hours)).isoformat()

    conn = sqlite3.connect(str(store.db_path), timeout=30)
    conn.row_factory = sqlite3.Row

    # Foreshocks + aftershocks in spatial window
    nearby = conn.execute("""
        SELECT id, timestamp, magnitude, depth_km, lat, lon, place
        FROM earthquakes
        WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """, (lat - radius, lat + radius, lon - radius, lon + radius,
          fore_start, after_end)).fetchall()

    evt_ts = evt_naive.isoformat()
    foreshocks = [dict(r) for r in nearby if r['timestamp'] < evt_ts]
    aftershocks = [dict(r) for r in nearby if r['timestamp'] > evt_ts]

    # Conditions: Kp, Dst, solar wind, IMF Bz around event (±24h window)
    cond_start = (evt_naive - timedelta(hours=24)).isoformat()
    cond_end = (evt_naive + timedelta(hours=24)).isoformat()
    MAX_SPARK = 200
    conditions = {}
    for metric, label in [('kp_index', 'kp'), ('dst_index', 'dst'),
                          ('solar_wind_speed', 'solar_wind'),
                          ('solar_wind_density', 'sw_density'),
                          ('imf_bz', 'imf_bz')]:
        rows = conn.execute("""
            SELECT timestamp, value FROM samples
            WHERE metric = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (metric, cond_start, cond_end)).fetchall()
        vals = [dict(r) for r in rows]
        if vals:
            values = [v['value'] for v in vals]
            at_idx = min(range(len(vals)),
                key=lambda i: abs(datetime.fromisoformat(
                    vals[i]['timestamp']).timestamp() - evt_naive.timestamp()))
            at_event_val = vals[at_idx]['value']
            if len(vals) > MAX_SPARK:
                step = len(vals) / MAX_SPARK
                sampled = [vals[int(i * step)] for i in range(MAX_SPARK)]
                ds_event_idx = round(at_idx / step)
                ds_event_idx = min(ds_event_idx, len(sampled) - 1)
            else:
                sampled = vals
                ds_event_idx = at_idx
            conditions[label] = {
                'values': sampled,
                'event_idx': ds_event_idx,
                'min': round(min(values), 2),
                'max': round(max(values), 2),
                'mean': round(sum(values) / len(values), 2),
                'at_event': at_event_val,
            }

    # GOES X-ray (solar flare indicator)
    xray = conn.execute("""
        SELECT timestamp, value FROM samples
        WHERE metric LIKE 'goes_xray_flux%' AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC LIMIT 1
    """, (cond_start, cond_end)).fetchone()
    if xray:
        conditions['xray_flux'] = {'at_event': xray['value'], 'timestamp': xray['timestamp']}

    # Nearest DART buoys (within 15° of event)
    dart_info = []
    try:
        dart_stations = conn.execute("""
            SELECT station_id, lat, lon, region FROM dart_stations
            WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
        """, (lat - 15, lat + 15, lon - 15, lon + 15)).fetchall()
        for ds in dart_stations[:5]:
            reading = conn.execute("""
                SELECT height_m, mode, timestamp FROM dart_readings
                WHERE station_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
            """, (ds['station_id'], after_end)).fetchone()
            dart_info.append({
                'station_id': ds['station_id'], 'lat': ds['lat'], 'lon': ds['lon'],
                'region': ds['region'],
                'latest': dict(reading) if reading else None,
            })
    except Exception:
        pass

    # Active zone context
    zone_context = None
    try:
        import json as _json
        with open(str(store.db_path).replace('quakewatch.db', 'tier2_watch.json')) as f:
            t2 = _json.load(f)
        for w in t2.get('watch', []):
            c = w.get('centroid', [0, 0])
            if abs(c[0] - lat) < 5 and abs(c[1] - lon) < 5:
                zone_context = {
                    'zone': w.get('zone'), 'alert_level': w.get('alert_level'),
                    'escalation_prob': w.get('escalation_prob_72h'),
                    'n_quakes': w.get('n_recent_quakes'),
                    'model_skill': w.get('model_skill'),
                }
                break
    except Exception:
        pass

    conn.close()
    return {
        'event': {'lat': lat, 'lon': lon, 'time': time_str, 'mag': mag},
        'foreshocks': foreshocks[-50:],
        'aftershocks': aftershocks[:50],
        'conditions': conditions,
        'dart': dart_info,
        'zone': zone_context,
    }


@app.get("/api/samples")
async def api_samples(
    source: str = Query(None), metric: str = Query(None),
    start: str = Query(None), end: str = Query(None),
    limit: int = Query(1000), bucket: int = Query(None),
):
    return store.query_samples(source=source, metric=metric, start=start, end=end, limit=limit, bucket_minutes=bucket)


@app.get("/api/sources")
async def api_sources():
    return {
        name: {
            "description": info["description"],
            "interval_min": info["interval_min"],
            "last_fetch": engine._last_fetch.get(name),
        }
        for name, info in SOURCE_REGISTRY.items()
    }


@app.get("/api/ingest/{source}")
async def api_ingest_source(source: str):
    result = engine.fetch_source(source)
    return result


@app.get("/api/ingest")
async def api_ingest_all():
    results = engine.fetch_all()
    return {"results": results}


backfill_status = {"running": False, "results": None}


@app.get("/api/backfill")
async def api_backfill(start_year: int = Query(2015)):
    if backfill_status["running"]:
        return {"status": "already_running"}

    def _run():
        backfill_status["running"] = True
        try:
            backfill_status["results"] = run_full_backfill(store, start_year=start_year)
        finally:
            backfill_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "start_year": start_year}


@app.get("/api/backfill/status")
async def api_backfill_status():
    return backfill_status


def _ensure_predictor():
    global predictor
    if predictor is None:
        predictor = QuakePredictor(store)
    return predictor


@app.get("/api/predict")
async def api_predict(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
):
    p = _ensure_predictor()
    return await asyncio.to_thread(p.predict_heuristic, lat, lon)


@app.get("/api/predict/faults")
async def api_predict_faults():
    p = _ensure_predictor()
    return await asyncio.to_thread(p.score_fault_segments)


@app.get("/api/predict/hotspots")
async def api_predict_hotspots():
    _ensure_predictor()
    return await asyncio.to_thread(scan_hotspots, store)


@app.get("/api/predict/grid")
async def api_predict_grid(
    step: float = Query(15.0, description="Grid step size in degrees"),
):
    p = _ensure_predictor()
    return await asyncio.to_thread(p.predict_grid, step=step)


def _ensure_threat_detector():
    global threat_detector
    if threat_detector is None:
        threat_detector = ThreatDetector(store)
    return threat_detector


@app.get("/api/threats")
async def api_threats():
    td = _ensure_threat_detector()
    return await asyncio.to_thread(td.scan)


@app.get("/api/seedlink/stations")
async def api_seedlink_stations():
    if seedlink_monitor:
        return seedlink_monitor.get_station_status()
    return []


@app.get("/api/dart/status")
async def api_dart_status():
    return await asyncio.to_thread(_fetch_dart_status)


def _fetch_dart_status():
    import sqlite3
    from lab.dart_ingest import DART_STATIONS, detect_mode_transitions
    conn = sqlite3.connect(str(store.db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        alerts = detect_mode_transitions(conn)
        stations = []
        for sid, info in DART_STATIONS.items():
            row = conn.execute(
                "SELECT timestamp, mode, height_m FROM dart_readings "
                "WHERE station_id = ? ORDER BY timestamp DESC LIMIT 1",
                (sid,)
            ).fetchone()
            if not row:
                continue
            mode_name = {1: "normal", 2: "event", 3: "tsunami"}.get(row["mode"], "unknown")

            # Compute 24h height deviation for signal strength
            deviation = 0.0
            stats = conn.execute(
                "SELECT AVG(height_m) as avg_h, "
                "       AVG(height_m * height_m) - AVG(height_m) * AVG(height_m) as var_h "
                "FROM dart_readings WHERE station_id = ? "
                "AND timestamp >= datetime(?, '-24 hours')",
                (sid, row["timestamp"])
            ).fetchone()
            if stats and stats["avg_h"] and row["height_m"]:
                std_h = (stats["var_h"] ** 0.5) if stats["var_h"] and stats["var_h"] > 0 else 0.001
                deviation = abs(row["height_m"] - stats["avg_h"]) / max(0.001, std_h)

            stations.append({
                "station_id": sid,
                "lat": info["lat"],
                "lon": info["lon"],
                "region": info.get("region", ""),
                "mode": mode_name,
                "last_reading": row["timestamp"],
                "height_m": round(row["height_m"], 3) if row["height_m"] else None,
                "deviation": round(deviation, 2),
            })
        return {"stations": stations, "alerts": alerts}
    finally:
        conn.close()


@app.get("/api/ionosphere/tec")
async def api_ionosphere_tec():
    """Latest global ionospheric TEC field (NOAA SWPC GloTEC) as a GeoJSON grid for the
    map heatmap overlay. 5184 cells @ 2.5x5 deg; properties: tec, anomaly, hmF2, NmF2."""
    return await asyncio.to_thread(_fetch_ionosphere_tec)


def _fetch_ionosphere_tec():
    import os, json as _json
    from ingest.sources import _GLOTEC_CACHE, fetch_glotec
    if not os.path.exists(_GLOTEC_CACHE):
        try:
            fetch_glotec()  # populate on first hit if the poller hasn't run yet
        except Exception:
            pass
    try:
        with open(_GLOTEC_CACHE) as f:
            return _json.load(f)
    except Exception:
        return {"type": "FeatureCollection", "features": [], "time_tag": None,
                "error": "GloTEC data not available yet"}


# ── Weather overlays (free NOAA/NWS + RainViewer; display-only, not model inputs) ──
_WX_CACHE = {}


def _wx_get(url, ttl, transform=None):
    import urllib.request, json as _json, time as _t
    now = _t.time(); c = _WX_CACHE.get(url)
    if c and now - c[0] < ttl:
        return c[1]
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "seismic-lab dashboard (ryan@axomlabs.ai)", "Accept": "application/geo+json"})
        data = _json.loads(urllib.request.urlopen(req, timeout=30).read())
        if transform:
            data = transform(data)
        _WX_CACHE[url] = (now, data)
        return data
    except Exception as e:
        return c[1] if c else {"error": str(e)}


@app.get("/api/weather/alerts")
async def api_weather_alerts():
    """Active severe-weather warnings/watches (NWS) as GeoJSON polygons — tornado,
    severe thunderstorm, hurricane/tropical, flash flood, storm surge. US coverage."""
    return await asyncio.to_thread(_weather_alerts)


def _weather_alerts():
    KEEP = ("Tornado", "Severe Thunderstorm", "Hurricane", "Tropical", "Flash Flood", "Storm Surge")
    def tx(j):
        feats = []
        for f in j.get("features", []):
            ev = (f.get("properties") or {}).get("event", "")
            if f.get("geometry") and any(k in ev for k in KEEP):
                p = f["properties"]
                feats.append({"type": "Feature", "geometry": f["geometry"],
                              "properties": {k: p.get(k) for k in
                                  ("event", "severity", "certainty", "urgency", "headline", "areaDesc", "ends")}})
        return {"type": "FeatureCollection", "generated": j.get("updated"), "features": feats}
    return _wx_get("https://api.weather.gov/alerts/active?status=actual", 120, tx)


@app.get("/api/weather/hurricanes")
async def api_weather_hurricanes():
    """Active tropical cyclones (NHC) — positions/intensity, Atlantic + E. Pacific."""
    return await asyncio.to_thread(lambda: _wx_get("https://www.nhc.noaa.gov/CurrentStorms.json", 600))


_NHC_TW = ("https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/"
           "NHC_tropical_weather_summary/MapServer")


@app.get("/api/weather/hurricane-cones")
async def api_weather_hurricane_cones():
    """NHC forecast cone-of-uncertainty + forecast track + forecast points as GeoJSON
    for any active tropical cyclones (empty out of season). The classic cone overlay."""
    return await asyncio.to_thread(_hurricane_cones)


def _hurricane_cones():
    def layer(n):
        return _wx_get(f"{_NHC_TW}/{n}/query?where=1%3D1&outFields=*&returnGeometry=true&f=geojson", 600)
    return {"cone": layer(7), "track": layer(6), "points": layer(5)}


@app.get("/api/weather/outlook")
async def api_weather_outlook():
    """SPC day-1 convective outlook — severe-weather risk polygons (US)."""
    return await asyncio.to_thread(lambda: _wx_get(
        "https://www.spc.noaa.gov/products/outlook/day1otlk_cat.lyr.geojson", 1800))


@app.get("/api/weather/radar")
async def api_weather_radar():
    """RainViewer global radar tile manifest — the FE builds tile URLs from host+frames."""
    return await asyncio.to_thread(lambda: _wx_get("https://api.rainviewer.com/public/weather-maps.json", 300))


@app.get("/api/volcanic/activity")
async def api_volcanic_activity():
    return await asyncio.to_thread(_fetch_volcanic_activity)


def _fetch_volcanic_activity():
    import sqlite3
    from datetime import datetime
    conn = sqlite3.connect(str(store.db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    cur_year = datetime.utcnow().year
    try:
        # USGS HANS authoritative alert levels (HANS vnum == GVP volcano_number)
        usgs_alerts = {}
        try:
            for a in conn.execute("SELECT vnum, alert_level FROM volcano_alerts").fetchall():
                usgs_alerts[str(a["vnum"])] = a["alert_level"]
        except Exception:
            pass

        # Source 1: FIRMS thermal hotspots ON the edifice (<= 15 km), last 2 days, FRP >= 15.
        # The distance cap is essential: without it, a distant wildfire is attributed to the
        # nearest volcano (e.g. a fire 90 km away flagged dormant Mt St Helens as "high").
        # And FIRMS thermal alone is NOT authoritative — only USGS can call it "high".
        rows = conn.execute("""
            SELECT v.volcano_number, v.name, v.lat, v.lon, v.elevation_m,
                   v.volcano_type, v.region,
                   COUNT(t.id) as hotspot_count,
                   MAX(t.brightness) as max_brightness,
                   MAX(t.frp) as max_frp,
                   ROUND(AVG(t.brightness), 1) as avg_brightness,
                   MAX(t.acq_date) as last_detection,
                   ROUND(MIN(t.nearest_volcano_dist_km), 1) as nearest_km
            FROM volcanoes v
            JOIN thermal_anomalies t ON t.nearest_volcano_id = v.volcano_number
            WHERE t.acq_date >= date('now', '-2 days')
              AND t.nearest_volcano_dist_km <= 15
            GROUP BY v.volcano_number
            HAVING max_frp >= 15
            ORDER BY max_frp DESC
        """).fetchall()

        volcanoes = []
        seen_ids = set()
        for r in rows:
            frp = r["max_frp"] or 0
            usgs = usgs_alerts.get(str(r["volcano_number"]))
            # USGS is authoritative for "high"; on-edifice FIRMS thermal alone is "elevated"
            # at most (could still be a flank fire — flagged via usgs_alert/source).
            if usgs in ("WARNING", "WATCH"):
                level = "high"
            elif usgs == "ADVISORY" or frp >= 15:
                level = "elevated"
            else:
                continue
            seen_ids.add(r["volcano_number"])
            volcanoes.append({
                "id": r["volcano_number"],
                "name": r["name"],
                "lat": r["lat"],
                "lon": r["lon"],
                "elevation_m": r["elevation_m"],
                "type": r["volcano_type"],
                "region": r["region"],
                "hotspot_count": r["hotspot_count"],
                "max_brightness": round(r["max_brightness"] or 0, 1),
                "max_frp": round(frp, 1),
                "avg_brightness": r["avg_brightness"],
                "last_detection": r["last_detection"],
                "thermal_dist_km": r["nearest_km"],
                "usgs_alert": usgs,
                "source": "USGS+FIRMS" if usgs else "FIRMS thermal (unconfirmed)",
                "level": level,
            })

        # Source 2: GVP-confirmed active volcanoes (current year or continuing eruption)
        gvp_rows = conn.execute("""
            SELECT DISTINCT v.volcano_number, v.name, v.lat, v.lon, v.elevation_m,
                   v.volcano_type, v.region
            FROM volcanoes v
            WHERE v.last_eruption_year >= ?
            UNION
            SELECT DISTINCT v.volcano_number, v.name, v.lat, v.lon, v.elevation_m,
                   v.volcano_type, v.region
            FROM volcanoes v
            JOIN eruptions e ON e.volcano_number = v.volcano_number
            WHERE e.continuing = 1
        """, (cur_year,)).fetchall()

        for r in gvp_rows:
            if r["volcano_number"] in seen_ids:
                continue
            seen_ids.add(r["volcano_number"])
            # Check for any recent thermal data (extended 30-day window)
            thermal = conn.execute("""
                SELECT COUNT(*) as cnt, MAX(t.frp) as max_frp,
                       MAX(t.brightness) as max_brightness,
                       ROUND(AVG(t.brightness), 1) as avg_brightness,
                       MAX(t.acq_date) as last_detection
                FROM thermal_anomalies t
                WHERE t.nearest_volcano_id = ?
                AND t.acq_date >= date('now', '-30 days')
            """, (r["volcano_number"],)).fetchone()

            frp = (thermal["max_frp"] or 0) if thermal else 0
            usgs = usgs_alerts.get(str(r["volcano_number"]))
            # USGS alert promotes a GVP-active volcano above the baseline "active"
            level = ("high" if usgs in ("WARNING", "WATCH")
                     else "elevated" if usgs == "ADVISORY" else "active")
            volcanoes.append({
                "id": r["volcano_number"],
                "name": r["name"],
                "lat": r["lat"],
                "lon": r["lon"],
                "elevation_m": r["elevation_m"],
                "type": r["volcano_type"],
                "region": r["region"],
                "hotspot_count": (thermal["cnt"] or 0) if thermal else 0,
                "max_brightness": round((thermal["max_brightness"] or 0), 1) if thermal else 0,
                "max_frp": round(frp, 1),
                "avg_brightness": (thermal["avg_brightness"] or 0) if thermal else 0,
                "last_detection": (thermal["last_detection"] or "GVP") if thermal else "GVP",
                "usgs_alert": usgs,
                "level": level,
                "source": "USGS+GVP" if usgs else "gvp",
            })

        volcanoes.sort(key=lambda v: (
            {"high": 0, "elevated": 1, "active": 2}.get(v["level"], 3),
            -(v["max_frp"] or 0)
        ))
        return {"volcanoes": volcanoes, "count": len(volcanoes)}
    finally:
        conn.close()


_tidal_cache = None
_tidal_cache_time = 0
_TIDAL_CACHE_TTL = 120

@app.get("/api/tidal/sensitivity")
async def api_tidal_sensitivity():
    return await asyncio.to_thread(_fetch_tidal_sensitivity)


_sf_ts = None
_sf_eph = None

def _get_skyfield():
    global _sf_ts, _sf_eph
    if _sf_ts is None:
        from skyfield.api import load as sf_load
        _sf_ts = sf_load.timescale()
        _sf_eph = sf_load('de421.bsp')
    return _sf_ts, _sf_eph

_GM_MOON = 4.902800e12
_GM_SUN  = 1.327124e20
_A_EARTH = 6.3781e6


def _realtime_tidal_v2(zone_latlons, utc_dt):
    """Compute degree-2 tidal potential at each zone from Moon+Sun ephemeris."""
    import math
    ts, eph = _get_skyfield()
    t = ts.from_datetime(utc_dt)
    earth = eph['earth']

    bodies = []
    for name, gm in [('moon', _GM_MOON), ('sun', _GM_SUN)]:
        ra, dec, dist = earth.at(t).observe(eph[name]).apparent().radec()
        bodies.append((gm, ra.radians, dec.radians, dist.m))

    gast_hours = t.gast
    results = []

    for lat_deg, lon_deg in zone_latlons:
        phi = math.radians(lat_deg)
        lst_rad = math.radians(((gast_hours + lon_deg / 15.0) % 24.0) * 15.0)

        v2 = 0.0
        v2_max = 0.0
        for gm, alpha, delta, R in bodies:
            H = lst_rad - alpha
            cos_z = (math.sin(phi) * math.sin(delta) +
                     math.cos(phi) * math.cos(delta) * math.cos(H))
            P2 = (3 * cos_z**2 - 1) / 2
            v2 += gm * _A_EARTH**2 / R**3 * P2
            v2_max += gm * _A_EARTH**2 / R**3
        stress = min(1.0, abs(v2) / v2_max) if v2_max > 0 else 0
        results.append((v2, v2_max, stress))

    return results


def _fetch_tidal_sensitivity():
    import sqlite3, math, time
    from datetime import datetime, timedelta, timezone
    global _tidal_cache, _tidal_cache_time
    now_mono = time.monotonic()
    if _tidal_cache is not None and (now_mono - _tidal_cache_time) < _TIDAL_CACHE_TTL:
        return _tidal_cache
    UTC = timezone.utc

    ZONES = [
        {"id": "socal",       "name": "S. California",     "lat": [31, 36],  "lon": [-121, -114], "clat": 33.5, "clon": -117.5},
        {"id": "norcal",      "name": "N. California",     "lat": [36, 42],  "lon": [-126, -119], "clat": 39.0, "clon": -122.5},
        {"id": "cascadia",    "name": "Cascadia",          "lat": [42, 50],  "lon": [-131, -119], "clat": 46.0, "clon": -124.0},
        {"id": "alaska",      "name": "Alaska",            "lat": [50, 65],  "lon": [-180, -130], "clat": 58.0, "clon": -153.0},
        {"id": "japan",       "name": "Japan",             "lat": [28, 46],  "lon": [128, 148],   "clat": 36.0, "clon": 140.0},
        {"id": "indonesia",   "name": "Indonesia",         "lat": [-12, 8],  "lon": [94, 136],    "clat": -2.0, "clon": 115.0},
        {"id": "chile",       "name": "Chile / Peru",      "lat": [-46, 2],  "lon": [-82, -65],   "clat": -22.0,"clon": -70.0},
        {"id": "mexico",      "name": "Mexico / C. America","lat": [7, 25],  "lon": [-115, -77],  "clat": 16.0, "clon": -99.0},
        {"id": "caribbean",   "name": "Caribbean",         "lat": [10, 22],  "lon": [-85, -60],   "clat": 16.0, "clon": -70.0},
        {"id": "turkey",      "name": "Mediterranean",     "lat": [33, 42],  "lon": [-6, 45],     "clat": 37.5, "clon": 28.0},
        {"id": "philippines", "name": "Philippines",       "lat": [4, 26],   "lon": [118, 128],   "clat": 14.0, "clon": 122.0},
        {"id": "nz",          "name": "New Zealand",       "lat": [-48, -34],"lon": [165, 180],   "clat": -42.0,"clon": 173.0},
        {"id": "himalaya",    "name": "Himalayas",         "lat": [24, 40],  "lon": [65, 100],    "clat": 30.0, "clon": 82.0},
    ]

    now = datetime.now(UTC)
    conn = sqlite3.connect(str(store.db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    # Real-time tidal V2 from JPL ephemeris (Moon + Sun)
    zone_latlons = [(z["clat"], z["clon"]) for z in ZONES]
    tidal_states = _realtime_tidal_v2(zone_latlons, now)

    # Historical tidal data for Schuster test phase computation
    import numpy as np
    tidal_cutoff = (now - timedelta(days=35)).isoformat()
    now_iso = now.isoformat()

    station_locs = conn.execute(
        "SELECT DISTINCT lat, lon FROM samples WHERE metric = 'tidal_potential'"
    ).fetchall()
    tidal_by_station = {}
    for slat, slon in station_locs:
        rows = conn.execute(
            "SELECT timestamp, value FROM samples "
            "WHERE metric = 'tidal_potential' AND timestamp >= ? AND timestamp <= ? "
            "AND lat = ? AND lon = ? ORDER BY timestamp",
            (tidal_cutoff, now_iso, slat, slon)
        ).fetchall()
        if rows:
            tidal_by_station[(slat, slon)] = rows

    zones = []

    for i, z in enumerate(ZONES):
        clat, clon = z["clat"], z["clon"]
        v2_now, v2_max, tidal_stress = tidal_states[i]

        # Load nearest station's historical data for Schuster test
        best_key = min(
            tidal_by_station.keys(),
            key=lambda s: (s[0] - clat) ** 2 + (s[1] - clon) ** 2,
            default=None
        ) if tidal_by_station else None
        tidal_rows = tidal_by_station.get(best_key, []) if best_key else []

        tidal_times, tidal_values = [], []
        for ts, val in tidal_rows:
            try:
                t = datetime.fromisoformat(ts)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=UTC)
                tidal_times.append(t.timestamp())
                tidal_values.append(val)
            except:
                continue

        # Schuster test on 30-day nearby seismicity
        t30 = (now - timedelta(days=30)).isoformat()
        rows = conn.execute(
            "SELECT timestamp FROM earthquakes "
            "WHERE timestamp >= ? AND magnitude >= 2.5 "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (t30, z["lat"][0], z["lat"][1], z["lon"][0], z["lon"][1])
        ).fetchall()

        phases = []
        t_arr = np.array(tidal_times) if tidal_times else np.array([])
        v_arr = np.array(tidal_values) if tidal_values else np.array([])

        for (ts,) in rows:
            try:
                t = datetime.fromisoformat(ts)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=UTC)
                epoch = t.timestamp()
                idx = np.searchsorted(t_arr, epoch)
                if idx <= 0 or idx >= len(v_arr):
                    continue
                v0, v1 = v_arr[idx - 1], v_arr[idx]
                deriv = v1 - v0
                frac = (epoch - t_arr[idx - 1]) / (t_arr[idx] - t_arr[idx - 1]) if t_arr[idx] != t_arr[idx - 1] else 0.5
                val = v0 + frac * (v1 - v0)
                phases.append(math.atan2(deriv, val))
            except:
                continue

        n = len(phases)
        if n >= 10:
            cos_sum = sum(math.cos(p) for p in phases)
            sin_sum = sum(math.sin(p) for p in phases)
            D_sq = (cos_sum ** 2 + sin_sum ** 2) / n
            p_val = math.exp(-D_sq)
            R = math.sqrt(cos_sum ** 2 + sin_sum ** 2) / n
        else:
            p_val, R = 1.0, 0.0

        # Threat = sensitivity (R) × real-time tidal stress at this zone
        if p_val < 0.05 and R > 0.02:
            threat = R * tidal_stress
        else:
            threat = 0.0

        zones.append({
            "id": z["id"],
            "name": z["name"],
            "lat": z["clat"],
            "lon": z["clon"],
            "p_value": round(p_val, 6),
            "R": round(R, 6),
            "n_quakes": n,
            "significant": p_val < 0.05,
            "tidal_stress": round(tidal_stress, 4),
            "v2": round(v2_now, 4),
            "threat": round(threat, 6),
        })

    conn.close()
    zones.sort(key=lambda z: z["p_value"])
    result = {"zones": zones}
    _tidal_cache = result
    _tidal_cache_time = now_mono
    return result


@app.get("/api/event-analyses")
async def api_event_analyses(
    hours: int = Query(72),
    min_risk: float = Query(0.0),
):
    from event_analyzer import EventAnalyzer
    analyzer = EventAnalyzer(store.db_path)
    return await asyncio.to_thread(analyzer.get_recent, hours, min_risk)


@app.get("/api/event-analyses/{earthquake_id}/llm")
async def api_event_llm_analysis(earthquake_id: str):
    from llm_analyzer import LLMAnalyzer
    llm = LLMAnalyzer(store.db_path)
    text = await asyncio.to_thread(llm.get_analysis, earthquake_id)
    if text is None:
        return JSONResponse({"error": "No LLM analysis found"}, status_code=404)
    return {"earthquake_id": earthquake_id, "analysis": text}


@app.get("/api/event-analyses/llm/run/{earthquake_id}")
async def api_event_llm_run(earthquake_id: str):
    """Trigger LLM analysis for a specific event on demand."""
    from event_analyzer import EventAnalyzer
    from llm_analyzer import LLMAnalyzer
    analyzer = EventAnalyzer(store.db_path)
    conn = analyzer._conn()
    try:
        row = conn.execute(
            "SELECT * FROM event_analyses WHERE earthquake_id = ?",
            (earthquake_id,)
        ).fetchone()
        if not row:
            return JSONResponse({"error": "Event analysis not found"}, status_code=404)
        analysis = dict(row)
    finally:
        conn.close()
    llm = LLMAnalyzer(store.db_path)
    result = await asyncio.to_thread(llm.analyze, analysis)
    if result is None:
        return JSONResponse({"error": "LLM analysis skipped or failed"}, status_code=500)
    return {"earthquake_id": earthquake_id, "analysis": result}


@app.get("/api/seismicity/summary")
async def api_seismicity_summary():
    return await asyncio.to_thread(_fetch_seismicity_summary)


def _fetch_seismicity_summary():
    import sqlite3
    conn = sqlite3.connect(str(store.db_path), timeout=30)
    try:
        totals = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN magnitude >= 4.5 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN magnitude >= 5.0 THEN 1 ELSE 0 END), MAX(magnitude) "
            "FROM earthquakes WHERE timestamp >= datetime('now', '-24 hours')"
        ).fetchone()
        bins = conn.execute(
            "SELECT CAST(strftime('%s', timestamp) AS INTEGER) / 3600 AS hr, COUNT(*) "
            "FROM earthquakes WHERE timestamp >= datetime('now', '-24 hours') "
            "GROUP BY hr ORDER BY hr"
        ).fetchall()
        hourly = [r[1] for r in bins] if bins else []
        return {
            "total_24h": totals[0] or 0,
            "m45_24h": totals[1] or 0,
            "m50_24h": totals[2] or 0,
            "max_mag_24h": round(totals[3], 1) if totals[3] else 0,
            "hourly_counts": hourly,
        }
    finally:
        conn.close()


@app.get("/api/seedlink/detections")
async def api_seedlink_detections(limit: int = Query(20)):
    if seedlink_monitor:
        return seedlink_monitor.get_recent_detections(limit)
    return []


@app.get("/api/seedlink/waveform/{station:path}")
async def api_seedlink_waveform(station: str, scale: str = Query("30m")):
    if seedlink_monitor:
        wf = seedlink_monitor.get_waveform(station, scale=scale)
        if wf:
            return wf
    return {"station": station, "scale": scale, "mode": "raw", "sample_rate": 0, "samples": [], "timestamp": ""}


@app.get("/api/seedlink/resonance/{station:path}")
async def api_seedlink_resonance(station: str):
    if seedlink_monitor:
        result = seedlink_monitor.get_station_resonance(station)
        if result:
            return result
    return {"station": station, "freqs": [], "power": [], "peak_freq": 0, "peak_power": 0}


@app.websocket("/ws/seismograph/{station:path}")
async def ws_seismograph(websocket: WebSocket, station: str):
    await websocket.accept()
    if not seedlink_monitor:
        await websocket.close(code=1011, reason="SeedLink not running")
        return

    wf = seedlink_monitor._waveforms.get(station)
    if wf:
        await websocket.send_json({
            "type": "init",
            "samples": wf["samples"],
            "sr": wf["sample_rate"],
            "ts": wf["timestamp"],
        })

    q = seedlink_monitor.subscribe(station)
    try:
        while True:
            msg = await q.get()
            await websocket.send_json({"type": "update", **msg})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        seedlink_monitor.unsubscribe(station, q)


@app.get("/api/predictions")
async def api_predictions(limit: int = Query(100)):
    from alerts import PREDICTION_LOG
    if not PREDICTION_LOG.exists():
        return []
    lines = PREDICTION_LOG.read_text().strip().splitlines()
    results = []
    for line in reversed(lines[-limit:]):
        try:
            results.append(json_mod.loads(line))
        except Exception:
            continue
    return results


def _ensure_prediction_engine():
    global prediction_engine
    if prediction_engine is None:
        prediction_engine = PredictionEngine(store)
        prediction_engine.load_models()
    return prediction_engine


@app.get("/api/forecast/active")
async def api_forecast_active():
    pe = _ensure_prediction_engine()
    return await asyncio.to_thread(pe.get_active_predictions)


@app.get("/api/forecast/run")
async def api_forecast_run():
    pe = _ensure_prediction_engine()
    td = _ensure_threat_detector()
    threat_data = await asyncio.to_thread(td.scan)
    active = await asyncio.to_thread(pe.issue_predictions, threat_data)
    resolved = await asyncio.to_thread(pe.resolve_predictions)
    for p in active:
        td._ml_probabilities[p["zone_id"]] = p["probability"]
    td.invalidate_cache()
    return {"active": active, "resolved": resolved}


@app.get("/api/forecast/history")
async def api_forecast_history(limit: int = Query(50)):
    pe = _ensure_prediction_engine()
    return await asyncio.to_thread(pe.get_prediction_history, limit)


@app.get("/api/forecast/metrics")
async def api_forecast_metrics():
    pe = _ensure_prediction_engine()
    return await asyncio.to_thread(pe.get_metrics)


_explain_cache = {}
_explain_cache_time = {}
_EXPLAIN_CACHE_TTL = 600

@app.get("/api/forecast/{zone_id}/explain")
async def api_forecast_explain(zone_id: str):
    now = time.monotonic()
    if zone_id in _explain_cache and (now - _explain_cache_time.get(zone_id, 0)) < _EXPLAIN_CACHE_TTL:
        return _explain_cache[zone_id]

    pe = _ensure_prediction_engine()
    td = _ensure_threat_detector()
    threat_data = await asyncio.to_thread(td.scan)
    result = await asyncio.to_thread(pe.explain_prediction, zone_id, threat_data)
    if result is None:
        return {"error": "Zone not found or prediction unavailable"}

    _explain_cache[zone_id] = result
    _explain_cache_time[zone_id] = time.monotonic()
    return result


@app.get("/api/tier2-watch")
async def api_tier2_watch():
    watch_path = Path(__file__).parent / "data" / "tier2_watch.json"
    if not watch_path.exists():
        return JSONResponse(content={"generated": None, "base_rate_72h": 0, "n_active_swarms": 0, "watch": []})
    return JSONResponse(content=json_mod.loads(watch_path.read_text()))


@app.get("/api/signals/latest")
async def api_signals_latest():
    results = {}
    for source, metric, label, unit in DASHBOARD_SIGNALS:
        samples = store.query_samples(source=source, metric=metric, limit=1)
        if samples:
            s = samples[0]
            results[f"{source}__{metric}"] = {
                "label": label,
                "value": s["value"],
                "timestamp": s["timestamp"],
                "unit": unit or s.get("unit", ""),
            }
    return results


@app.get("/api/signals/history")
async def api_signals_history(hours: int = Query(24)):
    start = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    results = {}
    for source, metric, label, unit in DASHBOARD_SIGNALS:
        samples = store.query_samples(source=source, metric=metric, start=start, limit=2000)
        samples.reverse()
        deduped = []
        for s in samples:
            if not deduped or deduped[-1]["v"] != s["value"]:
                deduped.append({"t": s["timestamp"], "v": s["value"]})
        results[f"{source}__{metric}"] = {
            "label": label,
            "unit": unit,
            "data": deduped,
        }
    return results


def _fetch_stream_quakes():
    recent = store.query_earthquakes(limit=100, min_mag=2.5)
    seven_days = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    significant = store.query_earthquakes(min_mag=4.5, start=seven_days, limit=100)
    seen = {q["id"] for q in recent}
    for q in significant:
        if q["id"] not in seen:
            recent.append(q)
            seen.add(q["id"])
    return recent


def _fetch_stream_signals():
    signals = {}
    for source, metric, label, unit in DASHBOARD_SIGNALS:
        samples = store.query_samples(source=source, metric=metric, limit=1)
        if samples:
            s = samples[0]
            signals[f"{source}__{metric}"] = {
                "label": label,
                "value": s["value"],
                "timestamp": s["timestamp"],
                "unit": unit or s.get("unit", ""),
            }
    return signals


def _fetch_hotspots():
    return scan_hotspots(store)


@app.get("/api/stream")
async def stream_events():
    async def generate():
        yield "event: ping\ndata: {}\n\n"
        cycle = 0
        while True:
            try:
                # Core data — fetch in parallel
                core_tasks = [
                    asyncio.to_thread(_fetch_stream_quakes),
                    asyncio.to_thread(_fetch_stream_signals),
                ]
                if cycle % 2 == 0:
                    td = _ensure_threat_detector()
                    core_tasks.append(asyncio.to_thread(td.scan))

                core_results = await asyncio.gather(*core_tasks, return_exceptions=True)

                quakes = core_results[0] if not isinstance(core_results[0], Exception) else []
                yield f"event: earthquakes\ndata: {json_mod.dumps(quakes)}\n\n"

                if not isinstance(core_results[1], Exception):
                    yield f"event: signals\ndata: {json_mod.dumps(core_results[1])}\n\n"

                if len(core_results) > 2 and not isinstance(core_results[2], Exception):
                    yield f"event: threats\ndata: {json_mod.dumps(core_results[2])}\n\n"

                if seedlink_monitor and cycle % 2 == 1:
                    sl_status = seedlink_monitor.get_station_status()
                    sl_detections = seedlink_monitor.get_recent_detections(5)
                    yield f"event: seedlink\ndata: {json_mod.dumps({'stations': sl_status, 'detections': sl_detections})}\n\n"

                # Heavy secondary data — fetch in parallel
                if cycle % 3 == 0:
                    secondary = await asyncio.gather(
                        asyncio.to_thread(_fetch_dart_status),
                        asyncio.to_thread(_fetch_tidal_sensitivity),
                        asyncio.to_thread(_fetch_volcanic_activity),
                        return_exceptions=True,
                    )
                    if not isinstance(secondary[0], Exception):
                        yield f"event: dart\ndata: {json_mod.dumps(secondary[0])}\n\n"
                    if not isinstance(secondary[1], Exception):
                        yield f"event: tidal\ndata: {json_mod.dumps(secondary[1])}\n\n"
                    if not isinstance(secondary[2], Exception):
                        yield f"event: volcanic\ndata: {json_mod.dumps(secondary[2])}\n\n"

                if cycle % 2 == 0:
                    try:
                        pe = _ensure_prediction_engine()
                        if pe._initialized:
                            td = _ensure_threat_detector()
                            td_data = await asyncio.to_thread(td.scan)
                            await asyncio.to_thread(pe.issue_predictions, td_data)
                            forecast_active, forecast_metrics, _ = await asyncio.gather(
                                asyncio.to_thread(pe.get_active_predictions),
                                asyncio.to_thread(pe.get_metrics),
                                asyncio.to_thread(pe.resolve_predictions),
                            )
                            for p in forecast_active:
                                td._ml_probabilities[p["zone_id"]] = p["probability"]
                            td.invalidate_cache()
                            now_mono = time.monotonic()
                            for p in forecast_active:
                                zid = p.get("zone_id") or p.get("zone")
                                if zid:
                                    try:
                                        result = pe.explain_prediction(zid, td_data)
                                        if result:
                                            _explain_cache[zid] = result
                                            _explain_cache_time[zid] = now_mono
                                    except Exception:
                                        pass
                            yield f"event: forecast\ndata: {json_mod.dumps({'active': forecast_active, 'metrics': forecast_metrics})}\n\n"
                    except Exception as e:
                        log.warning(f"Forecast stream error: {e}")

                if cycle % 6 == 0:
                    global predictor
                    if predictor is None:
                        try:
                            predictor = await asyncio.to_thread(QuakePredictor, store)
                        except Exception:
                            pass
                    if predictor and predictor.model is not None:
                        hotspots = await asyncio.to_thread(_fetch_hotspots)
                        yield f"event: hotspots\ndata: {json_mod.dumps(hotspots)}\n\n"
                        alerts = [h for h in hotspots if h.get("risk_score", 0) > 0.9]
                        if alerts:
                            yield f"event: alerts\ndata: {json_mod.dumps(alerts)}\n\n"

                cycle += 1
            except Exception as e:
                log.error(f"Stream error: {e}")
                yield f"event: error\ndata: {json_mod.dumps({'error': str(e)})}\n\n"

            await asyncio.sleep(15)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


app.mount("/static", StaticFiles(directory=str(dashboard_dir / "static")), name="static")


@app.on_event("startup")
async def startup():
    engine.start()
    td = _ensure_threat_detector()
    scanner = AlertScanner(td)
    asyncio.create_task(scanner.run_forever())

    global seedlink_monitor
    seedlink_monitor = SeedLinkMonitor(store)
    seedlink_monitor.start()

    pe = _ensure_prediction_engine()
    if pe._initialized:
        cached = pe.get_active_predictions()
        for p in cached:
            td._ml_probabilities[p["zone_id"]] = p["probability"]
        log.info(f"Prediction engine loaded — {len(cached)} cached predictions seeded")

        async def _prewarm_explain():
            try:
                td_data = await asyncio.to_thread(td.scan)
                now_mono = time.monotonic()
                for p in cached:
                    zid = p.get("zone_id") or p.get("zone")
                    if zid:
                        try:
                            result = await asyncio.to_thread(pe.explain_prediction, zid, td_data)
                            if result:
                                _explain_cache[zid] = result
                                _explain_cache_time[zid] = now_mono
                        except Exception:
                            pass
                log.info(f"Explain cache pre-warmed: {len(_explain_cache)} zones")
            except Exception as e:
                log.warning(f"Explain pre-warm failed: {e}")
        asyncio.create_task(_prewarm_explain())
    else:
        log.warning("Prediction engine: no models found — run lab/train_predictor.py")

    # Backfill event analyses for recent M4.5+ events
    try:
        from event_analyzer import EventAnalyzer
        analyzer = EventAnalyzer(store.db_path)
        backfilled = await asyncio.to_thread(analyzer.backfill, 72)
        log.info(f"EventAnalyzer backfill: {len(backfilled)} events analyzed")
    except Exception as e:
        log.warning(f"EventAnalyzer backfill failed: {e}")


    log.info("QuakeWatch started — ingestion + alerts + predictions + SeedLink running")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8422, log_level="warning")
