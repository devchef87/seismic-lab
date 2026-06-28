"""SeismicLab — Real-time earthquake prediction engine.

Loads the trained binary classifier and scores current conditions
for any lat/lon to produce P(M5.0+ within 24h).

Includes a heuristic composite scorer that produces meaningful
location-varying risk estimates using:
  - Distance to nearest plate boundary
  - Recent local/regional seismicity
  - Current geomagnetic conditions (Kp, Dst)
"""

import json
import math
import logging
import sqlite3
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ingest.store import QuakeStore
from ingest.features import (
    _extract_seismic_rate_features,
    _extract_signal_features,
    _find_available_signals,
    SEISMIC_RATE_FEATURE_NAMES,
    SIGNAL_METRICS,
    LOOKBACK_HOURS,
)

log = logging.getLogger("seismiclab.predict")
UTC = timezone.utc
DATA_DIR = Path(__file__).parent / "data"

PLATE_BOUNDARY_SEGMENTS = [
    [(-55,-70),(-46,-75),(-38,-73),(-33,-71),(-24,-70),(-18,-70),(-12,-76),(-5,-80),(2,-78)],
    [(2,-78),(8,-83),(14,-93),(17,-100),(20,-106),(23,-110)],
    [(23,-110),(28,-113),(32,-117),(34,-120),(37,-122),(40,-125),(44,-127),(48,-128),(51,-130)],
    [(51,-130),(55,-136),(57,-150),(56,-158),(53,-167),(52,-172),(51,178),(50,172),(50,165)],
    [(50,165),(52,160),(50,157),(46,152),(42,146),(38,142),(35,140),(32,135),(30,132)],
    [(30,132),(27,128),(24,124),(20,122),(16,121),(12,124),(8,126),(4,127),(0,124),(-3,120),(-7,115),(-9,110),(-7,105),(-5,100),(-3,95)],
    [(-46,167),(-42,174),(-38,178),(-34,-178),(-28,-176),(-22,-174),(-16,-173)],
    [(36,-10),(37,-5),(37,0),(38,5),(39,10),(38,15),(37,20),(38,28),(39,35),(38,42),(36,48),(34,52),(32,57)],
    [(32,57),(31,62),(30,67),(30,72),(29,77),(28,82),(27,86),(28,92),(26,95),(22,96)],
    [(65,-18),(62,-22),(55,-30),(45,-28),(35,-35),(25,-45),(15,-46),(5,-32),(0,-14),(-10,-13),(-20,-12),(-30,-14),(-40,-15),(-50,-8),(-55,-5)],
    [(12,42),(8,38),(4,36),(0,33),(-4,30),(-8,32),(-12,34),(-16,35),(-20,35)],
]

def _haversine_deg(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.sqrt((px - ax)**2 + (py - ay)**2)
    t = max(0, min(1, ((px - ax)*dx + (py - ay)*dy) / (dx*dx + dy*dy)))
    return math.sqrt((px - ax - t*dx)**2 + (py - ay - t*dy)**2)

def _min_fault_distance_km(lat, lon):
    min_deg = float('inf')
    for seg in PLATE_BOUNDARY_SEGMENTS:
        for i in range(len(seg) - 1):
            d = _point_to_segment_dist(lat, lon, seg[i][0], seg[i][1], seg[i+1][0], seg[i+1][1])
            if d < min_deg:
                min_deg = d
    return _haversine_deg(lat, lon, lat + min_deg, lon)


class QuakePredictor:
    def __init__(self, store: QuakeStore = None):
        self.store = store or QuakeStore()
        self.model = None
        self.feature_names = None
        self.keep_mask = None
        self._load_model()

    def _load_model(self):
        model_path = DATA_DIR / "model_binary.json"
        cache_path = DATA_DIR / "features_binary.npz"

        if not model_path.exists():
            log.warning("No trained model found — run train.py binary first")
            return

        try:
            import xgboost as xgb
        except ImportError:
            log.warning("xgboost not installed — ML model disabled, heuristic scoring still active")
            return

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))

        if cache_path.exists():
            data = np.load(str(cache_path), allow_pickle=True)
            self.feature_names = list(data["feature_names"])
            X = data["X"]
            y = data["y"]
            nan_frac = np.isnan(X).mean(axis=0)
            self.keep_mask = nan_frac < 0.5
            self.kept_features = [n for n, k in zip(self.feature_names, self.keep_mask) if k]
            X_neg = X[y == 0][:, self.keep_mask]
            self.medians = np.nanmedian(X_neg, axis=0)

            self._needed_signals = set()
            for fname in self.kept_features:
                parts = fname.split("__")
                if len(parts) >= 4:
                    self._needed_signals.add((parts[0], parts[1]))
            log.info(f"Needed signal pairs: {len(self._needed_signals)}")

            self._available_cache = None

            self._calibrate(X, y)
        else:
            log.warning("No feature cache — predictions may use wrong feature alignment")
            self._needed_signals = set()
            self._available_cache = None

        log.info(f"Model loaded: {self.model.n_features_in_} features")

    def _calibrate(self, X: np.ndarray, y: np.ndarray):
        """Build percentile-based calibration from training negatives.

        Instead of raw probability, we report how extreme the prediction
        is relative to confirmed quiet conditions. This eliminates the
        class-balance bias in raw probabilities.
        """
        X_clean = X[:, self.keep_mask].copy()
        for col in range(X_clean.shape[1]):
            mask = np.isnan(X_clean[:, col])
            if mask.any():
                X_clean[mask, col] = self.medians[col]

        neg_mask = y == 0
        pos_mask = y == 1

        neg_probs = self.model.predict_proba(X_clean[neg_mask])[:, 1]
        pos_probs = self.model.predict_proba(X_clean[pos_mask])[:, 1]

        self.neg_probs_sorted = np.sort(neg_probs)
        self.neg_p50 = float(np.median(neg_probs))
        self.neg_p90 = float(np.percentile(neg_probs, 90))
        self.neg_p99 = float(np.percentile(neg_probs, 99))
        self.pos_p50 = float(np.median(pos_probs))

        log.info(f"Calibration: neg median={self.neg_p50:.4f}, "
                 f"neg P90={self.neg_p90:.4f}, neg P99={self.neg_p99:.4f}, "
                 f"pos median={self.pos_p50:.4f}")

    def _calibrated_score(self, raw_prob: float) -> float:
        """Convert raw probability to calibrated risk score [0, 1].

        Percentile rank against the training negative distribution:
        0.0 = looks exactly like a quiet period
        0.5 = above-average concern
        1.0 = more extreme than any known quiet period
        """
        if not hasattr(self, 'neg_probs_sorted') or self.neg_probs_sorted is None:
            return raw_prob
        idx = np.searchsorted(self.neg_probs_sorted, raw_prob)
        return float(idx) / len(self.neg_probs_sorted)

    def predict(self, lat: float, lon: float, when: datetime = None) -> dict:
        if self.model is None:
            return {"error": "No model loaded"}

        if when is None:
            when = datetime.now(UTC)

        feature_vector = self._build_features(lat, lon, when)

        if self.keep_mask is not None:
            feature_vector = feature_vector[self.keep_mask]

        for i in range(len(feature_vector)):
            if np.isnan(feature_vector[i]):
                feature_vector[i] = self.medians[i] if self.medians is not None else 0.0

        X = feature_vector.reshape(1, -1)
        raw_prob = float(self.model.predict_proba(X)[0, 1])
        risk_score = self._calibrated_score(raw_prob)
        pred = 1 if risk_score > 0.9 else 0

        risk_level = "LOW"
        if risk_score > 0.5:
            risk_level = "MODERATE"
        if risk_score > 0.75:
            risk_level = "ELEVATED"
        if risk_score > 0.9:
            risk_level = "HIGH"
        if risk_score > 0.95:
            risk_level = "VERY HIGH"
        if risk_score > 0.99:
            risk_level = "EXTREME"

        top_contributors = self._get_top_contributors(X)

        return {
            "risk_score": round(risk_score, 4),
            "raw_probability": round(raw_prob, 4),
            "prediction": pred,
            "risk_level": risk_level,
            "target": "M5.0+ within 24h",
            "location": {"lat": lat, "lon": lon},
            "evaluated_at": when.isoformat(),
            "top_contributing_features": top_contributors,
        }

    def predict_grid(self, lat_range=(-60, 70), lon_range=(-180, 180),
                     step=10.0, when: datetime = None) -> list:
        if self.model is None:
            return []

        if when is None:
            when = datetime.now(UTC)

        results = []
        lat = lat_range[0]
        while lat <= lat_range[1]:
            lon = lon_range[0]
            while lon <= lon_range[1]:
                r = self.predict(lat, lon, when)
                if "error" not in r:
                    results.append(r)
                lon += step
            lat += step

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results

    def _build_features(self, lat: float, lon: float, when: datetime) -> np.ndarray:
        if self.feature_names is None:
            return np.zeros(self.model.n_features_in_)

        db_path = self.store.db_path

        if self._available_cache is None:
            self._available_cache = {
                (s, m) for s, m, _ in _find_available_signals(self.store)
            }

        needed = self._needed_signals & self._available_cache

        seismic_feats = _extract_seismic_rate_features(db_path, when, lat, lon)

        sig_cache = {}
        for source, metric in needed:
            sig_cache[(source, metric)] = _extract_signal_features(
                db_path, source, metric, when
            )

        row = []
        for fname in self.feature_names:
            if fname == "eq_lat":
                row.append(lat)
            elif fname == "eq_lon":
                row.append(lon)
            elif fname == "hour_utc":
                row.append(when.hour)
            elif fname == "day_of_year":
                row.append(when.timetuple().tm_yday)
            elif fname == "day_of_week":
                row.append(when.weekday())
            elif fname == "month":
                row.append(when.month)
            elif fname.startswith("seismic_"):
                row.append(seismic_feats.get(fname, 0.0))
            elif "__" in fname:
                parts = fname.split("__")
                if len(parts) >= 4:
                    source = parts[0]
                    metric = parts[1]
                    window_str = parts[2]
                    stat = parts[3]
                    hours = int(window_str.replace("h", ""))
                    sig_feats = sig_cache.get((source, metric), {})
                    f = sig_feats.get(hours, {})
                    row.append(f.get(stat, np.nan))
                else:
                    row.append(np.nan)
            else:
                row.append(np.nan)

        return np.array(row, dtype=np.float64)

    def predict_heuristic(self, lat: float, lon: float, when: datetime = None) -> dict:
        if when is None:
            when = datetime.now(UTC)

        fault_km = _min_fault_distance_km(lat, lon)
        fault_score = max(0, 1 - (fault_km / 1500)) ** 1.5

        db_path = self.store.db_path
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            start_7d = (when - timedelta(days=7)).isoformat()
            start_30d = (when - timedelta(days=30)).isoformat()

            rows_7d = conn.execute(
                "SELECT magnitude, lat, lon, timestamp, place FROM earthquakes "
                "WHERE timestamp >= ? AND timestamp < ?",
                (start_7d, when.isoformat())
            ).fetchall()

            rows_30d = conn.execute(
                "SELECT magnitude, lat, lon FROM earthquakes "
                "WHERE timestamp >= ? AND timestamp < ?",
                (start_30d, when.isoformat())
            ).fetchall()

            regional_7d = []
            regional_30d = []
            radius_deg = 5.0
            for mag, rlat, rlon, *rest in rows_7d:
                if rlat is not None and rlon is not None:
                    if abs(rlat - lat) <= radius_deg and abs(rlon - lon) <= radius_deg:
                        regional_7d.append(mag)

            for mag, rlat, rlon in rows_30d:
                if rlat is not None and rlon is not None:
                    if abs(rlat - lat) <= radius_deg and abs(rlon - lon) <= radius_deg:
                        regional_30d.append(mag)

            n_regional_7d = len(regional_7d)
            n_regional_30d = len(regional_30d)
            max_mag_7d = max(regional_7d) if regional_7d else 0
            max_mag_30d = max(regional_30d) if regional_30d else 0

            rate_score_7d = min(1.0, n_regional_7d / 25)
            rate_score_30d = min(1.0, n_regional_30d / 80)
            mag_score = min(1.0, max(0, max_mag_7d - 3.0) / 4.0)
            seismicity_score = rate_score_7d * 0.4 + rate_score_30d * 0.3 + mag_score * 0.3

            kp_row = conn.execute(
                "SELECT value FROM samples WHERE metric='kp_index' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            dst_row = conn.execute(
                "SELECT value FROM samples WHERE metric='dst_index' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            kp = kp_row[0] if kp_row else 2.0
            if kp >= 9.9:
                kp = 2.0
            dst = dst_row[0] if dst_row else -5.0

            kp_score = min(1.0, max(0, kp - 2) / 6)
            dst_score = min(1.0, max(0, -dst - 10) / 90)
            geomag_score = kp_score * 0.6 + dst_score * 0.4

            nearby_quakes = []
            for mag, rlat, rlon, ts, place in rows_7d:
                if rlat is not None and rlon is not None:
                    if abs(rlat - lat) <= radius_deg and abs(rlon - lon) <= radius_deg:
                        nearby_quakes.append({
                            "magnitude": mag, "lat": rlat, "lon": rlon,
                            "timestamp": ts, "place": place or "",
                        })
            nearby_quakes.sort(key=lambda x: x["magnitude"], reverse=True)

            # Coulomb proxy: distance-decayed stress from prior M5+ events
            coulomb_proxy = 0
            for mag, rlat, rlon, *_ in rows_7d:
                if rlat is None or rlon is None or mag < 5.0:
                    continue
                d_km = math.sqrt(((rlat - lat) * 111) ** 2 + ((rlon - lon) * 111 * math.cos(math.radians(lat))) ** 2)
                if d_km < 1:
                    d_km = 1
                coulomb_proxy += (10 ** (1.5 * mag)) / (d_km ** 2)
            coulomb_score = min(1.0, math.log10(max(1, coulomb_proxy)) / 20)

            # DART buoy event mode detection
            dart_score = 0
            dart_info = None
            try:
                from lab.dart_ingest import DART_STATIONS, detect_mode_transitions
                nearest_dart = None
                nearest_dart_dist = float("inf")
                for sid, info in DART_STATIONS.items():
                    d = math.sqrt((info["lat"] - lat) ** 2 + (info["lon"] - lon) ** 2) * 111
                    if d < nearest_dart_dist:
                        nearest_dart_dist = d
                        nearest_dart = sid

                if nearest_dart and nearest_dart_dist < 2000:
                    em_row = conn.execute(
                        "SELECT COUNT(*) FROM dart_readings "
                        "WHERE station_id = ? AND mode IN (2,3) AND timestamp >= ?",
                        (nearest_dart, start_7d)
                    ).fetchone()
                    em_count = em_row[0] if em_row else 0
                    em_12h = conn.execute(
                        "SELECT COUNT(*) FROM dart_readings "
                        "WHERE station_id = ? AND mode IN (2,3) AND timestamp >= ?",
                        (nearest_dart, (when - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"))
                    ).fetchone()
                    em_12h_count = em_12h[0] if em_12h else 0

                    if em_12h_count > 0:
                        dart_score = min(1.0, em_12h_count / 60)
                        dart_info = {
                            "station": nearest_dart,
                            "region": DART_STATIONS[nearest_dart].get("region", "?"),
                            "distance_km": round(nearest_dart_dist, 0),
                            "event_mode_12h": em_12h_count,
                            "event_mode_7d": em_count,
                        }
            except Exception:
                pass

        finally:
            conn.close()

        risk_score = (
            coulomb_score * 0.30 +
            seismicity_score * 0.30 +
            fault_score * 0.15 +
            geomag_score * 0.10 +
            dart_score * 0.15
        )
        risk_score = round(min(1.0, risk_score), 4)

        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.45:
            risk_level = "ELEVATED"
        elif risk_score >= 0.2:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        contributors = [
            {
                "feature": "Coulomb stress transfer",
                "importance": round(coulomb_score, 4),
                "value": round(coulomb_proxy, 1),
                "detail": f"Stress from prior M5+ events (score: {coulomb_score:.2f})",
            },
            {
                "feature": "Regional seismicity (7d)",
                "importance": round(seismicity_score, 4),
                "value": n_regional_7d,
                "detail": f"{n_regional_7d} events within 500km, max M{max_mag_7d:.1f}",
            },
            {
                "feature": "Fault proximity",
                "importance": round(fault_score, 4),
                "value": round(fault_km, 1),
                "detail": f"{fault_km:.0f} km to nearest plate boundary",
            },
            {
                "feature": "DART buoy signal",
                "importance": round(dart_score, 4),
                "value": dart_info["event_mode_12h"] if dart_info else 0,
                "detail": (f"Station {dart_info['station']} ({dart_info['region']}): "
                          f"{dart_info['event_mode_12h']} event-mode readings in 12h"
                          if dart_info else "No nearby DART buoy or no event mode"),
            },
            {
                "feature": "Geomagnetic activity",
                "importance": round(geomag_score, 4),
                "value": round(kp, 1),
                "detail": f"Kp={kp:.1f}, Dst={dst:.0f}nT" + (" (storm)" if kp >= 5 else ""),
            },
        ]

        readings = {}
        try:
            conn2 = sqlite3.connect(str(db_path), timeout=10)
            for source, metric, label, unit in [
                ("noaa_swpc", "kp_index", "Kp Index", ""),
                ("noaa_swpc", "dst_index", "Dst Index", "nT"),
                ("goes_magnetometer", "goes_mag_total", "GOES Mag |B|", "nT"),
                ("goes_magnetometer", "goes_mag_hp", "GOES Mag Hp", "nT"),
                ("usgs_magnetometer", "mag_z", "Ground Mag Z", "nT"),
                ("intermagnet", "imag_f", "INTERMAGNET F", "nT"),
            ]:
                row = conn2.execute(
                    "SELECT value, timestamp FROM samples "
                    "WHERE source=? AND metric=? ORDER BY timestamp DESC LIMIT 1",
                    (source, metric)
                ).fetchone()
                if row:
                    readings[f"{source}__{metric}"] = {
                        "label": label, "value": round(row[0], 2),
                        "timestamp": row[1], "unit": unit,
                    }
            conn2.close()
        except Exception:
            pass

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "target": "M5.0+ seismic risk",
            "location": {"lat": lat, "lon": lon},
            "evaluated_at": when.isoformat(),
            "fault_distance_km": round(fault_km, 1),
            "regional_events_7d": n_regional_7d,
            "regional_events_30d": n_regional_30d,
            "max_magnitude_7d": round(max_mag_7d, 1),
            "kp": round(kp, 1),
            "dst": round(dst, 1),
            "top_contributing_features": contributors,
            "readings": readings,
            "nearby_quakes": nearby_quakes[:8],
        }

    def score_fault_segments(self, when: datetime = None) -> list:
        if when is None:
            when = datetime.now(UTC)

        conn = sqlite3.connect(str(self.store.db_path), timeout=10)
        try:
            start_7d = (when - timedelta(days=7)).isoformat()
            start_30d = (when - timedelta(days=30)).isoformat()
            start_12h = (when - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")

            rows_7d = conn.execute(
                "SELECT magnitude, lat, lon FROM earthquakes "
                "WHERE timestamp >= ? AND timestamp < ?",
                (start_7d, when.isoformat())
            ).fetchall()
            rows_30d = conn.execute(
                "SELECT magnitude, lat, lon FROM earthquakes "
                "WHERE timestamp >= ? AND timestamp < ?",
                (start_30d, when.isoformat())
            ).fetchall()

            kp_row = conn.execute(
                "SELECT value FROM samples WHERE metric='kp_index' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            dst_row = conn.execute(
                "SELECT value FROM samples WHERE metric='dst_index' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

            dart_buoy_scores = {}
            try:
                from lab.dart_ingest import DART_STATIONS
                for sid, info in DART_STATIONS.items():
                    em_count = conn.execute(
                        "SELECT COUNT(*) FROM dart_readings "
                        "WHERE station_id = ? AND timestamp >= ? AND mode IN (2, 3)",
                        (sid, start_12h)
                    ).fetchone()[0]
                    if em_count > 0:
                        dart_buoy_scores[sid] = {
                            "lat": info["lat"], "lon": info["lon"],
                            "em_count": em_count,
                            "is_tsunami": conn.execute(
                                "SELECT COUNT(*) FROM dart_readings "
                                "WHERE station_id = ? AND timestamp >= ? AND mode = 3",
                                (sid, start_12h)
                            ).fetchone()[0] > 0,
                        }
            except Exception:
                pass
        finally:
            conn.close()

        kp = kp_row[0] if kp_row else 2.0
        if kp >= 9.9:
            kp = 2.0
        dst = dst_row[0] if dst_row else -5.0
        kp_mod = min(1.0, max(0, kp - 2) / 6) * 0.6 + min(1.0, max(0, -dst - 10) / 90) * 0.4

        segments = []
        for seg in PLATE_BOUNDARY_SEGMENTS:
            for i in range(len(seg) - 1):
                a, b = seg[i], seg[i + 1]
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

                count_7d = 0
                max_mag_7d = 0
                energy_7d = 0
                for mag, rlat, rlon in rows_7d:
                    if rlat is None or rlon is None:
                        continue
                    d = math.sqrt((rlat - mx) ** 2 + (rlon - my) ** 2)
                    if d < 8:
                        proximity = max(0, 1 - d / 8)
                        count_7d += 1
                        max_mag_7d = max(max_mag_7d, mag)
                        energy_7d += proximity * (mag ** 1.5)

                count_30d = 0
                for mag, rlat, rlon in rows_30d:
                    if rlat is None or rlon is None:
                        continue
                    d = math.sqrt((rlat - mx) ** 2 + (rlon - my) ** 2)
                    if d < 8:
                        count_30d += 1

                seis_score = min(1.0, energy_7d / 40)
                rate_score = min(1.0, count_30d / 60)

                coulomb = 0
                for mag, rlat, rlon in rows_7d:
                    if rlat is None or rlon is None or mag < 5.0:
                        continue
                    d_km = math.sqrt(((rlat - mx) * 111) ** 2 + ((rlon - my) * 111 * math.cos(math.radians(mx))) ** 2)
                    if d_km < 1:
                        d_km = 1
                    coulomb += (10 ** (1.5 * mag)) / (d_km ** 2)
                coulomb_score = min(1.0, math.log10(max(1, coulomb)) / 20)

                dart_score = 0.0
                nearest_dart_sid = None
                for sid, dinfo in dart_buoy_scores.items():
                    d_km = math.sqrt(
                        ((dinfo["lat"] - mx) * 111) ** 2 +
                        ((dinfo["lon"] - my) * 111 * math.cos(math.radians(mx))) ** 2
                    )
                    if d_km < 2000:
                        proximity = max(0, 1 - d_km / 2000)
                        raw = min(1.0, dinfo["em_count"] / 60)
                        if dinfo["is_tsunami"]:
                            raw = min(1.0, raw + 0.5)
                        candidate = raw * (0.4 + 0.6 * proximity)
                        if candidate > dart_score:
                            dart_score = candidate
                            nearest_dart_sid = sid

                base_risk = (
                    coulomb_score * 0.30 +
                    seis_score * 0.30 +
                    rate_score * 0.15 +
                    kp_mod * 0.10 +
                    dart_score * 0.15
                )

                if dart_score > 0.3:
                    boost = dart_score * 0.25
                    base_risk = min(1.0, base_risk + boost)

                risk = round(min(1.0, base_risk), 4)

                if risk > 0.03:
                    seg_data = {
                        "a": list(a), "b": list(b),
                        "risk": risk,
                        "events_7d": count_7d,
                        "events_30d": count_30d,
                        "max_mag_7d": round(max_mag_7d, 1),
                        "kp": round(kp, 1),
                    }
                    if nearest_dart_sid:
                        seg_data["dart_station"] = nearest_dart_sid
                        seg_data["dart_score"] = round(dart_score, 3)
                    segments.append(seg_data)

        segments.sort(key=lambda x: x["risk"], reverse=True)
        return {
            "segments": segments,
            "kp": round(kp, 1),
            "dst": round(dst, 1),
            "total_segments": len(segments),
        }

    def _get_top_contributors(self, X: np.ndarray, top_n: int = 5) -> list:
        if not hasattr(self, 'kept_features') or self.kept_features is None:
            return []

        importance = self.model.feature_importances_
        feature_vals = X[0]

        scored = []
        for i, (name, imp) in enumerate(zip(self.kept_features, importance)):
            if imp > 0.005:
                scored.append({
                    "feature": name,
                    "importance": round(float(imp), 4),
                    "value": round(float(feature_vals[i]), 4) if not np.isnan(feature_vals[i]) else None,
                })

        scored.sort(key=lambda x: x["importance"], reverse=True)
        return scored[:top_n]


def scan_hotspots(store: QuakeStore = None, when: datetime = None) -> list:
    predictor = QuakePredictor(store)

    hotspot_regions = [
        ("Venezuela (Yumare)", 10.5, -68.8),
        ("Venezuela (San Felipe)", 10.3, -68.7),
        ("Japan (Kuji)", 40.2, 141.8),
        ("Japan (Tokyo)", 35.7, 139.7),
        ("California (SF)", 37.8, -122.4),
        ("California (LA)", 34.1, -118.2),
        ("Chile (Santiago)", -33.4, -70.7),
        ("Indonesia (Jakarta)", -6.2, 106.8),
        ("Turkey (Istanbul)", 41.0, 29.0),
        ("New Zealand (Wellington)", -41.3, 174.8),
        ("Alaska (Anchorage)", 61.2, -149.9),
        ("Philippines (Manila)", 14.6, 121.0),
        ("Mexico (Mexico City)", 19.4, -99.1),
        ("Iran (Tehran)", 35.7, 51.4),
        ("Nepal (Kathmandu)", 27.7, 85.3),
        ("Papua New Guinea", -6.0, 147.0),
        ("Tonga", -21.2, -175.2),
        ("Peru (Lima)", -12.0, -77.0),
        ("Caribbean (Puerto Rico)", 18.2, -66.5),
        ("Greece (Athens)", 37.9, 23.7),
    ]

    results = []
    for name, lat, lon in hotspot_regions:
        r = predictor.predict(lat, lon, when)
        r["region"] = name
        results.append(r)

    results.sort(key=lambda x: x["risk_score"], reverse=True)
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    mode = sys.argv[1] if len(sys.argv) > 1 else "hotspots"

    if mode == "hotspots":
        results = scan_hotspots()
        print(f"\n{'='*70}")
        print(f"SeismicLab Global Hotspot Scan — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*70}")
        for r in results:
            score = r["risk_score"]
            bar = "#" * int(score * 30)
            print(f"  {r['risk_level']:>10s} | {score:.1%} | {bar:<30s} | {r['region']}")
        print(f"{'='*70}\n")

    elif mode == "point":
        lat = float(sys.argv[2])
        lon = float(sys.argv[3])
        predictor = QuakePredictor()
        r = predictor.predict(lat, lon)
        print(json.dumps(r, indent=2))
