"""QuakeWatch Event Analyzer — Deterministic seismic event classification and risk scoring.

Triggers on each M4.5+ event, pulls regional context (aftershock patterns, DART,
volcanic, tidal, Coulomb stress), computes structured features, and stores
an analysis alert for review and downstream use in the prediction engine.
"""

import json
import logging
import math
import sqlite3
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("quakewatch.event_analyzer")
UTC = timezone.utc

MIN_MAGNITUDE = 4.5
AFTERSHOCK_RADIUS_KM = 300
AFTERSHOCK_WINDOW_DAYS = 14


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _seismic_moment(mag):
    return 10 ** (1.5 * mag + 9.1)


class EventAnalyzer:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._ensure_table()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        conn = self._conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS event_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    earthquake_id TEXT UNIQUE,
                    analyzed_at TEXT,
                    magnitude REAL,
                    lat REAL,
                    lon REAL,
                    depth_km REAL,
                    place TEXT,
                    event_timestamp TEXT,

                    event_type TEXT,
                    risk_score REAL,
                    risk_level TEXT,

                    mainshock_mag REAL,
                    mainshock_dist_km REAL,
                    hours_since_mainshock REAL,
                    omori_expected_rate REAL,
                    omori_observed_rate REAL,
                    omori_residual REAL,

                    sequence_count_24h INTEGER,
                    sequence_count_72h INTEGER,
                    mag_trend TEXT,
                    migration_speed_km_day REAL,
                    migration_bearing REAL,
                    doublet_flag INTEGER DEFAULT 0,

                    coulomb_loading REAL,

                    dart_elevated_count INTEGER,
                    dart_max_deviation REAL,
                    volcanic_hotspots_500km INTEGER,
                    volcanic_max_frp REAL,

                    background_rate_30d REAL,
                    rate_anomaly REAL,
                    b_value_14d REAL,
                    b_value_trend TEXT,

                    summary TEXT,
                    watch_items TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_ea_timestamp
                    ON event_analyses(event_timestamp);
                CREATE INDEX IF NOT EXISTS idx_ea_risk
                    ON event_analyses(risk_score);
            """)
            conn.commit()
        finally:
            conn.close()

    def analyze(self, eq_id, mag, lat, lon, depth_km, place, timestamp_str):
        """Run full analysis on a single event. Returns the analysis dict."""
        if mag < MIN_MAGNITUDE:
            return None

        conn = self._conn()
        try:
            # Skip if already analyzed
            existing = conn.execute(
                "SELECT id FROM event_analyses WHERE earthquake_id = ?",
                (eq_id,)
            ).fetchone()
            if existing:
                return None

            t_event = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if t_event.tzinfo is None:
                t_event = t_event.replace(tzinfo=UTC)

            analysis = {
                "earthquake_id": eq_id,
                "magnitude": mag,
                "lat": lat,
                "lon": lon,
                "depth_km": depth_km,
                "place": place,
                "event_timestamp": timestamp_str,
                "analyzed_at": datetime.now(UTC).isoformat(),
            }

            # ── Aftershock / mainshock context ──
            mainshock = self._find_mainshock(conn, lat, lon, t_event, mag)
            if mainshock:
                analysis.update(mainshock)

            # ── Sequence features ──
            seq = self._sequence_features(conn, lat, lon, t_event, mag)
            analysis.update(seq)

            # ── Coulomb stress loading ──
            analysis["coulomb_loading"] = self._coulomb_loading(conn, lat, lon, t_event)

            # ── DART status ──
            dart = self._dart_status(conn)
            analysis.update(dart)

            # ── Volcanic context ──
            volc = self._volcanic_context(conn, lat, lon)
            analysis.update(volc)

            # ── Background seismicity ──
            bg = self._background_context(conn, lat, lon, t_event)
            analysis.update(bg)

            # ── Classification and risk score ──
            classification = self._classify(analysis)
            analysis.update(classification)

            # ── Store ──
            self._store(conn, analysis)

            level = analysis["risk_level"].upper()
            etype = analysis["event_type"]
            log.info(f"Event analysis: M{mag} {place} → {etype} [{level}] "
                     f"risk={analysis['risk_score']:.2f}")

            return analysis

        except Exception as e:
            log.error(f"Event analysis failed for {eq_id}: {e}")
            return None
        finally:
            conn.close()

    def _find_mainshock(self, conn, lat, lon, t_event, mag):
        """Check if this event is an aftershock of a recent larger event."""
        t_start = (t_event - timedelta(days=AFTERSHOCK_WINDOW_DAYS)).isoformat()
        candidates = conn.execute(
            "SELECT magnitude, lat, lon, timestamp, place FROM earthquakes "
            "WHERE magnitude > ? AND timestamp >= ? AND timestamp < ? "
            "ORDER BY magnitude DESC",
            (mag, t_start, t_event.isoformat())
        ).fetchall()

        for c in candidates:
            dist = _haversine_km(lat, lon, c["lat"], c["lon"])
            if dist > AFTERSHOCK_RADIUS_KM:
                continue

            t_main = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
            if t_main.tzinfo is None:
                t_main = t_main.replace(tzinfo=UTC)
            hours_since = (t_event - t_main).total_seconds() / 3600.0

            # Omori analysis: expected vs observed aftershock rate
            omori = self._omori_fit(conn, c["lat"], c["lon"], c["magnitude"],
                                    t_main, t_event)

            return {
                "mainshock_mag": c["magnitude"],
                "mainshock_dist_km": round(dist, 1),
                "hours_since_mainshock": round(hours_since, 1),
                "omori_expected_rate": omori["expected"],
                "omori_observed_rate": omori["observed"],
                "omori_residual": omori["residual"],
            }

        return {
            "mainshock_mag": None,
            "mainshock_dist_km": None,
            "hours_since_mainshock": None,
            "omori_expected_rate": None,
            "omori_observed_rate": None,
            "omori_residual": None,
        }

    def _omori_fit(self, conn, lat, lon, mainshock_mag, t_main, t_now):
        """Compare observed aftershock rate to Omori prediction."""
        hours_since = (t_now - t_main).total_seconds() / 3600.0
        days_since = hours_since / 24.0

        if days_since < 0.05:
            return {"expected": 0, "observed": 0, "residual": 0}

        # Count aftershocks in the last 24h (or since mainshock if <24h)
        window_start = max(t_main, t_now - timedelta(hours=24))
        observed = conn.execute(
            "SELECT COUNT(*) FROM earthquakes "
            "WHERE timestamp >= ? AND timestamp <= ? "
            "AND magnitude >= 2.0 "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (window_start.isoformat(), t_now.isoformat(),
             lat - 3, lat + 3, lon - 3, lon + 3)
        ).fetchone()[0]

        window_hours = (t_now - window_start).total_seconds() / 3600.0
        observed_rate = observed / max(1, window_hours) * 24  # per day

        # Omori expected: N(t) ≈ 10^(a - b*Mc) / (c + t)^p
        # Simplified: expected_rate ≈ 10^(mainshock_mag - 3.5) / (0.05 + days_since)
        expected_rate = 10 ** (mainshock_mag - 3.5) / (0.05 + days_since)

        residual = (observed_rate / max(1, expected_rate)) - 1.0

        return {
            "expected": round(expected_rate, 1),
            "observed": round(observed_rate, 1),
            "residual": round(residual, 3),
        }

    def _sequence_features(self, conn, lat, lon, t_event, mag):
        """Detect doublets, swarms, magnitude trends, migration."""
        t_24h = (t_event - timedelta(hours=24)).isoformat()
        t_72h = (t_event - timedelta(hours=72)).isoformat()

        nearby_24h = conn.execute(
            "SELECT magnitude, lat, lon, timestamp FROM earthquakes "
            "WHERE timestamp >= ? AND timestamp <= ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "ORDER BY timestamp",
            (t_24h, t_event.isoformat(), lat - 3, lat + 3, lon - 3, lon + 3)
        ).fetchall()

        nearby_72h = conn.execute(
            "SELECT magnitude, lat, lon, timestamp FROM earthquakes "
            "WHERE timestamp >= ? AND timestamp <= ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "ORDER BY timestamp",
            (t_72h, t_event.isoformat(), lat - 3, lat + 3, lon - 3, lon + 3)
        ).fetchall()

        result = {
            "sequence_count_24h": len(nearby_24h),
            "sequence_count_72h": len(nearby_72h),
            "doublet_flag": 0,
            "mag_trend": "stable",
            "migration_speed_km_day": 0.0,
            "migration_bearing": 0.0,
        }

        # Doublet detection: M4.5+ within 10 min and 100km
        for eq in nearby_24h:
            if eq["magnitude"] < 4.5:
                continue
            t_eq = datetime.fromisoformat(eq["timestamp"].replace("Z", "+00:00"))
            if t_eq.tzinfo is None:
                t_eq = t_eq.replace(tzinfo=UTC)
            dt_min = abs((t_event - t_eq).total_seconds()) / 60.0
            dist = _haversine_km(lat, lon, eq["lat"], eq["lon"])
            if 0.5 < dt_min < 10 and dist < 100:
                result["doublet_flag"] = 1
                break

        # Magnitude trend: compare first-half vs second-half magnitudes
        if len(nearby_72h) >= 6:
            mags = [eq["magnitude"] for eq in nearby_72h]
            half = len(mags) // 2
            first_mean = np.mean(mags[:half])
            second_mean = np.mean(mags[half:])
            if second_mean > first_mean + 0.3:
                result["mag_trend"] = "escalating"
            elif second_mean < first_mean - 0.3:
                result["mag_trend"] = "decaying"

        # Migration: centroid movement over time
        if len(nearby_72h) >= 4:
            half = len(nearby_72h) // 2
            early = nearby_72h[:half]
            late = nearby_72h[half:]
            early_lat = np.mean([eq["lat"] for eq in early])
            early_lon = np.mean([eq["lon"] for eq in early])
            late_lat = np.mean([eq["lat"] for eq in late])
            late_lon = np.mean([eq["lon"] for eq in late])

            dist_km = _haversine_km(early_lat, early_lon, late_lat, late_lon)
            t_early = datetime.fromisoformat(early[-1]["timestamp"].replace("Z", "+00:00"))
            t_late = datetime.fromisoformat(late[0]["timestamp"].replace("Z", "+00:00"))
            dt_days = max(0.01, (t_late - t_early).total_seconds() / 86400)

            result["migration_speed_km_day"] = round(dist_km / dt_days, 1)
            result["migration_bearing"] = round(
                math.degrees(math.atan2(late_lon - early_lon, late_lat - early_lat)) % 360, 1)

        return result

    def _coulomb_loading(self, conn, lat, lon, t_event):
        """Sum Coulomb stress transfer from recent M5+ events."""
        t_start = (t_event - timedelta(days=60)).isoformat()
        large = conn.execute(
            "SELECT magnitude, lat as elat, lon as elon FROM earthquakes "
            "WHERE magnitude >= 5.0 AND timestamp >= ? AND timestamp < ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (t_start, t_event.isoformat(), lat - 5, lat + 5, lon - 5, lon + 5)
        ).fetchall()

        total = 0.0
        for eq in large:
            r_km = max(1.0, _haversine_km(lat, lon, eq["elat"], eq["elon"]))
            total += _seismic_moment(eq["magnitude"]) / (r_km ** 3)

        return round(math.log10(total) if total > 0 else 0, 3)

    def _dart_status(self, conn):
        """Check DART buoy network for elevated stations."""
        try:
            from lab.dart_ingest import DART_STATIONS
        except ImportError:
            return {"dart_elevated_count": 0, "dart_max_deviation": 0}

        elevated = 0
        max_dev = 0.0

        for sid in DART_STATIONS:
            row = conn.execute(
                "SELECT timestamp, height_m FROM dart_readings "
                "WHERE station_id = ? ORDER BY timestamp DESC LIMIT 1",
                (sid,)
            ).fetchone()
            if not row or not row["height_m"]:
                continue

            stats = conn.execute(
                "SELECT AVG(height_m) as avg_h, "
                "       AVG(height_m * height_m) - AVG(height_m) * AVG(height_m) as var_h "
                "FROM dart_readings WHERE station_id = ? "
                "AND timestamp >= datetime(?, '-24 hours')",
                (sid, row["timestamp"])
            ).fetchone()
            if not stats or not stats["avg_h"] or not stats["var_h"] or stats["var_h"] <= 0:
                continue

            std = stats["var_h"] ** 0.5
            dev = abs(row["height_m"] - stats["avg_h"]) / max(0.001, std)
            if dev >= 1.5:
                elevated += 1
            max_dev = max(max_dev, dev)

        return {
            "dart_elevated_count": elevated,
            "dart_max_deviation": round(max_dev, 2),
        }

    def _volcanic_context(self, conn, lat, lon):
        """Count nearby volcanic hotspots."""
        try:
            hotspots = conn.execute(
                "SELECT COUNT(*) as cnt, MAX(frp) as max_frp "
                "FROM thermal_anomalies t "
                "JOIN volcanoes v ON t.nearest_volcano_id = v.volcano_number "
                "WHERE t.acq_date >= date('now', '-7 days') "
                "AND v.lat BETWEEN ? AND ? AND v.lon BETWEEN ? AND ?",
                (lat - 5, lat + 5, lon - 5, lon + 5)
            ).fetchone()
            return {
                "volcanic_hotspots_500km": hotspots["cnt"] if hotspots else 0,
                "volcanic_max_frp": round(hotspots["max_frp"] or 0, 1) if hotspots else 0,
            }
        except Exception:
            return {"volcanic_hotspots_500km": 0, "volcanic_max_frp": 0}

    def _background_context(self, conn, lat, lon, t_event):
        """Background seismicity rate and b-value."""
        t_30d = (t_event - timedelta(days=30)).isoformat()
        t_7d = (t_event - timedelta(days=7)).isoformat()
        t_14d = (t_event - timedelta(days=14)).isoformat()

        count_30d = conn.execute(
            "SELECT COUNT(*) FROM earthquakes "
            "WHERE timestamp >= ? AND timestamp < ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "AND magnitude >= 2.0",
            (t_30d, t_event.isoformat(), lat - 3, lat + 3, lon - 3, lon + 3)
        ).fetchone()[0]

        count_7d = conn.execute(
            "SELECT COUNT(*) FROM earthquakes "
            "WHERE timestamp >= ? AND timestamp < ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "AND magnitude >= 2.0",
            (t_7d, t_event.isoformat(), lat - 3, lat + 3, lon - 3, lon + 3)
        ).fetchone()[0]

        rate_30d = count_30d / 30.0
        rate_7d = count_7d / 7.0
        rate_anomaly = rate_7d / max(0.1, rate_30d)

        # b-value from 14-day magnitudes
        mags_14d = conn.execute(
            "SELECT magnitude FROM earthquakes "
            "WHERE timestamp >= ? AND timestamp < ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "AND magnitude >= 2.0",
            (t_14d, t_event.isoformat(), lat - 3, lat + 3, lon - 3, lon + 3)
        ).fetchall()
        mags = [r["magnitude"] for r in mags_14d]

        b_value = None
        if len(mags) >= 8:
            mc = 2.0
            above = [m for m in mags if m >= mc]
            if above and np.mean(above) > mc:
                b_value = round(1.0 / (math.log(10) * (np.mean(above) - mc)), 3)

        # b-value trend: compare 14d to prior 14d
        t_28d = (t_event - timedelta(days=28)).isoformat()
        mags_prior = conn.execute(
            "SELECT magnitude FROM earthquakes "
            "WHERE timestamp >= ? AND timestamp < ? "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "AND magnitude >= 2.0",
            (t_28d, t_14d, lat - 3, lat + 3, lon - 3, lon + 3)
        ).fetchall()
        mags_p = [r["magnitude"] for r in mags_prior]

        b_trend = "stable"
        if b_value and len(mags_p) >= 8:
            above_p = [m for m in mags_p if m >= 2.0]
            if above_p and np.mean(above_p) > 2.0:
                b_prior = 1.0 / (math.log(10) * (np.mean(above_p) - 2.0))
                if b_value < b_prior - 0.15:
                    b_trend = "dropping"
                elif b_value > b_prior + 0.15:
                    b_trend = "rising"

        return {
            "background_rate_30d": round(rate_30d, 2),
            "rate_anomaly": round(rate_anomaly, 2),
            "b_value_14d": b_value,
            "b_value_trend": b_trend,
        }

    def _classify(self, a):
        """Classify event type and compute composite risk score."""
        score = 0.0
        event_type = "isolated"
        watch = []
        findings = []

        mag = a["magnitude"]

        # Base magnitude contribution
        if mag >= 7.0:
            score += 0.4
            findings.append(f"M{mag:.1f} — major event")
        elif mag >= 6.0:
            score += 0.3
            findings.append(f"M{mag:.1f} — strong event")
        elif mag >= 5.0:
            score += 0.15
        else:
            score += 0.05

        # Aftershock context
        if a.get("mainshock_mag"):
            event_type = "aftershock"
            findings.append(
                f"Aftershock of M{a['mainshock_mag']:.1f} "
                f"({a['mainshock_dist_km']:.0f}km, {a['hours_since_mainshock']:.0f}h ago)")

            if a.get("omori_residual") and a["omori_residual"] > 0.5:
                score += 0.2
                watch.append("Aftershock rate ABOVE Omori prediction — stress may be redistributing")
            elif a.get("omori_residual") and a["omori_residual"] < -0.5:
                score -= 0.05
                findings.append("Aftershock rate below Omori prediction — normal decay")
        else:
            if mag >= 5.0:
                score += 0.1
                findings.append("No nearby mainshock — independent event")
                watch.append("Monitor for follow-on events in next 24-72h")

        # Doublet
        if a.get("doublet_flag"):
            event_type = "doublet"
            score += 0.15
            findings.append("DOUBLET detected — paired event within 10 min")
            watch.append("Doublets indicate complex rupture — watch for stress gap filling")

        # Magnitude trend
        if a.get("mag_trend") == "escalating":
            score += 0.15
            watch.append("Magnitudes ESCALATING — sequence not decaying normally")
        elif a.get("mag_trend") == "decaying":
            score -= 0.05
            findings.append("Magnitudes decaying — normal aftershock pattern")

        # Migration
        if a.get("migration_speed_km_day", 0) > 20:
            score += 0.1
            bearing = a.get("migration_bearing", 0)
            watch.append(
                f"Seismicity migrating at {a['migration_speed_km_day']:.0f} km/day "
                f"(bearing {bearing:.0f}°)")

        # Coulomb stress
        if a.get("coulomb_loading", 0) > 12:
            score += 0.1
            findings.append(f"High Coulomb stress loading: {a['coulomb_loading']:.1f}")

        # DART
        if a.get("dart_elevated_count", 0) > 0:
            score += 0.05 * min(3, a["dart_elevated_count"])
            findings.append(
                f"{a['dart_elevated_count']} DART buoys elevated "
                f"(max {a['dart_max_deviation']:.1f}σ)")

        # Volcanic
        if a.get("volcanic_max_frp", 0) >= 15:
            score += 0.05
            findings.append(
                f"Volcanic activity nearby: {a['volcanic_hotspots_500km']} hotspots, "
                f"max FRP {a['volcanic_max_frp']:.0f} MW")

        # Rate anomaly
        if a.get("rate_anomaly", 0) > 3.0:
            score += 0.1
            watch.append(
                f"Seismicity rate {a['rate_anomaly']:.1f}× above 30-day background")

        # b-value
        if a.get("b_value_trend") == "dropping":
            score += 0.1
            watch.append("b-value DROPPING — larger events becoming more likely")

        # Sequence size
        if a.get("sequence_count_72h", 0) > 20 and not a.get("mainshock_mag"):
            event_type = "swarm"
            findings.append(f"Swarm: {a['sequence_count_72h']} events in 72h without dominant mainshock")

        # Clamp
        score = max(0.0, min(1.0, score))

        if score >= 0.7:
            level = "critical"
        elif score >= 0.5:
            level = "high"
        elif score >= 0.3:
            level = "elevated"
        elif score >= 0.15:
            level = "moderate"
        else:
            level = "low"

        return {
            "event_type": event_type,
            "risk_score": round(score, 3),
            "risk_level": level,
            "summary": json.dumps(findings),
            "watch_items": json.dumps(watch),
        }

    def _store(self, conn, a):
        """Insert analysis into database."""
        conn.execute("""
            INSERT OR IGNORE INTO event_analyses (
                earthquake_id, analyzed_at, magnitude, lat, lon, depth_km,
                place, event_timestamp, event_type, risk_score, risk_level,
                mainshock_mag, mainshock_dist_km, hours_since_mainshock,
                omori_expected_rate, omori_observed_rate, omori_residual,
                sequence_count_24h, sequence_count_72h, mag_trend,
                migration_speed_km_day, migration_bearing, doublet_flag,
                coulomb_loading, dart_elevated_count, dart_max_deviation,
                volcanic_hotspots_500km, volcanic_max_frp,
                background_rate_30d, rate_anomaly, b_value_14d, b_value_trend,
                summary, watch_items
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
        """, (
            a["earthquake_id"], a["analyzed_at"], a["magnitude"],
            a["lat"], a["lon"], a["depth_km"],
            a["place"], a["event_timestamp"],
            a["event_type"], a["risk_score"], a["risk_level"],
            a.get("mainshock_mag"), a.get("mainshock_dist_km"),
            a.get("hours_since_mainshock"),
            a.get("omori_expected_rate"), a.get("omori_observed_rate"),
            a.get("omori_residual"),
            a.get("sequence_count_24h"), a.get("sequence_count_72h"),
            a.get("mag_trend"),
            a.get("migration_speed_km_day"), a.get("migration_bearing"),
            a.get("doublet_flag", 0),
            a.get("coulomb_loading"),
            a.get("dart_elevated_count"), a.get("dart_max_deviation"),
            a.get("volcanic_hotspots_500km"), a.get("volcanic_max_frp"),
            a.get("background_rate_30d"), a.get("rate_anomaly"),
            a.get("b_value_14d"), a.get("b_value_trend"),
            a.get("summary"), a.get("watch_items"),
        ))
        conn.commit()

    def get_recent(self, hours=24, min_risk=0.0):
        """Retrieve recent analyses above a risk threshold."""
        conn = self._conn()
        try:
            t_start = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
            rows = conn.execute(
                "SELECT * FROM event_analyses "
                "WHERE event_timestamp >= ? AND risk_score >= ? "
                "ORDER BY risk_score DESC",
                (t_start, min_risk)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def backfill(self, hours=72):
        """Analyze recent M4.5+ events that haven't been analyzed yet."""
        conn = self._conn()
        try:
            t_start = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
            events = conn.execute(
                "SELECT id, magnitude, lat, lon, depth_km, place, timestamp "
                "FROM earthquakes "
                "WHERE magnitude >= ? AND timestamp >= ? "
                "AND id NOT IN (SELECT earthquake_id FROM event_analyses) "
                "ORDER BY timestamp",
                (MIN_MAGNITUDE, t_start)
            ).fetchall()
            log.info(f"Backfilling {len(events)} events from last {hours}h")
        finally:
            conn.close()

        results = []
        for eq in events:
            r = self.analyze(
                eq["id"], eq["magnitude"], eq["lat"], eq["lon"],
                eq["depth_km"], eq["place"], eq["timestamp"])
            if r:
                results.append(r)

        return results
