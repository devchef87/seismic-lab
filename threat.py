"""QuakeWatch Tier-1 Threat Detector — multi-signal zone assessment with caching."""

import math
import sqlite3
import logging
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger("quakewatch.threat")
UTC = timezone.utc

M_MIN = 2.5
DM = 0.1
CACHE_TTL = 60

ZONES = [
    {"id": "socal",       "name": "Southern California",     "lat": [31, 36],  "lon": [-121, -114], "center": [33.5, -117.5]},
    {"id": "norcal",      "name": "N. California / Cascadia","lat": [36, 50],  "lon": [-131, -119], "center": [43, -125]},
    {"id": "alaska",      "name": "Alaska / Aleutians",      "lat": [50, 65],  "lon": [-180, -130], "center": [57, -155]},
    {"id": "japan",       "name": "Japan",                   "lat": [28, 46],  "lon": [128, 148],   "center": [37, 140]},
    {"id": "indonesia",   "name": "Indonesia",               "lat": [-12, 8],  "lon": [94, 136],    "center": [-2, 115]},
    {"id": "chile_peru",  "name": "Chile / Peru",            "lat": [-46, 2],  "lon": [-82, -65],   "center": [-22, -70]},
    {"id": "mediterranean","name": "Mediterranean / Turkey",  "lat": [33, 42],  "lon": [-6, 45],     "center": [38, 28]},
    {"id": "mexico_ca",   "name": "Mexico / Central America","lat": [7, 25],   "lon": [-115, -77],  "center": [16, -96]},
    {"id": "himalaya",    "name": "Himalayas",               "lat": [24, 40],  "lon": [65, 100],    "center": [32, 82]},
    {"id": "caribbean",   "name": "Caribbean",               "lat": [10, 22],  "lon": [-85, -60],   "center": [16, -72]},
    {"id": "iceland",     "name": "Iceland / N. Atlantic",   "lat": [55, 68],  "lon": [-25, -12],   "center": [64, -19]},
    {"id": "philippines", "name": "Philippines / Taiwan",     "lat": [4, 26],   "lon": [118, 128],   "center": [15, 123]},
]

ESCALATE_THRESHOLD = 0.5
ALERT_THRESHOLD = 0.75


class ThreatDetector:

    def __init__(self, store):
        self.store = store
        self._prev_scores = {}
        self._ml_probabilities = {}
        self._cache = None
        self._cache_time = 0

    def invalidate_cache(self):
        self._cache = None
        self._cache_time = 0

    def _tidal_stress_for_zones(self, when):
        try:
            from server import _realtime_tidal_v2
            latlons = [(z["center"][0], z["center"][1]) for z in ZONES]
            states = _realtime_tidal_v2(latlons, when)
            return {ZONES[i]["id"]: states[i] for i in range(len(ZONES))}
        except Exception as e:
            log.warning("Tidal computation unavailable: %s", e)
            return {}

    def scan(self, when=None):
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_time) < CACHE_TTL:
            return self._cache

        if when is None:
            when = datetime.now(UTC)
        conn = sqlite3.connect(self.store.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            results = [self._assess_zone(conn, z, when) for z in ZONES]
            mag_anom = self._mag_anomaly(conn)
            geo_onset = self._geomag_onset(conn)
            dart_signals = self._dart_zone_signals(conn)
            volcanic_signals = self._volcanic_zone_signals(conn)
        finally:
            conn.close()

        # Tidal stress boost (real-time Moon+Sun ephemeris)
        tidal_map = self._tidal_stress_for_zones(when)
        for r in results:
            tid = tidal_map.get(r["zone_id"])
            if tid:
                v2_now, v2_max, tidal_stress = tid
                r["tidal_stress"] = round(tidal_stress, 4)
                if tidal_stress > 0.4 and r["threat_score"] > 0.05:
                    base = r["threat_score"]
                    boosted = base * (1 + tidal_stress * 0.25)
                    r["threat_score"] = round(min(1.0, boosted), 4)
                    r["tidal_boost"] = round(r["threat_score"] - base, 4)
                    tier = "critical" if tidal_stress > 0.7 else "elevated"
                    r["signals"].append({
                        "signal": f"Gravitational stress {tier} ({tidal_stress*100:.0f}%)",
                        "detail": f"Moon/Sun tidal forcing at {tidal_stress*100:.0f}%",
                        "severity": tidal_stress,
                    })
            else:
                r["tidal_stress"] = 0.0

        onset_score = geo_onset["score"]
        for r in results:
            r["mag_anomaly"] = mag_anom
            r["geomag_onset"] = {"score": onset_score, "signals": geo_onset["signals"]}

            if onset_score > 0.1 and r["threat_score"] > 0.05:
                base = r["threat_score"]
                boosted = base * (1 + onset_score * 0.35)
                r["threat_score"] = round(min(1.0, boosted), 4)
                r["geomag_boost"] = round(r["threat_score"] - base, 4)

                for s in geo_onset["signals"]:
                    if not any(existing["signal"] == s["signal"] for existing in r["signals"]):
                        r["signals"].append(s)

        # DART buoy anomaly boost
        for r in results:
            dart = dart_signals.get(r["zone_id"])
            if dart:
                base = r["threat_score"]
                r["threat_score"] = round(min(1.0, base + dart["boost"]), 4)
                r["dart_boost"] = round(r["threat_score"] - base, 4)
                r["signals"].append(dart["signal"])

        # Volcanic proximity boost
        for r in results:
            volc = volcanic_signals.get(r["zone_id"])
            if volc:
                base = r["threat_score"]
                r["threat_score"] = round(min(1.0, base + volc["boost"]), 4)
                r["volcanic_boost"] = round(r["threat_score"] - base, 4)
                r["signals"].append(volc["signal"])

        # ML forecast boost
        for r in results:
            ml_prob = self._ml_probabilities.get(r["zone_id"], 0)
            r["ml_probability"] = round(ml_prob, 4)
            if ml_prob > 0.5 and r["threat_score"] > 0.05:
                base = r["threat_score"]
                boost_factor = (ml_prob - 0.5) * 1.2
                boosted = base * (1 + boost_factor)
                r["threat_score"] = round(min(1.0, boosted), 4)
                r["ml_boost"] = round(r["threat_score"] - base, 4)
                tier = "high" if ml_prob >= 0.75 else "likely"
                r["signals"].append({
                    "signal": f"ML forecast {tier} ({ml_prob*100:.0f}%)",
                    "detail": f"Multi-feature model — M5+ within 72h at {ml_prob*100:.0f}% confidence",
                    "severity": ml_prob,
                })

        # Re-evaluate levels after all boosts
        for r in results:
            if r["threat_score"] >= ALERT_THRESHOLD:
                r["level"] = "ALERT"
            elif r["threat_score"] >= ESCALATE_THRESHOLD:
                r["level"] = "ELEVATED"
            elif r["threat_score"] >= 0.25:
                r["level"] = "WATCH"

        results.sort(key=lambda x: x["threat_score"], reverse=True)

        for r in results:
            prev = self._prev_scores.get(r["zone_id"], r["threat_score"])
            r["trend"] = round(r["threat_score"] - prev, 4)
            self._prev_scores[r["zone_id"]] = r["threat_score"]

        self._cache = results
        self._cache_time = now
        return results

    def _assess_zone(self, conn, zone, when):
        lat_min, lat_max = zone["lat"]
        lon_min, lon_max = zone["lon"]

        start_30d = (when - timedelta(days=30)).isoformat()
        start_7d = (when - timedelta(days=7)).isoformat()
        start_72h = (when - timedelta(hours=72)).isoformat()
        start_24h = (when - timedelta(hours=24)).isoformat()
        start_6h = (when - timedelta(hours=6)).isoformat()
        start_1h = (when - timedelta(hours=1)).isoformat()
        now_iso = when.isoformat()

        rows = conn.execute(
            "SELECT magnitude, timestamp FROM earthquakes "
            "WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "AND timestamp >= ? AND timestamp < ? AND magnitude >= ?",
            (lat_min, lat_max, lon_min, lon_max, start_30d, now_iso, M_MIN)
        ).fetchall()

        mags_30d = [r[0] for r in rows]
        n_30d = len(mags_30d)
        n_7d = sum(1 for m, ts in rows if ts >= start_7d)
        n_24h = sum(1 for m, ts in rows if ts >= start_24h)
        n_6h = sum(1 for m, ts in rows if ts >= start_6h)
        n_1h = sum(1 for m, ts in rows if ts >= start_1h)

        # ── Foreshock rate z-score (6h bins over 30 days) ──
        bins_6h = self._rate_bins(rows, when, bin_hours=6, n_bins=120)
        current_6h = bins_6h[0] if bins_6h else n_6h

        if len(bins_6h) > 10:
            baseline = bins_6h[1:]
            mean_r = sum(baseline) / len(baseline)
            var_r = sum((x - mean_r) ** 2 for x in baseline) / len(baseline)
            std_r = math.sqrt(var_r) if var_r > 0 else max(1, mean_r * 0.5)
            rate_zscore = (current_6h - mean_r) / std_r
        else:
            rate_zscore = 0
            mean_r = 0

        # 1h spike check
        bins_1h = self._rate_bins(rows, when, bin_hours=1, n_bins=168)
        current_1h = bins_1h[0] if bins_1h else n_1h
        if len(bins_1h) > 20:
            bl_1h = bins_1h[1:]
            mean_1h = sum(bl_1h) / len(bl_1h)
            var_1h = sum((x - mean_1h) ** 2 for x in bl_1h) / len(bl_1h)
            std_1h = math.sqrt(var_1h) if var_1h > 0 else max(1, mean_1h * 0.5)
            rate_1h_zscore = (current_1h - mean_1h) / std_1h
        else:
            rate_1h_zscore = 0
            mean_1h = 0

        foreshock_score = min(1.0, max(0, max(rate_zscore, rate_1h_zscore * 0.8) / 4.5))

        # ── b-value delta (7d vs 30d) ──
        mags_7d = [m for m, ts in rows if ts >= start_7d]
        b_30d = self._b_value(mags_30d)
        b_7d = self._b_value(mags_7d)

        b_delta = 0
        if b_30d is not None and b_7d is not None:
            b_delta = b_30d - b_7d

        b_score = min(1.0, max(0, b_delta / 0.35))

        # ── Rate acceleration (is the rate increasing over time?) ──
        accel_score = 0
        if len(bins_6h) >= 8:
            recent_4 = sum(bins_6h[:4]) / 4
            older_4 = sum(bins_6h[4:8]) / 4
            if older_4 > 0:
                accel = (recent_4 - older_4) / older_4
                accel_score = min(1.0, max(0, accel / 2.0))

        # ── Significant recent event (M4.5+ in 24h/72h) ──
        mags_24h = [m for m, ts in rows if ts >= start_24h]
        mags_72h = [m for m, ts in rows if ts >= start_72h]
        max_24h = max(mags_24h) if mags_24h else 0
        max_72h = max(mags_72h) if mags_72h else 0

        sig_24h = min(1.0, max(0, (max_24h - 4.0) / 3.5))
        sig_72h = min(1.0, max(0, (max_72h - 4.0) / 3.5)) * 0.7
        sig_event_score = max(sig_24h, sig_72h)

        # ── Composite (tuned weights — foreshock + b-value are strongest predictors) ──
        threat_score = (
            foreshock_score * 0.35 +
            b_score * 0.25 +
            accel_score * 0.15 +
            sig_event_score * 0.25
        )

        if threat_score >= ALERT_THRESHOLD:
            level = "ALERT"
        elif threat_score >= ESCALATE_THRESHOLD:
            level = "ELEVATED"
        elif threat_score >= 0.25:
            level = "WATCH"
        else:
            level = "NORMAL"

        signals = []
        if rate_zscore > 2:
            signals.append({
                "signal": "Foreshock rate spike",
                "detail": f"{current_6h} events/6h — {rate_zscore:.1f}σ above baseline ({mean_r:.1f} avg)",
                "severity": min(1.0, rate_zscore / 5),
            })
        if rate_1h_zscore > 2 and current_1h >= 2:
            signals.append({
                "signal": "Hourly burst",
                "detail": f"{current_1h} events in the last hour ({rate_1h_zscore:.1f}σ)",
                "severity": min(1.0, rate_1h_zscore / 5),
            })
        if b_delta > 0.1 and b_7d is not None:
            signals.append({
                "signal": "b-value declining",
                "detail": f"{b_7d:.2f} (7d) vs {b_30d:.2f} (30d) — stress consolidating",
                "severity": min(1.0, b_delta / 0.4),
            })
        if accel_score > 0.2:
            signals.append({
                "signal": "Rate accelerating",
                "detail": f"Recent activity {accel_score * 2:.0f}x higher than prior period",
                "severity": accel_score,
            })
        if max_24h >= 4.5 or max_72h >= 5.0:
            best_mag = max(max_24h, max_72h)
            window = "24h" if max_24h >= max_72h else "72h"
            signals.append({
                "signal": "Significant recent event",
                "detail": f"M{best_mag:.1f} in the last {window} — aftershock risk elevated",
                "severity": min(1.0, (best_mag - 4.0) / 3.0),
            })

        return {
            "zone_id": zone["id"],
            "name": zone["name"],
            "center": zone["center"],
            "threat_score": round(threat_score, 4),
            "level": level,
            "foreshock_score": round(foreshock_score, 4),
            "sig_event_score": round(sig_event_score, 4),
            "max_mag_24h": round(max_24h, 1),
            "max_mag_72h": round(max_72h, 1),
            "rate_zscore_6h": round(rate_zscore, 2),
            "rate_zscore_1h": round(rate_1h_zscore, 2),
            "b_value_7d": round(b_7d, 3) if b_7d is not None else None,
            "b_value_30d": round(b_30d, 3) if b_30d is not None else None,
            "b_delta": round(b_delta, 3),
            "b_score": round(b_score, 4),
            "accel_score": round(accel_score, 4),
            "events_1h": n_1h,
            "events_6h": n_6h,
            "events_24h": n_24h,
            "events_7d": n_7d,
            "events_30d": n_30d,
            "signals": signals,
        }

    def _rate_bins(self, rows, when, bin_hours, n_bins):
        bins = [0] * n_bins
        bin_sec = bin_hours * 3600
        for mag, ts in rows:
            try:
                t = datetime.fromisoformat(ts)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=UTC)
                age_sec = (when - t).total_seconds()
                idx = int(age_sec / bin_sec)
                if 0 <= idx < n_bins:
                    bins[idx] += 1
            except (ValueError, TypeError):
                continue
        return bins

    def _b_value(self, magnitudes):
        filtered = [m for m in magnitudes if m >= M_MIN]
        if len(filtered) < 15:
            return None
        m_mean = sum(filtered) / len(filtered)
        denom = m_mean - (M_MIN - DM / 2)
        if denom <= 0.01:
            return None
        return math.log10(math.e) / denom

    def _mag_anomaly(self, conn):
        try:
            rows = conn.execute(
                "SELECT value, timestamp FROM samples "
                "WHERE metric IN ('mag_z','imag_f','goes_mag_total') "
                "AND timestamp >= datetime('now','-24 hours') "
                "ORDER BY timestamp DESC LIMIT 500"
            ).fetchall()
            if len(rows) < 20:
                return {"score": 0, "detail": "Insufficient data"}

            values = [r[0] for r in rows if r[0] is not None]
            if len(values) < 20:
                return {"score": 0, "detail": "Insufficient data"}

            recent = values[:len(values)//3]
            baseline = values[len(values)//3:]

            if not recent or not baseline:
                return {"score": 0, "detail": "Insufficient data"}

            mean_r = sum(recent) / len(recent)
            mean_b = sum(baseline) / len(baseline)
            var_r = sum((v - mean_r)**2 for v in recent) / len(recent)
            var_b = sum((v - mean_b)**2 for v in baseline) / len(baseline)

            cv_recent = math.sqrt(var_r) / abs(mean_r) if mean_r != 0 else 0
            cv_baseline = math.sqrt(var_b) / abs(mean_b) if mean_b != 0 else 0

            if cv_baseline > 0:
                ratio = cv_recent / cv_baseline
                score = min(1.0, max(0, (ratio - 1.0) / 2.0))
            else:
                score = 0

            if score > 0.3:
                detail = f"Field variance {ratio:.1f}x above baseline"
            else:
                detail = "Within normal range"

            return {"score": round(score, 3), "detail": detail}
        except Exception as e:
            log.warning(f"Mag anomaly check failed: {e}")
            return {"score": 0, "detail": "Error"}

    def _geomag_onset(self, conn):
        """Detect sudden geomagnetic onset — dDst/dt, GOES |B| compression, ground dB/dt."""
        result = {"score": 0, "signals": []}
        try:
            self._onset_dst(conn, result)
            self._onset_goes(conn, result)
            self._onset_ground_dbdt(conn, result)
        except Exception as e:
            log.warning(f"Geomag onset check failed: {e}")
        return result

    def _onset_dst(self, conn, result):
        rows = conn.execute(
            "SELECT value, timestamp FROM samples "
            "WHERE source='noaa_swpc' AND metric='dst_index' "
            "AND timestamp >= datetime('now', '-6 hours') "
            "ORDER BY timestamp ASC"
        ).fetchall()
        vals = [(r[0], r[1]) for r in rows if r[0] is not None]
        if len(vals) < 4:
            return

        n = len(vals)
        quarter = max(1, n // 4)
        recent = vals[-quarter:]
        earlier = vals[:quarter]

        avg_recent = sum(v for v, _ in recent) / len(recent)
        avg_earlier = sum(v for v, _ in earlier) / len(earlier)

        try:
            t0 = datetime.fromisoformat(earlier[-1][1])
            t1 = datetime.fromisoformat(recent[0][1])
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=UTC)
            if t1.tzinfo is None:
                t1 = t1.replace(tzinfo=UTC)
            hours = max(0.5, (t1 - t0).total_seconds() / 3600)
        except (ValueError, TypeError):
            return

        dst_rate = (avg_recent - avg_earlier) / hours

        if dst_rate < -8:
            severity = min(1.0, abs(dst_rate) / 25)
            result["signals"].append({
                "signal": "Sudden Dst drop (dDst/dt)",
                "detail": f"Dst declining at {dst_rate:.1f} nT/hr — storm commencement signature",
                "severity": round(severity, 3),
            })
            result["score"] = max(result["score"], round(severity, 3))

    def _onset_goes(self, conn, result):
        rows = conn.execute(
            "SELECT value, timestamp FROM samples "
            "WHERE source='goes_magnetometer' AND metric='goes_mag_total' "
            "AND timestamp >= datetime('now', '-6 hours') "
            "ORDER BY timestamp ASC"
        ).fetchall()
        vals = [r[0] for r in rows if r[0] is not None]
        if len(vals) < 6:
            return

        n = len(vals)
        third = max(2, n // 3)
        recent = vals[-third:]
        baseline = vals[:n - third]
        if not baseline:
            return

        avg_recent = sum(recent) / len(recent)
        avg_baseline = sum(baseline) / len(baseline)
        jump = avg_recent - avg_baseline

        if jump > 10:
            severity = min(1.0, jump / 40)
            result["signals"].append({
                "signal": "Magnetospheric compression",
                "detail": f"GOES |B| jumped +{jump:.1f} nT — solar wind pressure pulse",
                "severity": round(severity, 3),
            })
            result["score"] = max(result["score"], round(severity, 3))

        if jump < -15:
            severity = min(1.0, abs(jump) / 50)
            result["signals"].append({
                "signal": "Magnetospheric expansion",
                "detail": f"GOES |B| dropped {jump:.1f} nT — possible substorm onset",
                "severity": round(severity, 3),
            })
            result["score"] = max(result["score"], round(severity, 3))

    def _onset_ground_dbdt(self, conn, result):
        rows = conn.execute(
            "SELECT value, timestamp FROM samples "
            "WHERE metric IN ('mag_z', 'imag_f') "
            "AND timestamp >= datetime('now', '-3 hours') "
            "ORDER BY timestamp ASC"
        ).fetchall()
        vals = [r[0] for r in rows if r[0] is not None]
        if len(vals) < 12:
            return

        diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
        n = len(diffs)
        third = max(2, n // 3)
        recent_d = diffs[-third:]
        baseline_d = diffs[:n - third]
        if not baseline_d:
            return

        avg_recent = sum(recent_d) / len(recent_d)
        avg_baseline = sum(baseline_d) / len(baseline_d)

        if avg_baseline > 0.01:
            ratio = avg_recent / avg_baseline
            if ratio > 2.0:
                severity = min(1.0, (ratio - 1) / 4)
                result["signals"].append({
                    "signal": "Ground dB/dt elevated",
                    "detail": f"Rate-of-change {ratio:.1f}x above baseline — telluric current induction risk",
                    "severity": round(severity, 3),
                })
                result["score"] = max(result["score"], round(severity, 3))

    def _dart_zone_signals(self, conn):
        """Check DART buoy anomalies near each zone."""
        signals = {}
        try:
            rows = conn.execute(
                "SELECT station_id, mode, height_m, timestamp FROM dart_readings "
                "WHERE timestamp >= datetime('now', '-6 hours') "
                "ORDER BY timestamp DESC"
            ).fetchall()
            if not rows:
                return signals

            from lab.dart_ingest import DART_STATIONS

            station_latest = {}
            station_deviations = {}
            for sid, mode, height, ts in rows:
                if sid not in station_latest:
                    station_latest[sid] = {"mode": mode, "height": height}
                if sid not in station_deviations:
                    station_deviations[sid] = []
                station_deviations[sid].append(height)

            for sid, heights in station_deviations.items():
                if len(heights) >= 4:
                    mean_h = sum(heights) / len(heights)
                    var_h = sum((h - mean_h) ** 2 for h in heights) / len(heights)
                    std_h = math.sqrt(var_h) if var_h > 0 else 1.0
                    latest = station_latest[sid]["height"]
                    station_latest[sid]["deviation"] = abs(latest - mean_h) / std_h if std_h > 0.001 else 0

            for zone in ZONES:
                lat_min, lat_max = zone["lat"]
                lon_min, lon_max = zone["lon"]
                pad = 15
                best_dev = 0
                best_mode = 1
                best_sid = None
                n_elevated = 0

                for sid, info in DART_STATIONS.items():
                    slat, slon = info["lat"], info["lon"]
                    if not (lat_min - pad <= slat <= lat_max + pad and
                            lon_min - pad <= slon <= lon_max + pad):
                        continue
                    if sid not in station_latest:
                        continue
                    st = station_latest[sid]
                    mode = st["mode"]
                    dev = st.get("deviation", 0)
                    if mode >= 2 or dev >= 1.5:
                        n_elevated += 1
                    if mode > best_mode or (mode == best_mode and dev > best_dev):
                        best_mode = mode
                        best_dev = dev
                        best_sid = sid

                if best_mode >= 3:
                    signals[zone["id"]] = {
                        "boost": 0.15,
                        "signal": {
                            "signal": f"DART tsunami mode ({best_sid})",
                            "detail": f"Buoy {best_sid} in tsunami mode — seafloor displacement detected",
                            "severity": 0.9,
                        }
                    }
                elif best_mode >= 2:
                    signals[zone["id"]] = {
                        "boost": 0.10,
                        "signal": {
                            "signal": f"DART event mode ({best_sid})",
                            "detail": f"Buoy {best_sid} triggered event mode — pressure anomaly",
                            "severity": 0.7,
                        }
                    }
                elif n_elevated >= 2:
                    signals[zone["id"]] = {
                        "boost": 0.06,
                        "signal": {
                            "signal": f"DART anomaly ({n_elevated} buoys)",
                            "detail": f"{n_elevated} nearby buoys showing >1.5σ deviation",
                            "severity": 0.5,
                        }
                    }
        except Exception as e:
            log.warning("DART zone signal check failed: %s", e)
        return signals

    def _volcanic_zone_signals(self, conn):
        """Check volcanic thermal anomalies near each zone."""
        signals = {}
        try:
            rows = conn.execute(
                "SELECT v.name, v.lat, v.lon, MAX(t.frp) as max_frp, "
                "       COUNT(t.id) as hotspot_count "
                "FROM volcanoes v "
                "JOIN thermal_anomalies t ON t.nearest_volcano_id = v.volcano_number "
                "WHERE t.acq_date >= date('now', '-3 days') "
                "GROUP BY v.volcano_number "
                "HAVING max_frp >= 15 "
                "ORDER BY max_frp DESC"
            ).fetchall()
            if not rows:
                return signals

            for zone in ZONES:
                lat_min, lat_max = zone["lat"]
                lon_min, lon_max = zone["lon"]
                pad = 5
                zone_volcs = []

                for name, vlat, vlon, frp, count in rows:
                    if (lat_min - pad <= vlat <= lat_max + pad and
                            lon_min - pad <= vlon <= lon_max + pad):
                        zone_volcs.append({"name": name, "frp": frp, "count": count})

                if not zone_volcs:
                    continue

                top = max(zone_volcs, key=lambda v: v["frp"])
                n_active = len(zone_volcs)
                max_frp = top["frp"]

                if max_frp >= 100:
                    boost = 0.10
                    tier = "intense"
                elif max_frp >= 50:
                    boost = 0.06
                    tier = "high"
                else:
                    boost = 0.03
                    tier = "elevated"

                if n_active >= 3:
                    boost += 0.03

                signals[zone["id"]] = {
                    "boost": round(boost, 3),
                    "signal": {
                        "signal": f"Volcanic activity {tier} ({top['name']})",
                        "detail": f"{n_active} active volcano{'es' if n_active > 1 else ''} — peak FRP {max_frp:.0f} MW",
                        "severity": min(1.0, max_frp / 120),
                    }
                }
        except Exception as e:
            log.warning("Volcanic zone signal check failed: %s", e)
        return signals
