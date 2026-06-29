"""QuakeWatch Prediction Engine — ML-driven earthquake forecasting with tracking.

Issues structured predictions (zone, timeframe, magnitude, confidence),
logs them, and scores against actual events to track hit rate.
"""

import json
import logging
import sqlite3
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("quakewatch.prediction_engine")
UTC = timezone.utc

MODEL_DIR = Path(__file__).parent / "models"
PREDICTION_DB = Path(__file__).parent / "data" / "quakewatch.db"

# Prediction thresholds (overridden by model metadata if available)
PREDICT_THRESHOLD = 0.35
ELEVATED_THRESHOLD = 0.55
ALERT_THRESHOLD = 0.75

# Zone definitions (same as threat.py)
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


class PredictionEngine:

    def __init__(self, store):
        self.store = store
        self._models = {}
        self._feature_names = {}
        self._metadata = {}
        self._zone_mag_stats = {}
        self._initialized = False
        self._feature_cache = {}
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    zone_name TEXT NOT NULL,
                    window_hours INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    magnitude_est REAL,
                    magnitude_lo REAL,
                    magnitude_hi REAL,
                    threat_score REAL,
                    tidal_stress REAL,
                    top_features TEXT,
                    outcome TEXT DEFAULT 'pending',
                    resolved_at TEXT,
                    actual_mag REAL,
                    actual_event_id TEXT,
                    lead_time_hours REAL
                );

                CREATE INDEX IF NOT EXISTS idx_pred_zone
                    ON predictions(zone_id, issued_at);
                CREATE INDEX IF NOT EXISTS idx_pred_outcome
                    ON predictions(outcome);
                CREATE INDEX IF NOT EXISTS idx_pred_expires
                    ON predictions(expires_at);

                CREATE TABLE IF NOT EXISTS prediction_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    probability REAL NOT NULL,
                    threat_score REAL,
                    magnitude_est REAL
                );
                CREATE INDEX IF NOT EXISTS idx_snap_zone_ts
                    ON prediction_snapshots(zone_id, timestamp);
            """)
            conn.commit()
        finally:
            conn.close()

    def load_models(self):
        import joblib

        for wh in [24, 48, 72]:
            model_path = MODEL_DIR / f"forecast_{wh}h.joblib"
            meta_path = MODEL_DIR / f"forecast_{wh}h.json"
            if model_path.exists() and meta_path.exists():
                self._models[wh] = joblib.load(model_path)
                with open(meta_path) as f:
                    meta = json.load(f)
                self._metadata[wh] = meta
                self._feature_names[wh] = meta["feature_names"]
                model_type = "ensemble" if isinstance(self._models[wh], dict) else "pipeline"
                opt_thresh = meta.get("optimal_threshold")
                if opt_thresh:
                    log.info(f"Loaded {wh}h model ({model_type}): AUC={meta['auc']}, "
                             f"{meta['n_features']} features, threshold={opt_thresh}")
                else:
                    log.info(f"Loaded {wh}h model ({model_type}): AUC={meta['auc']}, "
                             f"{meta['n_features']} features")

        self._zone_mag_stats = self._compute_zone_mag_stats()
        self._mag_model = None
        self._mag_meta = None
        mag_path = MODEL_DIR / "magnitude_est.joblib"
        mag_meta_path = MODEL_DIR / "magnitude_est.json"
        if mag_path.exists() and mag_meta_path.exists():
            self._mag_model = joblib.load(mag_path)
            with open(mag_meta_path) as f:
                self._mag_meta = json.load(f)
            zone_aware = self._mag_meta.get("type", "").endswith("zone_aware")
            mae = self._mag_meta.get("cv_mae", "?")
            log.info(f"Loaded magnitude model: MAE={mae}, zone_aware={zone_aware}")

        self._initialized = bool(self._models)
        if not self._initialized:
            log.warning("No prediction models found — run lab/train_predictor.py first")
        return self._initialized

    def _compute_zone_mag_stats(self):
        """Precompute historical magnitude stats per zone for estimation."""
        stats = {}
        try:
            conn = sqlite3.connect(str(self.store.db_path), timeout=30)
            for zone in ZONES:
                rows = conn.execute(
                    "SELECT magnitude FROM earthquakes "
                    "WHERE magnitude >= 5.5 AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
                    (zone["lat"][0], zone["lat"][1], zone["lon"][0], zone["lon"][1])
                ).fetchall()
                mags = [r[0] for r in rows]
                if len(mags) >= 3:
                    arr = np.array(mags)
                    stats[zone["id"]] = {
                        "median": float(np.median(arr)),
                        "p75": float(np.percentile(arr, 75)),
                        "p90": float(np.percentile(arr, 90)),
                        "max": float(arr.max()),
                        "std": float(arr.std()),
                        "n": len(mags),
                    }
                else:
                    stats[zone["id"]] = {
                        "median": 6.0, "p75": 6.3, "p90": 6.5,
                        "max": 7.0, "std": 0.4, "n": len(mags),
                    }
                log.debug(f"Zone {zone['id']}: {stats[zone['id']]}")
            conn.close()
        except Exception as e:
            log.warning(f"Failed to compute zone mag stats: {e}")
        return stats

    def _extract_zone_features(self, conn, zone, when):
        """Extract features for a zone center at a given time."""
        from lab.experiment import extract_features, FEATURE_DEFS
        lat, lon = zone["center"]
        try:
            features = extract_features(conn, lat, lon, when.isoformat(), hours=72)
            return features
        except Exception as e:
            log.warning(f"Feature extraction failed for {zone['id']}: {e}")
            return None

    def _predict_model(self, model, features, feature_names):
        """Run prediction through either old Pipeline or new ensemble model."""
        if isinstance(model, dict):
            kept_names = model.get("kept_names", feature_names)
            vec = np.array([features.get(f, np.nan) for f in kept_names]).reshape(1, -1)
            X_imp = model["imputer"].transform(vec)
            # Generate interaction features
            pairs = model.get("interaction_indices", [])
            if pairs:
                interactions = [X_imp[0, i] * X_imp[0, j] for i, j in pairs]
                X_aug = np.hstack([X_imp, np.array(interactions).reshape(1, -1)])
            else:
                X_aug = X_imp
            gbm_p = model["gbm"].predict_proba(X_aug)[0, 1]
            lgb_p = model["lgb"].predict_proba(X_aug)[0, 1]
            return float((gbm_p + lgb_p) / 2.0)
        else:
            vec = np.array([features.get(f, np.nan) for f in feature_names]).reshape(1, -1)
            return float(model.predict_proba(vec)[0, 1])

    def _get_top_features(self, features, window_hours):
        """Return top contributing features for explanation."""
        if window_hours not in self._models:
            return []
        model = self._models[window_hours]
        kept_names = self._feature_names[window_hours]

        if isinstance(model, dict):
            base_names = model.get("kept_names", kept_names)
            vec = np.array([features.get(f, np.nan) for f in base_names]).reshape(1, -1)
            X_imp = model["imputer"].transform(vec)
            gbm_imp = model["gbm"].feature_importances_
            # Only report base features, not interactions
            n_base = len(base_names)
            importances = gbm_imp[:n_base]
            X_norm = X_imp[0] / (np.abs(X_imp[0]).max() + 1e-10)
            contributions = importances * np.abs(X_norm)
            ranked = sorted(zip(base_names, contributions),
                            key=lambda x: x[1], reverse=True)
        else:
            vec = np.array([features.get(f, np.nan) for f in kept_names]).reshape(1, -1)
            imputer = model.named_steps["impute"]
            X_imp = imputer.transform(vec)
            clf = model.named_steps["clf"]
            if not hasattr(clf, "feature_importances_"):
                return []
            importances = clf.feature_importances_
            X_norm = X_imp[0] / (np.abs(X_imp[0]).max() + 1e-10)
            contributions = importances * np.abs(X_norm)
            ranked = sorted(zip(kept_names, contributions),
                            key=lambda x: x[1], reverse=True)

        return [{"feature": name, "contribution": round(float(c), 4)}
                for name, c in ranked[:5]]

    def _get_extended_features(self, features, window_hours, top_n=12):
        """Return extended feature contributions with values, labels, and categories."""
        from lab.experiment import FEATURE_DEFS
        feat_meta = {f[0]: {"category": f[1], "description": f[2]} for f in FEATURE_DEFS}

        if window_hours not in self._models:
            return []
        model = self._models[window_hours]
        kept_names = self._feature_names[window_hours]

        if isinstance(model, dict):
            base_names = model.get("kept_names", kept_names)
            vec = np.array([features.get(f, np.nan) for f in base_names]).reshape(1, -1)
            X_imp = model["imputer"].transform(vec)
            gbm_imp = model["gbm"].feature_importances_
            n_base = len(base_names)
            importances = gbm_imp[:n_base]
            X_norm = X_imp[0] / (np.abs(X_imp[0]).max() + 1e-10)
            contributions = importances * np.abs(X_norm)
            ranked = sorted(zip(base_names, contributions, X_imp[0]),
                            key=lambda x: x[1], reverse=True)
        else:
            vec = np.array([features.get(f, np.nan) for f in kept_names]).reshape(1, -1)
            imputer = model.named_steps["impute"]
            X_imp = imputer.transform(vec)
            clf = model.named_steps["clf"]
            if not hasattr(clf, "feature_importances_"):
                return []
            importances = clf.feature_importances_
            X_norm = X_imp[0] / (np.abs(X_imp[0]).max() + 1e-10)
            contributions = importances * np.abs(X_norm)
            ranked = sorted(zip(kept_names, contributions, X_imp[0]),
                            key=lambda x: x[1], reverse=True)

        max_c = ranked[0][1] if ranked else 1.0
        results = []
        for name, c, val in ranked[:top_n]:
            meta = feat_meta.get(name, {"category": "other", "description": name})
            results.append({
                "feature": name,
                "label": meta["description"],
                "category": meta["category"],
                "contribution": round(float(c), 4),
                "contribution_pct": round(float(c / max_c * 100), 1) if max_c > 0 else 0,
                "value": round(float(val), 4) if not np.isnan(val) else None,
            })
        return results

    def predict_zones(self, when=None, threat_data=None):
        """Run ML prediction on all zones. Returns list of zone predictions."""
        if not self._initialized:
            if not self.load_models():
                return []

        if when is None:
            when = datetime.now(UTC)

        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.row_factory = sqlite3.Row

        # Get tidal stress for each zone
        try:
            from server import _realtime_tidal_v2
            latlons = [(z["center"][0], z["center"][1]) for z in ZONES]
            tidal_states = _realtime_tidal_v2(latlons, when)
        except Exception:
            tidal_states = [(0, 0, 0)] * len(ZONES)

        # Map threat detector data by zone if provided
        threat_map = {}
        if threat_data:
            for t in threat_data:
                threat_map[t["zone_id"]] = t

        predictions = []

        try:
            for i, zone in enumerate(ZONES):
                features = self._extract_zone_features(conn, zone, when)
                if features is None:
                    continue
                self._feature_cache[zone["id"]] = features

                _, _, tidal_stress = tidal_states[i]
                threat_info = threat_map.get(zone["id"], {})

                # Run each window model
                zone_pred = {
                    "zone_id": zone["id"],
                    "zone_name": zone["name"],
                    "center": zone["center"],
                    "tidal_stress": round(tidal_stress, 4),
                    "threat_score": threat_info.get("threat_score", 0),
                    "windows": {},
                }

                best_prob = 0
                best_window = 72

                for wh in sorted(self._models.keys()):
                    kept_names = self._feature_names[wh]

                    try:
                        prob = self._predict_model(
                            self._models[wh], features, kept_names)
                    except Exception:
                        prob = 0.0

                    zone_pred["windows"][wh] = round(prob, 4)

                    if prob > best_prob:
                        best_prob = prob
                        best_window = wh

                zone_pred["probability"] = round(best_prob, 4)
                zone_pred["window_hours"] = best_window

                # Zone-aware magnitude estimation
                mag_est, mag_lo, mag_hi = self._estimate_magnitude(
                    zone["id"], features, threat_info)
                zone_pred["magnitude_est"] = mag_est
                zone_pred["magnitude_lo"] = mag_lo
                zone_pred["magnitude_hi"] = mag_hi

                # Top contributing features
                if best_prob > PREDICT_THRESHOLD:
                    zone_pred["top_features"] = self._get_top_features(
                        features, best_window)
                else:
                    zone_pred["top_features"] = []

                predictions.append(zone_pred)

        finally:
            conn.close()

        predictions.sort(key=lambda p: p["probability"], reverse=True)
        return predictions

    def _estimate_magnitude(self, zone_id, features, threat_info=None):
        """Zone-aware magnitude estimation: trained model with heuristic fallback."""
        s = self._zone_mag_stats.get(zone_id, {
            "median": 6.0, "p75": 6.3, "p90": 6.5, "max": 7.0, "std": 0.4})

        # Try trained zone-aware model
        if (self._mag_model and isinstance(self._mag_model, dict)
                and self._mag_model.get("zone_aware")):
            try:
                mag_est = self._predict_magnitude_zone(zone_id, features)
                mag_est = max(s["median"], min(s["max"], mag_est))
                mae = self._mag_meta.get("cv_mae", 0.3) if self._mag_meta else 0.3
                lo = round(max(5.5, mag_est - mae), 1)
                hi = round(min(s["max"], mag_est + mae), 1)
                return round(mag_est, 1), lo, hi
            except Exception as e:
                log.debug(f"Magnitude model failed for {zone_id}: {e}")

        # Heuristic fallback
        return self._estimate_magnitude_heuristic(zone_id, features, threat_info, s)

    def _predict_magnitude_zone(self, zone_id, features):
        """Run zone-aware magnitude model."""
        from lab.experiment import FEATURE_DEFS
        feature_names = [f[0] for f in FEATURE_DEFS]
        zone_ids = [z["id"] for z in ZONES]

        vec = [features.get(f, np.nan) for f in feature_names]

        zs = self._zone_mag_stats.get(zone_id, {})
        vec += [zs.get("max", 7.0), zs.get("p75", 6.3), zs.get("p90", 6.5),
                zs.get("std", 0.4), zs.get("median", 6.0)]

        onehot = [1.0 if z == zone_id else 0.0 for z in zone_ids]
        vec += onehot

        kept_names = self._mag_model["kept_names"]
        all_names = feature_names + [
            "zone_hist_max", "zone_hist_p75", "zone_hist_p90",
            "zone_hist_std", "zone_hist_median"
        ] + [f"zone_{z}" for z in zone_ids]
        name_to_idx = {n: i for i, n in enumerate(all_names)}

        kept_vec = [vec[name_to_idx[n]] if n in name_to_idx else np.nan
                    for n in kept_names]
        X = np.array(kept_vec, dtype=float).reshape(1, -1)
        X_imp = self._mag_model["imputer"].transform(X)

        gbm_pred = self._mag_model["gbm"].predict(X_imp)[0]
        lgb_pred = self._mag_model["lgb"].predict(X_imp)[0]
        pred = float((gbm_pred + lgb_pred) / 2.0)

        # Floor: clamp to zone median from model or live stats
        model_zs = (self._mag_model.get("zone_stats") or {}).get(zone_id, {})
        z_floor = model_zs.get("hist_median", 0)
        if not z_floor:
            live_zs = self._zone_mag_stats.get(zone_id, {})
            z_floor = live_zs.get("median", 5.5)
        return max(pred, z_floor)

    def _estimate_magnitude_heuristic(self, zone_id, features, threat_info, s):
        """Heuristic fallback using zone stats + current conditions."""
        base = s["p75"]

        def _safe(key, default=0):
            v = features.get(key, default)
            return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        b_delta = _safe("b_delta")
        if b_delta > 0.05:
            base += min(0.6, b_delta * 2.0)
        elif b_delta < -0.05:
            base -= min(0.3, abs(b_delta))

        coulomb = _safe("coulomb_proxy")
        if coulomb > 0.3:
            base += min(0.4, coulomb * 0.3)

        moment_7d = _safe("moment_release_7d")
        if moment_7d > 15:
            base += min(0.4, (moment_7d - 15) * 0.05)

        max_mag_72h = _safe("max_mag_72h")
        if max_mag_72h >= 6.0:
            base = max(base, max_mag_72h + 0.3)

        threat_score = (threat_info or {}).get("threat_score", 0)
        if threat_score > 0.6:
            base += 0.2 * (threat_score - 0.6) / 0.4

        base = max(s["median"], min(s["max"], base))

        half_range = max(0.3, s["std"])
        lo = round(max(5.5, base - half_range), 1)
        hi = round(min(s["max"], base + half_range), 1)

        return round(base, 1), lo, hi

    def issue_predictions(self, threat_data=None):
        """Run predictions, log those above threshold, return active predictions."""
        when = datetime.now(UTC)
        all_preds = self.predict_zones(when=when, threat_data=threat_data)

        active = []
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")

        try:
            for p in all_preds:
                if p["probability"] < PREDICT_THRESHOLD:
                    continue

                # Check for existing active prediction in same zone
                existing = conn.execute(
                    "SELECT id, probability FROM predictions "
                    "WHERE zone_id = ? AND outcome = 'pending' "
                    "AND expires_at > ? ORDER BY probability DESC LIMIT 1",
                    (p["zone_id"], when.isoformat())
                ).fetchone()

                # Only issue new prediction if probability increased significantly
                # or no existing prediction
                if existing and p["probability"] < existing[1] * 1.1:
                    active.append({
                        "id": existing[0],
                        **p,
                        "issued_at": None,
                        "status": "active",
                    })
                    continue

                # Supersede all older pending predictions for this zone
                if existing:
                    conn.execute(
                        "UPDATE predictions SET outcome = 'superseded' "
                        "WHERE zone_id = ? AND outcome = 'pending' AND expires_at > ?",
                        (p["zone_id"], when.isoformat())
                    )

                wh = p["window_hours"]
                expires_at = when + timedelta(hours=wh)

                conn.execute(
                    "INSERT INTO predictions "
                    "(issued_at, expires_at, zone_id, zone_name, window_hours, "
                    "probability, magnitude_est, magnitude_lo, magnitude_hi, "
                    "threat_score, tidal_stress, top_features) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (when.isoformat(), expires_at.isoformat(),
                     p["zone_id"], p["zone_name"], wh,
                     p["probability"], p["magnitude_est"],
                     p["magnitude_lo"], p["magnitude_hi"],
                     p["threat_score"], p["tidal_stress"],
                     json.dumps(p.get("top_features", [])))
                )

                active.append({
                    **p,
                    "issued_at": when.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "status": "new",
                })

            conn.commit()
        finally:
            conn.close()

        # Save probability snapshots for ALL zones (for trend charts)
        self._save_snapshots(all_preds, when)

        log.info(f"Predictions: {len(active)} active "
                 f"({sum(1 for a in active if a['status'] == 'new')} new)")
        return active

    def _save_snapshots(self, predictions, when):
        """Save probability snapshots for trend tracking."""
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            ts = when.isoformat()
            for p in predictions:
                conn.execute(
                    "INSERT INTO prediction_snapshots "
                    "(timestamp, zone_id, probability, threat_score, magnitude_est) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ts, p["zone_id"], p["probability"],
                     p.get("threat_score", 0), p.get("magnitude_est"))
                )
            conn.commit()
        except Exception as e:
            log.warning(f"Snapshot save error: {e}")
        finally:
            conn.close()

    def get_zone_snapshots(self, zone_id, hours=72):
        """Return probability trend for a zone."""
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT timestamp, probability, threat_score, magnitude_est "
                "FROM prediction_snapshots "
                "WHERE zone_id = ? AND timestamp > ? "
                "ORDER BY timestamp ASC",
                (zone_id, cutoff)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def resolve_predictions(self):
        """Check expired/active predictions against actual events. Returns resolved count."""
        now = datetime.now(UTC)
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        resolved = 0

        try:
            # Get pending predictions
            pending = conn.execute(
                "SELECT id, issued_at, expires_at, zone_id, window_hours, "
                "magnitude_est FROM predictions WHERE outcome = 'pending'"
            ).fetchall()

            for pred_id, issued_at, expires_at, zone_id, window_hours, mag_est in pending:
                zone = next((z for z in ZONES if z["id"] == zone_id), None)
                if not zone:
                    continue

                t_issued = datetime.fromisoformat(issued_at)
                t_expires = datetime.fromisoformat(expires_at)
                lat_min, lat_max = zone["lat"]
                lon_min, lon_max = zone["lon"]

                # Check for M5.5+ events in zone during prediction window
                events = conn.execute(
                    "SELECT id, timestamp, magnitude FROM earthquakes "
                    "WHERE magnitude >= 5.5 "
                    "AND timestamp >= ? AND timestamp <= ? "
                    "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
                    (issued_at, expires_at,
                     lat_min, lat_max, lon_min, lon_max)
                ).fetchall()

                if events:
                    # HIT — prediction was correct
                    best_event = max(events, key=lambda e: e[2])
                    t_event = datetime.fromisoformat(best_event[1])
                    if t_event.tzinfo is None:
                        t_event = t_event.replace(tzinfo=UTC)
                    lead_time = (t_event - t_issued).total_seconds() / 3600

                    conn.execute(
                        "UPDATE predictions SET outcome = 'hit', resolved_at = ?, "
                        "actual_mag = ?, actual_event_id = ?, lead_time_hours = ? "
                        "WHERE id = ?",
                        (now.isoformat(), best_event[2], best_event[0],
                         round(lead_time, 1), pred_id)
                    )
                    resolved += 1
                    log.info(f"PREDICTION HIT: {zone_id} — M{best_event[2]:.1f} "
                             f"at {lead_time:.1f}h lead time "
                             f"(predicted M{mag_est:.1f})")

                elif now > t_expires:
                    # MISS / FALSE ALARM — window expired with no qualifying event
                    conn.execute(
                        "UPDATE predictions SET outcome = 'false_alarm', "
                        "resolved_at = ? WHERE id = ?",
                        (now.isoformat(), pred_id)
                    )
                    resolved += 1

            conn.commit()
        finally:
            conn.close()

        return resolved

    def get_active_predictions(self):
        """Return currently active (pending, not expired) predictions."""
        now = datetime.now(UTC)
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM predictions "
                "WHERE outcome = 'pending' AND expires_at > ? "
                "ORDER BY probability DESC",
                (now.isoformat(),)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_prediction_history(self, limit=50):
        """Return resolved predictions (most recent first)."""
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM predictions "
                "WHERE outcome != 'pending' "
                "ORDER BY resolved_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_metrics(self):
        """Calculate prediction accuracy metrics."""
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE outcome != 'pending'"
            ).fetchone()[0]

            hits = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE outcome = 'hit'"
            ).fetchone()[0]

            false_alarms = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE outcome = 'false_alarm'"
            ).fetchone()[0]

            pending = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE outcome = 'pending'"
            ).fetchone()[0]

            avg_lead = conn.execute(
                "SELECT AVG(lead_time_hours) FROM predictions WHERE outcome = 'hit'"
            ).fetchone()[0]

            avg_mag_error = conn.execute(
                "SELECT AVG(ABS(magnitude_est - actual_mag)) FROM predictions "
                "WHERE outcome = 'hit' AND actual_mag IS NOT NULL"
            ).fetchone()[0]

            # Per-window metrics
            window_metrics = {}
            for wh in [24, 48, 72]:
                wh_total = conn.execute(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE window_hours = ? AND outcome != 'pending'",
                    (wh,)
                ).fetchone()[0]
                wh_hits = conn.execute(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE window_hours = ? AND outcome = 'hit'",
                    (wh,)
                ).fetchone()[0]
                if wh_total > 0:
                    window_metrics[wh] = {
                        "total": wh_total,
                        "hits": wh_hits,
                        "hit_rate": round(wh_hits / wh_total, 3),
                    }

            hit_rate = round(hits / total, 3) if total > 0 else 0
            false_alarm_rate = round(false_alarms / total, 3) if total > 0 else 0

            return {
                "total_resolved": total,
                "hits": hits,
                "false_alarms": false_alarms,
                "pending": pending,
                "hit_rate": hit_rate,
                "false_alarm_rate": false_alarm_rate,
                "avg_lead_time_hours": round(avg_lead, 1) if avg_lead else None,
                "avg_magnitude_error": round(avg_mag_error, 2) if avg_mag_error else None,
                "by_window": window_metrics,
            }
        finally:
            conn.close()

    def explain_prediction(self, zone_id, threat_data=None):
        """Generate a rich explanation using stored prediction + cached features."""
        zone = next((z for z in ZONES if z["id"] == zone_id), None)
        if not zone:
            return None

        if not self._initialized:
            if not self.load_models():
                return None

        when = datetime.now(UTC)
        conn = sqlite3.connect(str(self.store.db_path), timeout=30)
        conn.row_factory = sqlite3.Row

        try:
            # Use the stored prediction from DB (matches what sidebar shows)
            stored = conn.execute(
                "SELECT * FROM predictions "
                "WHERE zone_id = ? AND outcome = 'pending' AND expires_at > ? "
                "ORDER BY probability DESC LIMIT 1",
                (zone_id, when.isoformat())
            ).fetchone()

            if stored:
                pred = dict(stored)
                best_prob = pred["probability"]
                best_window = pred["window_hours"]
                mag_est = pred["magnitude_est"]
                mag_lo = pred["magnitude_lo"]
                mag_hi = pred["magnitude_hi"]
                tidal_stress = pred.get("tidal_stress", 0)
                issued = pred["issued_at"]
                expires = pred["expires_at"]
                top_features_raw = pred.get("top_features")
            else:
                best_prob = 0
                best_window = 72
                mag_est = 6.2
                mag_lo = 5.2
                mag_hi = 7.2
                tidal_stress = 0
                issued = when.isoformat()
                expires = (when + timedelta(hours=72)).isoformat()
                top_features_raw = None

            # Use cached features or extract for just this zone
            features = self._feature_cache.get(zone_id)
            if not features:
                features = self._extract_zone_features(conn, zone, when)
                if features:
                    self._feature_cache[zone_id] = features

            evidence = []
            if features and best_prob > 0:
                evidence = self._get_extended_features(
                    features, best_window, top_n=12)

            # Zone prediction history
            history = conn.execute(
                "SELECT outcome, probability, magnitude_est, actual_mag, "
                "lead_time_hours, window_hours, issued_at "
                "FROM predictions WHERE zone_id = ? "
                "ORDER BY issued_at DESC LIMIT 20",
                (zone_id,)
            ).fetchall()
            zone_history = [dict(r) for r in history]

            zone_hits = sum(1 for h in zone_history if h["outcome"] == "hit")
            zone_resolved = sum(1 for h in zone_history
                                if h["outcome"] in ("hit", "false_alarm"))

            # Recent seismicity
            lat_lo, lat_hi = zone["lat"]
            lon_lo, lon_hi = zone["lon"]
            cutoff_72h = (when - timedelta(hours=72)).isoformat()
            recent_quakes = conn.execute(
                "SELECT magnitude, depth_km, timestamp, place FROM earthquakes "
                "WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
                "AND timestamp > ? ORDER BY magnitude DESC LIMIT 10",
                (lat_lo, lat_hi, lon_lo, lon_hi, cutoff_72h)
            ).fetchall()

            # Probability trend (snapshots)
            snapshots = conn.execute(
                "SELECT timestamp, probability, threat_score "
                "FROM prediction_snapshots "
                "WHERE zone_id = ? AND timestamp > ? "
                "ORDER BY timestamp ASC",
                (zone_id, (when - timedelta(hours=168)).isoformat())
            ).fetchall()
            trend = [{"t": r["timestamp"], "p": r["probability"],
                       "ts": r["threat_score"]} for r in snapshots]

            result = {
                "zone_id": zone_id,
                "zone_name": zone["name"],
                "center": zone["center"],
                "issued_at": issued,
                "expires_at": expires,
                "probability": round(best_prob, 4),
                "window_hours": best_window,
                "magnitude_est": round(mag_est, 1),
                "magnitude_lo": round(mag_lo, 1),
                "magnitude_hi": round(mag_hi, 1),
                "tidal_stress": round(tidal_stress, 4) if tidal_stress else 0,
                "evidence": evidence,
                "trend": trend,
                "recent_seismicity": [
                    {"mag": r["magnitude"], "depth": r["depth_km"],
                     "time": r["timestamp"], "place": r["place"]}
                    for r in recent_quakes
                ],
                "zone_track_record": {
                    "total": zone_resolved,
                    "hits": zone_hits,
                    "hit_rate": round(zone_hits / zone_resolved, 3)
                    if zone_resolved > 0 else None,
                    "recent": zone_history[:10],
                },
            }

            # Merge threat data
            if threat_data:
                td = next((t for t in threat_data
                           if t.get("zone_id") == zone_id), None)
                if td:
                    result["threat"] = {
                        "score": td.get("threat_score", 0),
                        "level": td.get("level", "NORMAL"),
                        "signals": td.get("signals", []),
                        "foreshock_score": td.get("foreshock_score", 0),
                        "b_score": td.get("b_score", 0),
                        "accel_score": td.get("accel_score", 0),
                        "sig_event_score": td.get("sig_event_score", 0),
                        "tidal_boost": td.get("tidal_boost", 0),
                        "geomag_boost": td.get("geomag_boost", 0),
                        "dart_boost": td.get("dart_boost", 0),
                        "volcanic_boost": td.get("volcanic_boost", 0),
                        "ml_boost": td.get("ml_boost", 0),
                    }

            return result
        finally:
            conn.close()
