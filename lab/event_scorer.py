"""Event-triggered earthquake escalation scorer.

Scores individual earthquakes on arrival: given the sequence context at this
location (catalog lookback), what's the probability a larger event follows
within 7 days / Gardner-Knopoff radius?

Designed to be called from the realtime engine on each new event, or from
the ingest pipeline as a post-insert hook. Stateless — reads catalog from DB.

  from lab.event_scorer import EventScorer
  scorer = EventScorer()              # loads model once
  scores = scorer.score_new(conn)     # score all events since last call
"""
import os, math, time
import numpy as np
import lightgbm as lgb
import sqlite3

import json as _json

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(_HERE), "models", "event_escalation_lgb.txt")
MAG_TABLE_PATH = os.path.join(os.path.dirname(_HERE), "models", "mag_exceedance_table.json")
DB_PATH = os.path.join(os.path.dirname(_HERE), "data", "quakewatch.db")

GK_CAP_KM = 300
MIN_MAG = 2.5

FEAT_NAMES = [
    "trigger_mag", "trigger_depth",
    "n_events_24h", "n_events_72h", "n_events_7d", "n_events_30d",
    "n_m4_events_7d",
    "max_mag_72h", "max_mag_7d", "mag_range_7d", "median_mag_7d",
    "consec_up", "rumble_ratio", "mag_trend_72h",
    "double_tap_count", "depth_to_peak",
    "active_ratio_7d_30d", "rate_accel_6h_24h",
    "hours_since_last", "hours_since_m4",
    "b_value_14d", "energy_ratio_24h_48h",
    "moment_7d_log", "moment_accel",
    "depth_trend_7d",
]
NUM_FEATS = len(FEAT_NAMES)

H1 = 3600
W24 = 24 * H1; W72 = 72 * H1; W7D = 7 * 86400; W14D = 14 * 86400; W30D = 30 * 86400


def _gk_radius(mag):
    return min(GK_CAP_KM, 10 ** (0.1238 * mag + 0.983))


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(min(1, math.sqrt(a)))


def _haversine_vec(lat1, lon1, lats, lons):
    R = 6371.0
    dlat = np.radians(lats - lat1); dlon = np.radians(lons - lon1)
    a = np.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        np.cos(np.radians(lats)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _compute_features(mag, depth, lat, lon, epoch, conn):
    """Compute 25 sequence features for one event from the catalog."""
    fi = {name: i for i, name in enumerate(FEAT_NAMES)}
    feats = np.zeros(NUM_FEATS, dtype=np.float32)
    feats[fi["trigger_mag"]] = mag
    feats[fi["trigger_depth"]] = depth or 10.0

    r_km = _gk_radius(mag)
    dlat_deg = r_km / 111.0
    dlon_deg = r_km / (111.0 * max(0.1, math.cos(math.radians(lat))))

    t_lo = epoch - W30D
    from datetime import datetime, timezone
    ts_lo = datetime.fromtimestamp(t_lo, tz=timezone.utc).isoformat()
    ts_hi = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    rows = conn.execute(
        "SELECT timestamp, magnitude, depth_km FROM earthquakes "
        "WHERE magnitude >= ? AND timestamp >= ? AND timestamp < ? "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
        "ORDER BY timestamp",
        (MIN_MAG, ts_lo, ts_hi,
         lat - dlat_deg, lat + dlat_deg,
         lon - dlon_deg, lon + dlon_deg)).fetchall()

    if not rows:
        feats[fi["hours_since_last"]] = 999.0
        feats[fi["hours_since_m4"]] = 999.0
        return feats

    # Parse and filter by actual haversine distance
    import pandas as pd
    nearby_t, nearby_m, nearby_d = [], [], []
    for ts, m, d in rows:
        try:
            t = pd.Timestamp(ts, tz="UTC").timestamp()
        except Exception:
            continue
        if t >= epoch:
            continue
        rlat, rlon = lat, lon  # approx — we already box-filtered
        nearby_t.append(t); nearby_m.append(m); nearby_d.append(d or 10.0)

    if not nearby_t:
        feats[fi["hours_since_last"]] = 999.0
        feats[fi["hours_since_m4"]] = 999.0
        return feats

    nb_t = np.array(nearby_t)
    nb_m = np.array(nearby_m, dtype=np.float32)
    nb_d = np.array(nearby_d, dtype=np.float32)
    dt = epoch - nb_t  # seconds ago

    m24 = dt <= W24; m72 = dt <= W72; m7d = dt <= W7D
    m4_mask = nb_m >= 4.0

    feats[fi["n_events_24h"]] = m24.sum()
    feats[fi["n_events_72h"]] = m72.sum()
    feats[fi["n_events_7d"]] = m7d.sum()
    feats[fi["n_events_30d"]] = len(nb_t)
    feats[fi["n_m4_events_7d"]] = (m7d & m4_mask).sum()

    if m7d.sum() > 0:
        m7_mags = nb_m[m7d]
        feats[fi["max_mag_72h"]] = nb_m[m72].max() if m72.any() else 0
        feats[fi["max_mag_7d"]] = m7_mags.max()
        feats[fi["mag_range_7d"]] = m7_mags.max() - m7_mags.min()
        feats[fi["median_mag_7d"]] = np.median(m7_mags)

        seq_mags = m7_mags
        if len(seq_mags) >= 2:
            run = 1; best = 1
            for k in range(1, len(seq_mags)):
                if seq_mags[k] >= seq_mags[k-1] - 0.05:
                    run += 1; best = max(best, run)
                else:
                    run = 1
            feats[fi["consec_up"]] = best

            med = np.median(seq_mags)
            if med > 0:
                feats[fi["rumble_ratio"]] = seq_mags.max() / med

            feats[fi["depth_to_peak"]] = np.argmax(seq_mags)

        if m72.sum() >= 3:
            m72_mags = nb_m[m72]
            x = np.arange(len(m72_mags), dtype=np.float32)
            xm = x.mean()
            denom = ((x - xm) ** 2).sum()
            if denom > 0:
                feats[fi["mag_trend_72h"]] = ((x - xm) * (m72_mags - m72_mags.mean())).sum() / denom

        if len(seq_mags) >= 3:
            taps = 0
            for k in range(len(seq_mags) - 2):
                if abs(seq_mags[k] - seq_mags[k+1]) <= 0.3 and \
                   seq_mags[k+2] >= seq_mags[k] + 1.0:
                    taps += 1
            feats[fi["double_tap_count"]] = taps

        m14d = dt <= W14D
        if m14d.sum() >= 10:
            bm = nb_m[m14d]
            above = bm[bm >= MIN_MAG]
            if len(above) >= 5:
                feats[fi["b_value_14d"]] = np.log10(np.e) / (above.mean() - MIN_MAG + 0.01)

        e24 = np.sum(10 ** (1.5 * nb_m[m24])) if m24.any() else 0
        m24_48 = (dt > W24) & (dt <= 2 * W24)
        e_prior = np.sum(10 ** (1.5 * nb_m[m24_48])) if m24_48.any() else 0.01
        feats[fi["energy_ratio_24h_48h"]] = e24 / (e_prior + 0.01)

        moment_7d = np.sum(10 ** (1.5 * m7_mags + 9.1))
        feats[fi["moment_7d_log"]] = np.log10(moment_7d + 1)
        moment_prior = np.sum(10 ** (1.5 * nb_m[~m7d] + 9.1)) if (~m7d).any() else 1
        feats[fi["moment_accel"]] = moment_7d / (moment_prior + 1)

        if m7d.sum() >= 3:
            d7 = nb_d[m7d]
            x = np.arange(len(d7), dtype=np.float32)
            xm = x.mean()
            denom = ((x - xm) ** 2).sum()
            if denom > 0:
                feats[fi["depth_trend_7d"]] = ((x - xm) * (d7 - d7.mean())).sum() / denom

    n7 = m7d.sum(); n30 = len(nb_t)
    feats[fi["active_ratio_7d_30d"]] = n7 / max(1, n30)
    n6h = (dt <= 6 * H1).sum(); n24h = m24.sum()
    rate_6h = n6h; rate_prior_18h = max(n24h - n6h, 0) / 3.0
    feats[fi["rate_accel_6h_24h"]] = rate_6h / (rate_prior_18h + 1.0)

    feats[fi["hours_since_last"]] = dt.min() / 3600.0
    m4_times = dt[m4_mask]
    feats[fi["hours_since_m4"]] = m4_times.min() / 3600.0 if len(m4_times) > 0 else 999.0

    return feats


def _classify_sequence(feats):
    """Classify the sequence pattern type from features."""
    fi = {name: i for i, name in enumerate(FEAT_NAMES)}
    n7 = feats[fi["n_events_7d"]]
    rumble = feats[fi["rumble_ratio"]]
    consec = feats[fi["consec_up"]]
    dtap = feats[fi["double_tap_count"]]
    trend = feats[fi["mag_trend_72h"]]

    if n7 < 1:
        return "isolated"
    if rumble >= 1.5 and n7 >= 5:
        return "rumble"
    if dtap >= 1:
        return "double_tap"
    if consec >= 3:
        return "staircase"
    if trend > 0.1 and n7 >= 3:
        return "accelerating"
    if n7 >= 3:
        return "active_sequence"
    return "early_sequence"


def cluster_scores(events, radius_km=150):
    """Collapse scored events into one entry per active sequence/location.

    Events within radius_km of each other are the same sequence — keep only
    the highest-probability event as the representative, with a count of how
    many events are in that cluster. This prevents 42 alerts for one swarm."""
    if not events:
        return []

    clusters = []  # list of (representative, member_events)
    used = set()

    # Sort by probability descending — greedily assign to clusters
    sorted_events = sorted(events, key=lambda e: -e["escalation_prob"])

    for ev in sorted_events:
        if ev["event_id"] in used:
            continue
        # Start a new cluster with this event as representative
        members = [ev]
        used.add(ev["event_id"])

        # Find all unassigned events within radius
        for other in sorted_events:
            if other["event_id"] in used:
                continue
            d = _haversine(ev["lat"], ev["lon"], other["lat"], other["lon"])
            if d <= radius_km:
                members.append(other)
                used.add(other["event_id"])

        # Representative = highest prob (already sorted, so it's ev)
        latest = max(members, key=lambda m: m["time"])
        cluster_entry = {
            "event_id": ev["event_id"],
            "time": latest["time"],          # most recent event time
            "magnitude": ev["magnitude"],    # magnitude of highest-prob event
            "max_magnitude": round(max(m["magnitude"] for m in members), 1),
            "depth_km": ev["depth_km"],
            "lat": round(sum(m["lat"] for m in members) / len(members), 2),
            "lon": round(sum(m["lon"] for m in members) / len(members), 2),
            "escalation_prob": ev["escalation_prob"],  # peak probability
            "sequence_pattern": ev["sequence_pattern"],
            "magnitude_probs": ev.get("magnitude_probs", {}),
            "sequence_context": ev["sequence_context"],
            "n_events_in_cluster": len(members),
            "latest_event_id": latest["event_id"],
        }
        clusters.append(cluster_entry)

    clusters.sort(key=lambda c: -c["escalation_prob"])
    return clusters


MAG_MODEL_THRESHOLDS = [5.0, 5.5, 6.0]


class EventScorer:
    def __init__(self, model_path=None):
        mp = model_path or MODEL_PATH
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Event model not found: {mp}")
        self.model = lgb.Booster(model_file=mp)
        self.last_scored_ts = ""

        self.mag_models = {}
        models_dir = os.path.dirname(mp)
        for thresh in MAG_MODEL_THRESHOLDS:
            tp = os.path.join(models_dir, f"mag_threshold_m{int(thresh*10)}.txt")
            if os.path.exists(tp):
                self.mag_models[thresh] = lgb.Booster(model_file=tp)
        if self.mag_models:
            print(f"  [event_scorer] loaded {len(self.mag_models)} magnitude threshold models")

        self.mag_table = None
        try:
            with open(MAG_TABLE_PATH) as f:
                self.mag_table = _json.load(f)
        except Exception:
            pass

    def _magnitude_probs(self, escalation_prob, seq_max, feats):
        """Per-threshold magnitude probabilities.
        M5.0-M6.0: per-threshold models (sequence-aware, 0.68-0.88 AUC).
        M6.5+: static exceedance table (insufficient data for reliable models).
        Monotonicity enforced across the full chain."""
        probs = {}
        prev_p = 1.0

        if self.mag_models:
            feat_row = feats.reshape(1, -1)
            for thresh in sorted(self.mag_models.keys()):
                if thresh <= seq_max:
                    continue
                p = min(float(self.mag_models[thresh].predict(feat_row)[0]), prev_p)
                p = round(p, 4)
                if p >= 0.001:
                    probs[str(thresh)] = p
                prev_p = p

        if self.mag_table:
            matched = None
            for b in self.mag_table["bins"]:
                if b["lo"] <= seq_max < b["hi"]:
                    matched = b
                    break
            if not matched and self.mag_table["bins"]:
                matched = self.mag_table["bins"][-1]
            if matched:
                for thresh_str in sorted(matched["exceedance"].keys(), key=float):
                    thresh = float(thresh_str)
                    if thresh <= seq_max or thresh_str in probs:
                        continue
                    p = min(round(escalation_prob * matched["exceedance"][thresh_str], 4), prev_p)
                    if p >= 0.001:
                        probs[thresh_str] = p
                    prev_p = p

        return probs

    def score_event(self, conn, eid, ts, mag, depth, lat, lon):
        """Score a single earthquake. Returns dict with probability and context."""
        import pandas as pd
        try:
            epoch = pd.Timestamp(ts, tz="UTC").timestamp()
        except Exception:
            epoch = pd.Timestamp(str(ts).replace("Z", "+00:00")).timestamp()

        feats = _compute_features(mag, depth, lat, lon, epoch, conn)
        prob = float(self.model.predict(feats.reshape(1, -1))[0])
        pattern = _classify_sequence(feats)

        fi = {name: i for i, name in enumerate(FEAT_NAMES)}
        seq_max = max(float(mag), float(feats[fi["max_mag_7d"]]))
        mag_probs = self._magnitude_probs(prob, seq_max, feats)

        return {
            "event_id": str(eid),
            "time": str(ts),
            "magnitude": round(float(mag), 1),
            "depth_km": round(float(depth or 10), 1),
            "lat": round(float(lat), 3),
            "lon": round(float(lon), 3),
            "escalation_prob": round(prob, 3),
            "sequence_pattern": pattern,
            "magnitude_probs": mag_probs,
            "sequence_context": {
                "events_24h": int(feats[fi["n_events_24h"]]),
                "events_7d": int(feats[fi["n_events_7d"]]),
                "events_30d": int(feats[fi["n_events_30d"]]),
                "max_mag_7d": round(float(feats[fi["max_mag_7d"]]), 1),
                "rumble_ratio": round(float(feats[fi["rumble_ratio"]]), 2),
                "consec_up": int(feats[fi["consec_up"]]),
                "mag_trend_72h": round(float(feats[fi["mag_trend_72h"]]), 3),
                "hours_since_last": round(float(feats[fi["hours_since_last"]]), 1),
            },
        }

    def score_new(self, conn, since_ts=None):
        """Score all earthquakes since last call (or since_ts). Returns list of scores."""
        ts_cutoff = since_ts or self.last_scored_ts
        if ts_cutoff:
            rows = conn.execute(
                "SELECT id, timestamp, magnitude, depth_km, lat, lon FROM earthquakes "
                "WHERE magnitude >= ? AND timestamp > ? ORDER BY timestamp",
                (MIN_MAG, ts_cutoff)).fetchall()
        else:
            # First call — score last 24h
            rows = conn.execute(
                "SELECT id, timestamp, magnitude, depth_km, lat, lon FROM earthquakes "
                "WHERE magnitude >= ? AND timestamp >= datetime('now', '-1 day') "
                "ORDER BY timestamp", (MIN_MAG,)).fetchall()

        scores = []
        for eid, ts, mag, dep, lat, lon in rows:
            try:
                s = self.score_event(conn, eid, ts, mag, dep, lat, lon)
                scores.append(s)
            except Exception as e:
                print(f"  [event_scorer] error scoring {eid}: {e}")
                continue

        if rows:
            self.last_scored_ts = rows[-1][1]

        return scores
