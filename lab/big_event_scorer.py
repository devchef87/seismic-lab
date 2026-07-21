"""Big-event (M5.5+/M6+) regional watch scorer.

Answers a different question than the escalation scorer: not "will this
sequence grow by M+1.0" but "will an M6+ occur within 100km of here in the
next 30 days". Trained with fixed-radius labels (scripts/train_big_event_model.py).

Honest performance (temporal holdout, first-big-event context — no M5+ within
100km in prior 30d): M6.0 AUC 0.715, top-0.1% alerts hit at 14.9% vs 1.4%
base rate (11x lift). This is screening, not prediction — surface sparingly.

Feature vector: 25 catalog/sequence features (shared with event_scorer) +
16 regional/physics features (100km/30d window + leakage-safe historical
priors). Order must match scripts/train_big_event_model.py ALL_FEAT_NAMES.
"""
import os, math
import numpy as np
import lightgbm as lgb

from lab.event_scorer import (_compute_features as _catalog_features,
                              load_calibration, apply_calibration)

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(_HERE), "models")

RADIUS_KM = 100.0
WINDOW_DAYS = 30
MIN_MAG = 2.5
FIRST_EVENT_GATE_MAG = 5.0
CATALOG_T0 = 1420070400.0        # 2015-01-01, matches training t_start
_B_CONST = math.log10(math.e)

THRESHOLDS = [5.5, 6.0]

# Watch bands on the M6.0 probability (see BIG_EVENT_WATCH_BRIEF.md).
# Defaults are raw-score bands from the pre-calibration era; when a
# calibration file exists (scripts/fit_calibration.py), bands come from it
# and probabilities are calibrated empirical frequencies.
WATCH_PROB = 0.30
ELEVATED_PROB = 0.55
_calib = load_calibration()
if _calib:
    import numpy as _np
    try:
        _z = _np.load(os.path.join(MODELS_DIR, "probability_calibration.npz"))
        if "be60_watch" in _z.files:
            WATCH_PROB = float(_z["be60_watch"])
            ELEVATED_PROB = float(_z["be60_elevated"])
    except Exception:
        pass

REGIONAL_FEAT_NAMES = [
    "r100_n_7d", "r100_n_30d", "r100_max_mag_30d", "r100_n_m4_30d",
    "r100_mag_range_30d", "r100_hours_since_m4",
    "r100_b_30d", "b_delta_bg", "drift_speed", "drift_approach",
    "quiescence", "dt_accel", "r100_depth_trend",
    "hist_m6_frac", "hist_m55_frac", "rate_anom_30d",
]


def _lon_conds(lo, hi):
    """SQL condition for a longitude range, handling +/-180 wrap."""
    if lo >= -180 and hi <= 180:
        return "lon >= ? AND lon < ?", [lo, hi]
    if lo < -180:
        return "(lon >= ? OR lon < ?)", [lo + 360, hi]
    return "(lon >= ? OR lon < ?)", [lo, hi - 360]


def _historical_priors(conn, lat, lon, ts):
    """Leakage-safe priors from the 3x3 1-degree cell neighborhood, matching
    the training grid: all M4+ events strictly before ts."""
    lat_lo = math.floor(lat) - 1
    lat_hi = math.floor(lat) + 2
    lon_lo = math.floor(lon) - 1
    lon_hi = math.floor(lon) + 2
    lc, lp = _lon_conds(lon_lo, lon_hi)
    row = conn.execute(
        f"SELECT COUNT(*), SUM(magnitude >= 5.5), SUM(magnitude >= 6.0), "
        f"AVG(magnitude) FROM earthquakes "
        f"WHERE magnitude >= 4.0 AND timestamp < ? "
        f"AND lat >= ? AND lat < ? AND {lc}",
        [ts, lat_lo, lat_hi] + lp).fetchone()
    n4 = row[0] or 0
    m6_frac = m55_frac = b_bg = 0.0
    if n4 >= 50:
        m55_frac = (row[1] or 0) / n4
        m6_frac = (row[2] or 0) / n4
    if n4 >= 30 and row[3]:
        b_bg = _B_CONST / (row[3] - 4.0 + 0.05)
    return n4, m6_frac, m55_frac, b_bg


def _regional_features(conn, lat, lon, epoch, ts):
    """The 16 regional/physics features for one event."""
    import pandas as pd
    from datetime import datetime, timezone
    feats = np.zeros(len(REGIONAL_FEAT_NAMES), dtype=np.float32)
    fi = {name: i for i, name in enumerate(REGIONAL_FEAT_NAMES)}

    n4_hist, m6_frac, m55_frac, b_bg = _historical_priors(conn, lat, lon, ts)
    feats[fi["hist_m6_frac"]] = m6_frac
    feats[fi["hist_m55_frac"]] = m55_frac

    dlat_deg = RADIUS_KM / 111.0
    coslat = math.cos(math.radians(lat))
    dlon_deg = RADIUS_KM / (111.0 * max(0.1, coslat))
    ts_lo = datetime.fromtimestamp(epoch - WINDOW_DAYS * 86400,
                                   tz=timezone.utc).isoformat()
    lc, lp = _lon_conds(lon - dlon_deg, lon + dlon_deg)
    rows = conn.execute(
        f"SELECT timestamp, magnitude, depth_km, lat, lon FROM earthquakes "
        f"WHERE magnitude >= ? AND timestamp >= ? AND timestamp < ? "
        f"AND lat BETWEEN ? AND ? AND {lc} ORDER BY timestamp",
        [MIN_MAG, ts_lo, ts, lat - dlat_deg, lat + dlat_deg] + lp).fetchall()

    nb_t, nb_m, nb_d, nb_x, nb_y = [], [], [], [], []
    for rts, m, d, rla, rlo in rows:
        try:
            t = pd.Timestamp(rts, tz="UTC").timestamp()
        except Exception:
            continue
        if t >= epoch:
            continue
        y = (rla - lat) * 111.0
        dlo_raw = rlo - lon
        if dlo_raw > 180: dlo_raw -= 360
        elif dlo_raw < -180: dlo_raw += 360
        x = dlo_raw * 111.0 * coslat
        if x * x + y * y > RADIUS_KM ** 2:
            continue
        nb_t.append(t); nb_m.append(m); nb_d.append(d or 10.0)
        nb_x.append(x); nb_y.append(y)

    n30 = len(nb_t)
    if n30 == 0:
        feats[fi["r100_hours_since_m4"]] = 999.0
        return feats

    nb_t = np.array(nb_t); nb_m = np.array(nb_m, dtype=np.float32)
    nb_d = np.array(nb_d, dtype=np.float32)
    nb_x = np.array(nb_x); nb_y = np.array(nb_y)
    dt = epoch - nb_t
    n7 = int((dt <= 7 * 86400).sum())
    m4 = nb_m >= 4.0

    feats[fi["r100_n_7d"]] = n7
    feats[fi["r100_n_30d"]] = n30
    feats[fi["r100_max_mag_30d"]] = nb_m.max()
    feats[fi["r100_n_m4_30d"]] = m4.sum()
    feats[fi["r100_mag_range_30d"]] = nb_m.max() - nb_m.min()
    feats[fi["r100_hours_since_m4"]] = dt[m4].min() / 3600.0 if m4.any() else 999.0

    above = nb_m[m4]
    b_now = _B_CONST / (above.mean() - 4.0 + 0.05) if len(above) >= 10 else 0.0
    feats[fi["r100_b_30d"]] = b_now
    if b_now > 0 and b_bg > 0:
        feats[fi["b_delta_bg"]] = b_now - b_bg

    if n30 >= 6:
        mid = n30 // 2
        ox, oy = nb_x[:mid].mean(), nb_y[:mid].mean()
        nx, ny = nb_x[mid:].mean(), nb_y[mid:].mean()
        dt_c = (nb_t[mid:].mean() - nb_t[:mid].mean()) / 86400.0
        if dt_c > 0.1:
            feats[fi["drift_speed"]] = math.hypot(nx - ox, ny - oy) / dt_c
        feats[fi["drift_approach"]] = math.hypot(ox, oy) - math.hypot(nx, ny)

    rate_prior = (n30 - n7) / 23.0
    rate_recent = n7 / 7.0
    feats[fi["quiescence"]] = (rate_prior - rate_recent) / (rate_prior + 1.0)

    if n30 >= 8:
        gaps = np.diff(nb_t)
        recent = gaps[nb_t[1:] >= epoch - 7 * 86400]
        prior = gaps[nb_t[1:] < epoch - 7 * 86400]
        if len(recent) >= 3 and len(prior) >= 3:
            med_p = np.median(prior)
            if med_p > 0:
                feats[fi["dt_accel"]] = min(float(np.median(recent) / med_p), 10.0)

    if n30 >= 5:
        x = np.arange(n30, dtype=np.float32)
        xm = x.mean()
        denom = ((x - xm) ** 2).sum()
        if denom > 0:
            feats[fi["r100_depth_trend"]] = \
                ((x - xm) * (nb_d - nb_d.mean())).sum() / denom

    age_days = (epoch - CATALOG_T0) / 86400.0
    if n4_hist > 0 and age_days > 60:
        expected_30d = n4_hist * 30.0 / age_days
        feats[fi["rate_anom_30d"]] = min(m4.sum() / (expected_30d + 0.1), 100.0)

    return feats


# Ordinal model tail mapping: class k = follower-magnitude bin
# [<5.0, 5.0-5.5, 5.5-6.0, 6.0-6.5, >=6.5]; P(>=thresh) = sum of tail classes.
ORDINAL_PATH = os.path.join(MODELS_DIR, "big_event_ordinal.txt")
ORDINAL_CALIB_PATH = os.path.join(MODELS_DIR, "big_event_ordinal_calib.npz")
ORDINAL_TAILS = {"5.5": 2, "6.0": 3, "6.5": 4}


class BigEventScorer:
    def __init__(self):
        self.ordinal = None
        self.ordinal_calib = {}
        if os.path.exists(ORDINAL_PATH):
            self.ordinal = lgb.Booster(model_file=ORDINAL_PATH)
            try:
                z = np.load(ORDINAL_CALIB_PATH)
                for k in z.files:
                    if k.endswith("_x") and k[:-2] + "_y" in z.files:
                        self.ordinal_calib[k[:-2]] = (z[k], z[k[:-2] + "_y"])
                global WATCH_PROB, ELEVATED_PROB
                if "be60_watch" in z.files:
                    WATCH_PROB = float(z["be60_watch"])
                    ELEVATED_PROB = float(z["be60_elevated"])
            except Exception:
                pass
            print(f"  [big_event] loaded ordinal model "
                  f"({len(self.ordinal_calib)} tail calibrations, "
                  f"watch>={WATCH_PROB:.3f})")

        self.models = {}
        if not self.ordinal:
            for thresh in THRESHOLDS:
                path = os.path.join(MODELS_DIR, f"big_event_m{int(thresh*10)}.txt")
                if os.path.exists(path):
                    self.models[thresh] = lgb.Booster(model_file=path)
            if not self.models:
                raise FileNotFoundError("No big-event models found")
            print(f"  [big_event] loaded {len(self.models)} big-event models "
                  f"({sorted(self.models.keys())})")

    def score_event(self, conn, ts, mag, depth, lat, lon):
        """Score one earthquake for big-event probability. Returns dict:
        probs per threshold, first_event flag, regional context."""
        import pandas as pd
        try:
            epoch = pd.Timestamp(ts, tz="UTC").timestamp()
        except Exception:
            epoch = pd.Timestamp(str(ts).replace("Z", "+00:00")).timestamp()

        cat = _catalog_features(mag, depth, lat, lon, epoch, conn)
        reg = _regional_features(conn, lat, lon, epoch, str(ts))
        row = np.concatenate([cat, reg]).reshape(1, -1)

        fi = {name: i for i, name in enumerate(REGIONAL_FEAT_NAMES)}
        probs = {}
        if self.ordinal:
            cls_p = self.ordinal.predict(row)[0]      # (5,) class probs
            prev = 1.0
            for tkey, cidx in ORDINAL_TAILS.items():
                tail = float(cls_p[cidx:].sum())
                cal = apply_calibration(self.ordinal_calib,
                                        f"be{tkey.replace('.', '')}", tail)
                p = min(cal, prev)   # safety only — tails are monotone pre-calibration
                probs[tkey] = round(p, 3)
                prev = p
        else:
            prev = 1.0
            for thresh in sorted(self.models.keys()):
                raw = float(self.models[thresh].predict(row)[0])
                cal = apply_calibration(_calib, f"be{int(thresh*10)}", raw)
                p = min(cal, prev)
                probs[str(thresh)] = round(p, 3)
                prev = p

        return {
            "probs": probs,
            "first_event": bool(reg[fi["r100_max_mag_30d"]] < FIRST_EVENT_GATE_MAG),
            "regional_context": {
                "events_100km_30d": int(reg[fi["r100_n_30d"]]),
                "max_mag_100km_30d": round(float(reg[fi["r100_max_mag_30d"]]), 1),
                "quiescence": round(float(reg[fi["quiescence"]]), 3),
                "b_value_30d": round(float(reg[fi["r100_b_30d"]]), 2),
                "hist_m6_frac": round(float(reg[fi["hist_m6_frac"]]), 4),
            },
        }
