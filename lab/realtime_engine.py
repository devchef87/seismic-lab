"""SeismicLab — realtime Tier-2 escalation engine (v2, event-driven sub-minute).

Architecture that makes it genuinely realtime without rescoring everything:
  - GLOBAL CACHE: solar/geomag/tidal/event signals change slowly -> compute once,
    refresh hourly.
  - EVENT-DRIVEN TICK (every ~60s): detect new M2.5+ events; rescore ONLY the
    affected cells (catalog features read live -> new foreshocks reflected at once),
    reusing cached globals. A few cells -> sub-second to seconds.
  - VELOCITY: per-cell probability trail -> RISING / FALLING / STEADY.
  - VOLCANIC OVERLAY: nearby real-time volcanic alert (USGS HANS) flagged as context.
  - Writes data/tier2_watch.json atomically for the UI.

Run:  python realtime_engine.py --tick 60        # event-driven, 60s ticks
      python realtime_engine.py --once           # single full pass (cron-friendly)
"""
import os, sys, json, time, argparse, tempfile
import numpy as np
import pandas as pd
import sqlite3
import importlib.util
import lightgbm as lgb
from sklearn.cluster import DBSCAN

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("te", os.path.join(_HERE, "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

MODEL = os.path.join(te.MODEL_DIR, "tier2_watch_lgb.txt")
CALIB = os.path.join(te.MODEL_DIR, "tier2_watch_calib.npz")
OUT_JSON = os.path.join(te.CACHE_DIR, "tier2_watch.json")
HISTORY_JSON = os.path.join(te.CACHE_DIR, "tier2_history.json")
CACHE_WINDOW_DAYS = 200    # history depth for trailing features
CACHE_REFRESH_H = 1.0      # rebuild global cache every hour
HISTORY_KEEP_H = 48
LEVEL_NAMES = ["NORMAL", "ADVISORY", "WATCH", "WARNING"]


def _load_bundle():
    model = lgb.Booster(model_file=MODEL)
    cz = np.load(CALIB, allow_pickle=True)
    ix, iy = cz["iso_x"], cz["iso_y"]; thr = cz["raw_thresholds"]; wp = float(cz["warn_prob"])
    base = float(cz["base_rate"])
    cal = lambda r: float(np.interp(r, ix, iy, left=iy[0], right=iy[-1]))
    def alert(r, c):
        return 3 if c >= wp else (2 if r >= thr[1] else (1 if r >= thr[0] else 0))
    return model, cal, alert, base


class GlobalCache:
    """Slow-changing global signals + the active-cell list. Rebuilt hourly."""
    def __init__(self):
        self.built_at = 0.0
        self.refresh()

    def refresh(self):
        conn = sqlite3.connect(te.DB_PATH, timeout=60)
        r = conn.execute("SELECT MAX(timestamp) FROM earthquakes").fetchone()
        end = pd.Timestamp(r[0], tz="UTC").floor("h")
        start = end - pd.Timedelta(days=CACHE_WINDOW_DAYS)
        self.hours = pd.date_range(start, end, freq="h", tz="UTC")
        self.hour_epochs = np.array([h.timestamp() for h in self.hours])
        self.sig = te.build_signal_features(conn, self.hours); self.sig.pop("_raw", None)
        self.evt = te.build_event_features(conn, self.hours, self.hour_epochs)
        rows = conn.execute("SELECT timestamp, value FROM samples WHERE metric='tidal_potential' "
                            "AND timestamp >= ? ORDER BY timestamp", (start.isoformat(),)).fetchall()
        tt, tv = [], []
        for ts, v in rows:
            try:
                tt.append(pd.Timestamp(ts, tz="UTC").timestamp()); tv.append(v)
            except Exception:
                pass
        self.tidal_times = np.array(tt); self.tidal_values = np.array(tv)
        # active cells (enough seismicity to model)
        self.cells = {}
        for c in te.generate_grid_cells(te.PARENT_ZONES):
            cnt = conn.execute("SELECT COUNT(*) FROM earthquakes WHERE magnitude>=? "
                               "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
                               (te.MIN_MAG_CATALOG, c["lat_range"][0], c["lat_range"][1],
                                c["lon_range"][0], c["lon_range"][1])).fetchone()[0]
            if cnt >= 100:
                self.cells[c["id"]] = c
        conn.close()
        self.built_at = time.time()
        print(f"  [cache] rebuilt: window ends {self.hours[-1]}, {len(self.cells)} active cells")


def _volcanic_alerts():
    """Recent real-time volcanic alerts (USGS HANS) -> list of (lat, lon, level, name)."""
    try:
        conn = sqlite3.connect(te.DB_PATH, timeout=10)
        rows = conn.execute("SELECT lat, lon, alert_level, volcano_name FROM volcano_alerts "
                            "WHERE alert_level IN ('WATCH','WARNING','ADVISORY')").fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _nearby_alert(cell, alerts):
    clat, clon = te._zone_center(cell)
    best = None
    for la, lo, lvl, name in alerts:
        if abs(la - clat) < 8 and abs(lo - clon) < 8 and te._haversine(clat, clon, la, lo) < 500:
            best = {"volcano": name, "level": lvl}
    return best


_NAMED_REGIONS = [  # friendly labels incl. gaps the fixed grid missed
    ("indonesia", -12, 8, 95, 140), ("japan_kurils", 25, 50, 128, 155),
    ("south_america", -60, 7, -82, -60), ("mexico_ca", 7, 25, -115, -77),
    ("himalaya", 25, 42, 60, 100), ("alaska", 50, 72, -180, -130),
    ("california", 32, 42, -125, -114), ("philippines", 8, 25, 117, 127),
    ("mediterranean", 34, 43, 19, 45), ("caribbean", 10, 20, -77, -60),
    ("new_zealand", -48, -34, 165, 180), ("png_solomon", -12, -3, 140, 160),
    ("kamchatka", 50, 62, 156, 168), ("tonga_fiji", -25, -14, 175, 180),
    ("tonga_fiji", -25, -14, -180, -173), ("vanuatu", -23, -11, 165, 172),
    ("iran", 25, 40, 44, 60), ("w_aleutians", 50, 56, 170, 180),
]

def _region_name(lat, lon):
    for name, a, b, c, d in _NAMED_REGIONS:
        if a <= lat <= b and c <= lon <= d:
            return name
    return f"{abs(lat):.0f}{'N' if lat >= 0 else 'S'}_{abs(lon):.0f}{'E' if lon >= 0 else 'W'}"


def detect_swarm_clusters(conn, hour_epochs, trail_h=None, eps_km=150.0, min_samples=None):
    """Cluster-centered swarm detection — replaces the fixed grid for serving.

    DBSCAN over recent small (M2.5-5) events ANYWHERE on Earth (haversine, so the
    antimeridian works); each dense cluster becomes a 10deg cell centered on its
    centroid. The region follows the swarm -> no grid gaps, no edge-splitting, and
    feature scale stays ~10deg to match training. Returns dynamic cell dicts.
    """
    trail_h = trail_h or te.TIER2_TRAIL_H
    min_samples = min_samples or te.TIER2_MIN_SMALL
    now = hour_epochs[-1]
    t0 = pd.Timestamp(now - trail_h * 3600, unit="s", tz="UTC").isoformat()
    rows = conn.execute(
        "SELECT id, timestamp, magnitude, depth_km, lat, lon, place FROM earthquakes "
        "WHERE magnitude >= ? AND magnitude < ? AND timestamp >= ? ",
        (te.TIER2_SMALL_MAG, te.MIN_MAG_TARGET, t0)).fetchall()
    # collapse duplicate catalog entries (same event from USGS + EMSC, or the legacy
    # emsc_ ingest path) so a swarm isn't double-counted; key on (time, lat~, lon~, mag),
    # prefer a USGS id when both exist
    _uniq = {}
    for r in rows:
        k = (r[1], round(float(r[4]), 2), round(float(r[5]), 2), round(float(r[2]), 1))
        if k not in _uniq or (str(r[0]).startswith("us") and not str(_uniq[k][0]).startswith("us")):
            _uniq[k] = r
    rows = list(_uniq.values())
    if len(rows) < min_samples:
        return []
    latlon = np.array([[r[4], r[5]] for r in rows], dtype=float)
    labels = DBSCAN(eps=eps_km / 6371.0, min_samples=min_samples,
                    metric="haversine").fit_predict(np.radians(latlon))
    cells = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue  # noise (isolated events, not a swarm)
        idx = np.where(labels == lbl)[0]
        m = latlon[idx]
        clat = float(m[:, 0].mean()); clon = float(m[:, 1].mean())
        # the exact small events the model clustered into this swarm — so the UI can
        # render precisely what the model saw (len == n_recent), most recent first
        quakes = sorted(
            [{"id": str(rows[i][0]), "time": str(rows[i][1]),
              "mag": round(float(rows[i][2]), 1),
              "depth_km": (round(abs(float(rows[i][3])), 1) if rows[i][3] is not None else None),
              "lat": round(float(rows[i][4]), 3), "lon": round(float(rows[i][5]), 3),
              "place": rows[i][6]} for i in idx],
            key=lambda q: q["time"], reverse=True)
        cells.append({
            "id": f"swarm_{clat:+.1f}_{clon:+.1f}", "parent": _region_name(clat, clon),
            # 10deg scoring footprint (INTERNAL — keeps feature scale = training; not for display)
            "lat_range": [clat - 5, clat + 5], "lon_range": [clon - 5, clon + 5],
            "centroid": [round(clat, 2), round(clon, 2)], "n_recent": int(len(idx)),
            # tight extent of the actual cluster quakes (for an optional small map area)
            "extent": [round(float(m[:, 0].min()), 2), round(float(m[:, 0].max()), 2),
                       round(float(m[:, 1].min()), 2), round(float(m[:, 1].max()), 2)],
            "quakes": quakes,  # the individual events behind n_recent (what the model saw)
        })
    return cells


def score_cell(conn, cell, cache, model, cal, alert):
    feats = te.build_cell_features(conn, cell, cache.hours, cache.hour_epochs,
                                   cache.sig, cache.evt, cache.tidal_times, cache.tidal_values)
    epi, _ = te.build_tier2_labels(conn, cell, cache.hour_epochs, te.MIN_MAG_TARGET)
    last = len(cache.hours) - 1
    if epi[last] < 0.5:
        return None  # no active swarm in this cell now
    raw = float(model.predict(feats[last:last + 1])[0])
    c = cal(raw)
    return {"raw": raw, "prob": c, "level": LEVEL_NAMES[alert(raw, c)]}


class Engine:
    def __init__(self, mode="cluster"):
        self.mode = mode  # "cluster" (dynamic, global) or "grid" (fixed zones)
        self.model, self.cal, self.alert, self.base = _load_bundle()
        self.cache = GlobalCache()
        self.history = self._load_history()
        self.last_event_ts = self._max_event_ts()
        self.full_rescore()

    def _cells(self, conn):
        """Cell set to score: dynamic swarm clusters (default) or the fixed grid."""
        if self.mode == "cluster":
            cells = detect_swarm_clusters(conn, self.cache.hour_epochs)
            return {c["id"]: c for c in cells}
        return self.cache.cells

    def _load_history(self):
        try:
            return json.load(open(HISTORY_JSON))
        except Exception:
            return {}

    def _max_event_ts(self):
        conn = sqlite3.connect(te.DB_PATH, timeout=30)
        r = conn.execute("SELECT MAX(timestamp) FROM earthquakes WHERE magnitude>=?",
                         (te.TIER2_SMALL_MAG,)).fetchone()[0]
        conn.close()
        return r or ""

    def _record(self, cid, prob, now_ep):
        h = [r for r in self.history.get(cid, []) if r[0] >= now_ep - HISTORY_KEEP_H * 3600]
        # velocity vs ~6h ago
        target = now_ep - 6 * 3600
        direction, delta, then = "NEW", 0.0, None
        if h:
            prior = min(h, key=lambda r: abs(r[0] - target))
            if abs(prior[0] - target) <= 6 * 3600:
                then = round(prior[1], 3); delta = round(prob - prior[1], 3)
                direction = "RISING" if delta > 0.01 else ("FALLING" if delta < -0.01 else "STEADY")
        h.append([now_ep, round(prob, 4)])
        self.history[cid] = h
        return then, delta, direction

    def full_rescore(self):
        conn = sqlite3.connect(te.DB_PATH, timeout=60)
        cells = self._cells(conn)
        self.watch_state = {}
        for cid, cell in cells.items():
            s = score_cell(conn, cell, self.cache, self.model, self.cal, self.alert)
            if s:
                self.watch_state[cid] = (cell, s)
        conn.close()
        self.write()

    def tick(self):
        # hourly global refresh of slow signals
        if time.time() - self.cache.built_at > CACHE_REFRESH_H * 3600:
            self.cache.refresh()
            self.full_rescore()
            return

        if self.mode == "cluster":
            # re-detect clusters + rescore (cluster set is dynamic; detection is cheap)
            conn = sqlite3.connect(te.DB_PATH, timeout=60)
            new = conn.execute("SELECT MAX(timestamp) FROM earthquakes WHERE magnitude>=?",
                               (te.TIER2_SMALL_MAG,)).fetchone()[0]
            conn.close()
            if new and new != self.last_event_ts:
                self.last_event_ts = new
                self.full_rescore()
                print(f"  [tick] new events -> re-detected swarms ({len(self.watch_state)} active)")
            return

        # grid mode: event-driven rescore of only the affected fixed cells
        conn = sqlite3.connect(te.DB_PATH, timeout=60)
        new = conn.execute("SELECT lat, lon, timestamp FROM earthquakes WHERE magnitude>=? "
                           "AND timestamp > ? ORDER BY timestamp",
                           (te.TIER2_SMALL_MAG, self.last_event_ts)).fetchall()
        if new:
            self.last_event_ts = new[-1][2]
            affected = set(self.watch_state.keys())
            for la, lo, _ in new:
                for cid, cell in self.cache.cells.items():
                    if (cell["lat_range"][0] <= la <= cell["lat_range"][1] and
                            cell["lon_range"][0] <= lo <= cell["lon_range"][1]):
                        affected.add(cid)
            for cid in affected:
                s = score_cell(conn, self.cache.cells[cid], self.cache, self.model, self.cal, self.alert)
                if s:
                    self.watch_state[cid] = (self.cache.cells[cid], s)
                else:
                    self.watch_state.pop(cid, None)
            print(f"  [tick] {len(new)} new events -> rescored {len(affected)} cells")
            self.write()
        conn.close()

    def write(self):
        now_ep = pd.Timestamp(self.cache.hours[-1]).timestamp()
        alerts = _volcanic_alerts()
        watch = []
        for cid, (cell, s) in self.watch_state.items():
            then, delta, direction = self._record(cid, s["prob"], now_ep)
            va = _nearby_alert(cell, alerts)
            watch.append({
                "cell": cid, "zone": str(cell.get("parent", cid)),
                "lat_range": cell.get("lat_range"), "lon_range": cell.get("lon_range"),
                "escalation_prob_72h": round(s["prob"], 3), "alert_level": s["level"],
                "lift_vs_base": round(s["prob"] / max(self.base, 1e-6), 1),
                "prob_6h_ago": then, "trend_6h": delta, "direction": direction,
                "nearby_volcanic_alert": va,
                "centroid": cell.get("centroid"), "n_recent_quakes": cell.get("n_recent"),
                "extent": cell.get("extent"),  # [minlat,maxlat,minlon,maxlon] of cluster quakes
                "quakes": cell.get("quakes"),  # exact events the model clustered (len == n_recent_quakes)
            })
        watch.sort(key=lambda w: -w["escalation_prob_72h"])
        self.history = {c: [r for r in h if r[0] >= now_ep - HISTORY_KEEP_H * 3600]
                        for c, h in self.history.items()}
        payload = {"generated": str(self.cache.hours[-1]), "base_rate_72h": round(self.base, 4),
                   "n_active_swarms": len(watch),
                   "alert_counts": {lv: sum(1 for w in watch if w["alert_level"] == lv) for lv in LEVEL_NAMES},
                   "watch": watch}
        _js = lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o))
        fd, tmp = tempfile.mkstemp(dir=te.CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=_js)
        os.replace(tmp, OUT_JSON)
        try:
            json.dump({c: h for c, h in self.history.items() if h}, open(HISTORY_JSON, "w"))
        except Exception:
            pass
        top = watch[0] if watch else None
        print(f"  [{payload['generated']}] {len(watch)} swarms {payload['alert_counts']}"
              + (f" | top {top['zone']} {top['escalation_prob_72h']} {top['direction']}" if top else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick", type=int, default=120, help="seconds between ticks")
    ap.add_argument("--mode", choices=["cluster", "grid"], default="cluster",
                    help="cluster = dynamic global swarm detection (default); grid = fixed zones")
    ap.add_argument("--once", action="store_true", help="single full pass then exit")
    args = ap.parse_args()
    if not (os.path.exists(MODEL) and os.path.exists(CALIB)):
        sys.exit(f"Model bundle missing — run tier2_watch.py first ({MODEL})")
    eng = Engine(mode=args.mode)
    if args.once:
        return
    print(f"  realtime engine live — {args.mode} mode, {args.tick}s ticks")
    while True:
        time.sleep(args.tick)
        try:
            eng.tick()
        except Exception as e:
            import traceback; print(f"  [ERROR] {e}"); traceback.print_exc()


if __name__ == "__main__":
    main()
