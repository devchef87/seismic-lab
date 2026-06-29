"""SeismicLab — production Tier-2 watch scoring job.

Runs on a schedule (hourly). Computes CURRENT-window features from the live DB,
gates to active swarms, scores + calibrates with the saved bundle, and writes
data/tier2_watch.json atomically for the UI to serve.

  python serve_tier2_watch.py            # score once (for cron)
  python serve_tier2_watch.py --loop 3600  # self-scheduling, every hour
  python serve_tier2_watch.py --no-refresh  # skip the recent-EMSC pull

Architecture: precompute -> static JSON. The UI never runs the model; it reads
the JSON. Nothing changes faster than ~hourly (72h swarm window), so hourly is ample.
"""
import os, sys, json, time, argparse, tempfile
import numpy as np
import pandas as pd
import importlib.util
import lightgbm as lgb

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("te", os.path.join(_HERE, "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

MODEL = os.path.join(te.MODEL_DIR, "tier2_watch_lgb.txt")
CALIB = os.path.join(te.MODEL_DIR, "tier2_watch_calib.npz")
OUT_JSON = os.path.join(te.CACHE_DIR, "tier2_watch.json")
HISTORY_JSON = os.path.join(te.CACHE_DIR, "tier2_history.json")  # per-cell prob trail -> velocity
BUILD_DAYS = 200   # recent window — enough history for all trailing features (90d Schuster, etc.)
HISTORY_KEEP_H = 48  # hours of probability history to retain per cell
LEVEL_NAMES = ["NORMAL", "ADVISORY", "WATCH", "WARNING"]


def _load_history():
    try:
        with open(HISTORY_JSON) as f:
            return json.load(f)
    except Exception:
        return {}


def _velocity(hist_cell, now_ep, cur_prob, window_h=6):
    """Trend of escalation prob vs ~window_h ago. Returns (prob_then, delta, direction)."""
    if not hist_cell:
        return None, 0.0, "NEW"
    target = now_ep - window_h * 3600
    prior = min(hist_cell, key=lambda r: abs(r[0] - target))
    if abs(prior[0] - target) > window_h * 3600:  # no comparable point
        return None, 0.0, "NEW"
    delta = cur_prob - prior[1]
    direction = "RISING" if delta > 0.01 else ("FALLING" if delta < -0.01 else "STEADY")
    return round(prior[1], 3), round(delta, 3), direction


def _load_bundle():
    model = lgb.Booster(model_file=MODEL)
    cz = np.load(CALIB, allow_pickle=True)
    iso_x, iso_y = cz["iso_x"], cz["iso_y"]
    thr = cz["raw_thresholds"]              # [ADVISORY_raw, WATCH_raw]
    warn_prob = float(cz["warn_prob"])
    base_rate = float(cz["base_rate"])
    calibrate = lambda raw: float(np.interp(raw, iso_x, iso_y, left=iso_y[0], right=iso_y[-1]))
    def alert(raw, cal):
        if cal >= warn_prob: return 3
        if raw >= thr[1]: return 2
        if raw >= thr[0]: return 1
        return 0
    return model, calibrate, alert, base_rate


def score_once(refresh_emsc=True, days=BUILD_DAYS):
    t0 = time.time()
    model, calibrate, alert, base_rate = _load_bundle()

    # 1. Keep the recent small-event catalog current (swarm gate needs M2.5+;
    #    USGS misses these offshore, so pull recent EMSC). Cheap for a short window.
    if refresh_emsc:
        try:
            from ingest.backfill import backfill_earthquakes_emsc, EMSC_ZONES
            from ingest.store import QuakeStore
            start_year = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).year
            backfill_earthquakes_emsc(QuakeStore(), start_year=start_year,
                                      end_year=pd.Timestamp.utcnow().year + 1)
        except Exception as e:
            print(f"  [warn] EMSC refresh skipped: {e}")

    # 2. Build features for the recent window (ends at the latest data in the DB)
    start_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    ds = te.build_full_dataset(te.HORIZON_HOURS, te.MIN_MAG_TARGET, use_grid=True,
                               start_date=start_date)
    cell_ids = list(ds["features"].keys())
    parents = ds["cell_parent_map"]
    geom = {c["id"]: c for c in te.generate_grid_cells(te.PARENT_ZONES)}
    hours = ds["hours"]; last = len(hours) - 1

    # 3. Gate to active swarms at the current hour, score the survivors + compute velocity
    now_ep = pd.Timestamp(hours[last]).timestamp()
    history = _load_history()
    watch = []
    for cid in cell_ids:
        if ds["episode"][cid][last] < 0.5:
            continue
        raw = float(model.predict(ds["features"][cid][last:last + 1])[0])
        cal = calibrate(raw)
        lvl = LEVEL_NAMES[alert(raw, cal)]
        g = geom.get(cid, {})
        prob_then, delta, direction = _velocity(history.get(cid, []), now_ep, cal)
        # append + prune this cell's history
        h = [r for r in history.get(cid, []) if r[0] >= now_ep - HISTORY_KEEP_H * 3600]
        h.append([now_ep, round(cal, 4)])
        history[cid] = h
        watch.append({
            "cell": cid, "zone": str(parents.get(cid, cid)),
            "lat_range": g.get("lat_range"), "lon_range": g.get("lon_range"),
            "escalation_prob_72h": round(cal, 3), "alert_level": lvl,
            "lift_vs_base": round(cal / max(base_rate, 1e-6), 1),
            "prob_6h_ago": prob_then, "trend_6h": delta, "direction": direction,
        })
    watch.sort(key=lambda w: -w["escalation_prob_72h"])
    # persist history (only keep cells seen recently)
    history = {c: [r for r in h if r[0] >= now_ep - HISTORY_KEEP_H * 3600]
               for c, h in history.items()}
    history = {c: h for c, h in history.items() if h}
    try:
        with open(HISTORY_JSON, "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"  [warn] history save failed: {e}")

    # 4. Atomic write
    payload = {"generated": str(hours[last]), "base_rate_72h": round(base_rate, 4),
               "n_active_swarms": len(watch),
               "alert_counts": {lv: sum(1 for w in watch if w["alert_level"] == lv)
                                for lv in LEVEL_NAMES},
               "watch": watch}
    def _js(o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        return str(o)
    fd, tmp = tempfile.mkstemp(dir=te.CACHE_DIR, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, default=_js)
    os.replace(tmp, OUT_JSON)

    top = watch[0] if watch else None
    print(f"  [{payload['generated']}] {len(watch)} active swarms "
          f"({payload['alert_counts']}) | top: "
          f"{top['zone']+' '+str(top['escalation_prob_72h']) if top else 'none'} "
          f"| {time.time()-t0:.0f}s -> {OUT_JSON}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between runs (0 = run once)")
    ap.add_argument("--no-refresh", action="store_true", help="skip recent-EMSC pull")
    ap.add_argument("--days", type=int, default=BUILD_DAYS)
    args = ap.parse_args()
    if not (os.path.exists(MODEL) and os.path.exists(CALIB)):
        sys.exit(f"Model bundle missing — run tier2_watch.py first ({MODEL})")
    while True:
        try:
            score_once(refresh_emsc=not args.no_refresh, days=args.days)
        except Exception as e:
            import traceback; print(f"  [ERROR] scoring failed: {e}"); traceback.print_exc()
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
