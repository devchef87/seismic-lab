#!/usr/bin/env python3
"""Forecast verification: replay the scorers over a past window and compare
against what actually happened.

Both production scorers are deterministic functions of the catalog as-of the
event time (all feature queries filter timestamp < t), so replaying them over
past events reconstructs exactly what the live system would have said — no
hindsight leaks into the scores. Outcomes are then read from the catalog:

  escalation model : did an event >= trigger+1.0 occur within GK(trigger)/7d?
  big-event model  : did an M5.5+/M6.0+ occur within 100km/30d?

Events whose outcome window extends past 'now' count as resolved if already
hit, otherwise 'pending' and excluded from calibration.

Usage:
  python3 -u scripts/verify_forecasts.py [--start 2026-07-01] [--end now]
Writes data/verification_report.json and prints a summary.
"""
import os, sys, math, json, sqlite3, argparse, time
from datetime import datetime, timezone
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)

from lab.event_scorer import EventScorer, _gk_radius
from lab.big_event_scorer import BigEventScorer, WATCH_PROB, ELEVATED_PROB

DB = os.path.join(ROOT, "data", "quakewatch.db")
OUT = os.path.join(ROOT, "data", "verification_report.json")
MIN_MAG = 2.5


def load_catalog(conn, ts_lo, ts_hi):
    rows = conn.execute(
        "SELECT timestamp, magnitude, lat, lon FROM earthquakes "
        "WHERE magnitude >= ? AND timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp", (MIN_MAG, ts_lo, ts_hi)).fetchall()
    ts, mags, lats, lons = [], [], [], []
    for t, m, la, lo in rows:
        try:
            ts.append(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
        except Exception:
            continue
        mags.append(m); lats.append(la); lons.append(lo)
    return (np.array(ts), np.array(mags, dtype=np.float32),
            np.array(lats, dtype=np.float32), np.array(lons, dtype=np.float32))


def max_mag_within(cat, epoch, lat, lon, radius_km, window_s):
    ts, mags, lats, lons = cat
    j0 = np.searchsorted(ts, epoch, side="right")
    j1 = np.searchsorted(ts, epoch + window_s, side="right")
    if j1 <= j0:
        return 0.0
    sl = slice(j0, j1)
    dla = (lats[sl] - lat) * 111.0
    dlo = (lons[sl] - lon) * 111.0 * math.cos(math.radians(lat))
    inr = dla * dla + dlo * dlo <= radius_km ** 2
    return float(mags[sl][inr].max()) if inr.any() else 0.0


def calib_table(pairs, bands):
    """pairs: list of (prob, outcome 0/1). Returns per-band realized rates."""
    out = []
    for lo, hi in bands:
        sel = [(p, o) for p, o in pairs if lo <= p < hi]
        if not sel:
            out.append({"band": f"{lo:.2f}-{hi:.2f}", "n": 0})
            continue
        rate = sum(o for _, o in sel) / len(sel)
        out.append({"band": f"{lo:.2f}-{hi:.2f}", "n": len(sel),
                    "mean_pred": round(float(np.mean([p for p, _ in sel])), 3),
                    "realized": round(rate, 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    end_ts = args.end or now.isoformat()
    start_ts = args.start

    conn = sqlite3.connect(DB, timeout=60)
    escorer = EventScorer()
    bscorer = BigEventScorer()

    events = conn.execute(
        "SELECT id, timestamp, magnitude, depth_km, lat, lon FROM earthquakes "
        "WHERE magnitude >= ? AND timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp", (MIN_MAG, start_ts, end_ts)).fetchall()
    print(f"Replaying {len(events):,} events  [{start_ts} .. {end_ts[:16]}]")

    # Outcome catalog: needs to extend 30d past the last scored event (bounded by now)
    cat = load_catalog(conn, start_ts, now.isoformat())
    now_epoch = now.timestamp()

    esc_pairs, m55_pairs, m6_pairs = [], [], []
    flagged = []          # events with p(M6) >= WATCH_PROB
    m6_actual = []        # actual M6+ events in the window
    records = []
    t0 = time.time()

    for k, (eid, ets, mag, dep, lat, lon) in enumerate(events):
        try:
            es = escorer.score_event(conn, eid, ets, mag, dep, lat, lon)
            be = bscorer.score_event(conn, ets, mag, dep, lat, lon)
        except Exception as e:
            print(f"  skip {eid}: {e}")
            continue
        epoch = datetime.fromisoformat(ets.replace("Z", "+00:00")).timestamp()

        # Outcomes
        gk = _gk_radius(mag)
        esc_max = max_mag_within(cat, epoch, lat, lon, gk, 7 * 86400)
        esc_hit = esc_max >= mag + 1.0
        esc_resolved = esc_hit or (epoch + 7 * 86400 <= now_epoch)

        big_max = max_mag_within(cat, epoch, lat, lon, 100.0, 30 * 86400)
        m55_hit = big_max >= 5.5
        m6_hit = big_max >= 6.0
        big_resolved = epoch + 30 * 86400 <= now_epoch

        if esc_resolved:
            esc_pairs.append((es["escalation_prob"], int(esc_hit)))
        if m55_hit or big_resolved:
            m55_pairs.append((float(be["probs"].get("5.5", 0)), int(m55_hit)))
        if m6_hit or big_resolved:
            m6_pairs.append((float(be["probs"].get("6.0", 0)), int(m6_hit)))

        p6 = float(be["probs"].get("6.0", 0))
        rec = {"id": str(eid), "time": ets, "mag": round(float(mag), 1),
               "lat": round(float(lat), 2), "lon": round(float(lon), 2),
               "esc_prob": es["escalation_prob"], "esc_hit": bool(esc_hit),
               "m6_prob": round(p6, 3), "m6_hit": bool(m6_hit),
               "m55_hit": bool(m55_hit),
               "first_event": be["first_event"],
               "resolved": bool(big_resolved)}
        records.append(rec)
        if p6 >= WATCH_PROB:
            flagged.append(rec)
        if mag >= 6.0:
            m6_actual.append(rec)

        if (k + 1) % 1000 == 0:
            print(f"  {k+1:,}/{len(events):,} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    conn.close()
    print(f"Replay done in {time.time()-t0:.0f}s")

    # ── Recall: for each actual M6+, was the region flagged in the prior 30d?
    m6_recall = []
    for q in m6_actual:
        qt = datetime.fromisoformat(q["time"].replace("Z", "+00:00")).timestamp()
        pre = [f for f in flagged
               if f["id"] != q["id"]
               and 0 < qt - datetime.fromisoformat(
                   f["time"].replace("Z", "+00:00")).timestamp() <= 30 * 86400
               and math.hypot((f["lat"] - q["lat"]) * 111.0,
                              (f["lon"] - q["lon"]) * 111.0 *
                              math.cos(math.radians(q["lat"]))) <= 100.0]
        first = min(pre, key=lambda f: f["time"]) if pre else None
        lead_h = (qt - datetime.fromisoformat(
            first["time"].replace("Z", "+00:00")).timestamp()) / 3600.0 if first else None
        m6_recall.append({"quake": q, "flagged_before": bool(pre),
                          "n_prior_flags": len(pre),
                          "lead_hours": round(lead_h, 1) if lead_h else None})

    bands = [(0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.55), (0.55, 1.01)]
    esc_bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]

    watch_res = [(p, o) for p, o in m6_pairs if p >= WATCH_PROB]
    report = {
        "generated": now.isoformat(),
        "window": {"start": start_ts, "end": end_ts},
        "n_events_scored": len(records),
        "escalation_calibration": calib_table(esc_pairs, esc_bands),
        "m55_calibration": calib_table(m55_pairs, bands),
        "m6_calibration": calib_table(m6_pairs, bands),
        "m6_watch_precision": {
            "threshold": WATCH_PROB,
            "n_flagged_events": len(flagged),
            "n_resolved": len(watch_res),
            "precision": round(sum(o for _, o in watch_res) /
                               max(len(watch_res), 1), 3),
        },
        "m6_actual": m6_recall,
        "flagged_events": flagged,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary
    print("\n" + "=" * 70)
    print("  FORECAST VERIFICATION" + f"  [{start_ts} .. {end_ts[:10]}]")
    print("=" * 70)
    print(f"\n  Escalation model calibration (resolved, n={len(esc_pairs):,}):")
    for b in report["escalation_calibration"]:
        if b["n"]:
            print(f"    pred {b['band']}  n={b['n']:>6,}  "
                  f"mean_pred={b['mean_pred']:.3f}  realized={b['realized']:.3f}")
    print(f"\n  Big-event M6 calibration (resolved, n={len(m6_pairs):,}):")
    for b in report["m6_calibration"]:
        if b["n"]:
            print(f"    pred {b['band']}  n={b['n']:>6,}  "
                  f"mean_pred={b['mean_pred']:.3f}  realized={b['realized']:.3f}")
    wp = report["m6_watch_precision"]
    print(f"\n  M6 watch (p >= {WATCH_PROB}): {wp['n_flagged_events']} flagged events, "
          f"{wp['n_resolved']} resolved, precision {wp['precision']:.1%}")
    print(f"\n  Actual M6+ events in window: {len(m6_recall)}")
    for r in m6_recall:
        q = r["quake"]
        tag = (f"FLAGGED {r['n_prior_flags']}x, first {r['lead_hours']:.0f}h before"
               if r["flagged_before"] else "not flagged")
        print(f"    M{q['mag']} {q['time'][:16]} ({q['lat']:.1f},{q['lon']:.1f})  {tag}")
    print(f"\n  Report: {OUT}")


if __name__ == "__main__":
    main()
