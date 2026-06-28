"""SeismicLab Lab — DART Case Study Analysis

Analyze DART buoy pressure signals before M6.5+ earthquakes.
Tests hypothesis: pre-seismic deformation creates a compression→extension
pressure pattern on nearby seafloor pressure sensors.

Venezuela M7.5 (2026-06-24) showed: 48h trend reversed from +23 to -41 mm/day
in 24 hours, with the closest station (42407, 537km) having an entirely local
signal (only 0.7% correlation with Atlantic basin reference).

Run:  python3 -m lab.dart_case_study [--min-mag 6.5] [--max-dist 1500]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings("ignore")

UTC = timezone.utc
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "seismiclab.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def detrend_and_analyze(conn, station_id, target_time, window_days=10):
    """Full detrending analysis for a DART station around a target time."""
    t_start = target_time - timedelta(days=window_days)
    t_end = target_time + timedelta(hours=6)

    rows = conn.execute(
        "SELECT timestamp, mode, height_m FROM dart_readings "
        "WHERE station_id = ? AND timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp",
        (station_id,
         t_start.strftime("%Y-%m-%d %H:%M:%S"),
         t_end.strftime("%Y-%m-%d %H:%M:%S"))
    ).fetchall()

    normal = []
    event_count = 0
    for r in rows:
        ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        if r["mode"] == 1:
            normal.append((ts, r["height_m"]))
        if r["mode"] in (2, 3) and ts <= target_time + timedelta(hours=2):
            event_count += 1

    if len(normal) < 96:
        return None

    t0 = normal[0][0]
    hours = np.array([(t - t0).total_seconds() / 3600 for t, h in normal])
    heights = np.array([h for t, h in normal])
    times = [t for t, h in normal]

    tidal_periods = [12.42, 12.00, 23.93, 25.82, 6.21]
    A = np.zeros((len(hours), 2 + 2 * len(tidal_periods)))
    A[:, 0] = 1
    A[:, 1] = hours
    for i, T in enumerate(tidal_periods):
        A[:, 2 + 2 * i] = np.sin(2 * np.pi * hours / T)
        A[:, 2 + 2 * i + 1] = np.cos(2 * np.pi * hours / T)

    coeffs, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)
    residuals = (heights - A @ coeffs) * 1000  # mm

    rms = np.std(residuals)

    result = {
        "rms_mm": rms,
        "event_mode_count": event_count,
        "n_readings": len(normal),
    }

    for hours_before in [72, 48, 24, 12, 6, 0]:
        t = target_time - timedelta(hours=hours_before)

        t_24h_ago = t - timedelta(hours=24)
        t_48h_ago = t - timedelta(hours=48)

        last_24h = [residuals[i] for i in range(len(times)) if t_24h_ago <= times[i] <= t]
        last_48h = [residuals[i] for i in range(len(times)) if t_48h_ago <= times[i] <= t]

        if len(last_24h) > 4:
            result[f"mean24h_T{hours_before}h"] = float(np.mean(last_24h))
        if len(last_48h) > 4:
            x = np.arange(len(last_48h))
            slope = np.polyfit(x, last_48h, 1)[0]
            rpd = len(last_48h) / 2.0
            result[f"trend48h_T{hours_before}h"] = float(slope * rpd)

    t48 = result.get("trend48h_T48h", 0)
    t12 = result.get("trend48h_T12h", 0)
    t0v = result.get("trend48h_T0h", 0)
    result["trend_reversal_48_12"] = t12 - t48
    result["trend_reversal_48_0"] = t0v - t48
    result["max_abs_trend"] = max(
        abs(result.get(f"trend48h_T{h}h", 0))
        for h in [72, 48, 24, 12, 6, 0]
    )

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-mag", type=float, default=6.5)
    parser.add_argument("--max-dist", type=float, default=1500)
    parser.add_argument("--baseline-samples", type=int, default=30)
    args = parser.parse_args()

    conn = get_conn()

    dart_stations = conn.execute(
        "SELECT station_id, lat, lon, depth_m, region FROM dart_stations"
    ).fetchall()

    dart_ranges = {}
    for r in conn.execute(
        "SELECT station_id, MIN(timestamp), MAX(timestamp), COUNT(*) "
        "FROM dart_readings WHERE mode = 1 GROUP BY station_id"
    ).fetchall():
        dart_ranges[r["station_id"]] = (r[1], r[2], r[3])

    events = conn.execute(
        "SELECT timestamp, lat, lon, magnitude, depth_km, place "
        "FROM earthquakes WHERE magnitude >= ? ORDER BY timestamp",
        (args.min_mag,)
    ).fetchall()

    print(f"\n{'=' * 80}")
    print(f"  DART PRE-SEISMIC PRESSURE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"  Min magnitude: {args.min_mag}")
    print(f"  Max station distance: {args.max_dist} km")
    print(f"  Total M{args.min_mag}+ events: {len(events)}")
    print(f"  DART stations with data: {len(dart_ranges)}")

    matches = []
    for ev in events:
        best_sid, best_dist = None, 9999
        for ds in dart_stations:
            dist = haversine_km(ev["lat"], ev["lon"], ds["lat"], ds["lon"])
            if dist < best_dist:
                best_dist = dist
                best_sid = ds["station_id"]

        if best_dist > args.max_dist or best_sid not in dart_ranges:
            continue

        min_t, max_t, cnt = dart_ranges[best_sid]
        ev_date = ev["timestamp"][:10]
        ev_7d_before = (datetime.strptime(ev_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        if min_t[:10] <= ev_7d_before and max_t[:10] >= ev_date and cnt > 500:
            matches.append((ev, best_sid, best_dist))

    print(f"  Events with DART coverage: {len(matches)}")

    if not matches:
        print("  No events with sufficient DART data.")
        conn.close()
        return

    print(f"\n{'─' * 80}")
    print(f"  CASE-BY-CASE ANALYSIS")
    print(f"{'─' * 80}")

    event_results = []

    for ev, sid, dist in matches:
        ev_time = datetime.strptime(ev["timestamp"][:19].replace("T", " "), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        region = next((d["region"] for d in dart_stations if d["station_id"] == sid), "?")

        result = detrend_and_analyze(conn, sid, ev_time)
        if not result:
            continue

        print(f"\n  M{ev['magnitude']:.1f} {ev['timestamp'][:16]}  {(ev['place'] or '?')[:45]}")
        print(f"  DART {sid} ({region}) at {dist:.0f}km  |  RMS={result['rms_mm']:.1f}mm  "
              f"event_mode={result['event_mode_count']}")

        print(f"    {'Lead':>6s}  {'Trend48h':>10s}  {'Mean24h':>10s}")
        for h in [72, 48, 24, 12, 6, 0]:
            trend = result.get(f"trend48h_T{h}h", float("nan"))
            mean = result.get(f"mean24h_T{h}h", float("nan"))
            marker = ""
            if not np.isnan(trend) and abs(trend) > 2 * result["rms_mm"]:
                marker = "  ◀ >2σ"
            print(f"    T-{h:2d}h  {trend:+10.1f}  {mean:+10.1f}{marker}")

        print(f"    Trend reversal (T-48→T-12): {result['trend_reversal_48_12']:+.1f} mm/day")
        print(f"    Max |trend48h|: {result['max_abs_trend']:.1f} mm/day")

        event_results.append({
            "label": f"M{ev['magnitude']:.1f} {ev['timestamp'][:10]}",
            "mag": ev["magnitude"],
            "dist_km": dist,
            "station": sid,
            **result,
        })

    if len(event_results) < 2:
        print("\n  Insufficient events for statistical analysis.")
        conn.close()
        return

    print(f"\n{'─' * 80}")
    print(f"  BASELINE COMPARISON (random quiet periods)")
    print(f"{'─' * 80}")

    import random
    random.seed(42)

    baseline_results = []
    stations_used = set(r["station"] for r in event_results)

    for sid in stations_used:
        if sid not in dart_ranges:
            continue
        min_t_str, max_t_str, cnt = dart_ranges[sid]
        min_t = datetime.strptime(min_t_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        max_t = datetime.strptime(max_t_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        total_hours = int((max_t - min_t).total_seconds() / 3600)

        if total_hours < 500:
            continue

        event_times = [
            datetime.strptime(ev["timestamp"][:19].replace("T", " "), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            for ev, s, d in matches if s == sid
        ]

        n = 0
        for _ in range(args.baseline_samples * 3):
            if n >= args.baseline_samples:
                break
            offset = random.randint(10 * 24, total_hours - 24)
            t_sample = min_t + timedelta(hours=offset)

            too_close = any(abs((t_sample - et).total_seconds()) < 14 * 86400
                           for et in event_times)
            if too_close:
                continue

            result = detrend_and_analyze(conn, sid, t_sample)
            if result and result["max_abs_trend"] > 0:
                baseline_results.append(result)
                n += 1

    print(f"  Collected {len(baseline_results)} baseline samples")

    if len(baseline_results) < 10:
        print("  Insufficient baseline samples.")
        conn.close()
        return

    ev_max_trends = [r["max_abs_trend"] for r in event_results]
    bl_max_trends = [r["max_abs_trend"] for r in baseline_results]
    ev_reversals = [abs(r["trend_reversal_48_12"]) for r in event_results]
    bl_reversals = [abs(r["trend_reversal_48_12"]) for r in baseline_results]

    print(f"\n{'─' * 80}")
    print(f"  STATISTICAL SUMMARY")
    print(f"{'─' * 80}")

    print(f"\n  Max |trend48h| across lead times:")
    print(f"    Pre-event (n={len(ev_max_trends)}): "
          f"mean={np.mean(ev_max_trends):.1f}, med={np.median(ev_max_trends):.1f}, "
          f"std={np.std(ev_max_trends):.1f}")
    print(f"    Baseline  (n={len(bl_max_trends)}): "
          f"mean={np.mean(bl_max_trends):.1f}, med={np.median(bl_max_trends):.1f}, "
          f"std={np.std(bl_max_trends):.1f}")
    ratio = np.mean(ev_max_trends) / np.mean(bl_max_trends) if np.mean(bl_max_trends) > 0 else 0
    print(f"    Ratio: {ratio:.2f}x")

    print(f"\n  |Trend reversal T-48→T-12|:")
    print(f"    Pre-event (n={len(ev_reversals)}): "
          f"mean={np.mean(ev_reversals):.1f}, med={np.median(ev_reversals):.1f}")
    print(f"    Baseline  (n={len(bl_reversals)}): "
          f"mean={np.mean(bl_reversals):.1f}, med={np.median(bl_reversals):.1f}")

    try:
        from scipy.stats import mannwhitneyu
        for label, ev_vals, bl_vals in [
            ("Max |trend48h|", ev_max_trends, bl_max_trends),
            ("|Trend reversal|", ev_reversals, bl_reversals),
        ]:
            stat, p = mannwhitneyu(ev_vals, bl_vals, alternative="greater")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {label}: Mann-Whitney p = {p:.4f} {sig}")
    except ImportError:
        pass

    p75_bl = np.percentile(bl_max_trends, 75)
    p90_bl = np.percentile(bl_max_trends, 90)
    n_above_75 = sum(1 for x in ev_max_trends if x > p75_bl)
    n_above_90 = sum(1 for x in ev_max_trends if x > p90_bl)
    print(f"\n  Pre-event above baseline P75 ({p75_bl:.1f}): {n_above_75}/{len(ev_max_trends)}")
    print(f"  Pre-event above baseline P90 ({p90_bl:.1f}): {n_above_90}/{len(ev_max_trends)}")

    print(f"\n  Per-event summary:")
    print(f"  {'Event':>30s}  {'Mag':>4s}  {'Dist':>6s}  {'MaxTrend':>9s}  {'Reversal':>9s}  {'Status':>10s}")
    print(f"  {'─' * 80}")
    for r in event_results:
        status = "SIGNAL" if r["max_abs_trend"] > p75_bl else "baseline"
        print(f"  {r['label']:>30s}  {r['mag']:>4.1f}  {r['dist_km']:>5.0f}km  "
              f"{r['max_abs_trend']:>8.1f}  {abs(r['trend_reversal_48_12']):>8.1f}  "
              f"{status:>10s}")

    conn.close()
    print(f"\n  Analysis complete.\n")


if __name__ == "__main__":
    main()
