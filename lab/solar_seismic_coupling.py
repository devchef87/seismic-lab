"""QuakeWatch Lab — Solar-Seismic Coupling Analysis

Test the hypothesis: do geomagnetic storms / solar wind events
precede large earthquakes more often than expected by chance?

The anomaly detector flagged Kp (geomagnetic index) as the #1
anomalous channel before M6.5+ events. This experiment rigorously
tests that finding across the full historical dataset.

Tests:
  1. Superposed epoch analysis: stack solar/geomag signals around M6.5+ events
  2. Rate ratio: M6.5+ rate during storms vs quiet periods
  3. Lag correlation: does elevated SW/Kp PRECEDE earthquakes?
  4. Bootstrap null test: could the correlation arise by chance?
  5. Confound controls: seasonal, solar cycle, temporal clustering

Run:
  python3 lab/solar_seismic_coupling.py
  python3 lab/solar_seismic_coupling.py --min-mag 7.0    # only M7+ events
  python3 lab/solar_seismic_coupling.py --lag-hours 48    # test longer lags
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings("ignore")

UTC = timezone.utc
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "quakewatch.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def get_hourly_signal(conn, metric, t_start, t_end):
    """Get hourly averages for a signal over a time range."""
    rows = conn.execute(
        "SELECT timestamp, value FROM samples WHERE metric = ? "
        "AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
        (metric, t_start, t_end)
    ).fetchall()

    hourly = {}
    for r in rows:
        try:
            ts_str = r["timestamp"]
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            hour_key = ts.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly:
                hourly[hour_key] = []
            hourly[hour_key].append(float(r["value"]))
        except (ValueError, TypeError):
            continue

    return {k: np.mean(v) for k, v in hourly.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-mag", type=float, default=6.5)
    parser.add_argument("--lag-hours", type=int, default=72)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    conn = get_conn()

    # ── Load all M6.5+ events ──
    events = conn.execute(
        "SELECT timestamp, magnitude, lat, lon, place FROM earthquakes "
        "WHERE magnitude >= ? ORDER BY timestamp",
        (args.min_mag,)
    ).fetchall()
    events = [dict(r) for r in events]

    # Parse timestamps
    for e in events:
        ts = e["timestamp"]
        e["dt"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if e["dt"].tzinfo is None:
            e["dt"] = e["dt"].replace(tzinfo=UTC)

    # Deduplicate (within 2h / 2° = same event)
    unique = []
    for e in events:
        is_dup = False
        for u in unique:
            dt_hours = abs((e["dt"] - u["dt"]).total_seconds()) / 3600
            if dt_hours < 2 and abs(e["lat"] - u["lat"]) < 2 and abs(e["lon"] - u["lon"]) < 2:
                is_dup = True
                break
        if not is_dup:
            unique.append(e)
    events = unique

    print(f"\n{'='*80}")
    print(f"  SOLAR-SEISMIC COUPLING ANALYSIS")
    print(f"  {len(events)} unique M{args.min_mag}+ events  |  lag window: ±{args.lag_hours}h")
    print(f"{'='*80}")

    # ── Load full solar/geomag time series ──
    signals = {
        "solar_wind_speed": "SW Speed (km/s)",
        "solar_wind_density": "SW Density (p/cm³)",
        "kp_index": "Kp Index",
        "dst_index": "Dst (nT)",
        "imf_bt": "IMF |B| (nT)",
        "proton_flux": "Proton Flux (pfu)",
        "neutron_count": "Neutron Count",
    }

    t_global_start = "2015-01-01T00:00:00"
    t_global_end = "2026-06-25T23:59:59"

    print(f"\n  Loading signal time series...")
    signal_data = {}
    for metric, label in signals.items():
        data = get_hourly_signal(conn, metric, t_global_start, t_global_end)
        signal_data[metric] = data
        print(f"    {label:30s}: {len(data):>7d} hourly points")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: Superposed Epoch Analysis
    # Stack each signal around event time, see if there's a consistent pattern
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  TEST 1: SUPERPOSED EPOCH ANALYSIS")
    print(f"  (Stack signal values ±{args.lag_hours}h around each M{args.min_mag}+ event)")
    print(f"{'='*80}")

    lag_range = range(-args.lag_hours, args.lag_hours + 1)

    for metric, label in signals.items():
        data = signal_data[metric]
        if len(data) < 100:
            print(f"\n  {label}: insufficient data ({len(data)} points), skipping")
            continue

        # For each event, extract signal at each lag
        epoch_stack = defaultdict(list)
        n_events_with_data = 0

        for e in events:
            has_any = False
            for lag_h in lag_range:
                t_lag = e["dt"] + timedelta(hours=lag_h)
                t_key = t_lag.replace(minute=0, second=0, microsecond=0)
                if t_key in data:
                    epoch_stack[lag_h].append(data[t_key])
                    has_any = True
            if has_any:
                n_events_with_data += 1

        if n_events_with_data < 10:
            print(f"\n  {label}: only {n_events_with_data} events with data, skipping")
            continue

        # Compute statistics at each lag
        pre_event_vals = []  # -72h to -6h (excluding near-event)
        post_event_vals = []  # +6h to +72h
        at_event_vals = []   # -6h to 0

        for lag_h in lag_range:
            vals = epoch_stack.get(lag_h, [])
            if not vals:
                continue
            mean_val = np.mean(vals)
            if -args.lag_hours <= lag_h <= -6:
                pre_event_vals.append(mean_val)
            elif -6 <= lag_h <= 0:
                at_event_vals.append(mean_val)
            elif 6 <= lag_h <= args.lag_hours:
                post_event_vals.append(mean_val)

        if not pre_event_vals or not at_event_vals:
            continue

        pre_mean = np.mean(pre_event_vals)
        at_mean = np.mean(at_event_vals)
        post_mean = np.mean(post_event_vals) if post_event_vals else np.nan

        # Baseline: average over all available hours
        all_vals = list(data.values())
        baseline_mean = np.mean(all_vals)
        baseline_std = np.std(all_vals)

        pre_sigma = (pre_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
        at_sigma = (at_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0

        marker = ""
        if abs(pre_sigma) > 0.3 or abs(at_sigma) > 0.3:
            marker = " ◀ SIGNAL"

        print(f"\n  {label} ({n_events_with_data} events):")
        print(f"    Baseline mean:       {baseline_mean:>10.2f}")
        print(f"    Pre-event (-72 to -6h): {pre_mean:>10.2f}  ({pre_sigma:+.3f}σ)")
        print(f"    At event (-6 to 0h):    {at_mean:>10.2f}  ({at_sigma:+.3f}σ){marker}")
        if not np.isnan(post_mean):
            post_sigma = (post_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
            print(f"    Post-event (+6 to +72h):{post_mean:>10.2f}  ({post_sigma:+.3f}σ)")

        # Time-resolved: show 12h bins
        print(f"    Time profile (12h bins):")
        bin_edges = list(range(-args.lag_hours, args.lag_hours + 1, 12))
        for i in range(len(bin_edges) - 1):
            b_start, b_end = bin_edges[i], bin_edges[i + 1]
            bin_vals = []
            for lag_h in range(b_start, b_end):
                bin_vals.extend(epoch_stack.get(lag_h, []))
            if bin_vals:
                bm = np.mean(bin_vals)
                bs = (bm - baseline_mean) / baseline_std if baseline_std > 0 else 0
                bar_len = int(abs(bs) * 20)
                bar = "█" * min(bar_len, 40)
                direction = "+" if bs > 0 else "-"
                label_str = f"T{b_start:+d}h to T{b_end:+d}h"
                print(f"      {label_str:>20s}: {bm:>10.2f} ({bs:+.3f}σ) {direction}{bar}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: Earthquake Rate During Storms vs Quiet
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  TEST 2: EARTHQUAKE RATE — STORMS vs QUIET PERIODS")
    print(f"{'='*80}")

    sw_data = signal_data.get("solar_wind_speed", {})
    kp_data = signal_data.get("kp_index", {})

    if sw_data:
        sw_values = np.array(list(sw_data.values()))
        sw_p75 = np.percentile(sw_values, 75)
        sw_p90 = np.percentile(sw_values, 90)
        sw_p95 = np.percentile(sw_values, 95)

        print(f"\n  Solar Wind Speed thresholds:")
        print(f"    75th pct: {sw_p75:.0f} km/s  |  90th: {sw_p90:.0f} km/s  |  95th: {sw_p95:.0f} km/s")

        for threshold_name, threshold_val in [("75th pct", sw_p75), ("90th pct", sw_p90), ("95th pct", sw_p95)]:
            # Count hours above/below threshold
            storm_hours = sum(1 for v in sw_data.values() if v >= threshold_val)
            quiet_hours = sum(1 for v in sw_data.values() if v < threshold_val)

            # Count events during storm vs quiet (within 24h of storm conditions)
            storm_events = 0
            quiet_events = 0
            for e in events:
                # Check if SW speed was above threshold in the 24h before event
                is_storm = False
                for lag_h in range(0, 25):
                    t_check = e["dt"] - timedelta(hours=lag_h)
                    t_key = t_check.replace(minute=0, second=0, microsecond=0)
                    if t_key in sw_data and sw_data[t_key] >= threshold_val:
                        is_storm = True
                        break
                if is_storm:
                    storm_events += 1
                else:
                    quiet_events += 1

            total_hours = storm_hours + quiet_hours
            storm_frac = storm_hours / total_hours if total_hours > 0 else 0
            total_events = storm_events + quiet_events
            storm_event_frac = storm_events / total_events if total_events > 0 else 0

            rate_ratio = (storm_event_frac / storm_frac) if storm_frac > 0 else 0

            print(f"\n    SW >= {threshold_val:.0f} km/s ({threshold_name}):")
            print(f"      Storm hours: {storm_hours:>6d}/{total_hours} ({100*storm_frac:.1f}% of time)")
            print(f"      Events in storm: {storm_events:>3d}/{total_events} ({100*storm_event_frac:.1f}% of events)")
            print(f"      Rate ratio: {rate_ratio:.2f}x")
            if rate_ratio > 1.2:
                print(f"      → Earthquakes are {rate_ratio:.1f}x MORE LIKELY during high SW speed")
            elif rate_ratio < 0.8:
                print(f"      → Earthquakes are {1/rate_ratio:.1f}x LESS LIKELY during high SW speed")
            else:
                print(f"      → No significant difference")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: Lag Correlation (Granger-style)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  TEST 3: LAG CORRELATION — Does Solar Activity PRECEDE Earthquakes?")
    print(f"{'='*80}")

    # Build daily earthquake rate time series
    daily_quake_rate = defaultdict(int)
    for e in events:
        day_key = e["dt"].date()
        daily_quake_rate[day_key] += 1

    # Build daily solar/geomag averages
    for metric, label in [("solar_wind_speed", "SW Speed"),
                          ("kp_index", "Kp Index"),
                          ("dst_index", "Dst Index"),
                          ("proton_flux", "Proton Flux")]:
        data = signal_data.get(metric, {})
        if len(data) < 100:
            continue

        daily_signal = defaultdict(list)
        for t, v in data.items():
            daily_signal[t.date()].append(v)
        daily_signal = {k: np.mean(v) for k, v in daily_signal.items()
                        if len(v) >= 3}

        # Common days
        common_days = sorted(set(daily_signal.keys()) & set(daily_quake_rate.keys()))
        if len(common_days) < 30:
            print(f"\n  {label}: only {len(common_days)} common days, skipping")
            continue

        all_days = sorted(daily_signal.keys())

        # Cross-correlation at different lags
        print(f"\n  {label} → M{args.min_mag}+ rate (positive lag = signal leads):")
        print(f"    {'Lag (days)':>10s}  {'Correlation':>12s}  {'Visual':>20s}")
        print(f"    {'─'*50}")

        best_lag = 0
        best_corr = 0

        for lag_days in range(-7, 8):
            sig_vals = []
            eq_vals = []
            for day in all_days:
                eq_day = day + timedelta(days=lag_days)
                if day in daily_signal:
                    sig_vals.append(daily_signal[day])
                    eq_vals.append(daily_quake_rate.get(eq_day, 0))

            if len(sig_vals) < 30:
                continue

            corr = np.corrcoef(sig_vals, eq_vals)[0, 1]
            bar_len = int(abs(corr) * 100)
            bar = "█" * min(bar_len, 30)
            sign = "+" if corr > 0 else "-"
            marker = " ◀" if abs(corr) > abs(best_corr) else ""

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag_days

            lag_label = f"{lag_days:+d}d" if lag_days != 0 else " 0d"
            print(f"    {lag_label:>10s}  {corr:>+12.4f}  {sign}{bar}{marker}")

        if abs(best_corr) > 0.02:
            if best_lag > 0:
                print(f"    → Peak correlation at lag={best_lag:+d}d: {label} leads earthquakes by {best_lag} day(s)")
            elif best_lag < 0:
                print(f"    → Peak at lag={best_lag:+d}d: earthquakes lead {label} (reverse causation?)")
            else:
                print(f"    → Peak at lag=0: simultaneous")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: Bootstrap Null Test
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  TEST 4: BOOTSTRAP NULL TEST")
    print(f"  (Shuffle event times {args.n_bootstrap}x, compare to observed)")
    print(f"{'='*80}")

    for metric, label in [("solar_wind_speed", "SW Speed"),
                          ("kp_index", "Kp Index")]:
        data = signal_data.get(metric, {})
        if len(data) < 100:
            continue

        # Observed: mean signal in 24h before each event
        observed_vals = []
        for e in events:
            vals = []
            for lag_h in range(0, 25):
                t_check = e["dt"] - timedelta(hours=lag_h)
                t_key = t_check.replace(minute=0, second=0, microsecond=0)
                if t_key in data:
                    vals.append(data[t_key])
            if vals:
                observed_vals.append(np.mean(vals))

        if not observed_vals:
            continue

        observed_mean = np.mean(observed_vals)

        # Bootstrap: randomly sample the same number of times
        all_times = list(data.keys())
        rng = np.random.RandomState(42)
        bootstrap_means = []

        for _ in range(args.n_bootstrap):
            random_times = rng.choice(len(all_times), size=len(events), replace=True)
            rand_vals = []
            for idx in random_times:
                t_base = all_times[idx]
                vals = []
                for lag_h in range(0, 25):
                    t_check = t_base - timedelta(hours=lag_h)
                    t_key = t_check.replace(minute=0, second=0, microsecond=0)
                    if t_key in data:
                        vals.append(data[t_key])
                if vals:
                    rand_vals.append(np.mean(vals))
            if rand_vals:
                bootstrap_means.append(np.mean(rand_vals))

        bootstrap_arr = np.array(bootstrap_means)
        p_value = np.mean(bootstrap_arr >= observed_mean)
        z_score = (observed_mean - np.mean(bootstrap_arr)) / np.std(bootstrap_arr)

        print(f"\n  {label}:")
        print(f"    Observed mean (24h pre-event): {observed_mean:.2f}")
        print(f"    Bootstrap mean (random times): {np.mean(bootstrap_arr):.2f}")
        print(f"    Bootstrap std:                 {np.std(bootstrap_arr):.2f}")
        print(f"    Z-score:                       {z_score:+.3f}")
        print(f"    P-value (one-tailed):           {p_value:.4f}")

        if p_value < 0.01:
            print(f"    → HIGHLY SIGNIFICANT (p={p_value:.4f}): {label} is elevated before M{args.min_mag}+ events")
        elif p_value < 0.05:
            print(f"    → SIGNIFICANT (p={p_value:.4f}): {label} tends to be elevated before events")
        elif p_value < 0.10:
            print(f"    → MARGINAL (p={p_value:.4f}): weak evidence of elevation")
        else:
            print(f"    → NOT SIGNIFICANT (p={p_value:.4f}): no evidence of solar-seismic coupling")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 5: Today's Event Case Study
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  CASE STUDY: June 24-25, 2026 Earthquake Sequence")
    print(f"{'='*80}")

    recent_quakes = [e for e in events if e["dt"] >= datetime(2026, 6, 24, tzinfo=UTC)]

    sw = signal_data.get("solar_wind_speed", {})

    print(f"\n  Solar Wind Speed Timeline (June 22-25):")
    print(f"  {'Time':>20s}  {'SW (km/s)':>10s}  {'Event':>40s}")
    print(f"  {'─'*75}")

    # 6-hour bins
    t = datetime(2026, 6, 22, 0, tzinfo=UTC)
    t_end = datetime(2026, 6, 25, 12, tzinfo=UTC)
    while t <= t_end:
        # Average SW in this 6h window
        sw_vals = []
        for h in range(6):
            t_key = (t + timedelta(hours=h)).replace(minute=0, second=0, microsecond=0)
            if t_key in sw:
                sw_vals.append(sw[t_key])

        sw_mean = np.mean(sw_vals) if sw_vals else np.nan
        sw_str = f"{sw_mean:.0f}" if not np.isnan(sw_mean) else "N/A"

        # Any events in this 6h window?
        event_str = ""
        for eq in recent_quakes:
            if t <= eq["dt"] < t + timedelta(hours=6):
                event_str = f"M{eq['magnitude']:.1f} {eq.get('place', '')[:30]}"

        bar_len = int((sw_mean - 300) / 10) if not np.isnan(sw_mean) and sw_mean > 300 else 0
        bar = "▓" * min(bar_len, 30)

        print(f"  {t.strftime('%b %d %H:%M'):>20s}  {sw_str:>10s}  {bar}  {event_str}")
        t += timedelta(hours=6)

    # Per-event detail
    print(f"\n  Per-Event Solar Conditions:")
    for eq in recent_quakes:
        # SW speed at event time and 24h before
        sw_at = []
        sw_24h = []
        for lag_h in range(0, 3):
            t_key = (eq["dt"] - timedelta(hours=lag_h)).replace(minute=0, second=0, microsecond=0)
            if t_key in sw:
                sw_at.append(sw[t_key])
        for lag_h in range(0, 25):
            t_key = (eq["dt"] - timedelta(hours=lag_h)).replace(minute=0, second=0, microsecond=0)
            if t_key in sw:
                sw_24h.append(sw[t_key])

        sw_now = np.mean(sw_at) if sw_at else np.nan
        sw_24 = np.mean(sw_24h) if sw_24h else np.nan
        sw_max_24 = max(sw_24h) if sw_24h else np.nan

        print(f"\n    M{eq['magnitude']:.1f} at {eq['dt'].strftime('%H:%M UTC')} — {eq.get('place', '')}")
        print(f"      SW speed: {sw_now:.0f} km/s (24h avg: {sw_24:.0f}, peak: {sw_max_24:.0f})")

    conn.close()
    print(f"\n  Analysis complete.\n")


if __name__ == "__main__":
    main()
