"""SeismicLab Lab — Tidal Triggering Analysis

Tests whether earthquakes preferentially occur at certain tidal phases.
If a fault responds to tidal forcing (~1 kPa), it's critically stressed
and close to failure — this sensitivity itself is the precursor signal.

Statistical methods:
  - Schuster test: are earthquake tidal phases uniformly distributed?
  - Per-region analysis: which zones show tidal sensitivity?
  - Temporal analysis: does tidal sensitivity change before large events?
  - Magnitude dependence: are larger quakes more tidally correlated?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "seismiclab.db")
UTC = timezone.utc

ZONES = [
    {"id": "socal",       "name": "Southern California",     "lat": [31, 36],  "lon": [-121, -114]},
    {"id": "norcal",      "name": "N. California / Cascadia","lat": [36, 50],  "lon": [-131, -119]},
    {"id": "alaska",      "name": "Alaska / Aleutians",      "lat": [50, 65],  "lon": [-180, -130]},
    {"id": "japan",       "name": "Japan",                   "lat": [28, 46],  "lon": [128, 148]},
    {"id": "indonesia",   "name": "Indonesia",               "lat": [-12, 8],  "lon": [94, 136]},
    {"id": "chile_peru",  "name": "Chile / Peru",            "lat": [-46, 2],  "lon": [-82, -65]},
    {"id": "mediterranean","name": "Mediterranean / Turkey",  "lat": [33, 42],  "lon": [-6, 45]},
    {"id": "mexico_ca",   "name": "Mexico / Central America","lat": [7, 25],   "lon": [-115, -77]},
    {"id": "caribbean",   "name": "Caribbean",               "lat": [10, 22],  "lon": [-85, -60]},
    {"id": "philippines", "name": "Philippines / Taiwan",     "lat": [4, 26],   "lon": [118, 128]},
    {"id": "himalaya",    "name": "Himalayas",               "lat": [24, 40],  "lon": [65, 100]},
    {"id": "nz_tonga",    "name": "NZ / Tonga / Fiji",       "lat": [-46, -14],"lon": [165, -170]},
]


def schuster_test(phases):
    """Schuster test for non-uniformity of circular data.
    Returns p-value — low p means quakes cluster at certain tidal phases."""
    n = len(phases)
    if n < 10:
        return 1.0, 0.0, 0.0
    cos_sum = sum(math.cos(p) for p in phases)
    sin_sum = sum(math.sin(p) for p in phases)
    D_sq = (cos_sum ** 2 + sin_sum ** 2) / n
    p_value = math.exp(-D_sq)
    mean_phase = math.atan2(sin_sum, cos_sum)
    R = math.sqrt(cos_sum ** 2 + sin_sum ** 2) / n
    return p_value, mean_phase, R


def compute_tidal_phase(tidal_values, tidal_times, eq_time):
    """Find tidal potential at earthquake time and compute phase.
    Phase is estimated from the local derivative (rising/falling/peak/trough)."""
    idx = np.searchsorted(tidal_times, eq_time)
    if idx <= 0 or idx >= len(tidal_values):
        return None

    t0, t1 = tidal_times[idx - 1], tidal_times[idx]
    v0, v1 = tidal_values[idx - 1], tidal_values[idx]
    frac = (eq_time - t0) / (t1 - t0) if t1 != t0 else 0.5
    value = v0 + frac * (v1 - v0)
    derivative = v1 - v0

    if idx >= 2:
        prev_deriv = tidal_values[idx - 1] - tidal_values[idx - 2]
    else:
        prev_deriv = derivative

    phase = math.atan2(derivative, value)
    return phase, value, derivative


def load_data():
    """Load tidal and earthquake data."""
    conn = sqlite3.connect(DB_PATH)

    print("Loading tidal data...")
    tidal_rows = conn.execute(
        "SELECT timestamp, value FROM samples "
        "WHERE metric = 'tidal_potential' "
        "ORDER BY timestamp"
    ).fetchall()

    tidal_times = []
    tidal_values = []
    for ts, val in tidal_rows:
        try:
            t = datetime.fromisoformat(ts)
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            tidal_times.append(t.timestamp())
            tidal_values.append(val)
        except:
            continue

    tidal_times = np.array(tidal_times)
    tidal_values = np.array(tidal_values)
    print(f"  {len(tidal_times)} tidal readings loaded")

    print("Loading earthquakes...")
    eq_rows = conn.execute(
        "SELECT timestamp, lat, lon, magnitude, depth_km FROM earthquakes "
        "WHERE magnitude >= 2.5 ORDER BY timestamp"
    ).fetchall()
    print(f"  {len(eq_rows)} earthquakes loaded")

    conn.close()
    return tidal_times, tidal_values, eq_rows


def assign_zone(lat, lon):
    for z in ZONES:
        if z["lat"][0] <= lat <= z["lat"][1] and z["lon"][0] <= lon <= z["lon"][1]:
            return z["id"]
    return None


def run_analysis():
    tidal_times, tidal_values, eq_rows = load_data()

    print("\n" + "=" * 80)
    print("  TIDAL TRIGGERING ANALYSIS")
    print("=" * 80)

    # Phase computation for all earthquakes
    print("\nComputing tidal phases for all earthquakes...")
    phases_by_mag = defaultdict(list)
    phases_by_zone = defaultdict(list)
    phases_all = []
    tidal_at_eq = []

    for ts, lat, lon, mag, depth in eq_rows:
        try:
            t = datetime.fromisoformat(ts)
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            eq_epoch = t.timestamp()
        except:
            continue

        result = compute_tidal_phase(tidal_values, tidal_times, eq_epoch)
        if result is None:
            continue

        phase, value, derivative = result
        phases_all.append(phase)
        tidal_at_eq.append((phase, value, derivative, mag, lat, lon, depth))

        if mag >= 7.0:
            phases_by_mag["M7+"].append(phase)
        elif mag >= 6.0:
            phases_by_mag["M6+"].append(phase)
        elif mag >= 5.0:
            phases_by_mag["M5+"].append(phase)
        elif mag >= 4.0:
            phases_by_mag["M4+"].append(phase)
        else:
            phases_by_mag["M2.5-4"].append(phase)

        zone = assign_zone(lat, lon)
        if zone:
            phases_by_zone[zone].append((phase, mag))

    print(f"  {len(phases_all)} earthquakes with tidal phase computed")

    # 1. Global Schuster test
    print("\n" + "-" * 60)
    print("  1. GLOBAL SCHUSTER TEST")
    print("-" * 60)
    p, mean_phase, R = schuster_test(phases_all)
    phase_deg = math.degrees(mean_phase)
    print(f"  All quakes (n={len(phases_all)}): p={p:.6f}, "
          f"mean phase={phase_deg:.1f}°, R={R:.6f}")
    print(f"  {'*** SIGNIFICANT ***' if p < 0.05 else 'Not significant'}")

    # 2. By magnitude
    print("\n" + "-" * 60)
    print("  2. MAGNITUDE DEPENDENCE")
    print("-" * 60)
    print(f"  {'Mag Range':<12} {'Count':>8} {'p-value':>12} {'Mean Phase':>12} {'R':>10} {'Sig?':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    for label in ["M2.5-4", "M4+", "M5+", "M6+", "M7+"]:
        if label not in phases_by_mag:
            continue
        ph = phases_by_mag[label]
        p, mean_p, R = schuster_test(ph)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {label:<12} {len(ph):>8} {p:>12.6f} {math.degrees(mean_p):>10.1f}° {R:>10.6f} {sig:>8}")

    # 3. By zone
    print("\n" + "-" * 60)
    print("  3. PER-ZONE TIDAL SENSITIVITY")
    print("-" * 60)
    print(f"  {'Zone':<25} {'Count':>7} {'p-value':>12} {'Mean Phase':>12} {'R':>10} {'Sig?':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")

    zone_results = []
    for z in ZONES:
        zid = z["id"]
        if zid not in phases_by_zone:
            continue
        ph = [p for p, m in phases_by_zone[zid]]
        p_val, mean_p, R = schuster_test(ph)
        zone_results.append((z["name"], len(ph), p_val, mean_p, R))

    zone_results.sort(key=lambda x: x[2])
    for name, n, p_val, mean_p, R in zone_results:
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {name:<25} {n:>7} {p_val:>12.6f} {math.degrees(mean_p):>10.1f}° {R:>10.6f} {sig:>6}")

    # 4. By zone, M5+ only (the real signal)
    print("\n" + "-" * 60)
    print("  4. PER-ZONE TIDAL SENSITIVITY — M5+ ONLY")
    print("-" * 60)
    print(f"  {'Zone':<25} {'Count':>7} {'p-value':>12} {'Mean Phase':>12} {'R':>10} {'Sig?':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")

    zone_m5_results = []
    for z in ZONES:
        zid = z["id"]
        if zid not in phases_by_zone:
            continue
        ph = [p for p, m in phases_by_zone[zid] if m >= 5.0]
        if len(ph) < 10:
            continue
        p_val, mean_p, R = schuster_test(ph)
        zone_m5_results.append((z["name"], len(ph), p_val, mean_p, R))

    zone_m5_results.sort(key=lambda x: x[2])
    for name, n, p_val, mean_p, R in zone_m5_results:
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {name:<25} {n:>7} {p_val:>12.6f} {math.degrees(mean_p):>10.1f}° {R:>10.6f} {sig:>6}")

    # 5. Tidal value distribution: quakes vs random
    print("\n" + "-" * 60)
    print("  5. TIDAL STRESS AT EARTHQUAKE TIMES vs RANDOM")
    print("-" * 60)

    eq_tidal_vals = [v for _, v, _, _, _, _, _ in tidal_at_eq]
    eq_tidal_rates = [d for _, _, d, _, _, _, _ in tidal_at_eq]

    # Compare earthquake tidal values to background distribution
    bg_mean = float(np.mean(tidal_values))
    bg_std = float(np.std(tidal_values))
    eq_mean = float(np.mean(eq_tidal_vals))
    eq_std = float(np.std(eq_tidal_vals))

    print(f"  Background tidal potential: mean={bg_mean:.4f}, std={bg_std:.4f}")
    print(f"  At earthquake times:        mean={eq_mean:.4f}, std={eq_std:.4f}")
    print(f"  Difference: {eq_mean - bg_mean:.6f} ({(eq_mean - bg_mean) / bg_std:.4f} σ)")

    # Do quakes prefer rising or falling tides?
    rising = sum(1 for _, _, d, _, _, _, _ in tidal_at_eq if d > 0)
    falling = len(tidal_at_eq) - rising
    print(f"\n  Rising tide: {rising} ({rising/len(tidal_at_eq)*100:.1f}%)")
    print(f"  Falling tide: {falling} ({falling/len(tidal_at_eq)*100:.1f}%)")
    expected = len(tidal_at_eq) / 2
    chi2 = (rising - expected) ** 2 / expected + (falling - expected) ** 2 / expected
    print(f"  Chi-squared: {chi2:.2f} (>3.84 = significant at p<0.05)")

    # 6. Temporal evolution — does tidal sensitivity increase before large events?
    print("\n" + "-" * 60)
    print("  6. TEMPORAL EVOLUTION — TIDAL SENSITIVITY BEFORE M6+ EVENTS")
    print("-" * 60)

    m6_events = [(ts, lat, lon, mag) for ts, lat, lon, mag, _ in eq_rows if mag >= 6.0]
    print(f"  Analyzing {len(m6_events)} M6+ events...")

    precursor_hits = 0
    precursor_total = 0

    for eq_ts, eq_lat, eq_lon, eq_mag in m6_events[:50]:
        try:
            t = datetime.fromisoformat(eq_ts)
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
        except:
            continue

        # Get all M3+ quakes within 3° and 90 days before this event
        before_start = (t - timedelta(days=90)).isoformat()
        before_end = t.isoformat()

        pre_phases = []
        for pts, plat, plon, pmag, pdep in eq_rows:
            if pmag < 3.0:
                continue
            try:
                pt = datetime.fromisoformat(pts)
                if pt.tzinfo is None:
                    pt = pt.replace(tzinfo=UTC)
            except:
                continue
            if pt >= t or pt < t - timedelta(days=90):
                continue
            dlat = abs(plat - eq_lat)
            dlon = abs(plon - eq_lon)
            if dlat > 3 or dlon > 3:
                continue

            result = compute_tidal_phase(tidal_values, tidal_times, pt.timestamp())
            if result:
                pre_phases.append(result[0])

        if len(pre_phases) >= 20:
            p_val, _, R = schuster_test(pre_phases)
            precursor_total += 1
            if p_val < 0.05:
                precursor_hits += 1

    if precursor_total > 0:
        print(f"  Events with sufficient precursory data: {precursor_total}")
        print(f"  Events where precursory quakes showed tidal sensitivity: {precursor_hits}")
        print(f"  Hit rate: {precursor_hits/precursor_total*100:.1f}%")
        baseline = 0.05
        print(f"  Expected by chance: {baseline*100:.1f}%")
        if precursor_hits / precursor_total > baseline * 3:
            print(f"  *** SIGNIFICANT: {precursor_hits/precursor_total/baseline:.1f}x above chance ***")
    else:
        print(f"  Insufficient precursory data for analysis")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_analysis()
