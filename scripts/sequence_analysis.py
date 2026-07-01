#!/usr/bin/env python3
"""Full sequence analysis: when a quake happens, extract the ENTIRE chain of
subsequent events in the vicinity and characterize the trajectory shape.

Key questions:
  1. How often does ANY later event exceed the trigger? (not just the next one)
  2. How many events deep is the peak? (is it event #2 or event #15?)
  3. Do escalating sequences have a recognizable shape? (staircase, acceleration,
     oscillation-then-spike)
  4. Does the base rate change with sequence shape?
  5. Window sweep: 24h / 72h / 7d / 14d / 30d

Uses numpy vectorization + time-sorted scanning to avoid O(n^2) brute force."""
import os, sys, math, sqlite3, time as _time
import numpy as np
from collections import defaultdict

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "quakewatch.db")
MIN_TRIG = 4.0
YEAR_MIN = "2015"

WINDOWS_H = [24, 72, 168, 336, 720]
RADII_KM = [50, 100, "gk"]
THRESHOLDS = [0.0, 0.5, 1.0]


def haversine_vec(lat1, lon1, lats, lons):
    """Vectorized haversine: one point vs array of points → distances in km."""
    R = 6371.0
    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = np.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        np.cos(np.radians(lats)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def gk_radius(mag):
    return 10 ** (0.1238 * mag + 0.983)


def load_events():
    conn = sqlite3.connect(DB, timeout=60)
    rows = conn.execute(
        "SELECT id, timestamp, magnitude, lat, lon FROM earthquakes "
        "WHERE magnitude >= ? AND substr(timestamp,1,4) >= ? "
        "ORDER BY timestamp",
        (MIN_TRIG, YEAR_MIN)).fetchall()
    conn.close()
    from datetime import datetime
    epochs, mags, lats, lons = [], [], [], []
    for eid, ts, mag, lat, lon in rows:
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            epochs.append(t.timestamp())
            mags.append(mag); lats.append(lat); lons.append(lon)
        except Exception:
            continue
    print(f"loaded {len(epochs):,} events (M{MIN_TRIG}+, {YEAR_MIN}+)")
    return (np.array(epochs), np.array(mags, dtype=np.float32),
            np.array(lats, dtype=np.float32), np.array(lons, dtype=np.float32))


def get_followers(i, epochs, mags, lats, lons, window_s, radius):
    """Get all followers of event i within window and radius. Returns mag array."""
    t0 = epochs[i]; m0 = mags[i]
    r_km = gk_radius(m0) if radius == "gk" else radius

    # Time window slice (already sorted by time)
    j_start = i + 1
    j_end = np.searchsorted(epochs, t0 + window_s, side='right')
    if j_start >= j_end:
        return np.array([], dtype=np.float32)

    # Rough lat/lon box filter first (much cheaper than haversine)
    dlat_deg = r_km / 111.0
    dlon_deg = r_km / (111.0 * max(0.1, math.cos(math.radians(lats[i]))))
    sl = slice(j_start, j_end)
    lat_ok = np.abs(lats[sl] - lats[i]) <= dlat_deg
    lon_ok = np.abs(lons[sl] - lons[i]) <= dlon_deg
    box_mask = lat_ok & lon_ok

    if not box_mask.any():
        return np.array([], dtype=np.float32)

    # Precise haversine on candidates only
    cand_idx = np.where(box_mask)[0]
    actual_idx = j_start + cand_idx
    dists = haversine_vec(lats[i], lons[i], lats[actual_idx], lons[actual_idx])
    within = dists <= r_km
    return mags[actual_idx[within]]


def run_sweep(epochs, mags, lats, lons):
    """Part 1: Window × Radius × Threshold sweep."""
    print("\n── PART 1: Escalation rate sweep (any event in chain exceeds trigger) ──")
    print(f"{'window':<8} {'radius':<8} {'threshold':<10} {'triggers':>10} "
          f"{'w/followers':>12} {'escalated':>10} {'rate':>8}")
    print("-" * 75)

    n = len(epochs)
    for wh in WINDOWS_H:
        ws = wh * 3600
        for rad in RADII_KM:
            # Pre-compute all follower sequences for this window/radius
            seqs = []
            t0_sweep = _time.time()
            for i in range(n):
                fs = get_followers(i, epochs, mags, lats, lons, ws, rad)
                seqs.append((mags[i], fs))
            elapsed = _time.time() - t0_sweep

            for thresh in THRESHOLDS:
                total = len(seqs)
                with_f = sum(1 for m, fs in seqs if len(fs) > 0)
                esc = sum(1 for m, fs in seqs if len(fs) > 0 and fs.max() >= m + thresh)
                rate = esc / total * 100 if total else 0
                r_label = "GK" if rad == "gk" else f"{rad}km"
                print(f"{wh:>4}h   {r_label:<8} +{thresh:<9.1f} {total:>10,} "
                      f"{with_f:>12,} {esc:>10,} {rate:>7.1f}%")
            r_label = "GK" if rad == "gk" else f"{rad}km"
            print(f"  [{wh}h/{r_label}: {elapsed:.1f}s]")
            sys.stdout.flush()


def run_depth_and_shapes(epochs, mags, lats, lons):
    """Parts 2-5: using 7d window, GK radius."""
    n = len(epochs)
    ws = 168 * 3600  # 7d
    print("\n\nComputing 7d/GK sequences for depth + shape analysis...")
    t0 = _time.time()
    seqs = []
    for i in range(n):
        fs = get_followers(i, epochs, mags, lats, lons, ws, "gk")
        seqs.append((float(mags[i]), list(fs)))
        if (i+1) % 50000 == 0:
            print(f"  {i+1:,}/{n:,} ({_time.time()-t0:.0f}s)")
            sys.stdout.flush()
    print(f"  done in {_time.time()-t0:.0f}s")

    # ── Part 2: Peak depth ──
    print("\n── PART 2: How deep in the sequence is the peak? ──")
    depth_bins = defaultdict(lambda: {"total": 0, "escalated": 0})
    peak_depths = []

    for m0, followers in seqs:
        if not followers:
            continue
        max_mag = max(followers)
        max_pos = followers.index(max_mag)
        escalated = max_mag > m0
        n_f = len(followers)

        if n_f == 1:
            label = "1 (single)"
        elif n_f <= 3:
            label = "2-3"
        elif n_f <= 10:
            label = "4-10"
        elif n_f <= 30:
            label = "11-30"
        else:
            label = "30+"

        depth_bins[label]["total"] += 1
        if escalated:
            depth_bins[label]["escalated"] += 1
            peak_depths.append((max_pos + 1, n_f, max_mag - m0, m0))

    print(f"\n{'seq_length':<22} {'count':>8} {'escalated':>10} {'rate':>8}")
    print("-" * 52)
    for label in ["1 (single)", "2-3", "4-10", "11-30", "30+"]:
        d = depth_bins[label]
        rate = d["escalated"] / d["total"] * 100 if d["total"] else 0
        print(f"{label:<22} {d['total']:>8,} {d['escalated']:>10,} {rate:>7.1f}%")

    if peak_depths:
        positions = [p for p, n, dm, m0 in peak_depths]
        print(f"\nWhen escalation happens, the peak is at position:")
        print(f"  median: event #{np.median(positions):.0f}")
        print(f"  mean:   event #{np.mean(positions):.1f}")
        print(f"  p90:    event #{int(np.percentile(positions, 90))}")
        print(f"  max:    event #{max(positions)}")
        not_first = sum(1 for p in positions if p > 1)
        pct = not_first / len(positions) * 100
        print(f"\n  Peak is NOT the immediate next event: {not_first}/{len(positions)} = {pct:.1f}%")
        print(f"  → {pct:.0f}% of escalations need multi-step lookback")

    # ── Part 3: Trajectory shapes ──
    print("\n── PART 3: Trajectory shapes → which predict escalation? ──")
    shape_stats = defaultdict(lambda: {"total": 0, "escalated": 0, "big_esc": 0, "gains": []})

    for m0, followers in seqs:
        if not followers:
            shape_stats["isolated"]["total"] += 1
            continue

        max_mag = max(followers)
        escalated = max_mag > m0
        big_esc = max_mag >= m0 + 1.0
        max_pos = followers.index(max_mag)
        n_f = len(followers)
        mags_seq = [m0] + followers

        if n_f == 1:
            shape = "single_follower"
        elif n_f <= 3:
            diffs = [mags_seq[k+1] - mags_seq[k] for k in range(len(mags_seq)-1)]
            if all(d >= -0.15 for d in diffs):
                shape = "monotonic_up"
            elif all(d <= 0.15 for d in diffs):
                shape = "monotonic_down"
            else:
                shape = "short_oscillating"
        else:
            arr = np.array(mags_seq)
            diffs = np.diff(arr)
            peak_rel = max_pos / n_f

            # Consecutive up-steps
            consec_up = 0; max_consec = 0
            for d in diffs:
                if d >= -0.05:
                    consec_up += 1
                    max_consec = max(max_consec, consec_up)
                else:
                    consec_up = 0

            # Trend in first vs second half
            mid = len(diffs) // 2
            first_trend = np.mean(diffs[:mid])
            second_trend = np.mean(diffs[mid:])

            if max_consec >= 3 and escalated:
                shape = "staircase_up"
            elif peak_rel > 0.6 and escalated and second_trend > first_trend + 0.05:
                shape = "accelerating"
            elif peak_rel > 0.6 and escalated:
                shape = "late_peak"
            elif peak_rel < 0.25 and n_f > 3:
                shape = "early_peak_decay"
            elif n_f >= 5 and max_mag >= np.median(followers) + 1.5:
                shape = "rumble_then_spike"
            else:
                shape = "oscillating"

        shape_stats[shape]["total"] += 1
        if escalated:
            shape_stats[shape]["escalated"] += 1
            shape_stats[shape]["gains"].append(max_mag - m0)
        if big_esc:
            shape_stats[shape]["big_esc"] += 1

    total_n = sum(s["total"] for s in shape_stats.values())
    total_esc = sum(s["escalated"] for s in shape_stats.values())
    base_rate = total_esc / total_n * 100

    print(f"\n{'shape':<25} {'count':>8} {'escalated':>10} {'rate':>8} "
          f"{'big(+1.0)':>10} {'avg_gain':>10}")
    print("-" * 75)
    for shape in sorted(shape_stats, key=lambda s: -(shape_stats[s]["escalated"] /
                        max(1, shape_stats[s]["total"]))):
        s = shape_stats[shape]
        rate = s["escalated"] / s["total"] * 100 if s["total"] else 0
        big_rate = s["big_esc"] / s["total"] * 100 if s["total"] else 0
        avg_gain = float(np.mean(s["gains"])) if s["gains"] else 0
        flag = " <<<" if rate > base_rate * 1.5 and s["total"] >= 30 else ""
        print(f"{shape:<25} {s['total']:>8,} {s['escalated']:>10,} "
              f"{rate:>7.1f}% {s['big_esc']:>10,} {avg_gain:>+9.2f}{flag}")
    print(f"\n  overall: {total_esc:,}/{total_n:,} = {base_rate:.1f}%")

    # ── Part 4: Quiescence ──
    print("\n── PART 4: Prior quiescence → escalation rate ──")
    ev_epochs = np.array([e for e, _, _, _ in
                          [(epochs[i], mags[i], lats[i], lons[i]) for i in range(len(epochs))]])

    quiet_results = {30: [0,0], 90: [0,0], 180: [0,0], 365: [0,0]}
    active_result = [0, 0]

    for idx, (m0, followers) in enumerate(seqs):
        t0_ev = epochs[idx]; lat0 = lats[idx]; lon0 = lons[idx]
        r_km = gk_radius(m0)
        escalated = bool(followers and max(followers) > m0)

        # Scan backward for nearest prior event within GK radius
        last_prior_days = None
        for j in range(idx - 1, max(idx - 3000, -1), -1):
            dt_back = t0_ev - epochs[j]
            if dt_back > 400 * 86400:
                break
            dlat = abs(lats[j] - lat0)
            if dlat > r_km / 111.0:
                continue
            d = haversine_vec(lat0, lon0,
                              np.array([lats[j]]), np.array([lons[j]]))[0]
            if d <= r_km:
                last_prior_days = dt_back / 86400
                break

        if last_prior_days is not None and last_prior_days < 30:
            active_result[0] += 1
            if escalated:
                active_result[1] += 1
        else:
            quiet_d = last_prior_days if last_prior_days is not None else 9999
            for threshold in sorted(quiet_results.keys()):
                if quiet_d >= threshold:
                    quiet_results[threshold][0] += 1
                    if escalated:
                        quiet_results[threshold][1] += 1

    print(f"\n{'condition':<30} {'count':>8} {'escalated':>10} {'rate':>8}")
    print("-" * 60)
    ar = active_result[1] / max(1, active_result[0]) * 100
    print(f"{'active (prior <30d)':<30} {active_result[0]:>8,} "
          f"{active_result[1]:>10,} {ar:>7.1f}%")
    for days in sorted(quiet_results):
        total, esc = quiet_results[days]
        rate = esc / max(1, total) * 100
        flag = " <<<" if rate > ar * 1.3 and total >= 30 else ""
        print(f"{'quiet >= ' + str(days) + 'd':<30} {total:>8,} {esc:>10,} {rate:>7.1f}%{flag}")

    # ── Part 5: Multi-step pattern signatures ──
    print("\n── PART 5: Multi-step escalation signatures ──")
    patterns = {
        "dip_then_rise":  [0, 0],   # drops ≥0.3, then later rises ≥1.0 above trigger
        "staircase_3up":  [0, 0],   # 3+ consecutive magnitude increases
        "rumble_5plus":   [0, 0],   # ≥5 events, peak ≫ rest
        "double_tap":     [0, 0],   # 2 events within ±0.3 of each other, then jump
        "quiet_burst":    [0, 0],   # ≥3h gap inside sequence before the peak
    }

    for m0, followers in seqs:
        if len(followers) < 3:
            continue
        max_mag = max(followers)
        max_pos = followers.index(max_mag)
        big_esc = max_mag >= m0 + 1.0
        mags_seq = [m0] + followers

        # Dip-then-rise
        if max_pos > 0:
            pre_peak = followers[:max_pos]
            if any(m <= m0 - 0.3 for m in pre_peak):
                patterns["dip_then_rise"][0] += 1
                if big_esc:
                    patterns["dip_then_rise"][1] += 1

        # Staircase: 3+ consecutive ≥ prior
        consec = 0; has_staircase = False
        for k in range(1, len(mags_seq)):
            if mags_seq[k] >= mags_seq[k-1] - 0.05:
                consec += 1
                if consec >= 3:
                    has_staircase = True
            else:
                consec = 0
        if has_staircase:
            patterns["staircase_3up"][0] += 1
            if big_esc:
                patterns["staircase_3up"][1] += 1

        # Rumble: ≥5 followers, peak well above the pack
        if len(followers) >= 5:
            med = np.median(followers)
            if max_mag >= med + 1.5:
                patterns["rumble_5plus"][0] += 1
                if big_esc:
                    patterns["rumble_5plus"][1] += 1

        # Double-tap: two consecutive events within ±0.3, then ≥+1.0 jump
        for k in range(len(followers) - 2):
            if abs(followers[k] - followers[k+1]) <= 0.3 and \
               followers[k+2] >= followers[k] + 1.0:
                patterns["double_tap"][0] += 1
                if big_esc:
                    patterns["double_tap"][1] += 1
                break

    print(f"\n{'pattern':<25} {'occurrences':>12} {'big_esc(+1.0)':>14} {'rate':>8}")
    print("-" * 63)
    for name in patterns:
        total, esc = patterns[name]
        rate = esc / max(1, total) * 100
        print(f"{name:<25} {total:>12,} {esc:>14,} {rate:>7.1f}%")

    print("\n" + "=" * 80)
    print("DONE")


def main():
    epochs, mags, lats, lons = load_events()

    # Part 1 is the slowest (runs all window/radius combos). Do 7d/GK detailed analysis first.
    run_depth_and_shapes(epochs, mags, lats, lons)

    print("\n\n" + "=" * 80)
    print("Now running full sweep (all window × radius × threshold combos)...")
    print("=" * 80)
    run_sweep(epochs, mags, lats, lons)


if __name__ == "__main__":
    main()
