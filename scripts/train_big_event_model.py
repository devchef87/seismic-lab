#!/usr/bin/env python3
"""Big-event model: fixed-radius labels (T1) + first-big-event split (T2).

T1 — Label geometry fix:
  The old label connected a trigger to a follower only within the trigger's
  Gardner-Knopoff radius (GK(M3.5) ~ 25km) and 7 days. Measured against the
  catalog, that design can reach only 35% of M6+ events. A fixed 100km radius
  and 30-day window reaches 77%. Labels here: max magnitude within 100km/30d
  AFTER each trigger event.

T2 — Honest evaluation split:
  'Aftershock context' triggers (an M5+ already occurred within 100km in the
  prior 30d) are the easy case — Omori bookkeeping. 'First big event' triggers
  (no prior M5+) are the real prize. Every threshold is evaluated on both
  subsets separately. (A dedicated first-event-only model was tested and LOST
  to the all-trigger model on its own subset at every threshold — training on
  the full distribution transfers; the sparse subset starves. Removed.)

T3 — Physics features aimed at the first-event subset:
  b-value delta vs local background, foreshock centroid migration (speed +
  approach), quiescence after buildup, inter-event time acceleration, depth
  trend, historical big-event productivity priors (leakage-safe cumulative
  grid), and regional rate anomaly. All catalog-derived.

Usage: python3 -u scripts/train_big_event_model.py
"""
import os, sys, math, time
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_event_model import (
    load_events, compute_catalog_features, assign_zones, train_and_eval,
    CATALOG_FEAT_NAMES,
)

LABEL_RADIUS_KM = 100.0
LABEL_WINDOW_DAYS = 30
THRESHOLDS = [5.5, 6.0, 6.5]
FIRST_EVENT_GATE_MAG = 5.0   # 'first big event' = no M5+ within 100km/30d prior

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models")

REGIONAL_FEAT_NAMES = [
    "r100_n_7d", "r100_n_30d", "r100_max_mag_30d", "r100_n_m4_30d",
    "r100_mag_range_30d", "r100_hours_since_m4",
    # T3 physics features
    "r100_b_30d",        # b-value of the 30d/100km set (M4+, needs >=10)
    "b_delta_bg",        # 30d b-value minus long-term local background b
    "drift_speed",       # foreshock centroid migration speed (km/day)
    "drift_approach",    # centroid moving toward trigger site (km, + = approaching)
    "quiescence",        # rate collapse after buildup (prior-23d rate vs last-7d)
    "dt_accel",          # inter-event time shortening (recent/prior median ratio)
    "r100_depth_trend",  # depth trend across the 30d set
    "hist_m6_frac",      # local historical M6+ fraction of M4+ (productivity prior)
    "hist_m55_frac",     # same for M5.5+
    "rate_anom_30d",     # current M4+ rate vs long-term local M4+ rate
]
ALL_FEAT_NAMES = CATALOG_FEAT_NAMES + REGIONAL_FEAT_NAMES

# Cumulative spatial grid for leakage-safe historical priors (1-degree cells,
# 3x3 neighborhood query ~ +/-150km). Magnitude histogram per cell.
GRID_MAG_LO = 2.5
GRID_MAG_BIN = 0.1
GRID_NBINS = 56  # 2.5 .. 8.1
_BIN_CENTERS = GRID_MAG_LO + (np.arange(GRID_NBINS) + 0.5) * GRID_MAG_BIN
_B_CONST = math.log10(math.e)


def _grid_cell(lat, lon):
    return (int(math.floor(lat)), int((math.floor(lon) + 180) % 360))


def _grid_query(cell_hist, lat, lon):
    cx = int(math.floor(lat))
    cy = int((math.floor(lon) + 180) % 360)
    total = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            h = cell_hist.get((cx + dx, (cy + dy) % 360))
            if h is not None:
                total = h.copy() if total is None else total + h
    return total


def _b_value(mag_arr, mc):
    above = mag_arr[mag_arr >= mc]
    if len(above) < 10:
        return 0.0
    return _B_CONST / (above.mean() - mc + 0.05)


def compute_regional(epochs, mags, depths, lats, lons):
    """Fixed-100km forward label basis + backward regional/physics features.

    Returns:
      fwd_max:  (n,) max magnitude within 100km / next 30d (0 if none)
      reg_feats: (n, 16) backward features within 100km / prior 30d,
                 plus grid-based historical priors (strictly pre-trigger)
    """
    n = len(epochs)
    window_s = LABEL_WINDOW_DAYS * 86400
    fwd_max = np.zeros(n, dtype=np.float32)
    reg = np.zeros((n, len(REGIONAL_FEAT_NAMES)), dtype=np.float32)
    fi = {name: i for i, name in enumerate(REGIONAL_FEAT_NAMES)}
    dlat_deg = LABEL_RADIUS_KM / 111.0

    cell_hist = {}          # (ix,iy) -> mag histogram of all PRIOR events
    t_start = epochs[0]
    m4_bin0 = int((4.0 - GRID_MAG_LO) / GRID_MAG_BIN)
    m55_bin0 = int((5.5 - GRID_MAG_LO) / GRID_MAG_BIN)
    m6_bin0 = int((6.0 - GRID_MAG_LO) / GRID_MAG_BIN)

    print("Computing 100km/30d labels + regional/physics features...")
    t0 = time.time()
    for i in range(n):
        lat_i, lon_i, t_i = float(lats[i]), float(lons[i]), epochs[i]
        dlon_deg = LABEL_RADIUS_KM / (111.0 * max(0.1, math.cos(math.radians(lat_i))))
        coslat = math.cos(math.radians(lat_i))

        # ── Historical priors from the cumulative grid (strictly prior events)
        hist = _grid_query(cell_hist, lat_i, lon_i)
        b_bg = 0.0
        n4_hist = 0.0
        if hist is not None:
            n4_hist = hist[m4_bin0:].sum()
            if n4_hist >= 50:
                reg[i, fi["hist_m6_frac"]] = hist[m6_bin0:].sum() / n4_hist
                reg[i, fi["hist_m55_frac"]] = hist[m55_bin0:].sum() / n4_hist
            if n4_hist >= 30:
                mean_m = (hist[m4_bin0:] * _BIN_CENTERS[m4_bin0:]).sum() / n4_hist
                b_bg = _B_CONST / (mean_m - 4.0 + 0.05)

        # ── Forward: max magnitude within 100km in next 30d (the label)
        j_end = np.searchsorted(epochs, t_i + window_s, side='right')
        if j_end > i + 1:
            sl = slice(i + 1, j_end)
            box = (np.abs(lats[sl] - lat_i) <= dlat_deg) & \
                  (np.abs(lons[sl] - lon_i) <= dlon_deg)
            if box.any():
                idx = np.where(box)[0] + i + 1
                dla = (lats[idx] - lat_i) * 111.0
                dlo = (lons[idx] - lon_i) * 111.0 * coslat
                inr = dla * dla + dlo * dlo <= LABEL_RADIUS_KM ** 2
                if inr.any():
                    fwd_max[i] = mags[idx][inr].max()

        # ── Backward: regional context within 100km in prior 30d
        j0 = np.searchsorted(epochs, t_i - window_s, side='left')
        if j0 < i:
            sl = slice(j0, i)
            box = (np.abs(lats[sl] - lat_i) <= dlat_deg) & \
                  (np.abs(lons[sl] - lon_i) <= dlon_deg)
            if box.any():
                idx = np.where(box)[0] + j0
                dla = (lats[idx] - lat_i) * 111.0
                dlo = (lons[idx] - lon_i) * 111.0 * coslat
                d2 = dla * dla + dlo * dlo
                inr = d2 <= LABEL_RADIUS_KM ** 2
                if inr.any():
                    nb = idx[inr]                 # ascending time order
                    nb_m = mags[nb]
                    nb_d = depths[nb]
                    nb_t = epochs[nb]
                    nb_x = dlo[inr]; nb_y = dla[inr]   # km east/north of trigger
                    dt = t_i - nb_t
                    n30 = len(nb)
                    n7 = int((dt <= 7 * 86400).sum())

                    reg[i, fi["r100_n_7d"]] = n7
                    reg[i, fi["r100_n_30d"]] = n30
                    reg[i, fi["r100_max_mag_30d"]] = nb_m.max()
                    m4 = nb_m >= 4.0
                    reg[i, fi["r100_n_m4_30d"]] = m4.sum()
                    reg[i, fi["r100_mag_range_30d"]] = nb_m.max() - nb_m.min()
                    reg[i, fi["r100_hours_since_m4"]] = \
                        dt[m4].min() / 3600.0 if m4.any() else 999.0

                    # b-value of the current 30d set, and delta vs background
                    b_now = _b_value(nb_m, 4.0)
                    reg[i, fi["r100_b_30d"]] = b_now
                    if b_now > 0 and b_bg > 0:
                        reg[i, fi["b_delta_bg"]] = b_now - b_bg

                    # Migration: centroid drift, older half -> recent half
                    if n30 >= 6:
                        mid = n30 // 2
                        ox, oy = nb_x[:mid].mean(), nb_y[:mid].mean()
                        nx, ny = nb_x[mid:].mean(), nb_y[mid:].mean()
                        dt_c = (nb_t[mid:].mean() - nb_t[:mid].mean()) / 86400.0
                        if dt_c > 0.1:
                            reg[i, fi["drift_speed"]] = \
                                math.hypot(nx - ox, ny - oy) / dt_c
                        reg[i, fi["drift_approach"]] = \
                            math.hypot(ox, oy) - math.hypot(nx, ny)

                    # Quiescence: buildup then quiet
                    rate_prior = (n30 - n7) / 23.0
                    rate_recent = n7 / 7.0
                    reg[i, fi["quiescence"]] = \
                        (rate_prior - rate_recent) / (rate_prior + 1.0)

                    # Inter-event time acceleration
                    if n30 >= 8:
                        gaps = np.diff(nb_t)
                        recent = gaps[nb_t[1:] >= t_i - 7 * 86400]
                        prior = gaps[nb_t[1:] < t_i - 7 * 86400]
                        if len(recent) >= 3 and len(prior) >= 3:
                            med_p = np.median(prior)
                            if med_p > 0:
                                reg[i, fi["dt_accel"]] = min(
                                    float(np.median(recent) / med_p), 10.0)

                    # Depth trend across the window
                    if n30 >= 5:
                        x = np.arange(n30, dtype=np.float32)
                        xm = x.mean()
                        denom = ((x - xm) ** 2).sum()
                        if denom > 0:
                            reg[i, fi["r100_depth_trend"]] = \
                                ((x - xm) * (nb_d - nb_d.mean())).sum() / denom

                    # Rate anomaly vs long-term local rate
                    age_days = (t_i - t_start) / 86400.0
                    if n4_hist > 0 and age_days > 60:
                        expected_30d = n4_hist * 30.0 / age_days
                        reg[i, fi["rate_anom_30d"]] = \
                            min(m4.sum() / (expected_30d + 0.1), 100.0)

        # Insert this event into the grid AFTER querying (strictly-prior priors)
        cell = _grid_cell(lat_i, lon_i)
        h = cell_hist.get(cell)
        if h is None:
            h = np.zeros(GRID_NBINS, dtype=np.float64)
            cell_hist[cell] = h
        b = min(max(int((mags[i] - GRID_MAG_LO) / GRID_MAG_BIN), 0), GRID_NBINS - 1)
        h[b] += 1

        if (i + 1) % 50000 == 0:
            print(f"  {i+1:,}/{n:,} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    print(f"  Done in {time.time()-t0:.0f}s")
    return fwd_max, reg


def precision_at_k(y_true, pred, fracs=(0.001, 0.005, 0.01)):
    """Precision in the top fraction of scored triggers, with lift vs base."""
    order = np.argsort(-pred)
    base = y_true.mean()
    out = []
    for f in fracs:
        k = max(1, int(len(pred) * f))
        p = y_true[order[:k]].mean()
        out.append((f, k, p, p / max(base, 1e-9)))
    return base, out


def report_subset(tag, y, pred):
    if 0 < y.sum() < len(y):
        auc = roc_auc_score(y, pred)
        base, pk = precision_at_k(y, pred)
        print(f"    {tag}: AUC {auc:.4f} | base {base*100:.2f}% | " +
              " | ".join(f"P@{f*100:.1f}%={p*100:.1f}% ({lift:.0f}x)"
                         for f, k, p, lift in pk))
        return auc
    print(f"    {tag}: degenerate labels, skipped")
    return float("nan")


def main():
    ids, epochs, mags, depths, lats, lons = load_events()

    fwd_max, reg_feats = compute_regional(epochs, mags, depths, lats, lons)

    print("\nComputing catalog features...")
    cat_feats = compute_catalog_features(epochs, mags, depths, lats, lons)
    feats = np.hstack([cat_feats, reg_feats])

    zones = assign_zones(lats, lons)

    # Trim the final 30d — forward labels there are truncated
    t_cut = epochs.max() - LABEL_WINDOW_DAYS * 86400
    keep = epochs <= t_cut
    feats, fwd_max, zones = feats[keep], fwd_max[keep], zones[keep]
    reg_max_prior = reg_feats[keep][:, REGIONAL_FEAT_NAMES.index("r100_max_mag_30d")]
    print(f"\nAfter 30d edge trim: {len(fwd_max):,} events")

    n = len(fwd_max)
    tr = int(n * 0.70); va = int(n * 0.85)
    print(f"Temporal split: train {tr:,} | val {va-tr:,} | test {n-va:,}")

    # T2 gate: no M5+ within 100km in prior 30d
    first_event = reg_max_prior < FIRST_EVENT_GATE_MAG
    print(f"First-big-event triggers (no prior M5+ in 100km/30d): "
          f"{first_event.sum():,} / {n:,} ({first_event.mean()*100:.0f}%)")

    for thresh in THRESHOLDS:
        labels = (fwd_max >= thresh).astype(np.float32)
        n_pos_te = labels[va:].sum()
        print(f"\n{'#'*65}\n#  THRESHOLD M{thresh:.1f}  "
              f"(positives: train {labels[:tr].sum():.0f}, test {n_pos_te:.0f})\n{'#'*65}")
        if n_pos_te < 20:
            print("  Too few test positives, skipping")
            continue

        # Stage A: all triggers, new labels
        model, macro, pooled, aucs = train_and_eval(
            feats[:tr], labels[:tr],
            feats[tr:va], labels[tr:va],
            feats[va:], labels[va:],
            zones[va:], ALL_FEAT_NAMES,
            f"M{thresh:.1f} | ALL triggers | 100km/30d labels")

        pred_te = model.predict(feats[va:])
        fe_te = first_event[va:]
        print(f"\n  T2 split on the ALL-trigger model:")
        report_subset("aftershock-context", labels[va:][~fe_te], pred_te[~fe_te])
        report_subset("FIRST-BIG-EVENT   ", labels[va:][fe_te], pred_te[fe_te])

        # T3 feature check: how much do the physics features contribute?
        imp = dict(zip(ALL_FEAT_NAMES,
                       model.feature_importance(importance_type="gain")))
        tot = sum(imp.values()) or 1
        t3_names = REGIONAL_FEAT_NAMES[6:]
        t3_share = sum(imp[f] for f in t3_names) / tot
        print(f"\n  T3 physics features combined gain share: {t3_share*100:.1f}%")
        for f in sorted(t3_names, key=lambda f: -imp[f]):
            print(f"    {f:<20} {100*imp[f]/tot:.2f}%")

        out = os.path.join(MODELS_DIR, f"big_event_m{int(thresh*10)}.txt")
        model.save_model(out)
        print(f"  Saved: {out}")

    print("\n" + "=" * 65)
    print("  Reference — pre-T3 run (25+6 features, same labels):")
    print("    M5.5 first-event: AUC 0.6570 | P@0.1%=17.9% (4x)")
    print("    M6.0 first-event: AUC 0.6440 | P@0.1%=10.4% (8x)")
    print("    M6.5 first-event: AUC 0.4357 (below chance)")
    print("=" * 65)


if __name__ == "__main__":
    main()
