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
  subsets separately, and a dedicated first-event model is trained to test
  whether specialization helps.

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
]
ALL_FEAT_NAMES = CATALOG_FEAT_NAMES + REGIONAL_FEAT_NAMES


def compute_regional(epochs, mags, lats, lons):
    """Fixed-100km forward label basis + backward regional features.

    Returns:
      fwd_max:  (n,) max magnitude within 100km / next 30d (0 if none)
      reg_feats: (n, 6) backward regional features within 100km / prior 30d
    """
    n = len(epochs)
    window_s = LABEL_WINDOW_DAYS * 86400
    fwd_max = np.zeros(n, dtype=np.float32)
    reg = np.zeros((n, len(REGIONAL_FEAT_NAMES)), dtype=np.float32)
    fi = {name: i for i, name in enumerate(REGIONAL_FEAT_NAMES)}
    dlat_deg = LABEL_RADIUS_KM / 111.0

    print("Computing 100km/30d forward labels + backward regional features...")
    t0 = time.time()
    for i in range(n):
        lat_i, lon_i, t_i = float(lats[i]), float(lons[i]), epochs[i]
        dlon_deg = LABEL_RADIUS_KM / (111.0 * max(0.1, math.cos(math.radians(lat_i))))
        coslat = math.cos(math.radians(lat_i))

        # Forward: max magnitude within 100km in next 30d
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

        # Backward: regional context within 100km in prior 30d
        j0 = np.searchsorted(epochs, t_i - window_s, side='left')
        if j0 < i:
            sl = slice(j0, i)
            box = (np.abs(lats[sl] - lat_i) <= dlat_deg) & \
                  (np.abs(lons[sl] - lon_i) <= dlon_deg)
            if box.any():
                idx = np.where(box)[0] + j0
                dla = (lats[idx] - lat_i) * 111.0
                dlo = (lons[idx] - lon_i) * 111.0 * coslat
                inr = dla * dla + dlo * dlo <= LABEL_RADIUS_KM ** 2
                if inr.any():
                    nb = idx[inr]
                    nb_m = mags[nb]
                    dt = t_i - epochs[nb]
                    reg[i, fi["r100_n_7d"]] = (dt <= 7 * 86400).sum()
                    reg[i, fi["r100_n_30d"]] = len(nb)
                    reg[i, fi["r100_max_mag_30d"]] = nb_m.max()
                    m4 = nb_m >= 4.0
                    reg[i, fi["r100_n_m4_30d"]] = m4.sum()
                    reg[i, fi["r100_mag_range_30d"]] = nb_m.max() - nb_m.min()
                    reg[i, fi["r100_hours_since_m4"]] = \
                        dt[m4].min() / 3600.0 if m4.any() else 999.0

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

    fwd_max, reg_feats = compute_regional(epochs, mags, lats, lons)

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

        out = os.path.join(MODELS_DIR, f"big_event_m{int(thresh*10)}.txt")
        model.save_model(out)
        print(f"  Saved: {out}")

        # Stage B: dedicated first-event model — does specialization help?
        fe_tr = first_event[:tr]; fe_va = first_event[tr:va]
        if labels[:tr][fe_tr].sum() >= 200 and labels[va:][fe_te].sum() >= 20:
            model_fe, macro_fe, pooled_fe, _ = train_and_eval(
                feats[:tr][fe_tr], labels[:tr][fe_tr],
                feats[tr:va][fe_va], labels[tr:va][fe_va],
                feats[va:][fe_te], labels[va:][fe_te],
                zones[va:][fe_te], ALL_FEAT_NAMES,
                f"M{thresh:.1f} | FIRST-EVENT-ONLY model")
            pred_fe = model_fe.predict(feats[va:][fe_te])
            base, pk = precision_at_k(labels[va:][fe_te], pred_fe)
            print(f"    P@k: " + " | ".join(
                f"P@{f*100:.1f}%={p*100:.1f}% ({lift:.0f}x)" for f, k, p, lift in pk))
            out_fe = os.path.join(MODELS_DIR, f"big_event_m{int(thresh*10)}_first.txt")
            model_fe.save_model(out_fe)
            print(f"  Saved: {out_fe}")
        else:
            print("  First-event-only model skipped (too few positives)")

    print("\n" + "=" * 65)
    print("  Reference (OLD GK/7d labels): M5.5 macro 0.753 | M6.0 macro 0.682")
    print("  Old labels could reach 35% of M6+ events; these reach 77%.")
    print("=" * 65)


if __name__ == "__main__":
    main()
