"""QuakeWatch — Magnitude / size prediction.

The classification model answers "will an M5+ happen?". This answers the harder
half of the goal: *how big?* Two framings, because magnitude is compressed near
the detection threshold (Gutenberg-Richter — ~71% of M5+ are M5.0-5.5):

  1. Regression       — predict the magnitude (MAE vs naive mean baseline)
  2. M6+ exceedance   — P(M>=6 | an event is coming). The useful one: given
                        precursors point to an event, is it the damaging "big
                        one" or a routine M5? Directly serves "Chile -> M6.1".

Trained only on positive event-hours (mag target present), reusing the
EMSC-enriched feature cache (no rebuild needed). Honest temporal split.
"""

import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
BIG_MAG = 6.0          # threshold for "the big one"
EVENT_MAG = 5.0        # an event-hour = max mag in window >= this


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    cell_ids = list(d["cell_ids"])
    parents = dict(zip(d["cell_ids"], d["cell_parents"]))
    feature_names = list(d["feature_names"])
    feats = {c: d[f"feat_{c}"] for c in cell_ids}
    mags = {c: d[f"mag_{c}"] for c in cell_ids}
    n_hours = len(d["hours"])
    return cell_ids, parents, feature_names, feats, mags, n_hours


def split_positives(cell_ids, feats, mags, n_hours, lo, hi):
    """Concatenate positive event-hours (mag>=EVENT_MAG) in [lo,hi) across cells,
    returning features, magnitudes, and parent-zone index per sample."""
    X, y, zone = [], [], []
    for ci, c in enumerate(cell_ids):
        m = mags[c][lo:hi]
        f = feats[c][lo:hi]
        pos = m >= EVENT_MAG
        if pos.sum() == 0:
            continue
        X.append(f[pos])
        y.append(m[pos])
        zone.append(np.full(pos.sum(), ci))
    if not X:
        return np.empty((0, 0)), np.empty(0), np.empty(0)
    return np.concatenate(X), np.concatenate(y), np.concatenate(zone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz"))
    args = ap.parse_args()

    import lightgbm as lgb

    print("=" * 70)
    print("  QUAKEWATCH — MAGNITUDE / SIZE PREDICTION")
    print(f"  Cache: {os.path.basename(args.cache)}")
    print("=" * 70)

    cell_ids, parents, feat_names, feats, mags, n_hours = load_cache(args.cache)
    train_end = int(n_hours * 0.70)
    val_end = int(n_hours * 0.85)

    Xtr, ytr, ztr = split_positives(cell_ids, feats, mags, n_hours, 0, train_end)
    Xva, yva, zva = split_positives(cell_ids, feats, mags, n_hours, train_end, val_end)
    Xte, yte, zte = split_positives(cell_ids, feats, mags, n_hours, val_end, n_hours)

    print(f"\n  Positive event-hours: train={len(ytr):,}  val={len(yva):,}  test={len(yte):,}")
    print(f"  Train mag: mean={ytr.mean():.3f}  std={ytr.std():.3f}")
    print(f"  Test M6+ rate: {(yte >= BIG_MAG).mean()*100:.1f}%")

    # ── 1. REGRESSION ──
    print(f"\n  -- REGRESSION (predict magnitude) --")
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feat_names)
    dva = lgb.Dataset(Xva, label=yva, feature_name=feat_names, reference=dtr)
    reg_params = {
        "objective": "regression_l1", "metric": "l1", "boosting_type": "gbdt",
        "num_leaves": 31, "max_depth": 6, "learning_rate": 0.01,
        "min_child_samples": 100, "colsample_bytree": 0.5, "subsample": 0.7,
        "subsample_freq": 1, "reg_alpha": 0.5, "reg_lambda": 2.0,
        "verbose": -1, "seed": 42,
    }
    reg = lgb.train(reg_params, dtr, num_boost_round=3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pred = reg.predict(Xte)

    # Naive baseline: always predict the training mean magnitude
    base = np.full_like(yte, ytr.mean())
    mae_model = mean_absolute_error(yte, pred)
    mae_base = mean_absolute_error(yte, base)
    rmse_model = mean_squared_error(yte, pred) ** 0.5
    rmse_base = mean_squared_error(yte, base) ** 0.5
    print(f"  Test MAE:  model={mae_model:.4f}  baseline(mean)={mae_base:.4f}  "
          f"({(1-mae_model/mae_base)*100:+.1f}% vs baseline)")
    print(f"  Test RMSE: model={rmse_model:.4f}  baseline={rmse_base:.4f}")

    imp = reg.feature_importance(importance_type="gain")
    print(f"  Top 15 size-predictive features:")
    for i in np.argsort(imp)[::-1][:15]:
        print(f"    {feat_names[i]:28s} {imp[i]:>12.1f}")

    # ── 2. M6+ EXCEEDANCE ("the big one") ──
    print(f"\n  -- M{BIG_MAG}+ EXCEEDANCE (is it the big one?) --")
    btr = (ytr >= BIG_MAG).astype(int)
    bva = (yva >= BIG_MAG).astype(int)
    bte = (yte >= BIG_MAG).astype(int)
    scale = (len(btr) - btr.sum()) / max(btr.sum(), 1)
    dtrb = lgb.Dataset(Xtr, label=btr, feature_name=feat_names)
    dvab = lgb.Dataset(Xva, label=bva, feature_name=feat_names, reference=dtrb)
    clf_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "max_depth": 6, "learning_rate": 0.01,
        "min_child_samples": 100, "colsample_bytree": 0.5, "subsample": 0.7,
        "subsample_freq": 1, "scale_pos_weight": min(scale, 10.0),
        "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1, "seed": 42,
    }
    clf = lgb.train(clf_params, dtrb, num_boost_round=3000, valid_sets=[dvab],
                    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pb = clf.predict(Xte)
    if bte.sum() > 0 and bte.sum() < len(bte):
        auc = roc_auc_score(bte, pb)
        print(f"  Test AUC (M6+ vs M5-6 | event): {auc:.4f}")
        print(f"  (base rate M6+: {bte.mean()*100:.1f}% — AUC>0.5 means real size skill)")
    impb = clf.feature_importance(importance_type="gain")
    print(f"  Top 15 'big one' features:")
    for i in np.argsort(impb)[::-1][:15]:
        print(f"    {feat_names[i]:28s} {impb[i]:>12.1f}")

    # ── Per-zone exceedance AUC ──
    print(f"\n  Per-parent-zone M6+ AUC:")
    zone_of = {ci: parents.get(c, c) for ci, c in enumerate(cell_ids)}
    pmap = {}
    for ci in np.unique(zte.astype(int)):
        mask = zte == ci
        pid = zone_of[ci]
        pmap.setdefault(pid, [[], []])
        pmap[pid][0].append(bte[mask]); pmap[pid][1].append(pb[mask])
    for pid in sorted(pmap):
        t = np.concatenate(pmap[pid][0]); p = np.concatenate(pmap[pid][1])
        if t.sum() > 0 and t.sum() < len(t):
            print(f"    {pid:>16s}  AUC={roc_auc_score(t, p):.4f}  "
                  f"n={len(t):>4d}  M6+={int(t.sum())}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    reg.save_model(os.path.join(MODEL_DIR, "magnitude_reg.txt"))
    clf.save_model(os.path.join(MODEL_DIR, "magnitude_m6_clf.txt"))
    print(f"\n  Saved magnitude models. Done.")


if __name__ == "__main__":
    main()
