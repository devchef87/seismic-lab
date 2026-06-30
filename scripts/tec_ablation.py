#!/usr/bin/env python3
"""Does ionospheric TEC add skill to the tier-2 model BEYOND the solar signals we
already feed it? Standalone ablation against the existing 2021 cache — computes TEC
features per cell from the `samples` store (igs_ionex + noaa_glotec), then trains
baseline vs baseline+TEC and compares within-zone (macro) AUC. No changes to the
deployed pipeline until/unless TEC proves out. Run AFTER the IONEX backfill completes."""
import os, sys, importlib.util
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("te", "lab/train_ensemble.py")
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
TEC_FEATS = ["tec_mean_7d", "tec_anomaly", "tec_anomaly_max_72h", "tec_trend_72h"]


def build_tec(conn, clat, clon, hours_ts):
    """4 TEC features for one cell from nearby grid points; anomaly is diurnal-detrended
    (current TEC vs trailing 27-day median at the same UTC hour) — the LAIC candidate."""
    n = len(hours_ts)
    out = np.full((n, len(TEC_FEATS)), np.nan, np.float32)
    rows = conn.execute(
        "SELECT timestamp, value FROM samples WHERE metric='tec' "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
        "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp",
        (clat - 15, clat + 15, clon - 15, clon + 15,
         hours_ts[0].isoformat(), hours_ts[-1].isoformat())).fetchall()
    if len(rows) < 50:
        return out
    df = pd.DataFrame(rows, columns=["t", "tec"])
    df["t"] = pd.to_datetime(df["t"], format="ISO8601", utc=True)
    df["h"] = df["t"].dt.floor("h")
    hourly = df.groupby("h")["tec"].mean()
    s = hourly.reindex(hours_ts).interpolate(limit=3).ffill(limit=6).bfill(limit=6)
    base = s.groupby(s.index.hour).transform(lambda x: x.rolling(27, min_periods=5).median())
    anom = s - base
    out[:, 0] = s.rolling(168, min_periods=6).mean().values.astype(np.float32)
    out[:, 1] = anom.values.astype(np.float32)
    out[:, 2] = anom.rolling(72, min_periods=6).max().values.astype(np.float32)
    out[:, 3] = te._rolling_slope(np.nan_to_num(anom.values, nan=0.0).astype(np.float32), 72)
    return out


def main():
    d = np.load(CACHE, allow_pickle=True)
    cell_ids = list(d["cell_ids"]); parents = dict(zip(d["cell_ids"], d["cell_parents"]))
    feats = {c: d[f"feat_{c}"] for c in cell_ids}
    epi = {c: d[f"epi_{c}"] for c in cell_ids}; esc = {c: d[f"esc_{c}"] for c in cell_ids}
    names = list(d["feature_names"])
    hours_ts = pd.DatetimeIndex(d["hours"])
    if hours_ts.tz is None:
        hours_ts = hours_ts.tz_localize("UTC")
    n = len(hours_ts); tr, va = int(n * 0.70), int(n * 0.85)

    geo = {c["id"]: c for c in te.generate_grid_cells(te.PARENT_ZONES)}
    conn = sqlite3.connect(te.DB_PATH, timeout=120); conn.execute("PRAGMA busy_timeout=120000")

    # TEC coverage sanity
    cov = conn.execute("SELECT COUNT(*), MIN(substr(timestamp,1,10)), MAX(substr(timestamp,1,10)) "
                       "FROM samples WHERE metric='tec'").fetchone()
    print(f"TEC store: {cov[0]:,} rows, {cov[1]} -> {cov[2]}")

    tec = {}
    for i, c in enumerate(cell_ids):
        cell = geo.get(c)
        if cell is None:
            tec[c] = np.full((n, len(TEC_FEATS)), np.nan, np.float32); continue
        clat, clon = te._zone_center(cell)
        tec[c] = build_tec(conn, clat, clon, hours_ts)
    nan_rate = np.mean([np.isnan(tec[c]).mean() for c in cell_ids])
    print(f"TEC feature NaN rate across cells: {nan_rate:.1%}")

    def assemble(lo, hi, with_tec):
        X, y, Z = [], [], []
        for c in cell_ids:
            m = epi[c][lo:hi] > 0.5
            if not m.sum():
                continue
            base = feats[c][lo:hi][m]
            X.append(np.hstack([base, tec[c][lo:hi][m]]) if with_tec else base)
            y.append(esc[c][lo:hi][m].astype(np.float32)); Z += [parents[c]] * int(m.sum())
        return np.concatenate(X), np.concatenate(y), np.array(Z)

    def run(tag, with_tec):
        fn = names + TEC_FEATS if with_tec else names
        Xtr, ytr, _ = assemble(0, tr, with_tec)
        Xva, yva, _ = assemble(tr, va, with_tec)
        Xte, yte, Zte = assemble(va, n, with_tec)
        model, _ = te.train_lgbm(Xtr, ytr, Xva, yva, fn, tag=tag)
        aucs = {}
        for z in set(Zte):
            mm = Zte == z; s = yte[mm].sum()
            if mm.sum() >= 30 and 0 < s < mm.sum():
                aucs[z] = roc_auc_score(yte[mm], model.predict(Xte[mm]))
        macro = float(np.mean(list(aucs.values())))
        pooled = roc_auc_score(yte, model.predict(Xte))
        print(f"\n##### {tag}: macro {macro:.4f} | pooled {pooled:.4f}")
        # TEC feature importance (if present)
        if with_tec:
            imp = dict(zip(fn, model.feature_importance(importance_type="gain")))
            tot = sum(imp.values()) or 1
            for f in TEC_FEATS:
                print(f"     {f}: {100*imp.get(f,0)/tot:.2f}% gain")
        return macro, aucs

    base_macro, base_aucs = run("baseline (no TEC)", False)
    tec_macro, tec_aucs = run("baseline + TEC", True)
    print("\n" + "=" * 56)
    print(f"  macro AUC: {base_macro:.4f} -> {tec_macro:.4f}  (Δ {tec_macro-base_macro:+.4f})")
    print("  per-zone Δ:")
    for z in sorted(base_aucs, key=lambda z: -(tec_aucs.get(z, 0) - base_aucs[z])):
        if z in tec_aucs:
            print(f"     {z:<16}{base_aucs[z]:.3f} -> {tec_aucs[z]:.3f}  ({tec_aucs[z]-base_aucs[z]:+.3f})")


if __name__ == "__main__":
    main()
