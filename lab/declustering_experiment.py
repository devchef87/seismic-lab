"""Is our AUC real prediction, or aftershock bookkeeping?

Decluster the catalog (Gardner-Knopoff 1974 space-time windows), split M5+ events
into independent MAINSHOCKS vs dependent AFTERSHOCKS/foreshocks, then compare:

  A) FULL target      — any M5+ in next 6h (our standard; should ~reproduce 0.70)
  B) MAINSHOCK target — only INDEPENDENT mainshocks; aftershock-only windows are
                        excluded entirely (not scored as positive OR negative)

Same features (existing cache), same pipeline. If B << A, the model was largely
scoring the Omori-law fact that aftershocks follow big events — trivial, not
prediction. If B holds, we have genuine independent-mainshock skill.
"""
import os, sys, math
import numpy as np
import pandas as pd
import sqlite3
import importlib.util

spec = importlib.util.spec_from_file_location("te", os.path.join(os.path.dirname(__file__), "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
HORIZON_H = 6
MIN_MAG = 5.0
NEG_RATIO = 4

# ── Gardner-Knopoff (1974) windows ──
def gk_window(mag):
    L = 10 ** (0.1238 * mag + 0.983)                 # distance, km
    T = 10 ** (0.032 * mag + 2.7389) if mag >= 6.5 else 10 ** (0.5409 * mag - 0.547)  # time, days
    return L, T

def hav_km(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1); dl = math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def decluster(ev):
    """ev: list of (epoch_sec, mag, lat, lon). Returns bool array: True = independent mainshock.
    Window method: largest events claim smaller events within their GK space-time window."""
    n = len(ev)
    dependent = [False] * n
    order = sorted(range(n), key=lambda i: -ev[i][1])  # magnitude descending
    for i in order:
        if dependent[i]:
            continue
        ti, mi, lai, loi = ev[i]
        L, T = gk_window(mi); Tsec = T * 86400.0
        for j in range(n):
            if j == i or dependent[j] or ev[j][1] >= mi:
                continue
            if abs(ev[j][0] - ti) <= Tsec and hav_km(lai, loi, ev[j][2], ev[j][3]) <= L:
                dependent[j] = True
    return [not d for d in dependent]


# ── Load cache + cell geometry ──
data = np.load(CACHE, allow_pickle=True)
cell_ids = list(data["cell_ids"])
parents = dict(zip(data["cell_ids"], data["cell_parents"]))
feats = {c: data[f"feat_{c}"] for c in cell_ids}
hour_epochs = data["hour_epochs"]
hours = pd.DatetimeIndex(data["hours"])
feat_names = list(data["feature_names"])
n = len(hours)
train_end = int(n * 0.70); val_end = int(n * 0.85)
cell_geom = {c["id"]: c for c in te.generate_grid_cells(te.PARENT_ZONES)}
cpm = {c: parents[c] for c in cell_ids}

print("=" * 70)
print("  DECLUSTERING EXPERIMENT — mainshock skill vs aftershock bookkeeping")
print("=" * 70)

# ── Build full / mainshock / aftershock targets per cell from declustered catalog ──
conn = sqlite3.connect(te.DB_PATH, timeout=60)
full_t, main_t, after_t = {}, {}, {}
tot_main = tot_after = 0
hsec = HORIZON_H * 3600
for c in cell_ids:
    g = cell_geom[c]
    rows = conn.execute(
        "SELECT timestamp, magnitude, lat, lon FROM earthquakes WHERE magnitude >= ? "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? ORDER BY timestamp",
        (MIN_MAG, g["lat_range"][0], g["lat_range"][1], g["lon_range"][0], g["lon_range"][1])).fetchall()
    ev = []
    for ts, m, la, lo in rows:
        try:
            t = pd.Timestamp(ts, tz="UTC").timestamp()
        except Exception:
            continue
        if t < hour_epochs[0] - hsec or t > hour_epochs[-1]:
            continue
        ev.append((t, float(m), la, lo))
    full = np.zeros(n, np.float32); main = np.zeros(n, np.float32); aft = np.zeros(n, np.float32)
    if ev:
        is_main = decluster(ev)
        for k, (t, m, la, lo) in enumerate(ev):
            lo_i = np.searchsorted(hour_epochs, t - hsec, "left")
            hi_i = np.searchsorted(hour_epochs, t, "left")
            lo_i, hi_i = max(lo_i, 0), min(hi_i, n)
            if hi_i > lo_i:
                full[lo_i:hi_i] = 1.0
                if is_main[k]:
                    main[lo_i:hi_i] = 1.0; tot_main += 1
                else:
                    aft[lo_i:hi_i] = 1.0; tot_after += 1
    full_t[c] = full; main_t[c] = main; after_t[c] = aft
conn.close()
print(f"  Declustered M5+ events: {tot_main} mainshocks, {tot_after} aftershocks/foreshocks "
      f"({tot_main/(tot_main+tot_after)*100:.0f}% independent)")

# ── Assemble train/val/test for an arm ──
def assemble(target, exclude, lo, hi, sample):
    X, y, z = [], [], []
    for ci, c in enumerate(cell_ids):
        f = feats[c][lo:hi]; t = target[c][lo:hi]
        valid = exclude[c][lo:hi] < 0.5 if exclude else np.ones(hi - lo, bool)
        pos = (t > 0.5) & valid
        neg = (t < 0.5) & valid
        if sample:
            pos_idx = np.where(pos)[0]; neg_idx = np.where(neg)[0]
            if len(pos_idx) == 0:
                continue
            rng = np.random.RandomState(42 + ci)
            ntarget = min(len(neg_idx), len(pos_idx) * NEG_RATIO)
            sel_neg = rng.choice(neg_idx, ntarget, replace=False) if ntarget > 0 else neg_idx
            keep = np.concatenate([pos_idx, sel_neg])
        else:
            keep = np.where(valid)[0]
        X.append(f[keep]); y.append(t[keep]); z.append(np.full(len(keep), ci))
    return np.concatenate(X), np.concatenate(y), np.concatenate(z)

def run_arm(target, exclude, label):
    Xtr, ytr, _ = assemble(target, exclude, 0, train_end, True)
    Xva, yva, _ = assemble(target, exclude, train_end, val_end, False)
    Xte, yte, zte = assemble(target, exclude, val_end, n, False)
    print(f"\n  [{label}] train {Xtr.shape[0]:,} ({int(ytr.sum())} pos) | "
          f"test {Xte.shape[0]:,} ({int(yte.sum())} pos)")
    model, _ = te.train_lgbm(Xtr, ytr, Xva, yva, feat_names, tag=label)
    probs = model.predict(Xte)
    macro, _, parent = te.evaluate_per_zone(probs, yte, zte, cell_ids, f"{label} TEST", cpm)
    paucs = [m["auc"] for m in parent.values()]
    return np.mean(paucs) if paucs else 0.0, parent

excl_after = {c: ((after_t[c] > 0.5) & (main_t[c] < 0.5)).astype(np.float32) for c in cell_ids}

macro_full, par_full = run_arm(full_t, None, "FULL")
macro_main, par_main = run_arm(main_t, excl_after, "MAINSHOCK-only")

print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)
print(f"  FULL target (any M5+):        macro AUC = {macro_full:.4f}")
print(f"  MAINSHOCK-only (declustered): macro AUC = {macro_main:.4f}")
print(f"  Δ = {(macro_main - macro_full)*100:+.2f}pp")
print(f"\n  {'Zone':>16s} {'FULL':>8s} {'MAIN':>8s} {'Δ':>7s}")
for z in sorted(set(parents.values())):
    a = par_full.get(z, {}).get("auc"); b = par_main.get(z, {}).get("auc")
    if a and b:
        print(f"  {z:>16s} {a:>8.4f} {b:>8.4f} {(b-a)*100:>+6.2f}")
if macro_main < macro_full - 0.05:
    print("\n  => Much of the skill was aftershock autocorrelation.")
elif macro_main < macro_full - 0.02:
    print("\n  => Some aftershock inflation, but real mainshock skill remains.")
else:
    print("\n  => Skill holds on independent mainshocks — genuine prediction.")
