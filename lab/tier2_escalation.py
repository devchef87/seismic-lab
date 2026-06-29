"""TIER-2: will an active swarm escalate to a mainshock, or fizzle?

The honest, well-posed version of the goal. Two stages:
  Stage 1 (gate): keep only cell-hours with an ACTIVE SWARM building — >= K small
     (M2.5+) events in the trailing window AND no M5+ yet (we're in buildup, not
     aftershock). No leakage: trailing-only.
  Stage 2 (label): does an INDEPENDENT mainshock (GK-declustered) M5+ strike within
     the next H hours? Positive = true foreshock sequence; negative = ordinary swarm.

The key test is the TWO ARMS:
  A) CATALOG-ONLY  — seismicity features only (foreshock rate, b-value, energy...)
  B) MULTI-SIGNAL  — + GPS strain, DART, tidal, FIRMS, stations, space weather
If B > A, the multi-modal data genuinely helps tell foreshocks from swarms — the
whole thesis. If not, escalation is a catalog-rate story and the rest is decoration.
"""
import os, sys, math, sqlite3
import numpy as np
import pandas as pd
import importlib.util
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

spec = importlib.util.spec_from_file_location("te", os.path.join(os.path.dirname(__file__), "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
TRAIL_H = 72        # swarm-detection window
MIN_SMALL = 3       # >= this many M2.5+ in trailing window = active swarm
ESC_H = 72          # escalation horizon (predict mainshock within next 72h)
MAIN_MAG = 5.0
SMALL_MAG = 2.5

def gk_window(mag):
    L = 10 ** (0.1238 * mag + 0.983)
    T = 10 ** (0.032 * mag + 2.7389) if mag >= 6.5 else 10 ** (0.5409 * mag - 0.547)
    return L, T

def hav_km(la1, lo1, la2, lo2):
    R = 6371.0; p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1); dl = math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def decluster(ev):
    n = len(ev); dep = [False] * n
    for i in sorted(range(n), key=lambda i: -ev[i][1]):
        if dep[i]: continue
        ti, mi, lai, loi = ev[i]; L, T = gk_window(mi); Ts = T * 86400
        for j in range(n):
            if j == i or dep[j] or ev[j][1] >= mi: continue
            if abs(ev[j][0] - ti) <= Ts and hav_km(lai, loi, ev[j][2], ev[j][3]) <= L:
                dep[j] = True
    return [not d for d in dep]

# ── load cache ──
data = np.load(CACHE, allow_pickle=True)
cell_ids = list(data["cell_ids"])
parents = dict(zip(data["cell_ids"], data["cell_parents"]))
feats = {c: data[f"feat_{c}"] for c in cell_ids}
hour_epochs = data["hour_epochs"].astype(float)
hours = pd.DatetimeIndex(data["hours"]); n = len(hours)
feat_names = list(data["feature_names"])
train_end = int(n * 0.70); val_end = int(n * 0.85)
geom = {c["id"]: c for c in te.generate_grid_cells(te.PARENT_ZONES)}
cpm = {c: parents[c] for c in cell_ids}
TRAIL_S = TRAIL_H * 3600; ESC_S = ESC_H * 3600

print("=" * 70)
print("  TIER-2 ESCALATION — does an active swarm become a mainshock?")
print(f"  Gate: >={MIN_SMALL} M{SMALL_MAG}+ in {TRAIL_H}h, no M5+ yet | Escalate: M5+ mainshock in {ESC_H}h")
print("=" * 70)

conn = sqlite3.connect(te.DB_PATH, timeout=60)
episode = {}; escalate = {}
n_epi = n_esc = 0
for c in cell_ids:
    g = geom[c]
    small = conn.execute(
        "SELECT timestamp FROM earthquakes WHERE magnitude >= ? AND magnitude < ? "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? ORDER BY timestamp",
        (SMALL_MAG, MAIN_MAG, g["lat_range"][0], g["lat_range"][1], g["lon_range"][0], g["lon_range"][1])).fetchall()
    bigrows = conn.execute(
        "SELECT timestamp, magnitude, lat, lon FROM earthquakes WHERE magnitude >= ? "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? ORDER BY timestamp",
        (MAIN_MAG, g["lat_range"][0], g["lat_range"][1], g["lon_range"][0], g["lon_range"][1])).fetchall()
    s_ep = np.array([pd.Timestamp(t, tz="UTC").timestamp() for (t,) in small]) if small else np.array([])
    bev = []
    for ts, m, la, lo in bigrows:
        bev.append((pd.Timestamp(ts, tz="UTC").timestamp(), float(m), la, lo))
    b_ep = np.array([e[0] for e in bev]) if bev else np.array([])

    # trailing small count + trailing big count, vectorized
    if len(s_ep):
        small_trail = np.searchsorted(s_ep, hour_epochs, "right") - np.searchsorted(s_ep, hour_epochs - TRAIL_S, "left")
    else:
        small_trail = np.zeros(n)
    if len(b_ep):
        big_trail = np.searchsorted(b_ep, hour_epochs, "right") - np.searchsorted(b_ep, hour_epochs - TRAIL_S, "left")
    else:
        big_trail = np.zeros(n)
    epi = (small_trail >= MIN_SMALL) & (big_trail == 0)

    # escalation: independent mainshock within next ESC_H
    esc = np.zeros(n, bool)
    if bev:
        is_main = decluster(bev)
        for k, (t, m, la, lo) in enumerate(bev):
            if not is_main[k]: continue
            lo_i = np.searchsorted(hour_epochs, t - ESC_S, "left")
            hi_i = np.searchsorted(hour_epochs, t, "left")
            esc[max(lo_i, 0):min(hi_i, n)] = True
    episode[c] = epi; escalate[c] = esc
    n_epi += int(epi.sum()); n_esc += int((epi & esc).sum())
conn.close()

base_rate = n_esc / max(n_epi, 1)
print(f"  Episode cell-hours: {n_epi:,}  |  of those, escalate: {n_esc:,}  (base rate {base_rate*100:.1f}%)")
print(f"  (vs ~1% for the old any-M5+ target — far more balanced, and it's the real question)")

# ── feature column groups ──
catalog_cols = [feat_names.index(f) for f in te.CATALOG_FEATURES if f in feat_names]
all_cols = list(range(len(feat_names)))
print(f"  Catalog-only features: {len(catalog_cols)} | Multi-signal: {len(all_cols)}")

def assemble(lo, hi):
    X, y, z = [], [], []
    for ci, c in enumerate(cell_ids):
        m = episode[c][lo:hi]
        if m.sum() == 0: continue
        X.append(feats[c][lo:hi][m]); y.append(escalate[c][lo:hi][m].astype(np.float32))
        z.append(np.full(int(m.sum()), ci))
    return np.concatenate(X), np.concatenate(y), np.concatenate(z)

Xtr, ytr, _ = assemble(0, train_end)
Xva, yva, _ = assemble(train_end, val_end)
Xte, yte, zte = assemble(val_end, n)
print(f"\n  Episode windows: train {len(ytr):,} ({ytr.mean()*100:.0f}% esc) | "
      f"val {len(yva):,} | test {len(yte):,} ({yte.mean()*100:.0f}% esc)")

def prec_at_recall(y, p, target_recall):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec[:-1] >= target_recall
    return prec[:-1][ok].max() if ok.any() else 0.0

def run_arm(cols, label):
    names = [feat_names[i] for i in cols]
    model, _ = te.train_lgbm(Xtr[:, cols], ytr, Xva[:, cols], yva, names, tag=label)
    p = model.predict(Xte[:, cols])
    auc = roc_auc_score(yte, p); ap = average_precision_score(yte, p)
    print(f"  >>> {label}: AUC={auc:.4f}  AP={ap:.4f}  "
          f"P@R0.3={prec_at_recall(yte,p,0.3):.3f}  P@R0.5={prec_at_recall(yte,p,0.5):.3f}")
    return auc, ap, p, model

print("\n  -- ARM A: CATALOG-ONLY --")
auc_a, ap_a, p_a, _ = run_arm(catalog_cols, "CATALOG")
print("\n  -- ARM B: MULTI-SIGNAL (all features) --")
auc_b, ap_b, p_b, mdl_b = run_arm(all_cols, "MULTI-SIGNAL")

print("\n" + "=" * 70)
print("  THE THESIS TEST")
print("=" * 70)
print(f"  Catalog-only:  AUC={auc_a:.4f}  AP={ap_a:.4f}")
print(f"  Multi-signal:  AUC={auc_b:.4f}  AP={ap_b:.4f}")
print(f"  Δ AUC = {(auc_b-auc_a)*100:+.2f}pp   Δ AP = {(ap_b-ap_a)*100:+.2f}pp")
print(f"  base rate (random AP) = {yte.mean()*100:.1f}%")
if auc_b > auc_a + 0.02:
    print("  => Multi-signal data ADDS real escalation skill. The thesis holds.")
elif auc_b > auc_a:
    print("  => Marginal multi-signal lift.")
else:
    print("  => No multi-signal lift — escalation is a catalog-rate story.")

# Multi-signal feature importance (what discriminates foreshock from swarm?)
imp = mdl_b.feature_importance(importance_type="gain")
print("\n  Top escalation discriminators (multi-signal arm):")
for i in np.argsort(imp)[::-1][:20]:
    tag = ""
    fn = feat_names[i]
    if fn.startswith("gps_"): tag = "  [GPS]"
    elif fn.startswith("dart"): tag = "  [DART]"
    elif fn.startswith("tidal"): tag = "  [TIDAL]"
    elif fn.startswith("stn_"): tag = "  [SEISMIC-WF]"
    elif fn.startswith("hotspot"): tag = "  [FIRMS]"
    print(f"    {fn:30s} {imp[i]:>12.1f}{tag}")
