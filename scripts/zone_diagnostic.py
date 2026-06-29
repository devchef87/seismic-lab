#!/usr/bin/env python3
"""Why do some zones score badly? Per-zone breakdown of sample size (can we learn /
can we even measure?), feature coverage (GPS/DART/waveform starvation), and base rate.
Runs on the existing feature cache — no rebuild."""
import os, sys, importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("te", "lab/train_ensemble.py")
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

data = np.load(os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz"),
               allow_pickle=True)
cell_ids = list(data["cell_ids"])
parents = dict(zip(data["cell_ids"], data["cell_parents"]))
feats = {c: data[f"feat_{c}"] for c in cell_ids}
episode = {c: data[f"epi_{c}"] for c in cell_ids}
escal = {c: data[f"esc_{c}"] for c in cell_ids}
feat_names = list(data["feature_names"])
n = len(pd.DatetimeIndex(data["hours"]))
train_end, val_end = int(n * 0.70), int(n * 0.85)

# feature-group column indices
def idx(pred): return [i for i, fn in enumerate(feat_names) if pred(fn)]
GRP = {
    "gps":   idx(lambda f: f.startswith("gps_")),
    "dart":  idx(lambda f: f.startswith("dart_")),
    "wave":  idx(lambda f: any(k in f for k in ("sta_lta", "amp_", "station", "_mag_72h", "mag_x", "mag_y", "mag_z"))),
}

# known per-zone holdout AUC from the experiment (for context)
AUC = {"south_america": 0.73, "alaska": 0.72, "new_zealand": 0.67, "japan_kurils": 0.59,
       "mexico_ca": 0.55, "indonesia": 0.56, "himalaya": 0.52, "california": 0.41,
       "kamchatka": 0.43, "png_solomon": 0.42, "philippines": 0.51, "caribbean": 0.36,
       "mediterranean": 0.10}

# stations per zone (waveform coverage proxy)
zbbox = {z["id"]: (z["lat_range"], z["lon_range"]) for z in te.PARENT_ZONES} if hasattr(te, "PARENT_ZONES") else {}
def n_stations(zone):
    if zone not in zbbox:
        return None
    (la0, la1), (lo0, lo1) = zbbox[zone]
    return sum(1 for s in getattr(te, "STATIONS", []) if la0 <= s["lat"] <= la1 and lo0 <= s["lon"] <= lo1)

def agg(zone, lo, hi):
    eps = pos = 0
    for c in cell_ids:
        if parents[c] != zone:
            continue
        m = episode[c][lo:hi] > 0.5
        eps += int(m.sum()); pos += int(escal[c][lo:hi][m].sum())
    return eps, pos

def nan_frac(zone, cols):
    if not cols:
        return float("nan")
    vals = []
    for c in cell_ids:
        if parents[c] != zone:
            continue
        m = episode[c] > 0.5
        if m.sum():
            vals.append(feats[c][m][:, cols])
    if not vals:
        return float("nan")
    A = np.concatenate(vals)
    return float(np.isnan(A).mean())

zones = sorted(set(parents.values()), key=lambda z: -AUC.get(z, 0))
print(f"{'zone':<15}{'AUC':>5}{'cells':>6}{'tr_eps':>8}{'tr_pos':>7}{'te_eps':>8}{'te_pos':>7}{'base%':>6}{'sta':>4}{'gpsNaN':>7}{'dartNaN':>8}")
print("-" * 86)
for z in zones:
    cells = sum(1 for c in cell_ids if parents[c] == z)
    tre, trp = agg(z, 0, train_end)
    tee, tep = agg(z, val_end, n)
    base = 100 * tep / tee if tee else 0
    sta = n_stations(z)
    print(f"{z:<15}{AUC.get(z,0):>5.2f}{cells:>6}{tre:>8}{trp:>7}{tee:>8}{tep:>7}{base:>6.1f}"
          f"{('-' if sta is None else sta):>4}{100*nan_frac(z,GRP['gps']):>6.0f}%{100*nan_frac(z,GRP['dart']):>7.0f}%")
print("\nte_pos = escalation events in the TEST window (need >~30 for a stable AUC).")
print("tr_pos = escalations available to LEARN from. sta = seismic stations in-zone.")
