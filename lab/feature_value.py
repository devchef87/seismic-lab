"""Rigorous feature value attribution for the tier-2 escalation model.

Not LightGBM 'gain' (biased in-sample split proxy). Three honest measures:
  1. UNIVARIATE AUC      — each feature alone as a score (standalone signal)
  2. PERMUTATION IMPORT. — shuffle each feature in the trained model, AUC drop
                           (true in-model contribution, accounts for redundancy)
  3. LEAVE-ONE-GROUP-OUT — retrain without each data modality (collective value)
"""
import os, sys
import numpy as np
import pandas as pd
import importlib.util
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("te", os.path.join(os.path.dirname(__file__), "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
data = np.load(CACHE, allow_pickle=True)
cell_ids = list(data["cell_ids"])
feats = {c: data[f"feat_{c}"] for c in cell_ids}
episode = {c: data[f"epi_{c}"] for c in cell_ids}
escal = {c: data[f"esc_{c}"] for c in cell_ids}
feat_names = list(data["feature_names"])
n = len(data["hours"]); train_end = int(n * 0.70); val_end = int(n * 0.85)

def assemble(lo, hi):
    X, y = [], []
    for c in cell_ids:
        m = episode[c][lo:hi] > 0.5
        if m.sum() == 0:
            continue
        X.append(feats[c][lo:hi][m]); y.append(escal[c][lo:hi][m].astype(np.float32))
    return np.concatenate(X), np.concatenate(y)

Xtr, ytr = assemble(0, train_end)
Xva, yva = assemble(train_end, val_end)
Xte, yte = assemble(val_end, n)
print("=" * 70)
print("  FEATURE VALUE ATTRIBUTION — tier-2 escalation")
print(f"  train {len(ytr):,} | test {len(yte):,} ({yte.mean()*100:.1f}% escalate)")
print("=" * 70)

GROUPS = {
    "catalog_seismicity": te.CATALOG_FEATURES,
    "tidal": te.TIDAL_FEATURES + te.TIDAL_TRIGGER_FEATURES,
    "dart": te.DART_FEATURES,
    "gps_deformation": te.GPS_FEATURES,
    "station_waveform": te.STATION_FEATURES,
    "volcanic": te.VOLCANIC_FEATURES,
    "firms_thermal": te.FIRMS_FEATURES,
    "ground_mag": te.GROUND_MAG_FEATURES,
    "solar_geomag": (te.GEOMAG_FEATURES + te.SOLAR_FEATURES + te.COSMIC_FEATURES +
                     te.IEF_FEATURES + te.CME_FEATURES + te.FLARE_FEATURES +
                     te.STORM_FEATURES + te.OLR_FEATURES),
    "sw_trajectory": te.TRAJECTORY_FEATURES + te.SHAPE_FEATURES + te.COUPLING_FEATURES,
    "interactions": te.INTERACTION_FEATURES,
    "velocity": te.VELOCITY_FEATURES,
}
name_to_idx = {fn: i for i, fn in enumerate(feat_names)}

# ── Full model baseline ──
full_model, _ = te.train_lgbm(Xtr, ytr, Xva, yva, feat_names, tag="FULL")
base_auc = roc_auc_score(yte, full_model.predict(Xte))
print(f"\n  FULL MODEL test AUC = {base_auc:.4f}")

# ── 1. Univariate AUC (standalone, instant) ──
uni = {}
for i, fn in enumerate(feat_names):
    col = np.nan_to_num(Xte[:, i], nan=0.0)
    if np.std(col) < 1e-12:
        uni[fn] = 0.5; continue
    a = roc_auc_score(yte, col)
    uni[fn] = max(a, 1 - a)  # directionless

# ── 2. Permutation importance (in-model contribution) ──
rng = np.random.RandomState(0)
perm = {}
for i, fn in enumerate(feat_names):
    saved = Xte[:, i].copy()
    Xte[:, i] = rng.permutation(saved)
    perm[fn] = base_auc - roc_auc_score(yte, full_model.predict(Xte))
    Xte[:, i] = saved

# ── 3. Leave-one-group-out (collective, retrain) ──
print("\n  Leave-one-group-out (retraining)...")
logo = {}
for gname, members in GROUPS.items():
    drop = set(name_to_idx[m] for m in members if m in name_to_idx)
    cols = [i for i in range(len(feat_names)) if i not in drop]
    names = [feat_names[i] for i in cols]
    m, _ = te.train_lgbm(Xtr[:, cols], ytr, Xva[:, cols], yva, names, tag=f"drop-{gname}")
    logo[gname] = base_auc - roc_auc_score(yte, m.predict(Xte[:, cols]))

# ── Report ──
print("\n" + "=" * 70)
print("  COLLECTIVE CONTRIBUTION — leave-one-group-out (AUC drop when removed)")
print("=" * 70)
print(f"  {'data stream':>20s} {'#feat':>5s} {'AUC drop':>9s}")
for g, d in sorted(logo.items(), key=lambda x: -x[1]):
    print(f"  {g:>20s} {len([m for m in GROUPS[g] if m in name_to_idx]):>5d} {d*100:>+8.2f}pp")

print("\n" + "=" * 70)
print("  TOP INDIVIDUAL FEATURES — permutation importance (in-model)")
print("=" * 70)
print(f"  {'feature':>26s} {'perm-drop':>10s} {'univariate':>11s}")
for fn, d in sorted(perm.items(), key=lambda x: -x[1])[:25]:
    print(f"  {fn:>26s} {d*100:>+9.3f}pp {uni[fn]:>10.4f}")

print("\n" + "=" * 70)
print("  STRONGEST STANDALONE FEATURES — univariate AUC (may be redundant)")
print("=" * 70)
print(f"  {'feature':>26s} {'univariate':>11s} {'perm-drop':>10s}")
for fn, a in sorted(uni.items(), key=lambda x: -x[1])[:20]:
    print(f"  {fn:>26s} {a:>10.4f} {perm[fn]*100:>+9.3f}pp")
