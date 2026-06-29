"""Dilution test: does pooling sparse/weak zones hurt the strong ones?

This run (13-zone pooled) dipped the original strong zones ~1pp vs the 7-zone run.
Hypothesis: feeding noisy/sparse/catalog-only zones (PNG, Caribbean, Philippines)
into the single pooled model blurs the shared physics for data-rich zones.

Test, on the existing cache (no rebuild):
  A) train pooled on ALL 13 zones
  B) train pooled on STRONG data-rich subduction zones only
Both evaluated on the SAME strong-zone test windows. If B > A on the strong zones,
pooling the weak zones is hurting, and regime-grouping is the way.
"""
import os, sys
import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("te", os.path.join(os.path.dirname(__file__), "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
STRONG = {"indonesia", "japan_kurils", "alaska", "south_america", "mexico_ca"}

data = np.load(CACHE, allow_pickle=True)
cell_ids = list(data["cell_ids"])
parents = dict(zip(data["cell_ids"], data["cell_parents"]))
feats = {c: data[f"feat_{c}"] for c in cell_ids}
tgts = {c: data[f"tgt_{c}"] for c in cell_ids}
hours = pd.DatetimeIndex(data["hours"]); n = len(hours)
feat_names = list(data["feature_names"])
train_end = int(n * 0.70); val_end = int(n * 0.85)

strong_cells = [c for c in cell_ids if parents[c] in STRONG]
cpm = {c: parents[c] for c in cell_ids}

print("=" * 70)
print("  DILUTION TEST — all-13 pooled vs strong-only pooled")
print(f"  Strong zones: {sorted(STRONG)}  ({len(strong_cells)} cells)")
print(f"  All zones: {len(cell_ids)} cells")
print("=" * 70)

# Shared evaluation set = strong zones' val/test
def make_eval(cells, lo, hi):
    X, y, z = [], [], []
    for ci, c in enumerate(cells):
        X.append(feats[c][lo:hi]); y.append(tgts[c][lo:hi])
        z.append(np.full(hi - lo, ci))
    return np.concatenate(X), np.concatenate(y), np.concatenate(z)

X_val, y_val, _ = make_eval(strong_cells, train_end, val_end)
X_test, y_test, test_z = make_eval(strong_cells, val_end, n)

def train_on(train_cells, label):
    Xtr, ytr = te.smart_negative_sample(feats, tgts, train_cells, 0, train_end,
                                        neg_ratio=4, boundary_hours=3)
    print(f"\n  [{label}] train: {Xtr.shape[0]:,} samples, {int(ytr.sum())} pos")
    model, _ = te.train_lgbm(Xtr, ytr, X_val, y_val, feat_names, tag=label)
    probs = model.predict(X_test)
    _, _, parent = te.evaluate_per_zone(probs, y_test, test_z, strong_cells,
                                        f"{label} TEST", cpm)
    aucs = [m["auc"] for m in parent.values()]
    return np.mean(aucs), parent

macro_all, par_all = train_on(cell_ids, "ALL-13")
macro_strong, par_strong = train_on(strong_cells, "STRONG-only")

print("\n" + "=" * 70)
print("  DILUTION RESULT (evaluated on strong zones)")
print("=" * 70)
print(f"  {'Zone':>16s} {'ALL-13':>8s} {'STRONG':>8s} {'Δ':>7s}")
for z in sorted(STRONG):
    a = par_all.get(z, {}).get("auc"); b = par_strong.get(z, {}).get("auc")
    if a and b:
        print(f"  {z:>16s} {a:>8.4f} {b:>8.4f} {(b-a)*100:>+6.2f}")
print(f"  {'MACRO':>16s} {macro_all:>8.4f} {macro_strong:>8.4f} {(macro_strong-macro_all)*100:>+6.2f}")
print(f"\n  {'STRONG-only helps' if macro_strong > macro_all else 'pooling all is fine'}")
