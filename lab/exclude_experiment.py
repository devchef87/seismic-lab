"""Train on quality zones, predict on all. Tests whether dropping the noisy
catalog-only/sparse zones FROM TRAINING (while keeping them as targets) lifts the
overall 13-zone result — capturing the dilution gain without losing breadth.

  A) train pooled on ALL 13 zones      (the v8 baseline)
  B) train pooled on all EXCEPT {png_solomon, caribbean, philippines}
Both evaluated on ALL 13 zones' test windows.
"""
import os, sys
import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("te", os.path.join(os.path.dirname(__file__), "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
EXCLUDE = {"png_solomon", "caribbean", "philippines"}  # weak (<0.60) AND sparse (2 cells)

data = np.load(CACHE, allow_pickle=True)
cell_ids = list(data["cell_ids"])
parents = dict(zip(data["cell_ids"], data["cell_parents"]))
feats = {c: data[f"feat_{c}"] for c in cell_ids}
tgts = {c: data[f"tgt_{c}"] for c in cell_ids}
hours = pd.DatetimeIndex(data["hours"]); n = len(hours)
feat_names = list(data["feature_names"])
train_end = int(n * 0.70); val_end = int(n * 0.85)
cpm = {c: parents[c] for c in cell_ids}

print("=" * 70)
print("  EXCLUDE-FROM-TRAIN TEST — keep all 13 as targets")
print(f"  Excluded from training: {sorted(EXCLUDE)}")
print("=" * 70)

# Eval set = ALL 13 zones
def make_eval(cells, lo, hi):
    X, y, z = [], [], []
    for ci, c in enumerate(cells):
        X.append(feats[c][lo:hi]); y.append(tgts[c][lo:hi]); z.append(np.full(hi - lo, ci))
    return np.concatenate(X), np.concatenate(y), np.concatenate(z)

X_val, y_val, _ = make_eval(cell_ids, train_end, val_end)
X_test, y_test, test_z = make_eval(cell_ids, val_end, n)

def run(train_cells, label):
    Xtr, ytr = te.smart_negative_sample(feats, tgts, train_cells, 0, train_end,
                                        neg_ratio=4, boundary_hours=3)
    print(f"\n  [{label}] train cells={len(train_cells)}  samples={Xtr.shape[0]:,}  pos={int(ytr.sum())}")
    model, _ = te.train_lgbm(Xtr, ytr, X_val, y_val, feat_names, tag=label)
    probs = model.predict(X_test)
    _, _, parent = te.evaluate_per_zone(probs, y_test, test_z, cell_ids, f"{label} TEST", cpm)
    return parent

par_all = run(cell_ids, "ALL-13")
par_clean = run([c for c in cell_ids if parents[c] not in EXCLUDE], "CLEAN-train")

print("\n" + "=" * 70)
print("  RESULT (all 13 zones as targets)")
print("=" * 70)
print(f"  {'Zone':>16s} {'ALL-13':>8s} {'CLEAN':>8s} {'Δ':>7s}")
zones = sorted(set(parents.values()))
a_aucs, c_aucs = [], []
for z in zones:
    a = par_all.get(z, {}).get("auc"); c = par_clean.get(z, {}).get("auc")
    if a and c:
        a_aucs.append(a); c_aucs.append(c)
        tag = "  <-excl-from-train" if z in EXCLUDE else ""
        print(f"  {z:>16s} {a:>8.4f} {c:>8.4f} {(c-a)*100:>+6.2f}{tag}")
print(f"  {'MACRO':>16s} {np.mean(a_aucs):>8.4f} {np.mean(c_aucs):>8.4f} {(np.mean(c_aucs)-np.mean(a_aucs))*100:>+6.2f}")
