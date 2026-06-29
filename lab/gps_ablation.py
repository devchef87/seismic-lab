"""GPS ablation: is crustal deformation the driver of the strong tier-2 zones?

Same tier-2 setup (all zones, swarm episodes, escalation target). Two arms differ
only by whether GPS features are present. If dropping GPS tanks the strong zones
(Alaska, NZ, South America — the GPS-rich high performers), deformation is the lever.
Runs on the current cache (episode/escalation labels already stored).
"""
import os, sys
import numpy as np
import pandas as pd
import importlib.util
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location("te", os.path.join(os.path.dirname(__file__), "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
STRONG = {"alaska", "new_zealand", "south_america"}

data = np.load(CACHE, allow_pickle=True)
cell_ids = list(data["cell_ids"])
parents = dict(zip(data["cell_ids"], data["cell_parents"]))
feats = {c: data[f"feat_{c}"] for c in cell_ids}
episode = {c: data[f"epi_{c}"] for c in cell_ids}
escal = {c: data[f"esc_{c}"] for c in cell_ids}
feat_names = list(data["feature_names"])
n = len(data["hours"]); train_end = int(n * 0.70); val_end = int(n * 0.85)
cpm = {c: parents[c] for c in cell_ids}

gps_cols = [i for i, fn in enumerate(feat_names) if fn.startswith("gps_")]
keep_cols = [i for i in range(len(feat_names)) if i not in gps_cols]
print("=" * 70)
print("  GPS ABLATION — does crustal deformation drive the strong tier-2 zones?")
print(f"  GPS features dropped in NO-GPS arm: {[feat_names[i] for i in gps_cols]}")
print("=" * 70)

def assemble(lo, hi):
    X, y, z = [], [], []
    for ci, c in enumerate(cell_ids):
        m = episode[c][lo:hi] > 0.5
        if m.sum() == 0:
            continue
        X.append(feats[c][lo:hi][m]); y.append(escal[c][lo:hi][m].astype(np.float32))
        z.append(np.full(int(m.sum()), ci))
    return np.concatenate(X), np.concatenate(y), np.concatenate(z)

Xtr, ytr, _ = assemble(0, train_end)
Xva, yva, _ = assemble(train_end, val_end)
Xte, yte, zte = assemble(val_end, n)
print(f"  Episodes — train {len(ytr):,} | test {len(yte):,} ({yte.mean()*100:.1f}% escalate)")

def run(cols, label):
    names = [feat_names[i] for i in cols]
    model, _ = te.train_lgbm(Xtr[:, cols], ytr, Xva[:, cols], yva, names, tag=label)
    p = model.predict(Xte[:, cols])
    _, _, par = te.evaluate_per_zone(p, yte, zte, cell_ids, "", cpm)
    return par, roc_auc_score(yte, p), average_precision_score(yte, p)

print("\n  -- ARM A: WITH GPS --")
par_g, auc_g, ap_g = run(list(range(len(feat_names))), "WITH-GPS")
print("\n  -- ARM B: NO GPS --")
par_n, auc_n, ap_n = run(keep_cols, "NO-GPS")

print("\n" + "=" * 70)
print("  GPS ABLATION RESULT")
print("=" * 70)
print(f"  Overall pooled AUC:  with-GPS {auc_g:.4f}  | no-GPS {auc_n:.4f}  | Δ {(auc_g-auc_n)*100:+.2f}pp")
print(f"  Overall pooled AP:   with-GPS {ap_g:.4f}  | no-GPS {ap_n:.4f}  | Δ {(ap_g-ap_n)*100:+.2f}pp")
print(f"\n  {'Zone':>16s} {'withGPS':>8s} {'noGPS':>8s} {'Δ':>7s}")
for z in sorted(set(parents.values())):
    a = par_g.get(z, {}).get("auc"); b = par_n.get(z, {}).get("auc")
    if a and b:
        star = "  <- strong/GPS-rich" if z in STRONG else ""
        print(f"  {z:>16s} {a:>8.4f} {b:>8.4f} {(a-b)*100:>+6.2f}{star}")
sg = [(par_g.get(z, {}).get("auc"), par_n.get(z, {}).get("auc")) for z in STRONG
      if par_g.get(z, {}).get("auc") and par_n.get(z, {}).get("auc")]
if sg:
    dg = np.mean([a - b for a, b in sg])
    print(f"\n  Strong-zone mean Δ from GPS: {dg*100:+.2f}pp")
    print("  => GPS is a real tier-2 driver." if dg > 0.01 else
          "  => GPS not the driver — strong zones are strong for other reasons.")
