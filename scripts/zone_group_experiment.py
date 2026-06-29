#!/usr/bin/env python3
"""Fast A/B on the EXISTING feature cache: does zone-grouping / dead-feature removal
beat the single pooled tier-2 model on within-zone (macro) AUC? No feature rebuild."""
import os, sys, importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("te", "lab/train_ensemble.py")
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
data = np.load(CACHE, allow_pickle=True)
cell_ids = list(data["cell_ids"])
parents = dict(zip(data["cell_ids"], data["cell_parents"]))
feats = {c: data[f"feat_{c}"] for c in cell_ids}
episode = {c: data[f"epi_{c}"] for c in cell_ids}
escal = {c: data[f"esc_{c}"] for c in cell_ids}
feat_names = list(data["feature_names"])
n = len(pd.DatetimeIndex(data["hours"]))
train_end, val_end = int(n * 0.70), int(n * 0.85)

# zones with genuine within-zone test skill from the pooled run (>~0.58)
STRONG = {"alaska", "new_zealand", "south_america", "japan_kurils", "california", "mexico_ca"}
DROP = set(getattr(te, "VELOCITY_FEATURES", [])) | {"omori_deficit"}
ALL = set(parents.values())


def colmask(drop):
    return np.array([fn not in drop for fn in feat_names])


def assemble(lo, hi, cells, cm):
    X, y, Z = [], [], []
    for c in cells:
        m = episode[c][lo:hi] > 0.5
        if m.sum() == 0:
            continue
        X.append(feats[c][lo:hi][m][:, cm])
        y.append(escal[c][lo:hi][m].astype(np.float32))
        Z += [parents[c]] * int(m.sum())
    return np.concatenate(X), np.concatenate(y), np.array(Z)


def macro(model, Xte, yte, Zte, zones):
    out = {}
    for z in zones:
        mm = Zte == z
        s = yte[mm].sum()
        if mm.sum() < 30 or s == 0 or s == mm.sum():
            continue
        out[z] = roc_auc_score(yte[mm], model.predict(Xte[mm]))
    return out


def run(name, train_cells, eval_zones, drop=set()):
    cm = colmask(drop)
    names = [fn for fn in feat_names if fn not in drop]
    Xtr, ytr, _ = assemble(0, train_end, train_cells, cm)
    Xva, yva, _ = assemble(train_end, val_end, train_cells, cm)
    ev_cells = [c for c in cell_ids if parents[c] in eval_zones]
    Xte, yte, Zte = assemble(val_end, n, ev_cells, cm)
    model, _ = te.train_lgbm(Xtr, ytr, Xva, yva, names, tag=name)
    aucs = macro(model, Xte, yte, Zte, eval_zones)
    m = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    print(f"\n##### {name}: MACRO AUC over {len(aucs)} zones = {m:.4f}")
    for z, a in sorted(aucs.items(), key=lambda x: -x[1]):
        print(f"     {z:<16}{a:.3f}")
    return m, aucs


print("\n" + "=" * 60)
res = {}
res["A_pooled_all"] = run("A: pooled / all zones", cell_ids, ALL)[0]
res["B_pooled_strong"] = run("B: pooled / eval strong", cell_ids, STRONG)[0]
res["C_pooled_dropdead_strong"] = run("C: pooled - dead feats / eval strong", cell_ids, STRONG, DROP)[0]
res["D_strongonly"] = run("D: strong-zones-only model", [c for c in cell_ids if parents[c] in STRONG], STRONG)[0]
res["E_strongonly_dropdead"] = run("E: strong-only - dead feats", [c for c in cell_ids if parents[c] in STRONG], STRONG, DROP)[0]

print("\n" + "=" * 60)
print("  SUMMARY (macro AUC, higher = better within-zone ranking)")
for k, v in res.items():
    print(f"    {k:<32}{v:.4f}")
