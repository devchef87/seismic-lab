#!/usr/bin/env python3
"""Decisive CA test: does the deep 1985+ catalog help if we use only features that
exist across ALL eras (catalog + tidal + static geophysics), dropping the post-2015
sensor features that are NaN in the old data? If this beats the 2015+ model (0.63),
deep history helps; if not, ~0.6 is California's physics ceiling."""
import os, sys, importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("te", "lab/train_ensemble.py")
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from1985-01-01_zcalifornia.npz")
d = np.load(CACHE, allow_pickle=True)
cells = list(d["cell_ids"]); names = list(d["feature_names"])
feats = {c: d[f"feat_{c}"] for c in cells}
epi = {c: d[f"epi_{c}"] for c in cells}
esc = {c: d[f"esc_{c}"] for c in cells}
n = len(pd.DatetimeIndex(d["hours"])); tr, va = int(n*0.70), int(n*0.85)

def assemble(lo, hi, cm):
    X, y = [], []
    for c in cells:
        m = epi[c][lo:hi] > 0.5
        if m.sum():
            X.append(feats[c][lo:hi][m][:, cm]); y.append(esc[c][lo:hi][m].astype(np.float32))
    return np.concatenate(X), np.concatenate(y)

# data-driven "available across all eras" = low NaN over all active episodes
allX = np.concatenate([feats[c][epi[c] > 0.5] for c in cells])
nan_rate = np.isnan(allX).mean(axis=0)
keep = nan_rate < 0.20
print(f"features kept (<20% NaN across 1985-2026): {keep.sum()}/{len(names)}")
print("dropped (sparse pre-2015):", [names[i] for i in range(len(names)) if not keep[i]][:20], "...")

def run(tag, cm):
    Xtr, ytr = assemble(0, tr, cm); Xva, yva = assemble(tr, va, cm); Xte, yte = assemble(va, n, cm)
    model, _ = te.train_lgbm(Xtr, ytr, Xva, yva, [names[i] for i in range(len(names)) if cm[i]], tag=tag)
    auc_va = roc_auc_score(yva, model.predict(Xva)); auc_te = roc_auc_score(yte, model.predict(Xte))
    print(f"\n##### {tag}: val {auc_va:.3f} | TEST {auc_te:.3f}  (train pos {int(ytr.sum())}, test pos {int(yte.sum())})")

run("CA 1985 ALL features", np.ones(len(names), bool))
run("CA 1985 catalog+tidal only", keep)
