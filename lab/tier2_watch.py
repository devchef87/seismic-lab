"""SeismicLab — Tier-2 Swarm-Escalation Watch (operational product layer).

Turns the tier-2 model into a deployable watch system:
  - trains the escalation model (swarm -> independent mainshock in 72h)
  - CALIBRATES raw scores to true probabilities (isotonic on validation)
  - defines ADVISORY / WATCH / WARNING alert levels from calibrated probability
  - scores the most recent window per cell -> current active-swarm watch list
  - saves a deployable bundle + writes tier2_watch.json for the UI

Output probability is the calibrated chance an active swarm escalates to an
independent M5+ mainshock within 72h.
"""
import os, sys, json
import numpy as np
import pandas as pd
import importlib.util
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

spec = importlib.util.spec_from_file_location("te", os.path.join(os.path.dirname(__file__), "train_ensemble.py"))
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
MODEL_DIR = te.MODEL_DIR
OUT_JSON = os.path.join(te.CACHE_DIR, "tier2_watch.json")

data = np.load(CACHE, allow_pickle=True)
cell_ids = list(data["cell_ids"])
parents = dict(zip(data["cell_ids"], data["cell_parents"]))
feats = {c: data[f"feat_{c}"] for c in cell_ids}
episode = {c: data[f"epi_{c}"] for c in cell_ids}
escal = {c: data[f"esc_{c}"] for c in cell_ids}
feat_names = list(data["feature_names"])
hours = pd.DatetimeIndex(data["hours"]); n = len(hours)
train_end = int(n * 0.70); val_end = int(n * 0.85)

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
Xva, yva, zva = assemble(train_end, val_end)
Xte, yte, zte = assemble(val_end, n)
print("=" * 70)
print("  TIER-2 SWARM-ESCALATION WATCH — operational build")
print(f"  train {len(ytr):,} | val {len(yva):,} | test {len(yte):,} episodes")
print("=" * 70)

# ── Train + calibrate ──
model, _ = te.train_lgbm(Xtr, ytr, Xva, yva, feat_names, tag="tier2-watch")
raw_va = model.predict(Xva); raw_te = model.predict(Xte)
iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va, yva)
cal_te = iso.predict(raw_te)

auc = roc_auc_score(yte, raw_te); ap = average_precision_score(yte, raw_te)
print(f"\n  Test AUC {auc:.4f}  AP {ap:.4f}  (base rate {yte.mean()*100:.1f}%)")

# ── Calibration check (reliability) ──
print("\n  Calibration (predicted vs actual escalation rate):")
bins = [0, 0.08, 0.15, 0.30, 1.01]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (cal_te >= lo) & (cal_te < hi)
    if m.sum() > 0:
        print(f"    pred {lo:.2f}-{hi:.2f}: actual {yte[m].mean()*100:5.1f}%  (n={int(m.sum())})")

# ── Alert levels by RISK PERCENTILE (set on validation -> always-populated tiers) ──
# Isotonic gives a bimodal prob distribution, so fixed-prob bands leave WATCH empty.
# Percentile thresholds on the raw score guarantee graded, populated tiers; we still
# DISPLAY the honest calibrated probability for each alert.
THR_ADV = float(np.percentile(raw_va, 80))    # top 20% of active swarms -> ADVISORY
THR_WATCH = float(np.percentile(raw_va, 95))  # top 5% -> WATCH
WARN_PROB = 0.30   # WARNING = calibrated prob >= 30% (rare, high-confidence subset)
def level(raw, cal):
    return np.where(cal >= WARN_PROB, 3, np.where(raw >= THR_WATCH, 2,
                    np.where(raw >= THR_ADV, 1, 0)))
LEVEL_NAMES = ["NORMAL", "ADVISORY", "WATCH", "WARNING"]
lv = level(raw_te, cal_te)
print("\n  Alert-level performance (test) — percentile-banded:")
print(f"  {'level':>10s} {'count':>7s} {'%swarms':>8s} {'precision':>9s} {'recall':>7s} {'typ.prob':>9s}")
for name, code in [("WARNING", 3), ("WATCH+", 2), ("ADVISORY+", 1)]:
    fire = lv >= code
    if fire.sum() > 0:
        prec = precision_score(yte, fire, zero_division=0)
        rec = recall_score(yte, fire, zero_division=0)
        typ = cal_te[fire].mean()
        print(f"  {name:>10s} {int(fire.sum()):>7d} {fire.mean()*100:>7.1f}% {prec:>9.3f} {rec:>7.3f} {typ*100:>8.1f}%")

# ── Current watch: score the most recent window per cell ──
geom = {c["id"]: c for c in te.generate_grid_cells(te.PARENT_ZONES)}
watch = []
last = n - 1
for ci, c in enumerate(cell_ids):
    if episode[c][last] < 0.5:
        continue  # no active swarm in this cell right now
    raw = model.predict(feats[c][last:last + 1])[0]
    p = float(iso.predict([raw])[0])
    lvl = LEVEL_NAMES[int(level(np.array([raw]), np.array([p]))[0])]
    g = geom.get(c, {})
    watch.append({
        "cell": c, "zone": parents.get(c, c),
        "lat_range": g.get("lat_range"), "lon_range": g.get("lon_range"),
        "escalation_prob_72h": round(p, 3),
        "alert_level": lvl,
        "lift_vs_base": round(p / max(yte.mean(), 1e-6), 1),
    })
watch.sort(key=lambda w: -w["escalation_prob_72h"])

print(f"\n  CURRENT ACTIVE-SWARM WATCH ({len(watch)} cells with active swarms):")
print(f"  {'zone':>16s} {'cell':>22s} {'P(esc72h)':>10s} {'level':>9s}")
for w in watch[:15]:
    print(f"  {w['zone']:>16s} {w['cell']:>22s} {w['escalation_prob_72h']:>10.3f} {w['alert_level']:>9s}")

# ── Save deployable bundle + UI JSON ──
model.save_model(os.path.join(MODEL_DIR, "tier2_watch_lgb.txt"))
np.savez(os.path.join(MODEL_DIR, "tier2_watch_calib.npz"),
         iso_x=iso.X_thresholds_, iso_y=iso.y_thresholds_,
         feature_names=np.array(feat_names),
         raw_thresholds=np.array([THR_ADV, THR_WATCH]), warn_prob=WARN_PROB,
         base_rate=float(yte.mean()))
def _jsafe(o):
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    return str(o)
with open(OUT_JSON, "w") as f:
    json.dump({"generated": str(hours[last]), "base_rate_72h": round(float(yte.mean()), 4),
               "n_active_swarms": len(watch), "watch": watch}, f, indent=2, default=_jsafe)
print(f"\n  Saved model + calibrator to {MODEL_DIR}")
print(f"  Wrote UI watch -> {OUT_JSON}")
print("  Done.")
