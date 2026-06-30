#!/usr/bin/env python3
"""Do planetary alignments / deeper lunar geometry add skill to the tier-2 model?
Same standalone ablation as TEC: compute global orbital features per hour (skyfield),
broadcast to all cells, train baseline vs baseline+orbital, compare within-zone AUC.

Lunar features have a real mechanism (tides — already partly in the model); planetary
ones are a falsifiable long shot (planet tides on Earth are ~1e-5 of the Moon's). The
feature-importance + AUC delta will prove or disprove each."""
import os, sys, importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("te", "lab/train_ensemble.py")
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")
LUNAR = ["lunar_syzygy", "lunar_phase_sin", "lunar_phase_cos", "lunar_tidal", "lunar_decl"]
PLANET = ["planet_tidal_total", "planet_align", "jupiter_tidal", "venus_tidal"]
ORBITAL = LUNAR + PLANET
MASS = {"venus": 4.867e24, "mars": 6.417e23, "jupiter barycenter": 1.898e27,
        "saturn barycenter": 5.683e26, "mercury": 3.301e23}


def orbital_features(hours_ts):
    from skyfield.api import load
    ts = load.timescale(); eph = load("de421.bsp")
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    t = ts.from_datetimes([h.to_pydatetime() for h in hours_ts])
    e = earth.at(t)
    ma = e.observe(moon).apparent(); sa = e.observe(sun).apparent()
    elong = ma.separation_from(sa).radians
    dmoon = ma.distance().km
    _, dec_m, _ = ma.radec()
    out = {
        "lunar_syzygy": np.abs(np.cos(elong)).astype(np.float32),       # 1 at new/full
        "lunar_phase_sin": np.sin(elong).astype(np.float32),
        "lunar_phase_cos": np.cos(elong).astype(np.float32),
        "lunar_tidal": ((384400.0 / dmoon) ** 3).astype(np.float32),    # perigee boost (~1)
        "lunar_decl": dec_m.degrees.astype(np.float32),
    }
    n = len(hours_ts)
    wsum = np.zeros(n); vec = np.zeros((3, n)); per = {}
    for key, mass in MASS.items():
        pa = e.observe(eph[key]).apparent()
        d = pa.distance().km
        w = mass / d ** 3
        per[key] = (w / w.mean()).astype(np.float32)
        xyz = pa.position.km; nrm = np.sqrt((xyz ** 2).sum(0))
        vec += w * (xyz / nrm); wsum += w
    out["planet_tidal_total"] = (wsum / wsum.mean()).astype(np.float32)
    out["planet_align"] = (np.sqrt((vec ** 2).sum(0)) / wsum).astype(np.float32)  # 0..1
    out["jupiter_tidal"] = per["jupiter barycenter"]
    out["venus_tidal"] = per["venus"]
    return np.column_stack([out[k] for k in ORBITAL])


def main():
    d = np.load(CACHE, allow_pickle=True)
    cell_ids = list(d["cell_ids"]); parents = dict(zip(d["cell_ids"], d["cell_parents"]))
    feats = {c: d[f"feat_{c}"] for c in cell_ids}
    epi = {c: d[f"epi_{c}"] for c in cell_ids}; esc = {c: d[f"esc_{c}"] for c in cell_ids}
    names = list(d["feature_names"])
    hours_ts = pd.DatetimeIndex(d["hours"])
    if hours_ts.tz is None:
        hours_ts = hours_ts.tz_localize("UTC")
    n = len(hours_ts); tr, va = int(n * 0.70), int(n * 0.85)

    print("computing orbital features (skyfield)...")
    orb = orbital_features(hours_ts)  # (n, len(ORBITAL)), global
    print("orbital ranges:")
    for j, nm in enumerate(ORBITAL):
        print(f"   {nm:<20} {orb[:,j].min():.3f} .. {orb[:,j].max():.3f}")

    def assemble(lo, hi, extra):
        X, y, Z = [], [], []
        og = orb[lo:hi]
        for c in cell_ids:
            m = epi[c][lo:hi] > 0.5
            if not m.sum():
                continue
            base = feats[c][lo:hi][m]
            X.append(np.hstack([base, og[m]]) if extra else base)
            y.append(esc[c][lo:hi][m].astype(np.float32)); Z += [parents[c]] * int(m.sum())
        return np.concatenate(X), np.concatenate(y), np.array(Z)

    def run(tag, extra):
        fn = names + ORBITAL if extra else names
        Xtr, ytr, _ = assemble(0, tr, extra); Xva, yva, _ = assemble(tr, va, extra)
        Xte, yte, Zte = assemble(va, n, extra)
        model, _ = te.train_lgbm(Xtr, ytr, Xva, yva, fn, tag=tag)
        aucs = {}
        for z in set(Zte):
            mm = Zte == z; s = yte[mm].sum()
            if mm.sum() >= 30 and 0 < s < mm.sum():
                aucs[z] = roc_auc_score(yte[mm], model.predict(Xte[mm]))
        macro = float(np.mean(list(aucs.values()))); pooled = roc_auc_score(yte, model.predict(Xte))
        print(f"\n##### {tag}: macro {macro:.4f} | pooled {pooled:.4f}")
        if extra:
            imp = dict(zip(fn, model.feature_importance(importance_type="gain")))
            tot = sum(imp.values()) or 1
            for f in ORBITAL:
                print(f"     {f:<20} {100*imp.get(f,0)/tot:.2f}% gain")
        return macro, aucs

    b_macro, b = run("baseline", False)
    o_macro, o = run("baseline + orbital", True)
    print("\n" + "=" * 56)
    print(f"  macro AUC: {b_macro:.4f} -> {o_macro:.4f}  (Δ {o_macro-b_macro:+.4f})")
    for z in sorted(b, key=lambda z: -(o.get(z, 0) - b[z])):
        if z in o:
            print(f"     {z:<16}{b[z]:.3f} -> {o[z]:.3f}  ({o[z]-b[z]:+.3f})")


if __name__ == "__main__":
    main()
