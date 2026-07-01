#!/usr/bin/env python3
"""Sequence-gated ablation: do environmental features gain skill when conditioned
on active sequences? Three experiments:

  A. Catalog-only model on ALL hours (baseline — how good is sequence alone?)
  B. Environmental-only model on ALL hours (how good is environment alone?)
  C. Environmental-only model on ACTIVE-SEQUENCE hours only (the key test —
     does environment gain skill when conditioned on "we're in a sequence"?)

If C >> B, the gated architecture is justified — environmental signals matter
for modulating active sequences, not for cold-start prediction."""
import os, sys, importlib.util, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("te", "lab/train_ensemble.py")
te = importlib.util.module_from_spec(spec); sys.modules["te"] = te; spec.loader.exec_module(te)

CACHE = os.path.join(te.CACHE_DIR, "ensemble_v7_grid10_m5.0_h6_from2021-07-01.npz")

CATALOG_ONLY = [
    "eq_count_1h", "eq_count_6h", "eq_count_24h", "eq_count_72h",
    "eq_count_7d", "eq_count_30d", "m4_count_24h", "m4_count_7d",
    "foreshock_accel", "rate_anomaly_7d",
    "b_value_14d", "b_value_30d", "b_delta",
    "energy_24h", "energy_ratio",
    "max_mag_72h", "max_mag_7d", "mag_range_7d",
    "moment_7d", "moment_30d", "moment_accel",
    "time_since_last", "depth_mean_7d",
    "b_value_accel", "coulomb_proxy", "omori_deficit",
    "subduction_proxy", "foreshock_curvature",
    "seq_consec_up_7d", "seq_rumble_ratio_7d", "seq_mag_trend_72h",
    "seq_double_tap_7d", "seq_depth_to_peak_7d", "seq_active_ratio",
]

ENV_ONLY = [f for f in te.ALL_FEATURES if f not in CATALOG_ONLY]


def main():
    print("Loading cached features...")
    d = np.load(CACHE, allow_pickle=True)
    cell_ids = list(d["cell_ids"])
    parents = dict(zip(d["cell_ids"], d["cell_parents"]))
    feats = {c: d[f"feat_{c}"] for c in cell_ids}
    epi = {c: d[f"epi_{c}"] for c in cell_ids}
    esc = {c: d[f"esc_{c}"] for c in cell_ids}
    all_names = list(d["feature_names"])
    hours_ts = pd.DatetimeIndex(d["hours"])
    if hours_ts.tz is None:
        hours_ts = hours_ts.tz_localize("UTC")
    n = len(hours_ts)
    tr, va = int(n * 0.70), int(n * 0.85)

    # Feature indices
    cat_idx = [all_names.index(f) for f in CATALOG_ONLY if f in all_names]
    env_idx = [all_names.index(f) for f in ENV_ONLY if f in all_names]
    cat_names = [all_names[i] for i in cat_idx]
    env_names = [all_names[i] for i in env_idx]

    # Activity threshold: eq_count_7d >= 3 (at least 3 events in 7 days)
    eq7d_idx = all_names.index("eq_count_7d")

    print(f"Catalog features: {len(cat_idx)}")
    print(f"Environmental features: {len(env_idx)}")
    print(f"Total hours: {n:,}, train/val/test: {tr}/{va-tr}/{n-va}")

    def assemble(lo, hi, feat_indices, active_only=False):
        X, y, Z = [], [], []
        for c in cell_ids:
            m = epi[c][lo:hi] > 0.5
            if active_only:
                act = feats[c][lo:hi][:, eq7d_idx] >= 3.0
                m = m & act
            if not m.sum():
                continue
            X.append(feats[c][lo:hi][m][:, feat_indices])
            y.append(esc[c][lo:hi][m].astype(np.float32))
            Z += [parents[c]] * int(m.sum())
        return np.concatenate(X), np.concatenate(y), np.array(Z)

    def run(tag, feat_indices, feat_names, active_only=False):
        Xtr, ytr, _ = assemble(0, tr, feat_indices, active_only)
        Xva, yva, _ = assemble(tr, va, feat_indices, active_only)
        Xte, yte, Zte = assemble(va, n, feat_indices, active_only)

        print(f"\n{'='*60}")
        print(f"  {tag}")
        print(f"  Train: {len(ytr):,} ({ytr.mean()*100:.1f}% pos) | "
              f"Test: {len(yte):,} ({yte.mean()*100:.1f}% pos)")
        if active_only:
            # Compare to full test set size
            _, y_full, _ = assemble(va, n, feat_indices, False)
            print(f"  Active filter kept {len(yte)/len(y_full)*100:.1f}% of test samples")
        print(f"{'='*60}")

        model, _ = te.train_lgbm(Xtr, ytr, Xva, yva, feat_names, tag=tag)
        aucs = {}
        for z in set(Zte):
            mm = Zte == z; s = yte[mm].sum()
            if mm.sum() >= 30 and 0 < s < mm.sum():
                aucs[z] = roc_auc_score(yte[mm], model.predict(Xte[mm]))
        macro = float(np.mean(list(aucs.values())))
        pooled = roc_auc_score(yte, model.predict(Xte))

        print(f"\n  >>> {tag}: macro {macro:.4f} | pooled {pooled:.4f}")
        print(f"  Per-zone:")
        for z in sorted(aucs, key=lambda z: -aucs[z]):
            print(f"    {z:<20} {aucs[z]:.4f}")

        # Top 10 feature importance
        imp = dict(zip(feat_names, model.feature_importance(importance_type="gain")))
        tot = sum(imp.values()) or 1
        top = sorted(imp, key=lambda f: -imp[f])[:10]
        print(f"  Top 10 features:")
        for f in top:
            print(f"    {f:<30} {100*imp[f]/tot:.1f}%")

        return macro, aucs

    # ── Experiment A: Catalog-only on ALL hours ──
    a_macro, a_aucs = run("A: CATALOG ONLY (all hours)",
                          cat_idx, cat_names, active_only=False)

    # ── Experiment B: Environment-only on ALL hours ──
    b_macro, b_aucs = run("B: ENVIRONMENT ONLY (all hours)",
                          env_idx, env_names, active_only=False)

    # ── Experiment C: Environment-only on ACTIVE sequences only ──
    c_macro, c_aucs = run("C: ENVIRONMENT ONLY (active sequences only, eq_count_7d >= 3)",
                          env_idx, env_names, active_only=True)

    # ── Experiment D: Catalog + Environment on ACTIVE only ──
    all_idx = cat_idx + env_idx
    all_n = cat_names + env_names
    d_macro, d_aucs = run("D: FULL MODEL (active sequences only)",
                          all_idx, all_n, active_only=True)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY — Does gating improve environmental signal?")
    print("=" * 70)
    print(f"  A. Catalog only (all hours):        macro {a_macro:.4f}")
    print(f"  B. Environment only (all hours):     macro {b_macro:.4f}")
    print(f"  C. Environment only (active only):   macro {c_macro:.4f}  {'<<<' if c_macro > b_macro + 0.02 else ''}")
    print(f"  D. Full model (active only):         macro {d_macro:.4f}")
    print()
    print(f"  B → C lift (environment gains from gating): {c_macro - b_macro:+.4f}")
    print(f"  A → D lift (adding environment to active):  {d_macro - a_macro:+.4f}")
    if c_macro > b_macro + 0.02:
        print(f"\n  ✓ GATING WORKS — environmental signals gain {(c_macro-b_macro)*100:.1f}pp when")
        print(f"    conditioned on active sequences. The two-stage architecture is justified.")
    elif c_macro > b_macro:
        print(f"\n  ~ MARGINAL — small lift ({(c_macro-b_macro)*100:.1f}pp). Gating helps but isn't transformative.")
    else:
        print(f"\n  ✗ NO LIFT — environmental signals don't improve with gating.")
        print(f"    The sequence IS the signal; environment is noise.")

    # Per-zone comparison
    print(f"\n  Per-zone B→C (environment skill change with gating):")
    all_zones = sorted(set(list(b_aucs.keys()) + list(c_aucs.keys())))
    for z in all_zones:
        bv = b_aucs.get(z, float('nan'))
        cv = c_aucs.get(z, float('nan'))
        delta = cv - bv if not (np.isnan(bv) or np.isnan(cv)) else float('nan')
        flag = " <<<" if delta > 0.05 else ""
        print(f"    {z:<20} {bv:.4f} → {cv:.4f}  ({delta:+.4f}){flag}")


if __name__ == "__main__":
    main()
