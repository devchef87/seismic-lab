# SeismicLab — Tier-2 Swarm-Escalation Watch (FE integration)

## What this model is (and is not)

SeismicLab's forecasting purpose has been refocused. We do **not** claim to predict
arbitrary earthquakes. The honest, validated capability is **swarm-escalation
detection**:

> Given an **active seismic swarm** (a cluster of small quakes building, with no
> large event yet), what is the probability it **escalates to an independent
> mainshock (M5+) within 72 hours** — versus fizzling out like most swarms do?

This matters because:
- ~94% of active swarms fizzle. The model isolates the minority that escalate.
- It is **declustered** — it predicts *independent mainshocks*, not aftershocks
  (predicting aftershocks is trivial Omori-law bookkeeping and was explicitly
  removed from the target).
- Probabilities are **calibrated**: when it says 30%, ~30% actually escalate.

## Data contract: `tier2_watch.json`

Regenerated each scoring cycle. Top-level:

```json
{
  "generated": "2026-06-29 01:00:00+00:00",   // UTC timestamp of the scored window
  "base_rate_72h": 0.0602,                      // baseline: P(any active swarm escalates in 72h)
  "n_active_swarms": 25,                         // # cells with an active swarm right now
  "watch": [ ... ]                              // one entry per active-swarm cell, sorted desc by prob
}
```

Detection is **cluster-centered**: swarms are found by clustering recent small events
anywhere on Earth (DBSCAN, haversine), and each cluster becomes a region centered on
its own centroid — so coverage isn't limited to fixed zones and a fault crossing a
zone boundary isn't split. Each `watch[]` entry:

| field | type | meaning |
|-------|------|---------|
| `cell` | string | dynamic swarm id, e.g. `swarm_-30.2_-070.0` (centroid-based) |
| `zone` | string | region label — a known zone name where applicable, else a coordinate label (e.g. `tonga_fiji`, `50N_157E`) for swarms outside the named zones |
| `centroid` | [lat, lon] | swarm centroid (deg) — **pin the map marker here** |
| `extent` | [minlat, maxlat, minlon, maxlon] | tight bbox of the actual cluster quakes — use this if you want a small area outline |
| `n_recent_quakes` | int | small (M2.5-5) events in the cluster over the trailing 72h |
| `lat_range` | [lo, hi] | **INTERNAL scoring footprint** (~10° box) — do NOT render; it's the feature-computation window, not a hazard region |
| `lon_range` | [lo, hi] | internal scoring footprint — do NOT render |
| `escalation_prob_72h` | float 0-1 | **calibrated** probability the swarm escalates to M5+ in 72h |
| `alert_level` | string | NORMAL / ADVISORY / WATCH / WARNING (see below) |
| `lift_vs_base` | float | `escalation_prob_72h / base_rate_72h` — how many× above baseline |

**Map rendering:** draw a **marker at `centroid`** (size/color by alert level), optionally
a small outline from `extent`. Do **not** draw `lat_range`/`lon_range` — that's the ~10°
internal scoring window; rendering it produces large overlapping boxes that aren't
meaningful regions. **Only active swarms appear.** Absence = no swarm building there
(not "all clear" in a forecasting sense).

(The engine also supports a fixed-grid mode (`--mode grid`) for the original 13 zones,
but cluster mode is the default and recommended — it covers anywhere with activity.)

## Alert levels

Levels are set by **risk percentile** among active swarms (always populated, graded),
not fixed probability cuts. Suggested UI treatment:

| level | meaning | ~population | UI |
|-------|---------|-------------|-----|
| `WARNING` | top ~1% risk — high-confidence escalation | rare | red, prominent |
| `WATCH` | top ~5% risk — notably elevated | uncommon | orange |
| `ADVISORY` | top ~20% risk — modestly elevated | common | yellow |
| `NORMAL` | active swarm, baseline risk | most | gray/info |

Always show the **calibrated probability** alongside the level — the level is the
triage bucket, the probability is the honest number. e.g.
*"WATCH — central Chile swarm, 18% chance of M5+ in 72h (3× baseline)."*

## Honest framing for the UI copy

- Lead with the **question**, not a prediction: *"Active swarm detected — escalation risk: X%."*
- Use **72h** explicitly; this is a 3-day escalation window, not an instant forecast.
- A WARNING is a *watch posture*, not an evacuation order — even at the top tier,
  it's "much more likely than usual," not certainty.
- Where there's no active swarm, the honest state is "no swarm building," not
  "no earthquake risk."

## Model bundle (backend)

- `models/tier2_watch_lgb.txt` — LightGBM escalation model (multi-signal: seismicity,
  tidal stress, DART seafloor loading, GPS deformation, volcanic prior, waveforms)
- `models/tier2_watch_calib.npz` — isotonic calibrator + percentile alert thresholds
- Scoring writes `data/tier2_watch.json` (the contract above)

## Performance (held-out test, 2025-09 → 2026-06)

- Escalation AUC ~0.67; calibration verified (30%+ band → 72% actual escalation)
- WARNING tier: high precision, low recall (cry-wolf-rarely, be-right-when-you-do)
- Strongest zones: Alaska, New Zealand, South America, Japan/Kurils
