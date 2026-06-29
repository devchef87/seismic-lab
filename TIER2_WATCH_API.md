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

Each `watch[]` entry:

| field | type | meaning |
|-------|------|---------|
| `cell` | string | grid-cell id, e.g. `south_america_-30_-072` |
| `zone` | string | parent zone (indonesia, japan_kurils, south_america, mexico_ca, himalaya, alaska, california, philippines, mediterranean, caribbean, new_zealand, png_solomon, kamchatka) |
| `lat_range` | [lo, hi] | cell latitude bounds (deg) — for map rendering |
| `lon_range` | [lo, hi] | cell longitude bounds (deg) |
| `escalation_prob_72h` | float 0-1 | **calibrated** probability the swarm escalates to M5+ in 72h |
| `alert_level` | string | NORMAL / ADVISORY / WATCH / WARNING (see below) |
| `lift_vs_base` | float | `escalation_prob_72h / base_rate_72h` — how many× above baseline |

**Only cells with an active swarm appear.** Absence of a cell = no active swarm
(not "all clear" in a forecasting sense — there's simply nothing building there).

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
