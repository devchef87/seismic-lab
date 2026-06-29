# Deployment Report — Replace "AI Forecast" with the Tier-2 Swarm-Escalation Watch

**Audience:** Frontend + Backend teams
**Companion spec:** `TIER2_WATCH_API.md` (JSON schema, field semantics, alert levels)
**Status:** model + calibrator + data contract shipped; needs a live scoring job + UI swap

---

## 1. Why we're changing it

The current **AI Forecast** panel ("Chile/Peru — M6 within 72h — 90% confidence", per-zone, always-on) is built on the old occurrence model. We discovered two problems and fixed them:

- **It was largely aftershock bookkeeping.** When we declustered the catalog, predicting *independent* mainshocks (the events that matter) scored near-random (~0.54). Most of the old "skill" was the trivial fact that aftershocks follow big quakes (Omori's law). The 90%-style confidences were overstated.
- **It answered the wrong question.** "Will an M6 happen somewhere in this zone in 72h" is both near-impossible (cold mainshocks) and not actionable.

The new model answers the **one question that is both honest and useful**:

> A seismic **swarm is active** here right now. What's the calibrated probability it
> **escalates to an independent M5+ mainshock within 72h** — vs. fizzling out (which
> ~94% of swarms do)?

It's **declustered** (no aftershock inflation), **calibrated** (30%→72% actually escalate — the numbers mean what they say), and **multi-signal** (seismicity + tidal stress + DART seafloor loading + GPS deformation + volcanic prior + waveforms).

---

## 2. What changes in the UI

| | OLD: AI Forecast | NEW: Swarm-Escalation Watch |
|---|---|---|
| Scope | every zone, always shown | only cells with an **active swarm** |
| Claim | "M6 within 72h, 90%" | "swarm active → X% escalation in 72h" |
| Granularity | parent zone | grid cell (roll up to zone for display) |
| Levels | HIGH / POSSIBLE | WARNING / WATCH / ADVISORY / NORMAL |
| Honesty | overstated | calibrated + declustered |

**The big UX shift:** the panel is no longer an always-on per-zone forecast. It's a
**live watchlist** that's often short (or empty) — and that's correct. No active swarm =
nothing building = no entry. Don't backfill it with zones to look busy.

---

## 3. Data contract (what FE consumes)

Single file/endpoint: **`data/tier2_watch.json`** (full schema in `TIER2_WATCH_API.md`).

```json
{
  "generated": "2026-06-29 01:00:00+00:00",
  "base_rate_72h": 0.0602,
  "n_active_swarms": 25,
  "watch": [
    { "cell": "south_america_-30_-072", "zone": "south_america",
      "lat_range": [-30,-20], "lon_range": [-72,-62],
      "escalation_prob_72h": 0.098, "alert_level": "ADVISORY", "lift_vs_base": 1.6 }
  ]
}
```

**FE rendering rules:**
1. Sort `watch` by `escalation_prob_72h` desc (already sorted in file).
2. Color by `alert_level`: WARNING=red, WATCH=orange, ADVISORY=yellow, NORMAL=gray.
3. Always show the calibrated probability next to the level — the level is the bucket,
   the probability is the honest number. *"WATCH — Chile swarm, 18% M5+ in 72h (3× normal)."*
4. **Empty/quiet state:** if `n_active_swarms == 0` or all NORMAL, show
   "No swarms currently building" — not an error, not "all clear from earthquakes."
5. Roll cells up to zones for a summary view (max alert level per zone), with cells
   as drill-down. `lat_range`/`lon_range` are provided for map polygons.

---

## 4. Backend wiring (the real work)

The model bundle is ready; the missing piece is a **periodic scoring job**. It must:

1. **Compute current-window features per grid cell** — reuse the feature builders in
   `lab/train_ensemble.py` (`build_full_dataset` logic) for the latest hour against the
   live DB. *This is the main effort: the scoring job needs the feature pipeline running
   on current data, not the cached historical .npz.*
2. **Gate to active swarms** — `build_tier2_labels()` episode rule (≥3 M2.5+ in trailing
   72h, no M5+ yet). Only these cells are scored.
3. **Score + calibrate** — load `models/tier2_watch_lgb.txt`, predict, apply the isotonic
   calibrator in `models/tier2_watch_calib.npz`.
4. **Assign alert level** — hybrid bands (from the calibrator bundle):
   - `WARNING` = calibrated prob ≥ `warn_prob` (0.30)
   - `WATCH` = raw score ≥ `raw_thresholds[1]` (95th pct)
   - `ADVISORY` = raw score ≥ `raw_thresholds[0]` (80th pct)
   - else `NORMAL`
5. **Write `data/tier2_watch.json`** (atomic write) and serve it.

**Cadence:** hourly is ample (the swarm gate uses a 72h trailing window; nothing changes
minute-to-minute). 6-hourly is fine too.

**Reference implementation:** `lab/tier2_watch.py` does steps 2-5 against the cache; the
production job is that logic with step 1 (live feature build) swapping the cache read.
We can extract this into `serve_tier2_watch.py` — see §6.

**Serving:** drop the JSON at the existing dashboard data path, or expose
`GET /api/tier2-watch`. Replace whatever currently feeds the AI Forecast panel
(`predict.py` / `predictions.jsonl` path).

---

## 5. Migration checklist

- [ ] Backend: build `serve_tier2_watch.py` (live feature build + score + write JSON)
- [ ] Backend: schedule it hourly; output to dashboard data path / endpoint
- [ ] Backend: retire the old AI Forecast feed (`predict.py` per-zone confidences)
- [ ] FE: replace AI Forecast panel with Swarm-Escalation Watch (consume `tier2_watch.json`)
- [ ] FE: alert-level styling + calibrated-probability display + empty state
- [ ] FE: cell→zone rollup + map polygons from `lat_range`/`lon_range`
- [ ] FE: update copy/labels to the swarm-escalation framing (see `TIER2_WATCH_API.md`)
- [ ] Both: keep the seismograph / live event feed as-is (independent of this panel)

---

## 6. Performance & honest-messaging notes (for copy)

Held-out test (2025-09 → 2026-06), calibrated:
- **WARNING** (≥30%): ~72% precision — rare, high-confidence. "Act on this."
- **WATCH** (top 5%): ~14% / ~2.4× baseline. "Notably elevated."
- **ADVISORY** (top 20%): ~12% / ~2× baseline. "Worth watching."
- Strongest zones: Alaska, New Zealand, South America, Japan/Kurils.

Copy guidance: lead with the question, not a prediction; always state the 72h window;
a WARNING is a heightened watch posture, not a certainty; "no swarm building" ≠ "no risk."

---

**Open item:** a waveform backfill is completing (~hours, mainly Alaska). A final model
refresh after it lands will marginally sharpen the strong zones — swap in the updated
bundle when ready; the JSON contract and UI do not change.
